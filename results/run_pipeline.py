#!/usr/bin/env python
"""Newspaper page -> articles -> OCR text -> sentiment + topics -> summaries.

Usage:
    python run_pipeline.py --input ../RAW_Images
    python run_pipeline.py --input ../RAW_Images --force          # redo every page
    python run_pipeline.py --input ../RAW_Images --topic-method bertopic

Per-page results are cached in output/pages/<page_id>/page.json, so a stopped
run resumes where it left off.  Your input files are only ever read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from newspaper_pipeline import (compare, config, labelling, layout, llm_sentiment, ocr,
                                overlay, pages, preprocess, sentiment, summarize, textclean,
                                topics)

HERE = Path(__file__).resolve().parent
log = logging.getLogger("pipeline")


# --------------------------------------------------------------------------- per page
def page_flags(page: dict) -> list[str]:
    flags = []
    kept = [a for a in page["articles"] if a["kept"]]
    n = len(kept)
    if n < config.FLAG_MIN_ARTICLES:
        flags.append(f"only {n} article(s) kept")
    if n > config.FLAG_MAX_ARTICLES:
        flags.append(f"{n} articles - likely over-segmented")
    tot = page["layout_stats"]["body_words_total"] or 1
    orphan = page["layout_stats"]["orphan_words"] / tot
    if orphan > config.FLAG_MAX_ORPHAN_SHARE:
        flags.append(f"{orphan:.0%} of body text had no headline above it")
    words = [a["n_words"] for a in kept]
    if words and max(words) / max(1, sum(words)) > config.FLAG_MAX_SINGLE_ARTICLE_SHARE and n > 1:
        flags.append("one article holds most of the page text - possible merge")
    confs = [a["ocr_conf"] for a in kept if a["ocr_conf"] is not None]
    if confs and sum(confs) / len(confs) < config.FLAG_MIN_MEAN_CONF:
        flags.append(f"low OCR confidence ({sum(confs) / len(confs):.0f})")
    if abs(page["skew_deg"]) > config.FLAG_MAX_SKEW_DEG:
        flags.append(f"large skew corrected ({page['skew_deg']} deg)")
    if page["page_type"] not in ("front", "editorial"):
        flags.append(f"page type is '{page['page_type']}' (not front/editorial)")
    headless = sum(1 for a in kept if not a["headline"])
    if headless:
        flags.append(f"{headless} kept article(s) without a headline")
    return flags


def process_page(p: pages.PageInfo, paths: dict, force: bool) -> dict:
    pdir = paths["pages"] / p.page_id
    out_json = pdir / "page.json"
    if out_json.exists() and not force:
        cached = json.loads(out_json.read_text())
        if cached.get("version") == config.PIPELINE_VERSION:
            log.info("  cached - skipping (use --force to redo)")
            return cached
    pdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    color = pages.render(p)
    pre = preprocess.preprocess(color)
    log.info("  preprocessed: skew %.2f deg, %d photos, %d/%d rules (%.0fs)",
             pre.skew_deg, len(pre.photos), len(pre.hrules), len(pre.vrules), time.time() - t0)
    lay = layout.segment(pre, p.filename_page_type)
    log.info("  segmented: body H=%.0f px, %d candidate articles (%.0fs)",
             lay["H"] or 0, len(lay["articles"]), time.time() - t0)

    raw = []
    for i, a in enumerate(lay["articles"], 1):
        raw.append({"n": i, "layout": a, **ocr.ocr_article(pre.binary, a)})
    textclean.set_page_vocab([r["body_raw"] for r in raw])
    for r in raw:
        r.update(textclean.clean_article(r["headline_raw"], r["deck_raw"], r["body_raw"]))
        r["kept"], r["drop_reason"] = True, ""
        if r["n_words"] < config.MIN_ARTICLE_WORDS:
            r["kept"], r["drop_reason"] = False, f"under {config.MIN_ARTICLE_WORDS} words"
    # drop pull-quotes / boxes that repeat another article's text
    sh = {r["n"]: textclean.shingles(r["text"]) for r in raw}
    for r in raw:
        if not r["kept"] or not sh[r["n"]]:
            continue
        for o in raw:
            if o is r or not o["kept"] or o["n_words"] <= r["n_words"]:
                continue
            if len(sh[r["n"]] & sh[o["n"]]) / len(sh[r["n"]]) >= config.DUPLICATE_SHINGLE_OVERLAP:
                r["kept"], r["drop_reason"] = False, f"repeats article {o['n']} (pull quote)"
                break

    page = {
        "version": config.PIPELINE_VERSION,
        **p.to_dict(),
        "page_type": lay["page_type"],
        "skew_deg": pre.skew_deg,
        "n_photos": len(pre.photos),
        "layout_stats": {k: lay[k] for k in ("H", "masthead_y", "n_lines", "mean_line_conf",
                                             "orphan_words", "body_words_total")},
        "articles": raw,
        "seconds": round(time.time() - t0, 1),
    }
    page["flags"] = page_flags(page)

    # text files
    tdir = paths["text"] / p.page_id
    tdir.mkdir(parents=True, exist_ok=True)
    page_txt = []
    for r in raw:
        if not r["kept"]:
            continue
        body = (r["headline"] + "\n\n" if r["headline"] else "") + \
               (r["deck"] + "\n\n" if r["deck"] else "") + r["text"]
        (tdir / f"article_{r['n']:02d}.txt").write_text(body + "\n")
        page_txt.append(f"===== ARTICLE {r['n']} =====\n{body}\n")
    (paths["text"] / f"{p.page_id}.txt").write_text("\n".join(page_txt))

    overlay.draw(pre.color, {"blocks": lay["blocks"]},
                 [{"n": r["n"], "layout": r["layout"], "kept": r["kept"]} for r in raw],
                 pre.photos, paths["overlays"] / f"{p.page_id}_overlay.jpg")
    (pdir / "blocks.json").write_text(json.dumps(lay["blocks"]))
    tmp = out_json.with_suffix(".tmp")
    tmp.write_text(json.dumps(page, indent=1))
    tmp.replace(out_json)              # atomic: a half-written page is never "done"
    log.info("  %d articles kept / %d candidates, %.0fs. Flags: %s",
             sum(r["kept"] for r in raw), len(raw), page["seconds"], page["flags"] or "none")
    return page


# --------------------------------------------------------------------------- corpus
def _hash(s):
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def sentiment_with_cache(df: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    df = df.copy()
    df["_h"] = [_hash(h + "\n" + t) for h, t in zip(df["headline"], df["text"])]
    cache = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame(columns=["_h"])
    todo = df[~df["_h"].isin(cache["_h"])]
    if len(todo):
        log.info("Sentiment: scoring %d articles (%d cached)", len(todo), len(df) - len(todo))
        res = pd.DataFrame(sentiment.score_articles(todo[["article_id", "headline", "text"]]
                                                    .to_dict("records")))
        res["_h"] = todo["_h"].values
        res = res.drop(columns=["article_id"])
        cache = pd.concat([cache, res], ignore_index=True).drop_duplicates("_h", keep="last")
        cache.to_csv(cache_path, index=False)
    return df.merge(cache, on="_h", how="left").drop(columns=["_h"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--input", required=True, type=Path, help="folder of page images / PDFs")
    ap.add_argument("--output", type=Path, default=HERE / "output")
    ap.add_argument("--force", action="store_true", help="re-process pages even if cached")
    ap.add_argument("--pages-only", action="store_true", help="stop after OCR (.txt files)")
    ap.add_argument("--topic-method", choices=["auto", "bertopic", "lda"], default="auto",
                    help="auto = BERTopic if >= 100 articles, else LDA")
    ap.add_argument("--llm", choices=["claude", "hf_api", "local", "off"], default=config.LLM_BACKEND,
                    help="LLM sentiment backend: claude = Claude Haiku 5.5 (default), hf_api, local, off")
    ap.add_argument("--llm-model", default=None, help="override the LLM model id")
    ap.add_argument("--no-compare", action="store_true",
                    help="don't also run BERTopic for comparison when LDA is chosen")
    args = ap.parse_args()

    paths = config.default_paths(HERE)
    paths = {k: args.output / v.relative_to(HERE / "output") if k != "output" else args.output
             for k, v in paths.items()}
    for v in paths.values():
        v.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(paths["output"] / "run.log")])

    found = pages.discover(args.input.resolve())
    if not found:
        sys.exit(f"No images or PDFs found in {args.input}")
    log.info("Found %d page files in %s", len(found), args.input)

    page_results = []
    for i, p in enumerate(found, 1):
        log.info("[%d/%d] %s", i, len(found), p.source_file)
        try:
            page_results.append(process_page(p, paths, args.force))
        except Exception as e:  # keep going; the page is retried next run
            log.exception("  FAILED: %s", e)

    # page type overrides (optional): page_type_overrides.csv with page_id,page_type
    ov_path = HERE / "page_type_overrides.csv"
    overrides = {}
    if ov_path.exists():
        overrides = dict(pd.read_csv(ov_path).astype(str).values)
    report = []
    rows = []
    for pg in page_results:
        ptype = overrides.get(pg["page_id"], pg["page_type"])
        kept = [a for a in pg["articles"] if a["kept"]]
        report.append({
            "page_id": pg["page_id"], "source_file": pg["source_file"], "date": pg["date"],
            "page_type": ptype, "articles_kept": len(kept),
            "candidates": len(pg["articles"]),
            "dropped_short": sum(1 for a in pg["articles"] if "words" in a["drop_reason"]),
            "dropped_duplicate": sum(1 for a in pg["articles"] if "repeats" in a["drop_reason"]),
            "words_kept": sum(a["n_words"] for a in kept),
            "mean_ocr_conf": round(sum(a["ocr_conf"] or 0 for a in kept) / max(1, len(kept)), 1),
            "skew_deg": pg["skew_deg"], "photos_removed": pg["n_photos"],
            "unreliable": bool(pg["flags"]), "flags": "; ".join(pg["flags"]),
        })
        for a in kept:
            rows.append({
                "article_id": f"{pg['page_id']}_a{a['n']:02d}",
                "date": pg["date"], "page_type": ptype, "source_file": pg["source_file"],
                "article_no": a["n"], "headline": a["headline"], "deck": a["deck"],
                "byline": a["byline"], "text": a["text"], "n_words": a["n_words"],
                "ocr_conf": round(a["ocr_conf"], 1) if a["ocr_conf"] else None,
                "no_headline": not a["headline"],
                "page_flagged": bool(pg["flags"]),
            })
    rep = pd.DataFrame(report)
    rep.to_csv(paths["tables"] / "page_report.csv", index=False)
    log.info("\n%s", rep[["page_id", "page_type", "articles_kept", "candidates",
                          "mean_ocr_conf", "flags"]].to_string(index=False))
    if args.pages_only or not rows:
        return
    df = pd.DataFrame(rows)

    # sentiment: transformer + VADER
    df = sentiment_with_cache(df, paths["output"] / "sentiment_cache.csv")
    # sentiment: LLM
    llm_cols = ["llm_label", "llm_score", "llm_reason", "llm_model"]
    if args.llm != "off":
        res = llm_sentiment.score_articles(
            df[["article_id", "headline", "text"]].to_dict("records"),
            paths["output"] / "llm_cache.jsonl", args.llm, args.llm_model)
        llm = pd.DataFrame.from_dict(res, orient="index").reindex(columns=llm_cols)
        df = df.merge(llm, left_on="article_id", right_index=True, how="left")
        log.info("LLM sentiment available for %d/%d articles", df["llm_label"].notna().sum(),
                 len(df))
    for c in llm_cols:
        if c not in df:
            df[c] = None

    # topics
    log.info("Topic modelling on %d articles (method=%s)", len(df), args.topic_method)
    tm = topics.run(df["text"].tolist(), args.topic_method)
    df["topic"] = tm["doc_topic"]
    df["topic_prob"] = tm["doc_topic_prob"]
    df["topic_label"] = [tm["topic_labels"][t] for t in tm["doc_topic"]]
    df["topic_method"] = tm["method"]
    tinfo = pd.DataFrame([{"topic": t, "label": tm["topic_labels"][t],
                           "n_articles": int((df["topic"] == t).sum()),
                           "top_words": ", ".join(w)} for t, w in tm["topic_words"].items()])
    tinfo.to_csv(paths["tables"] / "topics.csv", index=False)
    (paths["tables"] / "topic_model_info.json").write_text(json.dumps(
        {k: v for k, v in tm.items() if k not in ("doc_topic", "doc_topic_prob")}, indent=1,
        default=str))
    if tm["method"] == "LDA" and not args.no_compare:
        try:
            bt = topics.run(df["text"].tolist(), "bertopic")
            cmp_ = df[["article_id", "headline", "topic_label"]].rename(
                columns={"topic_label": "lda_topic"})
            cmp_["bertopic_topic"] = [bt["topic_labels"][t] for t in bt["doc_topic"]]
            cmp_.to_csv(paths["tables"] / "topics_bertopic_comparison.csv", index=False)
            log.info("BERTopic comparison: %d topics (saved, not used for summaries)",
                     bt["n_topics"])
        except Exception as e:
            log.warning("BERTopic comparison failed: %s", e)

    cols = ["article_id", "date", "page_type", "source_file", "article_no", "headline", "deck",
            "byline", "text", "n_words", "ocr_conf", "no_headline", "page_flagged",
            "topic", "topic_label", "topic_prob", "topic_method",
            "sentiment_label", "sentiment_score", "p_positive", "p_neutral", "p_negative",
            "sentiment_score_sd", "n_sentences", "share_pos_sentences", "share_neg_sentences",
            "headline_label", "headline_score", "headline_p_positive", "headline_p_neutral",
            "headline_p_negative", "vader_compound", "vader_label",
            "llm_label", "llm_score", "llm_reason", "llm_model"]
    df = df[cols]
    df.to_csv(paths["output"] / "articles.csv", index=False)
    stats = summarize.run(df, paths["tables"], paths["charts"])
    agree, dist = compare.agreement(df)
    agree.to_csv(paths["tables"] / "method_agreement.csv", index=False)
    dist.to_csv(paths["tables"] / "method_label_distribution.csv")
    compare.topic_chart(df, paths["charts"] / "methods_by_topic.png")
    try:
        from newspaper_pipeline import method_charts, topic_charts
        method_charts.run(df, paths["charts"] / "methods")
        topic_charts.run(df, paths["charts"] / "topics", paths["output"] / "embeddings.npz")
    except Exception as e:  # noqa: BLE001 - extra charts must never stop the run
        log.warning("extra charts failed: %s", e)
    log.info("Method agreement (no labels needed):\n%s", agree.to_string(index=False))
    created = labelling.export_sample(df, paths["labelling"] / "label_sample.xlsx")
    log.info("Label sample %s", "written" if created else "already exists - left untouched")
    log.info("Done: %d articles. Transformer/VADER label agreement %.0f%%, Spearman %.2f",
             len(df), 100 * stats["agreement_transformer_vader"],
             stats["spearman_transformer_vader"])


if __name__ == "__main__":
    main()
