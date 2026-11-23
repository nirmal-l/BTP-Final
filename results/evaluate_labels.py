#!/usr/bin/env python
"""Score the sentiment methods (transformer, VADER, Claude Haiku) against your hand labels.

1. Fill the 'my label' column of a labelling sheet (positive / neutral / negative).
2. Run:
       python evaluate_labels.py                                   # output/labelling/label_sample.xlsx
       python evaluate_labels.py --labels output/labelling/label_me.xlsx --run output_jan_week_pdf

--run is the output folder whose articles.csv the labelled article_ids come from.
Articles in that run that have no Claude Haiku score yet are scored first (cached).
Writes accuracy_report.md, method_accuracy.csv/.png, method_mcnemar.csv and
method_accuracy_breakdown.csv to <run>/labelling/.
"""
import argparse
from pathlib import Path

import pandas as pd

from newspaper_pipeline import config, labelling, llm_sentiment

HERE = Path(__file__).resolve().parent
LLM_COLS = ["llm_label", "llm_score", "llm_reason", "llm_model"]


def ensure_llm(run: Path) -> Path:
    """Return an articles csv for `run` that has Claude Haiku scores filled in."""
    arts = pd.read_csv(run / "articles.csv")
    if "llm_label" in arts and arts["llm_label"].notna().all():
        return run / "articles.csv"
    print(f"Scoring {len(arts)} articles of {run.name} with {config.LLM_MODEL_CLAUDE} (cached)...")
    res = llm_sentiment.score_articles(arts[["article_id", "headline", "text"]].to_dict("records"),
                                       run / "llm_cache.jsonl", "claude")
    llm = pd.DataFrame.from_dict(res, orient="index").reindex(columns=LLM_COLS)
    arts = arts.drop(columns=[c for c in LLM_COLS if c in arts]).merge(
        llm, left_on="article_id", right_index=True, how="left")
    out = run / "articles_with_llm.csv"          # articles.csv itself is left untouched
    arts.to_csv(out, index=False)
    print(f"Claude Haiku scores for {arts['llm_label'].notna().sum()}/{len(arts)} articles -> {out.name}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run", type=Path, default=HERE / "output")
    ap.add_argument("--labels", type=Path, default=None,
                    help="labelling sheet (default <run>/labelling/label_sample.xlsx)")
    ap.add_argument("--no-llm", action="store_true", help="don't score missing Claude Haiku labels")
    a = ap.parse_args()
    labels = a.labels or a.run / "labelling" / "label_sample.xlsx"
    arts = a.run / "articles.csv" if a.no_llm else ensure_llm(a.run)
    print(labelling.evaluate(labels, arts, a.run / "labelling"))


if __name__ == "__main__":
    main()
