"""Charts that explain each sentiment method on its own and how the three compare.

  methods/score_distributions.png   each method's score histogram, coloured by its own label
  methods/score_scatter.png         pairwise score scatter (hexbin) with Spearman rho
  methods/label_crosstabs.png       pairwise label cross-tables (row % and counts)
  methods/label_distribution.png    label mix per method: all / front / editorial
  methods/agreement_by_topic.png    pairwise label agreement + all-three agreement per topic
  methods/agreement_by_month.png    pairwise Cohen's kappa per month (stability over time)
  methods/llm_reasons.png           words Claude Haiku uses in its one-line reasons, per label
  methods/vader_misleading_words.png  positive-lexicon words in articles VADER calls positive
                                    but Claude Haiku calls negative
"""
from __future__ import annotations

import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from .summarize import GRID, INK, INK2, MUTED, NEG, NEU, POS, SURFACE, _style

DPI = 150
LABELS = ["negative", "neutral", "positive"]
LCOL = {"negative": NEG, "neutral": NEU, "positive": POS}
M = {"vader": ("VADER", "vader_label", "vader_compound", "#E69F00"),
     "transformer": ("Transformer", "sentiment_label", "sentiment_score", "#009E73"),
     "llm": ("Claude Haiku", "llm_label", "llm_score", "#0072B2")}
STOP = set("""a an the of to in and or for on with by at as is are was were be been being it its
this that these those from about over into than then but not no nor so such which who whom whose
what when where while how all any both each few more most other some own same very can will
just should now article articles report reports reporting tone overall news india indian
""".split())


def _avail(df):
    return [k for k in M if M[k][1] in df and df[M[k][1]].notna().mean() > 0.5]


def _fin(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def _spines(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def run(df: pd.DataFrame, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["llm_score"] = pd.to_numeric(df.get("llm_score"), errors="coerce")
    meths = _avail(df)
    made = []

    # 1 score distributions coloured by each method's own label --------------------------
    fig, axes = plt.subplots(1, len(meths), figsize=(13, 4.2), facecolor=SURFACE)
    bins = np.linspace(-1, 1, 41)
    for ax, k in zip(np.atleast_1d(axes), meths):
        name, lc, sc, col = M[k]
        bottom = np.zeros(len(bins) - 1)
        for lab in LABELS:
            h, _ = np.histogram(df.loc[df[lc] == lab, sc].dropna(), bins=bins)
            ax.bar(bins[:-1], h, width=np.diff(bins), align="edge", bottom=bottom,
                   color=LCOL[lab], edgecolor=SURFACE, linewidth=0.4,
                   label=f"{lab} {(df[lc] == lab).mean():.0%}")
            bottom += h
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_title(name, loc="left", fontsize=11, color=col, fontweight="bold")
        ax.set_xlabel({"vader": "VADER compound (sentence mean)",
                       "transformer": "P(pos) − P(neg) (sentence mean)",
                       "llm": "Claude Haiku score"}[k], fontsize=8.5, color=INK2)
        _spines(ax)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    np.atleast_1d(axes)[0].set_ylabel("articles", color=INK2)
    fig.suptitle("How each method scores and labels the same 5,648 articles"
                 if len(df) == 5648 else "How each method scores and labels the articles",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.text(0.01, 0.905, "bar colour = the label that method gave · VADER labels by fixed "
             "thresholds (±0.05) · Transformer by its largest probability · Claude Haiku "
             "chooses label and score itself", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_dir / "score_distributions.png", dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    made.append("score_distributions")

    pairs = list(combinations(meths, 2))
    # 2 pairwise score scatter -----------------------------------------------------------
    fig, axes = plt.subplots(1, len(pairs), figsize=(13, 4.6), facecolor=SURFACE)
    for ax, (a, b) in zip(np.atleast_1d(axes), pairs):
        d = df[[M[a][2], M[b][2]]].dropna()
        hb = ax.hexbin(d[M[a][2]], d[M[b][2]], gridsize=30, cmap="Greys", mincnt=1,
                       bins="log", extent=(-1, 1, -1, 1))
        rho = spearmanr(d[M[a][2]], d[M[b][2]]).statistic
        ax.axhline(0, color=GRID, lw=0.8)
        ax.axvline(0, color=GRID, lw=0.8)
        ax.set_xlabel(f"{M[a][0]} score", color=M[a][3], fontsize=9)
        ax.set_ylabel(f"{M[b][0]} score", color=M[b][3], fontsize=9)
        ax.set_title(f"Spearman ρ = {rho:.2f}", loc="left", fontsize=10, color=INK)
        _spines(ax)
    cb = fig.colorbar(hb, ax=axes, shrink=0.8, pad=0.01)
    cb.set_label("articles per cell (log scale)", fontsize=8.5, color=INK2)
    fig.suptitle("Do the methods rank articles the same way?  (scores, pairwise)", x=0.01,
                 ha="left", fontsize=13, color=INK)
    fig.savefig(out_dir / "score_scatter.png", dpi=DPI, facecolor=SURFACE,
                bbox_inches="tight")
    plt.close(fig)
    made.append("score_scatter")

    # 3 pairwise label cross-tables ------------------------------------------------------
    fig, axes = plt.subplots(1, len(pairs), figsize=(13, 4.4), facecolor=SURFACE)
    for ax, (a, b) in zip(np.atleast_1d(axes), pairs):
        ct = pd.crosstab(df[M[a][1]], df[M[b][1]]).reindex(index=LABELS, columns=LABELS,
                                                           fill_value=0)
        rp = ct.div(ct.sum(1).replace(0, np.nan), axis=0)
        ax.imshow(rp.values, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{rp.values[i, j]:.0%}\n({ct.values[i, j]})", ha="center",
                        va="center", fontsize=8.5,
                        color="white" if rp.values[i, j] > 0.55 else INK)
        agree = np.trace(ct.values) / ct.values.sum()
        kap = cohen_kappa_score(df[M[a][1]], df[M[b][1]], labels=LABELS)
        ax.set_xticks(range(3), ["neg", "neu", "pos"])
        ax.set_yticks(range(3), ["neg", "neu", "pos"])
        ax.set_ylabel(f"{M[a][0]} label", color=M[a][3], fontsize=9)
        ax.set_xlabel(f"{M[b][0]} label", color=M[b][3], fontsize=9)
        ax.set_title(f"agree {agree:.0%} · κ = {kap:.2f}", loc="left", fontsize=10, color=INK)
        ax.tick_params(colors=INK2, labelsize=8.5)
    fig.suptitle("Where the labels go: rows = first method's label, % of that row",
                 x=0.01, ha="left", fontsize=13, color=INK)
    _fin(fig, out_dir / "label_crosstabs.png")
    made.append("label_crosstabs")

    # 4 label mix per method x page type -------------------------------------------------
    rows = []
    for k in meths:
        for grp, g in [("all", df)] + [(p, df[df["page_type"] == p])
                                       for p in ("front", "editorial")
                                       if p in df["page_type"].unique()]:
            vc = g[M[k][1]].value_counts(normalize=True).reindex(LABELS).fillna(0)
            rows.append((f"{M[k][0]} · {grp}", k, *vc.values))
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(rows) + 1.8), facecolor=SURFACE)
    for i, (name, k, ng, nu, po) in enumerate(rows):
        left = 0
        for v, lab in zip((ng, nu, po), LABELS):
            ax.barh(i, v, left=left, color=LCOL[lab], height=0.65, edgecolor=SURFACE,
                    linewidth=1.5, label=lab if i == 0 else None)
            if v >= 0.06:
                ax.text(left + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                        color=INK if lab == "neutral" else "white")
            left += v
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    for t, r in zip(ax.get_yticklabels(), rows):
        t.set_color(M[r[1]][3])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=9)
    _style(ax, "Label mix per method", xlabel="share of articles",
           subtitle="same articles, three very different pictures")
    _fin(fig, out_dir / "label_distribution.png")
    made.append("label_distribution")

    # 5 agreement by topic ---------------------------------------------------------------
    if "topic_label" in df:
        tops = df["topic_label"].value_counts().index.tolist()
        res = []
        for t in tops:
            g = df[df["topic_label"] == t]
            r = {"topic": t, "n": len(g)}
            for a, b in pairs:
                r[f"{a}-{b}"] = (g[M[a][1]] == g[M[b][1]]).mean()
            if len(meths) == 3:
                r["all"] = ((g[M[meths[0]][1]] == g[M[meths[1]][1]])
                            & (g[M[meths[1]][1]] == g[M[meths[2]][1]])).mean()
            res.append(r)
        res = pd.DataFrame(res).sort_values("all" if len(meths) == 3 else res.columns[2])
        fig, ax = plt.subplots(figsize=(10, 0.45 * len(res) + 2), facecolor=SURFACE)
        y = np.arange(len(res))
        pcol = {0: "#CC79A7", 1: "#56B4E9", 2: "#009E73"}
        for i, (a, b) in enumerate(pairs):
            ax.plot(res[f"{a}-{b}"], y, "o", ms=6, color=pcol[i],
                    label=f"{M[a][0]} = {M[b][0]}", zorder=3)
        if "all" in res:
            ax.barh(y, res["all"], color=NEU, height=0.6, label="all three agree", zorder=1)
        ax.set_yticks(y, [f"{t}  (n={n})" for t, n in zip(res["topic"], res["n"])], fontsize=8.5)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.07),
                  fontsize=8.5)
        _style(ax, "Method agreement by topic", xlabel="share of articles with the same label",
               subtitle="grey bar = all three methods agree · dots = each pair")
        _fin(fig, out_dir / "agreement_by_topic.png")
        made.append("agreement_by_topic")

    # 6 agreement (kappa) by month -------------------------------------------------------
    df["month"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m")
    months = sorted(df["month"].unique())
    if len(months) >= 3:
        fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=SURFACE)
        for i, (a, b) in enumerate(pairs):
            kv = [cohen_kappa_score(g[M[a][1]], g[M[b][1]], labels=LABELS)
                  for _, g in df.groupby("month")]
            ax.plot(range(len(months)), kv, "-o", ms=4, lw=1.6,
                    color={0: "#CC79A7", 1: "#56B4E9", 2: "#009E73"}[i],
                    label=f"{M[a][0]} vs {M[b][0]}")
        ax.set_xticks(range(len(months)), [pd.Period(m).strftime("%b") for m in months])
        ax.set_ylim(-0.05, max(0.5, ax.get_ylim()[1]))
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1),
                  fontsize=9)
        _style(ax, "Agreement between methods, month by month", ylabel="Cohen's κ",
               subtitle="flat lines = the disagreement is stable, not caused by one period")
        _fin(fig, out_dir / "agreement_by_month.png")
        made.append("agreement_by_month")

    # 7 what Claude Haiku cites in its reasons ------------------------------------------
    if "llm_reason" in df and df["llm_reason"].notna().mean() > 0.5:
        tok = lambda s: [w for w in re.findall(r"[a-z][a-z'-]+", str(s).lower())
                         if w not in STOP and len(w) > 2]
        cnt = {lab: Counter(w for r in df.loc[df["llm_label"] == lab, "llm_reason"]
                            for w in set(tok(r))) for lab in LABELS}
        tot = {lab: (df["llm_label"] == lab).sum() for lab in LABELS}
        fig, axes = plt.subplots(1, 3, figsize=(13, 5), facecolor=SURFACE)
        for ax, lab in zip(axes, LABELS):
            # words most typical of this label: share in this label minus share elsewhere
            others = [o for o in LABELS if o != lab]
            sc = {w: c / tot[lab] - sum(cnt[o][w] for o in others) / sum(tot[o] for o in others)
                  for w, c in cnt[lab].items() if c >= 15}
            top = sorted(sc, key=sc.get, reverse=True)[:15]
            vals = [cnt[lab][w] / tot[lab] for w in top]
            ax.barh(range(len(top))[::-1], vals, color=LCOL[lab] if lab != "neutral" else MUTED,
                    height=0.7)
            ax.set_yticks(range(len(top))[::-1], top, fontsize=9)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
            ax.set_title(f"{lab} (n={tot[lab]})", loc="left", fontsize=11,
                         color=LCOL[lab] if lab != "neutral" else INK2, fontweight="bold")
            ax.set_xlabel("share of that label's reasons using the word", fontsize=8, color=INK2)
            _spines(ax)
        fig.suptitle("Why Claude Haiku chose each label: words typical of its one-line reasons",
                     x=0.01, ha="left", fontsize=13, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(out_dir / "llm_reasons.png", dpi=DPI, facecolor=SURFACE)
        plt.close(fig)
        made.append("llm_reasons")

    # 8 words that make VADER positive where Claude Haiku says negative -----------------
    if {"vader", "llm"} <= set(meths):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        lex = SentimentIntensityAnalyzer().lexicon
        sub = df[(df["vader_label"] == "positive") & (df["llm_label"] == "negative")]
        rest = df.drop(sub.index)
        def docfreq(texts):
            c = Counter()
            for t in texts:
                c.update({w for w in re.findall(r"[a-z]+", str(t).lower()) if lex.get(w, 0) > 0})
            return c
        cs, cr = docfreq(sub["text"]), docfreq(rest["text"])
        top = [w for w, _ in cs.most_common(60)]
        top = sorted(top, key=lambda w: cs[w] / len(sub) * lex[w], reverse=True)[:20]
        fig, ax = plt.subplots(figsize=(10, 6.5), facecolor=SURFACE)
        y = np.arange(len(top))[::-1]
        ax.barh(y, [cs[w] / len(sub) for w in top], color=POS, height=0.6,
                label=f"in the {len(sub)} disputed articles")
        ax.plot([cr[w] / max(len(rest), 1) for w in top], y, "o", color=INK, ms=5,
                label="in all other articles")
        ax.set_yticks(y, [f"{w}  (+{lex[w]:.1f})" for w in top], fontsize=9)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v * 100:g}%"))
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9, loc="lower right")
        _style(ax, "Why VADER reads bad news as positive",
               xlabel="share of articles containing the word",
               subtitle=f"articles VADER calls positive but Claude Haiku calls negative "
                        f"({len(sub) / len(df):.0%} of all) · positive VADER words, "
                        f"lexicon weight in brackets")
        _fin(fig, out_dir / "vader_misleading_words.png")
        made.append("vader_misleading_words")
    return made
