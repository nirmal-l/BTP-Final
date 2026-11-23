"""Comparing the three sentiment methods: transformer, VADER and LLM.

Two kinds of comparison:
  * without labels  - how often the methods agree with each other (all articles)
  * with hand labels - which method is closest to a human judge, and whether the
                       difference between two methods is more than chance (McNemar)
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr
from sklearn.metrics import cohen_kappa_score, f1_score

LABELS = ["positive", "neutral", "negative"]
# name -> (label column, score column)
METHODS = {
    "transformer": ("sentiment_label", "sentiment_score"),
    "vader": ("vader_label", "vader_compound"),
    "llm": ("llm_label", "llm_score"),
}
COLORS = {"transformer": "#009E73", "vader": "#E69F00", "llm": "#0072B2"}   # same as figures
NAMES = {"transformer": "Transformer", "vader": "VADER", "llm": "Claude Haiku"}
INK2, MUTED, GRID, SURFACE = "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def available(df: pd.DataFrame) -> dict:
    return {m: c for m, c in METHODS.items() if c[0] in df and df[c[0]].notna().any()}


def _wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


# --------------------------------------------------------------------------- no labels
def agreement(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    meth = available(df)
    rows = []
    for a, b in combinations(meth, 2):
        (la, sa), (lb, sb) = meth[a], meth[b]
        d = df.dropna(subset=[la, lb])
        rho = spearmanr(d[sa], d[sb]).statistic if len(d) > 2 else np.nan
        rows.append({
            "method_a": a, "method_b": b, "n": len(d),
            "label_agreement": round((d[la] == d[lb]).mean(), 3),
            "cohen_kappa": round(cohen_kappa_score(d[la], d[lb], labels=LABELS), 3),
            "spearman_scores": round(float(rho), 3),
        })
    dist = pd.DataFrame({m: df[c[0]].value_counts(normalize=True).reindex(LABELS).fillna(0)
                         for m, c in meth.items()}).T.round(3)
    dist.index.name = "method"
    return pd.DataFrame(rows), dist


def topic_chart(df: pd.DataFrame, path: Path):
    """Mean score per topic for each method, z-scored within method (the three
    scores live on different scales, so only the pattern across topics compares)."""
    meth = available(df)
    key = "llm" if "llm" in meth else "transformer"
    sc_key = pd.to_numeric(df[METHODS[key][1]], errors="coerce")
    topics = sc_key.groupby(df["topic_label"]).mean().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(topics) + 2.4), facecolor=SURFACE)
    y = np.arange(len(topics))
    for m, (_, sc) in meth.items():
        s = pd.to_numeric(df[sc], errors="coerce")
        z = (s - s.mean()) / s.std(ddof=0)
        means = z.groupby(df["topic_label"]).mean().reindex(topics)
        ax.plot(means, y, "o", ms=8, color=COLORS[m], mec=SURFACE, mew=1.5, label=NAMES[m],
                zorder=3)
    for yi, tp in enumerate(topics):   # thin line spanning the three dots = disagreement
        vals = [((pd.to_numeric(df[sc], errors="coerce") - pd.to_numeric(df[sc], errors="coerce").mean())
                 / pd.to_numeric(df[sc], errors="coerce").std(ddof=0))[df["topic_label"] == tp].mean()
                for _, sc in meth.values()]
        ax.plot([min(vals), max(vals)], [yi, yi], color=GRID, lw=3, zorder=1)
    ax.axvline(0, color=MUTED, lw=1)
    n = df["topic_label"].value_counts()
    ax.set_yticks(y, [f"{t}  (n={n[t]})" for t in topics])
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_xlabel("topic mean in standard deviations from the method's own average (0)",
                  color=INK2, fontsize=9)
    ax.set_title("Do the three methods rank topics the same way?", loc="left", fontsize=12,
                 pad=22)
    ax.text(0, 1.015, f"sorted by {NAMES[key]} · grey bar = spread between methods (short = "
            "they agree)", transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.07),
              fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _scores(y, p):
    n = len(y)
    k = int(sum(a == b for a, b in zip(y, p)))
    lo, hi = _wilson(k, n)
    return {"n": n, "accuracy": round(k / n, 3) if n else np.nan,
            "acc_95ci_low": round(lo, 3), "acc_95ci_high": round(hi, 3),
            "macro_f1": round(f1_score(y, p, labels=LABELS, average="macro", zero_division=0), 3)
            if n else np.nan,
            "cohen_kappa": round(cohen_kappa_score(y, p, labels=LABELS), 3) if n > 1 else np.nan}


def mcnemar(y, pa, pb):
    """Exact McNemar test: of the articles where exactly one method is right,
    is one method right significantly more often than the other?"""
    a_only = sum((x == t) and (z != t) for t, x, z in zip(y, pa, pb))
    b_only = sum((z == t) and (x != t) for t, x, z in zip(y, pa, pb))
    n = a_only + b_only
    p = binomtest(a_only, n, 0.5).pvalue if n else 1.0
    return a_only, b_only, p


def evaluate(m: pd.DataFrame, out_dir: Path) -> str:
    """m: articles merged with a 'my label' column (only labelled rows)."""
    meth = {k: v for k, v in available(m).items()}
    y = m["my label"].tolist()
    majority = max(LABELS, key=y.count)
    rows = []
    for name, (lc, _) in meth.items():
        d = m.dropna(subset=[lc])
        rows.append({"method": name, **_scores(d["my label"].tolist(), d[lc].tolist())})
    rows.append({"method": f"baseline: always '{majority}'", **_scores(y, [majority] * len(y))})
    overall = pd.DataFrame(rows)

    # pairwise significance on articles every method scored
    common = m.dropna(subset=[c[0] for c in meth.values()])
    yc = common["my label"].tolist()
    pairs = []
    for a, b in combinations(meth, 2):
        ao, bo, p = mcnemar(yc, common[meth[a][0]].tolist(), common[meth[b][0]].tolist())
        pairs.append({"method_a": a, "method_b": b, "n_common": len(common),
                      "only_a_correct": ao, "only_b_correct": bo, "p_value": round(p, 4),
                      "significant_at_0.05": p < 0.05})
    pairs = pd.DataFrame(pairs)

    # breakdowns
    parts = []
    for col in ("page_type", "topic_label"):
        for g, d in m.groupby(col):
            row = {"group_by": col, "group": g, "n": len(d)}
            for name, (lc, _) in meth.items():
                dd = d.dropna(subset=[lc])
                row[f"{name}_accuracy"] = round((dd["my label"] == dd[lc]).mean(), 3) \
                    if len(dd) else np.nan
            parts.append(row)
    breakdown = pd.DataFrame(parts)

    out_dir.mkdir(parents=True, exist_ok=True)
    overall.to_csv(out_dir / "method_accuracy.csv", index=False)
    pairs.to_csv(out_dir / "method_mcnemar.csv", index=False)
    breakdown.to_csv(out_dir / "method_accuracy_breakdown.csv", index=False)

    # chart: accuracy with 95% CI per method
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(overall) + 1.5), facecolor=SURFACE)
    yy = np.arange(len(overall))
    for i, r in overall.iterrows():
        c = COLORS.get(r["method"], MUTED)
        ax.plot([r["acc_95ci_low"], r["acc_95ci_high"]], [i, i], color=c, lw=2)
        ax.scatter([r["accuracy"]], [i], s=70, color=c, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.set_yticks(yy, overall["method"])
    ax.set_xlim(0, 1)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel(f"accuracy vs your labels (n = {len(m)}); line = 95% interval",
                  color=INK2, fontsize=9)
    ax.set_title("Which sentiment method matches the hand labels best?", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "method_accuracy.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    cms = []
    for name, (lc, _) in meth.items():
        d = m.dropna(subset=[lc])
        cm = pd.crosstab(pd.Categorical(d["my label"], LABELS), pd.Categorical(d[lc], LABELS),
                         rownames=["your label"], colnames=[name], dropna=False)
        cms += [f"### {name}", "", cm.to_markdown(), ""]

    best = overall.iloc[:-1].sort_values("macro_f1", ascending=False)
    sig = pairs[pairs["significant_at_0.05"]]
    verdict = (f"Highest macro-F1: **{best.iloc[0]['method']}** "
               f"({best.iloc[0]['macro_f1']}). ") if len(best) else ""
    verdict += ("No pair of methods differs significantly (McNemar p >= 0.05) - with this "
                "many labels the ranking may be chance." if sig.empty else
                "Significant differences: " + "; ".join(
                    f"{r.method_a} vs {r.method_b} (p={r.p_value})" for r in sig.itertuples()))
    text = "\n".join([
        "# Sentiment methods vs hand labels", "",
        f"Labelled articles: {len(m)}. Methods compared: {', '.join(meth)}.", "",
        "## Verdict", "", verdict, "",
        "## Overall", "", overall.to_markdown(index=False), "",
        "* accuracy = share of articles where the method's label equals yours",
        "* macro-F1 = average F1 over positive/neutral/negative - not fooled by a method "
        "that just says 'neutral' to everything",
        "* Cohen's kappa = agreement beyond chance (0 = chance, 1 = perfect)", "",
        "## Is the difference real? (exact McNemar test)", "",
        pairs.to_markdown(index=False), "",
        "## Accuracy by page type and topic", "",
        breakdown.to_markdown(index=False), "",
        "Small groups (n < 10) are anecdotal.", "",
        "## Confusion matrices (rows = your label)", "", *cms,
    ])
    (out_dir / "accuracy_report.md").write_text(text)
    return text
