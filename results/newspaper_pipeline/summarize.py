"""Summary tables and charts of sentiment by topic, page type, month and date.

Tables keep every method side by side (transformer columns first for backward
compatibility, then VADER and Claude Haiku).  Charts are drawn from Claude Haiku
when its scores are present - it is the method that agrees best with hand labels -
and fall back to the transformer otherwise.  Every chart names its method.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# diverging blue (positive) / grey (neutral) / red (negative)
POS, NEU, NEG = "#2a78d6", "#c3c2b7", "#e34948"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
PAGE_COLORS = {"front": "#0072B2", "editorial": "#E69F00", "other": "#999999"}
PAGE_STYLES = {"front": "-", "editorial": "--", "other": ":"}


def _style(ax, title, xlabel=None, ylabel=None, subtitle=None):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=22 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9, color=INK2,
                va="bottom")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def _ci(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce").dropna()
    n = len(s)
    if n < 2:
        return np.nan, np.nan
    h = 1.96 * s.std(ddof=1) / np.sqrt(n)
    return s.mean() - h, s.mean() + h


def group_table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    def agg(g):
        n = len(g)
        lo, hi = _ci(g["sentiment_score"])
        llo, lhi = _ci(g["llm_score"])
        llm = g["llm_label"]
        return pd.Series({
            "n_articles": n,
            # transformer
            "mean_score": g["sentiment_score"].mean(),
            "ci95_low": lo, "ci95_high": hi,
            "share_positive": (g["sentiment_label"] == "positive").mean(),
            "share_neutral": (g["sentiment_label"] == "neutral").mean(),
            "share_negative": (g["sentiment_label"] == "negative").mean(),
            "mean_headline_score": g["headline_score"].mean(),
            # VADER
            "mean_vader": g["vader_compound"].mean(),
            # Claude Haiku
            "mean_llm_score": pd.to_numeric(g["llm_score"], errors="coerce").mean(),
            "share_llm_positive": (llm == "positive").mean(),
            "share_llm_negative": (llm == "negative").mean(),
            "label_agreement_with_vader": (g["sentiment_label"] == g["vader_label"]).mean(),
            "llm_ci95_low": llo, "llm_ci95_high": lhi,
            "share_llm_neutral": (llm == "neutral").mean(),
            "llm_net": (llm == "positive").mean() - (llm == "negative").mean(),
            "share_vader_positive": (g["vader_label"] == "positive").mean(),
            "share_vader_negative": (g["vader_label"] == "negative").mean(),
        })
    out = df.groupby(by, dropna=False).apply(agg, include_groups=False).reset_index()
    out["n_articles"] = out["n_articles"].astype(int)
    return out.round(3)


# which method the charts show -------------------------------------------------------
def _chart_method(df):
    if "llm_label" in df and df["llm_label"].notna().mean() > 0.5:
        return {"name": "Claude Haiku", "mean": "mean_llm_score", "lo": "llm_ci95_low",
                "hi": "llm_ci95_high", "pos": "share_llm_positive", "neu": "share_llm_neutral",
                "neg": "share_llm_negative", "score": "llm_score", "label": "llm_label",
                "scale": "Claude Haiku score (−1 very negative … +1 very positive)"}
    return {"name": "Transformer", "mean": "mean_score", "lo": "ci95_low", "hi": "ci95_high",
            "pos": "share_positive", "neu": "share_neutral", "neg": "share_negative",
            "score": "sentiment_score", "label": "sentiment_label",
            "scale": "transformer score, P(pos) − P(neg)"}


def _bar_ci(t, by, M, title, path):
    t = t.sort_values(M["mean"])
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(t) + 1.9), facecolor=SURFACE)
    y = np.arange(len(t))
    ax.barh(y, t[M["mean"]], color=[POS if v > 0 else NEG for v in t[M["mean"]]], height=0.55)
    lo = (t[M["mean"]] - t[M["lo"]]).fillna(0).clip(lower=0)
    hi = (t[M["hi"]] - t[M["mean"]]).fillna(0).clip(lower=0)
    ax.errorbar(t[M["mean"]], y, xerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y, [f"{l}  (n={int(n)})" for l, n in zip(t[by].astype(str), t["n_articles"])])
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    _style(ax, title, xlabel=f"mean {M['scale']}\nwhiskers = 95% confidence interval",
           subtitle=f"Method: {M['name']} · most positive at top")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _stacked(t, by, M, title, path, order=None):
    if order is None:
        t = t.assign(_net=t[M["pos"]] - t[M["neg"]]).sort_values("_net")
    else:
        t = t.set_index(by).loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(t) + 2.3), facecolor=SURFACE)
    y = np.arange(len(t))
    left = np.zeros(len(t))
    for col, c, name in ((M["neg"], NEG, "negative"), (M["neu"], NEU, "neutral"),
                         (M["pos"], POS, "positive")):
        v = t[col].to_numpy(float)
        ax.barh(y, v, left=left, color=c, height=0.6, label=name, edgecolor=SURFACE, linewidth=2)
        for yi, (l, w) in enumerate(zip(left, v)):
            if w >= 0.07:
                ax.text(l + w / 2, yi, f"{w:.0%}", ha="center", va="center", fontsize=7.5,
                        color="white" if c != NEU else INK)
        left += v
    ax.set_yticks(y, [f"{l}  (n={int(n)})" for l, n in zip(t[by].astype(str), t["n_articles"])])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1 if len(t) > 4
                                                                          else -0.35), fontsize=9)
    sub = (f"Method: {M['name']} · " + ("sorted by net sentiment (most positive at top)"
                                        if order is None else "in calendar order"))
    _style(ax, title, xlabel="share of articles", subtitle=sub)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _month_line(df, t, M, path):
    months = sorted(t["month"])
    x = np.arange(len(months))
    labels = [pd.Period(m).strftime("%b") if pd.Period(months[0]).year == pd.Period(months[-1]).year
              else pd.Period(m).strftime("%b %y") for m in months]
    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor=SURFACE)
    tt = t.set_index("month").loc[months]
    ax.fill_between(x, tt[M["lo"]], tt[M["hi"]], color=INK2, alpha=0.12, lw=0,
                    label="all articles, 95% CI")
    ax.plot(x, tt[M["mean"]], color=INK, lw=2, marker="o", ms=4, label="all articles")
    for pt in [p for p in ("front", "editorial") if p in df["page_type"].unique()]:
        s = (pd.to_numeric(df.loc[df["page_type"] == pt, M["score"]], errors="coerce")
             .groupby(df.loc[df["page_type"] == pt, "month"]).mean().reindex(months))
        ax.plot(x, s.values, color=PAGE_COLORS[pt], ls=PAGE_STYLES[pt], lw=1.4, marker="o",
                ms=3, label=pt)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(x, labels)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    _style(ax, "Mean article sentiment by month", ylabel=f"mean {M['scale']}",
           subtitle=f"Method: {M['name']} · calendar order · n per month "
                    f"{int(tt['n_articles'].min())}–{int(tt['n_articles'].max())}")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _date_charts(df, t, M, path_score, path_labels, min_n=12, window=7):
    t = t.copy()
    t["date"] = pd.to_datetime(t["date"])
    t = t.set_index("date").sort_index()
    # rolling 7-day article-weighted mean (sum of scores / number of articles)
    d = df.assign(date=pd.to_datetime(df["date"]),
                  _s=pd.to_numeric(df[M["score"]], errors="coerce"))
    daily = d.groupby("date").agg(n=("_s", "count"), s=("_s", "sum"))
    full = daily.reindex(pd.date_range(daily.index.min(), daily.index.max()), fill_value=0)
    roll = (full["s"].rolling(window, center=True, min_periods=1).sum()
            / full["n"].rolling(window, center=True, min_periods=1).sum())

    fig, ax = plt.subplots(figsize=(11, 4.8), facecolor=SURFACE)
    ax.scatter(t.index, t[M["mean"]], s=np.clip(t["n_articles"], 2, 25) * 1.2, color=MUTED,
               alpha=0.45, lw=0, label="one day (dot size = articles that day)")
    ax.plot(roll.index, roll.values, color=INK, lw=2, label=f"{window}-day rolling mean")
    ax.axhline(0, color=MUTED, lw=1)
    ok = t[t["n_articles"] >= min_n]
    prev = None
    for i, (dt, r) in enumerate(ok.nsmallest(5, M["mean"]).sort_index().iterrows()):
        close = prev is not None and (dt - prev).days < 12      # stagger neighbouring labels
        ax.annotate(dt.strftime("%d %b"), (dt, r[M["mean"]]),
                    xytext=(14 if close else -2, -13 if not close else -2),
                    textcoords="offset points", ha="left" if close else "center", fontsize=8,
                    color=NEG)
        prev = dt
    for dt, r in ok.nlargest(3, M["mean"]).iterrows():
        ax.annotate(dt.strftime("%d %b"), (dt, r[M["mean"]]), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8, color=POS)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    _style(ax, "Daily article sentiment", ylabel=f"mean {M['scale']}",
           subtitle=f"Method: {M['name']} · labelled: 5 most negative and 3 most positive days "
                    f"with ≥ {min_n} articles")
    fig.tight_layout()
    fig.savefig(path_score, dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # rolling label shares
    cnt = pd.crosstab(d["date"], d[M["label"]]).reindex(columns=["negative", "neutral",
                                                                 "positive"], fill_value=0)
    cnt = cnt.reindex(full.index, fill_value=0).rolling(window, center=True, min_periods=1).sum()
    sh = cnt.div(cnt.sum(axis=1).replace(0, np.nan), axis=0)
    fig, ax = plt.subplots(figsize=(11, 4.4), facecolor=SURFACE)
    ax.stackplot(sh.index, sh["negative"], sh["neutral"], sh["positive"],
                 colors=[NEG, NEU, POS], labels=["negative", "neutral", "positive"], lw=0)
    ax.set_ylim(0, 1)
    ax.set_xlim(sh.index.min(), sh.index.max())
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    _style(ax, "Share of articles by sentiment label over the year",
           ylabel="share of articles", subtitle=f"Method: {M['name']} · {window}-day rolling window")
    fig.tight_layout()
    fig.savefig(path_labels, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _scatter(x, y, color, title, xlabel, ylabel, path, sub, by=None):
    fig, ax = plt.subplots(figsize=(6.5, 6.5), facecolor=SURFACE)
    if by is None:
        ax.scatter(x, y, s=9, color=color, alpha=0.35, lw=0)
    else:
        for pt, idx in by.groupby(by).groups.items():
            ax.scatter(x[idx], y[idx], s=9, color=PAGE_COLORS.get(pt, MUTED), alpha=0.35, lw=0,
                       label=pt)
        leg = ax.legend(frameon=False, fontsize=9, markerscale=2.5)
        for h in leg.legend_handles:
            h.set_alpha(1)
    ax.axhline(0, color=GRID)
    ax.axvline(0, color=GRID)
    _style(ax, title, xlabel=xlabel, ylabel=ylabel, subtitle=sub)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def run(df: pd.DataFrame, tables_dir: Path, charts_dir: Path) -> dict:
    tables_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["llm_score"] = pd.to_numeric(df.get("llm_score"), errors="coerce")
    df["month"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m")
    M = _chart_method(df)
    out = {}
    for by, name in (("topic_label", "topic"), ("page_type", "page_type"),
                     ("month", "month"), ("date", "date")):
        t = group_table(df, by)
        t.to_csv(tables_dir / f"sentiment_by_{name}.csv", index=False)
        out[name] = t
    _bar_ci(out["topic"], "topic_label", M, "Mean article sentiment by topic",
            charts_dir / "sentiment_by_topic.png")
    _stacked(out["topic"], "topic_label", M, "Sentiment labels by topic",
             charts_dir / "labels_by_topic.png")
    _bar_ci(out["page_type"], "page_type", M, "Mean article sentiment by page type",
            charts_dir / "sentiment_by_page_type.png")
    _stacked(out["page_type"], "page_type", M, "Sentiment labels by page type",
             charts_dir / "labels_by_page_type.png")
    _month_line(df, out["month"], M, charts_dir / "sentiment_by_month.png")
    months = sorted(out["month"]["month"])
    one_year = len({m[:4] for m in months}) == 1
    lab = {m: pd.Period(m).strftime("%b" if one_year else "%b %Y") for m in months}
    _stacked(out["month"].assign(month=out["month"]["month"].map(lab)), "month", M,
             "Sentiment labels by month", charts_dir / "labels_by_month.png",
             order=[lab[m] for m in months[::-1]])
    _date_charts(df, out["date"], M, charts_dir / "sentiment_by_date.png",
                 charts_dir / "labels_by_date.png")

    # topic x page type (transformer mean, plus Claude Haiku mean when present)
    ct = pd.crosstab(df["topic_label"], df["page_type"], values=df["sentiment_score"],
                     aggfunc="mean").round(3)
    if df["llm_score"].notna().any():
        ct2 = pd.crosstab(df["topic_label"], df["page_type"], values=df["llm_score"],
                          aggfunc="mean").round(3).add_prefix("llm_")
        ct = ct.join(ct2)
    ct.to_csv(tables_dir / "sentiment_topic_x_pagetype.csv")

    # transformer vs VADER
    r = df[["vader_compound", "sentiment_score"]].corr(method="spearman").iloc[0, 1]
    _scatter(df["vader_compound"], df["sentiment_score"], None,
             f"Transformer vs VADER (Spearman ρ = {r:.2f})", "VADER compound (sentence mean)",
             "transformer score, P(pos) − P(neg)", charts_dir / "transformer_vs_vader.png",
             "one dot per article · lower-right = VADER positive but transformer negative",
             by=df["page_type"])
    # headline vs body (transformer: the only method that scores headlines separately)
    h = df.dropna(subset=["headline_score"])
    rh = h[["headline_score", "sentiment_score"]].corr(method="spearman").iloc[0, 1]
    _scatter(h["headline_score"], h["sentiment_score"], "#0072B2",
             f"Headline vs body sentiment (Spearman ρ = {rh:.2f})", "headline score",
             "body score (sentence mean)", charts_dir / "headline_vs_body.png",
             f"Method: Transformer · {len(h)} articles with a detected headline")
    out["spearman_transformer_vader"] = float(r)
    out["spearman_headline_body"] = float(rh)
    out["agreement_transformer_vader"] = float((df["sentiment_label"] == df["vader_label"]).mean())
    out["chart_method"] = M["name"]
    return out
