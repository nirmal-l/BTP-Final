"""Publication figures (vector PDF + 600-dpi PNG, 85 mm / 174 mm widths, 7-9 pt text).

  fig1  pipeline schematic
  fig2  corpus profile: articles per period by page type + token-length density (512 line)
  fig3  gold-standard: kappa & macro-F1 with bootstrap 95% CIs + row-normalised confusion
        matrices                                           (needs hand labels)
  fig4  label shifts VADER -> Transformer -> LLM (alluvial) + agreement / F1 by length
  fig5  front vs editorial (diverging stacked bars) + topic x method net-sentiment heatmap
  fig6  sentiment over time, one panel per method, shared axes
  fig11 2-D UMAP of article embeddings coloured by method consensus

Methods that are missing (e.g. LLM before a token is set) are left out automatically.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

LABELS = ["negative", "neutral", "positive"]
# Okabe-Ito, fixed per method across every figure
METHOD_COLORS = {"vader": "#E69F00", "transformer": "#009E73", "llm": "#0072B2"}
METHOD_NAMES = {"vader": "VADER", "transformer": "Transformer", "llm": "Claude Haiku"}
LABEL_COLORS = {"negative": "#D55E00", "neutral": "#BBBBBB", "positive": "#56B4E9"}
METHOD_COLS = {"vader": ("vader_label", "vader_compound"),
               "transformer": ("sentiment_label", "sentiment_score"),
               "llm": ("llm_label", "llm_score")}
MM = 1 / 25.4
SINGLE, DOUBLE = 85 * MM, 174 * MM

plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8.5, "axes.labelsize": 7.5, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "pdf.fonttype": 42, "svg.fonttype": "none",
    "font.family": "sans-serif", "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def methods(df):
    return [m for m in ("vader", "transformer", "llm")
            if METHOD_COLS[m][0] in df and df[METHOD_COLS[m][0]].notna().sum() > 0]


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def _panel(ax, letter):
    ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="bottom", ha="left")


def net(labels: pd.Series) -> float:
    """Net sentiment = % positive - % negative (comparable across methods)."""
    labels = labels.dropna()
    return ((labels == "positive").mean() - (labels == "negative").mean()) if len(labels) else np.nan


# --------------------------------------------------------------------------- fig 1
def fig1_pipeline(out):
    fig, ax = plt.subplots(figsize=(DOUBLE, 62 * MM))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#F2F2F2", ec="#555555"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=5.6,
                linespacing=1.25)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                     lw=0.7, color="#555555"))

    y = 27
    steps = [("Page PDFs\n/ scans", 1), ("Preprocess\ndeskew,\nthreshold,\nphoto removal", 15),
             ("Segment\nlines → blocks\n→ articles", 29), ("OCR per\narticle +\ncleaning\n(≥ 50 words)", 43)]
    for t, x in steps:
        box(x, y - 1, 11, 11, t)
    for (_, x0), (_, x1) in zip(steps, steps[1:]):
        arrow(x0 + 11.5, y + 4.5, x1 - 0.5, y + 4.5)
    # branches
    mx = [("VADER\nlexicon", 62, METHOD_COLORS["vader"]),
          ("Transformer\nRoBERTa,\nsentence avg.", 62, METHOD_COLORS["transformer"]),
          ("LLM\nprompted label\n+ score", 62, METHOD_COLORS["llm"])]
    ys = [31.5, 22.5, 13.5]
    for (t, x, c), yy in zip(mx, ys):
        box(x, yy, 14, 7, t, fc="white", ec=c)
        arrow(55, y + 4.5, x - 0.5, yy + 3.5)
    box(62, 3, 14, 7, "BERTopic\nMiniLM + UMAP", fc="white", ec="#777777")
    arrow(55, y + 4.5, 61.5, 6.5)
    box(83, 16, 15, 15, "Join &\nevaluate\ntopic · section\n· time\nvs gold labels:\nκ, macro-F1,\nMcNemar")
    for yy in ys + [3]:
        arrow(76.5, yy + 3.5, 82.5, 23.5)
    _save(fig, out, "fig1_pipeline")


# --------------------------------------------------------------------------- fig 2
def _token_lengths(df):
    try:
        from transformers import AutoTokenizer
        from . import config
        tok = AutoTokenizer.from_pretrained(config.SENTIMENT_MODEL)
        return np.array([len(tok(t, add_special_tokens=True)["input_ids"]) for t in df["text"]])
    except Exception:
        return (df["n_words"] * 1.3).to_numpy()      # rough words -> tokens


def _period(df):
    d = pd.to_datetime(df["date"])
    if d.dt.to_period("M").nunique() >= 3:
        return d.dt.to_period("M").astype(str), "month"
    return d.dt.strftime("%b %d"), "day"


def _short_labels(order, unit):
    """'2025-01' -> 'Jan' (or 'Jan 25' when the data spans several years)."""
    if unit != "month":
        return list(order)
    years = {pd.Period(s).year for s in order}
    fmt = "%b" if len(years) == 1 else "%b %y"
    return [pd.Period(s).strftime(fmt) for s in order]


def fig2_corpus(df, out):
    df = df.copy()
    df["tokens"] = _token_lengths(df)
    per, unit = _period(df)
    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 58 * MM), gridspec_kw={"wspace": 0.35})
    ct = pd.crosstab(per, df["page_type"]).reindex(columns=[c for c in
                                                            ["front", "editorial", "other"]
                                                            if c in df["page_type"].unique()])
    order = sorted(ct.index, key=lambda s: pd.to_datetime(s, format="%b %d", errors="coerce")
                   if unit == "day" else pd.Period(s))
    ct = ct.loc[order]
    colors = {"front": "#0072B2", "editorial": "#E69F00", "other": "#BBBBBB"}
    bottom = np.zeros(len(ct))
    for c in ct.columns:
        a.bar(range(len(ct)), ct[c], bottom=bottom, color=colors[c], label=c,
              edgecolor="white", linewidth=0.6, width=0.7)
        bottom += ct[c].values
    a.set_xticks(range(len(ct)), _short_labels(ct.index, unit),
                 rotation=90 if len(ct) > 6 else 0, fontsize=6.5)
    a.set_ylabel("articles")
    a.set_title(f"Articles per {unit}")
    a.set_ylim(0, bottom.max() * 1.22)          # head-room so the legend never covers bars
    a.legend(frameon=False, ncol=len(ct.columns), loc="upper left", fontsize=6.5)
    _panel(a, "a")

    from scipy.stats import gaussian_kde
    t = df["tokens"].astype(float)
    xs = np.linspace(0, max(t.max() * 1.05, 600), 400)
    for c in ct.columns:
        v = t[df["page_type"] == c]
        if len(v) > 2:
            b.plot(xs, gaussian_kde(v)(xs), color=colors[c], lw=1.2, label=f"{c} (n={len(v)})")
    b.axvline(512, ls="--", color="#333333", lw=0.8)
    share = (t > 512).mean()
    b.text(520, b.get_ylim()[1] * 0.92, f"512 tokens\n{share:.0%} of articles longer",
           fontsize=6.5, va="top")
    b.set_xlabel("article length (RoBERTa tokens)")
    b.set_ylabel("density")
    b.set_yticks([])
    b.set_title("Article length")
    b.legend(frameon=False, loc="center right")
    _panel(b, "b")
    _save(fig, out, "fig2_corpus_profile")
    return df["tokens"]


# --------------------------------------------------------------------------- fig 3
def _boot(y, p, fn, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        vals.append(fn(y[i], p[i]))
    return np.nanpercentile(vals, [2.5, 97.5])


def fig3_gold(m, out):
    meths = methods(m)
    y = m["my label"].to_numpy()
    kap = lambda a, b: cohen_kappa_score(a, b, labels=LABELS)
    mf1 = lambda a, b: f1_score(a, b, labels=LABELS, average="macro", zero_division=0)
    stats = []
    for k in meths:
        p = m[METHOD_COLS[k][0]].fillna("missing").to_numpy()
        stats.append((k, kap(y, p), _boot(y, p, kap), mf1(y, p), _boot(y, p, mf1)))
    fig = plt.figure(figsize=(DOUBLE, 64 * MM))
    gs = fig.add_gridspec(1, 1 + len(meths), width_ratios=[1.3] + [1] * len(meths), wspace=0.45)
    a = fig.add_subplot(gs[0])
    for i, (k, kv, kci, fv, fci) in enumerate(stats):
        c = METHOD_COLORS[k]
        a.errorbar(kv, i + 0.15, xerr=[[kv - kci[0]], [kci[1] - kv]], fmt="o", color=c, ms=4,
                   capsize=2, lw=0.9)
        a.errorbar(fv, i - 0.15, xerr=[[fv - fci[0]], [fci[1] - fv]], fmt="s", color=c, ms=4,
                   capsize=2, lw=0.9, mfc="white")
    a.set_yticks(range(len(stats)), [METHOD_NAMES[s[0]] for s in stats])
    a.axvline(0, color="#999999", lw=0.6)
    a.set_xlim(-0.2, 1)
    a.set_xlabel("score (95% bootstrap CI)")
    a.plot([], [], "o", color="#555555", label="Cohen's κ")
    a.plot([], [], "s", color="#555555", mfc="white", label="macro-F1")
    a.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2,
             handletextpad=0.3, columnspacing=1.0)
    a.set_title(f"Agreement with gold (n={len(m)})")
    _panel(a, "a")
    for j, k in enumerate(meths):
        ax = fig.add_subplot(gs[j + 1])
        p = m[METHOD_COLS[k][0]].fillna("missing")
        cm = confusion_matrix(y, p, labels=LABELS).astype(float)
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(cmn, cmap="cividis", vmin=0, vmax=1)
        for r in range(3):
            for c in range(3):
                ax.text(c, r, f"{cmn[r, c]:.2f}\n({int(cm[r, c])})", ha="center", va="center",
                        fontsize=6, color="white" if cmn[r, c] < 0.6 else "black")
        ax.set_xticks(range(3), ["neg", "neu", "pos"])
        ax.set_yticks(range(3), ["neg", "neu", "pos"] if j == 0 else [])
        ax.set_xlabel("predicted")
        if j == 0:
            ax.set_ylabel("gold")
        ax.set_title(METHOD_NAMES[k], color=METHOD_COLORS[k])
        ax.spines[:].set_visible(False)
        if j == 0:
            _panel(ax, "b")
    _save(fig, out, "fig3_gold_standard")


# --------------------------------------------------------------------------- fig 4
def _alluvial(ax, df, cols, names):
    n = len(df)
    xs = np.arange(len(cols)) * 3.0
    w = 0.35
    gap = 0.03
    ypos = {}
    for x, c in zip(xs, cols):
        y = 0
        for lab in LABELS:
            h = (df[c] == lab).sum() / n
            ypos[(c, lab)] = [y, y + h]
            ax.add_patch(plt.Rectangle((x - w / 2, y), w, h, color=LABEL_COLORS[lab], lw=0))
            if h > 0.04:
                ax.text(x, y + h / 2, f"{h:.0%}", ha="center", va="center", fontsize=6)
            y += h + gap
    for (c0, x0), (c1, x1) in zip(zip(cols, xs), zip(cols[1:], xs[1:])):
        out_off = {lab: ypos[(c0, lab)][0] for lab in LABELS}
        in_off = {lab: ypos[(c1, lab)][0] for lab in LABELS}
        for l0 in LABELS:
            for l1 in LABELS:
                h = ((df[c0] == l0) & (df[c1] == l1)).sum() / n
                if h == 0:
                    continue
                ya, yb = out_off[l0], in_off[l1]
                t = np.linspace(0, 1, 30)
                s = 3 * t ** 2 - 2 * t ** 3
                xa = x0 + w / 2 + (x1 - w / 2 - x0 - w / 2) * t
                top = ya + h + (yb - ya) * s
                bot = ya + (yb - ya) * s
                ax.add_patch(Polygon(np.r_[np.c_[xa, bot], np.c_[xa[::-1], top[::-1]]],
                                     color=LABEL_COLORS[l0], alpha=0.45, lw=0))
                out_off[l0] += h
                in_off[l1] += h
    ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
    ax.set_ylim(-0.02, 1 + 2 * gap + 0.02)
    ax.set_xticks(xs, names)
    ax.set_yticks([])
    ax.spines[["left"]].set_visible(False)
    for lab in LABELS:
        ax.bar(0, 0, color=LABEL_COLORS[lab], label=lab)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))


def fig4_shifts(df, tokens, out, gold=None):
    meths = methods(df)
    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 66 * MM),
                               gridspec_kw={"wspace": 0.25, "width_ratios": [1, 0.8]})
    d = df.dropna(subset=[METHOD_COLS[k][0] for k in meths])
    _alluvial(a, d, [METHOD_COLS[k][0] for k in meths], [METHOD_NAMES[k] for k in meths])
    a.set_title(f"Label flows between methods (n={len(d)})")
    _panel(a, "a")

    t = pd.Series(np.asarray(tokens, float), index=df.index)
    edges = [0, 256, 512, 1024, np.inf]
    bins = pd.cut(t, edges, right=False)
    centers = t.groupby(bins, observed=False).median()
    for x, y_ in combinations(meths, 2):
        lx, ly = METHOD_COLS[x][0], METHOD_COLS[y_][0]
        ag = (df[lx] == df[ly]).groupby(bins, observed=False).mean()
        b.plot(centers.values, ag.values, marker="o", ms=3, lw=1,
               color=METHOD_COLORS[y_], ls="-" if x == "vader" else ":",
               label=f"{METHOD_NAMES[x]} = {METHOD_NAMES[y_]}")
    if gold is not None and len(gold) >= 20:
        gt = pd.Series(np.asarray(tokens, float), index=df.index).loc[gold.index]
        gb = pd.cut(gt, edges, right=False)
        for k in meths:
            f = gold.groupby(gb, observed=False).apply(
                lambda g: f1_score(g["my label"], g[METHOD_COLS[k][0]].fillna("x"),
                                   labels=LABELS, average="macro", zero_division=0)
                if len(g) >= 5 else np.nan)
            b.plot(centers.values, f.values, marker="s", ms=3, lw=1.4, color=METHOD_COLORS[k],
                   label=f"{METHOD_NAMES[k]} macro-F1 vs gold")
    b.axvline(512, ls="--", color="#333333", lw=0.8)
    b.text(530, 0.98, "512-token\nlimit", fontsize=6.5, va="top")
    b.set_xscale("log")
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
    b.xaxis.set_major_locator(FixedLocator([100, 200, 500, 1000, 2000]))
    b.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    b.xaxis.set_minor_locator(NullLocator())
    b.xaxis.set_minor_formatter(NullFormatter())
    b.set_ylim(0, 1)
    b.set_xlabel("article length (tokens, bin medians)")
    b.set_ylabel("share agreeing / macro-F1")
    b.set_title("Effect of article length")
    b.legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    counts = bins.value_counts(sort=False)
    b.text(0.99, 0.02, "n per bin: " + ", ".join(str(int(c)) for c in counts), fontsize=6,
           ha="right", transform=b.transAxes)
    _panel(b, "b")
    _save(fig, out, "fig4_label_shifts_length")


# --------------------------------------------------------------------------- fig 5
def fig5_sections_topics(df, out):
    meths = methods(df)
    secs = [s for s in ("front", "editorial") if s in df["page_type"].unique()]
    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 78 * MM),
                               gridspec_kw={"wspace": 1.05, "width_ratios": [1, 0.95]})
    rows = []
    for k in meths:
        for s in secs:
            lab = df.loc[df["page_type"] == s, METHOD_COLS[k][0]].dropna()
            rows.append((f"{METHOD_NAMES[k]} · {s}", k,
                         *(lab.value_counts(normalize=True).reindex(LABELS).fillna(0))))
    for i, (name, k, ng, nu, po) in enumerate(rows):
        a.barh(i, -ng, color=LABEL_COLORS["negative"], height=0.65)
        a.barh(i, -nu / 2, color=LABEL_COLORS["neutral"], height=0.65)
        a.barh(i, nu / 2, color=LABEL_COLORS["neutral"], height=0.65)
        a.barh(i, po, left=nu / 2, color=LABEL_COLORS["positive"], height=0.65)
        a.barh(i, -ng, left=-nu / 2, color=LABEL_COLORS["negative"], height=0.65)
    a.set_yticks(range(len(rows)), [r[0] for r in rows])
    for t, r in zip(a.get_yticklabels(), rows):
        t.set_color(METHOD_COLORS[r[1]])
    a.axvline(0, color="#333333", lw=0.6)
    a.set_xlim(-1, 1)
    a.set_xticks([-1, -0.5, 0, 0.5, 1], ["100%", "50%", "0", "50%", "100%"])
    a.set_xlabel("← negative   share of articles   positive →")
    a.invert_yaxis()
    a.set_title("Front page vs editorial")
    for lab in LABELS:
        a.barh(0, 0, color=LABEL_COLORS[lab], label=lab)
    a.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    _panel(a, "a")

    topics = df["topic_label"].value_counts().index.tolist()
    mat = np.array([[net(df.loc[df["topic_label"] == t, METHOD_COLS[k][0]]) for k in meths]
                    for t in topics])
    lim = max(0.1, np.nanmax(np.abs(mat)))
    im = b.imshow(mat, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(len(topics)):
        for j in range(len(meths)):
            v = mat[i, j]
            b.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6,
                   color="white" if abs(v) > 0.6 * lim else "black")
    n = df["topic_label"].value_counts()
    short = {t: " / ".join(str(t).split(" / ")[:3]) for t in topics}
    b.set_yticks(range(len(topics)), [f"{short[t]} ({n[t]})" for t in topics], fontsize=5.8)
    b.set_xticks(range(len(meths)), [METHOD_NAMES[k] for k in meths], rotation=30,
                 ha="right", rotation_mode="anchor")
    b.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=b, shrink=0.8, pad=0.02)
    cb.set_label("net sentiment (%pos − %neg)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    b.set_title("BERTopic topic × method")
    _panel(b, "b")
    _save(fig, out, "fig5_sections_topics")


# --------------------------------------------------------------------------- fig 6
def fig6_time(df, out):
    meths = methods(df)
    per, unit = _period(df)
    d = df.assign(period=per)
    order = sorted(d["period"].unique(), key=lambda s: pd.to_datetime(s, format="%b %d")
                   if unit == "day" else pd.Period(s))
    fig, axes = plt.subplots(1, len(meths), figsize=(DOUBLE, 55 * MM), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    axes = np.atleast_1d(axes)
    styles = {"front": "-", "editorial": "--", "other": ":"}
    for ax, k in zip(axes, meths):
        for s in [x for x in ("front", "editorial") if x in d["page_type"].unique()]:
            ser = d[d["page_type"] == s].groupby("period")[METHOD_COLS[k][0]].apply(net)
            ser = ser.reindex(order)
            ax.plot(range(len(order)), ser.values, ls=styles[s], marker="o", ms=2.5, lw=1.1,
                    color=METHOD_COLORS[k], label=s)
        allv = d.groupby("period")[METHOD_COLS[k][0]].apply(net).reindex(order)
        ax.plot(range(len(order)), allv.values, color="#555555", lw=0.7, alpha=0.7,
                label="all")
        ax.axhline(0, color="#999999", lw=0.5)
        ax.set_xticks(range(len(order)), _short_labels(order, unit), rotation=90,
                      fontsize=6.5)
        ax.set_title(METHOD_NAMES[k], color=METHOD_COLORS[k])
        ax.legend(frameon=False, fontsize=6)
    axes[0].set_ylabel("net sentiment (%pos − %neg)")
    fig.suptitle(f"Sentiment by {unit}", x=0.01, ha="left", fontsize=8.5, fontweight="bold",
                 y=1.02)
    _save(fig, out, "fig6_time_series")


# --------------------------------------------------------------------------- fig 11
def fig11_umap(df, out):
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from . import config
    meths = methods(df)
    emb = SentenceTransformer(config.EMBEDDING_MODEL).encode(df["text"].tolist(),
                                                            show_progress_bar=False)
    xy = UMAP(n_neighbors=min(15, len(df) - 1), min_dist=0.1, metric="cosine",
              random_state=config.RANDOM_SEED).fit_transform(emb)
    labs = df[[METHOD_COLS[k][0] for k in meths]]
    nuniq = labs.nunique(axis=1)
    if len(meths) >= 3:
        cat = nuniq.map({1: "all agree", 2: "2 of 3 agree", 3: "all differ"})
        palette = {"all agree": "#009E73", "2 of 3 agree": "#E69F00", "all differ": "#D55E00"}
    else:
        cat = nuniq.map({1: "methods agree", 2: "methods disagree"})
        palette = {"methods agree": "#009E73", "methods disagree": "#D55E00"}
    fig, ax = plt.subplots(figsize=(SINGLE * 1.35, SINGLE * 1.15))
    for c, col in palette.items():
        s = cat == c
        ax.scatter(xy[s, 0], xy[s, 1], s=14, color=col, edgecolor="white", lw=0.4,
                   label=f"{c} ({s.sum()})")
    for t, g in df.groupby("topic_label"):
        cx, cy = xy[g.index.map(df.index.get_loc)].mean(axis=0)
        ax.text(cx, cy, t.split(" / ")[0] + " / " + t.split(" / ")[1] if " / " in t else t,
                fontsize=6, ha="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Topic map: where the methods disagree")
    ax.legend(frameon=False, fontsize=6, loc="best")
    _save(fig, out, "fig11_umap_disagreement")


# --------------------------------------------------------------------------- driver
def _gold(df: pd.DataFrame, label_xlsx: Path | None):
    if label_xlsx is None or not Path(label_xlsx).exists():
        return None
    lab = pd.read_excel(label_xlsx, sheet_name="label_me")
    lab["my label"] = lab["my label"].astype(str).str.strip().str.lower()
    lab = lab[lab["my label"].isin(LABELS)][["article_id", "my label"]]
    if not len(lab):
        return None
    g = df.reset_index().merge(lab, on="article_id").set_index("index")
    return g if len(g) else None


def make_all(articles_csv: Path, label_xlsx: Path, out: Path,
             gold_articles_csv: Path | None = None) -> list[str]:
    """gold_articles_csv: articles of another run that the hand labels belong to (e.g. the
    first-week PDF run).  Then fig3 uses those; fig4's gold overlay needs the same corpus."""
    df = pd.read_csv(articles_csv).reset_index(drop=True)
    gold = _gold(df, label_xlsx)
    gold3 = gold
    if gold3 is None and gold_articles_csv is not None and Path(gold_articles_csv).exists():
        gold3 = _gold(pd.read_csv(gold_articles_csv).reset_index(drop=True), label_xlsx)
    done = []
    fig1_pipeline(out); done.append("fig1")
    tokens = fig2_corpus(df, out); done.append("fig2")
    if gold3 is not None and len(gold3) >= 10:
        fig3_gold(gold3, out); done.append("fig3")
    fig4_shifts(df, tokens, out, gold); done.append("fig4")
    fig5_sections_topics(df, out); done.append("fig5")
    fig6_time(df, out); done.append("fig6")
    fig11_umap(df, out); done.append("fig11")
    return done
