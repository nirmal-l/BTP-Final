"""BERTopic-style topic charts, drawn as static images from the saved topic assignment.

  topics/topic_map.png          intertopic distance map (bubble = topic, size = articles,
                                colour = Claude Haiku net sentiment)
  topics/topic_barchart.png     top words per topic with their c-TF-IDF weight
  topics/topic_heatmap.png      topic-topic similarity (cosine of topic centroids)
  topics/topic_hierarchy.png    hierarchical clustering of topics (dendrogram)
  topics/topics_over_time.png   share of each month's articles per topic
  topics/topic_sentiment_by_month.png  net sentiment per topic per month (Claude Haiku)

The topics themselves are NOT re-fitted: articles.csv already holds each article's
topic.  Word weights are recomputed with BERTopic's own class-based TF-IDF on the same
vectorizer, and topic positions from the mean MiniLM embedding of each topic's articles
(cached in output/embeddings.npz).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .summarize import GRID, INK, INK2, MUTED, NEG, POS, SURFACE, _style

DPI = 150


def short(label: str, k: int = 2) -> str:
    return " / ".join(str(label).split(" / ")[:k])


def embeddings(df: pd.DataFrame, cache: Path) -> np.ndarray:
    ids = np.array(df["article_id"].astype(str).tolist(), dtype=str)
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if len(z["ids"]) == len(ids) and (z["ids"] == ids).all():
            return z["emb"]
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer(config.EMBEDDING_MODEL).encode(df["text"].tolist(),
                                                             show_progress_bar=False)
    np.savez_compressed(cache, ids=ids, emb=emb)
    return emb


def ctfidf(df: pd.DataFrame):
    """Class-based TF-IDF exactly as BERTopic computes it (one merged document per topic)."""
    from bertopic.vectorizers import ClassTfidfTransformer
    from .topics import _vectorizer
    topics = sorted(df["topic"].unique())
    docs = [" ".join(df.loc[df["topic"] == t, "text"]) for t in topics]
    vec = _vectorizer(len(df), ngram=(1, 2), for_bertopic=True)
    X = vec.fit_transform(docs)
    W = ClassTfidfTransformer().fit_transform(X)
    return topics, np.asarray(vec.get_feature_names_out()), W


def _net(s):
    s = s.dropna()
    return (s.eq("positive").mean() - s.eq("negative").mean()) if len(s) else np.nan


def run(df: pd.DataFrame, out_dir: Path, cache: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.reset_index(drop=True).copy()
    df["month"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m")
    lab = df.groupby("topic")["topic_label"].first()
    n = df["topic"].value_counts()
    has_llm = "llm_label" in df and df["llm_label"].notna().mean() > 0.5
    sent = df.groupby("topic")["llm_label" if has_llm else "sentiment_label"].apply(_net)
    meth = "Claude Haiku" if has_llm else "Transformer"
    made = []

    emb = embeddings(df, cache)
    topics, words, W = ctfidf(df)
    cent = np.vstack([emb[df["topic"].to_numpy() == t].mean(0) for t in topics])
    cent_n = cent / np.linalg.norm(cent, axis=1, keepdims=True)
    sim = cent_n @ cent_n.T

    # hierarchy (average linkage on cosine distance) - also gives the heatmap order
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
    dist = np.clip(1 - sim, 0, None)
    np.fill_diagonal(dist, 0)
    Z = linkage(squareform(dist, checks=False), method="average")

    # 1 intertopic distance map ---------------------------------------------------------
    from sklearn.manifold import MDS
    xy = MDS(n_components=2, dissimilarity="precomputed", random_state=config.RANDOM_SEED,
             n_init=8, normalized_stress="auto").fit_transform(dist)
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=SURFACE)
    vmax = max(0.3, np.nanmax(np.abs(sent.reindex(topics))))
    sizes = np.array([n[t] for t in topics], float)
    sc = ax.scatter(xy[:, 0], xy[:, 1], s=sizes / sizes.max() * 4200 + 120,
                    c=[sent[t] for t in topics], cmap="RdBu", vmin=-vmax, vmax=vmax,
                    alpha=0.85, edgecolor="white", linewidth=1.5)
    for (x, y), t in zip(xy, topics):
        ax.annotate(f"{short(lab[t])}\n({n[t]})", (x, y), ha="center", va="center",
                    fontsize=8, color=INK, fontweight="bold")
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label(f"net sentiment, {meth} (% pos − % neg)", fontsize=9, color=INK2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0.12)
    _style(ax, "Intertopic distance map",
           subtitle="bubble = topic · area = number of articles · close bubbles = similar "
                    "content (MDS of topic-centroid cosine distance)")
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "topic_map.png", dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    made.append("topic_map")

    # 2 word bar charts -----------------------------------------------------------------
    order_n = sorted(topics, key=lambda t: -n[t])
    cols = 3
    rows = int(np.ceil(len(order_n) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.15 * rows + 0.8), facecolor=SURFACE)
    axes = np.atleast_1d(axes).ravel()
    for ax, t in zip(axes, order_n):
        i = topics.index(t)
        row = W[i].toarray().ravel()
        top = np.argsort(row)[::-1][:8]
        c = POS if sent[t] > 0.05 else NEG if sent[t] < -0.05 else MUTED
        ax.barh(range(len(top))[::-1], row[top], color=c, height=0.7)
        ax.set_yticks(range(len(top))[::-1], words[top], fontsize=8)
        ax.set_title(f"{short(lab[t], 3)}  (n={n[t]})", loc="left", fontsize=9, color=INK)
        ax.tick_params(axis="x", labelsize=7, colors=INK2)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.set_facecolor(SURFACE)
    for ax in axes[len(order_n):]:
        ax.axis("off")
    fig.suptitle("Topic word scores (c-TF-IDF)", x=0.01, ha="left", fontsize=13, color=INK)
    fig.text(0.01, 0.985 - 0.35 / (2.15 * rows + 0.8),
             f"8 most distinctive words per topic, largest topics first · bar colour = topic's "
             f"{meth} net sentiment (blue positive, red negative, grey ~0)",
             fontsize=9, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_dir / "topic_barchart.png", dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    made.append("topic_barchart")

    # 3 similarity heatmap (dendrogram order) -------------------------------------------
    leaves = dendrogram(Z, no_plot=True)["leaves"]
    S = sim[np.ix_(leaves, leaves)]
    names = [f"{short(lab[topics[i]])} ({n[topics[i]]})" for i in leaves]
    fig, ax = plt.subplots(figsize=(10, 8.5), facecolor=SURFACE)
    off = S[~np.eye(len(S), dtype=bool)]
    im = ax.imshow(S, cmap="Blues", vmin=off.min(), vmax=off.max())
    for i in range(len(S)):
        for j in range(len(S)):
            if i != j:
                ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if S[i, j] > off.min() + 0.7 * (off.max() - off.min())
                        else INK)
    ax.set_xticks(range(len(S)), names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(S)), names, fontsize=8)
    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("cosine similarity of topic centroids", fontsize=9, color=INK2)
    _style(ax, "Topic similarity",
           subtitle="ordered by the topic hierarchy · darker = more similar content")
    fig.tight_layout()
    fig.savefig(out_dir / "topic_heatmap.png", dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    made.append("topic_heatmap")

    # 4 hierarchy -----------------------------------------------------------------------
    from scipy.cluster.hierarchy import set_link_color_palette
    set_link_color_palette(["#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"])
    fig, ax = plt.subplots(figsize=(10, 7.5), facecolor=SURFACE)
    dendrogram(Z, orientation="left", ax=ax, color_threshold=0.7 * Z[:, 2].max(),
               above_threshold_color=MUTED,
               labels=[f"{short(lab[t], 3)} ({n[t]})" for t in topics], leaf_font_size=9)
    ax.set_xlabel("cosine distance at which topics merge (smaller = more alike)", color=INK2,
                  fontsize=9)
    _style(ax, "Topic hierarchy",
           subtitle="average-linkage clustering of topic centroids · branches = groups of "
                    "related topics")
    ax.tick_params(axis="y", colors=INK)
    fig.tight_layout()
    fig.savefig(out_dir / "topic_hierarchy.png", dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    made.append("topic_hierarchy")

    # 5 topics over time ----------------------------------------------------------------
    months = sorted(df["month"].unique())
    mlab = [pd.Period(m).strftime("%b") for m in months]
    share = pd.crosstab(df["topic"], df["month"], normalize="columns").reindex(
        index=order_n, columns=months)
    cnt = pd.crosstab(df["topic"], df["month"]).reindex(index=order_n, columns=months)
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(order_n) + 2), facecolor=SURFACE)
    im = ax.imshow(share.values * 100, cmap="Blues", aspect="auto")
    for i in range(share.shape[0]):
        for j in range(share.shape[1]):
            v = share.values[i, j] * 100
            ax.text(j, i, f"{cnt.values[i, j]}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.6 * np.nanmax(share.values * 100) else INK)
    ax.set_xticks(range(len(months)), mlab)
    ax.set_yticks(range(len(order_n)), [f"{short(lab[t], 3)}" for t in order_n], fontsize=8.5)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
    cb.set_label("% of that month's articles", fontsize=9, color=INK2)
    _style(ax, "Topics over time",
           subtitle="colour = topic's share of the month's articles · number = article count")
    fig.tight_layout()
    fig.savefig(out_dir / "topics_over_time.png", dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    made.append("topics_over_time")

    # 6 topic x month sentiment ---------------------------------------------------------
    col = "llm_label" if has_llm else "sentiment_label"
    g = df.groupby(["topic", "month"])[col]
    netm = g.apply(_net).unstack().reindex(index=order_n, columns=months)
    nm = g.size().unstack().reindex(index=order_n, columns=months).fillna(0)
    netm = netm.where(nm >= 8)
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(order_n) + 2), facecolor=SURFACE)
    lim = np.nanmax(np.abs(netm.values))
    im = ax.imshow(netm.values, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(netm.shape[0]):
        for j in range(netm.shape[1]):
            v = netm.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}".replace("+0.", "+.").replace("-0.", "−."),
                        ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.6 * lim else INK)
    ax.set_xticks(range(len(months)), mlab)
    ax.set_yticks(range(len(order_n)), [f"{short(lab[t], 3)}" for t in order_n], fontsize=8.5)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
    cb.set_label("net sentiment (% pos − % neg)", fontsize=9, color=INK2)
    _style(ax, "Topic sentiment by month",
           subtitle=f"Method: {meth} · blank = fewer than 8 articles that month")
    fig.tight_layout()
    fig.savefig(out_dir / "topic_sentiment_by_month.png", dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    made.append("topic_sentiment_by_month")
    return made
