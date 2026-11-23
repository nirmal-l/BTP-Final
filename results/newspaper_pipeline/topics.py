"""Topic modelling: BERTopic when there are enough documents, LDA otherwise."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

from . import config

STOP = sorted(set(ENGLISH_STOP_WORDS) | config.NEWS_STOPWORDS)
TOKEN = r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b"


def _label(words, n=4):
    return " / ".join(words[:n])


def _vectorizer(n_docs, ngram=(1, 1), for_bertopic=False):
    if for_bertopic:   # BERTopic fits this on one merged document per topic
        return CountVectorizer(stop_words=STOP, token_pattern=TOKEN, lowercase=True,
                               ngram_range=ngram)
    return CountVectorizer(stop_words=STOP, token_pattern=TOKEN, lowercase=True,
                           min_df=2 if n_docs >= 20 else 1, max_df=0.6, ngram_range=ngram)


# --------------------------------------------------------------------------- LDA
def _umass(top_idx, X):
    """UMass coherence of one topic's top words (higher = more coherent)."""
    Xb = (X > 0).astype(np.int32)
    score, pairs = 0.0, 0
    for i in range(1, len(top_idx)):
        for j in range(i):
            wi, wj = top_idx[i], top_idx[j]
            co = Xb[:, wi].multiply(Xb[:, wj]).sum()
            dj = Xb[:, wj].sum()
            score += np.log((co + 1) / max(dj, 1))
            pairs += 1
    return score / max(pairs, 1)


def run_lda(docs: list[str]) -> dict:
    from sklearn.decomposition import LatentDirichletAllocation
    vec = _vectorizer(len(docs))
    X = vec.fit_transform(docs)
    vocab = np.array(vec.get_feature_names_out())
    best = None
    tried = []
    for k in config.LDA_K_RANGE:
        if k >= len(docs):
            break
        lda = LatentDirichletAllocation(n_components=k, random_state=config.RANDOM_SEED,
                                        learning_method="batch", max_iter=200,
                                        doc_topic_prior=0.1, topic_word_prior=0.01)
        theta = lda.fit_transform(X)
        tops = [np.argsort(-comp)[:10] for comp in lda.components_]
        coh = float(np.mean([_umass(t, X.tocsc()) for t in tops]))
        tried.append({"k": k, "umass_coherence": round(coh, 4)})
        if best is None or coh > best[0]:
            best = (coh, k, lda, theta, tops)
    coh, k, lda, theta, tops = best
    topic_words = {t: list(vocab[idx]) for t, idx in enumerate(tops)}
    assign = theta.argmax(axis=1)
    return {
        "method": "LDA",
        "n_topics": k,
        "selection": tried,
        "doc_topic": assign.tolist(),
        "doc_topic_prob": theta.max(axis=1).round(4).tolist(),
        "topic_words": topic_words,
        "topic_labels": {t: _label(w) for t, w in topic_words.items()},
    }


# --------------------------------------------------------------------------- BERTopic
def run_bertopic(docs: list[str]) -> dict:
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from umap import UMAP

    n = len(docs)
    emb_model = SentenceTransformer(config.EMBEDDING_MODEL)
    emb = emb_model.encode(docs, show_progress_bar=False)
    umap = UMAP(n_neighbors=max(2, min(15, n - 1)), n_components=5, min_dist=0.0,
                metric="cosine", random_state=config.RANDOM_SEED)
    cluster_note = "HDBSCAN"
    if n >= config.BERTOPIC_HDBSCAN_MIN_DOCS:
        # n // 50 with min_samples=2 merged a 5.6k-article year into 2 topics; n // 110
        # (about 50 for a year) with min_samples=10 gives ~18 distinct topics, largest ~8%
        mcs = max(15, n // 110)
        cluster_note = f"HDBSCAN (min_cluster_size={mcs}, min_samples=10)"
        cluster = HDBSCAN(min_cluster_size=mcs, min_samples=10,
                          metric="euclidean", cluster_selection_method="eom",
                          prediction_data=True)
    else:
        # HDBSCAN needs dense regions; with a few hundred news articles it tends
        # to lump almost everything into one topic.  k-means on the same UMAP
        # space, with k chosen by silhouette, gives usable topics instead.
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        red = UMAP(n_neighbors=max(2, min(15, n - 1)), n_components=5, min_dist=0.0,
                   metric="cosine", random_state=config.RANDOM_SEED).fit_transform(emb)
        scores = {}
        for k in config.BERTOPIC_K_RANGE:
            if k >= n:
                break
            lab = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_SEED).fit_predict(red)
            scores[k] = silhouette_score(red, lab)
        k_best = max(scores, key=scores.get)
        cluster = KMeans(n_clusters=k_best, n_init=10, random_state=config.RANDOM_SEED)
        cluster_note = f"k-means (k={k_best}, silhouette={scores[k_best]:.3f})"
    model = BERTopic(embedding_model=emb_model, umap_model=umap, hdbscan_model=cluster,
                     vectorizer_model=_vectorizer(n, ngram=(1, 2), for_bertopic=True),
                     calculate_probabilities=False, verbose=False)
    topics, _ = model.fit_transform(docs, emb)
    n_outliers_before = int(sum(t == -1 for t in topics))
    if n_outliers_before and len(set(topics)) > 1:
        topics = model.reduce_outliers(docs, topics, strategy="embeddings", embeddings=emb)
        model.update_topics(docs, topics=topics,
                            vectorizer_model=_vectorizer(n, ngram=(1, 2), for_bertopic=True))
    topic_words = {}
    for t in sorted(set(topics)):
        words = model.get_topic(t) or []
        topic_words[int(t)] = [w for w, _ in words[:10]]
    # similarity of each doc to its topic centroid as a confidence proxy
    topics = np.array(topics)
    probs = []
    for i, t in enumerate(topics):
        members = emb[topics == t]
        c = members.mean(axis=0)
        probs.append(float(np.dot(emb[i], c) / (np.linalg.norm(emb[i]) * np.linalg.norm(c) + 1e-9)))
    return {
        "method": "BERTopic",
        "clustering": cluster_note,
        "n_topics": len(topic_words),
        "outliers_reassigned": n_outliers_before,
        "doc_topic": [int(t) for t in topics],
        "doc_topic_prob": [round(p, 4) for p in probs],
        "topic_words": topic_words,
        "topic_labels": {t: _label(w) for t, w in topic_words.items()},
    }


def run(docs: list[str], method: str = "auto") -> dict:
    if method == "auto":
        method = "bertopic" if len(docs) >= config.BERTOPIC_MIN_DOCS else "lda"
    return run_bertopic(docs) if method == "bertopic" else run_lda(docs)
