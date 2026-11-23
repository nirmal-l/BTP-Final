"""Three sentiment measures per article.

1. Transformer, article body: every sentence scored by a 3-class RoBERTa model,
   probabilities averaged over sentences.  score = P(pos) - P(neg), in [-1, 1].
2. Transformer, headline: the same model on the headline alone.
3. VADER (lexicon based), averaged over the same sentences - a cross-check.
"""
from __future__ import annotations

import numpy as np

from . import config
from .textclean import sentences

_pipe = None
_vader = None


def _device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _model():
    global _pipe
    if _pipe is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(config.SENTIMENT_MODEL)
        mdl = AutoModelForSequenceClassification.from_pretrained(config.SENTIMENT_MODEL)
        dev = _device()
        mdl.to(dev).eval()
        labels = [mdl.config.id2label[i].lower() for i in range(mdl.config.num_labels)]
        _pipe = (tok, mdl, dev, labels)
    return _pipe


def transformer_probs(texts: list[str], batch_size=32) -> np.ndarray:
    """Return an (n, 3) array of [neg, neu, pos] probabilities."""
    import torch
    tok, mdl, dev, labels = _model()
    order = [labels.index(k) for k in ("negative", "neutral", "positive")]
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=256,
                  return_tensors="pt").to(dev)
        with torch.no_grad():
            p = torch.softmax(mdl(**enc).logits.float(), dim=-1).cpu().numpy()
        out.append(p[:, order])
    return np.vstack(out) if out else np.zeros((0, 3))


def vader_scores(texts: list[str]) -> np.ndarray:
    global _vader
    if _vader is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader = SentimentIntensityAnalyzer()
    return np.array([_vader.polarity_scores(t)["compound"] for t in texts])


def _label(p):
    return ("negative", "neutral", "positive")[int(np.argmax(p))]


def _vader_label(c):
    if c >= config.VADER_POS:
        return "positive"
    if c <= config.VADER_NEG:
        return "negative"
    return "neutral"


def score_articles(rows: list[dict]) -> list[dict]:
    """rows need 'article_id', 'headline', 'text'. Returns one result dict per row."""
    sent_lists = [sentences(r["text"]) or [r["text"]] for r in rows]
    flat = [s for sl in sent_lists for s in sl]
    probs = transformer_probs(flat)
    vad = vader_scores(flat)
    heads = [r["headline"] for r in rows]
    head_idx = [i for i, h in enumerate(heads) if h and len(h.split()) >= 2]
    head_probs = transformer_probs([heads[i] for i in head_idx])
    head_map = dict(zip(head_idx, head_probs))

    results, k = [], 0
    for i, (r, sl) in enumerate(zip(rows, sent_lists)):
        p = probs[k:k + len(sl)]
        v = vad[k:k + len(sl)]
        k += len(sl)
        mean = p.mean(axis=0)
        sent_scores = p[:, 2] - p[:, 0]
        res = {
            "article_id": r["article_id"],
            "n_sentences": len(sl),
            "p_negative": round(float(mean[0]), 4),
            "p_neutral": round(float(mean[1]), 4),
            "p_positive": round(float(mean[2]), 4),
            "sentiment_label": _label(mean),
            "sentiment_score": round(float(mean[2] - mean[0]), 4),
            "sentiment_score_sd": round(float(sent_scores.std()), 4),
            "share_neg_sentences": round(float(np.mean(np.argmax(p, 1) == 0)), 4),
            "share_pos_sentences": round(float(np.mean(np.argmax(p, 1) == 2)), 4),
            "vader_compound": round(float(v.mean()), 4),
            "vader_label": _vader_label(float(v.mean())),
        }
        if i in head_map:
            hp = head_map[i]
            res.update({
                "headline_p_negative": round(float(hp[0]), 4),
                "headline_p_neutral": round(float(hp[1]), 4),
                "headline_p_positive": round(float(hp[2]), 4),
                "headline_label": _label(hp),
                "headline_score": round(float(hp[2] - hp[0]), 4),
            })
        else:
            res.update({"headline_p_negative": None, "headline_p_neutral": None,
                        "headline_p_positive": None, "headline_label": None,
                        "headline_score": None})
        results.append(res)
    return results
