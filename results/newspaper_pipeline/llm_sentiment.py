"""Sentiment method 3: an instruction-tuned LLM reads the article and labels its tone.

Backends, same prompt and output:
  * "claude" - Claude Haiku 5.5 through the Anthropic API (default; needs
               ANTHROPIC_API_KEY in results/.env).  Only Haiku models are allowed.
  * "hf_api" - Hugging Face Inference Providers (needs HF credits)
  * "local"  - the model runs on this machine with transformers (no key, slower)

Results are cached per (model, article text) in output/llm_cache.jsonl, written
after every article, so an interrupted run - or one that hits the free API
quota - resumes where it stopped.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path

from . import config

log = logging.getLogger("pipeline")

SYSTEM_PROMPT = (
    "You are a careful media analyst. You classify the overall tone of newspaper "
    "articles. You answer with JSON only."
)

USER_TEMPLATE = """Classify the overall sentiment (tone) of this newspaper article.

Definitions:
- positive: the article's overall tone is favourable, hopeful or reports good news / progress.
- negative: the overall tone is critical, alarming, or reports conflict, harm, failure or bad news.
- neutral: mainly factual or balanced reporting with no clear positive or negative slant.
Judge the tone of the text itself, not whether you agree with it. The text comes from OCR
and may contain small errors; ignore them.

Return exactly one JSON object and nothing else:
{{"label": "positive" | "neutral" | "negative", "score": <number from -1.0 (very negative) to 1.0 (very positive)>, "reason": "<at most 20 words>"}}

HEADLINE: {headline}

ARTICLE:
{text}"""


class QuotaExhausted(RuntimeError):
    pass


def load_env_file():
    """Copy KEY=value lines from results/.env into the environment (never printed)."""
    env = Path(__file__).resolve().parent.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            if k.strip() and v and not os.environ.get(k.strip()):
                os.environ[k.strip()] = v


# --------------------------------------------------------------------------- token
def get_token() -> str | None:
    """HF token from the environment, or from results/.env (HF_TOKEN=...)."""
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if tok:
        return tok.strip()
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    return None


# --------------------------------------------------------------------------- parsing
def _truncate(text: str, max_words: int) -> str:
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words]) + " [...]"


def parse_response(raw: str) -> dict:
    m = re.search(r"\{.*?\}", raw or "", re.S)
    if not m:
        raise ValueError(f"no JSON in response: {raw[:120]!r}")
    obj = json.loads(m.group(0))
    label = str(obj.get("label", "")).strip().lower()
    try:
        score = float(obj.get("score"))
    except (TypeError, ValueError):
        score = None
    if score is not None:
        score = max(-1.0, min(1.0, score))
    if label not in ("positive", "neutral", "negative"):
        if score is None:
            raise ValueError(f"bad label {label!r}")
        label = "positive" if score > 0.2 else "negative" if score < -0.2 else "neutral"
    if score is None:
        score = {"positive": 0.5, "neutral": 0.0, "negative": -0.5}[label]
    return {"llm_label": label, "llm_score": round(score, 3),
            "llm_reason": str(obj.get("reason", ""))[:200]}


# --------------------------------------------------------------------------- backends
class _Claude:
    """Claude Haiku via the official Anthropic SDK, with a validated JSON reply."""

    def __init__(self, model: str):
        import anthropic
        from pydantic import BaseModel, Field
        from typing import Literal

        if not model.startswith("claude-haiku"):
            raise RuntimeError(f"Only Claude Haiku is allowed for sentiment; got {model!r}")
        load_env_file()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("No ANTHROPIC_API_KEY found (add it to results/.env)")

        class Verdict(BaseModel):
            label: Literal["positive", "neutral", "negative"]
            score: float = Field(description="-1.0 very negative ... 1.0 very positive")
            reason: str = Field(description="at most 20 words")

        self.anthropic = anthropic
        self.Verdict = Verdict
        self.model = model
        self.client = anthropic.Anthropic(max_retries=5, timeout=120.0)

    def __call__(self, messages) -> str:
        a = self.anthropic
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        convo = [m for m in messages if m["role"] != "system"]
        try:
            resp = self.client.messages.parse(
                model=self.model,
                max_tokens=config.LLM_MAX_NEW_TOKENS,
                system=system,
                output_config={"effort": config.LLM_EFFORT},
                output_format=self.Verdict,
                messages=convo,
            )
        except a.AuthenticationError as e:
            raise QuotaExhausted("Anthropic API key rejected - check ANTHROPIC_API_KEY") from e
        except a.PermissionDeniedError as e:
            raise QuotaExhausted(f"Anthropic API permission denied: {e.message}") from e
        except a.BadRequestError as e:
            if "credit" in str(e.message).lower() or "billing" in str(e.message).lower():
                raise QuotaExhausted("Anthropic credit balance too low") from e
            raise
        if resp.stop_reason == "refusal":
            raise ValueError("model declined this article")
        if resp.stop_reason == "max_tokens" or resp.parsed_output is None:
            raise ValueError(f"no complete answer (stop_reason={resp.stop_reason})")
        return resp.parsed_output.model_dump_json()


class _HFApi:
    def __init__(self, model: str):
        from huggingface_hub import InferenceClient
        token = get_token()
        if not token:
            raise RuntimeError("No Hugging Face token found (set HF_TOKEN or create results/.env)")
        self.model = model
        self.client = InferenceClient(provider=config.LLM_PROVIDER, api_key=token,
                                      timeout=120)

    def __call__(self, messages) -> str:
        from huggingface_hub.errors import HfHubHTTPError
        delay = 5
        for attempt in range(5):
            try:
                out = self.client.chat_completion(messages=messages, model=self.model,
                                                  max_tokens=config.LLM_MAX_NEW_TOKENS,
                                                  temperature=0.0)
                return out.choices[0].message.content
            except HfHubHTTPError as e:
                code = getattr(e.response, "status_code", None)
                if code == 402:
                    raise QuotaExhausted("Hugging Face free credits used up (HTTP 402)") from e
                if code in (429, 500, 502, 503, 504) and attempt < 4:
                    log.info("  HF API %s - retrying in %ds", code, delay)
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise


class _Local:
    def __init__(self, model: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_name = model
        self.tok = AutoTokenizer.from_pretrained(model)
        dev = "mps" if torch.backends.mps.is_available() else \
              "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if dev != "cpu" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=dtype).to(dev).eval()
        self.dev = dev

    def __call__(self, messages) -> str:
        import torch
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True,
                                           return_tensors="pt").to(self.dev)
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=config.LLM_MAX_NEW_TOKENS,
                                      do_sample=False, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


# --------------------------------------------------------------------------- driver
def _key(model, headline, text):
    return hashlib.sha1(f"{model}\n{headline}\n{text}".encode()).hexdigest()[:16]


def score_articles(rows: list[dict], cache_path: Path, backend: str,
                   model: str | None = None) -> dict:
    """Returns {article_id: result dict}.  Never raises for API quota problems:
    articles not scored simply have no entry (and are retried next run)."""
    model = model or {"claude": config.LLM_MODEL_CLAUDE, "hf_api": config.LLM_MODEL_API,
                      "local": config.LLM_MODEL_LOCAL}[backend]
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                cache[rec["key"]] = rec
    todo = [r for r in rows if _key(model, r["headline"], r["text"]) not in cache]
    log.info("LLM sentiment (%s, %s): %d to score, %d cached", backend, model, len(todo),
             len(rows) - len(todo))
    if todo:
        try:
            call = {"claude": _Claude, "hf_api": _HFApi, "local": _Local}[backend](model)
        except Exception as e:
            log.warning("LLM sentiment skipped: %s", e)
            call = None
        if call is not None:
            import threading
            from concurrent.futures import ThreadPoolExecutor
            lock = threading.Lock()
            stop = threading.Event()
            counter = {"n": 0}

            def work(r):
                if stop.is_set():
                    return
                msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_TEMPLATE.format(
                            headline=r["headline"] if isinstance(r["headline"], str)
                            and r["headline"].strip() else "(none)",
                            text=_truncate(r["text"], config.LLM_MAX_WORDS))}]
                try:
                    raw = call(msgs)
                    try:
                        res = parse_response(raw)
                    except ValueError:
                        msgs += [{"role": "assistant", "content": raw},
                                 {"role": "user", "content": "Reply with the JSON object only."}]
                        res = parse_response(call(msgs))
                except QuotaExhausted as e:
                    if not stop.is_set():
                        log.warning("%s - stopping; re-run later to continue (finished "
                                    "articles are cached).", e)
                    stop.set()
                    return
                except Exception as e:  # noqa: BLE001
                    log.warning("  LLM failed on %s: %s", r["article_id"], str(e)[:150])
                    return
                rec = {"key": _key(model, r["headline"], r["text"]), "model": model, **res}
                with lock:
                    cache[rec["key"]] = rec
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    counter["n"] += 1
                    if counter["n"] % 50 == 0:
                        log.info("  LLM scored %d/%d", counter["n"], len(todo))

            workers = config.LLM_WORKERS if backend == "claude" else 1
            with cache_path.open("a") as fh, ThreadPoolExecutor(workers) as pool:
                list(pool.map(work, todo))
    out = {}
    for r in rows:
        rec = cache.get(_key(model, r["headline"], r["text"]))
        if rec:
            out[r["article_id"]] = {k: rec[k] for k in ("llm_label", "llm_score", "llm_reason")} \
                                   | {"llm_model": model}
    return out
