# Newspaper sentiment & topic pipeline

Turns newspaper page files (PDF e-paper pages or scanned JPG/PNG/TIFF) into
per-article text, sentiment scores, topics and summary tables/charts.
Your input folder is only ever read — nothing in it is changed.

## Setup (once)

```bash
brew install tesseract                     # OCR engine
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The first run downloads two models from Hugging Face (~600 MB):
`cardiffnlp/twitter-roberta-base-sentiment-latest` and
`sentence-transformers/all-MiniLM-L6-v2`.

## Step 0 — download the pages (Indian Express e-paper, Jaipur)

Needs an active e-paper subscription.

```bash
.venv/bin/python -m playwright install chromium        # once
.venv/bin/python scraper/login_once.py                 # once: log in in the window, press Enter
.venv/bin/python scraper/scrape_epaper.py --start 2025-01-01 --end 2025-12-31
```

* For each issue it finds the **front page** (skipping wrap-around ad pages; confirmed
  by 'The Second Page' on the next page) and the **editorial page** ('The Editorial Page'
  header, or the page before 'The Ideas Page'). **Sundays: front page only.**
* Pages are rebuilt from the e-paper reader's image tiles at the best resolution
  offered: 1600 px wide until ~30 Mar 2025, 2400 px after. No CAPTCHA is involved
  (the CAPTCHA only guards the separate PDF-download button, which is not used).
* Output: `raw_scraped/YYYY-MM-DD_front.png`, `..._editorial.png`, `manifest.csv`
  (page number, how each page was identified, resolution, the issue's own list of
  front-page headlines) and `scrape.log`. Finished days are skipped on re-run.
* Session expired? Run `login_once.py` again and re-run the same command.
* ~1 minute per day → a full year takes ~5-6 hours; it can be stopped and resumed.

Quality vs hand-downloaded PDFs (same 7 days): OCR confidence ~88 at 1600 px and ~91.5
at 2400 px, vs ~93 for PDFs; slightly more over-split small items at 1600 px.

Then process them with the pipeline below using `--input raw_scraped`.

## Run

```bash
.venv/bin/python run_pipeline.py --input ../RAW_Images
```

| Option | Effect |
|---|---|
| `--force` | re-process every page (otherwise cached pages are skipped) |
| `--pages-only` | stop after OCR — just the `.txt` files and overlays |
| `--topic-method auto\|bertopic\|lda` | `auto` = BERTopic if ≥ 100 articles, else LDA |
| `--output DIR` | write somewhere other than `./output` |

**Resuming:** each page is saved to `output/pages/<page>/page.json` as soon as it
finishes. If a run stops, run the same command again and finished pages are skipped.
Sentiment scores are cached too (`output/sentiment_cache.csv`).

**Page type** (front / editorial / other) is read from the page itself:
"JOURNALISM OF COURAGE" → front, "THE EDITORIAL PAGE" or "Founded by Ramnath
Goenka" → editorial, anything else → other. Filenames like `2026-11-10_front.jpg`
are also understood. To correct a page, create `page_type_overrides.csv`:

```
page_id,page_type
Jaipur_January_05_2025_page_1,front
```

## Three sentiment methods

| Method | Columns in `articles.csv` |
|---|---|
| Transformer (RoBERTa, per sentence, averaged) | `sentiment_label`, `sentiment_score`, `p_*` |
| VADER (lexicon, per sentence, averaged) | `vader_label`, `vader_compound` |
| LLM (reads the article, returns label + score + reason) | `llm_label`, `llm_score`, `llm_reason` |

**LLM setup (Claude Haiku 5.5 - the default)**
1. In the Anthropic Console create an API key.
2. Put it in `results/.env` on its own line (never commit or share this file):
   `ANTHROPIC_API_KEY=sk-ant-...`
3. Run the pipeline as usual (`--llm claude` is the default). Only `claude-haiku-5-5` is
   used; any non-Haiku model is refused. Each article is sent whole (safety cap 4,000
   words), 4 requests in parallel, low effort, answer validated against a
   label/score/reason schema. Cost is roughly $0.10 per million input tokens - about
   $1-2 for a year of articles.
   Alternatives: `--llm local` (small model on this Mac), `--llm hf_api` (needs HF
   credits), `--llm off`.

Results are cached in `output/llm_cache.jsonl`; if the key is wrong or credit runs out
the run logs it, keeps what was scored, and continues - re-run later to finish.

**Comparison** — every run writes `tables/method_agreement.csv` (pairwise label
agreement, Cohen's kappa, Spearman of scores), `tables/method_label_distribution.csv`
and `charts/methods_by_topic.png`. After hand-labelling, `evaluate_labels.py` reports
accuracy (95% CI), macro-F1 and kappa per method vs a majority baseline, an exact
McNemar test for each pair, and accuracy by page type and topic.

## Checking scraped pages

`scraper/verify_pages.py` OCRs the top of every downloaded editorial page (no network)
and lists files whose title is not "The Editorial Page" / "Wordly Wise" or that are
identical to that day's front page (`raw_scraped/verify.csv`).  From November 2025 the
front page carries a "The Editorial Page: ..." teaser, so the scraper never takes the
front page as the editorial and reads each page's own title strip first.

## Paper figures

```bash
.venv/bin/python make_figures.py
# hand labels that belong to another run (e.g. the first-week PDF run):
.venv/bin/python make_figures.py --labels output/labelling/label_me.xlsx --gold-run output_jan_week_pdf
```
Writes `output/figures/*.pdf` (vector) and `*.png` (600 dpi), sized for 174 mm
(double-column) / 85 mm (single-column) with 7-9 pt text, Okabe-Ito colours fixed per
method (VADER orange, Transformer green, LLM blue), cividis confusion matrices and
zero-centred RdBu heatmaps.

| Figure | Content |
|---|---|
| fig1 | pipeline schematic |
| fig2 | articles per period by page type + token-length density with 512-token line |
| fig3 | κ and macro-F1 with 95% bootstrap CIs + row-normalised confusion matrices (*needs hand labels*) |
| fig4 | alluvial label flows VADER → Transformer → LLM + agreement / F1 by article length |
| fig5 | front vs editorial diverging bars + BERTopic topic × method net-sentiment heatmap |
| fig6 | sentiment over time, one panel per method (daily if < 3 months of data) |
| fig11 | 2-D UMAP of articles coloured by method consensus, topic labels at centroids |

Re-run after hand-labelling; missing methods are skipped.
Not included: cost/latency Pareto (not measured), ROC/calibration (VADER and the LLM
give no class probabilities), SHAP (heavy, little added value here).

## Hand-label check

1. Open `output/labelling/label_sample.xlsx` (50 random articles; model
   predictions are hidden so they can't bias you) and fill **my label**
   (positive / neutral / negative).
2. Run `.venv/bin/python evaluate_labels.py` → `output/labelling/accuracy_report.md`
   (plus `method_accuracy.png`) comparing transformer, VADER and LLM.
   For labels from another run: `evaluate_labels.py --labels <sheet> --run <run folder>`
   (articles of that run without a Claude Haiku score are scored first, cached, and
   written to `<run>/articles_with_llm.csv`; its `articles.csv` is left untouched).

The current gold set is `output/labelling/label_me.xlsx`: 48 labelled articles from the
first-week PDF run, report in `output_jan_week_pdf/labelling/`.

The sample file is never overwritten once it exists.

## Outputs (`output/`)

| Path | Contents |
|---|---|
| `articles.csv` | one row per article: date, page type, source file, headline, deck, byline, text, topic, all sentiment scores |
| `text/<page>.txt`, `text/<page>/article_NN.txt` | cleaned OCR text per page and per article |
| `overlays/<page>_overlay.jpg` | article boxes drawn on each page — **check these** |
| `tables/page_report.csv` | articles found per page, OCR confidence, reliability flags |
| `tables/sentiment_by_{topic,page_type,month,date}.csv` | summaries with 95% CIs |
| `tables/topics.csv`, `topic_model_info.json` | topic labels, top words, method details |
| `charts/*.png` | sentiment by topic / page type / month / date, label mix, transformer vs VADER, headline vs body |
| `labelling/` | hand-label sheet and accuracy report |
| `run.log` | full log of the last run |

## How it works

1. **Render / load** — PDFs rendered at 4500 px wide (~360 dpi of a printed broadsheet), images scaled to the same width.
2. **Preprocess** (`preprocess.py`) — deskew (projection profile, ±5°), grayscale,
   photo/halftone/graphic detection (mid-tone and colour density) and removal,
   Sauvola adaptive threshold, detection and removal of printed rules (incl. dotted).
3. **Segment** (`layout.py`) — Tesseract finds text lines; lines are chained into
   blocks; blocks are classed by type size relative to the body text;
   masthead / footer / captions / text inside photos are skipped; kickers, headlines,
   standfirsts and op-ed bylines are grouped; each body column is attached to the
   headline above it (never across a rule, except an op-ed byline rule); a second
   pass recovers headlines the layout pass missed. Reading order = columns left→right, top→bottom.
4. **OCR** (`ocr.py`) — each article's blocks OCR'd separately (Tesseract LSTM, psm 6);
   pull quotes that repeat the article's own text are removed.
5. **Clean** (`textclean.py`) — rejoin hyphenated words, drop junk lines, page numbers,
   "continued on page", URLs, bylines/datelines; articles < 50 words dropped.
6. **Sentiment** (`sentiment.py`) — every sentence scored by a 3-class RoBERTa model;
   probabilities averaged → `p_positive/p_neutral/p_negative`, `sentiment_label` (argmax),
   `sentiment_score` = P(pos) − P(neg) ∈ [−1, 1]. Headline scored separately.
   VADER (sentence-averaged compound) as a cross-check.
7. **Topics** (`topics.py`) — BERTopic (MiniLM embeddings + UMAP; k-means with
   silhouette-chosen k below 500 docs, HDBSCAN above) or LDA (k chosen by UMass
   coherence) below 100 docs. Topic label = top 4 c-TF-IDF / LDA words.
8. **Summaries** (`summarize.py`).

All thresholds live in `newspaper_pipeline/config.py`.

## Known limitations — read before using the numbers

* **Segmentation is rule-based** and was tuned on the Indian Express Jaipur e-paper layout.
  Other papers' layouts will need checking via the overlays. Known failure modes:
  side boxes/teasers absorbed into a neighbouring article, the last lines of a column
  cut off by an inset box, occasional headline fragments ("puzzle"), the "Explained"
  sidebars merged with the main story, and the masthead logo merged with an op-ed
  headline on the same line.
* **Stories continued on another page** are only the front-page portion.
* **OCR**: mean Tesseract confidence ~93 on these vector PDFs. Scanned/photographed
  pages will be noticeably worse. Small-caps headlines get mis-cased ("HoMEWoRK").
* **Sentiment model** was trained on tweets, not news. Averaging sentence
  probabilities pulls most articles toward *neutral* (about 90% here). Use
  `sentiment_score` (continuous) rather than the label for comparisons, and judge it
  against your hand labels before trusting it.
* **Transformer vs VADER** agree on only ~30% of labels (Spearman ρ ≈ 0.5); VADER
  calls most news text positive. They measure different things — treat disagreement
  as a sign of uncertainty, not as one being right.
* **Small sample**: one week of pages → wide confidence intervals; "by month" has a
  single month. Topic assignments on ~100 articles are rough.
