"""All tunable settings in one place.

Pixel thresholds are expressed relative to the page's measured body-text line
height (``H``) wherever possible, so they survive a change of render resolution.
"""
from pathlib import Path

# Bump this when segmentation/OCR logic changes so cached pages are recomputed.
PIPELINE_VERSION = "1.6"

# --------------------------------------------------------------------------- input
# Rendering: e-paper PDFs are 1500 pt wide (a print broadsheet is ~12.5 in wide).
# 4500 px wide ~= 360 dpi relative to the printed page, which puts body text at
# roughly 30-35 px per em - the range Tesseract handles best.
RENDER_WIDTH_PX = 4500
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
PDF_EXTS = {".pdf"}

# --------------------------------------------------------------------------- preprocess
MAX_SKEW_DEG = 5.0          # search range for deskew
SAUVOLA_WINDOW = 51         # adaptive threshold window (px, odd)
SAUVOLA_K = 0.2
# Photo / halftone detection (fractions of pixels in a local window)
PHOTO_MIDTONE_FRAC = 0.40   # share of mid-grey pixels that marks a photo
PHOTO_INK_FRAC = 0.70       # share of non-white pixels that marks a photo
PHOTO_MIN_SIDE_PX = 150     # smaller blobs are ignored (kickers, coloured text)
PHOTO_MIN_AREA_PX = 60_000

# Rules (printed lines)
RULE_MIN_LEN_PX = 300
RULE_MAX_THICK_PX = 12

# --------------------------------------------------------------------------- layout
TESS_LAYOUT_CONFIG = "--psm 3 --oem 1"
TESS_BLOCK_CONFIG = "--psm 6 --oem 1"
HEADLINE_RATIO = 1.75       # line height >= this * H  -> headline type
DECK_RATIO = 1.25           # line height >= this * H  -> kicker / deck / standfirst
SMALL_RATIO = 0.80          # line height <  this * H  -> small print (credits etc.)
MASTHEAD_ZONE = 0.22        # top fraction of page searched for the masthead
FOOTER_ZONE = 0.025         # bottom fraction of page treated as page furniture
MIN_LINE_CONF = 30          # Tesseract line confidence below this is junk

# --------------------------------------------------------------------------- text
MIN_ARTICLE_WORDS = 50
DUPLICATE_SHINGLE_OVERLAP = 0.6   # pull-quotes that repeat >=60% of another article

# --------------------------------------------------------------------------- reliability flags
FLAG_MIN_ARTICLES = 2
FLAG_MAX_ARTICLES = 20
FLAG_MAX_ORPHAN_SHARE = 0.25
FLAG_MAX_SINGLE_ARTICLE_SHARE = 0.60
FLAG_MIN_MEAN_CONF = 75
FLAG_MAX_SKEW_DEG = 2.0

# --------------------------------------------------------------------------- models
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# LLM sentiment (method 3).  "claude" (default) needs ANTHROPIC_API_KEY in results/.env;
# "hf_api" needs HF_TOKEN + HF credits; "local" runs on this Mac.
LLM_BACKEND = "claude"
LLM_MODEL_CLAUDE = "claude-haiku-5-5"               # only Claude Haiku is used for sentiment
LLM_EFFORT = "low"                                  # simple classification: low effort
LLM_WORKERS = 4                                     # parallel requests (claude backend)
LLM_PROVIDER = "auto"                               # HF picks an available provider
LLM_MODEL_API = "Qwen/Qwen2.5-7B-Instruct"         # any chat model served on HF
LLM_MODEL_LOCAL = "Qwen/Qwen2.5-1.5B-Instruct"      # fits in 8 GB RAM
LLM_MAX_WORDS = 4000         # safety cap only - news articles are shorter, so none are cut
LLM_MAX_NEW_TOKENS = 2048       # room for brief thinking + the JSON answer
VADER_POS = 0.05
VADER_NEG = -0.05
BERTOPIC_MIN_DOCS = 100
BERTOPIC_HDBSCAN_MIN_DOCS = 500   # below this, BERTopic clusters with k-means
BERTOPIC_K_RANGE = range(5, 16)
LDA_K_RANGE = range(4, 11)
RANDOM_SEED = 42
LABEL_SAMPLE_SIZE = 50

# Extra stop-words for news topic modelling (on top of sklearn's English list)
NEWS_STOPWORDS = {
    "said", "says", "say", "also", "would", "could", "one", "two", "three", "new",
    "year", "years", "time", "per", "cent", "like", "just", "many", "made", "make",
    "even", "still", "may", "us", "mr", "ms", "including", "however", "last",
    "first", "according", "told", "since", "well", "way", "page", "continued",
    "december", "january", "february", "march", "april", "june", "july",
    "august", "september", "october", "november", "monday", "tuesday",
    "wednesday", "thursday", "friday", "saturday", "sunday", "delhi", "new delhi",
    "express", "indian express", "pti", "reuters", "ap",
}


def default_paths(project_dir: Path):
    out = project_dir / "output"
    return {
        "output": out,
        "pages": out / "pages",
        "text": out / "text",
        "overlays": out / "overlays",
        "tables": out / "tables",
        "charts": out / "charts",
        "labelling": out / "labelling",
    }
