"""Settings for the Indian Express e-paper scraper."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
COOKIES_FILE = HERE / "session_cookies.json"     # written by login_once.py - keep private
STORAGE_FILE = HERE / "storage_state.json"       # cookies + localStorage - keep private
DEBUG_DIR = HERE / "debug"

PRODUCT_ID = 4288                                # Jaipur edition
EDITION = "Jaipur"

EPAPER_HOME = "https://epaper.indianexpress.com/"
OAUTH_URL = "https://authentication.indianexpress.com/api/vendor/2/oauth-user/?token={token}"
ISSUES_URL = ("https://ecommerce.indianexpress.com/api/partner-delivery-data/"
              "?productId={pid}&month={month}&year={year}&size=31")
PAGEMETA_URL = "https://epaper.indianexpress.com/pagemeta/get/{issue_id}/1-40"

USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36")

# politeness: one request at a time, with pauses (seconds)
DELAY_BETWEEN_REQUESTS = 1.0
DELAY_BETWEEN_TILES = 0.3
DELAY_BETWEEN_PAGES = 2.0
DELAY_BETWEEN_DAYS = 4.0
FRONT_SCAN_DEPTH = 6          # jacket ads can push the real front page to page 3-5
RETRIES = 3
TIMEOUT = 60

RESOLUTION_LEVELS = ["level3", "level2", "level1", "leveldefault"]   # best available first
