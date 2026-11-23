import json
import requests
from pathlib import Path

COOKIES_PATH = Path("session_cookies.json")
ISSUE_LIST_URL = "https://ecommerce.indianexpress.com/api/partner-delivery-data/"

with open(COOKIES_PATH, "r", encoding="utf-8") as f:
    cookies = json.load(f)

s = requests.Session()
s.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://indianexpress.com/",
})
for cookie in cookies:
    s.cookies.set(cookie["name"], cookie["value"], domain=cookie.get("domain", "").lstrip("."))

params = {"productId": 4288, "month": 3, "year": 2025, "size": 31}
resp = s.get(ISSUE_LIST_URL, params=params, timeout=30)
print("STATUS", resp.status_code)
print(resp.text[:4000])
