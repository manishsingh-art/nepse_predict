import pandas as pd
import requests
from fetcher import HEADERS

url = "https://merolagani.com/LatestMarket.aspx"
try:
    r = requests.get(url, headers=HEADERS, timeout=10, verify=False)
    tables = pd.read_html(r.text)
    print(f"Merolagani: {len(tables)} tables")
    for i, t in enumerate(tables):
        print(f"Table {i} Head:\n{t.head(3)}")
except Exception as e:
    print(f"Merolagani Error: {e}")

url = "https://www.sharesansar.com/today-share-price"
try:
    r = requests.get(url, headers=HEADERS, timeout=10, verify=False)
    tables = pd.read_html(r.text)
    print(f"ShareSansar: {len(tables)} tables")
    for i, t in enumerate(tables):
        print(f"Table {i} Head:\n{t.head(3)}")
except Exception as e:
    print(f"ShareSansar Error: {e}")
