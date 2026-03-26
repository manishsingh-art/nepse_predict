import pandas as pd
import requests
from fetcher import HEADERS

url = "https://merolagani.com/LatestMarket.aspx"
r = requests.get(url, headers=HEADERS, timeout=10)
print(f"Status: {r.status_code}")

try:
    tables = pd.read_html(r.text)
    print(f"Found {len(tables)} tables")
    for i, t in enumerate(tables):
        print(f"Table {i} columns: {list(t.columns)}")
        # Check if Symbol is in columns
        if any("Symbol" in str(c) for c in t.columns):
            res = t[t.iloc[:, 0] == "AKJCL"]
            if not res.empty:
                print(f"Match in Table {i}!")
                print(res.iloc[0].to_dict())
except Exception as e:
    print(f"Error: {e}")
