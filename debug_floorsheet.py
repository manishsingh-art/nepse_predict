import requests
import pandas as pd
from bs4 import BeautifulSoup

def test_floorsheet(symbol="AKJCL"):
    url = f"https://www.sharesansar.com/floorsheet?symbol={symbol}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        try:
            tables = pd.read_html(r.text)
            for t in tables:
                if "buyer" in str(t.columns).lower():
                    print(f"Found floorsheet table for {symbol}:")
                    print(t.head())
                    return True
        except Exception as e:
            print(f"Error parsing table: {e}")
    else:
        print(f"Failed to fetch: {r.status_code}")
    return False

test_floorsheet()
