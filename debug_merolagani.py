import requests
import re
from fetcher import HEADERS

sym = "AKJCL"
url = "https://merolagani.com/LatestMarket.aspx"
r = requests.get(url, headers=HEADERS, timeout=10)
print(f"Status: {r.status_code}")

idx = r.text.find(f"symbol={sym}\"")
if idx != -1:
    snippet = r.text[idx:idx+300]
    print(f"Snippet: {snippet}")
    # Merolagani: <a ...>AKJCL</a></td><td>406.00</td>
    m = re.search(r'<td>([\d,.]+)</td>', snippet)
    if m:
        print(f"Found price: {m.group(1)}")
else:
    print("Symbol not found")
