import requests
import re
from fetcher import HEADERS

sym = "AKJCL"
url = "https://www.sharesansar.com/today-share-price"
r = requests.get(url, headers=HEADERS, timeout=10)
print(f"Status: {r.status_code}")

# Find the row for AKJCL
idx = r.text.find(f">{sym}<")
if idx != -1:
    snippet = r.text[idx:idx+500]
    print(f"Snippet: {snippet}")
    # ShareSansar columns: ...</td><td>LTP</td><td>Change</td>...
    m = re.search(r'<td>([\d,.]+)</td>', snippet)
    if m:
        print(f"Found price: {m.group(1)}")
else:
    print("Symbol not found")
