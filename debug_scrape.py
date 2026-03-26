import requests
import re
from fetcher import HEADERS

url = "https://merolagani.com/CompanyDetail.aspx?symbol=AKJCL"
r = requests.get(url, headers=HEADERS)
print(f"Status: {r.status_code}")
m = re.search(r'lblMarketPrice">([\d,.]+)<', r.text)
if m:
    print(f"Match 1: {m.group(1)}")
else:
    print("Match 1 failed")

# Try to find any price looking thing near Market Price text
m = re.search(r'Market Price</th>\s*<td>\s*([\d,.]+)', r.text, re.S | re.I)
if m:
    print(f"Match 2: {m.group(1)}")
else:
    print("Match 2 failed")

# Print a snippet of the page near Market Price
idx = r.text.find("Market Price")
if idx != -1:
    print(f"Snippet: {r.text[idx:idx+200]}")
else:
    print("Market Price not found in HTML")
