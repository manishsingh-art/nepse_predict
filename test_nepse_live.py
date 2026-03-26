import requests
from fetcher import HEADERS

# NEPSE Active Securities (Live LTPs)
url = "https://nepalstock.com.np/api/nots/market/active-securities"
try:
    r = requests.get(url, headers=HEADERS, timeout=10)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        # Find AKJCL
        for item in data:
            if item.get("symbol") == "AKJCL":
                print(f"AKJCL: {item.get('lastTradedPrice')}")
                break
        else:
            print("AKJCL not found in active securities")
    else:
        print(f"Body: {r.text[:200]}")
except Exception as e:
    print(f"Error: {e}")
