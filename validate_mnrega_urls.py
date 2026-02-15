import requests

urls = [
    # Arpita-deb: NREGA Data Analysis
    "https://raw.githubusercontent.com/Arpita-deb/NREGA-Data-Analysis/main/NREGA_data.csv",
    # hemanth929: NREGA MIS Data
    "https://raw.githubusercontent.com/hemanth929/NREGA-MIS-Data/master/NREGA_MIS_Data.csv",
    # Alternative branch check
    "https://raw.githubusercontent.com/Arpita-deb/NREGA-Data-Analysis/master/NREGA_data.csv"
]

print("Checking MNREGA URLs...")
for url in urls:
    try:
        resp = requests.get(url, stream=True, timeout=5)
        if resp.status_code == 200:
            print(f"✅ FOUND: {url}")
            # Peek content
            print(f"   Preview: {next(resp.iter_lines()).decode('utf-8')[:200]}")
        else:
            print(f"❌ NOT FOUND ({resp.status_code}): {url}")
    except Exception as e:
        print(f"❌ ERROR: {e}")
