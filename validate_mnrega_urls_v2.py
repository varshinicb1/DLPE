import requests

urls = [
    # Arpita-deb: Space encoded as %20
    "https://raw.githubusercontent.com/Arpita-deb/NREGA-Data-Analysis/main/NREGA%20Data.csv",
    # Arpita-deb: Space encoded as %20, master branch
    "https://raw.githubusercontent.com/Arpita-deb/NREGA-Data-Analysis/master/NREGA%20Data.csv",
    # Dataful sample (if available publicly, usually behind login, but checking common public mirrors)
    "https://raw.githubusercontent.com/in-rolls/mnrega/master/data/district_wise_stats.csv"
]

print("Checking Corrected MNREGA URLs...")
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
