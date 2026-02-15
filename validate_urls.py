import requests

urls = [
    # Ritveek19 Crop Production
    "https://raw.githubusercontent.com/ritveek19/EDA_CropProduction/master/crop_production.csv",
    # Osprey-DS Rainfall
    "https://raw.githubusercontent.com/Osprey-DS/Rainfall-Measurement-in-INDIA-in-the-time-of-1901-to-2015/master/district%20wise%20rainfall%20normal.csv",
    # Alternative branch 'main'
    "https://raw.githubusercontent.com/Osprey-DS/Rainfall-Measurement-in-INDIA-in-the-time-of-1901-to-2015/main/district%20wise%20rainfall%20normal.csv",
    # MeetDarkPow Rainfall
    "https://raw.githubusercontent.com/MeetDarkPow/Rainfall-Data-Analysis-India/master/district%20wise%20rainfall%20normal.csv"
]

print("Checking URLs...")
for url in urls:
    try:
        resp = requests.get(url, stream=True, timeout=5)
        if resp.status_code == 200:
            print(f"✅ FOUND: {url}")
            # Peek content
            print(f"   Preview: {next(resp.iter_lines()).decode('utf-8')}")
        else:
            print(f"❌ NOT FOUND ({resp.status_code}): {url}")
    except Exception as e:
        print(f"❌ ERROR: {e}")
