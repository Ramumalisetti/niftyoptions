import json
try:
    from nsepython import nse_optionchain_scrapper, nse_quote
    print("nsepython imported OK")

    print("Fetching NIFTY option chain...")
    oc = nse_optionchain_scrapper("NIFTY")
    print(f"Keys: {list(oc.keys())}")
    records = oc.get("records", {})
    print(f"  Underlying value: {records.get('underlyingValue')}")
    print(f"  Expiry dates: {records.get('expiryDates', [])[:3]}")
    data = records.get("data", [])
    print(f"  Total strikes: {len(data)}")
    if data:
        print(f"  Sample strike: {data[0]}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"ERROR: {e}")
