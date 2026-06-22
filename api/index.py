"""
Nifty SMC Options Intelligence Engine — Browser Edition v3.1
Uses Playwright (real Chromium) during market hours,
falls back to realistic synthetic data when NSE is closed.
"""

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from scipy.stats import norm
import math, requests, time, logging, json, asyncio, random
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Nifty SMC Options Intelligence", version="3.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_NIFTY_LOT_SIZE = 75
_cached_data: Optional[Dict] = None
_cache_time: float = 0
_CACHE_TTL = 90  # seconds

# ─────────────────────────────────────────────────────────────────────────────
# BLACK-SCHOLES GREEKS
# ─────────────────────────────────────────────────────────────────────────────

def bs_greeks(S, K, T, r, sigma, opt):
    if T <= 0 or sigma <= 0:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}
    try:
        d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        d2 = d1 - sigma*math.sqrt(T)
        gamma = norm.pdf(d1) / (S*sigma*math.sqrt(T))
        vega  = S*norm.pdf(d1)*math.sqrt(T)/100
        if opt == "CE":
            delta = norm.cdf(d1)
            theta = (-S*norm.pdf(d1)*sigma/(2*math.sqrt(T)) - r*K*math.exp(-r*T)*norm.cdf(d2))/365
        else:
            delta = -norm.cdf(-d1)
            theta = (-S*norm.pdf(d1)*sigma/(2*math.sqrt(T)) + r*K*math.exp(-r*T)*norm.cdf(-d2))/365
        return {"delta": round(delta,4), "gamma": round(gamma,6), "theta": round(theta,2), "vega": round(vega,4)}
    except Exception:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}


def implied_vol_approx(S, K, T, mkt_price, opt):
    if mkt_price <= 0 or T <= 0:
        return 0.0
    lo, hi = 0.01, 5.0
    for _ in range(80):
        mid = (lo+hi)/2
        d1 = (math.log(S/K) + (0.065+0.5*mid**2)*T)/(mid*math.sqrt(T))
        d2 = d1 - mid*math.sqrt(T)
        if opt=="CE":
            p = S*norm.cdf(d1) - K*math.exp(-0.065*T)*norm.cdf(d2)
        else:
            p = K*math.exp(-0.065*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        if abs(p-mkt_price)<0.05: return round(mid*100,2)
        if p < mkt_price: lo = mid
        else: hi = mid
    return round((lo+hi)/2*100, 2)


def days_to_expiry(expiry_str):
    try:
        exp = datetime.strptime(expiry_str, "%d-%b-%Y").date()
        return max((exp - date.today()).days, 0) / 365.0
    except Exception:
        return 7/365.0


# ─────────────────────────────────────────────────────────────────────────────
# LIVE SPOT PRICE (allIndices — always works)
# ─────────────────────────────────────────────────────────────────────────────

def get_live_spot() -> float:
    """Fetch Nifty 50 spot from allIndices endpoint (works even after market close)"""
    try:
        h = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0.0.0 Safari/537.36",
             "Accept": "*/*", "Referer": "https://www.nseindia.com/"}
        s = requests.Session()
        s.get("https://www.nseindia.com", headers=h, timeout=10)
        r = s.get("https://www.nseindia.com/api/allIndices", headers=h, timeout=10)
        if r.status_code == 200:
            for item in r.json().get("data", []):
                if item.get("index") == "NIFTY 50":
                    spot = float(item["last"])
                    logger.info(f"Live spot from allIndices: {spot}")
                    return spot
    except Exception as e:
        logger.warning(f"allIndices fetch failed: {e}")
    return 24000.0  # fallback


# ─────────────────────────────────────────────────────────────────────────────
# BROWSER FETCH (Playwright — intercepts NSE option chain API)
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_via_browser() -> Optional[Dict]:
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        logger.warning("Playwright not installed")
        return None

    logger.info("Launching headless Chromium browser...")
    result = None

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        ctx = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        )
        page = await ctx.new_page()
        future: asyncio.Future = asyncio.get_event_loop().create_future()

        async def on_response(resp):
            if "option-chain-indices" in resp.url and not future.done():
                try:
                    body = await resp.json()
                    if body.get("records", {}).get("data"):
                        future.set_result(body)
                        logger.info(f"✅ Intercepted OC data! Spot={body['records'].get('underlyingValue')}")
                except Exception:
                    pass

        page.on("response", on_response)
        try:
            await page.goto("https://www.nseindia.com/option-chain", timeout=30000, wait_until="domcontentloaded")
            await page.wait_for_timeout(5000)
            result = await asyncio.wait_for(future, timeout=15.0)
        except asyncio.TimeoutError:
            logger.warning("Browser intercept timed out — market closed or page slow")
        except Exception as e:
            logger.error(f"Browser error: {e}")
        finally:
            await browser.close()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC OPTION CHAIN (realistic, based on live spot)
# ─────────────────────────────────────────────────────────────────────────────

def build_synthetic_chain(spot: float) -> Dict:
    """
    Generates a realistic Nifty option chain when market is closed.
    Uses real spot price + Black-Scholes pricing with typical IV smile.
    """
    logger.info(f"Building synthetic option chain around spot={spot}")
    atm = round(spot / 50) * 50

    # Next weekly expiry
    today = date.today()
    days_to_thur = (3 - today.weekday()) % 7
    if days_to_thur == 0:
        days_to_thur = 7
    expiry_date = today + timedelta(days=days_to_thur)
    expiry_str  = expiry_date.strftime("%d-%b-%Y")
    T = max(days_to_thur, 1) / 365.0

    r     = 0.065
    atm_iv = 0.14  # 14% base IV
    records_data = []

    # Build strikes from ATM-2000 to ATM+2000 in steps of 50
    for K in range(int(atm) - 2000, int(atm) + 2050, 50):
        moneyness = (spot - K) / spot

        # IV smile: higher OTM IV
        iv_smile = atm_iv + 0.008 * abs(moneyness) * 10 + 0.002 * (moneyness * 10) ** 2
        iv_smile = max(iv_smile, 0.05)

        # CE price
        d1 = (math.log(spot/K) + (r + 0.5*iv_smile**2)*T) / (iv_smile*math.sqrt(T))
        d2 = d1 - iv_smile*math.sqrt(T)
        ce_price = spot*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
        pe_price = K*math.exp(-r*T)*norm.cdf(-d2) - spot*norm.cdf(-d1)

        ce_price = max(round(ce_price, 2), 0.05)
        pe_price = max(round(pe_price, 2), 0.05)

        # Realistic OI — higher near ATM, institutional blocks at key levels
        base_oi = max(500_000 - int(abs(K - atm) * 800), 10_000)
        # Add institutional OB clusters at round numbers
        if K % 500 == 0:
            base_oi = int(base_oi * 2.5)
        if K % 1000 == 0:
            base_oi = int(base_oi * 1.8)

        ce_oi = int(base_oi * (1.0 if K > atm else 0.6) * random.uniform(0.85, 1.15))
        pe_oi = int(base_oi * (1.0 if K < atm else 0.6) * random.uniform(0.85, 1.15))

        ce_oi_chg = int(ce_oi * random.uniform(-0.05, 0.08))
        pe_oi_chg = int(pe_oi * random.uniform(-0.04, 0.09))

        records_data.append({
            "strikePrice": K,
            "expiryDate": expiry_str,
            "CE": {
                "strikePrice": K,
                "expiryDate": expiry_str,
                "lastPrice": ce_price,
                "openInterest": ce_oi,
                "changeinOpenInterest": ce_oi_chg,
                "impliedVolatility": round(iv_smile * 100, 2),
            },
            "PE": {
                "strikePrice": K,
                "expiryDate": expiry_str,
                "lastPrice": pe_price,
                "openInterest": pe_oi,
                "changeinOpenInterest": pe_oi_chg,
                "impliedVolatility": round(iv_smile * 100, 2),
            }
        })

    return {
        "records": {
            "underlyingValue": spot,
            "expiryDates": [expiry_str],
            "data": records_data,
        },
        "_synthetic": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FETCH — Browser first, then synthetic fallback
# ─────────────────────────────────────────────────────────────────────────────

def fetch_nse_data() -> Dict:
    global _cached_data, _cache_time
    now = time.time()

    if _cached_data and (now - _cache_time) < _CACHE_TTL:
        logger.info("Returning cached data")
        return _cached_data

    # 1) Try real browser fetch
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        data = loop.run_until_complete(fetch_via_browser())
        loop.close()

        if data and data.get("records", {}).get("data"):
            logger.info("✅ Got LIVE data from browser")
            _cached_data = data
            _cache_time  = now
            return data
    except Exception as e:
        logger.warning(f"Browser fetch failed: {e}")

    # 2) Fallback: synthetic chain with live spot
    logger.info("⚠️  Using synthetic option chain (market closed / NSE blocking)")
    spot = get_live_spot()
    data = build_synthetic_chain(spot)
    _cached_data = data
    _cache_time  = now
    return data


# ─────────────────────────────────────────────────────────────────────────────
# SMC ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def calculate_max_pain(strikes):
    if not strikes: return 0
    best_k, best_pain = 0, float("inf")
    for t in strikes:
        K = t["strike"]
        pain = sum(max(0, K - s["strike"]) * s["ce_oi"] + max(0, s["strike"] - K) * s["pe_oi"] for s in strikes)
        if pain < best_pain:
            best_pain, best_k = pain, K
    return best_k


def analyze_order_flow(strikes, spot):
    above = [s for s in strikes if s["strike"] > spot]
    below = [s for s in strikes if s["strike"] < spot]
    supply_ob = max(above, key=lambda x: x["ce_oi"], default=None)
    demand_ob = max(below, key=lambda x: x["pe_oi"], default=None)
    top5 = sorted(strikes, key=lambda x: x["ce_oi"]+x["pe_oi"], reverse=True)[:5]
    lp   = [s["strike"] for s in top5]
    ce_chg = sum(s.get("ce_oi_chg",0) for s in strikes)
    pe_chg = sum(s.get("pe_oi_chg",0) for s in strikes)
    if pe_chg > 0 and ce_chg >= 0:    bias = "BULLISH"
    elif ce_chg > 0 and pe_chg >= 0:  bias = "BEARISH"
    elif ce_chg < 0 and pe_chg > 0:   bias = "BULLISH_UNWIND"
    elif pe_chg < 0 and ce_chg > 0:   bias = "BEARISH_UNWIND"
    else:                              bias = "NEUTRAL"
    atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i]["strike"]-spot), default=0)
    atm = strikes[atm_idx] if strikes else {}
    return {"supply_ob": supply_ob, "demand_ob": demand_ob,
            "liquidity_pools": lp, "bias": bias,
            "iv_skew": round(atm.get("pe_iv",15)-atm.get("ce_iv",15), 2),
            "ce_oi_chg_total": ce_chg, "pe_oi_chg_total": pe_chg}


def calculate_gex(strikes, spot):
    gex_list, total, prev, flip = [], 0.0, None, spot
    for s in sorted(strikes, key=lambda x: x["strike"]):
        net = (s.get("ce_gamma",0)*s["ce_oi"] - s.get("pe_gamma",0)*s["pe_oi"]) \
              * _NIFTY_LOT_SIZE * spot**2 * 0.01
        total += net
        gex_list.append({"strike": s["strike"], "gex": round(net, 2)})
        if prev is not None and prev*net < 0:
            flip = s["strike"]
        prev = net
    return gex_list, round(total, 2), flip


def analyze_market_structure(strikes, spot, pcr, total_gex, order_flow):
    if not strikes:
        return {"zone":"EQUILIBRIUM","bias":"NEUTRAL"}
    hs = [s["strike"] for s in strikes if s["strike"] > spot]
    ls = [s["strike"] for s in strikes if s["strike"] < spot]
    rh = max(hs) if hs else spot+500
    rl = min(ls) if ls else spot-500
    rng = rh - rl
    f618, f382 = rl+rng*0.618, rl+rng*0.382
    if spot > f618:     zone = "PREMIUM"
    elif spot < f382:   zone = "DISCOUNT"
    else:               zone = "EQUILIBRIUM"
    of = order_flow["bias"]
    if zone=="DISCOUNT" and of in ("BULLISH","BULLISH_UNWIND"):  sb = "STRONG_BULL"
    elif zone=="PREMIUM" and of in ("BEARISH","BEARISH_UNWIND"): sb = "STRONG_BEAR"
    elif zone=="EQUILIBRIUM":                                     sb = "RANGE_BOUND"
    else:                                                         sb = of
    return {"zone":zone,"bias":sb,"range_high":rh,"range_low":rl,
            "fib_618":round(f618,1),"fib_382":round(f382,1),
            "gex_regime":"POSITIVE" if total_gex>0 else "NEGATIVE"}


def generate_signals(spot, pcr, total_gex, gamma_flip, max_pain, market_str, order_flow, atm_iv, vix, atm_strike):
    zone = market_str["zone"]
    of   = order_flow["bias"]
    cs, cr = 0, []
    ps, pr = 0, []
    ss, sr = 0, []

    if zone=="DISCOUNT":     cs+=30; cr.append("Price in Demand Zone (Discount)")
    if of in ("BULLISH","BULLISH_UNWIND"): cs+=25; cr.append("Institutional PUT writing detected")
    if total_gex>0:          cs+=15; cr.append("Positive GEX — dealers buy dips")
    if pcr>1.2:              cs+=15; cr.append(f"PCR {pcr:.2f} — extreme puts = contrarian bull")
    if max_pain>spot:        cs+=10; cr.append(f"Max Pain ₹{max_pain} above — magnetic pull up")
    if spot<gamma_flip:      cs+=5;  cr.append("Below Gamma Flip — dealers long gamma")

    if zone=="PREMIUM":      ps+=30; pr.append("Price in Supply Zone (Premium)")
    if of in ("BEARISH","BEARISH_UNWIND"): ps+=25; pr.append("Institutional CALL writing detected")
    if total_gex<0:          ps+=15; pr.append("Negative GEX — dealers amplify falls")
    if pcr<0.8:              ps+=15; pr.append(f"PCR {pcr:.2f} — extreme calls = contrarian bear")
    if max_pain<spot:        ps+=10; pr.append(f"Max Pain ₹{max_pain} below — gravity pulls down")
    if spot>gamma_flip:      ps+=5;  pr.append("Above Gamma Flip — dealers short gamma")

    if total_gex>500:        ss+=30; sr.append("Very high positive GEX — strong pin force")
    if abs(spot-max_pain)/spot<0.01: ss+=25; sr.append(f"Spot near Max Pain ₹{max_pain} — expiry pin")
    if zone=="EQUILIBRIUM":  ss+=20; sr.append("Price at Equilibrium — no directional bias")
    if atm_iv>16:            ss+=15; sr.append(f"ATM IV {atm_iv:.1f}% — sell elevated premium")
    if vix and vix<16:       ss+=10; sr.append(f"VIX {vix:.1f} — calm = safe to sell premium")

    best = max([("LONG_CALL",cs),("LONG_PUT",ps),("SHORT_STRADDLE",ss)], key=lambda x:x[1])
    return {
        "LONG_CALL":       {"score":min(cs,100),"reasons":cr,"entry":f"Buy {atm_strike}CE","sl":f"SL: {round(spot*0.995)}","target":f"TGT: {round(spot*1.015)}"},
        "LONG_PUT":        {"score":min(ps,100),"reasons":pr,"entry":f"Buy {atm_strike}PE","sl":f"SL: {round(spot*1.005)}","target":f"TGT: {round(spot*0.985)}"},
        "SHORT_STRADDLE":  {"score":min(ss,100),"reasons":sr,"entry":f"Sell {atm_strike}CE+PE","sl":"SL: 2× premium","target":"TGT: 50% decay"},
    }, best[0]


# ─────────────────────────────────────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/nifty-analysis")
async def nifty_analysis(response: Response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    
    try:
        loop = asyncio.get_event_loop()
        raw  = await loop.run_in_executor(None, fetch_nse_data)

        is_synthetic = raw.get("_synthetic", False)
        records  = raw.get("records", {})
        spot     = float(records.get("underlyingValue", 24000))
        expiries = records.get("expiryDates", [])
        near_exp = expiries[0] if expiries else "26-Jun-2025"
        T        = days_to_expiry(near_exp)
        r        = 0.065
        atm_strike = round(spot / 50) * 50

        strike_map: Dict = {}
        for row in records.get("data", []):
            K = float(row.get("strikePrice", 0))
            if K not in strike_map:
                strike_map[K] = {"strike":K,"expiry":near_exp,
                    "ce_oi":0,"ce_oi_chg":0,"ce_iv":0,"ce_ltp":0,"ce_delta":0,"ce_gamma":0,"ce_theta":0,"ce_vega":0,
                    "pe_oi":0,"pe_oi_chg":0,"pe_iv":0,"pe_ltp":0,"pe_delta":0,"pe_gamma":0,"pe_theta":0,"pe_vega":0}
            e = strike_map[K]

            if "CE" in row and row["CE"]:
                ce  = row["CE"]
                oi  = float(ce.get("openInterest", 0))
                ltp = float(ce.get("lastPrice", 0))
                iv  = float(ce.get("impliedVolatility", 0)) or (implied_vol_approx(spot,K,T,ltp,"CE") if ltp>0 else 15.0)
                g   = bs_greeks(spot, K, T, r, iv/100, "CE")
                e.update({"ce_oi":int(oi),"ce_oi_chg":int(ce.get("changeinOpenInterest",0)),
                           "ce_iv":round(iv,2),"ce_ltp":round(ltp,2),
                           "ce_delta":g["delta"],"ce_gamma":g["gamma"],"ce_theta":g["theta"],"ce_vega":g["vega"]})

            if "PE" in row and row["PE"]:
                pe  = row["PE"]
                oi  = float(pe.get("openInterest", 0))
                ltp = float(pe.get("lastPrice", 0))
                iv  = float(pe.get("impliedVolatility", 0)) or (implied_vol_approx(spot,K,T,ltp,"PE") if ltp>0 else 15.0)
                g   = bs_greeks(spot, K, T, r, iv/100, "PE")
                e.update({"pe_oi":int(oi),"pe_oi_chg":int(pe.get("changeinOpenInterest",0)),
                           "pe_iv":round(iv,2),"pe_ltp":round(ltp,2),
                           "pe_delta":g["delta"],"pe_gamma":g["gamma"],"pe_theta":g["theta"],"pe_vega":g["vega"]})

        strikes_data = sorted(strike_map.values(), key=lambda x: x["strike"])
        total_ce_oi  = sum(s["ce_oi"] for s in strikes_data)
        total_pe_oi  = sum(s["pe_oi"] for s in strikes_data)
        pcr_oi       = round(total_pe_oi/total_ce_oi, 3) if total_ce_oi > 0 else 1.0

        atm_row = next((s for s in strikes_data if s["strike"]==atm_strike), None)
        atm_iv  = atm_row["ce_iv"] if atm_row and atm_row["ce_iv"]>0 else 15.0

        max_pain   = calculate_max_pain(strikes_data)
        order_flow = analyze_order_flow(strikes_data, spot)
        gex_data, total_gex, gamma_flip = calculate_gex(strikes_data, spot)
        market_str = analyze_market_structure(strikes_data, spot, pcr_oi, total_gex, order_flow)
        signals, best = generate_signals(spot, pcr_oi, total_gex, gamma_flip, max_pain,
                                         market_str, order_flow, atm_iv, None, atm_strike)
        display = [s for s in strikes_data if abs(s["strike"]-atm_strike) <= 1500]
        sob = order_flow.get("supply_ob")
        dob = order_flow.get("demand_ob")

        return {
            "timestamp": datetime.now().isoformat(),
            "is_synthetic": is_synthetic,
            "data_source": "SYNTHETIC (Market Closed)" if is_synthetic else "LIVE (NSE Browser)",
            "spot": spot, "atm_strike": atm_strike, "near_expiry": near_exp,
            "pcr_oi": pcr_oi, "max_pain": max_pain, "atm_iv": atm_iv,
            "vix": None, "total_gex": total_gex, "gamma_flip": gamma_flip,
            "order_blocks": {
                "supply_strike":   sob["strike"] if sob else None,
                "supply_oi":       sob["ce_oi"]  if sob else 0,
                "demand_strike":   dob["strike"] if dob else None,
                "demand_oi":       dob["pe_oi"]  if dob else 0,
                "liquidity_pools": order_flow["liquidity_pools"],
            },
            "order_flow": {k:v for k,v in order_flow.items() if k not in ("supply_ob","demand_ob")},
            "market_structure": market_str,
            "gex_data": gex_data, "strikes": display,
            "signals": signals, "best_signal": best,
        }

    except Exception as e:
        logger.exception("Analysis error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status":"ok","timestamp":datetime.now().isoformat()}


@app.get("/")
async def root():
    return FileResponse("index.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
