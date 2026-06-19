"""
Nifty SMC Options Intelligence Engine
Author: AI Options Trader
Concepts: Smart Money Concepts (SMC) + Order Flow + GEX + Black-Scholes Greeks

SMC Translation to Options:
- Order Blocks      = Strikes with massive OI (institutional money zones)
- Liquidity Pools   = ATM/near-ATM strikes (stop loss concentration)
- BOS/CHoCH         = PCR + OI velocity + GEX regime shift
- Premium Zone      = Spot above equilibrium (resistance pressure)
- Discount Zone     = Spot below equilibrium (support pressure)
- Gamma Flip        = Where dealers flip from long to short gamma (CRITICAL pivot)
- Inducement        = Fake OI buildup before actual move
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import numpy as np
from scipy.stats import norm
import math
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Nifty SMC Options Intelligence", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# NSE SESSION MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}

NSE_API_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://www.nseindia.com/option-chain",
    "X-Requested-With": "XMLHttpRequest",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
}

_session: Optional[requests.Session] = None
_session_time: float = 0
_NIFTY_LOT_SIZE = 75  # Current Nifty lot size


def get_nse_session() -> requests.Session:
    """Multi-step cookie warming for NSE — mimics real browser navigation"""
    global _session, _session_time
    now = time.time()
    if _session is None or (now - _session_time) > 240:
        logger.info("Warming new NSE session (3-step cookie handshake)...")
        sess = requests.Session()
        sess.headers.update({"User-Agent": NSE_HEADERS["User-Agent"]})

        # Step 1: Hit homepage to get initial cookies
        try:
            sess.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=15)
            time.sleep(1.0)
        except Exception:
            pass

        # Step 2: Hit option-chain page (sets nsit, nseappid cookies)
        try:
            sess.get("https://www.nseindia.com/option-chain", headers=NSE_HEADERS, timeout=15)
            time.sleep(0.8)
        except Exception:
            pass

        # Step 3: Pre-warm the API endpoint with a quiet ping
        try:
            sess.get(
                "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY",
                headers=NSE_API_HEADERS, timeout=15
            )
            time.sleep(0.5)
        except Exception:
            pass

        _session = sess
        _session_time = now
    return _session


def fetch_nse_data(symbol: str = "NIFTY") -> Dict:
    """Fetch NSE option chain with session retry and full error context"""
    global _session, _session_time

    for attempt in range(3):
        try:
            session = get_nse_session()
            url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
            resp = session.get(url, headers=NSE_API_HEADERS, timeout=20)

            if resp.status_code == 200:
                data = resp.json()
                if "records" in data:
                    return data
                # Got 200 but wrong payload — re-warm session
                logger.warning(f"Attempt {attempt+1}: 200 but bad payload, re-warming session")
            else:
                logger.warning(f"Attempt {attempt+1}: HTTP {resp.status_code}, re-warming session")

            # Force re-warm on next iteration
            _session = None
            _session_time = 0
            time.sleep(2)

        except requests.exceptions.Timeout:
            logger.warning(f"Attempt {attempt+1}: Timeout, retrying...")
            _session = None
            _session_time = 0
            time.sleep(2)
        except Exception as e:
            logger.error(f"Attempt {attempt+1}: {e}")
            _session = None
            _session_time = 0
            time.sleep(2)

    raise ConnectionError("NSE API unreachable after 3 attempts. Market may be closed or NSE is blocking requests. Try again in a few minutes.")


def fetch_vix() -> Optional[float]:
    """Fetch India VIX from NSE"""
    try:
        session = get_nse_session()
        url = "https://www.nseindia.com/api/allIndices"
        resp = session.get(url, headers=NSE_HEADERS, timeout=10)
        if resp.status_code == 200:
            indices = resp.json().get("data", [])
            for idx in indices:
                if idx.get("index") == "INDIA VIX":
                    return float(idx.get("last", 0))
    except Exception as e:
        logger.warning(f"VIX fetch failed: {e}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# BLACK-SCHOLES GREEKS ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def bs_greeks(spot: float, strike: float, r: float, T: float, sigma: float, opt_type: str) -> Dict:
    """
    Full Black-Scholes Greeks
    spot   : underlying price
    strike : option strike
    r      : risk-free rate (decimal, e.g. 0.065)
    T      : time to expiry in years
    sigma  : implied volatility (decimal, e.g. 0.15)
    opt_type: 'CE' or 'PE'
    """
    zero = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0,
            "rho": 0.0, "d1": 0.0, "d2": 0.0}

    if T <= 1e-6 or sigma <= 1e-6 or spot <= 0 or strike <= 0:
        return zero

    try:
        d1 = (math.log(spot / strike) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        Nd1 = norm.cdf(d1)
        Nd2 = norm.cdf(d2)
        nd1 = norm.pdf(d1)

        gamma = nd1 / (spot * sigma * math.sqrt(T))
        vega  = spot * nd1 * math.sqrt(T) / 100  # per 1% IV change

        if opt_type == "CE":
            delta = Nd1
            theta = (-(spot * nd1 * sigma) / (2 * math.sqrt(T))
                     - r * strike * math.exp(-r * T) * Nd2) / 365
            rho   = strike * T * math.exp(-r * T) * Nd2 / 100
        else:
            delta = Nd1 - 1
            theta = (-(spot * nd1 * sigma) / (2 * math.sqrt(T))
                     + r * strike * math.exp(-r * T) * norm.cdf(-d2)) / 365
            rho   = -strike * T * math.exp(-r * T) * norm.cdf(-d2) / 100

        return {
            "delta": round(delta, 4),
            "gamma": round(gamma, 7),
            "theta": round(theta, 2),
            "vega":  round(vega, 2),
            "rho":   round(rho, 2),
            "d1":    round(d1, 4),
            "d2":    round(d2, 4),
        }
    except Exception:
        return zero


# ─────────────────────────────────────────────────────────────────────────────
# GAMMA EXPOSURE (GEX) ENGINE — THE CORE OF DEALER HEDGING
# ─────────────────────────────────────────────────────────────────────────────

def calculate_gex(strikes_data: List[Dict], spot: float) -> Tuple[List[Dict], float, Optional[float]]:
    """
    GEX = OI × Gamma × Spot² × LotSize × 0.01
    Dealers are LONG gamma on calls they've sold → positive GEX per call
    Dealers are SHORT gamma on puts they've sold → negative GEX per put

    Positive total GEX = Stable, mean-reverting market
    Negative total GEX = Explosive, momentum market
    Gamma flip = The pivot where GEX changes sign (critical level!)
    """
    gex_by_strike = []
    total_gex = 0.0

    for s in strikes_data:
        ce_gex = s.get("ce_oi", 0) * s.get("ce_gamma", 0) * (spot ** 2) * _NIFTY_LOT_SIZE * 0.01 / 1e9
        pe_gex = -s.get("pe_oi", 0) * s.get("pe_gamma", 0) * (spot ** 2) * _NIFTY_LOT_SIZE * 0.01 / 1e9
        net = ce_gex + pe_gex
        total_gex += net
        gex_by_strike.append({
            "strike": s["strike"],
            "ce_gex": round(ce_gex, 4),
            "pe_gex": round(pe_gex, 4),
            "net_gex": round(net, 4),
        })

    # Find gamma flip (where cumulative GEX changes sign as we scan from low to high)
    gamma_flip = None
    cumulative = 0.0
    for i, g in enumerate(gex_by_strike):
        prev = cumulative
        cumulative += g["net_gex"]
        if i > 0 and prev * cumulative < 0:  # sign change
            gamma_flip = round((gex_by_strike[i - 1]["strike"] + g["strike"]) / 2)
            break

    return gex_by_strike, round(total_gex, 4), gamma_flip


# ─────────────────────────────────────────────────────────────────────────────
# MAX PAIN CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def calculate_max_pain(strikes_data: List[Dict]) -> float:
    """
    Max Pain = Strike where total monetary loss to option buyers is maximum
    (i.e., where option sellers/writers lose the least)
    This acts as a magnetic price target near expiry.
    """
    pain_map = {}
    for target in strikes_data:
        t_strike = target["strike"]
        pain = 0
        for s in strikes_data:
            if t_strike > s["strike"]:
                pain += (t_strike - s["strike"]) * s.get("ce_oi", 0)
            if t_strike < s["strike"]:
                pain += (s["strike"] - t_strike) * s.get("pe_oi", 0)
        pain_map[t_strike] = pain

    return min(pain_map, key=pain_map.get) if pain_map else 0


# ─────────────────────────────────────────────────────────────────────────────
# SMC ORDER FLOW ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_order_flow(strikes_data: List[Dict], spot: float) -> Dict:
    """
    SMC Order Flow Analysis:
    - Order Blocks: Strikes with massive OI = institutional interest zones
    - Liquidity Pools: Where stop losses cluster (highest combined OI)
    - OI Velocity: Rate of OI change (smart money accelerating/decelerating)
    - Absorption: Heavy OI without price movement = absorption (reversal signal)
    - CE/PE Writing Pressure: Who is the aggressor?
    """
    above = [s for s in strikes_data if s["strike"] > spot]
    below = [s for s in strikes_data if s["strike"] <= spot]
    near  = [s for s in strikes_data if abs(s["strike"] - spot) <= 500]

    # Supply Zone (CE Order Block) = Max CE OI above spot
    supply_ob = max(above, key=lambda x: x.get("ce_oi", 0)) if above else None
    # Demand Zone (PE Order Block) = Max PE OI below spot
    demand_ob = max(below, key=lambda x: x.get("pe_oi", 0)) if below else None

    # Liquidity pools = highest combined OI (buy/sell stops both sides)
    liq_pools = sorted(
        strikes_data,
        key=lambda x: x.get("ce_oi", 0) + x.get("pe_oi", 0),
        reverse=True
    )[:5]

    # OI Velocity — is institutional money accelerating?
    ce_oi_vel = sum(s.get("ce_oi_change", 0) for s in near)
    pe_oi_vel = sum(s.get("pe_oi_change", 0) for s in near)

    # Net writing pressure (positive = more call writing = bearish)
    # Institutions write calls when they expect market to stay below that level
    # Institutions write puts when they expect market to stay above that level
    ce_writing = sum(s.get("ce_oi_change", 0) for s in above if s.get("ce_oi_change", 0) > 0)
    pe_writing = sum(s.get("pe_oi_change", 0) for s in below if s.get("pe_oi_change", 0) > 0)

    # Unwinding signals
    ce_unwinding = sum(abs(s.get("ce_oi_change", 0)) for s in near if s.get("ce_oi_change", 0) < 0)
    pe_unwinding = sum(abs(s.get("pe_oi_change", 0)) for s in near if s.get("pe_oi_change", 0) < 0)

    # IV Skew (put IV vs call IV near ATM) — skew > 1 = bearish fear
    atm_ce_iv = np.mean([s.get("ce_iv", 0) for s in near if s.get("ce_iv", 0) > 0] or [15])
    atm_pe_iv = np.mean([s.get("pe_iv", 0) for s in near if s.get("pe_iv", 0) > 0] or [15])
    iv_skew = round(atm_pe_iv / atm_ce_iv, 3) if atm_ce_iv > 0 else 1.0

    # Volume imbalance (which side is being traded more aggressively)
    ce_vol = sum(s.get("ce_volume", 0) for s in near)
    pe_vol = sum(s.get("pe_volume", 0) for s in near)
    vol_ratio = round(pe_vol / ce_vol, 3) if ce_vol > 0 else 1.0

    # Put-Call Ratio by Volume (more accurate than OI for short-term sentiment)
    total_ce_vol = sum(s.get("ce_volume", 0) for s in strikes_data)
    total_pe_vol = sum(s.get("pe_volume", 0) for s in strikes_data)
    pcr_vol = round(total_pe_vol / total_ce_vol, 3) if total_ce_vol > 0 else 1.0

    return {
        "supply_ob": supply_ob,       # CE resistance order block
        "demand_ob": demand_ob,       # PE support order block
        "liquidity_pools": [lp["strike"] for lp in liq_pools],
        "ce_oi_velocity": ce_oi_vel,  # Positive = call OI building (bearish)
        "pe_oi_velocity": pe_oi_vel,  # Positive = put OI building (bullish)
        "ce_writing": ce_writing,     # Fresh CE writing above spot
        "pe_writing": pe_writing,     # Fresh PE writing below spot
        "ce_unwinding": ce_unwinding, # CE being closed near ATM (bullish)
        "pe_unwinding": pe_unwinding, # PE being closed near ATM (bearish)
        "iv_skew": iv_skew,           # >1.1 = bearish fear, <0.9 = bullish greed
        "vol_ratio": vol_ratio,       # PE vol / CE vol near ATM
        "pcr_vol": pcr_vol,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SMC MARKET STRUCTURE (BOS, CHoCH, Premium/Discount)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_market_structure(
    strikes_data: List[Dict],
    spot: float,
    pcr: float,
    total_gex: float,
    order_flow: Dict,
) -> Dict:
    """
    SMC Market Structure:
    - Equilibrium: Midpoint between max CE OI (supply) and max PE OI (demand)
    - Premium Zone: >60% of range (smart money sells here)
    - Discount Zone: <40% of range (smart money buys here)
    - BOS Up: Price broke above recent supply OB → continuation bullish
    - BOS Down: Price broke below recent demand OB → continuation bearish
    - CHoCH: Counter-trend OI shift → potential reversal
    """
    supply_ob = order_flow.get("supply_ob")
    demand_ob = order_flow.get("demand_ob")

    supply_strike = supply_ob["strike"] if supply_ob else (spot + 500)
    demand_strike = demand_ob["strike"] if demand_ob else (spot - 500)

    # Equilibrium = midpoint between supply and demand zones
    equilibrium = (supply_strike + demand_strike) / 2

    # Premium/Discount positioning
    total_range = supply_strike - demand_strike
    if total_range > 0:
        pos_pct = ((spot - demand_strike) / total_range) * 100
    else:
        pos_pct = 50.0

    if pos_pct > 61.8:  # Fibonacci 61.8% = Premium
        zone = "PREMIUM"
        zone_bias = "BEARISH"
        zone_desc = "Price in SUPPLY/PREMIUM zone — Smart Money distributing"
    elif pos_pct < 38.2:  # Fibonacci 38.2% = Discount
        zone = "DISCOUNT"
        zone_bias = "BULLISH"
        zone_desc = "Price in DEMAND/DISCOUNT zone — Smart Money accumulating"
    else:
        zone = "EQUILIBRIUM"
        zone_bias = "NEUTRAL"
        zone_desc = "Price at EQUILIBRIUM — No edge, wait for extreme"

    # GEX Regime Analysis
    if total_gex > 3:
        gex_regime = "STRONG_POSITIVE"
        gex_bias = "RANGE_BOUND"
        gex_desc = f"Strong Positive GEX ({total_gex:.2f}B) — Dealers heavily long gamma, market magnetized"
    elif total_gex > 0.5:
        gex_regime = "POSITIVE"
        gex_bias = "STABLE"
        gex_desc = f"Positive GEX ({total_gex:.2f}B) — Dealers buy dips, sell rips → mean reversion"
    elif total_gex > -0.5:
        gex_regime = "NEUTRAL"
        gex_bias = "TRANSITIONAL"
        gex_desc = f"Near-zero GEX ({total_gex:.2f}B) — Transitional phase, watch for flip"
    elif total_gex > -3:
        gex_regime = "NEGATIVE"
        gex_bias = "TRENDING"
        gex_desc = f"Negative GEX ({total_gex:.2f}B) — Dealers short gamma, amplifying moves"
    else:
        gex_regime = "STRONG_NEGATIVE"
        gex_bias = "EXPLOSIVE"
        gex_desc = f"Strong Negative GEX ({total_gex:.2f}B) — Dealers panic hedging, explosive volatility"

    # Institutional Bias (Order Flow)
    ce_vel = order_flow.get("ce_oi_velocity", 0)
    pe_vel = order_flow.get("pe_oi_velocity", 0)
    ce_writing = order_flow.get("ce_writing", 0)
    pe_writing = order_flow.get("pe_writing", 0)

    # Smart money is writing PUTS → they expect support → BULLISH
    # Smart money is writing CALLS → they expect resistance → BEARISH
    if pe_writing > ce_writing * 1.5 and pcr > 1.0:
        inst_bias = "BULLISH"
        inst_desc = "Institutions writing Puts (expecting floor) + High PCR = Bullish flow"
    elif ce_writing > pe_writing * 1.5 and pcr < 1.0:
        inst_bias = "BEARISH"
        inst_desc = "Institutions writing Calls (expecting ceiling) + Low PCR = Bearish flow"
    elif order_flow.get("ce_unwinding", 0) > order_flow.get("pe_unwinding", 0) * 1.3:
        inst_bias = "BULLISH"
        inst_desc = "CE OI Unwinding dominant — Call writers exiting → bullish momentum"
    elif order_flow.get("pe_unwinding", 0) > order_flow.get("ce_unwinding", 0) * 1.3:
        inst_bias = "BEARISH"
        inst_desc = "PE OI Unwinding dominant — Put writers exiting → bearish momentum"
    else:
        inst_bias = "NEUTRAL"
        inst_desc = "No clear institutional directional bias detected"

    # IV Skew interpretation
    iv_skew = order_flow.get("iv_skew", 1.0)
    if iv_skew > 1.15:
        skew_bias = "BEARISH"
        skew_desc = f"Put skew high ({iv_skew:.2f}) — Traders paying up for downside protection"
    elif iv_skew < 0.9:
        skew_bias = "BULLISH"
        skew_desc = f"Call IV elevated ({iv_skew:.2f}) — Demand for upside calls (bullish speculation)"
    else:
        skew_bias = "NEUTRAL"
        skew_desc = f"IV Skew neutral ({iv_skew:.2f}) — Balanced risk perception"

    return {
        "zone": zone,
        "zone_bias": zone_bias,
        "zone_desc": zone_desc,
        "position_pct": round(pos_pct, 1),
        "equilibrium": round(equilibrium),
        "supply_strike": supply_strike,
        "demand_strike": demand_strike,
        "gex_regime": gex_regime,
        "gex_bias": gex_bias,
        "gex_desc": gex_desc,
        "inst_bias": inst_bias,
        "inst_desc": inst_desc,
        "skew_bias": skew_bias,
        "skew_desc": skew_desc,
    }


# ─────────────────────────────────────────────────────────────────────────────
# AI SIGNAL ENGINE — SMC + ORDER FLOW CONFLUENCE
# ─────────────────────────────────────────────────────────────────────────────

def generate_signals(
    spot: float,
    pcr: float,
    total_gex: float,
    gamma_flip: Optional[float],
    max_pain: float,
    market_str: Dict,
    order_flow: Dict,
    atm_iv: float,
    vix: Optional[float],
    atm_strike: float,
) -> Tuple[List[Dict], Dict]:
    """
    AI Signal Engine — Confluence-based SMC trading signals
    Each signal is scored by confluences (0–100)
    """

    zone = market_str["zone"]
    gex_regime = market_str["gex_regime"]
    inst_bias = market_str["inst_bias"]
    skew_bias = market_str["skew_bias"]
    supply_ob = order_flow.get("supply_ob")
    demand_ob = order_flow.get("demand_ob")

    # ── LONG CALL ─────────────────────────────────────────────────────────
    call_score = 0
    call_reasons = []
    call_cautions = []

    # 1. Zone confluence
    if zone == "DISCOUNT":
        call_score += 22
        call_reasons.append(f"🟢 DISCOUNT Zone ({market_str['position_pct']:.0f}%) — Price in SMC demand zone, high-probability LONG area")
    elif zone == "PREMIUM":
        call_score -= 15
        call_cautions.append("⛔ PREMIUM Zone — Buying calls at supply is low-probability")

    # 2. Institutional bias
    if inst_bias == "BULLISH":
        call_score += 20
        call_reasons.append(f"🏛️ Institutional Flow: {market_str['inst_desc']}")

    # 3. PCR (Contrarian — extreme put activity = fuel for rally)
    if pcr > 1.3:
        call_score += 18
        call_reasons.append(f"📊 PCR={pcr:.2f} — Extreme put positioning = contrarian BULLISH fuel")
    elif pcr > 1.0:
        call_score += 8
        call_reasons.append(f"📊 PCR={pcr:.2f} — More puts than calls = mild bullish lean")

    # 4. Max Pain magnet
    if max_pain > spot:
        dist_pct = ((max_pain - spot) / spot) * 100
        call_score += min(int(dist_pct * 5), 15)
        call_reasons.append(f"🧲 Max Pain {max_pain:.0f} is {dist_pct:.1f}% ABOVE spot — Magnetic pull upward")

    # 5. Demand Order Block (SMC)
    if demand_ob and demand_ob["strike"] < spot:
        ob_oi_cr = demand_ob.get("pe_oi", 0) * _NIFTY_LOT_SIZE * demand_ob["strike"] / 1e7
        call_score += 15
        call_reasons.append(f"🏛️ PE Order Block at {demand_ob['strike']} ({ob_oi_cr:.0f}Cr OI) — Institutional DEMAND zone protecting downside")

    # 6. GEX Regime
    if gex_regime in ("POSITIVE", "STRONG_POSITIVE"):
        call_score += 10
        call_reasons.append(f"⚡ {gex_regime} GEX — Dealers buy dips, adds fuel to calls")
    elif gex_regime in ("NEGATIVE", "STRONG_NEGATIVE"):
        call_cautions.append("⚠️ Negative GEX — Explosive potential, use spreads not naked calls")

    # 7. CE Unwinding (call writers covering → bullish)
    ce_unwind = order_flow.get("ce_unwinding", 0)
    if ce_unwind > 0:
        call_score += 8
        call_reasons.append(f"📉 CE OI Unwinding ({ce_unwind/1000:.0f}K contracts) — Call writers covering = bullish momentum")

    # 8. IV Skew
    if skew_bias == "BULLISH":
        call_score += 7
        call_reasons.append(f"📈 {market_str['skew_desc']}")

    # 9. Gamma Flip
    if gamma_flip and spot > gamma_flip:
        call_score += 8
        call_reasons.append(f"🔥 Spot ({spot:.0f}) ABOVE Gamma Flip ({gamma_flip:.0f}) — Market in positive GEX territory")
    elif gamma_flip and spot < gamma_flip:
        call_cautions.append(f"⚠️ Spot below Gamma Flip ({gamma_flip:.0f}) — Be cautious, dealers may amplify moves")

    # Strategy
    call_entry = round(atm_strike / 50) * 50
    call_target = supply_ob["strike"] if supply_ob else int(spot * 1.015)
    call_sl = demand_ob["strike"] - 50 if demand_ob else int(spot * 0.99)
    call_strat = f"Buy {call_entry} CE (ATM) or {call_entry + 50} CE (1 OTM). Target: {call_target} CE OB. SL: {call_sl} break"

    # ── LONG PUT ──────────────────────────────────────────────────────────
    put_score = 0
    put_reasons = []
    put_cautions = []

    if zone == "PREMIUM":
        put_score += 22
        put_reasons.append(f"🔴 PREMIUM Zone ({market_str['position_pct']:.0f}%) — Price in SMC supply zone, high-probability SHORT area")
    elif zone == "DISCOUNT":
        put_score -= 15
        put_cautions.append("⛔ DISCOUNT Zone — Buying puts at demand is low-probability")

    if inst_bias == "BEARISH":
        put_score += 20
        put_reasons.append(f"🏛️ Institutional Flow: {market_str['inst_desc']}")

    if pcr < 0.7:
        put_score += 18
        put_reasons.append(f"📊 PCR={pcr:.2f} — Extreme call positioning = contrarian BEARISH fuel")
    elif pcr < 1.0:
        put_score += 8
        put_reasons.append(f"📊 PCR={pcr:.2f} — More calls than puts = mild bearish lean")

    if max_pain < spot:
        dist_pct = ((spot - max_pain) / spot) * 100
        put_score += min(int(dist_pct * 5), 15)
        put_reasons.append(f"🧲 Max Pain {max_pain:.0f} is {dist_pct:.1f}% BELOW spot — Magnetic pull downward")

    if supply_ob and supply_ob["strike"] > spot:
        ob_oi_cr = supply_ob.get("ce_oi", 0) * _NIFTY_LOT_SIZE * supply_ob["strike"] / 1e7
        put_score += 15
        put_reasons.append(f"🏛️ CE Order Block at {supply_ob['strike']} ({ob_oi_cr:.0f}Cr OI) — Institutional SUPPLY zone capping upside")

    if gex_regime in ("NEGATIVE", "STRONG_NEGATIVE"):
        put_score += 12
        put_reasons.append(f"⚡ {gex_regime} GEX — Dealers amplifying downward moves, puts gain faster")
    elif gex_regime in ("POSITIVE", "STRONG_POSITIVE"):
        put_cautions.append("⚠️ Positive GEX — Dealers buy dips, may cap put gains")

    pe_unwind = order_flow.get("pe_unwinding", 0)
    if pe_unwind > 0:
        put_score += 8
        put_reasons.append(f"📉 PE OI Unwinding ({pe_unwind/1000:.0f}K contracts) — Put writers covering = bearish acceleration")

    if skew_bias == "BEARISH":
        put_score += 7
        put_reasons.append(f"📉 {market_str['skew_desc']}")

    if gamma_flip and spot < gamma_flip:
        put_score += 8
        put_reasons.append(f"🔥 Spot ({spot:.0f}) BELOW Gamma Flip ({gamma_flip:.0f}) — Market in negative GEX, amplified downside")

    put_entry = round(atm_strike / 50) * 50
    put_target = demand_ob["strike"] if demand_ob else int(spot * 0.985)
    put_sl = supply_ob["strike"] + 50 if supply_ob else int(spot * 1.01)
    put_strat = f"Buy {put_entry} PE (ATM) or {put_entry - 50} PE (1 OTM). Target: {put_target} PE OB. SL: {put_sl} break"

    # ── SHORT STRADDLE ────────────────────────────────────────────────────
    straddle_score = 0
    straddle_reasons = []
    straddle_cautions = []

    if gex_regime in ("STRONG_POSITIVE", "POSITIVE"):
        straddle_score += 30
        straddle_reasons.append(f"⚡ {gex_regime} GEX ({total_gex:.2f}B) — Dealers will actively pin price, perfect for straddle")

    if atm_iv > 18:
        iv_bonus = min(int((atm_iv - 18) * 2), 25)
        straddle_score += iv_bonus
        straddle_reasons.append(f"💰 ATM IV={atm_iv:.1f}% — Expensive premium, ideal to sell (IV crush expected)")

    dist_pain = abs(spot - max_pain) / spot * 100
    if dist_pain < 1.0:
        straddle_score += 25
        straddle_reasons.append(f"🎯 Spot near Max Pain ({max_pain:.0f}, dist={dist_pain:.2f}%) — Maximum pinning force active")
    elif dist_pain < 2.0:
        straddle_score += 15
        straddle_reasons.append(f"🎯 Spot relatively close to Max Pain ({max_pain:.0f}, {dist_pain:.1f}%)")

    if zone == "EQUILIBRIUM":
        straddle_score += 20
        straddle_reasons.append("⚖️ Price at EQUILIBRIUM — No SMC directional edge, range-bound setup")

    if vix and vix < 16:
        straddle_score += 10
        straddle_reasons.append(f"📉 India VIX={vix:.1f} — Low fear, stable environment for premium selling")
    elif vix and vix > 22:
        straddle_cautions.append(f"🚨 VIX={vix:.1f} High — Dangerous for straddle (move risk!)")

    if inst_bias == "NEUTRAL":
        straddle_score += 10
        straddle_reasons.append("🔄 No directional institutional bias — Range-bound likely")

    # Cautions
    if gex_regime in ("NEGATIVE", "STRONG_NEGATIVE"):
        straddle_cautions.append("🚨 AVOID — Negative GEX = explosive move will blow up straddle!")
    if atm_iv < 12:
        straddle_cautions.append("⚠️ Low IV — Premium too cheap, risk/reward poor for straddle")
    if zone != "EQUILIBRIUM":
        straddle_cautions.append(f"⚠️ Price in {zone} zone — Directional risk exists")

    straddle_strike = round(spot / 50) * 50
    straddle_strat = (
        f"Sell {straddle_strike} CE + {straddle_strike} PE. "
        f"Max profit zone: {int(max_pain * 0.995)}-{int(max_pain * 1.005)}. "
        f"Hedge: Buy {straddle_strike + 200} CE + {straddle_strike - 200} PE (Iron Fly)"
    )

    signals = [
        {
            "type": "BUY_CALL",
            "label": "LONG CALL",
            "emoji": "🟢",
            "color": "#00ff88",
            "score": max(0, min(call_score, 100)),
            "strength": "STRONG" if call_score >= 65 else ("MODERATE" if call_score >= 40 else "WEAK"),
            "reasons": call_reasons,
            "cautions": call_cautions,
            "strategy": call_strat,
        },
        {
            "type": "BUY_PUT",
            "label": "LONG PUT",
            "emoji": "🔴",
            "color": "#ff4466",
            "score": max(0, min(put_score, 100)),
            "strength": "STRONG" if put_score >= 65 else ("MODERATE" if put_score >= 40 else "WEAK"),
            "reasons": put_reasons,
            "cautions": put_cautions,
            "strategy": put_strat,
        },
        {
            "type": "SHORT_STRADDLE",
            "label": "SHORT STRADDLE",
            "emoji": "⚡",
            "color": "#ffcc00",
            "score": max(0, min(straddle_score, 100)),
            "strength": "STRONG" if straddle_score >= 65 else ("MODERATE" if straddle_score >= 40 else "WEAK"),
            "reasons": straddle_reasons,
            "cautions": straddle_cautions,
            "strategy": straddle_strat,
        },
    ]

    best = max(signals, key=lambda x: x["score"])
    return signals, best


# ─────────────────────────────────────────────────────────────────────────────
# MAIN API ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/nifty-analysis")
async def nifty_analysis():
    try:
        raw = fetch_nse_data("NIFTY")
        vix = fetch_vix()

        spot       = float(raw["records"]["underlyingValue"])
        expiry_dates = raw["records"]["expiryDates"]
        near_expiry = expiry_dates[0]

        # Days to expiry
        exp_dt = datetime.strptime(near_expiry, "%d-%b-%Y").date()
        dte    = max((exp_dt - date.today()).days, 1)
        T      = dte / 365.0
        r      = 0.065  # RBI approximate risk-free rate

        atm_strike = round(spot / 50) * 50

        # ── Parse option chain ──────────────────────────────────────────
        strike_map: Dict[int, Dict] = {}
        for item in raw["records"]["data"]:
            if item.get("expiryDate") != near_expiry:
                continue
            k = item["strikePrice"]
            if abs(k - atm_strike) > 3000:
                continue
            if k not in strike_map:
                strike_map[k] = {"strike": k,
                                  "ce_oi": 0, "ce_oi_change": 0, "ce_volume": 0,
                                  "ce_iv": 0, "ce_ltp": 0,
                                  "ce_delta": 0, "ce_gamma": 0, "ce_theta": 0,
                                  "ce_vega": 0,
                                  "pe_oi": 0, "pe_oi_change": 0, "pe_volume": 0,
                                  "pe_iv": 0, "pe_ltp": 0,
                                  "pe_delta": 0, "pe_gamma": 0, "pe_theta": 0,
                                  "pe_vega": 0}
            s = strike_map[k]

            if "CE" in item:
                ce = item["CE"]
                iv  = float(ce.get("impliedVolatility") or 0)
                ltp = float(ce.get("lastPrice") or 0)
                g = bs_greeks(spot, k, r, T, iv / 100, "CE") if iv > 0 else {}
                s.update({
                    "ce_oi":        int(ce.get("openInterest") or 0),
                    "ce_oi_change": int(ce.get("changeinOpenInterest") or 0),
                    "ce_volume":    int(ce.get("totalTradedVolume") or 0),
                    "ce_iv":   round(iv, 2),
                    "ce_ltp":  round(ltp, 2),
                    "ce_delta": g.get("delta", 0),
                    "ce_gamma": g.get("gamma", 0),
                    "ce_theta": g.get("theta", 0),
                    "ce_vega":  g.get("vega", 0),
                })

            if "PE" in item:
                pe = item["PE"]
                iv  = float(pe.get("impliedVolatility") or 0)
                ltp = float(pe.get("lastPrice") or 0)
                g = bs_greeks(spot, k, r, T, iv / 100, "PE") if iv > 0 else {}
                s.update({
                    "pe_oi":        int(pe.get("openInterest") or 0),
                    "pe_oi_change": int(pe.get("changeinOpenInterest") or 0),
                    "pe_volume":    int(pe.get("totalTradedVolume") or 0),
                    "pe_iv":   round(iv, 2),
                    "pe_ltp":  round(ltp, 2),
                    "pe_delta": g.get("delta", 0),
                    "pe_gamma": g.get("gamma", 0),
                    "pe_theta": g.get("theta", 0),
                    "pe_vega":  g.get("vega", 0),
                })

        strikes_data = sorted(strike_map.values(), key=lambda x: x["strike"])

        # ── Aggregate metrics ───────────────────────────────────────────
        total_ce_oi = sum(s["ce_oi"] for s in strikes_data)
        total_pe_oi = sum(s["pe_oi"] for s in strikes_data)
        pcr_oi      = round(total_pe_oi / total_ce_oi, 3) if total_ce_oi > 0 else 1.0

        # ATM IV
        atm_row = next((s for s in strikes_data if s["strike"] == atm_strike), None)
        atm_iv  = atm_row["ce_iv"] if atm_row and atm_row["ce_iv"] > 0 else 15.0

        # ── Analysis ────────────────────────────────────────────────────
        max_pain    = calculate_max_pain(strikes_data)
        order_flow  = analyze_order_flow(strikes_data, spot)
        gex_data, total_gex, gamma_flip = calculate_gex(strikes_data, spot)
        market_str  = analyze_market_structure(strikes_data, spot, pcr_oi, total_gex, order_flow)
        signals, best_signal = generate_signals(
            spot, pcr_oi, total_gex, gamma_flip, max_pain,
            market_str, order_flow, atm_iv, vix, atm_strike
        )

        # Display strikes: ±1500 around ATM
        display = [s for s in strikes_data if abs(s["strike"] - atm_strike) <= 1500]

        # Serialize order blocks
        supply_ob = order_flow.get("supply_ob")
        demand_ob = order_flow.get("demand_ob")

        return {
            "timestamp":      datetime.now().strftime("%d %b %Y %H:%M:%S"),
            "spot":           round(spot, 2),
            "atm_strike":     int(atm_strike),
            "expiry":         near_expiry,
            "dte":            dte,
            "pcr_oi":         pcr_oi,
            "pcr_vol":        round(order_flow.get("pcr_vol", 1.0), 3),
            "total_ce_oi":    total_ce_oi,
            "total_pe_oi":    total_pe_oi,
            "atm_iv":         round(atm_iv, 2),
            "vix":            vix,
            "max_pain":       int(max_pain),
            "total_gex":      total_gex,
            "gamma_flip":     gamma_flip,
            "order_blocks": {
                "supply_strike":  supply_ob["strike"] if supply_ob else None,
                "supply_oi":      supply_ob["ce_oi"] if supply_ob else 0,
                "demand_strike":  demand_ob["strike"] if demand_ob else None,
                "demand_oi":      demand_ob["pe_oi"] if demand_ob else 0,
                "liquidity_pools": order_flow["liquidity_pools"],
            },
            "order_flow":     {
                k: v for k, v in order_flow.items()
                if k not in ("supply_ob", "demand_ob")
            },
            "market_structure": market_str,
            "gex_data":       gex_data,
            "strikes":        display,
            "signals":        signals,
            "best_signal":    best_signal,
        }

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"NSE fetch failed: {str(e)}")
    except Exception as e:
        logger.exception("Analysis error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
