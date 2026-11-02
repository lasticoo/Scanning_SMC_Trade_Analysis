#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Binance USDⓈ-M Futures - SMC (Smart Money Concepts) & Breakout Setup Scanner
==========================================================================

Memindai kontrak perpetual USDT di Binance Futures untuk menemukan:
1. SETUP SMC (Smart Money Concepts):
   - Order Block (OB) Bullish & Bearish (termasuk Sweep POI)
   - Liquidity Sweep & Liquidity Run (BSL & SSL)
   - Liquidity Range (Swing High/Low, Dealing Range, Discount & Premium %)
   - CHOCH (Change of Character) & BOS (Break of Structure)
   - Inducement (IDM) & IDM Taken/Swept
   - Supply & Demand (S&D) Zones
   - Candle Pin Bar (Rejection Wick Reversal)
   - CISD (Change in State of Delivery)
   - POI (Point of Interest) & Sniper Entry di Demand / FVG / OB Retest
2. SETUP BREAKOUT / BREAKDOWN:
   - Pola konsolidasi klasik: Ascending/Descending/Symmetrical Triangle, Wedge, Flag

- Hanya memakai endpoint publik Binance (tanpa API key, tanpa order).
- Semua sinyal berbasis candle yang SUDAH close (candle berjalan dibuang).
- Timeframe: 4H (tren utama), 1H (struktur pasar), 15M (setup & trigger).

Contoh:
    python binance.py                               # default scan SMC setup
    python binance.py --strategy smc                # scan khusus SMC
    python binance.py --strategy breakout           # scan khusus breakout konsolidasi
    python binance.py --strategy all                # scan kedua strategi sekaligus
    python binance.py --symbols BTCUSDT,ETHUSDT,SOLUSDT
    python binance.py --equity 1000 --risk-pct 1    # hitung ukuran posisi
    python binance.py --loop --telegram             # jalan terus tiap 15M + kirim Telegram
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import requests

try:
    from ai_validator import load_env
    load_env()
except Exception:
    pass

BASE_URL = "https://fapi.binance.com"
NAN = float("nan")

STABLE_BASES = {
    "USDC", "FDUSD", "TUSD", "USDP", "BUSD", "DAI", "USDD", "USDE", "PYUSD",
    "EURI", "AEUR", "XUSD", "USD1", "RLUSD", "BFUSD",
}


#  KONFIGURASI
@dataclass
class Config:
    # strategi
    strategy: str = "smc"           # "smc", "breakout", "all"
    smc_entry_score: int = 70
    smc_watch_score: int = 50
    # pola konsolidasi
    piv_len: int = 5
    min_pat: int = 15
    max_pat: int = 100
    conv_ratio: float = 0.85
    flat_tol: float = 0.6
    flag_impulse: float = 3.0
    imp_look: int = 20
    touch_tol: float = 0.25
    # breakout / breakdown
    bk_buffer: float = 0.1
    body_min: float = 0.6
    rel_vol_len: int = 20
    rel_vol_thr: float = 1.5
    ext_atr: float = 1.2
    retest_bars: int = 12
    retest_tol: float = 0.3
    event_recent_bars: int = 6      # event dianggap "aktif" jika terjadi dalam N bar 15M terakhir
    # risk management
    sl_buffer: float = 0.25
    min_risk_atr: float = 0.6
    mm_factor: float = 1.0
    min_rr: float = 2.0
    entry_score: int = 80
    watch_score: int = 65
    # konteks HTF
    htf_piv: int = 5
    # universe & likuiditas
    min_vol_usdt: float = 50e6
    max_spread_bps: float = 6.0
    min_age_days: int = 14
    top_n: int = 120
    # data
    limit_15m: int = 400
    limit_1h: int = 300
    limit_4h: int = 300
    min_bars_15m: int = 150
    workers: int = 6
    # evaluasi Full LLM (Groq)
    ai_audit: bool = True
    ai_top_n: int = 15


# =====================================================================
#  INDIKATOR
# =====================================================================
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def atr_wilder(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    out = rma(tr, n)
    out.iloc[:n] = np.nan
    return out


def rsi_wilder(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = rma(up, n) / rma(dn, n)
        out = 100 - 100 / (1 + rs)
    return out


def find_pivots(high: np.ndarray, low: np.ndarray, L: int):
    """Pivot terkonfirmasi: bar i adalah pivot jika extreme dari jendela [i-L, i+L]."""
    n = len(high)
    is_ph = np.zeros(n, dtype=bool)
    is_pl = np.zeros(n, dtype=bool)
    for i in range(L, n - L):
        wh = high[i - L:i + L + 1]
        wl = low[i - L:i + L + 1]
        if int(np.argmax(wh)) == L:
            is_ph[i] = True
        if int(np.argmin(wl)) == L:
            is_pl[i] = True
    return is_ph, is_pl


def structure_state(df: pd.DataFrame, L: int) -> int:
    """Tren struktur: +1 jika close terakhir menembus swing high, -1 jika menembus swing low."""
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    is_ph, is_pl = find_pivots(h, l, L)
    hv = lv = None
    hx = lx = True
    st = 0
    for t in range(len(df)):
        i = t - L
        if i >= L:
            if is_ph[i]:
                hv, hx = h[i], False
            if is_pl[i]:
                lv, lx = l[i], False
        if hv is not None and not hx and c[t] > hv:
            st, hx = 1, True
        if lv is not None and not lx and c[t] < lv:
            st, lx = -1, True
    return st


def htf_context(df: pd.DataFrame, L: int) -> dict:
    c = df["close"]
    e200 = ema(c, 200).iloc[-1] if len(df) >= 200 else NAN
    return {
        "st": structure_state(df, L),
        "close": float(c.iloc[-1]),
        "e20": float(ema(c, 20).iloc[-1]),
        "e50": float(ema(c, 50).iloc[-1]),
        "e200": float(e200),
        "hi": float(df["high"].iloc[-60:].max()),
        "lo": float(df["low"].iloc[-60:].min()),
        "rsi": float(rsi_wilder(c).iloc[-1]),
        "last_time": int(df["open_time"].iloc[-1]),
    }


def trend_text(cx: dict) -> str:
    if cx["close"] > cx["e50"] and cx["e20"] > cx["e50"]:
        e = "Bullish"
    elif cx["close"] < cx["e50"] and cx["e20"] < cx["e50"]:
        e = "Bearish"
    else:
        e = "Sideways"
    s = {1: "struktur naik", -1: "struktur turun", 0: "struktur netral"}[cx["st"]]
    return f"{e} ({s})"


def trend_short(cx: dict) -> str:
    if cx["close"] > cx["e50"] and cx["e20"] > cx["e50"]:
        e = "Bull"
    elif cx["close"] < cx["e50"] and cx["e20"] < cx["e50"]:
        e = "Bear"
    else:
        e = "Side"
    s = {1: "^", -1: "v", 0: "-"}[cx["st"]]
    return f"{e}|{s}"


# =====================================================================
#  FUNGSI PENILAIAN
# =====================================================================
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def f_align(d: int, cx: dict) -> float:
    """Keselarasan TF konteks dengan arah d (0..1): 50% struktur, 50% posisi vs EMA."""
    st = cx["st"]
    s_part = 1.0 if st == d else (0.4 if st == 0 else 0.0)
    c, e20, e50, e200 = cx["close"], cx["e20"], cx["e50"], cx["e200"]
    no200 = math.isnan(e200)
    e_part = 0.0
    if d == 1:
        e_part += 0.5 if c > e50 else 0.0
        e_part += 0.3 if e20 > e50 else 0.0
        e_part += 0.2 if (no200 or e50 > e200) else 0.0
    else:
        e_part += 0.5 if c < e50 else 0.0
        e_part += 0.3 if e20 < e50 else 0.0
        e_part += 0.2 if (no200 or e50 < e200) else 0.0
    return 0.5 * s_part + 0.5 * e_part


def f_quality(t, uN, uS, lN, lS, n, a, h, l, c, tol):
    """Jumlah sentuhan trendline & seberapa bersih harga berada di dalam pola."""
    tU = tL = ins = cnt = 0
    prevU = prevL = False
    for k in range(1, n + 1):
        idx = t - k
        if idx < 0:
            break
        up = uN - uS * k
        lo = lN - lS * k
        hitU = abs(h[idx] - up) <= tol * a
        hitL = abs(l[idx] - lo) <= tol * a
        if hitU and not prevU:
            tU += 1
        if hitL and not prevL:
            tL += 1
        prevU, prevL = hitU, hitL
        if c[idx] <= up + tol * a and c[idx] >= lo - tol * a:
            ins += 1
        cnt += 1
    return tU, tL, ins, cnt


def f_nearest(d: int, entry: float, a: float, arr, x1: float, x2: float) -> Optional[float]:
    """Level lawan terdekat: resistance untuk long, support untuk short."""
    gap = 0.3 * a
    vals = []
    for v in list(arr) + [x1, x2]:
        if v is None or math.isnan(v):
            continue
        if d == 1 and v > entry + gap:
            vals.append(v)
        if d == -1 and v < entry - gap:
            vals.append(v)
    if not vals:
        return None
    return min(vals) if d == 1 else max(vals)


def liquidity_score(meta: dict, d: int, atr_pct: float):
    """Skor likuiditas & eksekusi (maks 5): volume, spread, funding, volatilitas."""
    vol = meta.get("vol24", 0.0)
    spread = meta.get("spread_bps", 99.0)
    fund = meta.get("funding", 0.0)
    s_vol = 2.0 * clamp(vol / 1e8, 0.0, 1.0)
    s_spread = 1.5 if spread <= 2 else (1.0 if spread <= 5 else (0.5 if spread <= 10 else 0.0))
    crowd = d * fund      # positif = funding membebani posisi searah
    s_fund = 1.0 if crowd <= 0.0003 else (0.5 if crowd <= 0.001 else 0.0)
    s_vola = 0.5 if atr_pct <= 0.03 else 0.0
    return s_vol + s_spread + s_fund + s_vola


def f_score(d, al4, al1, q_pts, brk_pts, rv, rsi_v, above_e, rr, room_r, liq_pts, cfg: Config):
    s1 = 12.0 * al4 + 13.0 * al1
    s2 = q_pts
    s3 = brk_pts
    s_vol = 8.0 * clamp(rv / cfg.rel_vol_thr, 0.0, 1.0)
    if d == 1:
        s_rsi = 4.0 if 50 <= rsi_v <= 75 else (1.5 if rsi_v > 75 else (2.0 if rsi_v >= 40 else 0.0))
    else:
        s_rsi = 4.0 if 25 <= rsi_v <= 50 else (1.5 if rsi_v < 25 else (2.0 if rsi_v <= 60 else 0.0))
    s_mom = 3.0 if above_e else 0.0
    s4 = s_vol + s_rsi + s_mom
    s_rr = 8.0 if rr >= 2.5 else (6.5 if rr >= 2.0 else (3.0 if rr >= 1.5 else 0.0))
    s5 = s_rr + 7.0 * room_r
    s6 = liq_pts
    tot = s1 + s2 + s3 + s4 + s5 + s6
    return tot, [s1, s2, s3, s4, s5, s6]


SCORE_NAMES = [
    ("Struktur & tren", 25), ("Konsolidasi & trendline", 20), ("Konfirmasi breakout", 20),
    ("Volume & momentum", 15), ("RR & ruang target", 15), ("Likuiditas & eksekusi", 5),
]


# =====================================================================
#  STRUKTUR DATA HASIL
# =====================================================================
@dataclass
class Pat:
    ok: bool
    ptype: str
    bias: int
    uNow: float
    lNow: float
    uSlope: float
    lSlope: float
    patLen: int
    wNow: float
    wStart: float
    impulse: float


@dataclass
class Setup:
    symbol: str = ""
    direction: str = ""
    kind: str = ""
    pattern: str = ""
    bar_index: int = 0
    candle_time: str = ""
    level: float = NAN
    entry: float = NAN
    entry_zone: str = ""
    sl: float = NAN
    tp1: float = NAN
    tp2: float = NAN
    rr: float = NAN
    risk_pct: float = NAN
    tp1_pct: float = NAN
    tp2_pct: float = NAN
    score: float = 0.0
    parts: list = field(default_factory=list)
    status: str = ""
    confirmed: bool = False
    extended: bool = False
    rel_vol: float = NAN
    rsi: float = NAN
    atr: float = NAN
    atr_pct: float = NAN
    upper_line: float = NAN
    lower_line: float = NAN
    opp_level: Optional[float] = None
    trend_4h: str = ""
    struct_1h: str = ""
    trend_4h_short: str = ""
    struct_1h_short: str = ""
    confirm_txt: str = ""
    confirm_short: str = ""
    invalidation: str = ""
    notes: list = field(default_factory=list)
    # internal (disimpan untuk state retest)
    q_pts: float = 0.0
    body_pts: float = 0.0
    height: float = NAN
    # data meta & sizing
    meta: dict = field(default_factory=dict)
    last_candle: str = ""
    qty: float = NAN
    notional: float = NAN
    risk_usdt: float = NAN
    # audit AI SMC (Groq)
    ai_decision: str = ""
    ai_confidence: float = NAN
    ai_summary: str = ""
    ai_action_note: str = ""
    ai_model: str = ""
    # 10 elemen SMC institusional dari AI & data mentah Binance
    smc_elements: dict = field(default_factory=dict)
    raw_market_data: dict = field(default_factory=dict)


def fp(x) -> str:
    """Format harga adaptif."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    ax = abs(x)
    if ax == 0:
        return "0"
    dec = min(10, max(2, 5 - int(math.floor(math.log10(ax)))))
    return f"{x:.{dec}f}"


def ts_utc(ms: int) -> str:
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime("%Y-%m-%d %H:%M")


# =====================================================================
#  ANALISA SATU SIMBOL
# =====================================================================
def analyze_symbol(symbol: str, df15: pd.DataFrame, ctx4: dict, ctx1: dict, meta: dict, cfg: Config) -> Optional[Setup]:
    n = len(df15)
    if n < cfg.min_bars_15m:
        return None

    o = df15["open"].values
    h = df15["high"].values
    l = df15["low"].values
    c = df15["close"].values
    times = df15["open_time"].values
    atr_s = atr_wilder(df15).values
    vol_sma = df15["volume"].rolling(cfg.rel_vol_len).mean().shift(1)
    rel = (df15["volume"] / vol_sma).values
    ema20 = ema(df15["close"], 20).values
    rsi = rsi_wilder(df15["close"]).values
    is_ph, is_pl = find_pivots(h, l, cfg.piv_len)

    al4 = {1: f_align(1, ctx4), -1: f_align(-1, ctx4)}
    al1 = {1: f_align(1, ctx1), -1: f_align(-1, ctx1)}

    S = dict(pH1P=None, pH1I=None, pH2P=None, pH2I=None, pL1P=None, pL1I=None, pL2P=None, pL2I=None)
    piv_hi: list = []
    piv_lo: list = []
    bk = dict(dir=0, bar=None, level=None, height=None, type="", q=0.0, body=0.0, rv=0.0, retested=False)

    def pattern_at(t: int) -> Optional[Pat]:
        if None in (S["pH1P"], S["pH2P"], S["pL1P"], S["pL2P"]):
            return None
        a = atr_s[t]
        if np.isnan(a):
            return None
        uS = (S["pH2P"] - S["pH1P"]) / (S["pH2I"] - S["pH1I"])
        lS = (S["pL2P"] - S["pL1P"]) / (S["pL2I"] - S["pL1I"])
        uNow = S["pH2P"] + uS * (t - S["pH2I"])
        lNow = S["pL2P"] + lS * (t - S["pL2I"])
        start = min(S["pH1I"], S["pL1I"])
        plen = t - start
        wNow = uNow - lNow
        wStart = (uNow - uS * plen) - (lNow - lS * plen)
        spanU, spanL = uS * plen, lS * plen
        len_ok = cfg.min_pat <= plen <= cfg.max_pat
        recent_ok = (t - max(S["pH2I"], S["pL2I"])) <= max(cfg.piv_len * 8, plen / 2)
        ordered = uNow > lNow and wStart > 0
        is_conv = ordered and wNow <= wStart * cfg.conv_ratio and wNow >= 0.5 * a
        is_par = ordered and 0.75 * wStart <= wNow <= 1.3 * wStart
        flat_u = abs(spanU) <= cfg.flat_tol * a
        flat_l = abs(spanL) <= cfg.flat_tol * a

        impulse = 0.0
        if len_ok:
            i0 = t - plen
            i1 = i0 - cfg.imp_look
            if i1 >= 0:
                impulse = c[i0] - c[i1]

        ptype, bias = "-", 0
        if len_ok and recent_ok and ordered:
            if is_conv:
                if flat_u and spanL > cfg.flat_tol * a:
                    ptype, bias = "Ascending Triangle", 1
                elif flat_l and spanU < -cfg.flat_tol * a:
                    ptype, bias = "Descending Triangle", -1
                elif spanU < 0 and spanL > 0:
                    ptype, bias = "Symmetrical Triangle", 0
                elif spanU < 0 and spanL < 0:
                    ptype, bias = "Falling Wedge", 1
                elif spanU > 0 and spanL > 0:
                    ptype, bias = "Rising Wedge", -1
            elif is_par:
                if spanU < -0.2 * a and spanL < -0.2 * a and impulse >= cfg.flag_impulse * a:
                    ptype, bias = "Bullish Flag", 1
                elif spanU > 0.2 * a and spanL > 0.2 * a and impulse <= -cfg.flag_impulse * a:
                    ptype, bias = "Bearish Flag", -1
        return Pat(ptype != "-", ptype, bias, uNow, lNow, uS, lS, plen, wNow, wStart, impulse)

    def evaluate(d: int, kind: str, t: int, pat: Optional[Pat]) -> Setup:
        a = float(atr_s[t])
        use_bk = kind in ("RETEST", "AWAIT_RETEST")
        is_dev = kind == "DEVELOPING"
        utype = bk["type"] if use_bk else pat.ptype
        is_flag = utype in ("Bullish Flag", "Bearish Flag")
        lvl = bk["level"] if use_bk else (pat.uNow if d == 1 else pat.lNow)

        if kind == "AWAIT_RETEST":
            entry = lvl + d * 0.5 * cfg.retest_tol * a
        elif is_dev:
            entry = lvl + d * cfg.bk_buffer * a
        else:
            entry = float(c[t])

        # Stop Loss struktural
        if kind == "RETEST":
            sl_raw = (min(l[t], lvl) - cfg.sl_buffer * a) if d == 1 else (max(h[t], lvl) + cfg.sl_buffer * a)
        elif kind == "AWAIT_RETEST":
            sl_raw = lvl - d * cfg.sl_buffer * a
        else:
            sl_raw = (min(S["pL2P"], pat.lNow) - cfg.sl_buffer * a) if d == 1 else (max(S["pH2P"], pat.uNow) + cfg.sl_buffer * a)
        sl = min(sl_raw, entry - cfg.min_risk_atr * a) if d == 1 else max(sl_raw, entry + cfg.min_risk_atr * a)
        risk = abs(entry - sl)

        # Target: measured move, dibatasi level lawan terdekat
        hgt = bk["height"] if use_bk else (abs(pat.impulse) if is_flag else pat.wStart)
        tp_raw = entry + d * hgt * cfg.mm_factor
        opp = f_nearest(d, entry, a, piv_hi if d == 1 else piv_lo,
                        ctx4["hi"] if d == 1 else ctx4["lo"], ctx1["hi"] if d == 1 else ctx1["lo"])
        tp2 = tp_raw
        if opp is not None:
            tp2 = min(tp_raw, opp - 0.1 * a) if d == 1 else max(tp_raw, opp + 0.1 * a)
        tp2 = max(tp2, entry + 0.05 * a) if d == 1 else min(tp2, entry - 0.05 * a)
        tp1 = entry + 0.5 * (tp2 - entry)
        rr = abs(tp2 - entry) / risk
        room_r = 1.0 if opp is None else clamp(abs(opp - entry) / max(abs(tp_raw - entry), 1e-12), 0.0, 1.0)

        # kualitas pola & konfirmasi
        ext_flag = False
        if use_bk:
            q_pts, body_pts = bk["q"], bk["body"]
            rv_use = bk["rv"]
            brk_pts = 6.0 + body_pts + 4.0 + (5.0 if kind == "RETEST" else 0.0)
        else:
            rv_use = 0.0 if np.isnan(rel[t]) else float(rel[t])
            tU, tL, insC, cnt = f_quality(t, pat.uNow, pat.uSlope, pat.lNow, pat.lSlope, pat.patLen, a, h, l, c, cfg.touch_tol)
            touch_pts = min(tU, 4) + min(tL, 4)
            ins_ratio = insC / cnt if cnt > 0 else 0.0
            ins_pts = 6.0 * clamp((ins_ratio - 0.7) / 0.3, 0.0, 1.0)
            if is_flag:
                conv_pts = 6.0 * clamp(1.0 - abs(pat.wNow / pat.wStart - 1.0) / 0.3, 0.0, 1.0)
            else:
                conv_pts = 6.0 * clamp((1.0 - pat.wNow / pat.wStart) / 0.5, 0.0, 1.0)
            q_pts = touch_pts + ins_pts + conv_pts
            body_pts, brk_pts = 0.0, 0.0
            if not is_dev:
                rng = h[t] - l[t]
                body = abs(c[t] - o[t])
                close_pos = ((c[t] - l[t]) / rng if d == 1 else (h[t] - c[t]) / rng) if rng > 0 else 0.5
                body_pts = (3.0 if body >= cfg.body_min * a else 3.0 * body / (cfg.body_min * a)) + (2.0 if close_pos >= 0.6 else 0.0)
                ext_flag = (c[t] - lvl > cfg.ext_atr * a) if d == 1 else (lvl - c[t] > cfg.ext_atr * a)
                brk_pts = 6.0 + body_pts + (0.0 if ext_flag else 4.0)

        rsi_v = 50.0 if np.isnan(rsi[t]) else float(rsi[t])
        above_e = (c[t] > ema20[t]) if d == 1 else (c[t] < ema20[t])
        atr_pct = a / float(c[t])
        liq_pts = liquidity_score(meta, d, atr_pct)
        tot, parts = f_score(d, al4[d], al1[d], q_pts, brk_pts, rv_use, rsi_v, above_e, rr, room_r, liq_pts, cfg)

        # status
        if is_dev:
            status = "WATCHLIST (belum break)" if tot >= cfg.watch_score else "NO SETUP"
        elif kind == "AWAIT_RETEST":
            status = "WATCHLIST (tunggu retest)" if tot >= cfg.watch_score else "NO SETUP"
        elif ext_flag:
            status = "EXTENDED"
        elif tot >= cfg.entry_score and rr >= cfg.min_rr:
            status = "ENTRY READY"
        elif tot >= cfg.watch_score:
            status = "WATCHLIST"
        else:
            status = "REJECT"

        # teks
        side = "atas" if d == 1 else "bawah"
        if is_dev:
            confirm = f"Belum break. Garis {side} di {fp(lvl)} | RelVol {rv_use:.1f}x"
            conf_short = f"belum break (garis {fp(lvl)})"
            zone = f"> {fp(lvl)}" if d == 1 else f"< {fp(lvl)}"
        elif kind == "AWAIT_RETEST":
            confirm = f"Breakout sudah terjadi di {fp(lvl)}, menunggu retest | RelVol breakout {rv_use:.1f}x"
            conf_short = f"tunggu retest {fp(lvl)}"
            zone = f"{fp(lvl)} - {fp(lvl + d * cfg.retest_tol * a)}"
        elif kind == "RETEST":
            confirm = f"Retest ditahan di {fp(lvl)} | RelVol breakout {rv_use:.1f}x"
            conf_short = f"retest ditahan {fp(lvl)}"
            zone = fp(entry)
        else:
            body_atr = abs(c[t] - o[t]) / a
            confirm = (f"Close {'di atas' if d == 1 else 'di bawah'} garis {fp(lvl)} | RelVol {rv_use:.1f}x | "
                       f"Body {body_atr:.1f} ATR | RSI {rsi_v:.0f}")
            conf_short = f"close {'>' if d == 1 else '<'} garis, RV {rv_use:.1f}x"
            zone = fp(entry)

        if kind in ("BREAKOUT", "BREAKDOWN"):
            inval = (f"Close 15M kembali {'di bawah' if d == 1 else 'di atas'} {fp(lvl)} (masuk lagi ke pola) "
                     f"atau harga menyentuh SL {fp(sl)}")
        elif use_bk:
            inval = (f"Close 15M {'di bawah' if d == 1 else 'di atas'} {fp(lvl)} (retest gagal) atau SL {fp(sl)} tersentuh")
        else:
            inval = (f"Close 15M {'di bawah garis bawah' if d == 1 else 'di atas garis atas'} pola, atau pola melebar/rusak. "
                     f"Setelah break: close kembali ke dalam pola atau SL {fp(sl)} tersentuh")

        st_ = Setup(
            symbol=symbol, direction="LONG" if d == 1 else "SHORT", kind=kind, pattern=utype,
            bar_index=t, candle_time=ts_utc(int(times[t])), level=float(lvl), entry=float(entry), entry_zone=zone,
            sl=float(sl), tp1=float(tp1), tp2=float(tp2), rr=float(rr),
            risk_pct=risk / entry * 100, tp1_pct=abs(tp1 - entry) / entry * 100, tp2_pct=abs(tp2 - entry) / entry * 100,
            score=float(tot), parts=[(SCORE_NAMES[i][0], float(parts[i]), SCORE_NAMES[i][1]) for i in range(6)],
            status=status, confirmed=kind in ("BREAKOUT", "BREAKDOWN", "RETEST"), extended=ext_flag,
            rel_vol=rv_use, rsi=rsi_v, atr=a, atr_pct=atr_pct,
            upper_line=float(pat.uNow) if pat else NAN, lower_line=float(pat.lNow) if pat else NAN,
            opp_level=opp,
            trend_4h=trend_text(ctx4), struct_1h=trend_text(ctx1),
            trend_4h_short=trend_short(ctx4), struct_1h_short=trend_short(ctx1),
            confirm_txt=confirm, confirm_short=conf_short, invalidation=inval,
            q_pts=float(q_pts), body_pts=float(body_pts), height=float(hgt), meta=dict(meta),
            last_candle=ts_utc(int(times[-1])),
        )
        if np.isnan(ctx4["e200"]):
            st_.notes.append("EMA200 4H tidak tersedia (histori kurang dari 200 candle), poin terkait dikurangi.")
        if opp is None:
            st_.notes.append("Tidak ditemukan resistance/support lawan pada data yang diambil; ruang target dianggap penuh (belum tentu benar).")
        return st_

    events: list = []
    last_evt: Optional[int] = None
    last_pat: Optional[Pat] = None
    L = cfg.piv_len

    for t in range(n):
        i = t - L
        if i >= L:
            if is_ph[i]:
                S["pH1P"], S["pH1I"], S["pH2P"], S["pH2I"] = S["pH2P"], S["pH2I"], float(h[i]), i
                piv_hi.append(float(h[i]))
                if len(piv_hi) > 25:
                    piv_hi.pop(0)
            if is_pl[i]:
                S["pL1P"], S["pL1I"], S["pL2P"], S["pL2I"] = S["pL2P"], S["pL2I"], float(l[i]), i
                piv_lo.append(float(l[i]))
                if len(piv_lo) > 25:
                    piv_lo.pop(0)

        a = atr_s[t]
        if t < 1 or np.isnan(a):
            continue
        pat = pattern_at(t)
        last_pat = pat

        # state breakout: kadaluarsa atau gagal (harga kembali ke dalam pola)
        if bk["dir"] != 0:
            if t - bk["bar"] > cfg.retest_bars:
                bk["dir"] = 0
            elif t > bk["bar"] and bk["dir"] == 1 and c[t] < bk["level"] - cfg.bk_buffer * a:
                bk["dir"] = 0
            elif t > bk["bar"] and bk["dir"] == -1 and c[t] > bk["level"] + cfg.bk_buffer * a:
                bk["dir"] = 0

        kind, d = None, 0
        if pat is not None and pat.ok and bk["dir"] == 0:
            u_prev = pat.uNow - pat.uSlope
            l_prev = pat.lNow - pat.lSlope
            if pat.bias >= 0 and c[t] > pat.uNow + cfg.bk_buffer * a and c[t - 1] <= u_prev + cfg.bk_buffer * a:
                kind, d = "BREAKOUT", 1
            elif pat.bias <= 0 and c[t] < pat.lNow - cfg.bk_buffer * a and c[t - 1] >= l_prev - cfg.bk_buffer * a:
                kind, d = "BREAKDOWN", -1
        if kind is None and bk["dir"] != 0 and not bk["retested"] and t - bk["bar"] >= 1:
            if bk["dir"] == 1 and l[t] <= bk["level"] + cfg.retest_tol * a and c[t] > bk["level"] and c[t] > o[t]:
                kind, d = "RETEST", 1
            elif bk["dir"] == -1 and h[t] >= bk["level"] - cfg.retest_tol * a and c[t] < bk["level"] and c[t] < o[t]:
                kind, d = "RETEST", -1

        if kind is not None:
            s = evaluate(d, kind, t, pat)
            if kind in ("BREAKOUT", "BREAKDOWN"):
                bk.update(dir=d, bar=t, level=s.level, height=s.height, type=s.pattern,
                          q=s.q_pts, body=s.body_pts, rv=s.rel_vol, retested=False)
            else:
                bk["retested"] = True
            last_evt = t
            events.append(s)

    recent = [e for e in events if e.bar_index >= n - cfg.event_recent_bars]
    if recent:
        return recent[-1]

    t = n - 1
    if np.isnan(atr_s[t]):
        return None
    if bk["dir"] != 0 and not bk["retested"]:
        return evaluate(bk["dir"], "AWAIT_RETEST", t, None)
    if last_pat is not None and last_pat.ok and (last_evt is None or t - last_evt > 30):
        if last_pat.bias == 0:
            d = 1 if (al4[1] + al1[1]) >= (al4[-1] + al1[-1]) else -1
        else:
            d = last_pat.bias
        return evaluate(d, "DEVELOPING", t, last_pat)
    return None


# =====================================================================
#  ANALISA SMC (SMART MONEY CONCEPTS)
# =====================================================================
def analyze_smc_symbol(symbol: str, df15: pd.DataFrame, ctx4: dict, ctx1: dict, meta: dict, cfg: Config) -> Optional[Setup]:
    n = len(df15)
    if n < cfg.min_bars_15m:
        return None

    o = df15["open"].values
    h = df15["high"].values
    l = df15["low"].values
    c = df15["close"].values
    times = df15["open_time"].values
    atr_s = atr_wilder(df15).values
    cur_atr = float(atr_s[-1])
    cur_p = float(c[-1])
    if np.isnan(cur_atr) or cur_atr <= 0:
        return None

    vol_sma = df15["volume"].rolling(cfg.rel_vol_len).mean().shift(1)
    rel = (df15["volume"] / vol_sma).values
    cur_rv = 1.0 if np.isnan(rel[-1]) else float(rel[-1])
    rsi_s = rsi_wilder(df15["close"]).values
    cur_rsi = 50.0 if np.isnan(rsi_s[-1]) else float(rsi_s[-1])

    # 1. Swings (Pivots) - Liquidity Range & Dealing Range
    piv_len = cfg.piv_len
    is_ph, is_pl = find_pivots(h, l, piv_len)
    highs = [{"idx": i, "price": float(h[i]), "time": int(times[i])} for i in range(n) if is_ph[i]]
    lows = [{"idx": i, "price": float(l[i]), "time": int(times[i])} for i in range(n) if is_pl[i]]
    if not highs or not lows:
        return None

    recent_h_pool = [s["price"] for s in highs[-4:]]
    recent_l_pool = [s["price"] for s in lows[-4:]]
    range_high = max(recent_h_pool)
    range_low = min(recent_l_pool)
    eq_price = (range_high + range_low) / 2.0
    discount_pct = (cur_p - range_low) / max(range_high - range_low, 1e-12) * 100.0
    in_discount = cur_p < eq_price
    in_premium = cur_p > eq_price

    # 2. Fair Value Gaps (FVG / Imbalance)
    fvgs = []
    for i in range(2, n):
        if l[i] > h[i - 2]:
            fvgs.append({"idx": i, "dir": 1, "top": float(l[i]), "bot": float(h[i - 2]), "time": int(times[i])})
        elif h[i] < l[i - 2]:
            fvgs.append({"idx": i, "dir": -1, "top": float(l[i - 2]), "bot": float(h[i]), "time": int(times[i])})

    # 3. Pin Bars (Rejection Candles)
    pin_bars = []
    for i in range(n):
        rng = h[i] - l[i]
        if rng <= 0:
            continue
        body = abs(c[i] - o[i])
        l_wick = min(o[i], c[i]) - l[i]
        u_wick = h[i] - max(o[i], c[i])
        if l_wick >= 0.50 * rng and u_wick <= 0.25 * rng and body <= 0.35 * rng:
            pin_bars.append({"idx": i, "dir": 1, "wick_pct": float(l_wick / rng), "time": int(times[i])})
        elif u_wick >= 0.50 * rng and l_wick <= 0.25 * rng and body <= 0.35 * rng:
            pin_bars.append({"idx": i, "dir": -1, "wick_pct": float(u_wick / rng), "time": int(times[i])})

    # 4. Order Blocks (OB) & Sweep POI
    obs = []
    for i in range(1, n - 2):
        # Bullish OB: candle i merah, diikuti ekspansi kuat naik
        if c[i] < o[i] and c[i + 1] > h[i]:
            has_fvg = any(f["idx"] in (i + 1, i + 2) and f["dir"] == 1 for f in fvgs)
            expansion = (c[i + 1] - o[i + 1]) >= (h[i] - l[i]) * 1.0
            if has_fvg or expansion:
                swept = l[i] < l[i - 1]
                obs.append({"idx": i, "dir": 1, "top": float(max(o[i], h[i])), "bot": float(l[i]),
                            "swept": swept, "time": int(times[i])})
        # Bearish OB: candle i hijau, diikuti ekspansi kuat turun
        if c[i] > o[i] and c[i + 1] < l[i]:
            has_fvg = any(f["idx"] in (i + 1, i + 2) and f["dir"] == -1 for f in fvgs)
            expansion = (o[i + 1] - c[i + 1]) >= (h[i] - l[i]) * 1.0
            if has_fvg or expansion:
                swept = h[i] > h[i - 1]
                obs.append({"idx": i, "dir": -1, "top": float(h[i]), "bot": float(min(o[i], l[i])),
                            "swept": swept, "time": int(times[i])})

    active_bull_obs = [ob for ob in obs if ob["dir"] == 1 and not any(c[j] < ob["bot"] for j in range(ob["idx"] + 1, n))]
    active_bear_obs = [ob for ob in obs if ob["dir"] == -1 and not any(c[j] > ob["top"] for j in range(ob["idx"] + 1, n))]

    # 5. Struktur & Liquidity Events (BOS, CHOCH, Liquidity Sweep & Run)
    lookback = min(40, n - 1)
    struct_events = []
    trend = 0
    for t in range(n - lookback, n):
        known_h = [s for s in highs if s["idx"] <= t - piv_len]
        known_l = [s for s in lows if s["idx"] <= t - piv_len]
        if known_h:
            last_h = known_h[-1]["price"]
            if h[t] > last_h and c[t] <= last_h and h[t - 1] <= last_h:
                struct_events.append({"type": "SWEEP", "dir": -1, "idx": t, "level": last_h, "detail": "BSL Sweep"})
            elif c[t] > last_h and c[t - 1] <= last_h:
                ev_name = "BOS" if trend == 1 else "CHOCH"
                struct_events.append({"type": ev_name, "dir": 1, "idx": t, "level": last_h, "detail": f"Bullish {ev_name}"})
                trend = 1
        if known_l:
            last_l = known_l[-1]["price"]
            if l[t] < last_l and c[t] >= last_l and l[t - 1] >= last_l:
                struct_events.append({"type": "SWEEP", "dir": 1, "idx": t, "level": last_l, "detail": "SSL Sweep"})
            elif c[t] < last_l and c[t - 1] >= last_l:
                ev_name = "BOS" if trend == -1 else "CHOCH"
                struct_events.append({"type": ev_name, "dir": -1, "idx": t, "level": last_l, "detail": f"Bearish {ev_name}"})
                trend = -1

    # CISD (Change in State of Delivery)
    for t in range(n - 20, n):
        if c[t - 2] < o[t - 2] and c[t - 1] < o[t - 1] and c[t] > o[t - 2] and c[t - 1] <= o[t - 2]:
            struct_events.append({"type": "CISD", "dir": 1, "idx": t, "level": float(o[t - 2]), "detail": "Bullish CISD"})
        if c[t - 2] > o[t - 2] and c[t - 1] > o[t - 1] and c[t] < o[t - 2] and c[t - 1] >= o[t - 2]:
            struct_events.append({"type": "CISD", "dir": -1, "idx": t, "level": float(o[t - 2]), "detail": "Bearish CISD"})

    # Inducement (IDM)
    idm_bull_swept = False
    idm_bear_swept = False
    if highs and lows:
        if trend == 1:
            sh_idx = highs[-1]["idx"]
            pullbacks = [k for k in range(sh_idx + 1, n - 1) if l[k] < l[k - 1]]
            if pullbacks:
                idm_bull_swept = any(l[j] <= l[pullbacks[0]] for j in range(pullbacks[0] + 1, n))
        elif trend == -1:
            sl_idx = lows[-1]["idx"]
            pullbacks = [k for k in range(sl_idx + 1, n - 1) if h[k] > h[k - 1]]
            if pullbacks:
                idm_bear_swept = any(h[j] >= h[pullbacks[0]] for j in range(pullbacks[0] + 1, n))

    # Konfluensi terkini (15 bar terakhir)
    recent_ev = [e for e in struct_events if e["idx"] >= n - 15]
    recent_pins = [p for p in pin_bars if p["idx"] >= n - 5]

    has_bull_sweep = any(e["type"] == "SWEEP" and e["dir"] == 1 for e in recent_ev)
    has_bull_choch = any(e["type"] == "CHOCH" and e["dir"] == 1 for e in recent_ev)
    has_bull_bos = any(e["type"] == "BOS" and e["dir"] == 1 for e in recent_ev)
    has_bull_cisd = any(e["type"] == "CISD" and e["dir"] == 1 for e in recent_ev)
    has_bull_pin = any(p["dir"] == 1 for p in recent_pins)

    poi_bull = None
    poi_type_bull = ""
    for ob in reversed(active_bull_obs):
        if l[-1] <= ob["top"] + 0.3 * cur_atr and c[-1] >= ob["bot"] - 0.2 * cur_atr:
            poi_bull = ob
            poi_type_bull = "Bullish OB (Sweep POI)" if ob["swept"] else "Bullish Order Block"
            break
    if not poi_bull:
        for f in reversed([f for f in fvgs if f["dir"] == 1 and f["idx"] >= n - 35]):
            if l[-1] <= f["top"] + 0.2 * cur_atr and c[-1] >= f["bot"] - 0.2 * cur_atr:
                poi_bull = f
                poi_type_bull = "Bullish Fair Value Gap (FVG)"
                break

    has_bear_sweep = any(e["type"] == "SWEEP" and e["dir"] == -1 for e in recent_ev)
    has_bear_choch = any(e["type"] == "CHOCH" and e["dir"] == -1 for e in recent_ev)
    has_bear_bos = any(e["type"] == "BOS" and e["dir"] == -1 for e in recent_ev)
    has_bear_cisd = any(e["type"] == "CISD" and e["dir"] == -1 for e in recent_ev)
    has_bear_pin = any(p["dir"] == -1 for p in recent_pins)

    poi_bear = None
    poi_type_bear = ""
    for ob in reversed(active_bear_obs):
        if h[-1] >= ob["bot"] - 0.3 * cur_atr and c[-1] <= ob["top"] + 0.2 * cur_atr:
            poi_bear = ob
            poi_type_bear = "Bearish OB (Sweep POI)" if ob["swept"] else "Bearish Order Block"
            break
    if not poi_bear:
        for f in reversed([f for f in fvgs if f["dir"] == -1 and f["idx"] >= n - 35]):
            if h[-1] >= f["bot"] - 0.2 * cur_atr and c[-1] <= f["top"] + 0.2 * cur_atr:
                poi_bear = f
                poi_type_bear = "Bearish Fair Value Gap (FVG)"
                break

    # Penilaian keselarasan HTF (4H & 1H)
    def htf_score(d_dir: int) -> float:
        al4 = 10.0 if (ctx4["st"] == d_dir or (ctx4["close"] > ctx4["e50"] and d_dir == 1) or (ctx4["close"] < ctx4["e50"] and d_dir == -1)) else 4.0
        al1 = 10.0 if (ctx1["st"] == d_dir or (ctx1["close"] > ctx1["e50"] and d_dir == 1) or (ctx1["close"] < ctx1["e50"] and d_dir == -1)) else 4.0
        return al4 + al1

    bull_htf = htf_score(1)
    bear_htf = htf_score(-1)

    bull_poi_score = (15.0 if poi_bull else 0.0) + (5.0 if poi_bull and poi_bull.get("swept") else 0.0) + (10.0 if in_discount else 0.0)
    bear_poi_score = (15.0 if poi_bear else 0.0) + (5.0 if poi_bear and poi_bear.get("swept") else 0.0) + (10.0 if in_premium else 0.0)

    bull_struct_score = (12.0 if has_bull_sweep else 0.0) + (10.0 if has_bull_choch else (7.0 if has_bull_bos else 0.0)) + (5.0 if idm_bull_swept else 0.0)
    bear_struct_score = (12.0 if has_bear_sweep else 0.0) + (10.0 if has_bear_choch else (7.0 if has_bear_bos else 0.0)) + (5.0 if idm_bear_swept else 0.0)

    bull_trigger_score = (10.0 if has_bull_pin else 0.0) + (8.0 if has_bull_cisd else 0.0) + (5.0 if cur_rv >= 1.4 else 0.0)
    bear_trigger_score = (10.0 if has_bear_pin else 0.0) + (8.0 if has_bear_cisd else 0.0) + (5.0 if cur_rv >= 1.4 else 0.0)

    total_bull = bull_htf * 0.7 + bull_poi_score + bull_struct_score + bull_trigger_score
    total_bear = bear_htf * 0.7 + bear_poi_score + bear_struct_score + bear_trigger_score

    d = 0
    if total_bull >= 40.0 and total_bull >= total_bear:
        d = 1
    elif total_bear >= 40.0:
        d = -1
    else:
        return None

    entry = cur_p
    notes = []
    confluences = []

    if d == 1:
        htf_pts = bull_htf
        poi_pts = bull_poi_score
        struct_pts = bull_struct_score
        trig_pts = bull_trigger_score

        min_sl_base = min(cur_p, float(l[-1]))
        if poi_bull:
            min_sl_base = min(min_sl_base, poi_bull["bot"])
            confluences.append(poi_type_bull)
            top_lvl, bot_lvl = poi_bull["top"], poi_bull["bot"]
        else:
            top_lvl, bot_lvl = eq_price, range_low
            confluences.append("Demand Zone")

        sl = min_sl_base - cfg.sl_buffer * cur_atr
        sl = min(sl, entry - cfg.min_risk_atr * cur_atr)
        risk = abs(entry - sl)

        tp1 = eq_price if cur_p < eq_price - 0.5 * cur_atr else entry + 2.0 * risk
        tp2 = range_high
        if tp2 <= entry + 1.5 * risk:
            tp2 = entry + 3.0 * risk
        opp_lvl = range_high

        if in_discount:
            confluences.append(f"Discount Zone ({discount_pct:.1f}%)")
        if has_bull_sweep:
            confluences.append("SSL Liquidity Sweep")
        if has_bull_choch:
            confluences.append("Bullish CHOCH")
        if has_bull_bos:
            confluences.append("Bullish BOS")
        if idm_bull_swept:
            confluences.append("Inducement (IDM) Taken")
        if has_bull_cisd:
            confluences.append("Bullish CISD")
        if has_bull_pin:
            confluences.append("Bullish Pin Bar Rejection")

        if has_bull_sweep and (has_bull_choch or has_bull_cisd) and poi_bull:
            pat_name = "SMC Sniper Buy (Sweep + CHOCH + POI Tap)"
            kind_name = "SMC_SNIPER"
        elif poi_bull and "OB" in poi_type_bull:
            pat_name = "SMC Demand Bounce (Order Block Retest)"
            kind_name = "SMC_OB_RETEST"
        elif poi_bull and "FVG" in poi_type_bull:
            pat_name = "SMC FVG Retest Entry (Fair Value Gap)"
            kind_name = "SMC_FVG_FILL"
        elif has_bull_sweep:
            pat_name = "SMC SSL Liquidity Sweep Reversal"
            kind_name = "SMC_SWEEP"
        else:
            pat_name = "SMC Demand Setup"
            kind_name = "SMC_DEMAND"

        if poi_bull:
            inval = f"Close 15M di bawah level SL {fp(sl)} atau tembus bawah POI {fp(bot_lvl)}."
            zone_str = f"{fp(bot_lvl)} - {fp(top_lvl)}"
        else:
            inval = f"Close 15M di bawah level SL {fp(sl)}."
            zone_str = fp(entry)
        conf_short = ", ".join(confluences[:3])
        conf_txt = f"Setup SMC di area Discount ({discount_pct:.1f}%): " + ", ".join(confluences)

    else:  # SHORT
        htf_pts = bear_htf
        poi_pts = bear_poi_score
        struct_pts = bear_struct_score
        trig_pts = bear_trigger_score

        max_sl_base = max(cur_p, float(h[-1]))
        if poi_bear:
            max_sl_base = max(max_sl_base, poi_bear["top"])
            confluences.append(poi_type_bear)
            top_lvl, bot_lvl = poi_bear["top"], poi_bear["bot"]
        else:
            top_lvl, bot_lvl = range_high, eq_price
            confluences.append("Supply Zone")

        sl = max_sl_base + cfg.sl_buffer * cur_atr
        sl = max(sl, entry + cfg.min_risk_atr * cur_atr)
        risk = abs(entry - sl)

        tp1 = eq_price if cur_p > eq_price + 0.5 * cur_atr else entry - 2.0 * risk
        tp2 = range_low
        if tp2 >= entry - 1.5 * risk:
            tp2 = entry - 3.0 * risk
        opp_lvl = range_low

        if in_premium:
            confluences.append(f"Premium Zone ({discount_pct:.1f}%)")
        if has_bear_sweep:
            confluences.append("BSL Liquidity Sweep")
        if has_bear_choch:
            confluences.append("Bearish CHOCH")
        if has_bear_bos:
            confluences.append("Bearish BOS")
        if idm_bear_swept:
            confluences.append("Inducement (IDM) Taken")
        if has_bear_cisd:
            confluences.append("Bearish CISD")
        if has_bear_pin:
            confluences.append("Bearish Pin Bar Rejection")

        if has_bear_sweep and (has_bear_choch or has_bear_cisd) and poi_bear:
            pat_name = "SMC Sniper Short (Sweep + CHOCH + POI Tap)"
            kind_name = "SMC_SNIPER"
        elif poi_bear and "OB" in poi_type_bear:
            pat_name = "SMC Supply Rejection (Order Block Retest)"
            kind_name = "SMC_OB_RETEST"
        elif poi_bear and "FVG" in poi_type_bear:
            pat_name = "SMC FVG Retest Entry (Fair Value Gap)"
            kind_name = "SMC_FVG_FILL"
        elif has_bear_sweep:
            pat_name = "SMC BSL Liquidity Sweep Reversal"
            kind_name = "SMC_SWEEP"
        else:
            pat_name = "SMC Supply Setup"
            kind_name = "SMC_SUPPLY"

        if poi_bear:
            inval = f"Close 15M di atas level SL {fp(sl)} atau tembus atas POI {fp(top_lvl)}."
            zone_str = f"{fp(bot_lvl)} - {fp(top_lvl)}"
        else:
            inval = f"Close 15M di atas level SL {fp(sl)}."
            zone_str = fp(entry)
        conf_short = ", ".join(confluences[:3])
        conf_txt = f"Setup SMC di area Premium ({discount_pct:.1f}%): " + ", ".join(confluences)

    rr = abs(tp2 - entry) / max(risk, 1e-12)
    rr_pts = 10.0 if rr >= 3.0 else (8.0 if rr >= 2.5 else (6.0 if rr >= 2.0 else 2.0))
    atr_pct = cur_atr / entry
    liq_pts = liquidity_score(meta, d, atr_pct)

    p1 = min(20.0, htf_pts)
    p2 = min(25.0, poi_pts)
    p3 = min(25.0, struct_pts)
    p4 = min(15.0, trig_pts)
    p5 = min(10.0, rr_pts)
    p6 = min(5.0, liq_pts)
    tot_score = p1 + p2 + p3 + p4 + p5 + p6

    parts = [
        ("Konteks HTF & Tren", p1, 20),
        ("Kualitas POI & Zone", p2, 25),
        ("Struktur & Liquidity", p3, 25),
        ("Trigger & Rejection", p4, 15),
        ("Risk to Reward", p5, 10),
        ("Likuiditas Pasar", p6, 5),
    ]
    status = "ENTRY READY" if tot_score >= cfg.smc_entry_score and rr >= cfg.min_rr else ("WATCHLIST" if tot_score >= cfg.smc_watch_score else "REJECT")

    raw_data = {
        "current_price": cur_p,
        "range_high": range_high,
        "range_low": range_low,
        "equilibrium": eq_price,
        "discount_pct": discount_pct,
        "recent_highs": [h["price"] for h in highs[-3:]],
        "recent_lows": [l["price"] for l in lows[-3:]],
        "bull_obs": [{"top": ob["top"], "bot": ob["bot"], "swept": ob["swept"]} for ob in active_bull_obs[-2:]],
        "bear_obs": [{"top": ob["top"], "bot": ob["bot"], "swept": ob["swept"]} for ob in active_bear_obs[-2:]],
        "fvgs": [{"dir": f["dir"], "top": f["top"], "bot": f["bot"]} for f in fvgs[-3:]],
        "pin_bars": [{"dir": p["dir"], "wick_pct": p["wick_pct"]} for p in pin_bars[-3:]],
        "struct_events": [e["detail"] for e in struct_events[-5:]],
        "idm_bull_swept": idm_bull_swept,
        "idm_bear_swept": idm_bear_swept,
        "trend_4h": trend_text(ctx4),
        "trend_1h": trend_text(ctx1),
        "atr": cur_atr,
        "vol24_usdt": meta.get("vol24", 0),
        "funding_rate": meta.get("funding", 0.0),
    }

    st = Setup(
        symbol=symbol,
        direction="LONG" if d == 1 else "SHORT",
        kind=kind_name,
        pattern=pat_name,
        bar_index=n - 1,
        candle_time=ts_utc(int(times[-1])),
        level=float(eq_price),
        entry=float(entry),
        entry_zone=zone_str,
        sl=float(sl),
        tp1=float(tp1),
        tp2=float(tp2),
        rr=float(rr),
        risk_pct=abs(entry - sl) / entry * 100.0,
        tp1_pct=abs(tp1 - entry) / entry * 100.0,
        tp2_pct=abs(tp2 - entry) / entry * 100.0,
        score=float(tot_score),
        parts=parts,
        status=status,
        confirmed=True,
        extended=False,
        rel_vol=cur_rv,
        rsi=cur_rsi,
        atr=cur_atr,
        atr_pct=atr_pct,
        upper_line=float(top_lvl),
        lower_line=float(bot_lvl),
        opp_level=float(opp_lvl),
        trend_4h=trend_text(ctx4),
        struct_1h=trend_text(ctx1),
        trend_4h_short=trend_short(ctx4),
        struct_1h_short=trend_short(ctx1),
        confirm_txt=conf_txt,
        confirm_short=conf_short,
        invalidation=inval,
        notes=notes,
        meta=dict(meta),
        last_candle=ts_utc(int(times[-1])),
        raw_market_data=raw_data,
    )
    return st


# =====================================================================
#  POSITION SIZING (opsional)
# =====================================================================
def size_position(s: Setup, equity: float, risk_pct: float, fee_pct: float) -> None:
    step = s.meta.get("step", 0.0)
    min_notional = s.meta.get("min_notional", 5.0)
    risk_amt = equity * risk_pct / 100.0
    per_unit = abs(s.entry - s.sl) + (fee_pct / 100.0) * (s.entry + s.sl)   # fee taker masuk + keluar
    if per_unit <= 0:
        return
    qty = risk_amt / per_unit
    if step and step > 0:
        qty = math.floor(qty / step) * step
    s.qty = qty
    s.notional = qty * s.entry
    s.risk_usdt = qty * per_unit
    if s.notional < min_notional:
        s.notes.append(f"Notional {s.notional:.2f} USDT di bawah minimum kontrak ({min_notional}); ukuran posisi tidak valid.")


# =====================================================================
#  BINANCE API
# =====================================================================
class Binance:
    def __init__(self, timeout: int = 15):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": "setup-scanner/1.0"})
        self.timeout = timeout
        self.used_weight = 0
        self.lock = threading.Lock()

    def get(self, path: str, params: Optional[dict] = None):
        url = BASE_URL + path
        for attempt in range(6):
            with self.lock:
                heavy = self.used_weight > 2000
            if heavy:
                time.sleep(8)
                with self.lock:
                    self.used_weight = 0
            try:
                r = self.s.get(url, params=params, timeout=self.timeout)
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            w = r.headers.get("X-MBX-USED-WEIGHT-1M")
            if w and w.isdigit():
                with self.lock:
                    self.used_weight = int(w)
            if r.status_code in (418, 429):
                wait = int(r.headers.get("Retry-After", 2 ** attempt * 5))
                time.sleep(min(wait, 120))
                continue
            if r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"Gagal request {path} setelah beberapa percobaan")

    def server_time(self) -> int:
        return int(self.get("/fapi/v1/time")["serverTime"])

    def klines(self, symbol: str, interval: str, limit: int, server_ms: int) -> pd.DataFrame:
        raw = self.get("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})
        df = pd.DataFrame(raw, columns=["open_time", "open", "high", "low", "close", "volume", "close_time",
                                        "quote_volume", "trades", "tb_base", "tb_quote", "ignore"])
        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[col] = df[col].astype(float)
        df["open_time"] = df["open_time"].astype("int64")
        df["close_time"] = df["close_time"].astype("int64")
        df = df[df["close_time"] < server_ms].reset_index(drop=True)   # buang candle yang belum close
        return df


def build_universe(api: Binance, cfg: Config, only: Optional[list] = None):
    info = api.get("/fapi/v1/exchangeInfo")
    now_ms = int(time.time() * 1000)
    syms = {}
    for s in info["symbols"]:
        if s.get("contractType") != "PERPETUAL" or s.get("quoteAsset") != "USDT" or s.get("status") != "TRADING":
            continue
        if s.get("baseAsset") in STABLE_BASES or s.get("underlyingType") == "INDEX":
            continue
        onboard = s.get("onboardDate", 0)
        if onboard and (now_ms - onboard) / 86_400_000 < cfg.min_age_days:
            continue
        flt = {f["filterType"]: f for f in s.get("filters", [])}
        syms[s["symbol"]] = {
            "base": s["baseAsset"],
            "step": float(flt.get("LOT_SIZE", {}).get("stepSize", 0) or 0),
            "min_notional": float(flt.get("MIN_NOTIONAL", {}).get("notional", 5) or 5),
        }
    tick = {t["symbol"]: t for t in api.get("/fapi/v1/ticker/24hr")}
    book = {b["symbol"]: b for b in api.get("/fapi/v1/ticker/bookTicker")}
    prem = {p["symbol"]: p for p in api.get("/fapi/v1/premiumIndex")}

    rows = []
    for sym, m in syms.items():
        if only and sym not in only:
            continue
        t, b, p = tick.get(sym), book.get(sym), prem.get(sym)
        if not t or not b or not p:
            continue
        bid, ask = float(b["bidPrice"]), float(b["askPrice"])
        if bid <= 0 or ask <= 0:
            continue
        mid = (bid + ask) / 2
        m = dict(m)
        m["vol24"] = float(t["quoteVolume"])
        m["spread_bps"] = (ask - bid) / mid * 1e4
        m["funding"] = float(p.get("lastFundingRate") or 0.0)
        m["chg24"] = float(t["priceChangePercent"])
        rows.append((sym, m))

    total = len(rows)
    if only:
        return rows, total, total
    keep = [(s, m) for s, m in rows if m["vol24"] >= cfg.min_vol_usdt and m["spread_bps"] <= cfg.max_spread_bps]
    keep.sort(key=lambda x: x[1]["vol24"], reverse=True)
    return keep[: cfg.top_n], total, len(keep)


def process_symbol(api: Binance, sym: str, meta: dict, cfg: Config, server_ms: int):
    df4 = api.klines(sym, "4h", cfg.limit_4h, server_ms)
    df1 = api.klines(sym, "1h", cfg.limit_1h, server_ms)
    df15 = api.klines(sym, "15m", cfg.limit_15m, server_ms)
    if len(df4) < 60 or len(df1) < 60 or len(df15) < cfg.min_bars_15m:
        return sym, None
    ctx4 = htf_context(df4, cfg.htf_piv)
    ctx1 = htf_context(df1, cfg.htf_piv)

    if cfg.strategy == "smc":
        return sym, analyze_smc_symbol(sym, df15, ctx4, ctx1, meta, cfg)
    elif cfg.strategy == "breakout":
        return sym, analyze_symbol(sym, df15, ctx4, ctx1, meta, cfg)
    else:  # "all"
        s_smc = analyze_smc_symbol(sym, df15, ctx4, ctx1, meta, cfg)
        s_bk = analyze_symbol(sym, df15, ctx4, ctx1, meta, cfg)
        if s_smc and s_bk:
            return sym, s_smc if s_smc.score >= s_bk.score else s_bk
        return sym, s_smc or s_bk


# =====================================================================
#  LAPORAN
# =====================================================================
def tv_link(sym: str) -> str:
    return f"https://www.tradingview.com/chart/?symbol=BINANCE:{sym}.P&interval=15"


def trim(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def status_group(s: Setup, cfg: Config) -> str:
    if s.status == "ENTRY READY":
        return "LONG" if s.direction == "LONG" else "SHORT"
    if s.status.startswith("WATCHLIST"):
        return "WATCH"
    return "REJECT"


def reason_reject(s: Setup, cfg: Config) -> str:
    if getattr(s, "ai_decision", "") == "REJECTED":
        return f"Ditolak Full LLM: {trim(getattr(s, 'ai_summary', 'Kriteria SMC tidak terpenuhi'), 70)}"
    if s.status == "EXTENDED":
        return "Entry sudah extended (jangan dikejar)"
    if s.status == "NO SETUP":
        return f"Score {s.score:.0f} < {cfg.watch_score}"
    if s.score < cfg.watch_score:
        return f"Score {s.score:.0f} < {cfg.watch_score}"
    if s.rr < cfg.min_rr:
        return f"RR 1:{s.rr:.1f} < 1:{cfg.min_rr:.0f}"
    return "Kriteria tidak terpenuhi"


def describe(s: Setup, rank: int, btc: Optional[dict], cfg: Config) -> str:
    d = 1 if s.direction == "LONG" else -1
    m = s.meta
    L = []
    score_label = f"Skor Full LLM: {s.score:.0f}/100" if getattr(s, "ai_model", "") else f"Score {s.score:.0f}/100"
    L.append(f"#{rank}  {s.symbol}  {s.direction}  |  {s.pattern}  |  {s.status}  |  {score_label}")
    L.append(f"    Chart: {tv_link(s.symbol)}")
    L.append(f"    Sinyal: {s.kind} pada candle 15M {s.candle_time} UTC (candle terakhir yang dipakai: {s.last_candle} UTC)")
    if getattr(s, "smc_elements", None):
        el = s.smc_elements
        L.append("    1. Peta 10 Elemen Kunci SMC (Ditentukan AI dari Data Real-Time Binance):")
        L.append(f"       [1] Order Blocks (Bull/Bear) : {el.get('order_blocks', '-')}")
        L.append(f"       [2] Liquidity Sweeps & Runs  : {el.get('liquidity_sweeps_runs', '-')}")
        L.append(f"       [3] Liquidity Range & Zone   : {el.get('liquidity_range', '-')}")
        L.append(f"       [4] CHOCH (Change of Char)   : {el.get('choch', '-')}")
        L.append(f"       [5] BOS (Break of Structure) : {el.get('bos', '-')}")
        L.append(f"       [6] Inducement (IDM Level)   : {el.get('inducement', '-')}")
        L.append(f"       [7] Supply & Demand Zones    : {el.get('supply_demand', '-')}")
        L.append(f"       [8] Rejection Pin Bar Wick   : {el.get('pin_bar', '-')}")
        L.append(f"       [9] CISD (Delivery Change)   : {el.get('cisd', '-')}")
        L.append(f"       [10] High-Probability POI    : {el.get('poi', '-')}")
    elif s.parts:
        top = sorted(s.parts, key=lambda x: x[1] / x[2], reverse=True)[:2]
        weak = sorted(s.parts, key=lambda x: x[1] / x[2])[:1]
        pilar_lbl = "Pilar Full LLM" if getattr(s, "ai_model", "") else "Komponen"
        L.append("    1. Evaluasi SMC: pola " + s.pattern + f" di 15M; 4H {s.trend_4h}; 1H {s.struct_1h}. "
                 f"{pilar_lbl} terkuat: {top[0][0]} ({top[0][1]:.0f}/{top[0][2]}), {top[1][0]} ({top[1][1]:.0f}/{top[1][2]}); "
                 f"terlemah: {weak[0][0]} ({weak[0][1]:.0f}/{weak[0][2]}).")
    if not math.isnan(s.upper_line) and not math.isnan(s.lower_line):
        lv = f"zona [{fp(s.lower_line)} - {fp(s.upper_line)}]"
    else:
        lv = f"level break {fp(s.level)}"
    opp_txt = f"{fp(s.opp_level)}" if s.opp_level is not None else "tidak ditemukan"
    L.append(f"    2. Level & POI: {lv}; {'target resistance / BSL' if d == 1 else 'target support / SSL'} lawan {opp_txt}.")
    L.append(f"    3. Bukti konfirmasi: {s.confirm_txt}")
    state = {
        "ENTRY READY": "TERKONFIRMASI (candle close, tidak extended, score dan RR memenuhi).",
        "EXTENDED": "terkonfirmasi tetapi sudah extended; jangan dikejar, tunggu retest.",
        "REJECT": "terkonfirmasi tetapi score atau RR belum memenuhi.",
        "WATCHLIST": "kandidat berkembang / menyentuh POI, tunggu konfirmasi close.",
    }.get(s.status, "masih BERKEMBANG, belum ada konfirmasi close.")
    L.append(f"    4. Status entry: {state}")
    L.append(f"    5. Entry {s.entry_zone} | SL {fp(s.sl)} ({s.risk_pct:.2f}%) | TP1 {fp(s.tp1)} (+{s.tp1_pct:.2f}%) | "
             f"TP2 {fp(s.tp2)} (+{s.tp2_pct:.2f}%) | RR ke TP2 1:{s.rr:.1f} (RR ke TP1 1:{s.rr / 2:.1f}).")
    if not math.isnan(s.qty):
        L.append(f"       Ukuran posisi (risiko {s.risk_usdt:.2f} USDT termasuk fee taker): qty {s.qty:g}, notional {s.notional:.2f} USDT.")
    L.append(f"    6. Invalidasi: {s.invalidation}")
    fund = m.get("funding", 0.0) * 100
    crowd = d * m.get("funding", 0.0)
    fund_txt = "membebani posisi searah" if crowd > 0.0003 else "normal"
    risks = [f"volume 24j {m.get('vol24', 0) / 1e6:.0f} juta USDT, spread {m.get('spread_bps', 0):.1f} bps",
             f"funding {fund:+.4f}% ({fund_txt})",
             f"ATR 15M {s.atr_pct * 100:.2f}% dari harga",
             "risiko false breakout / liquidity hunt tetap ada"]
    if btc:
        btc_align = (btc["4h"]["st"] == d) or ((btc["4h"]["close"] > btc["4h"]["e50"]) == (d == 1))
        risks.append("BTC 4H " + ("searah" if btc_align else "berbeda arah") + " (informasi konteks makro, koin dianalisis independen)")
    L.append("    7. Risiko: " + "; ".join(risks) + ".")
    for nt in s.notes:
        L.append(f"       Catatan: {nt}")
    if getattr(s, "ai_decision", ""):
        icon = "✅" if s.ai_decision == "APPROVED" else ("⚠️" if s.ai_decision == "CONDITIONAL" else "❌")
        mdl_str = f" ({s.ai_model})" if getattr(s, "ai_model", "") else ""
        L.append(f"    8. Penilaian Mandiri Full LLM (Groq{mdl_str}): {icon} {s.ai_decision} (Skor SMC: {s.score:.0f}/100)")
        if s.ai_summary:
            L.append(f"       Analisis Institusional: {s.ai_summary}")
        if s.ai_action_note:
            L.append(f"       Rencana Eksekusi: {s.ai_action_note}")
    return "\n".join(L)


def build_report(results: list, cfg: Config, scan_ts: str, btc: Optional[dict], stats: dict, args) -> str:
    out = []
    out.append("=" * 110)
    out.append(f"BINANCE USDT-M FUTURES SETUP SCANNER   |   Strategi: {cfg.strategy.upper()}   |   Scan: {scan_ts} UTC")
    out.append("=" * 110)
    out.append(f"Universe: {stats['total']} kontrak USDT-perp aktif -> {stats['liquid']} lolos filter likuiditas "
               f"(vol24 >= {cfg.min_vol_usdt / 1e6:.0f} juta USDT, spread <= {cfg.max_spread_bps:g} bps) -> {stats['scanned']} dianalisa "
               f"({stats['errors']} gagal data)")
    if btc:
        out.append(f"Konteks BTC: 4H {trend_text(btc['4h'])} | 1H {trend_text(btc['1h'])} | candle terakhir 4H {ts_utc(btc['4h']['last_time'])} UTC "
                   f"(konteks saja, bukan filter otomatis)")
    out.append("Data: hanya candle yang sudah close. Semua angka dihitung dari data Binance yang diambil saat scan.")
    out.append("")

    valid = [r for r in results if r.status == "ENTRY READY" or r.status.startswith("WATCHLIST")]
    valid.sort(key=lambda r: (0 if r.status == "ENTRY READY" else 1, -r.score))
    longs = [r for r in valid if r.status == "ENTRY READY" and r.direction == "LONG"]
    shorts = [r for r in valid if r.status == "ENTRY READY" and r.direction == "SHORT"]
    watch = [r for r in valid if r.status.startswith("WATCHLIST")]
    rejected = [r for r in results if r.status not in ("ENTRY READY",) and not r.status.startswith("WATCHLIST") and r.status != "NO SETUP"]
    nosetup = [r for r in results if r.status == "NO SETUP"]

    if not valid:
        out.append("NO VALID SETUP: tidak ada kandidat yang memenuhi kriteria saat ini.")
        out.append("")
    else:
        rows = []
        for i, r in enumerate(valid, 1):
            ai_verdict = f"{r.ai_decision} ({r.ai_confidence:.0f}%)" if getattr(r, "ai_decision", "") else "-"
            rows.append({
                "Rank": i, "Symbol": r.symbol, "Direction": r.direction, "Setup Type": r.pattern,
                "4H Trend": r.trend_4h_short, "1H Structure": r.struct_1h_short,
                "15M Confirmation": trim(r.confirm_short, 30), "Entry Zone": r.entry_zone,
                "Stop Loss": fp(r.sl), "TP1": fp(r.tp1), "TP2": fp(r.tp2), "RR": f"1:{r.rr:.1f}",
                "Score": f"{r.score:.0f}", "AI Verdict": ai_verdict, "Status": r.status, "Signal UTC": r.candle_time,
            })
        pd.set_option("display.width", 300)
        pd.set_option("display.max_columns", 50)
        pd.set_option("display.max_colwidth", 40)
        out.append(pd.DataFrame(rows).to_string(index=False))
        out.append("")

    def section(title, items):
        out.append("-" * 110)
        out.append(f"{title}  ({len(items)})")
        out.append("-" * 110)
        if not items:
            out.append("  (kosong)")
        for r in items:
            out.append(f"  {r.symbol:<14} {r.direction:<5} {r.pattern:<22} score {r.score:>3.0f}  RR 1:{r.rr:.1f}  entry {r.entry_zone}  [{r.status}]")
        out.append("")

    section("A. LONG - confirmed (ENTRY READY)", longs)
    section("B. SHORT - confirmed (ENTRY READY)", shorts)
    section("C. WATCHLIST - developing / perlu konfirmasi", watch)
    out.append("-" * 110)
    out.append(f"D. REJECTED  ({len(rejected)} terkonfirmasi tapi gagal kriteria, {len(nosetup)} pola dengan score < {cfg.watch_score})")
    out.append("-" * 110)
    for r in sorted(rejected, key=lambda x: -x.score)[:15]:
        out.append(f"  {r.symbol:<14} {r.direction:<5} {r.pattern:<22} score {r.score:>3.0f}  -> {reason_reject(r, cfg)}")
    if not rejected:
        out.append("  (tidak ada kandidat terkonfirmasi yang ditolak)")
    out.append("")

    if valid:
        out.append("=" * 110)
        out.append("ANALISA DETAIL (maks. 5 kandidat teratas)")
        out.append("=" * 110)
        for i, r in enumerate(valid[:5], 1):
            out.append(describe(r, i, btc, cfg))
            out.append("")

    out.append("-" * 110)
    out.append("Catatan: deteksi pola & score adalah aturan mekanis; verifikasi chart sebelum entry. Ukuran posisi, fee, slippage, dan leverage")
    out.append("tetap harus dihitung sendiri. Tidak ada jaminan profit. Scanner ini tidak mengirim order.")
    return "\n".join(out)


def results_to_df(results: list, scan_ts: str) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {
            "scan_utc": scan_ts, "symbol": r.symbol, "direction": r.direction, "kind": r.kind, "pattern": r.pattern,
            "status": r.status, "score": round(r.score, 1), "rr": round(r.rr, 2), "entry": r.entry, "entry_zone": r.entry_zone,
            "stop_loss": r.sl, "tp1": r.tp1, "tp2": r.tp2, "risk_pct": round(r.risk_pct, 3), "tp2_pct": round(r.tp2_pct, 3),
            "signal_candle_utc": r.candle_time, "last_closed_candle_utc": r.last_candle,
            "trend_4h": r.trend_4h, "struct_1h": r.struct_1h, "confluences": r.confirm_short, "invalidation": r.invalidation,
            "rel_vol": round(r.rel_vol, 2), "rsi": round(r.rsi, 1),
            "vol24_usdt": r.meta.get("vol24"), "spread_bps": round(r.meta.get("spread_bps", NAN), 2),
            "funding": r.meta.get("funding"), "qty": r.qty, "notional": r.notional, "risk_usdt": r.risk_usdt,
            "ai_decision": getattr(r, "ai_decision", ""), "ai_confidence": getattr(r, "ai_confidence", NAN),
            "ai_summary": getattr(r, "ai_summary", ""), "ai_action_note": getattr(r, "ai_action_note", ""),
            "ai_model": getattr(r, "ai_model", ""),
            "chart": tv_link(r.symbol),
        }
        if getattr(r, "smc_elements", None):
            el = r.smc_elements
            row["smc_order_blocks"] = el.get("order_blocks", "")
            row["smc_liquidity_sweeps"] = el.get("liquidity_sweeps_runs", "")
            row["smc_liquidity_range"] = el.get("liquidity_range", "")
            row["smc_choch"] = el.get("choch", "")
            row["smc_bos"] = el.get("bos", "")
            row["smc_inducement"] = el.get("inducement", "")
            row["smc_supply_demand"] = el.get("supply_demand", "")
            row["smc_pin_bar"] = el.get("pin_bar", "")
            row["smc_cisd"] = el.get("cisd", "")
            row["smc_poi"] = el.get("poi", "")
        for name, val, mx in r.parts:
            row[f"score_{name}"] = round(val, 1)
        rows.append(row)
    return pd.DataFrame(rows)


# =====================================================================
#  TELEGRAM (opsional)
# =====================================================================
def send_telegram(text: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("[telegram] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID belum diatur, pesan dilewati.", file=sys.stderr)
        return
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat, "text": text[:4000], "disable_web_page_preview": True}, timeout=15)
    except requests.RequestException as e:
        print(f"[telegram] gagal kirim: {e}", file=sys.stderr)


def telegram_summary(results: list, scan_ts: str, sent: set) -> Optional[str]:
    ready = [r for r in results if r.status == "ENTRY READY"]
    new = []
    for r in ready:
        key = f"{r.symbol}|{r.direction}|{r.kind}|{r.candle_time}"
        if key not in sent:
            sent.add(key)
            new.append(r)
    if not new:
        return None
    lines = [f"Setup scanner {scan_ts} UTC"]
    for r in sorted(new, key=lambda x: -x.score):
        lines.append(f"{r.direction} {r.symbol} | {r.pattern} {r.kind} | entry {fp(r.entry)} SL {fp(r.sl)} "
                     f"TP1 {fp(r.tp1)} TP2 {fp(r.tp2)} | RR 1:{r.rr:.1f} | score {r.score:.0f}")
        if getattr(r, "ai_decision", ""):
            icon = "✅" if r.ai_decision == "APPROVED" else ("⚠️" if r.ai_decision == "CONDITIONAL" else "❌")
            lines.append(f"🤖 AI Verdict: {icon} {r.ai_decision} ({r.ai_confidence:.0f}%) | {r.ai_action_note}")
        lines.append(tv_link(r.symbol))
    lines.append("Bukan saran finansial. Verifikasi chart sebelum entry.")
    return "\n".join(lines)


# =====================================================================
#  ALUR UTAMA
# =====================================================================
def run_scan(cfg: Config, args, sent: set) -> list:
    api = Binance()
    server_ms = api.server_time()
    scan_ts = dt.datetime.fromtimestamp(server_ms / 1000, dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    only = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
    print(f"[{scan_ts} UTC] mengambil daftar kontrak ...", file=sys.stderr)
    universe, total, liquid = build_universe(api, cfg, only)
    print(f"  {total} kontrak aktif, {liquid} lolos filter, menganalisa {len(universe)} ...", file=sys.stderr)

    btc = None
    try:
        b4 = api.klines("BTCUSDT", "4h", cfg.limit_4h, server_ms)
        b1 = api.klines("BTCUSDT", "1h", cfg.limit_1h, server_ms)
        btc = {"4h": htf_context(b4, cfg.htf_piv), "1h": htf_context(b1, cfg.htf_piv)}
    except Exception as e:  # noqa: BLE001
        print(f"  konteks BTC gagal diambil: {e}", file=sys.stderr)

    results, errors = [], 0
    done = 0
    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        futs = {ex.submit(process_symbol, api, sym, meta, cfg, server_ms): sym for sym, meta in universe}
        for f in as_completed(futs):
            done += 1
            try:
                sym, setup = f.result()
                if setup is not None:
                    if args.equity:
                        size_position(setup, args.equity, args.risk_pct, args.fee_pct)
                    results.append(setup)
            except Exception as e:  # noqa: BLE001
                errors += 1
                print(f"  [{futs[f]}] gagal: {e}", file=sys.stderr)
            if done % 20 == 0:
                print(f"  progres {done}/{len(universe)}", file=sys.stderr)

    if cfg.ai_audit:
        try:
            import ai_validator
            ai_validator.audit_top_setups(results, btc_ctx=btc, max_audit=cfg.ai_top_n)
            # Urutkan ulang setup secara mutlak berdasarkan status dan skor yang diputuskan Full LLM!
            results.sort(
                key=lambda r: (
                    0 if r.status == "ENTRY READY" else (1 if r.status.startswith("WATCHLIST") else 2),
                    -r.score,
                )
            )
        except Exception as e:
            print(f"  [AI Validator] Peringatan: audit AI dilewati: {e}", file=sys.stderr)

    stats = {"total": total, "liquid": liquid, "scanned": len(universe) - errors, "errors": errors}
    report = build_report(results, cfg, scan_ts, btc, stats, args)
    print(report)

    os.makedirs(args.out, exist_ok=True)
    stamp = dt.datetime.fromtimestamp(server_ms / 1000, dt.timezone.utc).strftime("%Y%m%d_%H%M")
    path = os.path.join(args.out, f"scan_{stamp}.csv")
    if results:
        results_to_df(results, scan_ts).sort_values("score", ascending=False).to_csv(path, index=False)
        print(f"\nHasil disimpan: {path}")
    with open(os.path.join(args.out, f"scan_{stamp}.txt"), "w", encoding="utf-8") as fh:
        fh.write(report)

    if args.telegram:
        msg = telegram_summary(results, scan_ts, sent)
        if msg:
            send_telegram(msg)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Binance USDT-M Futures SMC & Breakout setup scanner")
    p.add_argument("--strategy", type=str, default="smc", choices=["smc", "breakout", "all"],
                   help="strategi analisa: 'smc' (Smart Money Concepts), 'breakout' (pola konsolidasi), 'all' (keduanya)")
    p.add_argument("--top", type=int, default=Config.top_n, help="jumlah pair teratas (berdasarkan volume) yang dianalisa")
    p.add_argument("--min-vol", type=float, default=Config.min_vol_usdt / 1e6, help="volume 24 jam minimal (juta USDT)")
    p.add_argument("--max-spread-bps", type=float, default=Config.max_spread_bps, help="spread maksimal dalam bps")
    p.add_argument("--symbols", type=str, default="", help="analisa simbol tertentu saja, pisahkan dengan koma (tanpa filter likuiditas)")
    p.add_argument("--equity", type=float, default=0.0, help="equity akun (USDT) untuk hitung ukuran posisi")
    p.add_argument("--risk-pct", type=float, default=1.0, help="risiko per trade (%% dari equity)")
    p.add_argument("--fee-pct", type=float, default=0.05, help="fee taker per sisi (%%)")
    p.add_argument("--min-score", type=int, default=0, help="override score minimal ENTRY READY (default 70 untuk SMC, 80 untuk breakout)")
    p.add_argument("--recent-bars", type=int, default=Config.event_recent_bars, help="event dianggap aktif jika terjadi dalam N bar 15M terakhir")
    p.add_argument("--workers", type=int, default=Config.workers)
    p.add_argument("--no-ai", action="store_true", help="nonaktifkan evaluasi Full LLM SMC (Groq)")
    p.add_argument("--ai-top", type=int, default=15, help="jumlah setup teratas yang dievaluasi Full LLM (default 15)")
    p.add_argument("--ai-model", type=str, default="", help="override model Groq AI (default dari .env: qwen/qwen3.8-27b)")
    p.add_argument("--out", type=str, default="scan_output", help="folder output CSV dan TXT")
    p.add_argument("--loop", action="store_true", help="jalan terus, scan tiap candle 15M close")
    p.add_argument("--telegram", action="store_true", help="kirim ENTRY READY baru ke Telegram (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)")
    args = p.parse_args()

    if args.ai_model:
        os.environ["GROQ_MODEL"] = args.ai_model

    smc_entry = args.min_score if args.min_score > 0 else 70
    bk_entry = args.min_score if args.min_score > 0 else 80
    cfg = Config(strategy=args.strategy, top_n=args.top, min_vol_usdt=args.min_vol * 1e6,
                 max_spread_bps=args.max_spread_bps, smc_entry_score=smc_entry, entry_score=bk_entry,
                 event_recent_bars=args.recent_bars, workers=args.workers,
                 ai_audit=not args.no_ai, ai_top_n=args.ai_top)
    sent: set = set()

    if not args.loop:
        run_scan(cfg, args, sent)
        return
    while True:
        try:
            run_scan(cfg, args, sent)
        except Exception as e:  # noqa: BLE001
            print(f"scan gagal: {e}", file=sys.stderr)
        now = time.time()
        nxt = (math.floor(now / 900) + 1) * 900 + 20      # 20 detik setelah candle 15M close
        print(f"\nScan berikutnya {dt.datetime.fromtimestamp(nxt, dt.timezone.utc).strftime('%H:%M:%S')} UTC ...", file=sys.stderr)
        time.sleep(max(1, nxt - now))


if __name__ == "__main__":
    main()