# -*- coding: utf-8 -*-
from __future__ import annotations

import datetime as dt
import math
import numpy as np
import pandas as pd

from .models import NAN, clamp


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def atr_wilder(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - pc).abs(),
            (df["low"] - pc).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out = rma(tr, n)
    out.iloc[: min(n, len(out))] = np.nan
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
    """Pivot terkonfirmasi bebas look-ahead: bar i adalah pivot jika ekstrem pada [i-L, i+L]."""
    n = len(high)
    is_ph = np.zeros(n, dtype=bool)
    is_pl = np.zeros(n, dtype=bool)
    if n < 2 * L + 1:
        return is_ph, is_pl
    for i in range(L, n - L):
        wh = high[i - L : i + L + 1]
        wl = low[i - L : i + L + 1]
        if int(np.argmax(wh)) == L:
            is_ph[i] = True
        if int(np.argmin(wl)) == L:
            is_pl[i] = True
    return is_ph, is_pl


def structure_state(df: pd.DataFrame, L: int) -> int:
    """Tren struktur: +1 jika close terakhir menembus swing high terkonfirmasi, -1 jika swing low."""
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
    atr_s = atr_wilder(df)
    e200 = float(ema(c, 200).iloc[-1]) if len(df) >= 200 else NAN
    cur_atr = float(atr_s.iloc[-1]) if len(atr_s) and not np.isnan(atr_s.iloc[-1]) else float((df["high"] - df["low"]).tail(14).mean())
    return {
        "st": structure_state(df, L),
        "close": float(c.iloc[-1]),
        "e20": float(ema(c, 20).iloc[-1]),
        "e50": float(ema(c, 50).iloc[-1]),
        "e200": e200,
        "atr": cur_atr,
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


def liquidity_score(meta: dict, d: int, atr_pct: float) -> float:
    """Skor likuiditas & eksekusi (maks 5): volume, spread, funding, volatilitas."""
    vol = meta.get("vol24", 0.0)
    spread = meta.get("spread_bps", 99.0)
    fund = meta.get("funding", 0.0)
    s_vol = 2.0 * clamp(vol / 1e8, 0.0, 1.0)
    s_spread = 1.5 if spread <= 2 else (1.0 if spread <= 5 else (0.5 if spread <= 10 else 0.0))
    crowd = d * fund      # positif = funding membebani posisi searah
    s_fund = 1.0 if crowd <= 0.0003 else (0.5 if crowd <= 0.001 else 0.0)
    s_vola = 0.5 if atr_pct <= 0.035 else 0.0
    return s_vol + s_spread + s_fund + s_vola


def compute_cvd_metrics(df: pd.DataFrame) -> dict:
    """
    Menghitung Cumulative Volume Delta (CVD), Taker Buy Ratio, dan Absorption/Divergence
    langsung dari kolom quote_volume & tb_quote kline publik Binance (0 ekstra API call).
    """
    n = len(df)
    if n < 20 or "tb_quote" not in df.columns or "quote_volume" not in df.columns:
        return {
            "taker_buy_ratio": 0.5,
            "cvd_dir": 0,
            "bull_absorption": False,
            "bear_absorption": False,
            "cvd_state": "Netral (data delta tidak tersedia)",
        }

    qv = df["quote_volume"].values.astype(float)
    tb = df["tb_quote"].values.astype(float)
    delta = 2.0 * tb - qv
    cvd = np.cumsum(delta)

    recent_qv = float(np.sum(qv[-3:]))
    recent_tb = float(np.sum(tb[-3:]))
    tbr_3 = recent_tb / recent_qv if recent_qv > 0 else 0.5
    last_tbr = float(tb[-1] / qv[-1]) if qv[-1] > 0 else 0.5

    # Tren CVD jangka pendek (10 bar terakhir vs rata-rata sebelumnya)
    cvd_change_10 = cvd[-1] - cvd[-10]
    avg_abs_delta = float(np.mean(np.abs(delta[-20:]))) + 1e-9
    norm_slope = cvd_change_10 / (10.0 * avg_abs_delta)
    cvd_dir = 1 if norm_slope > 0.15 else (-1 if norm_slope < -0.15 else 0)

    l = df["low"].values
    h = df["high"].values
    c = df["close"].values
    o = df["open"].values

    # Deteksi Bullish Absorption / CVD Divergence:
    # Harga menyentuh/membuat low rendah baru dalam 6 bar terakhir dibandingkan bar [-20:-6],
    # namun CVD membuat Higher Low atau bar pantulan didominasi Taker Buy (>= 54%).
    prev_low_idx = int(np.argmin(l[-22:-6])) + (n - 22)
    curr_low_idx = int(np.argmin(l[-6:])) + (n - 6)
    price_lower_or_eq = l[curr_low_idx] <= l[prev_low_idx] * 1.002
    cvd_higher_low = cvd[curr_low_idx] > cvd[prev_low_idx]
    bull_candle_absorb = (c[-1] > o[-1] and last_tbr >= 0.54) or (tbr_3 >= 0.545)
    bull_absorption = bool((price_lower_or_eq and cvd_higher_low) or (l[-1] <= np.min(l[-10:]) * 1.003 and bull_candle_absorb))

    # Deteksi Bearish Absorption / CVD Divergence:
    prev_high_idx = int(np.argmax(h[-22:-6])) + (n - 22)
    curr_high_idx = int(np.argmax(h[-6:])) + (n - 6)
    price_higher_or_eq = h[curr_high_idx] >= h[prev_high_idx] * 0.998
    cvd_lower_high = cvd[curr_high_idx] < cvd[prev_high_idx]
    bear_candle_absorb = (c[-1] < o[-1] and last_tbr <= 0.46) or (tbr_3 <= 0.455)
    bear_absorption = bool((price_higher_or_eq and cvd_lower_high) or (h[-1] >= np.max(h[-10:]) * 0.997 and bear_candle_absorb))

    parts = []
    if cvd_dir == 1:
        parts.append("CVD Naik")
    elif cvd_dir == -1:
        parts.append("CVD Turun")
    else:
        parts.append("CVD Datar")
    parts.append(f"TakerBuy {tbr_3 * 100:.1f}%")
    if bull_absorption:
        parts.append("Bullish CVD Absorption")
    if bear_absorption:
        parts.append("Bearish CVD Absorption")

    return {
        "taker_buy_ratio": round(tbr_3, 4),
        "last_taker_buy_ratio": round(last_tbr, 4),
        "cvd_dir": cvd_dir,
        "bull_absorption": bull_absorption,
        "bear_absorption": bear_absorption,
        "cvd_state": " | ".join(parts),
    }


def session_liquidity_context(df15: pd.DataFrame) -> dict:
    """
    Mengidentifikasi Killzone aktif (UTC) dan apakah likuiditas sesi Asia (00:00-07:00 UTC)
    telah disapu pada sesi London / New York hari ini.
    """
    if len(df15) < 10:
        return {"session": "Unknown", "asian_high": NAN, "asian_low": NAN, "asian_bsl_swept": False, "asian_ssl_swept": False}

    last_ms = int(df15["open_time"].iloc[-1])
    dt_now = dt.datetime.fromtimestamp(last_ms / 1000, dt.timezone.utc)
    hr = dt_now.hour

    if 0 <= hr < 7:
        sess = "Asian Range (00-07 UTC)"
    elif 7 <= hr < 11:
        sess = "London Killzone (07-11 UTC)"
    elif 11 <= hr < 12:
        sess = "Pre-NY Transition"
    elif 12 <= hr < 16:
        sess = "NY Open Killzone (12-16 UTC)"
    elif 16 <= hr < 20:
        sess = "London Close / NY PM (16-20 UTC)"
    else:
        sess = "Late NY (20-24 UTC)"

    day_start = dt.datetime(dt_now.year, dt_now.month, dt_now.day, 0, 0, tzinfo=dt.timezone.utc)
    asian_start_ms = int(day_start.timestamp() * 1000)
    asian_end_ms = asian_start_ms + 7 * 3600 * 1000

    asian_bars = df15[(df15["open_time"] >= asian_start_ms) & (df15["open_time"] < asian_end_ms)]
    post_asian = df15[df15["open_time"] >= asian_end_ms]

    if len(asian_bars) >= 8:
        a_hi = float(asian_bars["high"].max())
        a_lo = float(asian_bars["low"].min())
        bsl_swept = bool(len(post_asian) > 0 and float(post_asian["high"].max()) > a_hi and float(df15["close"].iloc[-1]) <= a_hi)
        ssl_swept = bool(len(post_asian) > 0 and float(post_asian["low"].min()) < a_lo and float(df15["close"].iloc[-1]) >= a_lo)
    else:
        a_hi, a_lo = NAN, NAN
        bsl_swept, ssl_swept = False, False

    return {
        "session": sess,
        "asian_high": a_hi,
        "asian_low": a_lo,
        "asian_bsl_swept": bsl_swept,
        "asian_ssl_swept": ssl_swept,
    }

