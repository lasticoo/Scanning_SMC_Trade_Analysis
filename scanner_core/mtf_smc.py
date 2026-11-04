# -*- coding: utf-8 -*-
"""
Top-Down Multi-Timeframe SMC Engine (4H -> 1H -> 15M) + Orderflow & Zoom-Out Check
==================================================================================
1. Lapis 4H : Peta 1-3 Macro Key Ranges & Tembok Supply/Demand 4H (Kompas & Pembatas TP).
2. Lapis 1H : Working Trading Range (1H Swing Low <-> 1H Swing High), HTF OB/FVG/EQH/EQL,
              dan Estimasi Durasi Posisi (Hold Time) agar tidak tergerus funding fee.
3. Lapis 15M: Likuiditas minor (BSL/SSL/IDM Sweep), CHOCH/BOS/CISD, Pin Bar, Breaker Block,
              CVD Absorption + Derivatives (OI & L/S Ratio), dengan Filter Proporsionalitas
              "Zoom-Out Check" berjangkar ATR 1H (anti-SL sehelai rambut & anti-SL bengkak).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .indicators import (
    atr_wilder,
    compute_cvd_metrics,
    find_pivots,
    liquidity_score,
    rsi_wilder,
    session_liquidity_context,
    trend_short,
    trend_text,
)
from .models import MAJOR_SYMBOLS, NAN, Config, Setup, clamp, fp, ts_utc
from .zone_radar import enrich_setup_with_zone_radar


def extract_tf_structure(df: Optional[pd.DataFrame], piv_len: int, tf_label: str, lookback_bars: int = 90) -> Optional[dict]:
    """
    Mengekstrak peta struktur lengkap satu timeframe secara bebas look-ahead:
    - Confirmed Swings (Highs & Lows) + Dealing Range adaptif (0..100%)
    - Unmitigated & Tapped-1x Order Blocks (Bullish & Bearish) + Sweep POI tag
    - Breaker Blocks (OB kuat yang ditembus impulsif dan berbalik fungsi)
    - Unmitigated Fair Value Gaps (FVG yang belum diisi penuh / belum invalid)
    - Equal Highs (EQH) & Equal Lows (EQL) sebagai kolam likuiditas magnet TP
    """
    if df is None or len(df) < max(30, 2 * piv_len + 5):
        return None

    n = len(df)
    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    times = df["open_time"].values.astype("int64")

    atr_s = atr_wilder(df).values
    fallback_atr = float(np.nanmean(h[-14:] - l[-14:])) if n >= 14 else float(np.mean(h - l))
    cur_atr = float(atr_s[-1]) if not np.isnan(atr_s[-1]) and atr_s[-1] > 0 else max(fallback_atr, 1e-12)
    cur_p = float(c[-1])

    is_ph, is_pl = find_pivots(h, l, piv_len)
    highs = [{"idx": i, "price": float(h[i]), "time": int(times[i])} for i in range(n) if is_ph[i]]
    lows = [{"idx": i, "price": float(l[i]), "time": int(times[i])} for i in range(n) if is_pl[i]]

    # Dealing Range adaptif: ekspansi jendela swing jika 4 pivot terakhir terlalu sempit (< 2.5 ATR)
    tail_hi = float(np.max(h[-min(n, piv_len + 3) :]))
    tail_lo = float(np.min(l[-min(n, piv_len + 3) :]))
    range_high = max([s["price"] for s in highs[-4:]] + [tail_hi]) if highs else float(np.max(h[-min(n, 40) :]))
    range_low = min([s["price"] for s in lows[-4:]] + [tail_lo]) if lows else float(np.min(l[-min(n, 40) :]))

    for k_piv in (6, 10, 16):
        if range_high - range_low >= 2.5 * cur_atr:
            break
        if len(highs) >= 2:
            range_high = max(range_high, max(s["price"] for s in highs[-k_piv:]))
        if len(lows) >= 2:
            range_low = min(range_low, min(s["price"] for s in lows[-k_piv:]))

    if range_high - range_low < 2.2 * cur_atr:
        win = min(n, lookback_bars)
        range_high = max(range_high, float(np.max(h[-win:])))
        range_low = min(range_low, float(np.min(l[-win:])))

    span = max(range_high - range_low, 1e-12)
    eq_price = (range_high + range_low) / 2.0
    pos_pct = clamp((cur_p - range_low) / span * 100.0, 0.0, 100.0)

    # 1. Unmitigated Fair Value Gaps (FVG)
    start_idx = max(3, n - lookback_bars)
    fvgs: List[dict] = []
    raw_fvg_idx = set()
    for i in range(start_idx, n):
        a = cur_atr if np.isnan(atr_s[i]) or atr_s[i] <= 0 else float(atr_s[i])
        # Bullish FVG
        if l[i] - h[i - 2] >= 0.12 * a:
            raw_fvg_idx.add((i, 1))
            r_top, r_bot = float(l[i]), float(h[i - 2])
            if i + 1 <= n - 2:
                sub_c = c[i + 1 : n - 1]
                sub_l = l[i + 1 : n - 1]
                # Invalid jika ada candle close di bawah dasar FVG atau gap sudah terisi >= 92%
                if np.min(sub_c) < r_bot or np.min(sub_l) <= r_bot + 0.08 * (r_top - r_bot):
                    continue
                eff_top = min(r_top, float(np.min(sub_l)))
            else:
                eff_top = r_top
            if eff_top - r_bot >= 0.08 * a:
                fvgs.append({
                    "tf": tf_label, "idx": i, "dir": 1, "top": eff_top, "bot": r_bot,
                    "raw_top": r_top, "ce": (eff_top + r_bot) / 2.0, "time": int(times[i]),
                    "kind": f"{tf_label} Bullish FVG",
                })
        # Bearish FVG
        elif l[i - 2] - h[i] >= 0.12 * a:
            raw_fvg_idx.add((i, -1))
            r_top, r_bot = float(l[i - 2]), float(h[i])
            if i + 1 <= n - 2:
                sub_c = c[i + 1 : n - 1]
                sub_h = h[i + 1 : n - 1]
                if np.max(sub_c) > r_top or np.max(sub_h) >= r_top - 0.08 * (r_top - r_bot):
                    continue
                eff_bot = max(r_bot, float(np.max(sub_h)))
            else:
                eff_bot = r_bot
            if r_top - eff_bot >= 0.08 * a:
                fvgs.append({
                    "tf": tf_label, "idx": i, "dir": -1, "top": r_top, "bot": eff_bot,
                    "raw_bot": r_bot, "ce": (r_top + eff_bot) / 2.0, "time": int(times[i]),
                    "kind": f"{tf_label} Bearish FVG",
                })

    # 2. Order Blocks (OB) dengan Validasi Displacement + Mitigasi + Breaker Block
    bull_obs: List[dict] = []
    bear_obs: List[dict] = []
    breakers: List[dict] = []

    for i in range(max(2, n - lookback_bars), n - 2):
        a = cur_atr if np.isnan(atr_s[i]) or atr_s[i] <= 0 else float(atr_s[i])
        end_d = min(n - 1, i + 3)

        # Kandidat Bullish OB: candle merah sebelum ekspansi impulsif naik 1-3 bar
        if c[i] < o[i] and np.max(c[i + 1 : end_d + 1]) > h[i]:
            disp_move = float(np.max(c[i + 1 : end_d + 1])) - o[i + 1]
            has_fvg = any((k, 1) in raw_fvg_idx for k in range(i + 1, end_d + 2))
            if has_fvg or disp_move >= 0.95 * a:
                ob_bot = float(l[i])
                ob_top = float(h[i]) if (h[i] - l[i]) <= 1.35 * a else float(min(h[i], max(o[i], l[i] + 0.85 * a)))
                swept = bool(l[i] < np.min(l[max(0, i - 4) : i]))

                broken_idx = None
                touches = 0
                for j in range(end_d + 1, n - 1):
                    if c[j] < ob_bot - 0.08 * a:
                        broken_idx = j
                        break
                    if l[j] <= ob_top:
                        touches += 1

                if broken_idx is None:
                    if touches <= 2:
                        bull_obs.append({
                            "tf": tf_label, "idx": i, "dir": 1, "top": ob_top, "bot": ob_bot,
                            "swept": swept, "touches": touches,
                            "fresh": touches == 0,
                            "time": int(times[i]),
                            "kind": f"{tf_label} Bullish OB (Sweep POI)" if swept else f"{tf_label} Bullish OB",
                        })
                else:
                    # Jika ditembus impulsif turun & belum ditembus balik ke atas -> Bearish Breaker
                    bj = broken_idx
                    if (o[bj] - c[bj]) >= 0.70 * a and (bj + 1 >= n - 1 or np.max(c[bj + 1 : n - 1]) <= ob_top + 0.1 * a):
                        breakers.append({
                            "tf": tf_label, "idx": bj, "dir": -1, "top": ob_top, "bot": ob_bot,
                            "swept": swept, "fresh": True, "time": int(times[bj]),
                            "kind": f"{tf_label} Bearish Breaker Block",
                        })

        # Kandidat Bearish OB: candle hijau sebelum ekspansi impulsif turun 1-3 bar
        if c[i] > o[i] and np.min(c[i + 1 : end_d + 1]) < l[i]:
            disp_move = o[i + 1] - float(np.min(c[i + 1 : end_d + 1]))
            has_fvg = any((k, -1) in raw_fvg_idx for k in range(i + 1, end_d + 2))
            if has_fvg or disp_move >= 0.95 * a:
                ob_top = float(h[i])
                ob_bot = float(l[i]) if (h[i] - l[i]) <= 1.35 * a else float(max(l[i], min(o[i], h[i] - 0.85 * a)))
                swept = bool(h[i] > np.max(h[max(0, i - 4) : i]))

                broken_idx = None
                touches = 0
                for j in range(end_d + 1, n - 1):
                    if c[j] > ob_top + 0.08 * a:
                        broken_idx = j
                        break
                    if h[j] >= ob_bot:
                        touches += 1

                if broken_idx is None:
                    if touches <= 2:
                        bear_obs.append({
                            "tf": tf_label, "idx": i, "dir": -1, "top": ob_top, "bot": ob_bot,
                            "swept": swept, "touches": touches,
                            "fresh": touches == 0,
                            "time": int(times[i]),
                            "kind": f"{tf_label} Bearish OB (Sweep POI)" if swept else f"{tf_label} Bearish OB",
                        })
                else:
                    bj = broken_idx
                    if (c[bj] - o[bj]) >= 0.70 * a and (bj + 1 >= n - 1 or np.min(c[bj + 1 : n - 1]) >= ob_bot - 0.1 * a):
                        breakers.append({
                            "tf": tf_label, "idx": bj, "dir": 1, "top": ob_top, "bot": ob_bot,
                            "swept": swept, "fresh": True, "time": int(times[bj]),
                            "kind": f"{tf_label} Bullish Breaker Block",
                        })

    # 3. Equal Highs (EQH) & Equal Lows (EQL) Liquidity Pools (belum disapu)
    eqh_pools: List[float] = []
    rec_h = highs[-10:]
    for idx_a in range(len(rec_h)):
        for idx_b in range(idx_a + 1, len(rec_h)):
            s1, s2 = rec_h[idx_a], rec_h[idx_b]
            if s2["idx"] - s1["idx"] < 3:
                continue
            lvl = max(s1["price"], s2["price"])
            if abs(s1["price"] - s2["price"]) <= 0.22 * cur_atr:
                after_slice = h[s2["idx"] + 1 : n - 1]
                if len(after_slice) == 0 or float(np.max(after_slice)) <= lvl + 0.05 * cur_atr:
                    if not any(abs(lvl - x) <= 0.15 * cur_atr for x in eqh_pools):
                        eqh_pools.append(lvl)

    eql_pools: List[float] = []
    rec_l = lows[-10:]
    for idx_a in range(len(rec_l)):
        for idx_b in range(idx_a + 1, len(rec_l)):
            s1, s2 = rec_l[idx_a], rec_l[idx_b]
            if s2["idx"] - s1["idx"] < 3:
                continue
            lvl = min(s1["price"], s2["price"])
            if abs(s1["price"] - s2["price"]) <= 0.22 * cur_atr:
                after_slice = l[s2["idx"] + 1 : n - 1]
                if len(after_slice) == 0 or float(np.min(after_slice)) >= lvl - 0.05 * cur_atr:
                    if not any(abs(lvl - x) <= 0.15 * cur_atr for x in eql_pools):
                        eql_pools.append(lvl)

    return {
        "tf": tf_label,
        "atr": cur_atr,
        "close": cur_p,
        "highs": highs,
        "lows": lows,
        "range_high": range_high,
        "range_low": range_low,
        "eq": eq_price,
        "pos_pct": pos_pct,
        "bull_obs": bull_obs,
        "bear_obs": bear_obs,
        "breakers": breakers,
        "fvgs": fvgs,
        "eqh": sorted(eqh_pools),
        "eql": sorted(eql_pools, reverse=True),
    }


def _zones_overlap(z1: dict, z2: dict, tol: float) -> bool:
    return (z1["bot"] - tol) <= z2["top"] and (z2["bot"] - tol) <= z1["top"]


def _format_hold_duration(h_tp1: float, h_tp2: float) -> str:
    lo = max(0.5, min(h_tp1, h_tp2 * 0.6))
    hi = max(lo + 1.0, h_tp2)
    if hi <= 6.0:
        tag = "Scalp-Intraday"
    elif hi <= 18.0:
        tag = "Intraday"
    elif hi <= 36.0:
        tag = "Short-Swing"
    else:
        tag = "Swing Panjang (pantau funding)"
    return f"{tag} (~{lo:.0f}-{hi:.0f} jam)"


def _format_orderflow_summary(cvd: dict, deriv: dict, funding: float, d: int) -> Tuple[str, float]:
    """Menyusun ringkasan orderflow publik & memberi skor konfluensi tambahan (0..6 poin)."""
    pts = 0.0
    tags = [cvd.get("cvd_state", "CVD Netral")]

    if d == 1:
        if cvd.get("bull_absorption"):
            pts += 2.5
        elif cvd.get("cvd_dir") == 1:
            pts += 1.5
    else:
        if cvd.get("bear_absorption"):
            pts += 2.5
        elif cvd.get("cvd_dir") == -1:
            pts += 1.5

    oi_1h = deriv.get("oi_change_1h_pct", NAN) if deriv else NAN
    oi_4h = deriv.get("oi_change_4h_pct", NAN) if deriv else NAN
    top_ls = deriv.get("top_ls_ratio", NAN) if deriv else NAN
    ret_ls = deriv.get("retail_ls_ratio", NAN) if deriv else NAN

    if not math.isnan(oi_1h):
        oi_str = f"OI 1H {oi_1h:+.2f}%"
        if not math.isnan(oi_4h):
            oi_str += f" (4H {oi_4h:+.2f}%)"
        tags.append(oi_str)
        # OI meningkat mengindikasikan uang baru masuk searah displacement
        if oi_1h >= 0.5:
            pts += 1.5
        elif oi_1h <= -1.5:
            tags.append("Likuidasi/Deleveraging")

    if not math.isnan(top_ls):
        tags.append(f"TopTrader L/S {top_ls:.2f}")
        if (d == 1 and top_ls >= 1.15) or (d == -1 and top_ls <= 0.90):
            pts += 1.0

    if not math.isnan(ret_ls):
        tags.append(f"Retail L/S {ret_ls:.2f}")
        # Kontrarian saat ritel terlalu berat sebelah berlawanan arah setup institusional
        if (d == 1 and ret_ls <= 0.85) or (d == -1 and ret_ls >= 1.85):
            pts += 1.0
            tags.append("Fuel Kontrarian Retail")

    if d * funding < -0.0002:
        pts += 0.5  # funding dibayar (menguntungkan posisi)

    return " | ".join(tags), min(6.0, pts)


def analyze_smc_symbol(
    symbol: str,
    df15: pd.DataFrame,
    ctx4: dict,
    ctx1: dict,
    meta: dict,
    cfg: Config,
    df4: Optional[pd.DataFrame] = None,
    df1: Optional[pd.DataFrame] = None,
    deriv: Optional[dict] = None,
) -> Optional[Setup]:
    """
    Mesin Utama Top-Down MTF SMC (4H -> 1H -> 15M):
    - Mempertahankan mesin seleksi arah & momentum yang terbukti tajam.
    - Menentukan POI, Entry Zone, Structural SL, dan TP1/TP2 menggunakan metode
      Top-Down 4H -> 1H -> 15M + Filter Proporsionalitas "Zoom-Out Check" (ATR 1H).
    """
    n = len(df15)
    if n < cfg.min_bars_15m:
        return None

    o = df15["open"].values.astype(float)
    h = df15["high"].values.astype(float)
    l = df15["low"].values.astype(float)
    c = df15["close"].values.astype(float)
    times = df15["open_time"].values.astype("int64")

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

    # =================================================================
    # LAPIS 1 (4H): 1-3 MACRO KEY RANGES & TEMBOK BESAR
    # =================================================================
    map4 = extract_tf_structure(df4, cfg.htf_piv, "4H", lookback_bars=80) if df4 is not None else None
    atr_4h = map4["atr"] if map4 else float(ctx4.get("atr", cur_atr * 4.2))
    if math.isnan(atr_4h) or atr_4h <= 0:
        atr_4h = cur_atr * 4.2

    wall_supply_4h: Optional[dict] = None
    wall_demand_4h: Optional[dict] = None
    in_supply_4h_now = False
    in_demand_4h_now = False
    if map4:
        r4_lo, r4_hi, r4_pos = map4["range_low"], map4["range_high"], map4["pos_pct"]
        all_sup_4h = (
            [z for z in map4["bear_obs"] if z["top"] > cur_p]
            + [z for z in map4["breakers"] if z["dir"] == -1 and z["top"] > cur_p]
            + [z for z in map4["fvgs"] if z["dir"] == -1 and z["top"] > cur_p]
        )
        in_supply_4h_now = any(z["bot"] <= cur_p <= z["top"] for z in all_sup_4h)
        # Tembok target atas adalah zona Supply 4H yang batas bawahnya berada di atas cur_p
        sup_above = [z for z in all_sup_4h if z["bot"] > cur_p]
        if sup_above:
            wall_supply_4h = min(sup_above, key=lambda z: z["bot"] - cur_p)
        elif all_sup_4h:
            wall_supply_4h = min(all_sup_4h, key=lambda z: z["top"] - cur_p)

        all_dem_4h = (
            [z for z in map4["bull_obs"] if z["bot"] < cur_p]
            + [z for z in map4["breakers"] if z["dir"] == 1 and z["bot"] < cur_p]
            + [z for z in map4["fvgs"] if z["dir"] == 1 and z["bot"] < cur_p]
        )
        in_demand_4h_now = any(z["bot"] <= cur_p <= z["top"] for z in all_dem_4h)
        dem_below = [z for z in all_dem_4h if z["top"] < cur_p]
        if dem_below:
            wall_demand_4h = max(dem_below, key=lambda z: z["top"] - cur_p)
        elif all_dem_4h:
            wall_demand_4h = max(all_dem_4h, key=lambda z: z["bot"] - cur_p)

        range_4h_str = f"[{fp(r4_lo)} - {fp(r4_hi)}] (Posisi Makro: {r4_pos:.1f}%)"
    else:
        r4_lo, r4_hi = float(ctx4.get("lo", cur_p * 0.94)), float(ctx4.get("hi", cur_p * 1.06))
        r4_pos = clamp((cur_p - r4_lo) / max(r4_hi - r4_lo, 1e-12) * 100.0, 0.0, 100.0)
        range_4h_str = f"[{fp(r4_lo)} - {fp(r4_hi)}] (Posisi Makro: {r4_pos:.1f}%)"

    w_sup_txt = f"Supply 4H [{fp(wall_supply_4h['bot'])}-{fp(wall_supply_4h['top'])}]" if wall_supply_4h else f"High 4H {fp(r4_hi)}"
    w_dem_txt = f"Demand 4H [{fp(wall_demand_4h['bot'])}-{fp(wall_demand_4h['top'])}]" if wall_demand_4h else f"Low 4H {fp(r4_lo)}"
    walls_4h_str = f"Batas Bawah: {w_dem_txt} | Batas Atas: {w_sup_txt}"

    # =================================================================
    # LAPIS 2 (1H): WORKING TRADING RANGE & HTF ZONES
    # =================================================================
    map1 = extract_tf_structure(df1, cfg.htf_piv, "1H", lookback_bars=96) if df1 is not None else None
    atr_1h = map1["atr"] if map1 else float(ctx1.get("atr", cur_atr * 2.1))
    if math.isnan(atr_1h) or atr_1h <= 0:
        atr_1h = cur_atr * 2.1

    # =================================================================
    # LAPIS 3 (15M): STRUKTUR MIKRO, SWEEP, TRIGGER, & ORDERFLOW
    # =================================================================
    map15 = extract_tf_structure(df15, cfg.piv_len, "15M", lookback_bars=95)
    if not map15 or not map15["highs"] or not map15["lows"]:
        return None

    if map1:
        w_low = map1["range_low"]
        w_high = map1["range_high"]
        w_eq = map1["eq"]
        w_pos_pct = map1["pos_pct"]
    else:
        w_low = map15["range_low"]
        w_high = map15["range_high"]
        w_eq = map15["eq"]
        w_pos_pct = map15["pos_pct"]

    in_discount_1h = w_pos_pct <= 50.0
    in_premium_1h = w_pos_pct >= 50.0

    # Pin Bars pada 15M
    pin_bars = []
    for i in range(max(0, n - 30), n):
        rng = h[i] - l[i]
        if rng <= 0:
            continue
        body = abs(c[i] - o[i])
        l_wick = min(o[i], c[i]) - l[i]
        u_wick = h[i] - max(o[i], c[i])
        if l_wick >= 0.50 * rng and u_wick <= 0.28 * rng and body <= 0.38 * rng:
            pin_bars.append({"idx": i, "dir": 1, "wick_pct": float(l_wick / rng), "time": int(times[i])})
        elif u_wick >= 0.50 * rng and l_wick <= 0.28 * rng and body <= 0.38 * rng:
            pin_bars.append({"idx": i, "dir": -1, "wick_pct": float(u_wick / rng), "time": int(times[i])})

    # Struktur & Liquidity Events 15M (BOS, CHOCH, BSL/SSL Sweep, CISD)
    highs_15 = map15["highs"]
    lows_15 = map15["lows"]
    piv_len = cfg.piv_len
    lookback = min(45, n - 1)
    struct_events = []
    trend_15 = 0

    for t in range(n - lookback, n):
        known_h = [s for s in highs_15 if s["idx"] <= t - piv_len]
        known_l = [s for s in lows_15 if s["idx"] <= t - piv_len]
        if known_h:
            last_h = known_h[-1]["price"]
            if h[t] > last_h and c[t] <= last_h and h[t - 1] <= last_h:
                struct_events.append({"type": "SWEEP", "dir": -1, "idx": t, "level": last_h, "detail": f"BSL Sweep ({fp(last_h)})"})
            elif c[t] > last_h and c[t - 1] <= last_h:
                ev_name = "BOS" if trend_15 == 1 else "CHOCH"
                struct_events.append({"type": ev_name, "dir": 1, "idx": t, "level": last_h, "detail": f"Bullish {ev_name} ({fp(last_h)})"})
                trend_15 = 1
        if known_l:
            last_l = known_l[-1]["price"]
            if l[t] < last_l and c[t] >= last_l and l[t - 1] >= last_l:
                struct_events.append({"type": "SWEEP", "dir": 1, "idx": t, "level": last_l, "detail": f"SSL Sweep ({fp(last_l)})"})
            elif c[t] < last_l and c[t - 1] >= last_l:
                ev_name = "BOS" if trend_15 == -1 else "CHOCH"
                struct_events.append({"type": ev_name, "dir": -1, "idx": t, "level": last_l, "detail": f"Bearish {ev_name} ({fp(last_l)})"})
                trend_15 = -1

    for t in range(max(2, n - 20), n):
        if c[t - 2] < o[t - 2] and c[t - 1] < o[t - 1] and c[t] > o[t - 2] and c[t - 1] <= o[t - 2]:
            struct_events.append({"type": "CISD", "dir": 1, "idx": t, "level": float(o[t - 2]), "detail": "Bullish CISD"})
        if c[t - 2] > o[t - 2] and c[t - 1] > o[t - 1] and c[t] < o[t - 2] and c[t - 1] >= o[t - 2]:
            struct_events.append({"type": "CISD", "dir": -1, "idx": t, "level": float(o[t - 2]), "detail": "Bearish CISD"})

    # Inducement (IDM) & Level IDM Spesifik
    idm_bull_swept = False
    idm_bear_swept = False
    idm_level: Optional[float] = None
    if highs_15 and lows_15:
        if trend_15 == 1:
            sh_idx = highs_15[-1]["idx"]
            pullbacks = [k for k in range(sh_idx + 1, n - 1) if l[k] < l[k - 1]]
            if pullbacks:
                idm_level = float(l[pullbacks[0]])
                idm_bull_swept = any(l[j] <= idm_level for j in range(pullbacks[0] + 1, n))
        elif trend_15 == -1:
            sl_idx = lows_15[-1]["idx"]
            pullbacks = [k for k in range(sl_idx + 1, n - 1) if h[k] > h[k - 1]]
            if pullbacks:
                idm_level = float(h[pullbacks[0]])
                idm_bear_swept = any(h[j] >= idm_level for j in range(pullbacks[0] + 1, n))

    sess_ctx = session_liquidity_context(df15)
    cvd_ctx = compute_cvd_metrics(df15)
    deriv_ctx = deriv or {}

    recent_ev = [e for e in struct_events if e["idx"] >= n - 15]
    recent_pins = [p for p in pin_bars if p["idx"] >= n - 5]

    # Cek juga sapuan likuiditas 1H (Swing High/Low 1H atau EQL/EQH 1H) pada 6 bar 15M terakhir
    h1_bull_sweep = False
    h1_bear_sweep = False
    if map1:
        for sw in map1.get("lows", [])[-4:]:
            lvl1 = float(sw["price"])
            if any(l[n - k] < lvl1 and c[n - k] >= lvl1 for k in range(1, min(7, n))) and cur_p >= lvl1 - 0.05 * atr_1h:
                h1_bull_sweep = True
                break
        for sw in map1.get("highs", [])[-4:]:
            lvl1 = float(sw["price"])
            if any(h[n - k] > lvl1 and c[n - k] <= lvl1 for k in range(1, min(7, n))) and cur_p <= lvl1 + 0.05 * atr_1h:
                h1_bear_sweep = True
                break
        if any(l[-1] < eq_l and cur_p >= eq_l for eq_l in map1.get("eql", [])[-3:]):
            h1_bull_sweep = True
        if any(h[-1] > eq_h and cur_p <= eq_h for eq_h in map1.get("eqh", [])[-3:]):
            h1_bear_sweep = True

    has_bull_sweep = any(e["type"] == "SWEEP" and e["dir"] == 1 for e in recent_ev) or sess_ctx["asian_ssl_swept"] or h1_bull_sweep
    has_bull_choch = any(e["type"] == "CHOCH" and e["dir"] == 1 for e in recent_ev)
    has_bull_bos = any(e["type"] == "BOS" and e["dir"] == 1 for e in recent_ev)
    has_bull_cisd = any(e["type"] == "CISD" and e["dir"] == 1 for e in recent_ev)
    has_bull_pin = any(p["dir"] == 1 for p in recent_pins)

    has_bear_sweep = any(e["type"] == "SWEEP" and e["dir"] == -1 for e in recent_ev) or sess_ctx["asian_bsl_swept"] or h1_bear_sweep
    has_bear_choch = any(e["type"] == "CHOCH" and e["dir"] == -1 for e in recent_ev)
    has_bear_bos = any(e["type"] == "BOS" and e["dir"] == -1 for e in recent_ev)
    has_bear_cisd = any(e["type"] == "CISD" and e["dir"] == -1 for e in recent_ev)
    has_bear_pin = any(p["dir"] == -1 for p in recent_pins)

    # =================================================================
    # PEMILIHAN NESTED POI (1H + 15M)
    # =================================================================
    def find_best_poi(d_dir: int) -> Tuple[Optional[dict], Optional[dict], str]:
        htf_cands: List[dict] = []
        if map1:
            if d_dir == 1:
                htf_cands = map1["bull_obs"] + [b for b in map1["breakers"] if b["dir"] == 1] + [f for f in map1["fvgs"] if f["dir"] == 1]
            else:
                htf_cands = map1["bear_obs"] + [b for b in map1["breakers"] if b["dir"] == -1] + [f for f in map1["fvgs"] if f["dir"] == -1]

        ltf_cands: List[dict] = []
        if d_dir == 1:
            ltf_cands = (
                map15["bull_obs"]
                + [b for b in map15["breakers"] if b["dir"] == 1]
                + [f for f in map15["fvgs"] if f["dir"] == 1 and f["idx"] >= n - 40]
            )
        else:
            ltf_cands = (
                map15["bear_obs"]
                + [b for b in map15["breakers"] if b["dir"] == -1]
                + [f for f in map15["fvgs"] if f["dir"] == -1 and f["idx"] >= n - 40]
            )

        # Filter zona yang relevan dengan harga terkini (sedang diuji atau menunggu pullback wajar di dalam range 1H)
        def near_price(z: dict, max_atr_1h: float) -> bool:
            if d_dir == 1:
                return (l[-1] <= z["top"] + max_atr_1h * atr_1h) and (cur_p >= z["bot"] - 0.25 * atr_1h)
            return (h[-1] >= z["bot"] - max_atr_1h * atr_1h) and (cur_p <= z["top"] + 0.25 * atr_1h)

        near_htf = [z for z in htf_cands if near_price(z, 0.90)]
        near_ltf = [z for z in ltf_cands if near_price(z, 0.65)]

        # Prioritas 1: Nested POI (zona 15M berada di dalam / bersinggungan dengan zona 1H!)
        for hz in reversed(near_htf):
            for lz in reversed(near_ltf):
                if _zones_overlap(hz, lz, 0.20 * atr_1h):
                    return hz, lz, f"Nested {hz['kind']} + {lz['kind']}"

        # Prioritas 2: Harga sedang bereaksi langsung di zona 1H aktif
        touching_htf = [z for z in near_htf if near_price(z, 0.35)]
        if touching_htf:
            hz = touching_htf[-1]
            return hz, None, hz["kind"]

        # Prioritas 3: Zona 15M segar yang selaras posisi 1H (tidak menggantung terlalu jauh)
        touching_ltf = [z for z in near_ltf if near_price(z, 0.35)]
        if touching_ltf:
            lz = touching_ltf[-1]
            return (near_htf[-1] if near_htf else None), lz, lz["kind"]

        if near_htf:
            return near_htf[-1], None, near_htf[-1]["kind"]
        if near_ltf:
            return None, near_ltf[-1], near_ltf[-1]["kind"]
        return None, None, ""

    htf_bull_poi, ltf_bull_poi, poi_type_bull = find_best_poi(1)
    htf_bear_poi, ltf_bear_poi, poi_type_bear = find_best_poi(-1)
    has_poi_bull = (htf_bull_poi is not None) or (ltf_bull_poi is not None)
    has_poi_bear = (htf_bear_poi is not None) or (ltf_bear_poi is not None)

    # =================================================================
    # MESIN SELEKSI ARAH (DIPERTAHANKAN + DIPERKAYA MTF & ORDERFLOW)
    # =================================================================
    def htf_score(d_dir: int) -> float:
        al4 = 10.0 if (ctx4["st"] == d_dir or (ctx4["close"] > ctx4["e50"] and d_dir == 1) or (ctx4["close"] < ctx4["e50"] and d_dir == -1)) else 4.0
        al1 = 10.0 if (ctx1["st"] == d_dir or (ctx1["close"] > ctx1["e50"] and d_dir == 1) or (ctx1["close"] < ctx1["e50"] and d_dir == -1)) else 4.0
        return al4 + al1

    bull_htf = htf_score(1)
    bear_htf = htf_score(-1)

    bull_nested_bonus = 4.0 if (htf_bull_poi and ltf_bull_poi) else (2.0 if htf_bull_poi else 0.0)
    bear_nested_bonus = 4.0 if (htf_bear_poi and ltf_bear_poi) else (2.0 if htf_bear_poi else 0.0)

    bull_swept_flag = bool((ltf_bull_poi and ltf_bull_poi.get("swept")) or (htf_bull_poi and htf_bull_poi.get("swept")))
    bear_swept_flag = bool((ltf_bear_poi and ltf_bear_poi.get("swept")) or (htf_bear_poi and htf_bear_poi.get("swept")))

    bull_poi_score = min(25.0, (14.0 if has_poi_bull else 0.0) + (4.0 if bull_swept_flag else 0.0) + bull_nested_bonus + (8.0 if in_discount_1h else (3.0 if w_pos_pct < 62.0 else 0.0)))
    bear_poi_score = min(25.0, (14.0 if has_poi_bear else 0.0) + (4.0 if bear_swept_flag else 0.0) + bear_nested_bonus + (8.0 if in_premium_1h else (3.0 if w_pos_pct > 38.0 else 0.0)))

    bull_struct_score = (12.0 if has_bull_sweep else 0.0) + (10.0 if has_bull_choch else (7.0 if has_bull_bos else 0.0)) + (5.0 if idm_bull_swept else 0.0)
    bear_struct_score = (12.0 if has_bear_sweep else 0.0) + (10.0 if has_bear_choch else (7.0 if has_bear_bos else 0.0)) + (5.0 if idm_bear_swept else 0.0)

    _, of_bull_pts = _format_orderflow_summary(cvd_ctx, deriv_ctx, meta.get("funding", 0.0), 1)
    _, of_bear_pts = _format_orderflow_summary(cvd_ctx, deriv_ctx, meta.get("funding", 0.0), -1)

    bull_trigger_score = min(15.0, (9.0 if has_bull_pin else 0.0) + (7.0 if has_bull_cisd else 0.0) + (4.0 if cur_rv >= 1.4 else 0.0) + of_bull_pts * 0.6)
    bear_trigger_score = min(15.0, (9.0 if has_bear_pin else 0.0) + (7.0 if has_bear_cisd else 0.0) + (4.0 if cur_rv >= 1.4 else 0.0) + of_bear_pts * 0.6)

    # Peringatan / penalti jika menabrak langsung tembok 4H lawan atau melawan tren kuat 4H+1H sekaligus
    bull_wall_pen = 8.0 if (wall_supply_4h and wall_supply_4h["bot"] - cur_p < 0.35 * atr_4h and cur_p >= wall_supply_4h["bot"] - 0.2 * atr_4h) else 0.0
    bear_wall_pen = 8.0 if (wall_demand_4h and cur_p - wall_demand_4h["top"] < 0.35 * atr_4h and cur_p <= wall_demand_4h["top"] + 0.2 * atr_4h) else 0.0
    counter_htf_bull_pen = 7.0 if bull_htf <= 8.0 else 0.0
    counter_htf_bear_pen = 7.0 if bear_htf <= 8.0 else 0.0

    total_bull = bull_htf * 0.85 + bull_poi_score + bull_struct_score + bull_trigger_score - bull_wall_pen - counter_htf_bull_pen
    total_bear = bear_htf * 0.85 + bear_poi_score + bear_struct_score + bear_trigger_score - bear_wall_pen - counter_htf_bear_pen

    if total_bull >= 40.0 and total_bull >= total_bear:
        d = 1
    elif total_bear >= 40.0:
        d = -1
    else:
        return None

    # =================================================================
    # PENENTUAN ZONA ENTRY, SL ("ZOOM-OUT CHECK" 1H), TP1/TP2, & DURASI
    # =================================================================
    notes: List[str] = []
    confluences: List[str] = []
    orderflow_str, of_pts = _format_orderflow_summary(cvd_ctx, deriv_ctx, meta.get("funding", 0.0), d)

    if w_pos_pct <= 45.0:
        zone_pos_lbl = f"Discount 1H ({w_pos_pct:.1f}%)"
    elif w_pos_pct >= 55.0:
        zone_pos_lbl = f"Premium 1H ({w_pos_pct:.1f}%)"
    else:
        zone_pos_lbl = f"Equilibrium 1H ({w_pos_pct:.1f}%)"

    htf_poi = htf_bull_poi if d == 1 else htf_bear_poi
    ltf_poi = ltf_bull_poi if d == 1 else ltf_bear_poi
    poi_label = poi_type_bull if d == 1 else poi_type_bear

    # Konstruksi batas zona POI yang proporsional (tidak sehelai rambut, tidak bengkak)
    if htf_poi and ltf_poi and _zones_overlap(htf_poi, ltf_poi, 0.20 * atr_1h):
        poi_low = max(htf_poi["bot"], ltf_poi["bot"])
        poi_high = min(htf_poi["top"], ltf_poi["top"])
        if poi_high <= poi_low:
            poi_low, poi_high = ltf_poi["bot"], ltf_poi["top"]
    elif ltf_poi:
        poi_low, poi_high = ltf_poi["bot"], ltf_poi["top"]
    elif htf_poi:
        poi_low, poi_high = htf_poi["bot"], htf_poi["top"]
        # Rampingkan zona 1H jika terlalu tebal (> 0.85 ATR 1H) ke separuh bagian depan
        if poi_high - poi_low > 0.85 * atr_1h:
            if d == 1:
                poi_low = poi_high - 0.75 * atr_1h
            else:
                poi_high = poi_low + 0.75 * atr_1h
    else:
        # Fallback zona sweep / swing struktural terdekat
        if d == 1:
            base_l = float(np.min(l[-6:]))
            poi_low, poi_high = base_l, base_l + 0.35 * atr_1h
        else:
            base_h = float(np.max(h[-6:]))
            poi_low, poi_high = base_h - 0.35 * atr_1h, base_h

    # Pastikan lebar zona proporsional: tidak sehelai rambut (min 0.14 ATR 1H) dan tidak membengkak melebihi batas SL (maks 0.80 ATR 1H / 3.5%)
    max_zone_w = min(0.80 * atr_1h, 0.035 * cur_p)
    if poi_high - poi_low > max_zone_w and max_zone_w > 0:
        if d == 1:
            poi_low = poi_high - max_zone_w
        else:
            poi_high = poi_low + max_zone_w

    if poi_high - poi_low < 0.14 * atr_1h:
        mid_z = (poi_low + poi_high) / 2.0
        poi_low = mid_z - 0.07 * atr_1h
        poi_high = mid_z + 0.07 * atr_1h

    limit_entry = (poi_low + poi_high) / 2.0
    dist_from_poi = (cur_p - poi_high) if d == 1 else (poi_low - cur_p)
    at_poi = dist_from_poi <= 0.45 * atr_1h

    # Jika harga sudah lari jauh dari POI (> 0.45 ATR 1H), jadikan rencana Limit Retest di POI (WATCHLIST)
    # kecuali sedang ada ekspansi BOS ber-volume tinggi dan ruang ke TP 1H masih sangat lega.
    if at_poi:
        plan_entry = cur_p
    else:
        plan_entry = limit_entry
        notes.append(f"Harga saat ini ({fp(cur_p)}) sudah berjarak dari POI; skenario optimal menunggu Limit Retest di {fp(poi_low)} - {fp(poi_high)}.")

    # --- STOP LOSS PROPORSIONAL ("ZOOM-OUT CHECK" BERJANGKAR ATR 1H & RANGE 1H) ---
    w_span = max(w_high - w_low, 2.5 * atr_1h)
    min_pct = cfg.min_sl_pct_major if symbol in MAJOR_SYMBOLS else cfg.min_sl_pct_alt
    raw_min_sl = max(
        cfg.min_sl_atr_1h * atr_1h,
        cfg.min_sl_range_1h * min(w_span, 5.0 * atr_1h),
        min_pct * plan_entry,
    )
    min_sl_dist = min(raw_min_sl, 0.038 * plan_entry)
    max_sl_dist = min(
        0.055 * plan_entry,
        max(min_sl_dist * 1.30, min(cfg.max_sl_atr_1h * atr_1h, cfg.max_sl_range_1h * max(w_span, 4.0 * atr_1h))),
    )

    if d == 1:
        local_swing_low = float(np.min(l[-12:]))
        anchor_low = min(poi_low, local_swing_low) if abs(plan_entry - local_swing_low) <= 1.8 * atr_1h else poi_low
        if htf_poi and abs(plan_entry - htf_poi["bot"]) <= 1.9 * atr_1h:
            anchor_low = min(anchor_low, htf_poi["bot"])
        sl_struct = anchor_low - cfg.sl_buffer * atr_1h
        raw_dist = max(0.0, plan_entry - sl_struct)
        final_sl_dist = clamp(raw_dist, min_sl_dist, max_sl_dist)
        sl = min(plan_entry - final_sl_dist, poi_low - 0.08 * cur_atr)
    else:
        local_swing_high = float(np.max(h[-12:]))
        anchor_high = max(poi_high, local_swing_high) if abs(local_swing_high - plan_entry) <= 1.8 * atr_1h else poi_high
        if htf_poi and abs(htf_poi["top"] - plan_entry) <= 1.9 * atr_1h:
            anchor_high = max(anchor_high, htf_poi["top"])
        sl_struct = anchor_high + cfg.sl_buffer * atr_1h
        raw_dist = max(0.0, sl_struct - plan_entry)
        final_sl_dist = clamp(raw_dist, min_sl_dist, max_sl_dist)
        sl = max(plan_entry + final_sl_dist, poi_high + 0.08 * cur_atr)

    risk = abs(plan_entry - sl)
    sl_atr_1h_ratio = risk / max(atr_1h, 1e-12)

    # --- TARGET TP1 & TP2 BERBASIS LIKUIDITAS 1H + PEMBATAS TEMBOK 4H + ESTIMASI DURASI ---
    htf_pts = bull_htf if d == 1 else bear_htf
    vel_factor = 0.50 if (cur_rv >= 1.4 and htf_pts >= 14.0) else 0.42
    hourly_vel = max(vel_factor * atr_1h, 1e-12)
    max_reach_dist = cfg.max_hold_hours * hourly_vel

    if d == 1:
        # Batas atas sebelum menabrak tembok Supply 4H (jangan pasang TP menembus tembok 4H di depan harga!)
        wall_cap = None
        if wall_supply_4h and wall_supply_4h["bot"] > plan_entry:
            wall_cap = max(plan_entry + 0.05 * atr_1h, wall_supply_4h["bot"] - 0.10 * atr_1h)

        cands_tp: List[float] = [w_eq, w_high]
        if map1:
            cands_tp.extend(map1["eqh"])
            cands_tp.extend([s["price"] for s in map1["highs"][-6:]])
            cands_tp.extend([z["bot"] for z in map1["bear_obs"] if z["bot"] > plan_entry])
            cands_tp.extend([z["bot"] for z in map1["fvgs"] if z["dir"] == -1 and z["bot"] > plan_entry])
        cands_tp.extend(map15["eqh"])
        if r4_hi > plan_entry:
            cands_tp.append(r4_hi)

        valid_tp2 = sorted([x for x in cands_tp if x >= plan_entry + 1.8 * risk and x <= plan_entry + max_reach_dist])
        if valid_tp2:
            # Pilih target likuiditas eksternal yang memberi ~2.2R - 4.2R
            ideal_tp2 = [x for x in valid_tp2 if x >= plan_entry + 2.1 * risk]
            tp2 = ideal_tp2[0] if ideal_tp2 else valid_tp2[-1]
        else:
            tp2 = plan_entry + min(2.8 * risk, max_reach_dist)

        if wall_cap is not None and tp2 > wall_cap:
            tp2 = wall_cap
            notes.append(f"TP2 dibatasi di bawah Tembok Supply 4H ({fp(wall_supply_4h['bot'])}) agar realistis tanpa tertahan berminggu-minggu.")

        valid_tp1 = sorted([x for x in cands_tp if plan_entry + 1.15 * risk <= x <= plan_entry + 0.68 * max(tp2 - plan_entry, 0)])
        tp1 = valid_tp1[0] if valid_tp1 else (plan_entry + 0.5 * (tp2 - plan_entry))
        opp_lvl = wall_supply_4h["bot"] if wall_supply_4h else w_high

    else:
        wall_cap = None
        if wall_demand_4h and wall_demand_4h["top"] < plan_entry:
            wall_cap = min(plan_entry - 0.05 * atr_1h, wall_demand_4h["top"] + 0.10 * atr_1h)

        cands_tp = [w_eq, w_low]
        if map1:
            cands_tp.extend(map1["eql"])
            cands_tp.extend([s["price"] for s in map1["lows"][-6:]])
            cands_tp.extend([z["top"] for z in map1["bull_obs"] if z["top"] < plan_entry])
            cands_tp.extend([z["top"] for z in map1["fvgs"] if z["dir"] == 1 and z["top"] < plan_entry])
        cands_tp.extend(map15["eql"])
        if r4_lo < plan_entry:
            cands_tp.append(r4_lo)

        valid_tp2 = sorted([x for x in cands_tp if x <= plan_entry - 1.8 * risk and x >= plan_entry - max_reach_dist], reverse=True)
        if valid_tp2:
            ideal_tp2 = [x for x in valid_tp2 if x <= plan_entry - 2.1 * risk]
            tp2 = ideal_tp2[0] if ideal_tp2 else valid_tp2[-1]
        else:
            tp2 = plan_entry - min(2.8 * risk, max_reach_dist)

        if wall_cap is not None and tp2 < wall_cap:
            tp2 = wall_cap
            notes.append(f"TP2 dibatasi di atas Tembok Demand 4H ({fp(wall_demand_4h['top'])}) agar posisi tidak tertahan terlalu lama.")

        valid_tp1 = sorted(
            [x for x in cands_tp if plan_entry - 0.68 * max(plan_entry - tp2, 0) <= x <= plan_entry - 1.15 * risk],
            reverse=True,
        )
        tp1 = valid_tp1[0] if valid_tp1 else (plan_entry - 0.5 * (plan_entry - tp2))
        opp_lvl = wall_demand_4h["top"] if wall_demand_4h else w_low

    rr = abs(tp2 - plan_entry) / max(risk, 1e-12)
    rr_tp1 = abs(tp1 - plan_entry) / max(risk, 1e-12)

    est_h1 = max(0.5, abs(tp1 - plan_entry) / hourly_vel)
    est_h2 = max(1.0, abs(tp2 - plan_entry) / hourly_vel)
    hold_label = _format_hold_duration(est_h1, est_h2)

    # =================================================================
    # PENAMAAN POLA, KONFLUENSI, & SKOR KOMPONEN AKHIR
    # =================================================================
    if d == 1:
        poi_pts = bull_poi_score
        struct_pts = bull_struct_score
        trig_pts = bull_trigger_score
        if poi_label:
            confluences.append(poi_label)
        else:
            confluences.append("1H/15M Demand Zone")
        confluences.append(zone_pos_lbl)
        if has_bull_sweep:
            confluences.append("SSL Sweep" + (" (Asian Low)" if sess_ctx["asian_ssl_swept"] else ""))
        if has_bull_choch:
            confluences.append("Bullish CHOCH")
        if has_bull_bos:
            confluences.append("Bullish BOS")
        if idm_bull_swept:
            confluences.append("IDM Taken")
        if has_bull_cisd:
            confluences.append("Bullish CISD")
        if has_bull_pin:
            confluences.append("Pin Bar Rejection")
        if cvd_ctx.get("bull_absorption"):
            confluences.append("CVD Bull Absorption")

        if has_bull_sweep and (has_bull_choch or has_bull_cisd) and has_poi_bull:
            pat_name = "MTF Sniper Buy (Sweep + CHOCH + 1H/15M POI)"
            kind_name = "SMC_SNIPER"
        elif htf_poi and ltf_poi:
            pat_name = "MTF Nested Demand (1H + 15M POI Alignment)"
            kind_name = "SMC_OB_RETEST"
        elif poi_label and "Breaker" in poi_label:
            pat_name = "SMC Bullish Breaker Block Retest"
            kind_name = "SMC_BREAKER"
        elif poi_label and "OB" in poi_label:
            pat_name = "SMC Demand Bounce (Order Block Retest)"
            kind_name = "SMC_OB_RETEST"
        elif poi_label and "FVG" in poi_label:
            pat_name = "SMC FVG Retest Entry (Unmitigated Gap)"
            kind_name = "SMC_FVG_FILL"
        elif has_bull_sweep:
            pat_name = "SMC SSL Liquidity Sweep Reversal"
            kind_name = "SMC_SWEEP"
        else:
            pat_name = "SMC MTF Demand Setup"
            kind_name = "SMC_DEMAND"

        inval = f"Close 1H/15M menembus bawah SL struktural {fp(sl)} (di bawah zona POI {fp(poi_low)})."
    else:
        poi_pts = bear_poi_score
        struct_pts = bear_struct_score
        trig_pts = bear_trigger_score
        if poi_label:
            confluences.append(poi_label)
        else:
            confluences.append("1H/15M Supply Zone")
        confluences.append(zone_pos_lbl)
        if has_bear_sweep:
            confluences.append("BSL Sweep" + (" (Asian High)" if sess_ctx["asian_bsl_swept"] else ""))
        if has_bear_choch:
            confluences.append("Bearish CHOCH")
        if has_bear_bos:
            confluences.append("Bearish BOS")
        if idm_bear_swept:
            confluences.append("IDM Taken")
        if has_bear_cisd:
            confluences.append("Bearish CISD")
        if has_bear_pin:
            confluences.append("Pin Bar Rejection")
        if cvd_ctx.get("bear_absorption"):
            confluences.append("CVD Bear Absorption")

        if has_bear_sweep and (has_bear_choch or has_bear_cisd) and has_poi_bear:
            pat_name = "MTF Sniper Short (Sweep + CHOCH + 1H/15M POI)"
            kind_name = "SMC_SNIPER"
        elif htf_poi and ltf_poi:
            pat_name = "MTF Nested Supply (1H + 15M POI Alignment)"
            kind_name = "SMC_OB_RETEST"
        elif poi_label and "Breaker" in poi_label:
            pat_name = "SMC Bearish Breaker Block Retest"
            kind_name = "SMC_BREAKER"
        elif poi_label and "OB" in poi_label:
            pat_name = "SMC Supply Rejection (Order Block Retest)"
            kind_name = "SMC_OB_RETEST"
        elif poi_label and "FVG" in poi_label:
            pat_name = "SMC FVG Retest Entry (Unmitigated Gap)"
            kind_name = "SMC_FVG_FILL"
        elif has_bear_sweep:
            pat_name = "SMC BSL Liquidity Sweep Reversal"
            kind_name = "SMC_SWEEP"
        else:
            pat_name = "SMC MTF Supply Setup"
            kind_name = "SMC_SUPPLY"

        inval = f"Close 1H/15M menembus atas SL struktural {fp(sl)} (di atas zona POI {fp(poi_high)})."

    zone_str = f"{fp(poi_low)} - {fp(poi_high)}"
    conf_short = ", ".join(confluences[:3])
    conf_txt = f"Top-Down MTF [{zone_pos_lbl} | Est. {hold_label}]: " + ", ".join(confluences)

    rr_pts = 10.0 if rr >= 2.8 else (8.5 if rr >= 2.3 else (6.5 if rr >= 2.0 else (2.5 if rr >= 1.5 else 0.0)))
    atr_pct = cur_atr / cur_p
    liq_pts = liquidity_score(meta, d, atr_pct)

    p1 = min(20.0, htf_pts)
    p2 = min(25.0, poi_pts)
    p3 = min(25.0, struct_pts)
    p4 = min(15.0, trig_pts)
    p5 = min(10.0, rr_pts)
    p6 = min(5.0, liq_pts)
    tot_score = max(0.0, p1 + p2 + p3 + p4 + p5 + p6 - (bull_wall_pen if d == 1 else bear_wall_pen))

    parts = [
        ("Konteks HTF & Tren", p1, 20),
        ("Kualitas POI & Zone", p2, 25),
        ("Struktur & Liquidity", p3, 25),
        ("Trigger & Orderflow", p4, 15),
        ("Risk to Reward", p5, 10),
        ("Likuiditas Pasar", p6, 5),
    ]

    # Konfirmasi reaksi candle aktual di POI (jangan masuk melawan candle impulsif yang belum close memantul!)
    recent_pin_2 = any(p["dir"] == d and p["idx"] >= n - 2 for p in pin_bars)
    recent_cisd_3 = any(e["type"] == "CISD" and e["dir"] == d and e["idx"] >= n - 3 for e in struct_events)
    if d == 1:
        candle_rejected = (c[-1] > o[-1] and c[-1] >= poi_low) or recent_pin_2 or recent_cisd_3
    else:
        candle_rejected = (c[-1] < o[-1] and c[-1] <= poi_high) or recent_pin_2 or recent_cisd_3

    # Untuk status otomatis ENTRY READY, wajib selaras minimal salah satu struktur/tren HTF (4H atau 1H);
    # jika melawan 4H dan 1H sekaligus (counter-trend penuh), tahan di WATCHLIST agar diverifikasi manual/Limit.
    htf_aligned = htf_pts >= 14.0

    # Jangan jadikan ENTRY READY otomatis bila harga sedang berada tepat di dalam tembok 4H berlawanan
    blocked_by_4h_wall = (d == 1 and in_supply_4h_now) or (d == -1 and in_demand_4h_now)

    # Pastikan tp2_pct bersih cukup besar menutupi fee round-trip (>= 1.4%)
    tp2_move_pct = abs(tp2 - plan_entry) / plan_entry * 100.0
    if at_poi and candle_rejected and htf_aligned and not blocked_by_4h_wall and tot_score >= cfg.smc_entry_score and rr >= cfg.min_rr and tp2_move_pct >= 1.4:
        status = "ENTRY READY"
    elif tot_score >= cfg.smc_watch_score and rr >= 1.6 and tp2_move_pct >= 1.0:
        status = "WATCHLIST"
        if blocked_by_4h_wall:
            notes.append("Harga saat ini berada di dalam zona Tembok 4H berlawanan; ditahan di WATCHLIST sampai tembok 4H berhasil ditembus.")
        elif at_poi and not candle_rejected:
            notes.append("Harga sedang menguji zona POI namun candle 15M terakhir belum menutup searah pantulan; tunggu candle konfirmasi.")
        elif not htf_aligned:
            notes.append("Setup berlawanan dengan tren 4H & 1H sekaligus (counter-trend); ditahan di WATCHLIST demi keamanan.")
    else:
        status = "REJECT"

    # Susun paket data lengkap 3 Timeframe + Orderflow untuk AI Validator (tidak ada data terbuang!)
    raw_data = {
        "current_price": cur_p,
        "plan_entry": plan_entry,
        "limit_entry": limit_entry,
        "at_poi": at_poi,
        "poi_low": poi_low,
        "poi_high": poi_high,
        "poi_label": poi_label,
        "min_sl_dist": min_sl_dist,
        "max_sl_dist": max_sl_dist,
        # Lapis 4H
        "range_4h_low": r4_lo,
        "range_4h_high": r4_hi,
        "range_4h_pos_pct": r4_pos,
        "range_4h_str": range_4h_str,
        "walls_4h_str": walls_4h_str,
        "atr_4h": atr_4h,
        "obs_4h": [
            {"dir": z["dir"], "bot": z["bot"], "top": z["top"], "kind": z["kind"]}
            for z in ((map4["bull_obs"][-2:] + map4["bear_obs"][-2:]) if map4 else [])
        ],
        # Lapis 1H
        "range_1h_low": w_low,
        "range_1h_high": w_high,
        "range_1h_eq": w_eq,
        "range_1h_pos_pct": w_pos_pct,
        "zone_pos_label": zone_pos_lbl,
        "atr_1h": atr_1h,
        "sl_atr_1h_ratio": sl_atr_1h_ratio,
        "hourly_velocity": hourly_vel,
        "est_hold_hours_tp1": round(est_h1, 1),
        "est_hold_hours_tp2": round(est_h2, 1),
        "est_hold_label": hold_label,
        "obs_1h": [
            {"dir": z["dir"], "bot": z["bot"], "top": z["top"], "swept": z.get("swept", False), "fresh": z.get("fresh", True), "kind": z["kind"]}
            for z in ((map1["bull_obs"][-2:] + map1["bear_obs"][-2:] + map1["breakers"][-2:]) if map1 else [])
        ],
        "fvgs_1h": [
            {"dir": f["dir"], "bot": f["bot"], "top": f["top"]}
            for f in (map1["fvgs"][-3:] if map1 else [])
        ],
        "eqh_1h": map1["eqh"][-3:] if map1 else [],
        "eql_1h": map1["eql"][-3:] if map1 else [],
        # Lapis 15M (kompatibel juga dengan nama kunci lama)
        "range_high": w_high,
        "range_low": w_low,
        "equilibrium": w_eq,
        "discount_pct": w_pos_pct,
        "range_15m_low": map15["range_low"],
        "range_15m_high": map15["range_high"],
        "recent_highs": [s["price"] for s in highs_15[-4:]],
        "recent_lows": [s["price"] for s in lows_15[-4:]],
        "bull_obs": [{"top": ob["top"], "bot": ob["bot"], "swept": ob["swept"], "fresh": ob["fresh"]} for ob in map15["bull_obs"][-3:]],
        "bear_obs": [{"top": ob["top"], "bot": ob["bot"], "swept": ob["swept"], "fresh": ob["fresh"]} for ob in map15["bear_obs"][-3:]],
        "breakers_15m": [{"dir": b["dir"], "top": b["top"], "bot": b["bot"], "kind": b["kind"]} for b in map15["breakers"][-2:]],
        "fvgs": [{"dir": f["dir"], "top": f["top"], "bot": f["bot"]} for f in map15["fvgs"][-4:]],
        "eqh_15m": map15["eqh"][-3:],
        "eql_15m": map15["eql"][-3:],
        "pin_bars": [{"dir": p["dir"], "wick_pct": round(p["wick_pct"], 2)} for p in pin_bars[-3:]],
        "struct_events": [e["detail"] for e in struct_events[-6:]],
        "idm_level": idm_level,
        "idm_bull_swept": idm_bull_swept,
        "idm_bear_swept": idm_bear_swept,
        # Sesi & Orderflow
        "session": sess_ctx["session"],
        "asian_high": sess_ctx["asian_high"],
        "asian_low": sess_ctx["asian_low"],
        "asian_bsl_swept": sess_ctx["asian_bsl_swept"],
        "asian_ssl_swept": sess_ctx["asian_ssl_swept"],
        "cvd_state": cvd_ctx["cvd_state"],
        "taker_buy_ratio": cvd_ctx["taker_buy_ratio"],
        "oi_change_1h_pct": deriv_ctx.get("oi_change_1h_pct", NAN),
        "oi_change_4h_pct": deriv_ctx.get("oi_change_4h_pct", NAN),
        "top_ls_ratio": deriv_ctx.get("top_ls_ratio", NAN),
        "retail_ls_ratio": deriv_ctx.get("retail_ls_ratio", NAN),
        "orderflow_summary": orderflow_str,
        "trend_4h": trend_text(ctx4),
        "trend_1h": trend_text(ctx1),
        "atr": cur_atr,
        "vol24_usdt": meta.get("vol24", 0),
        "spread_bps": meta.get("spread_bps", 0),
        "funding_rate": meta.get("funding", 0.0),
    }

    st = Setup(
        symbol=symbol,
        direction="LONG" if d == 1 else "SHORT",
        kind=kind_name,
        pattern=pat_name,
        bar_index=n - 1,
        candle_time=ts_utc(int(times[-1])),
        level=float(w_eq),
        entry=float(plan_entry),
        market_price=float(cur_p),
        limit_entry=float(limit_entry),
        entry_zone=zone_str,
        poi_low=float(poi_low),
        poi_high=float(poi_high),
        sl=float(sl),
        tp1=float(tp1),
        tp2=float(tp2),
        rr=float(rr),
        rr_tp1=float(rr_tp1),
        risk_pct=risk / plan_entry * 100.0,
        tp1_pct=abs(tp1 - plan_entry) / plan_entry * 100.0,
        tp2_pct=abs(tp2 - plan_entry) / plan_entry * 100.0,
        score=float(tot_score),
        python_score=float(tot_score),
        parts=parts,
        status=status,
        python_status=status,
        confirmed=at_poi,
        extended=not at_poi,
        rel_vol=cur_rv,
        rsi=cur_rsi,
        atr=cur_atr,
        atr_pct=atr_pct,
        atr_1h=atr_1h,
        atr_1h_pct=atr_1h / cur_p,
        sl_atr_1h_ratio=sl_atr_1h_ratio,
        upper_line=float(poi_high),
        lower_line=float(poi_low),
        opp_level=float(opp_lvl) if opp_lvl is not None else None,
        trend_4h=trend_text(ctx4),
        struct_1h=trend_text(ctx1),
        trend_4h_short=trend_short(ctx4),
        struct_1h_short=trend_short(ctx1),
        confirm_txt=conf_txt,
        confirm_short=conf_short,
        invalidation=inval,
        notes=notes,
        range_4h_str=range_4h_str,
        walls_4h_str=walls_4h_str,
        range_1h_low=float(w_low),
        range_1h_high=float(w_high),
        range_1h_eq=float(w_eq),
        range_1h_pos_pct=float(w_pos_pct),
        est_hold_hours_tp1=float(est_h1),
        est_hold_hours_tp2=float(est_h2),
        est_hold_label=hold_label,
        session_name=sess_ctx["session"],
        cvd_state=cvd_ctx["cvd_state"],
        taker_buy_ratio=float(cvd_ctx["taker_buy_ratio"]),
        oi_change_1h_pct=float(deriv_ctx.get("oi_change_1h_pct", NAN)),
        oi_change_4h_pct=float(deriv_ctx.get("oi_change_4h_pct", NAN)),
        top_ls_ratio=float(deriv_ctx.get("top_ls_ratio", NAN)),
        retail_ls_ratio=float(deriv_ctx.get("retail_ls_ratio", NAN)),
        orderflow_summary=orderflow_str,
        meta=dict(meta),
        last_candle=ts_utc(int(times[-1])),
        raw_market_data=raw_data,
    )
    enrich_setup_with_zone_radar(
        st,
        df15=df15,
        map4=map4,
        map1=map1,
        map15=map15,
        sess_ctx=sess_ctx,
        cvd_ctx=cvd_ctx,
        cfg=cfg,
        idm_level=idm_level,
        idm_bull_swept=idm_bull_swept,
        idm_bear_swept=idm_bear_swept,
        has_recent_choch=(has_bull_choch if d == 1 else has_bear_choch),
        has_recent_cisd=(has_bull_cisd if d == 1 else has_bear_cisd),
    )
    return st
