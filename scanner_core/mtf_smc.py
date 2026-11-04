# -*- coding: utf-8 -*-
"""
Dynamic 3-Archetype Top-Down MTF SMC/ICT Engine (4H -> 1H -> 15M)
=================================================================
Menggabungkan 3 variasi setup institusional secara dinamis pada setiap emiten:
1. TREND_CONT   : Follow-trend setelah konfirmasi reaksi di POI (Sweep + CHOCH/BOS/CISD + Nested POI).
2. PRE_REACTION : Deteksi SEBELUM harga bereaksi saat mendekati Supply/Demand/OB/Breaker SEGAR (Unmitigated)
                  dengan jarak dekat (<= 1.65x ATR 1H / <= 2.6%) + ukur gaya kedatangan (Kompresi vs Impulsif).
3. HTF_REVERSAL : Deteksi potensi KETERBALIKAN TREN (Counter-Trend / Wall-Bounce Inversion) saat harga
                  mendekati/menguji Tembok Mayor 4H/1H di ekstrem Premium/Discount + Exhaustion/Sweep.

Seluruh variasi dikunci oleh:
- Filter Proporsionalitas "Zoom-Out Check" berjangkar ATR 1H (anti-SL sehelai rambut & anti-SL bengkak).
- Estimasi Durasi Posisi (Hold Time <= 36 jam) agar modal tidak mengendap terkena funding fee.
- Forward Playbook (Skenario Utama If-Then & Skenario Antisipasi Breaker Flip).
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
from .zone_radar import (
    analyze_approach_dynamics,
    enrich_setup_with_zone_radar,
    get_live_price_snapshot,
)


def extract_tf_structure(df: Optional[pd.DataFrame], piv_len: int, tf_label: str, lookback_bars: int = 90) -> Optional[dict]:
    """
    Mengekstrak peta struktur lengkap satu timeframe secara bebas look-ahead:
    - Confirmed Swings (Highs & Lows) + Dealing Range adaptif (0..100%)
    - Unmitigated & Tapped-1x Order Blocks (Bullish & Bearish) + kekuatan Displacement + FVG tag
    - Breaker Blocks (OB kuat yang ditembus impulsif dan berbalik fungsi + hitungan touches jujur)
    - Unmitigated Fair Value Gaps (FVG yang belum diisi penuh / belum invalid)
    - Equal Highs (EQH) & Equal Lows (EQL) sebagai kolam likuiditas magnet TP / Inducement
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
                if np.min(sub_c) < r_bot or np.min(sub_l) <= r_bot + 0.08 * (r_top - r_bot):
                    continue
                eff_top = min(r_top, float(np.min(sub_l)))
            else:
                eff_top = r_top
            if eff_top - r_bot >= 0.08 * a:
                fvgs.append({
                    "tf": tf_label, "idx": i, "dir": 1, "top": eff_top, "bot": r_bot,
                    "raw_top": r_top, "ce": (eff_top + r_bot) / 2.0, "time": int(times[i]),
                    "touches": 0, "fresh": True, "disp_atr": round((r_top - r_bot) / max(a, 1e-12), 2), "has_fvg": True,
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
                    "touches": 0, "fresh": True, "disp_atr": round((r_top - r_bot) / max(a, 1e-12), 2), "has_fvg": True,
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
                disp_rat = round(disp_move / max(a, 1e-12), 2)

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
                            "swept": swept, "touches": touches, "fresh": touches == 0,
                            "disp_atr": disp_rat, "has_fvg": bool(has_fvg),
                            "time": int(times[i]),
                            "kind": f"{tf_label} Bullish OB (Sweep POI)" if swept else f"{tf_label} Bullish OB",
                        })
                else:
                    # Jika ditembus impulsif turun & belum ditembus balik ke atas -> Bearish Breaker
                    bj = broken_idx
                    if (o[bj] - c[bj]) >= 0.70 * a and (bj + 1 >= n - 1 or np.max(c[bj + 1 : n - 1]) <= ob_top + 0.1 * a):
                        brk_touches = sum(1 for j in range(bj + 1, n - 1) if h[j] >= ob_bot)
                        if brk_touches <= 2:
                            breakers.append({
                                "tf": tf_label, "idx": bj, "dir": -1, "top": ob_top, "bot": ob_bot,
                                "swept": swept, "touches": brk_touches, "fresh": brk_touches == 0,
                                "disp_atr": round(abs(o[bj] - c[bj]) / max(a, 1e-12), 2), "has_fvg": True,
                                "time": int(times[bj]),
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
                disp_rat = round(disp_move / max(a, 1e-12), 2)

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
                            "swept": swept, "touches": touches, "fresh": touches == 0,
                            "disp_atr": disp_rat, "has_fvg": bool(has_fvg),
                            "time": int(times[i]),
                            "kind": f"{tf_label} Bearish OB (Sweep POI)" if swept else f"{tf_label} Bearish OB",
                        })
                else:
                    bj = broken_idx
                    if (c[bj] - o[bj]) >= 0.70 * a and (bj + 1 >= n - 1 or np.min(c[bj + 1 : n - 1]) >= ob_bot - 0.1 * a):
                        brk_touches = sum(1 for j in range(bj + 1, n - 1) if l[j] <= ob_top)
                        if brk_touches <= 2:
                            breakers.append({
                                "tf": tf_label, "idx": bj, "dir": 1, "top": ob_top, "bot": ob_bot,
                                "swept": swept, "touches": brk_touches, "fresh": brk_touches == 0,
                                "disp_atr": round(abs(c[bj] - o[bj]) / max(a, 1e-12), 2), "has_fvg": True,
                                "time": int(times[bj]),
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


def _refine_zone_bounds(raw_lo: float, raw_hi: float, d: int, cur_p: float, max_w: float, min_w: float) -> Tuple[float, float]:
    """
    Merampingkan zona HTF yang terlalu lebar tanpa mendorong cur_p keluar dari zona
    apabila harga saat ini SUDAH berada di dalam zona tersebut.
    """
    lo, hi = (raw_hi, raw_lo) if raw_hi < raw_lo else (raw_lo, raw_hi)
    if max_w > 0 and (hi - lo) > max_w:
        if lo <= cur_p <= hi:
            if d == 1:
                hi = min(hi, max(cur_p + 0.25 * max_w, lo + max_w))
                lo = max(raw_lo, hi - max_w)
            else:
                lo = max(lo, min(cur_p - 0.25 * max_w, hi - max_w))
                hi = min(raw_hi, lo + max_w)
        else:
            if d == 1:
                lo = hi - max_w
            else:
                hi = lo + max_w

    if min_w > 0 and (hi - lo) < min_w:
        mid = (lo + hi) / 2.0
        lo = mid - 0.5 * min_w
        hi = mid + 0.5 * min_w
    return lo, hi


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
        if (d == 1 and ret_ls <= 0.85) or (d == -1 and ret_ls >= 1.85):
            pts += 1.0
            tags.append("Fuel Kontrarian Retail")

    if d * funding < -0.0002:
        pts += 0.5

    return " | ".join(tags), min(6.0, pts)


def _compute_trade_geometry(
    d: int,
    archetype: str,
    cur_p: float,
    poi_low: float,
    poi_high: float,
    htf_poi: Optional[dict],
    approach_code: str,
    symbol: str,
    cfg: Config,
    atr_1h: float,
    cur_atr: float,
    w_low: float,
    w_high: float,
    w_eq: float,
    r4_lo: float,
    r4_hi: float,
    wall_supply_4h: Optional[dict],
    wall_demand_4h: Optional[dict],
    map1: Optional[dict],
    map15: dict,
    l_arr: np.ndarray,
    h_arr: np.ndarray,
    htf_pts: float,
    cur_rv: float,
) -> dict:
    """
    Menghitung geometri Entry, Limit Entry, Structural SL ('Zoom-Out Check' ATR 1H),
    TP1/TP2 berbasis likuiditas 1H & pembatas Tembok 4H, serta Estimasi Durasi Posisi.
    """
    notes: List[str] = []
    max_zone_w = min(0.80 * atr_1h, 0.035 * cur_p)
    poi_low, poi_high = _refine_zone_bounds(poi_low, poi_high, d, cur_p, max_zone_w, 0.14 * atr_1h)

    span_poi = max(poi_high - poi_low, 1e-12)
    ce_50 = (poi_low + poi_high) / 2.0

    # Penempatan Limit Entry cerdas sesuai karakter volatilitas kripto & gaya kedatangan
    if archetype == "PRE_REACTION":
        if approach_code == "KOMPRESI_KOREKTIF":
            # Pendekatan kompresi rapi -> ambil 35% dari bibir zona agar tidak tertinggal (missed fill)
            limit_entry = (poi_high - 0.35 * span_poi) if d == 1 else (poi_low + 0.35 * span_poi)
        else:
            # Pendekatan normal/cepat -> ambil 50% Equilibrium (CE) untuk meredam tusukan wick kripto
            limit_entry = ce_50
        in_poi_now = (poi_low - 0.05 * atr_1h) <= cur_p <= (poi_high + 0.04 * atr_1h)
        at_poi = in_poi_now
        plan_entry = cur_p if in_poi_now else limit_entry
    elif archetype == "HTF_REVERSAL":
        limit_entry = ce_50
        dist_from_edge = (cur_p - poi_high) if d == 1 else (poi_low - cur_p)
        at_poi = dist_from_edge <= 0.25 * atr_1h
        plan_entry = cur_p if at_poi else limit_entry
    else:
        limit_entry = ce_50
        dist_from_poi = (cur_p - poi_high) if d == 1 else (poi_low - cur_p)
        at_poi = dist_from_poi <= 0.45 * atr_1h
        plan_entry = cur_p if at_poi else limit_entry
        if not at_poi:
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
        local_swing_low = float(np.min(l_arr[-12:]))
        use_local = archetype != "PRE_REACTION" and abs(plan_entry - local_swing_low) <= 1.8 * atr_1h
        anchor_low = min(poi_low, local_swing_low) if use_local else poi_low
        if htf_poi and abs(plan_entry - htf_poi["bot"]) <= 1.9 * atr_1h:
            anchor_low = min(anchor_low, htf_poi["bot"])
        sl_struct = anchor_low - cfg.sl_buffer * atr_1h
        raw_dist = max(0.0, plan_entry - sl_struct)
        final_sl_dist = clamp(raw_dist, min_sl_dist, max_sl_dist)
        sl = min(plan_entry - final_sl_dist, poi_low - 0.08 * cur_atr)
    else:
        local_swing_high = float(np.max(h_arr[-12:]))
        use_local = archetype != "PRE_REACTION" and abs(local_swing_high - plan_entry) <= 1.8 * atr_1h
        anchor_high = max(poi_high, local_swing_high) if use_local else poi_high
        if htf_poi and abs(htf_poi["top"] - plan_entry) <= 1.9 * atr_1h:
            anchor_high = max(anchor_high, htf_poi["top"])
        sl_struct = anchor_high + cfg.sl_buffer * atr_1h
        raw_dist = max(0.0, sl_struct - plan_entry)
        final_sl_dist = clamp(raw_dist, min_sl_dist, max_sl_dist)
        sl = max(plan_entry + final_sl_dist, poi_high + 0.08 * cur_atr)

    risk = max(abs(plan_entry - sl), 1e-12)
    sl_atr_1h_ratio = risk / max(atr_1h, 1e-12)

    # --- TARGET TP1 & TP2 + BATAS DURASI POSISI ---
    vel_factor = 0.50 if (cur_rv >= 1.4 and htf_pts >= 14.0) else 0.43
    hourly_vel = max(vel_factor * atr_1h, 1e-12)
    eff_max_hold = min(cfg.max_hold_hours, 28.0) if archetype == "HTF_REVERSAL" else cfg.max_hold_hours
    max_reach_dist = eff_max_hold * hourly_vel

    if d == 1:
        wall_cap = None
        if wall_supply_4h and wall_supply_4h["bot"] > plan_entry + 0.25 * atr_1h:
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

        min_r_tp2 = 1.95 if archetype == "HTF_REVERSAL" else 1.80
        valid_tp2 = sorted([x for x in cands_tp if x >= plan_entry + min_r_tp2 * risk and x <= plan_entry + max_reach_dist])
        if valid_tp2:
            ideal_tp2 = [x for x in valid_tp2 if x >= plan_entry + 2.1 * risk]
            tp2 = ideal_tp2[0] if ideal_tp2 else valid_tp2[-1]
        else:
            tp2 = plan_entry + min((2.5 if archetype == "HTF_REVERSAL" else 2.8) * risk, max_reach_dist)

        if wall_cap is not None and tp2 > wall_cap:
            tp2 = wall_cap
            notes.append(f"TP2 dibatasi di bawah Tembok Supply 4H ({fp(wall_supply_4h['bot'])}) agar realistis tanpa tertahan lama.")

        # Pada HTF_REVERSAL utamakan Equilibrium 1H sebagai TP1 cepat jika memberikan >= 1.10R
        if archetype == "HTF_REVERSAL" and (plan_entry + 1.10 * risk <= w_eq <= plan_entry + 0.72 * max(tp2 - plan_entry, 0)):
            tp1 = w_eq
        else:
            valid_tp1 = sorted([x for x in cands_tp if plan_entry + 1.15 * risk <= x <= plan_entry + 0.68 * max(tp2 - plan_entry, 0)])
            tp1 = valid_tp1[0] if valid_tp1 else (plan_entry + 0.5 * (tp2 - plan_entry))
        opp_lvl = wall_supply_4h["bot"] if wall_supply_4h else w_high

    else:
        wall_cap = None
        if wall_demand_4h and wall_demand_4h["top"] < plan_entry - 0.25 * atr_1h:
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

        min_r_tp2 = 1.95 if archetype == "HTF_REVERSAL" else 1.80
        valid_tp2 = sorted([x for x in cands_tp if x <= plan_entry - min_r_tp2 * risk and x >= plan_entry - max_reach_dist], reverse=True)
        if valid_tp2:
            ideal_tp2 = [x for x in valid_tp2 if x <= plan_entry - 2.1 * risk]
            tp2 = ideal_tp2[0] if ideal_tp2 else valid_tp2[-1]
        else:
            tp2 = plan_entry - min((2.5 if archetype == "HTF_REVERSAL" else 2.8) * risk, max_reach_dist)

        if wall_cap is not None and tp2 < wall_cap:
            tp2 = wall_cap
            notes.append(f"TP2 dibatasi di atas Tembok Demand 4H ({fp(wall_demand_4h['top'])}) agar posisi tidak tertahan terlalu lama.")

        if archetype == "HTF_REVERSAL" and (plan_entry - 0.72 * max(plan_entry - tp2, 0) <= w_eq <= plan_entry - 1.10 * risk):
            tp1 = w_eq
        else:
            valid_tp1 = sorted(
                [x for x in cands_tp if plan_entry - 0.68 * max(plan_entry - tp2, 0) <= x <= plan_entry - 1.15 * risk],
                reverse=True,
            )
            tp1 = valid_tp1[0] if valid_tp1 else (plan_entry - 0.5 * (plan_entry - tp2))
        opp_lvl = wall_demand_4h["top"] if wall_demand_4h else w_low

    rr = abs(tp2 - plan_entry) / risk
    rr_tp1 = abs(tp1 - plan_entry) / risk
    est_h1 = max(0.5, abs(tp1 - plan_entry) / hourly_vel)
    est_h2 = max(1.0, abs(tp2 - plan_entry) / hourly_vel)
    hold_label = _format_hold_duration(est_h1, est_h2)

    # Estimasi waktu jemput bibir POI (jam) jika harga masih mendekati zona
    if at_poi:
        est_tap_h = 0.0
    else:
        dist_edge = max(0.0, (cur_p - poi_high) if d == 1 else (poi_low - cur_p))
        est_tap_h = round(dist_edge / max(0.50 * atr_1h, 1e-12), 1)

    return {
        "poi_low": poi_low,
        "poi_high": poi_high,
        "limit_entry": limit_entry,
        "plan_entry": plan_entry,
        "at_poi": at_poi,
        "sl": sl,
        "risk": risk,
        "min_sl_dist": min_sl_dist,
        "max_sl_dist": max_sl_dist,
        "sl_atr_1h_ratio": sl_atr_1h_ratio,
        "tp1": tp1,
        "tp2": tp2,
        "rr": rr,
        "rr_tp1": rr_tp1,
        "opp_lvl": opp_lvl,
        "hourly_vel": hourly_vel,
        "est_h1": est_h1,
        "est_h2": est_h2,
        "hold_label": hold_label,
        "est_tap_h": est_tap_h,
        "notes": notes,
    }


def populate_forward_playbook(s: Setup) -> None:
    """
    Menyusun panduan taktis ke depan (Forward If-Then Playbook + Flip Scenario) secara spesifik
    sesuai Variasi Setup (TREND_CONT, PRE_REACTION, HTF_REVERSAL, BREAKOUT) & karakter liar crypto.
    """
    is_long = s.direction == "LONG"
    opp_dir = "SHORT" if is_long else "LONG"
    prox_edge = s.poi_high if is_long else s.poi_low
    ce_mid = (s.poi_low + s.poi_high) / 2.0 if (not math.isnan(s.poi_low) and not math.isnan(s.poi_high)) else s.entry
    dist_txt = f"{s.dist_to_poi_pct:.2f}%" if not math.isnan(s.dist_to_poi_pct) else "-"
    tap_txt = f"~{s.est_tap_hours:.1f} jam" if (not math.isnan(s.est_tap_hours) and s.est_tap_hours > 0) else "detik ini"

    if s.archetype == "PRE_REACTION":
        if s.exec_style == "LIMIT_PRE_TAP":
            s.forward_playbook = (
                f"SEBELUM BEREAKSI ({s.approach_style}): Harga berjarak {dist_txt} menuju zona segar [{s.entry_zone}] (est. jemput {tap_txt}). "
                f"Langkah Utama: Pasang Limit {s.direction} di {fp(s.entry)} (atau split 50% bibir {fp(prox_edge)} & 50% CE {fp(ce_mid)}), "
                f"SL {fp(s.sl)}; saat TP1 {fp(s.tp1)} tersentuh wajib ambil 50% & geser SL ke BEP ({s.est_hold_label})."
            )
        else:
            s.forward_playbook = (
                f"PENDEKATAN LIAR KRIPTO ({s.approach_style}): Harga mendekati [{s.entry_zone}] (jarak {dist_txt}) dengan momentum tajam. "
                f"Langkah Utama: JANGAN pasang limit buta di bibir {fp(prox_edge)} (rawan tusukan stop-hunt wick); tunggu harga menusuk ke "
                f"50% CE {fp(ce_mid)} + konfirmasi close 15M memantul (PinBar/CISD) baru entry {s.direction} dengan SL {fp(s.sl)}."
            )
        s.flip_scenario = (
            f"Antisipasi Jebol (Breaker Flip): Jika candle 15M/1H close impulsif ber-volume menembus SL {fp(s.sl)} tanpa pantulan, "
            f"batalkan {s.direction} — zona [{s.entry_zone}] berubah menjadi Breaker Block untuk cari retest {opp_dir}."
        )

    elif s.archetype == "HTF_REVERSAL":
        if s.status == "ENTRY READY":
            s.forward_playbook = (
                f"REVERSAL TERKONFIRMASI DI AREA KUNCI ({s.zone_badge}): Likuiditas sudah tersapu & struktur mikro berbalik. "
                f"Langkah Utama: Eksekusi {s.direction} di dekat {fp(s.entry)} (SL {fp(s.sl)}), wajib amankan 50% cepat di TP1 Equilibrium 1H "
                f"({fp(s.tp1)}) + geser SL ke BEP, sisanya kawal ke TP2 {fp(s.tp2)}."
            )
        else:
            s.forward_playbook = (
                f"POTENSI KETERBALIKAN TREN DI AREA PENTING [{s.entry_zone}] (Jarak {dist_txt}): "
                f"Langkah Utama: Hentikan kejar tren {opp_dir} di depan tembok ini! Tunggu Judas Swing / sapuan likuiditas + "
                f"konfirmasi close 15M CHOCH/CISD di dalam [{s.entry_zone}] sebelum eksekusi {s.direction} menuju EQ 1H {fp(s.tp1)}."
            )
        s.flip_scenario = (
            f"Skenario Lanjut Tren (Batal Reversal): Jika harga tidak bereaksi dan close candle 1H solid menembus {fp(s.sl)} "
            f"disertai lonjakan OI/Volume, tembok mayor resmi jebol — batalkan reversal & ikuti kelanjutan tren {opp_dir} saat retest."
        )

    elif s.archetype == "BREAKOUT":
        if s.status == "ENTRY READY":
            s.forward_playbook = (
                f"BREAKOUT AKTIF: Eksekusi {s.direction} dekat {fp(s.entry)} (SL {fp(s.sl)}), amankan 50% di TP1 {fp(s.tp1)} -> SL ke BEP, "
                f"target penuh TP2 {fp(s.tp2)} ({s.est_hold_label})."
            )
        else:
            s.forward_playbook = (
                f"PANTAU BREAKOUT/RETEST [{s.entry_zone}]: Tunggu close 15M valid ber-volume >= 1.5x rata-rata atau retest bersih di {fp(s.level)} "
                f"dengan SL {fp(s.sl)}."
            )
        s.flip_scenario = (
            f"Antisipasi Fakeout/Deviation: Jika breakout gagal dan close 15M masuk kembali menembus {fp(s.sl)}, waspadai sapuan likuiditas "
            f"(fakeout) berbalik tajam ke arah {opp_dir}."
        )

    else:
        # TREND_CONT (Post-Reaction Follow Trend)
        if s.status == "ENTRY READY":
            s.forward_playbook = (
                f"FOLLOW-TREND TERKONFIRMASI ({s.zone_badge}): Reaksi di POI [{s.entry_zone}] valid searah struktur. "
                f"Langkah Utama: Eksekusi {s.direction} di area {fp(s.entry)} (SL proporsional 1H {fp(s.sl)}), ambil 50% di TP1 {fp(s.tp1)} "
                f"& geser SL ke BEP, sisanya kawal ke TP2 {fp(s.tp2)} ({s.est_hold_label})."
            )
        elif s.zone_state in ("IN_POI_TESTING", "FIRST_TOUCH_NOW", "SWEEP_IN_POI"):
            s.forward_playbook = (
                f"SEDANG MENGUJI POI [{s.entry_zone}]: Harga sudah di dalam zona namun candle belum menutup searah pantulan. "
                f"Langkah Utama: Tunggu candle 15M close hijau/merah (PinBar/CISD) di atas/bawah bibir zona baru entry {s.direction} (SL {fp(s.sl)})."
            )
        else:
            s.forward_playbook = (
                f"SUDAH BEREAKSI ({s.zone_badge}): Struktur {s.direction} valid namun harga sudah berjarak {dist_txt} dari POI [{s.entry_zone}]. "
                f"Langkah Utama: Jangan FOMO kejar harga pasar; pasang Limit Retest di {fp(s.limit_entry)} (SL {fp(s.sl)}) atau batalkan order "
                f"jika harga mencapai TP1 {fp(s.tp1)} lebih dulu tanpa jemput limit."
            )
        s.flip_scenario = (
            f"Skenario Invalidasi / Flip: Jika retest gagal dan candle 15M/1H close menembus SL {fp(s.sl)}, struktur tren patah "
            f"(CHOCH ke {opp_dir}) — cut loss disiplin & pantau peluang balik arah {opp_dir}."
        )


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
    Mesin Utama Dinamis 3-Archetype Top-Down MTF SMC/ICT (4H -> 1H -> 15M):
    1. Mengevaluasi TREND_CONT (setup konfirmasi yang sudah berjalan tajam selama ini).
    2. Mengevaluasi PRE_REACTION (setup koin yang mendekati Fresh Supply/Demand/OB sebelum bereaksi).
    3. Mengevaluasi HTF_REVERSAL (setup potensi pembalikan tren saat mendekati/menguji Tembok Mayor 4H/1H).
    Memilih kandidat paling segar, proporsional (Zoom-Out Check ATR 1H), dan realistis durasinya.
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
    if np.isnan(cur_atr) or cur_atr <= 0:
        return None

    # Hybrid Live Price: gunakan harga live_bar untuk deteksi jarak/sentuhan real-time,
    # sementara seluruh indikator & struktur tetap dihitung dari 100% closed bar (bebas repaint!).
    cur_p, l_live, h_live, live_bar = get_live_price_snapshot(df15)
    atr_pct = cur_atr / cur_p

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
        r4_lo, r4_hi = map4["range_low"], map4["range_high"]
        r4_pos = clamp((cur_p - r4_lo) / max(r4_hi - r4_lo, 1e-12) * 100.0, 0.0, 100.0)
        all_sup_4h = (
            [z for z in map4["bear_obs"] if z["top"] > cur_p - 0.05 * atr_4h]
            + [z for z in map4["breakers"] if z["dir"] == -1 and z["top"] > cur_p - 0.05 * atr_4h]
            + [z for z in map4["fvgs"] if z["dir"] == -1 and z["top"] > cur_p - 0.05 * atr_4h]
        )
        in_supply_4h_now = any(z["bot"] <= cur_p <= z["top"] for z in all_sup_4h)
        sup_above = [z for z in all_sup_4h if z["bot"] > cur_p]
        if sup_above:
            wall_supply_4h = min(sup_above, key=lambda z: z["bot"] - cur_p)
        elif all_sup_4h:
            wall_supply_4h = min(all_sup_4h, key=lambda z: abs(z["top"] - cur_p))

        all_dem_4h = (
            [z for z in map4["bull_obs"] if z["bot"] < cur_p + 0.05 * atr_4h]
            + [z for z in map4["breakers"] if z["dir"] == 1 and z["bot"] < cur_p + 0.05 * atr_4h]
            + [z for z in map4["fvgs"] if z["dir"] == 1 and z["bot"] < cur_p + 0.05 * atr_4h]
        )
        in_demand_4h_now = any(z["bot"] <= cur_p <= z["top"] for z in all_dem_4h)
        dem_below = [z for z in all_dem_4h if z["top"] < cur_p]
        if dem_below:
            wall_demand_4h = max(dem_below, key=lambda z: z["top"] - cur_p)
        elif all_dem_4h:
            wall_demand_4h = max(all_dem_4h, key=lambda z: -abs(cur_p - z["bot"]))

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
    else:
        w_low = map15["range_low"]
        w_high = map15["range_high"]
        w_eq = map15["eq"]
    w_pos_pct = clamp((cur_p - w_low) / max(w_high - w_low, 1e-12) * 100.0, 0.0, 100.0)

    in_discount_1h = w_pos_pct <= 50.0
    in_premium_1h = w_pos_pct >= 50.0
    if w_pos_pct <= 45.0:
        zone_pos_lbl = f"Discount 1H ({w_pos_pct:.1f}%)"
    elif w_pos_pct >= 55.0:
        zone_pos_lbl = f"Premium 1H ({w_pos_pct:.1f}%)"
    else:
        zone_pos_lbl = f"Equilibrium 1H ({w_pos_pct:.1f}%)"

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

    # Cek sapuan likuiditas 1H pada 6 bar 15M terakhir (termasuk live_bar jika ada)
    h1_bull_sweep = False
    h1_bear_sweep = False
    if map1:
        for sw in map1.get("lows", [])[-4:]:
            lvl1 = float(sw["price"])
            if (any(l[n - k] < lvl1 and c[n - k] >= lvl1 for k in range(1, min(7, n))) or (live_bar and l_live < lvl1 <= cur_p)) and cur_p >= lvl1 - 0.05 * atr_1h:
                h1_bull_sweep = True
                break
        for sw in map1.get("highs", [])[-4:]:
            lvl1 = float(sw["price"])
            if (any(h[n - k] > lvl1 and c[n - k] <= lvl1 for k in range(1, min(7, n))) or (live_bar and h_live > lvl1 >= cur_p)) and cur_p <= lvl1 + 0.05 * atr_1h:
                h1_bear_sweep = True
                break
        if any((l[-1] < eq_l or l_live < eq_l) and cur_p >= eq_l for eq_l in map1.get("eql", [])[-3:]):
            h1_bull_sweep = True
        if any((h[-1] > eq_h or h_live > eq_h) and cur_p <= eq_h for eq_h in map1.get("eqh", [])[-3:]):
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

    def htf_score(d_dir: int) -> float:
        al4 = 10.0 if (ctx4["st"] == d_dir or (ctx4["close"] > ctx4["e50"] and d_dir == 1) or (ctx4["close"] < ctx4["e50"] and d_dir == -1)) else 4.0
        al1 = 10.0 if (ctx1["st"] == d_dir or (ctx1["close"] > ctx1["e50"] and d_dir == 1) or (ctx1["close"] < ctx1["e50"] and d_dir == -1)) else 4.0
        return al4 + al1

    bull_htf = htf_score(1)
    bear_htf = htf_score(-1)
    of_bull_str, of_bull_pts = _format_orderflow_summary(cvd_ctx, deriv_ctx, meta.get("funding", 0.0), 1)
    of_bear_str, of_bear_pts = _format_orderflow_summary(cvd_ctx, deriv_ctx, meta.get("funding", 0.0), -1)

    def make_setup_obj(
        d_val: int,
        archetype: str,
        archetype_lbl: str,
        exec_style: str,
        app_lbl: str,
        kind_name: str,
        pat_name: str,
        poi_lbl: str,
        z_fresh: bool,
        geom: dict,
        tot_score: float,
        parts: list,
        status_str: str,
        confluences: List[str],
        extra_notes: Optional[List[str]] = None,
    ) -> Setup:
        p_lo, p_hi = geom["poi_low"], geom["poi_high"]
        plan_e = geom["plan_entry"]
        sl_p, tp1_p, tp2_p = geom["sl"], geom["tp1"], geom["tp2"]
        zone_s = f"{fp(p_lo)} - {fp(p_hi)}"
        conf_s = ", ".join(confluences[:3])
        conf_full = f"[{archetype_lbl} | {zone_pos_lbl} | Est. {geom['hold_label']}]: " + ", ".join(confluences)
        if d_val == 1:
            inval_s = f"Close 1H/15M menembus bawah SL struktural {fp(sl_p)} (di bawah zona POI {fp(p_lo)})."
        else:
            inval_s = f"Close 1H/15M menembus atas SL struktural {fp(sl_p)} (di atas zona POI {fp(p_hi)})."

        all_notes = list(geom["notes"]) + (list(extra_notes) if extra_notes else [])
        of_str = of_bull_str if d_val == 1 else of_bear_str

        raw_data = {
            "archetype": archetype,
            "archetype_label": archetype_lbl,
            "exec_style": exec_style,
            "approach_style": app_lbl,
            "est_tap_hours": geom["est_tap_h"],
            "current_price": cur_p,
            "plan_entry": plan_e,
            "limit_entry": geom["limit_entry"],
            "at_poi": geom["at_poi"],
            "poi_low": p_lo,
            "poi_high": p_hi,
            "poi_label": poi_lbl,
            "min_sl_dist": geom["min_sl_dist"],
            "max_sl_dist": geom["max_sl_dist"],
            # Lapis 4H
            "range_4h_low": r4_lo,
            "range_4h_high": r4_hi,
            "range_4h_pos_pct": r4_pos,
            "range_4h_str": range_4h_str,
            "walls_4h_str": walls_4h_str,
            "atr_4h": atr_4h,
            "obs_4h": [
                {"dir": z["dir"], "bot": z["bot"], "top": z["top"], "fresh": z.get("fresh", True), "kind": z["kind"]}
                for z in ((map4["bull_obs"][-2:] + map4["bear_obs"][-2:] + map4["breakers"][-2:]) if map4 else [])
            ],
            # Lapis 1H
            "range_1h_low": w_low,
            "range_1h_high": w_high,
            "range_1h_eq": w_eq,
            "range_1h_pos_pct": w_pos_pct,
            "zone_pos_label": zone_pos_lbl,
            "atr_1h": atr_1h,
            "sl_atr_1h_ratio": geom["sl_atr_1h_ratio"],
            "hourly_velocity": geom["hourly_vel"],
            "est_hold_hours_tp1": round(geom["est_h1"], 1),
            "est_hold_hours_tp2": round(geom["est_h2"], 1),
            "est_hold_label": geom["hold_label"],
            "obs_1h": [
                {"dir": z["dir"], "bot": z["bot"], "top": z["top"], "swept": z.get("swept", False), "fresh": z.get("fresh", True), "kind": z["kind"]}
                for z in ((map1["bull_obs"][-2:] + map1["bear_obs"][-2:] + map1["breakers"][-2:]) if map1 else [])
            ],
            "fvgs_1h": [{"dir": f["dir"], "bot": f["bot"], "top": f["top"]} for f in (map1["fvgs"][-3:] if map1 else [])],
            "eqh_1h": map1["eqh"][-3:] if map1 else [],
            "eql_1h": map1["eql"][-3:] if map1 else [],
            # Lapis 15M
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
            "breakers_15m": [{"dir": b["dir"], "top": b["top"], "bot": b["bot"], "fresh": b.get("fresh", True), "kind": b["kind"]} for b in map15["breakers"][-2:]],
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
            "orderflow_summary": of_str,
            "trend_4h": trend_text(ctx4),
            "trend_1h": trend_text(ctx1),
            "atr": cur_atr,
            "vol24_usdt": meta.get("vol24", 0),
            "spread_bps": meta.get("spread_bps", 0),
            "funding_rate": meta.get("funding", 0.0),
        }

        return Setup(
            symbol=symbol,
            direction="LONG" if d_val == 1 else "SHORT",
            kind=kind_name,
            pattern=pat_name,
            bar_index=n - 1,
            candle_time=ts_utc(int(times[-1])),
            level=float(w_eq),
            entry=float(plan_e),
            market_price=float(cur_p),
            limit_entry=float(geom["limit_entry"]),
            entry_zone=zone_s,
            poi_low=float(p_lo),
            poi_high=float(p_hi),
            sl=float(sl_p),
            tp1=float(tp1_p),
            tp2=float(tp2_p),
            rr=float(geom["rr"]),
            rr_tp1=float(geom["rr_tp1"]),
            risk_pct=geom["risk"] / plan_e * 100.0,
            tp1_pct=abs(tp1_p - plan_e) / plan_e * 100.0,
            tp2_pct=abs(tp2_p - plan_e) / plan_e * 100.0,
            score=float(tot_score),
            python_score=float(tot_score),
            parts=parts,
            status=status_str,
            python_status=status_str,
            confirmed=bool(geom["at_poi"]),
            extended=not bool(geom["at_poi"]),
            rel_vol=cur_rv,
            rsi=cur_rsi,
            atr=cur_atr,
            atr_pct=atr_pct,
            atr_1h=atr_1h,
            atr_1h_pct=atr_1h / cur_p,
            sl_atr_1h_ratio=geom["sl_atr_1h_ratio"],
            upper_line=float(p_hi),
            lower_line=float(p_lo),
            opp_level=float(geom["opp_lvl"]) if geom["opp_lvl"] is not None else None,
            trend_4h=trend_text(ctx4),
            struct_1h=trend_text(ctx1),
            trend_4h_short=trend_short(ctx4),
            struct_1h_short=trend_short(ctx1),
            confirm_txt=conf_full,
            confirm_short=conf_s,
            invalidation=inval_s,
            notes=all_notes,
            range_4h_str=range_4h_str,
            walls_4h_str=walls_4h_str,
            range_1h_low=float(w_low),
            range_1h_high=float(w_high),
            range_1h_eq=float(w_eq),
            range_1h_pos_pct=float(w_pos_pct),
            est_hold_hours_tp1=float(geom["est_h1"]),
            est_hold_hours_tp2=float(geom["est_h2"]),
            est_hold_label=geom["hold_label"],
            session_name=sess_ctx["session"],
            cvd_state=cvd_ctx["cvd_state"],
            taker_buy_ratio=float(cvd_ctx["taker_buy_ratio"]),
            oi_change_1h_pct=float(deriv_ctx.get("oi_change_1h_pct", NAN)),
            oi_change_4h_pct=float(deriv_ctx.get("oi_change_4h_pct", NAN)),
            top_ls_ratio=float(deriv_ctx.get("top_ls_ratio", NAN)),
            retail_ls_ratio=float(deriv_ctx.get("retail_ls_ratio", NAN)),
            orderflow_summary=of_str,
            zone_fresh=z_fresh,
            archetype=archetype,
            archetype_label=archetype_lbl,
            exec_style=exec_style,
            approach_style=app_lbl,
            est_tap_hours=float(geom["est_tap_h"]),
            meta=dict(meta),
            last_candle=ts_utc(int(times[-1])),
            raw_market_data=raw_data,
        )

    candidates: List[Tuple[int, float, Setup]] = []  # (priority_tier, rank_score, Setup)

    # =================================================================
    # ARCHETYPE 1: TREND_CONT (Sistem Post-Reaction / Konfirmasi yang Dipertahankan Utuh)
    # =================================================================
    def find_best_poi_cont(d_dir: int) -> Tuple[Optional[dict], Optional[dict], str]:
        htf_cands: List[dict] = []
        if map4:
            if d_dir == 1:
                z4_c = map4["bull_obs"] + [b for b in map4["breakers"] if b["dir"] == 1]
                htf_cands.extend([z for z in z4_c if (l_live <= z["top"] + 0.35 * atr_1h) and (cur_p >= z["bot"] - 0.15 * atr_1h)])
            else:
                z4_c = map4["bear_obs"] + [b for b in map4["breakers"] if b["dir"] == -1]
                htf_cands.extend([z for z in z4_c if (h_live >= z["bot"] - 0.35 * atr_1h) and (cur_p <= z["top"] + 0.15 * atr_1h)])
        if map1:
            if d_dir == 1:
                htf_cands.extend(map1["bull_obs"] + [b for b in map1["breakers"] if b["dir"] == 1] + [f for f in map1["fvgs"] if f["dir"] == 1])
            else:
                htf_cands.extend(map1["bear_obs"] + [b for b in map1["breakers"] if b["dir"] == -1] + [f for f in map1["fvgs"] if f["dir"] == -1])

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

        def near_price(z: dict, max_atr_1h: float) -> bool:
            if d_dir == 1:
                return (l_live <= z["top"] + max_atr_1h * atr_1h) and (cur_p >= z["bot"] - 0.25 * atr_1h)
            return (h_live >= z["bot"] - max_atr_1h * atr_1h) and (cur_p <= z["top"] + 0.25 * atr_1h)

        near_htf = [z for z in htf_cands if near_price(z, 0.90)]
        near_ltf = [z for z in ltf_cands if near_price(z, 0.65)]

        for hz in reversed(near_htf):
            for lz in reversed(near_ltf):
                if _zones_overlap(hz, lz, 0.20 * atr_1h):
                    return hz, lz, f"Nested {hz['kind']} + {lz['kind']}"

        touching_htf = [z for z in near_htf if near_price(z, 0.35)]
        if touching_htf:
            hz = touching_htf[-1]
            return hz, None, hz["kind"]

        touching_ltf = [z for z in near_ltf if near_price(z, 0.35)]
        if touching_ltf:
            lz = touching_ltf[-1]
            return (near_htf[-1] if near_htf else None), lz, lz["kind"]

        if near_htf:
            return near_htf[-1], None, near_htf[-1]["kind"]
        if near_ltf:
            return None, near_ltf[-1], near_ltf[-1]["kind"]
        return None, None, ""

    htf_bull_poi, ltf_bull_poi, poi_type_bull = find_best_poi_cont(1)
    htf_bear_poi, ltf_bear_poi, poi_type_bear = find_best_poi_cont(-1)
    has_poi_bull = (htf_bull_poi is not None) or (ltf_bull_poi is not None)
    has_poi_bear = (htf_bear_poi is not None) or (ltf_bear_poi is not None)

    bull_nested_bonus = 4.0 if (htf_bull_poi and ltf_bull_poi) else (2.0 if htf_bull_poi else 0.0)
    bear_nested_bonus = 4.0 if (htf_bear_poi and ltf_bear_poi) else (2.0 if htf_bear_poi else 0.0)
    bull_swept_flag = bool((ltf_bull_poi and ltf_bull_poi.get("swept")) or (htf_bull_poi and htf_bull_poi.get("swept")))
    bear_swept_flag = bool((ltf_bear_poi and ltf_bear_poi.get("swept")) or (htf_bear_poi and htf_bear_poi.get("swept")))

    bull_poi_score = min(25.0, (14.0 if has_poi_bull else 0.0) + (4.0 if bull_swept_flag else 0.0) + bull_nested_bonus + (8.0 if in_discount_1h else (3.0 if w_pos_pct < 62.0 else 0.0)))
    bear_poi_score = min(25.0, (14.0 if has_poi_bear else 0.0) + (4.0 if bear_swept_flag else 0.0) + bear_nested_bonus + (8.0 if in_premium_1h else (3.0 if w_pos_pct > 38.0 else 0.0)))

    bull_struct_score = (12.0 if has_bull_sweep else 0.0) + (10.0 if has_bull_choch else (7.0 if has_bull_bos else 0.0)) + (5.0 if idm_bull_swept else 0.0)
    bear_struct_score = (12.0 if has_bear_sweep else 0.0) + (10.0 if has_bear_choch else (7.0 if has_bear_bos else 0.0)) + (5.0 if idm_bear_swept else 0.0)

    bull_trigger_score = min(15.0, (9.0 if has_bull_pin else 0.0) + (7.0 if has_bull_cisd else 0.0) + (4.0 if cur_rv >= 1.4 else 0.0) + of_bull_pts * 0.6)
    bear_trigger_score = min(15.0, (9.0 if has_bear_pin else 0.0) + (7.0 if has_bear_cisd else 0.0) + (4.0 if cur_rv >= 1.4 else 0.0) + of_bear_pts * 0.6)

    bull_wall_pen = 8.0 if (wall_supply_4h and wall_supply_4h["bot"] - cur_p < 0.35 * atr_4h and cur_p >= wall_supply_4h["bot"] - 0.2 * atr_4h) else 0.0
    bear_wall_pen = 8.0 if (wall_demand_4h and cur_p - wall_demand_4h["top"] < 0.35 * atr_4h and cur_p <= wall_demand_4h["top"] + 0.2 * atr_4h) else 0.0
    counter_htf_bull_pen = 7.0 if bull_htf <= 8.0 else 0.0
    counter_htf_bear_pen = 7.0 if bear_htf <= 8.0 else 0.0

    total_bull = bull_htf * 0.85 + bull_poi_score + bull_struct_score + bull_trigger_score - bull_wall_pen - counter_htf_bull_pen
    total_bear = bear_htf * 0.85 + bear_poi_score + bear_struct_score + bear_trigger_score - bear_wall_pen - counter_htf_bear_pen

    cont_dirs: List[int] = []
    if total_bull >= total_bear:
        if total_bull >= 40.0:
            cont_dirs.append(1)
        if total_bear >= 40.0:
            cont_dirs.append(-1)
    else:
        if total_bear >= 40.0:
            cont_dirs.append(-1)
        if total_bull >= 40.0:
            cont_dirs.append(1)

    for d_cont in cont_dirs:
        htf_poi = htf_bull_poi if d_cont == 1 else htf_bear_poi
        ltf_poi = ltf_bull_poi if d_cont == 1 else ltf_bear_poi
        poi_label = poi_type_bull if d_cont == 1 else poi_type_bear

        if htf_poi and ltf_poi and _zones_overlap(htf_poi, ltf_poi, 0.20 * atr_1h):
            p_lo = max(htf_poi["bot"], ltf_poi["bot"])
            p_hi = min(htf_poi["top"], ltf_poi["top"])
            if p_hi <= p_lo:
                p_lo, p_hi = ltf_poi["bot"], ltf_poi["top"]
        elif ltf_poi:
            p_lo, p_hi = ltf_poi["bot"], ltf_poi["top"]
        elif htf_poi:
            p_lo, p_hi = htf_poi["bot"], htf_poi["top"]
        else:
            if d_cont == 1:
                base_l = float(np.min(l[-6:]))
                p_lo, p_hi = base_l, base_l + 0.35 * atr_1h
            else:
                base_h = float(np.max(h[-6:]))
                p_lo, p_hi = base_h - 0.35 * atr_1h, base_h

        htf_pts_c = bull_htf if d_cont == 1 else bear_htf
        app_code_c, app_lbl_c, _ = analyze_approach_dynamics(df15, d_cont, cur_atr, live_bar=live_bar)
        geom_c = _compute_trade_geometry(
            d_cont, "TREND_CONT", cur_p, p_lo, p_hi, htf_poi, app_code_c, symbol, cfg,
            atr_1h, cur_atr, w_low, w_high, w_eq, r4_lo, r4_hi, wall_supply_4h, wall_demand_4h,
            map1, map15, l, h, htf_pts_c, cur_rv,
        )

        confs_c: List[str] = [poi_label or ("1H/15M Demand Zone" if d_cont == 1 else "1H/15M Supply Zone"), zone_pos_lbl]
        if d_cont == 1:
            if has_bull_sweep:
                confs_c.append("SSL Sweep" + (" (Asian Low)" if sess_ctx["asian_ssl_swept"] else ""))
            if has_bull_choch:
                confs_c.append("Bullish CHOCH")
            if has_bull_bos:
                confs_c.append("Bullish BOS")
            if idm_bull_swept:
                confs_c.append("IDM Taken")
            if has_bull_cisd:
                confs_c.append("Bullish CISD")
            if has_bull_pin:
                confs_c.append("Pin Bar Rejection")
            if cvd_ctx.get("bull_absorption"):
                confs_c.append("CVD Bull Absorption")

            if has_bull_sweep and (has_bull_choch or has_bull_cisd) and has_poi_bull:
                pat_c, kind_c = "MTF Sniper Buy (Sweep + CHOCH + 1H/15M POI)", "SMC_SNIPER"
            elif htf_poi and ltf_poi:
                pat_c, kind_c = "MTF Nested Demand (1H + 15M POI Alignment)", "SMC_OB_RETEST"
            elif poi_label and "Breaker" in poi_label:
                pat_c, kind_c = "SMC Bullish Breaker Block Retest", "SMC_BREAKER"
            elif poi_label and "OB" in poi_label:
                pat_c, kind_c = "SMC Demand Bounce (Order Block Retest)", "SMC_OB_RETEST"
            elif poi_label and "FVG" in poi_label:
                pat_c, kind_c = "SMC FVG Retest Entry (Unmitigated Gap)", "SMC_FVG_FILL"
            elif has_bull_sweep:
                pat_c, kind_c = "SMC SSL Liquidity Sweep Reversal", "SMC_SWEEP"
            else:
                pat_c, kind_c = "SMC MTF Demand Setup", "SMC_DEMAND"
        else:
            if has_bear_sweep:
                confs_c.append("BSL Sweep" + (" (Asian High)" if sess_ctx["asian_bsl_swept"] else ""))
            if has_bear_choch:
                confs_c.append("Bearish CHOCH")
            if has_bear_bos:
                confs_c.append("Bearish BOS")
            if idm_bear_swept:
                confs_c.append("IDM Taken")
            if has_bear_cisd:
                confs_c.append("Bearish CISD")
            if has_bear_pin:
                confs_c.append("Pin Bar Rejection")
            if cvd_ctx.get("bear_absorption"):
                confs_c.append("CVD Bear Absorption")

            if has_bear_sweep and (has_bear_choch or has_bear_cisd) and has_poi_bear:
                pat_c, kind_c = "MTF Sniper Short (Sweep + CHOCH + 1H/15M POI)", "SMC_SNIPER"
            elif htf_poi and ltf_poi:
                pat_c, kind_c = "MTF Nested Supply (1H + 15M POI Alignment)", "SMC_OB_RETEST"
            elif poi_label and "Breaker" in poi_label:
                pat_c, kind_c = "SMC Bearish Breaker Block Retest", "SMC_BREAKER"
            elif poi_label and "OB" in poi_label:
                pat_c, kind_c = "SMC Supply Rejection (Order Block Retest)", "SMC_OB_RETEST"
            elif poi_label and "FVG" in poi_label:
                pat_c, kind_c = "SMC FVG Retest Entry (Unmitigated Gap)", "SMC_FVG_FILL"
            elif has_bear_sweep:
                pat_c, kind_c = "SMC BSL Liquidity Sweep Reversal", "SMC_SWEEP"
            else:
                pat_c, kind_c = "SMC MTF Supply Setup", "SMC_SUPPLY"

        rr_c = geom_c["rr"]
        rr_pts_c = 10.0 if rr_c >= 2.8 else (8.5 if rr_c >= 2.3 else (6.5 if rr_c >= 2.0 else (2.5 if rr_c >= 1.5 else 0.0)))
        liq_pts_c = liquidity_score(meta, d_cont, atr_pct)
        p1 = min(20.0, htf_pts_c)
        p2 = min(25.0, bull_poi_score if d_cont == 1 else bear_poi_score)
        p3 = min(25.0, bull_struct_score if d_cont == 1 else bear_struct_score)
        p4 = min(15.0, bull_trigger_score if d_cont == 1 else bear_trigger_score)
        p5 = min(10.0, rr_pts_c)
        p6 = min(5.0, liq_pts_c)
        tot_c = max(0.0, p1 + p2 + p3 + p4 + p5 + p6 - (bull_wall_pen if d_cont == 1 else bear_wall_pen))
        parts_c = [
            ("Konteks HTF & Tren", p1, 20),
            ("Kualitas POI & Zone", p2, 25),
            ("Struktur & Liquidity", p3, 25),
            ("Trigger & Orderflow", p4, 15),
            ("Risk to Reward", p5, 10),
            ("Likuiditas Pasar", p6, 5),
        ]

        recent_pin_2 = any(p["dir"] == d_cont and p["idx"] >= n - 2 for p in pin_bars)
        recent_cisd_3 = any(e["type"] == "CISD" and e["dir"] == d_cont and e["idx"] >= n - 3 for e in struct_events)
        if d_cont == 1:
            candle_rejected = (c[-1] > o[-1] and c[-1] >= geom_c["poi_low"]) or recent_pin_2 or recent_cisd_3
        else:
            candle_rejected = (c[-1] < o[-1] and c[-1] <= geom_c["poi_high"]) or recent_pin_2 or recent_cisd_3

        htf_aligned = htf_pts_c >= 14.0
        blocked_by_4h_wall = (d_cont == 1 and in_supply_4h_now) or (d_cont == -1 and in_demand_4h_now)
        tp2_move_pct = abs(geom_c["tp2"] - geom_c["plan_entry"]) / geom_c["plan_entry"] * 100.0
        notes_c: List[str] = []

        if geom_c["at_poi"] and candle_rejected and htf_aligned and not blocked_by_4h_wall and tot_c >= cfg.smc_entry_score and rr_c >= cfg.min_rr and tp2_move_pct >= 1.4:
            status_c = "ENTRY READY"
            exec_c = "ENTRY_NOW"
            tier_c = 0
        elif tot_c >= cfg.smc_watch_score and rr_c >= 1.6 and tp2_move_pct >= 1.0:
            status_c = "WATCHLIST"
            exec_c = "WAIT_SWEEP_TRIGGER" if geom_c["at_poi"] else "LIMIT_RETEST"
            dist_from_edge = (cur_p - geom_c["poi_high"]) if d_cont == 1 else (geom_c["poi_low"] - cur_p)
            # Jika harga sudah di POI menunggu close candle -> tier 1; jika sudah memantul jauh -> tier 3 agar kalah dari PRE-TAP segar!
            tier_c = 1 if dist_from_edge <= 0.35 * atr_1h else 3
            if blocked_by_4h_wall:
                notes_c.append("Harga saat ini berada di dalam zona Tembok 4H berlawanan; ditahan di WATCHLIST.")
                tier_c = 3
            elif geom_c["at_poi"] and not candle_rejected:
                notes_c.append("Harga sedang menguji zona POI namun candle 15M terakhir belum menutup searah pantulan; tunggu konfirmasi.")
            elif not htf_aligned:
                notes_c.append("Setup berlawanan dengan tren 4H & 1H sekaligus; ditahan di WATCHLIST.")
        else:
            status_c = "REJECT"
            exec_c = "SKIP"
            tier_c = 4

        z_fresh_c = bool((ltf_poi and ltf_poi.get("fresh")) or (htf_poi and htf_poi.get("fresh")))
        st_cont = make_setup_obj(
            d_cont, "TREND_CONT", "Follow-Trend (Post-Reaction)", exec_c, app_lbl_c,
            kind_c, pat_c, poi_label, z_fresh_c, geom_c, tot_c, parts_c, status_c, confs_c, notes_c,
        )
        candidates.append((tier_c, tot_c, st_cont))

    # =================================================================
    # ARCHETYPE 2: PRE_REACTION (Deteksi SEBELUM Harga Bereaksi di Fresh Supply/Demand/OB)
    # =================================================================
    max_stalk_dist = min(cfg.max_stalk_atr_1h * atr_1h, (cfg.max_stalk_pct / 100.0) * cur_p)

    for d_pre in (1, -1):
        htf_p = bull_htf if d_pre == 1 else bear_htf
        blocked_wall = (d_pre == 1 and in_supply_4h_now) or (d_pre == -1 and in_demand_4h_now)
        if blocked_wall:
            continue
        # Wajib searah minimal salah satu HTF (4H atau 1H), atau berada di Discount/Premium makro 4H
        if htf_p < 14.0 and not ((d_pre == 1 and r4_pos <= 42.0) or (d_pre == -1 and r4_pos >= 58.0)):
            continue

        # Kumpulkan zona UNMITIGATED (touches == 0 / fresh == True) yang berada DI DEPAN harga (belum memantul di closed bar!)
        fresh_htf_zones: List[dict] = []
        if map4:
            z4_list = (map4["bull_obs"] + [b for b in map4["breakers"] if b["dir"] == 1]) if d_pre == 1 else (map4["bear_obs"] + [b for b in map4["breakers"] if b["dir"] == -1])
            fresh_htf_zones.extend([{**z, "_w": 3} for z in z4_list if z.get("fresh")])
        if map1:
            z1_list = (
                map1["bull_obs"] + [b for b in map1["breakers"] if b["dir"] == 1] + [f for f in map1["fvgs"] if f["dir"] == 1]
            ) if d_pre == 1 else (
                map1["bear_obs"] + [b for b in map1["breakers"] if b["dir"] == -1] + [f for f in map1["fvgs"] if f["dir"] == -1]
            )
            fresh_htf_zones.extend([{**z, "_w": 2} for z in z1_list if z.get("fresh")])

        fresh_ltf_zones = [
            z for z in (
                (map15["bull_obs"] + [b for b in map15["breakers"] if b["dir"] == 1]) if d_pre == 1
                else (map15["bear_obs"] + [b for b in map15["breakers"] if b["dir"] == -1])
            )
            if z.get("fresh")
        ]

        def is_ahead_and_unbounced(z: dict) -> Tuple[bool, float]:
            look_b = min(8, len(l))
            if d_pre == 1:
                dist = cur_p - z["top"]
                # Boleh sedikit masuk zona pada live_bar (first touch), asalkan closed bar terakhir belum memantul darinya!
                unbounced_closed = float(np.min(l[-look_b:])) > z["top"]
                ok = unbounced_closed and (cur_p >= z["bot"]) and (-0.15 * atr_1h <= dist <= max_stalk_dist)
                return ok, max(0.0, dist)
            else:
                dist = z["bot"] - cur_p
                unbounced_closed = float(np.max(h[-look_b:])) < z["bot"]
                ok = unbounced_closed and (cur_p <= z["top"]) and (-0.15 * atr_1h <= dist <= max_stalk_dist)
                return ok, max(0.0, dist)

        ahead_ltf_zones = [lz for lz in fresh_ltf_zones if is_ahead_and_unbounced(lz)[0]]
        valid_htf_pre = []
        for hz in fresh_htf_zones:
            ok_h, dist_h = is_ahead_and_unbounced(hz)
            if ok_h:
                # Hindari FVG 1H berdiri sendiri kecuali beririsan dengan 15M Fresh OB yang masih terjangkau
                has_nested_15 = any(_zones_overlap(hz, lz, 0.15 * atr_1h) for lz in ahead_ltf_zones)
                if "FVG" in hz["kind"] and not has_nested_15:
                    continue
                valid_htf_pre.append((dist_h, -hz["_w"], hz, has_nested_15))

        chosen_htf = None
        chosen_ltf = None
        if valid_htf_pre:
            valid_htf_pre.sort(key=lambda x: (0 if x[3] else 1, x[0], x[1]))
            _, _, chosen_htf, _ = valid_htf_pre[0]
            nested_matches = [lz for lz in ahead_ltf_zones if _zones_overlap(chosen_htf, lz, 0.18 * atr_1h)]
            if nested_matches:
                chosen_ltf = nested_matches[-1]
        elif htf_p >= 20.0 and ((d_pre == 1 and w_pos_pct <= 44.0) or (d_pre == -1 and w_pos_pct >= 56.0)):
            # Fallback Pre-Reaction: 15M Fresh Sweep-Origin OB di dalam Discount/Premium kuat saat tren 4H+1H selaras penuh
            ltf_sweep_pre = []
            for lz in fresh_ltf_zones:
                if lz.get("swept"):
                    ok_l, dist_l = is_ahead_and_unbounced(lz)
                    if ok_l and dist_l <= 0.95 * atr_1h:
                        ltf_sweep_pre.append((dist_l, lz))
            if ltf_sweep_pre:
                ltf_sweep_pre.sort(key=lambda x: x[0])
                chosen_ltf = ltf_sweep_pre[0][1]

        if chosen_htf is None and chosen_ltf is None:
            continue

        if chosen_htf and chosen_ltf:
            p_lo = max(chosen_htf["bot"], chosen_ltf["bot"])
            p_hi = min(chosen_htf["top"], chosen_ltf["top"])
            if p_hi <= p_lo:
                p_lo, p_hi = chosen_ltf["bot"], chosen_ltf["top"]
            poi_lbl_pre = f"Fresh Nested {chosen_htf['kind']} + {chosen_ltf['kind']}"
            z_hier_pts = 29.0
        elif chosen_htf:
            p_lo, p_hi = chosen_htf["bot"], chosen_htf["top"]
            poi_lbl_pre = f"Fresh {chosen_htf['kind']}"
            z_hier_pts = 26.0 if chosen_htf.get("tf") == "4H" else 23.0
            if chosen_htf.get("swept"):
                z_hier_pts = min(30.0, z_hier_pts + 3.0)
        else:
            p_lo, p_hi = chosen_ltf["bot"], chosen_ltf["top"]
            poi_lbl_pre = f"Fresh {chosen_ltf['kind']}"
            z_hier_pts = 20.0

        app_code_p, app_lbl_p, app_pts = analyze_approach_dynamics(df15, d_pre, cur_atr, live_bar=live_bar)
        geom_p = _compute_trade_geometry(
            d_pre, "PRE_REACTION", cur_p, p_lo, p_hi, chosen_htf, app_code_p, symbol, cfg,
            atr_1h, cur_atr, w_low, w_high, w_eq, r4_lo, r4_hi, wall_supply_4h, wall_demand_4h,
            map1, map15, l, h, htf_p, cur_rv,
        )
        ref_edge_dist = (cur_p - geom_p["poi_high"]) if d_pre == 1 else (geom_p["poi_low"] - cur_p)
        if geom_p["est_tap_h"] > 4.5 or ref_edge_dist > max_stalk_dist * 1.05:
            continue

        # Skor Kekuatan Displacement Awal Pembentuk Zona (0..18)
        ref_z = chosen_htf or chosen_ltf
        disp_v = float(ref_z.get("disp_atr", 1.1))
        has_fvg_z = bool(ref_z.get("has_fvg", True))
        disp_pts = 18.0 if (disp_v >= 1.6 and has_fvg_z) else (14.5 if (disp_v >= 1.15 or has_fvg_z) else 10.0)

        # Skor Inducement / Kolam Likuiditas Pengumpan di Depan POI (0..18)
        pre_pools: List[str] = []
        if d_pre == 1:
            all_l_pools = (map1["eql"] if map1 else []) + map15["eql"] + [s["price"] for s in lows_15[-4:]]
            if idm_level:
                all_l_pools.append(idm_level)
            near_pools = [x for x in all_l_pools if geom_p["poi_low"] <= x <= cur_p and (x - geom_p["poi_high"]) <= 0.75 * atr_1h]
            if near_pools:
                pre_pools.append(f"SSL/IDM Fuel @ {fp(near_pools[0])}")
        else:
            all_h_pools = (map1["eqh"] if map1 else []) + map15["eqh"] + [s["price"] for s in highs_15[-4:]]
            if idm_level:
                all_h_pools.append(idm_level)
            near_pools = [x for x in all_h_pools if cur_p <= x <= geom_p["poi_high"] and (geom_p["poi_low"] - x) <= 0.75 * atr_1h]
            if near_pools:
                pre_pools.append(f"BSL/IDM Fuel @ {fp(near_pools[0])}")

        if pre_pools:
            idm_fuel_pts = 17.0
        elif (d_pre == 1 and idm_bull_swept) or (d_pre == -1 and idm_bear_swept):
            idm_fuel_pts = 14.0
            pre_pools.append("IDM Taken")
        else:
            idm_fuel_pts = 8.0

        # Skor Konteks HTF, Posisi Range, R:R, & Orderflow (0..20)
        rr_p = geom_p["rr"]
        ctx_rr_pts = (
            (8.0 if htf_p >= 20.0 else (5.5 if htf_p >= 14.0 else 3.0))
            + (4.0 if ((d_pre == 1 and w_pos_pct <= 40.0) or (d_pre == -1 and w_pos_pct >= 60.0)) else 1.5)
            + (6.0 if rr_p >= 2.5 else (4.5 if rr_p >= 2.0 else (2.0 if rr_p >= 1.7 else 0.0)))
            + min(2.0, (of_bull_pts if d_pre == 1 else of_bear_pts) * 0.4)
        )
        ctx_rr_pts = min(20.0, ctx_rr_pts)

        tot_pre = clamp(z_hier_pts + disp_pts + idm_fuel_pts + app_pts + ctx_rr_pts, 0.0, 100.0)
        parts_pre = [
            ("Kesegaran & Hirarki POI", round(z_hier_pts, 1), 30),
            ("Kekuatan Displacement+FVG", round(disp_pts, 1), 18),
            ("Inducement / Magnet Depan", round(idm_fuel_pts, 1), 18),
            ("Dinamika Pendekatan Harga", round(app_pts, 1), 14),
            ("Konteks HTF, RR & Flow", round(ctx_rr_pts, 1), 20),
        ]

        confs_pre = [poi_lbl_pre, "Unmitigated (0 Touches)", app_lbl_p.split(" (")[0], zone_pos_lbl]
        confs_pre.extend(pre_pools[:1])

        tp2_pct_p = abs(geom_p["tp2"] - geom_p["plan_entry"]) / geom_p["plan_entry"] * 100.0
        notes_pre: List[str] = []
        # Limit buta sebelum reaksi (PRE-TAP LIMIT) wajib berjangkar pada Order Block (OB) HTF nyata agar SL dilindungi swing struktural (bukan FVG kosong)
        is_ob_anchor = (chosen_htf is not None) and (
            "OB" in chosen_htf["kind"]
            or (chosen_ltf is not None and "OB" in chosen_ltf["kind"] and chosen_ltf.get("swept"))
        )
        strong_htf_backing = is_ob_anchor and (htf_p >= 14.0) and (disp_v >= 1.20 or has_fvg_z)
        if strong_htf_backing and tot_pre >= 78.0 and rr_p >= max(cfg.min_rr, 2.2) and tp2_pct_p >= 1.45 and geom_p["est_h2"] <= cfg.max_hold_hours:
            if app_code_p == "IMPULSIF_LIAR":
                status_pre = "WATCHLIST"
                exec_pre = "WAIT_SWEEP_TRIGGER"
                tier_pre = 2
                notes_pre.append("Zona POI sangat segar, namun pendekatan candle saat ini tajam/impulsif; tunggu tusukan sweep/CISD sebelum entry.")
            else:
                status_pre = "PRE-TAP LIMIT"
                exec_pre = "LIMIT_PRE_TAP"
                tier_pre = 1
        elif tot_pre >= 60.0 and rr_p >= 1.80 and tp2_pct_p >= 1.0:
            status_pre = "WATCHLIST"
            exec_pre = "WAIT_SWEEP_TRIGGER" if (app_code_p == "IMPULSIF_LIAR" or not is_ob_anchor) else "LIMIT_PRE_TAP"
            tier_pre = 2
            if not is_ob_anchor:
                notes_pre.append("Zona Pre-Tap berbasis FVG/Breaker tanpa tameng Order Block HTF; tunggu sweep/reaksi mikro di dalam zona sebelum masuk.")
        else:
            continue

        pat_pre = (
            f"Pre-Tap Fresh Demand Stalker ({chosen_htf['tf'] if chosen_htf else '15M'} Unmitigated POI)"
            if d_pre == 1
            else f"Pre-Tap Fresh Supply Stalker ({chosen_htf['tf'] if chosen_htf else '15M'} Unmitigated POI)"
        )
        st_pre = make_setup_obj(
            d_pre, "PRE_REACTION", "Pre-Reaction (Sebelum Bereaksi)", exec_pre, app_lbl_p,
            "SMC_PRE_TAP", pat_pre, poi_lbl_pre, True, geom_p, tot_pre, parts_pre, status_pre, confs_pre, notes_pre,
        )
        candidates.append((tier_pre, tot_pre, st_pre))

    # =================================================================
    # ARCHETYPE 3: HTF_REVERSAL (Counter-Trend / Wall-Bounce Inversion di Area Penting 4H/1H)
    # =================================================================
    max_rev_dist = min(1.25 * atr_1h, 0.022 * cur_p)
    for d_rev in (1, -1):
        htf_r = bull_htf if d_rev == 1 else bear_htf
        # Reversal dikhususkan ketika setidaknya salah satu timeframe melawan arah (sedang tren menuju tembok!)
        if htf_r >= 20.0:
            continue

        extreme_range = (
            (d_rev == 1 and (r4_pos <= 25.0 or w_pos_pct <= 20.0))
            or (d_rev == -1 and (r4_pos >= 75.0 or w_pos_pct >= 80.0))
        )

        rev_walls: List[dict] = []
        if map4:
            z4_r = (map4["bull_obs"] + [b for b in map4["breakers"] if b["dir"] == 1]) if d_rev == 1 else (map4["bear_obs"] + [b for b in map4["breakers"] if b["dir"] == -1])
            rev_walls.extend([{**z, "_w": 3} for z in z4_r])
        if map1 and extreme_range:
            z1_r = (map1["bull_obs"] + [b for b in map1["breakers"] if b["dir"] == 1]) if d_rev == 1 else (map1["bear_obs"] + [b for b in map1["breakers"] if b["dir"] == -1])
            rev_walls.extend([{**z, "_w": 2} for z in z1_r if z.get("fresh") or z.get("swept")])

        look_w = min(8, len(l))
        near_walls = []
        for wz in rev_walls:
            if d_rev == 1:
                in_w = wz["bot"] - 0.08 * atr_1h <= cur_p <= wz["top"]
                dist_w = 0.0 if in_w else (cur_p - wz["top"] if cur_p > wz["top"] else 999.0)
                unbounced_w = float(np.min(l[-look_w:])) > wz["top"]
            else:
                in_w = wz["bot"] <= cur_p <= wz["top"] + 0.08 * atr_1h
                dist_w = 0.0 if in_w else (wz["bot"] - cur_p if cur_p < wz["bot"] else 999.0)
                unbounced_w = float(np.max(h[-look_w:])) < wz["bot"]
            if in_w or (0.0 <= dist_w <= max_rev_dist and (unbounced_w or dist_w <= 0.35 * atr_1h)):
                near_walls.append((0 if in_w else 1, dist_w, -wz["_w"], wz))

        if not near_walls:
            continue

        near_walls.sort(key=lambda x: (x[0], x[1], x[2]))
        in_wall_flag = near_walls[0][0] == 0
        wall_z = near_walls[0][3]

        # Cek bukti kelelahan tren (Exhaustion) / Liquidity Climax / Reversal Micro-Shift
        has_sw_r = has_bull_sweep if d_rev == 1 else has_bear_sweep
        has_shift_r = (has_bull_choch or has_bull_cisd) if d_rev == 1 else (has_bear_choch or has_bear_cisd)
        has_pin_r = has_bull_pin if d_rev == 1 else has_bear_pin
        rsi_ext = (d_rev == 1 and cur_rsi <= 35.0) or (d_rev == -1 and cur_rsi >= 65.0)
        cvd_abs = bool(cvd_ctx.get("bull_absorption") if d_rev == 1 else cvd_ctx.get("bear_absorption"))
        of_pts_r = of_bull_pts if d_rev == 1 else of_bear_pts

        # Jangan buat sinyal reversal jika belum di ekstrem range DAN tidak ada satupun tanda kelelahan/tembok 4H segar
        if not (has_sw_r or has_shift_r or rsi_ext or cvd_abs or (wall_z["_w"] == 3 and wall_z.get("fresh"))):
            continue

        p_lo, p_hi = wall_z["bot"], wall_z["top"]

        app_code_r, app_lbl_r, _ = analyze_approach_dynamics(df15, d_rev, cur_atr, live_bar=live_bar)
        geom_r = _compute_trade_geometry(
            d_rev, "HTF_REVERSAL", cur_p, p_lo, p_hi, wall_z, app_code_r, symbol, cfg,
            atr_1h, cur_atr, w_low, w_high, w_eq, r4_lo, r4_hi, wall_supply_4h, wall_demand_4h,
            map1, map15, l, h, htf_r, cur_rv,
        )

        wall_pts = (26.0 if wall_z["_w"] == 3 else 21.0) + (4.0 if wall_z.get("fresh") else 0.0)
        range_ext_pts = 15.0 if extreme_range else 8.0
        climax_pts = min(
            25.0,
            (12.0 if has_sw_r else 0.0)
            + (7.0 if rsi_ext else 0.0)
            + (6.0 if cvd_abs else 0.0)
            + min(4.0, of_pts_r * 0.8),
        )
        shift_pts = min(18.0, (11.0 if has_shift_r else 0.0) + (7.0 if has_pin_r else 0.0))
        rr_r = geom_r["rr"]
        rr_pts_r = 12.0 if rr_r >= 2.4 else (9.0 if rr_r >= 2.0 else (5.0 if rr_r >= 1.8 else 0.0))

        tot_rev = clamp(wall_pts + range_ext_pts + climax_pts + shift_pts + rr_pts_r, 0.0, 100.0)
        parts_rev = [
            ("Kekuatan Tembok 4H/1H", round(min(30.0, wall_pts), 1), 30),
            ("Ekstremitas Dealing Range", round(range_ext_pts, 1), 15),
            ("Liquidity Sweep & Climax", round(climax_pts, 1), 25),
            ("Shift Struktur 15M (CHOCH/CISD)", round(shift_pts, 1), 18),
            ("Ruang R:R ke Equilibrium", round(rr_pts_r, 1), 12),
        ]

        confs_rev = [f"Tembok {wall_z['kind']}", f"Makro 4H {r4_pos:.0f}% / 1H {w_pos_pct:.0f}%"]
        if has_sw_r:
            confs_rev.append("Liquidity Sweep")
        if has_shift_r:
            confs_rev.append("15M CHOCH/CISD Shift")
        if rsi_ext:
            confs_rev.append(f"RSI Ekstrem ({cur_rsi:.0f})")
        if cvd_abs:
            confs_rev.append("CVD Absorption")

        if d_rev == 1:
            candle_rej_r = (c[-1] > o[-1] and c[-1] >= geom_r["poi_low"]) or has_pin_r
        else:
            candle_rej_r = (c[-1] < o[-1] and c[-1] <= geom_r["poi_high"]) or has_pin_r

        tp2_pct_r = abs(geom_r["tp2"] - geom_r["plan_entry"]) / geom_r["plan_entry"] * 100.0
        notes_rev: List[str] = []

        if in_wall_flag and has_sw_r and has_shift_r and candle_rej_r and tot_rev >= cfg.smc_entry_score and rr_r >= cfg.min_rr and tp2_pct_r >= 1.4:
            status_rev = "ENTRY READY"
            exec_rev = "ENTRY_NOW"
            tier_rev = 0
            pat_rev = f"HTF Wall Reversal Confirmed ({wall_z['tf']} {'Demand' if d_rev==1 else 'Supply'} + Sweep + Shift)"
        elif tot_rev >= 58.0 and rr_r >= 1.80 and tp2_pct_r >= 1.1:
            status_rev = "REVERSAL WATCH"
            exec_rev = "WAIT_SWEEP_TRIGGER"
            tier_rev = 1 if (in_wall_flag or near_walls[0][1] <= 0.65 * atr_1h) else 2
            pat_rev = f"HTF Key-Area Reversal Stalker ({wall_z['kind']})"
            notes_rev.append(
                "Potensi keterbalikan tren (Counter-Trend) di tembok mayor: wajib tunggu Sweep likuiditas + close 15M CHOCH/CISD sebelum masuk."
            )
        else:
            continue

        st_rev = make_setup_obj(
            d_rev, "HTF_REVERSAL", "HTF Reversal (Area Penting)", exec_rev, app_lbl_r,
            "SMC_REVERSAL", pat_rev, wall_z["kind"], bool(wall_z.get("fresh")), geom_r, tot_rev, parts_rev, status_rev, confs_rev, notes_rev,
        )
        candidates.append((tier_rev, tot_rev, st_rev))

    if not candidates:
        return None

    # Pilih setup terbaik: Tier lebih kecil (0=ENTRY READY, 1=PRE-TAP LIMIT / ACTIVE REVERSAL / AT-POI, dst.) & Skor tertinggi
    candidates.sort(key=lambda item: (item[0], -item[1]))
    best_setup = candidates[0][2]

    best_d = 1 if best_setup.direction == "LONG" else -1
    enrich_setup_with_zone_radar(
        best_setup,
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
        has_recent_choch=(has_bull_choch if best_d == 1 else has_bear_choch),
        has_recent_cisd=(has_bull_cisd if best_d == 1 else has_bear_cisd),
    )
    populate_forward_playbook(best_setup)
    if isinstance(best_setup.raw_market_data, dict):
        best_setup.raw_market_data["forward_playbook"] = best_setup.forward_playbook
        best_setup.raw_market_data["flip_scenario"] = best_setup.flip_scenario
    return best_setup
