# -*- coding: utf-8 -*-
"""
Unified Live Zone, POI, Pre-Reaction Stalker, & Liquidity Sweep Detector (4H -> 1H -> 15M)
==========================================================================================
Mendeteksi secara presisi posisi harga real-time (termasuk hybrid live forming candle tanpa repaint) terhadap:
1. Demand & Supply Mayor (4H & 1H) beserta persentase kedalaman mitigasi (Zone Depth % & 50% CE).
2. Point of Interest spesifik: Fresh Order Block (OB), Breaker Block, & Unmitigated FVG (4H, 1H, & 15M).
3. Pemisahan tegas antara:
   - FIRST_TOUCH_NOW : Detik pertama candle berjalan menyentuh zona segar (sebelum candle 15M close).
   - PRE_TAP_NEAR / PRE_TAP_STALK : Harga mendekati zona segar yang BELUM bereaksi (siap pasang Limit!).
   - SWEEP_IN_POI / IN_POI_REJECTING / IN_POI_TESTING : Harga sedang bereaksi di dalam zona.
   - POST_BOUNCE_WAIT : Harga SUDAH menyentuh zona & memantul pergi (jangan dikejar!).
4. Dinamika pendekatan harga (Kompresi/Korektif vs Impulsif Liar) & Liquidity Sweep multi-layer.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .indicators import compute_cvd_metrics, session_liquidity_context, trend_short
from .models import NAN, Config, Setup, ZoneRadarHit, clamp, fp

ACTIVE_SETUP_STATUSES = ("ENTRY READY", "PRE-TAP LIMIT", "REVERSAL WATCH", "WATCHLIST")


def get_live_price_snapshot(df15: pd.DataFrame) -> Tuple[float, float, float, Optional[dict]]:
    """
    Mengambil harga live (termasuk candle 15M yang sedang berjalan di ujung real-time) secara AMAN:
    - Hanya aktif jika `live_bar["open_time"] == open_time_terakhir_closed + 900_000`.
    - Pada Walk-Forward Backtester (`df15.iloc[...]`), otomatis memakai 100% closed bar (0 look-ahead leak!).
    Return: (cur_price, recent_low, recent_high, active_live_bar_or_None)
    """
    h = df15["high"].values.astype(float)
    l = df15["low"].values.astype(float)
    c = df15["close"].values.astype(float)
    last_open_ms = int(df15["open_time"].iloc[-1])

    live_bar = df15.attrs.get("live_bar") if hasattr(df15, "attrs") else None
    use_live = (
        isinstance(live_bar, dict)
        and int(live_bar.get("open_time", 0)) == last_open_ms + 900_000
        and float(live_bar.get("close", 0.0)) > 0.0
    )
    if use_live:
        cur_p = float(live_bar["close"])
        l_rec = float(min(np.min(l[-min(2, len(l)):]), float(live_bar["low"])))
        h_rec = float(max(np.max(h[-min(2, len(h)):]), float(live_bar["high"])))
        return cur_p, l_rec, h_rec, live_bar

    cur_p = float(c[-1])
    l_rec = float(np.min(l[-min(2, len(l)):]))
    h_rec = float(np.max(h[-min(2, len(h)):]))
    return cur_p, l_rec, h_rec, None


def analyze_approach_dynamics(
    df15: pd.DataFrame,
    d: int,
    atr_15m: float,
    live_bar: Optional[dict] = None,
) -> Tuple[str, str, float]:
    """
    Mengukur gaya kedatangan harga menuju POI (d=1 turun ke Demand, d=-1 naik ke Supply):
    - KOMPRESI_KOREKTIF : candle pendek/saling tumpang-tindih (korektif sehat -> ideal Limit Pre-Tap).
    - PULLBACK_WAJAR    : koreksi normal -> aman Limit di 50% CE zona.
    - IMPULSIF_LIAR     : terjun bebas / spike vertikal marubozu -> rawan tembus / stop-hunt wick kripto,
                          wajib tunggu Sweep + 15M CISD/PinBar sebelum entry!
    Return: (approach_code, approach_label, score_pts_0_to_14)
    """
    a = max(float(atr_15m), 1e-12)
    win = min(4, len(df15))
    o = list(df15["open"].values[-win:].astype(float))
    h = list(df15["high"].values[-win:].astype(float))
    l = list(df15["low"].values[-win:].astype(float))
    c = list(df15["close"].values[-win:].astype(float))

    if live_bar is not None:
        o.append(float(live_bar["open"]))
        h.append(float(live_bar["high"]))
        l.append(float(live_bar["low"]))
        c.append(float(live_bar["close"]))

    recent_o = o[-3:]
    recent_h = h[-3:]
    recent_l = l[-3:]
    recent_c = c[-3:]

    # Body searah jatuhnya harga menuju zona (bearish body saat turun ke Demand, bullish body saat naik ke Supply)
    if d == 1:
        counter_bodies = [max(0.0, recent_o[i] - recent_c[i]) / a for i in range(len(recent_c))]
    else:
        counter_bodies = [max(0.0, recent_c[i] - recent_o[i]) / a for i in range(len(recent_c))]

    max_cb = max(counter_bodies) if counter_bodies else 0.0
    sum_cb = sum(counter_bodies)
    net_move = abs(recent_c[-1] - recent_o[0]) / a
    total_ranges = sum(max(recent_h[i] - recent_l[i], 1e-12) for i in range(len(recent_c))) / a
    overlap_eff = net_move / max(total_ranges, 1e-12)  # makin kecil = makin banyak tumpang tindih (kompresi)

    if max_cb >= 1.45 or (sum_cb >= 2.75 and overlap_eff >= 0.72):
        return "IMPULSIF_LIAR", "IMPULSIF TAJAM (Rawan Wick — Tunggu Sweep/CISD)", 3.0
    if max_cb <= 0.75 and (sum_cb <= 1.55 or overlap_eff <= 0.48):
        return "KOMPRESI_KOREKTIF", "KOMPRESI / KOREKTIF (Ideal Limit Pre-Tap)", 14.0
    return "PULLBACK_WAJAR", "PULLBACK WAJAR (Limit di 50% CE / Pantau)", 10.0


def _collect_directional_zones(
    map4: Optional[dict],
    map1: Optional[dict],
    map15: Optional[dict],
    cur_p: float,
    l_recent: float,
    h_recent: float,
) -> Tuple[List[dict], List[dict]]:
    """Kumpulkan seluruh zona Demand (dir=1) & Supply (dir=-1) aktif yang belum jebol."""
    bull_zones: List[dict] = []
    bear_zones: List[dict] = []

    for tf_map, weight in ((map4, 3), (map1, 2), (map15, 1)):
        if not tf_map:
            continue
        atr_tf = float(tf_map.get("atr", 0.0) or 0.0)
        for z in tf_map.get("bull_obs", []):
            if cur_p >= z["bot"] - 0.06 * atr_tf:
                bull_zones.append({**z, "weight": weight, "atr_tf": atr_tf, "ztype": "OB"})
        for z in tf_map.get("bear_obs", []):
            if cur_p <= z["top"] + 0.06 * atr_tf:
                bear_zones.append({**z, "weight": weight, "atr_tf": atr_tf, "ztype": "OB"})
        for b in tf_map.get("breakers", []):
            if b["dir"] == 1 and cur_p >= b["bot"] - 0.06 * atr_tf:
                bull_zones.append({**b, "weight": weight, "atr_tf": atr_tf, "ztype": "Breaker"})
            elif b["dir"] == -1 and cur_p <= b["top"] + 0.06 * atr_tf:
                bear_zones.append({**b, "weight": weight, "atr_tf": atr_tf, "ztype": "Breaker"})
        for f in tf_map.get("fvgs", []):
            # Untuk 15M FVG hanya ambil yang relatif baru (tidak kadaluarsa jauh)
            if tf_map.get("tf") == "15M" and len(tf_map.get("fvgs", [])) > 4 and f not in tf_map["fvgs"][-4:]:
                continue
            if f["dir"] == 1 and cur_p >= f["bot"] - 0.05 * atr_tf:
                bull_zones.append({**f, "fresh": True, "touches": 0, "weight": weight, "atr_tf": atr_tf, "ztype": "FVG"})
            elif f["dir"] == -1 and cur_p <= f["top"] + 0.05 * atr_tf:
                bear_zones.append({**f, "fresh": True, "touches": 0, "weight": weight, "atr_tf": atr_tf, "ztype": "FVG"})

    def annotate(zones: List[dict], d: int) -> List[dict]:
        out = []
        for z in zones:
            bot, top = float(z["bot"]), float(z["top"])
            span = max(top - bot, 1e-12)
            atr_tf = max(float(z.get("atr_tf", 0.0)), 1e-12)
            in_body = bot <= cur_p <= top
            if d == 1:
                wick_tap = (not in_body) and (l_recent <= top) and (l_recent >= bot - 0.15 * atr_tf) and (cur_p > top) and ((cur_p - top) / cur_p * 100.0 <= 0.65)
                dist_pct = 0.0 if in_body else (max(0.0, cur_p - top) / cur_p * 100.0 if cur_p > top else (bot - cur_p) / cur_p * 100.0)
                pen_p = min(cur_p, l_recent) if (in_body or wick_tap) else cur_p
                depth_pct = clamp((top - pen_p) / span * 100.0, 0.0, 100.0) if (in_body or wick_tap) else 0.0
                ahead_pre_tap = (cur_p > top) and bool(z.get("fresh", False)) and (not wick_tap)
            else:
                wick_tap = (not in_body) and (h_recent >= bot) and (h_recent <= top + 0.15 * atr_tf) and (cur_p < bot) and ((bot - cur_p) / cur_p * 100.0 <= 0.65)
                dist_pct = 0.0 if in_body else (max(0.0, bot - cur_p) / cur_p * 100.0 if cur_p < bot else (cur_p - top) / cur_p * 100.0)
                pen_p = max(cur_p, h_recent) if (in_body or wick_tap) else cur_p
                depth_pct = clamp((pen_p - bot) / span * 100.0, 0.0, 100.0) if (in_body or wick_tap) else 0.0
                ahead_pre_tap = (cur_p < bot) and bool(z.get("fresh", False)) and (not wick_tap)

            out.append({
                **z,
                "dir": d,
                "in_body": in_body,
                "wick_tap": wick_tap,
                "touched_now": in_body or wick_tap,
                "ahead_pre_tap": ahead_pre_tap,
                "dist_pct": float(dist_pct),
                "depth_pct": float(depth_pct),
                "at_ce_50": (in_body or wick_tap) and (40.0 <= depth_pct <= 68.0),
            })
        # Urutkan: yang sedang disentuh duluan (HTF lebih diutamakan), lalu jarak terdekat
        out.sort(key=lambda x: (0 if x["in_body"] else (1 if x["wick_tap"] else 2), x["dist_pct"], -x["weight"]))
        return out

    return annotate(bull_zones, 1), annotate(bear_zones, -1)


def _detect_multilayer_sweeps(
    df15: pd.DataFrame,
    map1: Optional[dict],
    map15: Optional[dict],
    sess_ctx: dict,
    piv_len_15: int = 5,
    idm_level: Optional[float] = None,
    idm_bull_swept: bool = False,
    idm_bear_swept: bool = False,
    cur_p_override: Optional[float] = None,
    live_bar: Optional[dict] = None,
) -> dict:
    """
    Mendeteksi Liquidity Sweep aktif & kolam likuiditas (EQH/EQL/Swing/Asian/IDM) lintas 1H & 15M:
    - Live Sweep   : terjadi pada live_bar berjalan atau 1..4 bar 15M terakhir (<= 1 jam terakhir)
    - Recent Sweep : terjadi dalam 5..12 bar 15M terakhir (1 - 3 jam terakhir)
    - Magnets      : kolam EQH/EQL/IDM terdekat (<= 1.8%) di depan harga sebagai magnet likuiditas
    """
    n = len(df15)
    h = df15["high"].values.astype(float)
    l = df15["low"].values.astype(float)
    c = df15["close"].values.astype(float)
    cur_p = float(cur_p_override) if cur_p_override is not None else float(c[-1])
    live_l = float(live_bar["low"]) if live_bar is not None else float(l[-1])
    live_h = float(live_bar["high"]) if live_bar is not None else float(h[-1])

    bull_sweeps: List[Tuple[int, str]] = []  # (bars_ago, label) -> SSL / EQL / Asian Low sweep
    bear_sweeps: List[Tuple[int, str]] = []  # (bars_ago, label) -> BSL / EQH / Asian High sweep
    magnets: List[str] = []

    # 1. Cek 1H Swing Low / High & 1H EQL / EQH Sweep dalam live_bar + 6 bar 15M terakhir
    if map1:
        atr1 = max(float(map1.get("atr", 0.0)), 1e-12)
        h1_lows = map1.get("lows", [])[-4:]
        for sw in h1_lows:
            lvl = float(sw["price"])
            if live_bar is not None and live_l < lvl and cur_p >= lvl:
                bull_sweeps.append((0, f"1H SSL Sweep ({fp(lvl)}, live!)"))
                continue
            for k in range(1, min(7, n)):
                idx = n - k
                if l[idx] < lvl and c[idx] >= lvl and cur_p >= lvl - 0.05 * atr1:
                    bull_sweeps.append((k - 1, f"1H SSL Sweep ({fp(lvl)}, {k-1}b lalu)"))
                    break

        h1_highs = map1.get("highs", [])[-4:]
        for sw in h1_highs:
            lvl = float(sw["price"])
            if live_bar is not None and live_h > lvl and cur_p <= lvl:
                bear_sweeps.append((0, f"1H BSL Sweep ({fp(lvl)}, live!)"))
                continue
            for k in range(1, min(7, n)):
                idx = n - k
                if h[idx] > lvl and c[idx] <= lvl and cur_p <= lvl + 0.05 * atr1:
                    bear_sweeps.append((k - 1, f"1H BSL Sweep ({fp(lvl)}, {k-1}b lalu)"))
                    break

        for lvl in map1.get("eql", [])[-3:]:
            if (live_l < lvl or l[-1] < lvl) and cur_p >= lvl:
                bull_sweeps.append((0, f"1H EQL Sweep ({fp(lvl)}, baru saja!)"))
            elif cur_p > lvl and (cur_p - lvl) / cur_p * 100.0 <= 1.80:
                magnets.append(f"EQL 1H @ {fp(lvl)} ({(cur_p - lvl)/cur_p*100:.2f}%)")

        for lvl in map1.get("eqh", [])[-3:]:
            if (live_h > lvl or h[-1] > lvl) and cur_p <= lvl:
                bear_sweeps.append((0, f"1H EQH Sweep ({fp(lvl)}, baru saja!)"))
            elif cur_p < lvl and (lvl - cur_p) / cur_p * 100.0 <= 1.80:
                magnets.append(f"EQH 1H @ {fp(lvl)} ({(lvl - cur_p)/cur_p*100:.2f}%)")

    # 2. Cek 15M Swing SSL / BSL & 15M EQL / EQH Sweep dalam 12 bar 15M terakhir
    if map15:
        highs_15 = map15.get("highs", [])
        lows_15 = map15.get("lows", [])
        if live_bar is not None and lows_15:
            last_l15 = float(lows_15[-1]["price"])
            if live_l < last_l15 and cur_p >= last_l15 and l[-1] >= last_l15:
                bull_sweeps.append((0, f"15M SSL Sweep ({fp(last_l15)}, live!)"))
        if live_bar is not None and highs_15:
            last_h15 = float(highs_15[-1]["price"])
            if live_h > last_h15 and cur_p <= last_h15 and h[-1] <= last_h15:
                bear_sweeps.append((0, f"15M BSL Sweep ({fp(last_h15)}, live!)"))

        for k in range(1, min(13, n)):
            t = n - k
            known_l = [s for s in lows_15 if s["idx"] <= t - piv_len_15]
            if known_l:
                last_l = float(known_l[-1]["price"])
                if l[t] < last_l and c[t] >= last_l and l[t - 1] >= last_l and cur_p >= last_l:
                    bull_sweeps.append((k - 1, f"15M SSL Sweep ({fp(last_l)}, {k-1}b lalu)"))
            known_h = [s for s in highs_15 if s["idx"] <= t - piv_len_15]
            if known_h:
                last_h = float(known_h[-1]["price"])
                if h[t] > last_h and c[t] <= last_h and h[t - 1] <= last_h and cur_p <= last_h:
                    bear_sweeps.append((k - 1, f"15M BSL Sweep ({fp(last_h)}, {k-1}b lalu)"))

        for lvl in map15.get("eql", [])[-3:]:
            if (live_l < lvl or l[-1] < lvl) and cur_p >= lvl:
                bull_sweeps.append((0, f"15M EQL Sweep ({fp(lvl)}, baru saja!)"))
            elif cur_p > lvl and (cur_p - lvl) / cur_p * 100.0 <= 1.50:
                magnets.append(f"EQL 15M @ {fp(lvl)} ({(cur_p - lvl)/cur_p*100:.2f}%)")

        for lvl in map15.get("eqh", [])[-3:]:
            if (live_h > lvl or h[-1] > lvl) and cur_p <= lvl:
                bear_sweeps.append((0, f"15M EQH Sweep ({fp(lvl)}, baru saja!)"))
            elif cur_p < lvl and (lvl - cur_p) / cur_p * 100.0 <= 1.50:
                magnets.append(f"EQH 15M @ {fp(lvl)} ({(lvl - cur_p)/cur_p*100:.2f}%)")

    # 3. Asian Range Sweep (hanya dihitung aktif jika terjadi dalam 8 bar 15M / 2 jam terakhir & masih dekat level sweep)
    a_lo = sess_ctx.get("asian_low", NAN)
    a_hi = sess_ctx.get("asian_high", NAN)
    atr1_ref = float(map1.get("atr", 0.0)) if map1 else float(map15.get("atr", 0.0) * 2.1 if map15 else cur_p * 0.015)
    if (sess_ctx.get("asian_ssl_swept") or (not math.isnan(a_lo) and live_l < a_lo <= cur_p)) and not math.isnan(a_lo) and cur_p >= a_lo:
        bars_touch = ([0] if live_l < a_lo else []) + [k - 1 for k in range(1, min(9, n)) if l[n - k] < a_lo]
        if bars_touch and (cur_p - a_lo) <= 1.5 * max(atr1_ref, 1e-12):
            bull_sweeps.append((min(bars_touch), f"Asian Low Sweep ({fp(a_lo)}, {min(bars_touch)}b lalu)"))
    if (sess_ctx.get("asian_bsl_swept") or (not math.isnan(a_hi) and live_h > a_hi >= cur_p)) and not math.isnan(a_hi) and cur_p <= a_hi:
        bars_touch = ([0] if live_h > a_hi else []) + [k - 1 for k in range(1, min(9, n)) if h[n - k] > a_hi]
        if bars_touch and (a_hi - cur_p) <= 1.5 * max(atr1_ref, 1e-12):
            bear_sweeps.append((min(bars_touch), f"Asian High Sweep ({fp(a_hi)}, {min(bars_touch)}b lalu)"))

    if idm_level is not None and not math.isnan(idm_level):
        if idm_bull_swept and cur_p >= idm_level * 0.995:
            idm_bars = [k - 1 for k in range(1, min(9, n)) if l[n - k] <= idm_level]
            if idm_bars:
                bull_sweeps.append((min(idm_bars), f"IDM Taken ({fp(idm_level)}, {min(idm_bars)}b lalu)"))
        elif idm_bear_swept and cur_p <= idm_level * 1.005:
            idm_bars = [k - 1 for k in range(1, min(9, n)) if h[n - k] >= idm_level]
            if idm_bars:
                bear_sweeps.append((min(idm_bars), f"IDM Taken ({fp(idm_level)}, {min(idm_bars)}b lalu)"))
        elif abs(cur_p - idm_level) / cur_p * 100.0 <= 1.60:
            magnets.append(f"IDM @ {fp(idm_level)} ({abs(cur_p - idm_level)/cur_p*100:.2f}%)")

    bull_sweeps.sort(key=lambda x: x[0])
    bear_sweeps.sort(key=lambda x: x[0])

    # Deduplikasi teks serupa
    def uniq_labels(items: List[Tuple[int, str]]) -> Tuple[List[str], Optional[int]]:
        seen = []
        min_age = items[0][0] if items else None
        for _, txt in items:
            prefix = txt.split("(")[0].strip()
            if not any(s.startswith(prefix) for s in seen):
                seen.append(txt)
        return seen[:3], min_age

    bull_list, bull_min_age = uniq_labels(bull_sweeps)
    bear_list, bear_min_age = uniq_labels(bear_sweeps)

    return {
        "bull_sweeps": bull_list,
        "bull_min_age": bull_min_age,
        "bull_live_sweep": bull_min_age is not None and bull_min_age <= 4,
        "bear_sweeps": bear_list,
        "bear_min_age": bear_min_age,
        "bear_live_sweep": bear_min_age is not None and bear_min_age <= 4,
        "magnets": magnets[:3],
    }


def _evaluate_candle_reaction(
    df15: pd.DataFrame,
    d: int,
    atr_15m: float,
    cvd_ctx: dict,
    has_recent_choch: bool = False,
    has_recent_cisd: bool = False,
    live_bar: Optional[dict] = None,
) -> Tuple[str, str]:
    """
    Mengevaluasi apakah harga di zona sedang memantul (REJECTING) atau masih jatuh tajam (TESTING / Falling Knife).
    Return: (state_code: 'REJECTING' | 'TESTING', detail_str)
    """
    o = df15["open"].values.astype(float)
    h = df15["high"].values.astype(float)
    l = df15["low"].values.astype(float)
    c = df15["close"].values.astype(float)

    if live_bar is not None:
        o_cur, h_cur, l_cur, c_cur = float(live_bar["open"]), float(live_bar["high"]), float(live_bar["low"]), float(live_bar["close"])
        bar_tag = "Live 15M"
    else:
        o_cur, h_cur, l_cur, c_cur = float(o[-1]), float(h[-1]), float(l[-1]), float(c[-1])
        bar_tag = "Close 15M"

    rng = max(h_cur - l_cur, 1e-12)
    body = abs(c_cur - o_cur)
    l_wick = min(o_cur, c_cur) - l_cur
    u_wick = h_cur - max(o_cur, c_cur)

    evidences: List[str] = []
    if d == 1:
        candle_green = c_cur > o_cur
        strong_wick = (l_wick / rng) >= 0.42 or (min(o[-1], c[-1]) - l[-1]) / max(h[-1] - l[-1], 1e-12) >= 0.48
        falling_knife = (c_cur < o_cur) and (body >= 0.60 * max(atr_15m, 1e-12)) and ((l_wick / rng) < 0.22)

        if strong_wick:
            evidences.append(f"Lower Wick {l_wick / rng * 100:.0f}%")
        if candle_green:
            evidences.append(f"{bar_tag} Hijau")
        if has_recent_cisd:
            evidences.append("Bull CISD")
        if has_recent_choch:
            evidences.append("Bull CHOCH")
        if cvd_ctx.get("bull_absorption"):
            evidences.append("CVD Bull Absorption")
        elif cvd_ctx.get("cvd_dir") == 1:
            evidences.append("CVD Naik")

        if falling_knife and not has_recent_cisd:
            return "TESTING", "TESTING (Candle merah impulsif / Falling Knife — tunggu close pantulan)"
        if candle_green or strong_wick or has_recent_cisd or cvd_ctx.get("bull_absorption"):
            return "REJECTING", "REJECTING (" + ", ".join(evidences[:3]) + ")"
        return "TESTING", "TESTING (Menguji Demand — tunggu PinBar / close hijau 15M)"
    else:
        candle_red = c_cur < o_cur
        strong_wick = (u_wick / rng) >= 0.42 or (h[-1] - max(o[-1], c[-1])) / max(h[-1] - l[-1], 1e-12) >= 0.48
        rising_spike = (c_cur > o_cur) and (body >= 0.60 * max(atr_15m, 1e-12)) and ((u_wick / rng) < 0.22)

        if strong_wick:
            evidences.append(f"Upper Wick {u_wick / rng * 100:.0f}%")
        if candle_red:
            evidences.append(f"{bar_tag} Merah")
        if has_recent_cisd:
            evidences.append("Bear CISD")
        if has_recent_choch:
            evidences.append("Bear CHOCH")
        if cvd_ctx.get("bear_absorption"):
            evidences.append("CVD Bear Absorption")
        elif cvd_ctx.get("cvd_dir") == -1:
            evidences.append("CVD Turun")

        if rising_spike and not has_recent_cisd:
            return "TESTING", "TESTING (Candle hijau impulsif — tunggu close penolakan)"
        if candle_red or strong_wick or has_recent_cisd or cvd_ctx.get("bear_absorption"):
            return "REJECTING", "REJECTING (" + ", ".join(evidences[:3]) + ")"
        return "TESTING", "TESTING (Menguji Supply — tunggu PinBar / close merah 15M)"


def _short_zone_tag(z: dict) -> str:
    tf = z.get("tf", "")
    zt = z.get("ztype", "POI")
    d_str = "DEM" if z.get("dir", 1) == 1 else "SUP"
    if zt == "OB":
        return f"{tf} OB"
    if zt == "Breaker":
        return f"{tf} BRK"
    if zt == "FVG":
        return f"{tf} FVG"
    return f"{tf} {d_str}"


def enrich_setup_with_zone_radar(
    setup: Setup,
    df15: pd.DataFrame,
    map4: Optional[dict],
    map1: Optional[dict],
    map15: Optional[dict],
    sess_ctx: dict,
    cvd_ctx: dict,
    cfg: Config,
    idm_level: Optional[float] = None,
    idm_bull_swept: bool = False,
    idm_bear_swept: bool = False,
    has_recent_choch: bool = False,
    has_recent_cisd: bool = False,
) -> None:
    """
    Memperkaya objek Setup (SMC maupun Breakout) dengan status presisi lokasi harga real-time
    terhadap POI, Demand/Supply 4H-1H-15M, pemisahan tegas PRE-TAP vs FIRST-TOUCH vs POST-BOUNCE,
    Liquidity Sweep, & Reaksi Candle.
    """
    h = df15["high"].values.astype(float)
    l = df15["low"].values.astype(float)
    c = df15["close"].values.astype(float)
    cur_p, l_rec, h_rec, live_bar = get_live_price_snapshot(df15)
    setup.market_price = float(cur_p)

    d = 1 if setup.direction == "LONG" else -1
    poi_lo = float(setup.poi_low) if not math.isnan(setup.poi_low) else float(setup.entry)
    poi_hi = float(setup.poi_high) if not math.isnan(setup.poi_high) else float(setup.entry)
    if poi_hi < poi_lo:
        poi_lo, poi_hi = poi_hi, poi_lo
    span_poi = max(poi_hi - poi_lo, 1e-12)
    atr_1h = float(setup.atr_1h) if not math.isnan(setup.atr_1h) and setup.atr_1h > 0 else float(setup.atr) * 2.1

    # Cek riwayat 10 candle closed terakhir: apakah zona POI ini SUDAH sempat disentuh & memantul?
    look_closed = min(10, len(l))
    if d == 1:
        tapped_in_closed = bool(np.min(l[-look_closed:]) <= poi_hi)
        live_pierced = live_bar is not None and float(live_bar["low"]) <= poi_hi and float(live_bar["low"]) >= poi_lo - 0.25 * atr_1h
    else:
        tapped_in_closed = bool(np.max(h[-look_closed:]) >= poi_lo)
        live_pierced = live_bar is not None and float(live_bar["high"]) >= poi_lo and float(live_bar["high"]) <= poi_hi + 0.25 * atr_1h

    first_touch_live = bool(live_pierced and not tapped_in_closed)

    in_poi_body = poi_lo <= cur_p <= poi_hi
    if d == 1:
        wick_tapped_poi = (not in_poi_body) and (l_rec <= poi_hi) and (l_rec >= poi_lo - 0.20 * atr_1h) and (cur_p > poi_hi) and ((cur_p - poi_hi) / cur_p * 100.0 <= 0.70)
        if in_poi_body:
            dist_pct = 0.0
        elif cur_p > poi_hi:
            dist_pct = (cur_p - poi_hi) / cur_p * 100.0
        else:
            dist_pct = (poi_lo - cur_p) / cur_p * 100.0
        pen_p = min(cur_p, l_rec) if (in_poi_body or wick_tapped_poi) else cur_p
        depth_pct = clamp((poi_hi - pen_p) / span_poi * 100.0, 0.0, 100.0) if (in_poi_body or wick_tapped_poi) else 0.0
    else:
        wick_tapped_poi = (not in_poi_body) and (h_rec >= poi_lo) and (h_rec <= poi_hi + 0.20 * atr_1h) and (cur_p < poi_lo) and ((poi_lo - cur_p) / cur_p * 100.0 <= 0.70)
        if in_poi_body:
            dist_pct = 0.0
        elif cur_p < poi_lo:
            dist_pct = (poi_lo - cur_p) / cur_p * 100.0
        else:
            dist_pct = (cur_p - poi_hi) / cur_p * 100.0
        pen_p = max(cur_p, h_rec) if (in_poi_body or wick_tapped_poi) else cur_p
        depth_pct = clamp((pen_p - poi_lo) / span_poi * 100.0, 0.0, 100.0) if (in_poi_body or wick_tapped_poi) else 0.0

    bull_zones, bear_zones = _collect_directional_zones(map4, map1, map15, cur_p, l_rec, h_rec)
    aligned_zones = bull_zones if d == 1 else bear_zones
    opposing_zones = bear_zones if d == 1 else bull_zones

    sweeps = _detect_multilayer_sweeps(
        df15, map1, map15, sess_ctx, cfg.piv_len, idm_level, idm_bull_swept, idm_bear_swept,
        cur_p_override=cur_p, live_bar=live_bar,
    )
    active_sweeps = sweeps["bull_sweeps"] if d == 1 else sweeps["bear_sweeps"]
    has_live_sweep = sweeps["bull_live_sweep"] if d == 1 else sweeps["bear_live_sweep"]

    touched_aligned = [z for z in aligned_zones if z["touched_now"]]
    near_aligned = [z for z in aligned_zones if not z["touched_now"] and z["dist_pct"] <= max(cfg.near_poi_pct, 1.20)]
    touched_opposing_htf = [z for z in opposing_zones if z["touched_now"] and z["weight"] >= 2]

    # Cek apakah zona utama masih Fresh (Unmitigated / sentuhan pertama)
    is_fresh = (
        bool(setup.zone_fresh)
        or first_touch_live
        or any(bool(z.get("fresh")) for z in touched_aligned)
        or (not tapped_in_closed and any(bool(z.get("fresh")) and z["dist_pct"] <= cfg.max_stalk_pct for z in aligned_zones[:3]))
    )

    # Ringkasan seluruh zona aktif 4H/1H/15M yang sedang diinjak atau sedang didekati (Pre-Tap)
    z_parts: List[str] = []
    if first_touch_live:
        z_parts.append(f"SENTUHAN PERTAMA LIVE [{setup.entry_zone}] (Depth {depth_pct:.0f}%, Unmitigated)")
    for z in touched_aligned[:3]:
        mode_lbl = "IN" if z["in_body"] else "WICK-TAP"
        fr_lbl = ", Fresh" if z.get("fresh") else ""
        ce_lbl = ", ~50% CE" if z.get("at_ce_50") else f", Depth {z['depth_pct']:.0f}%"
        z_parts.append(f"{mode_lbl} {z['kind']} [{fp(z['bot'])}-{fp(z['top'])}] ({ce_lbl.lstrip(', ')}{fr_lbl})")

    if not z_parts and (in_poi_body or wick_tapped_poi):
        mode_lbl = "IN POI" if in_poi_body else "WICK-TAP POI"
        z_parts.append(f"{mode_lbl} [{setup.entry_zone}] (Depth {depth_pct:.0f}%)")

    for z in near_aligned[:2]:
        tag_mode = "PRE-TAP SEGAR" if z.get("ahead_pre_tap") else "DEKAT"
        z_parts.append(f"{tag_mode} {z['kind']} [{fp(z['bot'])}-{fp(z['top'])}] (Jarak {z['dist_pct']:.2f}%)")

    if touched_opposing_htf and setup.archetype != "HTF_REVERSAL":
        oz = touched_opposing_htf[0]
        z_parts.append(f"PERINGATAN: Sedang menyentuh lawan {oz['kind']} [{fp(oz['bot'])}-{fp(oz['top'])}]")

    if not z_parts:
        if not tapped_in_closed:
            z_parts.append(f"Mendekati Fresh POI [{setup.entry_zone}] (Jarak {dist_pct:.2f}% — Belum Bereaksi)")
        else:
            z_parts.append(f"Sudah memantul dari POI [{setup.entry_zone}] (Jarak +{dist_pct:.2f}% — Tunggu Retest)")

    reac_code, reac_detail = _evaluate_candle_reaction(
        df15, d, float(setup.atr), cvd_ctx,
        has_recent_choch=has_recent_choch, has_recent_cisd=has_recent_cisd, live_bar=live_bar,
    )
    app_code, app_lbl, _ = analyze_approach_dynamics(df15, d, float(setup.atr), live_bar=live_bar)
    if not setup.approach_style:
        setup.approach_style = app_lbl if not tapped_in_closed else "SUDAH MEMANTUL DARI POI"

    body_in_htf_or_ltf = [z for z in touched_aligned if z["in_body"]]
    eff_depth = depth_pct if (in_poi_body or wick_tapped_poi) else (touched_aligned[0]["depth_pct"] if touched_aligned else NAN)
    top_tag = _short_zone_tag(body_in_htf_or_ltf[0]) if body_in_htf_or_ltf else (
        _short_zone_tag(touched_aligned[0]) if touched_aligned else (
            _short_zone_tag(near_aligned[0]) if near_aligned else ("DEMAND" if d == 1 else "SUPPLY")
        )
    )

    # Penentuan Zone State & Compact Badge (membedakan tegas FIRST TOUCH, PRE-TAP, IN-ZONE, vs POST-BOUNCE!)
    r_short = "Rej" if reac_code == "REJECTING" else "Test"
    is_pre_tap_zone = (not tapped_in_closed) or setup.archetype == "PRE_REACTION"

    if first_touch_live and (in_poi_body or wick_tapped_poi):
        zone_state = "FIRST_TOUCH_NOW"
        zone_badge = f"🔔 FIRST TOUCH {top_tag} ({depth_pct:.0f}%|{r_short})"
        reac_detail = f"SENTUHAN DETIK PERTAMA di zona segar! {reac_detail}"
    elif in_poi_body and has_live_sweep:
        zone_state = "SWEEP_IN_POI"
        zone_badge = f"🔥 SWEEP+IN {top_tag} ({depth_pct:.0f}%|{r_short})"
    elif in_poi_body and reac_code == "REJECTING":
        zone_state = "IN_POI_REJECTING"
        fr_s = "Fresh|" if is_fresh else ""
        zone_badge = f"🎯 IN {top_tag} ({fr_s}{depth_pct:.0f}%|Rej)"
    elif in_poi_body:
        zone_state = "IN_POI_TESTING"
        zone_badge = f"⚠️ IN {top_tag} ({depth_pct:.0f}%|Tunggu)"
    elif wick_tapped_poi and dist_pct <= 0.55:
        zone_state = "SWEEP_IN_POI" if has_live_sweep else ("IN_POI_REJECTING" if reac_code == "REJECTING" else "IN_POI_TESTING")
        icon = "🔥 SWEEP+TAP" if has_live_sweep else "⚡ WICK-TAP"
        zone_badge = f"{icon} {top_tag} ({dist_pct:.2f}%|{r_short})"
    elif body_in_htf_or_ltf and dist_pct <= 1.20 and not (setup.archetype == "PRE_REACTION" and dist_pct > 0.35):
        zone_state = "SWEEP_IN_POI" if has_live_sweep else ("IN_POI_REJECTING" if reac_code == "REJECTING" else "IN_POI_TESTING")
        hz_d = body_in_htf_or_ltf[0]["depth_pct"]
        icon = "🔥 SWEEP+IN" if has_live_sweep else "🎯 IN"
        zone_badge = f"{icon} {top_tag} ({hz_d:.0f}%|{r_short})"
    elif has_live_sweep and dist_pct <= 0.90:
        zone_state = "LIQ_SWEPT"
        sw_tag = "SSL" if d == 1 else "BSL"
        zone_badge = f"💧 {sw_tag} SWEPT ({dist_pct:.2f}%)"
    elif is_pre_tap_zone and dist_pct <= cfg.near_poi_pct:
        zone_state = "PRE_TAP_NEAR"
        zone_badge = f"🎯 PRE-TAP {top_tag} ({dist_pct:.2f}%)"
        reac_detail = f"SEBELUM BEREAKSI (Jarak {dist_pct:.2f}% ke bibir {top_tag} segar | Gerak: {app_lbl})"
    elif is_pre_tap_zone and dist_pct <= cfg.max_stalk_pct:
        zone_state = "PRE_TAP_STALK"
        zone_badge = f"🔭 STALK {top_tag} ({dist_pct:.2f}%)"
        reac_detail = f"MENDEKATI ZONA SEGAR (Jarak {dist_pct:.2f}% sebelum sentuhan | Gerak: {app_lbl})"
    elif dist_pct <= cfg.near_poi_pct:
        zone_state = "APPROACHING_POI"
        zone_badge = f"⚡ NEAR {top_tag} ({dist_pct:.2f}%)"
        reac_detail = f"MENDEKATI RETEST ZONA (Jarak {dist_pct:.2f}% — pantau reaksi)"
    else:
        zone_state = "POST_BOUNCE_WAIT" if tapped_in_closed else "WAITING_PULLBACK"
        zone_badge = f"🏃 POST-BOUNCE (+{dist_pct:.2f}%)" if tapped_in_closed else f"⏳ WAIT ({dist_pct:.2f}%)"
        reac_detail = (
            f"SUDAH BEREAKSI & MEMANTUL (+{dist_pct:.2f}% dari POI — tunggu Limit Retest, jangan kejar)"
            if tapped_in_closed
            else f"BELUM DI ZONA (Jarak {dist_pct:.2f}% dari POI)"
        )

    setup.zone_state = zone_state
    setup.zone_badge = zone_badge
    setup.dist_to_poi_pct = float(dist_pct)
    setup.zone_depth_pct = float(eff_depth) if not math.isnan(eff_depth) else NAN
    setup.zone_fresh = bool(is_fresh)
    setup.sweep_status = " | ".join(active_sweeps) if active_sweeps else "Belum ada sweep baru"
    setup.zone_reaction = reac_detail
    setup.active_zones_summary = " ; ".join(z_parts)
    setup.eq_magnet_info = " | ".join(sweeps["magnets"]) if sweeps["magnets"] else "-"

    if isinstance(setup.raw_market_data, dict):
        setup.raw_market_data.update({
            "current_price": cur_p,
            "zone_state": setup.zone_state,
            "zone_badge": setup.zone_badge,
            "dist_to_poi_pct": round(setup.dist_to_poi_pct, 3),
            "zone_depth_pct": round(setup.zone_depth_pct, 1) if not math.isnan(setup.zone_depth_pct) else None,
            "zone_fresh": setup.zone_fresh,
            "sweep_status": setup.sweep_status,
            "zone_reaction": setup.zone_reaction,
            "active_zones_summary": setup.active_zones_summary,
            "eq_magnet_info": setup.eq_magnet_info,
            "approach_code": app_code,
            "approach_style": setup.approach_style,
        })


def scan_live_zone_radar(
    symbol: str,
    df15: pd.DataFrame,
    ctx4: dict,
    ctx1: dict,
    map4: Optional[dict],
    map1: Optional[dict],
    map15: Optional[dict],
    cfg: Config,
    existing_setup: Optional[Setup] = None,
) -> Optional[ZoneRadarHit]:
    """
    Memindai apakah sebuah simbol SAAT INI:
    - Baru menyentuh zona segar di detik pertama (`FIRST_TOUCH_NOW`),
    - Sedang menginjak Demand/Supply 4H-1H-15M / Liquidity Sweep, ATAU
    - Sedang mendekati (`<= 1.0%`) zona Fresh 4H/1H OB/Breaker yang BELUM tersentuh (`PRE_TAP_HTF`)!
    """
    if df15 is None or len(df15) < 40 or map15 is None:
        return None

    h = df15["high"].values.astype(float)
    l = df15["low"].values.astype(float)
    cur_p, l_rec, h_rec, live_bar = get_live_price_snapshot(df15)
    atr_15m = float(map15.get("atr", 0.0) or 0.0)

    sess_ctx = session_liquidity_context(df15)
    cvd_ctx = compute_cvd_metrics(df15)
    bull_zones, bear_zones = _collect_directional_zones(map4, map1, map15, cur_p, l_rec, h_rec)
    sweeps = _detect_multilayer_sweeps(df15, map1, map15, sess_ctx, cfg.piv_len, cur_p_override=cur_p, live_bar=live_bar)

    # Kandidat Bullish Live / Pre-Tap Event (Demand / Bull OB / SSL-EQL Sweep)
    bull_touch = [z for z in bull_zones if z["touched_now"]]
    bull_pretap_htf = [
        z for z in bull_zones
        if not z["touched_now"] and z["weight"] >= 2 and z.get("ahead_pre_tap") and z["ztype"] in ("OB", "Breaker") and z["dist_pct"] <= 1.05
    ]
    bull_live_sw = sweeps["bull_live_sweep"]

    # Kandidat Bearish Live / Pre-Tap Event (Supply / Bear OB / BSL-EQH Sweep)
    bear_touch = [z for z in bear_zones if z["touched_now"]]
    bear_pretap_htf = [
        z for z in bear_zones
        if not z["touched_now"] and z["weight"] >= 2 and z.get("ahead_pre_tap") and z["ztype"] in ("OB", "Breaker") and z["dist_pct"] <= 1.05
    ]
    bear_live_sw = sweeps["bear_live_sweep"]

    def side_strength(touches: List[dict], pretap_htf: List[dict], live_sw: bool) -> float:
        pts = 0.0
        if touches:
            best_w = max(z["weight"] for z in touches)
            pts += 30.0 + best_w * 12.0 + (6.0 if any(z.get("fresh") for z in touches) else 0.0)
        elif pretap_htf:
            pts += 24.0 + pretap_htf[0]["weight"] * 5.0
        if live_sw:
            pts += 28.0
        return pts

    b_pts = side_strength(bull_touch, bull_pretap_htf, bull_live_sw)
    s_pts = side_strength(bear_touch, bear_pretap_htf, bear_live_sw)

    if existing_setup and existing_setup.status in ACTIVE_SETUP_STATUSES:
        if existing_setup.zone_state in (
            "FIRST_TOUCH_NOW", "SWEEP_IN_POI", "IN_POI_REJECTING", "IN_POI_TESTING",
            "LIQ_SWEPT", "PRE_TAP_NEAR", "PRE_TAP_STALK", "APPROACHING_POI",
        ):
            d = 1 if existing_setup.direction == "LONG" else -1
        elif max(b_pts, s_pts) < 25.0:
            return None
        else:
            d = 1 if b_pts >= s_pts else -1
    else:
        if max(b_pts, s_pts) < 28.0:
            return None
        d = 1 if b_pts >= s_pts else -1

    touches = bull_touch if d == 1 else bear_touch
    pretap_htf = bull_pretap_htf if d == 1 else bear_pretap_htf
    sw_list = sweeps["bull_sweeps"] if d == 1 else sweeps["bear_sweeps"]
    live_sw = bull_live_sw if d == 1 else bear_live_sw

    if not touches and not live_sw and not pretap_htf:
        return None
    if touches and max(z["weight"] for z in touches) == 1 and not live_sw and not existing_setup:
        if not any(z.get("fresh") and z.get("ztype") in ("OB", "Breaker") for z in touches):
            return None

    reac_code, reac_detail = _evaluate_candle_reaction(df15, d, atr_15m, cvd_ctx, live_bar=live_bar)
    _, app_lbl, _ = analyze_approach_dynamics(df15, d, atr_15m, live_bar=live_bar)

    primary = touches[0] if touches else (pretap_htf[0] if pretap_htf else (bull_zones[0] if (d == 1 and bull_zones) else (bear_zones[0] if bear_zones else None)))
    if primary:
        p_zone_txt = f"{primary['kind']} [{fp(primary['bot'])}-{fp(primary['top'])}]"
        z_lo, z_hi = float(primary["bot"]), float(primary["top"])
        dist_p = float(primary["dist_pct"])
        depth_p = float(primary["depth_pct"]) if primary["touched_now"] else NAN
        z_fresh = bool(primary.get("fresh"))
    elif existing_setup:
        p_zone_txt = f"POI [{existing_setup.entry_zone}]"
        z_lo, z_hi = float(existing_setup.poi_low), float(existing_setup.poi_high)
        dist_p = float(existing_setup.dist_to_poi_pct)
        depth_p = float(existing_setup.zone_depth_pct)
        z_fresh = bool(existing_setup.zone_fresh)
    else:
        p_zone_txt = "Liquidity Sweep Area"
        z_lo = z_hi = cur_p
        dist_p = 0.0
        depth_p = NAN
        z_fresh = False

    # Deteksi apakah sentuhan di primary adalah First Touch Live (candle closed belum pernah menyentuh)
    first_touch_radar = False
    if touches and live_bar is not None and z_fresh:
        if d == 1 and float(np.min(l[-min(8, len(l)):])) > z_hi and float(live_bar["low"]) <= z_hi:
            first_touch_radar = True
        elif d == -1 and float(np.max(h[-min(8, len(h)):])) < z_lo and float(live_bar["high"]) >= z_lo:
            first_touch_radar = True

    all_t_txt = " + ".join(
        f"{z['kind']}{' (Fresh)' if z.get('fresh') else ''}" for z in touches[:3]
    ) if touches else p_zone_txt

    has_in_body = any(z["in_body"] for z in touches)
    loc_word = "IN" if has_in_body else "TAP"

    if first_touch_radar:
        r_state = "FIRST_TOUCH_NOW"
        prio = 1
        badge = f"🔔 FIRST TOUCH {_short_zone_tag(touches[0])}"
        reac_detail = f"BARU SENTUH ZONA SEGAR! {reac_detail}"
    elif touches and live_sw:
        r_state = "SWEEP_IN_ZONE"
        prio = 1 if reac_code == "REJECTING" else 2
        badge = f"🔥 SWEEP + {loc_word} {_short_zone_tag(touches[0])}"
    elif touches and any(z["weight"] >= 2 for z in touches):
        r_state = "IN_HTF_ZONE"
        htf_z = [z for z in touches if z["weight"] >= 2][0]
        lw_h = "IN" if htf_z["in_body"] else "TAP"
        prio = 2 if (reac_code == "REJECTING" and htf_z["in_body"]) else 3
        badge = f"🎯 {lw_h} {_short_zone_tag(htf_z)} ({'Rej' if reac_code=='REJECTING' else 'Test'})"
    elif pretap_htf:
        r_state = "PRE_TAP_HTF"
        prio = 2 if dist_p <= 0.50 else 3
        badge = f"🎯 PRE-TAP {_short_zone_tag(pretap_htf[0])} ({dist_p:.2f}%)"
        reac_detail = f"SEBELUM BEREAKSI ({app_lbl})"
    elif live_sw:
        r_state = "LIVE_LIQ_SWEEP"
        prio = 3
        badge = f"💧 LIVE {'SSL' if d==1 else 'BSL'} SWEEP"
    else:
        r_state = "IN_LTF_ZONE"
        prio = 4
        badge = f"🎯 {loc_word} {_short_zone_tag(touches[0])} ({'Rej' if reac_code=='REJECTING' else 'Test'})"

    s_status = existing_setup.status if existing_setup else "-"
    s_score = float(existing_setup.score) if existing_setup else 0.0

    return ZoneRadarHit(
        symbol=symbol,
        market_price=cur_p,
        direction="LONG" if d == 1 else "SHORT",
        radar_state=r_state,
        radar_badge=badge,
        primary_zone=p_zone_txt,
        zone_low=z_lo,
        zone_high=z_hi,
        dist_pct=dist_p,
        depth_pct=depth_p,
        zone_fresh=z_fresh,
        all_zones_touched=all_t_txt,
        sweep_info=" | ".join(sw_list) if sw_list else "-",
        magnet_info=" | ".join(sweeps["magnets"]) if sweeps["magnets"] else "-",
        reaction_info=reac_detail,
        cvd_state=cvd_ctx.get("cvd_state", ""),
        trend_4h_short=trend_short(ctx4),
        struct_1h_short=trend_short(ctx1),
        setup_status=s_status,
        setup_score=s_score,
        priority=prio,
    )
