# -*- coding: utf-8 -*-
"""
Unified Live Zone, POI, & Liquidity Sweep Detector (4H -> 1H -> 15M)
====================================================================
Mendeteksi secara presisi posisi harga real-time (close & tusukan wick terbaru) terhadap:
1. Demand & Supply Mayor (4H & 1H) beserta persentase kedalaman mitigasi (Zone Depth % & 50% CE).
2. Point of Interest spesifik: Fresh Order Block (OB), Breaker Block, & Unmitigated FVG (1H & 15M).
3. Liquidity Sweep aktif: 1H/15M BSL & SSL Sweep, EQH/EQL Stop-Hunt, IDM Taken, & Asian High/Low Sweep.
4. Reaksi Candle & Orderflow di dalam zona (REJECTING vs TESTING / Anti-Falling-Knife).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .indicators import compute_cvd_metrics, session_liquidity_context, trend_short
from .models import NAN, Config, Setup, ZoneRadarHit, clamp, fp


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
                bull_zones.append({**f, "fresh": True, "weight": weight, "atr_tf": atr_tf, "ztype": "FVG"})
            elif f["dir"] == -1 and cur_p <= f["top"] + 0.05 * atr_tf:
                bear_zones.append({**f, "fresh": True, "weight": weight, "atr_tf": atr_tf, "ztype": "FVG"})

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
            else:
                wick_tap = (not in_body) and (h_recent >= bot) and (h_recent <= top + 0.15 * atr_tf) and (cur_p < bot) and ((bot - cur_p) / cur_p * 100.0 <= 0.65)
                dist_pct = 0.0 if in_body else (max(0.0, bot - cur_p) / cur_p * 100.0 if cur_p < bot else (cur_p - top) / cur_p * 100.0)
                pen_p = max(cur_p, h_recent) if (in_body or wick_tap) else cur_p
                depth_pct = clamp((pen_p - bot) / span * 100.0, 0.0, 100.0) if (in_body or wick_tap) else 0.0

            out.append({
                **z,
                "dir": d,
                "in_body": in_body,
                "wick_tap": wick_tap,
                "touched_now": in_body or wick_tap,
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
) -> dict:
    """
    Mendeteksi Liquidity Sweep aktif & kolam likuiditas (EQH/EQL/Swing/Asian/IDM) lintas 1H & 15M:
    - Live Sweep   : terjadi dalam 1..4 bar 15M terakhir (<= 1 jam terakhir)
    - Recent Sweep : terjadi dalam 5..12 bar 15M terakhir (1 - 3 jam terakhir)
    """
    n = len(df15)
    h = df15["high"].values.astype(float)
    l = df15["low"].values.astype(float)
    c = df15["close"].values.astype(float)
    cur_p = float(c[-1])

    bull_sweeps: List[Tuple[int, str]] = []  # (bars_ago, label) -> SSL / EQL / Asian Low sweep
    bear_sweeps: List[Tuple[int, str]] = []  # (bars_ago, label) -> BSL / EQH / Asian High sweep
    magnets: List[str] = []

    # 1. Cek 1H Swing Low / High & 1H EQL / EQH Sweep dalam 6 bar 15M terakhir
    if map1:
        atr1 = max(float(map1.get("atr", 0.0)), 1e-12)
        h1_lows = map1.get("lows", [])[-4:]
        for sw in h1_lows:
            lvl = float(sw["price"])
            for k in range(1, min(7, n)):
                idx = n - k
                if l[idx] < lvl and c[idx] >= lvl and cur_p >= lvl - 0.05 * atr1:
                    bull_sweeps.append((k - 1, f"1H SSL Sweep ({fp(lvl)}, {k-1}b lalu)"))
                    break

        h1_highs = map1.get("highs", [])[-4:]
        for sw in h1_highs:
            lvl = float(sw["price"])
            for k in range(1, min(7, n)):
                idx = n - k
                if h[idx] > lvl and c[idx] <= lvl and cur_p <= lvl + 0.05 * atr1:
                    bear_sweeps.append((k - 1, f"1H BSL Sweep ({fp(lvl)}, {k-1}b lalu)"))
                    break

        for lvl in map1.get("eql", [])[-3:]:
            if l[-1] < lvl and cur_p >= lvl:
                bull_sweeps.append((0, f"1H EQL Sweep ({fp(lvl)}, baru saja!)"))
            elif cur_p > lvl and (cur_p - lvl) / cur_p * 100.0 <= 0.65:
                magnets.append(f"EQL 1H @ {fp(lvl)} ({(cur_p - lvl)/cur_p*100:.2f}%)")

        for lvl in map1.get("eqh", [])[-3:]:
            if h[-1] > lvl and cur_p <= lvl:
                bear_sweeps.append((0, f"1H EQH Sweep ({fp(lvl)}, baru saja!)"))
            elif cur_p < lvl and (lvl - cur_p) / cur_p * 100.0 <= 0.65:
                magnets.append(f"EQH 1H @ {fp(lvl)} ({(lvl - cur_p)/cur_p*100:.2f}%)")

    # 2. Cek 15M Swing SSL / BSL & 15M EQL / EQH Sweep dalam 12 bar 15M terakhir
    if map15:
        highs_15 = map15.get("highs", [])
        lows_15 = map15.get("lows", [])
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
            if l[-1] < lvl and cur_p >= lvl:
                bull_sweeps.append((0, f"15M EQL Sweep ({fp(lvl)}, baru saja!)"))
            elif cur_p > lvl and (cur_p - lvl) / cur_p * 100.0 <= 0.45:
                magnets.append(f"EQL 15M @ {fp(lvl)} ({(cur_p - lvl)/cur_p*100:.2f}%)")

        for lvl in map15.get("eqh", [])[-3:]:
            if h[-1] > lvl and cur_p <= lvl:
                bear_sweeps.append((0, f"15M EQH Sweep ({fp(lvl)}, baru saja!)"))
            elif cur_p < lvl and (lvl - cur_p) / cur_p * 100.0 <= 0.45:
                magnets.append(f"EQH 15M @ {fp(lvl)} ({(lvl - cur_p)/cur_p*100:.2f}%)")

    # 3. Asian Range Sweep (hanya dihitung aktif jika terjadi dalam 8 bar 15M / 2 jam terakhir & masih dekat level sweep)
    a_lo = sess_ctx.get("asian_low", NAN)
    a_hi = sess_ctx.get("asian_high", NAN)
    atr1_ref = float(map1.get("atr", 0.0)) if map1 else float(map15.get("atr", 0.0) * 2.1 if map15 else cur_p * 0.015)
    if sess_ctx.get("asian_ssl_swept") and not math.isnan(a_lo) and cur_p >= a_lo:
        bars_touch = [k - 1 for k in range(1, min(9, n)) if l[n - k] < a_lo]
        if bars_touch and (cur_p - a_lo) <= 1.5 * max(atr1_ref, 1e-12):
            bull_sweeps.append((min(bars_touch), f"Asian Low Sweep ({fp(a_lo)}, {min(bars_touch)}b lalu)"))
    if sess_ctx.get("asian_bsl_swept") and not math.isnan(a_hi) and cur_p <= a_hi:
        bars_touch = [k - 1 for k in range(1, min(9, n)) if h[n - k] > a_hi]
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
        elif abs(cur_p - idm_level) / cur_p * 100.0 <= 0.50:
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
) -> Tuple[str, str]:
    """
    Mengevaluasi apakah harga di zona sedang memantul (REJECTING) atau masih jatuh tajam (TESTING / Falling Knife).
    Return: (state_code: 'REJECTING' | 'TESTING', detail_str)
    """
    o = df15["open"].values.astype(float)
    h = df15["high"].values.astype(float)
    l = df15["low"].values.astype(float)
    c = df15["close"].values.astype(float)

    rng = max(h[-1] - l[-1], 1e-12)
    body = abs(c[-1] - o[-1])
    l_wick = min(o[-1], c[-1]) - l[-1]
    u_wick = h[-1] - max(o[-1], c[-1])

    evidences: List[str] = []
    if d == 1:
        candle_green = c[-1] > o[-1]
        strong_wick = (l_wick / rng) >= 0.42 or (len(c) >= 2 and (min(o[-2], c[-2]) - l[-2]) / max(h[-2] - l[-2], 1e-12) >= 0.48)
        falling_knife = (c[-1] < o[-1]) and (body >= 0.60 * max(atr_15m, 1e-12)) and ((l_wick / rng) < 0.22)

        if strong_wick:
            evidences.append(f"Lower Wick {l_wick / rng * 100:.0f}%")
        if candle_green:
            evidences.append("Close 15M Hijau")
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
        candle_red = c[-1] < o[-1]
        strong_wick = (u_wick / rng) >= 0.42 or (len(c) >= 2 and (h[-2] - max(o[-2], c[-2])) / max(h[-2] - l[-2], 1e-12) >= 0.48)
        rising_spike = (c[-1] > o[-1]) and (body >= 0.60 * max(atr_15m, 1e-12)) and ((u_wick / rng) < 0.22)

        if strong_wick:
            evidences.append(f"Upper Wick {u_wick / rng * 100:.0f}%")
        if candle_red:
            evidences.append("Close 15M Merah")
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
    terhadap POI, Demand/Supply 4H-1H-15M, Liquidity Sweep, & Reaksi Candle.
    """
    h = df15["high"].values.astype(float)
    l = df15["low"].values.astype(float)
    c = df15["close"].values.astype(float)
    cur_p = float(c[-1])
    l_rec = float(np.min(l[-min(2, len(l)):]))
    h_rec = float(np.max(h[-min(2, len(h)):]))

    d = 1 if setup.direction == "LONG" else -1
    poi_lo = float(setup.poi_low) if not math.isnan(setup.poi_low) else float(setup.entry)
    poi_hi = float(setup.poi_high) if not math.isnan(setup.poi_high) else float(setup.entry)
    if poi_hi < poi_lo:
        poi_lo, poi_hi = poi_hi, poi_lo
    span_poi = max(poi_hi - poi_lo, 1e-12)
    atr_1h = float(setup.atr_1h) if not math.isnan(setup.atr_1h) and setup.atr_1h > 0 else float(setup.atr) * 2.1

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
        df15, map1, map15, sess_ctx, cfg.piv_len, idm_level, idm_bull_swept, idm_bear_swept
    )
    active_sweeps = sweeps["bull_sweeps"] if d == 1 else sweeps["bear_sweeps"]
    has_live_sweep = sweeps["bull_live_sweep"] if d == 1 else sweeps["bear_live_sweep"]

    touched_aligned = [z for z in aligned_zones if z["touched_now"]]
    near_aligned = [z for z in aligned_zones if not z["touched_now"] and z["dist_pct"] <= cfg.near_poi_pct]
    touched_opposing_htf = [z for z in opposing_zones if z["touched_now"] and z["weight"] >= 2]

    # Cek apakah salah satu zona aktif yang sedang diuji bersifat Fresh (Unmitigated)
    is_fresh = any(bool(z.get("fresh")) for z in touched_aligned) or any(
        bool(z.get("fresh")) and z["dist_pct"] <= 0.8 for z in aligned_zones[:2]
    )

    # Ringkasan seluruh zona aktif 4H/1H/15M yang sedang diinjak atau sangat dekat
    z_parts: List[str] = []
    for z in touched_aligned[:3]:
        mode_lbl = "IN" if z["in_body"] else "WICK-TAP"
        fr_lbl = ", Fresh" if z.get("fresh") else ""
        ce_lbl = ", ~50% CE" if z.get("at_ce_50") else f", Depth {z['depth_pct']:.0f}%"
        z_parts.append(f"{mode_lbl} {z['kind']} [{fp(z['bot'])}-{fp(z['top'])}] ({ce_lbl.lstrip(', ')}{fr_lbl})")

    if not z_parts and (in_poi_body or wick_tapped_poi):
        mode_lbl = "IN POI" if in_poi_body else "WICK-TAP POI"
        z_parts.append(f"{mode_lbl} [{setup.entry_zone}] (Depth {depth_pct:.0f}%)")

    for z in near_aligned[:2]:
        z_parts.append(f"DEKAT {z['kind']} [{fp(z['bot'])}-{fp(z['top'])}] (Jarak {z['dist_pct']:.2f}%)")

    if touched_opposing_htf:
        oz = touched_opposing_htf[0]
        z_parts.append(f"PERINGATAN: Sedang menyentuh lawan {oz['kind']} [{fp(oz['bot'])}-{fp(oz['top'])}]")

    if not z_parts:
        z_parts.append(f"Menunggu pullback ke POI [{setup.entry_zone}] (Jarak {dist_pct:.2f}%)")

    reac_code, reac_detail = _evaluate_candle_reaction(
        df15, d, float(setup.atr), cvd_ctx, has_recent_choch=has_recent_choch, has_recent_cisd=has_recent_cisd
    )

    body_in_htf_or_ltf = [z for z in touched_aligned if z["in_body"]]
    eff_depth = depth_pct if (in_poi_body or wick_tapped_poi) else (touched_aligned[0]["depth_pct"] if touched_aligned else NAN)
    top_tag = _short_zone_tag(body_in_htf_or_ltf[0]) if body_in_htf_or_ltf else (
        _short_zone_tag(touched_aligned[0]) if touched_aligned else ("DEMAND" if d == 1 else "SUPPLY")
    )

    # Penentuan Zone State & Compact Badge yang jujur terhadap posisi harga vs POI
    r_short = "Rej" if reac_code == "REJECTING" else "Test"
    if in_poi_body and has_live_sweep:
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
    elif body_in_htf_or_ltf and dist_pct <= 1.20:
        # Harga sedang di dalam zona HTF/LTF aktif sementara Entry Zone spesifik sedikit lebih dalam
        zone_state = "SWEEP_IN_POI" if has_live_sweep else ("IN_POI_REJECTING" if reac_code == "REJECTING" else "IN_POI_TESTING")
        hz_d = body_in_htf_or_ltf[0]["depth_pct"]
        icon = "🔥 SWEEP+IN" if has_live_sweep else "🎯 IN"
        zone_badge = f"{icon} {top_tag} ({hz_d:.0f}%|{r_short})"
    elif has_live_sweep and dist_pct <= 0.90:
        zone_state = "LIQ_SWEPT"
        sw_tag = "SSL" if d == 1 else "BSL"
        zone_badge = f"💧 {sw_tag} SWEPT ({dist_pct:.2f}%)"
    elif dist_pct <= cfg.near_poi_pct:
        zone_state = "APPROACHING_POI"
        near_tag = _short_zone_tag(near_aligned[0]) if near_aligned else "POI"
        zone_badge = f"⚡ NEAR {near_tag} ({dist_pct:.2f}%)"
        reac_detail = f"MENDEKATI ZONA (Jarak {dist_pct:.2f}% — siap pasang Limit / pantau trigger)"
    else:
        zone_state = "WAITING_PULLBACK"
        zone_badge = f"⏳ WAIT ({dist_pct:.2f}%)"
        reac_detail = f"BELUM DI ZONA (Jarak {dist_pct:.2f}% dari POI)"

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
            "zone_state": setup.zone_state,
            "zone_badge": setup.zone_badge,
            "dist_to_poi_pct": round(setup.dist_to_poi_pct, 3),
            "zone_depth_pct": round(setup.zone_depth_pct, 1) if not math.isnan(setup.zone_depth_pct) else None,
            "zone_fresh": setup.zone_fresh,
            "sweep_status": setup.sweep_status,
            "zone_reaction": setup.zone_reaction,
            "active_zones_summary": setup.active_zones_summary,
            "eq_magnet_info": setup.eq_magnet_info,
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
    Memindai apakah sebuah simbol SAAT INI sedang menginjak Demand/Supply 4H-1H-15M,
    OB/Breaker/FVG segar, atau baru saja melakukan Liquidity Sweep (bahkan bila belum
    memenuhi syarat R:R setup penuh), sehingga tidak ada momentum POI live yang terlewat.
    """
    if df15 is None or len(df15) < 40 or map15 is None:
        return None

    h = df15["high"].values.astype(float)
    l = df15["low"].values.astype(float)
    c = df15["close"].values.astype(float)
    cur_p = float(c[-1])
    l_rec = float(np.min(l[-2:]))
    h_rec = float(np.max(h[-2:]))
    atr_15m = float(map15.get("atr", 0.0) or 0.0)

    sess_ctx = session_liquidity_context(df15)
    cvd_ctx = compute_cvd_metrics(df15)
    bull_zones, bear_zones = _collect_directional_zones(map4, map1, map15, cur_p, l_rec, h_rec)
    sweeps = _detect_multilayer_sweeps(df15, map1, map15, sess_ctx, cfg.piv_len)

    # Kandidat Bullish Live Event (Demand / Bull OB / SSL-EQL Sweep)
    bull_touch = [z for z in bull_zones if z["touched_now"]]
    bull_near_htf = [z for z in bull_zones if not z["touched_now"] and z["weight"] >= 2 and z["dist_pct"] <= 0.35]
    bull_live_sw = sweeps["bull_live_sweep"]

    # Kandidat Bearish Live Event (Supply / Bear OB / BSL-EQH Sweep)
    bear_touch = [z for z in bear_zones if z["touched_now"]]
    bear_near_htf = [z for z in bear_zones if not z["touched_now"] and z["weight"] >= 2 and z["dist_pct"] <= 0.35]
    bear_live_sw = sweeps["bear_live_sweep"]

    # Skor prioritas kejadian live agar kita memilih sisi yang paling aktif saat ini
    def side_strength(touches: List[dict], near_htf: List[dict], live_sw: bool) -> float:
        pts = 0.0
        if touches:
            best_w = max(z["weight"] for z in touches)
            pts += 30.0 + best_w * 12.0 + (5.0 if any(z.get("fresh") for z in touches) else 0.0)
        elif near_htf:
            pts += 18.0 + near_htf[0]["weight"] * 4.0
        if live_sw:
            pts += 28.0
        return pts

    b_pts = side_strength(bull_touch, bull_near_htf, bull_live_sw)
    s_pts = side_strength(bear_touch, bear_near_htf, bear_live_sw)

    # Jika simbol memiliki setup aktif (ENTRY READY / WATCHLIST) yang sudah di/dekat POI, selaraskan arahnya
    if existing_setup and existing_setup.status in ("ENTRY READY", "WATCHLIST"):
        if existing_setup.zone_state in ("SWEEP_IN_POI", "IN_POI_REJECTING", "IN_POI_TESTING", "LIQ_SWEPT", "APPROACHING_POI"):
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
    near_htf = bull_near_htf if d == 1 else bear_near_htf
    sw_list = sweeps["bull_sweeps"] if d == 1 else sweeps["bear_sweeps"]
    live_sw = bull_live_sw if d == 1 else bear_live_sw

    # Hanya masukkan ke Radar Live jika benar-benar sedang di zona HTF/Nested/Fresh ATAU baru saja Live Sweep ATAU sangat dekat HTF
    if not touches and not live_sw and not near_htf:
        return None
    # Jika hanya menyentuh 15M tunggal tanpa HTF & tanpa sweep & bukan setup valid, saring hanya yang Fresh OB/Breaker
    if touches and max(z["weight"] for z in touches) == 1 and not live_sw and not existing_setup:
        if not any(z.get("fresh") and z.get("ztype") in ("OB", "Breaker") for z in touches):
            return None

    reac_code, reac_detail = _evaluate_candle_reaction(df15, d, atr_15m, cvd_ctx)

    primary = touches[0] if touches else (near_htf[0] if near_htf else (bull_zones[0] if (d == 1 and bull_zones) else (bear_zones[0] if bear_zones else None)))
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

    all_t_txt = " + ".join(
        f"{z['kind']}{' (Fresh)' if z.get('fresh') else ''}" for z in touches[:3]
    ) if touches else p_zone_txt

    has_in_body = any(z["in_body"] for z in touches)
    loc_word = "IN" if has_in_body else "TAP"

    if touches and live_sw:
        r_state = "SWEEP_IN_ZONE"
        prio = 1 if reac_code == "REJECTING" else 2
        badge = f"🔥 SWEEP + {loc_word} {_short_zone_tag(touches[0])}"
    elif touches and any(z["weight"] >= 2 for z in touches):
        r_state = "IN_HTF_ZONE"
        htf_z = [z for z in touches if z["weight"] >= 2][0]
        lw_h = "IN" if htf_z["in_body"] else "TAP"
        prio = 2 if (reac_code == "REJECTING" and htf_z["in_body"]) else 3
        badge = f"🎯 {lw_h} {_short_zone_tag(htf_z)} ({'Rej' if reac_code=='REJECTING' else 'Test'})"
    elif live_sw:
        r_state = "LIVE_LIQ_SWEEP"
        prio = 3
        badge = f"💧 LIVE {'SSL' if d==1 else 'BSL'} SWEEP"
    elif touches:
        r_state = "IN_LTF_ZONE"
        prio = 4
        badge = f"🎯 {loc_word} {_short_zone_tag(touches[0])} ({'Rej' if reac_code=='REJECTING' else 'Test'})"
    else:
        r_state = "APPROACHING_HTF"
        prio = 5
        badge = f"⚡ NEAR {_short_zone_tag(near_htf[0])} ({dist_p:.2f}%)"

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
