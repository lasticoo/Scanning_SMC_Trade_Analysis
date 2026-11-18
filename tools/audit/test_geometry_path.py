# -*- coding: utf-8 -*-
"""
Unit & Regression Tests for M02 (`test_geometry_path.py`):
a. LONG: supply 4H bot = entry + 0.25*ATR1H -> blocked_by_wall True and rr < 3 (REJECT-able). SHORT mirror.
b. Without structural candidate >= 3R: require_structural_tp2=True -> rr < 3; False -> legacy 3.1R.
c. Bear OB 1H at 1.5R vs >=3R and swing high at 5R: tp2 parked right before OB when room>=3R; blocked=True when OB at 1.5R.
d. Random invariant (200 synthetic scenarios, fixed seed): LONG always entry < tp1 <= tp2 < tp3 (SHORT mirror) and tp1 <= entry + 0.68*(tp2-entry).
e. With both flags False, output is 100% identical to legacy geometry (regression across 200 random scenarios & 282 synthetic windows).
Also reports synthetic dataset pass -> REJECT breakdown by (B1 / B2 / B3).
"""
from __future__ import annotations

import os
import sys
from typing import Optional
from unittest.mock import patch

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import scanner_core.mtf_smc as mtf_smc
from scanner_core.models import MAJOR_SYMBOLS, NAN, Config
from scanner_core.mtf_smc import (
    _cap_tp2_by_opposing_zones,
    _compute_trade_geometry,
    _format_hold_duration,
    _refine_zone_bounds,
    analyze_smc_symbol,
    fp,
)
from tools.audit.synth_data import iter_synthetic_windows


def _legacy_compute_trade_geometry(
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
    rev_ctx: Optional[dict] = None,
) -> dict:
    """Reference copy of pre-M02 _compute_trade_geometry for exact regression verification."""
    notes = []
    raw_poi_low, raw_poi_high = min(poi_low, poi_high), max(poi_low, poi_high)
    max_zone_w = min(0.80 * atr_1h, 0.035 * cur_p)
    poi_low, poi_high = _refine_zone_bounds(poi_low, poi_high, d, cur_p, max_zone_w, 0.14 * atr_1h)

    span_poi = max(poi_high - poi_low, 1e-12)
    ce_50 = (poi_low + poi_high) / 2.0

    if archetype == "PRE_REACTION":
        limit_entry = (poi_high - 0.15 * span_poi) if d == 1 else (poi_low + 0.15 * span_poi)
        in_poi_now = (poi_low - 0.05 * atr_1h) <= cur_p <= (poi_high + 0.05 * atr_1h)
        at_poi = in_poi_now
        plan_entry = cur_p if in_poi_now else limit_entry
    elif archetype == "HTF_REVERSAL":
        dist_from_edge = (cur_p - poi_high) if d == 1 else (poi_low - cur_p)
        at_poi = (poi_low - 0.08 * atr_1h) <= cur_p <= (poi_high + 0.08 * atr_1h) or (0.0 <= dist_from_edge <= 0.30 * atr_1h)
        limit_entry = ce_50
        plan_entry = cur_p
    else:
        dist_from_poi = (cur_p - poi_high) if d == 1 else (poi_low - cur_p)
        at_poi = (poi_low - 0.05 * atr_1h) <= cur_p <= (poi_high + 0.05 * atr_1h) or (0.0 <= dist_from_poi <= 0.35 * atr_1h)
        limit_entry = ce_50
        plan_entry = cur_p

    w_span = max(w_high - w_low, 2.5 * atr_1h)
    min_pct = cfg.min_sl_pct_major if symbol in MAJOR_SYMBOLS else cfg.min_sl_pct_alt
    raw_min_sl = max(
        cfg.min_sl_atr_1h * atr_1h,
        cfg.min_sl_range_1h * min(w_span, 5.0 * atr_1h),
        min_pct * plan_entry,
    )
    min_sl_dist = min(raw_min_sl, 0.038 * plan_entry)
    max_sl_dist = min(
        0.135 * plan_entry,
        max(min_sl_dist * 1.35, max(cfg.max_sl_atr_1h * atr_1h, cfg.max_sl_range_1h * max(w_span, 4.0 * atr_1h))),
    )

    liq_guard_info = ""
    sweep_entry_lvl = NAN
    sl_too_wide = False

    if d == 1:
        local_swing_low = float(np.min(l_arr[-12:]))
        use_local = abs(plan_entry - local_swing_low) <= 1.8 * atr_1h
        anchor_low = min(poi_low, local_swing_low) if use_local else poi_low
        if htf_poi and abs(plan_entry - htf_poi["bot"]) <= 2.0 * atr_1h:
            anchor_low = min(anchor_low, htf_poi["bot"])

        if archetype == "HTF_REVERSAL" and rev_ctx and not np.isnan(rev_ctx.get("sweep_extreme", NAN)):
            sw_ext = float(rev_ctx["sweep_extreme"])
            if abs(plan_entry - sw_ext) <= 2.4 * atr_1h:
                anchor_low = min(anchor_low, sw_ext)
                sweep_entry_lvl = sw_ext
                max_sl_dist = max(max_sl_dist, min(0.14 * plan_entry, 2.55 * atr_1h))
                liq_guard_info = f"SL di bawah Low Sweep ({fp(sw_ext)}) + buffer ATR"

        liq_cands_l = []
        if map1:
            for eq_l in map1.get("eql", []):
                liq_cands_l.append((float(eq_l), "EQL 1H"))
            for sw_l in map1.get("lows", [])[-4:]:
                liq_cands_l.append((float(sw_l["price"]), "Swing Low 1H"))
        for eq_l15 in map15.get("eql", []):
            liq_cands_l.append((float(eq_l15), "EQL 15M"))
        for sw_l15 in map15.get("lows", [])[-5:]:
            liq_cands_l.append((float(sw_l15["price"]), "Swing Low 15M"))
        if htf_poi:
            liq_cands_l.append((float(htf_poi["bot"]), f"Wick {htf_poi.get('tf', 'HTF')} OB"))
        if raw_poi_low < poi_low:
            liq_cands_l.append((float(raw_poi_low), "Ujung Bawah Zona HTF"))
        if wall_demand_4h:
            liq_cands_l.append((float(wall_demand_4h["bot"]), "Tembok Demand 4H"))

        valid_liq_l = [
            (lvl, lbl) for lvl, lbl in liq_cands_l
            if (plan_entry - 1.60 * atr_1h) <= lvl <= (poi_low + 0.04 * atr_1h)
        ]
        if valid_liq_l:
            valid_liq_l.sort(key=lambda x: x[0])
            liq_lvl, liq_lbl = valid_liq_l[0]
            anchor_low = min(anchor_low, liq_lvl)
            max_sl_dist = max(max_sl_dist, min(0.14 * plan_entry, 2.50 * atr_1h))
            if np.isnan(sweep_entry_lvl) and liq_lvl < poi_low - 0.02 * atr_1h:
                sweep_entry_lvl = liq_lvl
            if not liq_guard_info:
                liq_guard_info = f"SL diamankan di balik {liq_lbl} ({fp(liq_lvl)}) — tahan Judas Swing"

        buf_mult = 0.18 if archetype == "HTF_REVERSAL" else cfg.sl_buffer
        sl_struct = anchor_low - buf_mult * atr_1h
        raw_dist = max(0.0, plan_entry - sl_struct)
        if raw_dist > max_sl_dist * 1.15:
            sl_too_wide = True
        final_sl_dist = max(raw_dist, min_sl_dist)
        sl = min(plan_entry - final_sl_dist, anchor_low - 0.12 * cur_atr)
    else:
        local_swing_high = float(np.max(h_arr[-12:]))
        use_local = abs(local_swing_high - plan_entry) <= 1.8 * atr_1h
        anchor_high = max(poi_high, local_swing_high) if use_local else poi_high
        if htf_poi and abs(htf_poi["top"] - plan_entry) <= 2.0 * atr_1h:
            anchor_high = max(anchor_high, htf_poi["top"])

        if archetype == "HTF_REVERSAL" and rev_ctx and not np.isnan(rev_ctx.get("sweep_extreme", NAN)):
            sw_ext = float(rev_ctx["sweep_extreme"])
            if abs(sw_ext - plan_entry) <= 2.4 * atr_1h:
                anchor_high = max(anchor_high, sw_ext)
                sweep_entry_lvl = sw_ext
                max_sl_dist = max(max_sl_dist, min(0.14 * plan_entry, 2.55 * atr_1h))
                liq_guard_info = f"SL di atas High Sweep ({fp(sw_ext)}) + ekstra buffer Short Squeeze"

        liq_cands_h = []
        if map1:
            for eq_h in map1.get("eqh", []):
                liq_cands_h.append((float(eq_h), "EQH 1H"))
            for sw_h in map1.get("highs", [])[-4:]:
                liq_cands_h.append((float(sw_h["price"]), "Swing High 1H"))
        for eq_h15 in map15.get("eqh", []):
            liq_cands_h.append((float(eq_h15), "EQH 15M"))
        for sw_h15 in map15.get("lows", [])[-5:]:
            pass
        for sw_h15 in map15.get("highs", [])[-5:]:
            liq_cands_h.append((float(sw_h15["price"]), "Swing High 15M"))
        if htf_poi:
            liq_cands_h.append((float(htf_poi["top"]), f"Wick {htf_poi.get('tf', 'HTF')} OB"))
        if raw_poi_high > poi_high:
            liq_cands_h.append((float(raw_poi_high), "Ujung Atas Zona HTF"))
        if wall_supply_4h:
            liq_cands_h.append((float(wall_supply_4h["top"]), "Tembok Supply 4H"))

        valid_liq_h = [
            (lvl, lbl) for lvl, lbl in liq_cands_h
            if (poi_high - 0.04 * atr_1h) <= lvl <= (plan_entry + 1.60 * atr_1h)
        ]
        if valid_liq_h:
            valid_liq_h.sort(key=lambda x: -x[0])
            liq_lvl, liq_lbl = valid_liq_h[0]
            anchor_high = max(anchor_high, liq_lvl)
            max_sl_dist = max(max_sl_dist, min(0.14 * plan_entry, 2.50 * atr_1h))
            if np.isnan(sweep_entry_lvl) and liq_lvl > poi_high + 0.02 * atr_1h:
                sweep_entry_lvl = liq_lvl
            if not liq_guard_info:
                liq_guard_info = f"SL diamankan di balik {liq_lbl} ({fp(liq_lvl)}) — tahan Judas Swing"

        buf_mult = 0.24 if archetype == "HTF_REVERSAL" else cfg.sl_buffer
        sl_struct = anchor_high + buf_mult * atr_1h
        raw_dist = max(0.0, sl_struct - plan_entry)
        if raw_dist > max_sl_dist * 1.15:
            sl_too_wide = True
        final_sl_dist = max(raw_dist, min_sl_dist)
        sl = max(plan_entry + final_sl_dist, anchor_high + 0.15 * cur_atr)

    risk = max(abs(plan_entry - sl), 1e-12)
    sl_atr_1h_ratio = risk / max(atr_1h, 1e-12)
    if not liq_guard_info:
        liq_guard_info = f"SL Struktural 1H ({sl_atr_1h_ratio:.2f}x ATR 1H)"

    vel_factor = 0.50 if (cur_rv >= 1.4 and htf_pts >= 14.0) else 0.44
    hourly_vel = max(vel_factor * atr_1h, 1e-12)
    eff_max_hold = max(float(getattr(cfg, "max_hold_hours", 30.0)), 18.0)
    max_reach_dist = eff_max_hold * hourly_vel
    target_min_rr = max(float(getattr(cfg, "min_rr", 3.0)), 3.0)
    eq_4h = (r4_lo + r4_hi) / 2.0

    blocked_by_wall = False

    if d == 1:
        wall_cap = None
        if (
            wall_supply_4h
            and wall_supply_4h.get("touches", 0) <= 2
            and wall_supply_4h["bot"] > plan_entry + 0.30 * atr_1h
        ):
            wall_cap = max(plan_entry + 0.10 * atr_1h, wall_supply_4h["bot"] - 0.05 * atr_1h)

        cands_tp = [w_eq, w_high, eq_4h]
        if map1:
            cands_tp.extend(map1.get("eqh", []))
            cands_tp.extend([s["price"] for s in map1.get("highs", [])[-8:]])
            cands_tp.extend([z["bot"] for z in map1.get("bear_obs", []) if z["bot"] > plan_entry])
            cands_tp.extend([z["bot"] for z in map1.get("fvgs", []) if z["dir"] == -1 and z["bot"] > plan_entry])
        cands_tp.extend(map15.get("eqh", []))
        if wall_supply_4h and wall_supply_4h["bot"] > plan_entry:
            cands_tp.append(wall_supply_4h["bot"] - 0.05 * atr_1h)
            cands_tp.append(wall_supply_4h["top"])
        if r4_hi > plan_entry:
            cands_tp.append(r4_hi)

        struct_ceiling = max([x for x in cands_tp if x > plan_entry] + [r4_hi, w_high])
        reach_ceiling = min(struct_ceiling, plan_entry + max_reach_dist)
        if wall_cap is not None:
            reach_ceiling = min(reach_ceiling, wall_cap)

        valid_tp2 = sorted([x for x in cands_tp if (plan_entry + target_min_rr * risk) <= x <= reach_ceiling])
        if archetype == "HTF_REVERSAL" and (plan_entry + target_min_rr * risk) <= eq_4h <= reach_ceiling:
            tp2 = eq_4h
        elif valid_tp2:
            sweet_tp2 = [x for x in valid_tp2 if x <= plan_entry + 4.8 * risk]
            tp2 = sweet_tp2[0] if sweet_tp2 else min(valid_tp2[0], reach_ceiling)
        else:
            tp2 = min(plan_entry + (target_min_rr + 0.10) * risk, reach_ceiling)

        if wall_cap is not None and tp2 >= wall_cap - 1e-9 and (wall_cap - plan_entry) / risk < target_min_rr:
            blocked_by_wall = True
            notes.append(f"Ruang target terhalang Tembok Supply 4H ({fp(wall_supply_4h['bot'])}) sebelum mencapai 1:{target_min_rr:.1f} R:R.")

        h1_tp1_cands = []
        if map1:
            h1_tp1_cands.extend(map1.get("eqh", []))
            h1_tp1_cands.extend([s["price"] for s in map1.get("highs", [])[-8:]])
        h1_tp1_cands.append(w_eq)
        valid_tp1 = sorted([x for x in h1_tp1_cands if plan_entry + 1.30 * risk <= x <= plan_entry + 0.68 * max(tp2 - plan_entry, 0)])
        if valid_tp1:
            tp1 = valid_tp1[0]
        else:
            tp1 = plan_entry + 0.52 * max(tp2 - plan_entry, risk)
        opp_lvl = wall_supply_4h["bot"] if wall_supply_4h else w_high

        runner_cands = [x for x in cands_tp + [r4_hi, w_high] if x > tp2 + 0.8 * risk]
        if wall_supply_4h and wall_supply_4h["bot"] > tp2 + 0.5 * risk:
            runner_cands.append(wall_supply_4h["bot"])
        if runner_cands:
            tp3 = min(max(runner_cands), plan_entry + 8.0 * risk)
        else:
            tp3 = max(tp2 + 1.2 * risk, struct_ceiling)
    else:
        wall_cap = None
        if (
            wall_demand_4h
            and wall_demand_4h.get("touches", 0) <= 2
            and wall_demand_4h["top"] < plan_entry - 0.30 * atr_1h
        ):
            wall_cap = min(plan_entry - 0.10 * atr_1h, wall_demand_4h["top"] + 0.05 * atr_1h)

        cands_tp = [w_eq, w_low, eq_4h]
        if map1:
            cands_tp.extend(map1.get("eql", []))
            cands_tp.extend([s["price"] for s in map1.get("lows", [])[-8:]])
            cands_tp.extend([z["top"] for z in map1.get("bull_obs", []) if z["top"] < plan_entry])
            cands_tp.extend([z["top"] for z in map1.get("fvgs", []) if z["dir"] == 1 and z["top"] < plan_entry])
        cands_tp.extend(map15.get("eql", []))
        if wall_demand_4h and wall_demand_4h["top"] < plan_entry:
            cands_tp.append(wall_demand_4h["top"] + 0.05 * atr_1h)
            cands_tp.append(wall_demand_4h["bot"])
        if r4_lo < plan_entry:
            cands_tp.append(r4_lo)

        struct_floor = min([x for x in cands_tp if 0 < x < plan_entry] + [r4_lo, w_low])
        reach_floor = max(struct_floor, plan_entry - max_reach_dist, plan_entry * 0.25)
        if wall_cap is not None:
            reach_floor = max(reach_floor, wall_cap)

        valid_tp2 = sorted([x for x in cands_tp if reach_floor <= x <= (plan_entry - target_min_rr * risk)], reverse=True)
        if archetype == "HTF_REVERSAL" and reach_floor <= eq_4h <= (plan_entry - target_min_rr * risk):
            tp2 = eq_4h
        elif valid_tp2:
            sweet_tp2 = [x for x in valid_tp2 if x >= plan_entry - 4.8 * risk]
            tp2 = sweet_tp2[0] if sweet_tp2 else max(valid_tp2[0], reach_floor)
        else:
            tp2 = max(plan_entry - (target_min_rr + 0.10) * risk, reach_floor)

        if wall_cap is not None and tp2 <= wall_cap + 1e-9 and (plan_entry - wall_cap) / risk < target_min_rr:
            blocked_by_wall = True
            notes.append(f"Ruang target terhalang Tembok Demand 4H ({fp(wall_demand_4h['top'])}) sebelum mencapai 1:{target_min_rr:.1f} R:R.")

        h1_tp1_cands = []
        if map1:
            h1_tp1_cands.extend(map1.get("eql", []))
            h1_tp1_cands.extend([s["price"] for s in map1.get("lows", [])[-8:]])
        h1_tp1_cands.append(w_eq)
        valid_tp1 = sorted(
            [x for x in h1_tp1_cands if plan_entry - 0.68 * max(plan_entry - tp2, 0) <= x <= plan_entry - 1.30 * risk],
            reverse=True,
        )
        if valid_tp1:
            tp1 = valid_tp1[0]
        else:
            tp1 = plan_entry - 0.52 * max(plan_entry - tp2, risk)
        opp_lvl = wall_demand_4h["top"] if wall_demand_4h else w_low

        runner_cands = [x for x in cands_tp + [r4_lo, w_low] if 0 < x < tp2 - 0.8 * risk]
        if wall_demand_4h and wall_demand_4h["top"] < tp2 - 0.5 * risk:
            runner_cands.append(wall_demand_4h["top"])
        if runner_cands:
            tp3 = max(min(runner_cands), max(plan_entry * 0.25, plan_entry - 8.0 * risk))
        else:
            tp3 = max(plan_entry * 0.25, min(tp2 - 1.2 * risk, struct_floor))

    rr = (tp2 - plan_entry) / risk if d == 1 else (plan_entry - tp2) / risk
    rr = max(0.0, rr)
    rr_tp1 = abs(tp1 - plan_entry) / risk
    rr_tp3 = abs(tp3 - plan_entry) / risk
    est_h1 = max(0.5, abs(tp1 - plan_entry) / hourly_vel)
    est_h2 = max(1.0, abs(tp2 - plan_entry) / hourly_vel)
    hold_label = _format_hold_duration(est_h1, est_h2)

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
        "sl_too_wide": sl_too_wide,
        "blocked_by_wall": blocked_by_wall,
        "liq_guard_info": liq_guard_info,
        "sweep_entry_lvl": sweep_entry_lvl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "rr": rr,
        "rr_tp1": rr_tp1,
        "rr_tp3": rr_tp3,
        "opp_lvl": opp_lvl,
        "hourly_vel": hourly_vel,
        "est_h1": est_h1,
        "est_h2": est_h2,
        "hold_label": hold_label,
        "est_tap_h": est_tap_h,
        "notes": notes,
    }


def _base_arrays(cur_p: float = 100.0):
    l_arr = np.full(20, cur_p - 0.5, dtype=float)
    h_arr = np.full(20, cur_p + 0.5, dtype=float)
    return l_arr, h_arr


def _empty_map1():
    return {"eqh": [], "eql": [], "highs": [], "lows": [], "bear_obs": [], "bull_obs": [], "fvgs": [], "breakers": []}


def _empty_map15():
    return {"eqh": [], "eql": [], "highs": [], "lows": []}


def test_a_near_4h_wall_blocks_tp2() -> None:
    cfg = Config()
    l_arr, h_arr = _base_arrays(100.0)

    # LONG: supply 4H bot = entry + 0.25*ATR1H (100.25)
    g_long = _compute_trade_geometry(
        d=1,
        archetype="TREND_CONT",
        cur_p=100.0,
        poi_low=99.6,
        poi_high=100.0,
        htf_poi=None,
        approach_code="KOMPRESI_KOREKTIF",
        symbol="TESTUSDT",
        cfg=cfg,
        atr_1h=1.0,
        cur_atr=0.35,
        w_low=96.0,
        w_high=108.0,
        w_eq=104.5,
        r4_lo=94.0,
        r4_hi=110.0,
        wall_supply_4h={"bot": 100.25, "top": 101.00, "touches": 0, "kind": "4H Bearish OB"},
        wall_demand_4h=None,
        map1=_empty_map1(),
        map15=_empty_map15(),
        l_arr=l_arr,
        h_arr=h_arr,
        htf_pts=18.0,
        cur_rv=1.2,
    )
    assert g_long["blocked_by_wall"] is True, f"Expected LONG blocked_by_wall=True, got {g_long}"
    assert g_long["rr"] < 3.0, f"Expected LONG rr < 3.0, got {g_long['rr']}"
    assert any("terhalang" in n for n in g_long["notes"]), f"Missing terhalang note in {g_long['notes']}"

    # SHORT mirror: demand 4H top = entry - 0.25*ATR1H (99.75)
    g_short = _compute_trade_geometry(
        d=-1,
        archetype="TREND_CONT",
        cur_p=100.0,
        poi_low=100.0,
        poi_high=100.4,
        htf_poi=None,
        approach_code="KOMPRESI_KOREKTIF",
        symbol="TESTUSDT",
        cfg=cfg,
        atr_1h=1.0,
        cur_atr=0.35,
        w_low=92.0,
        w_high=104.0,
        w_eq=95.5,
        r4_lo=90.0,
        r4_hi=106.0,
        wall_supply_4h=None,
        wall_demand_4h={"bot": 99.00, "top": 99.75, "touches": 0, "kind": "4H Bullish OB"},
        map1=_empty_map1(),
        map15=_empty_map15(),
        l_arr=l_arr,
        h_arr=h_arr,
        htf_pts=18.0,
        cur_rv=1.2,
    )
    assert g_short["blocked_by_wall"] is True, f"Expected SHORT blocked_by_wall=True, got {g_short}"
    assert g_short["rr"] < 3.0, f"Expected SHORT rr < 3.0, got {g_short['rr']}"
    assert any("terhalang" in n for n in g_short["notes"]), f"Missing terhalang note in {g_short['notes']}"

    print(
        f"[PASS] Test (a): 4H wall at 0.25*ATR1H caps TP2 -> blocked_by_wall=True & rr<3 "
        f"(LONG rr={g_long['rr']:.2f}, SHORT rr={g_short['rr']:.2f})"
    )


def test_b_require_structural_tp2_toggle() -> None:
    l_arr, h_arr = _base_arrays(100.0)
    # Closest structure at 102.0 (~1.67R) and next structure at 120.0 (beyond max_reach_dist ~7.92 -> reach_ceiling=107.92)
    m1_long = _empty_map1()
    m1_long["highs"] = [{"price": 102.0, "idx": 10, "time": 0}]

    cfg_true = Config(require_structural_tp2=True, max_hold_hours=18.0)
    cfg_false = Config(require_structural_tp2=False, max_hold_hours=18.0)

    g_l_true = _compute_trade_geometry(
        d=1, archetype="TREND_CONT", cur_p=100.0, poi_low=99.6, poi_high=100.0, htf_poi=None,
        approach_code="KOMPRESI_KOREKTIF", symbol="TESTUSDT", cfg=cfg_true, atr_1h=1.0, cur_atr=0.35,
        w_low=96.0, w_high=120.0, w_eq=102.0, r4_lo=84.0, r4_hi=120.0,
        wall_supply_4h=None, wall_demand_4h=None, map1=m1_long, map15=_empty_map15(),
        l_arr=l_arr, h_arr=h_arr, htf_pts=18.0, cur_rv=1.0,
    )
    g_l_false = _compute_trade_geometry(
        d=1, archetype="TREND_CONT", cur_p=100.0, poi_low=99.6, poi_high=100.0, htf_poi=None,
        approach_code="KOMPRESI_KOREKTIF", symbol="TESTUSDT", cfg=cfg_false, atr_1h=1.0, cur_atr=0.35,
        w_low=96.0, w_high=120.0, w_eq=102.0, r4_lo=84.0, r4_hi=120.0,
        wall_supply_4h=None, wall_demand_4h=None, map1=m1_long, map15=_empty_map15(),
        l_arr=l_arr, h_arr=h_arr, htf_pts=18.0, cur_rv=1.0,
    )
    assert g_l_true["tp2"] == 102.0 and g_l_true["rr"] < 3.0, f"Expected LONG structural tp2=102.0 (<3R), got {g_l_true}"
    assert "TP2 struktural tidak tersedia >= 3R" in g_l_true["notes"]
    assert abs(g_l_false["rr"] - 3.10) < 1e-6, f"Expected legacy 3.10R when require_structural_tp2=False, got {g_l_false['rr']}"

    # SHORT mirror: closest structure at 98.0 (~1.82R), next at 80.0 (beyond reach_floor=92.08), eq_4h=(80+116)/2=98.0
    m1_short = _empty_map1()
    m1_short["lows"] = [{"price": 98.0, "idx": 10, "time": 0}]
    g_s_true = _compute_trade_geometry(
        d=-1, archetype="TREND_CONT", cur_p=100.0, poi_low=100.0, poi_high=100.4, htf_poi=None,
        approach_code="KOMPRESI_KOREKTIF", symbol="TESTUSDT", cfg=cfg_true, atr_1h=1.0, cur_atr=0.35,
        w_low=80.0, w_high=104.0, w_eq=98.0, r4_lo=80.0, r4_hi=116.0,
        wall_supply_4h=None, wall_demand_4h=None, map1=m1_short, map15=_empty_map15(),
        l_arr=l_arr, h_arr=h_arr, htf_pts=18.0, cur_rv=1.0,
    )
    g_s_false = _compute_trade_geometry(
        d=-1, archetype="TREND_CONT", cur_p=100.0, poi_low=100.0, poi_high=100.4, htf_poi=None,
        approach_code="KOMPRESI_KOREKTIF", symbol="TESTUSDT", cfg=cfg_false, atr_1h=1.0, cur_atr=0.35,
        w_low=80.0, w_high=104.0, w_eq=98.0, r4_lo=80.0, r4_hi=116.0,
        wall_supply_4h=None, wall_demand_4h=None, map1=m1_short, map15=_empty_map15(),
        l_arr=l_arr, h_arr=h_arr, htf_pts=18.0, cur_rv=1.0,
    )
    assert g_s_true["tp2"] == 98.0 and g_s_true["rr"] < 3.0
    assert "TP2 struktural tidak tersedia >= 3R" in g_s_true["notes"]
    assert abs(g_s_false["rr"] - 3.10) < 1e-6

    print(
        f"[PASS] Test (b): Without structural TP2 >= 3R -> require_structural_tp2=True yields rr={g_l_true['rr']:.2f} (<3), "
        f"False yields rr={g_l_false['rr']:.2f} (3.1R)"
    )


def test_c_opposing_1h_zones_corridor() -> None:
    cfg = Config()
    l_arr, h_arr = _base_arrays(100.0)

    # First get exact risk for this baseline setup
    g_ref = _compute_trade_geometry(
        d=1, archetype="TREND_CONT", cur_p=100.0, poi_low=99.6, poi_high=100.0, htf_poi=None,
        approach_code="KOMPRESI_KOREKTIF", symbol="TESTUSDT", cfg=cfg, atr_1h=1.0, cur_atr=0.35,
        w_low=96.0, w_high=112.0, w_eq=101.0, r4_lo=94.0, r4_hi=112.0,
        wall_supply_4h=None, wall_demand_4h=None, map1=_empty_map1(), map15=_empty_map15(),
        l_arr=l_arr, h_arr=h_arr, htf_pts=18.0, cur_rv=1.5,
    )
    entry = g_ref["plan_entry"]
    risk = g_ref["risk"]
    atr_1h = 1.0
    swing_5r = entry + 5.0 * risk

    # Case 1: Bear OB 1H at 1.5R and swing high at 5R -> blocked_by_wall=True
    m1_ob_15r = _empty_map1()
    m1_ob_15r["highs"] = [{"price": swing_5r, "idx": 12, "time": 0}]
    m1_ob_15r["bear_obs"] = [{
        "bot": entry + 1.5 * risk,
        "top": entry + 1.9 * risk,
        "touches": 0,
        "dir": -1,
        "kind": "1H Bearish OB",
    }]
    g_blocked = _compute_trade_geometry(
        d=1, archetype="TREND_CONT", cur_p=100.0, poi_low=99.6, poi_high=100.0, htf_poi=None,
        approach_code="KOMPRESI_KOREKTIF", symbol="TESTUSDT", cfg=cfg, atr_1h=atr_1h, cur_atr=0.35,
        w_low=96.0, w_high=entry + 7.0 * risk, w_eq=entry + 1.0 * risk, r4_lo=94.0, r4_hi=entry + 8.0 * risk,
        wall_supply_4h=None, wall_demand_4h=None, map1=m1_ob_15r, map15=_empty_map15(),
        l_arr=l_arr, h_arr=h_arr, htf_pts=18.0, cur_rv=1.5,
    )
    assert g_blocked["blocked_by_wall"] is True, f"Expected blocked_by_wall=True for 1.5R Bear OB, got {g_blocked}"
    assert any("Jalur TP2 terhalang" in n for n in g_blocked["notes"]), f"Missing terhalang note in {g_blocked['notes']}"

    # Case 2: Bear OB 1H at 3.5R (room >= 3R) and swing high at 5R -> tp2 parked at front - 0.05*atr_1h, blocked=False
    ob_35r_bot = entry + 3.5 * risk
    m1_ob_35r = _empty_map1()
    m1_ob_35r["highs"] = [{"price": swing_5r, "idx": 12, "time": 0}]
    m1_ob_35r["bear_obs"] = [{
        "bot": ob_35r_bot,
        "top": entry + 4.0 * risk,
        "touches": 1,
        "dir": -1,
        "kind": "1H Bearish OB",
    }]
    g_parked = _compute_trade_geometry(
        d=1, archetype="TREND_CONT", cur_p=100.0, poi_low=99.6, poi_high=100.0, htf_poi=None,
        approach_code="KOMPRESI_KOREKTIF", symbol="TESTUSDT", cfg=cfg, atr_1h=atr_1h, cur_atr=0.35,
        w_low=96.0, w_high=entry + 7.0 * risk, w_eq=entry + 1.0 * risk, r4_lo=94.0, r4_hi=entry + 8.0 * risk,
        wall_supply_4h=None, wall_demand_4h=None, map1=m1_ob_35r, map15=_empty_map15(),
        l_arr=l_arr, h_arr=h_arr, htf_pts=18.0, cur_rv=1.5,
    )
    expected_cap = ob_35r_bot - 0.05 * atr_1h
    assert g_parked["blocked_by_wall"] is False, f"Expected blocked_by_wall=False when room>=3R, got {g_parked}"
    assert abs(g_parked["tp2"] - expected_cap) < 1e-9, f"Expected parked tp2={expected_cap}, got {g_parked['tp2']}"
    assert g_parked["rr"] >= 3.0
    assert any("TP2 diparkir" in n for n in g_parked["notes"]), f"Missing 'TP2 diparkir' note in {g_parked['notes']}"

    # Direct helper check for SHORT with 1H Breaker (touches<=1) & 1H FVG (width >= 0.25*atr_1h)
    cap_s, blk_s, note_s = _cap_tp2_by_opposing_zones(
        d=-1,
        entry=100.0,
        tp2=95.0,
        risk=1.0,
        atr_1h=1.0,
        map1={
            "bull_obs": [],
            "breakers": [{"dir": 1, "bot": 96.0, "top": 96.4, "touches": 1, "kind": "1H Bullish Breaker"}],
            "fvgs": [{"dir": 1, "bot": 95.2, "top": 95.6, "kind": "1H Bullish FVG"}],
        },
        wall4=None,
        min_rr=3.0,
    )
    assert blk_s is False and abs(cap_s - (96.4 + 0.05)) < 1e-9 and "TP2 diparkir" in note_s

    print(
        f"[PASS] Test (c): Bear OB 1H at 1.5R -> blocked=True; at 3.5R -> TP2 parked at {g_parked['tp2']:.3f} "
        f"(RR={g_parked['rr']:.2f}) with note '{g_parked['notes'][0]}'"
    )


def _generate_200_random_scenarios(seed: int = 20261118):
    rng = np.random.default_rng(seed)
    scenarios = []
    archetypes = ["TREND_CONT", "PRE_REACTION", "HTF_REVERSAL"]
    for i in range(200):
        d = 1 if (i % 2 == 0) else -1
        arch = archetypes[i % 3]
        cur_p = float(rng.uniform(10.0, 500.0))
        atr_1h = cur_p * float(rng.uniform(0.008, 0.035))
        cur_atr = atr_1h * float(rng.uniform(0.25, 0.55))
        zone_w = atr_1h * float(rng.uniform(0.15, 0.70))
        if d == 1:
            poi_high = cur_p - atr_1h * float(rng.uniform(-0.1, 0.4))
            poi_low = poi_high - zone_w
        else:
            poi_low = cur_p + atr_1h * float(rng.uniform(-0.1, 0.4))
            poi_high = poi_low + zone_w

        w_low = cur_p - atr_1h * float(rng.uniform(1.0, 8.0))
        w_high = cur_p + atr_1h * float(rng.uniform(1.0, 8.0))
        w_eq = (w_low + w_high) / 2.0
        r4_lo = w_low - atr_1h * float(rng.uniform(0.0, 4.0))
        r4_hi = w_high + atr_1h * float(rng.uniform(0.0, 4.0))

        l_arr = cur_p - atr_1h * rng.uniform(0.1, 1.5, size=20)
        h_arr = cur_p + atr_1h * rng.uniform(0.1, 1.5, size=20)

        m1 = _empty_map1()
        for _ in range(int(rng.integers(0, 5))):
            m1["highs"].append({"price": cur_p + atr_1h * float(rng.uniform(0.3, 9.0)), "idx": 10, "time": 0})
            m1["lows"].append({"price": max(0.5 * cur_p, cur_p - atr_1h * float(rng.uniform(0.3, 9.0))), "idx": 10, "time": 0})
        for _ in range(int(rng.integers(0, 3))):
            ob_b = cur_p + atr_1h * float(rng.uniform(0.1, 6.0))
            m1["bear_obs"].append({
                "bot": ob_b,
                "top": ob_b + atr_1h * float(rng.uniform(0.2, 0.8)),
                "touches": int(rng.integers(0, 3)),
                "dir": -1,
                "kind": "1H Bearish OB",
            })
            ob_t = max(0.5 * cur_p, cur_p - atr_1h * float(rng.uniform(0.1, 6.0)))
            m1["bull_obs"].append({
                "bot": max(0.4 * cur_p, ob_t - atr_1h * float(rng.uniform(0.2, 0.8))),
                "top": ob_t,
                "touches": int(rng.integers(0, 3)),
                "dir": 1,
                "kind": "1H Bullish OB",
            })
        if rng.random() < 0.5:
            f_b = cur_p + d * atr_1h * float(rng.uniform(0.2, 5.0))
            f_w = atr_1h * float(rng.uniform(0.10, 0.60))
            m1["fvgs"].append({
                "dir": -d,
                "bot": min(f_b, f_b + d * f_w),
                "top": max(f_b, f_b + d * f_w),
                "kind": "1H FVG",
            })

        wall_sup = None
        wall_dem = None
        if rng.random() < 0.6:
            wb = cur_p + atr_1h * float(rng.uniform(0.05, 6.0))
            wall_sup = {"bot": wb, "top": wb + 0.6 * atr_1h, "touches": int(rng.integers(0, 3)), "kind": "4H Supply"}
        if rng.random() < 0.6:
            wt = max(0.5 * cur_p, cur_p - atr_1h * float(rng.uniform(0.05, 6.0)))
            wall_dem = {"bot": max(0.4 * cur_p, wt - 0.6 * atr_1h), "top": wt, "touches": int(rng.integers(0, 3)), "kind": "4H Demand"}

        scenarios.append({
            "d": d,
            "archetype": arch,
            "cur_p": cur_p,
            "poi_low": poi_low,
            "poi_high": poi_high,
            "htf_poi": None,
            "approach_code": "KOMPRESI_KOREKTIF",
            "symbol": "BTCUSDT" if (i % 5 == 0) else "SOLUSDT",
            "atr_1h": atr_1h,
            "cur_atr": cur_atr,
            "w_low": w_low,
            "w_high": w_high,
            "w_eq": w_eq,
            "r4_lo": r4_lo,
            "r4_hi": r4_hi,
            "wall_supply_4h": wall_sup,
            "wall_demand_4h": wall_dem,
            "map1": m1,
            "map15": _empty_map15(),
            "l_arr": l_arr,
            "h_arr": h_arr,
            "htf_pts": float(rng.uniform(10.0, 22.0)),
            "cur_rv": float(rng.uniform(0.8, 2.0)),
            "rev_ctx": None,
        })
    return scenarios


def test_d_random_200_ordering_invariant() -> None:
    cfg = Config()
    scenarios = _generate_200_random_scenarios(seed=20261118)
    for idx, sc in enumerate(scenarios):
        g = _compute_trade_geometry(cfg=cfg, **sc)
        entry = g["plan_entry"]
        tp1, tp2, tp3 = g["tp1"], g["tp2"], g["tp3"]
        risk = g["risk"]
        if sc["d"] == 1:
            assert entry < tp1 <= tp2 < tp3, (
                f"LONG ordering violated at scenario #{idx}: entry={entry}, tp1={tp1}, tp2={tp2}, tp3={tp3}"
            )
            assert tp1 <= entry + 0.68 * (tp2 - entry) + 1e-9, (
                f"LONG tp1 > entry + 0.68*(tp2-entry) at #{idx}: tp1={tp1}, limit={entry + 0.68*(tp2-entry)}"
            )
            assert tp3 >= tp2 + 1.2 * risk - 1e-9
        else:
            assert tp3 < tp2 <= tp1 < entry, (
                f"SHORT ordering violated at scenario #{idx}: entry={entry}, tp1={tp1}, tp2={tp2}, tp3={tp3}"
            )
            assert tp1 >= entry - 0.68 * (entry - tp2) - 1e-9, (
                f"SHORT tp1 < entry - 0.68*(entry-tp2) at #{idx}: tp1={tp1}, limit={entry - 0.68*(entry-tp2)}"
            )
            assert tp3 <= tp2 - 1.2 * risk + 1e-9

        # Verify rr_tp1, rr_tp3, est_h1, est_h2 are computed AFTER B5 adjustment
        assert abs(g["rr_tp1"] - abs(tp1 - entry) / risk) < 1e-9
        assert abs(g["rr_tp3"] - abs(tp3 - entry) / risk) < 1e-9
        assert abs(g["est_h1"] - max(0.5, abs(tp1 - entry) / g["hourly_vel"])) < 1e-9
        assert abs(g["est_h2"] - max(1.0, abs(tp2 - entry) / g["hourly_vel"])) < 1e-9

    print("[PASS] Test (d): 200/200 random scenarios satisfy entry < tp1 <= tp2 < tp3 (and SHORT mirror) + post-B5 RR/hold metrics")


def test_e_both_flags_false_regression_and_synth_impact() -> None:
    cfg_off = Config(require_structural_tp2=False, path_clear_check=False)
    scenarios = _generate_200_random_scenarios(seed=20261118)

    for idx, sc in enumerate(scenarios):
        g_new = _compute_trade_geometry(cfg=cfg_off, **sc)
        g_old = _legacy_compute_trade_geometry(cfg=cfg_off, **sc)
        for k in ("plan_entry", "sl", "risk", "tp1", "tp2", "tp3", "rr", "rr_tp1", "rr_tp3", "blocked_by_wall", "est_h1", "est_h2", "notes"):
            v1, v2 = g_new[k], g_old[k]
            if isinstance(v1, float):
                assert abs(v1 - v2) < 1e-12, f"Mismatch at scenario #{idx} key={k}: {v1} vs {v2}"
            else:
                assert v1 == v2, f"Mismatch at scenario #{idx} key={k}: {v1} vs {v2}"

    # Evaluate across all 282 synthetic windows from synth_data.py:
    # 1) Verify 282/282 identical when both flags are False
    # 2) Measure pass -> REJECT impact of B1 alone, B2 alone, B3 alone, and M02 combined
    cases = iter_synthetic_windows()
    cfg_all_on = Config(require_structural_tp2=True, path_clear_check=True)

    ident_282 = 0
    base_pass_ids = set()
    all_on_pass_ids = set()
    b1_rej_ids = set()
    b2_rej_ids = set()
    b3_rej_ids = set()

    def _run_case(c: dict, cfg_use: Config):
        return analyze_smc_symbol(
            c["symbol"], c["df15"], c["ctx4"], c["ctx1"], c["meta"], cfg_use,
            df4=c["df4"], df1=c["df1"], deriv=c["deriv"], btc=c["btc"],
        )

    for c in cases:
        cid = c["case_id"]
        st_off = _run_case(c, cfg_off)
        with patch.object(mtf_smc, "_compute_trade_geometry", _legacy_compute_trade_geometry):
            st_leg = _run_case(c, cfg_off)
        if (
            st_off.status == st_leg.status
            and abs(float(st_off.score) - float(st_leg.score)) < 1e-9
            and abs(float(st_off.entry) - float(st_leg.entry)) < 1e-9
            and abs(float(st_off.sl) - float(st_leg.sl)) < 1e-9
            and abs(float(st_off.tp1) - float(st_leg.tp1)) < 1e-9
            and abs(float(st_off.tp2) - float(st_leg.tp2)) < 1e-9
            and st_off.archetype == st_leg.archetype
        ):
            ident_282 += 1

        if st_off.status != "REJECT":
            base_pass_ids.add(cid)

        st_all = _run_case(c, cfg_all_on)
        if st_all.status != "REJECT":
            all_on_pass_ids.add(cid)
        elif cid in base_pass_ids:
            notes_off_str = " | ".join(st_off.notes)
            notes_all_str = " | ".join(st_all.notes)
            if "Ruang target terhalang Tembok" in notes_all_str and "Ruang target terhalang Tembok" not in notes_off_str:
                b1_rej_ids.add(cid)
            elif "TP2 struktural tidak tersedia >= 3R" in notes_all_str:
                b2_rej_ids.add(cid)
            elif "Jalur TP2 terhalang" in notes_all_str:
                b3_rej_ids.add(cid)

    assert ident_282 == len(cases), f"Expected 282/282 identical with both flags False, got {ident_282}/{len(cases)}"

    n_total = len(cases)
    n_base_pass = len(base_pass_ids)
    n_all_pass = len(all_on_pass_ids)
    b1_rej = len(b1_rej_ids)
    b2_rej = len(b2_rej_ids)
    b3_rej = len(b3_rej_ids)
    tot_rej = len(base_pass_ids - all_on_pass_ids)

    print(
        f"[PASS] Test (e): Both flags False regression 200/200 scenarios & {ident_282}/{n_total} synthetic setups 100% identical"
    )
    print(
        f"[INFO] Synthetic setups impact (total={n_total}): "
        f"Before (lolos)={n_base_pass} ({n_base_pass/n_total*100:.1f}%) -> "
        f"After M02 (lolos)={n_all_pass} ({n_all_pass/n_total*100:.1f}%), "
        f"berubah lolos->REJECT={tot_rej}/{n_base_pass} ({tot_rej/max(n_base_pass,1)*100:.1f}% dari lolos | {tot_rej/n_total*100:.1f}% dari 282) "
        f"[B1={b1_rej} ({b1_rej/max(n_base_pass,1)*100:.1f}% dari lolos | {b1_rej/n_total*100:.1f}% dari 282), "
        f"B2={b2_rej} ({b2_rej/max(n_base_pass,1)*100:.1f}% dari lolos | {b2_rej/n_total*100:.1f}% dari 282), "
        f"B3={b3_rej} ({b3_rej/max(n_base_pass,1)*100:.1f}% dari lolos | {b3_rej/n_total*100:.1f}% dari 282)]"
    )


if __name__ == "__main__":
    test_a_near_4h_wall_blocks_tp2()
    test_b_require_structural_tp2_toggle()
    test_c_opposing_1h_zones_corridor()
    test_d_random_200_ordering_invariant()
    test_e_both_flags_false_regression_and_synth_impact()
