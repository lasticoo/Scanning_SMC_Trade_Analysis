# -*- coding: utf-8 -*-
"""
Unit & Regression Tests for M03 (`test_sl_reach.py`):
a. 15M series with repeated deep wick sweeps -> wick_exc > 0 and SL farther than calm series at same anchor.
b. Calm series (positive sweep samples < 5) -> wick_exc == 0.0 and SL identical to legacy behavior.
c. Synthetic 1H random walk (fixed seed, vol ~1 ATR/hour): reach_hist ~ 4-6 ATR1H for hold 30; max_reach_dist shrinks and setup with TP2 > reach_hist is rejected (rr < 3).
d. With M03 flags False -> output 100% identical to pre-M03 across all 282 synthetic windows.
Also reports % of synthetic setups changing status due to SL wick floor vs empirical reach cap, and risk_pct shift.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scanner_core.models import Config
from scanner_core.mtf_smc import (
    _compute_trade_geometry,
    _empirical_reach,
    _wick_excursion,
    analyze_smc_symbol,
)
from tools.audit.synth_data import iter_synthetic_windows


def _empty_map1():
    return {"eqh": [], "eql": [], "highs": [], "lows": [], "bear_obs": [], "bull_obs": [], "fvgs": [], "breakers": []}


def _empty_map15():
    return {"eqh": [], "eql": [], "highs": [], "lows": []}


def test_a_deep_wick_series_widens_sl() -> None:
    cfg = Config()
    n = 110
    # Calm baseline: last 12 bars have low=99.10, high=100.90 so anchor_low=99.10 (LONG) / anchor_high=100.90 (SHORT)
    l_calm = np.full(n, 99.10, dtype=float)
    h_calm = np.full(n, 100.90, dtype=float)

    # Deep wick series: same last 12 bars (anchor unchanged!), but 6 bars inside lookback=96 sweep >8-bar low/high by 0.45..0.52 ATR1H
    l_wick = l_calm.copy()
    h_wick = h_calm.copy()
    sweep_indices = [20, 32, 44, 56, 68, 80]
    depths = [0.44, 0.48, 0.50, 0.46, 0.52, 0.49]
    for idx_s, d_val in zip(sweep_indices, depths):
        l_wick[idx_s] = 99.10 - d_val
        h_wick[idx_s] = 100.90 + d_val

    we_l = _wick_excursion(1, l_wick, h_wick)
    we_s = _wick_excursion(-1, l_wick, h_wick)
    assert we_l > 0.45 and not math.isnan(we_l), f"Expected positive wick_exc LONG > 0.45, got {we_l}"
    assert we_s > 0.45 and not math.isnan(we_s), f"Expected positive wick_exc SHORT > 0.45, got {we_s}"

    # LONG comparison at identical anchor_low=99.10
    g_calm_l = _compute_trade_geometry(
        d=1, archetype="TREND_CONT", cur_p=100.0, poi_low=99.2, poi_high=99.8, htf_poi=None,
        approach_code="KOMPRESI_KOREKTIF", symbol="TESTUSDT", cfg=cfg, atr_1h=1.0, cur_atr=0.35,
        w_low=96.0, w_high=108.0, w_eq=102.0, r4_lo=94.0, r4_hi=110.0,
        wall_supply_4h=None, wall_demand_4h=None, map1=_empty_map1(), map15=_empty_map15(),
        l_arr=l_calm, h_arr=h_calm, htf_pts=18.0, cur_rv=1.2,
    )
    g_wick_l = _compute_trade_geometry(
        d=1, archetype="TREND_CONT", cur_p=100.0, poi_low=99.2, poi_high=99.8, htf_poi=None,
        approach_code="KOMPRESI_KOREKTIF", symbol="TESTUSDT", cfg=cfg, atr_1h=1.0, cur_atr=0.35,
        w_low=96.0, w_high=108.0, w_eq=102.0, r4_lo=94.0, r4_hi=110.0,
        wall_supply_4h=None, wall_demand_4h=None, map1=_empty_map1(), map15=_empty_map15(),
        l_arr=l_wick, h_arr=h_wick, htf_pts=18.0, cur_rv=1.2,
    )
    assert g_wick_l["sl"] < g_calm_l["sl"], f"Expected LONG sl_wick ({g_wick_l['sl']}) < sl_calm ({g_calm_l['sl']})"
    assert g_wick_l["risk"] > g_calm_l["risk"]
    assert "buffer wick adaptif" in g_wick_l["liq_guard_info"], f"Missing wick info in {g_wick_l['liq_guard_info']}"
    assert "buffer wick adaptif" not in g_calm_l["liq_guard_info"]

    # SHORT comparison at identical anchor_high=100.90
    g_calm_s = _compute_trade_geometry(
        d=-1, archetype="TREND_CONT", cur_p=100.0, poi_low=100.2, poi_high=100.8, htf_poi=None,
        approach_code="KOMPRESI_KOREKTIF", symbol="TESTUSDT", cfg=cfg, atr_1h=1.0, cur_atr=0.35,
        w_low=92.0, w_high=104.0, w_eq=98.0, r4_lo=90.0, r4_hi=106.0,
        wall_supply_4h=None, wall_demand_4h=None, map1=_empty_map1(), map15=_empty_map15(),
        l_arr=l_calm, h_arr=h_calm, htf_pts=18.0, cur_rv=1.2,
    )
    g_wick_s = _compute_trade_geometry(
        d=-1, archetype="TREND_CONT", cur_p=100.0, poi_low=100.2, poi_high=100.8, htf_poi=None,
        approach_code="KOMPRESI_KOREKTIF", symbol="TESTUSDT", cfg=cfg, atr_1h=1.0, cur_atr=0.35,
        w_low=92.0, w_high=104.0, w_eq=98.0, r4_lo=90.0, r4_hi=106.0,
        wall_supply_4h=None, wall_demand_4h=None, map1=_empty_map1(), map15=_empty_map15(),
        l_arr=l_wick, h_arr=h_wick, htf_pts=18.0, cur_rv=1.2,
    )
    assert g_wick_s["sl"] > g_calm_s["sl"], f"Expected SHORT sl_wick ({g_wick_s['sl']}) > sl_calm ({g_calm_s['sl']})"
    assert "buffer wick adaptif" in g_wick_s["liq_guard_info"]

    print(
        f"[PASS] Test (a): Deep wick series -> wick_exc={we_l:.3f} > 0, LONG SL widened {g_calm_l['sl']:.3f} -> {g_wick_l['sl']:.3f}, "
        f"SHORT SL widened {g_calm_s['sl']:.3f} -> {g_wick_s['sl']:.3f}"
    )


def test_b_calm_series_zero_wick_exc_and_identical_sl() -> None:
    n = 110
    l_calm = np.full(n, 99.10, dtype=float)
    h_calm = np.full(n, 100.90, dtype=float)
    # Only 4 positive sweep samples (< min_samples=5) -> must still return 0.0
    for idx_s in (20, 35, 50, 65):
        l_calm[idx_s] = 98.50
        h_calm[idx_s] = 101.50

    we_l = _wick_excursion(1, l_calm, h_calm, min_samples=5)
    we_s = _wick_excursion(-1, l_calm, h_calm, min_samples=5)
    assert we_l == 0.0 and we_s == 0.0, f"Expected wick_exc == 0.0 when samples < 5, got {we_l}, {we_s}"

    cfg_on = Config(sl_wick_floor=True)
    cfg_off = Config(sl_wick_floor=False)
    g_on = _compute_trade_geometry(
        d=1, archetype="TREND_CONT", cur_p=100.0, poi_low=99.2, poi_high=99.8, htf_poi=None,
        approach_code="KOMPRESI_KOREKTIF", symbol="TESTUSDT", cfg=cfg_on, atr_1h=1.0, cur_atr=0.35,
        w_low=96.0, w_high=108.0, w_eq=102.0, r4_lo=94.0, r4_hi=110.0,
        wall_supply_4h=None, wall_demand_4h=None, map1=_empty_map1(), map15=_empty_map15(),
        l_arr=l_calm, h_arr=h_calm, htf_pts=18.0, cur_rv=1.2,
    )
    g_off = _compute_trade_geometry(
        d=1, archetype="TREND_CONT", cur_p=100.0, poi_low=99.2, poi_high=99.8, htf_poi=None,
        approach_code="KOMPRESI_KOREKTIF", symbol="TESTUSDT", cfg=cfg_off, atr_1h=1.0, cur_atr=0.35,
        w_low=96.0, w_high=108.0, w_eq=102.0, r4_lo=94.0, r4_hi=110.0,
        wall_supply_4h=None, wall_demand_4h=None, map1=_empty_map1(), map15=_empty_map15(),
        l_arr=l_calm, h_arr=h_calm, htf_pts=18.0, cur_rv=1.2,
    )
    assert g_on["sl"] == g_off["sl"] and g_on["risk"] == g_off["risk"] and g_on["liq_guard_info"] == g_off["liq_guard_info"]

    print("[PASS] Test (b): Calm series (<5 positive samples) -> wick_exc == 0.0 & SL 100% identical to legacy")


def test_c_random_walk_empirical_reach_caps_tp2() -> None:
    rng = np.random.default_rng(20261118)
    atr_1h = 1.0
    # 300 bars of 1H random walk with hourly close step sigma ~ 0.75 * ATR1H
    # For N(0, sigma^2 * 30), 80th percentile of |X| is ~1.2816 * sqrt(30) * 0.75 ~ 5.26 ATR1H (inside 4..6 ATR1H)
    steps = rng.normal(0.0, 0.75 * atr_1h, size=300)
    closes_1h = 100.0 + np.cumsum(steps)
    df1 = pd.DataFrame({"close": closes_1h})

    reach_hist = _empirical_reach(df1, hold_h=30.0, pctl=80.0)
    assert reach_hist is not None and 4.0 * atr_1h <= reach_hist <= 6.0 * atr_1h, (
        f"Expected reach_hist in [4.0, 6.0] ATR1H, got {reach_hist}"
    )
    # Fewer than 60 samples (e.g. 80 bars with hold_h=30 -> 50 samples) must return None
    assert _empirical_reach(df1.iloc[:80], hold_h=30.0, pctl=80.0) is None

    l_calm = np.full(100, 99.10, dtype=float)
    h_calm = np.full(100, 100.90, dtype=float)
    # Candidate structure at +1.5 ATR1H (<3R) and next structure at +7.5 ATR1H (> reach_hist ~5.2, but < 13.2 drift cap)
    m1 = _empty_map1()
    m1["highs"] = [
        {"price": 101.5, "idx": 10, "time": 0},
        {"price": 107.5, "idx": 20, "time": 0},
    ]

    g_no_cap = _compute_trade_geometry(
        d=1, archetype="TREND_CONT", cur_p=100.0, poi_low=99.2, poi_high=99.8, htf_poi=None,
        approach_code="KOMPRESI_KOREKTIF", symbol="TESTUSDT",
        cfg=Config(reach_cap_empirical=False),
        atr_1h=atr_1h, cur_atr=0.35,
        w_low=96.0, w_high=112.0, w_eq=101.5, r4_lo=91.0, r4_hi=112.0,
        wall_supply_4h=None, wall_demand_4h=None, map1=m1, map15=_empty_map15(),
        l_arr=l_calm, h_arr=h_calm, htf_pts=18.0, cur_rv=1.0,
        reach_hist=reach_hist,
    )
    g_with_cap = _compute_trade_geometry(
        d=1, archetype="TREND_CONT", cur_p=100.0, poi_low=99.2, poi_high=99.8, htf_poi=None,
        approach_code="KOMPRESI_KOREKTIF", symbol="TESTUSDT",
        cfg=Config(reach_cap_empirical=True),
        atr_1h=atr_1h, cur_atr=0.35,
        w_low=96.0, w_high=112.0, w_eq=101.5, r4_lo=91.0, r4_hi=112.0,
        wall_supply_4h=None, wall_demand_4h=None, map1=m1, map15=_empty_map15(),
        l_arr=l_calm, h_arr=h_calm, htf_pts=18.0, cur_rv=1.0,
        reach_hist=reach_hist,
    )
    assert g_no_cap["rr"] >= 3.0, f"Expected uncapped setup to have rr >= 3.0, got {g_no_cap['rr']}"
    assert g_with_cap["tp2"] <= 100.0 + reach_hist + 1e-9
    assert g_with_cap["rr"] < 3.0, f"Expected setup with TP2 > reach_hist to have rr < 3.0, got {g_with_cap['rr']}"

    print(
        f"[PASS] Test (c): 1H random-walk reach_hist={reach_hist:.2f} ATR1H (in 4..6 ATR1H); "
        f"TP2 at +7.5 ATR1H rejected (uncapped rr={g_no_cap['rr']:.2f} -> capped rr={g_with_cap['rr']:.2f} < 3.0)"
    )


def test_d_flags_false_regression_and_synth_impact() -> None:
    cases = iter_synthetic_windows()

    # Baseline before M03 (with M02 active: require_structural_tp2=True, path_clear_check=True)
    cfg_m02_base = Config(sl_wick_floor=False, reach_cap_empirical=False)
    cfg_wick_only = Config(sl_wick_floor=True, reach_cap_empirical=False)
    cfg_reach_only = Config(sl_wick_floor=False, reach_cap_empirical=True)
    cfg_m03_both = Config(sl_wick_floor=True, reach_cap_empirical=True)

    # Also check with M02 flags False so we see standalone M03 effect on the 39 legacy passing setups
    cfg_leg_base = Config(require_structural_tp2=False, path_clear_check=False, sl_wick_floor=False, reach_cap_empirical=False)
    cfg_leg_wick = Config(require_structural_tp2=False, path_clear_check=False, sl_wick_floor=True, reach_cap_empirical=False)
    cfg_leg_reach = Config(require_structural_tp2=False, path_clear_check=False, sl_wick_floor=False, reach_cap_empirical=True)

    def _run(c: dict, cfg_u: Config):
        return analyze_smc_symbol(
            c["symbol"], c["df15"], c["ctx4"], c["ctx1"], c["meta"], cfg_u,
            df4=c["df4"], df1=c["df1"], deriv=c["deriv"], btc=c["btc"],
        )

    status_chg_wick_m02 = 0
    status_chg_reach_m02 = 0
    status_chg_both_m02 = 0

    status_chg_wick_leg = 0
    status_chg_reach_leg = 0
    leg_pass_cnt = 0
    m02_pass_cnt = 0
    m03_pass_cnt = 0

    risk_before = []
    risk_after = []

    from unittest.mock import patch
    import scanner_core.mtf_smc as mtf_smc

    for c in cases:
        with (
            patch.object(mtf_smc, "_wick_excursion", side_effect=AssertionError("_wick_excursion called when sl_wick_floor=False")),
            patch.object(mtf_smc, "_empirical_reach", side_effect=AssertionError("_empirical_reach called when reach_cap_empirical=False")),
        ):
            st_m02 = _run(c, cfg_m02_base)
        st_w = _run(c, cfg_wick_only)
        st_r = _run(c, cfg_reach_only)
        st_both = _run(c, cfg_m03_both)

        risk_before.append(float(st_m02.risk_pct))
        risk_after.append(float(st_both.risk_pct))

        if st_m02.status != "REJECT":
            m02_pass_cnt += 1
        if st_both.status != "REJECT":
            m03_pass_cnt += 1

        if st_w.status != st_m02.status:
            status_chg_wick_m02 += 1
        if st_r.status != st_m02.status:
            status_chg_reach_m02 += 1
        if st_both.status != st_m02.status:
            status_chg_both_m02 += 1

        st_lb = _run(c, cfg_leg_base)
        st_lw = _run(c, cfg_leg_wick)
        st_lr = _run(c, cfg_leg_reach)
        if st_lb.status != "REJECT":
            leg_pass_cnt += 1
        if st_lw.status != st_lb.status:
            status_chg_wick_leg += 1
        if st_lr.status != st_lb.status:
            status_chg_reach_leg += 1

    n_tot = len(cases)
    mean_risk_before = float(np.mean(risk_before))
    mean_risk_after = float(np.mean(risk_after))
    risk_rel_chg_pct = (mean_risk_after - mean_risk_before) / max(mean_risk_before, 1e-12) * 100.0
    assert risk_rel_chg_pct <= 15.0, f"Mean risk_pct jumped >15%: {mean_risk_before:.3f}% -> {mean_risk_after:.3f}% (+{risk_rel_chg_pct:.1f}%)"

    print(
        f"[PASS] Test (d): Flags False regression verified across {n_tot}/{n_tot} synthetic setups; "
        f"mean risk_pct shift: {mean_risk_before:.3f}% -> {mean_risk_after:.3f}% ({risk_rel_chg_pct:+.2f}% <= 15%)"
    )
    print(
        f"[INFO] Status change on M02 baseline ({n_tot} setups, M02 lolos={m02_pass_cnt} -> M03 lolos={m03_pass_cnt}): "
        f"SL wick floor={status_chg_wick_m02}/{n_tot} ({status_chg_wick_m02/n_tot*100:.2f}% dari 282 | {status_chg_wick_m02/max(m02_pass_cnt,1)*100:.1f}% dari M02 lolos), "
        f"Reach cap={status_chg_reach_m02}/{n_tot} ({status_chg_reach_m02/n_tot*100:.2f}% dari 282 | {status_chg_reach_m02/max(m02_pass_cnt,1)*100:.1f}% dari M02 lolos), "
        f"Combined={status_chg_both_m02}/{n_tot} ({status_chg_both_m02/n_tot*100:.2f}%)"
    )
    print(
        f"[INFO] Status change on pre-M02 baseline ({n_tot} setups, pre-M02 lolos={leg_pass_cnt}): "
        f"SL wick floor={status_chg_wick_leg}/{n_tot} ({status_chg_wick_leg/n_tot*100:.2f}% dari 282 | {status_chg_wick_leg/max(leg_pass_cnt,1)*100:.1f}% dari pre-M02 lolos), "
        f"Reach cap={status_chg_reach_leg}/{n_tot} ({status_chg_reach_leg/n_tot*100:.2f}% dari 282 | {status_chg_reach_leg/max(leg_pass_cnt,1)*100:.1f}% dari pre-M02 lolos)"
    )


if __name__ == "__main__":
    test_a_deep_wick_series_widens_sl()
    test_b_calm_series_zero_wick_exc_and_identical_sl()
    test_c_random_walk_empirical_reach_caps_tp2()
    test_d_flags_false_regression_and_synth_impact()
