# -*- coding: utf-8 -*-
"""
Unit & Synthetic Tests for Phase 1:
1.1 MSS key fix (sweep_low/impulse_high, sweep_high/impulse_low)
1.2 Off-by-one mitigation up to bar n-1 (range(..., n) / slice [..., :n])
1.3 _count_visits(inside: np.ndarray) -> int (contiguous visit episodes = 1 touch)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scanner_core.mtf_smc import _count_visits, extract_tf_structure


def _make_base_df(n: int = 80, base: float = 100.0) -> pd.DataFrame:
    times = 1700000000000 + np.arange(n, dtype=np.int64) * 900_000
    o = np.full(n, base, dtype=float)
    c = np.full(n, base + 0.1, dtype=float)
    h = np.full(n, base + 0.5, dtype=float)
    l = np.full(n, base - 0.5, dtype=float)
    v = np.full(n, 1000.0, dtype=float)
    # Create gentle oscillation before bar 50
    for i in range(50):
        off = np.sin(i * 0.4) * 1.5
        o[i] = base + off
        c[i] = base + off + (0.2 if i % 2 == 0 else -0.2)
        h[i] = max(o[i], c[i]) + 0.4
        l[i] = min(o[i], c[i]) - 0.4
    return pd.DataFrame({
        "open_time": times,
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": v,
        "close_time": times + 899_999,
        "quote_volume": v * base,
        "trades": np.full(n, 100, dtype=int),
        "taker_buy_base": v * 0.5,
        "taker_buy_quote": v * base * 0.5,
    })


def test_count_visits_helper() -> None:
    assert _count_visits(np.array([], dtype=bool)) == 0
    assert _count_visits(np.array([False, False, False])) == 0
    assert _count_visits(np.array([True, True, True, True])) == 1
    assert _count_visits(np.array([False, True, True, True, False])) == 1
    assert _count_visits(np.array([True, False, True, True, False, True])) == 3
    print("[PASS] _count_visits helper unit tests")


def test_ob_last_bar_tap_and_multi_bar_single_visit() -> None:
    # Build synthetic series with a clear Bullish OB at bar 55:
    # bar 55: bearish candle [low=98.0, high=99.0, open=98.8, close=98.2]
    # bar 56-57: strong bullish impulse + FVG pushing up to 105.0
    df = _make_base_df(80, base=100.0)
    # Stabilize 50..54 around 99.5
    for k in range(50, 55):
        df.loc[k, ["open", "high", "low", "close"]] = [99.5, 99.8, 99.2, 99.4]
    # Bar 55: Bullish OB candle (bearish body, sweeps prior low 99.2 down to 98.0)
    df.loc[55, ["open", "high", "low", "close"]] = [99.0, 99.1, 98.0, 98.2]
    # Bar 56: Strong displacement candle up
    df.loc[56, ["open", "high", "low", "close"]] = [98.2, 102.5, 98.2, 102.3]
    # Bar 57: Follow-through leaving Bullish FVG (low 57 > high 55)
    df.loc[57, ["open", "high", "low", "close"]] = [102.3, 104.5, 101.0, 104.2]
    # Bars 58..78: price stays high above OB (around 104..105), untouched
    for k in range(58, 79):
        df.loc[k, ["open", "high", "low", "close"]] = [104.0, 104.8, 103.5, 104.2]

    # Case A: Last bar (bar 79, index n-1) taps down into the OB [98.0 - 99.1]
    df_tap_last = df.copy()
    df_tap_last.loc[79, ["open", "high", "low", "close"]] = [103.8, 104.0, 98.7, 101.5]
    res_last = extract_tf_structure(df_tap_last, piv_len=2, tf_label="15M", lookback_bars=60)
    assert res_last is not None
    ob_55 = [z for z in res_last["bull_obs"] if z["idx"] == 55]
    assert len(ob_55) == 1, f"Expected OB at idx 55, got {res_last['bull_obs']}"
    assert ob_55[0]["touches"] == 1, f"Expected touches=1 on last-bar tap, got {ob_55[0]['touches']}"
    assert ob_55[0]["fresh"] is False, f"Expected fresh=False on last-bar tap, got {ob_55[0]['fresh']}"
    print("[PASS] OB tap on last closed bar (n-1) recorded as touches=1, fresh=False")

    # Case B: Multi-bar single visit (4 consecutive bars 65..68 inside OB [98.0 - 99.1], then bounces back to 104)
    df_visit = df.copy()
    df_visit.loc[79, ["open", "high", "low", "close"]] = [104.0, 104.6, 103.8, 104.3]
    for k in range(65, 69):  # 4 consecutive bars touching/reacting inside OB
        df_visit.loc[k, ["open", "high", "low", "close"]] = [99.0, 99.5, 98.4, 99.1]
    res_visit = extract_tf_structure(df_visit, piv_len=2, tf_label="15M", lookback_bars=60)
    assert res_visit is not None
    ob_55_v = [z for z in res_visit["bull_obs"] if z["idx"] == 55]
    assert len(ob_55_v) == 1, "OB with 4-bar single visit must remain in bull_obs (not discarded as touches=4)"
    assert ob_55_v[0]["touches"] == 1, f"Expected touches=1 for 4 consecutive bars in 1 visit, got {ob_55_v[0]['touches']}"
    assert ob_55_v[0]["fresh"] is False
    print("[PASS] OB with 4 consecutive bars in single visit kept with touches=1, fresh=False")


if __name__ == "__main__":
    test_count_visits_helper()
    test_ob_last_bar_tap_and_multi_bar_single_visit()

