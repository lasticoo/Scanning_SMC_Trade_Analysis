# -*- coding: utf-8 -*-
"""
Unit & Synthetic Benchmark Tests for Tugas B (`_sweep_is_stale` in `scanner_core/evidence.py`):
1. Four required unit test cases on `_sweep_is_stale`:
   - (1H basi, 15M segar) -> False
   - (hanya 1H ada dan basi) -> True
   - (1H dan 15M keduanya basi) -> True
   - (tidak ada sweep) -> False
2. Synthetic 282-window frequency comparison (old OR >60 vs new `_sweep_is_stale(stale_1h_bars=60, stale_15m_bars=96)`).
"""
from __future__ import annotations

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scanner_core.evidence import _sweep_is_stale
from scanner_core.models import Config
from scanner_core.mtf_smc import analyze_smc_symbol
from tools.audit.synth_data import iter_synthetic_windows


def test_unit_cases() -> None:
    # Kasus 1: (1H basi, 15M segar) -> False
    sw1_stale = {"has_sweep": True, "sweep_age_bars": 85, "sweep_level": 100.0, "followed_by_mss_bos": True}
    sw15_fresh = {"has_sweep": True, "sweep_age_bars": 12, "sweep_level": 99.5, "followed_by_mss_bos": False}
    assert _sweep_is_stale(sw1_stale, sw15_fresh) is False, "Expected False when 1H is stale but 15M is fresh"
    print("[PASS] Case 1: (1H basi [85b > 60], 15M segar [12b <= 96]) -> False")

    # Kasus 2: (hanya 1H ada dan basi) -> True
    sw15_none = {"has_sweep": False, "sweep_age_bars": None, "sweep_level": None, "followed_by_mss_bos": False}
    assert _sweep_is_stale(sw1_stale, sw15_none) is True, "Expected True when only 1H exists and is stale"
    print("[PASS] Case 2: (hanya 1H ada dan basi [85b > 60], 15M tidak ada) -> True")

    # Kasus 3: (1H dan 15M keduanya basi) -> True
    sw15_stale = {"has_sweep": True, "sweep_age_bars": 110, "sweep_level": 98.0, "followed_by_mss_bos": True}
    assert _sweep_is_stale(sw1_stale, sw15_stale) is True, "Expected True when both 1H and 15M are stale"
    print("[PASS] Case 3: (1H [85b > 60] dan 15M [110b > 96] keduanya basi) -> True")

    # Kasus 4: (tidak ada sweep) -> False
    sw1_none = {"has_sweep": False, "sweep_age_bars": None, "sweep_level": None, "followed_by_mss_bos": False}
    assert _sweep_is_stale(sw1_none, sw15_none) is False, "Expected False when neither 1H nor 15M has a sweep"
    print("[PASS] Case 4: (tidak ada sweep di 1H maupun 15M) -> False")


def test_synthetic_frequency_and_errors() -> None:
    cfg = Config()
    cases = iter_synthetic_windows()
    old_cnt = 0
    new_cnt = 0
    ev_errors = 0

    for c in cases:
        st = analyze_smc_symbol(
            c["symbol"],
            c["df15"],
            c["ctx4"],
            c["ctx1"],
            c["meta"],
            cfg,
            df4=c["df4"],
            df1=c["df1"],
            deriv=c["deriv"],
            btc=c["btc"],
        )
        assert st is not None
        ev = st.raw_market_data.get("evidence") or {}
        if ev.get("error") is not None:
            ev_errors += 1
        sw1 = ev.get("sweep_1h") or {}
        sw15 = ev.get("sweep_15m") or {}
        old_cond = (bool(sw1.get("has_sweep")) and int(sw1.get("sweep_age_bars") or 0) > 60) or (
            bool(sw15.get("has_sweep")) and int(sw15.get("sweep_age_bars") or 0) > 60
        )
        if old_cond:
            old_cnt += 1
        if "SWEEP_STALE" in (ev.get("flags") or []):
            new_cnt += 1

    assert ev_errors == 0, f"Unexpected evidence errors: {ev_errors}"
    print(
        f"[PASS] Synthetic 282 setups: SWEEP_STALE before={old_cnt}/{len(cases)} -> after={new_cnt}/{len(cases)} "
        f"(evidence errors={ev_errors})"
    )


if __name__ == "__main__":
    test_unit_cases()
    test_synthetic_frequency_and_errors()
