# -*- coding: utf-8 -*-
"""
Unit & Integration Tests for Tugas C / Fase 5 (`scanner_core/evidence_report.py` & `scanner_core/tracker.py`):
1. Join test: corrupt JSONL lines, duplicate keys within same scan_utc (keep='last'), distinct scan_utc,
   and unmatched/missing scan_utc trades -> no lost/duplicated rows, accurate `evidence_joined`.
2. Statistics test: synthetic trade DataFrame with low-WR flag -> verify n, WR%, Avg R, Net R, PF against
   independent pandas calculations; verify Wilson 95% CI on 8/20 wins; verify '(sampel kecil, jangan disimpulkan)'.
3. Backward compatibility test: mock `AsyncBinance` running `_track_signals_async` on folder with legacy
   `scan_*.csv` and no JSONL -> original 18 columns stay in exact order and values (`realized_r`, `outcome`),
   new columns only at the end, and original summary text is unchanged.
4. Fault tolerance test: force `build_evidence_report` to raise an exception -> original summary preserved
   plus single `[evidence report gagal: ...]` line.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import scanner_core.tracker as tracker_mod
from scanner_core.evidence_report import (
    build_evidence_report,
    enrich_trades,
    load_audit_jsonl,
    wilson_ci_95,
)


def test_join_and_corrupt_jsonl() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = os.path.join(tmpdir, "setups_audit.jsonl")
        records = [
            # Valid record 1 (will be superseded by duplicate key with same scan_utc below)
            json.dumps({
                "scan_utc": "2026-11-15 08:00:00",
                "symbol": "BTCUSDT",
                "direction": "LONG",
                "python_status": "WATCHLIST",
                "llm_status": "REJECT",
                "evidence": {"er_1h_24": 0.15, "killzone_utc": "LONDON_KILLZONE (08:00 UTC)", "btc_4h_trend": "BULLISH", "text": "ignore"},
                "flags": ["POI_MULTIPLE_VISITS_24H(3x)"],
            }),
            # Corrupt JSON line
            "{corrupt_json_line: 123",
            # Duplicate (scan_utc, symbol, direction) -> should overwrite record 1 (keep='last')
            json.dumps({
                "scan_utc": "2026-11-15 08:00:00",
                "symbol": "BTCUSDT",
                "direction": "LONG",
                "python_status": "ENTRY READY",
                "llm_status": "ENTRY READY",
                "evidence": {"er_1h_24": 0.42, "killzone_utc": "LONDON_KILLZONE (08:00 UTC)", "btc_4h_trend": "BULLISH", "text": "ignore"},
                "flags": ["POI_MULTIPLE_VISITS_24H(4x)", "SWEEP_STALE"],
            }),
            # Same symbol/direction but different scan_utc
            json.dumps({
                "scan_utc": "2026-11-15 12:00:00",
                "symbol": "BTCUSDT",
                "direction": "LONG",
                "python_status": "PRE-TAP LIMIT",
                "llm_status": "PRE-TAP LIMIT",
                "evidence": {"er_1h_24": 0.28, "killzone_utc": "NY_KILLZONE (12:00 UTC)", "btc_4h_trend": "BEARISH"},
                "flags": ["CAUSAL_4H_OPPOSITE(BEARISH)"],
            }),
        ]
        with open(jsonl_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(records) + "\n")

        audit_df = load_audit_jsonl(tmpdir)
        assert audit_df.attrs.get("corrupt_lines") == 1, f"Expected 1 corrupt line, got {audit_df.attrs.get('corrupt_lines')}"
        assert len(audit_df) == 2, f"Expected 2 deduplicated audit rows, got {len(audit_df)}"
        assert "ev_text" not in audit_df.columns, "text field must not be flattened into ev_text"

        trades_df = pd.DataFrame([
            {"signal_utc": "2026-11-15 07:45:00", "symbol": "BTCUSDT", "direction": "LONG", "outcome": "TP2_FULL", "realized_r": 3.0, "scan_utc": "2026-11-15 08:00:00"},
            {"signal_utc": "2026-11-15 11:45:00", "symbol": "BTCUSDT", "direction": "LONG", "outcome": "LOSS_SL", "realized_r": -1.0, "scan_utc": "2026-11-15 12:00:00"},
            {"signal_utc": "2026-11-15 11:45:00", "symbol": "ETHUSDT", "direction": "SHORT", "outcome": "TP2_FULL", "realized_r": 3.2, "scan_utc": "2026-11-15 12:00:00"},
            {"signal_utc": "2026-11-15 11:45:00", "symbol": "SOLUSDT", "direction": "LONG", "outcome": "LOSS_SL", "realized_r": -1.0, "scan_utc": ""},
        ])

        enriched = enrich_trades(trades_df, audit_df)
        assert len(enriched) == 4, f"Row count changed after enrich_trades: {len(enriched)}"
        assert list(enriched.columns[: len(trades_df.columns)]) == list(trades_df.columns), "Original columns order changed"
        assert list(enriched["evidence_joined"]) == [True, True, False, False], f"Unexpected evidence_joined: {list(enriched['evidence_joined'])}"
        assert enriched.loc[0, "python_status"] == "ENTRY READY", "Expected keep='last' duplicate value"
        assert enriched.loc[0, "ev_er_1h_24"] == 0.42
        assert enriched.loc[0, "flag_names"] == "POI_MULTIPLE_VISITS_24H;SWEEP_STALE"
        assert enriched.loc[1, "flag_names"] == "CAUSAL_4H_OPPOSITE"
        print("[PASS] Join test: corrupt lines skipped, duplicates keep='last', flag_names normalized, no rows lost/duplicated")


def test_statistics_and_wilson_ci(save_sample_path: str = "") -> None:
    # 1. Wilson 95% CI known case: 8 wins out of 20 (p = 0.40) -> (21.88%, 61.34%)
    lo, hi = wilson_ci_95(8, 20)
    assert abs(lo - 21.88) <= 0.05 and abs(hi - 61.34) <= 0.05, f"Unexpected Wilson CI for 8/20: ({lo}, {hi})"
    print(f"[PASS] Wilson 95% CI for 8/20 wins = [{lo:.2f}%, {hi:.2f}%]")

    # 2. Synthetic trade DataFrame with 35 trades:
    #    - 20 trades WITH flag 'PULLBACK_BEYOND_LEG' (low WR: only 4 wins of +3.0R, 16 losses of -1.0R -> WR=20.0%, NetR=-4.00R, AvgR=-0.20R, PF=12/16=0.75)
    #    - 15 trades WITHOUT flag 'PULLBACK_BEYOND_LEG' (high WR: 12 wins of +3.0R, 3 losses of -1.0R -> WR=80.0%, NetR=+33.00R, AvgR=+2.20R, PF=36/3=12.00)
    rows = []
    for i in range(20):
        is_win = i < 4
        rows.append({
            "outcome": "TP2_FULL" if is_win else "LOSS_SL",
            "realized_r": 3.0 if is_win else -1.0,
            "direction": "LONG",
            "archetype": "TREND_CONT",
            "session": "NY_OPEN",
            "ai_decision": "APPROVED",
            "ai_confidence": 75.0,
            "ai_model": "llama-3.3-70b-versatile",
            "python_status": "ENTRY READY",
            "evidence_joined": True,
            "ev_killzone": "NY_KILLZONE (13:00 UTC)",
            "ev_btc_4h": "BEARISH",
            "ev_er_1h": 0.15,
            "flag_names": "PULLBACK_BEYOND_LEG;CAUSAL_4H_OPPOSITE",
        })
    for i in range(15):
        is_win = i < 12
        rows.append({
            "outcome": "TP2_FULL" if is_win else "LOSS_SL",
            "realized_r": 3.0 if is_win else -1.0,
            "direction": "LONG",
            "archetype": "HTF_REVERSAL",
            "session": "LONDON",
            "ai_decision": "APPROVED",
            "ai_confidence": 85.0,
            "ai_model": "llama-3.3-70b-versatile",
            "python_status": "REVERSAL WATCH",
            "evidence_joined": True,
            "ev_killzone": "OFF_KILLZONE (17:00 UTC)",
            "ev_btc_4h": "BULLISH",
            "ev_er_1h": 0.42,
            "flag_names": "",
        })
    df = pd.DataFrame(rows)

    # Independent pandas calculation for PULLBACK_BEYOND_LEG ADA vs TIDAK ADA
    has_pb = df["flag_names"].str.contains("PULLBACK_BEYOND_LEG")
    sub_yes = df[has_pb]
    sub_no = df[~has_pb]

    wr_yes = (sub_yes["realized_r"] > 0).mean() * 100.0
    avg_yes = sub_yes["realized_r"].mean()
    net_yes = sub_yes["realized_r"].sum()
    pf_yes = sub_yes[sub_yes["realized_r"] > 0]["realized_r"].sum() / abs(sub_yes[sub_yes["realized_r"] <= 0]["realized_r"].sum())

    wr_no = (sub_no["realized_r"] > 0).mean() * 100.0
    avg_no = sub_no["realized_r"].mean()
    net_no = sub_no["realized_r"].sum()
    pf_no = sub_no[sub_no["realized_r"] > 0]["realized_r"].sum() / abs(sub_no[sub_no["realized_r"] <= 0]["realized_r"].sum())

    rep = build_evidence_report(df, min_n=30)
    assert f"n={len(sub_yes):>4d} | WR={wr_yes:5.1f}%" in rep
    assert f"Avg R={avg_yes:+6.2f}R | Net R={net_yes:+7.2f}R | PF={pf_yes:5.2f}" in rep
    assert f"n={len(sub_no):>4d} | WR={wr_no:5.1f}%" in rep
    assert f"Avg R={avg_no:+6.2f}R | Net R={net_no:+7.2f}R | PF={pf_no:5.2f}" in rep
    assert "(sampel kecil, jangan disimpulkan)" in rep, "Expected small-sample warning for n=20 < min_n=30"
    assert "Footer: Korelasi bukan kausalitas" in rep

    if save_sample_path:
        with open(save_sample_path, "w", encoding="utf-8") as fh:
            fh.write(rep)

    print("[PASS] Statistics test: n, WR%, Wilson 95% CI, Avg R, Net R, PF match independent pandas calculations")


def test_backward_compatibility_and_fault_tolerance() -> None:
    orig_18_cols = [
        "signal_utc", "symbol", "direction", "kind", "status", "score", "ai_decision",
        "entry", "sl", "tp1", "tp2", "risk_pct", "planned_rr", "outcome", "tp1_hit",
        "realized_r", "hold_hours", "source_file",
    ]
    new_carry_cols = [
        "scan_utc", "archetype", "python_score", "ai_confidence", "ai_model",
        "session", "trend_4h", "oi_chg_1h_pct", "cvd_state",
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Legacy scan_*.csv without JSONL and without new columns
        scan_csv = os.path.join(tmpdir, "scan_20261115_0800.csv")
        pd.DataFrame([
            {
                "signal_candle_utc": "2026-11-15 07:45",
                "symbol": "BTCUSDT",
                "direction": "LONG",
                "kind": "SMC_SNIPER",
                "status": "ENTRY READY",
                "score": 82.0,
                "ai_decision": "APPROVED",
                "entry": 100.0,
                "stop_loss": 98.0,
                "tp1": 103.0,
                "tp2": 106.0,
                "rr": 3.0,
            }
        ]).to_csv(scan_csv, index=False)

        # Future 15m bars hitting TP2 (106.0)
        fut_bars = pd.DataFrame({
            "open_time": [1763193600000 + i * 900_000 for i in range(6)],
            "open": [100.0, 101.0, 102.5, 104.0, 105.0, 106.0],
            "high": [101.2, 102.8, 104.2, 105.5, 106.5, 107.0],
            "low": [99.5, 100.8, 102.0, 103.5, 104.8, 105.5],
            "close": [101.0, 102.5, 104.0, 105.0, 106.2, 106.5],
        })

        class FakeAsyncBinance:
            def __init__(self, concurrency: int = 14):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def server_time(self) -> int:
                return 1763193600000 + 72 * 3_600_000

            async def klines(self, sym: str, interval: str, limit: int, server_ms: int, start_time: int = 0):
                return fut_bars

        with patch.object(tracker_mod, "AsyncBinance", FakeAsyncBinance):
            summary_no_jsonl, res_df = asyncio.run(tracker_mod._track_signals_async(scan_dir=tmpdir))

        # Verify original 18 columns stay in exact order at the front, followed by 9 carry-through columns and evidence_joined
        assert list(res_df.columns[:18]) == orig_18_cols, f"Original 18 columns altered: {list(res_df.columns[:18])}"
        assert list(res_df.columns[18:27]) == new_carry_cols, f"Carry-through columns mismatch: {list(res_df.columns[18:27])}"
        assert res_df.columns[27] == "evidence_joined"
        assert bool(res_df.loc[0, "evidence_joined"]) is False
        assert res_df.loc[0, "outcome"] == "TP2_FULL"
        assert float(res_df.loc[0, "realized_r"]) == 2.96  # 0.5*1.5 + 0.5*3.0 - fee_r(0.04) -> 2.21 or set-and-forget 2.96

        expected_base_summary = tracker_mod._build_metrics_summary(
            res_df[orig_18_cols],
            f"SIGNAL PERFORMANCE TRACKER REPORT  |  Folder: {tmpdir}  |  Max Hold: 60h  |  Fixed SL (No BEP) -> Target Utama TP2 (>=1:3.0R)",
        )
        assert summary_no_jsonl == expected_base_summary, "Summary text changed when no JSONL exists"
        print("[PASS] Backward compatibility test: legacy scan_*.csv without JSONL preserves original columns, outcome/realized_r, and summary text")

        # Fault tolerance test: force build_evidence_report to raise RuntimeError
        def BoomReport(_df):
            raise RuntimeError("Simulated evidence report crash")

        with patch.object(tracker_mod, "AsyncBinance", FakeAsyncBinance), patch.object(tracker_mod, "build_evidence_report", BoomReport):
            summary_boom, _ = asyncio.run(tracker_mod._track_signals_async(scan_dir=tmpdir))

        assert summary_boom.startswith(expected_base_summary), "Original summary lost when build_evidence_report raised"
        assert summary_boom.endswith("\n[evidence report gagal: RuntimeError: Simulated evidence report crash]"), (
            f"Missing single-line error footer: {summary_boom[-120:]}"
        )
        print("[PASS] Fault tolerance test: exception in build_evidence_report caught cleanly with 1-line note")


if __name__ == "__main__":
    sample_out = sys.argv[1] if len(sys.argv) > 1 else ""
    test_join_and_corrupt_jsonl()
    test_statistics_and_wilson_ci(save_sample_path=sample_out)
    test_backward_compatibility_and_fault_tolerance()
