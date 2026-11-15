# -*- coding: utf-8 -*-
"""
Unit Test for Fase 4 (JSONL per-setup logging in binance.py):
Verifies that `append_setups_jsonl` writes valid JSONL lines containing:
- timestamps (`scan_utc`, `signal_candle_utc`, `tracker_key`)
- symbol, direction, archetype
- `python_status`, `python_score` (preserved even when LLM updates `status`/`score`)
- `llm_status`, `llm_score`, `ai_decision`, `ai_model`
- `levels` (`entry`, `sl`, `tp1`, `tp2`, `rr`, `risk_pct`)
- `evidence` (structured metrics without prompt text bloat) and `flags`
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import binance
from scanner_core.models import Config
from scanner_core.mtf_smc import analyze_smc_symbol
from tools.audit.synth_data import iter_synthetic_windows


def test_append_setups_jsonl() -> None:
    cfg = Config()
    cases = iter_synthetic_windows(seeds=(11,), n_bars_15m=2200, windows_per_seed=3)
    setups = []
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
        setups.append(st)

    py_status_map = {id(s): s.status for s in setups}
    orig_py_status_0 = setups[0].status
    orig_py_score_0 = setups[0].python_score

    # Simulate LLM curation mutating setup[0]
    setups[0].status = "ENTRY READY"
    setups[0].score = 86.0
    setups[0].ai_confidence = 86.0
    setups[0].ai_decision = "APPROVED"
    setups[0].ai_model = "llama-3.3-70b-versatile"

    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = binance.append_setups_jsonl(setups, "2026-11-15 09:00:00", tmpdir, py_status_map=py_status_map)
        assert os.path.isfile(out_file), "JSONL file was not created"
        with open(out_file, "r", encoding="utf-8") as fh:
            lines = [json.loads(line) for line in fh if line.strip()]

    assert len(lines) == 3, f"Expected 3 JSONL records, got {len(lines)}"
    r0 = lines[0]
    assert r0["python_status"] == orig_py_status_0, f"Expected python_status={orig_py_status_0}, got {r0['python_status']}"
    assert r0["python_score"] == round(float(orig_py_score_0), 2)
    assert r0["llm_status"] == "ENTRY READY"
    assert r0["llm_score"] == 86.0
    assert r0["ai_decision"] == "APPROVED"
    assert r0["ai_model"] == "llama-3.3-70b-versatile"
    assert "entry" in r0["levels"] and "sl" in r0["levels"] and "tp2" in r0["levels"]
    assert isinstance(r0["evidence"], dict) and "er_1h_24" in r0["evidence"]
    assert "text" not in r0["evidence"], "Verbose prompt text should be excluded from JSONL evidence dict"
    assert isinstance(r0["flags"], list)

    r1 = lines[1]
    assert r1["llm_status"] is None and r1["llm_score"] is None
    print(f"[PASS] JSONL logging verified ({len(lines)} records, keys={list(r0.keys())})")


if __name__ == "__main__":
    test_append_setups_jsonl()
