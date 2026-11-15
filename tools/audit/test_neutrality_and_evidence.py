# -*- coding: utf-8 -*-
"""
Unit & Integration Tests for Fase 2 (scanner_core/evidence.py & LLM Prompt Integration):
1. 100% Neutrality vs Fase 1 (status, score, entry, sl, tp1, tp2, archetype identical across all 282 windows).
2. Zero evidence errors (`evidence["error"] is None` on all 282 setups).
3. CPU overhead per setup (< 5 ms target).
4. `build_full_llm_smc_prompt` contains `[BUKTI KUALITAS TAMBAHAN DARI PYTHON` for all 3 archetypes
   (`HTF_REVERSAL`, `TREND_CONT`, `PRE_REACTION`).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ai_validator import build_full_llm_smc_prompt
from scanner_core.evidence import build_evidence
from scanner_core.models import Config
from scanner_core.mtf_smc import analyze_smc_symbol
from tools.audit.synth_data import iter_synthetic_windows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1-json", type=str, required=True, help="Path to phase1_baseline.json")
    ap.add_argument("--save-prompts", type=str, default="", help="Optional path to save 3 archetype prompts JSON")
    args = ap.parse_args()

    with open(args.phase1_json, "r", encoding="utf-8") as f:
        p1_data = json.load(f)
    p1_records = {r["case_id"]: r for r in p1_data["records"]}

    cfg = Config()
    cases = iter_synthetic_windows()

    total = 0
    identical = 0
    mismatches = []
    ev_errors = []
    ev_times_ms = []
    sample_prompts = {}

    for c in cases:
        total += 1
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
        assert st is not None, f"Unexpected None setup at case_id={c['case_id']}"

        # 1. Check neutrality vs Phase 1
        p1 = p1_records[c["case_id"]]
        cur_tuple = (
            st.status,
            round(float(st.score), 6),
            round(float(st.entry), 8),
            round(float(st.sl), 8),
            round(float(st.tp1), 8),
            round(float(st.tp2), 8),
            st.archetype,
        )
        p1_tuple = (
            p1["status"],
            p1["score"],
            p1["entry"],
            p1["sl"],
            p1["tp1"],
            p1["tp2"],
            p1["archetype"],
        )
        if cur_tuple == p1_tuple:
            identical += 1
        else:
            mismatches.append({"case_id": c["case_id"], "p1": p1_tuple, "p2": cur_tuple})

        # 2. Check evidence presence & error is None
        ev = st.raw_market_data.get("evidence") if isinstance(st.raw_market_data, dict) else None
        if not isinstance(ev, dict) or ev.get("error") is not None:
            ev_errors.append({"case_id": c["case_id"], "ev": ev})

        # 3. Measure isolated build_evidence CPU time
        t0 = time.perf_counter()
        _ = build_evidence(st, c["df4"], c["df1"], c["df15"], c["btc"])
        ev_times_ms.append((time.perf_counter() - t0) * 1000.0)

        # 4. Capture prompt per archetype
        if st.archetype not in sample_prompts:
            prompt_str = build_full_llm_smc_prompt(st, btc_ctx=c["btc"])
            sample_prompts[st.archetype] = {
                "case_id": c["case_id"],
                "symbol": c["symbol"],
                "status": st.status,
                "score": round(float(st.score), 2),
                "prompt": prompt_str,
            }

    mean_ms = sum(ev_times_ms) / max(len(ev_times_ms), 1)
    sorted_ms = sorted(ev_times_ms)
    p50_ms = sorted_ms[len(sorted_ms) // 2]
    p95_ms = sorted_ms[int(len(sorted_ms) * 0.95)]
    max_ms = sorted_ms[-1]

    print("=== FASE 2 VERIFICATION SUMMARY ===")
    print(f"Neutrality check        : {identical}/{total} identical (mismatches={len(mismatches)})")
    print(f"Evidence error count    : {len(ev_errors)}/{total}")
    print(f"build_evidence CPU time : mean={mean_ms:.3f} ms | median={p50_ms:.3f} ms | p95={p95_ms:.3f} ms | max={max_ms:.3f} ms")
    print(f"Archetypes captured     : {list(sample_prompts.keys())}")

    assert len(mismatches) == 0, f"Neutrality violated: {mismatches[:3]}"
    assert len(ev_errors) == 0, f"Evidence errors found: {ev_errors[:3]}"
    assert mean_ms < 5.0, f"Evidence CPU overhead too high: {mean_ms:.3f} ms >= 5.0 ms"

    for arch in ("HTF_REVERSAL", "TREND_CONT", "PRE_REACTION"):
        assert arch in sample_prompts, f"Missing sample prompt for {arch}"
        pr = sample_prompts[arch]["prompt"]
        assert "[BUKTI KUALITAS TAMBAHAN DARI PYTHON" in pr, f"Missing evidence header in {arch} prompt"
        print(f"[PASS] Prompt for {arch} contains '[BUKTI KUALITAS TAMBAHAN DARI PYTHON'")

    if args.save_prompts:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_prompts)), exist_ok=True)
        with open(args.save_prompts, "w", encoding="utf-8") as f:
            json.dump(sample_prompts, f, indent=2, ensure_ascii=False)
        print(f"Saved 3 archetype prompts to: {args.save_prompts}")


if __name__ == "__main__":
    main()
