# -*- coding: utf-8 -*-
"""
Run analyze_smc_symbol across synthetic windows to record baseline metrics,
count exceptions, and support Phase 1 vs Phase 2 neutrality comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scanner_core.models import Config
from scanner_core.mtf_smc import analyze_smc_symbol
from tools.audit.synth_data import iter_synthetic_windows


def run_suite(save_path: str = "", compare_path: str = "") -> dict:
    cfg = Config()
    cases = iter_synthetic_windows()
    records = []
    exc_counter: Counter = Counter()
    arch_counter: Counter = Counter()
    status_counter: Counter = Counter()
    t0 = time.perf_counter()

    for c in cases:
        try:
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
            if st is None:
                records.append({
                    "case_id": c["case_id"],
                    "outcome": "NONE",
                })
            else:
                arch_counter[st.archetype] += 1
                status_counter[st.status] += 1
                ev = st.raw_market_data.get("evidence") if isinstance(st.raw_market_data, dict) else None
                records.append({
                    "case_id": c["case_id"],
                    "outcome": "SETUP",
                    "status": st.status,
                    "score": round(float(st.score), 6),
                    "entry": round(float(st.entry), 8),
                    "sl": round(float(st.sl), 8),
                    "tp1": round(float(st.tp1), 8),
                    "tp2": round(float(st.tp2), 8),
                    "archetype": st.archetype,
                    "has_evidence": ev is not None,
                    "evidence_err": ev.get("error") if isinstance(ev, dict) else None,
                })
        except Exception as e:
            ename = f"{type(e).__name__}: {e}"
            exc_counter[ename] += 1
            records.append({
                "case_id": c["case_id"],
                "outcome": "EXCEPTION",
                "error": ename,
            })

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    n_cases = len(cases)
    n_setups = sum(1 for r in records if r["outcome"] == "SETUP")
    n_none = sum(1 for r in records if r["outcome"] == "NONE")
    n_exc = sum(1 for r in records if r["outcome"] == "EXCEPTION")

    print(f"Total windows evaluated : {n_cases}")
    print(f"Setups produced         : {n_setups}")
    print(f"None returned           : {n_none}")
    print(f"Exceptions raised       : {n_exc}")
    print(f"Total elapsed time      : {elapsed_ms:.1f} ms ({elapsed_ms / max(1, n_cases):.2f} ms/call)")
    if exc_counter:
        print("Exception breakdown:")
        for k, v in exc_counter.items():
            print(f"  {k}: {v}")
    print("Archetype breakdown     :", dict(arch_counter))
    print("Status breakdown        :", dict(status_counter))

    if save_path:
        with open(save_path, "w", encoding="utf-8") as fh:
            json.dump({"records": records, "elapsed_ms": elapsed_ms}, fh, indent=2)
        print(f"Saved snapshot to {save_path}")

    if compare_path:
        with open(compare_path, "r", encoding="utf-8") as fh:
            ref = json.load(fh)["records"]
        fields = ("outcome", "status", "score", "entry", "sl", "tp1", "tp2", "archetype")
        mismatches = []
        matched = 0
        for r_old, r_new in zip(ref, records):
            same = all(r_old.get(f) == r_new.get(f) for f in fields)
            if same:
                matched += 1
            else:
                mismatches.append((r_old["case_id"], r_old, r_new))
        print(f"Neutrality comparison   : {matched}/{len(records)} identical on {fields}")
        if mismatches:
            print(f"MISMATCHES ({len(mismatches)}):")
            for cid, ro, rn in mismatches[:5]:
                print(f"  {cid}: ref={ro} vs cur={rn}")
            sys.exit(1)

    return {"n_cases": n_cases, "n_setups": n_setups, "n_exc": n_exc, "elapsed_ms": elapsed_ms}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", type=str, default="")
    ap.add_argument("--compare", type=str, default="")
    args = ap.parse_args()
    run_suite(save_path=args.save, compare_path=args.compare)

