# -*- coding: utf-8 -*-
"""
Unit & Integration Tests for Fase 3 (binance.py async scaling via asyncio.to_thread):
1. Thread vs Serial equivalence across 18 synthetic symbols on (status, score, entry, tp2) + radar hits.
2. Fault isolation in Pass 1 (worker_one) and Pass 2 (derivative re-evaluation via asyncio.gather(return_exceptions=True)).
3. `error_types` Counter in `stats` and compatibility with `build_report`.
"""
from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import patch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import binance
from scanner_core.indicators import htf_context
from scanner_core.models import Config
from tools.audit.synth_data import iter_synthetic_windows


def _build_mock_universe(n_symbols: int = 10):
    cases = iter_synthetic_windows(seeds=(11, 22), n_bars_15m=2200, windows_per_seed=max(1, (n_symbols + 1) // 2))[:n_symbols]
    by_sym = {}
    universe = []
    for idx, c in enumerate(cases):
        sym = f"TEST{idx + 1:02d}USDT"
        meta = dict(c["meta"])
        by_sym[sym] = {
            "df4": c["df4"].copy(),
            "df1": c["df1"].copy(),
            "df15": c["df15"].copy(),
            "meta": meta,
            "deriv": {
                "oi_change_1h_pct": 2.4 if idx % 2 == 0 else -1.8,
                "oi_change_4h_pct": 5.1 if idx % 2 == 0 else -3.2,
                "top_ls_ratio": 1.18,
                "retail_ls_ratio": 0.92,
                "funding_rate": 0.0001,
            },
        }
        universe.append((sym, meta))
    btc_case = cases[0]
    return universe, by_sym, btc_case["df4"].copy(), btc_case["df1"].copy()


def _run_serial_reference(cfg: Config, universe, by_sym, b4, b1):
    btc = {"4h": htf_context(b4, cfg.htf_piv), "1h": htf_context(b1, cfg.htf_piv)}
    results = []
    radar_map = {}
    for sym, meta in universe:
        d = by_sym[sym]
        df4, df1, df15 = d["df4"], d["df1"], d["df15"]
        ctx4, ctx1, st, rh = binance._cpu_bundle(sym, df4, df1, df15, meta, cfg, btc)
        if rh is not None:
            radar_map[sym] = rh
        if st is not None:
            if cfg.fetch_derivatives:
                st_new = binance._eval_strategy(sym, df4, df1, df15, ctx4, ctx1, meta, cfg, deriv=d["deriv"], btc=btc)
                final_st = st_new if st_new is not None else st
            else:
                final_st = st
            results.append(final_st)
            if sym in radar_map:
                radar_map[sym].setup_status = final_st.status
                radar_map[sym].setup_score = float(final_st.score)
    return results, radar_map


def test_thread_vs_serial_identity() -> None:
    cfg = Config(fetch_derivatives=True, ai_audit=False, workers=6)
    universe, by_sym, b4, b1 = _build_mock_universe(10)

    serial_res, serial_radar = _run_serial_reference(cfg, universe, by_sym, b4, b1)

    class FakeAsyncBinance:
        def __init__(self, concurrency: int = 8):
            self.concurrency = concurrency

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def server_time(self) -> int:
            return 1731650000000

        async def build_universe(self, cfg_obj, only):
            return universe, len(universe), len(universe)

        async def klines(self, sym: str, interval: str, limit: int, server_ms: int):
            if sym == "BTCUSDT":
                return b4 if interval == "4h" else b1
            return by_sym[sym]["df4" if interval == "4h" else ("df1" if interval == "1h" else "df15")]

        async def fetch_symbol_klines(self, sym: str, cfg_obj, server_ms: int):
            d = by_sym[sym]
            return d["df4"], d["df1"], d["df15"]

        async def fetch_derivatives(self, sym: str):
            return by_sym[sym]["deriv"]

    with patch.object(binance, "AsyncBinance", FakeAsyncBinance):
        async_res, async_radar_list, _, stats, _ = asyncio.run(binance._scan_universe_async(cfg, None))

    assert len(async_res) == len(serial_res) == 10, f"Expected 10 setups, got {len(async_res)}"
    assert stats["errors"] == 0 and stats["error_types"] == {}, f"Unexpected errors: {stats}"

    s_map = {r.symbol: r for r in serial_res}
    a_map = {r.symbol: r for r in async_res}
    for sym in s_map:
        sr = s_map[sym]
        ar = a_map[sym]
        s_tup = (sr.status, round(float(sr.score), 6), round(float(sr.entry), 8), round(float(sr.tp2), 8), sr.archetype)
        a_tup = (ar.status, round(float(ar.score), 6), round(float(ar.entry), 8), round(float(ar.tp2), 8), ar.archetype)
        assert s_tup == a_tup, f"Mismatch on {sym}: serial={s_tup} vs thread={a_tup}"

    a_radar = {rh.symbol: rh for rh in async_radar_list}
    assert set(a_radar.keys()) == set(serial_radar.keys()), "Radar symbol keys mismatch"
    for sym, s_rh in serial_radar.items():
        a_rh = a_radar[sym]
        assert (s_rh.radar_state, s_rh.setup_status, round(s_rh.setup_score, 4), round(s_rh.dist_pct, 4)) == (
            a_rh.radar_state,
            a_rh.setup_status,
            round(a_rh.setup_score, 4),
            round(a_rh.dist_pct, 4),
        ), f"Radar mismatch on {sym}"

    print(f"[PASS] Thread vs Serial 100% identical across {len(async_res)} symbols (setups={len(async_res)}, radar_hits={len(async_radar_list)})")


def test_fault_isolation_and_error_types() -> None:
    cfg = Config(fetch_derivatives=True, ai_audit=False, workers=4)
    universe, by_sym, b4, b1 = _build_mock_universe(6)
    orig_eval = binance._eval_strategy

    def faulty_eval(sym, df4, df1, df15, ctx4, ctx1, meta, cfg_obj, deriv=None, btc=None):
        if sym == "TEST02USDT" and deriv is None:
            raise ValueError("Simulated Pass-1 kline/eval corruption")
        if sym == "TEST04USDT" and deriv is not None:
            raise RuntimeError("Simulated Pass-2 derivative re-eval failure")
        return orig_eval(sym, df4, df1, df15, ctx4, ctx1, meta, cfg_obj, deriv=deriv, btc=btc)

    class FakeAsyncBinance:
        def __init__(self, concurrency: int = 8):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def server_time(self) -> int:
            return 1731650000000

        async def build_universe(self, cfg_obj, only):
            return universe, len(universe), len(universe)

        async def klines(self, sym: str, interval: str, limit: int, server_ms: int):
            return b4 if interval == "4h" else b1

        async def fetch_symbol_klines(self, sym: str, cfg_obj, server_ms: int):
            d = by_sym[sym]
            return d["df4"], d["df1"], d["df15"]

        async def fetch_derivatives(self, sym: str):
            return by_sym[sym]["deriv"]

    with patch.object(binance, "AsyncBinance", FakeAsyncBinance), patch.object(binance, "_eval_strategy", faulty_eval):
        async_res, async_radar_list, btc, stats, _ = asyncio.run(binance._scan_universe_async(cfg, None))

    surviving_syms = [r.symbol for r in async_res]
    assert len(async_res) == 5, f"Expected 5 setups (TEST04USDT st_old retained), got {len(async_res)}"
    assert "TEST04USDT" in surviving_syms, "Expected TEST04USDT st_old to be retained when pass-2 derivative re-eval fails"
    assert "TEST02USDT" not in surviving_syms, "TEST02USDT failed in pass-1 and should not be in results"
    assert stats["errors"] == 2, f"Expected stats['errors']==2, got {stats['errors']}"
    assert stats["error_types"] == {"ValueError": 1, "RuntimeError": 1}, f"Unexpected error_types: {stats['error_types']}"

    class DummyArgs:
        equity = 0.0

    rep = binance.build_report(async_res, cfg, "2026-11-15 09:00:00", btc, stats, DummyArgs(), elapsed_s=1.2, radar_hits=async_radar_list)
    assert "2 gagal data: ValueError=1, RuntimeError=1" in rep, "Missing error_types summary in build_report output"
    print(f"[PASS] Fault isolation & st_old retention verified: surviving={len(async_res)}, TEST04USDT retained={('TEST04USDT' in surviving_syms)}, stats={stats}")


if __name__ == "__main__":
    test_thread_vs_serial_identity()
    test_fault_isolation_and_error_types()

