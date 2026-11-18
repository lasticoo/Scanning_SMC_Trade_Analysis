# -*- coding: utf-8 -*-
"""
Unit & Integration Tests for M01 (`test_tracker_realism.py`):
1. Limit penetration (limit_pen_bps=2): LONG touch low == entry does NOT fill; low == entry*(1 - 3e-4) fills.
2. LOSS_SL realized_r == round(-1 - fee_r - slip_r, 3) manually verified for both maker limit and taker market.
3. Zero-parameter regression matches legacy simulation behavior.
4. FETCH_ERROR when klines raises (mock) and is excluded from closed/win/loss; corrupt CSV files counted & reported.
5. `code_version` & `cfg_hash` written to JSONL & CSV, `cfg_hash` changes when one Config field changes,
   and `--since-version`, `ex-top10`, `TP2 realized vs planned`, and `--hold-compare` (30h vs 60h) work cleanly.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from unittest.mock import patch

import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import binance
import scanner_core.tracker as tracker_mod
from scanner_core.models import Config, Setup
from scanner_core.tracker import simulate_trade_on_bars


def _make_bars(rows):
    base_ms = 1763193600000
    recs = []
    for i, (o, h, l, c) in enumerate(rows):
        recs.append({
            "open_time": base_ms + i * 900_000,
            "open": float(o),
            "high": float(h),
            "low": float(l),
            "close": float(c),
        })
    return pd.DataFrame(recs)


def test_limit_penetration() -> None:
    entry, sl, tp1, tp2 = 100.0, 98.0, 103.0, 106.0

    # 1. Exact touch low == entry must NOT fill when limit_pen_bps=2.0
    bars_touch = _make_bars([
        (100.5, 101.0, 100.0, 100.8),
        (100.8, 106.5, 100.2, 106.2),
    ])
    res_touch = simulate_trade_on_bars(
        direction="LONG",
        entry_price=entry,
        poi_low=99.8,
        poi_high=100.2,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        require_limit_tap=True,
        future_df=bars_touch,
        limit_pen_bps=2.0,
    )
    assert res_touch["outcome"] == "UNFILLED", f"Expected UNFILLED on exact touch, got {res_touch}"

    # 2. Penetrating below limit trigger low == entry*(1 - 3e-4) = 99.97 MUST fill and hit TP2
    bars_pen = _make_bars([
        (100.5, 101.0, entry * (1.0 - 3e-4), 100.8),
        (100.8, 106.5, 100.2, 106.2),
    ])
    res_pen = simulate_trade_on_bars(
        direction="LONG",
        entry_price=entry,
        poi_low=99.8,
        poi_high=100.2,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        require_limit_tap=True,
        future_df=bars_pen,
        limit_pen_bps=2.0,
    )
    assert res_pen["outcome"] == "TP2_FULL", f"Expected TP2_FULL after 3bps penetration, got {res_pen}"

    # 3. SHORT symmetry: touch high == entry does NOT fill, 3bps above entry DOES fill
    s_bars_touch = _make_bars([
        (99.5, 100.0, 99.0, 99.2),
        (99.2, 99.8, 93.5, 93.8),
    ])
    s_res_touch = simulate_trade_on_bars(
        direction="SHORT",
        entry_price=100.0,
        poi_low=99.8,
        poi_high=100.2,
        sl=102.0,
        tp1=97.0,
        tp2=94.0,
        require_limit_tap=True,
        future_df=s_bars_touch,
        limit_pen_bps=2.0,
    )
    assert s_res_touch["outcome"] == "UNFILLED", f"Expected SHORT UNFILLED on exact touch, got {s_res_touch}"

    s_bars_pen = _make_bars([
        (99.5, 100.0 * (1.0 + 3e-4), 99.0, 99.2),
        (99.2, 99.8, 93.5, 93.8),
    ])
    s_res_pen = simulate_trade_on_bars(
        direction="SHORT",
        entry_price=100.0,
        poi_low=99.8,
        poi_high=100.2,
        sl=102.0,
        tp1=97.0,
        tp2=94.0,
        require_limit_tap=True,
        future_df=s_bars_pen,
        limit_pen_bps=2.0,
    )
    assert s_res_pen["outcome"] == "TP2_FULL", f"Expected SHORT TP2_FULL after 3bps penetration, got {s_res_pen}"

    print("[PASS] Limit penetration: low == entry UNFILLED with 2bps, low == entry*(1-3e-4) FILLED (LONG & SHORT)")


def test_sl_slippage_and_fees_manual() -> None:
    entry, sl, tp1, tp2 = 100.0, 98.0, 103.0, 106.0
    risk = entry - sl  # 2.0

    # Limit order (maker 0.02% + taker 0.04%) + SL slippage 5 bps
    bars_lim_sl = _make_bars([
        (100.2, 100.5, 99.95, 100.1),  # fills at 100.0 (o=100.2 >= 100.0, l=99.95 <= 99.98)
        (100.1, 100.4, 97.9, 98.0),    # hits SL
    ])
    res_lim = simulate_trade_on_bars(
        direction="LONG",
        entry_price=entry,
        poi_low=99.8,
        poi_high=100.2,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        require_limit_tap=True,
        future_df=bars_lim_sl,
        fee_pct=0.04,
        maker_fee_pct=0.02,
        slippage_bps=5.0,
        limit_pen_bps=2.0,
    )
    manual_fee_r = ((0.02 + 0.04) / 100.0 * 100.0) / risk     # 0.06 / 2.0 = 0.030
    manual_slip_r = (100.0 * 5.0 / 1e4) / risk                # 0.05 / 2.0 = 0.025
    expected_lim_r = round(-1.0 - manual_fee_r - manual_slip_r, 3)  # -1.055
    assert res_lim["outcome"] == "LOSS_SL"
    assert res_lim["realized_r"] == expected_lim_r, f"Expected {expected_lim_r}, got {res_lim['realized_r']}"

    # Market order (taker 0.04% + taker 0.04%) + SL slippage 5 bps
    bars_mkt_sl = _make_bars([
        (100.0, 100.5, 97.9, 98.0),
    ])
    res_mkt = simulate_trade_on_bars(
        direction="LONG",
        entry_price=entry,
        poi_low=99.8,
        poi_high=100.2,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        require_limit_tap=False,
        future_df=bars_mkt_sl,
        fee_pct=0.04,
        maker_fee_pct=0.02,
        slippage_bps=5.0,
        limit_pen_bps=2.0,
    )
    mkt_fee_r = ((0.04 + 0.04) / 100.0 * 100.0) / risk        # 0.08 / 2.0 = 0.040
    mkt_slip_r = (100.0 * 5.0 / 1e4) / risk                   # 0.025
    expected_mkt_r = round(-1.0 - mkt_fee_r - mkt_slip_r, 3)  # -1.065
    assert res_mkt["outcome"] == "LOSS_SL"
    assert res_mkt["realized_r"] == expected_mkt_r, f"Expected {expected_mkt_r}, got {res_mkt['realized_r']}"

    print(f"[PASS] Manual LOSS_SL realized_r = -1 - fee_r - slip_r verified (limit={expected_lim_r}, market={expected_mkt_r})")


def test_zero_param_regression() -> None:
    entry, sl, tp1, tp2 = 100.0, 98.0, 103.0, 106.0
    risk = entry - sl

    # Market trade with new params = 0 -> exact legacy behavior (-1 - 2*fee_pct*entry/(100*risk))
    bars_mkt = _make_bars([(100.0, 100.5, 97.9, 98.0)])
    res_mkt_zero = simulate_trade_on_bars(
        direction="LONG",
        entry_price=entry,
        poi_low=99.8,
        poi_high=100.2,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        require_limit_tap=False,
        future_df=bars_mkt,
        fee_pct=0.04,
        maker_fee_pct=0.0,
        slippage_bps=0.0,
        limit_pen_bps=0.0,
    )
    legacy_mkt_r = round(-1.0 - ((2.0 * 0.04 / 100.0) * 100.0) / risk, 3)
    assert res_mkt_zero["realized_r"] == legacy_mkt_r

    # Limit touch with limit_pen_bps=0 fills at low==entry; with maker_fee_pct=fee_pct and slippage_bps=0 matches legacy exact
    bars_lim = _make_bars([
        (100.5, 100.8, 100.0, 100.6),
        (100.6, 100.8, 97.9, 98.0),
    ])
    res_lim_legacy = simulate_trade_on_bars(
        direction="LONG",
        entry_price=entry,
        poi_low=99.8,
        poi_high=100.2,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        require_limit_tap=True,
        future_df=bars_lim,
        fee_pct=0.04,
        maker_fee_pct=0.04,
        slippage_bps=0.0,
        limit_pen_bps=0.0,
    )
    assert res_lim_legacy["outcome"] == "LOSS_SL"
    assert res_lim_legacy["realized_r"] == legacy_mkt_r

    # All-zero fees & penalties = raw -1.000 / +3.000
    res_raw_tp2 = simulate_trade_on_bars(
        direction="LONG",
        entry_price=entry,
        poi_low=99.8,
        poi_high=100.2,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        require_limit_tap=True,
        future_df=_make_bars([(100.5, 100.8, 100.0, 100.6), (100.6, 106.5, 100.5, 106.2)]),
        fee_pct=0.0,
        maker_fee_pct=0.0,
        slippage_bps=0.0,
        limit_pen_bps=0.0,
    )
    assert res_raw_tp2["outcome"] == "TP2_FULL" and res_raw_tp2["realized_r"] == 3.0

    print("[PASS] Zero-parameter regression matches legacy behavior")


def test_fetch_error_csv_error_hold_compare_and_since_version() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Write 1 corrupt CSV + 1 valid CSV with >10 trades so ex-top10 & hold-compare (30h vs 60h) are exercised
        with open(os.path.join(tmpdir, "scan_20261114_0000.csv"), "w", encoding="utf-8") as fh:
            fh.write("")  # raises EmptyDataError in pd.read_csv

        rows_csv = []
        base_t = pd.Timestamp("2026-11-14 01:00:00")
        for i in range(14):
            sym = f"COIN{i:02d}USDT"
            t_str = (base_t + pd.Timedelta(hours=i * 5)).strftime("%Y-%m-%d %H:%M")
            rows_csv.append({
                "scan_utc": "2026-11-14 01:00:00" if i < 12 else "2026-11-15 01:00:00",
                "signal_candle_utc": t_str,
                "symbol": sym,
                "direction": "LONG",
                "kind": "SMC_SNIPER",
                "archetype": "TREND_CONT",
                "status": "ENTRY READY",
                "score": 82.0,
                "ai_decision": "APPROVED",
                "entry": 100.0,
                "stop_loss": 98.0,
                "tp1": 103.0,
                "tp2": 106.0,
                "rr": 3.0,
            })
        pd.DataFrame(rows_csv).to_csv(os.path.join(tmpdir, "scan_20261114_0100.csv"), index=False)

        # JSONL: give COIN00..COIN11 code_version='legacy_v', and COIN12..COIN13 code_version='v_new99'
        jsonl_path = os.path.join(tmpdir, "setups_audit.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as fh:
            for i in range(14):
                sym = f"COIN{i:02d}USDT"
                rec = {
                    "scan_utc": "2026-11-14 01:00:00" if i < 12 else "2026-11-15 01:00:00",
                    "symbol": sym,
                    "direction": "LONG",
                    "code_version": "legacy" if i < 12 else "v_new99",
                    "cfg_hash": "1122334455",
                    "python_status": "ENTRY READY",
                    "llm_status": "ENTRY READY",
                    "evidence": {"er_1h_24": 0.36},
                    "flags": [],
                }
                fh.write(json.dumps(rec) + "\n")

        # Mock future bars:
        # - COIN00USDT throws ConnectionError -> FETCH_ERROR
        # - COIN01USDT hits TP2 at bar index 140 (hour 35.25 -> TIME_EXIT under 30h, TP2_FULL under 60h!)
        # - COIN02..COIN12 hit TP2 immediately at bar 2 (for testing ex-top10 & TP2 realized vs planned)
        # - COIN13 hits SL at bar 1
        base_ms = 1763100000000

        class FakeAsyncBinance:
            def __init__(self, concurrency: int = 14):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def server_time(self) -> int:
                return base_ms + 200 * 3_600_000

            async def klines(self, sym: str, interval: str, limit: int, server_ms: int, start_time: int = 0):
                if sym == "COIN00USDT":
                    raise ConnectionError("Simulated Binance timeout")
                if sym == "COIN01USDT":
                    # 160 bars: sideway 100.5 until bar 140 where it hits TP2=106.0
                    bars = [(100.5, 101.0, 99.5, 100.5) for _ in range(160)]
                    bars[140] = (101.0, 106.5, 100.5, 106.2)
                    return _make_bars(bars)
                if sym == "COIN13USDT":
                    return _make_bars([(100.0, 100.2, 97.5, 98.0), (98.0, 98.5, 97.0, 97.5)])
                return _make_bars([(100.0, 101.0, 99.5, 101.0), (101.0, 106.5, 100.5, 106.2)])

        with patch.object(tracker_mod, "AsyncBinance", FakeAsyncBinance):
            summary, res_df = asyncio.run(
                tracker_mod._track_signals_async(
                    scan_dir=tmpdir,
                    hold_compare=True,
                )
            )

        assert int((res_df["outcome"] == "FETCH_ERROR").sum()) == 1
        closed_df = res_df[res_df["outcome"].isin(["TP2_FULL", "TP1_BEP", "LOSS_SL", "TIME_EXIT"])]
        assert "COIN00USDT" not in list(closed_df["symbol"]), "FETCH_ERROR must not be counted in closed trades"
        assert "Fetch Error: 1 (ConnectionError=1)" in summary, f"Missing Fetch Error in summary:\n{summary[:600]}"
        assert "CSV Rusak/Dilewati                 : 1 file (EmptyDataError=1)" in summary, f"Missing CSV error summary:\n{summary[:600]}"
        assert "Performa ex-top10" in summary
        assert "TP2 realized vs planned" in summary
        assert "PERBANDINGAN HORIZON HOLD BERDAMPINGAN: 30h vs 60h" in summary

        # Verify COIN01USDT was TIME_EXIT in 30h (120 bars) and TP2_FULL in 60h (240 bars)
        c01 = res_df[res_df["symbol"] == "COIN01USDT"].iloc[0]
        assert c01["outcome"] == "TIME_EXIT", f"Expected COIN01USDT TIME_EXIT at 30h, got {c01['outcome']}"
        assert "Hold 60h" in summary and "TP2 +1" in summary, "Expected hold-compare to show +1 TP2 in 60h vs 30h"

        # Now test --since-version 'v_new99' filtering
        with patch.object(tracker_mod, "AsyncBinance", FakeAsyncBinance):
            summary_v, res_df_v = asyncio.run(
                tracker_mod._track_signals_async(
                    scan_dir=tmpdir,
                    since_version="v_new99",
                )
            )
        assert len(res_df_v) == 2 and set(res_df_v["symbol"]) == {"COIN12USDT", "COIN13USDT"}, f"Unexpected since_version result: {res_df_v}"
        assert all(res_df_v["code_version"] == "v_new99")
        assert "Since Version: v_new99" in summary_v

    print("[PASS] FETCH_ERROR, CSV errors, ex-top10, TP2 realized vs planned, 30h vs 60h compare, and --since-version verified")


def test_code_version_and_cfg_hash() -> None:
    cfg1 = Config()
    cfg2 = Config(max_hold_hours=45.0)
    h1 = binance.compute_cfg_hash(cfg1)
    h2 = binance.compute_cfg_hash(cfg2)
    assert len(h1) == 10 and len(h2) == 10 and h1 != h2, f"cfg_hash did not change on Config modification: {h1} vs {h2}"

    ver = binance.get_code_version()
    assert isinstance(ver, str) and len(ver) > 0

    st = Setup(
        symbol="TESTUSDT",
        direction="LONG",
        kind="SMC_SNIPER",
        pattern="Test",
        entry=100.0,
        market_price=100.0,
        limit_entry=100.0,
        sl=98.0,
        tp1=103.0,
        tp2=106.0,
        rr=3.0,
        status="ENTRY READY",
        score=82.0,
        python_score=82.0,
        archetype="TREND_CONT",
    )

    df_csv = binance.results_to_df([st], "2026-11-18 00:00:00", cfg=cfg1)
    assert df_csv.loc[0, "code_version"] == ver
    assert df_csv.loc[0, "cfg_hash"] == h1

    with tempfile.TemporaryDirectory() as tmpdir:
        jpath = binance.append_setups_jsonl([st], "2026-11-18 00:00:00", tmpdir, cfg=cfg1)
        rec = json.loads(open(jpath, encoding="utf-8").readline())
        assert rec["code_version"] == ver
        assert rec["cfg_hash"] == h1

    print(f"[PASS] code_version={ver} & cfg_hash ({h1} != {h2}) verified in JSONL & CSV")


def test_since_version_before_dedup_and_csv_priority() -> None:
    from scanner_core.evidence_report import version_matches  # noqa: F401 (import sanity)

    class SimpleFakeAsyncBinance:
        def __init__(self, concurrency: int = 14):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def server_time(self) -> int:
            return int(pd.Timestamp("2026-11-20 00:00:00").timestamp() * 1000)

        async def klines(self, sym: str, interval: str, limit: int, server_ms: int, start_time: int = 0):
            return _make_bars([
                (100.0, 101.0, 99.5, 101.0),
                (101.0, 106.5, 100.5, 106.2),
            ])

    def _base_csv_row(sym: str, scan_utc: str, sig_utc: str) -> dict:
        return {
            "scan_utc": scan_utc,
            "signal_candle_utc": sig_utc,
            "symbol": sym,
            "direction": "LONG",
            "kind": "SMC_SNIPER",
            "archetype": "TREND_CONT",
            "status": "ENTRY READY",
            "score": 82.0,
            "ai_decision": "APPROVED",
            "entry": 100.0,
            "stop_loss": 98.0,
            "tp1": 103.0,
            "tp2": 106.0,
            "rr": 3.0,
        }

    # 1. --since-version MUST run BEFORE 4h dedup so newer-version signals aren't shadowed by legacy
    with tempfile.TemporaryDirectory() as tmpdir_dedup:
        r_old = _base_csv_row("BTCUSDT", "2026-11-15 08:00:00", "2026-11-15 08:00")
        r_old["code_version"] = "legacy"
        r_old["cfg_hash"] = "0000000001"
        r_new = _base_csv_row("BTCUSDT", "2026-11-15 09:00:00", "2026-11-15 09:00")
        r_new["code_version"] = "new_sha123"
        r_new["cfg_hash"] = "0000000002"
        pd.DataFrame([r_old]).to_csv(os.path.join(tmpdir_dedup, "scan_20261115_0800.csv"), index=False)
        pd.DataFrame([r_new]).to_csv(os.path.join(tmpdir_dedup, "scan_20261115_0900.csv"), index=False)

        with patch.object(tracker_mod, "AsyncBinance", SimpleFakeAsyncBinance):
            _, res_all = asyncio.run(
                tracker_mod._track_signals_async(scan_dir=tmpdir_dedup, write_outputs=False)
            )
            _, res_filt = asyncio.run(
                tracker_mod._track_signals_async(
                    scan_dir=tmpdir_dedup,
                    since_version="new_sha123",
                    write_outputs=False,
                )
            )
        assert len(res_all) == 1 and res_all.iloc[0]["code_version"] == "legacy", (
            f"Unfiltered 4h dedup sanity check failed: {res_all}"
        )
        assert len(res_filt) == 1 and res_filt.iloc[0]["code_version"] == "new_sha123", (
            f"Version filter ran AFTER 4h dedup and dropped new signal: {res_filt}"
        )
        assert res_filt.iloc[0]["scan_utc"] == "2026-11-15 09:00:00"

    # 2a. CSV code_version is primary source; empty CSV falls back to JSONL; missing in both -> 'legacy'
    with tempfile.TemporaryDirectory() as tmpdir_prio:
        rows_prio = []
        for sym, cv, ch in [
            ("ETHUSDT", "v_csv", "csvhash001"),
            ("SOLUSDT", "", ""),
            ("BNBUSDT", "", ""),
        ]:
            r = _base_csv_row(sym, "2026-11-15 08:00:00", "2026-11-15 08:00")
            r["code_version"] = cv
            r["cfg_hash"] = ch
            rows_prio.append(r)
        pd.DataFrame(rows_prio).to_csv(os.path.join(tmpdir_prio, "scan_20261115_0800.csv"), index=False)

        with open(os.path.join(tmpdir_prio, "setups_audit.jsonl"), "w", encoding="utf-8") as fh:
            for sym, cv, ch in [
                ("ETHUSDT", "v_jsonl", "jsonlh001"),
                ("SOLUSDT", "v_jsonl", "jsonlh002"),
            ]:
                fh.write(json.dumps({
                    "scan_utc": "2026-11-15 08:00:00",
                    "symbol": sym,
                    "direction": "LONG",
                    "code_version": cv,
                    "cfg_hash": ch,
                }) + "\n")

        with patch.object(tracker_mod, "AsyncBinance", SimpleFakeAsyncBinance):
            _, df_unfilt = asyncio.run(
                tracker_mod._track_signals_async(scan_dir=tmpdir_prio, write_outputs=False)
            )
            _, df_v_csv = asyncio.run(
                tracker_mod._track_signals_async(scan_dir=tmpdir_prio, since_version="v_csv", write_outputs=False)
            )
            _, df_v_jsonl = asyncio.run(
                tracker_mod._track_signals_async(scan_dir=tmpdir_prio, since_version="v_jsonl", write_outputs=False)
            )

        cv_map = dict(zip(df_unfilt["symbol"], df_unfilt["code_version"]))
        ch_map = dict(zip(df_unfilt["symbol"], df_unfilt["cfg_hash"]))
        assert cv_map == {"ETHUSDT": "v_csv", "SOLUSDT": "v_jsonl", "BNBUSDT": "legacy"}, f"Unexpected cv_map: {cv_map}"
        assert ch_map["ETHUSDT"] == "csvhash001" and ch_map["SOLUSDT"] == "jsonlh002"
        assert set(df_v_csv["symbol"]) == {"ETHUSDT"}, f"Expected only ETHUSDT for v_csv, got {list(df_v_csv['symbol'])}"
        assert set(df_v_jsonl["symbol"]) == {"SOLUSDT"}, (
            f"Expected only SOLUSDT for v_jsonl (ETHUSDT overridden by CSV v_csv), got {list(df_v_jsonl['symbol'])}"
        )

    # 2b. Legacy CSV without any code_version column at all: falls back to JSONL or 'legacy'
    with tempfile.TemporaryDirectory() as tmpdir_nocol:
        pd.DataFrame([
            _base_csv_row("XRPUSDT", "2026-11-15 08:00:00", "2026-11-15 08:00"),
            _base_csv_row("ADAUSDT", "2026-11-15 08:00:00", "2026-11-15 08:00"),
        ]).to_csv(os.path.join(tmpdir_nocol, "scan_20261115_0800.csv"), index=False)
        with open(os.path.join(tmpdir_nocol, "setups_audit.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "scan_utc": "2026-11-15 08:00:00",
                "symbol": "XRPUSDT",
                "direction": "LONG",
                "code_version": "v_jonly",
                "cfg_hash": "jh00000001",
            }) + "\n")

        with patch.object(tracker_mod, "AsyncBinance", SimpleFakeAsyncBinance):
            _, df_nocol = asyncio.run(
                tracker_mod._track_signals_async(scan_dir=tmpdir_nocol, write_outputs=False)
            )
            _, df_jonly = asyncio.run(
                tracker_mod._track_signals_async(scan_dir=tmpdir_nocol, since_version="v_jonly", write_outputs=False)
            )
        assert dict(zip(df_nocol["symbol"], df_nocol["code_version"])) == {"XRPUSDT": "v_jonly", "ADAUSDT": "legacy"}
        assert list(df_jonly["symbol"]) == ["XRPUSDT"]

    print("[PASS] --since-version runs before 4h dedup; CSV code_version primary + JSONL fallback + 'legacy' verified")


def test_dirty_git_version_and_cfg_exclusions() -> None:
    from scanner_core.evidence_report import version_matches

    prev_cache = binance._CODE_VERSION_CACHE
    try:
        # 1. Clean repo -> no -dirty
        binance._CODE_VERSION_CACHE = None
        with patch.object(binance.subprocess, "check_output", side_effect=["abc1234\n", ""]):
            assert binance.get_code_version() == "abc1234"

        # 2. Only scan_output/ dirty -> still no -dirty
        binance._CODE_VERSION_CACHE = None
        with patch.object(
            binance.subprocess,
            "check_output",
            side_effect=["abc1234\n", "?? scan_output/scan_20261118.csv\n M scan_output/tracker_trades.csv\n"],
        ):
            assert binance.get_code_version() == "abc1234"

        # 3. Non-scan_output file dirty -> '-dirty', and per-process cached on second call
        binance._CODE_VERSION_CACHE = None
        with patch.object(
            binance.subprocess,
            "check_output",
            side_effect=["abc1234\n", "?? scan_output/x.csv\n M scanner_core/tracker.py\n"],
        ) as mock_co:
            v1 = binance.get_code_version()
            v2 = binance.get_code_version()
            assert v1 == "abc1234-dirty" and v2 == "abc1234-dirty"
            assert mock_co.call_count == 2, f"Expected cached result without extra subprocess calls, got {mock_co.call_count}"

        # 4. Fallback 'nogit' when git fails
        binance._CODE_VERSION_CACHE = None
        with patch.object(binance.subprocess, "check_output", side_effect=FileNotFoundError("no git")):
            assert binance.get_code_version() == "nogit"
    finally:
        binance._CODE_VERSION_CACHE = None

    assert version_matches("abc1234-dirty", "abc1234") is True
    assert version_matches("abc1234-dirty", "abc1234-dirty") is True
    assert version_matches("other123", "abc1234") is False

    # 5. compute_cfg_hash excludes non-signal runtime fields, keeps signal fields sensitive
    cfg_base = Config()
    cfg_nonsig = Config(
        workers=99,
        use_async=False,
        ai_audit=False,
        ai_top_n=1,
        fetch_derivatives=False,
        radar_mode=True,
        only_in_zone=True,
    )
    cfg_sig = Config(max_hold_hours=45.0)
    h_base = binance.compute_cfg_hash(cfg_base)
    h_nonsig = binance.compute_cfg_hash(cfg_nonsig)
    h_sig = binance.compute_cfg_hash(cfg_sig)
    assert h_base == h_nonsig, f"Non-signal fields changed cfg_hash: {h_base} vs {h_nonsig}"
    assert h_base != h_sig, f"Signal field max_hold_hours did not change cfg_hash: {h_base}"
    # Avoid leaving mock value cached if anything else touched it; prev_cache unused intentionally
    _ = prev_cache

    print("[PASS] get_code_version (-dirty excluding scan_output, cache, nogit fallback) & compute_cfg_hash exclusions verified")


def test_summary_resolved_wr_and_time_exit_share() -> None:
    df_syn = pd.DataFrame([
        {"symbol": "A", "outcome": "TP2_FULL", "realized_r": 3.0, "hold_hours": 4.0, "risk_pct": 2.0, "kind": "SMC_SNIPER", "ai_decision": "APPROVED", "planned_rr": 3.0},
        {"symbol": "B1", "outcome": "LOSS_SL", "realized_r": -1.0, "hold_hours": 3.0, "risk_pct": 2.0, "kind": "SMC_SNIPER", "ai_decision": "APPROVED", "planned_rr": 3.0},
        {"symbol": "B2", "outcome": "LOSS_SL", "realized_r": -1.0, "hold_hours": 3.0, "risk_pct": 2.0, "kind": "SMC_SNIPER", "ai_decision": "APPROVED", "planned_rr": 3.0},
        {"symbol": "B3", "outcome": "LOSS_SL", "realized_r": -1.0, "hold_hours": 3.0, "risk_pct": 2.0, "kind": "SMC_SNIPER", "ai_decision": "APPROVED", "planned_rr": 3.0},
        {"symbol": "C1", "outcome": "TIME_EXIT", "realized_r": 1.0, "hold_hours": 30.0, "risk_pct": 2.0, "kind": "SMC_SNIPER", "ai_decision": "APPROVED", "planned_rr": 3.0},
        {"symbol": "C2", "outcome": "TIME_EXIT", "realized_r": 1.0, "hold_hours": 30.0, "risk_pct": 2.0, "kind": "SMC_SNIPER", "ai_decision": "APPROVED", "planned_rr": 3.0},
    ])
    summary = tracker_mod._build_metrics_summary(df_syn, "TEST SUMMARY")
    expected_eff = "Effective Win Rate (termasuk time-exit mark-to-market) :  50.0%  (1 Full TP2 [16.7%], 0 TP1->BEP, 3 SL, 2 Time-Exit)"
    expected_res = "Win Rate resolved (TP2 vs SL saja)                 :  25.0%  (1/4 TP2 vs SL)  |  Kontribusi TIME_EXIT: +2.00 R (+100.0% dari total)"
    assert expected_eff in summary, f"Missing Effective WR label in summary:\n{summary}"
    assert expected_res in summary, f"Missing resolved WR / TIME_EXIT share in summary:\n{summary}"

    df_60 = df_syn.copy()
    df_60.loc[df_60["symbol"] == "C1", ["outcome", "realized_r", "hold_hours"]] = ["TP2_FULL", 3.0, 42.0]
    cmp_block = tracker_mod._build_hold_compare_block(df_syn, df_60)
    assert "Effective WR (termasuk time-exit)" in cmp_block
    assert "Win Rate resolved (TP2 vs SL)" in cmp_block
    assert "Kontribusi TIME_EXIT (R / %)" in cmp_block

    print("[PASS] Summary resolved WR (TP2 vs SL), Effective WR mark-to-market label & TIME_EXIT Net R % verified")


if __name__ == "__main__":
    test_limit_penetration()
    test_sl_slippage_and_fees_manual()
    test_zero_param_regression()
    test_fetch_error_csv_error_hold_compare_and_since_version()
    test_code_version_and_cfg_hash()
    test_since_version_before_dedup_and_csv_priority()
    test_dirty_git_version_and_cfg_exclusions()
    test_summary_resolved_wr_and_time_exit_share()
