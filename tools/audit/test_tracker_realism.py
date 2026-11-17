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


if __name__ == "__main__":
    test_limit_penetration()
    test_sl_slippage_and_fees_manual()
    test_zero_param_regression()
    test_fetch_error_csv_error_hold_compare_and_since_version()
    test_code_version_and_cfg_hash()
