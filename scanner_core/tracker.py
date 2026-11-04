# -*- coding: utf-8 -*-
"""
Signal Performance Tracker & Walk-Forward Historical Backtester
===============================================================
1. Signal Tracker (`--track`):
   Membaca seluruh histori `scan_output/scan_*.csv`, mengambil candle 15M Binance sesudahnya,
   dan menghitung performa riil (Win Rate, TP1+BEP Rate, Profit Factor, Expectancy R,
   Max Drawdown R, rata-rata jam hold, serta perbandingan AI Approved vs Non-AI).

2. Walk-Forward Backtester (`--backtest`):
   Menguji mesin Top-Down MTF (4H -> 1H -> 15M + Zoom-Out Check) secara bebas look-ahead
   pada data historis Binance selama N hari terakhir untuk membuktikan kualitas SL/TP baru.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import glob
import math
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .breakout import analyze_symbol
from .exchange_async import AsyncBinance
from .indicators import htf_context
from .models import NAN, Config, fp
from .mtf_smc import analyze_smc_symbol


def _parse_utc_ms(ts_str: str) -> Optional[int]:
    if not ts_str or not isinstance(ts_str, str):
        return None
    s = ts_str.strip().replace(" UTC", "")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            d = dt.datetime.strptime(s, fmt).replace(tzinfo=dt.timezone.utc)
            return int(d.timestamp() * 1000)
        except ValueError:
            continue
    return None


def _parse_zone_bounds(zone_str: str, fallback_entry: float) -> Tuple[float, float]:
    if not isinstance(zone_str, str) or not zone_str.strip():
        return fallback_entry, fallback_entry
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", zone_str)]
    if len(nums) >= 2:
        return min(nums[0], nums[1]), max(nums[0], nums[1])
    if len(nums) == 1:
        return nums[0], nums[0]
    return fallback_entry, fallback_entry


def simulate_trade_on_bars(
    direction: str,
    entry_price: float,
    poi_low: float,
    poi_high: float,
    sl: float,
    tp1: float,
    tp2: float,
    require_limit_tap: bool,
    future_df: pd.DataFrame,
    max_hold_hours: float = 48.0,
    fee_pct: float = 0.04,
    is_historical_complete: bool = True,
) -> dict:
    """
    Mensimulasikan hasil sinyal pada bar 15M setelah sinyal close:
    Manajemen posisi institusional:
    - Ambil 50% profit di TP1 + geser sisa 50% SL ke Breakeven (BEP).
    - Sisa 50% ditutup di TP2 (atau BEP jika berbalik).
    - Jika 1 bar menyentuh SL dan TP sekaligus pada bar yang sama, dihitung konservatif (SL duluan).
    """
    is_long = direction == "LONG"
    risk = abs(entry_price - sl)
    if risk <= 0 or entry_price <= 0 or future_df.empty:
        return {"outcome": "NO_DATA", "realized_r": 0.0, "hold_hours": 0.0, "tp1_hit": False}

    # Biaya round-trip dalam satuan R
    fee_r = ((2.0 * fee_pct / 100.0) * entry_price) / risk
    rr1 = abs(tp1 - entry_price) / risk
    rr2 = abs(tp2 - entry_price) / risk

    h = future_df["high"].values.astype(float)
    l = future_df["low"].values.astype(float)
    c = future_df["close"].values.astype(float)
    o = future_df["open"].values.astype(float)
    times = future_df["open_time"].values.astype("int64")

    max_bars = max(4, int(round(max_hold_hours * 4)))
    limit_wait_bars = min(len(future_df), 16)  # tunggu retest limit maks 4 jam (16 bar 15m)

    start_bar = 0
    actual_entry = entry_price

    if require_limit_tap:
        filled = False
        for idx in range(limit_wait_bars):
            # Batal jika harga sudah tembus SL atau sudah menyentuh TP1 sebelum jemput limit
            if (is_long and (l[idx] <= sl or h[idx] >= tp1)) or (not is_long and (h[idx] >= sl or l[idx] <= tp1)):
                return {"outcome": "UNFILLED", "realized_r": 0.0, "hold_hours": 0.0, "tp1_hit": False}
            if (is_long and l[idx] <= entry_price) or (not is_long and h[idx] >= entry_price):
                actual_entry = min(float(o[idx]), entry_price) if is_long else max(float(o[idx]), entry_price)
                start_bar = idx
                filled = True
                break
        if not filled:
            return {"outcome": "UNFILLED", "realized_r": 0.0, "hold_hours": 0.0, "tp1_hit": False}
    else:
        # Eksekusi pada open candle pertama sesudah sinyal close
        actual_entry = float(o[0])
        # Jika gap pembukaan sudah melewati SL, skip
        if (is_long and actual_entry <= sl) or (not is_long and actual_entry >= sl):
            return {"outcome": "UNFILLED", "realized_r": 0.0, "hold_hours": 0.0, "tp1_hit": False}

    risk = max(abs(actual_entry - sl), 1e-12)
    rr1 = abs(tp1 - actual_entry) / risk
    rr2 = abs(tp2 - actual_entry) / risk
    fee_r = ((2.0 * fee_pct / 100.0) * actual_entry) / risk

    tp1_hit = False
    entry_ms = int(times[start_bar])
    end_idx = min(len(future_df), start_bar + max_bars)

    for j in range(start_bar, end_idx):
        dur_h = max(0.25, (int(times[j]) - entry_ms + 900_000) / 3_600_000.0)
        if is_long:
            if not tp1_hit:
                # Konservatif: cek SL lebih dulu sebelum TP1 di bar yang sama
                if l[j] <= sl:
                    return {"outcome": "LOSS_SL", "realized_r": round(-1.0 - fee_r, 3), "hold_hours": round(dur_h, 2), "tp1_hit": False}
                if h[j] >= tp1:
                    tp1_hit = True
                    if h[j] >= tp2:
                        net_r = 0.5 * rr1 + 0.5 * rr2 - fee_r
                        return {"outcome": "TP2_FULL", "realized_r": round(net_r, 3), "hold_hours": round(dur_h, 2), "tp1_hit": True}
            else:
                # SL sudah digeser ke Breakeven (actual_entry)
                if l[j] <= actual_entry:
                    net_r = 0.5 * rr1 - fee_r
                    return {"outcome": "TP1_BEP", "realized_r": round(net_r, 3), "hold_hours": round(dur_h, 2), "tp1_hit": True}
                if h[j] >= tp2:
                    net_r = 0.5 * rr1 + 0.5 * rr2 - fee_r
                    return {"outcome": "TP2_FULL", "realized_r": round(net_r, 3), "hold_hours": round(dur_h, 2), "tp1_hit": True}
        else:
            if not tp1_hit:
                if h[j] >= sl:
                    return {"outcome": "LOSS_SL", "realized_r": round(-1.0 - fee_r, 3), "hold_hours": round(dur_h, 2), "tp1_hit": False}
                if l[j] <= tp1:
                    tp1_hit = True
                    if l[j] <= tp2:
                        net_r = 0.5 * rr1 + 0.5 * rr2 - fee_r
                        return {"outcome": "TP2_FULL", "realized_r": round(net_r, 3), "hold_hours": round(dur_h, 2), "tp1_hit": True}
            else:
                if h[j] >= actual_entry:
                    net_r = 0.5 * rr1 - fee_r
                    return {"outcome": "TP1_BEP", "realized_r": round(net_r, 3), "hold_hours": round(dur_h, 2), "tp1_hit": True}
                if l[j] <= tp2:
                    net_r = 0.5 * rr1 + 0.5 * rr2 - fee_r
                    return {"outcome": "TP2_FULL", "realized_r": round(net_r, 3), "hold_hours": round(dur_h, 2), "tp1_hit": True}

    last_idx = end_idx - 1
    dur_h = max(0.25, (int(times[last_idx]) - entry_ms + 900_000) / 3_600_000.0)
    last_close = float(c[last_idx])
    rem_r = ((last_close - actual_entry) / risk) if is_long else ((actual_entry - last_close) / risk)

    if len(future_df) - start_bar < max_bars and not is_historical_complete:
        unreal_r = (0.5 * rr1 + 0.5 * max(0.0, rem_r) - fee_r) if tp1_hit else (rem_r - fee_r)
        return {"outcome": "OPEN", "realized_r": round(unreal_r, 3), "hold_hours": round(dur_h, 2), "tp1_hit": tp1_hit}

    final_r = (0.5 * rr1 + 0.5 * max(0.0, rem_r) - fee_r) if tp1_hit else (rem_r - fee_r)
    return {"outcome": "TIME_EXIT", "realized_r": round(final_r, 3), "hold_hours": round(dur_h, 2), "tp1_hit": tp1_hit}


def _build_metrics_summary(df_trades: pd.DataFrame, title: str) -> str:
    lines = ["=" * 108, title, "=" * 108]
    if df_trades.empty:
        lines.append("Tidak ditemukan data trade yang dapat dievaluasi.")
        return "\n".join(lines)

    closed_mask = df_trades["outcome"].isin(["TP2_FULL", "TP1_BEP", "LOSS_SL", "TIME_EXIT"])
    closed = df_trades[closed_mask].copy()
    open_tr = df_trades[df_trades["outcome"] == "OPEN"]
    unfilled = df_trades[df_trades["outcome"] == "UNFILLED"]

    lines.append(
        f"Total Sinyal Unik: {len(df_trades)}  |  Selesai (Closed): {len(closed)}  |  "
        f"Masih Aktif (Open): {len(open_tr)}  |  Limit Tidak Tersentuh: {len(unfilled)}"
    )
    if closed.empty:
        lines.append("Belum ada sinyal yang selesai (mencapai TP/SL/batas waktu). Coba lagi setelah beberapa candle 15M berjalan.")
        return "\n".join(lines)

    r_vals = closed["realized_r"].values.astype(float)
    wins = closed[closed["realized_r"] > 0]
    losses = closed[closed["realized_r"] <= 0]
    tp2_cnt = int((closed["outcome"] == "TP2_FULL").sum())
    tp1_bep_cnt = int((closed["outcome"] == "TP1_BEP").sum())
    sl_cnt = int((closed["outcome"] == "LOSS_SL").sum())
    texit_cnt = int((closed["outcome"] == "TIME_EXIT").sum())

    win_rate = len(wins) / len(closed) * 100.0
    tp2_rate = tp2_cnt / len(closed) * 100.0
    gross_win = float(wins["realized_r"].sum()) if not wins.empty else 0.0
    gross_loss = abs(float(losses["realized_r"].sum())) if not losses.empty else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else (99.9 if gross_win > 0 else 0.0)
    expectancy = float(np.mean(r_vals))
    total_r = float(np.sum(r_vals))
    avg_hold = float(closed["hold_hours"].mean())
    avg_sl_pct = float(closed["risk_pct"].mean()) if "risk_pct" in closed.columns else NAN

    # Hitung Max Drawdown dalam satuan R
    eq_curve = np.cumsum(r_vals)
    peaks = np.maximum.accumulate(np.insert(eq_curve, 0, 0.0))[1:]
    dd = peaks - eq_curve
    max_dd_r = float(np.max(dd)) if len(dd) else 0.0

    lines.append("-" * 108)
    lines.append(
        f"Effective Win Rate (TP2 + TP1/BEP) : {win_rate:5.1f}%  ({tp2_cnt} Full TP2 [{tp2_rate:.1f}%], "
        f"{tp1_bep_cnt} TP1->BEP, {sl_cnt} SL, {texit_cnt} Time-Exit)"
    )
    lines.append(
        f"Total Net Return                   : {total_r:+.2f} R  |  Expectancy: {expectancy:+.2f} R/trade  |  Profit Factor: {pf:.2f}"
    )
    sl_info = f"  |  Rata-rata Lebar SL: {avg_sl_pct:.2f}%" if not math.isnan(avg_sl_pct) else ""
    lines.append(f"Max Drawdown                       : {max_dd_r:.2f} R   |  Rata-rata Durasi Posisi: {avg_hold:.1f} jam{sl_info}")
    lines.append("-" * 108)

    # Breakdown per jenis setup (kind)
    lines.append("PERFORMA PER JENIS SETUP:")
    grp_rows = []
    for k_name, grp in closed.groupby("kind"):
        w = (grp["realized_r"] > 0).mean() * 100.0
        gw = grp.loc[grp["realized_r"] > 0, "realized_r"].sum()
        gl = abs(grp.loc[grp["realized_r"] <= 0, "realized_r"].sum())
        g_pf = gw / gl if gl > 0 else (99.9 if gw > 0 else 0.0)
        grp_rows.append({
            "Setup Kind": k_name,
            "Trades": len(grp),
            "WinRate %": f"{w:.1f}%",
            "Net R": f"{grp['realized_r'].sum():+.2f}R",
            "Avg R": f"{grp['realized_r'].mean():+.2f}R",
            "Profit Factor": f"{g_pf:.2f}",
            "Avg Hold (h)": f"{grp['hold_hours'].mean():.1f}h",
        })
    if grp_rows:
        lines.append(pd.DataFrame(grp_rows).to_string(index=False))

    # Breakdown AI Verdict jika tersedia
    if "ai_decision" in closed.columns and closed["ai_decision"].astype(str).str.strip().ne("").any():
        lines.append("-" * 108)
        lines.append("PERBANDINGAN VERDICT AI vs NON-AI:")
        ai_rows = []
        closed_ai = closed.copy()
        closed_ai["ai_group"] = closed_ai["ai_decision"].fillna("").replace("", "UNAUDITED_PYTHON")
        for a_name, grp in closed_ai.groupby("ai_group"):
            w = (grp["realized_r"] > 0).mean() * 100.0
            ai_rows.append({
                "Kategori Evaluasi": a_name,
                "Trades": len(grp),
                "WinRate %": f"{w:.1f}%",
                "Net R": f"{grp['realized_r'].sum():+.2f}R",
                "Avg R": f"{grp['realized_r'].mean():+.2f}R",
            })
        lines.append(pd.DataFrame(ai_rows).to_string(index=False))

    lines.append("=" * 108)
    return "\n".join(lines)


async def _track_signals_async(
    scan_dir: str = "scan_output",
    max_hold_hours: float = 48.0,
    include_watchlist: bool = False,
    fee_pct: float = 0.04,
) -> Tuple[str, pd.DataFrame]:
    csv_files = sorted(glob.glob(os.path.join(scan_dir, "scan_*.csv")))
    if not csv_files:
        return f"Tidak ditemukan file scan_*.csv di folder '{scan_dir}'.", pd.DataFrame()

    frames = []
    for p in csv_files:
        try:
            df = pd.read_csv(p)
            if {"symbol", "direction", "status", "entry", "stop_loss", "tp1", "tp2"}.issubset(df.columns):
                df["_src_file"] = os.path.basename(p)
                frames.append(df)
        except Exception:
            continue

    if not frames:
        return "Tidak ada file CSV dengan skema sinyal valid.", pd.DataFrame()

    all_df = pd.concat(frames, ignore_index=True)
    if include_watchlist:
        st_col = all_df["status"].astype(str)
        mask = (
            st_col.isin(["ENTRY READY", "PRE-TAP LIMIT", "REVERSAL WATCH"])
            | st_col.str.startswith("WATCHLIST")
        )
    else:
        mask = all_df["status"] == "ENTRY READY"
    cand_df = all_df[mask].copy()
    if cand_df.empty:
        return "Belum ada baris berstatus ENTRY READY di histori CSV.", pd.DataFrame()

    # Deduplikasi sinyal berulang dalam jendela 4 jam pada simbol & arah yang sama
    cand_df["sig_ms"] = cand_df.apply(
        lambda r: _parse_utc_ms(str(r.get("signal_candle_utc") or r.get("scan_utc") or "")),
        axis=1,
    )
    cand_df = cand_df.dropna(subset=["sig_ms"]).sort_values("sig_ms").reset_index(drop=True)

    dedup_rows = []
    last_seen: Dict[Tuple[str, str], int] = {}
    for _, row in cand_df.iterrows():
        key = (str(row["symbol"]), str(row["direction"]))
        t_ms = int(row["sig_ms"])
        if key in last_seen and (t_ms - last_seen[key]) < 4 * 3_600_000:
            continue
        last_seen[key] = t_ms
        dedup_rows.append(row)

    uniq_df = pd.DataFrame(dedup_rows)
    print(f"[Signal Tracker] Mengevaluasi {len(uniq_df)} sinyal unik dari {len(csv_files)} file histori scan ...", file=sys.stderr)

    max_bars = max(12, int(round(max_hold_hours * 4)) + 20)
    trade_records = []

    async with AsyncBinance(concurrency=14) as api:
        server_ms = await api.server_time()

        async def eval_one(row: pd.Series) -> dict:
            sym = str(row["symbol"])
            sig_open_ms = int(row["sig_ms"])
            after_close_ms = sig_open_ms + 900_000  # mulai 1 bar 15M sesudah candle sinyal close
            try:
                fut_df = await api.klines(sym, "15m", min(1000, max_bars), server_ms, start_time=after_close_ms)
            except Exception:
                fut_df = pd.DataFrame()

            entry_p = float(row["entry"])
            sl_p = float(row["stop_loss"])
            tp1_p = float(row["tp1"])
            tp2_p = float(row["tp2"])
            z_lo, z_hi = _parse_zone_bounds(str(row.get("entry_zone", "")), entry_p)
            status_str = str(row["status"])
            need_limit = status_str != "ENTRY READY"

            is_complete = (server_ms - after_close_ms) >= max_hold_hours * 3_600_000
            sim = simulate_trade_on_bars(
                direction=str(row["direction"]),
                entry_price=entry_p,
                poi_low=z_lo,
                poi_high=z_hi,
                sl=sl_p,
                tp1=tp1_p,
                tp2=tp2_p,
                require_limit_tap=need_limit,
                future_df=fut_df,
                max_hold_hours=max_hold_hours,
                fee_pct=fee_pct,
                is_historical_complete=is_complete,
            )
            risk_pct = abs(entry_p - sl_p) / max(entry_p, 1e-12) * 100.0
            return {
                "signal_utc": row.get("signal_candle_utc") or row.get("scan_utc"),
                "symbol": sym,
                "direction": row["direction"],
                "kind": row.get("kind", "UNKNOWN"),
                "status": status_str,
                "score": row.get("score", 0.0),
                "ai_decision": row.get("ai_decision", ""),
                "entry": entry_p,
                "sl": sl_p,
                "tp1": tp1_p,
                "tp2": tp2_p,
                "risk_pct": round(risk_pct, 3),
                "planned_rr": row.get("rr", NAN),
                "outcome": sim["outcome"],
                "tp1_hit": sim["tp1_hit"],
                "realized_r": sim["realized_r"],
                "hold_hours": sim["hold_hours"],
                "source_file": row["_src_file"],
            }

        results = await asyncio.gather(*(eval_one(r) for _, r in uniq_df.iterrows()))
        trade_records.extend(results)

    res_df = pd.DataFrame(trade_records)
    summary = _build_metrics_summary(
        res_df,
        f"SIGNAL PERFORMANCE TRACKER REPORT  |  Folder: {scan_dir}  |  Max Hold: {max_hold_hours:.0f}h  |  Split TP1(50%)+TP2(50%)",
    )

    os.makedirs(scan_dir, exist_ok=True)
    res_df.to_csv(os.path.join(scan_dir, "tracker_trades.csv"), index=False)
    with open(os.path.join(scan_dir, "tracker_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(summary)

    return summary, res_df


def run_signal_tracker(
    scan_dir: str = "scan_output",
    max_hold_hours: float = 48.0,
    include_watchlist: bool = False,
    fee_pct: float = 0.04,
) -> Tuple[str, pd.DataFrame]:
    return asyncio.run(_track_signals_async(scan_dir, max_hold_hours, include_watchlist, fee_pct))


async def _backtest_symbol_walk_forward(
    api: AsyncBinance,
    symbol: str,
    meta: dict,
    cfg: Config,
    days: int,
    server_ms: int,
    step_bars_15m: int = 4,
    fee_pct: float = 0.04,
    include_pretap: bool = False,
) -> List[dict]:
    bars_15m_test = max(96, int(days * 96))
    need_15m = bars_15m_test + cfg.limit_15m
    need_1h = int(days * 24) + cfg.limit_1h
    need_4h = int(days * 6) + cfg.limit_4h

    df4, df1, df15 = await asyncio.gather(
        api.klines_history(symbol, "4h", need_4h, server_ms),
        api.klines_history(symbol, "1h", need_1h, server_ms),
        api.klines_history(symbol, "15m", need_15m, server_ms),
    )
    if len(df15) < cfg.min_bars_15m + 40 or len(df1) < 80 or len(df4) < 60:
        return []

    trades: List[dict] = []
    start_t = max(cfg.min_bars_15m, len(df15) - bars_15m_test)
    max_hold_bars = max(8, int(round(cfg.max_hold_hours * 4)))
    cooldown_until_idx = -1

    close_4h = df4["close_time"].values.astype("int64")
    close_1h = df1["close_time"].values.astype("int64")
    close_15m = df15["close_time"].values.astype("int64")

    t = start_t
    while t < len(df15) - 8:
        if t < cooldown_until_idx:
            t += step_bars_15m
            continue

        cur_close_ms = int(close_15m[t])
        # Potong histori 4H dan 1H secara ketat <= cur_close_ms (100% bebas look-ahead bias!)
        idx4 = int(np.searchsorted(close_4h, cur_close_ms, side="right"))
        idx1 = int(np.searchsorted(close_1h, cur_close_ms, side="right"))
        if idx4 < 60 or idx1 < 60:
            t += step_bars_15m
            continue

        sub4 = df4.iloc[max(0, idx4 - cfg.limit_4h) : idx4]
        sub1 = df1.iloc[max(0, idx1 - cfg.limit_1h) : idx1]
        sub15 = df15.iloc[max(0, t + 1 - cfg.limit_15m) : t + 1]

        ctx4 = htf_context(sub4, cfg.htf_piv)
        ctx1 = htf_context(sub1, cfg.htf_piv)

        setup = None
        if cfg.strategy == "smc":
            setup = analyze_smc_symbol(symbol, sub15, ctx4, ctx1, meta, cfg, df4=sub4, df1=sub1)
        elif cfg.strategy == "breakout":
            setup = analyze_symbol(symbol, sub15, ctx4, ctx1, meta, cfg, df4=sub4, df1=sub1)
        else:
            s1 = analyze_smc_symbol(symbol, sub15, ctx4, ctx1, meta, cfg, df4=sub4, df1=sub1)
            s2 = analyze_symbol(symbol, sub15, ctx4, ctx1, meta, cfg, df4=sub4, df1=sub1)
            if s1 and s2:
                setup = s1 if s1.score >= s2.score else s2
            else:
                setup = s1 or s2

        valid_bt_statuses = ("ENTRY READY", "PRE-TAP LIMIT") if include_pretap else ("ENTRY READY",)
        if setup and setup.status in valid_bt_statuses:
            is_limit_pre = setup.status == "PRE-TAP LIMIT"
            fut_df = df15.iloc[t + 1 : min(len(df15), t + 1 + max_hold_bars + 18)]
            sim = simulate_trade_on_bars(
                direction=setup.direction,
                entry_price=setup.entry,
                poi_low=setup.poi_low,
                poi_high=setup.poi_high,
                sl=setup.sl,
                tp1=setup.tp1,
                tp2=setup.tp2,
                require_limit_tap=is_limit_pre,
                future_df=fut_df,
                max_hold_hours=cfg.max_hold_hours,
                fee_pct=fee_pct,
                is_historical_complete=(len(df15) - (t + 1)) >= max_hold_bars,
            )
            if sim["outcome"] not in ("UNFILLED", "NO_DATA"):
                trades.append({
                    "signal_utc": setup.candle_time,
                    "symbol": symbol,
                    "direction": setup.direction,
                    "archetype": setup.archetype,
                    "kind": setup.kind,
                    "pattern": setup.pattern,
                    "status": setup.status,
                    "score": round(setup.score, 1),
                    "entry": setup.entry,
                    "sl": setup.sl,
                    "tp1": setup.tp1,
                    "tp2": setup.tp2,
                    "risk_pct": round(setup.risk_pct, 3),
                    "sl_atr_1h": round(setup.sl_atr_1h_ratio, 2),
                    "planned_rr": round(setup.rr, 2),
                    "outcome": sim["outcome"],
                    "tp1_hit": sim["tp1_hit"],
                    "realized_r": sim["realized_r"],
                    "hold_hours": sim["hold_hours"],
                })
                bars_held = max(4, int(round(sim["hold_hours"] * 4)))
                cooldown_until_idx = t + bars_held + 2
                t = cooldown_until_idx
                continue

        t += step_bars_15m

    return trades


async def _run_backtest_async(
    cfg: Config,
    only_symbols: Optional[List[str]] = None,
    days: int = 14,
    max_symbols: int = 15,
    out_dir: str = "scan_output",
    fee_pct: float = 0.04,
    include_pretap: bool = False,
) -> Tuple[str, pd.DataFrame]:
    async with AsyncBinance(concurrency=min(16, max(4, cfg.workers))) as api:
        server_ms = await api.server_time()
        universe, _, _ = await api.build_universe(cfg, only_symbols)
        targets = universe if only_symbols else universe[:max_symbols]

        print(
            f"[Backtester] Menguji Walk-Forward Top-Down MTF ({cfg.strategy.upper()}) selama {days} hari terakhir "
            f"pada {len(targets)} simbol ({', '.join(s for s, _ in targets[:8])}{'...' if len(targets)>8 else ''}) ...",
            file=sys.stderr,
        )

        all_trades: List[dict] = []
        batches = await asyncio.gather(
            *(
                _backtest_symbol_walk_forward(api, sym, meta, cfg, days, server_ms, fee_pct=fee_pct, include_pretap=include_pretap)
                for sym, meta in targets
            )
        )
        for b in batches:
            all_trades.extend(b)

    df_bt = pd.DataFrame(all_trades)
    if not df_bt.empty:
        df_bt = df_bt.sort_values("signal_utc").reset_index(drop=True)

    summary = _build_metrics_summary(
        df_bt,
        f"WALK-FORWARD BACKTEST REPORT (TOP-DOWN 4H->1H->15M + ZOOM-OUT CHECK)  |  Periode: {days} Hari  |  Pairs: {len(targets)}",
    )
    os.makedirs(out_dir, exist_ok=True)
    if not df_bt.empty:
        df_bt.to_csv(os.path.join(out_dir, "backtest_results.csv"), index=False)
    with open(os.path.join(out_dir, "backtest_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(summary)

    return summary, df_bt


def run_walk_forward_backtest(
    cfg: Config,
    only_symbols: Optional[List[str]] = None,
    days: int = 14,
    max_symbols: int = 15,
    out_dir: str = "scan_output",
    fee_pct: float = 0.04,
    include_pretap: bool = False,
) -> Tuple[str, pd.DataFrame]:
    return asyncio.run(_run_backtest_async(cfg, only_symbols, days, max_symbols, out_dir, fee_pct, include_pretap))
