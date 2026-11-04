#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Binance USDⓈ-M Futures - Top-Down MTF SMC (4H -> 1H -> 15M), Orderflow, & Breakout Scanner
==========================================================================================

Arsitektur Upgrade v2.0:
1. Top-Down Multi-Timeframe Refinement (4H -> 1H -> 15M):
   - Lapis 4H : Peta 1-3 Macro Key Ranges & Tembok Supply/Demand 4H (Kompas & Pembatas TP).
   - Lapis 1H : Working Trading Range, Unmitigated OB/FVG/Breaker, EQH/EQL, & Estimasi Durasi Posisi.
   - Lapis 15M: Likuiditas kecil (BSL/SSL/Asian Sweep, IDM), CHOCH/BOS/CISD, Pin Bar, & Nested POI,
                dengan Filter Proporsionalitas "Zoom-Out Check" berjangkar ATR 1H (anti-SL sempit).
2. Data Orderflow & Derivatif Publik (Tanpa API Key):
   - Cumulative Volume Delta (CVD), Taker Buy Ratio, & Bull/Bear Absorption dari klines.
   - Open Interest Change (1H/4H), Top Trader L/S Ratio, & Global Retail L/S Ratio.
3. Full LLM Top-Down Evaluator (Groq):
   - Menerima data 3 Timeframe + Orderflow utuh dengan presisi harga adaptif.
   - Tahan tag <think>, rate-limit backoff otomatis, & Proteksi Skor 0 (fallback aman ke skor MTF Python).
4. Kecepatan AsyncIO + Real-Time WebSocket Watcher + Signal Tracker & Walk-Forward Backtester:
   - Scan asinkron cepat (`aiohttp`), pantau sentuhan POI detik itu juga (`--watch-ws`),
     evaluasi akurasi histori CSV (`--track`), dan uji historis walk-forward (`--backtest`).

Contoh:
    python binance.py                               # Scan lengkap Top-Down MTF + Radar Zona/Sweep + AI
    python binance.py --in-zone                     # Tampilkan HANYA koin yang sudah di/dekat POI atau habis Sweep
    python binance.py --radar                       # Mode cepat (~4s) Radar Live Demand/Supply/OB/FVG/Liq Sweep
    python binance.py --no-ai                       # Scan cepat murni mesin MTF Python (tanpa AI)
    python binance.py --watch-ws                    # Scan lalu pantau live sentuhan POI & Sweep via WebSocket
    python binance.py --loop --watch-ws --telegram  # Otomatis tiap 15M + live alert POI ke Telegram
    python binance.py --track                       # Evaluasi performa seluruh histori scan_output/*.csv
    python binance.py --backtest --days 14          # Backtest walk-forward 14 hari terakhir
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import requests

try:
    from ai_validator import load_env
    load_env()
except Exception:
    pass

from scanner_core.breakout import analyze_symbol
from scanner_core.exchange_async import AsyncBinance, Binance
from scanner_core.indicators import (
    atr_wilder,
    ema,
    find_pivots,
    htf_context,
    liquidity_score,
    rma,
    rsi_wilder,
    structure_state,
    trend_short,
    trend_text,
)
from scanner_core.models import NAN, STABLE_BASES, Config, Pat, Setup, ZoneRadarHit, clamp, fp, size_position, ts_utc
from scanner_core.mtf_smc import analyze_smc_symbol, extract_tf_structure
from scanner_core.tracker import run_signal_tracker, run_walk_forward_backtest
from scanner_core.ws_watcher import watch_setups_realtime
from scanner_core.zone_radar import scan_live_zone_radar


def _eval_strategy(
    sym: str,
    df4: pd.DataFrame,
    df1: pd.DataFrame,
    df15: pd.DataFrame,
    ctx4: dict,
    ctx1: dict,
    meta: dict,
    cfg: Config,
    deriv: Optional[dict] = None,
) -> Optional[Setup]:
    if cfg.strategy == "smc":
        return analyze_smc_symbol(sym, df15, ctx4, ctx1, meta, cfg, df4=df4, df1=df1, deriv=deriv)
    if cfg.strategy == "breakout":
        return analyze_symbol(sym, df15, ctx4, ctx1, meta, cfg, df4=df4, df1=df1, deriv=deriv)
    s_smc = analyze_smc_symbol(sym, df15, ctx4, ctx1, meta, cfg, df4=df4, df1=df1, deriv=deriv)
    s_bk = analyze_symbol(sym, df15, ctx4, ctx1, meta, cfg, df4=df4, df1=df1, deriv=deriv)
    if s_smc and s_bk:
        return s_smc if s_smc.score >= s_bk.score else s_bk
    return s_smc or s_bk


def process_symbol(api: Binance, sym: str, meta: dict, cfg: Config, server_ms: int) -> Tuple[str, Optional[Setup]]:
    """Jalur sinkron (fallback --sync & kompatibilitas mundur)."""
    df4 = api.klines(sym, "4h", cfg.limit_4h, server_ms)
    df1 = api.klines(sym, "1h", cfg.limit_1h, server_ms)
    df15 = api.klines(sym, "15m", cfg.limit_15m, server_ms)
    if len(df4) < 60 or len(df1) < 60 or len(df15) < cfg.min_bars_15m:
        return sym, None
    ctx4 = htf_context(df4, cfg.htf_piv)
    ctx1 = htf_context(df1, cfg.htf_piv)
    setup = _eval_strategy(sym, df4, df1, df15, ctx4, ctx1, meta, cfg)
    if setup is not None and cfg.fetch_derivatives:
        deriv = api.fetch_derivatives(sym)
        setup = _eval_strategy(sym, df4, df1, df15, ctx4, ctx1, meta, cfg, deriv=deriv)
    return sym, setup


# =====================================================================
#  LAPORAN & FORMAT OUTPUT
# =====================================================================
def tv_link(sym: str) -> str:
    return f"https://www.tradingview.com/chart/?symbol=BINANCE:{sym}.P&interval=15"


def trim(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def reason_reject(s: Setup, cfg: Config) -> str:
    if getattr(s, "ai_decision", "") == "REJECTED":
        return f"Ditolak Full LLM: {trim(getattr(s, 'ai_summary', 'Kriteria MTF SMC tidak terpenuhi'), 72)}"
    if s.status == "EXTENDED":
        return "Entry sudah jauh dari POI (tunggu pullback limit)"
    min_w = cfg.smc_watch_score if s.kind.startswith("SMC") else cfg.watch_score
    if s.score < min_w:
        return f"Score {s.score:.0f} < {min_w}"
    if s.rr < cfg.min_rr:
        return f"RR 1:{s.rr:.1f} < 1:{cfg.min_rr:.1f} (terbatas tembok HTF / durasi)"
    return "Menunggu konfirmasi trigger / ruang target belum memadai"


def describe(s: Setup, rank: int, btc: Optional[dict], cfg: Config) -> str:
    d = 1 if s.direction == "LONG" else -1
    m = s.meta
    L = []
    has_ai_eval = getattr(s, "ai_decision", "") in ("APPROVED", "CONDITIONAL", "REJECTED")
    score_label = f"Skor Full LLM: {s.score:.0f}/100 (MTF Py: {s.python_score:.0f})" if has_ai_eval else f"Skor MTF: {s.score:.0f}/100"

    L.append(f"#{rank}  {s.symbol}  {s.direction}  |  {s.pattern}  |  {s.status}  |  {score_label}")
    L.append(f"    Chart: {tv_link(s.symbol)}  |  Sesi: {s.session_name or '-'}")
    L.append(f"    Sinyal: {s.kind} pada candle 15M {s.candle_time} UTC (close terakhir: {s.last_candle} UTC | Harga saat ini: {fp(s.market_price)})")

    # Peta Top-Down 3 Lapis (4H -> 1H -> 15M)
    L.append("    0. Peta Top-Down Multi-Timeframe (4H -> 1H -> 15M) & Proporsionalitas Zoom-Out:")
    if s.range_4h_str or s.walls_4h_str:
        L.append(f"       - [4H Kompas & Tembok] Range 4H {s.range_4h_str} | {s.walls_4h_str}")
    if not math.isnan(s.range_1h_low) and not math.isnan(s.range_1h_high):
        pos1_str = f" ({s.range_1h_pos_pct:.1f}%)" if not math.isnan(s.range_1h_pos_pct) else ""
        L.append(
            f"       - [1H Working Range]   Low {fp(s.range_1h_low)} <-> High {fp(s.range_1h_high)} | "
            f"EQ {fp(s.range_1h_eq)}{pos1_str} | Estimasi Durasi: {s.est_hold_label or '-'}"
        )
    sl_rat = f"{s.sl_atr_1h_ratio:.2f}x ATR 1H" if not math.isnan(s.sl_atr_1h_ratio) else "-"
    L.append(
        f"       - [15M Zoom-Out Check] Zona Ideal POI [{s.entry_zone}] | Lebar SL: {s.risk_pct:.2f}% ({sl_rat} — proporsional 1H)"
    )
    if s.zone_badge or s.active_zones_summary:
        L.append(f"       - [Lokasi Harga Live]  {s.zone_badge} | {s.active_zones_summary}")
    if s.sweep_status or s.zone_reaction:
        mag_str = f" | Magnet: {s.eq_magnet_info}" if (s.eq_magnet_info and s.eq_magnet_info != "-") else ""
        L.append(f"       - [Sweep & Reaksi POI] Sweep: {s.sweep_status or '-'}{mag_str} | {s.zone_reaction or '-'}")
    if s.orderflow_summary:
        L.append(f"       - [Orderflow Publik]   {s.orderflow_summary}")

    if getattr(s, "smc_elements", None):
        el = s.smc_elements
        L.append("    1. Peta 10 Elemen Kunci SMC (Evaluasi Top-Down AI dari Data Real-Time Binance):")
        L.append(f"       [1] Order Blocks (4H/1H/15M) : {el.get('order_blocks', '-')}")
        L.append(f"       [2] Liquidity Sweeps & Pools : {el.get('liquidity_sweeps_runs', '-')}")
        L.append(f"       [3] Dealing Range (4H & 1H)  : {el.get('liquidity_range', '-')}")
        L.append(f"       [4] CHOCH (Change of Char)   : {el.get('choch', '-')}")
        L.append(f"       [5] BOS (Break of Structure) : {el.get('bos', '-')}")
        L.append(f"       [6] Inducement (IDM Level)   : {el.get('inducement', '-')}")
        L.append(f"       [7] Supply & Demand Walls    : {el.get('supply_demand', '-')}")
        L.append(f"       [8] Pin Bar & CVD Absorption : {el.get('pin_bar', '-')}")
        L.append(f"       [9] CISD & Open Interest     : {el.get('cisd', '-')}")
        L.append(f"       [10] High-Probability POI    : {el.get('poi', '-')}")
    elif s.parts:
        top = sorted(s.parts, key=lambda x: x[1] / max(x[2], 1), reverse=True)[:2]
        weak = sorted(s.parts, key=lambda x: x[1] / max(x[2], 1))[:1]
        L.append(
            f"    1. Evaluasi MTF: {s.pattern} (4H {s.trend_4h}; 1H {s.struct_1h}). "
            f"Pilar terkuat: {top[0][0]} ({top[0][1]:.0f}/{top[0][2]}), {top[1][0]} ({top[1][1]:.0f}/{top[1][2]}); "
            f"terlemah: {weak[0][0]} ({weak[0][1]:.0f}/{weak[0][2]})."
        )

    lv = f"zona [{fp(s.lower_line)} - {fp(s.upper_line)}]" if (not math.isnan(s.upper_line) and not math.isnan(s.lower_line)) else f"level {fp(s.level)}"
    opp_txt = fp(s.opp_level) if s.opp_level is not None else "tidak ditemukan"
    L.append(f"    2. Level & Target Lawan: {lv}; {'tembok resistance / BSL' if d == 1 else 'tembok support / SSL'} utama di {opp_txt}.")
    L.append(f"    3. Bukti Konfirmasi: {s.confirm_txt}")

    state = {
        "ENTRY READY": "SIAP EKSEKUSI (harga berada di area POI dengan konfluensi MTF, SL proporsional 1H, & RR memenuhi).",
        "EXTENDED": "terkonfirmasi tetapi sudah lari dari POI; pasang Limit di POI, jangan kejar market.",
        "REJECT": "ditolak karena ruang RR terhadap tembok 4H/1H atau skor belum memadai.",
        "WATCHLIST": "MASUK PANTAUAN (pasang Limit / tunggu konfirmasi pantulan di zona POI 1H/15M).",
    }.get(s.status, "menunggu konfirmasi di zona POI.")
    L.append(f"    4. Status Eksekusi: {state}")

    rr1_val = s.rr_tp1 if not math.isnan(s.rr_tp1) else (s.rr / 2.0)
    L.append(
        f"    5. Acuan Entry {fp(s.entry)} (Zona Limit POI: {s.entry_zone}) | SL {fp(s.sl)} (-{s.risk_pct:.2f}%) | "
        f"TP1 {fp(s.tp1)} (+{s.tp1_pct:.2f}%, RR 1:{rr1_val:.1f}) | TP2 {fp(s.tp2)} (+{s.tp2_pct:.2f}%, RR 1:{s.rr:.1f})."
    )
    if not math.isnan(s.qty):
        L.append(f"       Ukuran posisi (risiko {s.risk_usdt:.2f} USDT termasuk fee taker): qty {s.qty:g}, notional {s.notional:.2f} USDT.")
    L.append(f"    6. Invalidasi: {s.invalidation}")

    fund = m.get("funding", 0.0) * 100
    crowd = d * m.get("funding", 0.0)
    fund_txt = "membebani posisi searah" if crowd > 0.0003 else ("menguntungkan/dibayar" if crowd < -0.0002 else "normal")
    risks = [
        f"vol24j {m.get('vol24', 0) / 1e6:.0f} juta USDT, spread {m.get('spread_bps', 0):.1f} bps",
        f"funding {fund:+.4f}% ({fund_txt})",
        f"ATR 1H {s.atr_1h_pct * 100:.2f}% | ATR 15M {s.atr_pct * 100:.2f}%",
    ]
    if btc:
        btc_align = (btc["4h"]["st"] == d) or ((btc["4h"]["close"] > btc["4h"]["e50"]) == (d == 1))
        risks.append("BTC 4H " + ("searah" if btc_align else "berbeda arah") + " (konteks makro saja)")
    L.append("    7. Profil Likuiditas & Risiko: " + "; ".join(risks) + ".")
    for nt in s.notes:
        L.append(f"       Catatan: {nt}")

    if has_ai_eval:
        icon = "✅" if s.ai_decision == "APPROVED" else ("⚠️" if s.ai_decision == "CONDITIONAL" else "❌")
        mdl_str = f" ({s.ai_model})" if getattr(s, "ai_model", "") else ""
        L.append(f"    8. Penilaian Mandiri Full LLM (Groq{mdl_str}): {icon} {s.ai_decision} (Skor: {s.score:.0f}/100)")
        if s.ai_summary:
            L.append(f"       Analisis Institusional: {s.ai_summary}")
        if s.ai_action_note:
            L.append(f"       Rencana Eksekusi: {s.ai_action_note}")
    elif getattr(s, "ai_decision", "") == "PYTHON_FALLBACK":
        L.append(f"    8. Catatan AI: Kuota/koneksi LLM sedang sibuk; mempertahankan evaluasi penuh Top-Down MTF Python ({s.score:.0f}/100).")
    return "\n".join(L)


LIVE_ZONE_STATES = {"SWEEP_IN_POI", "IN_POI_REJECTING", "IN_POI_TESTING", "LIQ_SWEPT", "APPROACHING_POI"}
ZONE_SORT_ORDER = {
    "SWEEP_IN_POI": 0,
    "IN_POI_REJECTING": 1,
    "IN_POI_TESTING": 2,
    "LIQ_SWEPT": 3,
    "APPROACHING_POI": 4,
    "WAITING_PULLBACK": 5,
}


def _is_in_or_near_zone(s: Setup) -> bool:
    return getattr(s, "zone_state", "") in LIVE_ZONE_STATES


def build_report(
    results: list,
    cfg: Config,
    scan_ts: str,
    btc: Optional[dict],
    stats: dict,
    args,
    elapsed_s: float = 0.0,
    radar_hits: Optional[List[ZoneRadarHit]] = None,
) -> str:
    out = []
    out.append("=" * 136)
    spd = f"   |   Durasi: {elapsed_s:.1f}s" if elapsed_s > 0 else ""
    mode_tag = " [RADAR LIVE ZONE]" if cfg.radar_mode else (" [FILTER: IN/NEAR POI]" if cfg.only_in_zone else "")
    out.append(
        f"BINANCE USDT-M FUTURES TOP-DOWN MTF + ZONE & SWEEP RADAR (4H->1H->15M){mode_tag}   |   Strategi: {cfg.strategy.upper()}   |   Scan: {scan_ts} UTC{spd}"
    )
    out.append("=" * 136)
    out.append(
        f"Universe: {stats['total']} kontrak USDT-perp aktif -> {stats['liquid']} lolos likuiditas "
        f"(vol24 >= {cfg.min_vol_usdt / 1e6:.0f}M USDT, spread <= {cfg.max_spread_bps:g} bps) -> {stats['scanned']} dianalisa "
        f"({stats['errors']} gagal data)"
    )
    if btc:
        out.append(
            f"Konteks BTC: 4H {trend_text(btc['4h'])} | 1H {trend_text(btc['1h'])} | "
            f"candle close 4H {ts_utc(btc['4h']['last_time'])} UTC (referensi makro saja, altcoin dinilai independen)"
        )
    out.append("Metode Terpadu: 4H Walls -> 1H Working Range & Hold -> 15M Zoom-Out SL + Detektor Live Demand/Supply/OB/FVG/Liq Sweep.")
    out.append("")

    all_valid = [r for r in results if r.status == "ENTRY READY" or r.status.startswith("WATCHLIST")]
    if cfg.only_in_zone:
        valid = [r for r in all_valid if r.status == "ENTRY READY" or _is_in_or_near_zone(r)]
        hidden_far = len(all_valid) - len(valid)
    else:
        valid = list(all_valid)
        hidden_far = 0

    # Urutkan: ENTRY READY teratas, lalu di dalam WATCHLIST utamakan koin yang SUDAH di/dekat POI atau baru Sweep!
    valid.sort(
        key=lambda r: (
            0 if r.status == "ENTRY READY" else 1,
            0 if _is_in_or_near_zone(r) else 1,
            ZONE_SORT_ORDER.get(r.zone_state, 6) if r.status != "ENTRY READY" else 0,
            -r.score,
        )
    )
    longs = [r for r in valid if r.status == "ENTRY READY" and r.direction == "LONG"]
    shorts = [r for r in valid if r.status == "ENTRY READY" and r.direction == "SHORT"]
    watch = [r for r in valid if r.status.startswith("WATCHLIST")]
    watch_live_cnt = sum(1 for r in watch if _is_in_or_near_zone(r))
    rejected = [r for r in results if r.status not in ("ENTRY READY",) and not r.status.startswith("WATCHLIST") and r.status != "NO SETUP"]
    nosetup = [r for r in results if r.status == "NO SETUP"]

    def render_radar_section() -> None:
        r_list = radar_hits or []
        out.append("=" * 136)
        out.append(
            f"RADAR LOKASI HARGA LIVE — Koin yang Detik Ini Sedang di Demand / Supply / POI OB-FVG / Liquidity Sweep ({len(r_list)})"
        )
        out.append("=" * 136)
        if not r_list:
            out.append("  Belum ada koin yang detik ini tepat menginjak zona HTF/Fresh POI atau baru saja melakukan Liquidity Sweep.")
            out.append("")
            return
        r_rows = []
        for idx_r, rh in enumerate(r_list[:25], 1):
            dep_s = f"{rh.depth_pct:.0f}%" if not math.isnan(rh.depth_pct) else f"Jrk {rh.dist_pct:.2f}%"
            st_lbl = f"{rh.setup_status} ({rh.setup_score:.0f})" if rh.setup_status != "-" else "RADAR ONLY"
            r_rows.append({
                "#": idx_r,
                "Symbol": rh.symbol,
                "Bias": rh.direction,
                "Price": fp(rh.market_price),
                "Status Zona / Sweep": rh.radar_badge,
                "Zona Aktif (4H/1H/15M)": trim(rh.primary_zone, 36),
                "Depth/Jrk": dep_s,
                "Liquidity Sweep Aktif": trim(rh.sweep_info, 32),
                "Reaksi Candle & CVD": trim(rh.reaction_info, 34),
                "Setup Trade": st_lbl,
            })
        pd.set_option("display.width", 380)
        pd.set_option("display.max_columns", 50)
        pd.set_option("display.max_colwidth", 38)
        out.append(pd.DataFrame(r_rows).to_string(index=False))
        out.append("")

    if cfg.radar_mode:
        render_radar_section()

    if not valid:
        out.append("NO VALID SETUP: tidak ada kandidat setup penuh yang memenuhi kriteria saat ini.")
        if hidden_far > 0:
            out.append(f"  (Catatan: {hidden_far} kandidat WATCHLIST disembunyikan karena masih jauh dari POI; jalankan tanpa --in-zone untuk melihat semua.)")
        out.append("")
    else:
        if hidden_far > 0:
            out.append(f"[Filter --in-zone Aktif] Menampilkan {len(valid)} koin yang SUDAH di/dekat POI atau baru Sweep ({hidden_far} koin jauh disembunyikan).")
            out.append("")
        rows = []
        for i, r in enumerate(valid, 1):
            dec = getattr(r, "ai_decision", "")
            if dec in ("APPROVED", "CONDITIONAL", "REJECTED"):
                ai_verdict = f"{dec} ({r.ai_confidence:.0f}%)"
            elif dec == "PYTHON_FALLBACK":
                ai_verdict = "PY_MTF"
            else:
                ai_verdict = "-"

            hold_short = "-"
            if not math.isnan(r.est_hold_hours_tp1) and not math.isnan(r.est_hold_hours_tp2):
                hold_short = f"~{r.est_hold_hours_tp1:.0f}-{r.est_hold_hours_tp2:.0f}j"

            rows.append({
                "Rank": i,
                "Symbol": r.symbol,
                "Dir": r.direction,
                "Setup Type": trim(r.pattern, 30),
                "4H": r.trend_4h_short,
                "1H": r.struct_1h_short,
                "Price": fp(r.market_price),
                "Entry Zone (POI)": r.entry_zone,
                "Live Zone / Sweep": trim(r.zone_badge or "-", 28),
                "Stop Loss": fp(r.sl),
                "SL%": f"{r.risk_pct:.2f}%",
                "TP1": fp(r.tp1),
                "TP2": fp(r.tp2),
                "RR": f"1:{r.rr:.1f}",
                "Hold": hold_short,
                "Score": f"{r.score:.0f}",
                "AI": ai_verdict,
                "Status": r.status,
            })
        pd.set_option("display.width", 380)
        pd.set_option("display.max_columns", 50)
        pd.set_option("display.max_colwidth", 38)
        out.append(pd.DataFrame(rows).to_string(index=False))
        out.append("")

    def section(title: str, items: List[Setup]) -> None:
        out.append("-" * 136)
        out.append(f"{title}  ({len(items)})")
        out.append("-" * 136)
        if not items:
            out.append("  (kosong)")
        for r in items:
            hold_s = f"~{r.est_hold_hours_tp1:.0f}-{r.est_hold_hours_tp2:.0f}j" if not math.isnan(r.est_hold_hours_tp2) else "-"
            z_b = trim(r.zone_badge or "-", 27)
            out.append(
                f"  {r.symbol:<13} {r.direction:<5} {z_b:<27}  score {r.score:>3.0f}  "
                f"SL {r.risk_pct:>4.2f}%  RR 1:{r.rr:<3.1f}  Hold {hold_s:<7}  Now {fp(r.market_price):<10}  POI {r.entry_zone}"
            )
        out.append("")

    section("A. LONG - confirmed at POI (ENTRY READY)", longs)
    section("B. SHORT - confirmed at POI (ENTRY READY)", shorts)
    section(f"C. WATCHLIST - {watch_live_cnt} sedang di/dekat Zona atau habis Sweep, {len(watch) - watch_live_cnt} tunggu pullback", watch)

    if not cfg.radar_mode:
        render_radar_section()

    out.append("-" * 136)
    out.append(f"D. REJECTED  ({len(rejected)} kandidat tidak memenuhi ruang RR/skor, {len(nosetup)} pola mentah)")
    out.append("-" * 136)
    for r in sorted(rejected, key=lambda x: -x.score)[:12]:
        z_hint = f" [{r.zone_badge}]" if _is_in_or_near_zone(r) else ""
        out.append(f"  {r.symbol:<14} {r.direction:<5} {trim(r.pattern, 28):<28} score {r.score:>3.0f}{z_hint}  -> {reason_reject(r, cfg)}")
    if not rejected:
        out.append("  (tidak ada kandidat yang ditolak)")
    out.append("")

    if valid:
        out.append("=" * 136)
        out.append("ANALISA DETAIL TOP-DOWN MTF, LOKASI ZONA LIVE, & ORDERFLOW (maks. 5 kandidat teratas)")
        out.append("=" * 136)
        for i, r in enumerate(valid[:5], 1):
            out.append(describe(r, i, btc, cfg))
            out.append("")

    out.append("-" * 136)
    out.append("Catatan: Gunakan '--in-zone' untuk hanya melihat koin yang sudah di POI/Sweep, atau '--radar' untuk pemindaian zona kilat.")
    out.append("Scanner hanya memakai endpoint publik tanpa akses akun/order.")
    return "\n".join(out)


def results_to_df(results: list, scan_ts: str) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {
            "scan_utc": scan_ts,
            "symbol": r.symbol,
            "direction": r.direction,
            "kind": r.kind,
            "pattern": r.pattern,
            "status": r.status,
            "score": round(r.score, 1),
            "python_score": round(r.python_score, 1),
            "rr": round(r.rr, 2),
            "rr_tp1": round(r.rr_tp1, 2) if not math.isnan(r.rr_tp1) else NAN,
            "market_price": r.market_price,
            "entry": r.entry,
            "limit_entry": r.limit_entry,
            "entry_zone": r.entry_zone,
            "zone_state": r.zone_state,
            "zone_badge": r.zone_badge,
            "dist_to_poi_pct": round(r.dist_to_poi_pct, 3) if not math.isnan(r.dist_to_poi_pct) else NAN,
            "zone_depth_pct": round(r.zone_depth_pct, 1) if not math.isnan(r.zone_depth_pct) else NAN,
            "zone_fresh": r.zone_fresh,
            "sweep_status": r.sweep_status,
            "zone_reaction": r.zone_reaction,
            "active_zones_summary": r.active_zones_summary,
            "eq_magnet_info": r.eq_magnet_info,
            "poi_low": r.poi_low,
            "poi_high": r.poi_high,
            "stop_loss": r.sl,
            "tp1": r.tp1,
            "tp2": r.tp2,
            "risk_pct": round(r.risk_pct, 3),
            "sl_atr_1h_ratio": round(r.sl_atr_1h_ratio, 2) if not math.isnan(r.sl_atr_1h_ratio) else NAN,
            "tp1_pct": round(r.tp1_pct, 3),
            "tp2_pct": round(r.tp2_pct, 3),
            "est_hold_h_tp1": round(r.est_hold_hours_tp1, 1) if not math.isnan(r.est_hold_hours_tp1) else NAN,
            "est_hold_h_tp2": round(r.est_hold_hours_tp2, 1) if not math.isnan(r.est_hold_hours_tp2) else NAN,
            "est_hold_label": r.est_hold_label,
            "range_4h": r.range_4h_str,
            "walls_4h": r.walls_4h_str,
            "range_1h_low": r.range_1h_low,
            "range_1h_high": r.range_1h_high,
            "range_1h_pos_pct": round(r.range_1h_pos_pct, 1) if not math.isnan(r.range_1h_pos_pct) else NAN,
            "session": r.session_name,
            "cvd_state": r.cvd_state,
            "taker_buy_ratio": r.taker_buy_ratio,
            "oi_chg_1h_pct": round(r.oi_change_1h_pct, 2) if not math.isnan(r.oi_change_1h_pct) else NAN,
            "oi_chg_4h_pct": round(r.oi_change_4h_pct, 2) if not math.isnan(r.oi_change_4h_pct) else NAN,
            "top_ls_ratio": r.top_ls_ratio,
            "retail_ls_ratio": r.retail_ls_ratio,
            "signal_candle_utc": r.candle_time,
            "last_closed_candle_utc": r.last_candle,
            "trend_4h": r.trend_4h,
            "struct_1h": r.struct_1h,
            "confluences": r.confirm_short,
            "invalidation": r.invalidation,
            "rel_vol": round(r.rel_vol, 2),
            "rsi": round(r.rsi, 1),
            "vol24_usdt": r.meta.get("vol24"),
            "spread_bps": round(r.meta.get("spread_bps", NAN), 2),
            "funding": r.meta.get("funding"),
            "qty": r.qty,
            "notional": r.notional,
            "risk_usdt": r.risk_usdt,
            "ai_decision": getattr(r, "ai_decision", ""),
            "ai_confidence": getattr(r, "ai_confidence", NAN),
            "ai_summary": getattr(r, "ai_summary", ""),
            "ai_action_note": getattr(r, "ai_action_note", ""),
            "ai_model": getattr(r, "ai_model", ""),
            "chart": tv_link(r.symbol),
        }
        if getattr(r, "smc_elements", None):
            el = r.smc_elements
            row["smc_order_blocks"] = el.get("order_blocks", "")
            row["smc_liquidity_sweeps"] = el.get("liquidity_sweeps_runs", "")
            row["smc_liquidity_range"] = el.get("liquidity_range", "")
            row["smc_choch"] = el.get("choch", "")
            row["smc_bos"] = el.get("bos", "")
            row["smc_inducement"] = el.get("inducement", "")
            row["smc_supply_demand"] = el.get("supply_demand", "")
            row["smc_pin_bar"] = el.get("pin_bar", "")
            row["smc_cisd"] = el.get("cisd", "")
            row["smc_poi"] = el.get("poi", "")
        for name, val, _ in r.parts:
            row[f"score_{name}"] = round(val, 1)
        rows.append(row)
    return pd.DataFrame(rows)


# =====================================================================
#  TELEGRAM NOTIFIER
# =====================================================================
def send_telegram(text: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("[telegram] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID belum diatur, pesan dilewati.", file=sys.stderr)
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat, "text": text[:4000], "disable_web_page_preview": True},
            timeout=15,
        )
    except requests.RequestException as e:
        print(f"[telegram] gagal kirim: {e}", file=sys.stderr)


def telegram_summary(results: list, scan_ts: str, sent: set) -> Optional[str]:
    ready = [r for r in results if r.status == "ENTRY READY"]
    new = []
    for r in ready:
        key = f"{r.symbol}|{r.direction}|{r.kind}|{r.candle_time}"
        if key not in sent:
            sent.add(key)
            new.append(r)
    if not new:
        return None
    lines = [f"⚡ Top-Down MTF Scanner {scan_ts} UTC"]
    for r in sorted(new, key=lambda x: -x.score):
        z_line = f"Lokasi Live: {r.zone_badge} | Sweep: {r.sweep_status}\n" if getattr(r, "zone_badge", "") else ""
        lines.append(
            f"{r.direction} {r.symbol} (Now {fp(r.market_price)}) | {r.pattern}\n"
            f"{z_line}"
            f"POI: {r.entry_zone} (Entry {fp(r.entry)}) | SL {fp(r.sl)} (-{r.risk_pct:.2f}%)\n"
            f"TP1 {fp(r.tp1)} | TP2 {fp(r.tp2)} | RR 1:{r.rr:.1f} | {r.est_hold_label} | Score {r.score:.0f}"
        )
        if getattr(r, "ai_decision", "") in ("APPROVED", "CONDITIONAL"):
            icon = "✅" if r.ai_decision == "APPROVED" else "⚠️"
            lines.append(f"🤖 AI: {icon} {r.ai_decision} ({r.ai_confidence:.0f}%) | {r.ai_action_note}")
        lines.append(tv_link(r.symbol))
    lines.append("Verifikasi chart sebelum eksekusi.")
    return "\n".join(lines)


# =====================================================================
#  ALUR PEMINDAIAN ASINKRON & SINKRON
# =====================================================================
def _compute_symbol_radar(
    sym: str,
    df4: pd.DataFrame,
    df1: pd.DataFrame,
    df15: pd.DataFrame,
    ctx4: dict,
    ctx1: dict,
    cfg: Config,
    st: Optional[Setup],
) -> Optional[ZoneRadarHit]:
    map4 = extract_tf_structure(df4, cfg.htf_piv, "4H", lookback_bars=80)
    map1 = extract_tf_structure(df1, cfg.htf_piv, "1H", lookback_bars=96)
    map15 = extract_tf_structure(df15, cfg.piv_len, "15M", lookback_bars=95)
    return scan_live_zone_radar(sym, df15, ctx4, ctx1, map4, map1, map15, cfg, existing_setup=st)


async def _scan_universe_async(cfg: Config, only: Optional[List[str]]) -> Tuple[list, List[ZoneRadarHit], Optional[dict], dict, int]:
    async with AsyncBinance(concurrency=max(4, cfg.workers)) as api:
        server_ms = await api.server_time()
        scan_ts = dt.datetime.fromtimestamp(server_ms / 1000, dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{scan_ts} UTC] mengambil daftar kontrak (AsyncIO) ...", file=sys.stderr)

        universe, total, liquid = await api.build_universe(cfg, only)
        print(f"  {total} kontrak aktif, {liquid} lolos likuiditas, memindai paralel {len(universe)} simbol ...", file=sys.stderr)

        btc = None
        try:
            b4, b1 = await asyncio.gather(
                api.klines("BTCUSDT", "4h", cfg.limit_4h, server_ms),
                api.klines("BTCUSDT", "1h", cfg.limit_1h, server_ms),
            )
            if len(b4) >= 60 and len(b1) >= 60:
                btc = {"4h": htf_context(b4, cfg.htf_piv), "1h": htf_context(b1, cfg.htf_piv)}
        except Exception as e:
            print(f"  konteks BTC gagal diambil: {e}", file=sys.stderr)

        cached_bundles: Dict[str, tuple] = {}
        radar_map: Dict[str, ZoneRadarHit] = {}
        results: List[Setup] = []
        errors = 0

        async def worker_one(sym: str, meta: dict) -> Optional[Setup]:
            nonlocal errors
            try:
                df4, df1, df15 = await api.fetch_symbol_klines(sym, cfg, server_ms)
                if len(df4) < 60 or len(df1) < 60 or len(df15) < cfg.min_bars_15m:
                    return None
                ctx4 = htf_context(df4, cfg.htf_piv)
                ctx1 = htf_context(df1, cfg.htf_piv)
                st = _eval_strategy(sym, df4, df1, df15, ctx4, ctx1, meta, cfg)
                if st is not None:
                    cached_bundles[sym] = (df4, df1, df15, ctx4, ctx1, meta)
                rh = _compute_symbol_radar(sym, df4, df1, df15, ctx4, ctx1, cfg, st)
                if rh is not None:
                    radar_map[sym] = rh
                return st
            except Exception as e:
                errors += 1
                print(f"  [{sym}] gagal: {e}", file=sys.stderr)
                return None

        raw_setups = await asyncio.gather(*(worker_one(sym, meta) for sym, meta in universe))
        prelim = [s for s in raw_setups if s is not None]

        # Tahap 2: Perkaya data Orderflow Derivatif publik (OI & L/S Ratio) hanya untuk kandidat lolos
        if prelim and cfg.fetch_derivatives:
            print(f"  mengambil data Open Interest & Long/Short Ratio untuk {len(prelim)} kandidat ...", file=sys.stderr)
            deriv_map_list = await asyncio.gather(*(api.fetch_derivatives(s.symbol) for s in prelim))
            for st_old, drv in zip(prelim, deriv_map_list):
                df4, df1, df15, ctx4, ctx1, meta = cached_bundles[st_old.symbol]
                st_new = _eval_strategy(st_old.symbol, df4, df1, df15, ctx4, ctx1, meta, cfg, deriv=drv)
                final_st = st_new if st_new is not None else st_old
                results.append(final_st)
                if final_st.symbol in radar_map:
                    radar_map[final_st.symbol].setup_status = final_st.status
                    radar_map[final_st.symbol].setup_score = float(final_st.score)
        else:
            results = prelim

        stats = {"total": total, "liquid": liquid, "scanned": len(universe) - errors, "errors": errors}
        return results, list(radar_map.values()), btc, stats, server_ms


def _scan_universe_sync(cfg: Config, only: Optional[List[str]]) -> Tuple[list, List[ZoneRadarHit], Optional[dict], dict, int]:
    api = Binance()
    server_ms = api.server_time()
    scan_ts = dt.datetime.fromtimestamp(server_ms / 1000, dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{scan_ts} UTC] mengambil daftar kontrak (Sync) ...", file=sys.stderr)
    universe, total, liquid = api.build_universe(cfg, only)
    print(f"  {total} kontrak aktif, {liquid} lolos likuiditas, menganalisa {len(universe)} ...", file=sys.stderr)

    btc = None
    try:
        b4 = api.klines("BTCUSDT", "4h", cfg.limit_4h, server_ms)
        b1 = api.klines("BTCUSDT", "1h", cfg.limit_1h, server_ms)
        btc = {"4h": htf_context(b4, cfg.htf_piv), "1h": htf_context(b1, cfg.htf_piv)}
    except Exception as e:
        print(f"  konteks BTC gagal diambil: {e}", file=sys.stderr)

    results, radar_hits, errors, done = [], [], 0, 0

    def sync_worker(sym: str, meta: dict) -> Tuple[str, Optional[Setup], Optional[ZoneRadarHit]]:
        df4 = api.klines(sym, "4h", cfg.limit_4h, server_ms)
        df1 = api.klines(sym, "1h", cfg.limit_1h, server_ms)
        df15 = api.klines(sym, "15m", cfg.limit_15m, server_ms)
        if len(df4) < 60 or len(df1) < 60 or len(df15) < cfg.min_bars_15m:
            return sym, None, None
        ctx4 = htf_context(df4, cfg.htf_piv)
        ctx1 = htf_context(df1, cfg.htf_piv)
        setup = _eval_strategy(sym, df4, df1, df15, ctx4, ctx1, meta, cfg)
        if setup is not None and cfg.fetch_derivatives:
            deriv = api.fetch_derivatives(sym)
            setup = _eval_strategy(sym, df4, df1, df15, ctx4, ctx1, meta, cfg, deriv=deriv)
        rh = _compute_symbol_radar(sym, df4, df1, df15, ctx4, ctx1, cfg, setup)
        return sym, setup, rh

    with ThreadPoolExecutor(max_workers=max(2, min(10, cfg.workers))) as ex:
        futs = {ex.submit(sync_worker, sym, meta): sym for sym, meta in universe}
        for f in as_completed(futs):
            done += 1
            try:
                _, setup, rh = f.result()
                if setup is not None:
                    results.append(setup)
                if rh is not None:
                    radar_hits.append(rh)
            except Exception as e:
                errors += 1
                print(f"  [{futs[f]}] gagal: {e}", file=sys.stderr)
            if done % 20 == 0:
                print(f"  progres {done}/{len(universe)}", file=sys.stderr)

    stats = {"total": total, "liquid": liquid, "scanned": len(universe) - errors, "errors": errors}
    return results, radar_hits, btc, stats, server_ms


def run_scan(cfg: Config, args, sent: set) -> list:
    t0 = time.monotonic()
    only = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] if args.symbols else None

    if cfg.use_async:
        results, radar_hits, btc, stats, server_ms = asyncio.run(_scan_universe_async(cfg, only))
    else:
        results, radar_hits, btc, stats, server_ms = _scan_universe_sync(cfg, only)

    scan_ts = dt.datetime.fromtimestamp(server_ms / 1000, dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    if args.equity:
        for s in results:
            size_position(s, args.equity, args.risk_pct, args.fee_pct)

    if cfg.ai_audit and results:
        try:
            import ai_validator
            ai_validator.audit_top_setups(
                results,
                btc_ctx=btc,
                max_audit=cfg.ai_top_n,
                equity=args.equity,
                risk_pct=args.risk_pct,
                fee_pct=args.fee_pct,
            )
        except Exception as e:
            print(f"  [AI Validator] Peringatan: audit AI dilewati: {e}", file=sys.stderr)

    # Sinkronkan status & skor pasca-AI ke daftar Radar Live
    setup_by_sym = {s.symbol: s for s in results}
    for rh in radar_hits:
        if rh.symbol in setup_by_sym:
            st_ref = setup_by_sym[rh.symbol]
            rh.setup_status = st_ref.status
            rh.setup_score = float(st_ref.score)

    radar_hits.sort(
        key=lambda rh: (
            rh.priority,
            0 if rh.setup_status == "ENTRY READY" else (1 if rh.setup_status.startswith("WATCHLIST") else 2),
            rh.dist_pct,
            -rh.setup_score,
        )
    )
    setattr(args, "_last_radar_hits", radar_hits)

    results.sort(
        key=lambda r: (
            0 if r.status == "ENTRY READY" else (1 if r.status.startswith("WATCHLIST") else 2),
            0 if _is_in_or_near_zone(r) else 1,
            ZONE_SORT_ORDER.get(r.zone_state, 6) if r.status != "ENTRY READY" else 0,
            -r.score,
        )
    )

    elapsed = time.monotonic() - t0
    report = build_report(results, cfg, scan_ts, btc, stats, args, elapsed_s=elapsed, radar_hits=radar_hits)
    print(report)

    os.makedirs(args.out, exist_ok=True)
    stamp = dt.datetime.fromtimestamp(server_ms / 1000, dt.timezone.utc).strftime("%Y%m%d_%H%M")
    csv_path = os.path.join(args.out, f"scan_{stamp}.csv")
    if results:
        results_to_df(results, scan_ts).sort_values("score", ascending=False).to_csv(csv_path, index=False)
        print(f"\nHasil disimpan: {csv_path}")
    if radar_hits and cfg.radar_mode:
        radar_csv = os.path.join(args.out, f"radar_{stamp}.csv")
        pd.DataFrame([vars(rh) for rh in radar_hits]).to_csv(radar_csv, index=False)
        print(f"Radar Zona disimpan: {radar_csv}")
    with open(os.path.join(args.out, f"scan_{stamp}.txt"), "w", encoding="utf-8") as fh:
        fh.write(report)

    if args.telegram:
        msg = telegram_summary(results, scan_ts, sent)
        if msg:
            send_telegram(msg)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Binance USDT-M Futures Top-Down MTF SMC, Orderflow, & Breakout Scanner")
    p.add_argument("--strategy", type=str, default="smc", choices=["smc", "breakout", "all"],
                   help="strategi analisa: 'smc' (Top-Down MTF SMC), 'breakout' (konsolidasi), 'all' (keduanya)")
    p.add_argument("--top", type=int, default=Config.top_n, help="jumlah pair teratas berdasarkan volume 24j")
    p.add_argument("--min-vol", type=float, default=Config.min_vol_usdt / 1e6, help="volume 24 jam minimal (juta USDT)")
    p.add_argument("--max-spread-bps", type=float, default=Config.max_spread_bps, help="spread maksimal dalam bps")
    p.add_argument("--symbols", type=str, default="", help="analisa simbol tertentu saja, pisahkan koma")
    p.add_argument("--equity", type=float, default=0.0, help="equity akun (USDT) untuk hitung ukuran posisi")
    p.add_argument("--risk-pct", type=float, default=1.0, help="risiko per trade (%% dari equity)")
    p.add_argument("--fee-pct", type=float, default=0.04, help="fee per sisi (%%)")
    p.add_argument("--min-score", type=int, default=0, help="override score minimal ENTRY READY (default 70 SMC, 80 breakout)")
    p.add_argument("--max-hold", type=float, default=Config.max_hold_hours, help="batas ideal durasi posisi dalam jam (default 36)")
    p.add_argument("--recent-bars", type=int, default=Config.event_recent_bars, help="event aktif N bar 15M terakhir")
    p.add_argument("--workers", type=int, default=Config.workers, help="konkurensi koneksi async/worker")
    p.add_argument("--sync", action="store_true", help="gunakan mode HTTP sinkron klasik (tanpa aiohttp)")
    p.add_argument("--no-deriv", action="store_true", help="lewati pengambilan data Open Interest & Long/Short Ratio")
    p.add_argument("--no-ai", action="store_true", help="nonaktifkan evaluasi Full LLM (gunakan murni skor Top-Down MTF Python)")
    p.add_argument("--ai-top", type=int, default=15, help="jumlah kandidat teratas yang dievaluasi Full LLM (default 15)")
    p.add_argument("--ai-model", type=str, default="", help="override model Groq AI")
    p.add_argument("--out", type=str, default="scan_output", help="folder output CSV dan TXT")
    p.add_argument("--loop", action="store_true", help="jalan terus setiap penutupan candle 15M")
    p.add_argument("--watch-ws", action="store_true", help="pantau live sentuhan zona POI, CE 50%%, Sweep, TP, & SL via WebSocket")
    p.add_argument("--telegram", action="store_true", help="kirim notifikasi ke Telegram (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)")
    # Mode Detektor & Filter Lokasi Harga Live (Demand/Supply/POI/Sweep)
    p.add_argument("--in-zone", "--at-poi", dest="in_zone", action="store_true",
                   help="tampilkan hanya koin yang harganya SUDAH di dalam/dekat Demand/Supply/POI atau baru saja Liquidity Sweep")
    p.add_argument("--near-pct", type=float, default=Config.near_poi_pct,
                   help="ambang jarak %% maksimal dari bibir zona untuk status NEAR POI (default 0.50%%)")
    p.add_argument("--radar", action="store_true",
                   help="mode kilat Radar Zona & Liquidity Sweep Live (fokus koin yang sedang di Demand/Supply/OB/FVG/Sweep)")
    p.add_argument("--with-ai", action="store_true",
                   help="tetap jalankan evaluasi Groq LLM saat memakai mode cepat --radar")
    # Mode Evaluasi Performa & Backtester
    p.add_argument("--track", action="store_true", help="evaluasi performa aktual seluruh sinyal histori di scan_output/*.csv")
    p.add_argument("--include-watchlist", action="store_true", help="sertakan sinyal WATCHLIST (limit fill) saat menjalankan --track")
    p.add_argument("--backtest", action="store_true", help="jalankan Walk-Forward Backtest historis mesin Top-Down MTF")
    p.add_argument("--days", type=int, default=14, help="jumlah hari pengujian historis untuk --backtest (default 14)")
    p.add_argument("--bt-top", type=int, default=12, help="jumlah pair likuid teratas untuk --backtest jika --symbols kosong")
    args = p.parse_args()

    if args.ai_model:
        os.environ["GROQ_MODEL"] = args.ai_model

    smc_entry = args.min_score if args.min_score > 0 else 70
    bk_entry = args.min_score if args.min_score > 0 else 80
    use_ai = (not args.no_ai) and (not args.radar or args.with_ai)
    cfg = Config(
        strategy=args.strategy,
        top_n=args.top,
        min_vol_usdt=args.min_vol * 1e6,
        max_spread_bps=args.max_spread_bps,
        smc_entry_score=smc_entry,
        entry_score=bk_entry,
        max_hold_hours=args.max_hold,
        event_recent_bars=args.recent_bars,
        workers=args.workers,
        use_async=not args.sync,
        fetch_derivatives=not args.no_deriv,
        ai_audit=use_ai,
        ai_top_n=args.ai_top,
        only_in_zone=bool(args.in_zone or args.radar),
        radar_mode=bool(args.radar),
        near_poi_pct=max(0.05, args.near_pct),
    )

    # 1. Mode Signal Performance Tracker pada histori CSV
    if args.track:
        summary, _ = run_signal_tracker(
            scan_dir=args.out,
            max_hold_hours=args.max_hold,
            include_watchlist=args.include_watchlist,
            fee_pct=args.fee_pct,
        )
        print(summary)
        return

    # 2. Mode Walk-Forward Historical Backtest
    if args.backtest:
        only = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] if args.symbols else None
        summary, _ = run_walk_forward_backtest(
            cfg=cfg,
            only_symbols=only,
            days=args.days,
            max_symbols=args.bt_top,
            out_dir=args.out,
            fee_pct=args.fee_pct,
        )
        print(summary)
        return

    # 3. Mode Live Scan (+ opsional Real-Time WebSocket POI Watcher & Loop 15M)
    sent: set = set()
    notify_fn = send_telegram if args.telegram else None

    if not args.loop:
        results = run_scan(cfg, args, sent)
        r_hits = getattr(args, "_last_radar_hits", None)
        if args.watch_ws and (results or r_hits):
            try:
                asyncio.run(watch_setups_realtime(results, duration_seconds=None, notify_cb=notify_fn, radar_hits=r_hits))
            except KeyboardInterrupt:
                print("\n[WebSocket Watcher] Dihentikan oleh pengguna.", file=sys.stderr)
        return

    while True:
        results = []
        try:
            results = run_scan(cfg, args, sent)
        except Exception as e:
            print(f"scan gagal: {e}", file=sys.stderr)

        now = time.time()
        nxt = (math.floor(now / 900) + 1) * 900 + 20  # 20 detik setelah candle 15M close
        wait_sec = max(1.0, nxt - now)
        nxt_str = dt.datetime.fromtimestamp(nxt, dt.timezone.utc).strftime("%H:%M:%S")
        print(f"\nScan berikutnya {nxt_str} UTC ...", file=sys.stderr)

        r_hits = getattr(args, "_last_radar_hits", None)
        if args.watch_ws and (results or r_hits):
            try:
                asyncio.run(watch_setups_realtime(results, duration_seconds=wait_sec, notify_cb=notify_fn, radar_hits=r_hits))
            except KeyboardInterrupt:
                print("\n[Loop + WebSocket] Dihentikan oleh pengguna.", file=sys.stderr)
                break
        else:
            time.sleep(wait_sec)


if __name__ == "__main__":
    main()
