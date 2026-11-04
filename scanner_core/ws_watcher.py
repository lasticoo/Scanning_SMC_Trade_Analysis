# -*- coding: utf-8 -*-
"""
Real-Time WebSocket POI, Zone Mitigation, & Liquidity Sweep Watcher
===================================================================
Memantau harga live simbol kandidat & Radar Zona menggunakan multiplex stream sub-detik Binance:
`wss://fstream.binance.com/stream?streams=<sym1>@bookTicker/<sym2>@bookTicker/...`
(0 kuota REST API & respons sub-detik!).

Memberi notifikasi detik itu juga saat koin ENTRY READY / WATCHLIST / RADAR:
1. Mendekati bibir POI (<= 0.25%)             -> "⚡ APPROACHING POI"
2. Masuk Zona Demand/Supply/OB/FVG            -> "🎯 POI TAPPED (IN ZONE)"
3. Menyentuh 50% Mean Threshold / Equilibrium -> "💎 50% CE POI TOUCHED"
4. Menyapu kolam likuiditas (EQH/EQL/IDM)     -> "💧 LIVE LIQUIDITY SWEEP"
5. Mencapai TP1 (geser SL ke BEP)             -> "💰 TP1 HIT"
6. Mencapai TP2 penuh                         -> "🏆 TP2 FULL TARGET"
7. Menyentuh Stop Loss / Invalidasi           -> "🛑 SETUP INVALID / SL TERSENTUH"
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import math
import sys
import time
from typing import Callable, Dict, List, Optional

import websockets

from .models import Setup, ZoneRadarHit, fp


class SetupWatchState:
    def __init__(self, setup: Setup):
        self.s = setup
        in_now = getattr(setup, "zone_state", "") in ("FIRST_TOUCH_NOW", "SWEEP_IN_POI", "IN_POI_REJECTING", "IN_POI_TESTING")
        self.poi_tapped = bool(setup.confirmed) or in_now
        self.near_alerted = in_now or getattr(setup, "zone_state", "") in ("APPROACHING_POI", "PRE_TAP_NEAR")
        self.poi_alerted = False
        self.ce_alerted = False
        self.tp1_alerted = False
        self.closed = False

        # Ekstrak level magnet likuiditas (EQL/EQH/IDM) untuk deteksi live sweep
        raw = getattr(setup, "raw_market_data", {}) or {}
        if setup.direction == "LONG":
            pools = list(raw.get("eql_1h", [])) + list(raw.get("eql_15m", []))
        else:
            pools = list(raw.get("eqh_1h", [])) + list(raw.get("eqh_15m", []))
        idm = raw.get("idm_level")
        if idm is not None and not (isinstance(idm, float) and math.isnan(idm)):
            pools.append(float(idm))
        self.liq_pools = [float(x) for x in pools if x]
        self.swept_pierced: Dict[float, bool] = {}
        self.swept_alerted: set = set()


class RadarOnlyWatchState:
    def __init__(self, hit: ZoneRadarHit):
        self.hit = hit
        self.alerted = hit.dist_pct <= 0.0
        self.closed = False


async def watch_setups_realtime(
    setups: List[Setup],
    duration_seconds: Optional[float] = None,
    notify_cb: Optional[Callable[[str], None]] = None,
    radar_hits: Optional[List[ZoneRadarHit]] = None,
) -> None:
    """
    Memantau kandidat ENTRY READY, WATCHLIST, serta koin Radar Zona mendekati HTF secara real-time.
    Jika duration_seconds diisi (mis. jeda antar scan 15M pada --loop), otomatis selesai saat waktu habis.
    """
    active_list = [
        s for s in setups
        if (s.status in ("ENTRY READY", "PRE-TAP LIMIT", "REVERSAL WATCH") or s.status.startswith("WATCHLIST"))
        and not math.isnan(s.poi_low)
        and not math.isnan(s.poi_high)
        and not math.isnan(s.sl)
    ]

    states: Dict[str, SetupWatchState] = {s.symbol: SetupWatchState(s) for s in active_list[:85]}
    radar_only: Dict[str, RadarOnlyWatchState] = {}
    if radar_hits:
        for rh in radar_hits:
            if rh.symbol not in states and not math.isnan(rh.zone_low) and not math.isnan(rh.zone_high) and rh.dist_pct > 0.0:
                if len(states) + len(radar_only) < 100:
                    radar_only[rh.symbol] = RadarOnlyWatchState(rh)

    all_syms = list(states.keys()) + list(radar_only.keys())
    if not all_syms:
        if duration_seconds and duration_seconds > 0:
            await asyncio.sleep(duration_seconds)
        return

    last_eval_ts: Dict[str, float] = {}
    streams_param = "/".join(f"{sym.lower()}@bookTicker" for sym in all_syms)
    ws_url = f"wss://fstream.binance.com/stream?streams={streams_param}"

    deadline = (time.monotonic() + duration_seconds) if duration_seconds else None
    dur_txt = f"selama {int(duration_seconds)} detik hingga scan berikutnya" if duration_seconds else "secara kontinu (Ctrl+C untuk berhenti)"
    preview_syms = ", ".join(all_syms[:8]) + ("..." if len(all_syms) > 8 else "")
    extra_r = f" + {len(radar_only)} Radar HTF" if radar_only else ""
    print(f"\n[WebSocket Watcher] Memantau live {len(states)} setup POI{extra_r} ({preview_syms}) {dur_txt} ...", file=sys.stderr)

    def emit(msg: str) -> None:
        now_str = dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%S UTC")
        line = f"[{now_str}] {msg}"
        print(line, file=sys.stderr)
        if notify_cb:
            try:
                notify_cb(line)
            except Exception:
                pass

    while True:
        if deadline is not None and time.monotonic() >= deadline:
            break
        if all(st.closed for st in states.values()) and all(rt.closed for rt in radar_only.values()):
            if deadline is not None:
                rem = max(0.0, deadline - time.monotonic())
                await asyncio.sleep(rem)
            break

        try:
            async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20, close_timeout=5) as ws:
                while True:
                    if deadline is not None:
                        rem = deadline - time.monotonic()
                        if rem <= 0:
                            return
                        recv_timeout = min(5.0, rem)
                    else:
                        recv_timeout = 30.0

                    try:
                        raw_msg = await asyncio.wait_for(ws.recv(), timeout=recv_timeout)
                    except asyncio.TimeoutError:
                        continue

                    payload = json.loads(raw_msg)
                    if isinstance(payload, dict) and "data" in payload:
                        ticks = payload["data"] if isinstance(payload["data"], list) else [payload["data"]]
                    elif isinstance(payload, list):
                        ticks = payload
                    elif isinstance(payload, dict):
                        ticks = [payload]
                    else:
                        continue

                    for tick in ticks:
                        sym = tick.get("s")
                        if not sym:
                            continue

                        now_m = time.monotonic()
                        if now_m - last_eval_ts.get(sym, 0.0) < 0.25:
                            continue
                        last_eval_ts[sym] = now_m

                        try:
                            if "b" in tick and "a" in tick:
                                p = (float(tick["b"]) + float(tick["a"])) / 2.0
                            else:
                                p = float(tick.get("c") or 0.0)
                        except (ValueError, TypeError):
                            continue
                        if p <= 0:
                            continue

                        # A. Pantauan koin Radar HTF murni
                        if sym in radar_only:
                            rt = radar_only[sym]
                            if not rt.closed and not rt.alerted:
                                rh = rt.hit
                                if rh.zone_low <= p <= rh.zone_high:
                                    rt.alerted = True
                                    rt.closed = True
                                    emit(
                                        f"🎯 [RADAR ZONE HIT] {sym} ({rh.direction}) masuk {rh.primary_zone} di {fp(p)}! Cek reaksi candle 15M."
                                    )
                            continue

                        if sym not in states:
                            continue
                        st = states[sym]
                        if st.closed:
                            continue

                        s = st.s
                        is_long = s.direction == "LONG"

                        # 1. Cek Invalidasi / SL
                        if (is_long and p <= s.sl) or (not is_long and p >= s.sl):
                            tag = "SL TERSENTUH" if st.poi_tapped else "SETUP BATAL (SL ditembus sebelum retest)"
                            emit(f"🛑 [{tag}] {sym} ({s.direction}) harga {fp(p)} melewati batas SL {fp(s.sl)}")
                            st.closed = True
                            continue

                        # 2. Deteksi Live Liquidity Pool Sweep (EQL/EQH/IDM)
                        for lvl in st.liq_pools:
                            if lvl in st.swept_alerted:
                                continue
                            if is_long:
                                if p < lvl and p > s.sl:
                                    st.swept_pierced[lvl] = True
                                elif st.swept_pierced.get(lvl) and p >= lvl * 1.0004:
                                    st.swept_alerted.add(lvl)
                                    emit(f"💧 [LIVE SSL/EQL SWEEP] {sym} (LONG) baru menyapu kolam likuiditas {fp(lvl)} & memantul ke {fp(p)}!")
                            else:
                                if p > lvl and p < s.sl:
                                    st.swept_pierced[lvl] = True
                                elif st.swept_pierced.get(lvl) and p <= lvl * 0.9996:
                                    st.swept_alerted.add(lvl)
                                    emit(f"💧 [LIVE BSL/EQH SWEEP] {sym} (SHORT) baru menyapu kolam likuiditas {fp(lvl)} & turun ke {fp(p)}!")

                        # 3. Peringatan Dini Mendekati Bibir POI (<= 0.25%)
                        if not st.near_alerted and not st.poi_tapped:
                            edge = s.poi_high if is_long else s.poi_low
                            dist_edge_pct = abs(p - edge) / p * 100.0
                            if ((is_long and p > s.poi_high) or (not is_long and p < s.poi_low)) and dist_edge_pct <= 0.25:
                                st.near_alerted = True
                                emit(
                                    f"⚡ [APPROACHING POI] {sym} ({s.direction}) tinggal {dist_edge_pct:.2f}% dari bibir POI [{fp(s.poi_low)} - {fp(s.poi_high)}] (harga {fp(p)})!"
                                )

                        # 4. Cek Sentuhan Zona Ideal POI (Demand/Supply/OB/FVG)
                        in_zone = s.poi_low <= p <= s.poi_high
                        if in_zone and not st.poi_alerted:
                            st.poi_tapped = True
                            st.poi_alerted = True
                            fr_tag = " [Fresh POI]" if getattr(s, "zone_fresh", False) else ""
                            tac_tag = f" [{s.status} | {getattr(s, 'exec_style', '')}]" if getattr(s, "status", "") else ""
                            emit(
                                f"🎯 [POI TAPPED]{fr_tag}{tac_tag} {sym} ({s.direction} | {s.pattern}) masuk zona [{fp(s.poi_low)} - {fp(s.poi_high)}] "
                                f"di {fp(p)}! SL {fp(s.sl)} | TP1 {fp(s.tp1)} | TP2 {fp(s.tp2)} (RR 1:{s.rr:.1f})"
                            )

                        # 5. Cek Mitigasi 50% Mean Threshold / Equilibrium Zona POI
                        if in_zone and not st.ce_alerted and (s.poi_high > s.poi_low):
                            ce_mid = (s.poi_low + s.poi_high) / 2.0
                            if (is_long and p <= ce_mid) or (not is_long and p >= ce_mid):
                                st.ce_alerted = True
                                emit(
                                    f"💎 [50% CE POI] {sym} ({s.direction}) mencapai titik tengah 50% Equilibrium POI ({fp(ce_mid)}) di {fp(p)}!"
                                )

                        # 6. Cek TP1 & TP2 setelah setup aktif/tapped
                        if st.poi_tapped:
                            if not st.tp1_alerted and ((is_long and p >= s.tp1) or (not is_long and p <= s.tp1)):
                                st.tp1_alerted = True
                                emit(f"💰 [TP1 HIT] {sym} ({s.direction}) menyentuh TP1 {fp(s.tp1)} (harga {fp(p)}) -> Amankan parsial & geser SL ke BEP!")

                            if (is_long and p >= s.tp2) or (not is_long and p <= s.tp2):
                                emit(f"🏆 [TP2 FULL TARGET] {sym} ({s.direction}) mencapai TP2 {fp(s.tp2)} (harga {fp(p)})! R:R 1:{s.rr:.1f} tuntas.")
                                st.closed = True

        except asyncio.CancelledError:
            raise
        except Exception as e:
            if deadline is not None and time.monotonic() >= deadline:
                break
            print(f"[WebSocket Watcher] Koneksi terputus ({e}), menyambung ulang dalam 3 detik ...", file=sys.stderr)
            await asyncio.sleep(3.0)
