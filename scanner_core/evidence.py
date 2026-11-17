# -*- coding: utf-8 -*-
"""
Modul Bukti Objektif (Evidence Builder) untuk Kurator LLM SMC/ICT
=================================================================
Menyusun fakta struktural, efisiensi pergerakan, kunjungan POI, kolam likuiditas,
dan metrik spesifik archetype (HTF_REVERSAL / TREND_CONT / PRE_REACTION) tanpa
mengubah skor, status, atau level setup Python (100% netral).
Hanya bergantung pada pustaka standar Python, NumPy, dan Pandas.
"""
from __future__ import annotations

import datetime as dt
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _fp(val: Optional[float]) -> str:
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return "-"
    av = abs(float(val))
    if av >= 1000:
        return f"{val:.2f}"
    if av >= 1:
        return f"{val:.4f}"
    if av >= 0.01:
        return f"{val:.6f}"
    return f"{val:.8f}"


def _fast_atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> float:
    n_all = len(c)
    if n_all == 0:
        return 1e-12
    if n_all < 2:
        return max(float(h[0] - l[0]), 1e-12)
    win_slice = min(n_all, period * 2 + 1)
    hs = h[-win_slice:]
    ls = l[-win_slice:]
    cs = c[-win_slice:]
    prev_c = cs[:-1]
    tr1 = np.maximum(hs[1:] - ls[1:], np.maximum(np.abs(hs[1:] - prev_c), np.abs(ls[1:] - prev_c)))
    return max(float(np.mean(tr1)), 1e-12)


def _find_confirmed_swings(h: np.ndarray, l: np.ndarray, piv_len: int) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """Deteksi fractal pivot bebas look-ahead secara ter-vektorisasi (hanya menguji hingga n - piv_len)."""
    n = len(h)
    limit = n - piv_len
    if limit <= piv_len:
        return [], []

    mid_h = h[piv_len:limit]
    mid_l = l[piv_len:limit]
    is_hi = np.ones(limit - piv_len, dtype=bool)
    is_lo = np.ones(limit - piv_len, dtype=bool)
    for k in range(1, piv_len + 1):
        is_hi &= (mid_h >= h[piv_len - k : limit - k]) & (mid_h > h[piv_len + k : limit + k])
        is_lo &= (mid_l <= l[piv_len - k : limit - k]) & (mid_l < l[piv_len + k : limit + k])

    hi_idx = np.nonzero(is_hi)[0] + piv_len
    lo_idx = np.nonzero(is_lo)[0] + piv_len
    highs = [(int(idx), float(h[idx])) for idx in hi_idx]
    lows = [(int(idx), float(l[idx])) for idx in lo_idx]
    return highs, lows


def _causal_structure_state_machine(df: Optional[pd.DataFrame], piv_len: int, tf_label: str, max_bars: int = 260) -> Dict[str, Any]:
    """
    State machine kausal BOS / CHOCH / SWEEP:
    - Setiap swing high/low yang sudah terkonfirmasi (di i + piv_len) hanya boleh dipakai SEKALI
      (setelah tersapu/ditembus, level ditandai consumed).
    - Mengembalikan bias struktur aktual ("BULLISH", "BEARISH", "NETRAL"), daftar 4 event terakhir
      beserta umurnya dalam satuan bar, dan daftar event mentah untuk cek sweep -> MSS/BOS.
    """
    if df is None or len(df) < max(20, 2 * piv_len + 4):
        return {"tf": tf_label, "bias": "NETRAL", "events_last4": [], "raw_events": [], "highs": [], "lows": []}

    h = df["high"].to_numpy(dtype=float, copy=False)[-max_bars:]
    l = df["low"].to_numpy(dtype=float, copy=False)[-max_bars:]
    c = df["close"].to_numpy(dtype=float, copy=False)[-max_bars:]
    n = len(c)

    highs, lows = _find_confirmed_swings(h, l, piv_len)
    # Urutkan swing berdasarkan bar konfirmasinya (idx + piv_len)
    h_ptr = 0
    l_ptr = 0
    active_high: Optional[Tuple[int, float, bool]] = None  # (idx, price, consumed)
    active_low: Optional[Tuple[int, float, bool]] = None   # (idx, price, consumed)

    bias = 0  # 1 = BULLISH, -1 = BEARISH, 0 = NETRAL
    events: List[Dict[str, Any]] = []

    for bar in range(2 * piv_len, n):
        # Aktifkan pivot yang baru selesai dikonfirmasi pada bar ini (confirm_bar = idx + piv_len)
        while h_ptr < len(highs) and highs[h_ptr][0] + piv_len <= bar:
            s_idx, s_val = highs[h_ptr]
            active_high = (s_idx, s_val, False)
            h_ptr += 1
        while l_ptr < len(lows) and lows[l_ptr][0] + piv_len <= bar:
            s_idx, s_val = lows[l_ptr]
            active_low = (s_idx, s_val, False)
            l_ptr += 1

        # Uji interaksi terhadap active_high (hanya jika belum pernah dipakai)
        if active_high is not None and not active_high[2]:
            sw_idx, sw_lvl, _ = active_high
            if bar > sw_idx:
                if c[bar] > sw_lvl:
                    ev_type = "BOS_BULL" if bias == 1 else "CHOCH_BULL"
                    bias = 1
                    events.append({
                        "type": ev_type,
                        "dir": 1,
                        "bar": bar,
                        "age_bars": n - 1 - bar,
                        "level": round(sw_lvl, 8),
                    })
                    active_high = (sw_idx, sw_lvl, True)
                elif h[bar] > sw_lvl and c[bar] <= sw_lvl:
                    events.append({
                        "type": "BSL_SWEEP",
                        "dir": -1,
                        "bar": bar,
                        "age_bars": n - 1 - bar,
                        "level": round(sw_lvl, 8),
                    })
                    active_high = (sw_idx, sw_lvl, True)

        # Uji interaksi terhadap active_low (hanya jika belum pernah dipakai)
        if active_low is not None and not active_low[2]:
            sw_idx, sw_lvl, _ = active_low
            if bar > sw_idx:
                if c[bar] < sw_lvl:
                    ev_type = "BOS_BEAR" if bias == -1 else "CHOCH_BEAR"
                    bias = -1
                    events.append({
                        "type": ev_type,
                        "dir": -1,
                        "bar": bar,
                        "age_bars": n - 1 - bar,
                        "level": round(sw_lvl, 8),
                    })
                    active_low = (sw_idx, sw_lvl, True)
                elif l[bar] < sw_lvl and c[bar] >= sw_lvl:
                    events.append({
                        "type": "SSL_SWEEP",
                        "dir": 1,
                        "bar": bar,
                        "age_bars": n - 1 - bar,
                        "level": round(sw_lvl, 8),
                    })
                    active_low = (sw_idx, sw_lvl, True)

    bias_str = "BULLISH" if bias == 1 else ("BEARISH" if bias == -1 else "NETRAL")
    last4 = [
        f"{e['type']}@{_fp(e['level'])}({e['age_bars']}b)"
        for e in events[-4:]
    ]
    return {
        "tf": tf_label,
        "bias": bias_str,
        "events_last4": last4,
        "raw_events": events,
        "highs": highs,
        "lows": lows,
    }


def _efficiency_ratio(c: np.ndarray, window: int) -> float:
    if len(c) < 3:
        return 0.0
    w = min(window, len(c) - 1)
    seg = c[-(w + 1) :]
    net = abs(float(seg[-1] - seg[0]))
    path = float(np.sum(np.abs(np.diff(seg))))
    return round(net / max(path, 1e-12), 3)


def _count_poi_visits_24h(df15: pd.DataFrame, poi_low: float, poi_high: float) -> Tuple[int, Optional[int]]:
    """
    Hitung jumlah kunjungan (episode berurutan = 1) ke dalam zona [poi_low, poi_high]
    dalam 24 jam terakhir (96 bar 15M sebelum bar terakhir) beserta umur kunjungan terakhir (bar).
    """
    if df15 is None or len(df15) < 2 or math.isnan(poi_low) or math.isnan(poi_high):
        return 0, None
    h = df15["high"].to_numpy(dtype=float, copy=False)
    l = df15["low"].to_numpy(dtype=float, copy=False)
    n = len(h)
    start = max(0, n - 96)
    sub_h = h[start:n]
    sub_l = l[start:n]
    inside = (sub_l <= poi_high) & (sub_h >= poi_low)
    if not np.any(inside):
        return 0, None
    visits = int(inside[0]) + int(np.sum(inside[1:] & ~inside[:-1]))
    # Cari indeks bar terakhir yang menyentuh POI sebelum bar saat ini (atau bar saat ini)
    touch_indices = np.nonzero(inside)[0]
    last_rel_idx = int(touch_indices[-1])
    last_age = (len(inside) - 1) - last_rel_idx
    return visits, last_age


def _killzone_utc(df15: Optional[pd.DataFrame]) -> str:
    if df15 is None or len(df15) == 0 or "open_time" not in df15.columns:
        return "UNKNOWN"
    ts_ms = int(df15["open_time"].iloc[-1])
    hr = dt.datetime.fromtimestamp(ts_ms / 1000.0, dt.timezone.utc).hour
    if 7 <= hr < 11:
        return f"LONDON_KILLZONE ({hr:02d}:00 UTC)"
    if 12 <= hr < 16:
        return f"NY_KILLZONE ({hr:02d}:00 UTC)"
    if 0 <= hr < 7:
        return f"ASIAN_SESSION ({hr:02d}:00 UTC)"
    return f"OFF_KILLZONE ({hr:02d}:00 UTC)"


def _aligned_sweep_and_mss(st_machine: Dict[str, Any], d: int) -> Dict[str, Any]:
    evs = st_machine.get("raw_events", [])
    sw_type = "SSL_SWEEP" if d == 1 else "BSL_SWEEP"
    sweeps = [e for e in evs if e["type"] == sw_type]
    if not sweeps:
        return {"has_sweep": False, "sweep_age_bars": None, "sweep_level": None, "followed_by_mss_bos": False}
    last_sw = sweeps[-1]
    sw_bar = last_sw["bar"]
    after_mss = any(
        e["bar"] >= sw_bar and e["dir"] == d and ("CHOCH" in e["type"] or "BOS" in e["type"])
        for e in evs
    )
    return {
        "has_sweep": True,
        "sweep_age_bars": int(last_sw["age_bars"]),
        "sweep_level": float(last_sw["level"]),
        "followed_by_mss_bos": bool(after_mss),
    }


def build_evidence(
    setup: Any,
    df4: Optional[pd.DataFrame],
    df1: Optional[pd.DataFrame],
    df15: Optional[pd.DataFrame],
    btc: Optional[dict] = None,
) -> dict:
    """
    Membangun kamus fakta objektif & ringkasan teks siap-prompt tanpa memutasi setup.
    """
    direction = str(getattr(setup, "direction", "LONG")).upper()
    d = 1 if direction == "LONG" else -1
    archetype = str(getattr(setup, "archetype", "TREND_CONT")).upper()
    entry = float(getattr(setup, "entry", 0.0) or 0.0)
    sl = float(getattr(setup, "sl", 0.0) or 0.0)
    tp1 = float(getattr(setup, "tp1", 0.0) or 0.0)
    tp2 = float(getattr(setup, "tp2", 0.0) or 0.0)
    poi_low = float(getattr(setup, "poi_low", entry) or entry)
    poi_high = float(getattr(setup, "poi_high", entry) or entry)
    risk = max(abs(entry - sl), 1e-12)

    c15 = df15["close"].to_numpy(dtype=float, copy=False) if (df15 is not None and len(df15) > 0) else np.array([entry])
    h15 = df15["high"].to_numpy(dtype=float, copy=False) if (df15 is not None and len(df15) > 0) else np.array([entry])
    l15 = df15["low"].to_numpy(dtype=float, copy=False) if (df15 is not None and len(df15) > 0) else np.array([entry])
    v15 = df15["volume"].to_numpy(dtype=float, copy=False) if (df15 is not None and len(df15) > 0 and "volume" in df15.columns) else np.ones_like(c15)
    cur_p = float(c15[-1])
    atr15 = _fast_atr(h15, l15, c15, 14)

    c1 = df1["close"].to_numpy(dtype=float, copy=False) if (df1 is not None and len(df1) > 0) else c15
    h1 = df1["high"].to_numpy(dtype=float, copy=False) if (df1 is not None and len(df1) > 0) else h15
    l1 = df1["low"].to_numpy(dtype=float, copy=False) if (df1 is not None and len(df1) > 0) else l15
    atr1h = _fast_atr(h1, l1, c1, 14)

    # 1. Struktur kausal 4H / 1H / 15M (state machine BOS/CHOCH/SWEEP, level 1x pakai)
    sm4 = _causal_structure_state_machine(df4, 3, "4H")
    sm1 = _causal_structure_state_machine(df1, 3, "1H")
    sm15 = _causal_structure_state_machine(df15, 2, "15M")

    # 2. Sweep searah terakhir (1H & 15M) + apakah diikuti MSS/BOS searah
    sw_info_1h = _aligned_sweep_and_mss(sm1, d)
    sw_info_15m = _aligned_sweep_and_mss(sm15, d)

    # 3. Posisi harga di range swing 1H (premium/discount)
    sw1_hi_list = [p for _, p in sm1["highs"][-6:]]
    sw1_lo_list = [p for _, p in sm1["lows"][-6:]]
    r1_hi = max(sw1_hi_list) if sw1_hi_list else float(np.max(h1[-min(len(h1), 48) :]))
    r1_lo = min(sw1_lo_list) if sw1_lo_list else float(np.min(l1[-min(len(l1), 48) :]))
    if r1_hi <= r1_lo:
        r1_hi = float(np.max(h1[-min(len(h1), 48) :]))
        r1_lo = float(np.min(l1[-min(len(l1), 48) :]))
    r1_span = max(r1_hi - r1_lo, 1e-12)
    r1_pos_pct = round(min(100.0, max(0.0, (cur_p - r1_lo) / r1_span * 100.0)), 1)
    r1_zone = "DISCOUNT" if r1_pos_pct < 45.0 else ("PREMIUM" if r1_pos_pct > 55.0 else "EQUILIBRIUM")

    # 4. Jumlah kunjungan sebelumnya ke POI dalam 24 jam & umur kunjungan terakhir
    poi_visits_24h, last_visit_age_bars = _count_poi_visits_24h(df15, poi_low, poi_high)

    # 5. Efficiency Ratio 1H (24 bar) & 15M (32 bar)
    er_1h_24 = _efficiency_ratio(c1, 24)
    er_15m_32 = _efficiency_ratio(c15, 32)

    # 6. Killzone UTC & Tren BTC 4H dan 1H
    killzone = _killzone_utc(df15)
    btc_4h_trend = "NETRAL"
    btc_1h_trend = "NETRAL"
    if isinstance(btc, dict):
        b4_st = btc.get("4h", {}).get("st", 0) if isinstance(btc.get("4h"), dict) else 0
        b1_st = btc.get("1h", {}).get("st", 0) if isinstance(btc.get("1h"), dict) else 0
        btc_4h_trend = "BULLISH" if b4_st == 1 else ("BEARISH" if b4_st == -1 else "NETRAL")
        btc_1h_trend = "BULLISH" if b1_st == 1 else ("BEARISH" if b1_st == -1 else "NETRAL")

    # 7. Likuiditas nyata di arah target (swing 1H/4H) beserta RR terhadap SL & jarak TP2 ke pool terdekat (ATR1H)
    target_pools: List[Dict[str, Any]] = []
    for tf_name, sm_obj in (("1H", sm1), ("4H", sm4)):
        piv_source = sm_obj["highs"][-6:] if d == 1 else sm_obj["lows"][-6:]
        for idx_p, lvl_p in piv_source:
            if (d == 1 and lvl_p > entry) or (d == -1 and lvl_p < entry):
                pool_rr = round(abs(lvl_p - entry) / risk, 2)
                target_pools.append({"tf": tf_name, "level": round(lvl_p, 8), "rr_vs_sl": pool_rr})
    target_pools.sort(key=lambda x: abs(x["level"] - entry))
    if target_pools:
        tp2_nearest_pool_dist_atr1h = round(min(abs(tp2 - p["level"]) for p in target_pools) / atr1h, 2)
    else:
        tp2_nearest_pool_dist_atr1h = round(abs(tp2 - (r1_hi if d == 1 else r1_lo)) / atr1h, 2)

    # 8. Metrik Spesifik per Archetype
    reversal_metrics: Optional[Dict[str, Any]] = None
    sniper_metrics: Optional[Dict[str, Any]] = None
    pretap_metrics: Optional[Dict[str, Any]] = None

    raw_md = getattr(setup, "raw_market_data", {}) or {}
    sw_ext = float(getattr(setup, "sweep_entry_lvl", float("nan")))
    if math.isnan(sw_ext):
        sw_ext = float(raw_md.get("sweep_entry_lvl", float("nan")))

    if archetype == "HTF_REVERSAL":
        if math.isnan(sw_ext):
            if sw_info_1h["sweep_level"] is not None:
                sw_ext = float(sw_info_1h["sweep_level"])
            else:
                sw_ext = float(np.min(l1[-16:])) if d == 1 else float(np.max(h1[-16:]))
        if d == 1:
            imp_peak = float(np.max(h15[-20:]))
            imp_size = max(0.0, imp_peak - sw_ext)
            imp_atr1h = round(imp_size / atr1h, 2)
            retrace_pct = round((imp_peak - cur_p) / max(imp_size, 1e-12) * 100.0, 1) if imp_size > 0 else 0.0
            sweep_failed = bool(cur_p < sw_ext)
        else:
            imp_trough = float(np.min(l15[-20:]))
            imp_size = max(0.0, sw_ext - imp_trough)
            imp_atr1h = round(imp_size / atr1h, 2)
            retrace_pct = round((cur_p - imp_trough) / max(imp_size, 1e-12) * 100.0, 1) if imp_size > 0 else 0.0
            sweep_failed = bool(cur_p > sw_ext)
        reversal_metrics = {
            "sweep_extreme": round(sw_ext, 8),
            "impulse_size_atr1h": imp_atr1h,
            "retrace_pct": retrace_pct,
            "sweep_failed": sweep_failed,
        }
    elif archetype == "TREND_CONT":
        # Persentase pullback dari leg 1H terakhir
        if sm1["highs"] and sm1["lows"]:
            last_h = sm1["highs"][-1][1]
            last_l = sm1["lows"][-1][1]
            leg_span = max( abs(last_h - last_l), 1e-12)
            if d == 1:
                pb_pct = round((max(last_h, float(np.max(h1[-12:]))) - cur_p) / leg_span * 100.0, 1)
            else:
                pb_pct = round((cur_p - min(last_l, float(np.min(l1[-12:])))) / leg_span * 100.0, 1)
        else:
            pb_pct = round((r1_hi - cur_p) / r1_span * 100.0 if d == 1 else (cur_p - r1_lo) / r1_span * 100.0, 1)
        sniper_metrics = {
            "pullback_leg_1h_pct": pb_pct,
        }
    elif archetype == "PRE_REACTION":
        look4 = min(4, len(c15) - 1)
        vel_atr15_per_bar = round(abs(float(c15[-1] - c15[-1 - look4])) / max(look4 * atr15, 1e-12), 2) if look4 > 0 else 0.0
        rng4_atr15 = round(float(np.max(h15[-4:]) - np.min(l15[-4:])) / atr15, 2)
        v4 = float(np.mean(v15[-4:])) if len(v15) >= 4 else float(np.mean(v15))
        v24 = float(np.mean(v15[-24:])) if len(v15) >= 24 else float(np.mean(v15))
        vol_ratio_4_24 = round(v4 / max(v24, 1e-12), 2)
        lo_between = min(cur_p, poi_high if d == 1 else poi_low)
        hi_between = max(cur_p, poi_low if d == 1 else poi_high)
        liq_between = sum(
            1 for _, p_lvl in (sm15["lows"][-8:] if d == 1 else sm15["highs"][-8:])
            if lo_between < p_lvl < hi_between
        )
        pretap_metrics = {
            "approach_vel_atr15_per_bar": vel_atr15_per_bar,
            "range_4bar_atr15": rng4_atr15,
            "vol_ratio_4_24": vol_ratio_4_24,
            "liquidity_levels_between_price_and_poi": int(liq_between),
        }

    # 9. Flags netral (observasi faktual untuk LLM, bukan veto)
    flags: List[str] = []
    if (d == 1 and sm4["bias"] == "BEARISH") or (d == -1 and sm4["bias"] == "BULLISH"):
        flags.append(f"CAUSAL_4H_OPPOSITE({sm4['bias']})")
    if (d == 1 and r1_pos_pct > 55.0) or (d == -1 and r1_pos_pct < 45.0):
        flags.append(f"1H_POS_{r1_zone}({r1_pos_pct:.0f}%)")
    if poi_visits_24h >= 2:
        flags.append(f"POI_MULTIPLE_VISITS_24H({poi_visits_24h}x)")
    if tp2_nearest_pool_dist_atr1h > 1.0:
        flags.append(f"TP2_FAR_FROM_SWING_POOL({tp2_nearest_pool_dist_atr1h:.2f}xATR1H)")
    if reversal_metrics and reversal_metrics["sweep_failed"]:
        flags.append("SWEEP_EXTREME_BREACHED")
    if (sw_info_1h["has_sweep"] and int(sw_info_1h["sweep_age_bars"] or 0) > 60) or (
        sw_info_15m["has_sweep"] and int(sw_info_15m["sweep_age_bars"] or 0) > 60
    ):
        flags.append("SWEEP_STALE")
    if sniper_metrics and sniper_metrics["pullback_leg_1h_pct"] > 100.0:
        flags.append("PULLBACK_BEYOND_LEG")
    if pretap_metrics and pretap_metrics["approach_vel_atr15_per_bar"] >= 0.85:
        flags.append(f"FAST_APPROACH({pretap_metrics['approach_vel_atr15_per_bar']:.2f}ATR15/bar)")

    # 10. Format ringkasan siap-prompt
    pools_txt = (
        ", ".join(f"{p['tf']}@{_fp(p['level'])}(RR {p['rr_vs_sl']:.1f})" for p in target_pools[:4])
        if target_pools
        else "Belum ada swing pool searah"
    )
    sw1_txt = (
        f"Ada ({sw_info_1h['sweep_age_bars']}b lalu @ {_fp(sw_info_1h['sweep_level'])}, MSS/BOS sesudahnya: {'YA' if sw_info_1h['followed_by_mss_bos'] else 'BELUM'})"
        if sw_info_1h["has_sweep"]
        else "Tidak ada"
    )
    sw15_txt = (
        f"Ada ({sw_info_15m['sweep_age_bars']}b lalu @ {_fp(sw_info_15m['sweep_level'])}, MSS/BOS sesudahnya: {'YA' if sw_info_15m['followed_by_mss_bos'] else 'BELUM'})"
        if sw_info_15m["has_sweep"]
        else "Tidak ada"
    )
    visit_age_txt = f"{last_visit_age_bars}b lalu" if last_visit_age_bars is not None else "-"

    if reversal_metrics is not None:
        arch_line = (
            f"- Khusus Reversal: Impuls dari Sweep={reversal_metrics['impulse_size_atr1h']:.2f}xATR1H | "
            f"Retrace={reversal_metrics['retrace_pct']:.1f}% | Lewat Ekstrem Sweep (Gagal)={'YA' if reversal_metrics['sweep_failed'] else 'TIDAK'}"
        )
    elif sniper_metrics is not None:
        arch_line = f"- Khusus Sniper: Pullback dari Leg 1H Terakhir={sniper_metrics['pullback_leg_1h_pct']:.1f}%"
    elif pretap_metrics is not None:
        arch_line = (
            f"- Khusus Pre-Tap: Kecepatan={pretap_metrics['approach_vel_atr15_per_bar']:.2f}ATR15/bar | "
            f"Range4Bar={pretap_metrics['range_4bar_atr15']:.2f}xATR15 | Vol(4/24)={pretap_metrics['vol_ratio_4_24']:.2f}x | "
            f"Likuiditas Antara Harga-POI={pretap_metrics['liquidity_levels_between_price_and_poi']}"
        )
    else:
        arch_line = f"- Archetype: {archetype}"

    text_block = (
        f"[BUKTI KUALITAS TAMBAHAN DARI PYTHON — FAKTA OBJEKTIF]\n"
        f"- Struktur Kausal (1x Pakai): 4H={sm4['bias']} [{', '.join(sm4['events_last4']) or '-'}] | "
        f"1H={sm1['bias']} [{', '.join(sm1['events_last4']) or '-'}] | "
        f"15M={sm15['bias']} [{', '.join(sm15['events_last4']) or '-'}]\n"
        f"- Sweep Searah Terakhir: 1H={sw1_txt} | 15M={sw15_txt}\n"
        f"- Posisi Swing 1H: {r1_pos_pct:.1f}% ({r1_zone}) | Kunjungan POI 24j: {poi_visits_24h}x (terakhir: {visit_age_txt})\n"
        f"- Efficiency Ratio: 1H(24b)={er_1h_24:.2f}, 15M(32b)={er_15m_32:.2f} | Killzone: {killzone} | BTC: 4H={btc_4h_trend}, 1H={btc_1h_trend}\n"
        f"- Pool Likuiditas Target: {pools_txt} | Jarak TP2 ke Pool Terdekat: {tp2_nearest_pool_dist_atr1h:.2f}xATR1H\n"
        f"{arch_line}\n"
        f"- Flags Netral: {', '.join(flags) if flags else 'NONE'}"
    )

    return {
        "error": None,
        "causal_4h": {"bias": sm4["bias"], "events": sm4["events_last4"]},
        "causal_1h": {"bias": sm1["bias"], "events": sm1["events_last4"]},
        "causal_15m": {"bias": sm15["bias"], "events": sm15["events_last4"]},
        "sweep_1h": sw_info_1h,
        "sweep_15m": sw_info_15m,
        "swing_1h_pos_pct": r1_pos_pct,
        "swing_1h_zone": r1_zone,
        "poi_visits_24h": poi_visits_24h,
        "poi_last_visit_age_bars": last_visit_age_bars,
        "er_1h_24": er_1h_24,
        "er_15m_32": er_15m_32,
        "killzone_utc": killzone,
        "btc_4h_trend": btc_4h_trend,
        "btc_1h_trend": btc_1h_trend,
        "target_pools": target_pools[:6],
        "tp2_nearest_pool_dist_atr1h": tp2_nearest_pool_dist_atr1h,
        "reversal_metrics": reversal_metrics,
        "sniper_metrics": sniper_metrics,
        "pretap_metrics": pretap_metrics,
        "flags": flags,
        "text": text_block,
    }
