# -*- coding: utf-8 -*-
"""
Dynamic 3-Archetype Top-Down MTF SMC/ICT Engine (4H -> 1H -> 15M)
=================================================================
Menggabungkan 3 variasi setup institusional secara dinamis pada setiap emiten:
1. TREND_CONT   : Follow-trend setelah konfirmasi reaksi di POI (Sweep + CHOCH/BOS/CISD + Nested POI).
2. PRE_REACTION : Deteksi SEBELUM harga bereaksi saat mendekati Supply/Demand/OB/Breaker SEGAR (Unmitigated)
                  dengan jarak dekat (<= 1.65x ATR 1H / <= 2.6%) + ukur gaya kedatangan (Kompresi vs Impulsif).
3. HTF_REVERSAL : Deteksi potensi KETERBALIKAN TREN (Counter-Trend / Wall-Bounce Inversion) saat harga
                  mendekati/menguji Tembok Mayor 4H/1H di ekstrem Premium/Discount + Exhaustion/Sweep.

Seluruh variasi dikunci oleh:
- Filter Proporsionalitas "Zoom-Out Check" berjangkar ATR 1H (anti-SL sehelai rambut & anti-SL bengkak).
- Estimasi Durasi Posisi (Hold Time <= 36 jam) agar modal tidak mengendap terkena funding fee.
- Forward Playbook (Skenario Utama If-Then & Skenario Antisipasi Breaker Flip).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .evidence import build_evidence
from .indicators import (
    atr_wilder,
    compute_cvd_metrics,
    find_pivots,
    liquidity_score,
    rsi_wilder,
    session_liquidity_context,
    trend_short,
    trend_text,
)
from .models import MAJOR_SYMBOLS, NAN, Config, Setup, clamp, fp, ts_utc
from .zone_radar import (
    analyze_approach_dynamics,
    enrich_setup_with_zone_radar,
    get_live_price_snapshot,
)


def _count_visits(inside: np.ndarray) -> int:
    """Hitung jumlah kunjungan (episode kontak berurutan = 1 kunjungan), bukan jumlah bar."""
    b = np.asarray(inside, dtype=bool)
    if b.size == 0:
        return 0
    return int(b[0]) + int(np.sum(b[1:] & ~b[:-1]))


def extract_tf_structure(df: Optional[pd.DataFrame], piv_len: int, tf_label: str, lookback_bars: int = 90) -> Optional[dict]:
    """
    Mengekstrak peta struktur lengkap satu timeframe secara bebas look-ahead:
    - Confirmed Swings (Highs & Lows) + Dealing Range adaptif (0..100%)
    - Unmitigated & Tapped-1x Order Blocks (Bullish & Bearish) + kekuatan Displacement + FVG tag
    - Breaker Blocks (OB kuat yang ditembus impulsif dan berbalik fungsi + hitungan touches jujur)
    - Unmitigated Fair Value Gaps (FVG yang belum diisi penuh / belum invalid)
    - Equal Highs (EQH) & Equal Lows (EQL) sebagai kolam likuiditas magnet TP / Inducement
    """
    if df is None or len(df) < max(30, 2 * piv_len + 5):
        return None

    n = len(df)
    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    times = df["open_time"].values.astype("int64")

    atr_s = atr_wilder(df).values
    fallback_atr = float(np.nanmean(h[-14:] - l[-14:])) if n >= 14 else float(np.mean(h - l))
    cur_atr = float(atr_s[-1]) if not np.isnan(atr_s[-1]) and atr_s[-1] > 0 else max(fallback_atr, 1e-12)
    cur_p = float(c[-1])

    is_ph, is_pl = find_pivots(h, l, piv_len)
    highs = [{"idx": i, "price": float(h[i]), "time": int(times[i])} for i in range(n) if is_ph[i]]
    lows = [{"idx": i, "price": float(l[i]), "time": int(times[i])} for i in range(n) if is_pl[i]]

    # Dealing Range adaptif: ekspansi jendela swing jika 4 pivot terakhir terlalu sempit (< 2.5 ATR)
    tail_hi = float(np.max(h[-min(n, piv_len + 3) :]))
    tail_lo = float(np.min(l[-min(n, piv_len + 3) :]))
    range_high = max([s["price"] for s in highs[-4:]] + [tail_hi]) if highs else float(np.max(h[-min(n, 40) :]))
    range_low = min([s["price"] for s in lows[-4:]] + [tail_lo]) if lows else float(np.min(l[-min(n, 40) :]))

    for k_piv in (6, 10, 16):
        if range_high - range_low >= 2.5 * cur_atr:
            break
        if len(highs) >= 2:
            range_high = max(range_high, max(s["price"] for s in highs[-k_piv:]))
        if len(lows) >= 2:
            range_low = min(range_low, min(s["price"] for s in lows[-k_piv:]))

    if range_high - range_low < 2.2 * cur_atr:
        win = min(n, lookback_bars)
        range_high = max(range_high, float(np.max(h[-win:])))
        range_low = min(range_low, float(np.min(l[-win:])))

    span = max(range_high - range_low, 1e-12)
    eq_price = (range_high + range_low) / 2.0
    pos_pct = clamp((cur_p - range_low) / span * 100.0, 0.0, 100.0)

    # Rata-rata body 20 candle untuk definisi objektif Displacement (body >= 1.5x avg_body_20)
    bodies = np.abs(c - o)
    rngs = np.maximum(h - l, 1e-12)

    def _avg_body20(idx_bar: int) -> float:
        st_b = max(0, idx_bar - 20)
        sl_b = bodies[st_b:idx_bar]
        return float(np.mean(sl_b)) if len(sl_b) >= 5 else float(np.mean(bodies[-min(n, 20):]))

    # 1. Unmitigated Fair Value Gaps (FVG) — ukuran minimal ~0.18-0.20x ATR (HTF) / ~0.15x ATR (15M)
    fvg_min_mult = 0.18 if tf_label in ("4H", "1H") else 0.14
    start_idx = max(3, n - lookback_bars)
    fvgs: List[dict] = []
    raw_fvg_idx = set()
    for i in range(start_idx, n):
        a = cur_atr if np.isnan(atr_s[i]) or atr_s[i] <= 0 else float(atr_s[i])
        # Bullish FVG: Low candle ke-3 (i) > High candle ke-1 (i-2)
        if l[i] - h[i - 2] >= fvg_min_mult * a:
            raw_fvg_idx.add((i, 1))
            r_top, r_bot = float(l[i]), float(h[i - 2])
            if i + 1 <= n - 1:
                sub_c = c[i + 1 : n]
                sub_l = l[i + 1 : n]
                if np.min(sub_c) < r_bot or np.min(sub_l) <= r_bot + 0.08 * (r_top - r_bot):
                    continue
                eff_top = min(r_top, float(np.min(sub_l)))
            else:
                eff_top = r_top
            if eff_top - r_bot >= 0.10 * a:
                fvgs.append({
                    "tf": tf_label, "idx": i, "dir": 1, "top": eff_top, "bot": r_bot,
                    "raw_top": r_top, "ce": (eff_top + r_bot) / 2.0, "time": int(times[i]),
                    "touches": 0, "fresh": True, "disp_atr": round((r_top - r_bot) / max(a, 1e-12), 2), "has_fvg": True,
                    "kind": f"{tf_label} Bullish FVG",
                })
        # Bearish FVG: High candle ke-3 (i) < Low candle ke-1 (i-2)
        elif l[i - 2] - h[i] >= fvg_min_mult * a:
            raw_fvg_idx.add((i, -1))
            r_top, r_bot = float(l[i - 2]), float(h[i])
            if i + 1 <= n - 1:
                sub_c = c[i + 1 : n]
                sub_h = h[i + 1 : n]
                if np.max(sub_c) > r_top or np.max(sub_h) >= r_top - 0.08 * (r_top - r_bot):
                    continue
                eff_bot = max(r_bot, float(np.max(sub_h)))
            else:
                eff_bot = r_bot
            if r_top - eff_bot >= 0.10 * a:
                fvgs.append({
                    "tf": tf_label, "idx": i, "dir": -1, "top": r_top, "bot": eff_bot,
                    "raw_bot": r_bot, "ce": (r_top + eff_bot) / 2.0, "time": int(times[i]),
                    "touches": 0, "fresh": True, "disp_atr": round((r_top - r_bot) / max(a, 1e-12), 2), "has_fvg": True,
                    "kind": f"{tf_label} Bearish FVG",
                })

    # 2. Order Blocks (OB) dengan Validasi Displacement (body >= 1.5x avg_body_20, close 70%/30%) + BOS + FVG
    bull_obs: List[dict] = []
    bear_obs: List[dict] = []
    breakers: List[dict] = []

    for i in range(max(2, n - lookback_bars), n - 2):
        a = cur_atr if np.isnan(atr_s[i]) or atr_s[i] <= 0 else float(atr_s[i])
        end_d = min(n - 1, i + 3)
        avg_b20 = max(_avg_body20(i), 1e-12)

        # Kandidat Bullish OB: candle bearish terakhir sebelum impuls yang menyebabkan MSS/BOS
        if c[i] < o[i] and np.max(c[i + 1 : end_d + 1]) > h[i]:
            disp_move = float(np.max(c[i + 1 : end_d + 1])) - o[i + 1]
            has_fvg = any((k, 1) in raw_fvg_idx for k in range(i + 1, end_d + 2))
            has_disp_candle = any(
                (c[k] > o[k]) and (bodies[k] >= 1.5 * avg_b20) and ((c[k] - l[k]) / rngs[k] >= 0.70)
                for k in range(i + 1, end_d + 1)
            )
            prior_sw_h = [s["price"] for s in highs if s["idx"] < i]
            internal_lh = float(np.max(h[max(0, i - 5) : i + 1]))
            ref_lh = min(prior_sw_h[-1], internal_lh) if prior_sw_h else internal_lh
            caused_bos = bool(np.max(c[i + 1 : min(n, i + 10)]) > ref_lh) or (has_fvg and has_disp_candle)

            if has_fvg or has_disp_candle or disp_move >= 0.95 * a:
                ob_bot = float(l[i])
                ob_top = float(h[i]) if (h[i] - l[i]) <= 1.35 * a else float(min(h[i], max(o[i], l[i] + 0.85 * a)))
                swept = bool(l[i] < np.min(l[max(0, i - 4) : i]))
                disp_rat = round(disp_move / max(a, 1e-12), 2)

                broken_idx = None
                for j in range(end_d + 1, n):
                    if c[j] < ob_bot - 0.08 * a:
                        broken_idx = j
                        break

                if broken_idx is None:
                    touches = _count_visits(l[end_d + 1 : n] <= ob_top)
                    if touches <= 2:
                        bull_obs.append({
                            "tf": tf_label, "idx": i, "dir": 1, "top": ob_top, "bot": ob_bot,
                            "swept": swept, "touches": touches, "fresh": touches == 0,
                            "disp_atr": disp_rat, "has_fvg": bool(has_fvg),
                            "has_disp": bool(has_disp_candle), "caused_bos": caused_bos,
                            "time": int(times[i]),
                            "kind": f"{tf_label} Bullish OB (Sweep POI)" if swept else f"{tf_label} Bullish OB",
                        })
                else:
                    bj = broken_idx
                    if (o[bj] - c[bj]) >= 0.70 * a and (bj + 1 >= n or np.max(c[bj + 1 : n]) <= ob_top + 0.1 * a):
                        brk_touches = _count_visits(h[bj + 1 : n] >= ob_bot)
                        if brk_touches <= 2:
                            breakers.append({
                                "tf": tf_label, "idx": bj, "dir": -1, "top": ob_top, "bot": ob_bot,
                                "swept": swept, "touches": brk_touches, "fresh": brk_touches == 0,
                                "disp_atr": round(abs(o[bj] - c[bj]) / max(a, 1e-12), 2), "has_fvg": True,
                                "has_disp": True, "caused_bos": True,
                                "time": int(times[bj]),
                                "kind": f"{tf_label} Bearish Breaker Block",
                            })

        # Kandidat Bearish OB: candle bullish terakhir sebelum impuls turun yang menyebabkan MSS/BOS
        if c[i] > o[i] and np.min(c[i + 1 : end_d + 1]) < l[i]:
            disp_move = o[i + 1] - float(np.min(c[i + 1 : end_d + 1]))
            has_fvg = any((k, -1) in raw_fvg_idx for k in range(i + 1, end_d + 2))
            has_disp_candle = any(
                (c[k] < o[k]) and (bodies[k] >= 1.5 * avg_b20) and ((c[k] - l[k]) / rngs[k] <= 0.30)
                for k in range(i + 1, end_d + 1)
            )
            prior_sw_l = [s["price"] for s in lows if s["idx"] < i]
            internal_hl = float(np.min(l[max(0, i - 5) : i + 1]))
            ref_hl = max(prior_sw_l[-1], internal_hl) if prior_sw_l else internal_hl
            caused_bos = bool(np.min(c[i + 1 : min(n, i + 10)]) < ref_hl) or (has_fvg and has_disp_candle)

            if has_fvg or has_disp_candle or disp_move >= 0.95 * a:
                ob_top = float(h[i])
                ob_bot = float(l[i]) if (h[i] - l[i]) <= 1.35 * a else float(max(l[i], min(o[i], h[i] - 0.85 * a)))
                swept = bool(h[i] > np.max(h[max(0, i - 4) : i]))
                disp_rat = round(disp_move / max(a, 1e-12), 2)

                broken_idx = None
                for j in range(end_d + 1, n):
                    if c[j] > ob_top + 0.08 * a:
                        broken_idx = j
                        break

                if broken_idx is None:
                    touches = _count_visits(h[end_d + 1 : n] >= ob_bot)
                    if touches <= 2:
                        bear_obs.append({
                            "tf": tf_label, "idx": i, "dir": -1, "top": ob_top, "bot": ob_bot,
                            "swept": swept, "touches": touches, "fresh": touches == 0,
                            "disp_atr": disp_rat, "has_fvg": bool(has_fvg),
                            "has_disp": bool(has_disp_candle), "caused_bos": caused_bos,
                            "time": int(times[i]),
                            "kind": f"{tf_label} Bearish OB (Sweep POI)" if swept else f"{tf_label} Bearish OB",
                        })
                else:
                    bj = broken_idx
                    if (c[bj] - o[bj]) >= 0.70 * a and (bj + 1 >= n or np.min(c[bj + 1 : n]) >= ob_bot - 0.1 * a):
                        brk_touches = _count_visits(l[bj + 1 : n] <= ob_top)
                        if brk_touches <= 2:
                            breakers.append({
                                "tf": tf_label, "idx": bj, "dir": 1, "top": ob_top, "bot": ob_bot,
                                "swept": swept, "touches": brk_touches, "fresh": brk_touches == 0,
                                "disp_atr": round(abs(c[bj] - o[bj]) / max(a, 1e-12), 2), "has_fvg": True,
                                "has_disp": True, "caused_bos": True,
                                "time": int(times[bj]),
                                "kind": f"{tf_label} Bullish Breaker Block",
                            })

    # 3. Equal Highs (EQH) & Equal Lows (EQL) Liquidity Pools (selisih <= 0.25x ATR(14) atau <= 0.15%)
    eqh_pools: List[float] = []
    rec_h = highs[-12:]
    for idx_a in range(len(rec_h)):
        for idx_b in range(idx_a + 1, len(rec_h)):
            s1, s2 = rec_h[idx_a], rec_h[idx_b]
            if s2["idx"] - s1["idx"] < 2:
                continue
            lvl = max(s1["price"], s2["price"])
            tol_eq = max(0.25 * cur_atr, 0.0015 * lvl)
            if abs(s1["price"] - s2["price"]) <= tol_eq:
                after_slice = h[s2["idx"] + 1 : n]
                if len(after_slice) == 0 or float(np.max(after_slice)) <= lvl + 0.05 * cur_atr:
                    if not any(abs(lvl - x) <= 0.15 * cur_atr for x in eqh_pools):
                        eqh_pools.append(lvl)

    eql_pools: List[float] = []
    rec_l = lows[-12:]
    for idx_a in range(len(rec_l)):
        for idx_b in range(idx_a + 1, len(rec_l)):
            s1, s2 = rec_l[idx_a], rec_l[idx_b]
            if s2["idx"] - s1["idx"] < 2:
                continue
            lvl = min(s1["price"], s2["price"])
            tol_eq = max(0.25 * cur_atr, 0.0015 * lvl)
            if abs(s1["price"] - s2["price"]) <= tol_eq:
                after_slice = l[s2["idx"] + 1 : n]
                if len(after_slice) == 0 or float(np.min(after_slice)) >= lvl - 0.05 * cur_atr:
                    if not any(abs(lvl - x) <= 0.15 * cur_atr for x in eql_pools):
                        eql_pools.append(lvl)

    # 4. Objektif Sweep SSL/BSL (reclaim/reject <= 3 candle), Breakdown/Breakout (3-6 candle), & MSS/CHoCH + OTE 62%-79%
    ssl_sweeps: List[dict] = []
    bsl_sweeps: List[dict] = []
    scan_sw_win = min(n - 1, 24)
    for t_bar in range(max(piv_len + 2, n - scan_sw_win), n):
        a_t = cur_atr if np.isnan(atr_s[t_bar]) or atr_s[t_bar] <= 0 else float(atr_s[t_bar])
        # Cek Sweep SSL terhadap swing low / EQL sebelumnya
        prior_lows = [s["price"] for s in lows if s["idx"] <= t_bar - piv_len]
        if prior_lows:
            ref_ssl = min(prior_lows[-3:]) if len(prior_lows) >= 2 else prior_lows[-1]
            # Termasuk swing low terakhir jika ditembus sedalam >= 0.08x ATR
            cand_ssls = sorted(list({ref_ssl, prior_lows[-1]}))
            for lvl_s in cand_ssls:
                pierce = lvl_s - float(l[t_bar])
                if pierce >= 0.08 * a_t and float(l[t_bar - 1]) >= lvl_s - 0.02 * a_t:
                    # Cari reclaim (close kembali di atas level) dalam 1-3 candle
                    reclaim_k = None
                    for k_r in range(t_bar, min(n, t_bar + 3)):
                        if float(c[k_r]) >= lvl_s:
                            reclaim_k = k_r - t_bar + 1
                            break
                    if reclaim_k is not None:
                        ext_lo = float(np.min(l[max(0, t_bar - 1) : min(n, t_bar + reclaim_k + 1)]))
                        # Pastikan sejak reclaim hingga bar terakhir tidak ada close di bawah low sweep (Section C: Close di bawah low sweep = batal)
                        after_reclaim_c = c[min(n - 1, t_bar + reclaim_k) : n]
                        if len(after_reclaim_c) == 0 or float(np.min(after_reclaim_c)) >= ext_lo:
                            is_dbl = any(abs(prev["level"] - lvl_s) <= 0.25 * a_t and t_bar - prev["idx"] >= 2 for prev in ssl_sweeps)
                            ssl_sweeps.append({
                                "idx": t_bar, "level": float(lvl_s), "extreme": ext_lo,
                                "pierce_atr": round(pierce / max(a_t, 1e-12), 2),
                                "reclaim_bars": reclaim_k, "double_sweep": is_dbl,
                                "vol_spike": bool(t_bar >= 5 and df["volume"].iloc[t_bar] >= 1.35 * float(df["volume"].iloc[max(0, t_bar - 15) : t_bar].mean())),
                            })
                            break

        # Cek Sweep BSL terhadap swing high / EQH sebelumnya
        prior_highs = [s["price"] for s in highs if s["idx"] <= t_bar - piv_len]
        if prior_highs:
            ref_bsl = max(prior_highs[-3:]) if len(prior_highs) >= 2 else prior_highs[-1]
            cand_bsls = sorted(list({ref_bsl, prior_highs[-1]}), reverse=True)
            for lvl_b in cand_bsls:
                pierce = float(h[t_bar]) - lvl_b
                if pierce >= 0.08 * a_t and float(h[t_bar - 1]) <= lvl_b + 0.02 * a_t:
                    reject_k = None
                    for k_r in range(t_bar, min(n, t_bar + 3)):
                        if float(c[k_r]) <= lvl_b:
                            reject_k = k_r - t_bar + 1
                            break
                    if reject_k is not None:
                        ext_hi = float(np.max(h[max(0, t_bar - 1) : min(n, t_bar + reject_k + 1)]))
                        # Pastikan sejak reject hingga bar terakhir tidak ada close di atas high sweep (Section C: Close di atas high sweep = batal)
                        after_reject_c = c[min(n - 1, t_bar + reject_k) : n]
                        if len(after_reject_c) == 0 or float(np.max(after_reject_c)) <= ext_hi:
                            is_dbl = any(abs(prev["level"] - lvl_b) <= 0.25 * a_t and t_bar - prev["idx"] >= 2 for prev in bsl_sweeps)
                            bsl_sweeps.append({
                                "idx": t_bar, "level": float(lvl_b), "extreme": ext_hi,
                                "pierce_atr": round(pierce / max(a_t, 1e-12), 2),
                                "reclaim_bars": reject_k, "double_sweep": is_dbl,
                                "vol_spike": bool(t_bar >= 5 and df["volume"].iloc[t_bar] >= 1.35 * float(df["volume"].iloc[max(0, t_bar - 15) : t_bar].mean())),
                            })
                            break

    # Deteksi Breakdown (close di bawah swing low >= 3 candle tanpa reclaim) & Breakout (close di atas swing high >= 3 candle)
    is_breakdown = False
    if lows and n >= 5:
        ref_l_bd = lows[-1]["price"]
        if lows[-1]["idx"] <= n - 4 and all(float(c[k]) < ref_l_bd for k in range(n - 3, n)):
            is_breakdown = True

    is_breakout = False
    if highs and n >= 5:
        ref_h_bo = highs[-1]["price"]
        if highs[-1]["idx"] <= n - 4 and all(float(c[k]) > ref_h_bo for k in range(n - 3, n)):
            is_breakout = True

    # Deteksi MSS / CHoCH Bullish & Bearish (dalam <= 20 candle setelah sweep/low) + Range Impuls & OTE 62%-79%
    mss_bull: Optional[dict] = None
    mss_bear: Optional[dict] = None

    if lows:
        sw_anchor_idx = ssl_sweeps[-1]["idx"] if ssl_sweeps else lows[-1]["idx"]
        sw_low_val = ssl_sweeps[-1]["extreme"] if ssl_sweeps else float(np.min(l[max(0, sw_anchor_idx - 1) : n]))
        if n - 1 - sw_anchor_idx <= 20:
            # Cari lower high internal sebelum atau di sekitar sweep
            cand_lh = [s["price"] for s in highs if s["idx"] <= sw_anchor_idx + 2 and s["price"] > sw_low_val]
            internal_lh = cand_lh[-1] if cand_lh else float(np.max(h[max(0, sw_anchor_idx - 6) : max(1, sw_anchor_idx + 1)]))
            for m_idx in range(max(sw_anchor_idx, 2), n):
                avg_b_m = max(_avg_body20(m_idx), 1e-12)
                disp_ok = (bodies[m_idx] >= 1.35 * avg_b_m and (c[m_idx] - l[m_idx]) / rngs[m_idx] >= 0.65) or any(
                    (k_f, 1) in raw_fvg_idx for k_f in range(max(2, m_idx - 1), min(n, m_idx + 2))
                )
                if float(c[m_idx]) > internal_lh and disp_ok:
                    imp_hi = float(np.max(h[sw_anchor_idx : n]))
                    imp_span = max(imp_hi - sw_low_val, 1e-12)
                    ote_top = imp_hi - 0.62 * imp_span
                    ote_bot = imp_hi - 0.79 * imp_span
                    mss_fvg = any(f["dir"] == 1 and f["idx"] >= sw_anchor_idx for f in fvgs)
                    mss_bull = {
                        "idx": m_idx, "level": float(internal_lh),
                        "sweep_low": sw_low_val, "impulse_high": imp_hi,
                        "ote_bot": ote_bot, "ote_top": ote_top,
                        "has_fvg": mss_fvg, "has_disp": True,
                        "bars_since_sweep": m_idx - sw_anchor_idx,
                    }
                    break

    if highs:
        sw_anchor_h_idx = bsl_sweeps[-1]["idx"] if bsl_sweeps else highs[-1]["idx"]
        sw_high_val = bsl_sweeps[-1]["extreme"] if bsl_sweeps else float(np.max(h[max(0, sw_anchor_h_idx - 1) : n]))
        if n - 1 - sw_anchor_h_idx <= 20:
            cand_hl = [s["price"] for s in lows if s["idx"] <= sw_anchor_h_idx + 2 and s["price"] < sw_high_val]
            internal_hl = cand_hl[-1] if cand_hl else float(np.min(l[max(0, sw_anchor_h_idx - 6) : max(1, sw_anchor_h_idx + 1)]))
            for m_idx in range(max(sw_anchor_h_idx, 2), n):
                avg_b_m = max(_avg_body20(m_idx), 1e-12)
                disp_ok = (bodies[m_idx] >= 1.35 * avg_b_m and (c[m_idx] - l[m_idx]) / rngs[m_idx] <= 0.35) or any(
                    (k_f, -1) in raw_fvg_idx for k_f in range(max(2, m_idx - 1), min(n, m_idx + 2))
                )
                if float(c[m_idx]) < internal_hl and disp_ok:
                    imp_lo = float(np.min(l[sw_anchor_h_idx : n]))
                    imp_span = max(sw_high_val - imp_lo, 1e-12)
                    ote_bot = imp_lo + 0.62 * imp_span
                    ote_top = imp_lo + 0.79 * imp_span
                    mss_fvg = any(f["dir"] == -1 and f["idx"] >= sw_anchor_h_idx for f in fvgs)
                    mss_bear = {
                        "idx": m_idx, "level": float(internal_hl),
                        "sweep_high": sw_high_val, "impulse_low": imp_lo,
                        "ote_bot": ote_bot, "ote_top": ote_top,
                        "has_fvg": mss_fvg, "has_disp": True,
                        "bars_since_sweep": m_idx - sw_anchor_h_idx,
                    }
                    break

    return {
        "tf": tf_label,
        "atr": cur_atr,
        "close": cur_p,
        "highs": highs,
        "lows": lows,
        "range_high": range_high,
        "range_low": range_low,
        "eq": eq_price,
        "pos_pct": pos_pct,
        "bull_obs": bull_obs,
        "bear_obs": bear_obs,
        "breakers": breakers,
        "fvgs": fvgs,
        "eqh": sorted(eqh_pools),
        "eql": sorted(eql_pools, reverse=True),
        "ssl_sweeps": ssl_sweeps,
        "bsl_sweeps": bsl_sweeps,
        "is_breakdown": is_breakdown,
        "is_breakout": is_breakout,
        "mss_bull": mss_bull,
        "mss_bear": mss_bear,
    }


def _zones_overlap(z1: dict, z2: dict, tol: float) -> bool:
    return (z1["bot"] - tol) <= z2["top"] and (z2["bot"] - tol) <= z1["top"]


def _refine_zone_bounds(raw_lo: float, raw_hi: float, d: int, cur_p: float, max_w: float, min_w: float) -> Tuple[float, float]:
    """
    Merampingkan zona HTF yang terlalu lebar tanpa mendorong cur_p keluar dari zona
    apabila harga saat ini SUDAH berada di dalam zona tersebut.
    """
    lo, hi = (raw_hi, raw_lo) if raw_hi < raw_lo else (raw_lo, raw_hi)
    if max_w > 0 and (hi - lo) > max_w:
        if lo <= cur_p <= hi:
            if d == 1:
                hi = min(hi, max(cur_p + 0.25 * max_w, lo + max_w))
                lo = max(raw_lo, hi - max_w)
            else:
                lo = max(lo, min(cur_p - 0.25 * max_w, hi - max_w))
                hi = min(raw_hi, lo + max_w)
        else:
            if d == 1:
                lo = hi - max_w
            else:
                hi = lo + max_w

    if min_w > 0 and (hi - lo) < min_w:
        mid = (lo + hi) / 2.0
        lo = mid - 0.5 * min_w
        hi = mid + 0.5 * min_w
    return lo, hi


def _format_hold_duration(h_tp1: float, h_tp2: float) -> str:
    lo = max(0.5, min(h_tp1, h_tp2 * 0.6))
    hi = max(lo + 1.0, h_tp2)
    if hi <= 6.0:
        tag = "Scalp-Intraday"
    elif hi <= 18.0:
        tag = "Intraday"
    elif hi <= 36.0:
        tag = "Short-Swing"
    else:
        tag = "Swing Panjang (pantau funding)"
    return f"{tag} (~{lo:.0f}-{hi:.0f} jam)"


def _format_orderflow_summary(cvd: dict, deriv: dict, funding: float, d: int) -> Tuple[str, float]:
    """Menyusun ringkasan orderflow publik & memberi skor konfluensi tambahan (0..6 poin)."""
    pts = 0.0
    tags = [cvd.get("cvd_state", "CVD Netral")]

    if d == 1:
        if cvd.get("bull_absorption"):
            pts += 2.5
        elif cvd.get("cvd_dir") == 1:
            pts += 1.5
    else:
        if cvd.get("bear_absorption"):
            pts += 2.5
        elif cvd.get("cvd_dir") == -1:
            pts += 1.5

    oi_1h = deriv.get("oi_change_1h_pct", NAN) if deriv else NAN
    oi_4h = deriv.get("oi_change_4h_pct", NAN) if deriv else NAN
    top_ls = deriv.get("top_ls_ratio", NAN) if deriv else NAN
    ret_ls = deriv.get("retail_ls_ratio", NAN) if deriv else NAN

    if not math.isnan(oi_1h):
        oi_str = f"OI 1H {oi_1h:+.2f}%"
        if not math.isnan(oi_4h):
            oi_str += f" (4H {oi_4h:+.2f}%)"
        tags.append(oi_str)
        if oi_1h >= 0.5:
            pts += 1.5
        elif oi_1h <= -1.5:
            tags.append("Likuidasi/Deleveraging")

    if not math.isnan(top_ls):
        tags.append(f"TopTrader L/S {top_ls:.2f}")
        if (d == 1 and top_ls >= 1.15) or (d == -1 and top_ls <= 0.90):
            pts += 1.0

    if not math.isnan(ret_ls):
        tags.append(f"Retail L/S {ret_ls:.2f}")
        if (d == 1 and ret_ls <= 0.85) or (d == -1 and ret_ls >= 1.85):
            pts += 1.0
            tags.append("Fuel Kontrarian Retail")

    if d * funding < -0.0002:
        pts += 0.5

    return " | ".join(tags), min(6.0, pts)


def _cap_tp2_by_opposing_zones(
    d: int,
    entry: float,
    tp2: float,
    risk: float,
    atr_1h: float,
    map1: Optional[dict],
    wall4: Optional[dict],
    min_rr: float,
) -> Tuple[float, bool, str]:
    """
    Mengecek koridor (entry, tp2) terhadap zona lawan 1H (OB/Breaker touches <= 1,
    FVG lebar >= 0.25*ATR 1H) serta tembok 4H terdekat.
    Jika ada zona lawan di jalur:
    - Bila ruang ke tepi depan zona (+/- 0.05*ATR 1H) >= min_rr: parkir TP2 di depan zona.
    - Bila ruang < min_rr: tandai blocked=True agar setup di-REJECT.
    """
    opp_zones: List[dict] = []
    if map1:
        obs_key = "bear_obs" if d == 1 else "bull_obs"
        for z in map1.get(obs_key, []):
            if z.get("touches", 0) <= 1:
                opp_zones.append(z)
        for z in map1.get("breakers", []):
            if z.get("dir") == -d and z.get("touches", 0) <= 1:
                opp_zones.append(z)
        for z in map1.get("fvgs", []):
            if z.get("dir") == -d and (float(z["top"]) - float(z["bot"])) >= 0.25 * atr_1h:
                opp_zones.append(z)
    if wall4:
        opp_zones.append(wall4)

    lo, hi = (entry, tp2) if d == 1 else (tp2, entry)
    buf = 0.10 * atr_1h
    in_path: List[dict] = []
    for z in opp_zones:
        z_bot = float(z["bot"])
        z_top = float(z["top"])
        ahead_of_entry = (z_bot >= entry) if d == 1 else (z_top <= entry)
        if ahead_of_entry and z_top > lo + buf and z_bot < hi - buf:
            in_path.append(z)

    if not in_path:
        return tp2, False, ""

    if d == 1:
        front_z = min(in_path, key=lambda z: float(z["bot"]))
        front = float(front_z["bot"])
        cap = front - 0.05 * atr_1h
        room_rr = (cap - entry) / max(risk, 1e-12)
    else:
        front_z = max(in_path, key=lambda z: float(z["top"]))
        front = float(front_z["top"])
        cap = front + 0.05 * atr_1h
        room_rr = (entry - cap) / max(risk, 1e-12)

    z_lbl = front_z.get("kind") or ("Tembok Supply/Demand 4H" if front_z is wall4 else "Zona Lawan 1H")
    if room_rr >= min_rr:
        return cap, False, f"TP2 diparkir di depan {z_lbl} ({fp(front)}, RR 1:{room_rr:.1f})."
    return tp2, True, f"Jalur TP2 terhalang {z_lbl} ({fp(front)}, ruang {room_rr:.2f}R < 1:{min_rr:.1f})."


def _compute_trade_geometry(
    d: int,
    archetype: str,
    cur_p: float,
    poi_low: float,
    poi_high: float,
    htf_poi: Optional[dict],
    approach_code: str,
    symbol: str,
    cfg: Config,
    atr_1h: float,
    cur_atr: float,
    w_low: float,
    w_high: float,
    w_eq: float,
    r4_lo: float,
    r4_hi: float,
    wall_supply_4h: Optional[dict],
    wall_demand_4h: Optional[dict],
    map1: Optional[dict],
    map15: dict,
    l_arr: np.ndarray,
    h_arr: np.ndarray,
    htf_pts: float,
    cur_rv: float,
    rev_ctx: Optional[dict] = None,
) -> dict:
    """
    Menghitung geometri Entry, Structural SL ('Zoom-Out Check' ATR 1H),
    TP1/TP2/TP3 berbasis likuiditas 1H & pembatas Tembok 4H, serta Estimasi Durasi Posisi.
    Aturan Tegas:
    - TIDAK memaksakan Limit Entry jauh demi memanipulasi R:R. Jika dari posisi harga/zona saat ini
      R:R efektif (setelah dipotong Tembok 4H penghalang & batas durasi hold) < 1:3.0, maka setup di-REJECT.
    """
    notes: List[str] = []
    raw_poi_low, raw_poi_high = min(poi_low, poi_high), max(poi_low, poi_high)
    max_zone_w = min(0.80 * atr_1h, 0.035 * cur_p)
    poi_low, poi_high = _refine_zone_bounds(poi_low, poi_high, d, cur_p, max_zone_w, 0.14 * atr_1h)

    span_poi = max(poi_high - poi_low, 1e-12)
    ce_50 = (poi_low + poi_high) / 2.0

    # Penempatan Entry Tanpa Limit Jauh Buatan (Prefer REJECT jika Trade Range / R:R tidak masuk)
    if archetype == "PRE_REACTION":
        # Ambil 15% dari bibir depan zona (Front-Edge Tap) agar tidak missed fill saat mengintai OB segar terdekat
        limit_entry = (poi_high - 0.15 * span_poi) if d == 1 else (poi_low + 0.15 * span_poi)
        in_poi_now = (poi_low - 0.05 * atr_1h) <= cur_p <= (poi_high + 0.05 * atr_1h)
        at_poi = in_poi_now
        plan_entry = cur_p if in_poi_now else limit_entry
    elif archetype == "HTF_REVERSAL":
        # Untuk Reversal Watch: gunakan harga pasar saat ini (cur_p) atau bibir terdekat zona jika sedang mendekat tipis.
        # Jangan pernah menarik plan_entry jauh ke belakang demi memalsukan R:R >= 1:3!
        dist_from_edge = (cur_p - poi_high) if d == 1 else (poi_low - cur_p)
        at_poi = (poi_low - 0.08 * atr_1h) <= cur_p <= (poi_high + 0.08 * atr_1h) or (0.0 <= dist_from_edge <= 0.30 * atr_1h)
        limit_entry = ce_50
        plan_entry = cur_p
    else:
        dist_from_poi = (cur_p - poi_high) if d == 1 else (poi_low - cur_p)
        at_poi = (poi_low - 0.05 * atr_1h) <= cur_p <= (poi_high + 0.05 * atr_1h) or (0.0 <= dist_from_poi <= 0.35 * atr_1h)
        limit_entry = ce_50
        plan_entry = cur_p

    # --- STOP LOSS BERBASIS TAMENG LIKUIDITAS / LOW-HIGH SWEEP ("LIQUIDITY-PROTECTED SL" + "ZOOM-OUT CHECK" ATR 1H) ---
    w_span = max(w_high - w_low, 2.5 * atr_1h)
    min_pct = cfg.min_sl_pct_major if symbol in MAJOR_SYMBOLS else cfg.min_sl_pct_alt
    raw_min_sl = max(
        cfg.min_sl_atr_1h * atr_1h,
        cfg.min_sl_range_1h * min(w_span, 5.0 * atr_1h),
        min_pct * plan_entry,
    )
    min_sl_dist = min(raw_min_sl, 0.038 * plan_entry)
    max_sl_dist = min(
        0.135 * plan_entry,
        max(min_sl_dist * 1.35, max(cfg.max_sl_atr_1h * atr_1h, cfg.max_sl_range_1h * max(w_span, 4.0 * atr_1h))),
    )

    liq_guard_info = ""
    sweep_entry_lvl = NAN
    sl_too_wide = False

    if d == 1:
        local_swing_low = float(np.min(l_arr[-12:]))
        use_local = abs(plan_entry - local_swing_low) <= 1.8 * atr_1h
        anchor_low = min(poi_low, local_swing_low) if use_local else poi_low
        if htf_poi and abs(plan_entry - htf_poi["bot"]) <= 2.0 * atr_1h:
            anchor_low = min(anchor_low, htf_poi["bot"])

        # Jika HTF_REVERSAL memiliki rangkaian sweep low, wajib gunakan low terendah dari rangkaian sweep tersebut
        if archetype == "HTF_REVERSAL" and rev_ctx and not math.isnan(rev_ctx.get("sweep_extreme", NAN)):
            sw_ext = float(rev_ctx["sweep_extreme"])
            if abs(plan_entry - sw_ext) <= 2.4 * atr_1h:
                anchor_low = min(anchor_low, sw_ext)
                sweep_entry_lvl = sw_ext
                max_sl_dist = max(max_sl_dist, min(0.14 * plan_entry, 2.55 * atr_1h))
                liq_guard_info = f"SL di bawah Low Sweep ({fp(sw_ext)}) + buffer ATR"

        # Deteksi kolam likuiditas (EQL 1H/15M, Swing Low 1H/15M, atau Wick 4H/1H) tepat di belakang POI
        liq_cands_l: List[Tuple[float, str]] = []
        if map1:
            for eq_l in map1.get("eql", []):
                liq_cands_l.append((float(eq_l), "EQL 1H"))
            for sw_l in map1.get("lows", [])[-4:]:
                liq_cands_l.append((float(sw_l["price"]), "Swing Low 1H"))
        for eq_l15 in map15.get("eql", []):
            liq_cands_l.append((float(eq_l15), "EQL 15M"))
        for sw_l15 in map15.get("lows", [])[-5:]:
            liq_cands_l.append((float(sw_l15["price"]), "Swing Low 15M"))
        if htf_poi:
            liq_cands_l.append((float(htf_poi["bot"]), f"Wick {htf_poi.get('tf', 'HTF')} OB"))
        if raw_poi_low < poi_low:
            liq_cands_l.append((float(raw_poi_low), "Ujung Bawah Zona HTF"))
        if wall_demand_4h:
            liq_cands_l.append((float(wall_demand_4h["bot"]), "Tembok Demand 4H"))

        valid_liq_l = [
            (lvl, lbl) for lvl, lbl in liq_cands_l
            if (plan_entry - 1.60 * atr_1h) <= lvl <= (poi_low + 0.04 * atr_1h)
        ]
        if valid_liq_l:
            valid_liq_l.sort(key=lambda x: x[0])
            liq_lvl, liq_lbl = valid_liq_l[0]
            anchor_low = min(anchor_low, liq_lvl)
            max_sl_dist = max(max_sl_dist, min(0.14 * plan_entry, 2.50 * atr_1h))
            if math.isnan(sweep_entry_lvl) and liq_lvl < poi_low - 0.02 * atr_1h:
                sweep_entry_lvl = liq_lvl
            if not liq_guard_info:
                liq_guard_info = f"SL diamankan di balik {liq_lbl} ({fp(liq_lvl)}) — tahan Judas Swing"

        buf_mult = 0.18 if archetype == "HTF_REVERSAL" else cfg.sl_buffer
        sl_struct = anchor_low - buf_mult * atr_1h
        raw_dist = max(0.0, plan_entry - sl_struct)
        if raw_dist > max_sl_dist * 1.15:
            sl_too_wide = True
        # Jangan pernah memotong SL lebih dangkal dari anchor_low - buffer! Biarkan jarak asli agar R:R diuji jujur.
        final_sl_dist = max(raw_dist, min_sl_dist)
        sl = min(plan_entry - final_sl_dist, anchor_low - 0.12 * cur_atr)
    else:
        local_swing_high = float(np.max(h_arr[-12:]))
        use_local = abs(local_swing_high - plan_entry) <= 1.8 * atr_1h
        anchor_high = max(poi_high, local_swing_high) if use_local else poi_high
        if htf_poi and abs(htf_poi["top"] - plan_entry) <= 2.0 * atr_1h:
            anchor_high = max(anchor_high, htf_poi["top"])

        # Jika HTF_REVERSAL SHORT memiliki rangkaian sweep high, gunakan high tertinggi + ekstra buffer squeeze
        if archetype == "HTF_REVERSAL" and rev_ctx and not math.isnan(rev_ctx.get("sweep_extreme", NAN)):
            sw_ext = float(rev_ctx["sweep_extreme"])
            if abs(sw_ext - plan_entry) <= 2.4 * atr_1h:
                anchor_high = max(anchor_high, sw_ext)
                sweep_entry_lvl = sw_ext
                max_sl_dist = max(max_sl_dist, min(0.14 * plan_entry, 2.55 * atr_1h))
                liq_guard_info = f"SL di atas High Sweep ({fp(sw_ext)}) + ekstra buffer Short Squeeze"

        # Deteksi kolam likuiditas (EQH 1H/15M, Swing High 1H/15M, atau Wick 4H/1H) tepat di belakang POI
        liq_cands_h: List[Tuple[float, str]] = []
        if map1:
            for eq_h in map1.get("eqh", []):
                liq_cands_h.append((float(eq_h), "EQH 1H"))
            for sw_h in map1.get("highs", [])[-4:]:
                liq_cands_h.append((float(sw_h["price"]), "Swing High 1H"))
        for eq_h15 in map15.get("eqh", []):
            liq_cands_h.append((float(eq_h15), "EQH 15M"))
        for sw_h15 in map15.get("highs", [])[-5:]:
            liq_cands_h.append((float(sw_h15["price"]), "Swing High 15M"))
        if htf_poi:
            liq_cands_h.append((float(htf_poi["top"]), f"Wick {htf_poi.get('tf', 'HTF')} OB"))
        if raw_poi_high > poi_high:
            liq_cands_h.append((float(raw_poi_high), "Ujung Atas Zona HTF"))
        if wall_supply_4h:
            liq_cands_h.append((float(wall_supply_4h["top"]), "Tembok Supply 4H"))

        valid_liq_h = [
            (lvl, lbl) for lvl, lbl in liq_cands_h
            if (poi_high - 0.04 * atr_1h) <= lvl <= (plan_entry + 1.60 * atr_1h)
        ]
        if valid_liq_h:
            valid_liq_h.sort(key=lambda x: -x[0])
            liq_lvl, liq_lbl = valid_liq_h[0]
            anchor_high = max(anchor_high, liq_lvl)
            max_sl_dist = max(max_sl_dist, min(0.14 * plan_entry, 2.50 * atr_1h))
            if math.isnan(sweep_entry_lvl) and liq_lvl > poi_high + 0.02 * atr_1h:
                sweep_entry_lvl = liq_lvl
            if not liq_guard_info:
                liq_guard_info = f"SL diamankan di balik {liq_lbl} ({fp(liq_lvl)}) — tahan Judas Swing"

        # Beri ruang ekstra untuk SHORT Reversal karena wick short squeeze sering lebih tajam (0.24x ATR 1H)
        buf_mult = 0.24 if archetype == "HTF_REVERSAL" else cfg.sl_buffer
        sl_struct = anchor_high + buf_mult * atr_1h
        raw_dist = max(0.0, sl_struct - plan_entry)
        if raw_dist > max_sl_dist * 1.15:
            sl_too_wide = True
        final_sl_dist = max(raw_dist, min_sl_dist)
        sl = max(plan_entry + final_sl_dist, anchor_high + 0.15 * cur_atr)

    risk = max(abs(plan_entry - sl), 1e-12)
    sl_atr_1h_ratio = risk / max(atr_1h, 1e-12)
    if not liq_guard_info:
        liq_guard_info = f"SL Struktural 1H ({sl_atr_1h_ratio:.2f}x ATR 1H)"

    # --- TARGET TP1 (CHECKPOINT H1), TP2 (EQUILIBRIUM 4H / STRUKTUR >= 1:3.0), & TP3 (SUPPLY/DEMAND 4H / PDH-PDL) ---
    vel_factor = 0.50 if (cur_rv >= 1.4 and htf_pts >= 14.0) else 0.44
    hourly_vel = max(vel_factor * atr_1h, 1e-12)
    eff_max_hold = max(float(getattr(cfg, "max_hold_hours", 30.0)), 18.0)
    max_reach_dist = eff_max_hold * hourly_vel
    target_min_rr = max(float(getattr(cfg, "min_rr", 3.0)), 3.0)
    eq_4h = (r4_lo + r4_hi) / 2.0

    blocked_by_wall = False
    req_struct_tp2 = bool(getattr(cfg, "require_structural_tp2", True))
    path_clear = bool(getattr(cfg, "path_clear_check", True))

    if d == 1:
        # Batasi TP2 pada Tembok Supply 4H aktif (touches <= 2) yang menghalangi jalur naik
        wall_cap = None
        if wall_supply_4h and wall_supply_4h.get("touches", 0) <= 2:
            if path_clear:
                if wall_supply_4h["bot"] > plan_entry:
                    wall_cap = max(plan_entry + 0.02 * atr_1h, wall_supply_4h["bot"] - 0.05 * atr_1h)
            elif wall_supply_4h["bot"] > plan_entry + 0.30 * atr_1h:
                wall_cap = max(plan_entry + 0.10 * atr_1h, wall_supply_4h["bot"] - 0.05 * atr_1h)

        cands_tp: List[float] = [w_eq, w_high, eq_4h]
        if map1:
            cands_tp.extend(map1.get("eqh", []))
            cands_tp.extend([s["price"] for s in map1.get("highs", [])[-8:]])
            if not path_clear:
                cands_tp.extend([z["bot"] for z in map1.get("bear_obs", []) if z["bot"] > plan_entry])
                cands_tp.extend([z["bot"] for z in map1.get("fvgs", []) if z["dir"] == -1 and z["bot"] > plan_entry])
            else:
                cands_tp.extend([z["bot"] for z in map1.get("bear_obs", []) if z["bot"] > plan_entry and z.get("touches", 0) > 1])
                cands_tp.extend([
                    z["bot"] for z in map1.get("fvgs", [])
                    if z["dir"] == -1 and z["bot"] > plan_entry and (float(z["top"]) - float(z["bot"])) < 0.25 * atr_1h
                ])
        cands_tp.extend(map15.get("eqh", []))
        if wall_supply_4h and wall_supply_4h["bot"] > plan_entry:
            cands_tp.append(wall_supply_4h["bot"] - 0.05 * atr_1h)
            cands_tp.append(wall_supply_4h["top"])
        if r4_hi > plan_entry:
            cands_tp.append(r4_hi)

        # Batas plafon struktural nyata (jangan mengarang target di luar struktur pasar atau melebihi batas jam hold)
        struct_ceiling = max([x for x in cands_tp if x > plan_entry] + [r4_hi, w_high])
        reach_ceiling = min(struct_ceiling, plan_entry + max_reach_dist)
        if wall_cap is not None:
            reach_ceiling = min(reach_ceiling, wall_cap)

        valid_tp2 = sorted([x for x in cands_tp if (plan_entry + target_min_rr * risk) <= x <= reach_ceiling])
        if archetype == "HTF_REVERSAL" and (plan_entry + target_min_rr * risk) <= eq_4h <= reach_ceiling:
            # Tahap 6 Reversal Watch: utamakan Equilibrium 4H sebagai TP2 jika sudah memenuhi >= 1:3.0 R:R
            tp2 = eq_4h
        elif valid_tp2:
            sweet_tp2 = [x for x in valid_tp2 if x <= plan_entry + 4.8 * risk]
            tp2 = sweet_tp2[0] if sweet_tp2 else min(valid_tp2[0], reach_ceiling)
        else:
            if req_struct_tp2:
                reachable = [x for x in cands_tp if plan_entry < x <= reach_ceiling]
                tp2 = max(reachable) if reachable else (plan_entry + 0.5 * risk)
                if wall_cap is not None:
                    tp2 = min(tp2, wall_cap)
                notes.append("TP2 struktural tidak tersedia >= 3R")
            else:
                # Jika tidak ada kandidat di atas 3.0R namun reach_ceiling masih >= 3.0R, gunakan min(3.1R, reach_ceiling).
                # Jika reach_ceiling < 3.0R (karena Tembok 4H atau Range sempit), biarkan tp2 = reach_ceiling (< 3.0R) agar di-REJECT!
                tp2 = min(plan_entry + (target_min_rr + 0.10) * risk, reach_ceiling)

        wall_blocks_rr = (
            (wall_cap - plan_entry) / risk < target_min_rr
            if (req_struct_tp2 or path_clear)
            else (tp2 >= wall_cap - 1e-9 and (wall_cap - plan_entry) / risk < target_min_rr)
        ) if wall_cap is not None else False
        if wall_cap is not None and wall_blocks_rr:
            blocked_by_wall = True
            notes.append(f"Ruang target terhalang Tembok Supply 4H ({fp(wall_supply_4h['bot'])}) sebelum mencapai 1:{target_min_rr:.1f} R:R.")

        if path_clear:
            tp2, path_blocked, path_note = _cap_tp2_by_opposing_zones(
                d, plan_entry, tp2, risk, atr_1h, map1, wall_supply_4h, target_min_rr
            )
            if path_blocked:
                blocked_by_wall = True
            if path_note:
                notes.append(path_note)

        # TP1: Internal High / BSL terdekat di H1 (~1.3R - 2.2R) TANPA menggeser SL ke BEP
        h1_tp1_cands = []
        if map1:
            h1_tp1_cands.extend(map1.get("eqh", []))
            h1_tp1_cands.extend([s["price"] for s in map1.get("highs", [])[-8:]])
        h1_tp1_cands.append(w_eq)
        valid_tp1 = sorted([x for x in h1_tp1_cands if plan_entry + 1.30 * risk <= x <= plan_entry + 0.68 * max(tp2 - plan_entry, 0)])
        if valid_tp1:
            tp1 = valid_tp1[0]
        else:
            tp1 = plan_entry + 0.52 * max(tp2 - plan_entry, risk)
        opp_lvl = wall_supply_4h["bot"] if wall_supply_4h else w_high

        # TP3: Supply 4H, PDH/PWH (High 4H / High 1H), atau Equal Highs besar
        runner_cands = [x for x in cands_tp + [r4_hi, w_high] if x > tp2 + 0.8 * risk]
        if wall_supply_4h and wall_supply_4h["bot"] > tp2 + 0.5 * risk:
            runner_cands.append(wall_supply_4h["bot"])
        if runner_cands:
            tp3 = min(max(runner_cands), plan_entry + 8.0 * risk)
        else:
            tp3 = max(tp2 + 1.2 * risk, struct_ceiling)

        if req_struct_tp2 or path_clear:
            tp2 = max(tp2, plan_entry + 0.02 * atr_1h)
            tp1 = min(tp1, plan_entry + 0.68 * (tp2 - plan_entry))
            tp3 = max(tp3, tp2 + 1.2 * risk)

    else:
        # Batasi TP2 pada Tembok Demand 4H aktif (touches <= 2) yang menghalangi jalur turun
        wall_cap = None
        if wall_demand_4h and wall_demand_4h.get("touches", 0) <= 2:
            if path_clear:
                if wall_demand_4h["top"] < plan_entry:
                    wall_cap = min(plan_entry - 0.02 * atr_1h, wall_demand_4h["top"] + 0.05 * atr_1h)
            elif wall_demand_4h["top"] < plan_entry - 0.30 * atr_1h:
                wall_cap = min(plan_entry - 0.10 * atr_1h, wall_demand_4h["top"] + 0.05 * atr_1h)

        cands_tp = [w_eq, w_low, eq_4h]
        if map1:
            cands_tp.extend(map1.get("eql", []))
            cands_tp.extend([s["price"] for s in map1.get("lows", [])[-8:]])
            if not path_clear:
                cands_tp.extend([z["top"] for z in map1.get("bull_obs", []) if z["top"] < plan_entry])
                cands_tp.extend([z["top"] for z in map1.get("fvgs", []) if z["dir"] == 1 and z["top"] < plan_entry])
            else:
                cands_tp.extend([z["top"] for z in map1.get("bull_obs", []) if z["top"] < plan_entry and z.get("touches", 0) > 1])
                cands_tp.extend([
                    z["top"] for z in map1.get("fvgs", [])
                    if z["dir"] == 1 and z["top"] < plan_entry and (float(z["top"]) - float(z["bot"])) < 0.25 * atr_1h
                ])
        cands_tp.extend(map15.get("eql", []))
        if wall_demand_4h and wall_demand_4h["top"] < plan_entry:
            cands_tp.append(wall_demand_4h["top"] + 0.05 * atr_1h)
            cands_tp.append(wall_demand_4h["bot"])
        if r4_lo < plan_entry:
            cands_tp.append(r4_lo)

        struct_floor = min([x for x in cands_tp if 0 < x < plan_entry] + [r4_lo, w_low])
        reach_floor = max(struct_floor, plan_entry - max_reach_dist, plan_entry * 0.25)
        if wall_cap is not None:
            reach_floor = max(reach_floor, wall_cap)

        valid_tp2 = sorted([x for x in cands_tp if reach_floor <= x <= (plan_entry - target_min_rr * risk)], reverse=True)
        if archetype == "HTF_REVERSAL" and reach_floor <= eq_4h <= (plan_entry - target_min_rr * risk):
            # Tahap 6 Reversal Watch SHORT: utamakan Equilibrium 4H sebagai TP2 jika sudah memenuhi >= 1:3.0 R:R
            tp2 = eq_4h
        elif valid_tp2:
            sweet_tp2 = [x for x in valid_tp2 if x >= plan_entry - 4.8 * risk]
            tp2 = sweet_tp2[0] if sweet_tp2 else max(valid_tp2[0], reach_floor)
        else:
            if req_struct_tp2:
                reachable = [x for x in cands_tp if reach_floor <= x < plan_entry]
                tp2 = min(reachable) if reachable else (plan_entry - 0.5 * risk)
                if wall_cap is not None:
                    tp2 = max(tp2, wall_cap)
                notes.append("TP2 struktural tidak tersedia >= 3R")
            else:
                tp2 = max(plan_entry - (target_min_rr + 0.10) * risk, reach_floor)

        wall_blocks_rr = (
            (plan_entry - wall_cap) / risk < target_min_rr
            if (req_struct_tp2 or path_clear)
            else (tp2 <= wall_cap + 1e-9 and (plan_entry - wall_cap) / risk < target_min_rr)
        ) if wall_cap is not None else False
        if wall_cap is not None and wall_blocks_rr:
            blocked_by_wall = True
            notes.append(f"Ruang target terhalang Tembok Demand 4H ({fp(wall_demand_4h['top'])}) sebelum mencapai 1:{target_min_rr:.1f} R:R.")

        if path_clear:
            tp2, path_blocked, path_note = _cap_tp2_by_opposing_zones(
                d, plan_entry, tp2, risk, atr_1h, map1, wall_demand_4h, target_min_rr
            )
            if path_blocked:
                blocked_by_wall = True
            if path_note:
                notes.append(path_note)

        # TP1: Internal Low / SSL terdekat di H1 (~1.3R - 2.2R) TANPA menggeser SL ke BEP
        h1_tp1_cands = []
        if map1:
            h1_tp1_cands.extend(map1.get("eql", []))
            h1_tp1_cands.extend([s["price"] for s in map1.get("lows", [])[-8:]])
        h1_tp1_cands.append(w_eq)
        valid_tp1 = sorted(
            [x for x in h1_tp1_cands if plan_entry - 0.68 * max(plan_entry - tp2, 0) <= x <= plan_entry - 1.30 * risk],
            reverse=True,
        )
        if valid_tp1:
            tp1 = valid_tp1[0]
        else:
            tp1 = plan_entry - 0.52 * max(plan_entry - tp2, risk)
        opp_lvl = wall_demand_4h["top"] if wall_demand_4h else w_low

        # TP3: Demand 4H, PDL/PWL (Low 4H / Low 1H), atau Equal Lows besar
        runner_cands = [x for x in cands_tp + [r4_lo, w_low] if 0 < x < tp2 - 0.8 * risk]
        if wall_demand_4h and wall_demand_4h["top"] < tp2 - 0.5 * risk:
            runner_cands.append(wall_demand_4h["top"])
        if runner_cands:
            tp3 = max(min(runner_cands), max(plan_entry * 0.25, plan_entry - 8.0 * risk))
        else:
            tp3 = max(plan_entry * 0.25, min(tp2 - 1.2 * risk, struct_floor))

        if req_struct_tp2 or path_clear:
            tp2 = min(tp2, plan_entry - 0.02 * atr_1h)
            tp1 = max(tp1, plan_entry - 0.68 * (plan_entry - tp2))
            tp3 = min(tp3, tp2 - 1.2 * risk)

    rr = (tp2 - plan_entry) / risk if d == 1 else (plan_entry - tp2) / risk
    rr = max(0.0, rr)
    rr_tp1 = abs(tp1 - plan_entry) / risk
    rr_tp3 = abs(tp3 - plan_entry) / risk
    est_h1 = max(0.5, abs(tp1 - plan_entry) / hourly_vel)
    est_h2 = max(1.0, abs(tp2 - plan_entry) / hourly_vel)
    hold_label = _format_hold_duration(est_h1, est_h2)

    # Estimasi waktu jemput bibir POI (jam) jika harga masih mendekati zona
    if at_poi:
        est_tap_h = 0.0
    else:
        dist_edge = max(0.0, (cur_p - poi_high) if d == 1 else (poi_low - cur_p))
        est_tap_h = round(dist_edge / max(0.50 * atr_1h, 1e-12), 1)

    return {
        "poi_low": poi_low,
        "poi_high": poi_high,
        "limit_entry": limit_entry,
        "plan_entry": plan_entry,
        "at_poi": at_poi,
        "sl": sl,
        "risk": risk,
        "min_sl_dist": min_sl_dist,
        "max_sl_dist": max_sl_dist,
        "sl_atr_1h_ratio": sl_atr_1h_ratio,
        "sl_too_wide": sl_too_wide,
        "blocked_by_wall": blocked_by_wall,
        "liq_guard_info": liq_guard_info,
        "sweep_entry_lvl": sweep_entry_lvl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "rr": rr,
        "rr_tp1": rr_tp1,
        "rr_tp3": rr_tp3,
        "opp_lvl": opp_lvl,
        "hourly_vel": hourly_vel,
        "est_h1": est_h1,
        "est_h2": est_h2,
        "hold_label": hold_label,
        "est_tap_h": est_tap_h,
        "notes": notes,
    }


def populate_forward_playbook(s: Setup) -> None:
    """
    Menyusun panduan taktis ke depan (Forward If-Then Playbook + Flip Scenario) sesuai gaya trader:
    - Wajib R:R Minimal 1:3.0 (Target Utama TP2 >= 3.0R, Runner TP3 4.5R - 8.0R+).
    - Set-and-Forget Fixed Structural SL di balik kolam likuiditas / low-high sweep (PANTANG geser SL ke BEP).
    """
    is_long = s.direction == "LONG"
    opp_dir = "SHORT" if is_long else "LONG"
    prox_edge = s.poi_high if is_long else s.poi_low
    ce_mid = (s.poi_low + s.poi_high) / 2.0 if (not math.isnan(s.poi_low) and not math.isnan(s.poi_high)) else s.entry
    dist_txt = f"{s.dist_to_poi_pct:.2f}%" if not math.isnan(s.dist_to_poi_pct) else "-"
    tap_txt = f"~{s.est_tap_hours:.1f} jam" if (not math.isnan(s.est_tap_hours) and s.est_tap_hours > 0) else "detik ini"
    runner_txt = f" atau lanjut Runner ke TP3 {fp(s.tp3)} (RR 1:{s.rr_tp3:.1f})" if not math.isnan(s.tp3) and not math.isnan(s.rr_tp3) else ""
    sw_ref_txt = (
        f" [Acuan Sweep: {fp(s.sweep_entry_lvl)} — SL tetap di {fp(s.sl)} di baliknya]"
        if not math.isnan(s.sweep_entry_lvl)
        else f" (SL Struktural Tetap: {fp(s.sl)})"
    )
    raw_d = s.raw_market_data if isinstance(s.raw_market_data, dict) else {}
    rev_stage_lbl = raw_d.get("rev_stage_label", "")
    rev_next_step = raw_d.get("rev_next_step", "")
    rev_pts_10 = s.python_score / 10.0

    if s.archetype == "PRE_REACTION":
        if s.exec_style == "LIMIT_PRE_TAP":
            s.forward_playbook = (
                f"SEBELUM BEREAKSI ({s.approach_style}): Harga berjarak {dist_txt} menuju OB segar 0-Touch [{s.entry_zone}] (est. jemput {tap_txt}). "
                f"Langkah Utama: Pasang Limit {s.direction} di bibir zona {fp(s.entry)} (atau CE {fp(ce_mid)}){sw_ref_txt}; "
                f"pertahankan SL tetap di {fp(s.sl)} (JANGAN geser SL ke BEP saat retest) dan tahan penuh menuju Target Utama >=1:3 di TP2 {fp(s.tp2)} (RR 1:{s.rr:.1f}){runner_txt}."
            )
        else:
            s.forward_playbook = (
                f"PENDEKATAN LIAR KRIPTO ({s.approach_style}): Harga mendekati [{s.entry_zone}] (jarak {dist_txt}) dengan momentum tajam. "
                f"Langkah Utama: Tunggu tusukan Judas Swing ke CE {fp(ce_mid)} / likuiditas {fp(s.sweep_entry_lvl if not math.isnan(s.sweep_entry_lvl) else (s.poi_low if is_long else s.poi_high))} "
                f"+ close 15M memantul (CHoCH/PinBar) baru entry {s.direction}{sw_ref_txt}; tahan SL tetap tanpa geser BEP menuju TP2 {fp(s.tp2)} (RR 1:{s.rr:.1f}){runner_txt}."
            )
        s.flip_scenario = (
            f"Antisipasi Jebol (Breaker Flip): Jika candle 15M/1H close impulsif ber-volume menembus SL {fp(s.sl)} tanpa pantulan, "
            f"terima risiko 1R secara disiplin — zona [{s.entry_zone}] berubah menjadi Breaker Block untuk cari retest {opp_dir}."
        )

    elif s.archetype == "HTF_REVERSAL":
        stage_prefix = f"[{rev_stage_lbl} | Skor {rev_pts_10:.1f}/10] " if rev_stage_lbl else f"[Skor {rev_pts_10:.1f}/10] "
        if s.status == "ENTRY READY":
            s.forward_playbook = (
                f"{stage_prefix}REVERSAL SMC LENGKAP (4H->H1->M15): Sweep {'SSL' if is_long else 'BSL'} + MSS H1 + Trigger CHoCH M15 ber-displacement valid di [{s.entry_zone}]. "
                f"Langkah Utama: Eksekusi {s.direction} dekat {fp(s.entry)}{sw_ref_txt}; pertahankan SL tetap di {fp(s.sl)} "
                f"(pantang geser SL ke BEP) dan tahan ke Target Utama TP2 (EQ 4H / Struktur) {fp(s.tp2)} (RR 1:{s.rr:.1f}){runner_txt} [Checkpoint H1 TP1: {fp(s.tp1)}]."
            )
        else:
            next_act = rev_next_step or (
                f"Tunggu Sweep {'SSL' if is_long else 'BSL'} + Reclaim <=3 candle H1 -> MSS H1 ber-displacement -> Retrace OTE 62-79% -> Trigger CHoCH M15 di [{s.entry_zone}]"
            )
            s.forward_playbook = (
                f"{stage_prefix}REVERSAL WATCH BERTAHAP [{s.entry_zone}] (Jarak {dist_txt}): {next_act}{sw_ref_txt}. "
                f"Setelah trigger M15 valid, tahan SL tetap di {fp(s.sl)} (tanpa geser BEP) menuju TP2 {fp(s.tp2)} (RR 1:{s.rr:.1f}){runner_txt}."
            )
        s.flip_scenario = (
            f"Syarat Pembatalan Reversal Watch (Cancel): Batalkan watch jika Close H1 bertahan {'di bawah Low Sweep' if is_long else 'di atas High Sweep'} {fp(s.sl)} (>=3 bar H1), "
            f"Close 4H menembus {'dasar Demand 4H' if is_long else 'puncak Supply 4H'}, MSS H1 tidak muncul dalam 20 bar H1, atau harga lari >50% impuls tanpa retrace."
        )

    elif s.archetype == "BREAKOUT":
        if s.status == "ENTRY READY":
            s.forward_playbook = (
                f"BREAKOUT AKTIF: Eksekusi {s.direction} dekat {fp(s.entry)} (SL tetap di {fp(s.sl)} tanpa geser BEP), "
                f"tahan menuju Target Utama TP2 {fp(s.tp2)} (RR 1:{s.rr:.1f}){runner_txt}."
            )
        else:
            s.forward_playbook = (
                f"PANTAU BREAKOUT/RETEST [{s.entry_zone}]: Tunggu close 15M valid ber-volume >= 1.5x rata-rata atau retest bersih di {fp(s.level)} "
                f"dengan SL tetap di {fp(s.sl)}."
            )
        s.flip_scenario = (
            f"Antisipasi Fakeout/Deviation: Jika breakout gagal dan close 15M masuk kembali menembus {fp(s.sl)}, waspadai sapuan likuiditas "
            f"(fakeout) berbalik tajam ke arah {opp_dir}."
        )

    else:
        # PILAR 3: SMC_SNIPER (Follow-Trend Post-Sweep + CHOCH di POI)
        if s.status == "ENTRY READY":
            s.forward_playbook = (
                f"SMC SNIPER FOLLOW-TREND TERKONFIRMASI ({s.zone_badge}): Likuiditas sudah disapu + CHOCH/CISD valid searah HTF di POI [{s.entry_zone}]. "
                f"Langkah Utama: Eksekusi {s.direction} di area {fp(s.entry)}{sw_ref_txt}, pertahankan SL tetap di {fp(s.sl)} "
                f"(JANGAN geser SL ke BEP saat retest) dan tahan penuh menuju Target Utama TP2 {fp(s.tp2)} (RR 1:{s.rr:.1f}){runner_txt}."
            )
        else:
            s.forward_playbook = (
                f"PANTAUAN SMC SNIPER FOLLOW-TREND ({s.zone_badge}): Struktur Sweep+CHOCH {s.direction} searah HTF sudah terbentuk di [{s.entry_zone}]{sw_ref_txt}. "
                f"Langkah Utama: Tunggu candle 15M menutup konfirmasi pantulan di dekat {fp(s.entry)}, pasang SL tetap di {fp(s.sl)} (tanpa geser BEP), dan bidik Target Utama TP2 {fp(s.tp2)} (RR 1:{s.rr:.1f}){runner_txt}."
            )
        s.flip_scenario = (
            f"Skenario Invalidasi / Flip: Jika candle 15M/1H close menembus SL {fp(s.sl)}, struktur tren patah "
            f"(CHOCH ke {opp_dir}) — terima risiko 1R disiplin & pantau peluang balik arah {opp_dir}."
        )


def _safe_evidence(
    setup: Setup,
    df4: Optional[pd.DataFrame],
    df1: Optional[pd.DataFrame],
    df15: Optional[pd.DataFrame],
    btc: Optional[dict] = None,
) -> dict:
    try:
        return build_evidence(setup, df4, df1, df15, btc)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "text": ""}


def analyze_smc_symbol(
    symbol: str,
    df15: pd.DataFrame,
    ctx4: dict,
    ctx1: dict,
    meta: dict,
    cfg: Config,
    df4: Optional[pd.DataFrame] = None,
    df1: Optional[pd.DataFrame] = None,
    deriv: Optional[dict] = None,
    btc: Optional[dict] = None,
) -> Optional[Setup]:
    """
    Mesin Utama Dinamis 3-Archetype Top-Down MTF SMC/ICT (4H -> 1H -> 15M):
    1. Mengevaluasi TREND_CONT (setup konfirmasi yang sudah berjalan tajam selama ini).
    2. Mengevaluasi PRE_REACTION (setup koin yang mendekati Fresh Supply/Demand/OB sebelum bereaksi).
    3. Mengevaluasi HTF_REVERSAL (setup potensi pembalikan tren saat mendekati/menguji Tembok Mayor 4H/1H).
    Memilih kandidat paling segar, proporsional (Zoom-Out Check ATR 1H), dan realistis durasinya.
    """
    n = len(df15)
    if n < cfg.min_bars_15m:
        return None

    o = df15["open"].values.astype(float)
    h = df15["high"].values.astype(float)
    l = df15["low"].values.astype(float)
    c = df15["close"].values.astype(float)
    times = df15["open_time"].values.astype("int64")

    atr_s = atr_wilder(df15).values
    cur_atr = float(atr_s[-1])
    if np.isnan(cur_atr) or cur_atr <= 0:
        return None

    # Hybrid Live Price: gunakan harga live_bar untuk deteksi jarak/sentuhan real-time,
    # sementara seluruh indikator & struktur tetap dihitung dari 100% closed bar (bebas repaint!).
    cur_p, l_live, h_live, live_bar = get_live_price_snapshot(df15)
    atr_pct = cur_atr / cur_p

    vol_sma = df15["volume"].rolling(cfg.rel_vol_len).mean().shift(1)
    rel = (df15["volume"] / vol_sma).values
    cur_rv = 1.0 if np.isnan(rel[-1]) else float(rel[-1])
    rsi_s = rsi_wilder(df15["close"]).values
    cur_rsi = 50.0 if np.isnan(rsi_s[-1]) else float(rsi_s[-1])

    # =================================================================
    # LAPIS 1 (4H): 1-3 MACRO KEY RANGES & TEMBOK BESAR
    # =================================================================
    map4 = extract_tf_structure(df4, cfg.htf_piv, "4H", lookback_bars=180) if df4 is not None else None
    atr_4h = map4["atr"] if map4 else float(ctx4.get("atr", cur_atr * 4.2))
    if math.isnan(atr_4h) or atr_4h <= 0:
        atr_4h = cur_atr * 4.2

    wall_supply_4h: Optional[dict] = None
    wall_demand_4h: Optional[dict] = None
    in_supply_4h_now = False
    in_demand_4h_now = False
    if map4:
        r4_lo, r4_hi = map4["range_low"], map4["range_high"]
        r4_pos = clamp((cur_p - r4_lo) / max(r4_hi - r4_lo, 1e-12) * 100.0, 0.0, 100.0)
        # Prioritaskan Order Block Supply/Demand 4H utuh (touches <= 2) sebagai Tembok 4H agar tidak terdistorsi Breaker / FVG minor
        ob_sup_4h = [
            z for z in map4["bear_obs"]
            if z.get("touches", 0) <= 2 and z["top"] > cur_p - 0.05 * atr_4h
        ]
        fvg_sup_4h = [
            z for z in map4["fvgs"]
            if z["dir"] == -1 and z.get("fresh", True) and (z["top"] - z["bot"]) >= 0.35 * atr_4h and z["top"] > cur_p - 0.05 * atr_4h
        ]
        all_sup_4h = ob_sup_4h if ob_sup_4h else fvg_sup_4h
        in_supply_4h_now = any(z["bot"] <= cur_p <= z["top"] for z in (ob_sup_4h + fvg_sup_4h))
        sup_above_ob = [z for z in ob_sup_4h if z["bot"] > cur_p]
        sup_above_all = [z for z in all_sup_4h if z["bot"] > cur_p]
        if sup_above_ob:
            wall_supply_4h = min(sup_above_ob, key=lambda z: z["bot"] - cur_p)
        elif sup_above_all:
            wall_supply_4h = min(sup_above_all, key=lambda z: z["bot"] - cur_p)
        elif ob_sup_4h:
            wall_supply_4h = min(ob_sup_4h, key=lambda z: abs(z["top"] - cur_p))

        ob_dem_4h = [
            z for z in map4["bull_obs"]
            if z.get("touches", 0) <= 2 and z["bot"] < cur_p + 0.05 * atr_4h
        ]
        fvg_dem_4h = [
            z for z in map4["fvgs"]
            if z["dir"] == 1 and z.get("fresh", True) and (z["top"] - z["bot"]) >= 0.35 * atr_4h and z["bot"] < cur_p + 0.05 * atr_4h
        ]
        all_dem_4h = ob_dem_4h if ob_dem_4h else fvg_dem_4h
        in_demand_4h_now = any(z["bot"] <= cur_p <= z["top"] for z in (ob_dem_4h + fvg_dem_4h))
        dem_below_ob = [z for z in ob_dem_4h if z["top"] < cur_p]
        dem_below_all = [z for z in all_dem_4h if z["top"] < cur_p]
        if dem_below_ob:
            wall_demand_4h = max(dem_below_ob, key=lambda z: z["top"] - cur_p)
        elif dem_below_all:
            wall_demand_4h = max(dem_below_all, key=lambda z: z["top"] - cur_p)
        elif ob_dem_4h:
            wall_demand_4h = max(ob_dem_4h, key=lambda z: -abs(cur_p - z["bot"]))

        range_4h_str = f"[{fp(r4_lo)} - {fp(r4_hi)}] (Posisi Makro: {r4_pos:.1f}%)"
    else:
        r4_lo, r4_hi = float(ctx4.get("lo", cur_p * 0.94)), float(ctx4.get("hi", cur_p * 1.06))
        r4_pos = clamp((cur_p - r4_lo) / max(r4_hi - r4_lo, 1e-12) * 100.0, 0.0, 100.0)
        range_4h_str = f"[{fp(r4_lo)} - {fp(r4_hi)}] (Posisi Makro: {r4_pos:.1f}%)"

    w_sup_txt = f"Supply 4H [{fp(wall_supply_4h['bot'])}-{fp(wall_supply_4h['top'])}]" if wall_supply_4h else f"High 4H {fp(r4_hi)}"
    w_dem_txt = f"Demand 4H [{fp(wall_demand_4h['bot'])}-{fp(wall_demand_4h['top'])}]" if wall_demand_4h else f"Low 4H {fp(r4_lo)}"
    walls_4h_str = f"Batas Bawah: {w_dem_txt} | Batas Atas: {w_sup_txt}"

    # =================================================================
    # LAPIS 2 (1H): WORKING TRADING RANGE & HTF ZONES
    # =================================================================
    map1 = extract_tf_structure(df1, cfg.htf_piv, "1H", lookback_bars=96) if df1 is not None else None
    atr_1h = map1["atr"] if map1 else float(ctx1.get("atr", cur_atr * 2.1))
    if math.isnan(atr_1h) or atr_1h <= 0:
        atr_1h = cur_atr * 2.1

    # =================================================================
    # LAPIS 3 (15M): STRUKTUR MIKRO, SWEEP, TRIGGER, & ORDERFLOW
    # =================================================================
    map15 = extract_tf_structure(df15, cfg.piv_len, "15M", lookback_bars=95)
    if not map15 or not map15["highs"] or not map15["lows"]:
        return None

    if map1:
        w_low = map1["range_low"]
        w_high = map1["range_high"]
        w_eq = map1["eq"]
    else:
        w_low = map15["range_low"]
        w_high = map15["range_high"]
        w_eq = map15["eq"]
    w_pos_pct = clamp((cur_p - w_low) / max(w_high - w_low, 1e-12) * 100.0, 0.0, 100.0)

    in_discount_1h = w_pos_pct <= 50.0
    in_premium_1h = w_pos_pct >= 50.0
    if w_pos_pct <= 45.0:
        zone_pos_lbl = f"Discount 1H ({w_pos_pct:.1f}%)"
    elif w_pos_pct >= 55.0:
        zone_pos_lbl = f"Premium 1H ({w_pos_pct:.1f}%)"
    else:
        zone_pos_lbl = f"Equilibrium 1H ({w_pos_pct:.1f}%)"

    # Pin Bars pada 15M
    pin_bars = []
    for i in range(max(0, n - 30), n):
        rng = h[i] - l[i]
        if rng <= 0:
            continue
        body = abs(c[i] - o[i])
        l_wick = min(o[i], c[i]) - l[i]
        u_wick = h[i] - max(o[i], c[i])
        if l_wick >= 0.50 * rng and u_wick <= 0.28 * rng and body <= 0.38 * rng:
            pin_bars.append({"idx": i, "dir": 1, "wick_pct": float(l_wick / rng), "time": int(times[i])})
        elif u_wick >= 0.50 * rng and l_wick <= 0.28 * rng and body <= 0.38 * rng:
            pin_bars.append({"idx": i, "dir": -1, "wick_pct": float(u_wick / rng), "time": int(times[i])})

    # Struktur & Liquidity Events 15M (BOS, CHOCH, BSL/SSL Sweep, CISD)
    highs_15 = map15["highs"]
    lows_15 = map15["lows"]
    piv_len = cfg.piv_len
    lookback = min(45, n - 1)
    struct_events = []
    trend_15 = 0

    for t in range(n - lookback, n):
        known_h = [s for s in highs_15 if s["idx"] <= t - piv_len]
        known_l = [s for s in lows_15 if s["idx"] <= t - piv_len]
        if known_h:
            last_h = known_h[-1]["price"]
            if h[t] > last_h and c[t] <= last_h and h[t - 1] <= last_h:
                struct_events.append({"type": "SWEEP", "dir": -1, "idx": t, "level": last_h, "detail": f"BSL Sweep ({fp(last_h)})"})
            elif c[t] > last_h and c[t - 1] <= last_h:
                ev_name = "BOS" if trend_15 == 1 else "CHOCH"
                struct_events.append({"type": ev_name, "dir": 1, "idx": t, "level": last_h, "detail": f"Bullish {ev_name} ({fp(last_h)})"})
                trend_15 = 1
        if known_l:
            last_l = known_l[-1]["price"]
            if l[t] < last_l and c[t] >= last_l and l[t - 1] >= last_l:
                struct_events.append({"type": "SWEEP", "dir": 1, "idx": t, "level": last_l, "detail": f"SSL Sweep ({fp(last_l)})"})
            elif c[t] < last_l and c[t - 1] >= last_l:
                ev_name = "BOS" if trend_15 == -1 else "CHOCH"
                struct_events.append({"type": ev_name, "dir": -1, "idx": t, "level": last_l, "detail": f"Bearish {ev_name} ({fp(last_l)})"})
                trend_15 = -1

    for t in range(max(2, n - 20), n):
        if c[t - 2] < o[t - 2] and c[t - 1] < o[t - 1] and c[t] > o[t - 2] and c[t - 1] <= o[t - 2]:
            struct_events.append({"type": "CISD", "dir": 1, "idx": t, "level": float(o[t - 2]), "detail": "Bullish CISD"})
        if c[t - 2] > o[t - 2] and c[t - 1] > o[t - 1] and c[t] < o[t - 2] and c[t - 1] >= o[t - 2]:
            struct_events.append({"type": "CISD", "dir": -1, "idx": t, "level": float(o[t - 2]), "detail": "Bearish CISD"})

    # Inducement (IDM) & Level IDM Spesifik
    idm_bull_swept = False
    idm_bear_swept = False
    idm_level: Optional[float] = None
    if highs_15 and lows_15:
        if trend_15 == 1:
            sh_idx = highs_15[-1]["idx"]
            pullbacks = [k for k in range(sh_idx + 1, n - 1) if l[k] < l[k - 1]]
            if pullbacks:
                idm_level = float(l[pullbacks[0]])
                idm_bull_swept = any(l[j] <= idm_level for j in range(pullbacks[0] + 1, n))
        elif trend_15 == -1:
            sl_idx = lows_15[-1]["idx"]
            pullbacks = [k for k in range(sl_idx + 1, n - 1) if h[k] > h[k - 1]]
            if pullbacks:
                idm_level = float(h[pullbacks[0]])
                idm_bear_swept = any(h[j] >= idm_level for j in range(pullbacks[0] + 1, n))

    sess_ctx = session_liquidity_context(df15)
    cvd_ctx = compute_cvd_metrics(df15)
    deriv_ctx = deriv or {}

    recent_ev = [e for e in struct_events if e["idx"] >= n - 15]
    recent_pins = [p for p in pin_bars if p["idx"] >= n - 5]

    # Cek sapuan likuiditas 1H pada 6 bar 15M terakhir (termasuk live_bar jika ada)
    h1_bull_sweep = False
    h1_bear_sweep = False
    if map1:
        for sw in map1.get("lows", [])[-4:]:
            lvl1 = float(sw["price"])
            if (any(l[n - k] < lvl1 and c[n - k] >= lvl1 for k in range(1, min(7, n))) or (live_bar and l_live < lvl1 <= cur_p)) and cur_p >= lvl1 - 0.05 * atr_1h:
                h1_bull_sweep = True
                break
        for sw in map1.get("highs", [])[-4:]:
            lvl1 = float(sw["price"])
            if (any(h[n - k] > lvl1 and c[n - k] <= lvl1 for k in range(1, min(7, n))) or (live_bar and h_live > lvl1 >= cur_p)) and cur_p <= lvl1 + 0.05 * atr_1h:
                h1_bear_sweep = True
                break
        if any((l[-1] < eq_l or l_live < eq_l) and cur_p >= eq_l for eq_l in map1.get("eql", [])[-3:]):
            h1_bull_sweep = True
        if any((h[-1] > eq_h or h_live > eq_h) and cur_p <= eq_h for eq_h in map1.get("eqh", [])[-3:]):
            h1_bear_sweep = True

    has_bull_sweep = any(e["type"] == "SWEEP" and e["dir"] == 1 for e in recent_ev) or sess_ctx["asian_ssl_swept"] or h1_bull_sweep
    has_bull_choch = any(e["type"] == "CHOCH" and e["dir"] == 1 for e in recent_ev)
    has_bull_bos = any(e["type"] == "BOS" and e["dir"] == 1 for e in recent_ev)
    has_bull_cisd = any(e["type"] == "CISD" and e["dir"] == 1 for e in recent_ev)
    has_bull_pin = any(p["dir"] == 1 for p in recent_pins)

    has_bear_sweep = any(e["type"] == "SWEEP" and e["dir"] == -1 for e in recent_ev) or sess_ctx["asian_bsl_swept"] or h1_bear_sweep
    has_bear_choch = any(e["type"] == "CHOCH" and e["dir"] == -1 for e in recent_ev)
    has_bear_bos = any(e["type"] == "BOS" and e["dir"] == -1 for e in recent_ev)
    has_bear_cisd = any(e["type"] == "CISD" and e["dir"] == -1 for e in recent_ev)
    has_bear_pin = any(p["dir"] == -1 for p in recent_pins)

    def htf_score(d_dir: int) -> float:
        al4 = 10.0 if (ctx4["st"] == d_dir or (ctx4["close"] > ctx4["e50"] and d_dir == 1) or (ctx4["close"] < ctx4["e50"] and d_dir == -1)) else 4.0
        al1 = 10.0 if (ctx1["st"] == d_dir or (ctx1["close"] > ctx1["e50"] and d_dir == 1) or (ctx1["close"] < ctx1["e50"] and d_dir == -1)) else 4.0
        return al4 + al1

    bull_htf = htf_score(1)
    bear_htf = htf_score(-1)
    of_bull_str, of_bull_pts = _format_orderflow_summary(cvd_ctx, deriv_ctx, meta.get("funding", 0.0), 1)
    of_bear_str, of_bear_pts = _format_orderflow_summary(cvd_ctx, deriv_ctx, meta.get("funding", 0.0), -1)

    def make_setup_obj(
        d_val: int,
        archetype: str,
        archetype_lbl: str,
        exec_style: str,
        app_lbl: str,
        kind_name: str,
        pat_name: str,
        poi_lbl: str,
        z_fresh: bool,
        geom: dict,
        tot_score: float,
        parts: list,
        status_str: str,
        confluences: List[str],
        extra_notes: Optional[List[str]] = None,
    ) -> Setup:
        p_lo, p_hi = geom["poi_low"], geom["poi_high"]
        plan_e = geom["plan_entry"]
        sl_p, tp1_p, tp2_p = geom["sl"], geom["tp1"], geom["tp2"]
        tp3_p = geom.get("tp3", tp2_p)
        rr_tp3_val = geom.get("rr_tp3", geom["rr"])
        liq_guard_s = geom.get("liq_guard_info", "")
        sw_entry_lvl = geom.get("sweep_entry_lvl", NAN)
        zone_s = f"{fp(p_lo)} - {fp(p_hi)}"
        conf_s = ", ".join(confluences[:3])
        conf_full = f"[{archetype_lbl} | {zone_pos_lbl} | Est. {geom['hold_label']}]: " + ", ".join(confluences)
        if d_val == 1:
            inval_s = f"Close 1H/15M menembus bawah SL struktural {fp(sl_p)} (di bawah zona POI {fp(p_lo)})."
        else:
            inval_s = f"Close 1H/15M menembus atas SL struktural {fp(sl_p)} (di atas zona POI {fp(p_hi)})."

        all_notes = list(geom["notes"]) + (list(extra_notes) if extra_notes else [])
        of_str = of_bull_str if d_val == 1 else of_bear_str

        raw_data = {
            "archetype": archetype,
            "archetype_label": archetype_lbl,
            "exec_style": exec_style,
            "approach_style": app_lbl,
            "est_tap_hours": geom["est_tap_h"],
            "current_price": cur_p,
            "plan_entry": plan_e,
            "limit_entry": geom["limit_entry"],
            "at_poi": geom["at_poi"],
            "poi_low": p_lo,
            "poi_high": p_hi,
            "poi_label": poi_lbl,
            "min_sl_dist": geom["min_sl_dist"],
            "max_sl_dist": geom["max_sl_dist"],
            "tp3": tp3_p,
            "rr_tp3": rr_tp3_val,
            "liq_guard_info": liq_guard_s,
            "sweep_entry_lvl": sw_entry_lvl,
            "blocked_by_wall": bool(geom.get("blocked_by_wall", False)),
            "sl_too_wide": bool(geom.get("sl_too_wide", False)),
            "wall_supply_bot": float(wall_supply_4h["bot"]) if wall_supply_4h else NAN,
            "wall_demand_top": float(wall_demand_4h["top"]) if wall_demand_4h else NAN,
            # Lapis 4H
            "range_4h_low": r4_lo,
            "range_4h_high": r4_hi,
            "range_4h_pos_pct": r4_pos,
            "range_4h_str": range_4h_str,
            "walls_4h_str": walls_4h_str,
            "atr_4h": atr_4h,
            "obs_4h": [
                {"dir": z["dir"], "bot": z["bot"], "top": z["top"], "fresh": z.get("fresh", True), "kind": z["kind"]}
                for z in ((map4["bull_obs"][-2:] + map4["bear_obs"][-2:] + map4["breakers"][-2:]) if map4 else [])
            ],
            # Lapis 1H
            "range_1h_low": w_low,
            "range_1h_high": w_high,
            "range_1h_eq": w_eq,
            "range_1h_pos_pct": w_pos_pct,
            "zone_pos_label": zone_pos_lbl,
            "atr_1h": atr_1h,
            "sl_atr_1h_ratio": geom["sl_atr_1h_ratio"],
            "hourly_velocity": geom["hourly_vel"],
            "est_hold_hours_tp1": round(geom["est_h1"], 1),
            "est_hold_hours_tp2": round(geom["est_h2"], 1),
            "est_hold_label": geom["hold_label"],
            "obs_1h": [
                {"dir": z["dir"], "bot": z["bot"], "top": z["top"], "swept": z.get("swept", False), "fresh": z.get("fresh", True), "kind": z["kind"]}
                for z in ((map1["bull_obs"][-2:] + map1["bear_obs"][-2:] + map1["breakers"][-2:]) if map1 else [])
            ],
            "fvgs_1h": [{"dir": f["dir"], "bot": f["bot"], "top": f["top"]} for f in (map1["fvgs"][-3:] if map1 else [])],
            "eqh_1h": map1["eqh"][-3:] if map1 else [],
            "eql_1h": map1["eql"][-3:] if map1 else [],
            # Lapis 15M
            "range_high": w_high,
            "range_low": w_low,
            "equilibrium": w_eq,
            "discount_pct": w_pos_pct,
            "range_15m_low": map15["range_low"],
            "range_15m_high": map15["range_high"],
            "recent_highs": [s["price"] for s in highs_15[-4:]],
            "recent_lows": [s["price"] for s in lows_15[-4:]],
            "bull_obs": [{"top": ob["top"], "bot": ob["bot"], "swept": ob["swept"], "fresh": ob["fresh"]} for ob in map15["bull_obs"][-3:]],
            "bear_obs": [{"top": ob["top"], "bot": ob["bot"], "swept": ob["swept"], "fresh": ob["fresh"]} for ob in map15["bear_obs"][-3:]],
            "breakers_15m": [{"dir": b["dir"], "top": b["top"], "bot": b["bot"], "fresh": b.get("fresh", True), "kind": b["kind"]} for b in map15["breakers"][-2:]],
            "fvgs": [{"dir": f["dir"], "top": f["top"], "bot": f["bot"]} for f in map15["fvgs"][-4:]],
            "eqh_15m": map15["eqh"][-3:],
            "eql_15m": map15["eql"][-3:],
            "pin_bars": [{"dir": p["dir"], "wick_pct": round(p["wick_pct"], 2)} for p in pin_bars[-3:]],
            "struct_events": [e["detail"] for e in struct_events[-6:]],
            "idm_level": idm_level,
            "idm_bull_swept": idm_bull_swept,
            "idm_bear_swept": idm_bear_swept,
            # Sesi & Orderflow
            "session": sess_ctx["session"],
            "asian_high": sess_ctx["asian_high"],
            "asian_low": sess_ctx["asian_low"],
            "asian_bsl_swept": sess_ctx["asian_bsl_swept"],
            "asian_ssl_swept": sess_ctx["asian_ssl_swept"],
            "cvd_state": cvd_ctx["cvd_state"],
            "taker_buy_ratio": cvd_ctx["taker_buy_ratio"],
            "oi_change_1h_pct": deriv_ctx.get("oi_change_1h_pct", NAN),
            "oi_change_4h_pct": deriv_ctx.get("oi_change_4h_pct", NAN),
            "top_ls_ratio": deriv_ctx.get("top_ls_ratio", NAN),
            "retail_ls_ratio": deriv_ctx.get("retail_ls_ratio", NAN),
            "orderflow_summary": of_str,
            "trend_4h": trend_text(ctx4),
            "trend_1h": trend_text(ctx1),
            "atr": cur_atr,
            "vol24_usdt": meta.get("vol24", 0),
            "spread_bps": meta.get("spread_bps", 0),
            "funding_rate": meta.get("funding", 0.0),
        }

        return Setup(
            symbol=symbol,
            direction="LONG" if d_val == 1 else "SHORT",
            kind=kind_name,
            pattern=pat_name,
            bar_index=n - 1,
            candle_time=ts_utc(int(times[-1])),
            level=float(w_eq),
            entry=float(plan_e),
            market_price=float(cur_p),
            limit_entry=float(geom["limit_entry"]),
            entry_zone=zone_s,
            poi_low=float(p_lo),
            poi_high=float(p_hi),
            sl=float(sl_p),
            tp1=float(tp1_p),
            tp2=float(tp2_p),
            tp3=float(tp3_p),
            rr=float(geom["rr"]),
            rr_tp1=float(geom["rr_tp1"]),
            rr_tp3=float(rr_tp3_val),
            risk_pct=geom["risk"] / plan_e * 100.0,
            tp1_pct=abs(tp1_p - plan_e) / plan_e * 100.0,
            tp2_pct=abs(tp2_p - plan_e) / plan_e * 100.0,
            tp3_pct=abs(tp3_p - plan_e) / plan_e * 100.0,
            liq_guard_info=liq_guard_s,
            sweep_entry_lvl=float(sw_entry_lvl) if not math.isnan(sw_entry_lvl) else NAN,
            score=float(tot_score),
            python_score=float(tot_score),
            parts=parts,
            status=status_str,
            python_status=status_str,
            confirmed=bool(geom["at_poi"]),
            extended=not bool(geom["at_poi"]),
            rel_vol=cur_rv,
            rsi=cur_rsi,
            atr=cur_atr,
            atr_pct=atr_pct,
            atr_1h=atr_1h,
            atr_1h_pct=atr_1h / cur_p,
            sl_atr_1h_ratio=geom["sl_atr_1h_ratio"],
            upper_line=float(p_hi),
            lower_line=float(p_lo),
            opp_level=float(geom["opp_lvl"]) if geom["opp_lvl"] is not None else None,
            trend_4h=trend_text(ctx4),
            struct_1h=trend_text(ctx1),
            trend_4h_short=trend_short(ctx4),
            struct_1h_short=trend_short(ctx1),
            confirm_txt=conf_full,
            confirm_short=conf_s,
            invalidation=inval_s,
            notes=all_notes,
            range_4h_str=range_4h_str,
            walls_4h_str=walls_4h_str,
            range_1h_low=float(w_low),
            range_1h_high=float(w_high),
            range_1h_eq=float(w_eq),
            range_1h_pos_pct=float(w_pos_pct),
            est_hold_hours_tp1=float(geom["est_h1"]),
            est_hold_hours_tp2=float(geom["est_h2"]),
            est_hold_label=geom["hold_label"],
            session_name=sess_ctx["session"],
            cvd_state=cvd_ctx["cvd_state"],
            taker_buy_ratio=float(cvd_ctx["taker_buy_ratio"]),
            oi_change_1h_pct=float(deriv_ctx.get("oi_change_1h_pct", NAN)),
            oi_change_4h_pct=float(deriv_ctx.get("oi_change_4h_pct", NAN)),
            top_ls_ratio=float(deriv_ctx.get("top_ls_ratio", NAN)),
            retail_ls_ratio=float(deriv_ctx.get("retail_ls_ratio", NAN)),
            orderflow_summary=of_str,
            zone_fresh=z_fresh,
            archetype=archetype,
            archetype_label=archetype_lbl,
            exec_style=exec_style,
            approach_style=app_lbl,
            est_tap_hours=float(geom["est_tap_h"]),
            meta=dict(meta),
            last_candle=ts_utc(int(times[-1])),
            raw_market_data=raw_data,
        )

    candidates: List[Tuple[int, float, Setup]] = []  # (priority_tier, rank_score, Setup)

    # =================================================================
    # ARCHETYPE 1: TREND_CONT (Sistem Post-Reaction / Konfirmasi yang Dipertahankan Utuh)
    # =================================================================
    def find_best_poi_cont(d_dir: int) -> Tuple[Optional[dict], Optional[dict], str]:
        htf_cands: List[dict] = []
        if map4:
            if d_dir == 1:
                z4_c = map4["bull_obs"] + [b for b in map4["breakers"] if b["dir"] == 1]
                htf_cands.extend([z for z in z4_c if (l_live <= z["top"] + 0.35 * atr_1h) and (cur_p >= z["bot"] - 0.15 * atr_1h)])
            else:
                z4_c = map4["bear_obs"] + [b for b in map4["breakers"] if b["dir"] == -1]
                htf_cands.extend([z for z in z4_c if (h_live >= z["bot"] - 0.35 * atr_1h) and (cur_p <= z["top"] + 0.15 * atr_1h)])
        if map1:
            if d_dir == 1:
                htf_cands.extend(map1["bull_obs"] + [b for b in map1["breakers"] if b["dir"] == 1] + [f for f in map1["fvgs"] if f["dir"] == 1])
            else:
                htf_cands.extend(map1["bear_obs"] + [b for b in map1["breakers"] if b["dir"] == -1] + [f for f in map1["fvgs"] if f["dir"] == -1])

        ltf_cands: List[dict] = []
        if d_dir == 1:
            ltf_cands = (
                map15["bull_obs"]
                + [b for b in map15["breakers"] if b["dir"] == 1]
                + [f for f in map15["fvgs"] if f["dir"] == 1 and f["idx"] >= n - 40]
            )
        else:
            ltf_cands = (
                map15["bear_obs"]
                + [b for b in map15["breakers"] if b["dir"] == -1]
                + [f for f in map15["fvgs"] if f["dir"] == -1 and f["idx"] >= n - 40]
            )

        def near_price(z: dict, max_atr_1h: float) -> bool:
            if d_dir == 1:
                return (l_live <= z["top"] + max_atr_1h * atr_1h) and (cur_p >= z["bot"] - 0.25 * atr_1h)
            return (h_live >= z["bot"] - max_atr_1h * atr_1h) and (cur_p <= z["top"] + 0.25 * atr_1h)

        near_htf = [z for z in htf_cands if near_price(z, 0.90)]
        near_ltf = [z for z in ltf_cands if near_price(z, 0.65)]

        for hz in reversed(near_htf):
            for lz in reversed(near_ltf):
                if _zones_overlap(hz, lz, 0.20 * atr_1h):
                    return hz, lz, f"Nested {hz['kind']} + {lz['kind']}"

        touching_htf = [z for z in near_htf if near_price(z, 0.35)]
        if touching_htf:
            hz = touching_htf[-1]
            return hz, None, hz["kind"]

        touching_ltf = [z for z in near_ltf if near_price(z, 0.35)]
        if touching_ltf:
            lz = touching_ltf[-1]
            return (near_htf[-1] if near_htf else None), lz, lz["kind"]

        if near_htf:
            return near_htf[-1], None, near_htf[-1]["kind"]
        if near_ltf:
            return None, near_ltf[-1], near_ltf[-1]["kind"]
        return None, None, ""

    htf_bull_poi, ltf_bull_poi, poi_type_bull = find_best_poi_cont(1)
    htf_bear_poi, ltf_bear_poi, poi_type_bear = find_best_poi_cont(-1)
    has_poi_bull = (htf_bull_poi is not None) or (ltf_bull_poi is not None)
    has_poi_bear = (htf_bear_poi is not None) or (ltf_bear_poi is not None)

    bull_nested_bonus = 4.0 if (htf_bull_poi and ltf_bull_poi) else (2.0 if htf_bull_poi else 0.0)
    bear_nested_bonus = 4.0 if (htf_bear_poi and ltf_bear_poi) else (2.0 if htf_bear_poi else 0.0)
    bull_swept_flag = bool((ltf_bull_poi and ltf_bull_poi.get("swept")) or (htf_bull_poi and htf_bull_poi.get("swept")))
    bear_swept_flag = bool((ltf_bear_poi and ltf_bear_poi.get("swept")) or (htf_bear_poi and htf_bear_poi.get("swept")))

    bull_poi_score = min(25.0, (14.0 if has_poi_bull else 0.0) + (4.0 if bull_swept_flag else 0.0) + bull_nested_bonus + (8.0 if in_discount_1h else (3.0 if w_pos_pct < 62.0 else 0.0)))
    bear_poi_score = min(25.0, (14.0 if has_poi_bear else 0.0) + (4.0 if bear_swept_flag else 0.0) + bear_nested_bonus + (8.0 if in_premium_1h else (3.0 if w_pos_pct > 38.0 else 0.0)))

    bull_struct_score = (12.0 if has_bull_sweep else 0.0) + (10.0 if has_bull_choch else (7.0 if has_bull_bos else 0.0)) + (5.0 if idm_bull_swept else 0.0)
    bear_struct_score = (12.0 if has_bear_sweep else 0.0) + (10.0 if has_bear_choch else (7.0 if has_bear_bos else 0.0)) + (5.0 if idm_bear_swept else 0.0)

    bull_trigger_score = min(15.0, (9.0 if has_bull_pin else 0.0) + (7.0 if has_bull_cisd else 0.0) + (4.0 if cur_rv >= 1.4 else 0.0) + of_bull_pts * 0.6)
    bear_trigger_score = min(15.0, (9.0 if has_bear_pin else 0.0) + (7.0 if has_bear_cisd else 0.0) + (4.0 if cur_rv >= 1.4 else 0.0) + of_bear_pts * 0.6)

    bull_wall_pen = 8.0 if (wall_supply_4h and wall_supply_4h["bot"] - cur_p < 0.35 * atr_4h and cur_p >= wall_supply_4h["bot"] - 0.2 * atr_4h) else 0.0
    bear_wall_pen = 8.0 if (wall_demand_4h and cur_p - wall_demand_4h["top"] < 0.35 * atr_4h and cur_p <= wall_demand_4h["top"] + 0.2 * atr_4h) else 0.0
    counter_htf_bull_pen = 7.0 if bull_htf <= 8.0 else 0.0
    counter_htf_bear_pen = 7.0 if bear_htf <= 8.0 else 0.0

    total_bull = bull_htf * 0.85 + bull_poi_score + bull_struct_score + bull_trigger_score - bull_wall_pen - counter_htf_bull_pen
    total_bear = bear_htf * 0.85 + bear_poi_score + bear_struct_score + bear_trigger_score - bear_wall_pen - counter_htf_bear_pen

    cont_dirs: List[int] = []
    if total_bull >= total_bear:
        if total_bull >= 40.0:
            cont_dirs.append(1)
        if total_bear >= 40.0:
            cont_dirs.append(-1)
    else:
        if total_bear >= 40.0:
            cont_dirs.append(-1)
        if total_bull >= 40.0:
            cont_dirs.append(1)

    for d_cont in cont_dirs:
        htf_poi = htf_bull_poi if d_cont == 1 else htf_bear_poi
        ltf_poi = ltf_bull_poi if d_cont == 1 else ltf_bear_poi
        poi_label = poi_type_bull if d_cont == 1 else poi_type_bear

        if htf_poi and ltf_poi and _zones_overlap(htf_poi, ltf_poi, 0.20 * atr_1h):
            p_lo = max(htf_poi["bot"], ltf_poi["bot"])
            p_hi = min(htf_poi["top"], ltf_poi["top"])
            if p_hi <= p_lo:
                p_lo, p_hi = ltf_poi["bot"], ltf_poi["top"]
        elif ltf_poi:
            p_lo, p_hi = ltf_poi["bot"], ltf_poi["top"]
        elif htf_poi:
            p_lo, p_hi = htf_poi["bot"], htf_poi["top"]
        else:
            if d_cont == 1:
                base_l = float(np.min(l[-6:]))
                p_lo, p_hi = base_l, base_l + 0.35 * atr_1h
            else:
                base_h = float(np.max(h[-6:]))
                p_lo, p_hi = base_h - 0.35 * atr_1h, base_h

        htf_pts_c = bull_htf if d_cont == 1 else bear_htf
        app_code_c, app_lbl_c, _ = analyze_approach_dynamics(df15, d_cont, cur_atr, live_bar=live_bar)
        geom_c = _compute_trade_geometry(
            d_cont, "TREND_CONT", cur_p, p_lo, p_hi, htf_poi, app_code_c, symbol, cfg,
            atr_1h, cur_atr, w_low, w_high, w_eq, r4_lo, r4_hi, wall_supply_4h, wall_demand_4h,
            map1, map15, l, h, htf_pts_c, cur_rv,
        )

        confs_c: List[str] = [poi_label or ("1H/15M Demand Zone" if d_cont == 1 else "1H/15M Supply Zone"), zone_pos_lbl]
        if d_cont == 1:
            if has_bull_sweep:
                confs_c.append("SSL Sweep" + (" (Asian Low)" if sess_ctx["asian_ssl_swept"] else ""))
            if has_bull_choch:
                confs_c.append("Bullish CHOCH")
            if has_bull_bos:
                confs_c.append("Bullish BOS")
            if idm_bull_swept:
                confs_c.append("IDM Taken")
            if has_bull_cisd:
                confs_c.append("Bullish CISD")
            if has_bull_pin:
                confs_c.append("Pin Bar Rejection")
            if cvd_ctx.get("bull_absorption"):
                confs_c.append("CVD Bull Absorption")

            if has_bull_sweep and (has_bull_choch or has_bull_cisd) and has_poi_bull:
                pat_c, kind_c = "MTF Sniper Buy (Sweep + CHOCH + 1H/15M POI)", "SMC_SNIPER"
            elif htf_poi and ltf_poi:
                pat_c, kind_c = "MTF Nested Demand (1H + 15M POI Alignment)", "SMC_OB_RETEST"
            elif poi_label and "Breaker" in poi_label:
                pat_c, kind_c = "SMC Bullish Breaker Block Retest", "SMC_BREAKER"
            elif poi_label and "OB" in poi_label:
                pat_c, kind_c = "SMC Demand Bounce (Order Block Retest)", "SMC_OB_RETEST"
            elif poi_label and "FVG" in poi_label:
                pat_c, kind_c = "SMC FVG Retest Entry (Unmitigated Gap)", "SMC_FVG_FILL"
            elif has_bull_sweep:
                pat_c, kind_c = "SMC SSL Liquidity Sweep Reversal", "SMC_SWEEP"
            else:
                pat_c, kind_c = "SMC MTF Demand Setup", "SMC_DEMAND"
        else:
            if has_bear_sweep:
                confs_c.append("BSL Sweep" + (" (Asian High)" if sess_ctx["asian_bsl_swept"] else ""))
            if has_bear_choch:
                confs_c.append("Bearish CHOCH")
            if has_bear_bos:
                confs_c.append("Bearish BOS")
            if idm_bear_swept:
                confs_c.append("IDM Taken")
            if has_bear_cisd:
                confs_c.append("Bearish CISD")
            if has_bear_pin:
                confs_c.append("Pin Bar Rejection")
            if cvd_ctx.get("bear_absorption"):
                confs_c.append("CVD Bear Absorption")

            if has_bear_sweep and (has_bear_choch or has_bear_cisd) and has_poi_bear:
                pat_c, kind_c = "MTF Sniper Short (Sweep + CHOCH + 1H/15M POI)", "SMC_SNIPER"
            elif htf_poi and ltf_poi:
                pat_c, kind_c = "MTF Nested Supply (1H + 15M POI Alignment)", "SMC_OB_RETEST"
            elif poi_label and "Breaker" in poi_label:
                pat_c, kind_c = "SMC Bearish Breaker Block Retest", "SMC_BREAKER"
            elif poi_label and "OB" in poi_label:
                pat_c, kind_c = "SMC Supply Rejection (Order Block Retest)", "SMC_OB_RETEST"
            elif poi_label and "FVG" in poi_label:
                pat_c, kind_c = "SMC FVG Retest Entry (Unmitigated Gap)", "SMC_FVG_FILL"
            elif has_bear_sweep:
                pat_c, kind_c = "SMC BSL Liquidity Sweep Reversal", "SMC_SWEEP"
            else:
                pat_c, kind_c = "SMC MTF Supply Setup", "SMC_SUPPLY"

        rr_c = geom_c["rr"]
        rr_pts_c = 10.0 if rr_c >= 3.5 else (8.5 if rr_c >= 3.0 else (4.0 if rr_c >= 2.5 else 0.0))
        liq_pts_c = liquidity_score(meta, d_cont, atr_pct)
        p1 = min(20.0, htf_pts_c)
        p2 = min(25.0, bull_poi_score if d_cont == 1 else bear_poi_score)
        p3 = min(25.0, bull_struct_score if d_cont == 1 else bear_struct_score)
        p4 = min(15.0, bull_trigger_score if d_cont == 1 else bear_trigger_score)
        p5 = min(10.0, rr_pts_c)
        p6 = min(5.0, liq_pts_c)
        tot_c = max(0.0, p1 + p2 + p3 + p4 + p5 + p6 - (bull_wall_pen if d_cont == 1 else bear_wall_pen))
        parts_c = [
            ("Konteks HTF & Tren", p1, 20),
            ("Kualitas POI & Zone", p2, 25),
            ("Struktur & Liquidity", p3, 25),
            ("Trigger & Orderflow", p4, 15),
            ("Risk to Reward", p5, 10),
            ("Likuiditas Pasar", p6, 5),
        ]

        recent_pin_2 = any(p["dir"] == d_cont and p["idx"] >= n - 2 for p in pin_bars)
        recent_cisd_3 = any(e["type"] == "CISD" and e["dir"] == d_cont and e["idx"] >= n - 3 for e in struct_events)
        if d_cont == 1:
            candle_rejected = (c[-1] > o[-1] and c[-1] >= geom_c["poi_low"]) or recent_pin_2 or recent_cisd_3
        else:
            candle_rejected = (c[-1] < o[-1] and c[-1] <= geom_c["poi_high"]) or recent_pin_2 or recent_cisd_3

        htf_aligned = htf_pts_c >= 14.0
        blocked_by_4h_wall = (
            (d_cont == 1 and in_supply_4h_now)
            or (d_cont == -1 and in_demand_4h_now)
            or bool(geom_c.get("blocked_by_wall", False))
        )
        sl_too_wide_c = bool(geom_c.get("sl_too_wide", False))
        hold_ok_c = geom_c["est_h2"] <= float(getattr(cfg, "max_hold_hours", 30.0))
        tp2_move_pct = abs(geom_c["tp2"] - geom_c["plan_entry"]) / geom_c["plan_entry"] * 100.0
        notes_c: List[str] = []
        z_fresh_c = bool((ltf_poi and ltf_poi.get("fresh")) or (htf_poi and htf_poi.get("fresh")))
        dist_from_edge = (cur_p - geom_c["poi_high"]) if d_cont == 1 else (geom_c["poi_low"] - cur_p)

        if getattr(cfg, "focus_3_pillars", True):
            # PILAR 3: Wajib 100% Follow-Trend (htf_aligned), dekat POI (<= 0.35x ATR 1H), R:R >= 1:3.0, & durasi <= max_hold_hours.
            # DILARANG KERAS mengambil Counter-Trend di SMC_SNIPER (Counter-Trend hanya boleh lewat Reversal Watch!).
            is_sniper_pillar = (
                kind_c == "SMC_SNIPER"
                and htf_aligned
                and tot_c >= 75.0
                and dist_from_edge <= 0.35 * atr_1h
                and not blocked_by_4h_wall
                and not sl_too_wide_c
                and hold_ok_c
            )
            is_first_touch_fvg = (
                kind_c == "SMC_FVG_FILL"
                and geom_c["at_poi"]
                and z_fresh_c
                and tot_c >= 78.0
                and candle_rejected
                and htf_aligned
                and not blocked_by_4h_wall
                and not sl_too_wide_c
                and hold_ok_c
            )
            if is_sniper_pillar or is_first_touch_fvg:
                if (
                    geom_c["at_poi"]
                    and candle_rejected
                    and htf_aligned
                    and not blocked_by_4h_wall
                    and not sl_too_wide_c
                    and hold_ok_c
                    and tot_c >= max(float(cfg.smc_entry_score), 75.0)
                    and rr_c >= cfg.min_rr
                    and tp2_move_pct >= 1.4
                ):
                    status_c = "ENTRY READY"
                    exec_c = "ENTRY_NOW"
                    tier_c = 0
                elif is_sniper_pillar and rr_c >= cfg.min_rr and tp2_move_pct >= 1.4:
                    status_c = "WATCHLIST"
                    exec_c = "WAIT_SWEEP_TRIGGER"
                    tier_c = 2
                    notes_c.append("SMC Sniper Follow-Trend di POI namun candle 15M terakhir belum menutup searah pantulan; tunggu close konfirmasi.")
                else:
                    status_c = "REJECT"
                    exec_c = "SKIP"
                    tier_c = 5
            else:
                status_c = "REJECT"
                exec_c = "SKIP"
                tier_c = 5
                if not htf_aligned:
                    notes_c.append("Ditolak: SMC Sniper wajib 100% Follow-Trend (Counter-Trend dialihkan ke Reversal Watch).")
                elif blocked_by_4h_wall or rr_c < cfg.min_rr:
                    notes_c.append(f"Ditolak: R:R efektif 1:{rr_c:.1f} < 1:{cfg.min_rr:.1f} atau terhalang Tembok 4H.")
                elif not hold_ok_c:
                    notes_c.append(f"Ditolak: Estimasi durasi hold ({geom_c['est_h2']:.1f}j) melebihi batas {cfg.max_hold_hours:.0f}j.")
                else:
                    notes_c.append(
                        "Disaring oleh Mode Fokus 3 Pilar Paten (mengabaikan OB Retest/Breaker pasif & fokus pada Reversal Watch, Pre-Tap Limit OB Segar, dan SMC Sniper Follow-Trend >= 75)."
                    )
        else:
            if (
                geom_c["at_poi"]
                and candle_rejected
                and htf_aligned
                and not blocked_by_4h_wall
                and not sl_too_wide_c
                and hold_ok_c
                and tot_c >= cfg.smc_entry_score
                and rr_c >= cfg.min_rr
                and tp2_move_pct >= 1.4
            ):
                status_c = "ENTRY READY"
                exec_c = "ENTRY_NOW"
                tier_c = 0
            elif htf_aligned and tot_c >= cfg.smc_watch_score and rr_c >= cfg.min_rr and tp2_move_pct >= 1.2 and hold_ok_c and not blocked_by_4h_wall:
                status_c = "WATCHLIST"
                exec_c = "WAIT_SWEEP_TRIGGER"
                tier_c = 2
            else:
                status_c = "REJECT"
                exec_c = "SKIP"
                tier_c = 5

        st_cont = make_setup_obj(
            d_cont, "TREND_CONT", "Follow-Trend (Post-Reaction)", exec_c, app_lbl_c,
            kind_c, pat_c, poi_label, z_fresh_c, geom_c, tot_c, parts_c, status_c, confs_c, notes_c,
        )
        if isinstance(st_cont.raw_market_data, dict):
            st_cont.raw_market_data["htf_aligned"] = bool(htf_aligned)
            st_cont.raw_market_data["valid_rr_and_range"] = bool(
                rr_c >= max(cfg.min_rr, 3.0)
                and not blocked_by_4h_wall
                and not sl_too_wide_c
                and hold_ok_c
                and tp2_move_pct >= 1.4
            )
        candidates.append((tier_c, tot_c, st_cont))

    # =================================================================
    # ARCHETYPE 2: PRE_REACTION (Deteksi SEBELUM Harga Bereaksi di Fresh Supply/Demand/OB)
    # =================================================================
    max_stalk_dist = min(cfg.max_stalk_atr_1h * atr_1h, (cfg.max_stalk_pct / 100.0) * cur_p)

    for d_pre in (1, -1):
        htf_p = bull_htf if d_pre == 1 else bear_htf
        blocked_wall = (d_pre == 1 and in_supply_4h_now) or (d_pre == -1 and in_demand_4h_now)
        if blocked_wall:
            continue
        # Pre-Tap Limit wajib searah tren HTF (htf_p >= 14.0) dan TIDAK jual di dasar Discount / beli di pucuk Premium!
        if htf_p < 14.0:
            continue
        if d_pre == 1 and (w_pos_pct > 55.0 or r4_pos > 70.0):
            continue
        if d_pre == -1 and (w_pos_pct < 45.0 or r4_pos < 30.0):
            continue

        # Kumpulkan zona UNMITIGATED (touches == 0 / fresh == True) yang berada DI DEPAN harga (belum memantul di closed bar!)
        fresh_htf_zones: List[dict] = []
        if map4:
            z4_list = (map4["bull_obs"] + [b for b in map4["breakers"] if b["dir"] == 1]) if d_pre == 1 else (map4["bear_obs"] + [b for b in map4["breakers"] if b["dir"] == -1])
            fresh_htf_zones.extend([{**z, "_w": 3} for z in z4_list if z.get("fresh")])
        if map1:
            z1_list = (
                map1["bull_obs"] + [b for b in map1["breakers"] if b["dir"] == 1] + [f for f in map1["fvgs"] if f["dir"] == 1]
            ) if d_pre == 1 else (
                map1["bear_obs"] + [b for b in map1["breakers"] if b["dir"] == -1] + [f for f in map1["fvgs"] if f["dir"] == -1]
            )
            fresh_htf_zones.extend([{**z, "_w": 2} for z in z1_list if z.get("fresh")])

        fresh_ltf_zones = [
            z for z in (
                (map15["bull_obs"] + [b for b in map15["breakers"] if b["dir"] == 1]) if d_pre == 1
                else (map15["bear_obs"] + [b for b in map15["breakers"] if b["dir"] == -1])
            )
            if z.get("fresh")
        ]

        def is_ahead_and_unbounced(z: dict) -> Tuple[bool, float]:
            look_b = min(8, len(l))
            if d_pre == 1:
                dist = cur_p - z["top"]
                unbounced_closed = float(np.min(l[-look_b:])) > z["top"]
                ok = unbounced_closed and (cur_p >= z["bot"]) and (-0.15 * atr_1h <= dist <= max_stalk_dist)
                return ok, max(0.0, dist)
            else:
                dist = z["bot"] - cur_p
                unbounced_closed = float(np.max(h[-look_b:])) < z["bot"]
                ok = unbounced_closed and (cur_p <= z["top"]) and (-0.15 * atr_1h <= dist <= max_stalk_dist)
                return ok, max(0.0, dist)

        ahead_ltf_zones = [lz for lz in fresh_ltf_zones if is_ahead_and_unbounced(lz)[0]]
        valid_htf_pre = []
        for hz in fresh_htf_zones:
            ok_h, dist_h = is_ahead_and_unbounced(hz)
            if ok_h:
                has_nested_15 = any(_zones_overlap(hz, lz, 0.15 * atr_1h) for lz in ahead_ltf_zones)
                if "FVG" in hz["kind"] and not has_nested_15:
                    continue
                valid_htf_pre.append((dist_h, -hz["_w"], hz, has_nested_15))

        chosen_htf = None
        chosen_ltf = None
        if valid_htf_pre:
            valid_htf_pre.sort(key=lambda x: (0 if x[3] else 1, x[0], x[1]))
            _, _, chosen_htf, _ = valid_htf_pre[0]
            nested_matches = [lz for lz in ahead_ltf_zones if _zones_overlap(chosen_htf, lz, 0.18 * atr_1h)]
            if nested_matches:
                chosen_ltf = nested_matches[-1]
        elif htf_p >= 20.0 and ((d_pre == 1 and w_pos_pct <= 44.0) or (d_pre == -1 and w_pos_pct >= 56.0)):
            ltf_sweep_pre = []
            for lz in fresh_ltf_zones:
                if lz.get("swept"):
                    ok_l, dist_l = is_ahead_and_unbounced(lz)
                    if ok_l and dist_l <= max_stalk_dist:
                        ltf_sweep_pre.append((dist_l, lz))
            if ltf_sweep_pre:
                ltf_sweep_pre.sort(key=lambda x: x[0])
                chosen_ltf = ltf_sweep_pre[0][1]

        if chosen_htf is None and chosen_ltf is None:
            continue

        if chosen_htf and chosen_ltf:
            p_lo = max(chosen_htf["bot"], chosen_ltf["bot"])
            p_hi = min(chosen_htf["top"], chosen_ltf["top"])
            if p_hi <= p_lo:
                p_lo, p_hi = chosen_ltf["bot"], chosen_ltf["top"]
            poi_lbl_pre = f"Fresh Nested {chosen_htf['kind']} + {chosen_ltf['kind']}"
            z_hier_pts = 29.0
        elif chosen_htf:
            p_lo, p_hi = chosen_htf["bot"], chosen_htf["top"]
            poi_lbl_pre = f"Fresh {chosen_htf['kind']}"
            z_hier_pts = 26.0 if chosen_htf.get("tf") == "4H" else 23.0
            if chosen_htf.get("swept"):
                z_hier_pts = min(30.0, z_hier_pts + 3.0)
        else:
            p_lo, p_hi = chosen_ltf["bot"], chosen_ltf["top"]
            poi_lbl_pre = f"Fresh {chosen_ltf['kind']}"
            z_hier_pts = 20.0

        app_code_p, app_lbl_p, app_pts = analyze_approach_dynamics(df15, d_pre, cur_atr, live_bar=live_bar)
        geom_p = _compute_trade_geometry(
            d_pre, "PRE_REACTION", cur_p, p_lo, p_hi, chosen_htf, app_code_p, symbol, cfg,
            atr_1h, cur_atr, w_low, w_high, w_eq, r4_lo, r4_hi, wall_supply_4h, wall_demand_4h,
            map1, map15, l, h, htf_p, cur_rv,
        )
        ref_edge_dist = (cur_p - geom_p["poi_high"]) if d_pre == 1 else (geom_p["poi_low"] - cur_p)
        if (
            geom_p["est_tap_h"] > 2.0
            or ref_edge_dist > max_stalk_dist
            or geom_p.get("sl_too_wide", False)
            or geom_p.get("blocked_by_wall", False)
            or geom_p["est_h2"] > float(getattr(cfg, "max_hold_hours", 30.0))
        ):
            continue

        # Skor Kekuatan Displacement Awal Pembentuk Zona (0..18)
        ref_z = chosen_htf or chosen_ltf
        disp_v = float(ref_z.get("disp_atr", 1.1))
        has_fvg_z = bool(ref_z.get("has_fvg", True))
        disp_pts = 18.0 if (disp_v >= 1.6 and has_fvg_z) else (14.5 if (disp_v >= 1.15 or has_fvg_z) else 10.0)

        # Skor Inducement / Kolam Likuiditas Pengumpan di Depan POI (0..18)
        pre_pools: List[str] = []
        if d_pre == 1:
            all_l_pools = (map1["eql"] if map1 else []) + map15["eql"] + [s["price"] for s in lows_15[-4:]]
            if idm_level:
                all_l_pools.append(idm_level)
            near_pools = [x for x in all_l_pools if geom_p["poi_low"] <= x <= cur_p and (x - geom_p["poi_high"]) <= 0.75 * atr_1h]
            if near_pools:
                pre_pools.append(f"SSL/IDM Fuel @ {fp(near_pools[0])}")
        else:
            all_h_pools = (map1["eqh"] if map1 else []) + map15["eqh"] + [s["price"] for s in highs_15[-4:]]
            if idm_level:
                all_h_pools.append(idm_level)
            near_pools = [x for x in all_h_pools if cur_p <= x <= geom_p["poi_high"] and (geom_p["poi_low"] - x) <= 0.75 * atr_1h]
            if near_pools:
                pre_pools.append(f"BSL/IDM Fuel @ {fp(near_pools[0])}")

        if pre_pools:
            idm_fuel_pts = 17.0
        elif (d_pre == 1 and idm_bull_swept) or (d_pre == -1 and idm_bear_swept):
            idm_fuel_pts = 14.0
            pre_pools.append("IDM Taken")
        else:
            idm_fuel_pts = 8.0

        # Skor Konteks HTF, Posisi Range, R:R >= 1:3, & Orderflow (0..20)
        rr_p = geom_p["rr"]
        ctx_rr_pts = (
            (8.0 if htf_p >= 20.0 else (5.5 if htf_p >= 14.0 else 3.0))
            + (4.0 if ((d_pre == 1 and w_pos_pct <= 40.0) or (d_pre == -1 and w_pos_pct >= 60.0)) else 1.5)
            + (6.0 if rr_p >= 3.5 else (5.0 if rr_p >= 3.0 else (2.0 if rr_p >= 2.5 else 0.0)))
            + min(2.0, (of_bull_pts if d_pre == 1 else of_bear_pts) * 0.4)
        )
        ctx_rr_pts = min(20.0, ctx_rr_pts)

        tot_pre = clamp(z_hier_pts + disp_pts + idm_fuel_pts + app_pts + ctx_rr_pts, 0.0, 100.0)
        parts_pre = [
            ("Kesegaran & Hirarki POI", round(z_hier_pts, 1), 30),
            ("Kekuatan Displacement+FVG", round(disp_pts, 1), 18),
            ("Inducement / Magnet Depan", round(idm_fuel_pts, 1), 18),
            ("Dinamika Pendekatan Harga", round(app_pts, 1), 14),
            ("Konteks HTF, RR & Flow", round(ctx_rr_pts, 1), 20),
        ]

        confs_pre = [poi_lbl_pre, "Unmitigated (0 Touches)", app_lbl_p.split(" (")[0], zone_pos_lbl]
        confs_pre.extend(pre_pools[:1])

        tp2_pct_p = abs(geom_p["tp2"] - geom_p["plan_entry"]) / geom_p["plan_entry"] * 100.0
        notes_pre: List[str] = []
        is_ob_anchor = (
            (chosen_htf is not None and "OB" in chosen_htf["kind"] and chosen_htf.get("caused_bos", True))
            or (chosen_ltf is not None and "OB" in chosen_ltf["kind"] and chosen_ltf.get("swept"))
        )
        strong_htf_backing = is_ob_anchor and htf_p >= 14.0 and (disp_v >= 1.15 or has_fvg_z)

        if not is_ob_anchor or not strong_htf_backing:
            continue
        if tot_pre >= 75.0 and rr_p >= max(cfg.min_rr, 3.0) and tp2_pct_p >= 1.50:
            if app_code_p == "IMPULSIF_LIAR":
                status_pre = "WATCHLIST"
                exec_pre = "WAIT_SWEEP_TRIGGER"
                tier_pre = 2
                notes_pre.append("Zona OB POI sangat segar (0 touch), namun pendekatan candle saat ini impulsif; tunggu tusukan sweep/CISD sebelum entry.")
            else:
                status_pre = "PRE-TAP LIMIT"
                exec_pre = "LIMIT_PRE_TAP"
                tier_pre = 1
        else:
            continue

        pat_pre = (
            f"Pre-Tap Fresh Demand Stalker ({chosen_htf['tf'] if chosen_htf else '15M'} Unmitigated POI)"
            if d_pre == 1
            else f"Pre-Tap Fresh Supply Stalker ({chosen_htf['tf'] if chosen_htf else '15M'} Unmitigated POI)"
        )
        st_pre = make_setup_obj(
            d_pre, "PRE_REACTION", "Pre-Reaction (Sebelum Bereaksi)", exec_pre, app_lbl_p,
            "SMC_PRE_TAP", pat_pre, poi_lbl_pre, True, geom_p, tot_pre, parts_pre, status_pre, confs_pre, notes_pre,
        )
        candidates.append((tier_pre, tot_pre, st_pre))

    # =================================================================
    # ARCHETYPE 3: REVERSAL WATCH SMC BERTAHAP (4H -> H1 -> M15, LONG & SHORT)
    # =================================================================
    # Urutan Tahap Objektif:
    # Tahap 0: Konteks 4H (Discount <50% untuk LONG / Premium >50% untuk SHORT + POI 4H BOS + Likuiditas SSL/BSL)
    # Tahap 1: WATCH (Menunggu Sweep di dekat/dalam POI 4H)
    # Tahap 2: Sweep Terkonfirmasi (Wick >= 0.1x ATR + Reclaim/Reject <= 3 candle H1)
    # Tahap 3: MSS di H1 (CHoCH H1 + Displacement >= 1.5x body20 + FVG H1 dalam <= 20 candle H1)
    # Tahap 4: Retrace ke POI H1 / OTE 62-79% (OB H1 + FVG H1 + OTE, belum lari > 50% impuls berikutnya)
    # Tahap 5: Trigger di M15 (CHoCH M15 + Displacement di dalam POI H1/4H)
    # Tahap 6: SL di balik Low/High Sweep + Target TP1 (BSL/SSL H1), TP2 (EQ 4H, wajib RR efektif >= 1:3.0), TP3 (Supply/Demand 4H)
    max_rev_dist = min(cfg.max_stalk_atr_1h * atr_1h, (cfg.max_stalk_pct / 100.0) * cur_p)

    btc_4h_ctx = btc.get("4h") if isinstance(btc, dict) else None
    btc_1h_ctx = btc.get("1h") if isinstance(btc, dict) else None

    for d_rev in (1, -1):
        # --- TAHAP 0: KONTEKS 4H ---
        # 1 & 2. Harga wajib berada di Discount 4H (<= 50%) untuk LONG, atau Premium 4H (>= 50%) untuk SHORT.
        if d_rev == 1 and r4_pos > 50.0:
            continue
        if d_rev == -1 and r4_pos < 50.0:
            continue

        # 3. Cari POI 4H / Demand-Supply Base 4H / POI 1H di area Discount 4H (<50%) atau Premium 4H (>50%)
        rev_pois: List[dict] = []
        if map4:
            z4_list = (
                (map4["bull_obs"] + [b for b in map4["breakers"] if b["dir"] == 1])
                if d_rev == 1
                else (map4["bear_obs"] + [b for b in map4["breakers"] if b["dir"] == -1])
            )
            for z in z4_list:
                if z.get("touches", 0) <= 3:
                    rev_pois.append({**z, "_w": 3})
        if map1:
            z1_list = (
                (map1["bull_obs"] + [b for b in map1["breakers"] if b["dir"] == 1])
                if d_rev == 1
                else (map1["bear_obs"] + [b for b in map1["breakers"] if b["dir"] == -1])
            )
            for z1 in z1_list:
                if z1.get("touches", 0) <= 2:
                    if (d_rev == 1 and r4_pos <= 45.0) or (d_rev == -1 and r4_pos >= 55.0) or z1.get("swept"):
                        rev_pois.append({**z1, "_w": 2})

        # Tambahkan juga basis Demand/Supply di ujung Dealing Range 4H (<= 24% atau >= 76%) jika harga sedang menguji low/high range 4H
        if (d_rev == 1 and r4_pos <= 24.0) or (d_rev == -1 and r4_pos >= 76.0):
            if d_rev == 1:
                rev_pois.append({
                    "dir": 1,
                    "bot": float(r4_lo),
                    "top": float(r4_lo + 0.35 * atr_4h),
                    "touches": 1,
                    "fresh": False,
                    "swept": bool(h1_bull_sweep or (map1 and map1.get("ssl_sweeps"))),
                    "has_fvg": True,
                    "has_disp": True,
                    "caused_bos": True,
                    "disp_atr": 1.4,
                    "tf": "4H",
                    "kind": "4H Demand Base (Low Range)",
                    "_w": 2,
                })
            else:
                rev_pois.append({
                    "dir": -1,
                    "bot": float(r4_hi - 0.35 * atr_4h),
                    "top": float(r4_hi),
                    "touches": 1,
                    "fresh": False,
                    "swept": bool(h1_bear_sweep or (map1 and map1.get("bsl_sweeps"))),
                    "has_fvg": True,
                    "has_disp": True,
                    "caused_bos": True,
                    "disp_atr": 1.4,
                    "tf": "4H",
                    "kind": "4H Supply Base (High Range)",
                    "_w": 2,
                })

        if not rev_pois:
            continue

        # Filter POI yang sedang diuji atau berada dalam jarak dekat (<= max_rev_dist) agar tidak menunggu lama
        near_rev_pois = []
        for wz in rev_pois:
            if d_rev == 1:
                # Close 4H di bawah dasar POI 4H -> Pembatalan (Bagian C)
                if wz["_w"] == 3 and float(ctx4.get("close", cur_p)) < wz["bot"] - 0.08 * atr_4h:
                    continue
                in_w = (wz["bot"] - 0.20 * atr_1h) <= cur_p <= (wz["top"] + 0.15 * atr_1h)
                dist_w = 0.0 if in_w else (cur_p - wz["top"] if cur_p > wz["top"] else (wz["bot"] - cur_p))
            else:
                # Close 4H di atas puncak POI 4H -> Pembatalan (Bagian C)
                if wz["_w"] == 3 and float(ctx4.get("close", cur_p)) > wz["top"] + 0.08 * atr_4h:
                    continue
                in_w = (wz["bot"] - 0.15 * atr_1h) <= cur_p <= (wz["top"] + 0.20 * atr_1h)
                dist_w = 0.0 if in_w else (wz["bot"] - cur_p if cur_p < wz["bot"] else (cur_p - wz["top"]))

            if in_w or (0.0 <= dist_w <= max_rev_dist):
                # Bagian E (Aturan Penentu): Pilih yang fresh (touches == 0), paling dekat, dan konfluensi terbanyak
                conf_cnt = (1 if wz.get("fresh") else 0) + (1 if wz.get("has_fvg") else 0) + (1 if wz.get("has_disp") else 0) + (1 if wz.get("swept") else 0)
                near_rev_pois.append((0 if in_w else 1, -wz["_w"], wz.get("touches", 0), dist_w, -conf_cnt, wz))

        if not near_rev_pois:
            continue

        near_rev_pois.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
        in_poi_4h = near_rev_pois[0][0] == 0
        dist_poi_4h = near_rev_pois[0][3]
        wall_z = near_rev_pois[0][5]

        # --- CEK KONDISI PEMBATALAN (BAGIAN C) ---
        # 1. Breakdown (LONG) atau Breakout (SHORT) selama >= 3 candle H1 tanpa reclaim/reject
        if map1:
            if d_rev == 1 and map1.get("is_breakdown", False):
                continue
            if d_rev == -1 and map1.get("is_breakout", False):
                continue

        # 2. Cek Relative Strength/Weakness vs BTC (Bagian D) & Pembatalan Struktur BTC 4H (Bagian C)
        rel_vs_btc_ok = False
        if btc_1h_ctx and symbol != "BTCUSDT":
            coin_pos_1h = w_pos_pct
            btc_lo1 = float(btc_1h_ctx.get("lo", 0.0))
            btc_hi1 = float(btc_1h_ctx.get("hi", 1.0))
            btc_c1 = float(btc_1h_ctx.get("close", 0.5))
            btc_pos_1h = (btc_c1 - btc_lo1) / max(btc_hi1 - btc_lo1, 1e-12) * 100.0
            if d_rev == 1 and coin_pos_1h >= btc_pos_1h + 5.0:
                rel_vs_btc_ok = True  # Koin tidak membuat low sedalam BTC saat BTC turun
            elif d_rev == -1 and coin_pos_1h <= btc_pos_1h - 5.0:
                rel_vs_btc_ok = True  # Koin membuat high lebih lemah saat BTC naik

        btc_aligned = True
        if symbol != "BTCUSDT" and btc_4h_ctx:
            btc_st4 = int(btc_4h_ctx.get("st", 0))
            btc_c4 = float(btc_4h_ctx.get("close", 0.0))
            btc_lo4 = float(btc_4h_ctx.get("lo", btc_c4))
            btc_hi4 = float(btc_4h_ctx.get("hi", btc_c4))
            # Batalkan setup altcoin jika BTC sedang jebol aktif menembus Low/High 4H TANPA adanya Relative Strength pada altcoin
            if d_rev == 1 and btc_st4 == -1 and btc_c4 <= btc_lo4 * 1.002 and not rel_vs_btc_ok:
                continue
            if d_rev == -1 and btc_st4 == 1 and btc_c4 >= btc_hi4 * 0.998 and not rel_vs_btc_ok:
                continue
            btc_aligned = (d_rev == 1 and (btc_st4 >= 0 or int(btc_1h_ctx.get("st", 0)) >= 0)) or (
                d_rev == -1 and (btc_st4 <= 0 or int(btc_1h_ctx.get("st", 0)) <= 0)
            )

        # 4 (Tahap 0): Cek keberadaan Likuiditas SSL (LONG) / BSL (SHORT) di dekat atau di dalam POI
        liq_pools_near: List[Tuple[float, str]] = []
        if d_rev == 1:
            if map4:
                for eq4 in map4.get("eql", []):
                    if wall_z["bot"] - 0.50 * atr_1h <= eq4 <= wall_z["top"] + 0.80 * atr_1h:
                        liq_pools_near.append((float(eq4), "EQL 4H"))
            if map1:
                for eq1 in map1.get("eql", []):
                    if wall_z["bot"] - 0.50 * atr_1h <= eq1 <= wall_z["top"] + 0.80 * atr_1h:
                        liq_pools_near.append((float(eq1), "EQL 1H"))
                for sw1 in map1.get("lows", [])[-5:]:
                    if wall_z["bot"] - 0.45 * atr_1h <= sw1["price"] <= wall_z["top"] + 0.65 * atr_1h:
                        liq_pools_near.append((float(sw1["price"]), "SSL 1H"))
            if abs(w_low - wall_z["bot"]) <= 0.80 * atr_1h:
                liq_pools_near.append((float(w_low), "PDL/Low 1H"))
            if abs(r4_lo - wall_z["bot"]) <= 0.90 * atr_1h:
                liq_pools_near.append((float(r4_lo), "Low Range 4H"))
        else:
            if map4:
                for eq4 in map4.get("eqh", []):
                    if wall_z["bot"] - 0.80 * atr_1h <= eq4 <= wall_z["top"] + 0.50 * atr_1h:
                        liq_pools_near.append((float(eq4), "EQH 4H"))
            if map1:
                for eq1 in map1.get("eqh", []):
                    if wall_z["bot"] - 0.80 * atr_1h <= eq1 <= wall_z["top"] + 0.50 * atr_1h:
                        liq_pools_near.append((float(eq1), "EQH 1H"))
                for sw1 in map1.get("highs", [])[-5:]:
                    if wall_z["bot"] - 0.65 * atr_1h <= sw1["price"] <= wall_z["top"] + 0.45 * atr_1h:
                        liq_pools_near.append((float(sw1["price"]), "BSL 1H"))
            if abs(w_high - wall_z["top"]) <= 0.80 * atr_1h:
                liq_pools_near.append((float(w_high), "PDH/High 1H"))
            if abs(r4_hi - wall_z["top"]) <= 0.90 * atr_1h:
                liq_pools_near.append((float(r4_hi), "High Range 4H"))

        # --- EVALUASI TAHAP 2: SWEEP TERKONFIRMASI (RECLAIM/REJECT <= 3 CANDLE H1) ---
        sw_1h_list = (map1.get("ssl_sweeps", []) if d_rev == 1 else map1.get("bsl_sweeps", [])) if map1 else []
        sw_4h_list = (map4.get("ssl_sweeps", []) if d_rev == 1 else map4.get("bsl_sweeps", [])) if map4 else []
        n1_bars = len(df1) if df1 is not None else 0

        recent_h1_sweeps = [
            s for s in sw_1h_list
            if (n1_bars - 1 - int(s["idx"])) <= 20
            and (
                (d_rev == 1 and s["extreme"] <= wall_z["top"] + 0.55 * atr_1h and s["extreme"] >= wall_z["bot"] - 0.80 * atr_1h)
                or (d_rev == -1 and s["extreme"] >= wall_z["bot"] - 0.55 * atr_1h and s["extreme"] <= wall_z["top"] + 0.80 * atr_1h)
            )
        ]
        has_4h_sweep = len(sw_4h_list) > 0
        has_live_h1_sweep = h1_bull_sweep if d_rev == 1 else h1_bear_sweep
        stage2_sweep_ok = bool(recent_h1_sweeps or has_4h_sweep or (in_poi_4h and has_live_h1_sweep))
        is_double_sweep = any(s.get("double_sweep") for s in recent_h1_sweeps) or any(s.get("double_sweep") for s in sw_4h_list)
        has_vol_spike_sweep = any(s.get("vol_spike") for s in recent_h1_sweeps) or cur_rv >= 1.45

        # Bagian E: Jika ada rangkaian sweep berurutan, gunakan low terendah / high tertinggi dari rangkaian sweep untuk SL
        sweep_extreme = NAN
        if recent_h1_sweeps:
            if d_rev == 1:
                sweep_extreme = float(min(s["extreme"] for s in recent_h1_sweeps))
                # Batal jika Close H1 saat ini sudah di bawah low sweep (Bagian C)
                if float(ctx1.get("close", cur_p)) < sweep_extreme:
                    continue
            else:
                sweep_extreme = float(max(s["extreme"] for s in recent_h1_sweeps))
                # Batal jika Close H1 saat ini sudah di atas high sweep (Bagian C)
                if float(ctx1.get("close", cur_p)) > sweep_extreme:
                    continue
        elif sw_4h_list:
            sweep_extreme = float(sw_4h_list[-1]["extreme"])

        # Jika belum ada sweep DAN tidak ada kolam likuiditas SSL/BSL di dekat POI, batalkan Tahap 0
        if not stage2_sweep_ok and not liq_pools_near:
            continue

        # --- EVALUASI TAHAP 3: MSS DI H1 (CHoCH H1 + DISPLACEMENT + FVG H1 <= 20 BAR) ---
        mss_h1 = (map1.get("mss_bull") if d_rev == 1 else map1.get("mss_bear")) if map1 else None
        stage3_mss_h1_ok = False
        has_fvg_h1 = False
        ote_bot = NAN
        ote_top = NAN
        in_ote_zone = False

        if mss_h1 and int(mss_h1.get("bars_since_sweep", 99)) <= 20:
            # Pastikan MSS H1 masih relevan dengan posisi harga saat ini (belum kedaluwarsa & belum ditembus balik)
            if d_rev == 1:
                imp_lo = float(mss_h1["sweep_low"])
                imp_hi = float(mss_h1["impulse_high"])
            else:
                imp_hi = float(mss_h1["sweep_high"])
                imp_lo = float(mss_h1["impulse_low"])
            imp_span = max(imp_hi - imp_lo, 1e-12)
            if d_rev == 1:
                # Batal jika harga close di bawah low sweep impuls MSS, atau sudah lari jauh > 50% impuls berikutnya tanpa retrace
                if cur_p < imp_lo or cur_p > imp_hi + 0.50 * imp_span:
                    mss_h1 = None
                else:
                    stage3_mss_h1_ok = True
                    has_fvg_h1 = bool(mss_h1.get("has_fvg", False))
                    ote_bot = float(mss_h1["ote_bot"])
                    ote_top = float(mss_h1["ote_top"])
                    if math.isnan(sweep_extreme):
                        sweep_extreme = imp_lo
                    in_ote_zone = (ote_bot - 0.10 * atr_1h) <= cur_p <= (ote_top + 0.15 * atr_1h)
            else:
                if cur_p > imp_hi or cur_p < imp_lo - 0.50 * imp_span:
                    mss_h1 = None
                else:
                    stage3_mss_h1_ok = True
                    has_fvg_h1 = bool(mss_h1.get("has_fvg", False))
                    ote_bot = float(mss_h1["ote_bot"])
                    ote_top = float(mss_h1["ote_top"])
                    if math.isnan(sweep_extreme):
                        sweep_extreme = imp_hi
                    in_ote_zone = (ote_bot - 0.15 * atr_1h) <= cur_p <= (ote_top + 0.10 * atr_1h)

        # Cek FVG H1 dan OB H1 aktif di sekitar POI
        h1_poi_ob = None
        h1_poi_fvg = None
        if map1:
            obs_1h_dir = map1["bull_obs"] if d_rev == 1 else map1["bear_obs"]
            for z1 in reversed(obs_1h_dir):
                if _zones_overlap(z1, wall_z, 0.35 * atr_1h) or (
                    not math.isnan(ote_bot) and _zones_overlap(z1, {"bot": ote_bot, "top": ote_top}, 0.20 * atr_1h)
                ):
                    h1_poi_ob = z1
                    break
            fvgs_1h_dir = [f for f in map1["fvgs"] if f["dir"] == d_rev]
            for f1 in reversed(fvgs_1h_dir):
                if _zones_overlap(f1, wall_z, 0.35 * atr_1h) or (
                    not math.isnan(ote_bot) and _zones_overlap(f1, {"bot": ote_bot, "top": ote_top}, 0.20 * atr_1h)
                ):
                    h1_poi_fvg = f1
                    has_fvg_h1 = True
                    break

        # --- EVALUASI TAHAP 4: RETRACE KE POI H1 / OTE 62-79% ---
        p_lo, p_hi = wall_z["bot"], wall_z["top"]
        if stage3_mss_h1_ok and not math.isnan(ote_bot) and not math.isnan(ote_top):
            # Gunakan irisan POI H1 + OTE 62-79% jika tersedia
            if h1_poi_ob and _zones_overlap(h1_poi_ob, {"bot": ote_bot, "top": ote_top}, 0.25 * atr_1h):
                p_lo = min(h1_poi_ob["bot"], ote_bot)
                p_hi = max(h1_poi_ob["top"], ote_top)
            else:
                p_lo = min(wall_z["bot"], ote_bot)
                p_hi = max(wall_z["top"], ote_top)
                if (p_hi - p_lo) > 0.90 * atr_1h:
                    p_lo, p_hi = ote_bot, ote_top

        stage4_retrace_ok = bool(
            stage3_mss_h1_ok
            and (
                in_ote_zone
                or (h1_poi_ob and (h1_poi_ob["bot"] - 0.10 * atr_1h <= cur_p <= h1_poi_ob["top"] + 0.10 * atr_1h))
                or (p_lo - 0.10 * atr_1h <= cur_p <= p_hi + 0.10 * atr_1h)
            )
        )

        # --- EVALUASI TAHAP 5: TRIGGER DI M15 (CHoCH M15 + DISPLACEMENT DI DALAM POI H1/4H) ---
        mss_15 = map15.get("mss_bull") if d_rev == 1 else map15.get("mss_bear")
        has_m15_mini_sweep = (has_bull_sweep if d_rev == 1 else has_bear_sweep) or bool(
            map15.get("ssl_sweeps") if d_rev == 1 else map15.get("bsl_sweeps")
        )
        has_m15_choch_disp = bool(
            mss_15 is not None
            or ((has_bull_choch if d_rev == 1 else has_bear_choch) and any(
                f["dir"] == d_rev and f["idx"] >= n - 12 for f in map15.get("fvgs", [])
            ))
        )
        in_active_poi_now = (p_lo - 0.10 * atr_1h) <= cur_p <= (p_hi + 0.10 * atr_1h) or in_poi_4h
        stage5_m15_trigger_ok = bool(
            in_active_poi_now
            and (stage2_sweep_ok or stage3_mss_h1_ok)
            and has_m15_choch_disp
        )

        # Jika Tahap 5 sudah aktif dan ada OB M15 atau FVG M15 hasil CHoCH di dekat harga, pertajam zona entry M15
        m15_entry_ob = None
        m15_entry_fvg = None
        obs_15_dir = map15["bull_obs"] if d_rev == 1 else map15["bear_obs"]
        for z15 in reversed(obs_15_dir[-3:]):
            if _zones_overlap(z15, {"bot": p_lo, "top": p_hi}, 0.25 * atr_1h):
                m15_entry_ob = z15
                break
        fvgs_15_dir = [f for f in map15["fvgs"][-4:] if f["dir"] == d_rev]
        for f15 in reversed(fvgs_15_dir):
            if _zones_overlap(f15, {"bot": p_lo, "top": p_hi}, 0.25 * atr_1h):
                m15_entry_fvg = f15
                break

        app_code_r, app_lbl_r, _ = analyze_approach_dynamics(df15, d_rev, cur_atr, live_bar=live_bar)

        # Bagian E (Khusus SHORT): Hindari short koin yang sedang parabolic pump liar kecuali Sweep BSL + MSS H1 sudah terkonfirmasi!
        if d_rev == -1 and app_code_r == "IMPULSIF_LIAR" and not (stage2_sweep_ok and (stage3_mss_h1_ok or stage5_m15_trigger_ok)):
            continue

        # --- TAHAP 6: GEOMETRI SL, TP1, TP2 (EQ 4H), TP3 & UJI R:R EFEKTIF >= 1:3.0 ---
        htf_r = bull_htf if d_rev == 1 else bear_htf
        rev_ctx_dict = {
            "sweep_extreme": sweep_extreme,
            "ote_bot": ote_bot,
            "ote_top": ote_top,
        }
        geom_r = _compute_trade_geometry(
            d_rev, "HTF_REVERSAL", cur_p, p_lo, p_hi, wall_z, app_code_r, symbol, cfg,
            atr_1h, cur_atr, w_low, w_high, w_eq, r4_lo, r4_hi, wall_supply_4h, wall_demand_4h,
            map1, map15, l, h, htf_r, cur_rv, rev_ctx=rev_ctx_dict,
        )

        rr_r = geom_r["rr"]
        tp2_pct_r = abs(geom_r["tp2"] - geom_r["plan_entry"]) / geom_r["plan_entry"] * 100.0
        hold_ok_r = geom_r["est_h2"] <= float(getattr(cfg, "max_hold_hours", 30.0))

        # --- KONFIRMASI TAMBAHAN (BAGIAN D): DERIVATIF & VOLUME CLIMAX ---
        funding_val = float(meta.get("funding", 0.0) or 0.0)
        cvd_abs = bool(cvd_ctx.get("bull_absorption") if d_rev == 1 else cvd_ctx.get("bear_absorption"))
        deriv_fav = (
            (d_rev == 1 and funding_val < -0.00005)
            or (d_rev == -1 and funding_val > 0.00010)
            or cvd_abs
            or (of_bull_pts >= 2.5 if d_rev == 1 else of_bear_pts >= 2.5)
        )

        # --- SKORING 10 POIN OBJEKTIF (BAGIAN F & G: 10 KOMPONEN x 10 = 100 POIN) ---
        # 1. Di Discount 4H (LONG) / Di Premium 4H (SHORT)
        s1_disc_prem = 10.0 if ((d_rev == 1 and r4_pos <= 35.0) or (d_rev == -1 and r4_pos >= 65.0)) else 8.0
        # 2. POI 4H Fresh dan Menyebabkan BOS
        if wall_z["_w"] == 3 and wall_z.get("fresh") and wall_z.get("caused_bos", True):
            s2_poi_4h = 10.0
        elif wall_z.get("caused_bos", True) and wall_z.get("touches", 0) <= 1:
            s2_poi_4h = 8.0
        else:
            s2_poi_4h = 5.5
        # 3. Sweep SSL/BSL Jelas dan Reclaim/Reject Cepat (<= 3 bar H1)
        if stage2_sweep_ok and (is_double_sweep or has_4h_sweep or len(liq_pools_near) >= 2):
            s3_sweep = 10.0
        elif stage2_sweep_ok:
            s3_sweep = 8.5
        elif has_m15_mini_sweep and in_poi_4h:
            s3_sweep = 6.5
        else:
            s3_sweep = 4.0  # Tahap 1: Kolam likuiditas siap di POI, menunggu sapuan
        # 4. MSS H1 dengan Displacement
        if stage3_mss_h1_ok and mss_h1 and mss_h1.get("has_disp"):
            s4_mss_h1 = 10.0
        elif stage3_mss_h1_ok or (stage2_sweep_ok and app_code_r == "KOMPRESI_KOREKTIF"):
            s4_mss_h1 = 7.5
        else:
            s4_mss_h1 = 4.0
        # 5. FVG H1 Meninggalkan Imbalance
        if has_fvg_h1:
            s5_fvg_h1 = 10.0
        elif wall_z.get("has_fvg") or m15_entry_fvg is not None:
            s5_fvg_h1 = 7.5
        else:
            s5_fvg_h1 = 4.0
        # 6. CHoCH M15 di Dalam POI H1/4H
        if stage5_m15_trigger_ok:
            s6_choch_m15 = 10.0
        elif in_active_poi_now and (has_bull_choch or has_bull_cisd if d_rev == 1 else has_bear_choch or has_bear_cisd):
            s6_choch_m15 = 8.0
        elif in_active_poi_now and (has_bull_pin if d_rev == 1 else has_bear_pin):
            s6_choch_m15 = 6.0
        else:
            s6_choch_m15 = 3.5
        # 7. Konfluensi OB + FVG + OTE (62-79%)
        conf_ob = (h1_poi_ob is not None) or (m15_entry_ob is not None) or ("OB" in wall_z["kind"])
        conf_fvg = has_fvg_h1 or (m15_entry_fvg is not None) or bool(wall_z.get("has_fvg"))
        conf_ote = in_ote_zone or ((d_rev == 1 and w_pos_pct <= 32.0) or (d_rev == -1 and w_pos_pct >= 68.0))
        conf_count_3 = int(bool(conf_ob)) + int(bool(conf_fvg)) + int(bool(conf_ote))
        s7_confluence = 10.0 if conf_count_3 == 3 else (8.0 if conf_count_3 == 2 else 5.0)
        # 8. Struktur BTC Mendukung & Relative Strength/Weakness
        if btc_aligned and rel_vs_btc_ok:
            s8_btc = 10.0
        elif btc_aligned or rel_vs_btc_ok:
            s8_btc = 7.5
        else:
            s8_btc = 5.0
        # 9. Derivatif (Funding/OI) & Volume Spike Sweep Mendukung
        if deriv_fav and has_vol_spike_sweep:
            s9_deriv = 10.0
        elif deriv_fav or has_vol_spike_sweep:
            s9_deriv = 7.5
        else:
            s9_deriv = 5.0
        # 10. RR Efektif >= 1:3 Tanpa Supply/Demand 4H Penghalang
        if rr_r >= 3.5 and not geom_r.get("blocked_by_wall", False):
            s10_rr = 10.0
        elif rr_r >= 3.0 and not geom_r.get("blocked_by_wall", False):
            s10_rr = 8.5
        else:
            s10_rr = 0.0

        tot_rev = clamp(
            s1_disc_prem + s2_poi_4h + s3_sweep + s4_mss_h1 + s5_fvg_h1
            + s6_choch_m15 + s7_confluence + s8_btc + s9_deriv + s10_rr,
            0.0,
            100.0,
        )
        parts_rev = [
            ("1. Posisi 4H (Discount/Premium)", round(s1_disc_prem, 1), 10),
            ("2. POI 4H Fresh & BOS", round(s2_poi_4h, 1), 10),
            ("3. Sweep SSL/BSL & Reclaim <=3H1", round(s3_sweep, 1), 10),
            ("4. MSS H1 + Displacement", round(s4_mss_h1, 1), 10),
            ("5. FVG H1 Imbalance", round(s5_fvg_h1, 1), 10),
            ("6. Trigger CHoCH M15 di POI", round(s6_choch_m15, 1), 10),
            ("7. Konfluensi OB + FVG + OTE", round(s7_confluence, 1), 10),
            ("8. Struktur BTC & Rel. Strength", round(s8_btc, 1), 10),
            ("9. Derivatif & Vol Spike Sweep", round(s9_deriv, 1), 10),
            ("10. RR Efektif >=1:3 Tanpa Halangan", round(s10_rr, 1), 10),
        ]

        # Tentukan Tahap Aktif (Tahap 1 s/d Tahap 5) & Langkah Selanjutnya
        sw_dir_lbl = "SSL" if d_rev == 1 else "BSL"
        if stage5_m15_trigger_ok:
            stage_num = 5
            stage_lbl = "Tahap 5/5: Trigger CHoCH M15 Valid"
            next_step_txt = f"Tahap 5 Tercapai (Sweep {sw_dir_lbl} + Shift Struktur + CHoCH M15 di POI)"
        elif stage4_retrace_ok:
            stage_num = 4
            stage_lbl = "Tahap 4/5: Retrace POI H1 / OTE 62-79%"
            next_step_txt = f"Harga sudah di Retrace POI H1 / OTE [{fp(p_lo)}-{fp(p_hi)}] -> Tunggu Trigger CHoCH M15 ber-displacement"
        elif stage3_mss_h1_ok:
            stage_num = 3
            stage_lbl = "Tahap 3/5: MSS H1 Terkonfirmasi"
            next_step_txt = f"MSS H1 + Displacement valid -> Tunggu retrace ke OTE 62-79% [{fp(p_lo)}-{fp(p_hi)}] & CHoCH M15"
        elif stage2_sweep_ok:
            stage_num = 2
            stage_lbl = f"Tahap 2/5: Sweep {sw_dir_lbl} Terkonfirmasi"
            next_step_txt = f"Sweep {sw_dir_lbl} & Reclaim H1 valid -> Tunggu MSS H1 ber-displacement (<=20 bar H1) + Trigger CHoCH M15"
        else:
            stage_num = 1
            stage_lbl = f"Tahap 1/5: WATCH (Menunggu Sweep {sw_dir_lbl})"
            liq_target_str = f"{liq_pools_near[0][1]} {fp(liq_pools_near[0][0])}" if liq_pools_near else fp(p_lo if d_rev == 1 else p_hi)
            next_step_txt = f"Menunggu Sweep {sw_dir_lbl} ({liq_target_str}) di POI 4H + Reclaim <=3 candle H1 -> MSS H1 -> CHoCH M15"

        confs_rev = [
            stage_lbl,
            f"POI {wall_z['kind']}",
            f"4H {'Discount' if d_rev == 1 else 'Premium'} ({r4_pos:.0f}%)",
        ]
        if stage2_sweep_ok:
            confs_rev.append(f"Double Sweep {sw_dir_lbl}" if is_double_sweep else f"Sweep {sw_dir_lbl} Reclaimed")
        elif liq_pools_near:
            confs_rev.append(f"Target {liq_pools_near[0][1]} @ {fp(liq_pools_near[0][0])}")
        if stage3_mss_h1_ok:
            confs_rev.append("MSS H1 + Disp")
        if in_ote_zone:
            confs_rev.append("OTE 62-79%")
        if stage5_m15_trigger_ok:
            confs_rev.append("CHoCH M15 Trigger")
        if rel_vs_btc_ok:
            confs_rev.append("Rel. Strength vs BTC" if d_rev == 1 else "Rel. Weakness vs BTC")

        has_pin_r = has_bull_pin if d_rev == 1 else has_bear_pin
        if d_rev == 1:
            candle_rej_r = (c[-1] > o[-1] and c[-1] >= geom_r["poi_low"]) or has_pin_r
        else:
            candle_rej_r = (c[-1] < o[-1] and c[-1] <= geom_r["poi_high"]) or has_pin_r

        # Ambang Skor Bagian F & G:
        # - Skor >= 8/10 (80.0) diprioritaskan.
        # - LONG minimal 6.5/10 (65.0) untuk langsung masuk Watch Python (atau >= 6.0/10 dikirim ke Kurator AI).
        # - SHORT lebih ketat (Bagian E & G): minimal 7.0/10 (70.0) jika BTC bearish, atau 7.2/10 (72.0) jika BTC tidak bearish.
        btc_4h_bear = bool(btc_4h_ctx and btc_4h_ctx.get("st", 0) == -1)
        min_watch_rev = 65.0 if d_rev == 1 else (70.0 if btc_4h_bear else 72.0)
        notes_rev: List[str] = [
            f"Progres Reversal Watch SMC: {stage_lbl} | Skor Objektif: {tot_rev / 10.0:.1f}/10 ({tot_rev:.0f}/100)."
        ]

        valid_rr_and_range = (
            rr_r >= max(cfg.min_rr, 3.0)
            and not geom_r.get("blocked_by_wall", False)
            and not geom_r.get("sl_too_wide", False)
            and hold_ok_r
            and tp2_pct_r >= 1.35
        )

        if (
            valid_rr_and_range
            and stage5_m15_trigger_ok
            and candle_rej_r
            and tot_rev >= 80.0
        ):
            status_rev = "ENTRY READY"
            exec_rev = "ENTRY_NOW"
            tier_rev = 0
            pat_rev = f"Reversal Watch SMC [{stage_lbl}] ({wall_z['tf']} {'Demand' if d_rev == 1 else 'Supply'} + Sweep + CHoCH M15)"
        elif valid_rr_and_range and tot_rev >= min_watch_rev:
            status_rev = "REVERSAL WATCH"
            exec_rev = "WAIT_SWEEP_TRIGGER"
            tier_rev = 1 if (tot_rev >= 75.0 or stage_num >= 2 or in_poi_4h) else 2
            pat_rev = f"Reversal Watch SMC [{stage_lbl}] ({wall_z['kind']})"
            notes_rev.append(next_step_txt)
        else:
            status_rev = "REJECT"
            exec_rev = "SKIP"
            # Jika geometri R:R >= 1:3.0 valid dan skor >= 6.0/10, beri tier 3 agar tetap terpilih & dikirim ke API AI sebagai penentu akhir!
            tier_rev = 3 if (valid_rr_and_range and tot_rev >= 60.0) else 4
            pat_rev = f"Reversal Watch SMC [{stage_lbl}] ({wall_z['kind']})"
            if geom_r.get("sl_too_wide", False):
                r_pct = geom_r["risk"] / max(geom_r["plan_entry"], 1e-12) * 100.0
                notes_rev.append(
                    f"Ditolak (Prefer REJECT): Jarak SL di balik Sweep terlalu lebar ({r_pct:.1f}% | {geom_r['sl_atr_1h_ratio']:.2f}x ATR 1H)."
                )
            elif not valid_rr_and_range:
                notes_rev.append(f"Ditolak (Prefer REJECT): R:R efektif 1:{rr_r:.1f} < 1:{cfg.min_rr:.1f} atau terhalang Tembok 4H / batas durasi.")
            else:
                notes_rev.append(f"Ditolak Python (Menunggu Putusan AI): Skor {tot_rev / 10.0:.1f}/10 di bawah ambang awal ({min_watch_rev / 10.0:.1f}/10).")

        st_rev = make_setup_obj(
            d_rev, "HTF_REVERSAL", f"Reversal Watch ({stage_lbl.split(':')[0]})", exec_rev, app_lbl_r,
            "SMC_REVERSAL", pat_rev, wall_z["kind"], bool(wall_z.get("fresh")), geom_r, tot_rev, parts_rev, status_rev, confs_rev, notes_rev,
        )
        if isinstance(st_rev.raw_market_data, dict):
            st_rev.raw_market_data["rev_stage_num"] = stage_num
            st_rev.raw_market_data["rev_stage_label"] = stage_lbl
            st_rev.raw_market_data["rev_next_step"] = next_step_txt
            st_rev.raw_market_data["rev_score_10"] = round(tot_rev / 10.0, 2)
            st_rev.raw_market_data["valid_rr_and_range"] = bool(valid_rr_and_range)
        candidates.append((tier_rev, tot_rev, st_rev))

    if not candidates:
        return None

    # Pilih setup terbaik: utamakan Pilar 1 (HTF_REVERSAL / Reversal Watch) & Pilar 2 (PRE_REACTION) pada tier setara
    candidates.sort(
        key=lambda item: (
            item[0],
            0 if item[2].archetype == "HTF_REVERSAL" else (1 if item[2].archetype == "PRE_REACTION" else 2),
            -item[1],
        )
    )
    best_setup = candidates[0][2]

    best_d = 1 if best_setup.direction == "LONG" else -1
    enrich_setup_with_zone_radar(
        best_setup,
        df15=df15,
        map4=map4,
        map1=map1,
        map15=map15,
        sess_ctx=sess_ctx,
        cvd_ctx=cvd_ctx,
        cfg=cfg,
        idm_level=idm_level,
        idm_bull_swept=idm_bull_swept,
        idm_bear_swept=idm_bear_swept,
        has_recent_choch=(has_bull_choch if best_d == 1 else has_bear_choch),
        has_recent_cisd=(has_bull_cisd if best_d == 1 else has_bear_cisd),
    )
    populate_forward_playbook(best_setup)
    if isinstance(best_setup.raw_market_data, dict):
        best_setup.raw_market_data["forward_playbook"] = best_setup.forward_playbook
        best_setup.raw_market_data["flip_scenario"] = best_setup.flip_scenario
        best_setup.raw_market_data["evidence"] = _safe_evidence(best_setup, df4, df1, df15, btc)
    return best_setup
