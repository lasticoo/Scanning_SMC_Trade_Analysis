# -*- coding: utf-8 -*-
"""
Consolidation Breakout / Breakdown Engine
=========================================
Mendeteksi pola konsolidasi klasik (Triangle, Wedge, Flag) dengan penguatan
proporsionalitas Stop Loss berbasis ATR 1H (Zoom-Out Check) & Estimasi Durasi Posisi.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from .indicators import (
    atr_wilder,
    compute_cvd_metrics,
    ema,
    f_align,
    find_pivots,
    liquidity_score,
    rsi_wilder,
    session_liquidity_context,
    trend_short,
    trend_text,
)
from .models import MAJOR_SYMBOLS, NAN, Config, Pat, Setup, clamp, fp, ts_utc
from .mtf_smc import extract_tf_structure, populate_forward_playbook
from .zone_radar import enrich_setup_with_zone_radar

SCORE_NAMES = [
    ("Struktur & tren", 25),
    ("Konsolidasi & trendline", 20),
    ("Konfirmasi breakout", 20),
    ("Volume & momentum", 15),
    ("RR & ruang target", 15),
    ("Likuiditas & eksekusi", 5),
]


def f_quality(t, uN, uS, lN, lS, n, a, h, l, c, tol):
    """Jumlah sentuhan trendline & seberapa bersih harga berada di dalam pola."""
    tU = tL = ins = cnt = 0
    prevU = prevL = False
    for k in range(1, n + 1):
        idx = t - k
        if idx < 0:
            break
        up = uN - uS * k
        lo = lN - lS * k
        hitU = abs(h[idx] - up) <= tol * a
        hitL = abs(l[idx] - lo) <= tol * a
        if hitU and not prevU:
            tU += 1
        if hitL and not prevL:
            tL += 1
        prevU, prevL = hitU, hitL
        if c[idx] <= up + tol * a and c[idx] >= lo - tol * a:
            ins += 1
        cnt += 1
    return tU, tL, ins, cnt


def f_nearest(d: int, entry: float, a: float, arr, x1: float, x2: float) -> Optional[float]:
    """Level lawan terdekat: resistance untuk long, support untuk short."""
    gap = 0.5 * a
    vals = []
    for v in list(arr) + [x1, x2]:
        if v is None or math.isnan(v):
            continue
        if d == 1 and v > entry + gap:
            vals.append(v)
        if d == -1 and v < entry - gap:
            vals.append(v)
    if not vals:
        return None
    return min(vals) if d == 1 else max(vals)


def f_score(d, al4, al1, q_pts, brk_pts, rv, rsi_v, above_e, rr, room_r, liq_pts, cfg: Config):
    s1 = 12.0 * al4 + 13.0 * al1
    s2 = q_pts
    s3 = brk_pts
    s_vol = 8.0 * clamp(rv / cfg.rel_vol_thr, 0.0, 1.0)
    if d == 1:
        s_rsi = 4.0 if 50 <= rsi_v <= 75 else (1.5 if rsi_v > 75 else (2.0 if rsi_v >= 40 else 0.0))
    else:
        s_rsi = 4.0 if 25 <= rsi_v <= 50 else (1.5 if rsi_v < 25 else (2.0 if rsi_v <= 60 else 0.0))
    s_mom = 3.0 if above_e else 0.0
    s4 = s_vol + s_rsi + s_mom
    s_rr = 8.0 if rr >= 2.5 else (6.5 if rr >= 2.0 else (3.0 if rr >= 1.5 else 0.0))
    s5 = s_rr + 7.0 * room_r
    s6 = liq_pts
    tot = s1 + s2 + s3 + s4 + s5 + s6
    return tot, [s1, s2, s3, s4, s5, s6]


def analyze_symbol(
    symbol: str,
    df15: pd.DataFrame,
    ctx4: dict,
    ctx1: dict,
    meta: dict,
    cfg: Config,
    df4: Optional[pd.DataFrame] = None,
    df1: Optional[pd.DataFrame] = None,
    deriv: Optional[dict] = None,
) -> Optional[Setup]:
    n = len(df15)
    if n < cfg.min_bars_15m:
        return None

    o = df15["open"].values.astype(float)
    h = df15["high"].values.astype(float)
    l = df15["low"].values.astype(float)
    c = df15["close"].values.astype(float)
    times = df15["open_time"].values.astype("int64")
    atr_s = atr_wilder(df15).values
    vol_sma = df15["volume"].rolling(cfg.rel_vol_len).mean().shift(1)
    rel = (df15["volume"] / vol_sma).values
    ema20 = ema(df15["close"], 20).values
    rsi = rsi_wilder(df15["close"]).values
    is_ph, is_pl = find_pivots(h, l, cfg.piv_len)

    atr_1h = float(ctx1.get("atr", NAN))
    if math.isnan(atr_1h) or atr_1h <= 0:
        atr_1h = float(atr_s[-1]) * 2.1 if not np.isnan(atr_s[-1]) else 0.0

    cvd_ctx = compute_cvd_metrics(df15)
    sess_ctx = session_liquidity_context(df15)
    deriv_ctx = deriv or {}

    al4 = {1: f_align(1, ctx4), -1: f_align(-1, ctx4)}
    al1 = {1: f_align(1, ctx1), -1: f_align(-1, ctx1)}

    S = dict(pH1P=None, pH1I=None, pH2P=None, pH2I=None, pL1P=None, pL1I=None, pL2P=None, pL2I=None)
    piv_hi: list = []
    piv_lo: list = []
    bk = dict(dir=0, bar=None, level=None, height=None, type="", q=0.0, body=0.0, rv=0.0, retested=False)

    def pattern_at(t: int) -> Optional[Pat]:
        if None in (S["pH1P"], S["pH2P"], S["pL1P"], S["pL2P"]):
            return None
        a = atr_s[t]
        if np.isnan(a):
            return None
        uS = (S["pH2P"] - S["pH1P"]) / (S["pH2I"] - S["pH1I"])
        lS = (S["pL2P"] - S["pL1P"]) / (S["pL2I"] - S["pL1I"])
        uNow = S["pH2P"] + uS * (t - S["pH2I"])
        lNow = S["pL2P"] + lS * (t - S["pL2I"])
        start = min(S["pH1I"], S["pL1I"])
        plen = t - start
        wNow = uNow - lNow
        wStart = (uNow - uS * plen) - (lNow - lS * plen)
        spanU, spanL = uS * plen, lS * plen
        len_ok = cfg.min_pat <= plen <= cfg.max_pat
        recent_ok = (t - max(S["pH2I"], S["pL2I"])) <= max(cfg.piv_len * 8, plen / 2)
        ordered = uNow > lNow and wStart > 0
        is_conv = ordered and wNow <= wStart * cfg.conv_ratio and wNow >= 0.5 * a
        is_par = ordered and 0.75 * wStart <= wNow <= 1.3 * wStart
        flat_u = abs(spanU) <= cfg.flat_tol * a
        flat_l = abs(spanL) <= cfg.flat_tol * a

        impulse = 0.0
        if len_ok:
            i0 = t - plen
            i1 = i0 - cfg.imp_look
            if i1 >= 0:
                impulse = c[i0] - c[i1]

        ptype, bias = "-", 0
        if len_ok and recent_ok and ordered:
            if is_conv:
                if flat_u and spanL > cfg.flat_tol * a:
                    ptype, bias = "Ascending Triangle", 1
                elif flat_l and spanU < -cfg.flat_tol * a:
                    ptype, bias = "Descending Triangle", -1
                elif spanU < 0 and spanL > 0:
                    ptype, bias = "Symmetrical Triangle", 0
                elif spanU < 0 and spanL < 0:
                    ptype, bias = "Falling Wedge", 1
                elif spanU > 0 and spanL > 0:
                    ptype, bias = "Rising Wedge", -1
            elif is_par:
                if spanU < -0.2 * a and spanL < -0.2 * a and impulse >= cfg.flag_impulse * a:
                    ptype, bias = "Bullish Flag", 1
                elif spanU > 0.2 * a and spanL > 0.2 * a and impulse <= -cfg.flag_impulse * a:
                    ptype, bias = "Bearish Flag", -1
        return Pat(ptype != "-", ptype, bias, uNow, lNow, uS, lS, plen, wNow, wStart, impulse)

    def evaluate(d: int, kind: str, t: int, pat: Optional[Pat]) -> Setup:
        a = float(atr_s[t])
        a1 = atr_1h if atr_1h > 0 else a * 2.1
        use_bk = kind in ("RETEST", "AWAIT_RETEST")
        is_dev = kind == "DEVELOPING"
        utype = bk["type"] if use_bk else pat.ptype
        is_flag = utype in ("Bullish Flag", "Bearish Flag")
        lvl = bk["level"] if use_bk else (pat.uNow if d == 1 else pat.lNow)

        if kind == "AWAIT_RETEST":
            entry = lvl + d * 0.5 * cfg.retest_tol * a
        elif is_dev:
            entry = lvl + d * cfg.bk_buffer * a
        else:
            entry = float(c[t])

        # Stop Loss struktural dengan pengaman Zoom-Out Check proporsional 1H
        if kind == "RETEST":
            sl_raw = (min(l[t], lvl) - cfg.sl_buffer * a1) if d == 1 else (max(h[t], lvl) + cfg.sl_buffer * a1)
        elif kind == "AWAIT_RETEST":
            sl_raw = lvl - d * 0.8 * a1
        else:
            sl_raw = (min(S["pL2P"], pat.lNow) - cfg.sl_buffer * a1) if d == 1 else (max(S["pH2P"], pat.uNow) + cfg.sl_buffer * a1)

        min_pct = cfg.min_sl_pct_major if symbol in MAJOR_SYMBOLS else cfg.min_sl_pct_alt
        min_sl_d = max(cfg.min_sl_atr_1h * a1, min_pct * entry)
        max_sl_d = max(min_sl_d * 1.35, min(cfg.max_sl_atr_1h * a1, 0.065 * entry))
        raw_d = abs(entry - sl_raw)
        clamped_d = clamp(raw_d, min_sl_d, max_sl_d)
        sl = (entry - clamped_d) if d == 1 else (entry + clamped_d)
        risk = abs(entry - sl)

        # Target: measured move, dibatasi level lawan terdekat & durasi wajar
        hgt = bk["height"] if use_bk else (abs(pat.impulse) if is_flag else pat.wStart)
        tp_raw = entry + d * max(hgt * cfg.mm_factor, 2.0 * risk)
        opp = f_nearest(
            d, entry, a1, piv_hi if d == 1 else piv_lo,
            ctx4["hi"] if d == 1 else ctx4["lo"], ctx1["hi"] if d == 1 else ctx1["lo"],
        )
        tp2 = tp_raw
        if opp is not None:
            tp2 = min(tp_raw, opp - 0.1 * a1) if d == 1 else max(tp_raw, opp + 0.1 * a1)
        tp2 = max(tp2, entry + 0.2 * a1) if d == 1 else min(tp2, entry - 0.2 * a1)
        tp1 = entry + 0.5 * (tp2 - entry)
        rr = abs(tp2 - entry) / max(risk, 1e-12)
        rr_tp1 = abs(tp1 - entry) / max(risk, 1e-12)
        room_r = 1.0 if opp is None else clamp(abs(opp - entry) / max(abs(tp_raw - entry), 1e-12), 0.0, 1.0)

        hourly_vel = max(0.45 * a1, 1e-12)
        est_h1 = max(0.5, abs(tp1 - entry) / hourly_vel)
        est_h2 = max(1.0, abs(tp2 - entry) / hourly_vel)
        hold_lbl = f"Intraday (~{est_h1:.0f}-{est_h2:.0f} jam)" if est_h2 <= 18 else f"Short-Swing (~{est_h1:.0f}-{est_h2:.0f} jam)"

        ext_flag = False
        if use_bk:
            q_pts, body_pts = bk["q"], bk["body"]
            rv_use = bk["rv"]
            brk_pts = 6.0 + body_pts + 4.0 + (5.0 if kind == "RETEST" else 0.0)
        else:
            rv_use = 0.0 if np.isnan(rel[t]) else float(rel[t])
            tU, tL, insC, cnt = f_quality(t, pat.uNow, pat.uSlope, pat.lNow, pat.lSlope, pat.patLen, a, h, l, c, cfg.touch_tol)
            touch_pts = min(tU, 4) + min(tL, 4)
            ins_ratio = insC / cnt if cnt > 0 else 0.0
            ins_pts = 6.0 * clamp((ins_ratio - 0.7) / 0.3, 0.0, 1.0)
            if is_flag:
                conv_pts = 6.0 * clamp(1.0 - abs(pat.wNow / pat.wStart - 1.0) / 0.3, 0.0, 1.0)
            else:
                conv_pts = 6.0 * clamp((1.0 - pat.wNow / pat.wStart) / 0.5, 0.0, 1.0)
            q_pts = touch_pts + ins_pts + conv_pts
            body_pts, brk_pts = 0.0, 0.0
            if not is_dev:
                rng = h[t] - l[t]
                body = abs(c[t] - o[t])
                close_pos = ((c[t] - l[t]) / rng if d == 1 else (h[t] - c[t]) / rng) if rng > 0 else 0.5
                body_pts = (3.0 if body >= cfg.body_min * a else 3.0 * body / (cfg.body_min * a)) + (2.0 if close_pos >= 0.6 else 0.0)
                ext_flag = (c[t] - lvl > cfg.ext_atr * a) if d == 1 else (lvl - c[t] > cfg.ext_atr * a)
                brk_pts = 6.0 + body_pts + (0.0 if ext_flag else 4.0)

        rsi_v = 50.0 if np.isnan(rsi[t]) else float(rsi[t])
        above_e = (c[t] > ema20[t]) if d == 1 else (c[t] < ema20[t])
        atr_pct = a / float(c[t])
        liq_pts = liquidity_score(meta, d, atr_pct)
        tot, parts = f_score(d, al4[d], al1[d], q_pts, brk_pts, rv_use, rsi_v, above_e, rr, room_r, liq_pts, cfg)

        if is_dev:
            status = "WATCHLIST (belum break)" if tot >= cfg.watch_score else "NO SETUP"
        elif kind == "AWAIT_RETEST":
            status = "WATCHLIST (tunggu retest)" if tot >= cfg.watch_score else "NO SETUP"
        elif ext_flag:
            status = "EXTENDED"
        elif tot >= cfg.entry_score and rr >= cfg.min_rr:
            status = "ENTRY READY"
        elif tot >= cfg.watch_score:
            status = "WATCHLIST"
        else:
            status = "REJECT"

        side = "atas" if d == 1 else "bawah"
        z_lo = min(lvl, entry)
        z_hi = max(lvl, entry)
        if is_dev:
            confirm = f"Belum break. Garis {side} di {fp(lvl)} | RelVol {rv_use:.1f}x | Est. {hold_lbl}"
            conf_short = f"belum break (garis {fp(lvl)})"
            zone = f"> {fp(lvl)}" if d == 1 else f"< {fp(lvl)}"
        elif kind == "AWAIT_RETEST":
            confirm = f"Breakout di {fp(lvl)}, tunggu retest | RelVol {rv_use:.1f}x | Est. {hold_lbl}"
            conf_short = f"tunggu retest {fp(lvl)}"
            z_lo = min(lvl, lvl + d * cfg.retest_tol * a)
            z_hi = max(lvl, lvl + d * cfg.retest_tol * a)
            zone = f"{fp(z_lo)} - {fp(z_hi)}"
        elif kind == "RETEST":
            confirm = f"Retest ditahan di {fp(lvl)} | RelVol {rv_use:.1f}x | Est. {hold_lbl}"
            conf_short = f"retest ditahan {fp(lvl)}"
            zone = fp(entry)
        else:
            body_atr = abs(c[t] - o[t]) / a
            confirm = (
                f"Close {'di atas' if d == 1 else 'di bawah'} garis {fp(lvl)} | RelVol {rv_use:.1f}x | "
                f"Body {body_atr:.1f} ATR | RSI {rsi_v:.0f} | Est. {hold_lbl}"
            )
            conf_short = f"close {'>' if d == 1 else '<'} garis, RV {rv_use:.1f}x"
            zone = fp(entry)

        if kind in ("BREAKOUT", "BREAKDOWN"):
            inval = f"Close 15M kembali {'di bawah' if d == 1 else 'di atas'} {fp(lvl)} atau harga menyentuh SL {fp(sl)}"
        elif use_bk:
            inval = f"Close 15M {'di bawah' if d == 1 else 'di atas'} {fp(lvl)} (retest gagal) atau SL {fp(sl)} tersentuh"
        else:
            inval = f"Close 15M {'di bawah garis bawah' if d == 1 else 'di atas garis atas'} pola atau SL {fp(sl)} tersentuh"

        st_ = Setup(
            symbol=symbol,
            direction="LONG" if d == 1 else "SHORT",
            kind=kind,
            pattern=utype,
            bar_index=t,
            candle_time=ts_utc(int(times[t])),
            level=float(lvl),
            entry=float(entry),
            market_price=float(c[-1]),
            limit_entry=float(lvl),
            entry_zone=zone,
            poi_low=float(z_lo),
            poi_high=float(z_hi),
            sl=float(sl),
            tp1=float(tp1),
            tp2=float(tp2),
            rr=float(rr),
            rr_tp1=float(rr_tp1),
            risk_pct=risk / entry * 100,
            tp1_pct=abs(tp1 - entry) / entry * 100,
            tp2_pct=abs(tp2 - entry) / entry * 100,
            score=float(tot),
            python_score=float(tot),
            parts=[(SCORE_NAMES[i][0], float(parts[i]), SCORE_NAMES[i][1]) for i in range(6)],
            status=status,
            python_status=status,
            confirmed=kind in ("BREAKOUT", "BREAKDOWN", "RETEST"),
            extended=ext_flag,
            rel_vol=rv_use,
            rsi=rsi_v,
            atr=a,
            atr_pct=atr_pct,
            atr_1h=a1,
            atr_1h_pct=a1 / float(c[-1]),
            sl_atr_1h_ratio=risk / max(a1, 1e-12),
            upper_line=float(pat.uNow) if pat else NAN,
            lower_line=float(pat.lNow) if pat else NAN,
            opp_level=opp,
            trend_4h=trend_text(ctx4),
            struct_1h=trend_text(ctx1),
            trend_4h_short=trend_short(ctx4),
            struct_1h_short=trend_short(ctx1),
            confirm_txt=confirm,
            confirm_short=conf_short,
            invalidation=inval,
            range_1h_low=float(ctx1["lo"]),
            range_1h_high=float(ctx1["hi"]),
            est_hold_hours_tp1=est_h1,
            est_hold_hours_tp2=est_h2,
            est_hold_label=hold_lbl,
            session_name=sess_ctx["session"],
            cvd_state=cvd_ctx["cvd_state"],
            taker_buy_ratio=float(cvd_ctx["taker_buy_ratio"]),
            oi_change_1h_pct=float(deriv_ctx.get("oi_change_1h_pct", NAN)),
            oi_change_4h_pct=float(deriv_ctx.get("oi_change_4h_pct", NAN)),
            top_ls_ratio=float(deriv_ctx.get("top_ls_ratio", NAN)),
            retail_ls_ratio=float(deriv_ctx.get("retail_ls_ratio", NAN)),
            orderflow_summary=cvd_ctx["cvd_state"],
            q_pts=float(q_pts),
            body_pts=float(body_pts),
            height=float(hgt),
            meta=dict(meta),
            last_candle=ts_utc(int(times[-1])),
        )
        if np.isnan(ctx4["e200"]):
            st_.notes.append("EMA200 4H tidak tersedia (histori kurang dari 200 candle).")
        return st_

    events: list = []
    last_evt: Optional[int] = None
    last_pat: Optional[Pat] = None
    L = cfg.piv_len

    for t in range(n):
        i = t - L
        if i >= L:
            if is_ph[i]:
                S["pH1P"], S["pH1I"], S["pH2P"], S["pH2I"] = S["pH2P"], S["pH2I"], float(h[i]), i
                piv_hi.append(float(h[i]))
                if len(piv_hi) > 25:
                    piv_hi.pop(0)
            if is_pl[i]:
                S["pL1P"], S["pL1I"], S["pL2P"], S["pL2I"] = S["pL2P"], S["pL2I"], float(l[i]), i
                piv_lo.append(float(l[i]))
                if len(piv_lo) > 25:
                    piv_lo.pop(0)

        a = atr_s[t]
        if t < 1 or np.isnan(a):
            continue
        pat = pattern_at(t)
        last_pat = pat

        if bk["dir"] != 0:
            if t - bk["bar"] > cfg.retest_bars:
                bk["dir"] = 0
            elif t > bk["bar"] and bk["dir"] == 1 and c[t] < bk["level"] - cfg.bk_buffer * a:
                bk["dir"] = 0
            elif t > bk["bar"] and bk["dir"] == -1 and c[t] > bk["level"] + cfg.bk_buffer * a:
                bk["dir"] = 0

        kind, d = None, 0
        if pat is not None and pat.ok and bk["dir"] == 0:
            u_prev = pat.uNow - pat.uSlope
            l_prev = pat.lNow - pat.lSlope
            if pat.bias >= 0 and c[t] > pat.uNow + cfg.bk_buffer * a and c[t - 1] <= u_prev + cfg.bk_buffer * a:
                kind, d = "BREAKOUT", 1
            elif pat.bias <= 0 and c[t] < pat.lNow - cfg.bk_buffer * a and c[t - 1] >= l_prev - cfg.bk_buffer * a:
                kind, d = "BREAKDOWN", -1
        if kind is None and bk["dir"] != 0 and not bk["retested"] and t - bk["bar"] >= 1:
            if bk["dir"] == 1 and l[t] <= bk["level"] + cfg.retest_tol * a and c[t] > bk["level"] and c[t] > o[t]:
                kind, d = "RETEST", 1
            elif bk["dir"] == -1 and h[t] >= bk["level"] - cfg.retest_tol * a and c[t] < bk["level"] and c[t] < o[t]:
                kind, d = "RETEST", -1

        if kind is not None:
            s = evaluate(d, kind, t, pat)
            if kind in ("BREAKOUT", "BREAKDOWN"):
                bk.update(dir=d, bar=t, level=s.level, height=s.height, type=s.pattern, q=s.q_pts, body=s.body_pts, rv=s.rel_vol, retested=False)
            else:
                bk["retested"] = True
            last_evt = t
            events.append(s)

    final_setup: Optional[Setup] = None
    recent = [e for e in events if e.bar_index >= n - cfg.event_recent_bars]
    if recent:
        final_setup = recent[-1]
    else:
        t = n - 1
        if np.isnan(atr_s[t]):
            return None
        if bk["dir"] != 0 and not bk["retested"]:
            final_setup = evaluate(bk["dir"], "AWAIT_RETEST", t, None)
        elif last_pat is not None and last_pat.ok and (last_evt is None or t - last_evt > 30):
            if last_pat.bias == 0:
                d = 1 if (al4[1] + al1[1]) >= (al4[-1] + al1[-1]) else -1
            else:
                d = last_pat.bias
            final_setup = evaluate(d, "DEVELOPING", t, last_pat)

    if final_setup is not None:
        final_setup.archetype = "BREAKOUT"
        final_setup.archetype_label = "Breakout Konsolidasi"
        final_setup.exec_style = "ENTRY_NOW" if final_setup.status == "ENTRY READY" else "WAIT_SWEEP_TRIGGER"
        map4 = extract_tf_structure(df4, cfg.htf_piv, "4H", lookback_bars=80) if df4 is not None else None
        map1 = extract_tf_structure(df1, cfg.htf_piv, "1H", lookback_bars=96) if df1 is not None else None
        map15 = extract_tf_structure(df15, cfg.piv_len, "15M", lookback_bars=95)
        enrich_setup_with_zone_radar(final_setup, df15, map4, map1, map15, sess_ctx, cvd_ctx, cfg)
        populate_forward_playbook(final_setup)
    return final_setup

