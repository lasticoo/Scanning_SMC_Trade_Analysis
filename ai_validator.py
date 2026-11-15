# -*- coding: utf-8 -*-
"""
AI Modern Crypto SMC/ICT Curator & Validator Module for Binance Futures Scanner
===============================================================================
Memanfaatkan LLM (Groq Cloud API) sebagai Kurator Utama Top-Down MTF SMC/ICT (4H -> 1H -> 15M)
yang disesuaikan khusus dengan karakter liar pasar Crypto Futures:
1. Menilai lintas 3 Variasi Setup Dinamis:
   - TREND_CONT   : Follow-trend setelah konfirmasi reaksi di POI.
   - PRE_REACTION : Mendekati Supply/Demand/OB SEGAR sebelum bereaksi (kurasi Limit Pre-Tap vs Tunggu Sweep).
   - HTF_REVERSAL : Potensi keterbalikan tren saat harga mendekati/menguji Tembok Mayor 4H/1H.
2. Menyesuaikan taktik terhadap Stop-Hunt Wick / Judas Swing kripto, kolam EQH/EQL/IDM di depan POI,
   gaya kedatangan harga (Kompresi vs Impulsif Liar), serta menjaga jarak dekat & durasi posisi singkat.
3. Menghasilkan Forward Playbook (Skenario Utama If-Then & Skenario Antisipasi Breaker Flip).
4. Dilengkapi parser JSON tahan tag <think>, koneksi TLS persisten cepat, PROTEKSI SKOR 0,
   serta pagar pengaman "Zoom-Out Check" (ATR 1H) pasca-AI.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import requests

from scanner_core.models import clamp, fp, size_position

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
_RATE_LOCK = threading.Lock()
_LAST_CALL_TS = 0.0
_MIN_CALL_INTERVAL = 0.30  # pacing halus lintas worker agar tidak memicu burst 429
_UNAVAILABLE_MODELS: set = set()
_MODEL_COOLDOWN_UNTIL: Dict[str, float] = {}
_MODEL_REMAINING_TPM: Dict[str, int] = {}
_MODEL_INFLIGHT: Dict[str, int] = {}
_TLS_LOCAL = threading.local()

VALID_STATUSES = ("ENTRY READY", "PRE-TAP LIMIT", "REVERSAL WATCH", "WATCHLIST", "REJECT")
VALID_EXEC_STYLES = ("ENTRY_NOW", "LIMIT_PRE_TAP", "WAIT_SWEEP_TRIGGER", "LIMIT_RETEST", "SKIP")


def _get_http_session() -> requests.Session:
    sess = getattr(_TLS_LOCAL, "session", None)
    if sess is None:
        sess = requests.Session()
        _TLS_LOCAL.session = sess
    return sess


def load_env(env_path: Optional[str] = None) -> None:
    """Membaca file .env sederhana ke os.environ tanpa ketergantungan library luar."""
    if env_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        env_path = os.path.join(base_dir, ".env")
        if not os.path.exists(env_path):
            teis_env = os.path.expanduser("~/.gemini/antigravity-ide/scratch/teis/.env")
            if os.path.exists(teis_env):
                env_path = teis_env

    if not os.path.exists(env_path):
        return

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception as e:
        print(f"[AI Validator] Peringatan: gagal membaca .env: {e}", file=sys.stderr)


load_env()


def get_groq_key() -> str:
    return os.environ.get("GROQ_API_KEY", "").strip()


def get_default_model() -> str:
    return os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b").strip()


def get_fallback_model() -> str:
    return os.environ.get("GROQ_FALLBACK_MODEL", "openai/gpt-oss-120b").strip()


def get_safety_model() -> str:
    return os.environ.get("GROQ_SAFETY_MODEL", "openai/gpt-oss-20b").strip()


def _fmt_zones(zones: list, limit: int = 3) -> str:
    if not zones:
        return "Tidak ada zona aktif terdekat"
    items = []
    for z in zones[-limit:]:
        d_lbl = "Bull" if z.get("dir", 1) == 1 else "Bear"
        kind = z.get("kind", f"{d_lbl} Zone")
        extra = []
        if z.get("swept"):
            extra.append("SweepPOI")
        if z.get("fresh") is True:
            extra.append("Unmitigated")
        tag = f" ({', '.join(extra)})" if extra else ""
        items.append(f"{kind} [{fp(z.get('bot'))} - {fp(z.get('top'))}]{tag}")
    return "; ".join(items)


def _fmt_fvgs(fvgs: list, limit: int = 3) -> str:
    if not fvgs:
        return "Tidak ada FVG unmitigated"
    items = []
    for f in fvgs[-limit:]:
        d_lbl = "Bull FVG" if f.get("dir", 1) == 1 else "Bear FVG"
        items.append(f"{d_lbl} [{fp(f.get('bot'))} - {fp(f.get('top'))}]")
    return "; ".join(items)


def build_full_llm_smc_prompt(setup: Any, btc_ctx: Optional[dict] = None) -> str:
    """
    Menyusun prompt Kurator Modern SMC/ICT Crypto Futures (4H -> 1H -> 15M) + Orderflow +
    Konteks Variasi Setup (TREND_CONT / PRE_REACTION / HTF_REVERSAL) secara presisi.
    """
    direction = getattr(setup, "direction", "UNKNOWN")
    symbol = getattr(setup, "symbol", "")
    pattern = getattr(setup, "pattern", "")
    archetype = getattr(setup, "archetype", "TREND_CONT")
    arch_lbl = getattr(setup, "archetype_label", archetype)
    exec_style = getattr(setup, "exec_style", "WAIT_SWEEP_TRIGGER")
    app_style = getattr(setup, "approach_style", "-")
    tap_h = getattr(setup, "est_tap_hours", 0.0)
    tap_txt = f"~{tap_h:.1f} jam" if (isinstance(tap_h, (int, float)) and not math.isnan(tap_h) and tap_h > 0) else "Sudah di area / detik ini"

    entry_zone = getattr(setup, "entry_zone", "")
    plan_entry = getattr(setup, "entry", 0.0)
    sl = getattr(setup, "sl", 0.0)
    tp1 = getattr(setup, "tp1", 0.0)
    tp2 = getattr(setup, "tp2", 0.0)
    tp3 = getattr(setup, "tp3", tp2)
    rr = getattr(setup, "rr", 0.0)
    rr3 = getattr(setup, "rr_tp3", rr)
    liq_guard = getattr(setup, "liq_guard_info", "")
    sw_entry = getattr(setup, "sweep_entry_lvl", float("nan"))
    t4 = getattr(setup, "trend_4h", "Netral")
    s1 = getattr(setup, "struct_1h", "Netral")
    confirm = getattr(setup, "confirm_txt", "")
    invalidation = getattr(setup, "invalidation", "")
    opp_lvl = getattr(setup, "opp_level", None)
    raw = getattr(setup, "raw_market_data", {}) or {}

    cur_p = raw.get("current_price", getattr(setup, "market_price", plan_entry))
    at_poi = raw.get("at_poi", True)
    atr_4h = raw.get("atr_4h", 0.0)
    atr_1h = raw.get("atr_1h", getattr(setup, "atr_1h", 0.0))
    atr_15m = raw.get("atr", getattr(setup, "atr", 0.0))
    min_sl_d = raw.get("min_sl_dist", abs(plan_entry - sl) * 0.8)
    max_sl_d = raw.get("max_sl_dist", abs(plan_entry - sl) * 1.6)

    # Data Lapis 4H
    r4_str = raw.get("range_4h_str", getattr(setup, "range_4h_str", "-"))
    w4_str = raw.get("walls_4h_str", getattr(setup, "walls_4h_str", "-"))
    obs_4h_txt = _fmt_zones(raw.get("obs_4h", []))

    # Data Lapis 1H
    r1_lo = raw.get("range_1h_low", getattr(setup, "range_1h_low", 0.0))
    r1_hi = raw.get("range_1h_high", getattr(setup, "range_1h_high", 0.0))
    r1_eq = raw.get("range_1h_eq", getattr(setup, "range_1h_eq", 0.0))
    z_pos_lbl = raw.get("zone_pos_label", f"{raw.get('range_1h_pos_pct', 50.0):.1f}%")
    hold_lbl = raw.get("est_hold_label", getattr(setup, "est_hold_label", "-"))
    obs_1h_txt = _fmt_zones(raw.get("obs_1h", []))
    fvgs_1h_txt = _fmt_fvgs(raw.get("fvgs_1h", []))
    eqh_1h = ", ".join(fp(x) for x in raw.get("eqh_1h", [])) or "-"
    eql_1h = ", ".join(fp(x) for x in raw.get("eql_1h", [])) or "-"

    # Data Lapis 15M & Orderflow
    bull_obs_15 = [{"dir": 1, "kind": "15M Bull OB", **z} for z in raw.get("bull_obs", [])]
    bear_obs_15 = [{"dir": -1, "kind": "15M Bear OB", **z} for z in raw.get("bear_obs", [])]
    brk_15 = raw.get("breakers_15m", [])
    obs_15_txt = _fmt_zones(bull_obs_15 + bear_obs_15 + brk_15, limit=5)
    fvgs_15_txt = _fmt_fvgs(raw.get("fvgs", []), limit=4)
    eqh_15 = ", ".join(fp(x) for x in raw.get("eqh_15m", [])) or "-"
    eql_15 = ", ".join(fp(x) for x in raw.get("eql_15m", [])) or "-"
    recent_events = ", ".join(raw.get("struct_events", [])) or confirm

    idm_lvl = raw.get("idm_level")
    idm_taken = raw.get("idm_bull_swept") if direction == "LONG" else raw.get("idm_bear_swept")
    idm_txt = f"Level {fp(idm_lvl)} ({'SUDAH DISAPU / IDM Taken' if idm_taken else 'Belum disapu — umpan likuiditas'})" if idm_lvl else "Tidak ada IDM spesifik"

    pins = raw.get("pin_bars", [])
    pin_txt = ", ".join(f"{'Bull' if p['dir']==1 else 'Bear'} PinBar (wick {p['wick_pct']*100:.0f}%)" for p in pins) or "Belum ada pin bar baru"

    sess_txt = raw.get("session", getattr(setup, "session_name", "-"))
    of_txt = raw.get("orderflow_summary", getattr(setup, "orderflow_summary", "-"))
    fund_pct = float(raw.get("funding_rate", 0.0)) * 100.0
    opp_txt = fp(opp_lvl) if opp_lvl is not None else "Belum teridentifikasi"

    # Data Detektor Lokasi Harga Live
    z_state = getattr(setup, "zone_state", raw.get("zone_state", "-"))
    z_badge = getattr(setup, "zone_badge", raw.get("zone_badge", "-"))
    dist_poi = getattr(setup, "dist_to_poi_pct", raw.get("dist_to_poi_pct", 0.0))
    depth_poi = getattr(setup, "zone_depth_pct", float("nan"))
    depth_txt = f"{depth_poi:.0f}%" if (isinstance(depth_poi, (int, float)) and not math.isnan(depth_poi)) else "Belum masuk zona"
    active_zones_txt = getattr(setup, "active_zones_summary", raw.get("active_zones_summary", "-")) or "-"
    sweep_live_txt = getattr(setup, "sweep_status", raw.get("sweep_status", "-")) or "-"
    reac_live_txt = getattr(setup, "zone_reaction", raw.get("zone_reaction", "-")) or "-"
    magnet_txt = getattr(setup, "eq_magnet_info", raw.get("eq_magnet_info", "-")) or "-"

    btc_note = "Konteks BTC Netral"
    if btc_ctx and "4h" in btc_ctx:
        b4 = btc_ctx["4h"].get("st", 0)
        b4_str = "Bullish" if b4 == 1 else ("Bearish" if b4 == -1 else "Netral")
        btc_note = f"BTC 4H: {b4_str} (Referensi makro saja; altcoin dinilai 100% independen dari struktur koin sendiri)"

    tp3_str = f" | TP3 Runner: {fp(tp3)} (RR 1:{rr3:.1f})" if (isinstance(tp3, (int, float)) and not math.isnan(tp3) and tp3 != tp2) else ""
    guard_str = f" | Proteksi SL di balik: {liq_guard}" if liq_guard else ""
    sw_entry_str = f" (Batas Sweep-Entry: {fp(sw_entry)})" if (isinstance(sw_entry, (int, float)) and not math.isnan(sw_entry)) else ""

    py_score = float(getattr(setup, "python_score", getattr(setup, "score", 0.0)))
    parts_list = getattr(setup, "parts", []) or []
    parts_txt = " | ".join(f"{nm}: {pt:.1f}/{mx}" for nm, pt, mx in parts_list) if parts_list else "-"

    rev_stage_num = int(raw.get("rev_stage_num", 0))
    rev_stage_lbl = str(raw.get("rev_stage_label", "-"))
    rev_next_step = str(raw.get("rev_next_step", "-"))
    rev_score_10 = float(raw.get("rev_score_10", round(py_score / 10.0, 2)))
    if archetype == "HTF_REVERSAL":
        stage_info_line = (
            f"- [PROGRES REVERSAL WATCH BERTAHAP PYTHON] Status Tahap: {rev_stage_lbl} (Tahap {rev_stage_num}/5) | "
            f"Skor Objektif Python: {rev_score_10:.1f}/10 ({py_score:.0f}/100) | Langkah Berikutnya: {rev_next_step}\n"
            f"- [RINCIAN 10 POIN SKOR OBJEKTIF PYTHON] {parts_txt}\n"
        )
    else:
        stage_info_line = (
            f"- [TEMUAN RULES & SKOR OBJEKTIF PYTHON] Skor Awal Python: {py_score:.0f}/100 | Rincian Komponen: {parts_txt}\n"
        )

    prompt = f"""Anda adalah HAKIM & KURATOR PENILAIAN AKHIR 3 Pilar Paten SMC/ICT Modern Crypto Futures (4H->1H->15M & Orderflow).
Python bertugas sebagai tracker, penemu struktur objektif, dan pengukur geometri. Anda (API AI) yang memegang keputusan penilaian akhir (decision, status, smc_score, playbook).

[PRINSIP MUTLAK TRADER — WAJIB DIIKUTI]
1. TANPA PAKSAAN LIMIT (PREFER REJECT): Dilarang mengarang/menggeser limit entry jauh hanya demi memaksakan R:R! Jika secara trade range, penghalang Supply/Demand 4H lawan, atau R:R efektif ke TP2 tidak mencapai minimal 1:3.0, WAJIB putuskan "decision": "REJECTED" dan "status": "REJECT".
2. MINIMAL R:R 1:3.0 & PANTANG GESER SL KE BEP: Target Utama TP2 wajib memiliki R:R efektif >= 1:3.0. DILARANG KERAS menyarankan geser SL ke BEP / Break-Even dalam kondisi apa pun (Fixed SL Set-and-Forget di balik low/high sweep).

[SETUP {symbol}] Harga: {fp(cur_p)} | Sesi: {sess_txt} | {btc_note}
- Archetype: {archetype} ({arch_lbl}) | {direction} ({pattern}) | Taktik Awal: {exec_style} | Approach: {app_style} | Est. Sentuh: {tap_txt}
- Lokasi Live: {z_badge} ({z_state} | Jarak POI: {dist_poi:.2f}% | Depth: {depth_txt}) | Zona Aktif: {active_zones_txt}
- Sweep: {sweep_live_txt} | Magnet EQH/EQL/IDM: {magnet_txt} | Reaksi/CVD: {reac_live_txt}
{stage_info_line}[4H MAKRO] Tren: {t4} | ATR4H: {fp(atr_4h)} | Range4H: {r4_str} | Tembok4H: {w4_str} | OB4H: {obs_4h_txt}
[1H RANGE] Tren: {s1} | ATR1H: {fp(atr_1h)} | Range1H: {fp(r1_lo)}-{fp(r1_hi)} (EQ {fp(r1_eq)}, Posisi {z_pos_lbl}) | Hold: {hold_lbl}
- OB1H: {obs_1h_txt} | FVG1H: {fvgs_1h_txt} | EQH1H[{eqh_1h}] EQL1H[{eql_1h}]
[15M & ORDERFLOW] ATR15M: {fp(atr_15m)} | Struktur15M: {recent_events} | IDM: {idm_txt} | PinBar: {pin_txt}
- OB15M: {obs_15_txt} | FVG15M: {fvgs_15_txt} | EQH15M[{eqh_15}] EQL15M[{eql_15}]
- Orderflow: {of_txt} | Funding: {fund_pct:+.4f}%

[LEVEL & FIXED LIQUIDITY-PROTECTED SL (NO BEP)] Entry: {fp(plan_entry)}{sw_entry_str} (POI: {entry_zone} | {'Di POI' if at_poi else 'Menunggu Sentuhan'})
- SL Tetap: {fp(sl)}{guard_str} | Checkpoint TP1: {fp(tp1)} | Target Utama TP2: {fp(tp2)} (RR 1:{rr:.1f}, WAJIB >= 1:3.0){tp3_str} | Tembok Lawan: {opp_txt} | Invalidasi: {invalidation}

[ATURAN PENILAIAN AKHIR KURATOR AI — 3 PILAR PATEN & REVERSAL WATCH BERTAHAP (4H -> H1 -> M15)]
A. PILAR 1 — REVERSAL WATCH SMC BERTAHAP (archetype="HTF_REVERSAL", LONG & SHORT):
   - Urutan Tahap Objektif:
     * Tahap 0 (Konteks 4H): LONG wajib di Discount 4H (<50%) & ada Demand/OB 4H penyebab BOS + SSL; SHORT wajib di Premium 4H (>50%) & ada Supply/OB 4H penyebab BOS + BSL.
     * Tahap 1 (WATCH): Menunggu sweep SSL/BSL di dekat POI 4H (belum boleh entry).
     * Tahap 2 (Sweep Terkonfirmasi): Wick menembus swing low/high atau EQL/EQH >=0.1x ATR lalu close kembali (reclaim/reject) dalam <=3 candle H1. Jika close bertahan di luar >=3 candle H1 tanpa reclaim/reject, itu Breakdown/Breakout -> WAJIB "REJECT".
     * Tahap 3 (MSS di H1): Close CHoCH H1 dengan displacement (body >=1.5x rata-rata 20 candle) + FVG H1 dalam <=20 candle H1 setelah sweep.
     * Tahap 4 (Retrace ke POI H1 / OTE 62-79%): Harga kembali ke konfluensi OB H1 + FVG H1 + OTE 62-79%. Jika lari >50% impuls tanpa retrace -> WAJIB "REJECT".
     * Tahap 5 (Trigger di M15): CHoCH M15 ber-displacement di dalam POI H1/4H.
     * Tahap 6 (SL & Target): SL tetap di bawah low_sweep (LONG) atau di atas high_sweep + ekstra buffer squeeze (SHORT); TP1 = BSL/SSL H1; TP2 = Equilibrium 4H dengan R:R efektif >= 1:3.0 setelah memperhitungkan Supply/Demand 4H penghalang; TP3 = Supply/Demand 4H / PDH-PDL.
   - Keputusan Status & Skoring 10 Poin (0-100):
     * Hanya beri "ENTRY READY" ("ENTRY_NOW") jika sudah mencapai Tahap 5 (CHoCH M15 di POI), berada di POI, dan smc_score >= 80 (8.0/10).
     * Jika berada di Tahap 1 s/d 4 (atau Tahap 5 menunggu konfirmasi close), beri "REVERSAL WATCH" ("WAIT_SWEEP_TRIGGER") bila LONG smc_score >= 60 (prioritas >= 80) atau SHORT smc_score >= 70 (SHORT wajib lebih ketat karena risiko short squeeze kripto). Di bawah itu -> "REJECT".
B. PILAR 2 — PRE-TAP LIMIT ("PRE-TAP LIMIT" / "LIMIT_PRE_TAP"):
   - Khusus archetype="PRE_REACTION" searah tren HTF, mendekati OB segar (0 touch, jarak <=1.35%) dengan approach KOMPRESI/WAJAR & R:R efektif >= 1:3.0 -> smc_score >= 75.
C. PILAR 3 — SMC SNIPER FOLLOW-TREND ("ENTRY READY" atau "WATCHLIST"):
   - Khusus archetype="TREND_CONT" yang 100% searah tren HTF (Counter-Trend dilarang di Sniper!), dekat POI, R:R efektif >= 1:3.0, smc_score >= 75.

Balas HANYA JSON valid murni dlm Bahasa Indonesia lugas (tiap sub-elemen smc_elements cukup 1 kalimat padat <=16 kata; fokus tajam pada detailed_analysis, forward_playbook, flip_scenario TANPA kata BEP/Break-Even):
{{
  "smc_elements": {{
    "order_blocks": "status OB/Breaker 4H-1H-15M & kesegarannya",
    "liquidity_sweeps_runs": "status BSL/SSL sweep & reclaim <=3 candle H1 atau kolam EQH/EQL",
    "liquidity_range": "posisi harga dlm Range 4H & 1H (Discount/Premium & OTE 62-79%)",
    "choch": "status MSS H1 & Trigger CHoCH M15 di dalam POI",
    "bos": "status BOS 4H/1H & displacement >=1.5x body",
    "inducement": "status IDM di depan POI",
    "supply_demand": "zona Supply/Demand & ruang bersih ke tembok 4H lawan >=1:3",
    "pin_bar": "reaksi wick/approach & absorpsi CVD",
    "cisd": "status CISD, OI, Funding, & Relative Strength vs BTC",
    "poi": "tahap Reversal Watch (1-5) / zona POI, taktik entry, & durasi hold"
  }},
  "decision": "APPROVED" | "CONDITIONAL" | "REJECTED",
  "status": "ENTRY READY" | "PRE-TAP LIMIT" | "REVERSAL WATCH" | "WATCHLIST" | "REJECT",
  "execution_style": "ENTRY_NOW" | "LIMIT_PRE_TAP" | "WAIT_SWEEP_TRIGGER" | "LIMIT_RETEST" | "SKIP",
  "smc_score": 80,
  "direction": "{direction}",
  "entry_zone": "{entry_zone}",
  "sl": {sl},
  "tp1": {tp1},
  "tp2": {tp2},
  "rr": {rr:.2f},
  "detailed_analysis": "ulasan padat penilaian akhir AI atas tahapan setup, struktur, & kelayakan R:R >=1:3 tanpa limit paksaan",
  "forward_playbook": "Langkah Utama (If-Then): kondisi tahap sekarang, pemicu spesifik menuju tahap berikutnya/entry, tahan SL tetap di balik sweep (tanpa geser BEP), & kawal ke TP2/TP3 >=1:3",
  "flip_scenario": "Syarat Pembatalan (Cancel/Flip): level close H1/4H di luar sweep/POI, batas waktu 20 bar H1 MSS, atau skenario balik arah jika invalid"
}}"""
    return prompt


def parse_llm_json(text: str) -> dict:
    """
    Mengekstrak objek JSON secara tangguh dari respons LLM:
    - Membuang blok reasoning <think>...</think>
    - Membuang pembungkus markdown ```json ... ```
    - Mencari kurung kurawal seimbang terluar {...} & membersihkan trailing comma
    """
    if not text:
        raise ValueError("Respons LLM kosong")

    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if "<think>" in cleaned.lower():
        # Jika blok <think> terpotong sebelum </think>, buang seluruh teks sebelum kurung '{' JSON
        cleaned_after_think = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE).strip()
        if "{" in cleaned_after_think:
            cleaned = cleaned_after_think
        else:
            raise ValueError("Respons reasoning <think> terpotong sebelum objek JSON terbentuk")

    cleaned = cleaned.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    if start == -1:
        raise ValueError("Tidak ditemukan kurung '{' JSON pada respons LLM")

    depth = 0
    in_str = False
    esc = False
    end = -1
    for idx in range(start, len(cleaned)):
        ch = cleaned[idx]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = idx
                    break

    candidate = cleaned[start : end + 1] if end != -1 else cleaned[start:]
    candidate_fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
    obj = json.loads(candidate_fixed)
    if not isinstance(obj, dict):
        raise ValueError("JSON hasil parse bukan dict")
    return obj


def _parse_reset_seconds(val: str) -> float:
    """Mengurai header durasi Groq seperti '952ms', '5.4s', '1m12.3s', atau angka detik murni."""
    if not val:
        return 0.0
    s = val.strip().lower()
    if s.replace(".", "", 1).isdigit():
        return float(s)
    total = 0.0
    m_min = re.search(r"(\d+(?:\.\d+)?)m(?!s)", s)
    if m_min:
        total += float(m_min.group(1)) * 60.0
    m_ms = re.search(r"(\d+(?:\.\d+)?)ms", s)
    if m_ms:
        total += float(m_ms.group(1)) / 1000.0
    s_no_ms = re.sub(r"\d+(?:\.\d+)?ms", "", s)
    m_sec = re.search(r"(\d+(?:\.\d+)?)s", s_no_ms)
    if m_sec:
        total += float(m_sec.group(1))
    return total


def _record_ratelimit_headers(mdl: str, hdrs: Any, is_429: bool = False) -> float:
    """Mencatat sisa kuota TPM & waktu cooldown per model dari header respons Groq."""
    rem_str = hdrs.get("x-ratelimit-remaining-tokens", "")
    rst_tok = _parse_reset_seconds(hdrs.get("x-ratelimit-reset-tokens", ""))
    retry_s = _parse_reset_seconds(hdrs.get("Retry-After", ""))
    wait_s = retry_s or rst_tok
    with _RATE_LOCK:
        now = time.monotonic()
        if rem_str.isdigit():
            rem_tok = int(rem_str)
            _MODEL_REMAINING_TPM[mdl] = rem_tok
            # Jika sisa token tidak cukup untuk 1 audit penuh (~1.950 token), alihkan sementara ke model berikutnya
            if not is_429 and rem_tok < 1950:
                cd = min(max(wait_s, 3.0), 12.0)
                _MODEL_COOLDOWN_UNTIL[mdl] = max(_MODEL_COOLDOWN_UNTIL.get(mdl, 0.0), now + cd)
        if is_429:
            cd = min(max(wait_s if wait_s > 0 else 8.0, 3.0), 24.0)
            _MODEL_COOLDOWN_UNTIL[mdl] = max(_MODEL_COOLDOWN_UNTIL.get(mdl, 0.0), now + cd)
            _MODEL_REMAINING_TPM[mdl] = 0
            return cd
    return wait_s


def _pace_request() -> None:
    global _LAST_CALL_TS
    with _RATE_LOCK:
        now = time.monotonic()
        wait = _MIN_CALL_INTERVAL - (now - _LAST_CALL_TS)
        if wait > 0:
            time.sleep(wait)
        _LAST_CALL_TS = time.monotonic()


def _order_models_by_capacity(base_models: List[str]) -> List[str]:
    """Mengurutkan model berdasarkan kesiapan kuota TPM & jumlah request paralel aktif."""
    with _RATE_LOCK:
        now = time.monotonic()
        scored = []
        for idx, m in enumerate(base_models):
            if m in _UNAVAILABLE_MODELS:
                continue
            cd_left = max(0.0, _MODEL_COOLDOWN_UNTIL.get(m, 0.0) - now)
            inflight = _MODEL_INFLIGHT.get(m, 0)
            rem_tpm = _MODEL_REMAINING_TPM.get(m, 8000)
            est_free = rem_tpm - (inflight * 2000)
            # Prioritaskan model yang bebas cooldown & estimasi tokennya masih cukup untuk 1 audit penuh
            busy_rank = 0 if (cd_left <= 0.0 and est_free >= 1900) else (1 if cd_left <= 0.0 else 2)
            scored.append((busy_rank, round(cd_left, 1), inflight, idx, m))
        scored.sort()
        return [item[4] for item in scored]


def call_groq_api(prompt: str, model: Optional[str] = None, api_key: Optional[str] = None) -> Optional[dict]:
    """Mengirim permintaan ke Groq Cloud API dengan penyeimbang TPM multi-model, failover 429 instan, & proteksi JSON."""
    key = api_key or get_groq_key()
    if not key:
        return None

    primary_model = model or get_default_model()
    fallback_model = get_fallback_model()
    safety_model = get_safety_model()

    base_models: List[str] = []
    for m in (primary_model, fallback_model, safety_model):
        if m and m not in base_models and m not in _UNAVAILABLE_MODELS:
            base_models.append(m)
    if not base_models:
        base_models = [safety_model]

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    sess = _get_http_session()
    last_err: Optional[str] = None

    for pool_pass in range(3):
        ordered = _order_models_by_capacity(base_models)
        if not ordered:
            break

        # Jika seluruh model sedang cooldown TPM, tidur tepat sampai kuota terdekat pulih
        with _RATE_LOCK:
            now = time.monotonic()
            waits = [max(0.0, _MODEL_COOLDOWN_UNTIL.get(m, 0.0) - now) for m in ordered]
        min_cd = min(waits) if waits else 0.0
        if min_cd > 0.25:
            time.sleep(min(min_cd + 0.5, 24.5))

        for mdl in ordered:
            if mdl in _UNAVAILABLE_MODELS:
                continue

            use_low_reasoning = "gpt-oss" in mdl.lower()
            payload: Dict[str, Any] = {
                "model": mdl,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Anda adalah Kurator Utama Modern SMC/ICT Crypto Futures. "
                            "Kembalikan HANYA objek JSON valid murni tanpa teks pengantar apa pun. "
                            "WAJIB gunakan Bahasa Indonesia pada seluruh isi nilai JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 2400,
            }
            if use_low_reasoning:
                payload["reasoning_effort"] = "low"

            with _RATE_LOCK:
                _MODEL_INFLIGHT[mdl] = _MODEL_INFLIGHT.get(mdl, 0) + 1
            try:
                _pace_request()
                res = sess.post(GROQ_ENDPOINT, headers=headers, json=payload, timeout=26)
                if res.status_code == 200:
                    _record_ratelimit_headers(mdl, res.headers, is_429=False)
                    data = res.json()
                    choices = data.get("choices", [])
                    if choices:
                        content = choices[0].get("message", {}).get("content", "")
                        parsed = parse_llm_json(content)
                        parsed["model"] = mdl
                        return parsed
                elif res.status_code == 429:
                    # Langsung catat cooldown TPM model ini & lompat instan ke model sehat berikutnya tanpa tidur sia-sia
                    _record_ratelimit_headers(mdl, res.headers, is_429=True)
                    last_err = f"HTTP 429 rate-limit TPM pada {mdl}"
                    continue
                elif res.status_code == 400 and "reasoning_effort" in payload:
                    payload.pop("reasoning_effort", None)
                    res2 = sess.post(GROQ_ENDPOINT, headers=headers, json=payload, timeout=26)
                    if res2.status_code == 200:
                        _record_ratelimit_headers(mdl, res2.headers, is_429=False)
                        choices = res2.json().get("choices", [])
                        if choices:
                            parsed = parse_llm_json(choices[0].get("message", {}).get("content", ""))
                            parsed["model"] = mdl
                            return parsed
                    _UNAVAILABLE_MODELS.add(mdl)
                    last_err = f"HTTP {res2.status_code} pada {mdl}"
                    continue
                elif res.status_code in (400, 404):
                    _UNAVAILABLE_MODELS.add(mdl)
                    last_err = f"HTTP {res.status_code} (model tidak tersedia: {mdl})"
                    continue
                else:
                    last_err = f"HTTP {res.status_code} pada {mdl}"
                    time.sleep(0.8 * (pool_pass + 1))
            except Exception as e:
                last_err = f"{mdl}: {e}"
                time.sleep(0.6 * (pool_pass + 1))
            finally:
                with _RATE_LOCK:
                    _MODEL_INFLIGHT[mdl] = max(0, _MODEL_INFLIGHT.get(mdl, 1) - 1)

    if last_err:
        print(f"[AI Validator] Seluruh model cadangan sibuk/gagal ({last_err})", file=sys.stderr)
    return None


def audit_smc_setup(setup: Any, btc_ctx: Optional[dict] = None, model: Optional[str] = None) -> dict:
    """Mengaudit satu setup MTF SMC menggunakan Full LLM dengan disiplin status & fallback aman."""
    key = get_groq_key()
    if not key:
        return {"decision": "NO_API_KEY", "ok": False}

    prompt = build_full_llm_smc_prompt(setup, btc_ctx)
    result = call_groq_api(prompt, model=model, api_key=key)
    if result:
        decision = str(result.get("decision", "CONDITIONAL")).upper()
        if decision not in ("APPROVED", "CONDITIONAL", "REJECTED"):
            decision = "CONDITIONAL"

        try:
            score = int(round(float(result.get("smc_score", getattr(setup, "python_score", 65)))))
            score = int(clamp(score, 0, 100))
        except (ValueError, TypeError):
            score = int(round(getattr(setup, "python_score", 65)))

        raw = getattr(setup, "raw_market_data", {}) or {}
        arch = getattr(setup, "archetype", "TREND_CONT")
        app_code = raw.get("approach_code", "")
        at_poi = bool(raw.get("at_poi", True))
        rev_stage_num = int(raw.get("rev_stage_num", 0))

        status = str(result.get("status", "")).upper().strip()
        if decision == "REJECTED" or score < 55:
            status = "REJECT"
        elif status not in VALID_STATUSES:
            if arch == "PRE_REACTION" and score >= 70 and app_code != "IMPULSIF_LIAR":
                status = "PRE-TAP LIMIT"
            elif arch == "HTF_REVERSAL" and score >= 60:
                status = "ENTRY READY" if (rev_stage_num >= 5 and score >= 80 and decision == "APPROVED" and at_poi) else "REVERSAL WATCH"
            elif score >= 75 and decision == "APPROVED" and at_poi:
                status = "ENTRY READY"
            else:
                status = "WATCHLIST"

        # Disiplin tahapan Reversal Watch & lokasi pasca keputusan AI:
        if arch == "HTF_REVERSAL":
            d_dir = getattr(setup, "direction", "LONG")
            min_rev_ai = 60 if d_dir == "LONG" else 70
            if score < min_rev_ai or decision == "REJECTED":
                status = "REJECT"
                decision = "REJECTED"
            elif status == "ENTRY READY" and (rev_stage_num < 5 or not at_poi or score < 80):
                # Sesuai Aturan Tahap 1-5: sebelum Tahap 5 (Trigger CHoCH M15 di POI) & Skor >= 80, wajib tetap REVERSAL WATCH
                status = "REVERSAL WATCH"
            elif status in ("PRE-TAP LIMIT", "WATCHLIST"):
                status = "REVERSAL WATCH"
        else:
            if status == "REVERSAL WATCH":
                status = "PRE-TAP LIMIT" if (arch == "PRE_REACTION" and app_code != "IMPULSIF_LIAR") else "WATCHLIST"
            if status == "ENTRY READY" and not at_poi:
                if arch == "PRE_REACTION" and app_code != "IMPULSIF_LIAR":
                    status = "PRE-TAP LIMIT"
                else:
                    status = "WATCHLIST"
            if status == "PRE-TAP LIMIT" and (arch != "PRE_REACTION" or app_code == "IMPULSIF_LIAR"):
                status = "WATCHLIST"

        exec_st = str(result.get("execution_style", "")).upper().strip()
        if status == "REJECT":
            exec_st = "SKIP"
        elif status == "REVERSAL WATCH":
            exec_st = "WAIT_SWEEP_TRIGGER"
        elif status == "ENTRY READY":
            exec_st = "ENTRY_NOW"
        elif status == "PRE-TAP LIMIT":
            exec_st = "LIMIT_PRE_TAP"
        elif exec_st not in VALID_EXEC_STYLES:
            exec_st = "WAIT_SWEEP_TRIGGER" if at_poi else "LIMIT_RETEST"

        smc_elements = result.get("smc_elements", {})
        if not isinstance(smc_elements, dict):
            smc_elements = {}

        fwd_play = str(result.get("forward_playbook", result.get("action_plan", ""))).strip()
        flip_sc = str(result.get("flip_scenario", "")).strip()

        return {
            "ok": True,
            "decision": decision,
            "smc_score": score,
            "status": status,
            "execution_style": exec_st,
            "direction": str(result.get("direction", "")).upper(),
            "entry_zone": str(result.get("entry_zone", "")).strip(),
            "sl": result.get("sl"),
            "tp1": result.get("tp1"),
            "tp2": result.get("tp2"),
            "rr": result.get("rr"),
            "smc_elements": smc_elements,
            "detailed_analysis": str(result.get("detailed_analysis", result.get("reason", ""))).strip(),
            "action_plan": fwd_play,
            "forward_playbook": fwd_play,
            "flip_scenario": flip_sc,
            "model": str(result.get("model", "")),
        }

    return {"decision": "FAILED", "ok": False}


def _apply_safe_ai_levels(s: Any, res: dict) -> None:
    """
    Memvalidasi level SL/TP usulan AI dengan koridor "Zoom-Out Check" (ATR 1H) dan
    plafon Tembok 4H lawan, tanpa mengizinkan AI menggeser entry_zone menjadi limit jauh
    (sesuai prinsip Prefer REJECT).
    """
    raw = getattr(s, "raw_market_data", {}) or {}
    ai_dir = res.get("direction")
    if ai_dir in ("LONG", "SHORT") and ai_dir == s.direction:
        d = 1 if s.direction == "LONG" else -1
        entry_p = float(s.entry)
        min_sl_d = float(raw.get("min_sl_dist", abs(entry_p - s.sl) * 0.85))
        max_sl_d = float(raw.get("max_sl_dist", abs(entry_p - s.sl) * 1.35))

        try:
            cand_sl = float(res["sl"]) if res.get("sl") is not None else s.sl
            cand_tp1 = float(res["tp1"]) if res.get("tp1") is not None else s.tp1
            cand_tp2 = float(res["tp2"]) if res.get("tp2") is not None else s.tp2
        except (ValueError, TypeError):
            cand_sl, cand_tp1, cand_tp2 = s.sl, s.tp1, s.tp2

        valid_order = (
            (d == 1 and cand_sl < entry_p < cand_tp1 <= cand_tp2)
            or (d == -1 and cand_sl > entry_p > cand_tp1 >= cand_tp2)
        )
        if valid_order:
            orig_tp2 = float(s.tp2)
            orig_sl_dist = abs(entry_p - float(s.sl))
            sl_dist = clamp(abs(entry_p - cand_sl), 0.85 * min_sl_d, 1.25 * max_sl_d)
            # Jika SL sudah dilindungi di balik kolam likuiditas / Low-High Sweep, cegah AI mempersempit SL!
            if getattr(s, "liq_guard_info", ""):
                sl_dist = max(sl_dist, orig_sl_dist * 0.98)
            s.sl = (entry_p - sl_dist) if d == 1 else (entry_p + sl_dist)

            # Pastikan TP2 usulan AI tidak menembus Tembok Supply/Demand 4H lawan yang sudah dipetakan Python
            wall_sup_bot = float(raw.get("wall_supply_bot", float("nan")))
            wall_dem_top = float(raw.get("wall_demand_top", float("nan")))
            if d == 1 and not math.isnan(wall_sup_bot) and wall_sup_bot > entry_p:
                cand_tp2 = min(cand_tp2, wall_sup_bot)
            elif d == -1 and not math.isnan(wall_dem_top) and wall_dem_top < entry_p:
                cand_tp2 = max(cand_tp2, wall_dem_top)

            max_tp2_dist = max(abs(orig_tp2 - entry_p) * 1.35, 5.5 * sl_dist)
            cand_rr2 = abs(cand_tp2 - entry_p) / max(sl_dist, 1e-12)
            orig_rr2 = abs(orig_tp2 - entry_p) / max(sl_dist, 1e-12)
            # Hanya terima TP2 usulan AI jika tetap memenuhi standar R:R >= 1:3.0 (jangan biarkan AI memangkas TP2 di bawah 3R!)
            if abs(cand_tp2 - entry_p) <= max_tp2_dist and (cand_rr2 >= 3.0 or orig_rr2 < 3.0) and (
                (d == 1 and cand_tp1 <= cand_tp2) or (d == -1 and cand_tp1 >= cand_tp2)
            ):
                s.tp1 = cand_tp1
                s.tp2 = cand_tp2

    # Jangan timpa s.entry_zone dengan limit jauh buatan AI (kunci sesuai zona POI aktual Python)

    risk = max(abs(s.entry - s.sl), 1e-12)
    s.risk_pct = risk / s.entry * 100.0
    s.tp1_pct = abs(s.tp1 - s.entry) / s.entry * 100.0
    s.tp2_pct = abs(s.tp2 - s.entry) / s.entry * 100.0
    s.rr_tp1 = abs(s.tp1 - s.entry) / risk
    s.rr = abs(s.tp2 - s.entry) / risk
    tp3_v = getattr(s, "tp3", float("nan"))
    if isinstance(tp3_v, (int, float)) and not math.isnan(tp3_v):
        d_sign = 1 if s.direction == "LONG" else -1
        if (d_sign == 1 and tp3_v < s.tp2 + 0.8 * risk) or (d_sign == -1 and tp3_v > s.tp2 - 0.8 * risk):
            tp3_v = (s.tp2 + 1.2 * risk) if d_sign == 1 else (s.tp2 - 1.2 * risk)
            s.tp3 = tp3_v
        s.tp3_pct = abs(tp3_v - s.entry) / s.entry * 100.0
        s.rr_tp3 = abs(tp3_v - s.entry) / risk

    atr_1h = getattr(s, "atr_1h", float("nan"))
    if not math.isnan(atr_1h) and atr_1h > 0:
        s.sl_atr_1h_ratio = risk / atr_1h
        vel = float(raw.get("hourly_velocity", 0.45 * atr_1h))
        if vel > 0:
            s.est_hold_hours_tp1 = max(0.5, abs(s.tp1 - s.entry) / vel)
            s.est_hold_hours_tp2 = max(1.0, abs(s.tp2 - s.entry) / vel)

    if s.direction == "LONG":
        s.invalidation = f"Close 1H/15M menembus bawah SL struktural {fp(s.sl)} (zona POI {s.entry_zone})."
    else:
        s.invalidation = f"Close 1H/15M menembus atas SL struktural {fp(s.sl)} (zona POI {s.entry_zone})."

    z_st = getattr(s, "zone_state", "")
    arch = getattr(s, "archetype", "TREND_CONT")
    rev_stage_num = int(raw.get("rev_stage_num", 0))
    if s.rr < 2.95 or getattr(s, "ai_decision", "") == "REJECTED":
        s.status = "REJECT"
        s.exec_style = "SKIP"
    elif arch == "HTF_REVERSAL":
        if s.status == "ENTRY READY" and (rev_stage_num < 5 or z_st in ("POST_BOUNCE_WAIT", "WAITING_PULLBACK", "PRE_TAP_STALK") or s.score < 80.0):
            s.status = "REVERSAL WATCH"
            s.exec_style = "WAIT_SWEEP_TRIGGER"
    elif s.status == "ENTRY READY" and (
        s.rr < 2.98 or z_st in ("POST_BOUNCE_WAIT", "WAITING_PULLBACK", "IN_POI_TESTING", "PRE_TAP_STALK")
    ):
        if arch == "PRE_REACTION" and s.rr >= 2.98 and raw.get("approach_code") != "IMPULSIF_LIAR":
            s.status = "PRE-TAP LIMIT"
            s.exec_style = "LIMIT_PRE_TAP"
        else:
            s.status = "WATCHLIST"


def _select_balanced_ai_slate(candidates: List[Any], max_audit: int) -> List[Any]:
    """
    Memilih kandidat kurasi AI secara proporsional lintas variasi (TREND_CONT, PRE_REACTION, HTF_REVERSAL)
    sehingga AI tidak hanya mengaudit satu tipe setup saja.
    """
    if len(candidates) <= max_audit:
        return candidates

    selected: List[Any] = []
    seen_syms: set = set()

    # 1. Pastikan seluruh ENTRY READY terambil lebih dulu
    for r in candidates:
        if getattr(r, "status", "") == "ENTRY READY" and len(selected) < max_audit:
            selected.append(r)
            seen_syms.add(r.symbol)

    # 2. Pastikan perwakilan terbaik dari HTF_REVERSAL dan PRE_REACTION masuk kuota kurasi AI
    for arch_target, quota in (("HTF_REVERSAL", max(3, max_audit // 3)), ("PRE_REACTION", max(2, max_audit // 3))):
        cnt = 0
        for r in candidates:
            if len(selected) >= max_audit or cnt >= quota:
                break
            if r.symbol not in seen_syms and getattr(r, "archetype", "") == arch_target:
                selected.append(r)
                seen_syms.add(r.symbol)
                cnt += 1

    # 3. Isi sisa slot berdasarkan urutan prioritas global
    for r in candidates:
        if len(selected) >= max_audit:
            break
        if r.symbol not in seen_syms:
            selected.append(r)
            seen_syms.add(r.symbol)

    return selected


def audit_top_setups(
    results: List[Any],
    btc_ctx: Optional[dict] = None,
    max_audit: int = 10,
    workers: int = 3,
    equity: float = 0.0,
    risk_pct: float = 1.0,
    fee_pct: float = 0.05,
) -> None:
    """
    Melakukan kurasi Full LLM sebagai HAKIM PENILAIAN AKHIR pada seluruh kandidat 3 Pilar Paten
    maupun kandidat borderline yang sudah memenuhi syarat mutlak geometri R:R >= 1:3.0 (tanpa terhalang Tembok 4H):
    - Jika API Groq rate-limit / timeout / gagal parse, SKOR & STATUS PYTHON DIPERTAHANKAN UTUH (Proteksi Skor 0).
    - Memperbarui skor akhir, status, Forward Playbook, & Flip Scenario dengan keputusan taktis AI.
    """
    if not results or not get_groq_key():
        return

    def _is_ai_eligible(r: Any) -> bool:
        if getattr(r, "rr", 0.0) < 2.95:
            return False
        st = str(getattr(r, "status", ""))
        if st in ("ENTRY READY", "PRE-TAP LIMIT", "REVERSAL WATCH"):
            return True
        if st.startswith("WATCHLIST") and getattr(r, "score", 0.0) >= 70.0:
            return True
        # Sertakan juga kandidat borderline yang secara geometri R:R >= 1:3.0 & Tembok 4H sudah valid agar AI yang menjadi penentu akhir!
        raw_r = getattr(r, "raw_market_data", {}) or {}
        if bool(raw_r.get("valid_rr_and_range", False)):
            arch_r = getattr(r, "archetype", "")
            sc_r = float(getattr(r, "score", 0.0))
            if arch_r == "HTF_REVERSAL" and sc_r >= 60.0:
                return True
            if arch_r == "TREND_CONT" and bool(raw_r.get("htf_aligned", False)) and getattr(r, "kind", "") == "SMC_SNIPER" and sc_r >= 70.0:
                return True
        return False

    candidates = [r for r in results if _is_ai_eligible(r)]
    st_prio = {
        "ENTRY READY": 0,
        "PRE-TAP LIMIT": 1,
        "REVERSAL WATCH": 1,
    }
    z_prio = {
        "FIRST_TOUCH_NOW": 0,
        "SWEEP_IN_POI": 0,
        "IN_POI_REJECTING": 1,
        "PRE_TAP_NEAR": 1,
        "IN_POI_TESTING": 2,
        "PRE_TAP_STALK": 2,
        "LIQ_SWEPT": 3,
        "APPROACHING_POI": 4,
        "POST_BOUNCE_WAIT": 5,
        "WAITING_PULLBACK": 6,
    }
    candidates.sort(
        key=lambda r: (
            st_prio.get(getattr(r, "status", ""), 2),
            z_prio.get(getattr(r, "zone_state", ""), 6),
            -getattr(r, "score", 0.0),
        )
    )
    targets = _select_balanced_ai_slate(candidates, max_audit)
    if not targets:
        return

    print(
        f"\n[AI Kurator SMC/ICT] Mengkurasi paralel {len(targets)} kandidat lintas variasi bersama Groq LLM ...",
        file=sys.stderr,
    )

    with ThreadPoolExecutor(max_workers=min(max(1, workers), len(targets))) as pool:
        futures = {pool.submit(audit_smc_setup, s, btc_ctx): s for s in targets}
        for f in as_completed(futures):
            s = futures[f]
            sym = getattr(s, "symbol", "")
            try:
                res = f.result()
                if not res.get("ok"):
                    s.ai_decision = "PYTHON_FALLBACK"
                    s.ai_confidence = float(s.score)
                    s.ai_summary = "Evaluasi LLM dilewati sementara (rate-limit/timeout API); mempertahankan evaluasi & playbook MTF Python."
                    print(
                        f"  [Groq SMC] {sym:12s} -> ⏭️ FALLBACK    | Skor Python dipertahankan: {s.score:>4.0f} | Status: {s.status}",
                        file=sys.stderr,
                    )
                    continue

                s.smc_elements = res.get("smc_elements", {})
                s.score = float(res["smc_score"])
                s.status = res["status"]
                s.exec_style = res["execution_style"]
                s.ai_decision = res["decision"]
                s.ai_confidence = float(res["smc_score"])
                clean_bep = lambda txt: re.sub(
                    r"(?:,?\s*(?:dan|&|\+)?\s*geser\s+sl\s+ke\s+(?:bep|break[- ]?even)[^.,;]*)",
                    ", tahan SL tetap di tempat (tanpa geser BEP)",
                    txt,
                    flags=re.IGNORECASE,
                )
                s.ai_summary = clean_bep(res["detailed_analysis"])
                s.ai_action_note = clean_bep(res["action_plan"])
                if res.get("forward_playbook"):
                    s.forward_playbook = clean_bep(res["forward_playbook"])
                if res.get("flip_scenario"):
                    s.flip_scenario = res["flip_scenario"]
                s.ai_model = res["model"]

                _apply_safe_ai_levels(s, res)
                if equity > 0:
                    size_position(s, equity, risk_pct, fee_pct)

                dec = s.ai_decision
                icon = "✅" if dec == "APPROVED" else ("⚠️" if dec == "CONDITIONAL" else "❌")
                arch_short = getattr(s, "archetype", "SMC")[:10]
                print(
                    f"  [Groq SMC] {sym:12s} [{arch_short:<10s}] -> {icon} {dec:<11s} | Skor: {s.score:>3.0f} | RR 1:{s.rr:.1f} | {s.status}",
                    file=sys.stderr,
                )
            except Exception as e:
                print(f"  [Groq SMC] {sym:12s} -> ⏭️ Error ({e}); skor Python {s.score:.0f} dipertahankan.", file=sys.stderr)
