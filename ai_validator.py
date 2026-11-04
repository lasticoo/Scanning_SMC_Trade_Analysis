# -*- coding: utf-8 -*-
"""
AI Top-Down MTF SMC Validator Module for Binance Futures Scanner
================================================================
Memanfaatkan LLM (Groq Cloud API) sebagai Kepala Analis Top-Down MTF SMC (4H -> 1H -> 15M):
1. Menerima peta struktur 3 Timeframe UTUH (4H Key Ranges & Walls -> 1H Working Range, HTF OB/FVG,
   & Estimasi Durasi Posisi -> 15M Likuiditas Kecil, Trigger, CVD Absorption, OI, & L/S Ratio).
2. Memakai format harga adaptif presisi penuh (aman untuk altcoin mikro maupun BTC).
3. Dilengkapi parser JSON tahan tag <think>...</think>, rate-limit pacing + backoff otomatis,
   serta PROTEKSI SKOR 0 (jika API limit/timeout, skor & status MTF Python dipertahankan utuh).
4. Menerapkan pagar pengaman "Zoom-Out Check" (ATR 1H) pasca-AI agar SL/TP selalu realistis,
   serta menyinkronkan ulang risk_pct, RR, durasi, invalidasi, dan ukuran posisi (qty).
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
_MIN_CALL_INTERVAL = 0.45  # cegah burst 429 serentak ke Groq
_UNAVAILABLE_MODELS: set = set()  # simpan model deprecated/404 agar tidak dicoba berulang kali


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
    Menyusun prompt Top-Down MTF (4H -> 1H -> 15M) + Orderflow lengkap dengan format harga
    adaptif (tanpa kehilangan presisi desimal) agar AI tidak menebak atau mengarang level.
    """
    direction = getattr(setup, "direction", "UNKNOWN")
    symbol = getattr(setup, "symbol", "")
    pattern = getattr(setup, "pattern", "")
    entry_zone = getattr(setup, "entry_zone", "")
    plan_entry = getattr(setup, "entry", 0.0)
    sl = getattr(setup, "sl", 0.0)
    tp1 = getattr(setup, "tp1", 0.0)
    tp2 = getattr(setup, "tp2", 0.0)
    rr = getattr(setup, "rr", 0.0)
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
    idm_txt = f"Level {fp(idm_lvl)} ({'SUDAH DISAPU / IDM Taken' if idm_taken else 'Belum disapu'})" if idm_lvl else "Tidak ada IDM spesifik"

    pins = raw.get("pin_bars", [])
    pin_txt = ", ".join(f"{'Bull' if p['dir']==1 else 'Bear'} PinBar (wick {p['wick_pct']*100:.0f}%)" for p in pins) or "Tidak ada pin bar baru"

    sess_txt = raw.get("session", getattr(setup, "session_name", "-"))
    of_txt = raw.get("orderflow_summary", getattr(setup, "orderflow_summary", "-"))
    fund_pct = float(raw.get("funding_rate", 0.0)) * 100.0
    opp_txt = fp(opp_lvl) if opp_lvl is not None else "Belum teridentifikasi"

    # Data Detektor Lokasi Harga Live (Demand/Supply/POI/Sweep/Reaction)
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

    prompt = f"""Anda adalah Master Trader Institusional Crypto Futures spesialis Top-Down Multi-Timeframe SMC (4H -> 1H -> 15M) & Orderflow.
Evaluasi setup berikut menggunakan kerangka Top-Down 3 Lapis tanpa keterikatan kaku pada BTC:

[DATA REAL-TIME BINANCE FUTURES: {symbol}]
- Harga Saat Ini: {fp(cur_p)} | Sesi Aktif: {sess_txt}
- Arah & Setup Kandidat: {direction} — {pattern} ({'Harga sudah di area POI' if at_poi else 'Harga menunggu pullback Limit ke POI'})
- Konteks Makro BTC: {btc_note}

[RADAR LOKASI HARGA LIVE DI DEMAND / SUPPLY / POI / LIQUIDITY SWEEP]
- Status Lokasi Live: {z_badge} (State: {z_state} | Jarak ke POI: {dist_poi:.2f}% | Kedalaman Mitigasi Zona: {depth_txt})
- Zona Demand/Supply/OB/FVG Aktif yang Disentuh/Dekat: {active_zones_txt}
- Deteksi Liquidity Sweep Terkini: {sweep_live_txt} | Magnet Likuiditas Terdekat (EQH/EQL/IDM): {magnet_txt}
- Reaksi Candle & Orderflow di Zona: {reac_live_txt}

[LAPIS 1 — 4H MACRO KEY RANGES & TEMBOK BESAR]
- Tren & Struktur 4H: {t4} | ATR 4H: {fp(atr_4h)}
- Dealing Range 4H: {r4_str}
- Peta Tembok 4H: {w4_str}
- Order Blocks 4H Aktif: {obs_4h_txt}

[LAPIS 2 — 1H WORKING TRADING RANGE & DURASI POSISI]
- Tren & Struktur 1H: {s1} | ATR 1H: {fp(atr_1h)}
- Working Range 1H: Low {fp(r1_lo)} <-> High {fp(r1_hi)} | Equilibrium 1H: {fp(r1_eq)} | Posisi Harga: {z_pos_lbl}
- Zona OB / Breaker 1H Aktif: {obs_1h_txt}
- Unmitigated FVG 1H: {fvgs_1h_txt}
- Kolam Likuiditas 1H: EQH [{eqh_1h}] | EQL [{eql_1h}]
- Estimasi Lama Posisi ke TP: {hold_lbl}

[LAPIS 3 — 15M REFINEMENT, LIKUIDITAS KECIL, & ORDERFLOW]
- ATR 15M: {fp(atr_15m)} | Event Struktur 15M: {recent_events}
- Zona OB / Breaker 15M Aktif: {obs_15_txt}
- Unmitigated FVG 15M: {fvgs_15_txt}
- Likuiditas 15M: EQH [{eqh_15}] | EQL [{eql_15}] | Inducement (IDM): {idm_txt}
- Rejection Candle: {pin_txt}
- Data Orderflow & Derivatif Publik: {of_txt} | Funding Rate: {fund_pct:+.4f}%

[RENCANA LEVEL TOP-DOWN & ATURAN WAJIB "ZOOM-OUT CHECK" (PROPORSIONAL 1H)]
- Acuan Entry: {fp(plan_entry)} (Zona POI Ideal: {entry_zone})
- Stop Loss Struktural Proporsional 1H: {fp(sl)} | TP1: {fp(tp1)} | TP2: {fp(tp2)} (R:R 1:{rr:.1f}) | Target Tembok Lawan: {opp_txt}
- Invalidasi: {invalidation}
- PENTING ("ZOOM-OUT CHECK"): Di crypto futures, DILARANG memasang SL sempit mikro 15M yang mudah terkena stop-hunt wick! Jarak SL dari Entry WAJIB proporsional terhadap 1H, yaitu di antara {fp(min_sl_d)} hingga {fp(max_sl_d)} (0.75x - 2.20x ATR 1H), serta TP2 tidak boleh menembus tembok besar 4H lawan. Pertahankan level SL/TP Top-Down di atas kecuali ada penyempurnaan struktural di dalam koridor tersebut.

[KEPUTUSAN AKHIR]
- "ENTRY READY": Konfluensi MTF 4H->1H->15M kuat, harga sudah dekat/di dalam POI, R:R >= 2.0, durasi wajar.
- "WATCHLIST": Struktur & arah bagus tetapi harga masih menunggu retest ke POI 1H/15M atau menunggu trigger konfirmasi.
- "REJECT": Melawan tembok 4H tepat di depan mata, R:R < 1.8, atau jebakan likuiditas buruk.

Balas HANYA dengan JSON valid murni (tanpa teks pengantar atau markdown):
{{
  "smc_elements": {{
    "order_blocks": "peta OB/Breaker 4H, 1H, dan 15M yang relevan beserta harga presisinya",
    "liquidity_sweeps_runs": "status BSL/SSL sweep, EQH/EQL, atau Asian sweep",
    "liquidity_range": "peta Macro Range 4H & Working Range 1H (Low, High, EQ, posisi Discount/Premium)",
    "choch": "status CHOCH di 1H/15M",
    "bos": "status BOS searah struktur",
    "inducement": "level IDM spesifik dan status IDM Taken",
    "supply_demand": "zona Supply & Demand 4H/1H dan batas tembok lawan",
    "pin_bar": "status rejection wick & konfirmasi orderflow/CVD absorption",
    "cisd": "status CISD & dukungan Open Interest / L-S Ratio",
    "poi": "zona POI utama (Nested 1H+15M) beserta estimasi durasi posisi"
  }},
  "decision": "APPROVED" | "CONDITIONAL" | "REJECTED",
  "status": "ENTRY READY" | "WATCHLIST" | "REJECT",
  "smc_score": 78,
  "direction": "LONG" | "SHORT",
  "entry_zone": "{entry_zone}",
  "sl": {sl},
  "tp1": {tp1},
  "tp2": {tp2},
  "rr": {rr:.2f},
  "detailed_analysis": "ulasan institusional Top-Down (4H->1H->15M) + Orderflow mengapa diambil/ditunggu/ditolak",
  "action_plan": "langkah eksekusi taktis (limit vs trigger, manajemen BEP saat TP1, batas waktu hold)"
}}"""
    return prompt


def parse_llm_json(text: str) -> dict:
    """
    Mengekstrak objek JSON secara tangguh dari respons LLM:
    - Membuang blok reasoning <think>...</think> (termasuk model Qwen / DeepSeek)
    - Membuang pembungkus markdown ```json ... ```
    - Mencari kurung kurawal seimbang terluar {...} & membersihkan trailing comma
    """
    if not text:
        raise ValueError("Respons LLM kosong")

    # 1. Hapus blok <think>...</think>
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Jika <think> terpotong tanpa tag penutup sebelum '{', buang teks sebelum JSON
    if "<think>" in cleaned.lower():
        cleaned = re.sub(r"^.*?<think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    cleaned = cleaned.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    # 2. Coba parse langsung
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # 3. Ekstrak blok {...} seimbang pertama yang valid
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
    # Perbaiki trailing comma umum dari LLM sebelum '}' atau ']'
    candidate_fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
    obj = json.loads(candidate_fixed)
    if not isinstance(obj, dict):
        raise ValueError("JSON hasil parse bukan dict")
    return obj


def _pace_request() -> None:
    global _LAST_CALL_TS
    with _RATE_LOCK:
        now = time.monotonic()
        wait = _MIN_CALL_INTERVAL - (now - _LAST_CALL_TS)
        if wait > 0:
            time.sleep(wait)
        _LAST_CALL_TS = time.monotonic()


def call_groq_api(prompt: str, model: Optional[str] = None, api_key: Optional[str] = None) -> Optional[dict]:
    """Mengirim permintaan ke Groq Cloud API dengan rate-pacing, exponential backoff, & multi-model fallback."""
    key = api_key or get_groq_key()
    if not key:
        return None

    primary_model = model or get_default_model()
    fallback_model = get_fallback_model()
    safety_model = "llama-3.3-70b-versatile"

    models_to_try: List[str] = []
    for m in (primary_model, fallback_model, safety_model):
        if m and m not in models_to_try and m not in _UNAVAILABLE_MODELS:
            models_to_try.append(m)
    if not models_to_try:
        models_to_try = [safety_model]

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    for mdl in models_to_try:
        payload = {
            "model": mdl,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Anda adalah Kepala Analis Top-Down MTF SMC Crypto Futures. "
                        "Kembalikan HANYA objek JSON valid tanpa teks pengantar apa pun."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 2300,
        }
        for attempt in range(3):
            _pace_request()
            try:
                res = requests.post(GROQ_ENDPOINT, headers=headers, json=payload, timeout=28)
                if res.status_code == 200:
                    data = res.json()
                    choices = data.get("choices", [])
                    if choices:
                        content = choices[0].get("message", {}).get("content", "")
                        parsed = parse_llm_json(content)
                        parsed["model"] = mdl
                        return parsed
                elif res.status_code == 429:
                    retry_hdr = res.headers.get("Retry-After", "")
                    wait_s = float(retry_hdr) if retry_hdr.replace(".", "", 1).isdigit() else (2.5 * (attempt + 1))
                    time.sleep(min(wait_s, 15.0))
                    continue
                elif res.status_code in (400, 404):
                    # Model tidak tersedia / deprecated -> catat agar tidak diulang & pindah ke fallback
                    _UNAVAILABLE_MODELS.add(mdl)
                    break
                else:
                    time.sleep(1.5 * (attempt + 1))
            except Exception as e:
                if attempt == 2:
                    print(f"[AI Validator] Model {mdl} gagal pada percobaan terakhir: {e}", file=sys.stderr)
                time.sleep(1.2 * (attempt + 1))

    return None


def audit_smc_setup(setup: Any, btc_ctx: Optional[dict] = None, model: Optional[str] = None) -> dict:
    """Mengaudit satu setup MTF SMC menggunakan Full LLM dengan fallback aman."""
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

        status = str(result.get("status", "")).upper()
        if status not in ("ENTRY READY", "WATCHLIST", "REJECT"):
            status = "ENTRY READY" if (score >= 70 and decision == "APPROVED") else ("WATCHLIST" if score >= 52 else "REJECT")

        # Jika harga sebenarnya jauh dari POI (extended), tahan di WATCHLIST agar trader pasang Limit, bukan kejar market
        raw = getattr(setup, "raw_market_data", {}) or {}
        if status == "ENTRY READY" and raw.get("at_poi") is False:
            status = "WATCHLIST"

        smc_elements = result.get("smc_elements", {})
        if not isinstance(smc_elements, dict):
            smc_elements = {}

        return {
            "ok": True,
            "decision": decision,
            "smc_score": score,
            "status": status,
            "direction": str(result.get("direction", "")).upper(),
            "entry_zone": str(result.get("entry_zone", "")).strip(),
            "sl": result.get("sl"),
            "tp1": result.get("tp1"),
            "tp2": result.get("tp2"),
            "rr": result.get("rr"),
            "smc_elements": smc_elements,
            "detailed_analysis": str(result.get("detailed_analysis", result.get("reason", ""))).strip(),
            "action_plan": str(result.get("action_plan", "")).strip(),
            "model": str(result.get("model", "")),
        }

    return {"decision": "FAILED", "ok": False}


def _apply_safe_ai_levels(s: Any, res: dict) -> None:
    """
    Memvalidasi level SL/TP usulan AI dengan koridor "Zoom-Out Check" (ATR 1H)
    agar AI tidak pernah merusak SL menjadi terlalu sempit/terbalik, lalu
    menghitung ulang risk_pct, tp1_pct, tp2_pct, RR, durasi hold, dan invalidasi.
    """
    raw = getattr(s, "raw_market_data", {}) or {}
    # Cegah AI membalik arah tanpa dasar level lengkap (jika berlawanan arah, pertahankan geometri Python)
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

        # Cek urutan arah level valid
        valid_order = (
            (d == 1 and cand_sl < entry_p < cand_tp1 <= cand_tp2)
            or (d == -1 and cand_sl > entry_p > cand_tp1 >= cand_tp2)
        )
        if valid_order:
            orig_tp1, orig_tp2 = float(s.tp1), float(s.tp2)
            # Pagar Zoom-Out Check: tolak SL AI jika lebih sempit dari 85% batas minimal 1H!
            sl_dist = clamp(abs(entry_p - cand_sl), 0.85 * min_sl_d, 1.25 * max_sl_d)
            s.sl = (entry_p - sl_dist) if d == 1 else (entry_p + sl_dist)
            # Pagar Anti-Typo TP2 AI: cegah salah ketik digit LLM (mis. 0.8755 -> 0.755) yang melampaui tembok 4H/durasi
            max_tp2_dist = max(abs(orig_tp2 - entry_p) * 1.35, 4.5 * sl_dist)
            if abs(cand_tp2 - entry_p) <= max_tp2_dist:
                s.tp1 = cand_tp1
                s.tp2 = cand_tp2

    if res.get("entry_zone"):
        s.entry_zone = res["entry_zone"]

    # Hitung ulang seluruh metrik turunan secara konsisten!
    risk = max(abs(s.entry - s.sl), 1e-12)
    s.risk_pct = risk / s.entry * 100.0
    s.tp1_pct = abs(s.tp1 - s.entry) / s.entry * 100.0
    s.tp2_pct = abs(s.tp2 - s.entry) / s.entry * 100.0
    s.rr_tp1 = abs(s.tp1 - s.entry) / risk
    s.rr = abs(s.tp2 - s.entry) / risk

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

    # Penjaga kualitas R:R & status lokasi zona pasca-penyesuaian AI
    z_st = getattr(s, "zone_state", "")
    if s.rr < 1.55:
        s.status = "REJECT"
    elif s.status == "ENTRY READY" and (s.rr < 1.96 or z_st in ("WAITING_PULLBACK", "IN_POI_TESTING")):
        s.status = "WATCHLIST"


def audit_top_setups(
    results: List[Any],
    btc_ctx: Optional[dict] = None,
    max_audit: int = 15,
    workers: int = 3,
    equity: float = 0.0,
    risk_pct: float = 1.0,
    fee_pct: float = 0.05,
) -> None:
    """
    Melakukan evaluasi Full LLM pada kandidat setup terbaik secara aman:
    - Jika API Groq rate-limit / timeout / gagal parse, SKOR & STATUS PYTHON DIPERTAHANKAN UTUH
      (tidak pernah lagi menimpa BTC/ETH/koin utama menjadi skor 0!).
    - Menyinkronkan kembali position sizing setelah level divalidasi.
    """
    if not results or not get_groq_key():
        return

    candidates = [
        r for r in results
        if (getattr(r, "status", "") in ("ENTRY READY", "WATCHLIST") or getattr(r, "score", 0) >= 48)
        and getattr(r, "rr", 0.0) >= 1.55
    ]
    z_prio = {
        "SWEEP_IN_POI": 0,
        "IN_POI_REJECTING": 1,
        "IN_POI_TESTING": 2,
        "LIQ_SWEPT": 3,
        "APPROACHING_POI": 4,
        "WAITING_PULLBACK": 5,
    }
    candidates.sort(
        key=lambda r: (
            0 if getattr(r, "status", "") == "ENTRY READY" else 1,
            z_prio.get(getattr(r, "zone_state", ""), 6),
            -getattr(r, "score", 0.0),
        )
    )
    targets = candidates[:max_audit]
    if not targets:
        return

    print(f"\n[AI Evaluator] Mengevaluasi {len(targets)} kandidat Top-Down MTF bersama Groq LLM ...", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=min(max(1, workers), len(targets))) as pool:
        futures = {pool.submit(audit_smc_setup, s, btc_ctx): s for s in targets}
        for f in as_completed(futures):
            s = futures[f]
            sym = getattr(s, "symbol", "")
            try:
                res = f.result()
                if not res.get("ok"):
                    # PROTEKSI SKOR 0: Pertahankan skor, status, dan level Top-Down Python!
                    s.ai_decision = "PYTHON_FALLBACK"
                    s.ai_confidence = float(s.score)
                    s.ai_summary = "Evaluasi LLM dilewati sementara (rate-limit/timeout API); menggunakan skor & level Top-Down MTF Python."
                    print(
                        f"  [Full LLM] {sym:12s} -> ⏭️ FALLBACK    | Skor MTF Python dipertahankan: {s.score:>4.0f} | Status: {s.status}",
                        file=sys.stderr,
                    )
                    continue

                s.smc_elements = res.get("smc_elements", {})
                s.score = float(res["smc_score"])
                s.status = res["status"]
                s.ai_decision = res["decision"]
                s.ai_confidence = float(res["smc_score"])
                s.ai_summary = res["detailed_analysis"]
                s.ai_action_note = res["action_plan"]
                s.ai_model = res["model"]

                _apply_safe_ai_levels(s, res)
                if equity > 0:
                    size_position(s, equity, risk_pct, fee_pct)

                dec = s.ai_decision
                icon = "✅" if dec == "APPROVED" else ("⚠️" if dec == "CONDITIONAL" else "❌")
                print(
                    f"  [Full LLM] {sym:12s} -> {icon} {dec:<11s} | Skor: {s.score:>3.0f} | RR 1:{s.rr:.1f} | Status: {s.status}",
                    file=sys.stderr,
                )
            except Exception as e:
                print(f"  [Full LLM] {sym:12s} -> ⏭️ Error ({e}); skor Python {s.score:.0f} dipertahankan.", file=sys.stderr)
