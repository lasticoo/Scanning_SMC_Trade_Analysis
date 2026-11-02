# -*- coding: utf-8 -*-
"""
AI SMC Validator Module for Binance Futures Scanner
===================================================
Modul ini memanfaatkan LLM (Groq Cloud API) dengan kunci API dari aplikasi TEIS
sebagai PENILAI MANDIRI & TUNGGAL (Full LLM SMC Evaluator) dari data real-time Binance.

Tugas AI:
1. Menganalisis koin SECARA INDEPENDEN (TIDAK terikat/berpatokan kaku pada tren BTC).
2. Menentukan secara spesifik 10 Elemen Kunci SMC:
   - Order Block Bearish & Bullish
   - Liquidity Sweep & Liquidity Run (BSL / SSL)
   - Liquidity Range (Swing Low & Swing High, Equilibrium, Diskon/Premium)
   - CHOCH (Change of Character)
   - Inducement (IDM & status mitigasi)
   - Supply & Demand Zones
   - BOS (Break of Structure)
   - Candle Pin Bar (Rejection Wick di POI)
   - CISD (Change in State of Delivery)
   - POI (Order Block yang terbentuk setelah liquidity sweep)
3. Memutuskan secara mandiri apakah MASUK (ENTRY READY), WATCHLIST, atau DITOLAK (REJECT).

Model default: qwen/qwen3.8-27b (fallback: openai/gpt-oss-120b)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import requests

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"


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


def build_full_llm_smc_prompt(setup: Any, btc_ctx: Optional[dict] = None) -> str:
    """Menyusun prompt analitis institusional lengkap dari data pasar real-time Binance."""
    direction = getattr(setup, "direction", "UNKNOWN")
    symbol = getattr(setup, "symbol", "")
    pattern = getattr(setup, "pattern", "")
    kind = getattr(setup, "kind", "")
    entry_zone = getattr(setup, "entry_zone", "")
    entry_p = getattr(setup, "entry", 0.0)
    sl = getattr(setup, "sl", 0.0)
    tp1 = getattr(setup, "tp1", 0.0)
    tp2 = getattr(setup, "tp2", 0.0)
    rr = getattr(setup, "rr", 0.0)
    t4 = getattr(setup, "trend_4h", "Netral")
    s1 = getattr(setup, "struct_1h", "Netral")
    confirm = getattr(setup, "confirm_txt", "")
    invalidation = getattr(setup, "invalidation", "")
    opp_lvl = getattr(setup, "opp_level", None)
    raw = getattr(setup, "raw_market_data", {})

    opp_txt = f"{opp_lvl:.4f}" if opp_lvl is not None else "Belum teridentifikasi"

    # Detail data teknikal real-time Binance
    cur_p = raw.get("current_price", entry_p)
    rng_h = raw.get("range_high", 0.0)
    rng_l = raw.get("range_low", 0.0)
    eq_p = raw.get("equilibrium", 0.0)
    disc_pct = raw.get("discount_pct", 50.0)
    recent_events = ", ".join(raw.get("struct_events", [])) or confirm

    # Info BTC hanya untuk referensi makro sekunder, BUKAN penentu mutlak
    btc_note = "Konteks BTC Netral"
    if btc_ctx and "4h" in btc_ctx:
        b4 = btc_ctx["4h"].get("st", 0)
        b4_str = "Bullish" if b4 == 1 else ("Bearish" if b4 == -1 else "Netral")
        btc_note = f"BTC 4H: {b4_str} (Hanya referensi makro; koin altcoin sering bergerak independen)"

    prompt = f"""Anda adalah Master Trader Institusional dan Kepala Analis Smart Money Concepts (SMC) & ICT (Inner Circle Trader).
Tugas Anda: Dari data pasar real-time Binance di bawah ini, tentukan secara mandiri ke-10 Elemen Kunci SMC dan putuskan apakah aset ini LAYAK MASUK (ENTRY READY) atau TIDAK (WATCHLIST / REJECT).

[PANDUAN PENTING MENGENAI BTC]
Jangan mendiskualifikasi atau menolak koin hanya karena arahnya berbeda dengan Bitcoin (BTC). Banyak altcoin bergerak mandiri karena momentum, rotasi likuiditas, dan ekosistem independen. Nilai koin ini 100% MURNI berdasarkan Price Action dan struktur SMC koin itu sendiri!

[DATA PASAR REAL-TIME BINANCE]
- Simbol Pasangan: {symbol}
- Harga Terkini: {cur_p:.4f}
- Dealing Range 15M: High {rng_h:.4f} | Low {rng_l:.4f} | Equilibrium: {eq_p:.4f} (Posisi: {disc_pct:.1f}% Discount/Premium)
- Tren Struktur 4H Koin: {t4}
- Tren Struktur 1H Koin: {s1}
- Arah Awal yang Dianalisis: {direction} ({pattern})
- Event Struktur Terdeteksi: {recent_events}
- Acuan Level Eksekusi: Entry {entry_zone} | Batas Risiko (SL) {sl:.4f} | TP1 {tp1:.4f} | TP2 {tp2:.4f} (R:R 1:{rr:.1f})
- Target Likuiditas Lawan: {opp_txt}
- Invalidasi Struktural Awal: {invalidation}
- Konteks Makro BTC: {btc_note}

[TUGAS ANDA: TENTUKAN 10 ELEMEN SMC INSTITUSIONAL]
Identifikasi dan jelaskan secara ringkas 10 komponen berikut:
1. Order Blocks: Sebutkan zona Order Block Bullish dan Bearish yang aktif.
2. Liquidity Sweeps & Runs: Apakah ada BSL/SSL Sweep atau Liquidity Run terkini?
3. Liquidity Range: Swing High, Swing Low, Equilibrium, dan apakah harga di zona Discount (ideal Buy) atau Premium (ideal Sell).
4. CHOCH: Change of Character pembalikan arah karakter pasar.
5. Inducement: Level Inducement (IDM) dan apakah sudah disapu (IDM Taken) untuk menjebak retail.
6. Supply & Demand: Pemetaan zona Supply dan Demand aktif terdekat.
7. BOS: Break of Structure konfirmasi kelanjutan tren searah.
8. Pin Bar: Deteksi candle penolakan sumbu panjang (rejection wick >= 50%) di zona POI.
9. CISD: Change in State of Delivery (peralihan algoritma pengiriman harga).
10. POI: Point of Interest utama (High-Probability Order Block yang terbentuk setelah liquidity sweep).

[KEPUTUSAN AKHIR]
- Status:
  * "ENTRY READY" : Setup matang, konfluensi SMC kuat, R:R >= 2.0, ada konfirmasi rejection/trigger.
  * "WATCHLIST"   : Menjanjikan tapi harga belum mencapai POI atau butuh konfirmasi LTF.
  * "REJECT"      : Kriteria tidak terpenuhi, R:R jelek, atau trap likuiditas.
- Skor SMC (0 - 100): Berikan skor institusional objektif.

Output HARUS berupa JSON murni tanpa teks pengantar:
{{
  "smc_elements": {{
    "order_blocks": "deskripsi OB bullish dan bearish",
    "liquidity_sweeps_runs": "deskripsi sweep BSL/SSL dan liquidity run",
    "liquidity_range": "swing high, swing low, EQ, dan discount/premium",
    "choch": "status change of character",
    "bos": "status break of structure",
    "inducement": "level IDM dan status mitigasi",
    "supply_demand": "zona demand dan supply aktif",
    "pin_bar": "analisis rejection wick candle",
    "cisd": "change in state of delivery",
    "poi": "Point of Interest utama setelah sweep"
  }},
  "decision": "APPROVED" | "CONDITIONAL" | "REJECTED",
  "status": "ENTRY READY" | "WATCHLIST" | "REJECT",
  "smc_score": (angka integer 0-100),
  "direction": "LONG" | "SHORT" | "NONE",
  "entry_zone": "level atau rentang harga entry optimal",
  "sl": (angka float Stop Loss),
  "tp1": (angka float Target 1),
  "tp2": (angka float Target 2),
  "rr": (angka float rasio R:R ke TP2),
  "detailed_analysis": "ulasan institusional komprehensif mengapa setup ini diambil/ditolak",
  "action_plan": "instruksi taktis langkah demi langkah untuk trader"
}}"""
    return prompt


def clean_json_text(text: str) -> str:
    """Membersihkan markdown backtick ```json ... ``` dari respons LLM."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def call_groq_api(prompt: str, model: Optional[str] = None, api_key: Optional[str] = None) -> Optional[dict]:
    """Mengirim permintaan ke Groq Cloud API dengan penanganan kuota dan fallback model otomatis."""
    key = api_key or get_groq_key()
    if not key:
        return None

    primary_model = model or get_default_model()
    fallback_model = get_fallback_model()
    models_to_try = [primary_model]
    if fallback_model and fallback_model != primary_model:
        models_to_try.append(fallback_model)

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
                    "content": "Anda adalah Master Trader Institusional dan Kepala Analis SMC. Selalu berikan output murni JSON sesuai format yang diminta.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 1500,
        }
        for attempt in range(2):
            try:
                res = requests.post(GROQ_ENDPOINT, headers=headers, json=payload, timeout=20)
                if res.status_code == 200:
                    data = res.json()
                    choices = data.get("choices", [])
                    if choices:
                        content = choices[0].get("message", {}).get("content", "")
                        cleaned = clean_json_text(content)
                        parsed = json.loads(cleaned)
                        parsed["model"] = mdl
                        return parsed
                elif res.status_code == 429:
                    time.sleep(2.0)
                    continue
                else:
                    print(f"[AI Validator] Groq model {mdl} error {res.status_code}: {res.text[:150]}", file=sys.stderr)
            except Exception as e:
                print(f"[AI Validator] Permintaan ke Groq ({mdl}) gagal: {e}", file=sys.stderr)

    return None


def audit_smc_setup(setup: Any, btc_ctx: Optional[dict] = None, model: Optional[str] = None) -> dict:
    """Mengaudit satu setup SMC secara mandiri menggunakan Full LLM."""
    key = get_groq_key()
    if not key:
        return {
            "decision": "NO_API_KEY",
            "smc_score": 0,
            "status": "REJECT",
            "smc_elements": {},
            "detailed_analysis": "GROQ_API_KEY tidak ditemukan di .env.",
            "action_plan": "Isi GROQ_API_KEY di file .env untuk mengaktifkan AI Evaluator.",
            "model": "",
        }

    prompt = build_full_llm_smc_prompt(setup, btc_ctx)
    result = call_groq_api(prompt, model=model, api_key=key)
    if result:
        decision = str(result.get("decision", "CONDITIONAL")).upper()
        if decision not in ("APPROVED", "CONDITIONAL", "REJECTED"):
            decision = "CONDITIONAL"

        try:
            score = int(float(result.get("smc_score", 50)))
        except (ValueError, TypeError):
            score = 50

        status = str(result.get("status", "REJECT")).upper()
        if status not in ("ENTRY READY", "WATCHLIST", "REJECT"):
            status = "ENTRY READY" if score >= 70 and decision == "APPROVED" else ("WATCHLIST" if score >= 50 else "REJECT")

        smc_elements = result.get("smc_elements", {})
        if not isinstance(smc_elements, dict):
            smc_elements = {}

        return {
            "decision": decision,
            "smc_score": score,
            "status": status,
            "direction": str(result.get("direction", "")).upper(),
            "entry_zone": str(result.get("entry_zone", "")),
            "sl": result.get("sl"),
            "tp1": result.get("tp1"),
            "tp2": result.get("tp2"),
            "rr": result.get("rr"),
            "smc_elements": smc_elements,
            "detailed_analysis": str(result.get("detailed_analysis", result.get("reason", ""))).strip(),
            "action_plan": str(result.get("action_plan", "")).strip(),
            "model": str(result.get("model", "")),
        }

    return {
        "decision": "FAILED",
        "smc_score": 0,
        "status": "REJECT",
        "smc_elements": {},
        "detailed_analysis": "Evaluasi Full LLM tidak dapat diselesaikan (timeout / kuota habis).",
        "action_plan": "Periksa koneksi internet atau kuota Groq API.",
        "model": "",
    }


def audit_top_setups(
    results: List[Any],
    btc_ctx: Optional[dict] = None,
    max_audit: int = 15,
    workers: int = 2,
) -> None:
    """
    Melakukan evaluasi Full LLM serentak pada kandidat setup SMC.
    Hasil dari LLM menentukan 10 elemen kunci SMC dan secara langsung
    MENIMPA (OVERWRITE) skor, status, dan level dari rumus awal Python.
    """
    if not results or not get_groq_key():
        return

    # Ambil semua kandidat yang memiliki potensi struktur setup
    candidates = [r for r in results if getattr(r, "status", "") in ("ENTRY READY", "WATCHLIST") or getattr(r, "score", 0) >= 40]
    candidates.sort(key=lambda r: (0 if getattr(r, "status", "") == "ENTRY READY" else 1, -getattr(r, "score", 0.0)))
    targets = candidates[:max_audit]
    if not targets:
        return

    print(f"\n[AI Evaluator] Menyerahkan evaluasi kepada Full LLM Groq (SMC Head Trader) untuk {len(targets)} kandidat ...", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=min(workers, len(targets))) as pool:
        futures = {pool.submit(audit_smc_setup, s, btc_ctx): s for s in targets}
        for f in as_completed(futures):
            s = futures[f]
            try:
                res = f.result()
                score = res["smc_score"]
                status = res["status"]

                # Simpan 10 elemen kunci SMC dari AI
                s.smc_elements = res.get("smc_elements", {})

                # Timpa skor dan status dengan otoritas Full LLM
                s.score = float(score)
                s.status = status
                s.ai_decision = res["decision"]
                s.ai_confidence = float(score)
                s.ai_summary = res["detailed_analysis"]
                s.ai_action_note = res["action_plan"]
                s.ai_model = res["model"]

                # Jika AI memberikan arahan dan level spesifik, adopsi
                if res.get("direction") in ("LONG", "SHORT"):
                    s.direction = res["direction"]
                if res.get("entry_zone"):
                    s.entry_zone = res["entry_zone"]
                if res.get("sl") is not None:
                    try:
                        s.sl = float(res["sl"])
                    except (ValueError, TypeError):
                        pass
                if res.get("tp1") is not None:
                    try:
                        s.tp1 = float(res["tp1"])
                    except (ValueError, TypeError):
                        pass
                if res.get("tp2") is not None:
                    try:
                        s.tp2 = float(res["tp2"])
                    except (ValueError, TypeError):
                        pass
                if res.get("rr") is not None:
                    try:
                        s.rr = float(res["rr"])
                    except (ValueError, TypeError):
                        pass

                sym = getattr(s, "symbol", "")
                dec = s.ai_decision
                icon = "✅" if dec == "APPROVED" else ("⚠️" if dec == "CONDITIONAL" else "❌")
                print(f"  [Full LLM] {sym:12s} -> {icon} {dec:<11s} | Skor: {score:>3d} | Status: {status}", file=sys.stderr)
            except Exception as e:
                print(f"  [Full LLM] {getattr(s, 'symbol', '')} error: {e}", file=sys.stderr)
