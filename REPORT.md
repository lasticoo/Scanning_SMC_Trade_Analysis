# Laporan Implementasi Audit Inkremental (`fix/audit-develop2`)

Basis commit awal: `2eeb6fd` (`develop2`)  
Branch kerja: `fix/audit-develop2`  
Lingkungan pengujian:
- **Python**: `3.10.6 (tags/v3.10.6:9c7b4bd, Aug  1 2022, 21:53:49) [MSC v.1932 64 bit (AMD64)]`
- **pandas**: `2.3.3`
- **numpy**: `2.2.6`

---

## 1. Ringkasan Commit & Diff terhadap `2eeb6fd`

### Daftar Commit per Fase
| Fase | Commit Hash | Pesan Commit |
|---|---|---|
| **Fase 0** | `e03293a` | `test(audit): add Phase 0 synthetic data generator and baseline harness` |
| **Fase 1** | `b8a0b89` | `fix(smc): fix MSS key error, off-by-one mitigation, and visit-based touches` |
| **Fase 2** | `a91c3a2` | `feat(evidence): add objective evidence module for LLM curator` |
| **Fase 3** | `0d43b4d` | `perf(async): offload CPU bundle and derivative pass to threads with error_types tracking` |
| **Fase 4** | `1bbb603` | `feat(logging): log per-setup evidence and Python/LLM status to JSONL` |

### Output `git diff --stat 2eeb6fd..1bbb603`
```text
 ai_validator.py                             |   5 +-
 binance.py                                  | 156 ++++++++-
 scanner_core/evidence.py                    | 479 ++++++++++++++++++++++++++++
 scanner_core/mtf_smc.py                     |  67 ++--
 tools/audit/run_baseline.py                 | 127 ++++++++
 tools/audit/synth_data.py                   | 162 ++++++++++
 tools/audit/test_jsonl_logging.py           |  85 +++++
 tools/audit/test_neutrality_and_evidence.py | 141 ++++++++
 tools/audit/test_ob_mitigation.py           | 105 ++++++
 tools/audit/test_thread_vs_serial.py        | 192 +++++++++++
 10 files changed, 1484 insertions(+), 35 deletions(-)
```

---

## 2. Rincian Per Fase, Perintah Uji, Output Asli, dan Status Kriteria

### FASE 0 — Baseline
- **File yang ditambahkan**:
  - `tools/audit/synth_data.py`: Generator OHLCV sintetis 15M (`5000` bar per seed dengan *volatility clustering* / rezim GARCH-like, di-*resample* ke 1H dan 4H) lintas 6 seed (`11, 22, 33, 44, 55, 66`) $\times$ `47` jendela evaluasi = **282 evaluasi**.
  - `tools/audit/run_baseline.py`: Harness pemanggil `analyze_smc_symbol` untuk merekam metrik baseline (`status, score, entry, sl, tp1, tp2, archetype`) dan menghitung exception.
- **Perintah uji**:
  ```powershell
  python binance.py --help
  python tools/audit/run_baseline.py --save <scratch>/phase0_baseline.json
  ```
- **Output asli pada commit `2eeb6fd` (sebelum perbaikan Fase 1)**:
  ```text
  Total windows evaluated : 282
  Setups produced         : 265
  None returned           : 2
  Exceptions raised       : 15
  Total elapsed time      : 15299.9 ms (54.25 ms/call)
  Exception breakdown:
    KeyError: 'imp_low': 15
  Archetype breakdown     : {'HTF_REVERSAL': 113, 'TREND_CONT': 150, 'PRE_REACTION': 2}
  Status breakdown        : {'REJECT': 229, 'REVERSAL WATCH': 25, 'ENTRY READY': 6, 'WATCHLIST': 3, 'PRE-TAP LIMIT': 2}
  ```

---

### FASE 1 — Perbaikan Bug di `scanner_core/mtf_smc.py`
- **File & Fungsi yang diubah**:
  - `scanner_core/mtf_smc.py`:
    1. **1.1 (`analyze_smc_symbol`, jalur `HTF_REVERSAL`)**: Memperbaiki pembacaan key `mss_h1` sesuai key yang dihasilkan `extract_tf_structure`: untuk LONG (`d_rev == 1`) menggunakan `imp_lo = float(mss_h1["sweep_low"])` dan `imp_hi = float(mss_h1["impulse_high"])`; untuk SHORT (`d_rev == -1`) menggunakan `imp_hi = float(mss_h1["sweep_high"])` dan `imp_lo = float(mss_h1["impulse_low"])`.
    2. **1.2 (`extract_tf_structure`)**: Memperbaiki batas *off-by-one* dari `n - 1` menjadi `n` pada:
       - FVG Bullish & Bearish: guard `if i + 1 <= n - 1:` dengan slice `c[i + 1 : n]`, `l[i + 1 : n]`, `h[i + 1 : n]`.
       - Mitigasi Bullish OB & Bearish OB: loop `for j in range(end_d + 1, n):`.
       - Breaker Bullish & Bearish: guard `bj + 1 >= n` dan slice `c[bj + 1 : n]`.
       - EQH & EQL: `after_slice = h[s2["idx"] + 1 : n]` dan `l[s2["idx"] + 1 : n]`.
    3. **1.3 (`_count_visits` & `extract_tf_structure`)**: Menambahkan helper `_count_visits(inside: np.ndarray) -> int` yang menghitung episode kontak berurutan sebagai `1` kunjungan, diterapkan pada `touches` Bull/Bear OB dan `brk_touches` Breaker.
  - `tools/audit/test_ob_mitigation.py`: Unit test sintetis untuk memverifikasi mitigasi bar terakhir (`n - 1`) dan perhitungan kunjungan berurutan (`_count_visits`).
- **Perintah uji**:
  ```powershell
  python tools/audit/test_ob_mitigation.py
  python tools/audit/run_baseline.py --save <scratch>/phase1_baseline.json
  ```
- **Output asli**:
  ```text
  [PASS] _count_visits helper unit tests
  [PASS] OB tap on last closed bar (n-1) recorded as touches=1, fresh=False
  [PASS] OB with 4 consecutive bars in single visit kept with touches=1, fresh=False
  ```
  ```text
  Total windows evaluated : 282
  Setups produced         : 282
  None returned           : 0
  Exceptions raised       : 0
  Total elapsed time      : 10827.8 ms (38.40 ms/call)
  Archetype breakdown     : {'HTF_REVERSAL': 166, 'TREND_CONT': 115, 'PRE_REACTION': 1}
  Status breakdown        : {'REJECT': 243, 'REVERSAL WATCH': 33, 'ENTRY READY': 4, 'PRE-TAP LIMIT': 1, 'WATCHLIST': 1}
  ```
- **Status Kriteria Fase 1**:
  - Exception dari data acak turun dari `15` (`> 0`) menjadi `0`: **LULUS**
  - Tap OB pada bar close terakhir tercatat (`touches=1, fresh=False`) dan reaksi 4 bar berturut-turut dalam 1 kunjungan tetap `touches=1`: **LULUS**
  - Tidak ada perubahan perilaku di luar tiga poin ini: **LULUS**

---

### FASE 2 — Modul Bukti Objektif untuk LLM (`scanner_core/evidence.py`)
- **File & Fungsi yang diubah**:
  - `scanner_core/evidence.py`: Modul baru (hanya bergantung stdlib + `numpy` + `pandas`) berisi `build_evidence(setup, df4, df1, df15, btc) -> dict` yang menghitung struktur kausal 4H/1H/15M (state machine 1x pakai), sweep searah + MSS/BOS sesudahnya, posisi range swing 1H, kunjungan POI 24 jam + umur kunjungan terakhir, Efficiency Ratio 1H(24b)/15M(32b), Killzone UTC, tren BTC 4H/1H, pool likuiditas target + RR + jarak TP2 dalam ATR1H, metrik khusus Reversal/Sniper/Pre-Tap, `flags` netral, dan blok `text` siap-prompt.
  - `scanner_core/mtf_smc.py`: Menambahkan `_safe_evidence(setup, df4, df1, df15, btc)` dan menyisipkan `best_setup.raw_market_data["evidence"] = _safe_evidence(...)` di akhir `analyze_smc_symbol` setelah `populate_forward_playbook`.
  - `ai_validator.py`: Menyisipkan `raw["evidence"]["text"]` tepat setelah baris `- Orderflow: ... | Funding: ...` di `build_full_llm_smc_prompt` tanpa mengubah bagian prompt lain.
  - `tools/audit/test_neutrality_and_evidence.py`: Skrip verifikasi netralitas Fase 1 vs Fase 2, error count, benchmark latensi CPU `build_evidence`, dan pemeriksaan prompt ketiga archetype.
- **Perintah uji**:
  ```powershell
  python tools/audit/test_neutrality_and_evidence.py --phase1-json <scratch>/phase1_baseline.json --save-prompts <scratch>/prompt_examples.json
  ```
- **Output asli**:
  ```text
  === FASE 2 VERIFICATION SUMMARY ===
  Neutrality check        : 282/282 identical (mismatches=0)
  Evidence error count    : 0/282
  build_evidence CPU time : mean=2.586 ms | median=1.659 ms | p95=6.015 ms | max=28.014 ms
  Archetypes captured     : ['HTF_REVERSAL', 'TREND_CONT', 'PRE_REACTION']
  [PASS] Prompt for HTF_REVERSAL contains '[BUKTI KUALITAS TAMBAHAN DARI PYTHON'
  [PASS] Prompt for TREND_CONT contains '[BUKTI KUALITAS TAMBAHAN DARI PYTHON'
  [PASS] Prompt for PRE_REACTION contains '[BUKTI KUALITAS TAMBAHAN DARI PYTHON'
  ```
- **Status Kriteria Fase 2**:
  - Netralitas Fase 1 vs Fase 2 pada `(status, score, entry, sl, tp1, tp2, archetype)`: **282/282 (100% identik) — LULUS**
  - Semua setup memiliki `evidence` dengan `error is None`: **0/282 error — LULUS**
  - Prompt ketiga archetype memuat `[BUKTI KUALITAS TAMBAHAN DARI PYTHON`: **LULUS**
  - Tambahan CPU per setup `< 5 ms`: **Rata-rata 2.586 ms (median 1.659 ms) — LULUS**
  - `evidence.py` hanya bergantung pada `numpy`/`pandas`/stdlib: **LULUS**

---

### FASE 3 — Scaling Ringan Async (`binance.py`)
- **File & Fungsi yang diubah**:
  - `binance.py`:
    - Menambahkan fungsi sinkron `_cpu_bundle(sym, df4, df1, df15, meta, cfg, btc, deriv=None, st_prev=None)` yang membungkus `htf_context`, `_eval_strategy`, dan `_compute_symbol_radar`.
    - Di `_scan_universe_async` (`worker_one`), memanggil `await asyncio.to_thread(_cpu_bundle, sym, df4, df1, df15, meta, cfg, btc)`. Penulisan ke `cached_bundles` dan `radar_map` tetap berada di event loop utama di luar thread.
    - Pada pass kedua (re-evaluasi derivatif), menjalankan `_eval_strategy` paralel via `asyncio.to_thread` di dalam `await asyncio.gather(..., return_exceptions=True)` sehingga kegagalan satu simbol tidak menjatuhkan simbol lain dan dihitung sebagai error.
    - Menambahkan `error_types: Counter` ke dalam dict `stats` (`stats["error_types"]`), mencetak nama class exception pada pesan error per simbol serta ringkasan jenis error, dan menampilkannya secara kompatibel di `build_report`.
    - Jalur `--sync` (`_scan_universe_sync`) tidak diubah.
  - `tools/audit/test_thread_vs_serial.py`: Pengujian kesamaan hasil eksekusi via thread vs serial serta isolasi kegagalan Pass 1 dan Pass 2.
- **Pemeriksaan State Global di Thread**:
  - `htf_context`, `_eval_strategy` (`analyze_smc_symbol` / `analyze_symbol`), dan `_compute_symbol_radar` (`extract_tf_structure` / `scan_live_zone_radar`) hanya membaca DataFrame/dict input dan konstanta (`MAJOR_SYMBOLS`, `NAN`) serta mengembalikan objek `Setup` dan `ZoneRadarHit` baru tanpa memutasi state global apa pun. Mutasi `cached_bundles` dan `radar_map` dilakukan di event loop sesudah `await asyncio.to_thread(...)` selesai.
- **Perintah uji**:
  ```powershell
  python tools/audit/test_thread_vs_serial.py
  ```
- **Output asli**:
  ```text
  [2024-11-15 05:53:20 UTC] mengambil daftar kontrak (AsyncIO) ...
    10 kontrak aktif, 10 lolos likuiditas, memindai paralel 10 simbol ...
    mengambil data Open Interest & Long/Short Ratio untuk 10 kandidat ...
  [2024-11-15 05:53:20 UTC] mengambil daftar kontrak (AsyncIO) ...
    6 kontrak aktif, 6 lolos likuiditas, memindai paralel 6 simbol ...
    [TEST02USDT] gagal (ValueError): Simulated Pass-1 kline/eval corruption
    mengambil data Open Interest & Long/Short Ratio untuk 5 kandidat ...
    [TEST04USDT] gagal (RuntimeError): Simulated Pass-2 derivative re-eval failure
    ringkasan jenis error (2 total): ValueError=1, RuntimeError=1
  [PASS] Thread vs Serial 100% identical across 10 symbols (setups=10, radar_hits=10)
  [PASS] Fault isolation & error_types verified: stats={'total': 6, 'liquid': 6, 'scanned': 4, 'errors': 2, 'error_types': {'ValueError': 1, 'RuntimeError': 1}}
  ```
- **Status Kriteria Fase 3**: **LULUS**

---

### FASE 4 — Pencatatan JSONL per Setup (`binance.py`)
- **File & Fungsi yang diubah**:
  - `binance.py`:
    - Menambahkan `append_setups_jsonl(results, scan_ts, out_dir, py_status_map=None, filename="setups_audit.jsonl")` yang mencatat waktu (`scan_utc`, `signal_candle_utc`, `last_closed_candle_utc`), kunci penghubung tracker (`tracker_key`), `symbol`, `direction`, `archetype`, `pattern`, `python_status`, `python_score`, `status`, `score`, `llm_status`, `llm_score`, `ai_decision`, `ai_model`, `levels` (`entry`, `sl`, `tp1`, `tp2`, `tp3`, `rr`, `risk_pct`), `evidence` (seluruh metrik numerik & struktural), serta `flags`.
    - Menyimpan `py_status_map` sebelum `ai_validator.audit_top_setups` dijalankan di `run_scan` agar `python_status` asli tidak tertimpa ketika LLM mengubah `s.status`.
  - `tools/audit/test_jsonl_logging.py`: Unit test verifikasi penulisan dan skema `setups_audit.jsonl`.
- **Perintah uji**:
  ```powershell
  python tools/audit/test_jsonl_logging.py
  ```
- **Output asli**:
  ```text
  [PASS] JSONL logging verified (3 records, keys=['scan_utc', 'signal_candle_utc', 'last_closed_candle_utc', 'tracker_key', 'symbol', 'direction', 'archetype', 'pattern', 'python_status', 'python_score', 'status', 'score', 'llm_status', 'llm_score', 'ai_decision', 'ai_model', 'levels', 'evidence', 'flags'])
  ```
- **Status Kriteria Fase 4**: **LULUS**

---

## 3. Penyimpangan dari Spesifikasi

Tidak ada. (Catatan: file referensi opsional `smc_update_develop2.patch` tidak tersedia di direktori kerja, sehingga seluruh fase diimplementasikan langsung di atas commit `2eeb6fd` sesuai spesifikasi).

---

## 4. Hal yang Tidak Bisa Diuji di Lingkungan Ini

1. **Pemindaian Live ke Endpoint REST/WebSocket Binance (`AsyncBinance` / `watch_setups_realtime`)**: Sesuai aturan mutlak ke-4, pemindaian live tidak dijalankan dan `.env` tidak disentuh; pengujian jalur async dilakukan menggunakan mock `AsyncBinance` deterministik di `tools/audit/test_thread_vs_serial.py`.
2. **Respons Kurasi Live dari API Groq LLM (`audit_top_setups`)**: Tidak memanggil API eksternal Groq; pengujian integrasi dilakukan dengan memverifikasi output `build_full_llm_smc_prompt` secara langsung untuk ketiga archetype dan mensimulasikan hasil kurasi LLM pada pengujian JSONL.

---

## 5. Temuan Baru / Risiko yang Terlihat Selama Audit (Dilaporkan Tanpa Diubah)

1. **Tracking `.pyc` di dalam Git (`scanner_core/__pycache__/*.pyc`)**: Beberapa file bytecode `.pyc` di `scanner_core/__pycache__/` terlacak di dalam repositori pada branch `develop2`, sehingga setiap kompilasi ulang Python memicu status *modified* pada file `.pyc` jika tidak diabaikan via `git rm --cached` + `.gitignore`.
2. **Pemotongan Candle Terakhir di `htf_context` (`scanner_core/indicators.py`)**: Fungsi `htf_context` memotong bar terakhir (`df = df.iloc[:-1]` jika `len(df) > 20`) dengan asumsi bar terakhir masih berjalan (*unclosed*), sedangkan `AsyncBinance.klines` sudah memfilter hanya candle yang `close_time <= server_ms` (sudah *closed*). Akibatnya, `htf_context` tertinggal 1 bar *closed* pada 4H dan 1H.
3. **Mutasi `s.status` dan `s.score` secara *in-place* oleh `ai_validator.audit_top_setups`**: Sebelum Fase 4 menyimpan `py_status_map`, status asli dari mesin Python langsung ditimpa oleh `s.status = res["status"]` tanpa menyimpan atribut `s.python_status` pada dataclass `Setup` (berbeda dengan `s.python_score` yang sudah ada di `Setup`).

---

## 6. Contoh Prompt LLM Lengkap per Archetype (`build_full_llm_smc_prompt`)

### 6.1 Archetype `HTF_REVERSAL` (`SYN_11_00`)
```text
Anda adalah HAKIM & KURATOR PENILAIAN AKHIR 3 Pilar Paten SMC/ICT Modern Crypto Futures (4H->1H->15M & Orderflow).
Python bertugas sebagai tracker, penemu struktur objektif, dan pengukur geometri. Anda (API AI) yang memegang keputusan penilaian akhir (decision, status, smc_score, playbook).

[PRINSIP MUTLAK TRADER — WAJIB DIIKUTI]
1. TANPA PAKSAAN LIMIT (PREFER REJECT): Dilarang mengarang/menggeser limit entry jauh hanya demi memaksakan R:R! Jika secara trade range, penghalang Supply/Demand 4H lawan, atau R:R efektif ke TP2 tidak mencapai minimal 1:3.0, WAJIB putuskan "decision": "REJECTED" dan "status": "REJECT".
2. MINIMAL R:R 1:3.0 & PANTANG GESER SL KE BEP: Target Utama TP2 wajib memiliki R:R efektif >= 1:3.0. DILARANG KERAS menyarankan geser SL ke BEP / Break-Even dalam kondisi apa pun (Fixed SL Set-and-Forget di balik low/high sweep).

[SETUP SYN11USDT] Harga: 35.4750 | Sesi: London Close / NY PM (16-20 UTC) | BTC 4H: Bullish (Referensi makro saja; altcoin dinilai 100% independen dari struktur koin sendiri)
- Archetype: HTF_REVERSAL (Reversal Watch (Tahap 1/5)) | SHORT (Reversal Watch SMC [Tahap 1/5: WATCH (Menunggu Sweep BSL)] (1H Bearish OB (Sweep POI))) | Taktik Awal: SKIP | Approach: KOMPRESI / KOREKTIF (Ideal Limit Pre-Tap) | Est. Sentuh: Sudah di area / detik ini
- Lokasi Live: ⚡ NEAR 1H OB (0.28%) (APPROACHING_POI | Jarak POI: 0.28% | Depth: Belum masuk zona) | Zona Aktif: DEKAT 1H Bearish OB (Sweep POI) [35.5740-36.1712] (Jarak 0.28%) ; DEKAT 4H Bearish OB (Sweep POI) [35.5913-37.8941] (Jarak 0.33%)
- Sweep: Belum ada sweep baru | Magnet EQH/EQL/IDM: EQH 15M @ 35.8975 (1.19%) | EQH 15M @ 35.9727 (1.40%) | IDM @ 35.2794 (0.55%) | Reaksi/CVD: MENDEKATI RETEST ZONA (Jarak 0.28% — pantau reaksi)
- [PROGRES REVERSAL WATCH BERTAHAP PYTHON] Status Tahap: Tahap 1/5: WATCH (Menunggu Sweep BSL) (Tahap 1/5) | Skor Objektif Python: 6.8/10 (68/100) | Langkah Berikutnya: Menunggu Sweep BSL (BSL 1H 35.3623) di POI 4H + Reclaim <=3 candle H1 -> MSS H1 -> CHoCH M15
- [RINCIAN 10 POIN SKOR OBJEKTIF PYTHON] 1. Posisi 4H (Discount/Premium): 10.0/10 | 2. POI 4H Fresh & BOS: 5.5/10 | 3. Sweep SSL/BSL & Reclaim <=3H1: 6.5/10 | 4. MSS H1 + Displacement: 4.0/10 | 5. FVG H1 Imbalance: 4.0/10 | 6. Trigger CHoCH M15 di POI: 8.0/10 | 7. Konfluensi OB + FVG + OTE: 8.0/10 | 8. Struktur BTC & Rel. Strength: 7.5/10 | 9. Derivatif & Vol Spike Sweep: 5.0/10 | 10. RR Efektif >=1:3 Tanpa Halangan: 10.0/10
[4H MAKRO] Tren: Bullish (struktur naik) | ATR4H: 1.45093 | Range4H: [28.3684 - 36.1712] (Posisi Makro: 91.1%) | Tembok4H: Batas Bawah: Demand 4H [29.3627-31.2235] | Batas Atas: Supply 4H [35.5913-37.8941] | OB4H: 4H Bearish OB (Sweep POI) [35.5913 - 37.8941]; 4H Bearish OB [35.8029 - 37.0251]; 4H Bullish Breaker Block [12.9963 - 13.4416] (Unmitigated)
[1H RANGE] Tren: Bullish (struktur naik) | ATR1H: 0.714399 | Range1H: 31.5665-36.1712 (EQ 33.8688, Posisi Premium 1H (84.9%)) | Hold: Intraday (~4-10 jam)
- OB1H: 1H Bullish OB (Sweep POI) [34.0552 - 34.7280] (SweepPOI); 1H Bullish OB [34.5062 - 35.2888] (Unmitigated); 1H Bearish OB (Sweep POI) [35.5740 - 36.1712] (SweepPOI) | FVG1H: Bull FVG [33.8435 - 33.9530] | EQH1H[-] EQL1H[33.9530, 31.5665]
[15M & ORDERFLOW] ATR15M: 0.375071 | Struktur15M: SSL Sweep (34.2149), BSL Sweep (35.2788), Bearish CHOCH (34.7774), BSL Sweep (35.2888), Bullish CHOCH (35.2888), Bearish CISD | IDM: Level 35.2794 (Belum disapu — umpan likuiditas) | PinBar: Bear PinBar (wick 59%), Bear PinBar (wick 60%), Bull PinBar (wick 68%)
- OB15M: 15M Bull OB [34.2907 - 34.7576]; 15M Bull OB [34.5497 - 34.7093]; 15M Bull OB [34.5037 - 34.8299] (SweepPOI, Unmitigated); 15M Bullish Breaker Block [34.7774 - 35.1055] | FVG15M: Tidak ada FVG unmitigated | EQH15M[35.8975, 35.9727] EQL15M[34.5037, 34.1387, 34.0552]
- Orderflow: Netral (data delta tidak tersedia) | OI 1H +2.40% (4H +5.10%) | TopTrader L/S 1.25 | Retail L/S 0.85 | Funding: -0.0100%
[BUKTI KUALITAS TAMBAHAN DARI PYTHON — FAKTA OBJEKTIF]
- Struktur Kausal (1x Pakai): 4H=BULLISH [CHOCH_BULL@33.3878(34b), SSL_SWEEP@30.1393(25b), BOS_BULL@33.8843(20b), BOS_BULL@35.0789(6b)] | 1H=BULLISH [BSL_SWEEP@32.7979(45b), BSL_SWEEP@33.4045(38b), BOS_BULL@35.3623(23b), BOS_BULL@35.4012(1b)] | 15M=BEARISH [SSL_SWEEP@34.2149(23b), BSL_SWEEP@35.2788(11b), BOS_BEAR@34.7774(10b), BSL_SWEEP@35.2888(6b)]
- Sweep Searah Terakhir: 1H=Ada (38b lalu @ 33.4045, MSS/BOS sesudahnya: BELUM) | 15M=Ada (6b lalu @ 35.2888, MSS/BOS sesudahnya: BELUM)
- Posisi Swing 1H: 88.4% (PREMIUM) | Kunjungan POI 24j: 6x (terakhir: 2b lalu)
- Efficiency Ratio: 1H(24b)=0.08, 15M(32b)=0.12 | Killzone: OFF_KILLZONE (17:00 UTC) | BTC: 4H=BULLISH, 1H=BEARISH
- Pool Likuiditas Target: 1H@34.1387(RR 1.5), 1H@34.0552(RR 1.6), 1H@33.9530(RR 1.8), 1H@31.9189(RR 4.1) | Jarak TP2 ke Pool Terdekat: 0.48xATR1H
- Khusus Reversal: Impuls dari Sweep=2.29xATR1H | Retrace=58.2% | Lewat Ekstrem Sweep (Gagal)=TIDAK
- Flags Netral: CAUSAL_4H_OPPOSITE(BULLISH), POI_MULTIPLE_VISITS_24H(6x)

[LEVEL & FIXED LIQUIDITY-PROTECTED SL (NO BEP)] Entry: 35.4750 (Batas Sweep-Entry: 36.1712) (POI: 35.5740 - 36.1455 | Di POI)
- SL Tetap: 36.3426 | Proteksi SL di balik: SL diamankan di balik Swing High 1H (36.1712) — tahan Judas Swing | Checkpoint TP1: 34.1387 | Target Utama TP2: 32.2698 (RR 1:3.7, WAJIB >= 1:3.0) | TP3 Runner: 28.5339 (RR 1:8.0) | Tembok Lawan: 31.2235 | Invalidasi: Close 1H/15M menembus atas SL struktural 36.3426 (di atas zona POI 36.1455).

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
{
  "smc_elements": {
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
  },
  "decision": "APPROVED" | "CONDITIONAL" | "REJECTED",
  "status": "ENTRY READY" | "PRE-TAP LIMIT" | "REVERSAL WATCH" | "WATCHLIST" | "REJECT",
  "execution_style": "ENTRY_NOW" | "LIMIT_PRE_TAP" | "WAIT_SWEEP_TRIGGER" | "LIMIT_RETEST" | "SKIP",
  "smc_score": 80,
  "direction": "SHORT",
  "entry_zone": "35.5740 - 36.1455",
  "sl": 36.342622101935845,
  "tp1": 34.13868117742927,
  "tp2": 32.26976169099953,
  "rr": 3.69,
  "detailed_analysis": "ulasan padat penilaian akhir AI atas tahapan setup, struktur, & kelayakan R:R >=1:3 tanpa limit paksaan",
  "forward_playbook": "Langkah Utama (If-Then): kondisi tahap sekarang, pemicu spesifik menuju tahap berikutnya/entry, tahan SL tetap di balik sweep (tanpa geser BEP), & kawal ke TP2/TP3 >=1:3",
  "flip_scenario": "Syarat Pembatalan (Cancel/Flip): level close H1/4H di luar sweep/POI, batas waktu 20 bar H1 MSS, atau skenario balik arah jika invalid"
}
```

### 6.2 Archetype `TREND_CONT` (`SYN_11_03`)
```text
Anda adalah HAKIM & KURATOR PENILAIAN AKHIR 3 Pilar Paten SMC/ICT Modern Crypto Futures (4H->1H->15M & Orderflow).
Python bertugas sebagai tracker, penemu struktur objektif, dan pengukur geometri. Anda (API AI) yang memegang keputusan penilaian akhir (decision, status, smc_score, playbook).

[PRINSIP MUTLAK TRADER — WAJIB DIIKUTI]
1. TANPA PAKSAAN LIMIT (PREFER REJECT): Dilarang mengarang/menggeser limit entry jauh hanya demi memaksakan R:R! Jika secara trade range, penghalang Supply/Demand 4H lawan, atau R:R efektif ke TP2 tidak mencapai minimal 1:3.0, WAJIB putuskan "decision": "REJECTED" dan "status": "REJECT".
2. MINIMAL R:R 1:3.0 & PANTANG GESER SL KE BEP: Target Utama TP2 wajib memiliki R:R efektif >= 1:3.0. DILARANG KERAS menyarankan geser SL ke BEP / Break-Even dalam kondisi apa pun (Fixed SL Set-and-Forget di balik low/high sweep).

[SETUP SYN11USDT] Harga: 32.0749 | Sesi: NY Open Killzone (12-16 UTC) | BTC 4H: Bullish (Referensi makro saja; altcoin dinilai 100% independen dari struktur koin sendiri)
- Archetype: TREND_CONT (Follow-Trend (Post-Reaction)) | SHORT (MTF Nested Supply (1H + 15M POI Alignment)) | Taktik Awal: SKIP | Approach: KOMPRESI / KOREKTIF (Ideal Limit Pre-Tap) | Est. Sentuh: ~2.1 jam
- Lokasi Live: 🏃 POST-BOUNCE (+2.25%) (POST_BOUNCE_WAIT | Jarak POI: 2.25% | Depth: 85%) | Zona Aktif: WICK-TAP 15M Bearish OB [32.2737-32.5702] (Depth 85%) ; DEKAT 15M Bearish OB (Sweep POI) [32.3483-32.8303] (Jarak 0.85%)
- Sweep: IDM Taken (32.4895, 1b lalu) | Magnet EQH/EQL/IDM: - | Reaksi/CVD: SUDAH BEREAKSI & MEMANTUL (+2.25% dari POI — tunggu Limit Retest, jangan kejar)
- [TEMUAN RULES & SKOR OBJEKTIF PYTHON] Skor Awal Python: 63/100 | Rincian Komponen: Konteks HTF & Tren: 20.0/20 | Kualitas POI & Zone: 18.0/25 | Struktur & Liquidity: 12.0/25 | Trigger & Orderflow: 7.9/15 | Risk to Reward: 0.0/10 | Likuiditas Pasar: 5.0/5
[4H MAKRO] Tren: Sideways (struktur turun) | ATR4H: 1.41170 | Range4H: [28.3684 - 36.6215] (Posisi Makro: 44.9%) | Tembok4H: Batas Bawah: Demand 4H [29.3627-31.2235] | Batas Atas: Supply 4H [33.3914-34.8534] | OB4H: 4H Bearish OB [33.3914 - 34.8534]; 4H Bearish OB [34.3468 - 35.4332] (Unmitigated); 4H Bullish Breaker Block [12.9963 - 13.4416] (Unmitigated)
[1H RANGE] Tren: Bearish (struktur turun) | ATR1H: 0.678851 | Range1H: 31.8807-36.6215 (EQ 34.2511, Posisi Discount 1H (4.1%)) | Hold: Scalp-Intraday (~2-3 jam)
- OB1H: 1H Bearish OB [32.6142 - 33.0337] (Unmitigated); 1H Bearish Breaker Block [34.5767 - 35.2970] (SweepPOI); 1H Bearish Breaker Block [33.8415 - 34.8609] | FVG1H: Bear FVG [33.8728 - 34.4248]; Bear FVG [33.0337 - 33.1955] | EQH1H[35.3623, 35.5146] EQL1H[-]
[15M & ORDERFLOW] ATR15M: 0.329388 | Struktur15M: Bearish CHOCH (33.0238), SSL Sweep (32.6310), Bearish BOS (32.6310), SSL Sweep (32.1242), Bearish BOS (32.1242), Bearish CISD | IDM: Level 32.4895 (SUDAH DISAPU / IDM Taken) | PinBar: Bull PinBar (wick 57%), Bear PinBar (wick 68%), Bear PinBar (wick 75%)
- OB15M: 15M Bear OB [32.6142 - 32.8256]; 15M Bear OB [32.2737 - 32.5702]; 15M Bear OB [32.3483 - 32.8303] (SweepPOI); 15M Bearish Breaker Block [34.3179 - 34.7824]; 15M Bearish Breaker Block [34.5419 - 34.7842] (Unmitigated) | FVG15M: Bear FVG [34.1103 - 34.2266]; Bear FVG [33.2378 - 33.2939]; Bear FVG [33.0337 - 33.1125] | EQH15M[33.5748, 33.6990, 33.8728] EQL15M[-]
- Orderflow: Netral (data delta tidak tersedia) | OI 1H +2.40% (4H +5.10%) | TopTrader L/S 1.25 | Retail L/S 1.65 | Funding: +0.0150%
[BUKTI KUALITAS TAMBAHAN DARI PYTHON — FAKTA OBJEKTIF]
- Struktur Kausal (1x Pakai): 4H=BULLISH [BOS_BULL@33.8843(31b), BOS_BULL@35.0789(17b), BSL_SWEEP@36.1712(9b), SSL_SWEEP@34.0552(6b)] | 1H=BEARISH [BSL_SWEEP@36.1446(36b), SSL_SWEEP@34.5767(29b), SSL_SWEEP@33.3914(13b), CHOCH_BEAR@32.8848(4b)] | 15M=BEARISH [BSL_SWEEP@33.5327(20b), CHOCH_BEAR@33.0238(18b), SSL_SWEEP@32.6310(12b), SSL_SWEEP@32.1242(2b)]
- Sweep Searah Terakhir: 1H=Ada (36b lalu @ 36.1446, MSS/BOS sesudahnya: YA) | 15M=Ada (20b lalu @ 33.5327, MSS/BOS sesudahnya: YA)
- Posisi Swing 1H: 0.0% (DISCOUNT) | Kunjungan POI 24j: 5x (terakhir: 5b lalu)
- Efficiency Ratio: 1H(24b)=0.43, 15M(32b)=0.30 | Killzone: NY_KILLZONE (13:00 UTC) | BTC: 4H=BULLISH, 1H=BEARISH
- Pool Likuiditas Target: 4H@30.1649(RR 1.5), 4H@30.1393(RR 1.5), 4H@28.9470(RR 2.5), 4H@28.3684(RR 3.0) | Jarak TP2 ke Pool Terdekat: 1.56xATR1H
- Khusus Sniper: Pullback dari Leg 1H Terakhir=23.8%
- Flags Netral: CAUSAL_4H_OPPOSITE(BULLISH), 1H_POS_DISCOUNT(0%), POI_MULTIPLE_VISITS_24H(5x), TP2_FAR_FROM_SWING_POOL(1.56xATR1H)

[LEVEL & FIXED LIQUIDITY-PROTECTED SL (NO BEP)] Entry: 32.0749 (Batas Sweep-Entry: 33.0337) (POI: 32.7968 - 32.9893 | Menunggu Sentuhan)
- SL Tetap: 33.3313 | Proteksi SL di balik: SL diamankan di balik Swing High 15M (33.0337) — tahan Judas Swing | Checkpoint TP1: 31.4216 | Target Utama TP2: 31.2574 (RR 1:0.7, WAJIB >= 1:3.0) | TP3 Runner: 28.3684 (RR 1:3.0) | Tembok Lawan: 31.2235 | Invalidasi: Close 1H/15M menembus atas SL struktural 33.3313 (di atas zona POI 32.9893).

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
{
  "smc_elements": {
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
  },
  "decision": "APPROVED" | "CONDITIONAL" | "REJECTED",
  "status": "ENTRY READY" | "PRE-TAP LIMIT" | "REVERSAL WATCH" | "WATCHLIST" | "REJECT",
  "execution_style": "ENTRY_NOW" | "LIMIT_PRE_TAP" | "WAIT_SWEEP_TRIGGER" | "LIMIT_RETEST" | "SKIP",
  "smc_score": 80,
  "direction": "SHORT",
  "entry_zone": "32.7968 - 32.9893",
  "sl": 33.33125059714068,
  "tp1": 31.421557543732483,
  "tp2": 31.25742644112943,
  "rr": 0.65,
  "detailed_analysis": "ulasan padat penilaian akhir AI atas tahapan setup, struktur, & kelayakan R:R >=1:3 tanpa limit paksaan",
  "forward_playbook": "Langkah Utama (If-Then): kondisi tahap sekarang, pemicu spesifik menuju tahap berikutnya/entry, tahan SL tetap di balik sweep (tanpa geser BEP), & kawal ke TP2/TP3 >=1:3",
  "flip_scenario": "Syarat Pembatalan (Cancel/Flip): level close H1/4H di luar sweep/POI, batas waktu 20 bar H1 MSS, atau skenario balik arah jika invalid"
}
```

### 6.3 Archetype `PRE_REACTION` (`SYN_44_29`)
```text
Anda adalah HAKIM & KURATOR PENILAIAN AKHIR 3 Pilar Paten SMC/ICT Modern Crypto Futures (4H->1H->15M & Orderflow).
Python bertugas sebagai tracker, penemu struktur objektif, dan pengukur geometri. Anda (API AI) yang memegang keputusan penilaian akhir (decision, status, smc_score, playbook).

[PRINSIP MUTLAK TRADER — WAJIB DIIKUTI]
1. TANPA PAKSAAN LIMIT (PREFER REJECT): Dilarang mengarang/menggeser limit entry jauh hanya demi memaksakan R:R! Jika secara trade range, penghalang Supply/Demand 4H lawan, atau R:R efektif ke TP2 tidak mencapai minimal 1:3.0, WAJIB putuskan "decision": "REJECTED" dan "status": "REJECT".
2. MINIMAL R:R 1:3.0 & PANTANG GESER SL KE BEP: Target Utama TP2 wajib memiliki R:R efektif >= 1:3.0. DILARANG KERAS menyarankan geser SL ke BEP / Break-Even dalam kondisi apa pun (Fixed SL Set-and-Forget di balik low/high sweep).

[SETUP SYN44USDT] Harga: 41.7230 | Sesi: London Close / NY PM (16-20 UTC) | BTC 4H: Bearish (Referensi makro saja; altcoin dinilai 100% independen dari struktur koin sendiri)
- Archetype: PRE_REACTION (Pre-Reaction (Sebelum Bereaksi)) | LONG (Pre-Tap Fresh Demand Stalker (1H Unmitigated POI)) | Taktik Awal: LIMIT_PRE_TAP | Approach: KOMPRESI / KOREKTIF (Ideal Limit Pre-Tap) | Est. Sentuh: ~1.2 jam
- Lokasi Live: 🔭 STALK 4H OB (1.28%) (PRE_TAP_STALK | Jarak POI: 1.28% | Depth: 36%) | Zona Aktif: IN 4H Bullish OB [40.2303-41.9580] (Depth 36%) ; IN 4H Bullish Breaker Block [41.3156-42.2211] (Depth 97%) ; WICK-TAP 1H Bullish Breaker Block [41.1654-41.7138] (Depth 68%)
- Sweep: 1H SSL Sweep (41.5878, 0b lalu) | Magnet EQH/EQL/IDM: - | Reaksi/CVD: MENDEKATI ZONA SEGAR (Jarak 1.28% sebelum sentuhan | Gerak: KOMPRESI / KOREKTIF (Ideal Limit Pre-Tap))
- [TEMUAN RULES & SKOR OBJEKTIF PYTHON] Skor Awal Python: 81/100 | Rincian Komponen: Kesegaran & Hirarki POI: 26.0/30 | Kekuatan Displacement+FVG: 18.0/18 | Inducement / Magnet Depan: 8.0/18 | Dinamika Pendekatan Harga: 14.0/14 | Konteks HTF, RR & Flow: 14.9/20
[4H MAKRO] Tren: Bullish (struktur turun) | ATR4H: 1.90103 | Range4H: [33.6302 - 49.7138] (Posisi Makro: 50.3%) | Tembok4H: Batas Bawah: Demand 4H [39.0709-39.9034] | Batas Atas: Supply 4H [44.3505-46.2470] | OB4H: 4H Bearish OB [44.3505 - 46.2470] (Unmitigated); 4H Bullish Breaker Block [37.1586 - 38.5904] (Unmitigated); 4H Bullish Breaker Block [37.4960 - 38.5906] (Unmitigated)
[1H RANGE] Tren: Bearish (struktur turun) | ATR1H: 0.880200 | Range1H: 41.3394-46.3446 (EQ 43.8420, Posisi Discount 1H (7.7%)) | Hold: Intraday (~4-7 jam)
- OB1H: 1H Bearish OB [42.2181 - 43.0390] (Unmitigated); 1H Bearish Breaker Block [42.4235 - 45.1949]; 1H Bearish Breaker Block [42.7462 - 43.3142] | FVG1H: Bear FVG [46.7762 - 47.1242]; Bear FVG [44.9292 - 45.2659] | EQH1H[46.3446, 49.7138] EQL1H[39.0709, 35.9279]
[15M & ORDERFLOW] ATR15M: 0.412033 | Struktur15M: Bullish BOS (42.7134), Bearish CHOCH (42.2181), SSL Sweep (41.9007), Bearish BOS (41.9007), Bearish CISD, Bullish CISD | IDM: Level 42.0837 (Belum disapu — umpan likuiditas) | PinBar: Bull PinBar (wick 54%), Bear PinBar (wick 82%), Bear PinBar (wick 54%)
- OB15M: 15M Bear OB [42.4212 - 42.9476]; 15M Bear OB [42.1004 - 42.5604]; 15M Bear OB [41.7589 - 42.0837]; 15M Bearish Breaker Block [41.5878 - 42.2147]; 15M Bearish Breaker Block [42.2428 - 42.6755] | FVG15M: Bear FVG [43.9682 - 44.4217]; Bear FVG [42.1130 - 42.2068] | EQH15M[42.6744, 43.0398, 43.5512] EQL15M[-]
- Orderflow: Netral (data delta tidak tersedia) | OI 1H -1.20% (4H -2.50%) | TopTrader L/S 1.25 | Retail L/S 1.65 | Funding: +0.0150%
[BUKTI KUALITAS TAMBAHAN DARI PYTHON — FAKTA OBJEKTIF]
- Struktur Kausal (1x Pakai): 4H=BULLISH [CHOCH_BEAR@37.1229(43b), CHOCH_BULL@37.7467(27b), SSL_SWEEP@42.4770(4b), SSL_SWEEP@41.5878(0b)] | 1H=BEARISH [BSL_SWEEP@49.5789(67b), CHOCH_BEAR@47.0574(52b), BOS_BEAR@44.3505(25b), BOS_BEAR@41.7654(1b)] | 15M=BEARISH [SSL_SWEEP@42.0183(24b), BOS_BULL@42.7134(23b), CHOCH_BEAR@42.2181(12b), SSL_SWEEP@41.9007(7b)]
- Sweep Searah Terakhir: 1H=Ada (120b lalu @ 35.9658, MSS/BOS sesudahnya: YA) | 15M=Ada (7b lalu @ 41.9007, MSS/BOS sesudahnya: BELUM)
- Posisi Swing 1H: 24.3% (DISCOUNT) | Kunjungan POI 24j: 0x (terakhir: -)
- Efficiency Ratio: 1H(24b)=0.32, 15M(32b)=0.09 | Killzone: OFF_KILLZONE (17:00 UTC) | BTC: 4H=BEARISH, 1H=BEARISH
- Pool Likuiditas Target: 1H@43.0398(RR 2.1), 1H@43.9682(RR 3.1), 1H@46.2470(RR 5.5), 1H@46.3446(RR 5.6) | Jarak TP2 ke Pool Terdekat: 0.00xATR1H
- Khusus Pre-Tap: Kecepatan=0.12ATR15/bar | Range4Bar=1.63xATR15 | Vol(4/24)=0.54x | Likuiditas Antara Harga-POI=0
- Flags Netral: NONE

[LEVEL & FIXED LIQUIDITY-PROTECTED SL (NO BEP)] Entry: 41.0814 (Batas Sweep-Entry: 40.3229) (POI: 40.4829 - 41.1871 | Menunggu Sentuhan)
- SL Tetap: 40.1468 | Proteksi SL di balik: SL diamankan di balik Wick 1H OB (40.3229) — tahan Judas Swing | Checkpoint TP1: 43.0398 | Target Utama TP2: 43.9641 (RR 1:3.1, WAJIB >= 1:3.0) | TP3 Runner: 48.5585 (RR 1:8.0) | Tembok Lawan: 44.3505 | Invalidasi: Close 1H/15M menembus bawah SL struktural 40.1468 (di bawah zona POI 40.4829).

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
{
  "smc_elements": {
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
  },
  "decision": "APPROVED" | "CONDITIONAL" | "REJECTED",
  "status": "ENTRY READY" | "PRE-TAP LIMIT" | "REVERSAL WATCH" | "WATCHLIST" | "REJECT",
  "execution_style": "ENTRY_NOW" | "LIMIT_PRE_TAP" | "WAIT_SWEEP_TRIGGER" | "LIMIT_RETEST" | "SKIP",
  "smc_score": 80,
  "direction": "LONG",
  "entry_zone": "40.4829 - 41.1871",
  "sl": 40.146814060631186,
  "tp1": 43.03983044900351,
  "tp2": 43.96409325301749,
  "rr": 3.08,
  "detailed_analysis": "ulasan padat penilaian akhir AI atas tahapan setup, struktur, & kelayakan R:R >=1:3 tanpa limit paksaan",
  "forward_playbook": "Langkah Utama (If-Then): kondisi tahap sekarang, pemicu spesifik menuju tahap berikutnya/entry, tahan SL tetap di balik sweep (tanpa geser BEP), & kawal ke TP2/TP3 >=1:3",
  "flip_scenario": "Syarat Pembatalan (Cancel/Flip): level close H1/4H di luar sweep/POI, batas waktu 20 bar H1 MSS, atau skenario balik arah jika invalid"
}
```
