# Binance Futures 3-Pilar Paten Top-Down MTF SMC/ICT (`4H` → `1H` → `15M`), Liquidity-Protected Fixed SL, & Min 1:3 R:R Scanner v3.5

Scanner kuantitatif & institusional berbasis Python untuk mendeteksi setup trading berprobabilitas tinggi pada pasar **Binance USDⓈ-M Perpetual Futures** dengan pendekatan **Top-Down Multi-Timeframe Refinement (`4H` → `1H` → `15M`)**, terfokus pada **3 Pilar Paten SMC Crypto (*HTF Wall Reversal*, *Pre-Tap Limit OB Segar*, & *SMC Sniper Skor >= 75*)**, **Standar Wajib R:R Minimal `1:3.0` (`TP2 >= 3.0R` & `TP3 Runner 4.8R–8.0R`)**, **Liquidity-Protected Fixed SL (*Set-and-Forget* di balik kolam likuiditas tanpa geser SL ke BEP)**, **Hybrid Live Candle (`FIRST TOUCH`)**, **Orderflow Publik**, serta **Kurator SMC/ICT Kripto Modern berbasis Groq LLM**.

Hanya menggunakan **Public Endpoint & WebSocket Binance** (tanpa API key Binance, tanpa akses akun, dan aman).

---

```powershell
# 1. Scan standar Fokus 3 Pilar Paten (Reversal Watch + Pre-Tap Limit OB Segar + SMC Sniper >= 75) + Groq AI
python binance.py

# 2. Tampilkan HANYA koin yang sedang di/dekat POI, First Touch, Pre-Tap Segar, atau baru Liquidity Sweep
python binance.py --in-zone

# 3. Mode kilat (~4 detik) Radar Lokasi Harga Live (First Touch, In-POI, Pre-Tap HTF Segar, & Sweep)
python binance.py --radar

# 4. Scan super cepat murni mesin kuantitatif 3-Pilar Python (~5-7 detik untuk 45 koin)
python binance.py --no-ai --top 45

# 5. Aktifkan kembali seluruh setup lama (OB Retest/Breaker pasif di luar 3 Pilar Paten) jika diperlukan
python binance.py --all-setups

# 6. Scan lalu pantau live sentuhan zona Limit Pre-Tap, 50% CE, Reversal Sweep, & TP/SL via WebSocket
python binance.py --watch-ws

# 7. Mode otomatis tiap penutupan 15M + WebSocket POI Watcher + Notifikasi Telegram
python binance.py --loop --watch-ws --telegram

# 8. Evaluasi akurasi seluruh histori sinyal di folder scan_output/ (simulasi Fixed SL tanpa BEP & Target >= 1:3)
python binance.py --track --include-watchlist

# 9. Backtest Walk-Forward 14 hari terakhir (bebas look-ahead bias)
python binance.py --backtest --days 14 --bt-top 10
```

---

## 🚀 Fitur Utama v3.5 (3 Pilar Paten SMC Crypto + Min R:R `1:3.0` + Fixed Liquidity-Protected SL)

### 1. Spesialisasi Mesin pada "3 Pilar Paten SMC Crypto" (`mtf_smc.py`)
Menyaring ~70%–75% *noise* setup pasif (seperti *OB Retest* biasa, *Breaker* lemah, atau *Post-Bounce* yang sudah lari jauh) dan memusatkan seluruh kekuatan analisa & kuota token AI pada **3 Pilar yang terbukti memiliki *Win Rate* & *Profit Factor* tertinggi di pasar kripto**:
- **`[Pilar 1] HTF_REVERSAL` — Counter-Trend / Wall-Bounce Inversion (`REVERSAL WATCH` & `ENTRY READY`)**:
  Menangkap pembalikan arah (*reversal*) berprobabilitas tinggi ketika harga menghantam **Tembok Mayor `4H`/`1H`** di ekstrem *Dealing Range* disertai *Liquidity Climax* (`BSL`/`SSL` Sweep, `RSI` ekstrem, *CVD Absorption*, atau pergeseran struktur `15M CHOCH/CISD`).
- **`[Pilar 2] PRE_REACTION` — Pre-Tap Limit & First Touch di OB Segar (`PRE-TAP LIMIT`)**:
  Mengintai koin yang sedang mendekati zona **Order Block (`4H`/`1H` atau *Nested Sweep OB* `15M`) yang 100% segar (`0 touches` / *unmitigated*)** dalam jarak rapat (`<= 1.65x ATR 1H` atau `<= 2.60%`, `Skor >= 74`) sebelum harga bereaksi. Otomatis membuang *Pre-Tap* berbasis FVG kosong tanpa tameng Order Block.
- **`[Pilar 3] SMC_SNIPER (>= 75)` — Post-Sweep + CHOCH Continuation di Dekat POI**:
  Hanya meloloskan setup penerusan tren yang memenuhi syarat ketat **Liquidity Sweep + `CHOCH`/`CISD` + berada di dekat POI** dengan **`Skor >= 75`** (atau *First Touch* FVG segar di POI dengan `Skor >= 78`).

### 2. Standar Wajib R:R Minimal `1:3.0` & Manajemen *Set-and-Forget Fixed SL* (Tanpa Geser BEP)
Dirancang khusus untuk gaya *trader* SMC yang mengutamakan *asymmetric payoff* (`3R–8R`) tanpa terpotong prematur oleh koreksi/*retest* kripto:
- **Wajib R:R Minimal `1:3.0` (`min_rr = 3.0`)**: Target utama **`TP2` wajib memiliki R:R minimal `1:3.0` (`3.0R–4.8R`)** menuju struktur likuiditas `1H`/`4H` (`High/Low 1H`, `EQH/EQL 1H`, `Tembok 4H`). Jika ruang menuju tembok lawan `< 3.0R`, setup otomatis ditolak (`REJECT`).
- **Target `TP3 (Macro Runner 4.8R–8.0R)`**: Menyediakan target *runner* menuju ujung *Dealing Range* `4H` untuk menangkap pergerakan besar (*home-run* `5R–8R+`).
- **Liquidity-Protected Fixed SL (*Pantang Geser SL ke BEP*)**:
  - Mesin secara otomatis memindai kolam likuiditas terdekat di belakang POI (`EQL`/`EQH` `1H`/`15M`, `Swing Low/High` `1H`/`15M`, serta *wick* terluar OB `4H`/`1H` dalam radius `<= 1.60x ATR 1H`) dan menempatkan **SL struktural di balik kolam likuiditas tersebut** agar tahan terhadap sapuan *Judas Swing* / *stop-hunt*.
  - Menampilkan **`Batas Sweep-Entry`** (titik likuiditas di belakang POI) dan **`Proteksi SL di balik: ...`** pada laporan & CSV.
  - **Tanpa Geser SL ke BEP**: Baik *Forward Playbook* Python, Kurator AI Groq, maupun simulator `--track`/`--backtest` mengunci prinsip **SL tetap di tempat (*Set-and-Forget*)** hingga menyentuh target utama `>= 1:3` (`TP2`/`TP3`) atau *close* invalidasi SL (`TP1` hanya sebagai *checkpoint* struktur `1H`).

### 3. Kurator AI SMC/ICT Kripto Modern Hemat Token & Bebas *Rate-Limit* (`ai_validator.py`)
- **Smart TPM Load-Balancing Pool (`24,000 TPM`)**: Mendistribusikan beban kurasi secara otomatis lintas 3 model aktif Groq (`qwen/qwen3.8-27b`, `openai/gpt-oss-120b`, & `openai/gpt-oss-20b`) dengan pembacaan *header* `x-ratelimit-remaining-tokens` secara *real-time* dan *instant failover* tanpa antrean panjang (~15–25 detik untuk 8 kandidat unggulan).
- **Guardrail SL & R:R Pasca-AI**: Validator mengunci agar AI tidak pernah mempersempit *Liquidity-Protected SL* ke area *stop-hunt*, tidak menurunkan `TP2` di bawah `1:3.0`, dan tidak menyarankan geser SL ke BEP.

### 4. Hybrid Live Candle (`FIRST TOUCH`), Orderflow Publik, WebSocket, Tracker & Backtester
- **Zero-Repaint Hybrid Live Bar (`exchange_async.py`, `zone_radar.py`)**: Seluruh struktur (`EMA`, `ATR`, `OB`, `BOS`/`CHOCH`) dihitung 100% dari *closed bar*, sementara *unclosed live bar* `15M` menangkap **`🔔 FIRST TOUCH`** dan dinamika *approach* (`KOMPRESI_KOREKTIF`, `PULLBACK_WAJAR`, `IMPULSIF_LIAR`) secara *real-time*.
- **Orderflow Publik Tanpa API Key**: *Cumulative Volume Delta (`CVD`)* & *Absorption*, perubahan *Open Interest (`OI` 1H & 4H)*, *Top Trader L/S*, *Retail L/S*, dan penyapuan likuiditas sesi Asia/London/NY.
- **Tracker (`--track`) & Walk-Forward Backtester (`--backtest`)**: Mengevaluasi performa historis menggunakan simulasi *Fixed SL (No BEP)* menuju target penuh `TP2 >= 1:3.0`.

---

## 🛠️ Instalasi & Penggunaan Cepat

```powershell
# Instalasi dependensi
pip install -r requirements.txt

# Scan lengkap dengan posisi sizing otomatis (modal 74 USDT, risiko 1.5% per trade)
python binance.py --equity 74 --risk-pct 1.5

# Scan strategi gabungan (3-Pilar SMC + Breakout Konsolidasi)
python binance.py --strategy all
```
