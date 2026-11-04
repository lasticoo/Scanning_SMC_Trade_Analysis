# Binance Futures Top-Down MTF SMC (`4H` → `1H` → `15M`), Orderflow, & Breakout Scanner v2.0

Scanner kuantitatif & institusional berbasis Python untuk mendeteksi setup trading berprobabilitas tinggi pada pasar **Binance USDⓈ-M Perpetual Futures** dengan pendekatan **Top-Down Multi-Timeframe Refinement (`4H` → `1H` → `15M`)**, **Orderflow Publik**, **Consolidation Breakout**, serta evaluasi **Full LLM (Groq)**.

Hanya menggunakan **Public Endpoint & WebSocket Binance** (tanpa API key Binance, tanpa akses akun, dan aman).

---
# 1. Scan standar Top-Down MTF (4H -> 1H -> 15M) + Radar Zona Live + Orderflow + Evaluasi Groq AI
python binance.py

# 2. Tampilkan HANYA koin yang harganya SUDAH di/dekat Demand/Supply/POI OB-FVG atau baru Liquidity Sweep
python binance.py --in-zone

# 3. Mode kilat (~4 detik) Radar Lokasi Harga Live (Demand/Supply/OB/FVG/Sweep tanpa tunggu antrean AI)
python binance.py --radar

# 4. Scan super cepat murni mesin Top-Down MTF Python (~4-5 detik untuk 40 koin)
python binance.py --no-ai --top 40

# 5. Scan lalu pantau live sentuhan zona Limit POI, 50% CE, & Liquidity Sweep via WebSocket
python binance.py --watch-ws

# 6. Mode otomatis tiap penutupan 15M + WebSocket POI Watcher + Notifikasi Telegram
python binance.py --loop --watch-ws --telegram

# 7. Evaluasi akurasi seluruh histori sinyal di folder scan_output/
python binance.py --track --include-watchlist

# 8. Backtest Walk-Forward 14 hari terakhir (bebas look-ahead)
python binance.py --backtest --days 14 --bt-top 10

## 🚀 Apa yang Baru di Upgrade v2.0?

### 1. Metode Top-Down Multi-Timeframe Refinement (`4H` → `1H` → `15M`)
Mengotomatiskan alur kerja analis institusional tanpa menambah beban kuota API:
- **Lapis `4H` (Kompas Makro & 1–3 Tembok Besar)**:
  Memetakan *Macro Dealing Range* serta **Tembok Supply `4H`** (atas) dan **Tembok Demand `4H`** (bawah) agar sinyal tidak menabrak tembok besar berlawanan dan target `TP2` tidak tersangkut berhari-hari.
- **Lapis `1H` (*Working Trading Range* & Estimasi Durasi Posisi)**:
  Mengunci rentang kerja aktif (`1H Swing Low` ↔ `1H Swing High`), *Unmitigated Order Block*, *Breaker Block*, *Unmitigated FVG*, serta kolam likuiditas **`EQH` / `EQL`**. Menghitung **Estimasi Durasi Posisi (`Hold Time`)** berbasis kecepatan `ATR 1H` agar modal tidak tergerus *funding fee* 8 jam akibat target terlalu jauh.
- **Lapis `15M` (Likuiditas Kecil + Filter Proporsionalitas *"Zoom-Out Check"*)**:
  Mencari *Nested POI* (`1H` + `15M`), *BSL/SSL/Asian Range Sweep*, `CHOCH`, `BOS`, `IDM Taken`, `CISD`, dan `Pin Bar`. Jarak *Stop Loss* dikunci secara matematis terhadap **`ATR 1H` (`0.75x – 2.20x ATR 1H`)** dan proporsi *Working Range* `1H`—menghilangkan SL "sehelai rambut" (`0.05%–0.4%`) yang mudah tersapu *wick*, sekaligus menjaga agar SL tidak membengkak seperti SL `4H` murni.

### 2. Data Orderflow & Derivatif Futures Publik (Tanpa API Key)
- **CVD (*Cumulative Volume Delta*) & Absorption**: Dihitung langsung dari volume *Taker Buy* (`tb_quote`) vs *Taker Sell* klines publik untuk mendeteksi *Bullish / Bearish CVD Absorption* di area POI.
- **Open Interest (`OI`) & Long/Short Ratio**: Mengambil perubahan `OI` (`1H` & `4H`), *Top Trader L/S Ratio*, dan *Global Retail L/S Ratio* untuk membedakan masuknya uang institusional baru vs *squeeze* / jebakan ritel.
- **Killzone & Likuiditas Sesi Asia**: Mendeteksi sesi aktif (*Asian*, *London Killzone*, *NY Open Killzone*) dan penyapuan *Asian High / Asian Low*.

### 3. Evaluator AI Top-Down Tangguh (`ai_validator.py`)
- **Data 3 Timeframe Utuh & Presisi Harga Adaptif**: Seluruh koordinat `4H`, `1H`, `15M`, dan *Orderflow* dikirim lengkap tanpa pembulatan kaku `:.4f` (aman untuk altcoin harga mikro seperti `1000PEPEUSDT` maupun `BTCUSDT`).
- **Tahan Tag `<think>` & Proteksi Skor `0`**: Dilengkapi ekstraktor JSON seimbang (mendukung model *reasoning* seperti Qwen/DeepSeek), *rate-pacing* + *exponential backoff* otomatis, serta **Proteksi Skor `0`**—jika API Groq sedang terkena *rate-limit* atau *timeout*, skor dan level Top-Down MTF Python dipertahankan utuh (tidak pernah lagi menjatuhkan koin bagus menjadi `score 0`).
- **Validasi Pasca-AI**: Setiap level `SL`/`TP` dari AI divalidasi ulang terhadap pagar *Zoom-Out Check* `1H`, lalu `risk_pct`, `RR`, durasi *hold*, dan *position sizing* (`qty`) disinkronkan ulang otomatis.

### 4. Arsitektur Modular, Kecepatan `AsyncIO`, WebSocket Live Alert, Tracker, & Backtester
- **Struktur Modular (`scanner_core/`)**: Terpisah rapi menjadi `models.py`, `indicators.py`, `exchange_async.py`, `mtf_smc.py`, `zone_radar.py`, `breakout.py`, `tracker.py`, dan `ws_watcher.py`.
- **Detektor Lokasi Harga Live & Liquidity Sweep Terpadu (`zone_radar.py`)**:
  - Mendeteksi secara presisi apakah harga **detik ini sudah di dalam** Demand/Supply `4H`/`1H`, *Fresh Order Block*, *Breaker Block*, atau *Unmitigated FVG* beserta persentase kedalaman mitigasi zona (*Zone Depth %* & *50% Mean Threshold / CE*).
  - Mendeteksi *Liquidity Sweep* lintas `1H` & `15M` (`BSL`/`SSL` Swing Sweep, `EQH`/`EQL` Stop-Hunt, `IDM Taken`, dan *Asian High/Low Sweep*), lengkap dengan deteksi reaksi candle (*REJECTING* vs *TESTING / Falling Knife*).
  - Otomatis mengurutkan kandidat `WATCHLIST` agar koin yang **sedang di zona / baru saja *sweep*** muncul paling atas, dilengkapi tabel khusus **Radar Lokasi Harga Live** serta filter `--in-zone` & `--radar`.
- **High-Speed Async Scanner (`aiohttp`)**: Memindai puluhan hingga ratusan kontrak beserta data derivatif hanya dalam hitungan detik.
- **Real-Time WebSocket POI & Sweep Watcher (`--watch-ws`)**: Memantau harga *live* via stream `@bookTicker` sub-detik (0 beban REST API) dan memberi peringatan `⚡ APPROACHING POI`, `🎯 POI TAPPED`, `💎 50% CE POI`, `💧 LIVE LIQUIDITY SWEEP`, `💰 TP1`, `🏆 TP2`, hingga `🛑 SL`.
- **Signal Performance Tracker (`--track`) & Walk-Forward Backtester (`--backtest`)**: Mengukur *Win Rate*, *Profit Factor*, *Expectancy (R)*, *Max Drawdown*, dan rata-rata durasi *hold* baik dari histori CSV nyata maupun simulasi historis *walk-forward* bebas *look-ahead bias*.

---

## 🛠️ Instalasi & Penggunaan

### 1. Instalasi Dependensi
```powershell
pip install -r requirements.txt
```

### 2. Pemindaian Live (Top-Down MTF SMC / Breakout)
```powershell
# Scan cepat Async Top-Down MTF SMC + Orderflow + Evaluasi AI Groq (Default)
python binance.py

# Scan murni mesin kuantitatif Top-Down MTF Python (tanpa AI)
python binance.py --no-ai

# Scan simbol tertentu + hitung ukuran posisi otomatis (modal 1000 USDT, risiko 1%)
python binance.py --symbols BTCUSDT,ETHUSDT,SOLUSDT,DOGEUSDT,AVAXUSDT --equity 1000 --risk-pct 1

# Scan strategi gabungan (SMC + Breakout Konsolidasi)
python binance.py --strategy all
```

### 3. Pantauan Harga Real-Time via WebSocket (`--watch-ws`)
```powershell
# Scan satu kali lalu langsung pantau live sentuhan zona POI, TP1, TP2, & SL
python binance.py --watch-ws

# Jalan otomatis tiap 15 menit + pantauan WebSocket real-time di sela candle + kirim Telegram
python binance.py --loop --watch-ws --telegram
```

### 4. Evaluasi Performa Sinyal (`--track`) & Backtester Historis (`--backtest`)
```powershell
# Evaluasi performa nyata seluruh histori sinyal di folder scan_output/*.csv
python binance.py --track

# Sertakan juga skenario Limit Order dari sinyal WATCHLIST pada tracker
python binance.py --track --include-watchlist

# Jalankan Walk-Forward Backtest (bebas look-ahead bias) selama 14 hari terakhir
python binance.py --backtest --days 14 --bt-top 10

# Backtest khusus pada koin pilihan Anda
python binance.py --backtest --days 21 --symbols BTCUSDT,ETHUSDT,SOLUSDT,DOGEUSDT,AVAXUSDT
```
