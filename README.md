# Binance Futures Dynamic 3-Archetype Top-Down MTF SMC/ICT (`4H` → `1H` → `15M`), Pre-Reaction, Reversal, & Orderflow Scanner v3.0

Scanner kuantitatif & institusional berbasis Python untuk mendeteksi setup trading berprobabilitas tinggi pada pasar **Binance USDⓈ-M Perpetual Futures** dengan pendekatan **Top-Down Multi-Timeframe Refinement (`4H` → `1H` → `15M`)**, **3 Variasi Setup Dinamis (*Follow-Trend Post-Reaction*, *Pre-Reaction Limit Stalker*, & *HTF Wall Reversal*)**, **Hybrid Live Candle (`FIRST TOUCH`)**, **Orderflow Publik**, **Consolidation Breakout**, serta **Kurator SMC/ICT Kripto Modern berbasis Groq LLM** dan **Forward If–Then Playbook**.

Hanya menggunakan **Public Endpoint & WebSocket Binance** (tanpa API key Binance, tanpa akses akun, dan aman).

---

```powershell
# 1. Scan standar 3 Variasi Dinamis (Follow-Trend + Pre-Tap Limit + HTF Reversal) + Kurator Groq AI
python binance.py

# 2. Tampilkan HANYA koin yang sedang di/dekat POI, First Touch, Pre-Tap Segar, atau baru Liquidity Sweep
python binance.py --in-zone

# 3. Mode kilat (~4 detik) Radar Lokasi Harga Live (First Touch, In-POI, Pre-Tap HTF Segar, & Sweep)
python binance.py --radar

# 4. Scan super cepat murni mesin kuantitatif 3-Archetype Python (~5-7 detik untuk 50 koin)
python binance.py --no-ai --top 50

# 5. Atur batas jarak maksimal mengintai zona segar Pre-Reaction (default: 2.60%) & kuota kurasi AI
python binance.py --max-stalk-pct 2.2 --ai-top 10

# 6. Scan lalu pantau live sentuhan zona Limit Pre-Tap, 50% CE, Reversal Sweep, & TP/SL via WebSocket
python binance.py --watch-ws

# 7. Mode otomatis tiap penutupan 15M + WebSocket POI Watcher + Notifikasi Telegram
python binance.py --loop --watch-ws --telegram

# 8. Evaluasi akurasi seluruh histori sinyal di folder scan_output/ (termasuk PRE-TAP LIMIT & REVERSAL WATCH)
python binance.py --track --include-watchlist

# 9. Backtest Walk-Forward 14 hari terakhir (bebas look-ahead bias)
python binance.py --backtest --days 14 --bt-top 10
```

---

## 🚀 Fitur Utama v3.0 (Dynamic 3-Archetype + Pre-Reaction + AI Crypto Curator)

### 1. Mesin 3 Variasi Setup Dinamis (`mtf_smc.py`)
Tidak lagi kaku hanya mengikuti tren setelah harga terlanjur memantul (*post-reaction*), melainkan memindai 3 variasi setup secara simultan dan memilih setup terbaik untuk setiap koin:
- **`[1] TREND_CONT` — Follow-Trend (*Post-Reaction* / Konfirmasi di POI)**:
  Mempertahankan utuh mesin pembaca arah & konfirmasi reaksi (`ENTRY READY` / `WATCHLIST`) saat harga menyentuh POI `4H`/`1H`/`15M` disertai *Liquidity Sweep*, `CHOCH`/`BOS`, `CISD`, atau `Pin Bar`.
- **`[2] PRE_REACTION` — Pre-Tap Limit (*Mendeteksi SEBELUM Harga Bereaksi*)**:
  Mendeteksi koin yang sedang mendekati zona **Supply / Demand / Order Block / Nested Breaker `4H`/`1H` yang 100% segar (`0 touches` / *unmitigated*)** dalam jarak rapat (`<= 1.65x ATR 1H` atau `<= 2.60%`, estimasi jemput `~0.5–4.5 jam`). Membedakan tegas antara **`🔭 STALK / PRE-TAP`** (belum tersentuh) vs **`🏃 POST-BOUNCE`** (sudah memantul pergi).
- **`[3] HTF_REVERSAL` — Counter-Trend / Wall-Bounce Inversion (`REVERSAL WATCH`)**:
  Membaca kelelahan tren & potensi keterbalikan arah ketika harga menghantam **Tembok Besar `4H`/`1H`** di ekstrem *Dealing Range* disertai *Liquidity Climax* (`BSL`/`SSL` Sweep, `RSI` ekstrem, *CVD Absorption*, atau pergeseran struktur `15M CHOCH/CISD`).

### 2. Hybrid Unclosed Live Candle & Analisa Gaya Kedatangan Harga (`exchange_async.py`, `zone_radar.py`)
- **Zero-Repaint Hybrid Live Bar**: Seluruh indikator struktural (`EMA`, `ATR`, `OB`, `BOS`/`CHOCH`) tetap 100% dihitung dari candle yang sudah `close`, sementara candle `15M` yang sedang berjalan (*unclosed live bar*) disimpan terpisah untuk menangkap **`🔔 FIRST TOUCH`** detik itu juga tanpa menunggu penutupan candle 15 menit.
- **Analisa Dinamika Pendekatan (*Approach Style*)**: Membedakan pendekatan **`KOMPRESI_KOREKTIF`** (ideal untuk `Limit Pre-Tap`), **`PULLBACK_WAJAR`** (ideal `Limit 50% CE`), dan **`IMPULSIF_LIAR`** (*falling knife* / marubozu — otomatis dilarang pasang limit buta dan dialihkan ke `Wait Sweep/CISD`).

### 3. Kurator AI SMC/ICT Kripto Modern (`ai_validator.py`)
- **Kilasan Cepat Python + Kurasi Paralel Groq**: Menampilkan ringkasan hasil Python secara instan dalam ~6–7 detik, lalu memanggil Groq LLM secara paralel (*persistent HTTP session*, `4 workers`) pada daftar berimbang lintas 3 variasi (`ENTRY READY`, `PRE-TAP LIMIT`, `REVERSAL WATCH`, dan `WATCHLIST`).
- **Adaptif terhadap Karakter Liar Kripto**: AI mengevaluasi risiko *stop-hunt wick* (*Judas Swing* / `EQH`/`EQL` di depan POI), kecepatan *approach*, kualitas *orderflow* (`CVD`, `OI`, *Top Trader vs Retail L/S*), dan memilih taktik eksekusi terbaik (`ENTRY_NOW`, `LIMIT_PRE_TAP`, `WAIT_SWEEP_TRIGGER`, `LIMIT_RETEST`, atau `SKIP`), lengkap dengan *post-AI safety guardrails* dan proteksi skor `0`.

### 4. Panduan Langkah Kedepan (*Forward If–Then Playbook & Antisipasi Flip*)
Setiap sinyal dilengkapi peta tindakan konkret ke depan:
- **`[Skenario Utama]`**: Apa yang sedang terjadi sekarang, di harga berapa & dengan pemicu apa harus masuk, serta aturan manajemen risiko (`50%` *partial TP1* + geser `SL` ke `BEP`).
- **`[Skenario Flip/Batal]`**: Apa yang harus dilakukan jika area penting justru dijebol secara impulsif (*Breaker Block Flip* / pembatalan limit & peluang balik arah).

### 5. Fondasi Top-Down MTF (`4H` → `1H` → `15M`), Orderflow Publik, WebSocket, Tracker & Backtester
- **Zoom-Out Check `ATR 1H` & Batas Durasi Posisi (`<= 36 jam`)**: Jarak *Stop Loss* proporsional terhadap `ATR 1H` (`0.75x – 2.20x ATR 1H`, bebas SL terlalu sempit maupun terlalu lebar) dan durasi *hold* dijaga tetap *Scalp-Intraday* / *Intraday* / *Short-Swing* agar tidak tergerus *funding fee* 8 jam.
- **Orderflow Publik Tanpa API Key**: *Cumulative Volume Delta (`CVD`)* & *Absorption*, perubahan *Open Interest (`OI` 1H & 4H)*, *Top Trader L/S*, *Retail L/S*, dan penyapuan likuiditas sesi Asia/London/NY.
- **WebSocket Watcher (`--watch-ws`), Tracker (`--track`), & Walk-Forward Backtester (`--backtest`)**: Mendukung penuh pemantauan real-time dan evaluasi performa untuk seluruh status termasuk `PRE-TAP LIMIT` dan `REVERSAL WATCH`.

---

## 🛠️ Instalasi & Penggunaan Cepat

```powershell
# Instalasi dependensi
pip install -r requirements.txt

# Scan lengkap dengan posisi sizing otomatis (modal 1000 USDT, risiko 1% per trade)
python binance.py --equity 1000 --risk-pct 1

# Scan strategi gabungan (3-Archetype SMC + Breakout Konsolidasi)
python binance.py --strategy all
```
