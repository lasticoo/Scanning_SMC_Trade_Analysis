# Binance Futures SMC (Smart Money Concepts) & Breakout Setup Scanner

Scanner otomatis berbasis Python untuk mendeteksi setup trading berprobabilitas tinggi pada pasar **Binance USDⓈ-M Perpetual Futures** menggunakan analisis **Smart Money Concepts (SMC)** dan **Pola Konsolidasi Breakout**.

Hanya menggunakan **Public Endpoint Binance** (tanpa API key, tanpa order otomatis, dan aman).

---

## 🎯 Fitur Smart Money Concepts (SMC)

Skrip mendeteksi seluruh elemen kunci SMC secara otomatis:

1. **Order Block (OB) Bullish & Bearish**:
   - Mendeteksi candle institusional sebelum pergerakan impulsif yang meninggalkan FVG / menembus struktur.
   - Mengelompokkan status OB: *Unmitigated*, *Mitigated (Tapped)*, dan *Sweep POI*.
2. **Liquidity Sweep & Liquidity Run**:
   - **BSL Sweep**: Sumbu harga menyapu *Buy-Side Liquidity* (Swing High) tetapi candle *close* kembali ke bawah.
   - **SSL Sweep**: Sumbu harga menyapu *Sell-Side Liquidity* (Swing Low) tetapi candle *close* kembali ke atas.
   - **Liquidity Run**: Penembusan agresif dengan ekspansi momentum tinggi.
3. **Liquidity Range & Dealing Range**:
   - Menghitung Swing High & Swing Low lokal.
   - Menentukan zona **Discount** (< 50% Equilibrium) untuk mencari peluang BUY murah, dan zona **Premium** (> 50% Equilibrium) untuk mencari peluang SELL mahal.
4. **CHOCH (Change of Character)**:
   - Sinyal awal pembalikan arah tren institusional saat harga menembus swing struktural utama berlawanan arah.
5. **BOS (Break of Structure)**:
   - Konfirmasi kelanjutan tren saat harga menembus swing struktural searah tren.
6. **Inducement (IDM)**:
   - Mendeteksi internal swing pertama (*trap level*) dan mengonfirmasi apakah likuiditas IDM telah disapu (*IDM taken*) sebelum menyentuh POI utama.
7. **Supply & Demand (S&D) Zones**:
   - Memetakan zona Demand kuat di area Discount dan zona Supply kuat di area Premium.
8. **Candle Pin Bar (Rejection Wick)**:
   - Mendeteksi candle penolakan sumbu panjang (sumbu >= 50% dari total rentang candle) di area POI.
9. **CISD (Change in State of Delivery)**:
   - Konsep ICT/SMC modern: Mendeteksi peralihan algoritma pengiriman harga saat harga menutup melampaui harga pembukaan rangkaian candle berlawanan.
10. **POI (Point of Interest) & Sweep OB**:
    - Memberikan bobot tertinggi pada Order Block yang terbentuk setelah menyerap likuiditas candle/swing sebelumnya (*High Probability POI*).
11. **Setup SMC Eksekusi (Sniper Entry, OB Retest, FVG Tap)**:
    - **SMC Sniper Entry**: Kombinasi *Liquidity Sweep + CHOCH + Retest POI* di area Discount/Premium.
    - **SMC Demand Bounce (OB Retest)**: Reaksi pemantulan pada Bullish Order Block.
    - **SMC Supply Rejection (OB Retest)**: Reaksi penolakan pada Bearish Order Block.
    - **SMC FVG Retest Entry**: Pengisian celah *Fair Value Gap* (Imbalance).
    - **SMC SSL/BSL Liquidity Sweep Reversal**: Reaksi tajam setelah pembersihan kolam likuiditas eksternal.

---

## 🚀 Cara Menjalankan

### 1. Instalasi Kebutuhan
```powershell
pip install -r requirements.txt
```

### 2. Pilihan Strategi

#### A. Mode SMC (Default)
Memindai setup SMC pada koin-koin likuid teratas:
```powershell
python binance.py
# atau secara eksplisit:
python binance.py --strategy smc
```

#### B. Mode Breakout Konsolidasi (Pola Klasik)
Memindai pola konsolidasi: *Triangle, Wedge, Flag*:
```powershell
python binance.py --strategy breakout
```

#### C. Mode Gabungan (All)
Memindai kedua strategi sekaligus dan memilih setup dengan skor tertinggi untuk tiap aset:
```powershell
python binance.py --strategy all
```

---

## ⚡ Contoh Penggunaan Praktis

- **Scan Cepat pada Koin Tertentu**:
  ```powershell
  python binance.py --symbols BTCUSDT,ETHUSDT,SOLUSDT,DOGEUSDT
  ```

- **Scan dengan Manajemen Risiko & Ukuran Posisi Otomatis**:
  ```powershell
  python binance.py --equity 1000 --risk-pct 1
  ```
  *(Menghitung lot/qty posisi berdasarkan modal $1.000 dengan risiko maksimal 1% = $10 per trade)*

- **Scan dengan Filter Likuiditas Khusus**:
  ```powershell
  python binance.py --top 50 --min-vol 100
  ```
  *(50 pair teratas dengan volume 24 jam minimal $100 juta USDT)*

- **Mode Otomatis (Looping Tiap 15 Menit)**:
  ```powershell
  python binance.py --loop
  ```

- **Notifikasi Telegram**:
  ```powershell
  $env:TELEGRAM_BOT_TOKEN="token_bot_anda"
  $env:TELEGRAM_CHAT_ID="id_chat_anda"
  python binance.py --loop --telegram
  ```

- **Evaluasi Full LLM SMC (Aktif Otomatis)**:
  ```powershell
  python binance.py
  ```
  *(Full LLM Groq bertindak sebagai Penilai Tunggal / Head Trader yang menentukan skor 0-100, status ENTRY READY/WATCHLIST/REJECT, dan 4 pilar SMC institusional).*

- **Opsi Konfigurasi AI**:
  ```powershell
  python binance.py --ai-top 20                       # evaluasi 20 kandidat setup teratas
  python binance.py --ai-model openai/gpt-oss-120b    # gunakan model alternatif Groq
  python binance.py --no-ai                           # jalankan rumus mekanis Python klasik jika tanpa AI
  ```

---

## 🧠 Evaluator Penuh Full LLM (Head SMC Trader)

Scanner kini beroperasi dengan arsitektur **Full LLM Authority**:
1. **Analisis Koin Independen (Tanpa Keterikatan Kaku pada BTC)**:
   - Banyak altcoin memiliki rotasi likuiditas, ekosistem, dan momentum mandiri.
   - AI mengevaluasi setiap koin 100% murni berdasarkan Price Action dan struktur SMC koin itu sendiri tanpa mendiskualifikasi setup hanya karena BTC sedang berbeda arah.
2. **Pemetaan Mandiri 10 Elemen Kunci SMC (Ditentukan AI dari Data Binance Real-Time)**:
   - **Order Blocks (Bullish & Bearish)**: Mengidentifikasi OB aktif yang belum termitigasi.
   - **Liquidity Sweeps & Runs**: Mendeteksi penyapuan likuiditas BSL/SSL dan ekspansi agresif.
   - **Liquidity Range**: Memetakan Swing High, Swing Low, Equilibrium, dan zona Diskon vs Premium.
   - **CHOCH (Change of Character)**: Mengonfirmasi peralihan awal karakter pasar.
   - **Inducement (IDM)**: Mendeteksi level jebakan likuiditas retail dan status pembersihannya (*IDM Taken*).
   - **Supply & Demand Zones**: Memetakan area penawaran dan permintaan institusional.
   - **BOS (Break of Structure)**: Mengonfirmasi kelanjutan tren searah struktur.
   - **Pin Bar Rejection**: Mengukur sumbu penolakan panjang (rejection wick >= 50%) di zona penting.
   - **CISD (Change in State of Delivery)**: Mendeteksi peralihan algoritma pengiriman harga.
   - **High-Probability POI**: Point of Interest utama berupa Order Block yang terbentuk setelah sweep likuiditas.
3. **Keputusan Eksekusi Mandiri (Masuk atau Tidak)**:
   - AI menentukan status: `ENTRY READY` (siap dieksekusi), `WATCHLIST` (menunggu konfirmasi), atau `REJECT`.
   - Menghitung level presisi: Entry Zone, Stop Loss (SL), TP1, TP2, dan Rasio Risk:Reward (RR).
   - Menyertakan rencana aksi langkah demi langkah (*tactical action plan*).


---

## 📊 Format Laporan & Skor (0 - 100)

Sistem penilaian SMC membagi skor ke dalam 6 komponen objektif:
1. **Konteks HTF & Tren (Maks 20)**: Keselarasan tren 4H dan 1H.
2. **Kualitas POI & Zone (Maks 25)**: Validitas Order Block, FVG, dan posisi Dealing Range (Discount/Premium).
3. **Struktur & Liquidity (Maks 25)**: Adanya Liquidity Sweep, CHOCH, BOS, dan Inducement taken.
4. **Trigger & Rejection (Maks 15)**: Pin Bar rejection, CISD, dan volume konfirmasi.
5. **Risk to Reward (Maks 10)**: RR minimal 1:2.0, hingga 1:3.0+ ke target likuiditas lawan.
6. **Likuiditas Pasar (Maks 5)**: Volume pasar 24 jam dan toleransi spread.

**Status Setup**:
- `ENTRY READY`: Skor >= 70 dan RR >= 2.0 (setup matang dengan konfirmasi).
- `WATCHLIST`: Skor >= 50 (harga mendekati/menyentuh POI, menunggu trigger).
- `REJECT`: Skor < 50 atau RR di bawah 2.0.

Laporan otomatis dicetak di terminal dan disimpan di folder `scan_output/` dalam format `.txt` dan `.csv`.

