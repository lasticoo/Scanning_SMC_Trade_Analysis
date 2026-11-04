# -*- coding: utf-8 -*-
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Optional

NAN = float("nan")

STABLE_BASES = {
    "USDC", "FDUSD", "TUSD", "USDP", "BUSD", "DAI", "USDD", "USDE", "PYUSD",
    "EURI", "AEUR", "XUSD", "USD1", "RLUSD", "BFUSD",
}

MAJOR_SYMBOLS = {
    "BTCUSDT", "ETHUSDT",
}


@dataclass
class Config:
    # strategi
    strategy: str = "smc"           # "smc", "breakout", "all"
    smc_entry_score: int = 70
    smc_watch_score: int = 52
    # pola konsolidasi (breakout)
    piv_len: int = 5
    min_pat: int = 15
    max_pat: int = 100
    conv_ratio: float = 0.85
    flat_tol: float = 0.6
    flag_impulse: float = 3.0
    imp_look: int = 20
    touch_tol: float = 0.25
    # breakout / breakdown
    bk_buffer: float = 0.1
    body_min: float = 0.6
    rel_vol_len: int = 20
    rel_vol_thr: float = 1.5
    ext_atr: float = 1.2
    retest_bars: int = 12
    retest_tol: float = 0.3
    event_recent_bars: int = 6      # event dianggap "aktif" jika terjadi dalam N bar 15M terakhir
    # risk management & Top-Down MTF "Zoom-Out Check" (proporsional 1H)
    sl_buffer: float = 0.30         # buffer struktural (dikalikan ATR 1H pada MTF SMC)
    min_risk_atr: float = 0.6       # fallback kompatibilitas lama
    min_sl_atr_1h: float = 1.15     # batas bawah anti-terlalu-sempit: minimal 1.15x ATR 1H
    max_sl_atr_1h: float = 2.25     # batas atas anti-terlalu-lebar: maksimal 2.25x ATR 1H
    min_sl_range_1h: float = 0.16   # minimal 16% dari lebar Working Range 1H
    max_sl_range_1h: float = 0.38   # maksimal 38% dari lebar Working Range 1H
    min_sl_pct_major: float = 0.0080  # 0.80% floor untuk BTC/ETH
    min_sl_pct_alt: float = 0.0115    # 1.15% floor untuk altcoin & high-beta futures
    max_hold_hours: float = 36.0    # batas ideal durasi posisi agar tidak lama terkena funding
    mm_factor: float = 1.0
    min_rr: float = 2.0
    entry_score: int = 80
    watch_score: int = 65
    # konteks HTF
    htf_piv: int = 4
    # universe & likuiditas
    min_vol_usdt: float = 50e6
    max_spread_bps: float = 6.0
    min_age_days: int = 14
    top_n: int = 120
    # data
    limit_15m: int = 400
    limit_1h: int = 300
    limit_4h: int = 300
    min_bars_15m: int = 150
    workers: int = 12               # async concurrency / thread workers
    use_async: bool = True
    fetch_derivatives: bool = True  # ambil Open Interest & Long/Short Ratio publik
    # evaluasi Full LLM (Groq)
    ai_audit: bool = True
    ai_top_n: int = 15
    # filter & radar lokasi harga di Demand/Supply/POI/Sweep
    only_in_zone: bool = False      # tampilkan hanya koin yang sudah di/dekat POI atau baru saja sweep
    radar_mode: bool = False        # fokus tampilan Radar Zona & Liquidity Sweep Live
    near_poi_pct: float = 0.50      # radius % dari bibir zona untuk status APPROACHING_POI


@dataclass
class Pat:
    ok: bool
    ptype: str
    bias: int
    uNow: float
    lNow: float
    uSlope: float
    lSlope: float
    patLen: int
    wNow: float
    wStart: float
    impulse: float


@dataclass
class Setup:
    symbol: str = ""
    direction: str = ""
    kind: str = ""
    pattern: str = ""
    bar_index: int = 0
    candle_time: str = ""
    level: float = NAN
    entry: float = NAN              # harga eksekusi acuan rencana (market saat di POI / limit mid saat tunggu retest)
    market_price: float = NAN       # harga close terkini saat scan
    limit_entry: float = NAN        # titik tengah ideal limit retest di POI 1H/15M
    entry_zone: str = ""
    poi_low: float = NAN
    poi_high: float = NAN
    sl: float = NAN
    tp1: float = NAN
    tp2: float = NAN
    rr: float = NAN
    rr_tp1: float = NAN
    risk_pct: float = NAN
    tp1_pct: float = NAN
    tp2_pct: float = NAN
    score: float = 0.0
    python_score: float = 0.0       # simpan skor murni Python sebelum AI
    parts: list = field(default_factory=list)
    status: str = ""
    python_status: str = ""
    confirmed: bool = False
    extended: bool = False
    rel_vol: float = NAN
    rsi: float = NAN
    atr: float = NAN                # ATR 15M
    atr_pct: float = NAN
    atr_1h: float = NAN             # ATR 1H (jangkar Zoom-Out Check)
    atr_1h_pct: float = NAN
    sl_atr_1h_ratio: float = NAN    # rasio jarak SL terhadap ATR 1H
    upper_line: float = NAN
    lower_line: float = NAN
    opp_level: Optional[float] = None
    trend_4h: str = ""
    struct_1h: str = ""
    trend_4h_short: str = ""
    struct_1h_short: str = ""
    confirm_txt: str = ""
    confirm_short: str = ""
    invalidation: str = ""
    notes: list = field(default_factory=list)
    # Top-Down MTF (4H -> 1H -> 15M) & Estimasi Durasi Posisi
    range_4h_str: str = ""
    walls_4h_str: str = ""
    range_1h_low: float = NAN
    range_1h_high: float = NAN
    range_1h_eq: float = NAN
    range_1h_pos_pct: float = NAN   # 0% = low 1H, 100% = high 1H
    est_hold_hours_tp1: float = NAN
    est_hold_hours_tp2: float = NAN
    est_hold_label: str = ""
    session_name: str = ""
    # Orderflow & Derivatives publik (tanpa API key)
    cvd_state: str = ""
    taker_buy_ratio: float = NAN
    oi_change_1h_pct: float = NAN
    oi_change_4h_pct: float = NAN
    top_ls_ratio: float = NAN
    retail_ls_ratio: float = NAN
    orderflow_summary: str = ""
    # Detektor Lokasi Harga Live (Demand / Supply / POI OB-FVG-Breaker / Liquidity Sweep)
    zone_state: str = ""            # SWEEP_IN_POI, IN_POI_REJECTING, IN_POI_TESTING, LIQ_SWEPT, APPROACHING_POI, WAITING_PULLBACK
    zone_badge: str = ""            # label ringkas status zona & jarak untuk tabel utama
    dist_to_poi_pct: float = NAN    # 0.0% jika harga sudah di dalam [poi_low, poi_high]
    zone_depth_pct: float = NAN     # 0..100% kedalaman penetrasi harga/wick di dalam zona POI
    zone_fresh: bool = False        # apakah POI masih segar (unmitigated / sentuhan pertama)
    sweep_status: str = ""          # status BSL/SSL/EQH/EQL/IDM/Asian sweep terkini
    zone_reaction: str = ""         # REJECTING / TESTING / MENUNGGU SENTUHAN + bukti candle/CVD
    active_zones_summary: str = ""  # ringkasan seluruh Demand/Supply/OB/FVG 4H-1H-15M yang sedang disentuh/dekat
    eq_magnet_info: str = ""        # kolam EQH/EQL/IDM terdekat yang belum tersapu
    # internal (disimpan untuk state retest breakout)
    q_pts: float = 0.0
    body_pts: float = 0.0
    height: float = NAN
    # data meta & sizing
    meta: dict = field(default_factory=dict)
    last_candle: str = ""
    qty: float = NAN
    notional: float = NAN
    risk_usdt: float = NAN
    # audit AI SMC (Groq)
    ai_decision: str = ""
    ai_confidence: float = NAN
    ai_summary: str = ""
    ai_action_note: str = ""
    ai_model: str = ""
    # 10 elemen SMC institusional dari AI & data mentah Binance
    smc_elements: dict = field(default_factory=dict)
    raw_market_data: dict = field(default_factory=dict)


@dataclass
class ZoneRadarHit:
    """Hasil deteksi live koin yang detik ini sedang di Demand/Supply/OB/FVG/Breaker atau baru Liquidity Sweep."""
    symbol: str = ""
    market_price: float = NAN
    direction: str = ""             # LONG (Demand/SSL) atau SHORT (Supply/BSL)
    radar_state: str = ""           # SWEEP_IN_ZONE, IN_HTF_ZONE, IN_LTF_ZONE, LIVE_LIQ_SWEEP, APPROACHING_HTF
    radar_badge: str = ""
    primary_zone: str = ""
    zone_low: float = NAN
    zone_high: float = NAN
    dist_pct: float = NAN
    depth_pct: float = NAN
    zone_fresh: bool = False
    all_zones_touched: str = ""
    sweep_info: str = ""
    magnet_info: str = ""
    reaction_info: str = ""
    cvd_state: str = ""
    trend_4h_short: str = ""
    struct_1h_short: str = ""
    setup_status: str = "-"
    setup_score: float = 0.0
    priority: int = 99


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def fp(x: Optional[float]) -> str:
    """Format harga adaptif sesuai skala desimal koin (aman untuk koin mikro s.d. BTC)."""
    if x is None or (isinstance(x, (int, float)) and (math.isnan(x) or math.isinf(x))):
        return "-"
    x = float(x)
    ax = abs(x)
    if ax == 0:
        return "0"
    dec = min(10, max(2, 5 - int(math.floor(math.log10(ax)))))
    return f"{x:.{dec}f}"


def ts_utc(ms: int) -> str:
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime("%Y-%m-%d %H:%M")


def size_position(s: Setup, equity: float, risk_pct: float, fee_pct: float) -> None:
    """Hitung ukuran posisi (qty, notional, risk_usdt) berdasarkan entry & SL terkini."""
    if equity <= 0 or math.isnan(s.entry) or math.isnan(s.sl):
        return
    step = s.meta.get("step", 0.0)
    min_notional = s.meta.get("min_notional", 5.0)
    risk_amt = equity * risk_pct / 100.0
    per_unit = abs(s.entry - s.sl) + (fee_pct / 100.0) * (s.entry + s.sl)   # fee taker masuk + keluar
    if per_unit <= 0:
        return
    qty = risk_amt / per_unit
    if step and step > 0:
        qty = math.floor(qty / step) * step
    s.qty = qty
    s.notional = qty * s.entry
    s.risk_usdt = qty * per_unit
    # Bersihkan catatan notional lama sebelum menambah baru (jika dihitung ulang pasca-AI)
    s.notes = [nt for nt in s.notes if not nt.startswith("Notional ")]
    if s.notional < min_notional:
        s.notes.append(f"Notional {s.notional:.2f} USDT di bawah minimum kontrak ({min_notional}); ukuran posisi tidak valid.")
