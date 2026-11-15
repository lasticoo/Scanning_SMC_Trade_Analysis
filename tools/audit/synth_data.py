# -*- coding: utf-8 -*-
"""
Deterministic Synthetic OHLCV Generator (Random Walk + Volatility Clustering)
for SMC Audit & Neutrality Verification (15M / 1H / 4H).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from scanner_core.indicators import htf_context
from scanner_core.models import Config


def make_synthetic_15m(n_bars: int = 5000, seed: int = 42, base_price: float = 100.0) -> pd.DataFrame:
    """
    Generate 15M OHLCV DataFrame with GARCH-like volatility clustering,
    regime shifts (uptrend, downtrend, sweep-and-reverse), and realistic taker volume.
    """
    rng = np.random.default_rng(seed)
    start_ms = 1700000000000  # fixed UTC epoch ms aligned to 15m
    start_ms = (start_ms // 900_000) * 900_000
    open_times = start_ms + np.arange(n_bars, dtype=np.int64) * 900_000
    close_times = open_times + 900_000 - 1

    sigma = np.zeros(n_bars, dtype=float)
    sigma[0] = 0.0045
    ret = np.zeros(n_bars, dtype=float)

    # Regime drift cycles to induce swings, sweeps, OBs, FVGs, and reversals
    cycle = np.sin(np.linspace(0, 18 * np.pi, n_bars)) * 0.00045
    burst_mask = rng.random(n_bars) < 0.035

    for i in range(1, n_bars):
        shock = rng.standard_normal()
        if burst_mask[i]:
            shock *= rng.uniform(2.2, 3.8)
        var = 1e-6 + 0.12 * (ret[i - 1] ** 2) + 0.84 * (sigma[i - 1] ** 2)
        sigma[i] = float(np.clip(np.sqrt(var), 0.0018, 0.025))
        ret[i] = cycle[i] + sigma[i] * shock

    close = base_price * np.exp(np.cumsum(ret))
    open_p = np.empty(n_bars, dtype=float)
    open_p[0] = base_price
    open_p[1:] = close[:-1]

    wick_up = np.abs(rng.normal(0.0, sigma * 0.75, size=n_bars)) * close
    wick_dn = np.abs(rng.normal(0.0, sigma * 0.75, size=n_bars)) * close
    # Occasional stop-hunt wicks
    sweep_up = rng.random(n_bars) < 0.04
    sweep_dn = rng.random(n_bars) < 0.04
    wick_up[sweep_up] *= rng.uniform(2.0, 3.5, size=int(np.sum(sweep_up)))
    wick_dn[sweep_dn] *= rng.uniform(2.0, 3.5, size=int(np.sum(sweep_dn)))

    high = np.maximum(open_p, close) + wick_up
    low = np.minimum(open_p, close) - wick_dn
    low = np.maximum(low, 1e-6)

    base_vol = 150_000.0 * (1.0 + (sigma / 0.0045) ** 1.5)
    vol = base_vol * rng.lognormal(0.0, 0.35, size=n_bars)
    buy_ratio = np.clip(0.50 + 0.22 * np.tanh((close - open_p) / np.maximum(high - low, 1e-9)), 0.15, 0.85)
    taker_buy_base = vol * buy_ratio
    quote_vol = vol * ((open_p + high + low + close) / 4.0)
    taker_buy_quote = quote_vol * buy_ratio

    return pd.DataFrame({
        "open_time": open_times,
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": vol,
        "close_time": close_times,
        "quote_volume": quote_vol,
        "trades": rng.integers(200, 5000, size=n_bars),
        "taker_buy_base": taker_buy_base,
        "taker_buy_quote": taker_buy_quote,
    })


def resample_ohlcv(df15: pd.DataFrame, group_bars: int) -> pd.DataFrame:
    """Resample 15M DataFrame into 1H (group_bars=4) or 4H (group_bars=16) without look-ahead."""
    n = (len(df15) // group_bars) * group_bars
    sub = df15.iloc[:n]
    grp = np.arange(n) // group_bars
    agg = sub.groupby(grp, sort=False).agg(
        open_time=("open_time", "first"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        close_time=("close_time", "last"),
        quote_volume=("quote_volume", "sum"),
        trades=("trades", "sum"),
        taker_buy_base=("taker_buy_base", "sum"),
        taker_buy_quote=("taker_buy_quote", "sum"),
    )
    return agg.reset_index(drop=True)


def iter_synthetic_windows(
    seeds: Tuple[int, ...] = (11, 22, 33, 44, 55, 66),
    n_bars_15m: int = 5000,
    windows_per_seed: int = 47,
) -> List[Dict]:
    """
    Generate deterministic multi-timeframe (4H/1H/15M) slices across seeds.
    Default: 6 seeds * 47 windows = 282 slices.
    """
    cfg = Config()
    btc_15m = make_synthetic_15m(n_bars=n_bars_15m, seed=999, base_price=65000.0)
    cases: List[Dict] = []

    for s_idx, seed in enumerate(seeds):
        df15_full = make_synthetic_15m(n_bars=n_bars_15m, seed=seed, base_price=10.0 * (s_idx + 1))
        step = max(16, (n_bars_15m - 2000) // max(1, windows_per_seed))
        for w_idx in range(windows_per_seed):
            end_15 = 2000 + w_idx * step
            end_15 = (end_15 // 16) * 16  # align to 4H boundary
            if end_15 > n_bars_15m:
                end_15 = (n_bars_15m // 16) * 16

            slice_15_all = df15_full.iloc[:end_15]
            df15 = slice_15_all.iloc[-cfg.limit_15m :].reset_index(drop=True)
            df1 = resample_ohlcv(slice_15_all, 4).iloc[-cfg.limit_1h :].reset_index(drop=True)
            df4 = resample_ohlcv(slice_15_all, 16).iloc[-cfg.limit_4h :].reset_index(drop=True)

            btc_slice = btc_15m.iloc[:end_15]
            b1 = resample_ohlcv(btc_slice, 4).iloc[-cfg.limit_1h :].reset_index(drop=True)
            b4 = resample_ohlcv(btc_slice, 16).iloc[-cfg.limit_4h :].reset_index(drop=True)

            ctx4 = htf_context(df4, cfg.htf_piv)
            ctx1 = htf_context(df1, cfg.htf_piv)
            btc = {"4h": htf_context(b4, cfg.htf_piv), "1h": htf_context(b1, cfg.htf_piv)}
            meta = {
                "vol24": 120_000_000.0,
                "spread_bps": 1.8,
                "funding": -0.0001 if (w_idx % 2 == 0) else 0.00015,
            }
            deriv = {
                "oi_change_1h_pct": 2.4 if (w_idx % 3 == 0) else -1.2,
                "oi_change_4h_pct": 5.1 if (w_idx % 3 == 0) else -2.5,
                "top_ls_ratio": 1.25,
                "retail_ls_ratio": 0.85 if (w_idx % 2 == 0) else 1.65,
            }
            cases.append({
                "case_id": f"SYN_{seed}_{w_idx:02d}",
                "symbol": f"SYN{seed}USDT",
                "df4": df4,
                "df1": df1,
                "df15": df15,
                "ctx4": ctx4,
                "ctx1": ctx1,
                "meta": meta,
                "deriv": deriv,
                "btc": btc,
            })
    return cases

