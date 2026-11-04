# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import threading
import time
from typing import Dict, List, Optional, Tuple

import aiohttp
import pandas as pd
import requests

from .models import NAN, STABLE_BASES, Config

BASE_URL = "https://fapi.binance.com"

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "tb_base", "tb_quote", "ignore",
]

FLOAT_KLINE_COLS = ["open", "high", "low", "close", "volume", "quote_volume", "tb_base", "tb_quote"]


def _parse_klines_df(raw: list, server_ms: int) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=KLINE_COLUMNS)
    df = pd.DataFrame(raw, columns=KLINE_COLUMNS)
    for col in FLOAT_KLINE_COLS:
        df[col] = df[col].astype(float)
    df["open_time"] = df["open_time"].astype("int64")
    df["close_time"] = df["close_time"].astype("int64")
    df = df[df["close_time"] < server_ms].reset_index(drop=True)  # buang candle yang belum close
    return df


def _filter_universe(
    info: dict,
    tick_list: list,
    book_list: list,
    prem_list: list,
    cfg: Config,
    only: Optional[List[str]] = None,
) -> Tuple[List[Tuple[str, dict]], int, int]:
    now_ms = int(time.time() * 1000)
    syms: Dict[str, dict] = {}
    for s in info.get("symbols", []):
        if s.get("contractType") != "PERPETUAL" or s.get("quoteAsset") != "USDT" or s.get("status") != "TRADING":
            continue
        if s.get("baseAsset") in STABLE_BASES or s.get("underlyingType") == "INDEX":
            continue
        onboard = s.get("onboardDate", 0)
        if not only and onboard and (now_ms - onboard) / 86_400_000 < cfg.min_age_days:
            continue
        flt = {f["filterType"]: f for f in s.get("filters", [])}
        syms[s["symbol"]] = {
            "base": s["baseAsset"],
            "step": float(flt.get("LOT_SIZE", {}).get("stepSize", 0) or 0),
            "min_notional": float(flt.get("MIN_NOTIONAL", {}).get("notional", 5) or 5),
        }

    tick = {t["symbol"]: t for t in tick_list}
    book = {b["symbol"]: b for b in book_list}
    prem = {p["symbol"]: p for p in prem_list}

    rows = []
    for sym, m in syms.items():
        if only and sym not in only:
            continue
        t, b, p = tick.get(sym), book.get(sym), prem.get(sym)
        if not t or not b or not p:
            continue
        bid, ask = float(b.get("bidPrice") or 0), float(b.get("askPrice") or 0)
        if bid <= 0 or ask <= 0:
            continue
        mid = (bid + ask) / 2.0
        m_out = dict(m)
        m_out["vol24"] = float(t.get("quoteVolume") or 0.0)
        m_out["spread_bps"] = (ask - bid) / mid * 1e4
        m_out["funding"] = float(p.get("lastFundingRate") or 0.0)
        m_out["chg24"] = float(t.get("priceChangePercent") or 0.0)
        rows.append((sym, m_out))

    total = len(rows)
    if only:
        return rows, total, total
    keep = [(s, m) for s, m in rows if m["vol24"] >= cfg.min_vol_usdt and m["spread_bps"] <= cfg.max_spread_bps]
    keep.sort(key=lambda x: x[1]["vol24"], reverse=True)
    return keep[: cfg.top_n], total, len(keep)


def _compute_derivatives_summary(oi_raw: list, top_ls_raw: list, glob_ls_raw: list) -> dict:
    out = {
        "oi_change_1h_pct": NAN,
        "oi_change_4h_pct": NAN,
        "top_ls_ratio": NAN,
        "retail_ls_ratio": NAN,
    }
    if isinstance(oi_raw, list) and len(oi_raw) >= 2:
        try:
            vals = [float(x.get("sumOpenInterestValue") or x.get("sumOpenInterest") or 0.0) for x in oi_raw]
            if vals[-2] > 0:
                out["oi_change_1h_pct"] = (vals[-1] - vals[-2]) / vals[-2] * 100.0
            idx4 = max(0, len(vals) - 5)
            if vals[idx4] > 0:
                out["oi_change_4h_pct"] = (vals[-1] - vals[idx4]) / vals[idx4] * 100.0
        except (ValueError, TypeError):
            pass

    if isinstance(top_ls_raw, list) and top_ls_raw:
        try:
            out["top_ls_ratio"] = float(top_ls_raw[-1].get("longShortRatio") or NAN)
        except (ValueError, TypeError):
            pass

    if isinstance(glob_ls_raw, list) and glob_ls_raw:
        try:
            out["retail_ls_ratio"] = float(glob_ls_raw[-1].get("longShortRatio") or NAN)
        except (ValueError, TypeError):
            pass
    return out


class AsyncBinance:
    """Client HTTP asinkron berkecepatan tinggi untuk endpoint publik Binance USDⓈ-M Futures."""

    def __init__(self, concurrency: int = 16, timeout: int = 15):
        self.concurrency = max(2, concurrency)
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.sem = asyncio.Semaphore(self.concurrency)
        self.used_weight = 0
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "AsyncBinance":
        connector = aiohttp.TCPConnector(limit=max(32, self.concurrency * 2), ttl_dns_cache=300)
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=self.timeout,
            headers={"User-Agent": "mtf-smc-scanner/2.0"},
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def get(self, path: str, params: Optional[dict] = None, soft_fail: bool = False):
        assert self._session is not None, "Gunakan 'async with AsyncBinance() as api'"
        url = BASE_URL + path
        for attempt in range(5):
            if self.used_weight > 2050:
                await asyncio.sleep(6.0)
                self.used_weight = 0
            async with self.sem:
                try:
                    async with self._session.get(url, params=params) as r:
                        w = r.headers.get("X-MBX-USED-WEIGHT-1M")
                        if w and w.isdigit():
                            self.used_weight = int(w)
                        if r.status in (418, 429):
                            wait = int(r.headers.get("Retry-After", (2 ** attempt) * 4))
                            await asyncio.sleep(min(wait, 60))
                            continue
                        if r.status >= 500:
                            await asyncio.sleep(1.5 ** attempt)
                            continue
                        if r.status != 200:
                            if soft_fail:
                                return None
                            r.raise_for_status()
                        return await r.json()
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    if soft_fail and attempt >= 1:
                        return None
                    await asyncio.sleep(1.5 ** attempt)
        if soft_fail:
            return None
        raise RuntimeError(f"Gagal request async {path} setelah beberapa percobaan")

    async def server_time(self) -> int:
        res = await self.get("/fapi/v1/time")
        return int(res["serverTime"])

    async def build_universe(self, cfg: Config, only: Optional[List[str]] = None) -> Tuple[List[Tuple[str, dict]], int, int]:
        info, tick, book, prem = await asyncio.gather(
            self.get("/fapi/v1/exchangeInfo"),
            self.get("/fapi/v1/ticker/24hr"),
            self.get("/fapi/v1/ticker/bookTicker"),
            self.get("/fapi/v1/premiumIndex"),
        )
        return _filter_universe(info, tick, book, prem, cfg, only)

    async def klines(
        self,
        symbol: str,
        interval: str,
        limit: int,
        server_ms: int,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> pd.DataFrame:
        params: dict = {"symbol": symbol, "interval": interval, "limit": min(limit, 1500)}
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(end_time)
        raw = await self.get("/fapi/v1/klines", params)
        return _parse_klines_df(raw, server_ms)

    async def klines_history(self, symbol: str, interval: str, total_bars: int, server_ms: int) -> pd.DataFrame:
        """Mengambil histori klines >1500 bar secara bertahap (untuk backtester)."""
        chunks: List[pd.DataFrame] = []
        rem = total_bars
        end_ms: Optional[int] = server_ms
        while rem > 0:
            batch = min(1500, rem + 2)
            df = await self.klines(symbol, interval, batch, server_ms, end_time=end_ms)
            if df.empty:
                break
            chunks.append(df)
            earliest = int(df["open_time"].iloc[0])
            end_ms = earliest - 1
            rem -= len(df)
            if len(df) < 10:
                break
        if not chunks:
            return pd.DataFrame(columns=KLINE_COLUMNS)
        merged = pd.concat(reversed(chunks), ignore_index=True)
        merged = merged.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
        return merged.tail(total_bars).reset_index(drop=True)

    async def fetch_symbol_klines(self, symbol: str, cfg: Config, server_ms: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df4, df1, df15 = await asyncio.gather(
            self.klines(symbol, "4h", cfg.limit_4h, server_ms),
            self.klines(symbol, "1h", cfg.limit_1h, server_ms),
            self.klines(symbol, "15m", cfg.limit_15m, server_ms),
        )
        return df4, df1, df15

    async def fetch_derivatives(self, symbol: str) -> dict:
        """Mengambil Open Interest History & Long/Short Ratio publik tanpa API key (aman soft-fail)."""
        oi_raw, top_raw, glob_raw = await asyncio.gather(
            self.get("/futures/data/openInterestHist", {"symbol": symbol, "period": "1h", "limit": 6}, soft_fail=True),
            self.get("/futures/data/topLongShortPositionRatio", {"symbol": symbol, "period": "1h", "limit": 2}, soft_fail=True),
            self.get("/futures/data/globalLongShortAccountRatio", {"symbol": symbol, "period": "1h", "limit": 2}, soft_fail=True),
        )
        return _compute_derivatives_summary(oi_raw or [], top_raw or [], glob_raw or [])


class Binance:
    """Client HTTP sinkron (fallback & kompatibilitas mundur)."""

    def __init__(self, timeout: int = 15):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": "mtf-smc-scanner/2.0"})
        self.timeout = timeout
        self.used_weight = 0
        self.lock = threading.Lock()

    def get(self, path: str, params: Optional[dict] = None, soft_fail: bool = False):
        url = BASE_URL + path
        for attempt in range(5):
            with self.lock:
                heavy = self.used_weight > 2050
            if heavy:
                time.sleep(6)
                with self.lock:
                    self.used_weight = 0
            try:
                r = self.s.get(url, params=params, timeout=self.timeout)
            except requests.RequestException:
                if soft_fail and attempt >= 1:
                    return None
                time.sleep(1.8 ** attempt)
                continue
            w = r.headers.get("X-MBX-USED-WEIGHT-1M")
            if w and w.isdigit():
                with self.lock:
                    self.used_weight = int(w)
            if r.status_code in (418, 429):
                wait = int(r.headers.get("Retry-After", (2 ** attempt) * 4))
                time.sleep(min(wait, 90))
                continue
            if r.status_code >= 500:
                time.sleep(1.8 ** attempt)
                continue
            if r.status_code != 200:
                if soft_fail:
                    return None
                r.raise_for_status()
            return r.json()
        if soft_fail:
            return None
        raise RuntimeError(f"Gagal request {path} setelah beberapa percobaan")

    def server_time(self) -> int:
        return int(self.get("/fapi/v1/time")["serverTime"])

    def build_universe(self, cfg: Config, only: Optional[List[str]] = None) -> Tuple[List[Tuple[str, dict]], int, int]:
        info = self.get("/fapi/v1/exchangeInfo")
        tick = self.get("/fapi/v1/ticker/24hr")
        book = self.get("/fapi/v1/ticker/bookTicker")
        prem = self.get("/fapi/v1/premiumIndex")
        return _filter_universe(info, tick, book, prem, cfg, only)

    def klines(
        self,
        symbol: str,
        interval: str,
        limit: int,
        server_ms: int,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> pd.DataFrame:
        params: dict = {"symbol": symbol, "interval": interval, "limit": min(limit, 1500)}
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(end_time)
        raw = self.get("/fapi/v1/klines", params)
        return _parse_klines_df(raw, server_ms)

    def fetch_derivatives(self, symbol: str) -> dict:
        oi_raw = self.get("/futures/data/openInterestHist", {"symbol": symbol, "period": "1h", "limit": 6}, soft_fail=True)
        top_raw = self.get("/futures/data/topLongShortPositionRatio", {"symbol": symbol, "period": "1h", "limit": 2}, soft_fail=True)
        glob_raw = self.get("/futures/data/globalLongShortAccountRatio", {"symbol": symbol, "period": "1h", "limit": 2}, soft_fail=True)
        return _compute_derivatives_summary(oi_raw or [], top_raw or [], glob_raw or [])

