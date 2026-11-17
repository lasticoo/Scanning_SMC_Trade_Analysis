# -*- coding: utf-8 -*-
"""
Modul Pelaporan Performa Berbasis Bukti (Evidence Performance Report)
=====================================================================
Membaca `setups_audit.jsonl`, menggabungkannya secara aman (left join) dengan
hasil simulasi `tracker.py`, dan menyajikan tabel statistik performa (Win Rate,
Wilson 95% CI, Avg R, Net R, Profit Factor) per kategori & flag bukti.
Hanya menggunakan pustaka standar Python, NumPy, dan Pandas (100% read-only/analitik).
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

CLOSED_OUTCOMES = ("TP2_FULL", "TP1_BEP", "LOSS_SL", "TIME_EXIT")
_VERSION_MATCH_CACHE: Dict[Tuple[str, str], bool] = {}


def _normalize_flag_name(raw_flag: str) -> str:
    """Buang bagian dalam tanda kurung dari nama flag, mis. 'POI_MULTIPLE_VISITS_24H(6x)' -> 'POI_MULTIPLE_VISITS_24H'."""
    if not raw_flag:
        return ""
    cleaned = re.sub(r"\([^)]*\)", "", str(raw_flag)).strip()
    return cleaned


def normalize_flags_list(flags: Any) -> str:
    """Normalisasi daftar atau string flag menjadi string unik dipisah ';'."""
    if flags is None:
        return ""
    if isinstance(flags, str):
        raw_items = [x.strip() for x in flags.split(";") if x.strip()]
    elif isinstance(flags, (list, tuple, set)):
        raw_items = [str(x).strip() for x in flags if str(x).strip()]
    else:
        return ""

    seen: List[str] = []
    for item in raw_items:
        norm = _normalize_flag_name(item)
        if norm and norm.upper() != "NONE" and norm not in seen:
            seen.append(norm)
    return ";".join(seen)


def wilson_ci_95(wins: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """Hitung interval kepercayaan Wilson 95% (dalam persen 0..100) untuk proporsi wins/n."""
    if n <= 0:
        return 0.0, 0.0
    k = max(0, min(int(wins), int(n)))
    p = k / float(n)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    rad = max(0.0, (p * (1.0 - p)) / n + z2 / (4.0 * n * n))
    half = (z * math.sqrt(rad)) / denom
    lo = max(0.0, (center - half) * 100.0)
    hi = min(100.0, (center + half) * 100.0)
    return round(lo, 2), round(hi, 2)


def load_audit_jsonl(scan_dir: str, filename: str = "setups_audit.jsonl") -> pd.DataFrame:
    """
    Membaca file JSONL audit secara toleran:
    - File tidak ada -> DataFrame kosong.
    - Baris rusak dilewati dan dihitung di df.attrs['corrupt_lines'].
    - Flatten setiap kunci skalar (int/float/bool/str) di record['evidence'] (kecuali 'text')
      menjadi kolom berawalan 'ev_'.
    - Normalisasi 'flags' menjadi 'flag_names' (tanpa bagian dalam kurung, digabung ';').
    - Pertahankan 'python_status' dan 'llm_status'.
    - Duplikat pada (scan_utc, symbol, direction) disimpan yang terakhir (keep='last').
    """
    path = os.path.join(scan_dir, filename)
    if not os.path.isfile(path):
        empty_df = pd.DataFrame()
        empty_df.attrs["corrupt_lines"] = 0
        return empty_df

    rows: List[Dict[str, Any]] = []
    corrupt_lines = 0

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                s = line.strip()
                if not s:
                    continue
                try:
                    rec = json.loads(s)
                    if not isinstance(rec, dict):
                        corrupt_lines += 1
                        continue
                except Exception:
                    corrupt_lines += 1
                    continue

                scan_utc = str(rec.get("scan_utc") or "").strip()
                symbol = str(rec.get("symbol") or "").strip()
                direction = str(rec.get("direction") or "").strip()
                if not scan_utc or not symbol or not direction:
                    corrupt_lines += 1
                    continue

                cv = str(rec.get("code_version") or "").strip()
                ch = str(rec.get("cfg_hash") or "").strip()
                flat: Dict[str, Any] = {
                    "scan_utc": scan_utc,
                    "symbol": symbol,
                    "direction": direction,
                    "python_status": rec.get("python_status"),
                    "llm_status": rec.get("llm_status"),
                    "code_version": cv if cv else "legacy",
                    "cfg_hash": ch,
                }

                ev = rec.get("evidence")
                if isinstance(ev, dict):
                    for k, v in ev.items():
                        if k == "text":
                            continue
                        if v is None or isinstance(v, (int, float, bool, str)):
                            flat[f"ev_{k}"] = v
                    # Sediakan alias standar jika memakai nama dari evidence.py
                    if "killzone_utc" in ev and "ev_killzone" not in flat:
                        flat["ev_killzone"] = ev.get("killzone_utc")
                    if "btc_4h_trend" in ev and "ev_btc_4h" not in flat:
                        flat["ev_btc_4h"] = ev.get("btc_4h_trend")
                    if "er_1h_24" in ev and "ev_er_1h" not in flat:
                        flat["ev_er_1h"] = ev.get("er_1h_24")

                raw_flags = rec.get("flags")
                if raw_flags is None and isinstance(ev, dict):
                    raw_flags = ev.get("flags")
                flat["flag_names"] = normalize_flags_list(raw_flags)
                rows.append(flat)
    except Exception:
        empty_df = pd.DataFrame()
        empty_df.attrs["corrupt_lines"] = corrupt_lines
        return empty_df

    if not rows:
        empty_df = pd.DataFrame()
        empty_df.attrs["corrupt_lines"] = corrupt_lines
        return empty_df

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["scan_utc", "symbol", "direction"], keep="last").reset_index(drop=True)
    df.attrs["corrupt_lines"] = corrupt_lines
    return df


def enrich_trades(res_df: pd.DataFrame, audit_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left join res_df dengan audit_df pada (scan_utc, symbol, direction):
    - Tidak ada baris trade yang hilang atau berganda.
    - Menambahkan kolom bool 'evidence_joined' di akhir bersama kolom audit hasil join.
    - Baris tanpa scan_utc atau tanpa pasangan JSONL tetap ada dengan evidence_joined=False.
    """
    if res_df is None or res_df.empty:
        out = res_df.copy() if isinstance(res_df, pd.DataFrame) else pd.DataFrame()
        out["evidence_joined"] = pd.Series(dtype=bool)
        return out

    orig_cols = list(res_df.columns)
    key_cols = ["scan_utc", "symbol", "direction"]

    if (
        audit_df is None
        or audit_df.empty
        or not all(k in res_df.columns for k in key_cols)
        or not all(k in audit_df.columns for k in key_cols)
    ):
        out = res_df.copy()
        out["evidence_joined"] = False
        if "code_version" not in out.columns:
            out["code_version"] = "legacy"
        if "cfg_hash" not in out.columns:
            out["cfg_hash"] = ""
        return out

    # Pastikan audit_df unik pada key_cols dan bebas dari scan_utc kosong
    aud = audit_df.copy()
    aud["scan_utc"] = aud["scan_utc"].fillna("").astype(str).str.strip()
    aud["symbol"] = aud["symbol"].fillna("").astype(str).str.strip()
    aud["direction"] = aud["direction"].fillna("").astype(str).str.strip()
    aud = aud[(aud["scan_utc"] != "") & (aud["symbol"] != "") & (aud["direction"] != "")]
    aud = aud.drop_duplicates(subset=key_cols, keep="last").copy()

    if aud.empty:
        out = res_df.copy()
        out["evidence_joined"] = False
        if "code_version" not in out.columns:
            out["code_version"] = "legacy"
        if "cfg_hash" not in out.columns:
            out["cfg_hash"] = ""
        return out

    # Hindari tabrakan nama kolom non-key yang sudah ada di res_df
    extra_cols = [c for c in aud.columns if c not in key_cols and c not in orig_cols]
    aud_sub = aud[key_cols + extra_cols].copy()
    aud_sub["_audit_matched_flag"] = True

    left = res_df.copy()
    left["_join_scan_utc"] = left["scan_utc"].fillna("").astype(str).str.strip()
    left["_join_symbol"] = left["symbol"].fillna("").astype(str).str.strip()
    left["_join_direction"] = left["direction"].fillna("").astype(str).str.strip()

    aud_sub = aud_sub.rename(
        columns={
            "scan_utc": "_join_scan_utc",
            "symbol": "_join_symbol",
            "direction": "_join_direction",
        }
    )

    merged = pd.merge(
        left,
        aud_sub,
        on=["_join_scan_utc", "_join_symbol", "_join_direction"],
        how="left",
        sort=False,
    )

    valid_key_mask = (merged["_join_scan_utc"] != "") & (merged["_join_symbol"] != "") & (merged["_join_direction"] != "")
    merged["evidence_joined"] = merged["_audit_matched_flag"].notna() & valid_key_mask

    if "code_version" in merged.columns:
        merged["code_version"] = (
            merged["code_version"].fillna("legacy").astype(str).str.strip().replace({"": "legacy", "nan": "legacy", "None": "legacy"})
        )
    else:
        merged["code_version"] = "legacy"
        if "code_version" not in orig_cols:
            extra_cols.append("code_version")
    if "cfg_hash" in merged.columns:
        merged["cfg_hash"] = merged["cfg_hash"].fillna("").astype(str).str.strip().replace({"nan": "", "None": ""})

    drop_temp = ["_join_scan_utc", "_join_symbol", "_join_direction", "_audit_matched_flag"]
    merged = merged.drop(columns=drop_temp)

    # Pastikan urutan kolom: orig_cols tetap identik di depan, diikuti evidence_joined dan extra_cols di akhir
    final_cols = orig_cols + ["evidence_joined"] + [c for c in extra_cols if c in merged.columns]
    return merged[final_cols]


def version_matches(code_ver: str, since_version: str) -> bool:
    """
    Cek apakah code_version sebuah baris cocok dengan filter `--since-version <sha>`:
    - Baris tanpa versi ("legacy" / "nogit") tidak lolos filter SHA nyata.
    - Cocok persis (case-insensitive) atau prefix SHA >= 4 karakter, atau
    - Jika keduanya adalah commit git valid, lolos bila `since_version` merupakan ancestor dari `code_ver`.
    """
    sv = str(since_version or "").strip()
    if not sv:
        return True
    cv = str(code_ver or "").strip() or "legacy"
    key = (cv, sv)
    if key in _VERSION_MATCH_CACHE:
        return _VERSION_MATCH_CACHE[key]

    res = False
    if cv == sv:
        res = True
    elif cv in ("legacy", "nogit"):
        res = False
    elif len(sv) >= 4 and len(cv) >= 4 and (cv.lower().startswith(sv.lower()) or sv.lower().startswith(cv.lower())):
        res = True
    else:
        try:
            rc = subprocess.run(
                ["git", "merge-base", "--is-ancestor", sv, cv],
                cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3,
            ).returncode
            res = rc == 0
        except Exception:
            res = False

    _VERSION_MATCH_CACHE[key] = res
    return res


def filter_trades_by_version(df: pd.DataFrame, since_version: str) -> pd.DataFrame:
    """Filter DataFrame berdasarkan kolom code_version menggunakan `version_matches`."""
    sv = str(since_version or "").strip()
    if not sv or df is None or df.empty:
        return df
    if "code_version" not in df.columns:
        return df.iloc[0:0].copy()
    mask = df["code_version"].apply(lambda v: version_matches(str(v), sv))
    return df[mask].reset_index(drop=True)


def _compute_stats(sub_df: pd.DataFrame) -> Dict[str, Any]:
    n = len(sub_df)
    if n == 0:
        return {
            "n": 0,
            "wins": 0,
            "wr_pct": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "avg_r": 0.0,
            "net_r": 0.0,
            "pf": 0.0,
        }
    r_vals = sub_df["realized_r"].to_numpy(dtype=float, copy=False)
    wins = int(np.sum(r_vals > 0))
    wr_pct = (wins / float(n)) * 100.0
    ci_low, ci_high = wilson_ci_95(wins, n)
    avg_r = float(np.mean(r_vals))
    net_r = float(np.sum(r_vals))
    gross_win = float(np.sum(r_vals[r_vals > 0]))
    gross_loss = float(np.abs(np.sum(r_vals[r_vals <= 0])))
    pf = gross_win / gross_loss if gross_loss > 0 else (99.9 if gross_win > 0 else 0.0)
    return {
        "n": n,
        "wins": wins,
        "wr_pct": wr_pct,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "avg_r": avg_r,
        "net_r": net_r,
        "pf": pf,
    }


def _fmt_stat_row(label: str, st: Dict[str, Any], min_n: int) -> str:
    small_note = "  (sampel kecil, jangan disimpulkan)" if st["n"] < min_n else ""
    return (
        f"  {label:<34s} | n={st['n']:>4d} | WR={st['wr_pct']:5.1f}% "
        f"[Wilson95: {st['ci_low']:5.1f}%-{st['ci_high']:5.1f}%] | "
        f"Avg R={st['avg_r']:+6.2f}R | Net R={st['net_r']:+7.2f}R | PF={st['pf']:5.2f}{small_note}"
    )


def _is_in_killzone(val: Any) -> bool:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return False
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    s = str(val).strip().upper()
    if not s or s in ("FALSE", "0", "NO", "TIDAK", "NONE", "NAN", "UNKNOWN"):
        return False
    if s.startswith("OFF_KILLZONE") or s.startswith("ASIAN_SESSION"):
        return False
    return True


def _btc_alignment_label(btc_val: Any, direction_val: Any) -> Optional[str]:
    if btc_val is None or direction_val is None:
        return None
    if isinstance(btc_val, float) and math.isnan(btc_val):
        return None
    b = str(btc_val).strip().upper()
    d = str(direction_val).strip().upper()
    if not b or not d or b in ("NAN", "NONE"):
        return None
    is_bull = b in ("BULLISH", "BULL", "1", "UP")
    is_bear = b in ("BEARISH", "BEAR", "-1", "DOWN")
    if not is_bull and not is_bear:
        return "NETRAL_BTC_4H"
    if (d == "LONG" and is_bull) or (d == "SHORT" and is_bear):
        return "SEARAH_BTC_4H"
    if (d == "LONG" and is_bear) or (d == "SHORT" and is_bull):
        return "BERLAWANAN_BTC_4H"
    return "NETRAL_BTC_4H"


def build_evidence_report(trades_df: pd.DataFrame, min_n: int = 30) -> str:
    """
    Menyusun laporan performa berbasis bukti untuk seluruh trade berstatus closed
    (TP2_FULL, TP1_BEP, LOSS_SL, TIME_EXIT).
    Jika trades_df berasal dari folder tanpa setups_audit.jsonl (evidence_joined semua False
    dan tidak ada kolom bukti/flag), kembalikan string kosong agar output tracker lama tidak berubah.
    """
    if trades_df is None or trades_df.empty:
        return ""

    has_ev_cols = any(c.startswith("ev_") or c in ("flag_names", "python_status", "llm_status") for c in trades_df.columns)
    if "evidence_joined" in trades_df.columns and not bool(trades_df["evidence_joined"].any()) and not has_ev_cols:
        return ""

    if "outcome" not in trades_df.columns or "realized_r" not in trades_df.columns:
        return "[Evidence Report] Kolom 'outcome' atau 'realized_r' tidak tersedia — dilewati."

    closed = trades_df[trades_df["outcome"].isin(CLOSED_OUTCOMES)].copy()
    n_closed = len(closed)
    n_joined = (
        int(closed["evidence_joined"].fillna(False).astype(bool).sum())
        if "evidence_joined" in closed.columns
        else n_closed
    )

    lines: List[str] = [
        "=" * 108,
        f"EVIDENCE PERFORMANCE REPORT (CLOSED TRADES: {n_closed} | EVIDENCE JOINED: {n_joined} | MIN SAMPEL: {min_n})",
        "=" * 108,
    ]

    if n_closed == 0:
        lines.append("Belum ada trade berstatus closed (TP2_FULL / TP1_BEP / LOSS_SL / TIME_EXIT) untuk dianalisis.")
        lines.append(
            "Footer: Korelasi bukan kausalitas; banyak perbandingan menghasilkan temuan palsu; jangan dijadikan aturan otomatis."
        )
        lines.append("=" * 108)
        return "\n".join(lines)

    # 0. Baseline Keseluruhan
    base_st = _compute_stats(closed)
    lines.append("[0] BASELINE KESELURUHAN (CLOSED TRADES):")
    lines.append(_fmt_stat_row("ALL_CLOSED", base_st, min_n))
    r_all = closed["realized_r"].to_numpy(dtype=float, copy=False)
    if len(r_all) > 10:
        r_ex10 = np.sort(r_all)[:-10]
        lines.append(
            f"  {'EX_TOP10':<34s} | n={len(r_ex10):>4d} | Expectancy={float(np.mean(r_ex10)):+6.2f}R | Net R={float(np.sum(r_ex10)):+7.2f}R"
        )
    else:
        lines.append(
            f"  {'EX_TOP10':<34s} | n={len(r_all):>4d} | <= 10 trade closed (butuh >10 untuk ex-top10)"
        )
    if "planned_rr" in closed.columns:
        tp2_sub = closed[closed["outcome"] == "TP2_FULL"]
        p_rr = pd.to_numeric(tp2_sub["planned_rr"], errors="coerce") if not tp2_sub.empty else pd.Series(dtype=float)
        r_rr = pd.to_numeric(tp2_sub["realized_r"], errors="coerce") if not tp2_sub.empty else pd.Series(dtype=float)
        valid_tp2 = p_rr.notna() & r_rr.notna()
        if valid_tp2.any():
            med_diff = float((p_rr[valid_tp2] - r_rr[valid_tp2]).median())
            lines.append(
                f"  {'TP2 realized vs planned':<34s} | n={int(valid_tp2.sum()):>4d} | median(planned_rr - realized_r) = {med_diff:+.3f} R"
            )

    def _add_categorical_section(sec_title: str, col_name: str) -> None:
        lines.append(f"\n{sec_title}:")
        if col_name not in closed.columns:
            lines.append(f"  [Catatan] Kolom '{col_name}' tidak tersedia — bagian ini dilewati.")
            return
        ser = closed[col_name].fillna("").astype(str).str.strip()
        valid_mask = (ser != "") & (ser.str.upper() != "NAN") & (ser.str.upper() != "NONE")
        if not valid_mask.any():
            lines.append(f"  [Catatan] Tidak ada data valid pada kolom '{col_name}' — dilewati.")
            return
        sub = closed[valid_mask].copy()
        sub["_grp"] = ser[valid_mask]
        for grp_val in sorted(sub["_grp"].unique()):
            grp_df = sub[sub["_grp"] == grp_val]
            lines.append(_fmt_stat_row(str(grp_val), _compute_stats(grp_df), min_n))

    # 1. Archetype
    _add_categorical_section("[1] PER ARCHETYPE", "archetype")

    # 2. Session
    _add_categorical_section("[2] PER SESSION", "session")

    # 3. AI Decision
    _add_categorical_section("[3] PER AI DECISION", "ai_decision")

    # 4. Bucket AI Confidence (<70, 70-79, 80-89, >=90)
    lines.append("\n[4] PER BUCKET AI CONFIDENCE (<70, 70-79, 80-89, >=90):")
    conf_col = "ai_confidence" if "ai_confidence" in closed.columns else ("llm_score" if "llm_score" in closed.columns else None)
    if conf_col is None:
        lines.append("  [Catatan] Kolom 'ai_confidence' tidak tersedia — bagian ini dilewati.")
    else:
        conf_num = pd.to_numeric(closed[conf_col], errors="coerce")
        if conf_num.notna().any():
            buckets = [
                ("<70", conf_num < 70.0),
                ("70-79", (conf_num >= 70.0) & (conf_num < 80.0)),
                ("80-89", (conf_num >= 80.0) & (conf_num < 90.0)),
                (">=90", conf_num >= 90.0),
            ]
            for b_lbl, b_mask in buckets:
                b_df = closed[b_mask & conf_num.notna()]
                if len(b_df) > 0:
                    lines.append(_fmt_stat_row(f"CONF_{b_lbl}", _compute_stats(b_df), min_n))
        else:
            lines.append("  [Catatan] Tidak ada nilai numerik pada 'ai_confidence' — dilewati.")

    # 5. AI Model
    _add_categorical_section("[5] PER AI MODEL", "ai_model")

    # 6. Python Status
    _add_categorical_section("[6] PER PYTHON STATUS", "python_status")

    # 7. Killzone (ev_killzone / ev_killzone_utc ada vs tidak)
    lines.append("\n[7] KILLZONE (ADA vs TIDAK ADA):")
    kz_col = "ev_killzone" if "ev_killzone" in closed.columns else ("ev_killzone_utc" if "ev_killzone_utc" in closed.columns else None)
    if kz_col is None:
        lines.append("  [Catatan] Kolom 'ev_killzone' / 'ev_killzone_utc' tidak tersedia — bagian ini dilewati.")
    else:
        kz_valid = closed[closed[kz_col].notna()].copy()
        if kz_valid.empty:
            lines.append(f"  [Catatan] Tidak ada data pada kolom '{kz_col}' — dilewati.")
        else:
            kz_bool = kz_valid[kz_col].apply(_is_in_killzone)
            for lbl, mask_b in (("KILLZONE_ADA", kz_bool), ("KILLZONE_TIDAK_ADA", ~kz_bool)):
                sub_k = kz_valid[mask_b]
                if len(sub_k) > 0:
                    lines.append(_fmt_stat_row(lbl, _compute_stats(sub_k), min_n))

    # 8. Searah / Berlawanan BTC 4H
    lines.append("\n[8] KESELARASAN ARAH TRADE vs BTC 4H:")
    btc_col = "ev_btc_4h" if "ev_btc_4h" in closed.columns else ("ev_btc_4h_trend" if "ev_btc_4h_trend" in closed.columns else None)
    if btc_col is None or "direction" not in closed.columns:
        lines.append("  [Catatan] Kolom 'ev_btc_4h' / 'direction' tidak tersedia — bagian ini dilewati.")
    else:
        align_ser = closed.apply(lambda r: _btc_alignment_label(r.get(btc_col), r.get("direction")), axis=1)
        valid_al = align_ser.notna()
        if not valid_al.any():
            lines.append("  [Catatan] Tidak ada data valid pada keselarasan BTC 4H — dilewati.")
        else:
            for al_lbl in ("SEARAH_BTC_4H", "BERLAWANAN_BTC_4H", "NETRAL_BTC_4H"):
                al_df = closed[align_ser == al_lbl]
                if len(al_df) > 0:
                    lines.append(_fmt_stat_row(al_lbl, _compute_stats(al_df), min_n))

    # 9. Bucket Efficiency Ratio 1H (<0.20, 0.20-0.35, >=0.35)
    lines.append("\n[9] PER BUCKET EFFICIENCY RATIO 1H (<0.20, 0.20-0.35, >=0.35):")
    er_col = "ev_er_1h" if "ev_er_1h" in closed.columns else ("ev_er_1h_24" if "ev_er_1h_24" in closed.columns else None)
    if er_col is None:
        lines.append("  [Catatan] Kolom 'ev_er_1h' / 'ev_er_1h_24' tidak tersedia — bagian ini dilewati.")
    else:
        er_num = pd.to_numeric(closed[er_col], errors="coerce")
        if er_num.notna().any():
            er_buckets = [
                ("ER1H_<0.20", er_num < 0.20),
                ("ER1H_0.20-0.35", (er_num >= 0.20) & (er_num < 0.35)),
                ("ER1H_>=0.35", er_num >= 0.35),
            ]
            for e_lbl, e_mask in er_buckets:
                e_df = closed[e_mask & er_num.notna()]
                if len(e_df) > 0:
                    lines.append(_fmt_stat_row(e_lbl, _compute_stats(e_df), min_n))
        else:
            lines.append("  [Catatan] Tidak ada nilai numerik pada Efficiency Ratio 1H — dilewati.")

    # 10. Per Flag Bukti (flag ADA vs flag TIDAK ADA)
    lines.append("\n[10] PER FLAG BUKTI (FLAG ADA vs FLAG TIDAK ADA):")
    if "flag_names" not in closed.columns:
        lines.append("  [Catatan] Kolom 'flag_names' tidak tersedia — bagian ini dilewati.")
    else:
        fl_valid = closed[closed["flag_names"].notna()].copy()
        if fl_valid.empty:
            lines.append("  [Catatan] Tidak ada baris dengan data 'flag_names' — dilewati.")
        else:
            parsed_sets: List[Set[str]] = []
            all_flags: Set[str] = set()
            for raw_f in fl_valid["flag_names"]:
                norm_str = normalize_flags_list(raw_f)
                fset = {x for x in norm_str.split(";") if x}
                parsed_sets.append(fset)
                all_flags.update(fset)

            if not all_flags:
                lines.append("  [Catatan] Tidak ada flag aktif yang tercatat pada sampel closed trades.")
            else:
                for fname in sorted(all_flags):
                    has_mask = np.array([fname in fset for fset in parsed_sets], dtype=bool)
                    df_yes = fl_valid[has_mask]
                    df_no = fl_valid[~has_mask]
                    lines.append(_fmt_stat_row(f"{fname} ADA", _compute_stats(df_yes), min_n))
                    lines.append(_fmt_stat_row(f"{fname} TIDAK ADA", _compute_stats(df_no), min_n))

    lines.append("-" * 108)
    lines.append(
        "Footer: Korelasi bukan kausalitas; banyak perbandingan menghasilkan temuan palsu; jangan dijadikan aturan otomatis."
    )
    lines.append("=" * 108)
    return "\n".join(lines)
