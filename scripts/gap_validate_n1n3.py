# -*- coding: utf-8 -*-
"""EDA v2 缺口 N1+N3 验证: micro_price 特征族 + spread_std → Protocol-v2 残差门禁

缺口来源: docs/eda-v2-2026-08-15.md
- N1: micro_gap_last (micro−mid 末值) vs target +0.050 (f0726 零覆盖, 新方向通道)
- N5: micro_gap_std vs |t| +0.189 (同族)
- N3: spread_std vs |t| +0.265 (f0726 无 spread 波动列)

特征设计 (7 列, 全部 per-sample 确定性聚合, 无跨样本统计量 → 无泄漏, 一份表四 outer 共用):
1. micro_gap_last   (micro_p − mid1) 末 bar
2. micro_gap_mean   per-sample 均值
3. micro_gap_std    per-sample 标准差
4. micro_gap_rel    micro_gap_last / mid1_last (相对量, 漂移鲁棒)
5. spread_std       spread1 标准差
6. spread_mean      spread1 均值
7. spread_rel       spread_std / spread_mean (变异系数, 无量纲)

0 价格哨兵处理 (缺口 B 修复): 聚合前 filter ask>0 & bid>0 (丢 <0.5% 行)。
market 分块 scan (30 万样本/块), 只读 6 列, 峰值内存 <2GB。

管线: save_p3_features → load_p3_frame → run_m01a_outer × 4 → summarize_m01a
门禁: PSEUDO Δ≥+0.0015 ∧ ≥3/4 outer 正 ∧ worst≥-0.0005
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import polars as pl

from mscapital.models.m01a import run_m01a_outer, summarize_m01a
from mscapital.residual import CanonicalOOF

from p3_common import load_p3_frame, save_p3_features

CANONICAL = Path(r"D:\mscapital-kaggle\output\canonical_residual_oof\canonical_residual_oof.npz")
MARKET = Path(r"D:\mscapital-forecasting\data\raw\train\market.feather")
BASELINE_ROOT = Path(r"D:\mscapital-kaggle\output\c4_protocol_closed_final\clean-baseline-v2")
OUT = Path(r"D:\mscapital-kaggle\output\gap_validation_formal")
FEATURE_OUT = Path(r"D:\mscapital-kaggle\output\gap_validation_features")

CHUNK = 300_000  # 每块样本数
SMOKE = int(__import__("os").environ.get("GAP_SMOKE", "0"))

FEATURE_NAMES = (
    "micro_gap_last", "micro_gap_mean", "micro_gap_std", "micro_gap_rel",
    "spread_std", "spread_mean", "spread_rel",
)

MARKET_COLS = [
    "sample_id", "seconds_before_predict",
    "ask_price_1", "bid_price_1", "ask_volume_1", "bid_volume_1",
]


def build_micro_gap_features(canonical: CanonicalOOF) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """分块 scan market → per-sample 聚合, 对齐 canonical 顺序。"""
    sids = canonical.sample_id.astype(np.int64)
    if SMOKE:
        sids = sids[:SMOKE]
    n = len(sids)
    n_chunks = int(np.ceil(n / CHUNK))
    print(f"building micro_gap features: {n} canonical samples, {n_chunks} chunks")

    rows = []
    for i in range(n_chunks):
        chunk_ids = sids[i*CHUNK:(i+1)*CHUNK]
        c = (
            pl.scan_ipc(MARKET)
            .filter(pl.col("sample_id").is_in(chunk_ids))
            .filter(pl.col("ask_price_1") > 0)  # 0 价格哨兵排除 (缺口 B)
            .filter(pl.col("bid_price_1") > 0)
            .select(MARKET_COLS)
            .with_columns([
                ((pl.col("ask_price_1") + pl.col("bid_price_1")) / 2).cast(pl.Float32).alias("mid1"),
                (pl.col("ask_price_1") - pl.col("bid_price_1")).cast(pl.Float32).alias("spread1"),
                ((pl.col("bid_volume_1") * pl.col("ask_price_1")
                  + pl.col("ask_volume_1") * pl.col("bid_price_1"))
                 / (pl.col("bid_volume_1") + pl.col("ask_volume_1") + 1.0)).cast(pl.Float32).alias("micro_p"),
            ])
            .group_by("sample_id")
            .agg([
                (pl.col("micro_p") - pl.col("mid1")).last().alias("micro_gap_last"),
                (pl.col("micro_p") - pl.col("mid1")).mean().alias("micro_gap_mean"),
                (pl.col("micro_p") - pl.col("mid1")).std().alias("micro_gap_std"),
                pl.col("spread1").std().alias("spread_std"),
                pl.col("spread1").mean().alias("spread_mean"),
                pl.col("mid1").last().alias("mid1_last"),
            ])
            .collect(engine="streaming")
        )
        rows.append(c)
        print(f"  chunk {i+1}/{n_chunks} done ({len(c)} samples)", flush=True)

    agg = pl.concat(rows)
    # 对齐 canonical 顺序 (允许个别样本因 0 价格过滤整组消失 → 均值填充)
    agg = agg.sort("sample_id")
    ids = agg["sample_id"].to_numpy().astype(np.int64)
    order = np.searchsorted(ids, sids)
    ok = (order < ids.size) & (ids[np.minimum(order, ids.size-1)] == sids)
    missing = int((~ok).sum())
    if missing:
        print(f"  WARN: {missing} canonical samples missing (0-price filtered) -> mean impute")
    order[~ok] = 0  # 占位, 下面用均值填充

    gap_last = agg["micro_gap_last"].to_numpy()[order]
    gap_mean = agg["micro_gap_mean"].to_numpy()[order]
    gap_std = agg["micro_gap_std"].to_numpy()[order]
    sp_std = agg["spread_std"].to_numpy()[order]
    sp_mean = agg["spread_mean"].to_numpy()[order]
    mid_last = agg["mid1_last"].to_numpy()[order]

    gap_rel = gap_last / np.maximum(mid_last, 1e-12)
    sp_rel = sp_std / np.maximum(sp_mean, 1e-12)

    values = np.column_stack([
        gap_last, gap_mean, gap_std, gap_rel, sp_std, sp_mean, sp_rel,
    ]).astype(np.float32)
    # NaN 兜底 + 缺失样本填充: 全量均值 (确定性, 无泄漏)
    col_mean = np.nanmean(values, axis=0, keepdims=True)
    values = np.where(np.isnan(values), col_mean, values)
    values[~ok] = col_mean

    months = np.asarray(canonical.month)
    targets = np.asarray(canonical.target, dtype=np.float64)
    if SMOKE:
        months = months[:SMOKE]
        targets = targets[:SMOKE]
    return sids, months, targets, values


def main() -> None:
    canonical = CanonicalOOF(**{
        k: np.asarray(np.load(CANONICAL)[k]) for k in
        ("sample_id", "month", "target", "baseline_oof", "source_train_end")
    })
    canonical.validate()
    print(f"canonical: {canonical.sample_id.size} rows, months {canonical.month.min()}-{canonical.month.max()}")

    sids, months, targets, values = build_micro_gap_features(canonical)
    print(f"features: {values.shape}")

    # 一次性打包 (确定性特征, 四 outer 共用)
    feat_path = FEATURE_OUT / "gap_features.parquet"
    save_p3_features(
        feat_path, "gap-n1n3-microgap-spreadstd", FEATURE_NAMES,
        sids, months, targets, values,
        extra={"source": "docs/eda-v2-2026-08-15.md N1/N3/N5", "rows": int(sids.size)},
    )
    frame = load_p3_frame(feat_path, FEATURE_NAMES)
    print(f"frame validated: {frame.values.shape}, names={frame.feature_names}")

    if SMOKE:
        print("SMOKE: feature build validated, skipping outer pipeline")
        return

    # 四 outer 残差验证
    for outer in ("PSEUDO", "H2", "T3", "T4"):
        print(f"\n=== outer {outer} ===", flush=True)
        diag = run_m01a_outer(canonical, frame, BASELINE_ROOT, OUT, outer)
        print(f"  {outer}: delta={diag['delta_vs_baseline']:+.9f} score={diag['final_score']:.9f}")

    summary = summarize_m01a(OUT)
    print("\n=== GAP N1+N3 gate ===")
    for row in summary["rows"]:
        print(f"  {row['outer']}: delta={row['delta_vs_baseline']:+.9f}")
    print(f"mean delta={summary['mean_delta']:+.9f} gate={summary['gate']}")


if __name__ == "__main__":
    main()
