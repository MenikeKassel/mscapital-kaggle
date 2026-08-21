# -*- coding: utf-8 -*-
"""三方对比联合特征表: 152 + 13 cancel + 73 Z (P9-A x Z 叠加).

来源: f0726_train_z_f32.parquet (sample_id+152+73Z+target) join pkgA cancel 13 列.
防泄漏: 两表均为 per-sample 聚合, 无跨样本统计.
输出: output/p9_lite/pkgJ/train_aug.parquet
"""
from __future__ import annotations

import os

import polars as pl

from p9_lite_build import PKG_COLS

Z = r"D:\mscapital-forecasting\data\processed\f0726_train_z_f32.parquet"
CANCEL = r"D:\mscapital-kaggle\output\p9_lite\pkgA\train_aug.parquet"
OUT = r"D:\mscapital-kaggle\output\p9_lite\pkgJ\train_aug.parquet"


def main():
    z = pl.read_parquet(Z)
    c = pl.read_parquet(CANCEL, columns=["sample_id"] + PKG_COLS["A"])
    # 列名去重检查
    zc = set(z.columns); cc = set(c.columns)
    dup = zc & (cc - {"sample_id"})
    assert not dup, f"列名冲突: {dup}"
    j = z.join(c, on="sample_id", how="left")
    assert j.height == z.height, "join broken"
    j = j.with_columns([pl.col(x).fill_null(0.0) for x in PKG_COLS["A"]])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    j.write_parquet(OUT)
    print(f"joint {j.shape} (feats={len([x for x in j.columns if x not in ('sample_id','target','month')])}) -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
