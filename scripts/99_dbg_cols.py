# -*- coding: utf-8 -*-
"""调试: test 特征列 vs train feature_names"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\mscapital-kaggle\src")
from mscapital.models.clean_table import load_table_frame, apply_r2

DATA = r"D:\mscapital-forecasting\data\processed"
RAW = r"D:\mscapital-forecasting\data\raw"

frame = load_table_frame(f"{DATA}/train_features.parquet", f"{DATA}/micro_features_train.parquet", f"{RAW}/train/label.feather")
te = pd.read_parquet(f"{DATA}/test_features.parquet")
micro_te = pd.read_parquet(f"{DATA}/micro_features_test.parquet")

official = [c for c in te.columns if c != "sample_id"]
print("official count:", len(official))
print("frame.feature_names count:", len(frame.feature_names))
print("frame names[:5]:", frame.feature_names[:5])
print("frame names[-5:]:", frame.feature_names[-5:])
print("official[:5]:", official[:5])
print("official[-5:]:", official[-5:])
missing = [c for c in frame.feature_names if c not in te.columns]
extra = [c for c in te.columns if c != "sample_id" and c not in frame.feature_names]
print("in frame not in te:", missing[:10], len(missing))
print("in te not in frame:", extra[:10], len(extra))
# R2 检查
from mscapital.models.clean_table import R2_REPLACEMENTS
print("R2_REPLACEMENTS:", list(R2_REPLACEMENTS.keys()))
r2_in_te = [k for k in R2_REPLACEMENTS if k in te.columns]
print("R2 outputs already in te:", r2_in_te)
