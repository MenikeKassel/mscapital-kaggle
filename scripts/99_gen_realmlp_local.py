# -*- coding: utf-8 -*-
"""生成 41_realmlp_local.py: 从 realmlp_source.py 适配本地路径"""
src = open(r"D:\mscapital-forecasting\reference\realmlp_source.py", encoding="utf-8").read()

# 1. BASE_PATH
src = src.replace(
    "BASE_PATH = '/kaggle/input/competitions/ms-capital-real-financial-market-forecasting'",
    "BASE_PATH = r'D:\\mscapital-forecasting\\data\\raw'")

# 2. 数据加载: read_csv → parquet
src = src.replace(
    'train = pd.read_csv("/kaggle/input/notebooks/yunsuxiaozi/rfmf-0726data/train.csv").sort_values(\'sample_id\')',
    'import polars as pl\ntrain = pl.read_parquet(r"D:\\mscapital-forecasting\\data\\processed\\f0726_train.parquet").to_pandas().sort_values(\'sample_id\')')
src = src.replace(
    'test = pd.read_csv("/kaggle/input/notebooks/yunsuxiaozi/rfmf-0726data/test.csv").sort_values(\'sample_id\')',
    'test = pl.read_parquet(r"D:\\mscapital-forecasting\\data\\processed\\f0726_test.parquet").to_pandas().sort_values(\'sample_id\')')

# 3. submission 输出本地 + 保存 test pred npz
src = src.replace(
    "sample_submission = pd.read_csv(f'{BASE_PATH}/submission.csv')",
    "sample_submission = pd.read_csv(f'{BASE_PATH}/submission.csv')\nimport numpy as np\nnp.savez(r'D:\\mscapital-forecasting\\data\\processed\\p12_out\\realmlp_test_pred.npz', pred=test_preds, test_ids=sample_submission['sample_id'].to_numpy())")

# 4. 输出路径
src = src.replace(
    "sample_submission.to_csv('submission.csv', index=False)",
    "sample_submission.to_csv(r'D:\\mscapital-kaggle\\output\\submissions\\realmlp_submission.csv', index=False)")

open(r"D:\mscapital-kaggle\scripts\41_realmlp_local.py", "w", encoding="utf-8").write(src)
print("41_realmlp_local.py generated")
print("checks:", "f0726_train" in src, "D:\\mscapital-forecasting" in src)
