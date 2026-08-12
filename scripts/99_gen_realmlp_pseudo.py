# -*- coding: utf-8 -*-
"""生成云端 realmlp_pseudo.py: RealMLP 训练改用 PSEUDO 切分 (m0-32/33-70)"""
src = open(r"D:\mscapital-kaggle\scripts\41_realmlp_local.py", encoding="utf-8").read()

# 1. 路径改为 Kaggle
src = src.replace(r"BASE_PATH = r'D:\mscapital-forecasting\data\raw'",
                  "BASE_PATH = '/kaggle/input/competitions/ms-capital-real-financial-market-forecasting'")
src = src.replace(
    'import polars as pl\ntrain = pl.read_parquet(r"D:\\mscapital-forecasting\\data\\processed\\f0726_train.parquet").to_pandas().sort_values(\'sample_id\')',
    'import polars as pl\nimport pandas as pd\ntrain = pl.read_parquet("/kaggle/input/msc-f0726-data/f0726_train_f32.parquet").to_pandas().sort_values(\'sample_id\')\nlabel = pl.read_ipc(f\'{BASE_PATH}/train/label.feather\', memory_map=False).to_pandas().sort_values(\'sample_id\')\ntrain = train.merge(label[[\'sample_id\', \'month\']], on=\'sample_id\', how=\'left\')')
src = src.replace(
    'test = pl.read_parquet(r"D:\\mscapital-forecasting\\data\\processed\\f0726_test.parquet").to_pandas().sort_values(\'sample_id\')',
    'test = pl.read_parquet("/kaggle/input/msc-f0726-data/f0726_test_f32.parquet").to_pandas().sort_values(\'sample_id\')')

# 2. 切分改为 PSEUDO (month)
src = src.replace(
    """train_size = 800000

X_num_train = train[NUMS].iloc[:train_size].values
X_cat_train = train[CATS].iloc[:train_size].values
y_train = train[target_col].iloc[:train_size].round(4).values

X_num_val = train[NUMS].iloc[train_size:].values
X_cat_val = train[CATS].iloc[train_size:].values
y_val = train[target_col].iloc[train_size:].values""",
    """# PSEUDO 切分: m0-32 train / m33-70 val
tr_mask = train['month'] <= 32
va_mask = (train['month'] > 32) & (train['month'] <= 70)
print(f"PSEUDO split: train {tr_mask.sum():,} val {va_mask.sum():,}")

X_num_train = train[NUMS][tr_mask].values
X_cat_train = train[CATS][tr_mask].values
y_train = train[target_col][tr_mask].round(4).values

X_num_val = train[NUMS][va_mask].values
X_cat_val = train[CATS][va_mask].values
y_val = train[target_col][va_mask].values""")

# 3. 保存 PSEUDO pred (在测试集推理前插入, 替换 np.savez test pred 部分)
src = src.replace(
    "np.savez(r'D:\\mscapital-forecasting\\data\\processed\\p12_out\\realmlp_test_pred.npz', pred=test_preds, test_ids=sample_submission['sample_id'].to_numpy())",
    """import numpy as np
# PSEUDO valid pred (best model, EMA applied)
val_model = model if not use_ema else model
val_preds = []
with torch.no_grad():
    for i in range(0, len(X_num_val_tensor), 2048):
        pred = val_model(X_num_val_tensor[i:i+2048], X_cat_val_tensor[i:i+2048], return_codes=False).mean(dim=1).squeeze()
        val_preds.append(pred.cpu().numpy())
val_preds = np.concatenate(val_preds)
np.savez('/kaggle/working/realmlp_pseudo_pred.npz', pred=val_preds, y=y_val)
print(f"PSEUDO pred saved: {val_preds.shape}, cos={cosine_similarity_score(val_preds, y_val):.6f}")""")

# 4. test 预测输出也存
src = src.replace(
    "sample_submission.to_csv(r'D:\\mscapital-kaggle\\output\\submissions\\realmlp_submission.csv', index=False)",
    "np.savez('/kaggle/working/realmlp_test_pred.npz', pred=test_preds, test_ids=sample_submission['sample_id'].to_numpy())\nsample_submission.to_csv('/kaggle/working/realmlp_submission.csv', index=False)")

open(r"D:\mscapital-kaggle\scripts\kaggle_0726ds\realmlp_pseudo.py", "w", encoding="utf-8").write(src)
print("realmlp_pseudo.py generated")
for k in ["PSEUDO split", "msc-f0726-data", "realmlp_pseudo_pred", "month"]:
    print(f"  check '{k}':", k in src)
