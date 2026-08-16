# Kaggle 提交格式规范（MSCapital）

## 铁律：列名必须是 `sample_id,prediction`

2026-08-16 事故：P10 生产提交用了 `sample_id,target`，Kaggle 评分直接
`FAILED — Evaluation metric raised an unexpected error`。修正为
`sample_id,prediction` 后正常。

## 核对清单（提交前逐项检查）

- [ ] 列名：`sample_id, prediction`（**不是** `target`）
- [ ] 行数：647,896（含表头 647,897 行）
- [ ] sample_id 从 0 连续递增（i64）
- [ ] 无 NaN / Inf（`prediction` 列 null_count = 0）
- [ ] 按 sample_id 排序
- [ ] 精度：默认 float 输出即可（历史提交均未指定格式）

## 参考格式（历史成功提交）

```
sample_id,prediction
0,-0.0024015937628481085
1,0.0011284182871875697
```

## 提交命令（venv kaggle CLI 有 uv trampoline bug，走 Python API）

```bash
cd /d/mscapital-kaggle && python -c "
import sys, os
os.environ['KAGGLE_CONFIG_DIR'] = os.path.expanduser('~/.kaggle')
sys.path.insert(0, '.venv/Lib/site-packages')
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
res = api.competition_submit('OUT.csv', 'MESSAGE', 'ms-capital-real-financial-market-forecasting')
print('SUBMIT OK:', res)
"
```

## 提交状态轮询

```bash
python -c "
import sys, os, time
os.environ['KAGGLE_CONFIG_DIR'] = os.path.expanduser('~/.kaggle')
sys.path.insert(0, '.venv/Lib/site-packages')
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
for i in range(40):
    for s in api.competition_submissions('ms-capital-real-financial-market-forecasting'):
        if s.ref == 'REF':
            print(f'[{i}] status={s.status} score={s.score}')
            if s.status in ('complete', 'error'): sys.exit(0)
    time.sleep(30)
"
```
