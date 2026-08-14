# MSCapital 交接报告 (Hermes → Codex)

> 交接时间: 2026-08-12 18:20 | 撰写: Hermes | 接收: Codex
> 状态: **LB 0.142, 排名 #30/107**, 提交冻结中(用户拍板: 先本地验证)

## 1. 比赛概况

- **比赛**: MSCapital – Real Financial Market Forecasting(Kaggle 社区赛, 个人赛)
- **指标**: 未中心化 cosine similarity(预测 target 与真实 target 的 cos)
- **截止**: 2026-10-09 | 提交配额: 每天 5 次(太平洋时间重置)
- **目标**: 前 10 ≈ 0.151(当前 0.142, 差 0.009);榜首 0.155
- 数据: 高频订单簿/成交/行情(market 222M 行, order 170M, transaction 104M), 125.8 万 train 样本 / 64.8 万 test 样本, 按月(0-70)时序

## 2. 当前成绩与提交历史

| 版本 | 内容 | PSEUDO | LB |
|---|---|---|---|
| v1-v3 | 官方特征树融合 | 0.129-0.130 | 0.122 |
| v4 | R2 归一化 | 0.1313 | 0.123 |
| v5 | +22 微观特征 | 0.1349 | 0.125 |
| v6 | +TCN(w=0.07) | 0.1379 | **0.082 ❌** |
| v7 | +RealMLP 复刻(w=0.2) | **0.1397** | **0.135** |
| v8a/b | +lb142ref(w=0.3/0.5) | 无法本地验证 | **0.139 / 0.142** |

**三次大台阶**: 0.122 →(R2+micro)0.125 →(事件动力学表示+RealMLP)0.135 →(外部正交 alpha)0.142

## 3. 核心科研资产(文档)

| 文档 | 位置 | 内容 |
|---|---|---|
| RESULTS.md | D:\mscapital-kaggle\ | 全部实验记录 + 负面结果库 N001-N007 |
| docs/calibration.md | 同上 | **两层校准体系 + 提交门禁(必读!)** |
| docs/plan-v1.3.0.md | 同上 | P2 阶段方案(GPT1 评审吸收版) |
| research/METHODS.md | 同上 | 方法原语库(A-F)+ 祖先比赛 + 来源映射 |
| docs/project_report_v3.md | 同上 | 阶段报告 |

## 4. 环境与路径

```
代码: D:\mscapital-kaggle\ (scripts/ + docs/ + research/ + output/)
数据: D:\mscapital-forecasting\data\ (raw/ + processed/ + p12_out/)
      processed/f0726_{train,test}_f32.parquet = 152 事件动力学特征 (float32)
参考: D:\mscapital-forecasting\reference\ (realmlp_source.py, rfmf_0726data_source.py,
      lb142/ 完整开源包, lgb_baseline, salute_rib, transformer)
venv: D:\mscapital-kaggle\.venv (python 3.11, polars/lightgbm/xgboost/catboost/torch2.6+cu124/pyarrow/sklearn)
```

**⚠️ 环境坑(必读)**:
- venv 用**绝对路径**调用: `D:\mscapital-kaggle\.venv\Scripts\python`(`source activate` 会被 Hermes 环境劫持)
- Windows git-bash: `export PYTHONPATH=` 清空(污染); 不要写内联 `python -c`(会被卡); 中文路径转英文
- kaggle CLI 用 `python -m kaggle`(uv trampoline 失效)
- 网络: `export HTTPS_PROXY=http://127.0.0.1:7897`(Clash 代理, Kaggle API 必须)
- 凭证: `KAGGLE_API_TOKEN` 环境变量(账号 kassel menike, token 值在 Hermes 对话中, 不落盘)

## 5. 云端资产(Kaggle 账号 kasselmenike)

| 资源 | 名称 | 用途 |
|---|---|---|
| kernel | msc-f0726-trees | 152特征×树(PSEUDO 0.127, 已 COMPLETE, test pred 已下载) |
| kernel | **msc-realmlp-pseudo v12** | **RealMLP PSEUDO 已 COMPLETE；最佳 EMA 产物已下载** |
| dataset | msc-f0726-pq (v1) | f0726 parquet(挂载成功的关键: 全新 dataset) |
| dataset | msc-f0726-data / msc-0726-featbuild | 旧 dataset(v2+ 挂载有坑, 勿用) |
| dataset | kasselmenike/msc-realmlp-code | 废弃 |

**⚠️ 云端坑(9 轮调试血泪)**:
1. dataset 版本更新(v2+)在 kernel 挂载时**可能丢文件** → 用全新 dataset(v1)最稳
2. 多 dataset 挂载可能只挂一个 → 单 dataset 挂载
3. 代码 exec dataset 里的 py 不可靠 → **自包含代码**(kernel 主文件)
4. Kaggle 分配 **P100**(sm_60)时预装 torch 不支持 → 自动降级 torch==2.2.2 + numpy<2；必须用版本幂等保护避免 `execv` 无限重启(已内置 v12)
5. train join label 的 month 列必须从特征排除(KeyError)

## 6. 已完成的关键接手任务

**realmlp-pseudo v12 已 COMPLETE**:
- 产出: `output/rlps_v12/realmlp_pseudo_pred.npz`(PSEUDO m0-32/33-70，最佳 EMA)
- RealMLP PSEUDO = 0.138560；v5 表格重放 = 0.134871
- v7 (`0.8×table + 0.2×RealMLP`) PSEUDO = **0.139683**
- Regime B gap = PSEUDO − LB = **0.004683**
- 诊断: `output/rlps_v12/v7_pseudo_diagnostics.json`

## 7. 待办清单(按优先级)

1. **【已完成】realmlp-pseudo v12 → 下载 pred → v7 PSEUDO=0.139683 → calibration.md Regime B 已更新**
2. **三路融合验证**(文件已生成 output/submissions/submission_v9_t5r30 ~ v9_t10r50): v7 × f0726树 × lb142ref
   - 已知: corr(f0726树, ref)=0.776(最低, 独立信息); corr(v7, ref)=0.823
   - 决策: 是否提交 1 次(用户拍板!当前冻结)
3. **P2 路线**(plan-v1.3.0): Dynamics V2 特征工厂(多尺度 diff/event intensity/velocity)、Market State KNN(替代 alpha)
4. **提交门禁检查**(calibration.md Gate 1-5)通过后才占用配额

## 8. 决策与纪律(用户明确)

- **提交冻结**: 只有本地验证达标(PSEUDO 提升 + 分布/尺度/corr 门禁全过)才提交;不拿 Public LB 当验证集(权重扫描 0.6/0.7 已明确禁止)
- 目标 0.145 → PSEUDO 需 ≥ 0.155(Regime A)
- 序列模型(TCN)路线冻结;新模型必须过 4 重门禁
- 用户偏好: 先看效果再提交; 长任务上 Kaggle 云端(本地轻载); 科研纪律(每步验证, 负面结果入库)
- 外部预测(lb142ref)只作 ensemble probe, 不进入本地校准

## 9. 快速参考(常用命令)

```bash
# 查提交/榜单
cd /d/mscapital-kaggle && export KAGGLE_API_TOKEN=$TOKEN && export HTTPS_PROXY=http://127.0.0.1:7897
D:/mscapital-kaggle/.venv/Scripts/python -m kaggle competitions submissions -c ms-capital-real-financial-market-forecasting
# 提交
... competitions submit -c ... -f <csv> -m "<msg>"
# 云端 kernel
... kernels status kasselmenike/msc-realmlp-pseudo
... kernels output kasselmenike/msc-realmlp-pseudo -p output/xxx
# 本地验证
export PYTHONPATH= && D:/mscapital-kaggle/.venv/Scripts/python scripts/XX.py
```

## 10. 下一步建议(给 Codex)

1. 先确认 realmlp-pseudo 状态(重点!)
2. 完成 v7 PSEUDO 定标 → 更新校准表
3. 三路融合: 用 v9 文件提交 1 次验证(需用户拍板)
4. 然后按 plan-v1.3.0 推进 P2(Dynamics V2 特征工厂最优先, 40% 资源)

## 11. Codex 接手更新 (2026-08-12 19:25)

- `msc-realmlp-pseudo` v10 实际处于 P100 torch 降级无限重启；已取消。
- v11 修复版本幂等保护后发现 PSEUDO 产物口径仍错误（无关全量训练在前、未恢复最佳 EMA）；已取消。
- v12 已修复为 PSEUDO-only + best EMA，Kaggle 状态 **COMPLETE**。
- RealMLP PSEUDO = **0.138560**；v5 表格重放 = **0.134871**。
- 实际 v7 (`0.8 table + 0.2 RealMLP`) PSEUDO = **0.139683**；LB=0.135，Regime B gap=**0.004683**。
- valid/test corr(table, RealMLP)=0.8658/0.8263；v7 test/valid std=0.7447，存在幅度迁移。
- 未进行任何新比赛提交；提交冻结仍有效。
- 下一步从待办 2/3 继续：先做 ref forensic / Dynamics V2，不将本地 raw 最优权重 0.03 直接转化为提交。
