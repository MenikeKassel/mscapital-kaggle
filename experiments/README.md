# experiments/ — 实验逻辑层

> 本目录是 MSCapital 实验的 **canonical 逻辑索引**: 每个正式实验一个目录 + README,
> 总台账见 `registry.csv`。物理资产 (scripts/output/docs) 保持原位以维护可运行性,
> README 内以相对路径指向实际位置。

## 阶段目录

- `baseline/` — Baseline 表格阶梯 (2026-08-10/11)
- `P0_protocol/` — P0 Protocol 验证 (2026-08-11)
- `P1_representation/` — P1 表示与序列 (2026-08-11/12)
- `P2_calibration/` — P2 校准 (2026-08-12)
- `C_clean_baseline/` — C 系列 Clean Baseline v2 (2026-08-13)
- `P3_nextgen/` — P3 下一代方法 (2026-08-14)
- `P4_hidden_info/` — P4 隐藏信息调查 (2026-08-13/14)
- `M_representation/` — M 系列残差表示 (2026-08-13)
- `E_conditional/` — E 系列状态条件化 (2026-08-13)
- `P5_market/` — P5 市场探针 (2026-08-14/15)
- `P6_production/` — P6/P6R 生产与检索 (2026-08-14/15)
- `P7_amplitude/` — P7 幅度门控终裁 (2026-08-15)
- `_unclassified/` — 未归属文件暂存
- `_tools/` — 生成工具 (experiment_data.py / build_registry.py)

## 用法

- 阅读: 从 `../docs/experiment-index.md` 进入
- 机器读取: `pd.read_csv('registry.csv')`
- 新增实验: 在 `_tools/experiment_data.py` 加条目 → 重跑 `python _tools/build_registry.py`
