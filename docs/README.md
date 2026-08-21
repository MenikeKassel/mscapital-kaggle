# MSCapital 文档索引

## 权威链

`CONTEXT.md` → `experiments/registry.csv` → `experiments/routes.yaml` → `submissions/registry.csv` → [`project-status.md`](project-status.md) → 生成视图 → 单实验报告 → `RESULTS.md` 历史日志。

## 当前视图

- [`all-experiments.md`](all-experiments.md)：全部 116 条实验
- [`experiment-index.md`](experiment-index.md)：一次性时间线
- [`experiment-lineage.md`](experiment-lineage.md)：唯一 DAG
- [`method-map.md`](method-map.md)：20 条主路线状态
- [`methods-tried-zh.md`](methods-tried-zh.md)：大白话路线总结
- [`failed-experiments.md`](failed-experiments.md)：按失败机制汇总
- [`current-research-queue.md`](current-research-queue.md)：只列 active/candidate
- [`next-direction-2026-08-21.md`](next-direction-2026-08-21.md)：下一方向预注册

## 维护

运行 `python experiments/_tools/build_project_views.py` 重建视图，运行 `--check` 检查重复生成稳定性。实验事实和路线决策先写入 CSV/YAML，再更新生成视图；不把历史计划文档当作当前状态来源。

旧计划、实验总览和方法调研文档保留在仓库中作为历史快照，不删除、不承担 current 权威。
