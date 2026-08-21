# MSCapital 项目状态

更新时间：2026-08-21。当前没有运行中的训练或 Kaggle 提交。

## 两条成绩轨

- external-assisted：v8b/lb142，Public LB **0.142**，只作为外部取证。
- self-owned：P6-ORIG，Public LB **0.132**，作为纯原创锚点。

## 冻结资产

- Clean Baseline v2：C4 严格嵌套协议，四折约 0.142649 / 0.141762 / 0.143515 / 0.156924，RMS 生产规则。
- 152+73Z：SCFI/Z 条件线，当前生产特征资产。

## 已关闭或不足

TCN、无监督 latent、幅度 gate、残差检索、NCL/V-REx、M01–M05 均未达到稳定晋级门槛；Cancel 被证明是 Z 的重叠信息。M06 因缺少 train/test 共有的资产/时间键而 `not_identifiable`。E01 四折均正但 PSEUDO 未达 +0.0015，E02/E03 仅诊断。P10 RQ 生产运行因跨历史范围定标不一致而 `protocol_invalid`。

## 当前推荐

科学探索优先 BLSM-G1；工程提分其次是 RealMLP recipe 组合；H7 refit ensemble 只用于降方差。三者在本阶段均不启动，预注册见 `docs/next-direction-2026-08-21.md`。

## 权威链

`CONTEXT.md` → `experiments/registry.csv` → `experiments/routes.yaml` → `submissions/registry.csv` → 本页 → 生成视图 → 单实验报告 → `RESULTS.md`。
