# 下一方向预注册（2026-08-21）

本阶段只做研究整理，不运行模型、不生成预测、不提交 Kaggle。

## 选择

1. **BLSM-G1 新信息验证（科学探索首选）**：G0 的 behavior state 与 activity、volatility、month 近似独立，尚未证明能提升预测。G1 使用相同 split、RealMLP、seed 和预处理，对比 `152` 与 `152+z_B`；scaler/PCA 必须 fold-local，不能复用 G0 全数据 PCA。以全部 `1,257,637` 个 label ID 左连接，缺失路径填零并加 missing indicator。`Δ<+0.0003` 且无月度方向则关闭；`+0.0003～+0.0008` 记为不足；`>+0.0008` 且多数月份为正才进入 G2。
2. **RealMLP recipe 组合（工程提分第二顺位）**：ParamMish、PL/PBLD、schedreg 和 coslog4 的单项结果是 candidate 证据，但必须重新做同协议组合与消融。
3. **H7 refit ensemble（利用现有信息降低方差）**：优先作为 exploitation，不把它宣称为新信息源，也不在本阶段冻结生产权重。
4. **Z 之外的 flow×market-response 条件特征**：只有在前述路线关闭或授权后再立项。

外部文献仅支持立项方向，不修改本地 gate： [LOB resiliency](http<local-path>)、[meso-scale order flow](http<local-path>)、[state-first conditional order flow](http<local-path>)。

## 不做

Q01–Q07、TabM、SimLOB、SSL、Recent-Regime、Zero-sum、额外 M01/M02 窗口扫描和任何 Kaggle competition submission 均不在当前执行队列。
