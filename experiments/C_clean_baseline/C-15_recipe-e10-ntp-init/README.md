# C-15 — RealMLP recipe E10: NT parametrization + 数据驱动 init

## Metadata

- ID: `C-15` (canonical)
- Phase: C_clean_baseline | Created: 2026-08-16
- Status: `completed` | Decision: RED
- Parent: C-05 | Successor: -
- Tags: realmlp|recipe|paper
- Script: scripts/c_recipe.py | Artifact: output/c_recipe/c15_ntp_init_results.json

## Research Question

单变量替换锚点 C-05 (robust+clip + β2=0.999 + constant LR + ReLU 3x256 + MSE), 该组件在 MSCapital PSEUDO fold 上的净增益是多少?

## Hypothesis

论文 (Better by Default) 消融: 该组件移除代价 ~1% 相对误差。

## Protocol

- PSEUDO fold: train m0-32 / eval m33-70, 严格 temporal
- 训练 MSE, 选 checkpoint 全局 cosine
- 30 epochs, batch 512, LR 1e-3, 3x256 MLP
- 实现忠实 pytabkit 源码 (research/paper-reading-2026-08/pytabkit_code/)

## Result

- Score: 0.075803 (PSEUDO eval 33-70, best ep 28)
- Delta: vs C-05 -0.037458
- Conclusion: NT 参数化+数据驱动 init 灾难 -0.0375: NTK 改变梯度尺度, 与 LR=1e-3 不匹配 (论文配 lr=0.2+256ep+coslog4); best ep 28 持续爬升, 30ep 远不够 → 组件交互案例, NT 非独立组件

## Verdict

RED
