# C-14 — RealMLP recipe E9: cosine -> coslog4

## Metadata

- ID: `C-14` (canonical)
- Phase: C_clean_baseline | Created: 2026-08-16
- Status: `completed` | Decision: GREEN
- Parent: C-05 | Successor: -
- Tags: realmlp|recipe|paper
- Script: scripts/c_recipe.py | Artifact: output/c_recipe/c14_coslog4_results.json

## Research Question

单变量替换锚点 C-05 (robust+clip + β2=0.999 + constant LR + ReLU 3x256 + MSE), 该组件在 MSCapital PSEUDO fold 上的净增益是多少?

## Hypothesis

论文 (Better by Default) 消融: 该组件移除代价 +0.4% 相对误差。

## Protocol

- PSEUDO fold: train m0-32 / eval m33-70, 严格 temporal
- 训练 MSE, 选 checkpoint 全局 cosine
- 30 epochs, batch 512, LR 1e-3, 3x256 MLP
- 实现忠实 pytabkit 源码 (research/paper-reading-2026-08/pytabkit_code/)

## Result

- Score: 0.115357 (PSEUDO eval 33-70, best ep 23)
- Delta: vs C-05 +0.002096
- Conclusion: coslog4 周期调度 +0.0021 (论文 ns +0.4%, 我们更明显): 4 周期重启跳出局部最优, best ep 23 持续学习

## Verdict

GREEN
