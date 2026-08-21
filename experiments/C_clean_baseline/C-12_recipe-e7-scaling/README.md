# C-12 — RealMLP recipe E7: Learnable scaling layer

## Metadata

- ID: `C-12` (canonical)
- Phase: C_clean_baseline | Created: 2026-08-16
- Status: `completed` | Decision: RED
- Parent: C-05 | Successor: -
- Tags: realmlp|recipe|paper
- Script: scripts/c_recipe.py | Artifact: output/c_recipe/c12_scaling_results.json

## Research Question

单变量替换锚点 C-05 (robust+clip + β2=0.999 + constant LR + ReLU 3x256 + MSE), 该组件在 MSCapital PSEUDO fold 上的净增益是多少?

## Hypothesis

论文 (Better by Default) 消融: 该组件移除代价 +1.0% 相对误差。

## Protocol

- PSEUDO fold: train m0-32 / eval m33-70, 严格 temporal
- 训练 MSE, 选 checkpoint 全局 cosine
- 30 epochs, batch 512, LR 1e-3, 3x256 MLP
- 实现忠实 pytabkit 源码 (research/paper-reading-2026-08/pytabkit_code/)

## Result

- Score: 0.112446 (PSEUDO eval 33-70, best ep 9)
- Delta: vs C-05 -0.000815
- Conclusion: Learnable scaling layer -0.0008: robust+clip 已归一化, 软特征选择无增量; lr×6 下早停 ep9

## Verdict

RED
