# C-09 — RealMLP recipe E4: ReLU -> Parametric Mish

## Metadata

- ID: `C-09` (canonical)
- Phase: C_clean_baseline | Created: 2026-08-16
- Status: `completed` | Decision: GREEN
- Parent: C-05 | Successor: -
- Tags: realmlp|recipe|paper
- Script: scripts/c_recipe.py | Artifact: output/c_recipe/c09_paramish_results.json

## Research Question

单变量替换锚点 C-05 (robust+clip + β2=0.999 + constant LR + ReLU 3x256 + MSE), 该组件在 MSCapital PSEUDO fold 上的净增益是多少?

## Hypothesis

论文 (Better by Default) 消融: 该组件移除代价 +4.8% 相对误差。

## Protocol

- PSEUDO fold: train m0-32 / eval m33-70, 严格 temporal
- 训练 MSE, 选 checkpoint 全局 cosine
- 30 epochs, batch 512, LR 1e-3, 3x256 MLP
- 实现忠实 pytabkit 源码 (research/paper-reading-2026-08/pytabkit_code/)

## Result

- Score: 0.122404 (PSEUDO eval 33-70, best ep 23)
- Delta: vs C-05 +0.009143
- Conclusion: Parametric Mish 大增益 +0.0091 (论文 reg +4.8% 相对误差, 我们更强): ReLU 死区在 robust+clip 数据上代价高, 参数化激活修复它

## Verdict

GREEN
