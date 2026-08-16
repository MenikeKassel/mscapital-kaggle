# C-08 — RealMLP recipe E3: constant LR -> cosine decay

## Metadata

- ID: `C-08` (canonical)
- Phase: C_clean_baseline | Created: 2026-08-16
- Status: `completed` | Decision: RED
- Parent: C-05 | Successor: -
- Tags: realmlp|recipe|paper
- Script: scripts/c_recipe.py | Artifact: output/c_recipe/c08_cosine_results.json

## Research Question

单变量替换锚点 C-05 (robust+clip + β2=0.999 + constant LR + ReLU 3x256 + MSE), 该组件在 MSCapital PSEUDO fold 上的净增益是多少?

## Hypothesis

论文 (Better by Default) 消融: 该组件移除代价 +13.5% 相对误差。

## Protocol

- PSEUDO fold: train m0-32 / eval m33-70, 严格 temporal
- 训练 MSE, 选 checkpoint 全局 cosine
- 30 epochs, batch 512, LR 1e-3, 3x256 MLP
- 实现忠实 pytabkit 源码 (research/paper-reading-2026-08/pytabkit_code/)

## Result

- Score: 0.112410 (PSEUDO eval 33-70, best ep 9)
- Delta: vs C-05 -0.000851
- Conclusion: cosine decay (1→0) 在 30ep 短训练下为负 -0.0009: 后期 LR 过低学不动, best ep 9 早停; 论文 +13.5% 是 256ep+meta-tuned LR 的结论, 短训练协议不适用

## Verdict

RED
