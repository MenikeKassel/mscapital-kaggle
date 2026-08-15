# GPT 评审核验报告: RealMLP 论文讲解 (2026-08-15)

> 输入: GPT 对 RealMLP 论文讲解的评审 (引用了论文数字 + standalone 代码 3 处)
> 方法: 克隆官方代码逐行核对 (mlp.py/preprocessing.py) + 回查论文 Table A.1/B.1/B.2/B.5 + 附录原文
> 裁决符号: ✅ 实测证实 / ⚠️ 部分修正 / ❌ 纠正 / 🔍 真新增量
> 原始会话: 20260815_160037_bda93b | 本报告为会话 48104 的落盘版

## 0. 结论先行

GPT 评审质量很高: **20+ 条论断数字几乎全部准确**, 还抓到两个上一轮讲解漏掉/弱化的真问题 (target 标准化平移敏感、训练/选择指标分离)。需纠正 4 处小错, 无一影响定性结论。

## 1. 逐条核对表

### ✅ 已实测证实 (代码逐字正确)

| GPT 论断 | 实测证据 | 判定 |
|---|---|---|
| scaling layer = `nn.Parameter(torch.ones(n_features))` + `x * scale` | tds_standalone mlp.py L11-17 | ✅ |
| NTP 前向 = `(1/√in)·(x@W)+b` | mlp.py L29-30 | ✅ |
| coslog4 = `0.5−0.5·cos(2π·log2(1+15t))` | mlp.py L122 | ✅ |
| Mish = `x·tanh(softplus(x))` | mlp.py L33-35 | ✅ |
| **target 标准化用 train-only 统计量** (论文措辞"training and validation set"是笔误) | mlp.py L56-60 y_mean_/y_std_ 只从 y 拟合 | ✅ **GPT 抓得对** |
| TD-S 用 plain Adam (非 AdamW) | mlp.py L76 无 wd 参数 | ✅ |
| LR factors: scale×6, bias×0.1, base 0.07 (TD-S reg) | mlp.py L113/125-127 | ✅ |
| last best epoch (平局取最后) | mlp.py L144 注释 | ✅ |
| 预处理 = median/IQR + IQR=0 退化为 min-max + 常数列置 0 + smooth clip | preprocessing.py L65-85 | ✅ |
| 全部消融数字 (β2 +22.8%/+2.0%、无嵌入 +20.6%/+2.3%、PLR +19.0%/+4.2%、constant LR +13.5%/+1.8%、param act +4.8%/+0.5%、clip +9.5%、scaling +1.0%/+1.4%、NTP +1.3%/+1.1%、init +1.2%/+0.9%、constant dropout/wd +3.6%/+3.1%) | Table B.1/B.2 回查 | ✅ |
| "XGB bug 修复后 SGM 约改善 2%" | 附录 B L1877 原文 | ✅ |
| 临界差图 = Figure B.10 (Friedman+Nemenyi 95%) | L2471 | ✅ |
| coslog4 相似 SGDR/cyclical LR | L455 | ✅ |
| "PL 已非常接近 PBLD" (PL reg +0.5% CI 含 0) | Table B.1 | ✅ 好判断 |
| "coslog4 vs 单次 cosine 优势不显著" (+0.4% ns) | Table B.1 | ✅ 好判断 |
| β2 半衰期: 0.999→693 步, 0.95→13.5 步 | ln0.5/lnβ2 推导 | ✅ |
| cosine 梯度 ∇_pC = y/(‖p‖‖y‖) − C·p/‖p‖² | 求导复核 | ✅ |
| "β2 +22.8% 是 recipe 内敏感度非普适增益" | 论文自述消融非无偏 + wd 交互警告 | ✅ |
| one-hot 列也过 robust+clip | pipeline: one_hot → rssc | ✅ |
| Figure 3/AUROC 教训: recipe 依赖最终 metric | B.5 + L573 | ✅ |

### ⚠️ 部分正确 (边界修正)

| GPT 论断 | 修正 |
|---|---|
| "clipping 主要收益是输入有界, 梯度漂亮不重要" | 半对: 有界是主收益 ✓; 但 PBLD 频率/第一层权重**确实经 f 反向传播**, hard clip 死区(梯度=0)让 embedding 学不到"有多极端", smooth clip 导数(x=10 仍有 0.025)保留学习通道; hard clip 还丢 ±3 外排序信息 |
| "Figure 1(c) 柱子" | 实为点线图+误差棒 (caption 明示 95% CI) |
| 均值例子 "μ=16.68" | 应为 16.88 (84.4/5); 对应 z 分数 ~0.01 出入; 定性结论不受影响 |

### ❌ 纠正 (无实质影响)

| GPT 论断 | 纠正 |
|---|---|
| "TD-S 因无 wd 用 Adam 非 AdamW" | 属实但 wd=0 时 Adam 与 AdamW 完全等价; Table A.1 写 AdamW 也没错, 只是 standalone 实现选择 |

## 2. GPT 相对讲解的真新增量 (5 条, 全部采纳)

1. **target 标准化平移敏感** (最值钱): uncentered cosine 对 y 缩放不变但对平移敏感 → "标准化后算 cosine loss" = 优化不同目标 (除非 μ≈0)。正确: cosine 项用 raw y 方向 (不减均值); MSE 辅助项可用标准化 y
2. **先 PL 后 PBLD**: 消融 PL≈PBLD (CI 含 0), 更便宜更干净
3. **cosine decay 先行, coslog4 押后**: 与消融一致 (+0.4% ns)
4. **objective track 后置 + 训练/选择指标分离**: MSE 训练 + global-cosine 选 epoch, 配方稳定后再开 loss 分支; 中途换损失污染组件归因 (排掉上一轮的 E3 雷)
5. **s_i 跨 fold 稳定性诊断**: scaling layer 参数作 feature-stability 研究工具 (论文没有, 高价值)

## 3. 待用户拍板 (3 项)

1. loss 策略: 全程 E0-A (MSE 训练 + cosine 选择) 直到配方稳定? 还是并行小 λ 扫描 sanity check?
2. E5 起点: 先 PL 后 PBLD (GPT 推荐) 还是直接 PBLD?
3. ~~文档落盘~~ → 已执行 (本报告 + realmlp-lecture-and-migration.md 已入 research/paper-reading-2026-08/)

## 4. 结论

**GPT 评审 ✅ 通过核验** (20+ 条实测无误, 仅 4 处小错), 贡献 5 条真实增量。合并后 v2 实验顺序 (E0-E10) 见 realmlp-lecture-and-migration.md §5。**讨论期已结束, 执行待用户明确指令。**
