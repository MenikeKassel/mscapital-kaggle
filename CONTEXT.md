# MSCapital 研究语境与证据词典

本文件定义项目 SSOT 中的术语，避免把诊断、候选和生产资产混为一谈。

- **Experiment**：一个有明确输入、目标、验证协议和复现指针的研究记录。
- **Diagnostic**：解释或稳定性检查，不产生可提交预测。
- **Candidate**：通过当前门禁、允许进入下一层验证的方法；不是生产资产。
- **Production Asset**：已冻结、协议闭合、可用于生产回放的表示或模型配置。
- **Submission**：提交行为及其 Public LB 结果，和本地模型证据分开登记。
- **Route**：一条方法谱系；每条实验只能属于一个主路线。
- **Self-owned**：仅依赖本地数据、代码和本地协议的证据。
- **External-assisted**：使用外部 LB、参考预测或外部线索的证据；不得进入本地训练选择。
- **Validated**：在声明的验证协议下可复查，可能仍然不足以晋级。
- **Insufficient**：方向有信号，但未满足稳定晋级门槛。
- **Negative**：在冻结协议下没有可接受增益或发生退化。
- **Invalid**：协议、数据来源或生产映射不合法，分数不承担模型结论。
- **Not identifiable**：当前 schema 无法识别所需对象，不能合法产出预测。

权威链：`experiments/registry.csv`（事实）→ `experiments/routes.yaml`（路线决策）→ `submissions/registry.csv`（提交）→ `docs/project-status.md`（当前叙述）→ 生成视图 → 单实验报告 → `RESULTS.md`（历史 append-only）。
