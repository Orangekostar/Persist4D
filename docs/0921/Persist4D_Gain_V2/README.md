# Persist4D Gain V2 执行指令包

本包用于指导Codex开发、运行和发布V2实验；不是已经完成的V2代码或训练结果。

## 使用

将 `Persist4D_Codex_Gain_V2_FINAL.md` 的完整内容交给Codex，作为本轮唯一执行规范。不要用V1旧指令同时覆盖V2的任务顺序。Codex应在用户已有Persist4D仓库和计算环境中核对资产、实施代码与实验，再按主文件实际提交和上传GitHub。

## 文件

| 文件 | 用途 |
|---|---|
| `Persist4D_Codex_Gain_V2_FINAL.md` | 唯一完整执行指令：两条独立主线、有限关联补测、选模、确认、发布 |
| `Persist4D_Codex_Gain_V2_EVIDENCE.md` | 固定提交来源和任务映射补充，不是另一套计划 |
| `Persist4D_Codex_Gain_V2_REVIEW.md` | 交付前语义审核、已修订的边界、审核限制 |
| `V2_REVIEW_CHECKS.json` | 文档静态和数值策略示例检查；不是服务器测试 |
| `SHA256SUMS.txt` | 上述交付文件与README的校验值，不包含自己 |

核查起点：`8b5e93795817682fe70864daa545db219e1443c9`。新分支：`research/persist4d-perception-gain-v2`。GPU预算沿用V1累计上限并扣除历史消耗；没有额外授权自动追加192 GPU-hours。

核心调整：低学习率适配配对；冻结R1立即运行NEW_ONLY与OLD_NEW修复对照；只延长一个新机制及其同LR C0；有限12配置关联作为独立候选；预算优先保留完整确认；真实上传模型/结果/交接，而不只发布空Release。
