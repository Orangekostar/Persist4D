# ReScene Q-P / M-N 执行包

将 `ReScene_QP_MN_Codex_FINAL.md` 完整交给Codex，作为唯一执行指令。

本轮验证两个干预：保留父分数的Q-P，以及仅按有效形状监督归一化的M-N。Q-A0和M-C是配对控制，Q2-F是本地修复后的旧配方桥接。暂不做关系头、困难候选采样、点级解码、长期实验或组合。

## 文件

- `ReScene_QP_MN_Codex_FINAL.md`：完整执行规范，含源绑定、公式、数据、选模、复核和实际发布。
- `qp_mn_targeted_v1.design.yaml`：与规范对应的结构化设计，不是已经存在的仓库CLI配置。Codex须将其适配成实际config。
- `EVIDENCE_AND_REVIEW.md`：已读依据、局限和交付前纠正的问题。
- `review_checks.py` / `REVIEW_CHECKS.json`：本包实际CPU公式与文档检查及结果；不等于仓库测试。
- `SHA256SUMS.txt`：交付文件校验。

## 重要前提

远端核实的旧版本为4bf00902；用户报告的修复版本465f37a尚不能通过本次GitHub连接读取。Codex必须在原服务器的本地仓库中解析完整SHA，读取CONCLUSION与REQUIREMENT_REVIEW，绑定正确标签与NOAUG资产；不允许回退旧标签。

本包没有运行新服务器实验，没有替用户push代码。执行后的结果与小头、指标证据和HANDOFF必须由Codex实际推送GitHub。本轮不要求确认名单非空。
