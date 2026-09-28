# ReScene 首轮单模块筛选指令包

本包落实“先逐个接入模块、筛选有效模块，暂不组合”的要求。

交给 Codex 执行的主文件：**ReScene_Round1_Single_Module_Codex_FINAL.md**。

## 文件

- `ReScene_Round1_Single_Module_Codex_FINAL.md`：本轮完整执行范围、模块合同、数据、训练、筛选和发布。
- `short_module_screen_v1.protocol.yaml`：对应的结构化设计配置。需要由 Codex 实现解析，不是旧仓库现成可执行配置。
- `ReScene_Round1_REVIEW.md`：本次交付前审核与修订记录，明确没有进行服务器实验。
- `REFERENCE_ONLY_24Papers_Roadmap.md`：上一轮24篇文献路线的原文副本，只供背景查证。其中后续扩展、组合与正式确认章节**不属于本轮执行授权**。
- `SHA256SUMS.txt`：包内文件校验值，不包含自身和外层压缩包。

## 本轮终点

冻结 R1；独立训练 Q1/Q2/Q3/M0/M1/M2；完成各自 CAL/SEL 和符合条件的固定点复核；给出多候选收益清单及真实单模块成本；将实际代码、小头、结果和交接推送 GitHub。

不运行 Q+M 或其它组合，不执行390/450续训，不默认替换当前部署，不在本轮重复正式PB/LOCAL长表。没有正收益也必须交付负结果和可复现证据。

这些文件是新实验的执行设计。生成本包时没有运行新训练、没有实际筛出胜出模块、也没有向用户仓库写入开发代码。
