## 决策解释与训练边界

本轮未确认全面质量增益，最终锁定 R1-D0，不增加感知适配、修复器或关联模块。下表中的差值均为 0–1 AP，均值是不同 T 差值的算术平均，不是新的 pooled AP。

| 候选 | 实际主模型训练终点 | CAL 选择 | full / SEL 决定及原因 |
| --- | --- | --- | --- |
| C0-H | 导入 V1 750 | pilot 250 | 高 LR 控制；不再进行 full 训练 |
| S-BAL-H | 从保存的 350 恢复到 750 | pilot 250 | pilot 未超过 Q-SEM-L；379 仅为 V1 未保存进度，未用作恢复边界 |
| C0-L | 3000 | 250 | 同 LR full 控制；SEL mean −0.003017、min −0.009739、long −0.009432，未采纳 |
| S-BAL-L | 750 | 750 | pilot 排名未晋级，不称训练失败 |
| A-OPEN-L | 750 | 750 | pilot 排名未晋级，不称训练失败 |
| Q-SEM-L | 3000 | 250 | 唯一新机制 full 晋级；SEL mean −0.000542、min −0.011577、long −0.006102，未采纳 |
| S-WORST-L | 750 | 250 | pilot 排名未晋级，不称训练失败 |
| R1 + NEW_ONLY | 修复头 1500，父模型 0 | 修复头 0 | 与 parent 相同，未达到修复采纳门槛 |
| R1 + OLD_NEW | 修复头 1500，父模型 0 | 修复头 0 | 与 parent 相同，未证明相对 NEW-best 的历史证据收益 |
| A0-U τ=0.73 | 无训练参数 | 12 配置中的 CAL 冠军 | SEL mean +0.000242、min −0.000379、long −0.000111，保持 D0 |

H=5e−5、L=1e−5；新低 LR 配对从 R1 开始，不恢复高 LR optimizer。C0-L 与 Q-SEM-L 使用同一 3000 步计划，各完成 24,000 个 local batches / 96,000 个全局 draws；两个 rank 的 RNG、optimizer 和 scheduler 状态见各 `FULL_ENDPOINT_AUDIT.json`。Q-SEM 的 scorer 是固定的既有 500 步组件，不将它计入主模型 3000 步，也未在 seed46 重训。

R1 两修复头共享 32 个 TRAIN reference、64 个 episode、5,220 条训练候选与相同初始化 SHA；各完成 1500 步。NEW 修正 63,618 个原错误、误修 6,799 个原正确 segment；PAIR 分别为 63,616 和 6,795。训练段级纠错不等于完整 CAL/SEL pooled AP 增益；两头的训练点均未胜过 0 步。因此本轮既不声称一般修复收益，也不声称历史 soft 证据的额外收益。

P=R1，因此第二父模型的修复训练 NOT_APPLICABLE；表内 P/NOT_RUN 行仅表示未另造一份重复训练。最终未部署任何新增训练组件，seed46 适配复核 NOT_APPLICABLE；LOCAL 的评价 seeds45/46/47 不是三次训练。未选 LR 的辅助臂与全部 seed46 配方文件属于预先登记的可执行配置，不代表这些配方实际训练过。

## 确认结论与可复现性限制

FINAL_LOCK 于 2026-09-26T13:49:22.974496+00:00 固定，SHA256 为 `e474309f47e856332b8c49cee10d050fd91520d8572fa789ba317ad968504d26`。之后的正确性恢复未改变权重、阈值、人口、组件选择或最终配方。

PB 每个方法覆盖 129 个 logical units / 516 个 T2–T5 前缀、6 个 references。R1-D0 相对 FH-R1-native 的 T2/T3/T4/T5 差值依次为 +0.022170、+0.000267、−0.021255、−0.021346，因此没有全 T 超过 FH。ADDITIONAL 的 T2/T3/T4 分别覆盖 111/77/32 单元、40/23/8 references；它没有 T5，不与 PB 混为同一人口。

LOCAL 保留原始 154 个序列、三评价种子；实际覆盖 46 references（原元数据写 41，已在锁定前审计更正）。FH-R1-native 三种子平均 temporal mAP 为 0.305460；同 LR C0-best-native 为 0.302875。最终 P=R1，与自身原生基线没有新增感知收益。

真实 A40 profile 使用每个 PB reference 的首个 canonical master，完整序列 warmup 一次、测量三次。FINAL 与 R1-D0 身份相同，计时去重；两方法共 144 条测量。T4/T5 测量中位数的 D0/FH 时延比 0.462945、显存 allocated 峰值比 0.449286；这是描述性比较，不代表统计稳定性证明。冷输入 IO 单列，不把其等待混入部署前向时延。

原始 R1 历史预测无法复现，原因仍为 UNDETERMINED。旧结果保留在 `foundation/original-runtime/`，当前 CAL/SEL 对照来自更正后重复一致的 runtime；不能把历史数值直接作为当前因果比较。数据 staging 的 hardlink 元数据/大小检查不构成原始文件逐字节不可变证明。

750 步恢复时的 checkpoint 重写已修复；更正后的完整 CAL 750 重跑与此前输出 SHA、指标一致。原生 PB 顺序名称错误已修复并完整重跑；LOCAL CUDA 诊断 loss 的确定性兼容仅作用于 no-grad criterion。旧失败结果与费用均保留。LOCAL 预测导出另发生旧首条缓存冲突，新缓存现绑定执行源码、输入与数值环境。首次补导出的线程/cuBLAS 环境未匹配，已中断计费并归档排除，见 `confirmation/LOCAL_EXPORT_RECOVERY_AUDIT.json`；按原环境进行的正式补导出见 `confirmation/LOCAL_EXPORT_MATCHED_RUNTIME_AUDIT.json`。不能仅凭评分 COMPLETE 声称预测包完整。

正式补导出六组全部通过：每组154/154序列、46 references，所有六项指标与原确认逐项完全一致。此前环境不匹配的尝试被排除，未据其数值更改结论或选择。全部规定人口的预测资产现已完整准备；另保留六份明确标为 PARTIAL 的原失败导出，不能将这些诊断档案计作额外独立确认。

部署 bundle 使用完整替换张量及变化 buffer，不是浮点差分。FH、D0、C0-L 的真实 panel 重载输出 SHA 均与直接推理一致。基础 R1/Concerto 权重和 GT 不再分发。Git/Release 发布状态由最终 external receipt 确定；READY、模型本地可重载、Release 可下载是三个不同状态。
