# ReScene4D 与当前实验的评测总体核对

核对日期：2026-10-13。结论：基础数据来源相同，但不能将所有结果称为同一评测集合上的结果。

## 原论文公开设置

ReScene4D 使用 3RScan 验证集报告结果，主要协议是随机排列扫描后构造滑动窗口的 T2 序列；论文主表的指标覆盖 18 个实例类别。训练还混入 ScanNet 单扫描数据，采样比为 3RScan:ScanNet = 1.0:0.8。

来源：[论文第 5 节及附录 B](https://arxiv.org/html/2601.11508v1)。论文未在正文提供可用于本次逐条核对的完整评测扫描对清单及随机排列种子，因此本次不声明已验证论文结果的样本完全一致性。

## 本地已绑定总体

| 总体 | 数据来源 | 独立参考场景 | 评测输入 |
|---|---|---:|---|
| LOCAL-T2 | 3RScan 验证集原生索引 | 46 | T2 共 154 条 |
| PB | 3RScan 验证集原生索引；筛选可构造 T5 的序列并固定三种顺序 | 6 | T2、T3、T4、T5 各 129 条 |
| CAL | 3RScan 官方训练来源的内部保留角色 | 8 | T1–T5：31/19/14/14/6 |
| SEL | 3RScan 官方训练来源的内部保留角色 | 8 | T1–T5：39/21/21/15/15 |

LOCAL-T2 最接近论文的验证 T2 设置；本地还启用了无有效监督序列过滤，不能仅凭“官方验证来源”宣称与论文逐条相同。CAL/SEL 使用训练来源索引，与论文的验证总体不同。PB 与 LOCAL-T2 的场景存在重叠，不能称为两个独立外部测试集。

## 本地扫描对直接核对

将 `artifacts/native_long_postselection_v1/protocol/FORMAL_POPULATION.json` 中记录的 `physical_scan_ids` 转为有顺序的扫描对，对 PB 的 T2 输入和 LOCAL-T2 输入进行集合核对：

- LOCAL-T2：154 条输入、154 个不同的有序扫描对。
- PB-T2：129 条输入、97 个不同的有序扫描对。
- 共同有序扫描对：16 个；PB 独有 81 个，LOCAL-T2 独有 138 个。
- 共同参考场景：6 个。

129 个 PB 输入含重复有序扫描对，不能等同于 129 个独立场景或独立扫描对。上述核对仅证明两套本地输入不同，不构成对原论文清单的一致性证明。

## 比较边界

PB 长序列分数、内部 CAL/SEL 分数和论文主表 T2 分数不能直接作为同一基准上的性能差值。有效方法比较需要在同一输入清单、序列长度、类别与指标配置、预测输出协议下运行双方。

本次核对读取的本地证据：

- `artifacts/P6A/protocol_b_manifest.json`
- `artifacts/native_long_postselection_v1/protocol/FORMAL_POPULATION.json`
- `artifacts/perception_gain_v2/foundation/LOCAL_POPULATION_AUDIT.json`
- `artifacts/revisit_prompt_v1/DATA_MANIFEST.json`
- `artifacts/revisit_prompt_v1/DATASET_POPULATION_AUDIT.json`
- `scripts/perception_gain_local_evaluation.py`
- `conf/config_p2_rescene4d_concerto_t2.yaml`

这是一项评测总体核对，不表示 MPR 开发、真实训练或最终提示词审查已经完成。
