# ReScene：Q-P 评分保留与 M-N 形状监督归一化——最终执行指令

日期：2026-09-29。任务类型：开发、真实小头训练、受控选模、诊断及 GitHub 交付。

**本文是本轮唯一执行规范。先阅读全文，再执行。不要叠加旧六臂任务的范围、晋级规则或 publish-only 恢复行为。** 本文没有预先声称涨点，也没有声称作者已经访问本地 `465f37a` 或运行服务器实验。

## 0. 本轮结论要回答什么

本轮只验证两个干预：

1. **Q-P**：保留 R1 原分数的直接通路，用零初始化小头学习实数分数修正。
2. **M-N**：在保持形状分配、采样、网络、残差幅度及正则不变时，将形状损失由“全部候选位置平均”改为“有实际形状监督的位置平均”。

最终同时回答“相对同条件 B0 是否改善”和“相对必要机制控制是否改善”。没有净收益就保留 B0；不得用打败退化控制替代打败 B0。

### 本轮运行范围

| ID | 类型 | 地位 | 是否要有完整 CAL/SEL 结果 |
|---|---|---|---|
| B0 | 冻结 R1 原生输出 | 唯一共同基准 | 是 |
| Q2-F | 本地修复后的旧 Q2：sigmoid 直接评分 | 历史配方桥接；满足身份条件可复用训练 | 是 |
| Q-A0 | 新建线性评分头，最后一层为零，`s=r(x)` | Q-P 的严格直接控制 | 是 |
| Q-P | 与 Q-A0 同结构同初始化，`s=s_R1+r(x)` | 新评分干预 | 是 |
| M-C | 本地修复后的 M1 结构；全部抽样位置平均形状项 | M-N 的严格直接控制 | 是 |
| M-N | 与 M-C 同结构同初始化；仅改变形状项分母 | 新形状干预 | 是 |

**固定四条配对训练轨迹：Q-A0、Q-P、M-C、M-N。** Q2-F 优先复用已经修复的同配方检查点；不满足复用条件时仅补这一条，不能顺势重开 Q1/Q3/M0/M2。M-C 也可在严格身份相同且完整轨迹可验证时复用，否则重新训练。所有复用都要写明来源和本轮实际新增更新数。

Q-A0 必不可少：旧 Q2 使用 sigmoid 和普通末层初始化；若直接比较 Q-P 与旧 Q2，会把跳接、输出非线性和初始化同时改变。Q-P−Q-A0 才隔离“保留原评分通路”；Q-A0−Q2-F 只解释桥接配方整体，不拆成未经验证的因果结论。

### 明确不执行

不执行 Q+M 组合、集成、跨头串联、默认模型替换；不执行 Q-R 候选关系头、M-S 重采样/多对一、M-L 点级解码器。它们只能写入下一阶段决策文件，不根据本轮小幅正数自动启动。也不做 390/450 续训、LoRA、解冻 R1、增加 query 数、改变分辨率、新关联、D0/lag1、T3–T5、PB/LOCAL/ADDITIONAL 正式计分。正式测试与组合另开后续任务。

以上是有意把上轮“先两项、再条件扩展”落实为一轮边界清楚的任务，并非把后续思路当成已经验证失败。

---

## 1. 证据身份和不能跳过的本地基线绑定

### 1.1 已读事实与未知项

- 可远端核实的代码：`Orangekostar/Persist4D@4bf00902c9428795f7f47547bf462540aa304cc6`。
- 用户报告已在本地完成监督/缓存修复、受控重跑、无增强核验，提交短 SHA 为 `465f37a`。
- 本指令编写时，GitHub 对 `465f37a` 返回未找到，匹配的远端 ReScene 分支仍只有旧实验分支；Library 未检索到新的 `CONCLUSION.md` / `REQUIREMENT_REVIEW.md`。
- 因此，旧源码只作为阅读和接口依据，**真正执行必须从本地修复提交绑定，禁止自动回退到旧版**。

已知依据：[E01] 新评分直接替换原分数；[E02] M 形状分割只在合法配对位置计算，但全部位置参与平均；[E03] 每步 3 个真实 T2、1 个真实 T1，每输入4候选；[E04] M1 原日志仅2223/24000抽样位置具有形状监督资格。它们支持设计对照，不证明原因贡献或新方法收益。

### 1.2 定位与冻结步骤

在现有 `persist4d` 环境和已知项目仓库中执行，不新建一套 CUDA/conda 环境。

1. 用 `git rev-parse --show-toplevel` 和 `git worktree list --porcelain` 找到实际仓库；优先查用户给出的 `paper5/.worktrees/rescene-code-first-audit-v1`，不扫描整个 `/home` 或磁盘。
2. 用 `git rev-parse --verify '465f37a^{commit}'` 获取唯一完整 SHA，记录为 `REPAIRED_BASE_COMMIT`。检查它与旧 `4bf00902...` 的祖先关系及实际 diff。若不是后者的直接后继，但有清楚的等价修复历史，写明来源差异；不能静默选择别的 commit。
3. 读取该提交实际包含的：
   - `artifacts/rescene_code_first_audit_v1/CONCLUSION.md`
   - `artifacts/rescene_code_first_audit_v1/REQUIREMENT_REVIEW.md`
   - 这两份报告引用的 repaired label、cache、no-augmentation config、Q/M 控制结果、成本账本和本地资产索引。
4. 不猜修复后类名或字段名。建立一个薄适配层，把实际修复接口绑定为本轮的 `quality_target`、`quality_weight`、`shape_assignment`、`shape_usable` 等逻辑字段。每项记录原文件/符号、含义与摘要。
5. 单独核实用户说的“1项既有测试失败”：若属于本轮直接调用链且会影响模型/标签/指标，必须有限修复后再执行；若只是无关旧 checkpoint 或另一条管线，记录排除理由，不为其重造历史资产。
6. 用新 worktree 从完整修复 SHA 开分支 `research/rescene-qp-mn-targeted-v1`，产物目录 `artifacts/qp_mn_targeted_v1/`。同名分支属于同一任务则恢复；不相关冲突用最小 `-r2` 后缀并记录。保留用户未提交改动，不 reset/clean、不移动旧 tag。

如果本机也没有修复对象或必要文件，只在已知原 worktree/common git objects 和用户给定资产位置检查一次。确实缺失则记 `BLOCKED_REPAIRED_BASE`，完成可完成的只读分析及交接；不在旧标签上训练新方法，也不伪造修复已通过。不要把“本地有提交、远端没有”误认为实验不可执行。

### 1.3 本地核验的最低语义，不重跑整套旧审计

从已有修复产物和两个定向用例确认：

- quality 标签独立于当前 `pred_scores`；partial-ignore 的监督资格按真实阈值处理，而非统一用0.5过滤所有阈值；明确 unknown/ambiguity 不强改为负例。
- Q 的有效性更改不会隐式改 M 的一对一训练分配。M-C/M-N 必须共享显式、冻结的形状资格与分配。
- 父预测缓存和标签缓存身份可以分开；代码或标签变更不会被旧 `COMPLETE` / `PUBLISH_STATE` 误判成已重跑。
- 无增强模式只关闭增强，保留同一 split、数据清单、标签格式、体素和后处理；不直接把 dataset `mode` 改为会改变标签格式的 `test`。

保留原72份复算结果的证据，不为凑验证数量再复算全部72份。只验证本轮实际复用的控制和新结果。

生成 `BASE_BINDING.json` 和 `CODE_BINDINGS.md` 后继续。它们记录事实，不是可独立代替训练的成果。

---

## 2. 代码阅读、改动与实验的对应关系

下列已知符号来自远端旧提交；在本地修复版按实际位置重新绑定。新文件名是待实现建议，不声称已经存在。

| 既有文件/符号 | 必须理解的行为 | 本轮最小改动/调用 |
|---|---|---|
| `models/short_module_heads.py::quality_inputs` | q/h归一化、阶段差值、类别、原分数和支持统计 | 评分系列原样复用，不同时加点数/关系/边界特征 |
| 同文件 `QualityHead/build_head/apply_module` | 旧Q sigmoid输出、候选谱系与score插槽 | 新建线性零末层头；通过单枚举区分Q-A0/Q-P |
| 同文件 `MaskHead` | 共享h、128维F、每阶段系数、`2*tanh` | M-C/M-N严格复用同一结构，不扩大残差 |
| `scripts/short_module_data.py::geometry_assignment/low_segment_targets/geometry_loss` | shape资格、同一GT、点数权重、分割项与正则的平均顺序 | 只提取可微candidate loss向量；两个分母分支，正则不变 |
| `scripts/short_module_training.py::build_sample_plan/train_arm` | 四输入/16位置、每阶段最多2048segments、1500步调度、检查点 | 复用固定抽样；新头独立初始化，不把R1加入optimizer |
| `scripts/short_module_native.py::NativeSession.produce` | 真正H1/H2读取、冻结前向、特征导出 | 使用修复版noaug adapter；不重复父前向四遍 |
| `scripts/rescene_task_postprocess.py::extract_official_task_prediction/materialize_segment_logits` | 一次top-k/filter与真实bool物化 | Q只换分数；M只换mask；materialize不进入训练loss |
| `scripts/short_module_evaluation.py` 及修复版对应入口 | CALSEL锁点、独立metric、结果缓存身份 | 本轮目录与枚举；禁止import旧常量后把输出写回旧任务 |
| `scripts/p6a_metrics.py::OfficialMetricAccumulator` 与锁定stmetrics | H1空间AP、H2temporalAP、score排序/ignore | 不修改官方阈值和评价算法；接受有限实数评分 |
| `scripts/short_module_results.py/profile/delivery.py` 及修复版 | 复核、部署计时、Git/Release | 复用窄函数；删除本轮对旧包目录、只publish恢复的隐式依赖 |

推荐最多新增：
`models/qp_mn_heads.py`、`scripts/qp_mn_adapter.py`、`scripts/qp_mn_campaign.py`、一个配置以及相关少量测试。可复用现有函数，不必机械复制文件；不要创建新数据库、通用任务DSL、完整训练框架或大量schema平台。

新头的运行模式只能是枚举 `B0/Q2_F/Q_A0/Q_P/M_C/M_N`。不允许组合列表。独立实验可以共享只读R1缓存、描述符和抽样计划，但不能共享训练后的新头参数。

---

## 3. 数据、父模型与输入条件

### 3.1 固定父模型

R1 为已完成epoch390的原selected checkpoint：

- SHA256：`629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd`
- bytes：`754813672`
- Concerto SHA256：`845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07`

从已有资产索引定位，启动时各核对一次；不每step hash父权重。R1始终eval、requires_grad=False，新增头单独train。已知旧维度为F=128、q=128、类别18、类概率19，但以真实张量校验，不补零裁切迁就文档。

### 3.2 人口不可重新挑选

继承 `SHORT_POPULATION.json` 和修复报告对应的物理reference分工。旧轮记录为TRAIN64 refs/188 T2/199 T1、CAL4 refs/23 T2/23 T1、SEL4 refs/24 T2/24 T1。**这些是旧证据计数，不是新轮预填PASS。** 实际读取后对齐并记录；修复报告若有合法人口纠错，保留纠错依据，不混用两个分母。

T1是相同角色T2扫描集合按physical scan去重后的独立单扫描读取、时间归零和独立前向。T2只输入两个扫描，没有额外X1历史前向、D0、lag1或X3。两阶段中的条件AP另列，不能冒充T1。

TRAIN不能补入CAL/SEL/PB/LOCAL/ADDITIONAL reference。父R1历史训练暴露和各开发人口已经反复使用的事实如实记录，不重新称为untouched。候选缺失的输入仍参加评价，不能按是否有GT或预测高分选择容易样本。

### 3.3 本轮主评价明确改为无训练增强

- **TRAIN条件**：继承本地修复后用于受控重跑的同源TRAIN父预测和描述符，保留其既有固定增强/seed处理，以减少新变量。Q-A0/Q-P/M-C/M-N全部使用同一组输入；不能某臂重导无增强TRAIN、另一臂沿用增强TRAIN。
- **主开发条件**：`NOAUG_NATIVE`，使用修复版已经核验的关闭训练增强路径，对同一CAL/SEL扫描独立执行原生H1/H2。全部臂（含B0、Q2-F）共用。
- 这属于本轮预先规定的评价条件，不称为对旧增强CAL分数的直接修复增益。每个条件必须有自己的B0。
- 旧增强CAL/SEL可在选点锁定后，仅对固定点作补充对照；优先复用，不根据其数值重新选点。本轮不以补充增强结果覆盖主NOAUG负结果。
- 修复版NOAUG缓存可验证复用；缺时一次导出，不改变数据split或原标签/分区格式。禁止仅调用model.eval()就声称关闭了数据增强。

### 3.4 预测—标签—训练—结果身份分离

预测缓存身份包括R1/Concerto SHA、真实input/scan顺序/H、数据与分区身份、数值环境、实际producer/postprocess依赖、NOAUG或TRAIN_INPUT条件。标签身份另包括修复版matcher/evaluator、dataset spec、标签函数、阈值和assignment规则。训练身份包括上述身份、head结构/初始化/目标/归一化/采样/调度。结果身份包括真实apply函数、物化、metric、输入条件和检查点。

改变标签无需重新计算未变的父特征；改变NOAUG输入则必须新父预测；改变训练目标不能继续加载旧optimizer装成同一训练。`COMPLETE`必须伴随身份比较。`run --resume`只重用有效节点，不能因存在发布回执跳过计算。

---

## 4. 修复版适配器：不猜测最新字段，不回退旧标签

在本轮运行目录写 `REPAIRED_CONTRACT.json`，绑定下列逻辑内容到本地真实来源：

```
input_id / candidate_key / horizon
parent_scores, masks, classes, query_ids, class_ids
F, q, raw_segment_logits, h, segment_stages
full_to_low_segment, inverse_map, full_eval_partition
quality_target_scalar, quality_weight_or_validity
shape_gt_id, shape_usable, segment_targets, n_valid_points
label_version, prediction_version, assignment_version
```

### 4.1 Q的目标

Q2-F/Q-A0/Q-P采用同一个**修复后Q2标量temporal质量标签与权重规则**。训练标签必须分数无关，显式处理partial-ignore与未知歧义。Q-A0/Q-P不得重新设计一套与修复版不同的GT最佳配对或negative规则。

若修复版Q2使用二值valid，则取w=0/1；若使用预先确定的非负importance weight，则原样保留。需验证其归一化形式与真实修复版loss一致。文档下面的MSE式以 `sum(w*error²)/sum(w)` 为标准表达；若本地实现不同，先建立数值等价适配，不擅自改所有臂的loss。

若本地所谓“修复后的Q2”已经改变为关系排序、另一个目标或非MSE网络，不能假装仍是这里的Q2-F。记 `BLOCKED_Q_CONTROL_MISMATCH`，说明差异，独立M工作仍可继续。

### 4.2 M的目标

M-C/M-N共享修复版明确采用的M1形状assignment、shape_usable和segment_targets。不要直接用新Q valid代替shape_usable。若修复版为了控制变量保留旧M assignment，就原样使用并标明其版本；不偷偷增加多对一、提高正例阈值、重新抽样或给未配对项补GT。

target始终是同一candidate跨所有阶段对应同一GT实体。该实体在某阶段确实缺失，则那一阶段仍监督为空。ignore点不当作普通负点；真实背景/其他实例仍是负点。

### 4.3 必需的不变量

对一条真实TRAIN记录和一个partial-ignore构造例，验证标签不随score改变或列重排而变。Q的修复权重、M的shape资格是两个不同字段。任何GT派生资格不得用于推理gate或描述符选择。

这一验证只确认当前适配器没有退回旧缺口，不再重做整套历史审计。

---

## 5. Q系列：把“评分参数化”与“保留原分数”拆开

### 5.1 公共输入

复用修复版的原 `quality_inputs()`：q、阶段h均值与差值、类别概率/one-hot、父score、支持/前景概率统计及H。保持字段顺序、维度、归一化和缺省值一致。不加入点数加权、候选关系或新不确定性特征。

### 5.2 网络与输出

公共trunk：`Linear(Din,128) → GELU → Linear(128,64) → GELU`。

| ID | 末层 | 输出 | 0步性质 |
|---|---|---|---|
| Q2-F | 修复旧Q2原初始化 | sigmoid(linear(trunk)) | 非B0；只作历史诊断 |
| Q-A0 | Linear(64,1)，weight和bias全0 | `r(x)` | 全0评分，不称B0 |
| Q-P | 与Q-A0逐参数相同的初始状态 | `parent_score + r(x)` | 必须严格等于B0原score |

Q-A0和Q-P从同seed相同trunk、相同零末层重新开始，不加载训练过的Q2。除父分数直接相加外二者的代码分支一致。父分数也仍作为共同输入，所以Q-A0并非被禁止使用该信息，Q-P改变的是直接通路与学习起点。

输出为有限实数排名分数，不是校准概率。不得clip、sigmoid、min-max rescale、再乘class/parent score或按GT门控；这些都会形成另一种实验。部署入口和官方metric如有未经证实的[0,1]断言，要在**新头适配边界**允许有限实数，不修改官方AP算法。

测试中用有限负分数和>1分数经锁定evaluator确认能评分。可用严格递增仿射变换验证排序一致；不利用容易浮点饱和的sigmoid作为相同排序的证明。

### 5.3 损失

两个线性臂使用同一个修复Q2的scalar MSE：

`L_Q = sum_i w_i * (score_i - target_i)^2 / sum_i w_i`。

全batch权重为0时，返回连接到head的零loss，并记录无有效监督更新。不要重抽该batch。

**第一版不新增 score-residual 正则。** 之前讨论中的弱正则属于可选设计，此处为保持可归因性暂不加入；两臂共同沿用AdamW weight decay。需要正则再另行登记，不根据SEL临时调lambda。

除了Q-P零步保证以外，不保证训练后自动不退化。以真实B0差值与固定点复核决定。

### 5.4 原生输出隔离

先由R1固定原top-k/filter得到retained候选，再替换最终score。所有mask、class、candidate key与B0一致；不再次按新分数过滤、NMS或选top-k。一个query可以对应多个class候选，唯一键包含input/query/class/retained-index。

Q2-F只用于桥接现有修复结果；Q-P最终成功需同时报告相对B0、Q-A0和Q2-F。不能因Q-A0很差就认为Q-P有部署价值。

---

## 6. M系列：只改形状loss分母，不同时改正例采样与正则

### 6.1 结构不动

M-C/M-N使用旧M1的shared h、原MaskHead、零末层和原始物化。每个candidate输出a[D]和b：

`delta(s,i)=2*tanh(F(s)·a_i/sqrt(D)+b_i)`

`new_logits=old_logits+delta`。

R1 F/q/z只读；原类别/score/candidate集合固定。残差施于原来所有有效segment，不只是正mask内。T1与T2均沿原真实物化，零残差不靠if分支绕过。

### 6.2 先提取未归一化的形状项

每步固定J=4个输入，每输入Cj=4个抽样candidate，总位置B=16。每candidate按现有每阶段最多2048个有效segment子集计算：

- BCE：按 `n_valid_points` 加权；
- Dice：沿用原soft target、相同权重和smoothing=1；
- 各有实际监督点的已观测阶段等权平均；
- 没有合法GT/shape_usable或没有有效监督点时，形状项为0。

定义 `p_ji = shape_usable AND assigned_gt>=0 AND has_sampled_valid_shape_points`。
定义 `ell_ji` 为candidate完整形状损失，保留梯度。`N_pos=sum p_ji`，这是本batch实际有形状项的位置数，不是GT实体数，也不是所有positive-IoU候选数。

正则与旧M1保持一致：

`R = (1/J) * sum_j mean_over_all_segments_and_sampled_slots(delta_j^2)`。

它包括ignore/未匹配输入的有限残差，**不**只在正例上计算，**不**改成按所有records拼接segments后的全局平均。

### 6.3 唯一两个公式

```
M-C: L = (sum_ji p_ji*ell_ji)/B + 0.01*R
M-N: L = (sum_ji p_ji*ell_ji)/N_pos + 0.01*R, if N_pos>0
     L = connected_zero + 0.01*R,              if N_pos==0
```

四输入各4位置保证旧“每record平均再平均”与 `/B` 等价；用真实batch检查M-C与修复旧M1的loss和梯度数值一致。若原修复版只改变了batch形状，先恢复本轮规定的4×4抽样再建立等价控制。若本地M1已经不是“全部候选位置归一化＋0.01原正则”，不能冒充等价：记录 `BLOCKED_M_CONTROL_MISMATCH`，保留实际差异，Q线继续；不擅自同时改assignment、正则来凑相同数字。

**只对形状项换分母。禁止把整个loss乘 B/N_pos**，否则0.01正则也被放大，实验就不再只检验形状学习权重。也不能随N_pos调整LR、改clip norm、换Dice或增大正例数量。

SoftGroup的公开实现使用有效mask标签权重的计数归一化mask loss，并对前景候选的IoU loss单独归一化；这支持任务分工的动机。**我们的candidate-level正例平均不是SoftGroup原公式的逐字复现。** [P02][E10]

### 6.4 训练记录必须解释有效监督

每100步记录sum/count，不只记录均值：B、N_pos、matched-but-no-sampled-points、unmatched、unknown、shape numerator、shape mean(all)、shape mean(pos)、0.01R、clip前梯度范数和clip触发次数。报告的是抽样位置出现次数，不伪称独立物体数。

在预选的3个TRAIN诊断batch上（第一个有监督batch，以及抽样计划500/1500对应batch；若无有效项则顺延到最近有效batch），分别计算shape和regularizer的梯度范数及其比值。不更新权重，不计作新训练；仅几次，不每步双重backward。

由此判断是有效监督稀释、梯度裁剪、正则影响还是几何表达不足，而不是见loss下降就认定成功。

---

## 7. 统一训练计划与旧控制复用

### 7.1 必须相同的项目

| 项目 | 本轮值 |
|---|---|
| new head seed | 首轮45；复核46 |
| optimizer | AdamW, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=1e-4 |
| scheduler | 一开始固定1500 updates，warmup75，然后cosine到1e-4；复用当前公式 |
| precision / clip | FP32；clip norm=1.0 |
| batch | 3个T2输入+1个T1输入；每输入4候选，共16位置 |
| sampling | reference→同H输入→candidate；与修复版固定seed计划一致；不按loss重抽 |
| shape segments | 每record每stage均匀≤2048有效segment，同seed同index |
| checkpoints | 0/500/1000/1500；last每100步用于恢复 |
| parent | 全程冻结eval；不进入optimizer |
| concurrency | 最多2张已确认空闲GPU，每卡1个独立小头任务；单卡可以顺序完成 |

Q-A0/Q-P同初始化同抽样；M-C/M-N同初始化同抽样；四臂可共享全局抽样计划，但不同监督资格不靠重抽进行平衡。不能将loss为0的更新计成伪造非零梯度；记录实际监督次数。

### 7.2 控制可复用的充分条件

Q2-F或M-C旧检查点只有同时满足下列条件才可代替新训练：父权重、TRAIN预测/descriptor、修复标签/assignment、抽样计划、网络/初始化、loss公式、optimizer参数、1500步scheduler horizon、seed、保存步数全部一致；本地完整state可读取，相关source和weight SHA已保存。使用抽样计划中的少量真实batch比较旧/新适配输出和loss即可，不重演整段训练来补日志。

条件不满足则重新训练对应控制。不能拿旧未修复Q2作控制；不能拿不同输入或不同分配的M1作M-C。数据一致但评价改为NOAUG时，可复用训练权重，必须在新评价条件下重新计分。

运行时记录 `updates_reused / updates_new / source_run`，避免重复收费或把复用更新说成本轮新增。

### 7.3 异常恢复

checkpoint应保存head、optimizer、update、scheduler horizon、sample-plan SHA、必要随机状态、父/标签/源码身份。新公式不能恢复旧公式的optimizer作为同一轨迹。代码实际修补导致训练数值变化时，归档该臂旧attempt，重跑受影响臂；无关臂继续。恢复不能重写旧点后还使用旧CAL分数。

第一次真正有监督的两步检查参数变化；零末层使trunk首步梯度可能为0是正常的，不能因此错误阻断任务。检查的是末层首步能动及随后trunk能获得梯度。不要每step比较整个R1字节。

---

## 8. 诊断必须用来解释结果，而不是开启无边界搜索

所有GT诊断只在TRAIN/CAL进行；SEL只按锁定配方验证和报告，不用来生成新网络/阈值。诊断不得改变正式选模规则。

### D-Q1：原评分、新评分与目标排序

在固定TRAIN诊断panel和完整NOAUG-CAL上，报告B0、Q2-F、Q-A0、Q-P的：修复标签加权MSE、分数分位数、按类别的排序错位、重复候选位置、partial-ignore候选升/降分情况、原来正确的排序被破坏数。

另做一次 **GT质量评分诊断**：固定同一候选集合，用分数无关的真实pairwise temporal质量替代分数，走官方evaluator。明确它是GT辅助诊断，不是可实现方法或AP上界。不能因候选在某个阈值被ignore，就从其他阈值或正式输出中删除它。保留重复候选；明确未知关系继续采用一个固定、预先写入诊断规范的无增益默认分数，例如原score，不依据AP择优。

输出分成“已能计算几何质量的候选子集”与“完整预测集辅助结果”，不得用子集AP冒充全体AP。若真实质量排序也不优，不自动判定所有学习评分不可行，只把目标与重复处理列为下轮候选。

### D-Q2：小头能否泛化

固定TRAIN panel只由预先hash最小的8个合法TRAIN reference、每ref至多2个T2及其去重T1组成；不足按实际。保存0/500/1000/1500的预测统计，区分TRAIN与CAL差距。采用同一panel，不在全TRAIN每step跑AP，也不从CAL挑困难训练样本。

只在 `Q-P` 与 `Q-A0` 的TRAIN目标误差均未优于一个仅输出TRAIN加权标签均值的常数基准时，仅对Q-P执行一次最多200步、固定16个可监督TRAIN候选的memorization探针（使用新初始化的临时head，不进入候选选择）。它验证输入/优化链路能否拟合，结果保存 `DIAGNOSTIC_ONLY`；探针失败不是数学不可能证明，不启动新LR网格。

### D-M1：有效监督与实际改动范围

继承assignment，对TRAIN panel及CAL的固定候选记录：原IoU、是否配对、监督次数、原正确/错误点数、`|z|>2`的错误segment和点数比例、实际delta均值/分位数、raw sign翻转数，以及最终full-eval多数决策后仍改变的点数。

严格大于2的同号logit在实数意义下不能被±2残差翻号；等于边界与FP32饱和另外计数，不把它们硬套严格推导。可修复比例是计数诊断，不是预测涨点。

### D-M2：宽松可达范围，不新建上界求解器

对固定candidate执行原单调物化T：`L=T(z-2)`，`U=T(z+2)`。原输出应位于L/U之间。对给定GT构造 `L ∪ (G ∩ (U\L))`，得到允许每个full point独立选择时的宽松mask。

它放松了共享segment和128维系数限制，只作为raw-mask IoU的乐观可达诊断；**不称官方AP上界**，尤其官方min-region处理可能改变细节。普通IoU与官方重新评价分别命名。所有GT构造只存诊断，不进部署或训练采样。

若宽松诊断已没有空间，说明当前残差/物化值得重新考虑；若有空间但小头没学到，仍须区分特征容量、监督和泛化，不能直接宣布“多对一即可解决”。

### D-M3：有限的拟合能力探针

只有M-N的主CAL没有正增益，并且TRAIN panel上至少存在8个合法配对、原IoU<0.9且宽松诊断提升≥0.05的candidate时，才执行此探针。

从该TRAIN集合按固定key取16个candidate（不足16但≥8取全部），只训练一个新seed145的共享MaskHead最多500步，lr=1e-3，按正例形状均值加原0.01正则。此panel全部具有有效形状目标，M-C与M-N分母相同，因此只需一个 `M_FIT_DIAGNOSTIC`，不重复训练两次来制造实验数。单独目录，0/250/500记录raw与物化后几何；不进入CAL/SEL选择，不重抽直到成功，不把训练拟合当泛化结果。若不满足条件记NOT_APPLICABLE并给真实计数。

### 下一轮候选只输出，不执行

生成 `NEXT_STAGE_DECISION.json`：
- 原评分被破坏/重复候选排序问题有实测证据：可建议Q-R；
- M-N缓解稀释但困难正例监督覆盖仍不足：可建议M-S；
- 现有表示/硬边界限制或局部拟合能力有实测缺口：可建议M-L；
- 当前正模块：建议后续独立确认，而不是自动组合。

每个建议带本轮真实行号/结果路径和限制。此文件没有自动执行权限；没有证据就写“仍需定位”，不填预期涨点。

---

## 9. 选模、跨种子和成果等级

### 9.1 CAL规则

全部四个新/配对臂及Q2-F桥接完成1500轨迹后，在NOAUG-CAL的500/1000/1500中，依次按T2官方pooled t-mAP、真正T1空间mAP、较早step选择。数值平局容差1e-6。0步只作诊断，不作为“学到的模块”入选；B0是独立fallback。

同一轮一次性锁定全部selected step后，才对所有臂评价完整NOAUG-SEL的T1/T2。某控制很差也要报告，不只给新方法表。原有无增强检查点结果可复用，但选择策略由本轮CAL重新固定，不能把以前看到的SEL称为未暴露。

**机制对照**：
- Q-P−Q-A0（主）；Q-P−Q2-F（桥接工程比较）；
- M-N−M-C（主）。

同时输出各自CAL选中点差值、CAL同step差值、SEL固定1500终点同step差值。SEL的1500行是预先规定的配对诊断，不能参与选点。selected step=1500则去重。

### 9.2 收益单位和门槛

所有机器文件用AP的[0,1]单位；报告同时显示%及pp，`0.001 = 0.1pp`，不得写成1pp。

`d2 = module_NOAUG_SEL_T2 - B0_NOAUG_SEL_T2`，`d1`同理。

- d2 < -1e-6：`NEGATIVE`；绝对值≤1e-6：`NO_GAIN`。
- d2 >1e-6但T1缺失或d1< -0.002-1e-6：`T2_ONLY_TRADEOFF`。
- d2>1e-6、d1≥-0.002-1e-6：`POSITIVE_SINGLE_SEED`；其中d2≥0.005-1e-6另记`TARGET_MAGNITUDE`。
- 0.5pp目标和0.2pp保护只是前瞻筛选规则，不代表统计显著。
- reference一致性单列；4个SEL reference至少3个d2>1e-6才记`REFERENCE_CONSISTENT`。即使没有3个，微弱正例也可执行固定seed复核，但不能称稳定。

没有达到0.5pp不自动当负结果；真实小增益与波动要区分。不用“确认名单必须非空”驱动阈值/epoch修改。

### 9.3 seed46固定复核

对Q-P或M-N中所有符合T1保护且d2>1e-6的候选，从相同R1重新初始化对应head，用seed46的新共同抽样计划。其直接控制同时从新初始化训练，调度horizon仍1500；只训练到所需固定比较点的最大值，不为了达到1500无条件多训。

在新方法的seed45选中step评价该方法与直接控制。若控制自己的已锁定step不同，最多再给控制该点，仍禁止seed46选epoch。去重同一arm/step。若Q-A0或M-C自己相对B0为正，也可作为工程候选按其已锁定step复核，不能因它是“控制”隐藏收益。

Q2-F不为了桥接例行再训seed46；只有它自身超过B0且需要作为工程候选保留时才按同规则复核。全部追加工作受预算约束，缺复核写`NOT_REPLICATED`。

`DEVELOPMENT_REPLICATED`要求两seed d2都>1e-6、T1都通过，并且seed45 reference一致性通过。另一seed负或零记`MIXED_SEED`；不挑正seed替换负seed。机制贡献另看同step控制差值，不能混成工程胜出。

### 9.4 本轮结束

本轮仅产出单模块开发名单。任何名单可以为空；不部署新默认，不运行Q+M，不声称超过作者官方ReScene。PB/LOCAL/ADDITIONAL正式确认不在本轮执行范围。

---

## 10. 资源与最小充分测试

### 10.1 成本

启动读真实最新账本，包含V2、旧六臂以及本地465f37a修复重跑。旧已知79.982308 GPUh只是旧轮累计，**不能忽略之后的修复费用直接计算余额**。按event_id和已结算prior快照去重；不把包含历史subtotal的账本再次全额叠加。

新增上限建议固定为 `min(16 GPUh, 192 GPUh - reconciled_prior)`，不是再授权192。预算表：

| 类别 | GPUh分配 |
|---|---:|
| 共享缓存补充/NOAUG导出 | 4 |
| 主要小头训练 | 2 |
| CALSEL及固定种子复核 | 4 |
| 有条件诊断探针 | 2 |
| 部署profile/重载 | 1 |
| 恢复余量 | 3 |
| 合计 | 16 |

可内部转移未花额度，不扩大总cap。开始前保留至少3 GPUh用于完整开发评价/profile/交付；用计划内前20步和前几个输入实测吞吐，不另开一套benchmark。预算不足先跳过诊断训练和第二seed，仍优先完成四个基本配对臂及完整CAL/SEL；无法完成就真实partial，不缩开发集挑容易数据。

CPU workers≤8、RAM≤96GiB、新增父缓存≤32GiB。用已有本地合法数据副本，NFS异常不无限等待。CPU评价不必占卡；按真实是否保留GPU进程记录reserved GPUh，父前向共享不等于部署免费。新增与累计费用分开。

### 10.2 必要测试

只覆盖本轮直接受影响链：

1. 465修复标签与partial-ignore/score-invariance，Q与shape资格分离。
2. Q-A0/Q-P初始化相同，QP零步score逐元素等于B0；QA0零步不是B0。
3. M-C/M-N网络/初始化/抽样相同，MC旧loss等价、MN只改形状分母；0正例也有有限梯度/正则。
4. 第一/第二个有效update头梯度可用，父R1不在optimizer。
5. Q只变score，M只变mask；多个class共享query时candidate key不混。
6. 真正NOAUG H1/H2输入和原物化；没有X3、没有通过GT分组推理。
7. 改label/head/apply/metric会让相应缓存失效；publish receipt不能跳过计算。
8. 完整分母、独立metric state、固定点、负结果/缺失为null；保存重载一致。

结束一次相关回归、一次定向lint和diff check。目标CPU测试≤20分钟、GPU smoke≤0.25GPUh；真实bug修复可超出并说明，但不扩成全库安全、fuzz、几百schema负例或重复旧checkpoint回归。测试数不是完成度。

---

## 11. 真实运行入口和依赖恢复

新增薄CLI，下面是待实现接口，不是声称当前仓库已有命令：

```bash
export RESCENE_QPMN_ROOT="${RESCENE_QPMN_ROOT:-$HOME/persist4d_runs/qp_mn_targeted_v1}"
# 按实际已安装解释器/环境执行，不要求安装新环境
python -m scripts.qp_mn_campaign run \
  --config configs/qp_mn_targeted_v1.yaml \
  --root "$RESCENE_QPMN_ROOT" \
  --base-commit 465f37a --resume
python -m scripts.qp_mn_campaign status --root "$RESCENE_QPMN_ROOT"
python -m scripts.qp_mn_campaign report --root "$RESCENE_QPMN_ROOT"
python -m scripts.qp_mn_campaign publish --root "$RESCENE_QPMN_ROOT" --resume
```

实现`run/status/report/publish`即可；内部阶段记录：
`bind → adapt/export → B0 → train-pairs → CAL-lock → SEL → replicate → diagnostics → profile → report → publish`。

既有代码或修复版已经提供同等入口则包一层适配，不复制旧campaign全部逻辑。`--root`和artifact_root必须参数化，禁止旧ARTIFACTS/V2_ROOT常量导致输出到旧目录。

run恢复按依赖检查重用。Q修复对照缺失只阻塞Q臂，不取消M；M assignment无法绑定只阻塞M，不取消Q。共同R1或评价人口不可用才共同阻塞。无论partial/negative都报告并发布真实已有工作。

---

## 12. 交付与GitHub实际同步

### 12.1 最小目录

```
artifacts/qp_mn_targeted_v1/
  EXECUTION_INSTRUCTION.md
  BASE_BINDING.json
  CODE_BINDINGS.md
  REPAIRED_CONTRACT.json
  RESOLVED_CONFIG.json
  INPUT_MANIFEST.json
  RUN_STATE.json
  EXECUTION_LOG.jsonl
  BUDGET_LEDGER.jsonl
  training/<arm>/seed<seed>/{resolved_config.json,metrics.csv,checkpoints/,checkpoint_manifest.json}
  evaluation/{CAL_ALL.csv,SEL_LOCKED.csv,SEL_ENDPOINT_1500.csv,BY_REFERENCE.csv,PAIRED_DELTAS.csv}
  selection/{CAL_LOCK.json,SHORTLIST.json}
  diagnostics/{QUALITY_RANKING.csv,SHAPE_SUPERVISION.csv,RESIDUAL_REACHABILITY.csv,NEXT_STAGE_DECISION.json}
  evidence/official_metric_states.zip
  resources/PROFILE.csv
  REQUIREMENT_REVIEW.md
  FINAL_REPORT.md
  HANDOFF.md
  ARTIFACT_MANIFEST.json
```

文件数量不作为目标，等价合并可以；缺失的任务写status和预期分母，不写空PASS。head包包含mode、输入schema、父SHA、label/assignment版本、训练step；raw R1/Concerto、原数据、原始GT不重分发。

### 12.2 表格必须呈现

主表至少列：B0、Q2-F、Q-A0、Q-P、M-C、M-N；实际新/复用步数、选中step、NOAUG CAL/SEL T2、delta B0、T1/delta、机制对照、seed46固定差值、正reference数、真实latency/VRAM、证据等级。补充增强结果另表，不与主表混算。

解释要回答：
- Q-P是否保住原排序？相对QA0的收益是否转成相对B0的收益？
- MN是否增加实际形状梯度，而不只是总loss数值变大？是否增加了高IoU候选、是否破坏原正确轨迹？
- 训练拟合、开发泛化、表达边界和评分不一致分别有哪些实际证据？
- 哪些属于已知代码修复，哪些是本轮主动方法改变？

### 12.3 真实部署profile与包重载

NOAUG-SEL每ref按固定hash取一对，共4对；相同空闲A40/原设备，B0及所有选中单模块各一遍warmup、三遍计时。包含父前向、描述符、小头、物化和可用输出，不把父特征共享当成部署免费；磁盘IO另列，metric/hash不计入。

每个选中包在已有真实输入上保存/重载比对一次；不额外全PB导出。通路等价而重复的控制可去重计时，但要明确别名与不重复收费。

### 12.4 发布，不再人为让小指标包依赖Release

用户已授权本任务代码与结果同步GitHub；应实际push，不只给命令。先用现有Git/Release权限，不索要token，不读取其他应用凭据。

- 代码、配置、曲线、所有正负结果、小头和不含raw GT的**充分统计指标状态小包**一起入本轮Git分支。
- 默认每小包≤20MiB，总新增二进制≤50MiB；能直接放Git的几MB指标包不要人为规定只能Release。不要把大权重切碎规避限制。
- 超过上述范围的必需大包，用已有授权Release；无授权则仍完成Git并列资产名/SHA/bytes/本地位置/可用补发命令。缺失必需资产就记CODE_ONLY，而不是用空Release冒充完成。
- 原修复提交如果只在本地，新分支push应使本次依赖的修复代码随祖先历史可取得；若修复证据在ignored本地文件里，复制必要的脱敏结果摘要/适配说明到新目录并绑定原SHA。不声称用户服务器raw cache也已上传。

发布顺序：
1. 只stage本轮明确代码/配置/产物路径，提交实验A。
2. 写HANDOFF引用A、生成manifest，提交B；manifest不hash自己，不把B嵌进B自身。旧tag/旧提交不重写。
3. 推送新分支，创建新tag `rescene-qp-mn-targeted-v1`（冲突最小-r2，不移动旧tag），推送并验证remote SHA。
4. 对本轮必需资产清单验证实际Git blob或Release可取得及bytes/SHA。发布完成状态写运行根外部 `publication/PUBLICATION_RECEIPT.json`，引用A/B/tag，不产生自引用。
5. 若全部必需资产已通过Git提供，允许 `GIT_COMPLETE`，Release未建记NOT_REQUIRED而不是伪造VERIFIED_RELEASE。只有额外可选Release缺失不降低科学状态；必需资产仍缺则CODE_ONLY。

不得合并main、force-push、改默认模型。最终答复给真实分支、完整B、tag、报告/交接/小头位置、增益清单和发布状态；没有涨点就明确没有。

---

## 13. 收口前一次审核：逐项接受真实结果

| 必查项 | 判定 |
|---|---|
| 本地465完整SHA/修复报告实际读取，未回退4bf标签 | 必须 |
| 4个配对轨迹及Q2-F桥接均完整或真实标阻塞 | 必须 |
| Q-P−QA0只改变父分数通路；MC/MN只改变形状分母 | 必须 |
| 新input NOAUG主评价、TRAIN条件共同、独立H1、参考人口无换样 | 必须 |
| partial-ignore修复和shape资格解耦，缓存不会无效复用 | 必须 |
| 选点来自CAL、所有固定SEL及1500端点预先规定 | 必须 |
| seed46固定点，控制和新模块不偷换调度 | 若触发 |
| 原分数/几何隔离、零步真实等价、完整指标可复算 | 必须 |
| 探针只在TRAIN、不进入选模、不扩大搜索 | 若触发 |
| 真实费用包括修复历史、不重复计prior；部署成本含R1 | 必须 |
| 真实push、tag和必需资产可得；缺项明确 | 必须 |

科学状态与发布状态分开。允许 `COMPLETE_NO_POSITIVE_MODULES`；若方法未跑完或证据不完整则PARTIAL/BLOCKED，不能用测试通过替代实验。允许单一方法或控制成为开发候选，不要求两个新方向都成功，不把未执行的Q-R/M-S/M-L写成失败。

---

## 14. 来源索引及证据边界

固定旧源S=`4bf00902c9428795f7f47547bf462540aa304cc6`。这些来源在编写指令时已读；执行基线仍必须绑定本地465。更多解释见同包 `EVIDENCE_AND_REVIEW.md`，不是另一份覆盖本文的指令。

- [E01] `models/short_module_heads.py`：https://github.com/Orangekostar/Persist4D/blob/4bf00902c9428795f7f47547bf462540aa304cc6/models/short_module_heads.py
- [E02] `scripts/short_module_data.py`：https://github.com/Orangekostar/Persist4D/blob/4bf00902c9428795f7f47547bf462540aa304cc6/scripts/short_module_data.py
- [E03] `scripts/short_module_training.py`：https://github.com/Orangekostar/Persist4D/blob/4bf00902c9428795f7f47547bf462540aa304cc6/scripts/short_module_training.py
- [E04] M1训练日志：https://github.com/Orangekostar/Persist4D/blob/4bf00902c9428795f7f47547bf462540aa304cc6/artifacts/short_module_screen_v1/training/M1/metrics.csv
- [E05] `scripts/short_module_native.py`：https://github.com/Orangekostar/Persist4D/blob/4bf00902c9428795f7f47547bf462540aa304cc6/scripts/short_module_native.py
- [E06] `scripts/rescene_task_postprocess.py`：https://github.com/Orangekostar/Persist4D/blob/4bf00902c9428795f7f47547bf462540aa304cc6/scripts/rescene_task_postprocess.py
- [E07] `scripts/p6a_metrics.py`：https://github.com/Orangekostar/Persist4D/blob/4bf00902c9428795f7f47547bf462540aa304cc6/scripts/p6a_metrics.py
- [E08] `scripts/short_module_evaluation.py`：https://github.com/Orangekostar/Persist4D/blob/4bf00902c9428795f7f47547bf462540aa304cc6/scripts/short_module_evaluation.py
- [E09] 原HANDOFF/4.7MB指标包：https://github.com/Orangekostar/Persist4D/blob/4bf00902c9428795f7f47547bf462540aa304cc6/artifacts/short_module_screen_v1/HANDOFF.md
- [E10] SoftGroup作者代码 `instance_loss`，本次读到blob `267cdc166e6a1765132d939829b0fe95a94fde7f`：https://github.com/thangvubk/SoftGroup/blob/main/softgroup/model/softgroup.py
- [P01] Mask Scoring R-CNN，CVPR2019：https://arxiv.org/abs/1903.00241
- [P02] SoftGroup，CVPR2022：https://openaccess.thecvf.com/content/CVPR2022/html/Vu_SoftGroup_for_3D_Instance_Segmentation_on_Point_Clouds_CVPR_2022_paper.html
- [P03] Rank & Sort Loss，ICCV2021（仅下轮排序动机）：https://openaccess.thecvf.com/content/ICCV2021/html/Oksuz_Rank__Sort_Loss_for_Object_Detection_and_Instance_Segmentation_ICCV_2021_paper.html
- [P04] Relation Networks for Object Detection，CVPR2018（仅下轮关系动机）：https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Relation_Networks_for_CVPR_2018_paper.html

文献支持机制动机，不提供本任务的预期涨点保证。本轮超参数是前瞻试验设计，不能写成论文原值或已有最优参数。
