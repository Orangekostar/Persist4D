# ReScene 第一轮：独立模块接入、单模块消融与多候选筛选

**日期：2026-09-28｜用途：交给 Codex 的执行指令｜本文没有声称新实验已经运行。**

## 0. 本轮唯一目标与边界

将上一份《ReScene T1–T2：24篇顶会论文与分阶段实验路线》中的首轮想法落实成可运行、可复算的模块实验，回答：**各模块单独加到同一个冻结 R1 上，究竟有没有收益？**

本轮只做 B0 加六个独立实验配方：Q1、Q2、Q3、M0、M1、M2。完成开发、真实训练、CAL 选点、SEL 筛选、必要的固定点第二种子复核、单模块成本、结果与 GitHub 交接。**不做任何 Q+M、两个 Q、两个 M 的联合推理、联合训练、集成或串联；不选最终组合模型，不替换现有默认部署。**

这里的“六个”是六个实验配方，不是六个都能叠加的独立网络：
- Q1/Q2/Q3 是同一评分插槽的互斥方案。
- M0/M1/M2 是同一 mask 修正插槽的互斥方案；M0 主要是监督对照。
- M1 是监督策略变化，不是可单独插在推理链上的网络层。
- M2 使用与 M1 一样的监督以控制变量，但从 R1 和新初始化的小头独立训练；**不加载 M1 学到的参数，不先运行 M1 再运行 M2**。

本轮产物是 `MODULE_SCREENING.csv` 和 `SHORTLIST.json` 等真实结果，可以保留多个合格单模块，也可以为空。不要沿用 V2 “只许一个 full 冠军”的选择方式。论文收益是设计动机，不是我们的预期实测数字。

### 固定身份

| 项目 | 值 |
|---|---|
| 仓库 | `Orangekostar/Persist4D` |
| 代码起点 | `1ddab2aee88e5f5d8aa50b73a7896341802b2a26` |
| 新分支 | `research/rescene-short-module-screen-v1` |
| 新产物目录 | `artifacts/short_module_screen_v1/` |
| 本地运行根 | `${RESCENE_SHORT_MODULE_ROOT:-$HOME/persist4d_runs/short_module_screen_v1}` |
| 父感知权重 | R1，已完成 epoch390 的 selected checkpoint |
| R1 SHA256 | `629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd` |
| R1 bytes | `754813672` |
| Concerto SHA256 | `845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07` |
| 主指标 | 原生双扫描输入上的官方 pooled T2 t-mAP |
| 保护指标 | 真正只输入一个扫描的 T1 空间实例 mAP |
| 实验性新增占卡上限 | `min(48 GPUh, 核实后的原192 GPUh累计余额)`，不自动另获192 |

只使用现有许可的数据、已知路径及权重。不改历史 V1/V2 产物、tag、分数或训练身份。从固定提交开独立 worktree；发现用户未提交改动应保留。分支已存在且属于同一任务时恢复，无关同名分支用最小 `-r2` 后缀。

**本轮明确不做：**390/450续训、LoRA、全量解冻、Q-SEM查询初始化、S-BAL/S-WORST、A-OPEN、12配置关联复跑、D0/lag1/重激活、T3–T5训练和计分；不做 Q4/Q5/M3/M4，不产生第二轮组合结果。输入侧不增加 X3 或未来扫描。

---

## 1. 依据与本轮对上稿的精化

### 1.1 已核实事实，不作为新实验成绩

1. 当前 R1 在一次查询解码循环中反复使用同一 mask 特征，mask logits 来自特征与 query-derived embedding 的点积。[E01]
2. 已有 `OfficialTaskSoftEvidence` 包含 logits、query/segment features、类别和候选谱系，可在同一次真实前向后导出。[E02]
3. `materialize_segment_logits()` 会 detach、搬到 CPU 并以 `>0` 二值化；它适合评价，**不适合作为几何头训练的可微 loss 路径**。[E02]
4. 已有 `official_temporal_match_trace()` 返回的是官方按分数匹配后的判定；不能将其 TP/FP 直接当成新评分器的独立质量监督。[E03]
5. V2 未选中任何新组件；原修复数据及局部纠错不能当作本轮 native-T2 模块的效果。[E04]
6. `LOCAL-T2` 原始154序列实际覆盖46个reference，且与PB/ADDITIONAL重叠；原代码已记录更正。[E05]

### 1.2 采用的论文思想

| 实验 | 主要来源 | 此处是怎样的迁移 |
|---|---|---|
| Q1 | Mask Scoring R-CNN / IoU-Net | 从最终实例信息学习几何质量；不是完整复现原网络 |
| Q2 | Mask Scoring R-CNN / GFL / VarifocalNet | 将监督换成官方 temporal overlap；保持输入和头结构 |
| Q3 | GFL / Rank-DETR 等质量与排名思想 | 直接建模多个官方 IoU 阈值事件；具体头为本项目设计 |
| M0 | CondInst / QueryInst 的实例条件预测 | 公共残差头与空目标监督控制，不是V2复现实验 |
| M1 | SoftGroup 的分类/形状监督分工 | 不再强迫未匹配候选整张清空；此臂不运行任何新质量头 |
| M2 | SyncVIS / QueryInst | 同一实例身份、分阶段局部表示，只有局部描述符的聚合方式变化 |

SoftGroup 的正例标准、SyncVIS 的视频输入、Mask Scoring R-CNN 的分类乘分方式不直接照搬。我们分别明确数据、标签和输出契约，不声称上述论文已经在 ReScene 上验证本方案。[P01–P06]

**对上稿的一个精化：**上稿 M1 同时提出“允许可靠重复候选获得多对一目标”和“不把未匹配都置空”。为隔离原因，本轮先不增加多对一分配；M0/M1 共用同一冻结的一对一分配，仅改变未匹配项的形状监督。多对一扩展以后另测，不在本轮偷偷加入。

---

## 2. 模块表：每个配方从同一个 B0 独立出发

| ID | 开关 | 允许改变 | 不允许改变 | 主对照 |
|---|---|---|---|---|
| B0 | `module=none` | 无 | 所有原始输出 | 固定基准 |
| Q1 | `module=quality_concat` | 最终分数 | mask、类别、候选集合、NMS/top-k、输出谱系 | B0 |
| Q2 | `module=quality_temporal` | 最终分数 | 同Q1 | B0、Q1 |
| Q3 | `module=quality_threshold` | 最终分数 | 同Q1 | B0、Q2 |
| M0 | `module=mask_shared_empty` | 保留候选的mask | 原分数、类别、候选集合、原R1权重 | B0，监督控制 |
| M1 | `module=mask_shared_preserve` | 同M0 | 同M0 | B0、M0 |
| M2 | `module=mask_stage_preserve` | 同M0 | 同M0 | B0、M1 |

配置只含一个枚举 `module`，不要采用可同时打开多个模块的布尔开关集合。代码可以设计统一插槽，但本轮运行器明确拒绝组合列表。每个 head 的初始化、优化器、checkpoint 和输出目录独立；公共 R1、只读缓存和训练抽样计划可以共享。

---

## 3. 最小代码阅读与绑定

先读以下已有符号及直接调用方，生成一张真实 `CODE_BINDINGS.md`，标注当前代码起止行和本轮接入点；不要重新进行全仓库审计。

| 现有位置 | 阅读目的 / 最小动作 |
|---|---|
| `models/rescene.py::forward/aggregate_features/mask_module` | 明确F、q与原logits来源；无需修改主decoder训练，导出只读特征 |
| `scripts/rescene_task_postprocess.py::OfficialTaskSoftEvidence` | 复用soft字段与retained candidate谱系；增加不影响默认行为的适配 |
| 同文件 `extract_official_task_prediction/materialize_segment_logits` | 分离评分/几何路径，保留原候选，建立真实零残差等价 |
| `trainer/trainer.py::_get_predictions/_get_mask_and_scores/_filter_and_sort_predictions` | 原函数可能修改传入预测字典；同一前向只处理一次，禁止二次softmax或混用已修改logits |
| `scripts/system_comparison_inference.py::FullHistoryPredictionProducer/postprocess_full_history_output` | 复用原生T2数据与输出规则，不通过D0输入缓存 |
| `scripts/perception_gain_native_evaluation.py::run_native_checkpoint_evaluation` | CAL/SEL 的真实输入顺序、标签映射与T2人口 |
| `scripts/perception_gain_local_evaluation.py::run_local_t2_evaluation/audit_local_population` | 了解已暴露确认人口；本轮不把LOCAL拿来选模块 |
| `datasets/semseg.py::load_scan_indices` 与真实collator/voxelizer | T1只载入一个扫描；TRAIN GT、ignore、point/segment映射的来源 |
| `models/perception_gain.py::derive_segment_stage_ids` | 严格阶段映射；不得用平均阶段掩盖跨扫描混合分区 |
| `scripts/p6a_metrics.py::OfficialMetricAccumulator/official_temporal_iou_thresholds` | 唯一正式评价口径，读官方阈值 |
| 本地已安装 `stmetrics/instances/matcher.py` 与 `evaluator.py` | 使用其pairwise重叠、有效性、ignore/ambiguity语义，不用按分数匹配结果生成质量标签 |
| V2 `DATA_ROLES.json`、`INPUT_MANIFEST.json`、`data/STAGING_MANIFEST.json`、预算账本 | 复用已知数据/权重定位和累计费用；不复用不等价soft缓存 |

建议新增一个小模型文件、一个数据/标签适配文件、一个运行器，例如：
`models/short_module_heads.py`、`scripts/short_module_data.py`、`scripts/short_module_screen.py`。
必要时拆出 evaluator，但不要复制整套 V2 campaign、另建审计平台或分布式状态库。上述文件名/CLI为**需要实现**的接口，不是宣称现在已存在。

### 3.1 一致的输入输出合同

父系统按原 top-k/filter 执行后得到 `N_c` 个候选，**N_c不必等于原始query数100**。候选键为：
`(input_id, source_query_id, source_class_id, retained_index)`。

同一raw query可能对应多个类别候选，不能用query_id作为唯一候选键。几何头可以按已保留的类别候选独立修正，公共输入包含该候选的类别one-hot；所有几何对照都采用同一规则。类别本身不修改。

只读输入至少包含：
- `F[S,D]`：FP32 segment mask特征；`q[Nc,Dq]`：归一化query特征；`z[S,Nc]`：原始logits；
- 原始保留的 `scores/classes/source_query_ids/source_class_ids`；
- 原始全分辨率bool mask，以及low point2segment、inverse map、full eval partition；
- **每个扫描自身的**原始vertex ID、扫描ID、阶段、原始坐标，及由其确定的segment centroid；
- 已观测阶段数H=1或2。不能填入GT presence、GT框或GT prompt。

`D/Dq/C`由实际checkpoint与张量读取并冻结到run config；不因文档中常见128而静默裁切/补零。公共头输出分别是 `new_scores[Nc]` 或 `delta_logits[S,Nc]`，不改列数。

预测数据和 `train_targets` 分开序列化；GT可以生成训练标签与评价诊断，但推理wrapper不接受GT参数。仅训练采样用到的candidate标签有效性不得进入推理support选择或head输入。`point2segment`仅允许来自原有无监督预处理分区，不用GT实例分组替代。

---

## 4. 原生输入、数据和缓存

### 4.1 本轮角色

使用V2固定reference分工，不重新随机挑容易的CAL/SEL：
- CAL：4 refs/23逻辑单元，取各单元已定义顺序的前两个扫描形成T2输入。
- SEL：4 refs/24逻辑单元，同样只载入前两个扫描。
- 若上述逻辑单元对应重复的物理扫描对，保留原逻辑权重用于可比性，另报unique pair数，不能说23/24是独立场景。
- T1-CAL/T1-SEL：各自T2输入中出现的扫描按**物理scan_id去重**后，每次只输入一个扫描。不得将T2预测切出第一阶段充作T1。
- TRAIN：从V2的TRAIN refs中，按 `SHA256('short-module-v1:'+reference_id)` 固定取前64个有合法真实T2样本的reference；每ref按同样hash顺序取最多4个有向扫描对，不足4按实际保留。该选择只看合法元数据，不看GT难度、预测分数或验证结果。
- T1-TRAIN：上述扫描对中的所有unique扫描，每个只读取一次扫描输入并独立前向。
- 少于64 refs仍可用实际全集；不足8个合法训练reference则该训练范围 `BLOCKED_TRAIN_COVERAGE`，不能拿CAL补训练。

沿用原R1处理分辨率、序列内时间编码、坐标处理、类表、preprocessing、min-region-size；不混入ScanNet新头训练来同时改变域混合。本轮是RIO小头筛选，不是复现R1完整训练。

启动前实际建立 `SHORT_POPULATION.json`，记录TRAIN/CAL/SEL全量输入键、unique refs/scans/pairs、各H分母。读V2 manifest核实TRAIN与CAL/SEL/PB/LOCAL/ADDITIONAL之间物理reference没有交集。既有父模型预训练暴露单列，不能以新头TRAIN干净声称父模型从未见过数据。

**本轮不评分PB/LOCAL/ADDITIONAL，不产出全T胜出结论。** 它们留给单模块筛选结束后另行授权的确认/组合阶段；历史暴露不能撤销，也不称其全新untouched。

### 4.2 T1专用路径

不能直接假设旧native脚本接受horizon=1。复用 `load_scan_indices` 和原collator，以单个scan index构造独立样本，时间归零；确认原始文件读取、坐标归一化、邻域、特征和soft数据均未使用另一个扫描。旧stage2专用change loss不参与父网络只读前向；T1评分用 `LegacyAPEvaluator` 的空间AP。T2中的stage1/stage2 AP单列为 `AP_stage1_given_T2/AP_stage2_given_T2`。

### 4.3 固定运行与缓存

R1所有参数 `requires_grad=False`、始终eval；新增头独立train。父R1用FP32、固定eval seed45和锁定数值环境。训练seed45/46只改变小头初始化与TRAIN抽样，不改变CAL/SEL父预测或评价种子。

缓存键至少包含父weight SHA、实际producer源码digest、input manifest、H/scan顺序、推理配置、seed、库版本、TF32/cuBLAS/thread配置、postprocess策略及分区身份。不要把V2 D0/lag1缓存、T2切片或别的checkpoint数据标成native B0。

一次真实父前向导出供六个模块共享；缓存落本地，32GiB上限，超限按input分片/流式，先删除可重建的中间副本，不重复完整父前向六遍。数据GT不发布。训练期间保留可继续运行所需输入；原始dataset staging与新soft缓存空间分开计算。

native soft导出须与原native输出在一个真实CAL pair上的mask/class/score一致，再完成全CAL/SEL B0。旧历史分数只参考；不根据是否重现旧数字换环境择高分。基线发生实际漂移，记录当前输入和输出并统一重生成受影响缓存，不无限排查历史。

---

## 5. 一次必要诊断，不作为关闭所有模块的门槛

用同源B0-CAL和TRAIN候选，随导出完成：
1. 在两阶段均有有效GT、无明确身份歧义的对象上，计算
   `U_joint=max_i min_t IoU(i,g,t)`、`U_separate=min_t max_i IoU(i,g,t)`；分别报告每个官方阈值的计数和per-reference。
2. 候选分成有可靠对应、无同类GT、低重叠、重复候选、unknown/ignore；报告真实分母，不沿用旧81.1%。
3. 对低/高分候选列出几何质量分布，记录排序错位；不以oracle score做部署结果。
4. 新增/消失实例单列；同一阶段双方均无mask的处理完全来自当前official matcher。

这些都是诊断量，不是可实现AP上界。第一轮不增加完整超点上界求解器或大规模可视化项目。诊断失败只阻塞依赖它的解释项；只要正式标签与评价正确，六个模块继续。

---

## 6. 公共局部描述符：固定，不按验证集调参

对每个candidate i和每个已观测阶段t：
- `A_it`：该阶段原segment logit > 0的集合。
- 若A非空，取其segment centroid的轴对齐包围盒，每轴固定扩展0.10m；`R_it`为盒内该扫描segments。
- A为空时，用该阶段logit最大的最多32个segment作局部support，按stable segment key打破平局。
- 另外从该阶段未在R中的segments按固定hash取最多32个exploration segments。推理与训练使用同一规则，不使用GT边界。
- `h_it = concat(mean(F[A_it]), mean(F[R_it union exploration]))`；A为空时第一项零向量。平均均为按segment的等权平均，避免不同scan点密度私下改变模块。

因此 `h_it`维度2D。T1只有h_i1，不生成假第二帧。

Q系列共同输入：`q_i`、阶段h的均值、H=2时两h的绝对差(H=1时零)、父类概率向量、候选类别one-hot、父原score，以及两阶段的支持比例/前景内部平均概率的min和max、观测阶段数H。所有字段顺序、归一化、缺省值写manifest并共同使用。LayerNorm只用于q/h且不学习父特征；标量本来在[0,1]，H用H/2。不增加Q4专属不确定性特征。

M系列共同输入为 `q_i + h + candidate_class_one_hot` 的拼接；M0/M1的h是所有输入阶段h的等权平均，M2对每阶段使用自己的h_it。q已经包含双扫描上下文，所以M0/M1不称“完全无时序信息”。

---

## 7. Q1/Q2/Q3：独立质量评分模块

### 7.1 不循环的质量标签

从本机**已锁定版本**的stmetrics matcher产生的pairwise重叠和有效性记录中取标签；在输入类别相容的有效GT中，以temporal overlap最大为第一条件、拼接IoU次之、GT stable ID最后打破平局，确定同一个 `g*_i`。
- Q1标签为i与g*的拼接IoU；Q2标签为同一对的official temporal overlap。
- 无同类有效GT且可判定为有效预测负例时，两者标签为0。
- 明确unknown、官方 `_proportion_ignore` 大于主t-mAP最小阈值、或涉及无法独立解析的明确ambiguity组时，从三个Q头的监督中共同排除，计数与理由入表。比例分母为0的候选不计算伪比例；正式评价仍保留全部预测、采用官方ignore/ambiguity规则。若原标注未提供ambiguities，记录 `AMBIGUITY_METADATA_UNAVAILABLE`，不得声称已排除所有现实歧义，也不凭此把整个人口删除。
- 用当前evaluator的 `_valid_gt/_proportion_ignore/_select_overlap` 语义及pairwise记录，不调用 `_evaluate_class/_match` 的分数排序结果来构造质量标签。
- 若候选全部时间阶段都为空或未达到official prediction有效条件，明确标 `INVALID_PRED/EMPTY`，不伪造成功质量；必要时监督为0仅用于可判定负例，其余mask掉。
- 修改pred_scores或打乱候选列顺序，按candidate key重对齐后的标签必须不变。

Q3：`y_ik=1[official_temporal_overlap(i,g*) > tau_k]`。tau读 `official_temporal_iou_thresholds()`，原始精度与数组写run config，另外检查该helper与本地head计算主t-mAP的阈值一致。缺失阶段与双方空阶段的定义不能用手写“两个阶段都必须有前景”替代。[E03][E06]

Q1/Q2中的“同一g*”是刻意控制目标归属；Q1不是每个候选在拼接口径下独立最优GT的oracle实验。多候选可有高几何标签；重复抑制不由本轮新规则决定。

### 7.2 头结构和损失

公共MLP：`Linear(Din,128) → GELU → Linear(128,64) → GELU → output`。实现时避免对类别one-hot/H等标志与连续特征作不一致归一化；以第6节已经规范化的拼接作为MLP输入，**不再做覆盖整个拼接向量的额外LayerNorm**。

- Q1/Q2：output=1，sigmoid得到[0,1]分数；loss=MSE；权重1。
- Q3：output=K。令 `a1=r1; ak=a(k-1)-softplus(rk)`，`p_k=sigmoid(a_k)`；loss为各阈值BCE均值，最终score=mean(p_k)。increment bias初始化-4、首logit bias0；不要把硬cummin用于训练而忽略梯度。
- Q1/Q2同seed使用完全相同参数初始化。Q3共享同seed的trunk初始化，输出维度改变必然有少量参数差，需如实列出，不声称参数数目严格相同。
- 综合标签包含预测类别正确与否，因此头输出直接作为新score，不再乘父score或父类概率。父score可作为输入，但不是另一个部署乘项。
- 没有新增top-k、NMS、阈值删除或mask变形；评价器可按新score排序，但candidate key集合严格不变。

**Q头的0步不是恒等变换。** 可以记录其未训练输出，但不要要求它等于原分数，也不要让未训练的偶然排序成为入选模块。禁用头走B0；实际选点仅为500/1000/1500。Q2/Q3在T1时使用同一个头，标签退化为单扫描overlap/阈值事件；不能用T2父预测伪装T1。

质量MAE/Brier和同类排序错位只做诊断。训练是reference-balanced抽样，输出首先称“质量排序分数”，不能未经校准评估宣称真实部署概率已经校准。

---

## 8. M0/M1/M2：独立几何修正模块

### 8.1 公共网络

`A = Linear(Dq+2D+C,128) → GELU → Linear(128,D+1)`，最后线性层weight/bias全部0。输出 `a_it[D], b_it[1]`。

对全部该阶段segment s：
```
r_it(s) = (F_t(s) dot a_it) / sqrt(D) + b_it
Delta_it(s) = 2*tanh(r_it(s))
z'_it(s) = z_it(s) + Delta_it(s)
```

- M0/M1使用同一合并h，因此同candidate两阶段共享a/b；M2使用h_it，每阶段计算a/b，但A的权重跨阶段共享，无新时间位置embedding。
- 三头参数数量、随机初始状态完全相同；彼此不加载训练过的权重。T1时共享/局部聚合等价，但独立训练后结果不强求相同。
- 残差施加于全部有效segment，不只原正mask内，避免结构上禁止补回遗漏。
- ±2是这轮固定可表达范围，不是论文结论；不得依据CAL临时调成4/8。零残差经真实物化精确回到B0，不以特判直接返回旧bool绕过检查。
- R1 F、q、z只读，head启用autograd。训练时不要把head包在`inference_mode`里。现有`materialize_segment_logits`仅用于评价；几何loss直接作用于float logits。

### 8.2 同一个冻结的形状目标分配

在TRAIN上，以保留candidate为单位，按预测类别与GT类别相容性建立拼接mask IoU矩阵；每类用现有稳定的一对一最优分配，接受IoU≥0.10的配对。只使用有效、可解析GT；明确歧义候选不监督而计数。这个GT分配只生成新增头的训练目标，不更改R1 matcher、候选身份或推理。

一个candidate一旦分到GT实体，两阶段必须使用**该同一个实体ID**的mask。不能分阶段选最好的不同GT。GT在某阶段确实不出现，该阶段目标为零，不因存在性变化取消惩罚。

M0/M1/M2共用完全相同的assignment manifest和candidate采样顺序。M1/M2首轮**不增加多对一分配**。

### 8.3 唯一监督差别

| 训练项 | M0 | M1/M2 |
|---|---|---|
| 有合法GT配对 | BCE+Dice分割 | 完全相同 |
| 可判定但未配对 | 全零mask目标BCE+Dice | 不做擦空监督，只做零残差保持正则 |
| 明确unknown/ignore | 不做无依据的mask监督 | 相同 |
| 全体可用输入上的正则 | `.01 * mean(Delta^2)` | 相同 |

M1并未运行任何新评分器。所谓“监督分工”在本轮仅表示把候选抑制从形状头中撤出；原R1评分负责现有排序，新质量评分的组合留到以后。

### 8.4 segment目标与可微训练

由 `low_point2segment[voxel_inverse]` 得到每个full vertex对应的**低分辨率输出segment**，以该segment内有可靠标注的full vertices中GT占比作为soft target，计数n_valid作为权重。不要使用名字相似但不同的full-eval partition来给训练segment编号。

background/stuff与其他实例是合法负点；255/unknown不能未经原数据语义处理一律当负点。有效点mask由真实原始标签/预处理映射提取，记录来源；不得用“所有GT mask的并集取反”推断全部ignore。缺原始细粒度unknown信息时沿用已绑定数据的明确语义并披露限制，不能声称新增了完整消歧。

每candidate每阶段最多均匀抽2048个有效segment，不足取全部；采样种子与索引共享。BCE和soft Dice都按n_valid加权；每candidate对有输入且存在有效监督点的阶段等权平均，然后在同一batch的**全部抽样candidate位置**上平均。M1未匹配项的分割loss为0，分母不改成只有正例数，从而不暗中提高正例权重；正则仍正常计算。Dice smoothing固定1，空匹配阶段也按这一公式处理。

评价顺序：原保留candidate logits加Delta → low point2segment展开 → `>0` → inverse map → 原full-eval segment多数决策。原score/class固定，candidate列固定；修正为空的列保留在导出中，正式matcher按原规则处理，不借二次过滤制造增益。

---

## 9. 统一训练和运行计划

### 9.1 初筛六臂全部完成，不只延长冠军

| 参数 | 固定值 |
|---|---|
| 新头训练seed | 45 |
| optimizer | AdamW，lr=1e-3，weight_decay=1e-4，betas=(.9,.999)，eps=1e-8 |
| schedule | 从启动就定1500步；warmup75，然后cosine到峰值的0.1 |
| grad clip / precision | 1.0 / FP32 |
| checkpoint | 0、500、1000、1500；每100步last用于异常恢复 |
| 每步采样 | 4个输入记录：3个T2、1个T1；每记录均匀取4个可训练candidate，共16 |
| 数据流 | 先均匀采reference，再均匀取该H记录，再均匀candidate；不足4候选有放回；所有模块同seed用相同表 |
| 上游 | 冻结eval R1，不新增R1 optimizer |
| 并发 | 最多2张已确认空闲A40，每卡一个任务；不干扰用户其它作业 |

没有candidate的记录不进入头训练抽样，但仍完整参加评价，记录其数量。不同模块的label有效性掩码可以不同；不能按其loss重抽容易候选。整个batch无分割正例时如实执行可用的正则/评分监督，记录，不伪造非零更新。

每100步记录loss分项、learning rate、监督正/负/unknown数、实际更新数；第一次两个有有效监督的更新检查head梯度/参数确实变化、父权重不变。只做必要检查，不每step hash几百MB父权重。

六臂在同一1500预算下完成。异常只影响本臂，其它能运行的继续；不要以Q1没有增益就取消Q2/Q3，也不要因M0失败阻断M1/M2。有限预算不足时标partial，不能把未跑的方案归为无效。

### 9.2 预算

原公开累计约78.94GPUh只是起点参考，启动先读实际账本并计入此后新增占用，旧已消耗不归零。当前轮计划新增≤48GPUh且受总192余额约束：证据/基线导出20、小头训练和固定点复核8、CAL/SEL/成本测量16、必要恢复4。各项是分配，不是运行时间承诺；未花额度可内部转移，不扩大总cap。

先保留完整CAL/SEL以及本地模型重载和Git发布所需费用。用真实前20个计划内update/前几个输入估计剩余，不另开benchmark。若48或累计余额无法覆盖，缩减**尚未启动的第二种子**，保留所有s45基础单臂与对照优先；连六臂和完整开发评测都无法容纳则真实部分交付，不改分母、不缩小CAL/SEL挑容易样本。

CPU workers≤8，RAM预算96GiB；标签/评价尽量CPU，GPU只用于实际前向/新头训练。失败、等待占卡和重跑计费；小头缓存训练快不是漏计父特征生成成本的理由。

---

## 10. 评价、筛选与多候选保留

### 10.1 官方指标

T2使用锁定stmetrics的TemporalEvaluator及原类表/ignore/歧义/大小规则，pooled汇总整个人口。T1用只输入单扫描的LegacyAPEvaluator空间AP；stage|T2另表。每模块、每checkpoint、每H独立metric state，不能复用已更新的accumulator。

Q臂逐候选几何tIoU、mask/class谱系必须与B0相同；官方召回可能受排序匹配影响，不据此声称新增几何召回。M臂score/class谱系必须与B0相同；同时报告joint通过率、最差阶段IoU、修复/破坏轨迹数。AP未提升但几何改善，记录 `GEOMETRY_SIGNAL_ONLY`，不标正向t-mAP。

### 10.2 CAL只选每模块的一个固定点

完整完成1500轨迹后，每臂在500/1000/1500中按CAL T2 AP降序、CAL T1 AP降序、step较早排序，1e-6作数值平局容差。M/Q未训练0步单列但不作为“学到的增益”入选；B0是所有头的独立fallback。

全部六臂的CAL-selected step一次性写 `CAL_LOCK.json`，再对每臂这一个点做完整SEL的T1/T2。**六臂都测SEL，不限只测冠军**；不根据SEL再换step/LR/阈值，也不把达到目标前持续调参算第一轮。

### 10.3 两个相互独立的收益字段

令 `d2=SEL_T2(module)-SEL_T2(B0)`；`d1=SEL_T1(module)-SEL_T1(B0)`，均按[0,1]存储。`ref_positive_count`为各SEL物理reference的T2差值>1e-6的数量。

- `NEGATIVE`：d2<-1e-6。
- `NO_GAIN`：|d2|≤1e-6。
- `T2_ONLY_TRADEOFF`：d2>1e-6但d1<-.002-1e-6，或T1未覆盖完整。
- `MODEST_GAIN`：0<d2<.005，且d1≥-.002（按1e-6比较）；如T1缺失不进入此类。
- `TARGET_GAIN`：d2≥.005且d1≥-.002。

以上是本轮筛选阈值，不是统计显著性结论。再单列一致性：完整4个SEL reference时，至少3个d2>1e-6才记 `REFERENCE_CONSISTENT`，否则 `REFERENCE_MIXED`。不把129个相关单元当独立样本。

**机制收益另列：**Q2−Q1、Q3−Q2、M1−M0、M2−M1，分别提供“各自CAL选中点”和“相同步数”差值。超过B0但不超过机制控制时，可保留工程收益，但不宣称该新机制贡献已证实。M0如有收益也如实报告，不因它是控制而隐藏。

### 10.4 固定点seed46复核，不做新冠军搜索

对于完整SEL后d2>1e-6且T1保护通过的所有单臂，预算允许时从同R1、重新初始化head，seed46训练到其s45选定step；只在固定step评价，不再次选择epoch。为机制比较，给其直接控制补同固定step的seed46，按arm/step去重；若某控制已有更长轨迹，使用中间固定checkpoint。

例如M2通过但M1未通过，仍需M1-46匹配点来判断局部表示是否有收益；这不是部署组合。最多仍是六种头，各自最多1500，不能据此新增搜索配方。

`DEVELOPMENT_REPLICATED`：两seed相对B0的d2都>1e-6、T1保护都通过，且s45满足reference一致性；附真实per-reference表。第二种子失败记 `MIXED_SEED`；没跑记 `NOT_REPLICATED`。不以正seed替换负seed，不声称跨数据集普遍有效。

`SHORTLIST.json`保留所有满足开发复核的独立arm，不限数量；单seed正信号另列 `provisional`，T2-only/几何-only另列tradeoff与diagnostic。同一插槽的方案标 `ALTERNATIVES_NOT_STACKABLE`，不要出现“Q1+Q2+Q3”的建议组合。

本轮只生成未来兼容性元数据（score_only/mask_only、父权重、谱系、标签版本）。**不执行未来2×2、不重新训练新mask上的质量头、不生成最终组合分数。**

---

## 11. 真实单模块开销与交付验证

从SEL每个物理reference固定取hash最小的一对扫描，共4对（不足按实际并说明）。同一空闲A40，B0与各CAL选中单臂分别一遍warmup、三遍测量：输入准备/H2D、父网络、局部描述符、新头、物化、可用输出端到端、peak allocated/reserved、CPU RSS。单独记录磁盘IO；metric与hash不计部署时延。

研究阶段可共享F/q，部署profile必须包含每个方法实际需要的R1前向，不能只测0.1ms小头就称整个系统更快。不强行设置成本否决门槛；质量与成本分别报告。

小头部署包记录mode、父weight SHA、D/Dq/C、目标/阈值、描述符、postprocess、输入H支持及训练step。至少每臂在已有真实输入上做一次保存/重载输出一致检查；不重复原所有PB预测导出。

---

## 12. 最小充分测试与有限恢复

复用已有用例，新增只覆盖真正影响分数的接口：
1. module只能取一个枚举；各臂从R1独立加载。
2. disabled严格回B0；M零残差经真实bool路径等价；Q0不错误宣称恒等。
3. 评分不改mask/class/candidate；几何不改score/class/candidate；同query多class谱系不混。
4. T1不读第二扫描，T2不读第三扫描；不同物理扫描不按vertex ID硬对应。
5. quality标签不随pred_scores或候选顺序变化；正确出现/消失、双方空、ignore和歧义用official规则。
6. M训练有head梯度；materialize仅用于评价；segment标签来自低分辨率分区映射；一条candidate全阶段同一GT。
7. M0/M1同assignment/同抽样/同初始；M1未匹配项不改变loss平均分母；M2只改描述符聚合。
8. metric独立state、完整分母、缺失为null；SEL只选一个已锁定step、多个模块可保留。
9. checkpoint模式/数据/父SHA一致，真实重载输出可复算。

结束一次相关回归、一次定向lint、一次diff check。目标CPU测试墙钟≤20分钟、GPU smoke≤0.5GPUh；必要实际前向计入预算。不做全仓库安全测试、fuzz、成百上千schema负例，不重建无关历史checkpoint。真实bug需修并重跑受影响用例，不能拿时间限额放过错误。

NFS/缺数据使用已有本地合法副本；只查已知manifest路径，不全盘找文件。无可读输入时不无限等待，保存本任务checkpoint与阻塞原因；有完整只读缓存的其他小头、CPU评测和发布继续。一次相同外部错误不盲重试，依赖恢复或实际修补后才重跑受影响任务。

---

## 13. 实际任务顺序与需实现CLI

1. `prepare`：固定代码/权重/本地数据/预算、读取绑定、创建SHORT_POPULATION，确认单次soft导出等价。
2. `export`：共享生成TRAIN/CAL/SEL的H1/H2只读证据与训练专用标签，建立完整B0。
3. `train`：六臂从同父独立训练，最多两作业，不串行依赖另一个head训练成功。
4. `evaluate-cal`：六臂所有规定点完整评价，写每臂CAL_LOCK。
5. `screen`：六臂各自固定点完整SEL，生成多模块初筛表。
6. `replicate`：符合条件的独立模块及必要控制，按固定点seed46复核。
7. `profile`：每个单臂选中点的真实成本及本地包重载。
8. `report`、`publish`：完整或partial都生成真实交接并实际推送。

需要新增一个小入口（下面是待实现CLI合同，不是当前现成命令）：
```bash
export RESCENE_SHORT_MODULE_ROOT="${RESCENE_SHORT_MODULE_ROOT:-$HOME/persist4d_runs/short_module_screen_v1}"
conda run -n persist4d python -m scripts.short_module_screen run \
  --config configs/short_module_screen_v1.yaml \
  --root "$RESCENE_SHORT_MODULE_ROOT" --resume
conda run -n persist4d python -m scripts.short_module_screen status \
  --root "$RESCENE_SHORT_MODULE_ROOT"
```

实现`run/status/report/publish`足够，阶段名写日志与状态即可，不必为每阶段建一套通用DSL。`run`范围固定为以上六臂，配置传入组合时拒绝。每次真实命令、时间、退出状态、head/source/input身份、更新范围与费用记入JSONL；partial任务恢复不能清零已花成本。

---

## 14. 必须交付的内容及GitHub同步

轻量结果至少包括：
```
artifacts/short_module_screen_v1/
  EXECUTION_INSTRUCTION.md
  CODE_BINDINGS.md
  RUN_CONFIG.json
  SHORT_POPULATION.json
  INPUT_MANIFEST.json
  BASELINE.json
  EXECUTION_LOG.jsonl
  BUDGET_LEDGER.jsonl
  training/<arm>/resolved_config.json
  training/<arm>/metrics.csv
  training/<arm>/checkpoint_manifest.json
  evaluation/CAL_ALL_CHECKPOINTS.csv
  evaluation/SEL_FIXED_CHECKPOINTS.csv
  evaluation/BY_REFERENCE.csv
  evaluation/PAIRED_MECHANISM_DELTAS.csv
  diagnostics/CANDIDATE_FAILURES.csv
  diagnostics/GEOMETRY_TRANSITIONS.csv
  selection/CAL_LOCK.json
  MODULE_SCREENING.csv
  SHORTLIST.json
  resources/PROFILE.csv
  FINAL_REPORT.md
  HANDOFF.md
  ARTIFACT_MANIFEST.json
```

缺数据时结果行status=NOT_RUN/BLOCKED，AP=null且保留实际应测分母；不先写PASS空表。每臂保存全部已评点，不只正结果。分组table必须同时包含B0和各自机制控制。

主结果表列：arm、slot、actual_updates、selected_step、seed、CAL/SEL T2、delta_B0、T1 AP/delta、机制对照delta、正收益reference数、几何变化/破坏、latency/VRAM、evidence_level。几何-only、已训练但负结果、未训练、预算跳过、外部阻塞分开。

**GitHub实际执行：**
- early检查现有Git与Release权限，不打印token；不要求用户粘贴密钥，不擅自找其它应用凭据。
- 代码、配置、曲线、负结果、HANDOFF和小模型参数入本轮独立分支；只stage本轮明确路径，不全工作树add/reset/clean。
- 新增小头通常很小，实际≤20MiB且允许分发时直接入Git，带README/模式/父SHA；**Release无权限不能让这些可发布的小头只剩本地路径**。
- 原R1/Concerto、数据和GT不重分发。较大可复算预测包按现有授权放Release，单asset≤1GiB，不能用Git切碎大权重绕过限制。
- 提交实验代码和结果A，再生成HANDOFF/manifest为B；manifest不hash自己，不把B写进B自身。推送分支并检查remote SHA；使用新tag `rescene-short-module-screen-v1`，冲突时最小-r2，不移动旧tag。
- 有权限创建draft prerelease，上传全部必需资产并检查bytes/digest或必要下载回验后发布。无权限仍完成Git并列缺失资产、补发命令。
- 最终发布回执写运行根外部，记录A/B/tag、远端引用和资产可访问性；可选上传draft期回执不造成自引用。状态VERIFIED要求本轮必需资产实际可得；否则CODE_ONLY/BLOCKED并明示。

本轮不改默认模型为任何新单臂，更不创建融合模型。最终答复必须给分支/B/tag、报告/交接/可取小头链接，多候选名单和真实增益；不得只报测试数或“全部完成”。

---

## 15. 完成前的一次审核

审核真实结果而不是审核漂亮叙事：
- 是否六臂均只有一个runtime module，父R1 SHA相同？
- M2是否fresh训练而不是加载M1？是否未运行任何Q+M？
- Q1/Q2同输入同结构；Q3参数差、标签语义、严格阈值是否说明？
- M0/M1只有未匹配监督不同，M2只有阶段聚合不同？
- 六臂CAL是否完整，SEL是否各自锁定点，T1是否真实单扫描？
- 所有“正收益”是否相对B0，不是只比退化控制高？是否另列机制控制？
- SHORTLIST是否允许多个或空集合，同插槽是否标互斥？
- 父前向/成本是否实测并计入，发布是否实际完成？

允许最终 `COMPLETE / NO_POSITIVE_MODULES`。本轮没有改善不能升级成“全部论文方向无效”；同样，只有单seed或开发集正结果不能称“已超过官方ReScene”。最终实验状态与外部发布状态分别记录。

---

## 16. 源码与论文索引

E01–E05均以固定仓库提交为准；E06是这次读到的作者metric接口参考，服务器运行时仍锁定原已安装版本，不自动升级。

- [E01] `models/rescene.py`：https://github.com/Orangekostar/Persist4D/blob/1ddab2aee88e5f5d8aa50b73a7896341802b2a26/models/rescene.py
- [E02] `scripts/rescene_task_postprocess.py`：https://github.com/Orangekostar/Persist4D/blob/1ddab2aee88e5f5d8aa50b73a7896341802b2a26/scripts/rescene_task_postprocess.py
- [E03] `scripts/p6a_metrics.py`：https://github.com/Orangekostar/Persist4D/blob/1ddab2aee88e5f5d8aa50b73a7896341802b2a26/scripts/p6a_metrics.py
- [E04] V2解释与限制：https://github.com/Orangekostar/Persist4D/blob/1ddab2aee88e5f5d8aa50b73a7896341802b2a26/artifacts/perception_gain_v2/validation/FINAL_INTERPRETATION.md
- [E05] 原生LOCAL接口：https://github.com/Orangekostar/Persist4D/blob/1ddab2aee88e5f5d8aa50b73a7896341802b2a26/scripts/perception_gain_local_evaluation.py
- [E06] 官方evaluator，本次读到blob `6c558fad20ecc033c1763864732eb98f2902108e`：https://github.com/GradientSpaces/stmetrics/blob/main/stmetrics/instances/evaluator.py
- [P01] Mask Scoring R-CNN，CVPR2019：https://arxiv.org/html/1903.00241v1
- [P02] Generalized Focal Loss，NeurIPS2020：https://proceedings.neurips.cc/paper/2020/hash/f0bda020d2470f2e74990a07a607ebd9-Abstract.html
- [P03] SoftGroup，CVPR2022：https://arxiv.org/html/2203.01509v1
- [P04] QueryInst，ICCV2021：https://arxiv.org/html/2105.01928v3
- [P05] SyncVIS，NeurIPS2024：https://arxiv.org/html/2412.00882v1
- [P06] 其余24篇的完整题名、会议信息和迁移边界见随包上一份文献路线；本指令没有声称再次完整复现/阅读24个作者仓库。

**本文件及随包YAML是执行设计，不是新实验成绩。没有新增服务器训练、单模块胜出或Git发布在本文生成时被预先确认。**
