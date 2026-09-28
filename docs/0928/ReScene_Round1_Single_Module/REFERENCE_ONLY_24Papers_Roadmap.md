# ReScene T1–T2：24篇顶会论文与分阶段实验路线

日期：2026-09-28。性质：文献研究和实验设计，未执行新增服务器训练；不是已验证涨点报告，也不是已经实现的CLI。

## 0. 核心结论

优先检验两条机制：**temporal质量评分**与**共享身份＋阶段局部mask解码**。首轮6个新增训练臂，先只训练附加头、冻结R1，避免重复最近的主模型续训退化；后续最多选择1个评分扩展和1个mask扩展，再进行2×2消融。

文献计数为24篇不同的主会论文，覆盖ECCV/CVPR/ICCV/NeurIPS/ICLR，2018–2026。不是全部2026论文；不将未确认接收的预印本、期刊扩展或ReScene任务基线充数。对所有条目核对核心机制与发表信息；重点方法使用原文方法段，其余按原文摘要及作者资料提取可迁移机制。并未逐个复现24个作者仓库。IKNE的部分CVF访问返回403，机制以作者机构摘要与官方代码说明交叉确认；其README仍提示完整模型版本待发布，不把它当现成完整可运行基线。

## 1. 本地已核实起点与证据

仓库：Orangekostar/Persist4D；本轮源码依据固定提交 `1ddab2aee88e5f5d8aa50b73a7896341802b2a26`，不自动跟随远端分支移动。

| 事实 | 依据 | 对新实验的约束 |
|---|---|---|
| R1查询解码循环沿用同一份mask特征，主要更新查询；mask由特征与实例embedding点积产生 | [E01] | M2/M3应真正改变局部mask生成，而非再次给查询换名 |
| 原生分数是类别概率×已预测前景内部平均置信度 | [E02] | 质量头首先只改变最终分数，固定候选、类别和mask |
| V2修复训练5220候选，4234空目标、986匹配、1256歧义候选被排除 | [E03] | 81.11%空目标是本地修复数据现象，不是ReScene全部训练集比例 |
| V2训练候选IoU>.5修复成功10、破坏63；NEW/PAIR均未成为主系统 | [E04][E05] | 首先隔离监督分工；segment错误下降不是成功判据 |
| V2已做阶段损失、初始attention、Q-SEM及12个关联网格 | [E05] | 不直接重复同一配置；不让这些旧门槛关闭新两条线 |
| 新任务主指标改为原生T2 t-mAP，T1为只输入单扫描的空间AP | [E06]与本次任务范围 | 双扫描条件下第一阶段AP不得冒充T1 |

本轮把R1作为共同父模型。390/450尾段续训可以另开小实验，但不要与本路线的机制比较同时改变父权重；父模型更换后所有缓存、对照和评分标签必须重建。

## 2. 文献矩阵

“迁移”列为本项目实验建议，不是论文已经在ReScene上验证的收益。

| ID | 论文 | 主会 | 原工作中的可借鉴机制 | 本项目迁移与限制 |
|---|---|---|---|---|
| P01 | [IoU-Net: Acquisition of Localization Confidence for Accurate Object Detection][P01] | ECCV 2018 | 定位质量不等于类别置信度；学习IoU并用于输出选择。 | Q1–Q3；只迁移质量建模，不把框优化直接用于离散mask。 |
| P02 | [Mask Scoring R-CNN][P02] | CVPR 2019 | 用实例特征和预测mask估计mask IoU，校准实例分数。 | Q1的主要基线；原方法不是temporal IoU预测。 |
| P03 | [Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection][P03] | NeurIPS 2020 | 类别与定位质量联合表示，连续质量标签。 | Q2/Q3的监督设计参考；不直接移植bbox分布回归。 |
| P04 | [Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection][P04] | CVPR 2021 | 根据定位分布的统计信息估计质量。 | Q4；mask熵/分位数是否有用需要重新验证，不能把尖峰必然等于正确当定理。 |
| P05 | [VarifocalNet: An IoU-aware Dense Object Detector][P05] | CVPR 2021 | 学习IoU-aware分类分数，并重新权衡正负样本。 | Q3/Q5；类别有效性与几何质量合并时避免重复乘类别概率。 |
| P06 | [Rank & Sort Loss for Object Detection and Instance Segmentation][P06] | ICCV 2021 | 分开处理正负排序和正样本内部质量排序。 | Q5；用于检验低回归误差是否真正带来正确排序。 |
| P07 | [TOOD: Task-aligned One-stage Object Detection][P07] | ICCV 2021 | 分类与定位特征及目标分配对齐。 | 后续联合优化参考；首轮不修改ReScene matcher。 |
| P08 | [Rank-DETR for High Quality Object Detection][P08] | NeurIPS 2023 | 将排序与高定位质量目标对齐。 | Q5参考；不要误记为CVPR 2024，也不整套替换DETR训练。 |
| P09 | [PointRend: Image Segmentation as Rendering][P09] | CVPR 2020 | 在不确定位置自适应预测，融合细粒度和粗分割信息。 | M4；把二维采样迁移成真实3D局部点采样。 |
| P10 | [Conditional Convolutions for Instance Segmentation (CondInst)][P10] | ECCV 2020 | 实例条件生成轻量mask预测器。 | M0–M2；使用查询条件残差，不复制二维卷积栅格。 |
| P11 | [RefineMask: Towards High-Quality Instance Segmentation with Fine-Grained Features][P11] | CVPR 2021 | 逐级融合细节以改善边界。 | M4；若最后仍强制同一超点同值，细节分支可能被抹掉。 |
| P12 | [Instances as Queries (QueryInst)][P12] | ICCV 2021 | 查询生成动态mask头，实现实例条件的信息交互。 | M0–M2；不是仅增加query-to-query注意力。 |
| P13 | [Mask Transfiner for High-Quality Instance Segmentation][P13] | CVPR 2022 | 稀疏错误易发区域与多尺度细节精修。 | M4；用边界/不确定点，而非全场景高分辨率计算。 |
| P14 | [SoftGroup for 3D Instance Segmentation on Point Clouds][P14] | CVPR 2022 | 软分组及top-down refinement；分类含负样本，分割和IoU头只在正样本上训练。 | M1监督分工；其50%正例阈值不直接强加到ReScene。 |
| P15 | [Mask DINO: Towards A Unified Transformer-based Framework for Object Detection and Segmentation][P15] | CVPR 2023 | 共享查询，结合检测、分割和去噪训练。 | 保留统一实例监督；不把Mask DINO误称为迭代更新mask特征的直接证据。 |
| P16 | [OneFormer3D: One Transformer for Unified Point Cloud Segmentation][P16] | CVPR 2024 | 统一3D语义与实例分割的Transformer设计。 | 3D实现与辅助语义监督参考；本轮不更换整个骨干。 |
| P17 | [Insightful Instance Features for 3D Instance Segmentation (IKNE)][P17] | CVPR 2025 | 聚合同实例候选的互补信息，并通过结构与降噪线索增强实例表示。 | 候选互补诊断与M3邻近参考；不是其完整方法复现。 |
| P18 | [CompetitorFormer: Mitigating Query Conflicts for 3D Instance Segmentation via Competitive Strategy][P18] | CVPR 2026 | 处理多个查询竞争相同实例的冲突。 | 重复候选诊断；已有查询竞争实验不能换名字重做。 |
| P19 | [Video Mask Transfiner for High-Quality Video Instance Segmentation][P19] | ECCV 2022 | 实例查询引导跨时空稀疏区域精修。 | M2/M4；视频中的3D是x-y-time，不等于物理3D点云。 |
| P20 | [DVIS: Decoupled Video Instance Segmentation Framework][P20] | ICCV 2023 | 分开建模分割、追踪与时序精修。 | 实验分工与冻结父模型；不搬入完整视频tracker。 |
| P21 | [A Generalized Framework for Video Instance Segmentation (GenVIS)][P21] | CVPR 2023 | 缩小训练与推理的时序输入/目标分配差异。 | 训练使用真实预测而非完美GT条件；不宣称此前TALA路线未研究。 |
| P22 | [CTVIS: Consistent Training for Online Video Instance Segmentation][P22] | ICCV 2023 | 用与推理一致的历史表示构造训练对比项。 | 真实预测误差与鲁棒性训练参考；不新增长期记忆主线。 |
| P23 | [SyncVIS: Synchronized Video Instance Segmentation][P23] | NeurIPS 2024 | 同步共享视频级与帧级实例表示。 | M2最直接参考：共享身份与阶段局部mask embedding分离。 |
| P24 | [SAM 2: Segment Anything in Images and Videos][P24] | ICLR 2025 | 双向更新提示与图像特征，分开预测mask质量及对象存在性。 | M3与质量/存在性分工；不用GT点击或GT mask作为推理提示。 |


## 3. 先做一次候选失败分解，不新增网络

### D1：候选完整性与双扫描共同正确性

对类别相容、两阶段均存在且没有身份歧义的有效GT g：

- `U_joint(g) = max_i min_t IoU(M_i,t, G_g,t)`。
- `U_separate(g) = min_t max_i IoU(M_i,t, G_g,t)`。

两者都低：单阶段几何/候选覆盖不足；separate高而joint低：两侧好预测没有落在同一个query/轨迹里；joint高而AP低：进一步分析评分、类别、重复及官方匹配。另对新增/消失实例按官方存在性规则单列，不把简式用于所有案例。

这些是候选诊断，不是可实现系统或AP上界：不同GT可能争用同一个候选，独立逐阶段选择也不是合法轨迹。报告每个阈值的计数、各reference、类别和尺寸桶，不以平均值替代全部分布。

### D2：重复/不匹配与分区表达能力

对未获一对一匹配的训练候选，区分无相容GT、低重叠、与GT高重叠但属于重复候选、未知/ignore。不要预先把它们都判为完全背景。

在可行范围内，评估允许组合整块超点能达到的最好mask IoU，以及不同阶段的边界混合程度。只有求解或有效证书支持“最佳”时才叫上界；若只用了多数标签分配，写成可达到的诊断值，不冒充严格上界。实际输出还包含低分辨率分区到full-res多数决策，诊断须覆盖两种分区和实际物化路径。

只用新同源native T2预测，按manifest一次导出原始q、各阶段特征、logits、原类别/分数、候选谱系、输入/运行身份。TRAIN的GT单独存储；CAL用于诊断，SEL不反复产生新配置；确认集不提供开发条件。

## 4. 全部实验的共同控制

- B0：冻结原R1，native双扫描输入和官方后处理；无D0/lag1额外第一扫描前向、无X3未来信息、无新记忆。
- 评分臂Q1–Q5：不改任何mask/类别/候选集合，保持原top-k、过滤结果，只在最终输出处替换分数。早期top-k重排和NMS变化是不同实验，首轮不做。
- 几何臂M0–M4：初期只改mask，冻结原类头和分数；固定query/候选谱系。若mask变空可如实输出空列，不通过候选过滤制造收益。
- R1始终eval且不回传梯度，新增分支独立train；全局q本来已包含双扫描信息，因此“局部控制”不等于完全无时序信息。
- 几何监督对一条query使用同一GT实体在两阶段的标签。即使采用多对一proposal监督，也不得让同一query在两个阶段选择不同GT。
- 不同物理扫描不能按vertex ID或最近世界坐标当作同点；vertex ID严格对应只适用于同一物理扫描的不同处理副本。
- 改变mask后，质量标签必须重算。评分器看到新mask特征的分布也变化，不直接把旧头输出当新mask质量。
- GT只生成训练标签与评价诊断，不进入质量头输入、推理prompt、阶段选择或修复门控。

## 5. 评分系列：Q1–Q5

### 公共输入和目标

使用父query特征q、各阶段mask内/邻域的池化特征h1/h2、原类别概率及支持信息。首轮不改变特征来源或隐藏宽度；比较目标定义时不同时扩网络。

在TRAIN上，选择同一个类别相容GT轨迹g*，计算官方规则下的temporal overlap；选择不能依赖模型当前分数或官方分数排序后的TP/FP结果，防止循环监督。无相容有效GT时质量为0；ignore/无法判定样本不硬置0。每个候选均应得到该规则下的标签，重复候选可能有高几何质量，单列其影响。

Q1/Q2/Q3包含“类别是否相容”的失败标签，因此第一版直接将头输出作为综合质量分数，原类别概率可以作为输入；不再乘原分数而重复惩罚类别。若后续使用纯条件几何IoU头×类别概率，必须另立配方和对应监督范围，不混合两种定义。

| 实验 | 唯一主要变化 | 参照 | 预期回答而非保证 |
|---|---|---|---|
| Q1 CONCAT-QUALITY | 预测两扫描拼接mask的IoU；使用同一g*和同一输入 | P01/P02 | 普通mask质量估计是否比原自信分数更有用？ |
| Q2 TEMPORAL-QUALITY | 网络与Q1相同，监督改为官方temporal overlap标量 | P02/P03/P05的迁移 | 不把好阶段平均掉，是否改善T2排序？ |
| Q3 THRESHOLD-QUALITY | 预测各官方阈值下“同一候选对应的整条轨迹通过”的事件 | P03/P05/P08的迁移 | 平均IoU回归是否忽略了过阈值边界？ |
| Q4 UNCERTAINTY-AUGMENTED | 只在最佳基础头上加入mask熵、概率分位数、边界不确定、两阶段差异；同容量输入控制 | P04 | 预测分布统计能否提供额外质量信息？ |
| Q5 RANK-QUALITY | 最佳头上加入/替换为质量排序目标，保持输入和容量 | P06/P07/P08 | 质量MAE降低是否真正转化为AP排序改善？ |

Q3的标签为 `y_i,k = 1[u_i > tau_k]`；tau数组从当前已锁定官方evaluator读取。输出按tau递增应单调不增，可用累计非负logit间隔参数化；分数是各阈值输出的平均值。该平均值是可比较阈值成功的排序代理，不是AP本身，也没有保证对应官方一对一匹配精度。

存在性是关键边界：正确的消失对象不应因“双阶段都必须有前景”而被判0。训练标签由官方存在性规则得到；推理只使用自身预测与已观测输入，不能读取GT有效阶段mask。

记录quality MAE、阈值Brier/reliability、同类排序错位及正式t-mAP。几何候选覆盖和每个候选的几何tIoU必须不变；官方REC可能因评分影响匹配次序而改变，不能据此宣称几何召回提高。若做正负再采样，保存抽样概率并采用适当权重，或明确只将输出用作排序分数，不虚称校准概率。

首轮Q1/Q2/Q3；Q4和Q5最多取一项进入第二轮。没有证据说明所有质量差都由排序造成，几何候选缺失时评分臂应降低优先级。

## 6. 阶段局部mask系列：M0–M4

### M0：共享残差头＋旧式空目标监督（新路径中的公平控制）

在native双扫描R1最终mask head之后，加一个零初始化的轻量共享实例残差头。输入是q与两阶段合并的实例特征，两个阶段使用同一修正embedding。可使用查询生成的小型动态层，不启动整套CondInst/QueryInst。

M0保留V2式“一对一未匹配候选采用空目标”的监督作为显式控制；不是直接拿旧lag1修复器数字比较。该控制的目的在于将监督变化和网络/输入变化分开。不要再次扩大这种配方搜索。

### M1：同结构，只把候选抑制与形状监督分离

网络、初始化、候选流和训练更新与M0相同。一个query若未分到一对一GT但与相容GT有可靠几何重叠，可在形状分支按唯一的最佳GT赋多对一proposal目标；这一额外assignment只用于新增形状头，不改变R1 matcher和身份输出。正例阈值及歧义规则在TRAIN预先固定，不根据CAL尝试多档。

真正无可靠实例目标的候选不再强制整张mask清零；给低权重的零残差保持约束，存在性/类别质量由评分分支学习。匹配到实体但其中一个阶段确实缺失，该阶段仍须监督为空；对象内部和周围背景仍是负像素。未知标注不当负像素。

M1-M0检验监督分工，不能直接解释为历史融合收益。可以用P14作为方法动机，但并非原样复现SoftGroup：它原论文采用IoU>0.5正例，当前任务应按固定训练配方决定可靠阈值。

### M2：共享身份＋阶段局部mask embedding（首选方法主线）

在M1监督下，仅把合并的实例特征改为分阶段特征：

`h_i,t = Pool/Attention(F_t, q_i, parent_mask_i,t)`

`e_i,t = e_i + A(q_i, h_i,t)`

`z_i,t(s) = F_t(s)^T e_i,t`

A的末层零初始化。M1的同容量控制为 `e_i' = e_i + A(q_i, Pool_all(F))`；两者输入维度、参数数量、候选和训练预算尽量一致。

同一个q和同一轨迹列负责身份；e_i,1/e_i,2负责各扫描的形状。只在最后一级加入一次修正，避免同时改全部decoder层。局部信息优先来自预测mask与固定扩展邻域，并保留少量探索点，防止漏分区域完全无法进入修复。探索范围由几何规则固定，不用GT。

该设计受SyncVIS、CondInst、QueryInst启发，不声称完整移植其框架。阶段查询完全独立重新检出再配GT，会改变问题，不能冒充共享身份的改进。

首要观察：同一GT的joint通过率是否提高、较差阶段IoU是否提高、原正确轨迹被破坏多少；T2官方mAP与T1空间AP都报告。

### M3：查询—mask特征双向更新（条件扩展）

当现有表示能表达GT、但M2仍受局部特征混淆限制时，在最终层增加一次轻量反馈：

`F'_t = F_t + W_out * CrossAttention(F_t, Q)`

`W_out`零初始化；将计算限制到候选支持/局部邻域，避免常驻Npoints×Nqueries×D中间张量。需有参数量相近的逐点MLP控制，以区分多参数与实例反馈。

直接结构参考是SAM/SAM2式两向更新；IKNE支持候选互补/结构提示这一邻近方向，但其IKA/ISG不是该公式本身，不能误引。禁止GT提示；R1的预测q即实例条件。多个query可能覆盖相同点，必须通过规范化聚合而非任意顺序覆盖。

### M4：稀疏点级边界修复（条件扩展）

只在分区表达诊断表明整块超点不足时启动。参考PointRend/RefineMask/Mask Transfiner/VMT，在实例边界和不确定区域读取聚合前点特征、局部几何和q，预测稀疏点残差。

固定每实例点预算、采样规则与random-point同数量控制。测试中不按GT挑边界。

**关键接口：若精修后仍强制经过原full-res超点多数决策，新点级细节可能再次被压平。** 新路径必须在明确记录的位置输出精修点mask，未选点保留原输出；建立真实zero/no-op等价测试，不用检测delta=0直接返回旧mask掩盖路径变化。一次前向内不增加未来扫描。

M3/M4依据D1/D2最多选一个。其正结果代表特征/表示路径有效，不意味着更长时序已经解决。

## 7. 首轮与后续排程

### 首轮：一个冻结基线＋六个小训练臂

| 系列 | 实验 | 主比较 |
|---|---|---|
| 不训练 | B0：R1 native | 固定起点 |
| 评分 | Q1、Q2、Q3 | Q2-Q1检验temporal目标，Q3-Q2检验阈值事件建模 |
| mask | M0、M1、M2 | M1-M0检验监督分工，M2-M1检验阶段局部表示 |

建议小头统一先做1500 optimizer updates，保存0/500/1000/1500；同系列固定seed45、同候选清单及抽样流。该步数是试验预算起点，不是收敛证明。评分臂和冻结R1后置mask头尽可能共享一次父特征导出；缓存超限按sequence流式，不重复前向所有大模型。

这轮不同时做390/450续训、LoRA、Q增加、分辨率提高和新association。若后来确需解冻R1最后decoder，应另设相同预算的原始C0控制，清楚区别“冻结头试验”和“联合训练”。

### 第二轮：只扩展两个最有希望的机制

评分扩展最多Q4或Q5一个；mask扩展最多M3或M4一个。保留对应父臂。无需将24论文各自完整训练，也不做所有组合的笛卡尔积。

### 最后：2×2验证真正的互补性

| | 原分数 | 与对应mask一起重新生成标签/校准的质量头 |
|---|---|---|
| 原mask | B0 | 最佳Q* |
| 最佳M*的mask | M*，原分数不变 | M*+Q*重新适配 |

不能沿用旧mask的quality标签训练组合头。若质量头只是记住旧父模型的错误模式，组合收益可能消失，需如实报告。两个单组件有收益不保证组合相加。

## 8. 数据、判定与成本规则

TRAIN拟合；CAL选择既定点/配方；SEL验证开发迁移；固定后才运行正式LOCAL-T2及PB-T2/ADDITIONAL-T2。历史CAL/SEL/PB和LOCAL暴露如实标明，不重新命名untouched。同一physical reference所有扫描都必须归于同一角色；评测seed不是训练seed。

主指标为同人口官方pooled T2 t-mAP，T1是真实单扫描空间mAP。另报stage1|T2、stage2|T2、joint候选覆盖、原正确轨迹破坏数、类别与重复情况。各种指标不能混成一个“准确率”。

建议推进目标：SEL T2增加至少0.5pp，T1不下降超过0.2pp，并在逐reference表上无少数场景完全主导；这些是新研究目标而非论文保证。两训练种子固定配方/固定step复核；不在seed46再次选最优epoch。用同reference成对差值或合法reference-block pooled bootstrap；缺实现就不虚报统计显著。

T2低于目标但稳定小增益的候选保留为MODEST_GAIN；技术跑通而精度无益是NEGATIVE，不包装主模型。仅评分提高不声称几何变好；仅mask IoU提高而正式AP不升则进一步检查质量排序，不能按辅助指标强行部署。

预算启动时核对V1/V2既有成本。该方案不授权新增192 GPUh；优先利用现有权重和缓存，以实测吞吐给当前可用余额安排限额。若沿用原192累计上限，先扣去已记78.94及其后真实费用。训练/原生前向/CPU评测/部署时延分开记账；共享父前向不能当部署免费计算。保留完整确认与本地模型重载资源。

测试只保护真实会改变成绩的边界：模式关闭/zero-init等价、跨阶段同一GT、ignore/消失处理、评分分支mask完全不变、坐标/分区映射、缓存源绑定与不读未来。复用现有测试，结束一次相关回归；不重复全库安全/schema/fuzz审计。

## 9. 代码—实验绑定

以下“新增动作”均为待实现建议，不是声称仓库已有类或CLI。

| 现有文件/符号 | 新实验的接入建议 |
|---|---|
| models/rescene.py::forward/aggregate_features/mask_module | M0–M3末级共享/阶段embedding残差；M3新增特征反馈返回值，不能只改变q却报告F被更新 |
| trainer/trainer.py::_get_mask_and_scores | 保留原分数作B0；Q臂另在最终保留候选上输出新分数，不全局改旧函数影响历史方法 |
| scripts/rescene_task_postprocess.py::extract_official_task_prediction | 导出candidate/query/class谱系、各阶段soft信息；Q/M输出分离 |
| scripts/rescene_task_postprocess.py::materialize_segment_logits | M0–M3沿用真实物化；M4单列点级绕开旧多数决策的明确接口 |
| scripts/train_perception_refiner.py::build_refiner_training_records | 参考其数据绑定，不直接沿用所有unmatched空目标；新增仅TRAIN几何目标构造 |
| models/perception_gain.py::derive_segment_stage_ids | 复用严格阶段对应；不同扫描不强行vertex对齐 |
| scripts/p6a_metrics.py::OfficialMetricAccumulator/official_temporal_match_trace | 指标/存在性/ignore规则；质量监督不能从当前评分的greedy TP结果反推 |
| scripts/perception_gain_native_evaluation.py与local_evaluation.py | 原生T2确认人口；补真正单扫描T1入口并测试输入中没有第二扫描 |

## 10. 应交付的最小证据

生成文献→实验绑定、实际数据清单与源hash、每臂resolved config/训练曲线/所有规定检查点、D1/D2诊断表、Q/M分组对照、2×2表、逐reference结果、固定种子复核、真实时延及HANDOFF。代码和轻量结果同步到独立GitHub分支；模型/预测大包按已有权限发布，不能用空Release声称完整上传。当前文件仅为研究方案，没有执行这些写操作。

## 11. 结论

首选Q2/Q3验证“最弱阶段质量”的排序价值；首选M2验证“共享身份但不强迫相同形状embedding”的几何价值。M1不可省，否则新mask结构的结果可能只是修正了错误监督。点级精修与双向特征反馈按诊断投入，不同时堆叠。

重要的论文叙事候选是：**将跨扫描身份一致性、阶段局部分割以及轨迹质量估计分别建模，再通过受控实验确认它们是否互补。** 这个叙事及新公式是我们的待验证设计，不是24篇论文已替ReScene证明的涨点结论。

## 文献与本地证据链接

[P01]: https://arxiv.org/abs/1807.11590
[P02]: https://arxiv.org/abs/1903.00241
[P03]: https://proceedings.neurips.cc/paper/2020/hash/f0bda020d2470f2e74990a07a607ebd9-Abstract.html
[P04]: https://openaccess.thecvf.com/content/CVPR2021/html/Li_Generalized_Focal_Loss_V2_Learning_Reliable_Localization_Quality_Estimation_for_CVPR_2021_paper.html
[P05]: https://arxiv.org/abs/2008.13367
[P06]: https://arxiv.org/abs/2107.11669
[P07]: https://arxiv.org/abs/2108.07755
[P08]: https://arxiv.org/abs/2310.08854
[P09]: https://arxiv.org/abs/1912.08193
[P10]: https://arxiv.org/abs/2003.05664
[P11]: https://arxiv.org/abs/2104.08569
[P12]: https://arxiv.org/abs/2105.01928
[P13]: https://arxiv.org/abs/2111.13673
[P14]: https://openaccess.thecvf.com/content/CVPR2022/html/Vu_SoftGroup_for_3D_Instance_Segmentation_on_Point_Clouds_CVPR_2022_paper.html
[P15]: https://arxiv.org/abs/2206.02777
[P16]: https://arxiv.org/abs/2311.14405
[P17]: https://openaccess.thecvf.com/content/CVPR2025/html/Roh_Insightful_Instance_Features_for_3D_Instance_Segmentation_CVPR_2025_paper.html
[P18]: https://openaccess.thecvf.com/content/CVPR2026/html/Wang_CompetitorFormer_Mitigating_Query_Conflicts_for_3D_Instance_Segmentation_via_Competitive_CVPR_2026_paper.html
[P19]: https://arxiv.org/abs/2207.14012
[P20]: https://arxiv.org/abs/2306.03413
[P21]: https://arxiv.org/abs/2211.08834
[P22]: https://arxiv.org/abs/2307.12616
[P23]: https://proceedings.neurips.cc/paper_files/paper/2024/hash/58acaf7b9aecc9604c7a6aac2eb81035-Abstract-Conference.html
[P24]: https://proceedings.iclr.cc/paper_files/paper/2025/hash/45c1f6a8cbf2da59ebf2c802b4f742cd-Abstract-Conference.html
[E01]: https://github.com/Orangekostar/Persist4D/blob/1ddab2aee88e5f5d8aa50b73a7896341802b2a26/models/rescene.py
[E02]: https://github.com/GradientSpaces/rescene4d/blob/fb2fe42eb8f1e926567c48eea9acb874e608ee10/trainer/trainer.py
[E03]: https://github.com/Orangekostar/Persist4D/blob/1ddab2aee88e5f5d8aa50b73a7896341802b2a26/artifacts/perception_gain_v2/refinement/PAIR_COVERAGE.csv
[E04]: https://github.com/Orangekostar/Persist4D/blob/1ddab2aee88e5f5d8aa50b73a7896341802b2a26/artifacts/perception_gain_v2/refiner/R1/POST_TRAIN_DIAGNOSTICS.json
[E05]: https://github.com/Orangekostar/Persist4D/blob/1ddab2aee88e5f5d8aa50b73a7896341802b2a26/artifacts/perception_gain_v2/validation/FINAL_INTERPRETATION.md
[E06]: https://arxiv.org/html/2601.11508v2

重点方法段补充：

- [SoftGroup 正负候选监督](https://arxiv.org/html/2203.01509v1)
- [SyncVIS 阶段/全局表示](https://arxiv.org/html/2412.00882v1)
- [SAM2 双向mask解码、存在性及IoU头](https://arxiv.org/html/2408.00714v2)
- [QueryInst 动态mask头](https://arxiv.org/html/2105.01928v3)
- [VMT 实例引导稀疏精修](https://arxiv.org/html/2207.14012v1)
- [Mask Scoring R-CNN 质量头](https://arxiv.org/html/1903.00241v1)
- [GFLv2 分布统计](https://arxiv.org/html/2011.12885v1)
- [IKNE 作者机构摘要](https://pure.kaist.ac.kr/en/publications/insightful-instance-features-for-3d-instance-segmentation/)
- [IKNE 官方仓库及完整版本限制](https://github.com/kuai-lab/cvpr25_IKNE)
