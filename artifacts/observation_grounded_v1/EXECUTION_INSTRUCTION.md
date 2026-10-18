# Persist4D OGR-V1：Codex 最终执行指令

版本：2026-10-18。任务：开发、训练、选模、复验、评测、交接及 Git/LFS 发布。

本文件是本轮新研究的执行指令，不是已完成实验报告。它将《SUPPLEMENTARY_EXPERIMENT_DESIGN.md》中的待定实现收敛为一条确定路线。新配置中的数值是本轮设计默认值或确定性选择规则，不冒充历史最优超参数。主文件可以单独交给 Codex；文末附带的配置是本文件的一部分。

## 0. 任务授权、完成范围与优先级

你是此任务的开发和实验主负责人。执行下述完整任务，不以“已经写好计划”“单元测试通过”“进程已启动”替代完成。无需再次询问是否开发、是否运行锁定实验、是否上传本任务产物。按确定规则解决可从仓库、当前环境和日志查明的字段。

唯一研究问题：**从当前预测掩码支持区域读取观测证据，并将它反馈给分割器，是否优于直接实体监督、事后身份读出及普通查询反馈？**

必须交付四个可运行训练配置 R-TARGET、R-ID、R-QFB、R-OBS；在可核实的剩余额度内完成统一长度训练、CAL 选模、R-OBS 与锁定强对照的第二适配种子、完整因果评测、机制解释和发布。即使结果为负，也交付代码、全部预设结果、真实权重和报告。不要自动改投其他研究路线。

执行优先级：用户当前任务 > 本文件的具体规则及附录配置 > 补充实验设计中未定的建议 > 旧 MPR 执行指令。旧实验结论、锁、账本和文件不可追溯修改。旧报告中的“不继续训练该候选”不阻止本轮新方法研究，但禁止把本轮称为旧 MPR 的补跑或改写旧 SCREEN_NO_GO。

不能取得资产、没有资源或关键实现无法正确运行时，记录准确的 BLOCKED 原因，继续不依赖该问题的开发、整理和发布。不要假造数据、权重、完整实验状态或远端成功；不要仅因缺外部确认集而停止本轮内部实验。不要以增加测试数量作为进度目标。

## 1. 固定起点与事实依据

- 仓库：`Orangekostar/Persist4D`。
- 本轮 BASE：`758b5aa4bd2e9119ae5e05f725faa360c2120d88`。
- 父训练源码：`6b1b779172afdeb51e2719a737cd05e7eeeca982`。
- 旧发布分支：`research/persist4d-mpr-v1-formal-20261014`。
- 新分支：`research/persist4d-ogr-v1-20261018`。
- 新标签：`persist4d-ogr-v1-20261018-v1`。
- 新结果目录：`artifacts/observation_grounded_v1/`，下称 OUT。
- 工作目录：从已存在的本项目 checkout 建独立 worktree；运行目录从已有 resolver 查得后建立该任务专属目录，不把私有绝对路径提交到 Git。

2026-10-18 的连接器读取确认上述旧分支 HEAD 为 BASE；不能据此声称所有其他分支都没有新工作。若执行时本地 HEAD 不同，记录差异，在 BASE 建新 worktree，不 reset 用户工作区。

必须先读下列已有文件的相关实现；证据索引中的“既有事实”只覆盖已读范围，不代表全仓逐行审计：

| ID | 既有路径／符号 | 已知事实及本轮绑定 |
|---|---|---|
| C01 | `artifacts/revisit_prompt_v1/REPORT.md`、`HANDOFF.md`、`RUN_LOCK.json` | MPR 已完成、SCREEN_NO_GO；核实真实父资产和累计预算。 |
| C02 | `models/rescene.py::forward,before_first_mask,after_decoder_stage,mask_module` | 首 mask 前钩子及每个 FFN 后钩子均已存在；本轮不复制整个 ReScene。 |
| C03 | `models/revisit_prompt.py::RevisitPromptReScene` | M4 的提示、选槽、坐标偏移、冻结主干、输出清理；复用但不要沿用旧变体白名单冒充新身份。 |
| C04 | `models/revisit_prompt_modules.py::segment_context,PromptAdapter,select_history_slots` | 完整当前超点检索、硬 anchor、最多 32 历史查询；复用 M4 提示前端。 |
| C05 | `models/criterion.py::SetCriterion.forward,get_loss,loss_labels,loss_masks` | 原 matcher 为集合分配，每层重新匹配；新建受限实体分配，不向原损失后面重复加一份相同掩码监督。 |
| C06 | `models/task_memory_criterion.py`、`models/task_memory_supervision.py` | 借用显式 assignment、训练标签隔离和歧义处理，不把 TALA 当新意，不强套旧 proposal route。 |
| C07 | `models/revisit_prompt_loss.py::TrainingLabelLedger,bind_training_births,evidence_labels` | birth 后可靠绑定；UNKNOWN 不作为负例；原辅助标签是 anchor 监督，不是最终实体掩码监督。 |
| C08 | `trainer/revisit_prompt_trainer.py`、`trainer/trainer.py::_configured_objective_loss` | 保留 raw_sum 的实际损失语义、逐阶段反向和优化器边界；新增前缀必须参与真实训练和保存。 |
| C09 | `datasets/revisit_prompt_episode.py` | UUID、train/validation 索引、共同因果前缀真值、局部 pair、坐标偏移已有；CONFIRM/full_prefix 需显式新适配。 |
| C10 | `models/task_memory_state.py`、`models/task_memory_routing.py::EntityRoute,commit_entities` | K100、无淘汰，route 绑定旧状态与元数据；新关联必须生成合法一对一路由，不能只改输出 ID 列。 |
| C11 | `models/revisit_prompt_state.py::RevisitGeometryState` | sidecar 只在已接受当前支持后更新；新观测原型沿用相同生命周期。 |
| C12 | `scripts/revisit_prompt_runtime.py` | 原支持概率同时影响关联与分数；新 R 配置禁止这样缩放，M4 原参照保持原定义。 |
| C13 | `scripts/rescene_task_postprocess.py`、`scripts/system_comparison_v2_inference.py` | 候选必须通过 source_query_ids 绑定身份；commit0、mean 分数，保留 ephemeral。 |
| C14 | `scripts/evaluate_revisit_prompt.py`、`scripts/system_comparison_metrics.py`、`scripts/p6a_metrics.py` | 原官方充分状态池化、当前目标裁切、身份指标复用；不另写替代 AP。 |
| C15 | `scripts/revisit_prompt_assets.py`、`artifacts/revisit_prompt_v1/MODEL_MANIFEST.json` | 父包、编码器、旧 delta 分开；新 bundle 是新契约，不删原身份校验。 |
| C16 | `scripts/select_revisit_prompt.py` | 旧 M0–M6、三重复、门槛均硬编码；新增选模入口。 |
| C17 | `models/revisit_prompt_{numerics,native_numerics,centroid_numerics,sparse_reference}.py`、`scripts/evaluate_revisit_prompt_sparse.py` | V4 包含稳定归约、Native 稀疏邻接、保序卷积；稀疏参考显式拒绝梯度。 |
| C18 | `scripts/revisit_prompt_sparse_policy.py` | 旧政策与 C3 源码绑定，不能把新任务写入旧命名空间；新建派生政策。 |
| C19 | `conf/model/rescene.yaml`、`conf/revisit_prompt/default.yaml` | 默认共享解码器、Q100、D128、无增强、stage truncation=1；实际解析父配置后核实。 |
| C20 | `scripts/native_long_budget.py`、`scripts/revisit_prompt_campaign.py` | 复用计费和进程锁思想，新任务避免旧任务路径和阶段常量污染。 |

历史 PB：M0 的 S_long/S_current 为 7.124311%/34.796831%，M4 为 7.106387%/34.513384%；M4 相对 M0 的当前恢复／丢失为 181/214。它们解释任务动机，不能填进新方法结果。三次进程重复不是三个适配种子。MPR 的充分状态包不能替代未保存的特征或检索 logits。

父包：`artifacts/native_long_postselection_v1/deployment/E1-23760.pt`；SHA256=`8c9d0d57044ce23c7964ddb1e7a2d8bd18e388e6ee657a7f4b02b1368b5b3a98`。
独立编码器 SHA256=`845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07`。taxonomy 从原 MODEL_ASSET_MANIFEST 的真实条目绑定。不要重新发布编码器、原始点云或逐点 GT。

## 2. P0：工作区、资产、数据与预算解析

### 2.1 工作区和引用

先记录 `git status --short`、`git rev-parse HEAD`、`git remote get-url origin`、旧分支/tag 实际引用。不要覆盖未提交修改或移动旧标签。已有同名新分支且任务标识和 BASE 相同则恢复；若同名属于其他任务，选择最小未使用后缀 `-r2/-r3/...` 并写入 START_STATE，后续统一使用该实际分支和对应标签。

读取本仓库 `assets.local.json` 及既有 run root，不凭空假定服务器路径、GPU 型号、CUDA 环境或新确认数据存在。资产只核验一次；未改变的文件复用已有有效回执及文件大小/mtime 缓存，发生改变再算哈希。LFS 指针不是模型实体。GitHub Release API 不可用不妨碍 Git/SSH 与 LFS 发布。

### 2.2 预算不得归零

从旧 RUN_LOCK、旧最终账本及可能存在的本轮 ledger 解析真实账务。固定历史参考为 1760.667754119 GPUh；其中旧 MPR 起点 1714.8140614156707、MPR 新耗用 45.853692703，不可重复加算。旧记录已包含此前约 0.931 GPUh 的前驱任务。

旧可核实 lifetime cap 为 1823 GPUh、预留 8 GPUh。本轮最多 50 GPUh：

`available_new_total = max(0, min(50, verified_lifetime_cap - latest_settled_prior_excluding_this_run - active_external_reservations - 8))`。

`available_remaining = max(0, available_new_total - this_run_settled - this_run_active_reservations)`。

恢复时 prior 明确排除本轮事件，本轮已消费与reservation各扣一次；不可在恢复时重新获得50GPUh。后文 available_new 指首次锁定的本轮总可用额，调度时必须检查 available_remaining。

执行时若存在更新授权或账单，以真实记录为准，但本文件不授权突破本轮 50 GPUh，也不授权新增付费云算力。结果不得回落到旧 1800 上限或把 1823+50 当新上限。

阶段上限：D_DIAGNOSTIC=2、T_FIRST=16、T_REPLICATION=12、E_EVALUATION=12、F_PROFILE_CONFIRM=4、I_INTEGRATION_REPAIR=4 GPUh，总和 50。探测、失败、重试、预热、启动均计入各自阶段；CPU 统计不虚构 GPU 消费。不能重复把预留记成已消费。

新 ledger 只写本轮事件，以旧累计值作为一次性 prior。每个 GPU 任务采用唯一 job_id、进程锁及共享的额度 reservation；结算用实际占用卡数×墙钟时间，恢复按已结算事件去重。预估上界和并发 reservations 共同受限，不能两个 worker 同时花同一余额。每任务 1 GPU，最多 2 个任务并行；先检查设备是否空闲，不中止其他用户进程。性能测量时该卡不并发本任务其他作业。

若可用额度不足统一 N=400 的完整核心包，完成代码、诊断和可用资产发布，标记 BUDGET_BLOCKED；不得只缩短控制组或漏掉负结果后声称完成四臂比较。

### 2.3 数据角色和可比较输入

从旧 DATA_MANIFEST 与 native POPULATION、FORMAL_POPULATION 建新只读输入清单；保持物理 UUID、ambiguities、原顺序和索引空间。已知 CAL 输入数 T1..5=31/19/14/14/6；SEL=39/21/21/15/15；PB T2..5各129，不存在 PB-T1。若执行端事实不符，解释源文件差异，不凑数。

CAL/SEL/PB 全部标记 HISTORICALLY_EXPOSED。TRAIN 与这些面板的 reference 交集按实际清单核验一次。加载阶段 j 的网络输入只能来自当前已允许窗口，标签共同前缀只能是0..j。官方 GT 投影留在训练／评价适配层，不进入 prompt、identity head、反馈或 commit。

新 CONFIRM 是显式新角色，dataset_scope 独立字段；禁止借用 PB 标签绕过索引契约。仅检查已挂载且已有授权的标签资产。候选要求：不在本轮适配 TRAIN、不在已知父任务 TRAIN、不在所有可核实的历史选模/诊断/评价 reference 清单中，且映射和标注确实可用。暴露记录不完整即 UNKNOWN_EXPOSURE，不称独立。

满足者按 reference UUID 排序，最多8个；每 reference 使用可用扫描的既定 metadata 顺序、最多前5个唯一扫描，并生成 T2..min(5,Tmax) 前缀，方法无关，不按表现选择。真实时间未核实就叫规定顺序，不叫按日期。至少含2个 reference 且有一个T≥3才运行；不足则 CONFIRM=UNAVAILABLE/INSUFFICIENT，记录原因，继续核心实验。独立性只针对已核实任务训练/选模，不默认排除基础编码器预训练暴露。

## 3. 固定的新方法与四臂定义

不要在执行中继续发散到 RGB 教师、额外基础模型、骨干微调、容量淘汰、更多 query 或多轮闭环。首轮 Q=100、D=128、K=100，沿用最多32历史提示（recent16、dormant16、余位按旧规则补齐），其余 FPS 查询不改。保留68个查询不意味着发现精度一定不损失。

四臂均从 E1-23760 父包和**共享的全新 M4 提示初始化**开始，不加载旧 M4-200 的适配权重。第一适配 seed=246，第二 seed=247；父 seed45不变。各模块以独立 RNG 命名空间初始化，共享部分同种子逐张量相同，增加模块不能改变父网络初始化/RNG。

| 配置 | 受限实体分割监督 | 区域身份头 | 最终关联 | 中途反馈 |
|---|---|---|---|---|
| R-TARGET | 有 | 无 | 旧共同 route/commit | 无 |
| R-ID | 有 | 有 | 新区域证据部分匹配 | 无 |
| R-QFB | 有 | 有 | 新区域证据部分匹配 | 有，value=条件化 query |
| R-OBS | 有 | 有 | 新区域证据部分匹配 | 有，value=当前区域池化 |

R-QFB 与 R-OBS 的**对应概率 P 和历史证据汇总完全采用同一种区域身份读出定义**，只改变反馈的直接 value 来源。R-QFB 仍通过 P 间接使用区域证据，不能把它叫“完全没有观测证据”。这组对照隔离的是直接反馈内容来源，不是所有可能的区域证据用途。

两臂使用同一中间层、相同参数、相同损失和执行次数。R-TARGET、R-ID 是不同大小的合法消融，不声称等参数。R-ID 的身份损失可能反向改变掩码，其跨模型效果也不能称纯后处理效应。

## 4. P1：34条固定输入的目标对应诊断

使用完整 CAL T3–T5 共34条输入，仅终点进行干预；其历史由 M0 独立预测生成。复用标签账本在分析端绑定可靠历史、生成几何及新观测 sidecar。模型输入中不得含该账本。

同终点用相同 M0 预测历史、当前结构和 RNG 运行旧 M4-200；这是共享历史的单步诊断，不是 M4 的自历史部署分数。有限预算内依序完成全部34条，无法完成标明实际范围，不默默删除大场景。

### 4.1 TARGET_FLOW

每个提示记录 input_id/reference/scan_uuid/absolute_stage、source_slot/logical_id/generation、source_query_id、可靠 GT 绑定状态、当前可评价目标状态、真实 anchor、top1/top5 段及命中、邻域点数加权纯度、同anchor碰撞、raw_target_IoU、raw_sameclass_best_IoU、官方保留和 published_target_IoU、最终发布 ID。

`raw_target_IoU` 对源提示 query 与源实体当前 GT 计算；缺预测的有效目标计0。`published_target_IoU` 只看该 source_query_id 对应且类别正确的官方候选，不从全体 query 事后挑一个最好者。无可靠目标记 NA 并计数。IoU阈值0.5；同类最佳IoU仅诊断“分对类却分错实体”，不参与选模。

提示条目与唯一实体事件分开。重复历史槽绑定同一个GT时，主条件指标用§7.1的规范槽，不把重复槽当独立对象。报告全部被选提示、可靠且当前可评价提示、存在正段提示三个分母。logical ID 与 GT实例ID处于不同空间，不能直接比较整数是否相等判断正确。

### 4.2 正确落点参考

每条输入最多2个可靠、当前有正段的提示，按源 logical_id/generation/slot排序，不按失败程度选。正段须有唯一可靠多数且纯度≥0.5；从纯度最高的段选 anchor，并以最小稳定段ID打破并列。

每个干预只替换该提示的 anchor 及由原 nearest32规则计算的邻域、z、当前位置；保留原历史特征、全部其他提示、解码权重与状态。与该输入同一真实anchor运行配对。不要给模型GT掩码或GT原型。没有正段记录 NA。

输出 ANCHOR_REFERENCE，注明 GT_ASSISTED_DIAGNOSTIC。它是诊断参考，不是正式成绩或理论上界。不以“正确anchor没有涨点”作为取消整个任务的门槛；四臂的直接实体监督本来就在检验该问题。

### 4.3 训练覆盖

首先读取实际旧 run root 的阶段日志，汇总 auxiliary_labels、birth_labels、selected_slots。远端 TRAINING.csv 曾返回空文本，不据此断言原训练日志为空，也不声称已经重新计算全部日志。旧日志不可得则明确 UNAVAILABLE；新训练必须直接输出每步汇总覆盖，不为补旧日志重跑六条旧训练。

## 5. P2：模型、观测 sidecar 和反馈的确定实现

### 5.1 新文件与继承

新建 `models/observation_grounded.py::ObservationGroundedReScene`，基于现有 RevisitPromptReScene；共同提示前端按 M4 运行，独立 `ogr_variant` 决定新模块。覆盖 train()：所有父参数和buffer保持eval，只有新提示/身份/反馈模块参与train。禁用 mpr_read、mpr_support、native_long_feedback、native_long_sampling、save_segment_info；新 R 配置的 mpr_support_probabilities 固定为全1。

优先用现有 before_first_mask 和 after_decoder_stage，不编辑原 ReScene forward。解析父配置并核实 hlevels、shared_decoder；不通过“解冻最后一轮”悄悄训练共享解码参数。

### 5.2 未直接注入本次提示的 F_obs

从 before_first_mask 的 dense_features 在修改查询前捕获 `F_obs [B,S_pad,128]`，父特征detach；保留padding、current_indices和行序。它可包含局部双扫描 backbone 上下文，只称“未直接注入本次实体提示的当前支持区域特征”。

点数 `n_s` 由 StageMeta.full_resolution_point2segment 与 local_stage_ids 在当前扫描上 bincount，长度等于完整有效段数；padding、前一扫描段权重0。禁止用实例GT面积或类别生成 n_s。B=1，所有shape在输入边界检查一次。

对于中间或最终 `mask_logits [S,Q]`，`p=sigmoid(mask_logits)`，

`v_q = sum_s(n_s*p_sq*F_obs_s) / clamp_min(sum_s n_s*p_sq, 1e-6)`。

只在current段求和，FP32；分母低于1e-6时 v=0、valid=false，身份只能null。训练不得detach p/v/P，不得对这段使用no_grad。无需构造全分辨率[点数,Q,128]张量，直接使用加权矩阵乘法。几何仍用于旧提示前端；本轮新身份头不额外堆叠新的几何分支。

### 5.3 ObservationEvidenceState

新建 `models/observation_evidence_state.py::ObservationEvidenceState`，字段：prototype[B,K,128]、valid[B,K]、quality[B,K]、logical_ids/generations[B,K]、last_updated[B,K]；明确dtype/bytes。它是运行状态，不是nn.Parameter，也不是可训练checkpoint的一部分。

初始全空；在最终 commit 已接受 current support 后更新一次。birth使用当前最终 v 的L2归一化值；既有槽使用与 common state 相同的 confidence_ema 速率，再L2归一化；quality同速率更新。occupied 槽 ID/generation必须与 common state一致，即使prototype暂invalid也保持身份绑定。无当前支持则保留旧值，不擦除、不把“未观测”当物理删除。

四个R配置都维护同一sidecar、K100、无淘汰；M0参照在共同局部wrapper中也维护sidecar但不读取，以保持状态更新可核对。报告旧M0和新wrapper差异及额外状态开销。整个主干前向仍一次，不读未来点云；episode结束状态清空。

### 5.4 身份头

新建 `models/observation_grounded_modules.py::IdentityEvidenceHead`：Wc/Wh 是无bias的128→128 Linear，初始化为单位阵；一个可学习 null_logit，初始0。两投影不是随机独立空间，以免无依据破坏同一观测特征域。

有效历史列是occupied & prototype.valid；按slot索引稳定排列。对于当前区域v与历史原型u：

`L_qk = cosine(Wc(v_q), Wh(u_k))/0.1 + 0.25*log(clamp(dot(c_q_fg,c_k_fg),1e-6))`。

c是原类别概率除最后no-object后的前景概率重新归一化，分母不足则类先验项0。c在此detach，避免把ID损失变成另一个分类器目标。L2 eps=1e-6，null为单独一列。`P=softmax(concat(L,null_logit), dim=history)`；无有效历史/无有效区域时Pnull=1。

这里是行归一化的软对应，不假称双随机或Sinkhorn。真实一对一约束在最终部分匹配实现。训练ID监督见§7.2。

### 5.5 只执行一次反馈

在 `execution_stage_idx == len(hlevels)-1`（默认3）计算中间mask、区域v和P。R-ID也在此计算相同中间身份损失，但不更新query。R-TARGET无身份头。

对被选历史提示query计算：

- `r_q = 1-P_q,null`；
- `hbar_q = sum_k P_qk*u_k / clamp_min(r_q,1e-6)`；
- `hsource_q = old_state.embedding[source_slot]`（无梯度）；
- R-QFB：`e_q=decoder_norm(query_q)`；R-OBS：`e_q=v_q`。

反馈适配器输入是 `[LN(e_q),LN(hbar_q),LN(hsource_q)]`，每项128，总384；LN无可训练affine。`Linear(384,128)+GELU+Linear(128,128)`，最后Linear权重/bias初始化0。输出：`q_new = q_old + r_q*MLP(...)`。

不要把q重复拼接两次作为R-QFB输入，否则人为降低对照的有效输入秩。R-QFB与R-OBS采用同一个MLP定义和相同初始化，只替换e。除选中的提示queries外其余query不直接更新；后续自注意力可能间接改变它们，不能称完全保护发现分支。

不修改query_pos，不再取一次hard anchor，不增加第二遍backbone或额外完整decoder。若历史为空，提示、位置、反馈全部严格noop；T1相同属于结构约束，不叫学会了保留短时能力。

保留中间P/v/queries到当前forward的返回字典供损失使用，避免超类finally清理使梯度信息丢失；阶段结束释放，不把计算图放进持久状态。用实际钩子后产生的第一个预测位置核实aux索引，默认feedback后从aux4起受影响，不把hook前的aux3称为反馈后输出。

### 5.6 预期参数范围与注册

共同旧M4提示模块应为84,224可训练参数。身份头预计32,769；反馈适配器预计65,792；因此四臂预计84,224 / 116,993 / 182,785 / 182,785。以本节结构生成的实际参数数核实；差异必须解释后修正实现，不能将“未使用参数”计入等参数证据。

新可训练前缀精确允许：`model.mpr_prompt.`、`model.ogr_identity.`、`model.ogr_feedback.`；R-TARGET没有后两项，R-ID没有反馈。父mask_features_head、query_projection、class/mask heads、共享decoder、Concerto任务decoder及criterion内部父参数全部冻结，旧训练温度参数不能漏进optimizer。

## 6. P2：最终身份匹配与状态提交

对最终raw query输出调用原 build_final_prediction_observation：原confidence≥0.5、mask≥0.5、minimum current support=1。新身份头只为当前支持且原valid的query参与匹配；无当前支持不因历史高相似产生对象。

R-TARGET用原route_entities；R-ID/R-QFB/R-OBS用区域P生成部分匹配：

1. 行按query_id升序，实列按slot升序，每行追加一个仅该行可用的dummy列；dummy cost=0。
2. 实边只有P_qk≥0.5且P_qk>P_q,null且当前有效时可用；cost=`log(Pnull+1e-6)-log(Pqk+1e-6)`。不可用边设有限大正值1e6。
3. 用SciPy linear_sum_assignment；结果按原query/slot索引还原。平价最优按固定矩阵顺序与绑定SciPy版本解释，不加入会改变有效阈值的随机噪声。一个历史槽至多分给一个query，dummy表示没有历史身份。
4. 构造合法 EntityRoute：query_to_slot、slot_to_query、route_score、prior_logical_id、prior_generation、current/previous support、valid、stage_indices、旧state/metadata绑定均齐全。复用原验证与hash构造函数，不只替换track_step中的ID。
5. 调用原commit_entities；未匹配且当前合理的query按原birth规则成为新实体；容量不足保留ephemeral，不丢掉官方候选。旧无淘汰规则不变。
6. 只在commit后更新common state、geometry、observation sidecar。返回raw query索引对应的新ID；官方候选通过source_query_ids引用，不按候选列号猜query。

**身份P绝不乘到官方候选分数、类别概率或common observation.confidence。**所有新R配置保留原掩码、类别分数及mean轨迹聚合；ID低只表示不认领旧实体，不表示分割本身无效。最终提交保持commit0，只追加当前输出，不回写过去mask。各方法正式评价都使用自己的预测历史。

## 7. P2：损失与标签分配的精确定义

### 7.1 受限实体集合分配

新建 `models/observation_grounded_loss.py`，包装/派生原SetCriterion，复用get_loss及原contrastive计算，只改变assignment与必要的忽略行处理。不得既调用完整原criterion又对同一目标追加完整掩码loss。

每阶段只在训练端读取ledger和可靠GT；构造提示来源身份：(reference_uuid, instance_id)。定义当前可评价目标为共同因果前缀接受、在当前扫描有真实mask支持且不在ambiguity组的实例。

对于每个当前可评价目标：若被多个选中历史槽指向，按旧state.confidence降序、last_seen降序、logical_id升序、generation升序、slot升序选唯一规范源query。该query在最终层与所有aux层固定监督该目标。对应GT从剩余Hungarian目标中移除。

同目标的其他重复提示query本阶段不参与task CE/mask和IDloss，记DUPLICATE_SOURCE；在CE中使用显式ignore mask，不能让原loss_labels把它误当no-object，也不能删掉其正式预测。未知、歧义、无当前可评价对应的提示不强制身份；它们与发现query一起在剩余GT上使用原Hungarian。这个选择不等于确认它们物理消失。

剩余匹配每个aux层按原方式独立计算并还原全局query/GT索引。合并后query与GT各至多出现一次。no-object与empty GT沿用原语义；忽略行不进入CE分子或加权分母，所有行被忽略时返回与图连接的0。掩码loss对所有已分配目标使用原local pair的完整目标mask；固定来源目标因此既监督当前，也保留该local pair中可见的过去部分，不额外复制当前loss。主结果仍只发布当前扫描。

原raw_sum实际返回的各task/aux/contrastive键及尺度必须保留，不能偷偷切换到weight_dict加权；新身份项单独加一次。没有任何forced/ignore条目时，新criterion要在相同RNG下与原criterion逐损失一致，作为一次针对性检查。

### 7.2 区域身份损失

R-ID/R-QFB/R-OBS在反馈前与最终各有一个区域P，均以同一当前目标分配规则确定query的可靠GT身份。反馈前使用该层受限assignment，不用最终mask事后挑最佳query。只针对assigned且当前有可靠GT支持的query。

监督候选列使用有效观测原型并有可靠ledger绑定的历史槽；未知历史列保留在模型forward的P中，但不进入监督归一化，不能将其当负身份。若真实GT身份在多个可靠历史槽中重复，正标签为整组，用`logsumexp(all_known+null)-logsumexp(positive_group)`，不随意挑其中一个。

当前可靠目标没有任何已知历史对应，且所有有效历史列身份都可核实时，才监督null（“当前对象没有被这份可核实记忆表示”）；只要有未知/歧义历史列则该null监督UNKNOWN。未匹配到GT的预测、不进入官方GT的对象和没有标注保证的空白，不自动做null负例。不同可靠实体自然作为对比负列，包括同类对象。

`L_id = 0.5*(L_id_mid + L_id_final)`；每一项按该阶段有效query均值，无有效项返回连图0。记录known positive/null/unknown、duplicate groups和reliable coverage，不因UNKNOWN删掉主评价输入。

`L_total = L_task_constrained_raw_sum + lambda_id * L_id`。R-TARGET的L_id=0。旧M5检索loss、旧M6support BCE均关闭，不叠加新变量。

### 7.3 lambda_id 只用TRAIN决定一次

工程探测采用四个配置各20真实更新，前4为warmup，其余计时；探测权重不进入正式训练。R-ID探测暂lambda=0.1，只在最后8个计时更新中有可靠ID标签的阶段，记录相对于共同`mpr_prompt.output_layer.weight`的task/id独立梯度范数（用autograd.grad，不额外执行训练任务）。这也是本轮唯一的量级探针。

取正且有限的范数各自中位数Gtask/Gid，候选{0.01,0.03,0.1,0.3}中选择使`abs(log10(lambda*Gid/Gtask)-log10(0.1))`最小者，并列取较小lambda。三身份臂共用，正式开始前锁定；不看CAL来选lambda。无有效标签则记录并在同一探测限额内读取紧接的最多4个预定TRAIN episode；仍无有效值，默认lambda=0.1并记录UNCALIBRATED_NO_VALID_GRADIENT，不伪称校准成功。若数据合同显示整个训练总体不存在可靠对应，则该主张NOT_TESTABLE，勿运行明知没有监督的长训练冒充验证。

梯度探针只检验该路径和量级，不做每更新全模型梯度扫描。零初始化时早期上游反馈参数梯度为0可能正常，必须在至少一次输出投影更新后检查有效路径，不能以step0上游0梯度直接判坏。

## 8. P2：训练与推理的数值政策

建立 `ogr-frozen-backbone-v4-derived-v1`，不使用旧RUN_LOCK作为新任务授权，不改旧V4文件。

新训练和评价都复用原V4的冻结主干计算方式：稳定group mean/centroid、Native邻接和ordered_subm。新builder显式pin_native_convolutions；训练时仅对冻结backbone使用`torch.no_grad()`和ordered sparse context。离开backbone后恢复正常grad。不要在训练前向外层用inference_mode，也不要从inference_mode创建的缓存直接送入需要反传的线性层。

旧`evaluation_builders`只patch旧builder，新builder不能假定会自动被覆盖。旧`install_evaluation_policy`会改全局类方法；在新独立worker内创建有恢复动作的上下文，或显式调用相同函数，记录新源码绑定。不要修改旧函数的“inference only”检查。mean/centroid只作用于冻结输入特征和坐标，新软mask池化是独立可微运算。

实际环境优先复用已成功的2.6.0+cu126/spconv2.3.8环境及原第三方源码；核实时若不同则记录并尝试现有已安装的正确环境，不做无上限跨版本重装。FP32、TF32关闭、未启用trainer AMP；若继承的Concerto内部仍用FP16 FlashAttention，应如实记录，不声称所有算子FP32。

只做两类有限重复：新路径小面板两新进程；最终关键端点同一小面板两新进程。固定数据/离散映射必须一致，报张量差和官方指标差。可接受的测量噪声上限是每长度AP极差≤0.2pp，真实差异不抹掉；若跨越选点差值或改变关键离散关联，标记NUMERICALLY_CONFOUNDED，不据小差值宣称机制胜出。超过范围最多一次具体修复和同面板复查；不重跑全部候选直到偶然一致。仍失败则交付数值受限结果，不把错误协议混进主比较。

## 9. P3：统一训练量、运行顺序与恢复

### 9.1 在CAL揭晓前锁定N

基于20更新探测，按各配置计时更新的p90、实际startup及每GPU卡数预测。第一种子成本=`1.25*sum_v(startup_v+N*t_v)/3600`；第二种子保守预留=`1.25*((startup_OBS+N*t_OBS)+max_control(startup_c+N*t_c))/3600`。这些数字不可用旧9.124GPUh直接替代实测。

候选N按1200、800、400降序，选择满足第一种子16GPUh、第二种子12GPUh、全部必需评价12GPUh、其他阶段与总available_new约束的最大值。N不由哪组CAL更好决定。正式checkpoint为N/4,N/2,3N/4,N；所有新臂相同。探测N=20与正式N分开，正式从共享新初始化和全新optimizer/update0启动。

评测预测用实际固定面板每阶段时间、1.5倍率及所有必需输入数；先预留44个完整评价job和有限机制/重复面板。可在probe使用CAL结构做无成绩计时，但不得读取其指标决定N/超参数。第一完整CAL只能在RUN_PLAN冻结后执行。

没有任何N可行：写NO_FEASIBLE_COMMON_N，发布代码/探测/诊断，不用零训练或少于400的结果冒充正式四臂。已经启动某N后不因CAL结果换N或重置scheduler；真实成本超界则在优化器边界保存，记录中断和未完成分母。

### 9.2 正式训练

AdamW lr1e-4、betas(.9,.999)、eps1e-8、weight_decay.01，grad clip1.0；5% warmup、cosine降到0.1×lr，复用lr_scale的边界语义。stage truncation=1，augmentation=false，data workers=0，per GPU1 episode。

每更新包含T2/T3/T4/T5四episode，各阶段loss按`1/(4*horizon)`归一化，合计14次真实stage forward。每episode状态和标签账本从空开始，下一stage只能看到已提交的预测；每stage反向后状态detach，不泄漏跨episode图。

seed246四个臂同draw计划；seed247两臂共享另一draw计划。完成度按optimizer updates、episode count、stage count三种计数分别记录。T1空历史阶段无新参数梯度属于设计事实；不能用累计stage数冒充更新数。

每50更新保存精确resume，包含新增参数/缓冲区、optimizer/scheduler/RNG、next_draw=step*4、源代码/数据/父包/lambda/N身份。每个选点保存delta。若在未完成某更新时中断，恢复到上一个完整更新边界并记录实际重复费用，不把未完成microstep当已完成。

## 10. P4：CAL选模和第二种子

### 10.1 指标单位和完整性

所有机器数值统一0–1，论文表显示×100，pp=(a-b)*100。`S_long=mean(AP_T3..T5)`；`S_current=mean(current_AP_T3..T5)`，每个AP来自官方充分状态池化。无GT或分母0为NA，不填0。

每个CAL检查点一次完整评价，不把旧三重复要求带入新选择。全部四点必须完整有限且拥有相同input_id/reference/horizon集合；缺点的变体不可列为胜者，保留其状态并在预算内修复具体任务。

每臂在其四点中先取相对新wrapper M0满足`S_current>=M0-0.002`且`T2>=M0-0.002`的点。按S_long、S_current、T2降序（差≤1e-6视同），最终较早step优先。具体实现按step升序稳定扫描四点，以该逐指标容差规则更新最优；不依赖可能非传递的容差比较器全局排序。全部不通过时仍锁定描述性最佳，但`eligible=false`，不能命名验证成功。

对照从R-TARGET/R-ID/R-QFB各自锁定点中按相同eligible优先、S_long/S_current/T2、较少实际参数、名称排序决定；按控制名称升序稳定扫描应用该比较规则。R-OBS固定为预定研究候选，不把CAL表现更好的其他配置改名为R-OBS。

### 10.2 第二种子不再由旧MPR门槛取消

在任何SEL/PB新结果揭晓前锁定seed246各端点及`control_star`。然后**无论第一种子的方向正负，只要此前锁定预算仍允许且实现有效，都完成seed247的R-OBS与control_star同N全程训练**。这正是本轮预留12GPUh的用途，不复用旧seed147 gate。

seed247也用自己的完整四点CAL和相同选择规则，控制家族不变，不在seed247重选对照种类。两种子均报告，不挑其中最好者。不将seed246和247叫独立E1训练复现。

若两臂中一臂失败且一次具体修复无效，配对复验标记INCOMPLETE，另一臂结果照实保留；不能把失败臂悄悄换成M0。

### 10.3 正式评价清单（不含CONFIRM）

- M0、旧M4-200：CAL/SEL/PB各1次，共6jobs；可复用旧结果只在新输入、父输出、数值、候选与状态语义逐项相同时，仍在job清单登记REUSED_EQUIVALENT，不默许复用旧均值。
- seed246四臂×4 CAL点：16jobs。
- seed246四臂锁定端点×SEL/PB：8jobs。
- seed247两臂×4 CAL点：8jobs。
- seed247两锁定端点×SEL/PB：4jobs。
- FH-COMMIT0×SEL/PB：2jobs。

合计44个完整评价job。诊断、probe、有限重复、profile不是上述44个，不混合计数。导出 EXPECTED_EVALUATIONS.json，逐项有COMPLETE/REUSED_EQUIVALENT/BLOCKED/FAILED状态，所有指标数量从实际任务推导，不硬编码“应该已有多少状态zip成员”。

## 11. 完整前缀对照与CONFIRM

### 11.1 FH-COMMIT0 的唯一含义

使用同一E1父模型、无提示、相同共同持久route/commit；阶段j网络输入为X0..Xj完整前缀，而不是local pair。当前候选经同一官方后处理，只提交j当前扫描，mean轨迹分数，早先mask不回写。没有GT进入关联。它是本轮采用完整因果上下文的部署对照，**不是ReScene原论文官方主表、也不是理论上界**。

新loader支持window_mode=full_prefix并正确构造所有local时间、offsets、段数、原始点索引；共同GT仍是同一0..j前缀，current stage=窗口最后阶段。每j真的跑prefix，不在最后使用全序列给过去预测补分。默认不增加历史encoder缓存或其他加速变量。

如果完整前缀资源失败，按相同输入保存OOM/耗时及缺失状态，不删大输入；主表该行INCOMPLETE，不能用旧FH-B2或官方论文数字替代。其缺失限制全历史比较，不使其他四臂内部结果失效。

### 11.2 CONFIRM

只有P0已锁定且满足曝光核查的数据才执行，模型/阈值/训练量已冻结。运行M0、seed246 R-ID、seed246 R-QFB、两种子R-OBS及配对control（重复端点去重）；若control为R-TARGET也纳入相应种子。预算4GPUh含profile，超界整面板记NOT_RUN_BUDGET，不根据效果挑subset。没有合法CONFIRM则不运行假确认，最终独立性状态单列。

## 12. P5：机制、身份与统计

### 12.1 单步机制与同类提示置换

锁定端点后，在同一34条CAL诊断输入与M0共享预测历史上比较R-ID/R-QFB/R-OBS(seed246)。使用相同目标事件集合、输入、旧state和RNG。每事件记录§4.1的目标指标；shared-history只解释局部机制。

R-OBS同checkpoint再做反馈off（仅残差置0，其余头保留）对照，记录指定targetIoU和当前候选，而不只存changed=true。最多每输入一对可靠同类A/B提示，按GT可靠性及源logicalID排序取第一对；交换完整历史描述/几何及来源元数据后，期望目标对应也同步交换，最终commit中的逻辑state不改。只记录单步结果，不把干预状态继续写入正式部署。没有同类对保留NA。

该置换诊断检查“输出是否跟随指定实体”，不是训练GT提示，也不是新的独立数据。off属于可能离分布的干预，不能替代R-ID/R-QFB重新训练对照。

若某面板调用数量超阶段剩余额度，先完成不干预的共同34输入比较，再按input_id排序执行off/交换，报告覆盖。不按哪个案例好看来删留。

### 12.2 各自历史的完整运行

SEL/PB/CONFIRM所有主指标均从各模型自己的空状态完整滚动，不读M0、GT或全未来历史。保留原官方全量输入和类别协议，当前mask评价包括ephemeral官方候选。对比M0的恢复／丢失使用相同目标事件，身份统计复用原定义并公布correct/opportunities、correct/attempts、switches/comparable分子分母。

gap由规定序列中的可评价可见记录间隔定义，不叫真实物理移除或天数。真实搬动标签未核实则MOVING_SUBSET=NOT_AVAILABLE，不新增质心标签。主评价保留未知/歧义对象，只有指定身份诊断排除并报覆盖。

### 12.3 统计与结论

每个评价job保存总体与每reference官方充分状态。原AP实现作为结果源；CPU加速复算只在2个固定实状态池化上与原接口对齐一次，不重写官方积分、抽样单位或所有旧统计。

SEL/PB分别按物理reference配对bootstrap1000次seed9101。每次包含抽中reference的全部序列和重复抽中的多重性，重新池化每方法官方状态。seed246的R-OBS对三控制分别计算；两种子的R-OBS对control_star先各自算差再取种子均值，**不将多个种子的预测拼成一个更大测试集**。同一批reference抽样在全部比较间复用。

无可评价GT导致的抽样记NA、不重抽；有效draw≥950时报告有效draw条件下百分位区间并醒目标记NA比例，少于950则区间NA。该规则在看结果前锁定。当前面板reference<10须注明区间局限；重复过程、顺序和前缀不增加独立场景数。

科学判定不是停训门槛，必须如实分维度写：

- **TARGET_SUPERVISION_ONLY**：R-TARGET已改善，而R-OBS没有独立额外证据；不强保留反馈为贡献。
- **ASSOCIATION_ONLY**：R-ID改善时序/身份，R-OBS不能改善目标mask/current AP。
- **PROMISING_INCONCLUSIVE**：点估计为正但区间跨零、种子方向不稳或未超过关键控制。
- **SCREEN_NO_GO**：候选不能超过M0和关键控制，或净恢复/当前质量存在明确代价；保留负结果，不能事后改目标。
- **MECHANISM_SUPPORTED_INTERNAL**：两适配种子相对锁定control在SEL与PB的S_long/S_current点差均正；种子均值配对区间下界均>0；seed246 R-OBS相对R-ID及R-QFB的current AP和共享目标published_target_IoU均正，且未以降低有效覆盖换得；无关键数值混杂。这个保守标签是内部证据分级，不是CVPR标准。

多条不能归入同一标签时保留各项判定及PRIMARY=INCONCLUSIVE，不制造单一好看标签。CONFIRM可用且正向也不自动写READY_FOR_CVPR；独立性、外部系统覆盖、训练范围及统计结论分别陈述。旧成果和新结果分别成表。

## 13. P6：资源、必要检查与避免浪费

### 13.1 资源面板

从34条CAL输入按input_id选前6条，M0、M4、四R(seed246)和FH-COMMIT0共7配置。同一物理GPU每配置一次warmup、三次测量；每次用相同初始空state。计时用同步CUDA及端到端墙钟，关闭大型诊断和逐张量指纹；记录日志开关。报每阶段与episode累计、I/O/主干/提示读出反馈/关联/后处理、allocated/reserved、common+geometry+observation state bytes和真实占用槽。输出历史存储另列，不算作固定决策状态。

不将此计时与旧含大量审计开销P5相减；不宣称无限历史无损、显存减半或全局确定性。不上无关容量饱和/压力测试。

### 13.2 只保留四组关键正确性检查

**A 数据与因果性**：一个真实pair含非空GT、一个真实空/未知对应路径，检查UUID/time/point2segment/current counts；GT ids/classes/masks置换但不改结构，预测路径不变。数据角色交集一次检查即可。

**B 监督与梯度**：小张量检查forced/duplicate/unknown/remnant assignment，empty GT；一个真实更新证明新增参数改变、父参数不变；一次区域pool→mask→prompt梯度有效检查和零初始化后梯度检查。R-QFB/R-OBS actual active参数及初始化相同。不创建百种异常输入测试。

**C 路由与输出**：小矩阵检查两query争一slot、全null、无历史、容量满保留ephemeral、候选source_query_ids乱序映射；一次真实episode全链路通过。检查ID概率未进入分数缩放。

**D 资产与发布**：一次optimizer边界resume恢复参数/optimizer/RNG/cursor；真实锁定R-OBS与控制各一次公开delta重载前向；全部最终LFS资产一次空缓存下载和字节核验。不是逐模型反复重载数十次。

源代码新增边界的单元案例可合并到四个测试文件，总GPU接通/故障限额4GPUh。重试必须有具体错误和代码/配置修复，同一错误最多1次额外重试。无上限数值追踪、全仓回归、重装集群、重复哈希大文件、重复出版审计禁止。科学负结果不是重试理由。

## 14. 代码文件、CLI与工作包验收

以下均为**要求新建的接口**，不是宣称仓库已存在。命令执行前先实现并通过--help/参数解析；不得把不存在的入口写成已运行命令。

| 工作包 | 新路径／职责 | 输出与验收 |
|---|---|---|
| W0 | `scripts/ogr_assets.py`、`scripts/ogr_policy.py` | BASE/parent/data/budget解析；新模型全覆盖加载；无旧锁修改 |
| W1 | `datasets/observation_grounded_episode.py` | local_pair/full_prefix/CONFIRM显式协议；旧role映射保持 |
| W2 | `models/observation_evidence_state.py`、`models/observation_grounded_modules.py` | 区域池化、观测sidecar、身份头、等参数反馈 |
| W3 | `models/observation_grounded.py` | 首mask提示＋首轮后一次反馈；返回当前loss所需图 |
| W4 | `models/observation_grounded_loss.py`、`trainer/observation_grounded_trainer.py` | 受限assignment、组正ID监督、四臂训练／resume |
| W5 | `scripts/ogr_runtime.py`、`scripts/ogr_evaluation.py`、`scripts/ogr_diagnostics.py` | 一对一部分route、commit0、官方池化、目标诊断 |
| W6 | `scripts/ogr_selection.py`、`scripts/ogr_statistics.py` | 完整四点选择、控制锁、bootstrap与准确分母 |
| W7 | `scripts/ogr_campaign.py` | 子进程调度、共享reservation、断点恢复、44job清单 |
| W8 | `scripts/ogr_delivery.py` | 表/报告/模型/manifest、A/B、Git/LFS远端核验 |
| W9 | `tests/test_ogr_data.py`、`test_ogr_learning.py`、`test_ogr_runtime.py`、`test_ogr_delivery.py` | 仅四组关键检查，不跑所有历史tests |

配置入口：`conf/observation_grounded_v1/default.yaml`。按文末配置生成，不从旧MPR YAML原地修改。主命令必须实现：

```bash
python -m scripts.ogr_campaign resolve --config conf/observation_grounded_v1/default.yaml
python -m scripts.ogr_campaign diagnose --config conf/observation_grounded_v1/default.yaml
python -m scripts.ogr_campaign probe --config conf/observation_grounded_v1/default.yaml
python -m scripts.ogr_campaign lock --config conf/observation_grounded_v1/default.yaml
python -m scripts.ogr_campaign run --config conf/observation_grounded_v1/default.yaml
python -m scripts.ogr_campaign finalize --config conf/observation_grounded_v1/default.yaml
python -m scripts.ogr_delivery publish --config conf/observation_grounded_v1/default.yaml
```

`run`完成first train+CAL→control lock→second train+CAL→SEL/PB/FH→stat/机制/profile/合法CONFIRM；不需要用户逐阶段输入“继续”。`finalize`不启动额外训练。各入口可重复执行：只跳过身份相同、分母完整且已完成的job，失败/旧源/不完整不可误认为完成。实际参数、stdout日志路径和return code写COMMANDS，不预填结果。

开发提交和实验提交的源身份分开。所有正式训练用一个冻结数值实现；只改报告可继续。若修复会改变已训练数值行为，保留原attempt与费用，为受影响臂重建同条件比较，预算不够时明确INCOMPLETE，不能混用修复前后结果。

## 15. 必须生成的报告和机器数据

OUT至少包含：

- `EXECUTION_INSTRUCTION.md`（本指令原文）、`SOURCE_BINDINGS.json`（实际源码符号及diff）、`START_STATE.json`；
- `DATA_MANIFEST.json`、`EXPOSURE_MAP.csv`、`CONFIRM_STATUS.json`；
- `BUDGET_LOCK.json`、`COST_LEDGER.jsonl`、`RUN_PLAN.json`、`NUMERIC_POLICY.json`、`PARAMETER_AND_LOAD_AUDIT.json`；
- `SUPERVISION_COVERAGE.csv`、`TARGET_FLOW.csv`、`ANCHOR_REFERENCE.csv`、`TARGET_INTERVENTIONS.csv`；
- `TRAINING.csv`及必要的阶段级轻量JSONL、`CAL_ALL.csv`、`CAL_SELECTION_LOCK.json`、`REPLICATION_LOCK.json`、`EXPECTED_EVALUATIONS.json`；
- `MAIN_RESULTS.csv`、`BY_REFERENCE.csv`、`IDENTITY_COUNTS.csv`、`MASK_GAIN_LOSS.csv`、`SAME_CLASS.csv`、`BOOTSTRAP_RESULTS.json`、`PROFILE.csv`；
- 所有24个预设delta选点（6条训练×4点，若未完整则实际数量）、每条完整轨迹的final resume、每条轨迹CAL-selected的明确别名／manifest条目；
- `evidence/official_states.zip`：只存真实可重新池化充分状态，成员数量从实际文件枚举；每个文件指回方法、seed、step、role、horizon、reference；
- `REPORT.md`、`HANDOFF.md`、`DEPLOYMENT.md`、`COMMANDS.md`、`MODEL_MANIFEST.json`、`ARTIFACT_MANIFEST.json`、`FINAL_REVIEW.md`。

不要为了维持“24”保存假的checkpoint；若轨迹未完成，manifest明确缺失和last_complete_step。所有四臂负结果、失效guard、失败attempt都保留。

REPORT必须讲清：数据/父包/冻结范围、四臂真实机制差异、CAL和SEL/PB主表、强对照和两种子差值、共享历史机制与自历史部署区别、current AP与目标IoU的证据、同类/gap净变化、真实成本、未完成部分与可支持主张。没有数据写NA及原因，不复制旧正向数字给新方法。

只生成三种数据图：T2–T5质量曲线；目标实体IoU/错配及覆盖；恢复与丢失和资源。用真实CSV绘图，成功和失败案例按固定清单展示。不要在任务末尾新发起插图生成或外部基础模型实验。

## 16. P7：Git/LFS发布与交接（必须实际执行）

### 16.1 本地完成与资产

先确认本任务训练/评测worker均已退出，reservation结算，ledger与实际GPU费用一致。不结束其他任务的进程。四个状态分开记录：CODE、EXPERIMENT、SCIENCE、PUBLICATION；缺CONFIRM可CORE_COMPLETE+CONFIRM_UNAVAILABLE，不叫已完成独立泛化验证。

遵循现有.gitattributes并只增加OUT的模型/zip LFS条目。全部真实delta/resume与指标包按manifest逐项上传；不要将父包/编码器再次打包。打包复制文件不反复计算同一大资产；原始点云、逐点GT、巨大训练特征缓存、tokens、私有路径不提交。提交前做一次针对改动文件的凭据/路径检查，不做全磁盘安全扫描。

### 16.2 两提交，避免最终SHA自引用

1. 实验提交A：代码、配置、真实结果、模型manifest及ARTIFACT_MANIFEST；manifest排除自身、后续HANDOFF和PUBLICATION_RECEIPT，不把未来A/B的SHA预填进自身内容。记录A之后不要在A内部补写A自己的SHA。
2. 交接提交B：HANDOFF引用A的真实完整SHA，说明父资产、各端点、重载和复现命令、实际阶段状态、预计发布分支/tag；不写B自身SHA。
3. 新分支推送到B；创建指向B的annotated tag并推送。不得force push、改旧tag、擅自合并main。
4. 使用干净的远端验证clone并指定独立空LFS storage，只fetch本tag/最终manifest列出的实体。逐项真实下载一次，比对bytes/SHA；`git lfs ls-files`出现指针不算成功。只核验当前最终交付实体，不重下全仓全部历史LFS。
5. `git ls-remote`核验分支对象B和tag剥离引用`refs/tags/<tag>^{}`均为B。annotated tag对象本身与B不同是正常现象，不能比较错对象。

### 16.3 外部回执与Release

核验完成后在运行目录写`publication/PUBLICATION_RECEIPT.json`，包含A、B、实际branch/tag、资产下载结果、远端引用、Git/LFS状态、Release API能力与错误。此回执不再提交到A/B，以免移动已经核验的tag。

已有可用API凭据时，创建或更新本任务同tag的Release，将回执作为附件；不修改其他Release。没有API凭据但Git/LFS成功：`GIT_LFS=PUSH_VERIFIED`、`RELEASE=UNAVAILABLE_API_CREDENTIALS`、回执本地实际路径；这是允许的完整Git/LFS发布分支，不得谎称Release已创建。Git或LFS失败则PUBLICATION=INCOMPLETE，保留本地commit和真实错误，不能仅以计划命令宣称已发布。

最终向用户报告代码/实验/科学/发布四项状态、选点、关键正负差值、总费用、A/B、branch/tag、报告/交接/manifest远端链接及回执存放方式。不要只说“已完成全部测试”。

## 17. 交付前内容审核：一次主审核，修复受影响处

在finalize和publish之间执行一次：

1. 源码与四臂身份是否对应，R-QFB/R-OBS是否只改变直接反馈value？
2. 新变量是否真的参与optimizer、保存、重载，所有父参数和buffer是否保留？
3. forced target、duplicate、UNKNOWN、null各有唯一规则；是否发生重复GT监督或GT进入运行状态？
4. V4是否仅在冻结主干no_grad复用，softmask/ID/反馈是否有真实梯度？
5. P是否未被当成掩码质量分数；source_query_ids和EntityRoute是否正确？
6. N、lambda、四点、最强控制是否在确认面板揭晓前锁定？第二种子是否真正完成同N而非推理重复？
7. 44项核心评价是否逐项有实际状态，missing/NA未被藏掉，官方池化与reference/bootstrap是否正确？
8. shared-M0-history诊断和own-history正式部署是否分表，条件目标指标覆盖是否一致？
9. 旧MPR数字、旧数值限制与新研究结果是否混淆，是否把现象写成已证明因果？
10. 账本、剩余额度、失败费用、资源范围是否一致；未启动新无授权大任务？
11. 报告是否保留负结果和真实独立性，是否从工程完成推导论文成功？
12. Git/LFS实体、A/B、annotated tag、Release与外部回执是否按实际事实对应？

每项记录所查文件/行或结果字段、发现、修正和最终状态。这是内容一致性审核，不是再运行一遍全部实验。只复查修复影响到的内容。科学结果不达标不算审核失败；数据或比较造假才是必须纠正的错误。

## 18. 研究范围与文献边界

沿用原CCFA-Skills的人类可读论证与最小充分证据原则：实验先回答问题，不按模板堆测试；拟定机制不写成已实测。原补充设计已读取；本文件精确化了此前未定的ID头、反馈value、分配、预算、选模和发布规则。

Cutie已研究对象记忆与稠密特征交互，故不能把“加对象反馈”本身当首次创新。这里预定的比较只支持“在本设置与强控制下，当前区域value反馈的增量”。未跑外部原系统就不写官方复现，不宣称最新SOTA。当前任务不再进行无限文献搜索或更换主题。

主要依据：
- 固定仓库源码及报告：`https://github.com/Orangekostar/Persist4D/tree/758b5aa4bd2e9119ae5e05f725faa360c2120d88`，详见C01–C20。
- CCFA实验设计：`https://github.com/mikubaka88/CCFA-Skills/blob/main/ccf-experiment-designer/SKILL.md`，适用规则已在本指令落实，不要求执行端重新安装整个skill家族。
- Cutie原论文摘要：`https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_Putting_the_Object_Back_into_Video_Object_Segmentation_CVPR_2024_paper.html`。
- PyTorch autograd官方说明：`https://docs.pytorch.org/docs/stable/notes/autograd`。no_grad产生的特征可用于后续grad计算；inference_mode不是训练时的等价替代。此文档用于语义依据，不授权升级绑定环境。

## 附录 A：机器配置（可直接生成default.yaml）

下列固定字段与正文配套。N、lambda、真实路径、实际资源计时和CONFIRM清单只可按正文规则从执行事实解析，写入RUN_PLAN；不能把它们的缺失当作随意选值的许可。

```yaml
schema: persist4d-ogr-execution-v1
date: '2026-10-18'
repository: Orangekostar/Persist4D
base_commit: 758b5aa4bd2e9119ae5e05f725faa360c2120d88
parent_source_commit: 6b1b779172afdeb51e2719a737cd05e7eeeca982
branch: research/persist4d-ogr-v1-20261018
tag: persist4d-ogr-v1-20261018-v1
artifact_root: artifacts/observation_grounded_v1
parent_bundle: artifacts/native_long_postselection_v1/deployment/E1-23760.pt
parent_sha256: 8c9d0d57044ce23c7964ddb1e7a2d8bd18e388e6ee657a7f4b02b1368b5b3a98
encoder_sha256: 845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07
variants:
- R-TARGET
- R-ID
- R-QFB
- R-OBS
baselines:
- M0
- M4-200
- FH-COMMIT0
budget:
  historical_reference_gpu_hours: 1760.667754119
  lifetime_cap_gpu_hours: 1823.0
  reserve_gpu_hours: 8.0
  new_cap_gpu_hours: 50.0
  phase_caps_gpu_hours:
    D_DIAGNOSTIC: 2.0
    T_FIRST: 16.0
    T_REPLICATION: 12.0
    E_EVALUATION: 12.0
    F_PROFILE_CONFIRM: 4.0
    I_INTEGRATION_REPAIR: 4.0
  max_parallel_jobs: 2
  gpus_per_job: 1
  training_cost_factor: 1.25
  evaluation_cost_factor: 1.5
  max_same_error_retries: 1
seeds:
  parent: 45
  first_adapter: 246
  replication_adapter: 247
  evaluation: 9001
  bootstrap: 9101
data:
  train_role: TRAIN
  selection_role: CAL
  historical_eval_roles:
  - SEL
  - PB
  confirmation_role: CONFIRM
  confirmation_default: UNAVAILABLE_UNTIL_EXPOSURE_VERIFIED
  expected_inputs:
    CAL:
      1: 31
      2: 19
      3: 14
      4: 14
      5: 6
    SEL:
      1: 39
      2: 21
      3: 21
      4: 15
      5: 15
    PB:
      2: 129
      3: 129
      4: 129
      5: 129
  diagnostic_horizons:
  - 3
  - 4
  - 5
  diagnostic_inputs: 34
  anchor_reference_max_prompts_per_input: 2
  same_class_prompt_pairs_max_per_input: 1
  confirmation_max_references: 8
model:
  num_queries: 100
  capacity: 100
  feature_dim: 128
  max_prompt_slots: 32
  recent_slots: 16
  dormant_slots: 16
  retrieval_neighbors: 32
  retrieval_temperature: 0.1
  freeze_parent: true
  augmentation: false
  tbptt_steps: 1
  feedback_after: len(hlevels)-1
  feedback_calls_per_nonempty_episode_stage: 1
  feedback_hidden_dim: 128
  feedback_input_dim: 384
  identity_projection_dim: 128
  identity_temperature: 0.1
  identity_null_logit_initial: 0.0
  identity_log_class_prior_weight: 0.25
  identity_acceptance_probability: 0.5
  identity_require_better_than_null: true
  pool_mass_epsilon: 1.0e-06
  pool_norm_epsilon: 1.0e-06
  norm_affine: false
  confidence_threshold: 0.5
  mask_threshold: 0.5
  minimum_mask_support: 1
  association_class_weight: 0.25
  association_threshold: 0.5
  update_mode: confidence_ema
  update_rate: 0.2
  max_update_rate: 0.2
  eviction: false
  feedback_assignment_source: regional_evidence_for_both_R-QFB_and_R-OBS
  feedback_value_source:
    R-QFB: decoder_norm_query
    R-OBS: current_region_pool
  candidate_score_identity_scaling: false
  trajectory_score_reducer: mean
  output_commit: commit0
training:
  candidate_updates:
  - 400
  - 800
  - 1200
  probe_updates_per_variant: 20
  probe_warmup_updates: 4
  probe_total_schedule: 20
  formal_checkpoint_fractions:
  - 0.25
  - 0.5
  - 0.75
  - 1.0
  resume_interval: 50
  episode_horizons:
  - 2
  - 3
  - 4
  - 5
  episodes_per_update: 4
  stage_forwards_per_update: 14
  optimizer: AdamW
  lr: 0.0001
  betas:
  - 0.9
  - 0.999
  eps: 1.0e-08
  weight_decay: 0.01
  gradient_clip: 1.0
  warmup_fraction: 0.05
  final_lr_fraction: 0.1
  identity_weight_candidates:
  - 0.01
  - 0.03
  - 0.1
  - 0.3
  identity_gradient_target_ratio: 0.1
  identity_loss_stages:
  - pre_feedback
  - final
  identity_loss_reduction: equal_mean_of_two_stages
  old_retrieval_aux_loss_weight: 0.0
  old_support_aux_loss_weight: 0.0
  objective: raw_sum_with_constrained_assignment
  replication_policy: R-OBS_and_CAL_locked_strongest_control_even_if_first_seed_is_negative
selection:
  metric_unit: fraction
  guard_current_drop: 0.002
  guard_T2_drop: 0.002
  tie_tolerance: 1.0e-06
  sort_keys:
  - S_long_desc
  - S_current_desc
  - T2_desc
  - step_asc
  control_variants:
  - R-TARGET
  - R-ID
  - R-QFB
  no_guard_pass: descriptive_best_not_validated_winner
  test_reselection: false
evaluation:
  full_process_repeats_per_checkpoint: 1
  finite_repeat_panel_new_processes: 2
  repeat_panel_inputs: 6
  numeric_AP_range_limit_pp: 0.2
  bootstrap_samples: 1000
  bootstrap_unit: physical_reference
  pooled_AP: true
  NA_draws: preserve_no_redraw
  NA_interval_min_valid_draws: 950
  same_seed_processes_are_training_seeds: false
  full_jobs_without_confirm: 44
  official_task_thresholds: inherit_pinned_evaluator
  target_iou_success_threshold: 0.5
numeric_policy:
  new_policy_id: ogr-frozen-backbone-v4-derived-v1
  frozen_backbone_context: torch.no_grad
  train_inference_mode: false
  precision: 32-true
  tf32: false
  reuse_legacy_policy_lock: false
  global_determinism_claim: false
publication:
  force_push: false
  overwrite_old_tag: false
  experiment_commit: A
  handoff_commit: B
  branch_and_tag_target: B
  manifest_excludes:
  - ARTIFACT_MANIFEST.json
  - HANDOFF.md
  - PUBLICATION_RECEIPT.json
  required_remote_verification: empty_LFS_cache_one_fetch_per_final_asset
  release_without_api: record_Git_LFS_success_and_Release_unavailable_separately_save_local_receipt
```
