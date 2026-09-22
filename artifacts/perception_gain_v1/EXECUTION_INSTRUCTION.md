# Persist4D：以 ReScene 原生指标提升为目标的 Codex 最终执行指令

**版本：Perception-Gain V1 / 2026-09-21 / 静态交付审核修订版**  
**任务类型：实际开发 → 真实训练 → 开发集选模 → 锁定 → 完整评测 → GitHub 发布。**  
**目标仓库：`Orangekostar/Persist4D`**  
**已核验起点：`6ef77620aa20926311eff3124a794a6ca2e32727`**  
**源分支：`research/persist4d-crosswindow-evidence-v1`**  
**新工作分支：`research/persist4d-perception-gain-v1`**

> 给 Codex：把本文作为本轮的唯一执行规范。先读指定源码和证据，再实现并运行；不要只交计划、脚手架、缓存分析或测试通过记录。预算内的训练、选模、确认和发布已包含在本任务中，无需逐阶段再次征求授权。算法是否涨点由真实结果决定，不得承诺或制造胜出。发现某条支线的真实阻塞时，记录原因并继续独立可执行支线，最后交付实际达到的完成状态。
>
> 本文中的新增模块名、配置名、CLI 和超参数是**本轮开发规定**，不代表它们已存在或已验证有效。现有代码事实由第 2 节固定版本来源支持；服务器文件可用性须在启动时实测。不得把网页检索、本文静态审核或历史 GPU 实验说成本轮实测。

---

## 0. 本轮目标、范围和成功口径

### 0.1 首要目标

提高官方 ReScene temporal evaluator 下的 t-mAP。保留两类互不混算的评测：

1. **LOCAL-T2**：继承 root-cause 阶段的 official-like 原生 T2 population、类映射、后处理和 evaluator，用同一 population 对比冻结 R1、同预算继续训练 C0 和新感知模型。
2. **PB-T2/T3/T4/T5**：继承已有 Protocol-B 的 43 masters × 3 orders，即 129 个逻辑序列单元；每方法应产生 516 个 T2–T5 逻辑前缀结果。比较原生 FH-R1、固定 D0、同预算 C0-D0 和最终候选。

43 masters、129 order-units 不是独立场景数。当前 PB 来自 6 个物理 reference；不得将相关前缀、顺序或重复观测当独立样本做显著性声明。[E01][E09]

**本轮精度是主目标，成本是必须实测的独立结果。** 不再要求一个有效精度候选必须同时降低延迟才允许晋级。不能省略成本测量，也不能用缓存重放时间代替网络前向时间。

### 0.2 必做开发范围

- F0：修复 E1 的 temporal 发布失败判定，并接通原生 FH 与当前 D0 的真实前向及计时。
- C0：冻结 R1 权重作为初始化的同预算继续训练对照。
- S-BAL：按真实阶段均衡的 mask 损失。
- S-WORST：在 S-BAL 上平滑强调弱阶段。
- Q-SEM：固定 Q=100、语义辅助且阶段/空间覆盖受控的查询位置初始化。
- A-OPEN：只在最早一次 cross-attention 去除预测 mask 门控，保留 padding mask。
- R-REFINE：在已选定感知父模型上，利用同一扫描 old/new 软证据训练局部 mask 修复器。
- 有限训练晋级、CAL/SEL 分工、冻结检查点、复核、PB/LOCAL 确认、GitHub 代码/结果/交接发布。

**本轮不做**：完整 backbone 替换、全历史教师蒸馏、新关联头、额外关联网格、多原型记忆重做、查询竞争模块重做、学习型轨迹评分、类别聚合改造、超点重新分割以及上述模块的组合网格。它们保留为下一轮方向，不得未经本文分支规则自行扩张。

### 0.3 结果状态分开表达

使用以下独立字段，不用一个笼统 PASS 覆盖全部事实：

- `execution_status`：`COMPLETE` / `PARTIAL_WITH_BLOCKERS`。
- `development_selection`：`NEW_PERCEPTION_SELECTED` / `CONTINUATION_ONLY` / `KEEP_R1`。
- `refinement_selection`：`REFINER_SELECTED` / `KEEP_PARENT` / `BLOCKED`。
- `local_t2_gain`：是否在同 population 上超过 R1；另列相对 C0 的差值。
- `pb_all_t_vs_native_fh`：四个 T 都严格高于实测 FH-R1-native，数值容差 `1e-6`。
- `pb_all_t_vs_d0`：四个 T 都严格高于实测固定 D0，容差同上。
- `replication_status`：`SUPPORTED` / `MIXED` / `NOT_APPLICABLE` / `NOT_RUN_BUDGET` / `BLOCKED`。
- `resource_status`：`IMPROVED` / `TRADEOFF` / `REGRESSED` / `UNCONFIRMED`，另附原始数值。
- `publication_status`：`VERIFIED` / `CODE_ONLY` / `BLOCKED`。

`local_t2_gain`、`pb_all_t_vs_native_fh`、`pb_all_t_vs_d0` 均采用 `true/false/null` 三态：对应所需覆盖完整才计算 true/false；缺基线、缺单元或评价失败时为 null，并提供 `comparison_status`。运行中允许独立的 `phase_status=PENDING/RUNNING/DONE/BLOCKED`；最终结果枚举不要用 PENDING 代替结论。规定的预算剪枝属于已执行决策，不是技术失败；必做臂因资产/代码/总预算缺失未完成时，execution_status 为 PARTIAL_WITH_BLOCKERS。

只有 `pb_all_t_vs_native_fh=true` 才能说“本 PB 协议下各 T 均超过原生 FH-R1”。论文参考值 34.8% 与本地 R1、PB 不是同一数值基线。只有 population、模型/输入协议等可比时才讨论论文指标；不能用不同 population 的数字相减来宣称超过论文。[R01]

---

## 1. 启动、环境和资产绑定

### 1.1 工作树

在现有克隆中只检查一次 `git status --short`、remote URL、HEAD 和目标分支。fetch 源分支后核对完整 SHA。

- 起点固定为上述 6ef7762 完整 SHA；不得自动使用 default branch、旧 P5 提交或远端新 tip。
- 已有工作树存在用户未提交修改时，在同一克隆创建独立 worktree，从固定起点建新分支，不 reset、clean 或覆盖用户修改。
- 新分支已存在且包含本轮同一指令/配置时恢复；若是无关实验，使用最小未使用后缀 `-r2`、`-r3`，在交接中记录。
- 若源分支已更新，记录新 tip，但本轮仍从固定起点工作；只有确定必须补入的基础修复才 cherry-pick，并在 `RUN_CONFIG.json` 记录补丁 SHA 和理由。

### 1.2 Python、GPU、目录

优先使用现有 `persist4d` Conda 环境。不要重装 CUDA、重建环境或升级全套依赖。只有导入/执行真实失败且能定位到依赖时才修最小问题。

新增输出环境变量：`PERSIST4D_PERCEPTION_RUN_ROOT`。若未设置，默认 `${HOME}/persist4d_runs/perception_gain_v1`，创建并检查写权限；这是**本轮规定的默认位置，不是已验证的服务器路径**。不得误把旧 `PERSIST4D_RUN_ROOT` 当作本轮可覆盖的输出目录。

输入优先级：本轮显式 CLI 参数 → 已设置的输入环境变量 → 现有 untracked `artifacts/task_memory_retention_v2/external_assets.local.json` → 已有 manifest 指向的已知位置。复用 `scripts/crosswindow_cache.py::resolve_assets()` 的键和别名；若需要调整输出键，在新 runner 做转换，不更改旧实验行为。[E02][E10]

必须定位：

| 输入 | 核验要求 |
|---|---|
| R1 checkpoint | SHA256 `629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd`；754,813,672 bytes |
| Concerto 预训练权重 | SHA256 `845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07` |
| 数据根、3RScan 元数据 | 使用实际路径，关联原 scene_id 与 reference_id |
| `metric_dataset_spec` | 保留原 rio 类别定义；不要凭记忆重建类别表 |
| DATA_CONTRACT、PB manifest、CAL/SEL 数据角色 | 读取既有文件，记录 SHA256，不重划 PB |
| R1 resolved config、LOCAL-T2 evaluation manifest | 从 root-cause 的 FINAL_MANIFEST/所指检查点 provenance 解析；模板 YAML 的默认值不是最终 R1 配置 |
| 旧 D0 base/supplement caches、旧 FH caches | 若存在可重用；缺缓存不是缺权重，必须实际尝试生成 |

不扫描整块文件系统或对原始数据集逐文件哈希。已知根目录的检查点候选只按文件大小初筛，再对实际选中的文件算 SHA256；输入权重一次、数据/协议 manifest 一次、新结果写入时一次即可。

### 1.3 预算：本轮研发上限，不是历史实测

下列 **192 GPU-hours 是本指令提出的新默认上限**，不是用户曾确认过的历史预算，也不是已消耗资源。可在首次 bootstrap 前由调用配置显式调低；将实际采用的总额与分类额写入 `RUN_CONFIG.json`，pilot 前冻结。Codex不得自行调高总额；不能将本轮默认值误写成此前32 GPU-hours合同的剩余额度。使用较低上限时仍按第9.7节执行同一删减规则，必做工作无法容纳则如实部分交付。

默认分类上限：

| 类别 | GPU-hours 上限 |
|---|---:|
| F0、CAL/SEL/最终前向和确认 | 32 |
| 感知模型训练，含 C0、scorer、候选晋级及第二种子 | 136 |
| 修复器缓存生成及训练 | 16 |
| 真实 profiling | 4 |
| 必要 smoke/一次故障恢复余量 | 4 |
| **总计** | **192** |

GPU-hour = 实际占用卡数 × 占用时长；不是墙钟小时。上限优先于计划步数，不能靠重新命名 run 清零。

默认每训练作业使用同节点 2 张 A40，`batch_size=2/GPU`、梯度累积 8，物理 global batch=4、有效 batch=32，FP32；这是沿用已验证 R1 物理批量，不尝试先扩大到 batch 8/16。[E03][E11] 两卡不同时可用但有一卡时，使用 `batch_size=2`、累积 16，记录执行差异，所有同种子比较臂用同一部署设置。

最多并行两个独立训练作业/4 张空闲 GPU；不占用其他已在运行的作业，不开展跨节点 DDP 调优。CPU 同时最多 8 workers，主机内存预算 96 GiB，新缓存总预算 32 GiB；超过缓存上限采用流式读取/按 shard 清理本轮可再生临时缓存，不删除任何原始输入或唯一检查点。

第一组真实训练的前 50 个 optimizer updates 同时用于吞吐估计，不另开性能实验。按该测量预测各阶段消耗，显示预算缺口；如不足，按第 9 节已规定顺序删减可选工作，不缩小 PB population、改变评价门槛或静默减少某个对照的步数。

---

## 2. 已核实证据及必须阅读的最小代码集合

下面只陈述固定提交中已核验的事实。历史诊断不自动成为当前选中 epoch-390 模型的根因证明。

| 证据 | 已核实事实 | 本轮判断与约束 |
|---|---|---|
| E01 | CrossWindow 最终保留 D0+M-new；PB t-mAP 为 0.249229/0.155346/0.090114/0.064331；native FH 与真实计时没有完整确认 | 以 D0 为固定系统起点；不能直接沿用空的 confirm 结果 |
| E02 | crosswindow YAML 固定窗口 2、Q/K=100、lag1/mean、eval seed=45；R1/Concerto SHA 明确 | 固定这些变量，新增感知机制不顺手换发布/关联规则 |
| E03 | R1 改 raw-sum objective 后取得受控改进；feature-seeded FPS 完整实验只有部分支持 | 采用真正 R1，不误继承模板的 weighted 默认值；Q-SEM 不重复 `use_np_features=true` |
| E04 | `loss_masks()` 在拼接的点/segment 支持上算 BCE/Dice，不按阶段独立归一 | S-BAL/S-WORST 是训练目标实验，不宣称修复了“官方错误” |
| E05 | `initialize_queries()` 在 FPS 位置使用零查询或可选特征投影；位置选择不因 `use_np_features` 改变 | Q-SEM 改位置生成；第一版仍用零内容查询，避免额外 query-content 混杂 |
| E06 | mask attention 以 detached sigmoid<0.5 屏蔽；全屏蔽才重置；有 execution-stage 计数 | A-OPEN 仅改变 execution_stage_idx=0 的 memory_mask |
| E07 | 历史 epoch-90 诊断中 background query fraction≈0.6508、earliest allowed GT≈0.0557 | 提供实验动机；先用小 panel 核查当前 R1，不把旧比例当当前模型实测 |
| E08 | E1 `_published_match_by_gt()` 对拼接 mask 用固定 0.5；coverage 对不同 tau 复用其结果 | 必须改成与各 tau 相符的官方 temporal 判定；该 E1 不再控制本轮感知训练授权 |
| E09 | 已有 TaskMemory 训练、真实 profiler、D0 prediction-only producer；V-CORE 未稳定胜出 | 复用工程入口，不从头搭建大平台，不重做多代表记忆 |
| E10 | CanonicalFrame/StageMeta 已有 scan、vertex、stage 对应关系 | 软缓存沿用这些对应，不能按 query 编号或数组位置跨窗口匹配 |
| E12 | official postprocess 的 dataclass 只输出 bool mask 等；full-res heatmap 被计算但没作为字段返回 | R-REFINE 要新增真实软证据导出；不能从 bool mask 假造“原始 logits” |
| E13 | mask features 在 decoder 循环前形成；`aggregate_features()` 返回 segment features 和 coordinates | 可作为 scorer/repair 输入；本轮不同时改成完整 mask-feature 更新解码器 |
| R01 | t-IoU 按阶段取最小值，双空阶段排除，并有歧义组分配 | evaluator 结果须覆盖缺失、严格阈值、歧义处理；不是 concat IoU |
| R02 | LaSSM 的语义与空间联合查询设计及 TCSVT 接收有作者/论文来源 | 只借鉴查询定位思想，明确写 LaSSM-inspired，不声称完整复现 |
| R03 | AQ3D 预印本讨论 query/feature 更新 | 仅为局部证据修复的邻近参考；本轮 refiner 是项目提出的新假设，不能声称论文已验证 |

### 2.1 源码 → 本轮工作绑定

完整阅读下列函数/类及直接调用点；不要求遍历所有历史目录：

| 现有文件/入口 | 要读懂什么 | 本轮动作 |
|---|---|---|
| `models/criterion.py::SetCriterion.loss_masks/forward` | mask 排列、匹配索引、采样、aux outputs、返回 loss 键 | 增加 legacy/balanced/worst 模式，保持默认 legacy |
| `models/matcher.py` 的 Hungarian matcher | 匹配发生在损失之前、点/segment 轴语义 | 本轮不改匹配 cost，仅验证新 loss 不误改其输入 |
| `trainer/trainer.py::_configured_objective_loss`、`InstanceSegmentation` | raw-sum 路径、freeze、optimizer/目标/数据 | 新训练器或窄扩展；不把 debug 标量加入 loss 字典 |
| `conf/config_rescene4d_concerto_rootcause.yaml` + R1 resolved config | 模板默认 R0/weighted 与最终 R1 的区别 | 新配置显式继承实际 R1 的 raw-sum、冻结、EOS、数据处理 |
| `models/rescene.py::initialize_queries/aggregate_features/forward/attn_mask` | 稀疏坐标、真实坐标、查询位置编码、decoder execution index | Q-SEM、A-OPEN、无 GT 的 feature 导出 |
| `datasets/task_memory_episode.py::StageMeta`、collator | local/absolute stages、original_vertex_ids、p2s、inverse maps | 在新 pair loader 导出同语义 StageMeta；不要平均得到浮点 stage |
| `scripts/crosswindow_cache.py::resolve_assets/align_mask/build_canonical_frame` | 资产与点序、候选 lineage | 复用；新增独立软证据 sidecar，不破坏旧 bool cache |
| `scripts/diagnose_crosswindow_failures.py::_published_match_by_gt/diagnose_candidate_coverage` | 当前错误分母、阈值复用、GT 辅助规则 | temporal 修复与事件 trace，不新增关联训练 |
| `scripts/rescene_task_postprocess.py::extract_official_task_prediction` | scorer/top-k/filter/full-res/lineage | 增加可选软证据返回，默认返回完全不变 |
| `scripts/task_memory_output.py`、`scripts/replay_crosswindow_association.py` | D0 身份与 lag1/mean、归档、publish | 原 D0 关联不改；在写入上一扫描输出时插入 refiner |
| `scripts/run_task_memory_controls.py` | frozen R1 prediction-only producer 与历史 supplement 的关系 | 参数化 checkpoint/config/新 cache 命名空间，生成新感知预测 |
| `scripts/system_comparison_inference.py` | FH cache key、全前缀输入、provenance | 给新 runner 接通真实 FH，而非固定 None |
| `scripts/p6a_metrics.py::OfficialMetricAccumulator`、`scripts/system_comparison_metrics.py::CausalTaskAccumulator` | pooled AP、类与点轴、完整前缀 | 所有正式分数复用该链，不实现简化 AP 代替 |
| `scripts/train_task_memory.py`、`trainer/task_memory_trainer.py` | checkpoint、episode、调度/恢复组件 | 复用可用函数；不把新臂塞进旧固定 VARIANTS 并改旧合同 |
| `scripts/profile_task_memory.py` | model-update/end-to-end/序列累积计时 | 接入本轮 D0/最终候选/FH 路径，关闭缓存计时替代 |

在 `CODE_BINDINGS.md` 中为每项记录：实际类/函数、起止行、调用方、新改动位置、对应实验 ID。运行时若发现函数名与本文不一致，以固定提交源码为准，在同一表纠正；不能因为名字不同就停止，也不能编造调用成功。

---

## 3. 新增结构、数据角色与只做一次的配置锁定

### 3.1 新增文件是任务接口，不是声称已存在

允许采用以下紧凑结构；若仓库有等价实现，直接复用并在 CODE_BINDINGS 说明，勿复制多套：

```text
configs/perception_gain_v1.yaml               # 新实验总配置
conf/perception_gain_v1/                      # 新 Hydra 臂配置
models/perception_gain.py                    # scorer、查询选择、refiner
trainer/perception_gain_trainer.py            # 最小训练适配
scripts/perception_gain_campaign.py           # 实际可运行的总 CLI
scripts/perception_gain_data.py               # split/pair/soft-cache 适配
scripts/perception_gain_evaluation.py         # checkpoint→真实输入→评测
artifacts/perception_gain_v1/                 # 轻量、可提交证据和交接
```

新模块全部默认关闭。禁止将历史 artifact 文件原地更新成本轮结论；F0 修正后的输出写入新目录并注明替代了哪个旧诊断。

### 3.2 数据角色

读取 `artifacts/task_memory_retention_v2/DATA_CONTRACT.json`、`artifacts/crosswindow_evidence_v1/DATA_ROLES.json` 和其中实际引用的 manifests：

- **TRAIN**：仅 DATA_CONTRACT 的 `adaptation` reference；从原 R1 3RScan 训练库过滤出这些 reference。保留原 ScanNet 训练子集与 1.0:0.8 nominal mixing，所有臂共用同一抽样流。先记录实际抽样规则，不能把 1.0:0.8 错解释成每 batch 必定 10:8。
- **CAL**：原 crosswindow DEV-CAL reference 与对应序列/顺序。只用于 pilot 排序和 checkpoint/配置选择。
- **SEL**：原 crosswindow DEV-SEL reference；只在候选配置和 CAL 检查点选定后评价。refiner 固定配方的 parent/no-refiner 比较也可用 SEL，但明确 SEL 仍是开发集，不是未触碰测试。
- **PB**：原 129 logical order-units，不改 master、order、prefix 或类别。只在最终 recipe 锁定后确认，不用于训练、调阈值、选择 checkpoint 或改 alpha。
- **LOCAL-T2**：原 official-like population；只做锁定后的确认，不能借它重新挑模型。
- **ADDITIONAL**：DATA_CONTRACT 中 `additional_native_refs` 的全部实际可读 reference；锁定后一次性评价其真实可用 horizons，单独汇报，不能把缺 T5 补成复制扫描。

这些角色可能已经被历史基础模型或人工结果查看暴露；不得因此写成“完全独立”“untouched”。将新增训练期是否使用、R1 历史是否使用、历史结果是否暴露分开记录。

TRAIN 与 CAL/SEL/PB/LOCAL/ADDITIONAL 的物理 reference 交集必须为空；若旧 TRAIN 角色与最终确认 population 有交集，**从本轮 TRAIN 中减掉交集**，保持评测人口不变，并写出删去的 reference。不得通过复制 scene ID 或只隔离 order 来绕过。

原 DATA_ROLES 若无法解析：优先修适配到已有 schema，不能重分一份“看起来相似”的 CAL/SEL。若文件确实缺失/损坏且无法从固定提交恢复，记录阻塞，不在看过 PB 后补划开发集。

### 3.3 episode/pair 规则

感知训练仍为单扫描 ScanNet + 两扫描 3RScan，不直接训练 full-history，也不引入 query-memory 状态。

3RScan 训练 pair 从 TRAIN reference 的真实 scans 产生，使用原生序列的相邻 pair；不同 epoch 的顺序随机化沿用原机制并固定训练种子。两扫描采用共同的几何/颜色增强变换。原数据没有的阶段不复制补齐。样本/增强随机流由 `(train_seed, epoch_or_draw_index, reference_id, pair_id)` 决定，并与模型内部 RNG 分开；不能仅设置一次全局 seed，就假定不同模块消耗随机数后仍使用相同数据流。

refiner 的 TRAIN 源是至少 3 次真实扫描的 TRAIN reference；上游固定父模型在两相邻窗口做真实前向，再按同一物理扫描及 vertex 对应产生样本。GT 只在离线训练标签、损失与诊断使用，不进入 scorer/refiner/association 的推理输入。

### 3.4 一次性配置记录

`RUN_CONFIG.json` 保存完整 resolved 配置、完整起点 SHA、权重哈希、population hashes、train/eval seeds、GPU 配置、预算、所有新参数、评价门槛和 rank 规则。pilot 前冻结。新增数据角色或评价规则变更必须作为显式版本变更，不能静默覆盖原结果。

新 Hydra 配置从正常 base configuration 组合并逐项覆盖为实际 R1 数值语义；旧 root-cause variant manifest、旧 common-initialization 授权检查、旧 OneCycle 总步数不作为本轮新臂的执行门槛。保留原数值行为与必要已有错误检查，不通过删除所有检查来取得可运行性。用一个已安排的真实 minibatch确认 `_configured_objective_loss` 实际为 raw-sum，而非模板默认 weighted。

---

## 4. F0：必要修复与真实前向打通

### 4.1 E1 的 temporal 失败判定

修复 `_published_match_by_gt()`，显式接收 `tau`，且每个 tau 重新判定。`diagnose_candidate_coverage()` 不再跨阈值复用 `published_failure`。

正式诊断取值集合为官方 t-mAP 的阈值数组；额外保留 0.25、0.75 展示行。阈值数组从锁定 evaluator 读取，不另凭印象生成 0.50:0.05:0.95。本地 p6a 常量为 0.50–0.90，最终以实际运行 evaluator 配置为准并记录。[E14]

**实现方式**：沿 `OfficialMetricAccumulator` 追到实际官方 evaluator，增加不改变数值路径的可选 match/ignore/ambiguity trace 或窄包装。用其一对一分配、有效 GT、ignore、歧义组处理来得到每个 eligible GT/组的成功/失败，不能只在诊断中调用普通 Hungarian 后冒充官方分配。

- 普通非歧义 pair 的 overlap：相关阶段逐阶段 IoU，双空排除，单空为 0，判定严格 `>`。
- 计数 key：`reference_id/master_id/order_id/prefix_T/gt_id_or_group/tau/pool`；不同 tau/pool 不混分母。
- `published_failure` 的分母只包含该 tau 下 evaluator 认定 eligible 的 GT；excluded/ambiguous-unresolved 单独计数。
- `candidate_complete` 仍只是每个所需阶段有候选的必要条件。报告“候选齐全但发布失败事件占比”，不命名为关联错误率或理论 AP 上界。
- 旧 GT-ID-LAG1 保留为历史启发式诊断，不再凭其退步关闭训练；不实现新的昂贵 oracle 搜索。

本轮只重算受此修复影响的 CAL E1 表和门槛解释。无论它最终支持哪种关联诊断，S/Q/A/R 的规定实验都继续。

### 4.2 原生 FH 与 D0 网络路径

先执行一个 CAL 的真实 T2 和一个 T5 prefix，使用固定 R1、原生 ReScene forward 和原生官方后处理。随后完成 CAL/SEL baseline：FH-R1-native、D0-R1。

D0 的当前网络 producer 要与旧 supplement 来源对应，而非把旧 B4/base 的不同预测混入。先对一个真实序列进行 cached D0 和 live D0 的 mask/class/score/lineage 比较。若不等价，定位 loader/seed/postprocess/config 并修实际差异；报告中保留旧缓存结果为历史行，不拿它冒充 live baseline。

原生 FH 每个 prefix 的输入为 `[X1,…,Xt]`，采用 native output policy；D0 为 pair-window/lag1/mean。最终同时保留政策匹配补充分析与原生系统主对照，不能把 FH-lag1 当 FH-native。

**禁止**：确认函数预先写死 `native_fh_tmap=None`；因为 cache root 缺失直接输出不可运行；只运行一段 import 就称为 GPU 前向完成。

### 4.3 当前诊断 panel

从 CAL 每个 reference 按 hash 顺序取首个可用序列，对固定 R1 收集查询覆盖、最早 attention 可达率、各阶段最佳候选 IoU；只做这一轮，不重复历史全部诊断。

可附加 whole-superpoint 上限：对每 GT 的每 segment 统计 `a_s=GT内点数`、`b_s=GT外点数`，按 `a_s/b_s` 降序扫描前缀，计算 `sum(a_s)/(GT总点数+sum(b_s))` 的最大值；`b_s=0,a_s>0` 置最前，`a_s=0` 不选。这是单 GT 可组合整超点的 IoU 上限，不是全系统 AP 上界，不能忽略各 GT 竞争后称为全局 oracle。

该 panel 用于解释后续效果，不作为关闭规定训练的前置门槛。

---

## 5. S-BAL / S-WORST：阶段 mask 损失

### 5.1 不变项

R1 的 matcher、分类/变化/对比学习损失、EOS、aux layers 和 raw-sum reducer 保持原样。新逻辑仅替换现有 `loss_mask`、`loss_dice` 的内部计算；不要在返回字典额外加一个可求导 `loss_stage_total`，否则 raw-sum 会重复计入。

`stage_debug`、per-stage IoU、梯度统计通过独立日志结构导出。不要修改公开日志键的物理含义后继续沿用旧列名。

### 5.2 真正的阶段对应

对 Hungarian 匹配后的第 i 个对象，第 t 个阶段的有效输出支持记为 `S_t`。训练在 segment 层则以与 `pred_masks` 同轴的 segment IDs 构造；在 point 层则以 point stage 构造。

使用 StageMeta 的 `segment_stage_ids`，或从真实 `temporal_stages + point2segment` 得到。每个 segment 内的有效 stage 最小值必须等于最大值；跨阶段混合则修复 segment namespace/offset，不能取均值、amax 掩盖混合，也不能把张量简单对半切。

对窗口中真实存在且有有效输入支持的每个阶段都计算损失，包括该 GT 在本阶段不存在、目标 mask 全零的情况。否则会奖励 ghost。只有阶段本身没有有效监督支持时才排除，并记录计数。

### 5.3 数学定义

令 `p_is = sigmoid(z_is)`、`y_is ∈ {0,1}`。

```text
b_it = mean_{s∈S_t} BCEWithLogits(z_is, y_is)
d_it = 1 - (2*sum_{s∈S_t}(p_is*y_is)+1)
             /(sum_{s∈S_t}(p_is)+sum_{s∈S_t}(y_is)+1)
```

这里平滑常数 1 与现有 Dice helper 对齐；如果实际 helper 不同，用其真实数值并在 config 写明，不默改 R1 helper。

S-BAL：`A(x_i) = mean_t(x_it)`。

S-WORST：分别对 BCE 与 Dice 使用：

```text
A(x_i) = (1-alpha)*mean_t(x_it)
         + alpha*beta*(logsumexp_t(x_it/beta)-log(n_valid_stages))
alpha = 0.5
beta  = 0.25
```

使用稳定 `torch.logsumexp`；不是先将 BCE/Dice 求和再重复输出两次。`loss_mask = sum_b mean_i A(b_i)`，`loss_dice = sum_b mean_i A(d_i)`，保持原 batch/匹配实例归一的外层口径；匹配实例为空时返回与输出图相连的零，而非 `stack([])`。

- 单阶段时两者退化为原逐实例 BCE/Dice（相同采样下应一致）。
- `mask_loss_mode=legacy` 必须走原代码路径，作为禁用等价基线。
- S-BAL 与 S-WORST 使用相同点/segment 采样。若原 `num_points!=-1`，在各阶段按原采样比例无放回取样，至少保留 1 个有效支持；不使用 GT 正负来决定采样位置。比例与输入坐标单位均沿用 R1。
- aux prediction 的每个 mask loss 也使用同一模式与 stage 对应，不只改最后一层。
- 本轮 alpha/beta 固定，不开展阈值网格。不因为某个 T 不好就给它专设 loss 权重。

### 5.4 必须输出

每个验证 checkpoint：官方 t-mAP/T、各阶段候选覆盖、非歧义 GT 的 worst-stage IoU 分布、单空阶段错误支持量。后三项是解释性诊断，选模仍按第 9 节官方指标。

---

## 6. Q-SEM / A-OPEN：查询位置与早期注意力

### 6.1 Q-SEM：先训练小 scorer，再固定 scorer 训练查询位置方案

不要直接把未经训练的随机分类头用作 hard top-k；也不要用测试 GT 选查询。分为两个可复现步骤：

**Scorer 数据与训练**

1. 从 TRAIN 中有合法pair的reference按 `SHA256('pgv1-scorer:'+reference_id)` 排序；各reference内部按 `SHA256('pgv1-scorer:'+reference_id+':'+pair_id)` 排pair。按reference轮转各取1个尚未使用pair，直到64个或耗尽。这样先覆盖reference，再补同reference的pair；不足64用全部，实际可用reference少于8时Q-SEM标 `BLOCKED_TRAIN_COVERAGE`，其余臂继续。
2. 冻结 R1、eval 模式真实前向，保存每 segment 的 128 维 mask feature、stage、坐标和仅供训练的监督。模型特征来自 `aggregate_features()`，不得打开读取 `x.gt_targets` 的 `save_segment_info` 推理通道。
3. 二分类 scorer：`LayerNorm(128) → Linear(128,64) → GELU → Linear(64,1)`。标签来自训练注释：已知前景 thing=1、已知 stuff/background=0、未知/未标注/ignore 排除。按实际 dataset 类映射生成 mask，不把“所有不在 GT 实例里的点”一律当负例。
4. segment 标签为其有效已标注点的前景比例，BCE soft target；训练每 batch 1024 个 segments，先均匀抽 reference，再抽 segment。固定 500 updates、AdamW `lr=1e-3`、weight_decay `1e-4`、seed45，不调参选 checkpoint，使用 update500。
5. scorer 训练结束冻结权重；感知 Q-SEM 训练期间 scorer 对当前 mask features 打分，但 scorer 本身不更新。打分和选索引不使用梯度；其它原 R1 可训练参数照常学习。固定 scorer 的额外训练预算单独报告，不能称为“零训练成本”。

**选点算法**

输入为每个真实阶段的 segment features、segment mean coordinates、stage_id。空间 FPS 仅使用归一化 xyz；最终 query positional encoding 保留原 ReScene 的完整位置/时间坐标及原归一方式。不能将 voxel index 坐标当 raw-coordinate 输入原 positional encoding；从真实坐标通过 p2s 聚合得到坐标中心。[E05][E13]

- 总 Q=100，不改 D=128、K=100。
- t 个实际阶段先分配 `floor(100/t)` 个查询，余数按阶段索引从小到大依次补 1；任何 stage 无候选时将其配额转给按索引排序的有候选 stage，循环分配。
- 每 stage 配额 q：`q_sem=floor(0.75*q)`、`q_explore=q-q_sem`。
- 按 scorer 概率选出 `min(S_t, 4*q_sem)` 个语义候选，概率相同用 canonical segment key 排序。
- 在语义候选上做确定性 xyz-FPS，选 q_sem 个；再从该阶段剩余候选选 q_explore 个最大化与已选点最小距离的探索点。初始 FPS 点取 scorer 最高/稳定 key 最小者，余下 argmax 平局以稳定 key 处理。
- 候选不足时先取全部不重复候选，再按已选稳定顺序循环重复以达到 Q；记录重复数。不可通过删除场景规避。
- 本版 query content 仍为零，位置编码与 R1一致；`use_np_features=false`。因此不是旧 feature-seeded FPS 的再次运行。

Q-SEM 是“额外小 scorer + 查询位置策略”的方法包；不把相对 C0 的全部差值声称是某一个单独超参数的因果效应。无需复制 LaSSM 的 SSM、400/500 queries 或外部 CUDA 模块。

### 6.2 A-OPEN：准确地只开放第一次 cross-attention

在 `models/rescene.py::forward()` 的第一个 execution stage（全局 `execution_stage_idx==0`，不是每个 shared decoder 重复一次）令该次 `memory_mask=None` 或全 False。其余保持：

- `memory_key_padding_mask` 原样传递。
- 相同 feature sampling 数量、positional encoding、cross/self attention 与 FFN。
- execution stage≥1 完全恢复原 mask attention。
- 不开放全部层，不打开 temporal_masking，不改注意力温度，不更改 query 初始化。

记录实际开放调用次数=每 forward 1 次、padding 仍生效，以及有效支持量。不要把“调用了 hook”当作行为已经改变的证据。

---

## 7. C0 与感知候选的真实训练

### 7.1 初始化与优化器

所有感知臂 `C0/S-BAL/S-WORST/Q-SEM/A-OPEN` 从同一个 R1 checkpoint 加载模型权重。新实验使用新 optimizer/scheduler；不续接 R1 已到 epoch390 的 OneCycle 状态，也不从 epoch0 全部重训。

这里“继续训练”指 warm-start 权重，不是沿用旧 optimizer。一次生成并校验公共初始模型状态，保留旧键严格匹配；只允许新增 scorer/配置扩展相关键，不允许用全局 `strict=False` 吞掉 backbone 权重缺失。

新增训练计划（本轮规定，不是历史超参）：

```text
train_seed                = 45
optimizer                 = AdamW
lr_existing_trainable     = 5e-5
betas                     = (0.9, 0.999)
eps                       = 1e-8
weight_decay              = 0.01
precision                 = FP32
frozen_encoder_policy     = 继承实际 R1 resolved config
total_optimizer_updates   = 3000
warmup_updates            = 150
minimum_lr_fraction       = 0.1
grad_clip                 = 1.0
pilot_endpoint            = 750
CAL_eval_updates          = [0, 250, 750, 1500, 2250, 3000]
```

LR 调度明确为 150-step 线性 warmup + 余下 cosine 到 0.1×峰值；从 update0 就按总3000步构造，不为 pilot 建750步完整退火调度后再重新启动。

C0、S/Q/A 共用训练 sample stream、增强、effective batch、冻结策略和所有其它 loss。各臂 GPU batch 若因显存限制调整，应先定位真正 OOM；最多采用一次同 cohort 的共同 microbatch 调整，并对受影响同预算对照从最近一致边界恢复。不要为一个候选单独降低分辨率、裁掉困难场景或增加数据。

### 7.2 必须完成的第一批

在技术可运行且预算允许时，五个臂都运行到 update750，不能因为 S-BAL 先通过就不开发/不运行 Q-SEM、A-OPEN。

C0 的 update0 使用公共 R1 模型；由于训练配置改变人口过滤而原始权重不变，baseline 仍是同一 R1 在固定 CAL 评测的结果。每臂 save `last.ckpt` 及规定 eval updates；恢复包含 optimizer、scheduler、RNG、sample cursor。

**训练证据最低要求**：实际 optimizer update 数、可训练参数列表、2个必要 step 的非零梯度/参数变化、train loss 曲线、验证输出、checkpoint SHA。不能用构造模型成功、loss.backward 无异常或者 0 GPU-hours 来声称模型训练完成。

### 7.3 旧训练器如何复用

`train_task_memory.py` 固定了旧 variants/预算与 M3 parents，不能靠把新名字映射成 Q-TALA 复用其语义。新 runner 复用资产定位、Hydra composition、R1 load、CSV logger、checkpoint 和 schedule 组件；感知训练使用 `InstanceSegmentation` 的正常 pair 监督，stage metadata 以窄接口传给 criterion。[E09][E11][E15]

---

## 8. R-REFINE：先选感知父模型，再训练一次局部修复器

### 8.1 顺序与父模型

先按第 9 节选择并冻结感知父模型 P*（可能仍为原 R1 或 C0）。R-REFINE 只在 P* 上训练一次，不同时在多个旧/新 backbone 上训练 refiner。

R-REFINE 必须实际开发并做训练/验证；只有真实缺少训练数据/软导出无法修复/预算耗尽才标阻塞。不能因为旧 E1 未支持关联空间就跳过。

### 8.2 正确的因果窗口

收到扫描 t 时，扫描 t-1 在 `(t-2,t-1)` 与 `(t-1,t)` 中具有 old/new 观察。t=2 时 old 是首个单扫描 `(1)`、new 是 `(1,2)`；只有已经到达的扫描可用于修复。

- 只修复正在 lag1 提交的同一个物理扫描。
- 最后一个当前扫描没有未来窗口，保持原 new 输出。
- 固定 D0 的 group/identity assignment；从不使用修复后的 mask 更新 D0 association/state evidence。它只改变发布/归档中的 mask 内容，因此相同 P* 的 parent 与 refiner 关联事件必须一致。
- same query_id 跨窗口不代表同实体。先通过 P* 下 D0 的 logical_id/generation 与旧 occurrence lineage 找对应，再在同 scan_id 内按 canonical vertex ID 对齐。 配对key固定为 `(scan_id, logical_id, generation, source_class_id)`，再限定old属于紧邻的上一窗口；候选保留/filter后恰有1个old才配对，0个或多于1个均回退new。不要为提高配对率在多old中用GT或随意挑分数最高者。
- 没有唯一合法 old 对应（出生、被挤出、类别/分组歧义）时直接原样保留 new，不按 GT 找 old。

### 8.3 新的软证据 sidecar

现有 bool-cache 不足以训练修复器。扩展 live producer 导出：

```text
producer_checkpoint_sha256 / resolved_config_sha256 / source_commit
reference_id / episode_id / order_id / absolute_stage / source_window
scan_id / canonical_vertex_ids / point2segment / segment_stage_ids
source_query_id / source_class_id / candidate_index
raw mask logits 或原生未二值化 probability（明确记录 representation）
128D mask features（无 GT） / 原 official score / 原 official bool mask
```

训练标签另存于训练专用 payload；推理 payload 不含 gt_ids/gt_masks/gt_classes。不要从 bool 的 0/1 做 logit 变换当“原始 soft evidence”。

导出必须复用官方 `_get_batch_masks/_get_full_res_mask` 的真实 inverse/p2s 路径以及 filter 后 candidate lineage，既不跳过官方 top-k/filter，也不重新排序候选而丢失来源。若保留的是概率，使用 `logit(clamp(p,1e-4,1-1e-4))` 作为修复输入；该转换只适用于真实概率，不适用于 bool。

为避免两次扫描超点划分不同导致错误对应：以new occurrence的segment partition为目标，将old端真实logits先sigmoid成point-wise probability，再按vertex ID对齐；对每个new segment所对应的full-res原始vertices求概率算术平均，得到p_old，z_old=logit(clamp(p_old,1e-4,1-1e-4))。new端z_new直接取原FP32 segment logit，p_new=sigmoid(z_new)，不从聚合概率反推new logit。full-res vertex→new低分辨率segment的映射由 `new_low_point2segment[new_voxel_inverse]` 得到，不与full-res评测segment编号混用。不能直接zip两份segment数组。两次增强坐标变换必须相同或先转换回共同坐标，frame/transform ID写入cache key。

缓存仅保留两窗口需要的 soft evidence；历史 binary 输出可以归档但不得无限保留所有 soft features。

### 8.4 修复网络与训练

候选 i、new segment s 的输入：

```text
LayerNorm(F_new[s])                         # 128
clip(z_old[s],-8,8), clip(z_new[s],-8,8)     # 2
clip(z_old[s]-z_new[s],-8,8)                 # 1
p_old[s], p_new[s]                          # 2
old_score, new_score                        # 2
总输入维度 = 135
MLP = Linear(135,64) → GELU → Linear(64,1)
delta = 2*tanh(MLP(input))
z_refined = z_new + delta
```

其中F_new的归一化采用不带可训练affine的LayerNorm（eps=1e-5）；只训练两个Linear。最后Linear的weight/bias全零初始化。P*、scorer、D0完全冻结，只训练refiner。默认 `AdamW lr=1e-3, weight_decay=1e-4, seed45, 1500 updates`；warmup75、cosine到0.1×峰值。CAL 在 updates `[0,500,1000,1500]` 评价。

训练缓存选择：在TRAIN中按 `SHA256('pgv1-refiner:'+reference_id)` 排序取前32个至少有3次扫描的reference。沿用已有native master构建器产生每reference的canonical真实扫描序列；多master时按sequence_id字典序取首个。截取前 `min(5,真实扫描数)` 个不同扫描，再取其canonical与reverse两种顺序，禁止用数组复制补阶段。不足32用全部，少于8个reference标真实数据不足，不编造episode。cache生成时父模型eval、关闭随机增强；训练refiner不额外改变已缓存几何。这些顺序是实验输入顺序，不在缺采集时间戳时声称真实时间顺序。

训练minibatch为16个可配对候选：均匀抽reference→episode→candidate；每候选从该new扫描的全部有效segments均匀无放回抽最多2048个，不限于预测为前景的位置。采样不按GT类别再分配；“覆盖前景、边界和背景”表示候选全集不裁掉这些区域，不保证每个随机batch都有三类点。数据少时允许候选有放回，不复制物理扫描。

训练目标：在 TRAIN 用未修复 new candidates 与 eligible GT 做类别相容的一对一匹配，IoU≥0.1 才赋对应 GT；未匹配 prediction 用 empty mask 作为 FP 抑制目标。歧义组无法稳定赋值的训练候选跳过并记录比例，不能将其标成确定负例。GT 只用于离线标签。每个候选损失为 `BCE + Dice + 0.01*mean(delta^2)`，候选均权。

### 8.5 二值化、分数和 no-op 一致性

阈值使用原 official mask 决策阈值；禁止为 T2/T5 分别设阈值。原官方 score、class、候选数量、排序与 D0 identity 原样保留，采用已有 occurrence mean reducer。

修复可在同一new扫描的全部有效segment上增减该候选支持，不限于原bool前景，否则无法补回漏分割；不能新造候选、加GT mask或改变原置信度。

固定版本的具体后处理顺序已经核实：[E21]

```text
new segment logits（按已保留source_query_id取列）
 → 用low-resolution point2segment展开到低分辨率点
 → 严格 logits > 0 得到低分辨率0/1 mask
 → 用inverse_map展开到full-resolution点
 → 若eval_on_segments=true：按full-resolution point2segment求0/1均值
 → 严格均值 > 0.5，多数决策后展开回点
 → 只替换原已保留candidate的该扫描mask
```

`is_heatmap=True` 分支只进行inverse映射，并不做上述bool多数决策；因此不能对full-res heatmap求均值再直接阈值，假定它与原bool相同。refiner应在第一行的new segment logits上加残差，再复用原后续映射。以概率保存原soft证据时，精确零阈值附近的数值必须保留；默认new端保存FP32原logits及上述映射，old端可保留真实概率用于特征。不得用clamp/logit往返误差改掉new端零残差输出。

原候选的top-k/filter/score计算先按**未修复输出**执行并冻结候选集合；修复后不重新调用mask相关重筛选。零残差必须重新经过以上真实映射得到相同bool，不能仅检测delta为零就返回旧bool掩盖映射错误。新开关关闭时则允许直接走原路径。

`refiner_enabled=false` 必须直接使用原 bool；零初始化启用时也必须在一个真实序列上证明与 parent 的 bool/class/score/identity 一致。若不一致，先修 representation/mapping，不能把额外二值化变化计为 refiner 增益。

CAL 选定 refiner checkpoint 后，用 SEL 比较 `P*-D0` 与 `P*-D0+R-REFINE`。门槛见第9节。未晋级也保留代码、训练结果与负例解释，最终系统回退到 P*。

---

## 9. 完整的训练晋级、选模、复核与回退规则

所有 AP 存为 `[0,1]`。`0.001=0.1pp`，`0.005=0.5pp`；表格转换百分数时只乘100一次。

### 9.1 统一指标与排名

对固定 population 和方法 m，使用 official accumulator 汇总的四个 pooled t-mAP `A_m(T)`。不把逐scene AP平均冒充 pooled AP。

相对固定 D0-R1 的差值 `d_T=A_m(T)-A_D0(T)`：

```text
S_mean = (d2+d3+d4+d5)/4
S_long = (d4+d5)/2
S_min  = min(d2,d3,d4,d5)
```

排序为 `S_mean` 降序 → `S_long` 降序 → `S_min` 降序 → 新增参数更少 → optimizer update 更早 → method_id 字典序。比较容差统一1e-6。主排序不再先排 S_min，避免只因一个T略有噪声就压掉有总体收益的候选；性能保护另用明确门槛。

### 9.2 第一轮 pilot → full

1. 五个臂均到750（真正技术失败的臂除外）。pilot 的候选点为250、750；每架构按 CAL 排名取一个点用于 pilot 决策，但 full 必须从其 **last/update750** 恢复，不从已选更早 checkpoint 改写训练轨迹。
2. C0 必须继续到3000，作为所有 full 臂的同预算对照。
3. 四个新感知臂中，满足 `CAL S_mean>=-0.01 且 S_min>=-0.02` 的按上述规则最多选2个继续到3000。此处宽松门槛用于避免过早关闭学习，不是正式胜出门槛。
4. 若无臂满足，且最佳臂 `S_mean>=-0.03`，只允许该1臂继续；若所有臂均低于该值，停止新感知延长，只完成 C0 和后续 refiner。
5. 不因为 E1、query覆盖诊断或某个早期 loss不漂亮额外关闭臂；也不在未入选臂上跑第三套配置。

### 9.3 full 的 checkpoint 选择

每个已完成full的架构，候选 checkpoint 为 `[0,250,750,1500,2250,3000]` 中实际存在的规定点。只在 CAL 按统一排名选择1个；C0也同样选择，update0代表R1。保存 `CAL_CHECKPOINT_SELECTION.json`，然后才评价 SEL。

对候选所在 update k，还要与 C0 update k 做配对差值，作为新增方法因果归因表；正式选模使用更强的 `C0-best-CAL`，不能专挑同预算继续训练的差点来衬托新模块。

只完成pilot、没有full资格的臂不参与最终部署选模，但其pilot结果必须进入汇总表。预算取消延长或外部故障导致未到3000的臂，同样不冒充full候选。

update0语义必须分清：C0/S-BAL/S-WORST的update0推理与R1相同；A-OPEN或已训练scorer的Q-SEM在update0即可改变推理。后两者可以作为该已完成full架构的CAL选择结果，但报告为“selected_adaptation_updates=0”，将scorer已完成的500步单列，不能声称选择到经过3000步适配的模型。C0若选update0，方法身份归并为R1，不能记录CONTINUATION_ONLY。

### 9.4 SEL 选择感知父模型 P*

固定 CAL 检查点后，计算各候选、C0-best、D0-R1 的 SEL 表。

感知部署基础门槛，相对固定D0-R1：

```text
S_min >= -0.001
S_mean >= 0.005
S_long >= 0
```

新方法要获得 `NEW_PERCEPTION_SELECTED`，还须相对 `C0-best-CAL` 满足：四T平均增益≥0.002，且每T退步不超过0.002。其余身份指标（wrong merge、wrong reactivation、ID-switch等）列出但不作为额外隐藏否决门槛。

- 多个新方法通过：按 SEL 统一排名选择唯一 P*。
- 无新方法通过，但 C0-best 通过基础门槛：P*=C0-best，状态 `CONTINUATION_ONLY`；不能称新模块贡献。
- 否则 P*=R1，保留D0，状态 `KEEP_R1`。

不组合两个感知臂，不进行二次SEL调参；本轮唯一可在 P* 上叠加的模块为 R-REFINE。

### 9.5 SEL 选择 refiner

CAL 为 R-REFINE 选择 `[0,500,1000,1500]` 中唯一 checkpoint，规则同上但差值父系统改为 `P*-D0`。更新0为严格no-op。

在 SEL 相对 `P*-D0`，须同时满足 `S_min>=-0.001`、`S_mean>=0.003`、`S_long>=0` 才启用。否则最终使用P*不带refiner。不要改分数、阈值或关联来挽救未晋级refiner。

### 9.6 两级冻结与第二种子

感知选择后写 `PERCEPTION_LOCK.json`；refiner选择后写 `FINAL_LOCK.json`。后者包括完整 recipe、所有weight/config/数据hash、选中的update、输出政策与判定阈值。

预算允许时，对已锁定的新增训练 recipe 做 seed46适配训练复核，且不重新选超参/检查点：

- 新感知入选：seed46从相同R1初始化训练到seed45锁定的update；C0-seed46也到相同步数作配对。Q-SEM沿用同一个固定scorer，明确这是固定scorer条件下的适配训练重复。
- P*=C0：只做C0-seed46，不重复训练一份完全相同对照。
- P*=R1且无refiner：没有新增训练recipe需要第二种子，状态为`NOT_APPLICABLE`。Q-SEM/A-OPEN若选择update0，也没有待复核的感知适配训练；记录其感知适配复核`NOT_APPLICABLE`。本轮不通过反复评估同一固定scorer/同一权重伪造训练重复。
- refiner入选：在对应seed46父模型重新生成训练soft证据，refiner-seed46到已锁定update；预算不够则只报告已完成的感知复核，不把它说成整个pipeline复核。

第二种子只验证，不换冠军、不改训练长度、不在PB后回CAL挑其它checkpoint。配对比较对象固定：新感知seed46对C0-seed46同update；C0-seed46对固定R1；refiner-seed46对其对应未修复parent；完整pipeline另对固定D0-R1。

逐组件写 `replication_scope`、完成范围与状态。`SUPPORTED`要求该已复核部分在SEL的四T平均相对上述配对对象为正，且单T退步均不超过0.002；否则`MIXED`。总状态只有所有实际选中且需要训练复核的组件都完整支持时为SUPPORTED；部分未运行时总状态按NOT_RUN_BUDGET或BLOCKED，并保留已支持组件行。全部无新增待复核训练时为NOT_APPLICABLE。不用“3个评估seed”冒充3次独立训练。

### 9.7 预算不足的确定性顺序

预算不足时依次取消：第二种子复核 → 第二个full晋级臂的延长。不能取消 C0 配对对照、seed45唯一主候选、R-REFINE 的规定基本实验及最终原生确认来换取更多候选。

如果按实测吞吐连必做内容也无法在剩余预算完成：保留已完成seed45阶段，保存checkpoint和确切未完成项，`execution_status=PARTIAL_WITH_BLOCKERS`。不能私自增加预算、声称全部完成或利用不同步数对照晋级。最终报告与GitHub发布仍必须执行。

确认/资源预算单独预留，训练不得挪用；不得因测试和文档反复检查耗尽GPU预算。

---

## 10. 最终确认：真正回答是否超过 ReScene

### 10.1 FINAL_LOCK 后的必需方法

在同一PB population上测量以下方法，相同者去重但保留方法别名：

| 方法 | 定义 |
|---|---|
| FH-R1-native | 固定R1，全prefix输入，native官方输出 |
| D0-R1 | 固定R1，真实pair producer，原D0/lag1/mean |
| C0-best-D0 | CAL锁定的C0，原D0/lag1/mean |
| FH-P*-native | 锁定感知P*，全prefix输入，native官方输出 |
| P*-D0 | 锁定感知P*，原D0/lag1/mean，不带refiner |
| FINAL | P*-D0以及按SEL规则锁定的refiner开关 |

同一P*下 FH-P* 与 P*-D0的差值才有助于解释短窗口机制本身；相对FH-R1的胜出不能全部归因于memory。对refiner相对P*-D0另列差值。可做同输出policy的补充行，但不得替换native主行。

LOCAL-T2使用原官方式population，测R1-native、C0-native、P*-native；refiner属于额外两次观察/lag1系统，不在LOCAL-T2表冒充完全相同的native单次前向结果。若要附加refinerLOCAL-T2结果，单列实际信息/前向预算，标`SYSTEM_EXTENSION`。

ADDITIONAL使用其真实可用的T2–T5前缀，parent/FINAL/FH-R1共用同一子population；每T列reference数与序列数。不拿不同T人口差异当纯长度退化效果。

### 10.2 每条结果的最小记录

```text
method_id, population_id, population_hash,
reference_id, master_id, order_id, prefix_T, observed_scan_ids,
checkpoint_sha256, config_sha256, source_commit, eval_seed,
producer_id, output_policy, reducer,
prediction_artifact_key, status, failure_reason
```

PB汇总必须含 `expected_units=129`、`completed_units`、`expected_prefixes=516`、`completed_prefixes`，以及实际去重物理前向数；LOCAL/ADDITIONAL使用各自manifest算出的分母，不套用129/516。缓存中的115个独立物理文件或645个历史stage引用不能直接替代129/516这两个覆盖分母。[E09]

缓存可用于评测复用，但前提是同checkpoint/config/输入point order/postprocess/seed；改变感知模型后必须新前向，不能给旧cache改producer名称。只对完全同一内容键去重。

### 10.3 官方 evaluator 与不完整结果

正式 t-mAP/AP/REC必须由锁定官方评价链产生。**不得**：丢掉失败单元后仍声称全PB；以NaN/None当0；把OOM的FH当负无穷；把mean(scene_AP)当pooled AP；用t-REC/ID-switch替代t-mAP胜出。

FH真实T5若OOM：一次清理本进程缓存/隔离设备后，以同输入同精度重试；仍失败记录设备、peak、异常、单元ID。不得自动裁点、降分辨率后沿用相同方法名。共同可完成子集可以做明确命名的描述性比较；全PB `pb_all_t_vs_native_fh=null`，并附`comparison_status=INCOMPLETE_NATIVE_COVERAGE`，不能宣称全T胜出。

PB结果不用于返回开发阶段改方法；所有锁定候选无论成败都报告。若发现实际代码正确性bug，修复并只重跑受影响结果，但将被影响的选择结果作废、记录暴露事实，不能把修复后PB称为未触碰测试。

### 10.4 默认部署推荐与科研结论

- `pb_all_t_vs_native_fh=true`且FINAL相对D0四T平均为正、各T最多退步0.001：可推荐本轮FINAL，并如实列出未严格超过D0的T。
- 不满足：保留D0作为默认部署版本，仍发布锁定候选和完整负结果；不是从PB挑另一候选取代它。
- `pb_all_t_vs_d0`仍按“四T都严格更高”独立判断，不因部署容差放宽而改成true。
- 对外宣称重复训练稳定，需要相应seed复核证据；无复核可报告本轮测得增益，但不能扩大为普遍稳定结论。

---

## 11. 真实成本测量与统计范围

复用 `profile_task_memory.py`，增加所需模型/输出适配即可，不另造资源评价平台。

方法至少为FH-R1-native、D0-R1、FINAL；P*不等于R1时另测FH-P*-native。顺序运行在同一张空闲A40，相同精度与实际输入。每个PB reference取首个canonical master，共6个固定序列；一遍完整序列warmup，三遍完整序列实测。这个样本量报告median与范围，不报告有统计稳定性暗示的p99。

每个T分开记录：

- `network_forward_ms`：CUDA event或同步后计时，包含该方法实际需要的encoder/decoder/scorer/refiner网络调用。
- `state_update_ms`：association/状态维护，不含读取已有预测缓存。
- `materialize_ms`：输出重建、pack/unpack及必要CPU传输。
- `end_to_end_ms`：真实输入准备→前向→状态→可用输出，明确是否含磁盘冷读；所有方法一致。
- `prefix_cumulative_ms`：在线从第1步实际处理到T的累积，不把最后一步时间叫整序列时间。
- `peak_allocated_bytes`、`peak_reserved_bytes`、CPU RSS、resident state bytes、soft-buffer bytes、历史输出archive bytes分别列。

计时前后正确cuda同步、reset peak；每次测量使用相同前缀起始状态。若为了测量复制状态，复制成本不得偷偷混入一种方法却从另一方法移除；写清scope。

不能因为状态K固定就称系统全部内存O(1)：累计历史输出可能增长。部署时C0/新感知producer、scorer、refiner实际需要的网络调用不能漏计；离线训练/训练缓存生成的GPU-hours在研发成本表单列，不加进单次部署延迟，也不能用一句“训练时已预计算”省略部署仍必须执行的特征提取。未经等价验证不采用FH encoder cache加速；缓存重放仅在独立`CACHE_REPLAY_ONLY`列中出现。

统计：以物理reference为单位输出每T配对差值。可做固定seed45的1000次reference-block bootstrap，复用存储的官方metric sufficient state重聚合，不重复GPU前向；需完整重算pooled AP，不能将等权reference均值CI冒充pooled CI。实现不方便时只报告逐reference表与范围，省掉bootstrap，不另写大型统计框架。

---

## 12. 最小充分测试：禁止过度安全测试和平台化审计

测试只保护本轮可能影响结论的行为。禁止把本轮变成通用安全审计、输入fuzzing、数百个防御性schema测试、反复全量回归、渗透扫描或循环哈希验收。

### 12.1 需要覆盖的行为（约12–16个小用例，不是额外平台）

| 必须覆盖 | 验证内容 |
|---|---|
| Temporal判定 | `[1,1,0]`不能按concat成功；双空/单空；严格大于；tau变化会改变事件；一组官方歧义trace一致 |
| Stage loss | T1与原损失一致；阶段支持量不同时确实均衡；空GT阶段受惩罚；梯度有限；legacy关闭等价 |
| Query/attention | Q固定100、平局/少点行为确定；GT不作为输入；只开放第一次且padding保留 |
| Soft mapping/refiner | 非恒等vertex permutation正确；不同segment partition正确；禁止同query_id假匹配；zero/no-op等价；无old/末帧回退 |
| 运行链 | 1次真实两步训练的参数变化；1个真实序列live推理/official evaluator；同一小run恢复不重置schedule；pipeline结果文件可重读 |
| 发布 | 新交接/代码/结果纳入git；远端SHA等于本地；release文件清单与字节/校验一致 |

能用现有测试覆盖的直接复用，不为凑数字新建重复测试。实际用例数可以因参数化超过16，但不扩张测试的语义范围。

### 12.2 执行频率与上限

- 开发过程中只跑所修改模块的目标测试；最终跑一次本轮相关回归集合。
- Ruff只检查本轮修改Python文件，`git diff --check`一次。旧无关lint问题记录但不全面修复。
- 测试/审核总CPU墙钟目标≤30分钟；GPU smoke实际占用≤1 GPU-hour，计入第1节4GPU-hour余量。
- 不在相同代码、输入与配置上反复重试同一种失败；每次复验必须对应已定位原因和实际修改，只跑相关用例。必要的代码调试不机械限制为“只能修一次”，但计入上述预算，耗尽时保留最小失败证据并标该支线阻塞。真实数值/映射错误未修复时，不得为了通过测试上限而继续产生有效性结论。
- 不要求每个epoch/每个commit重新hash全部权重、数据、artifact；结果写入时hash即可，发布时只验证新增交付文件。
- 文档、审计、测试不能代替真正的训练、确认和发布，也不能因“想把测试再做完善一点”无限延期实验。

---

## 13. CLI、阶段安排与恢复

以下为**必须实现的新CLI合同**；不声称旧仓库已支持这些命令。先完成最小runner及绑定，再运行。

所有子命令接受 `--config`、`--external-root`；`run`接受`--through`和`--resume`。阶段状态持久化至本轮`RUN_STATE.json`。

```bash
python -m scripts.perception_gain_campaign bootstrap \
  --config configs/perception_gain_v1.yaml \
  --external-root "$PERSIST4D_PERCEPTION_RUN_ROOT"

python -m scripts.perception_gain_campaign run \
  --config configs/perception_gain_v1.yaml \
  --external-root "$PERSIST4D_PERCEPTION_RUN_ROOT" \
  --through publish --resume

python -m scripts.perception_gain_campaign status \
  --config configs/perception_gain_v1.yaml \
  --external-root "$PERSIST4D_PERCEPTION_RUN_ROOT"
```

`--through`可用值与执行顺序固定为：

```text
bind → foundation → scorer → perception_pilot → perception_full
     → perception_select → refinement → final_lock → replication
     → confirm → profile → report → publish
```

| 阶段 | 实际工作 | 结束产物/后继 |
|---|---|---|
| bind | 读源码、定位资产、数据角色、配置冻结 | CODE_BINDINGS/RUN_CONFIG/DATA_ROLES |
| foundation | E1修复、live D0/native FH、小panel | F0表、真实执行日志；不控制感知臂授权 |
| scorer | 只训练Q-SEM固定scorer | scorer权重/来源；失败只影响Q |
| perception_pilot | 五臂750步 | CAL结果、2名以内晋级清单 |
| perception_full | C0+晋级臂3000 | CAL检查点选择 |
| perception_select | SEL按规则选P* | PERCEPTION_LOCK |
| refinement | P*软缓存、refiner1500、CAL/SEL | parent/refiner表 |
| final_lock | 锁定recipe/权重/政策 | FINAL_LOCK |
| replication | 预算内seed46复核 | 固定recipe复核表或明确未运行状态 |
| confirm | PB+LOCAL+ADDITIONAL真实确认 | 覆盖表/official metrics/逐reference结果 |
| profile | 同设备live profile | 原始测量CSV与summary |
| report | 从结果生成最终报告与交接 | FINAL_REPORT/HANDOFF/manifest |
| publish | 提交、push、release并验证 | 可访问GitHub链接和发布receipt |

runner可调用现有Python函数/脚本，不必自己重写trainer/evaluator。所有实际子进程命令、工作目录、PID/设备、exit code、输入/输出key写入精简执行日志。失败后不得仅写一段“建议以后补跑”而跳过仍可执行的阶段。

恢复优先复用已有同key缓存和checkpoint；`--resume`在配置/父检查点不匹配时标不同run，不覆写旧证据。耗时训练正常保存last，外部中断后继续当前步数；不把恢复开销说成新实验样本。

某阶段受阻不等于全流程停止：scorer阻塞继续S/A；refiner阻塞继续P*确认；FH个别prefix阻塞继续其余前向并发布比较不完整状态；GitHub权限失败保留完整本地交付并明确没有完成远端发布。

---

## 14. 必须交付的代码、结果和交接文件

代码写在本轮分支；轻量证据写入 `artifacts/perception_gain_v1/`。控制文件数量，不为每一个小检查建立新的“合同/签名/授权”文档。

```text
EXECUTION_INSTRUCTION.md           # 本文原文
RUN_CONFIG.json                   # 真实resolved配置、输入与协议绑定
DATA_ROLES.json                   # 本轮TRAIN/CAL/SEL/PB等及暴露标签
CODE_BINDINGS.md                   # 原代码→改动→实验的唯一映射表
RUN_STATE.json                    # 阶段、预算、失败和下一实际命令
EXECUTION_LOG.jsonl                # 精简实际调用和exit code
foundation/                       # 修正E1、baseline parity、当前panel
training/                         # scorer/各臂配置、曲线、检查点manifest
selection/                        # pilot/full/CAL/SEL与两次LOCK
confirmation/                     # LOCAL/PB/ADDITIONAL覆盖与官方指标
resources/                        # 原始计时和资源scope
FINAL_REPORT.md                   # 数值与结论
HANDOFF.md                        # 给下一位执行者的完整交接
ARTIFACT_MANIFEST.json             # 新增交付文件清单、SHA256、bytes
RELEASE_PLAN.json                 # 预定tag、待上传文件、外部大文件位置
```

`FINAL_REPORT.md`至少包含六张表：

1. 每个任务的实际开发/训练步数/运行状态，不把SKIPPED写成FAIL实验。
2. CAL/SEL各臂及其C0配对对照；每个架构只使用锁定checkpoint。
3. LOCAL-T2同population的R1/C0/P*结果，历史参考值分表。
4. PB各T的FH-R1/D0/C0/FH-P*/P*/FINAL，以及相对固定起点差值。
5. 每reference、每seed结果和ADDITIONAL人口数。
6. 真实成本、GPU/CPU预算、发布状态与实际阻塞。

`HANDOFF.md`用以下固定小节，不省略负结果：

```text
1. 仓库、起点SHA、工作分支、实验代码/结果提交SHA
2. 本轮目标与最终结论：哪项是实测、哪项仍未确认
3. 已改文件与函数、训练/推理数据流
4. 确认的R1初始化、最终模型/头的哈希、所需基础权重
5. 数据population、物理reference数量与历史暴露
6. 每个臂实际步数、checkpoint选择和淘汰理由
7. 最终锁定recipe与默认部署回退决定
8. LOCAL/PB/ADDITIONAL结果、相对R1与C0的差值
9. 当前native FH与真实profile完成程度
10. 测试命令与实际结果；没有运行的部分
11. 真实复现命令、环境、资产解析方法
12. 大文件release链接、manifest、取得基础权重的方法
13. 剩余阻塞：已执行命令、异常、最小下一动作
14. GitHub发布验证方法与release receipt位置
```

复现命令必须从实际执行日志生成，不保留 `/actual/path`、`TODO`、`<CHECKPOINT>` 占位符。公共命令使用可由资产manifest解析的环境变量；服务器绝对路径放在本地untracked locator中即可。新读者应能从manifest知道缺的是哪一种资产及其hash，不必猜目录。

`ARTIFACT_MANIFEST.json`不包含自己的hash，也不hash不断变化的执行状态产生递归依赖；只冻结实际交付快照。公共输出不提交凭据、原始扫描、完整GT标签或大缓存；只提交允许公开的预测索引/统计，必要的大结果包剥离GT字段。推理预测的大包可含vertex对应与预测mask/score，用用户自有合法GT复算；若官方metric sufficient state内嵌GT，不直接公开该state。

---

## 15. 实际同步并上传 GitHub：不是只给用户 git push 建议

### 15.1 上传范围与大小

用户已要求上传本轮代码、结果与交接，故本轮最后必须实际尝试推送，不再逐次询问确认。

- Git：代码、配置、测试、EXECUTION_INSTRUCTION、HANDOFF、FINAL_REPORT、完整小型CSV/JSON、结果manifest。单文件本轮内部上限20MiB；较大预测表压缩/分片或进入Release，不把数百MB checkpoint直接git add。
- GitHub Release：最终选中模型的部署bundle、refiner/scorer必要权重、C0部署权重，以及可复核的大型预测结果包。若新方法未入选，可额外上传最优探索候选的部署权重，明确`EXPLORATORY_NOT_SELECTED`，无需上传所有中间checkpoint。
- 部署bundle可以存相对固定R1的**完整替换参数键值与变化buffer**，配合base hash恢复；不能用有精度损失的浮点差分相加却声称bitwise还原。至少在一个已用过的真实panel验证重载输出相同；不再跑全PB。
- 原始R1/Concerto等第三方基础权重可保留现有获取方式和hash，不重复分发原始数据/未授权权重。本轮新增可公开训练参数必须实际上传，不仅留服务器路径。

GitHub普通Git限制及Release每asset<2GiB有官方文档依据。[R04][R05] 本轮统一按≤1GiB分片（若需要），并附分片顺序/整体SHA，避免临界值与大文件重传。

### 15.2 不产生“提交内嵌自己SHA”的循环

采用一次内容冻结和外部receipt：

1. 冻结代码/科学结果，创建实验提交A。
2. HANDOFF记录A、分支、预定release tag与复现命令，生成artifact manifest与RELEASE_PLAN；创建交接提交B。
3. B为本轮发布源码/结果快照。`git push -u origin "$BRANCH"`，比较 `git rev-parse HEAD` 与 `git ls-remote origin "refs/heads/$BRANCH"` 得到同一B。
4. 在B创建轻量或annotated tag，默认`persist4d-perception-gain-v1`。若该tag已用于无关内容，选择最小空闲`-r2`后缀，不移动既有tag或force push。
5. 用现有GitHub登录态和`gh release create/upload`，或现有授权API，创建prerelease并上传bundle/大结果。科学失败的本轮照样可发布，但release标题/说明必须写真实结果。
6. 最后产生 `PUBLICATION_RECEIPT.json`：记录B、远端分支SHA、tag目标B、release URL/ID、每个asset名称/字节/hash/URL、验证时间和状态。该receipt存本地run root并作为Release附件上传；不把含B的receipt再提交回B制造无限循环。

若git auth可用但没有Release写权限，仍push全部小型结果与交接，并记`CODE_ONLY`，明确哪些权重/结果未上传。不得把“本地打包完成”写成“已上传GitHub”。

### 15.3 发布验证只做必要步骤

- 远端分支SHA等于本地B，tag指向B。
- 通过GitHub读取HANDOFF和一份结果summary，确认本轮内容确实在远端。
- 列举Release assets，检查文件名、bytes与上传清单一致；平台返回digest时核对SHA；无digest则对bundle或小manifest下载回读一次。仅清单/字节检查时，receipt应写`verification_level=REMOTE_METADATA`，不虚称端到端hash已验证。
- 上传失败允许一次针对性修复重试；不得删除、覆盖无关分支/Release，也不得force push。

Git中提交B的RUN_STATE/HANDOFF是发布前冻结快照：以 `publication_phase=READY`、`publication_status=null` 指向预定receipt位置，不预写尚未发生的VERIFIED。最终发布结论以外部receipt与远端读取结果为准。receipt不包含自己的hash，也不要求在receipt自身中记录其上传后才获知的asset_id，防止另一种自引用循环。

最终给用户的答复必须包含：分支链接、B提交链接、HANDOFF链接、FINAL_REPORT链接、Release链接、主要结果、训练/确认是否完整、真实阻塞。未达到的项目直接写未达到。

---

## 16. 结束前的一次审核清单

这是交付审核，不是新一轮安全测试。由结果/文件直接回答下面问题，并在HANDOFF测试小节总结；不再建独立的庞大审计系统。

| 核查问题 | 合格条件 |
|---|---|
| 起点与权重正确？ | 起点完整SHA、R1/Concerto哈希与本文一致或有显式基础修复记录 |
| 新增接口真能运行？ | 新CLI实际调用真实trainer/evaluator，不是TODO或固定返回表 |
| 原R1是否保持？ | 新开关关闭后的真实输出等价；loss没有重复计入 |
| 阶段是否真实对应？ | stage与p2s维度一致；没有对半切、跨阶段平均或漏掉empty GT惩罚 |
| 是否实际训练？ | optimizer steps/非零参数变化/曲线/checkpoint与GPU-hours相互一致 |
| 对照是否公平？ | 公共初始化、样本流、预算匹配；候选对C0-k与C0-best均有表 |
| 选模有无越界？ | CAL选ckpt、SEL选recipe、FINAL_LOCK在PB之前；没有PB后挑赢家 |
| 修复是否改变了预期变量？ | 仅同一扫描mask内容；ID/score/class政策不变；soft不是bool伪造 |
| native FH是否真实？ | 真正完整prefix前向、native输出；覆盖不足时不宣称全T胜出 |
| 计时是否完整？ | 不是cache replay；包含新producer/scorer/refiner及正确累计scope |
| 状态是否诚实？ | 模块失败、未训练、部分覆盖、预算取消、方法无增益分别记录 |
| GitHub是否实际同步？ | B=remote branch SHA，远端交接可读取，Release附件有验证receipt |

**完成本轮的含义**：在预算和真实资产允许的范围内执行全部规定阶段，并如实发布选中或未选中的结果；不要求伪造“方法成功”。如果全部开发/选模/确认/发布均已完成而新方法没有稳定增益，应报告 `COMPLETE / NO_CONFIRMED_SUPERIORITY / VERIFIED`，而不是不断增加模块直到凑出一个正结果。

---

## 17. 来源索引：固定提交与一手论文/官方文档

下面链接是编写本指令时核查的证据入口。Codex应优先在已checkout的固定提交中读取对应文件；无必要反复联网搜索相同事实。外部论文描述不构成本项目预期涨点的证明。

- [E01]：CrossWindow最终报告：固定D0、PB数值、FH/profile未确认。
- [E02]：CrossWindow配置：checkpoint/hash、窗口、query/state/输出政策。
- [E03]：R1 root-cause交接：目标损失、历史训练与诊断、选中checkpoint；历史“三seed”需以真实checkpoint来源判断其含义。
- [E04]：现有criterion的mask损失实现。
- [E05]：查询初始化与decoder前向实现。
- [E06]：同一模型的mask attention实现。
- [E07]：历史epoch-90 decoder诊断及其provenance。
- [E08]：当前E1错误判定及候选覆盖统计实现。
- [E09]：TaskMemory最终报告：D0来源、负结果、成本scope、暴露限制。
- [E10]：crosswindow资产/点序/候选结构。
- [E11]：R1训练模板；其中默认weighted不是最终R1，必须解析resolved配置。
- [E12]：官方后处理：bool输出、热图、lineage。
- [E13]：mask feature聚合、mask module与attention源码。
- [E14]：official metric adapter与阈值常量。
- [E15]：旧训练脚本及其固定variants/步数/parent约束。
- [E16]：实际训练、prediction-only producer与profiler历史命令。
- [E17]：StageMeta、真实阶段和vertex对应。
- [E18]：前缀官方metric accumulator调用。
- [E19]：native FH输入/cache/provenance结构。
- [E20]：旧TaskMemory训练适配、`InstanceSegmentation`和目标reducer调用。
- [E21]：`trainer/trainer.py::_get_mask_and_scores/_get_full_res_mask/_filter_and_sort_predictions`：logits>0、full-res多数决策、候选过滤顺序。
- [R01]：ReScene4D，arXiv 2601.11508v2；主要依据§4与附录D.3的temporal评价语义。
- [R02]：LaSSM，arXiv 2602.11007，作者页面/摘要注明IEEE TCSVT接收；只借鉴semantic-spatial query思想。
- [R03]：AQ3D，arXiv 2608.30618v1；预印本参考，不声称已确认顶会接收或本项目可复用权重。
- [R04]：GitHub Release官方限制。
- [R05]：GitHub普通Git大文件官方限制。

[E01]: https://github.com/Orangekostar/Persist4D/blob/6ef77620aa20926311eff3124a794a6ca2e32727/artifacts/crosswindow_evidence_v1/FINAL_REPORT.md
[E02]: https://github.com/Orangekostar/Persist4D/blob/6ef77620aa20926311eff3124a794a6ca2e32727/configs/crosswindow_evidence_v1.yaml
[E03]: https://github.com/Orangekostar/Persist4D/blob/6ef77620aa20926311eff3124a794a6ca2e32727/artifacts/rescene_task_learning_root_cause_v1/HANDOFF.md
[E04]: https://github.com/Orangekostar/Persist4D/blob/6ef77620aa20926311eff3124a794a6ca2e32727/models/criterion.py
[E05]: https://github.com/Orangekostar/Persist4D/blob/6ef77620aa20926311eff3124a794a6ca2e32727/models/rescene.py
[E06]: https://github.com/Orangekostar/Persist4D/blob/6ef77620aa20926311eff3124a794a6ca2e32727/models/rescene.py
[E07]: https://github.com/Orangekostar/Persist4D/blob/6ef77620aa20926311eff3124a794a6ca2e32727/artifacts/rescene_task_learning_root_cause_v1/decoder_diagnostics/DECODER_DIAGNOSTICS.json
[E08]: https://github.com/Orangekostar/Persist4D/blob/6ef77620aa20926311eff3124a794a6ca2e32727/scripts/diagnose_crosswindow_failures.py
[E09]: https://github.com/Orangekostar/Persist4D/blob/6ef77620aa20926311eff3124a794a6ca2e32727/artifacts/task_memory_retention_v2/FINAL_REPORT.md
[E10]: https://github.com/Orangekostar/Persist4D/blob/6ef77620aa20926311eff3124a794a6ca2e32727/scripts/crosswindow_cache.py
[E11]: https://github.com/Orangekostar/Persist4D/blob/6ef77620aa20926311eff3124a794a6ca2e32727/conf/config_rescene4d_concerto_rootcause.yaml
[E12]: https://github.com/Orangekostar/Persist4D/blob/6ef77620aa20926311eff3124a794a6ca2e32727/scripts/rescene_task_postprocess.py
[E13]: https://github.com/Orangekostar/Persist4D/blob/6ef77620aa20926311eff3124a794a6ca2e32727/models/rescene.py
[E14]: https://github.com/Orangekostar/Persist4D/blob/6ef77620aa20926311eff3124a794a6ca2e32727/scripts/p6a_metrics.py
[E15]: https://github.com/Orangekostar/Persist4D/blob/6ef77620aa20926311eff3124a794a6ca2e32727/scripts/train_task_memory.py
[E16]: https://github.com/Orangekostar/Persist4D/blob/6ef77620aa20926311eff3124a794a6ca2e32727/artifacts/task_memory_retention_v2/COMMANDS.md
[E17]: https://github.com/Orangekostar/Persist4D/blob/6ef77620aa20926311eff3124a794a6ca2e32727/datasets/task_memory_episode.py
[E18]: https://github.com/Orangekostar/Persist4D/blob/6ef77620aa20926311eff3124a794a6ca2e32727/scripts/system_comparison_metrics.py
[E19]: https://github.com/Orangekostar/Persist4D/blob/6ef77620aa20926311eff3124a794a6ca2e32727/scripts/system_comparison_inference.py
[E20]: https://github.com/Orangekostar/Persist4D/blob/6ef77620aa20926311eff3124a794a6ca2e32727/trainer/task_memory_trainer.py
[E21]: https://github.com/Orangekostar/Persist4D/blob/6ef77620aa20926311eff3124a794a6ca2e32727/trainer/trainer.py
[R01]: https://arxiv.org/html/2601.11508v2
[R02]: https://arxiv.org/abs/2602.11007
[R03]: https://arxiv.org/html/2608.30618v1
[R04]: https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases
[R05]: https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github
