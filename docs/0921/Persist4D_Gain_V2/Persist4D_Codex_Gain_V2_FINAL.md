# Persist4D Gain V2：双主线增益实验、有限关联补测与完整发布

**日期：2026-09-22｜交付性质：给 Codex 的唯一执行指令，不是实验完成报告。**

| 项目 | 固定值 |
|---|---|
| 仓库 | `Orangekostar/Persist4D` |
| 已读取的起点提交 | `8b5e93795817682fe70864daa545db219e1443c9` |
| 起点分支 | `research/persist4d-perception-gain-v1` |
| 新工作分支 | `research/persist4d-perception-gain-v2` |
| 新证据目录 | `artifacts/perception_gain_v2/` |
| 新运行根变量 | `PERSIST4D_GAIN_V2_ROOT` |
| 新总入口，需实现 | `python -m scripts.perception_gain_v2` |
| 全程累计 GPU 上限 | 沿用 V1 的 192 GPU-hours；V2 使用核对后的余额，不重新获得 192 |

> **执行要求。** 阅读本指令绑定的源码与产物，完成必要开发、实际训练、CAL/SEL 选模、最终锁定、真实确认、成本测量和 GitHub 同步。不要只交计划、模块定义、测试通过记录或伪装成完成的空表。任务范围内无需逐阶段再请求许可；预算、可用数据或权限确实不足时，保留真实结果与具体阻塞，继续独立可执行任务，并实际尝试发布。
>
> **效力。** 本文是依据 V1 已暴露结果制定的 V2 协议。V1 的文件、tag、结果和部分完成状态保持不变；不把 V2 调参追溯成 V1 的预注册实验。本文完整规定 V2 的优先级与决策，不要求同时执行 V1 的所有流程。历史事实与新设计分开标记，新增 CLI/配方不表示已有实现或已验证涨点。
>
> **目标。** 提高真实 ReScene temporal evaluator 下的质量。0.5pp 平均增益、1pp 长时增益是研发目标，不是保证。未测不能算输或赢；继续训练优于退化 C0，也不等于超过冻结 R1。

---

## 1. 本轮要交付的答案与已核实起点

### 1.1 四个必须回答的问题

1. 同源预测下，降低适配学习率、阶段损失或查询/注意力改动，是否优于冻结 R1，并优于相同预算的 C0？
2. **完全冻结 R1** 时，同一扫描的 old/new 证据是否比仅用 new 证据的同容量修复器更有效？
3. 此前未完整执行的 12 个关联配置，在同源预测上有没有一个可确认的有效候选？
4. 唯一锁定的最终系统，在完整 PB、LOCAL-T2 和额外 reference 上表现如何；相对 native FH 的质量与真实成本分别怎样？

两条主线独立：A=感知适配；B=冻结 R1 的修复器。C=关联补测为有限辅助线。B 不等待 A 选模，A 不以 C 或 E1 门槛为前置条件。最终确认和发布有预留资源。

### 1.2 已核实的 V1 事实，不能重新包装成 V2 增益

| 事实 | 依据与解释 |
|---|---|
| V1 为 `PARTIAL_WITH_BLOCKERS`，无正式 PB/LOCAL/profile 结论 | [E01][E02]；代码发布不等于实验完成 |
| C0 完成 750；S-BAL 恢复 checkpoint 为 350，精度只测到 250 | [E01][E03]；379 是未保存观察进度，不是可恢复边界 |
| scorer 完成 500，64 references/64 pairs | [E02][E04]；可复用，不是主模型 Q-SEM 已训练 500 |
| V1 CAL：4 references、23 logical units，D0/lag1/mean | [E05]；不能与 PB 的 24.923/15.535/9.011/6.433% 混算 |
| C0-750 相对 R1 四 T 平均为 −1.659443pp | [E05]；动机是检验适配配方，不是已证明 LR 导致退化 |
| S-BAL-250 相对 C0-250：−0.511295/+0.166801/+0.413175/+0.498748pp | [E05]；平均 +0.141857pp，但相对 R1 平均仍 −0.885029pp |
| 修正 E1 得到 94/337=27.893175%，门槛支持关联空间 | [E06]；是候选齐全但发布失败事件，不是可恢复 AP 或确证关联错误率 |
| E1 重算仍读取旧 base/supplement；旧缓存与一个 live 序列不等价 | [E07]；27.89% 不得直接称为 V2 live 预测的当前错误率 |
| 当前 soft refiner 已实现 `new + 2*tanh(MLP)`，尚无训练收益 | [E08]；可表达的残差受限，不等于任意修复 mask |
| V1 编排硬编码 192 和固定臂集合，部分路径写死 source_commit/root | [E09][E10]；V2 必须参数化实际运行身份，不能只改 YAML 绕过 |

CAL 参考数值（原始 AP，范围 [0,1]）：

| checkpoint | T2 | T3 | T4 | T5 |
|---|---:|---:|---:|---:|
| R1 / C0-0 | 0.36262261867523193 | 0.2918131351470947 | 0.24670922756195068 | 0.22270338237285614 |
| C0-H-250 | 0.35375484824180603 | 0.27594244480133057 | 0.2354133278131485 | 0.21766230463981628 |
| C0-H-750 | 0.3342858552932739 | 0.280689001083374 | 0.22574535012245178 | 0.21675042808055878 |
| S-BAL-H-250 | 0.34864190220832825 | 0.2776104509830475 | 0.239545077085495 | 0.22264978289604187 |
| Q-SEM-0 | 0.36067473888397217 | 0.2956312298774719 | 0.24083933234214783 | 0.2222689390182495 |

H/L 是 V2 的学习率标识：H=5e-5，L=1e-5；不是新架构。上述历史值只在输入、recipe、publisher、metric、seed 均等价时复用为当前比较值，否则保留为历史行并重新评价相应权重。

---

## 2. 源码阅读、改动绑定和实现边界

先阅读以下类/函数及直接调用方，生成一张 `CODE_BINDINGS.md`。每行填写真实起止行、调用方、改动位置和实验 ID；行号从 checkout 源码提取，不沿用旧文档声称“已记录”却没有行号的表。[E11]

| 源码/产物 | 已确认的作用 | V2 的精确动作 |
|---|---|---|
| `scripts/train_perception_gain.py::compose_variant_config/run/_build_loader` | V1 训练入口、Hydra、pair 数据、sample plan、checkpoint | 增加显式 V2 recipe/config/root 输入；保留默认 V1 行为，不复制完整 trainer |
| 同文件 `build_weighted_sample_plan/PerceptionDrawDataset/DeterministicPerceptionCollator` | 独立 draw RNG 与确定性增强 | 复用相同 draw 计划；LR/模块变化不得改变数据流 |
| `trainer/perception_gain_trainer.py::PerceptionGainTrainer.configure_optimizers` | 已从 settings 读取 LR，raw-sum 继承基础 trainer | 通过真实 config 传入 H/L；不要硬改全局 optimizer |
| 同文件 `PerceptionProgress/on_save_checkpoint/on_load_checkpoint` | optimizer、scheduler、rank RNG、sample cursor 恢复 | 导入 V1-350 时恢复原状态；新低 LR 从 R1 重新开始；resume 校验 recipe |
| `conf/perception_gain_v1/common.yaml` | 5e-5、encoder eval=false、FP32、batch 32、3000-step schedule | V2 显式 overlay；除了规定变量，不同时改冻结、EOS、增强和 loss reducer |
| `models/criterion.py::loss_masks/_loss_masks_by_stage` | legacy/balanced/worst 及 aux 接入 | 复用；不改变 matcher，不将 debug 量加入 raw-sum |
| `models/perception_gain.py::stage_aware_mask_losses/derive_segment_stage_ids` | 阶段均衡、弱阶段损失和严格 stage 映射 | 保持 alpha=.5、beta=.25；空 GT 阶段仍受监督 |
| `models/rescene.py::initialize_queries/forward` | Q-SEM、第一次 cross-attention 开放、特征导出 | 保持现有单开关；不叠加新 decoder，不调 Q/K |
| `scripts/perception_gain_evaluation.py::run_checkpoint_evaluation/_load_evaluation_weights` | checkpoint→live 数据→D0→official metrics | 使用实际 recipe、角色与执行代码 SHA；支持 H/L，不把低 LR 权重标成 H 配方 |
| `scripts/perception_gain_native_evaluation.py` | native 全前缀前向 | 读其真实入口，接通 CAL/SEL 完整基线与最终 FH；不只跑 smoke |
| `scripts/perception_gain_local_evaluation.py` | LOCAL-T2 专用评价 | 继承原 population 和配置；不要把 CAL 当 LOCAL |
| `scripts/prepare_perception_refiner.py::run/advance_d0_identity` | 显式父模型和真实 soft 数据生成，固定 D0 路由 | 直接使用 R1/update0；不等感知锁定；增加 V2 目录参数 |
| `scripts/train_perception_refiner.py::CausalRefinerRevisionTransform` | 配对、vertex 对齐、残差、重新物化 | 新增显式 input_mode；NEW_ONLY 与 OLD_NEW 使用同一配对范围 |
| 同文件 `align_old_probabilities_to_new_segments/build_refiner_training_records` | new partition 对齐和离线训练目标 | 共用一次真实缓存；训练标签与推理 payload 分离；不伪造 logits |
| `models/perception_gain.py::CausalMaskRefiner` | 135→64→1、zero-init、±2 残差 | 保持结构；只增加 mode-aware 输入适配，不新增 mask scorer |
| `scripts/rescene_task_postprocess.py::extract_official_task_prediction/materialize_segment_logits` | 原始 soft sidecar、bool/热图映射 | 零残差必须经过真实 bool 决策路径验证，不用捷径返回旧 bool |
| `scripts/task_memory_output.py::RevisionMaskRequest/LagOnePublisher` | D0 的上一扫描修订回调 | 保持未修复证据驱动的 identity/state；仅改归档 mask |
| `models/overlap_entity_association.py::preregistered_association_configs/associate` | 已有 3+3+6 配置及 assignment | 原样复用 12 配置；不新增学习型关联头 |
| `scripts/replay_crosswindow_association.py` | canonical frame、关联、另一套 publisher/accumulator | 用其真实适配器；不能将这里的 dataclass 当 TaskMemory publisher 的同类对象 |
| `scripts/p6a_metrics.py::OfficialMetricAccumulator/official_temporal_match_trace` | official pooled AP 与逐 tau trace | 正式评分唯一口径；不以简化 AP 替代 |
| `scripts/perception_gain_profile.py` | V1 实际网络/状态/物化计时 | 新增完整 recipe 接入及 NEW/PAIR模式；不能漏计 scorer/refiner |
| `scripts/perception_gain_publish.py` | Git→tag→Release，但根/分支和验证规则固定 | 参数化 V2、检查非空必需 assets；按第17节发布，不复制旧 READY 判定 |

绑定来源：[E08]–[E20]。`scripts/perception_gain_campaign.py` 的全量 validator、固定 PILOT_VARIANTS 和根目录**不作为 V2 总驱动器**；复用其中纯函数可以，不能将单支线 BLOCKED 传播成全图停机。

### 2.1 最小开发结构

新增 `configs/perception_gain_v2.yaml`、`scripts/perception_gain_v2.py`，以及确需的一个 V2 配置/运行身份辅助模块。其余优先在现有训练/评价/修复/发布函数增加可选参数，缺省仍走 V1。不要复制 4000 行旧 campaign、另建通用实验平台或新安全框架。

必须统一 `recipe_id` 与 `architecture_variant`：前者例如 `C0-L-s45`，后者是既有 `C0`。两个 LR 的模型结构可以相同，运行身份、优化器与训练 recipe 不能合并。

以下均为**必须新增并实测的接口合同**，不是声称当前仓库已有：

- `compose_variant_config(..., recipe_config=None)` 或等价窄适配：解析明确的数值 overlay 并导出 resolved config。
- 训练/评价/缓存/Profile 使用同一份 resolved recipe；不要训练时 L、评价报告却重新 compose 默认 H。
- 所有新路径显式接收 `artifact_root/external_root/assets_path/roles_path`，不写回 V1。
- V2 refiner frozen/resume schema 明确记录 `input_mode, parent_recipe_hash, parent_weight_hash, source_shards_hash, seed, updates, residual_bound`。训练、评价、profile、bundle loader 同步识别；不只改训练端。
- 小型 V2 调度器根据每任务依赖和资源状态运行；不检查“必须五臂全部 COMPLETE”才允许任何后继任务。
- V2 live资产解析只要求R1/Concerto/data_root/rio_metadata/metric_dataset_spec；旧 `_resolve_cache_assets()` 强制要求的 dev_base_cache_root/dev_supplement_root 只在显式历史导入时需要。不能因为不相关旧缓存不可读而阻断新live前向。[E21]

---

## 3. 启动、数据可靠性与可恢复运行

### 3.1 工作树与环境

1. 在现有克隆检查一次 remote、HEAD、工作树；从固定 B 提交创建 V2 分支/独立 worktree。存在用户修改时保留；禁止 reset/clean/force push。
2. V2 分支已有同一指令和 run identity 时恢复；无关同名分支使用最小 `-r2` 后缀。起点分支后来移动也不自动升级本轮起点。
3. 使用既有 `persist4d` Conda 环境。只有具体导入/运行失败才做局部修复，不升级 CUDA、PyTorch 或全套依赖。
4. 输出根默认 `${HOME}/persist4d_runs/perception_gain_v2`；优先已有用户设置的 `PERSIST4D_GAIN_V2_ROOT`。旧输入按 V1 manifest、已知 `assets.local.json` 和显式环境变量解析，找不到即报告具体键；不扫描所有磁盘。

### 3.2 必需资产

| 资产 | 核验值/规则 |
|---|---|
| R1 | `629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd`；754813672 bytes |
| Concerto | `845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07` |
| 已训 scorer | `5bf58cdde393493daeb29bd023cb058db51bec56e76d39ba14ff68a6f911380c`；缺失才按第7节补训 |
| S-BAL-H 的 V1 last | `49062639ec7e1f38280c6be2a42be81fe913cd6e1169dff862b9fc8f551b20ce`；读取 resume payload 确认350 |
| C0-H-250 | `362a8575fb0d011b079b60af4eb92f57fcb5734484230c76c468bb9718007b87` |
| C0-H-750 | `384884e3e86de79b5ca92661aa887b1f34612d2d8c90e93a66bbf70b87e30143` |
| 数据与评价 | 实际 data_root、rio_metadata、原 metric dataset spec、R1 resolved config、V1 DATA_ROLES、PB/LOCAL manifest |

只对选中的输入 checkpoint/hash 和 manifest 核查一次，写入资产表。V1 scorer/评价结果可复用不等于其服务器文件现在可读，必须实际定位。

### 3.3 不让 NFS 再拖住所有任务

V1 的 RPC/NCCL 错误是历史记录，先对已知依赖做一次有超时的当前检查。优先已有本地副本；按**实际被 manifest 引用的文件**准备本地热数据。路径映射保留原相对结构与 scene ID，检查本地根和其 symlink 目标确实不落回失效 NFS。

- 本地盘空间先实测，不假定充足；不能复制空目录、少量子集冒充完整 TRAIN/CAL/SEL。
- 依赖恢复前，无可读合法副本的主模型训练为 `BLOCKED_DATA`；已完整缓存到本地的 refiner、关联重放、报告、Git 发布仍运行。
- 本地 dataset staging 不改变 split、sample list、增强、输入分辨率、GT 或 difficulty；原始数据 staging 容量与“新预测缓存32GiB”分开计。
- 不自动改管理员 NFS 服务、不重启其它进程、不永久修改挂载配置。执行者有现成可用的授权本地副本时可切换数据根；没有就如实阻塞。
- 新数据清单记录 relative path、bytes、来源 manifest hash；传输校验只针对本轮复制文件，不对全部原始数据反复全盘 hash。

### 3.4 输入停滞控制与恢复

V2 多 worker loader 使用可配置的 `timeout_seconds=180`；num_workers=0 时 timeout=0，由作业 watchdog 观察“完成 batch/update”进度。先用计划内真实 batch 测量，若正常 batch 确实超过180秒，将超时固定为 `min(600,max(180,10*observed_max_batch_seconds))`，在训练前记录，不根据分数调节。[R03]

DataLoader timeout 不是修复 NFS、也不保证终止内核不可中断 I/O。异常后终止本任务的其它 ranks/子进程、释放可释放的 GPU；不得延长 NCCL 超时后无限等待。一次相同外部错误后不循环重启；只有依赖恢复或本地输入源切换且最小读取检查通过时重试。

- 权重、日志、last.ckpt 写本地；每50 updates保存 last，规定评价点保存固定 checkpoint。不要逐step hash权重。
- 350 checkpoint 恢复包含 optimizer/scheduler/rank RNG/draw cursor，不从379开始，不重新warmup。
- 缺少旧350但有完整250 resume checkpoint，允许从250恢复并记录实际回退；只有部署权重而无optimizer时不称exact resume。恢复需要的本地控制文件/路径可重定位，但不能把old run目录的未完成日志覆盖；V2单列导入来源和增量费用。
- 旧高 LR 资产缺失时，只阻塞其连续性任务；低 LR 从 R1 的新训练与 R1-refiner照常执行。不要为补齐旧表默认重训完整高 LR C0。

---

## 4. 累计预算与排程

### 4.1 余额不是新的192

V1公开RUN_STATE：foundation/evaluation=3.653669649958901；perception training=8.15015840478406；合计 **11.803828054742961 GPU-hours**。名义余额 **180.19617194525705**。[E02]

启动时读取本地更完整日志，按任务/时间区间去重，确认是否有尚未纳入的 V1 占卡成本。中断记录中的估计值标 `ESTIMATED`，不伪装遥测；同一watchdog时间不重复记账。不因缺日志而把已知消耗归零。若有新增已占用时间，增加累计账本并减少余额。

`remaining = max(0, cumulative_cap - reconciled_prior_usage - V2_usage)`。
默认 cumulative_cap=192；可启动前调低，不可由 Codex自行调高。较低cap不是schema错误。推理/scorer/refiner的占卡、阻塞占卡、故障重跑、测试占卡均记录；复用历史artifact不二次记历史费用。

### 4.2 默认 V2 分配，均受真实余额约束

| 项目 | 计划额度（GPU-hours） |
|---|---:|
| 感知短跑与唯一晋级机制+同LR C0 | 70 |
| 基础live前向、CAL/SEL、开发期证据 | 34 |
| R1/P父模型soft缓存与两类refiner | 16 |
| 最终PB/LOCAL/ADDITIONAL及profile预留 | 40，其中profile最多4 |
| 第二种子/必要故障恢复/未使用余额 | 名义20.19617194525705 |

这些是V2调度计划，不是精确耗时预测。按V1吞吐和本轮前50个计划内updates更新预测；不另开大规模benchmark。新缓存最多32GiB；同时CPU workers≤8、RAM预算96GiB。关联搜索最多16 CPU-core-hours，只跑固定配置，不增加GPU训练头。

先确保**最终确认预留**：按本轮CAL真实前向/metric耗时估计完整确认所需，取1.25倍并与40额度比较。超出40优先从未承诺的训练/复核余额转入，在任何PB分数产生前写明；没有资金则减少探索任务，不缩减正式人口冒充完整。

额度在未查看PB前可从已完成任务的余量转移给另一任务，写一条预算事件即可，不做新审批系统。不得突破全程累计cap。

### 4.3 不足预算的确定性取舍

按以下顺序剪枝：

1. 第二种子中尚未启动的完整pipeline复核；保留已完成范围。
2. 入选新P上的refiner再训练（**R1上NEW/PAIR不取消**）。
3. 尚未启动的辅助pilot，依次S-WORST、Q-SEM、A-OPEN。
4. 尚未启动的新感知full延长；不能只延长候选而不给同LR C0相同预算。

不删原始R1基线、已承诺的低LR配对短跑、R1上两类refiner的配对关系、最终报告/发布来换更多候选。当实际余额低于默认分配之和时，先预留按实测估计的完整确认成本，再按上述顺序取消尚未承诺任务；各剩余额度之和必须≤真实余额，不将固定计划额度当作可用资金。若真实余额连核心配对任务也容纳不了，保存恢复点并如实部分交付；有可用锁定候选的确认/计时继续。预算剪枝叫 `SKIPPED_BUDGET`，不是模型失败。

### 4.4 设备

同节点双A40、batch2/GPU、accumulation8、FP32作为感知默认，effective batch32。每次最多两个独立训练作业；全局最多占4张已确认空闲GPU（包括refiner/evaluation）。refiner与评价各可用单卡；不抢占其它作业，不做跨节点DDP调优。

若仅一张卡可用，新低LR cohort可以batch2、accumulation16启动，但两臂必须相同。此时与V1高LR双卡结果标为不同runtime cohort，不用于LR因果比较；辅助臂统一用L。旧350双卡resume等待同拓扑，不能伪称单卡exact resume。修复器、关联、基础评价不因此等待。

---

## 5. 数据、基线与正式评价身份

### 5.1 固定population

继承V1 `artifacts/perception_gain_v1/DATA_ROLES.json`及其上游原始manifests，不重新随机划分。

| 角色 | 用途 | 已知规模/执行要求 |
|---|---|---|
| TRAIN | 感知/scorer/refiner拟合 | V1记录377物理references；ScanNet原训练子集照旧；实际过滤数量核对 |
| CAL | checkpoint、LR/配置排序 | 4 refs/23 logical units；每方法92个T2–T5前缀 |
| SEL | 固定候选/recipe开发选择 | 4 refs/24 logical units；每方法96个前缀 |
| PB | FINAL_LOCK之后唯一正式确认 | 6 refs、43masters×3orders=129 logical units；每方法516前缀 |
| LOCAL-T2 | 原official-like T2确认 | 41 refs，原manifest154序列；实际原evaluation seeds分别继承 |
| ADDITIONAL | 固定额外reference确认 | V1已绑定T2=111单位/40refs，T3=77/23，T4=32/8；不要伪造T5 |

TRAIN与CAL/SEL/PB/LOCAL/ADDITIONAL的reference交集须为空；保留V1既有过滤，不按成绩删除难例。发现真实不一致要修身份映射并声明哪些结果失效，不重划更有利的split。

CAL/SEL是已暴露开发集，PB也是历史暴露benchmark；如实记录base训练暴露和本轮适配暴露。LOCAL与ADDITIONAL可能重叠，分别列出交集，不称两份独立证明。不同T的ADDITIONAL人口不同，不当作纯长度退化曲线。

### 5.2 基础前向要覆盖完整CAL/SEL

先在1个实际CAL序列上确认：

- R1 strict load和新开关关闭行为；live repeated结果一致。
- D0 live producer与实际canonical输入同源；如果历史cache不等价，旧cache仅用于历史诊断。
- **D0评价路径与refiner禁用的parent路径**在同一sequence的mask/class/score/ID/指标一致。二者内部publisher不同，不能凭都叫D0就假定等价。
- zero refiner通过原materialize路径仍等价；不靠检测delta=0直接返回旧bool。

随后完整生成CAL/SEL的 `R1-D0` 与 `FH-R1-native`，每个角色两方法都完成才标 `BASELINES_COMPLETE`。SMOKE_PASS是独立字段。基线未齐时其他独立训练可运行；组件选择只要求其candidate与R1-D0/parent的CAL/SEL完整，FH单独blocked不自动禁止这种选择，但native比较保持null、基础阶段保持部分完成。缺candidate/parent本身的完整SEL则不允许该组件晋级。

PB预测和评分在FINAL_LOCK后才开始；不能提前借PB寻找最强baseline或改超参。

### 5.3 缓存与版本

缓存键必须包含实际 `code_commit + relevant_source_digest + weight_hash + inference_config_hash + input_manifest_hash + point_order/transform + role/sequence/order + eval_seed + postprocess/publisher`。不能因某个status文件PASS且单一文件hash相同，就复用依赖已变化的缓存。

区分 `parent_commit`、`executed_code_commit`、`training_recipe_hash`、`inference_recipe_hash`。代码提交先于正式运行；若必要修补，提交新代码，只废除依赖受影响的结果。新结果不能继续写死6ef7762。H/L参数若不影响推理仍保留训练来源，不偷换其训练recipe。

导出预测后可释放GPU、在CPU重算official metric；记录实际前向次数、GPU占用与CPU评价时间。研究期多头共享一次上游前向可节约成本，但最终部署profile不得借用其它方法的免费特征。

### 5.4 不同评价实现不能混用

正式 AP 使用 `OfficialMetricAccumulator/CausalTaskAccumulator`，保持原类表、min_region_size、ignored/ambiguity规则、严格阈值和pooled汇总。阈值数组从已绑定stmetrics读取，不硬写常见0.50:0.05:0.95。不能拿平均scene AP或手写_temporal_scores冒充official AP。

已修正E1不重新开发。可以在准备好的V2 CAL R1输出上重算一次受影响trace以解释关联实验，明确fresh-vs-history来源；元数据缺失保持unknown。该trace不控制两条主线，也不以27.89%承诺涨点。

---

## 6. 主线A：低学习率配对与V1连续性

### 6.1 第一组四格

| recipe_id | 架构/损失 | LR峰值 | 来源 | 初始任务 |
|---|---|---:|---|---|
| C0-H-s45 | C0/legacy | 5e-5 | 优先V1有效checkpoint与评测 | 复用或重评250/750，不默认重训 |
| S-BAL-H-s45 | balanced | 5e-5 | 原350完整resume | 恢复到750，并保存250已有/750评测 |
| C0-L-s45 | C0/legacy | 1e-5 | 同一冻结R1，新optimizer | 新训练到750 |
| S-BAL-L-s45 | balanced | 1e-5 | 同一冻结R1，新optimizer | 新训练到750 |

不将低LR覆盖在原350 checkpoint的optimizer上。两个新L臂使用完全相同样本/增强流；H/L因果比较仅在runtime与其它recipe可比时成立。没有H资产时报告高LR比较缺失，L两臂及修复器继续。

### 6.2 统一训练数值

除LR及规定模块/损失外继承真实R1与V1配方：

```text
init_weight_sha = 629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd
seed = 45
optimizer = AdamW; betas=(0.9,0.999); eps=1e-8; weight_decay=0.01
schedule_horizon = 3000 optimizer updates（从第一次启动就固定）
warmup = 150 updates; 其后cosine; final_lr_ratio = 0.1
grad_clip = 1.0; precision = FP32
mask/class/change/contrastive reducer = 真实R1 raw_sum
freeze = backbone_encoder; frozen_encoder_eval = false
Q=100; K=100; D0=LAST; publish=lag1; track_score=mean
pilot_eval = [0,250,750]; full_eval_additional = [1500,2250,3000]
```

单次调用可以在750停止，但scheduler不变成750总长。新初始化在构造任何新增随机参数之前设置seed，模型RNG与draw/augmentation RNG分离。原始ScanNet/RIO nominal mixing及真实抽样函数保留，不将1.0:0.8写成每batch强制比例。

每50updates写last；每个评价点记录actual optimizer step、LR、原loss项、梯度/参数变化最小检查、sample cursor、可训练参数和checkpoint hash。不同loss定义的train loss数值不能直接作为效果比较。

### 6.3 阶段损失原样复用

只修改已有mask BCE/Dice内部聚合，matcher、分类、对比损失、aux数量不变。phase identity从真实 `temporal_stages + point2segment` 导出，segment跨阶段时修namespace，不能取平均/amax掩盖混合。

对每个匹配对象、每个有有效输入支持的阶段计算BCE与Dice，包括GT缺失的全零阶段。S-BAL按阶段均值；S-WORST对BCE和Dice分别用：

`(1-.5)*mean(x) + .5*.25*(logsumexp(x/.25)-log(n_stages))`。

debug不进入raw-sum字典；单阶段、空匹配、aux都保持已规定语义。[E12]

### 6.4 先看750，但不把C0是否涨点当总开关

第一组技术可运行时均到750（已有C0-H复用），不因250分数差就结束所有路线。任何数值错误/数据错误即时修最小问题或标该臂blocked，不以丢场景继续。

定义第10节的R1基准S_mean/S_long/S_min；低LR与高LR各自取C0或S-BAL在250/750中**CAL最高排名点**，得到该LR的probe分数。仅用可比较cohort选择 `LR*`；按S_mean→S_long→S_min排序，1e-6平局优先L。H缺失/不同cohort则LR*=L。

这只是为后续辅助臂选一个固定LR，不是部署选模，也不证明LR因果。不能仅因C0-L退化就关闭S-BAL-L或修复器。把完整四格和每点相对R1、同LR同步数C0的差值写入结果。

---

## 7. 辅助感知臂与唯一full晋级

### 7.1 三个辅助臂按固定顺序、预算运行

先A-OPEN，再Q-SEM，再S-WORST；全部使用LR*，从R1 warm-start，不从S-BAL权重叠加新开关。预算足够时三个都做750短跑；预算剪枝按第4节，未跑的不叫失败。

- **A-OPEN**：先CAL update0。只在global execution_stage_idx=0设memory_mask=None，padding照旧；后续层原样。若update0同时 `S_mean<-0.03 且 S_min<-0.05`，可按协议取消其适配，标 `SKIPPED_SEVERE_ZERO_STEP`，不宣称训练方案失败；否则做250/750。
- **Q-SEM**：先加载V1 scorer并检查六个参数键；存在等价V1零步结果可以复用。相同严重零步规则。Q=100、按真实阶段均分配额、75% semantic与25% exploration、raw-coordinate PE、zero query content保持V1实现。scorer固定不更新。
- **S-WORST**：alpha=.5、beta=.25固定；与同LR C0/S-BAL比较。不依赖S-BAL必须先达到正式胜出门槛，仍允许预算内短跑。

scorer丢失且有训练数据时，仅补训一次：按既有64-reference轮转pair方案、128→64→1、已知前景soft标签、500steps/1024segments/AdamW1e-3/WD1e-4/seed45。未知/ignore不当负例。少于8个有效references则Q-SEM阻塞，其他不等待。[E04][E13]

### 7.2 只延长一个新机制，并保留其同LR C0

候选为完成750的 S-BAL-H、S-BAL-L、A-OPEN、Q-SEM、S-WORST。每臂从250/750按CAL排名取一个pilot点用于比较，**full从last750恢复**。

1. 先筛 `S_mean>=-0.01 且 S_min>=-0.02` 的候选；有则按第10节选唯一最高者。
2. 没有通过但最佳候选 `S_mean>=-0.03` 且从250到750的S_mean未继续下降超过0.005，可允许这唯一候选延长，标 `EXPLORATORY_FULL`。
3. 仍无合适新机制时，只检查已完成750的C0：若某同cohort C0在CAL达到mean≥.005、min≥−.001、long≥0，按排名选择这一个C0延长到3000，作为CONTINUATION_ONLY候选；否则结束感知延长，P*=R1。修复/关联/最终确认继续，不盲目延长全部C0。
4. 被选机制及其**同LR、同runtime的C0**均从750继续到3000。已有同key结果不重复运行，预算只计增量。没有配对C0资产则先补它，预算不足时不能单独延长候选伪造公平比较。
5. 不再延长第二个新机制，不叠加Q+S+A；资源优先给修复、正式确认和可行的复核。

full已启动后按固定3000调度完成，除真实异常/资源硬上限；不根据中途PB或SEL值选择停止。超过总预算立即保存状态并部分交付。没有完整规定终点的模型不冒充full候选。

### 7.3 CAL选择检查点，SEL选择P*

对已完成full的新机制与C0，从 `[0,250,750,1500,2250,3000]` 实际规定点按CAL排名各选一个，写冻结表再评价SEL。若按第7.2节只延长了C0，则只对这个C0做CAL/SEL选择，不要求补一个不存在的新机制。

- C0/S-BAL/S-WORST的0步推理归并R1；A-OPEN与Q-SEM的0步可改变推理。
- 如完整架构的最佳点是0步，报告selected_adaptation_updates=0；Q仍单列scorer500。不能声称选中了3000步训练模型。
- 相对R1-D0的SEL基础门槛：`S_mean>=.005, S_min>=-.001, S_long>=0`。
- 新机制还要相对**同LR C0-best-CAL**满足mean≥.002、每T差≥−.002，才成为P*。
- 新机制未通过、C0-best通过基础门槛：P*=C0；否则P*=R1。
- 新机制/控制某一个真的blocked时，不能绕过缺失的新增机制归因；仍可把完整且通过基础门槛的C0作为CONTINUATION_ONLY，或回退R1。

另列candidate-selected-update与C0-same-update差值。SEL不用于再改LR、alpha、checkpoint或训练时长。

---

## 8. 主线B：固定R1的两类同扫描修复器

### 8.1 立即独立启动，不等待主线A

依赖仅为R1权重、TRAIN可读且数量充分、原D0及soft映射最小校验通过。固定 parent=`R1/update0`，D0=LAST、lag1/mean，所有基础权重冻结。

生成一次真实训练soft数据，在其上训练：

| repair_recipe | input_mode | 实际输入 |
|---|---|---|
| R-NEW-R1 | `NEW_ONLY` | old_logits替换为new_logits、old_score替换为new_score，再构造所有差值/概率特征 |
| R-PAIR-R1 | `OLD_NEW` | 真实同扫描old/new logits、各自score和new features |

二者使用**相同eligible候选、相同D0路由、相同有唯一old对应的修复范围、相同训练样本流、相同参数初始化**。没有唯一合法old时二者都回退new，末帧都不修。

New-only仍使用公共历史路由/配对可用性筛选，准确表述为“相同路由与支持范围下不使用old外观/soft证据的控制”，不是完全无历史系统。不能NEW修全候选而PAIR只修配对候选。

### 8.2 父预测与D0不得被修复结果反向改变

父模型产生原始prediction/observation后先进行同一D0路由和状态提交；两修复头只修改即将提交的上一扫描mask。修复后的mask不再参与association/state/evidence。score、class、候选数、顺序、identity均与父系统一致。每个头的publisher/归档状态独立。

训练标签匹配使用原new候选与训练GT；不得用GT选择old occurrence、路由对象、修复窗口或评价阈值。

### 8.3 同一扫描和soft映射

收到t时修t−1：old来自(t−2,t−1)，new来自(t−1,t)；t=2的old来自单扫描1。配对key为 `(scan_id,logical_id,generation,source_class_id)`，且old来自紧邻上一窗口。恰有一个old才配对，零个/多个都回退。不按query_id相等配对。

保持V1严格映射：old真实point概率按canonical vertex ID对齐，再按new低分辨率segment partition所覆盖full vertices做算术均值，clamp到[1e-4,1−1e-4]取logit；new端保持原始FP32 segment logit，不做概率往返。

使用 `new_low_point2segment[new_voxel_inverse]` 映射full vertices到new低分辨率segment；不要混用full-res评价superpoint编号。两窗口须共同coordinate_frame/augmentation_transform。任何不一致按具体输入键修复或标错误，不能丢掉困难sequence重新报告满覆盖。[E14][E15]

### 8.4 训练数据和模式传播

从固定TRAIN中按 `SHA256('pgv1-refiner:'+reference_id)` 取前32个有至少3次真实扫描的reference；每ref首个canonical真实master，前min(5,T)扫描，canonical/reverse两顺序。少于32用全部，少于8才数据不足。与既有V1选择器保持一致。

上游parent eval、训练缓存不额外随机改变几何。原始GT只存训练专用payload。两个模式共用相同离线目标和候选/segment抽样索引。不要训练时替换old，评价/profile却忘记替换；`input_mode`进入recipe/hash/checkpoint/bundle并由同一输入适配函数实现。

`R-NEW` 输入始终从new构造全部old派生字段；任意扰动给入old的数值而不改变配对资格，NEW输出必须不变。

### 8.5 网络、目标与训练

保持已有网络，参数量由实际named_parameters计算，不依赖文档硬写：

```text
input = LayerNorm_no_affine(F_new, eps=1e-5)[128]
      + clip(z_old,-8,8), clip(z_new,-8,8), clip(z_old-z_new,-8,8)
      + sigmoid(z_old), sigmoid(z_new), old_score, new_score
input_dim = 135
MLP = Linear(135,64) -> GELU -> Linear(64,1)
最后Linear权重/bias=0
delta=2*tanh(MLP(input)); refined=new_logit+delta
AdamW lr=1e-3, weight_decay=1e-4, seed=45
1500 updates; warmup75; cosine至0.1峰值
16 candidates/batch; 每候选最多2048个有效segments
checkpoint=[0,500,1000,1500]
loss=BCE + Dice + .01*mean(delta^2)
```

标签沿用 `build_refiner_training_records()`：原new候选与类别相容GT一对一匹配，IoU≥0.1；未匹配用empty target；有明确歧义/平局按现有规则排除并记比例。每new segment目标使用其full vertices内GT比例。不要把ignore/未知标注一律变负例；若原metadata无法给出明确歧义，只声明该限制，不声称完整消歧。

先固定两个head完全一致的随机初始第一层和零输出层；用独立、共享seed的sampling generator令两臂实际batch序列相同。一个头提前出现错误不改变另一个的抽样流。

### 8.6 残差可改性与互补诊断，只生成一次

随真实缓存计算，不另跑网络：

- pair coverage（分母为所有本来需要修订的有效new候选）、回退原因；
- 原误分segment中 `abs(z_new)<2` 的数量/比例，边界±2单列；
- old较new更接近GT的segment比例，区分可改区间与区间外；
- 头训练后修正原错误/破坏原正确的计数及候选通过各tau的转移。

由于tanh残差幅度<2，|z_new|≥2不能直接翻转logit符号。这是表达能力诊断，不是整体mask/AP上界；full-res多数决策与对象竞争还会影响最终结果。**本轮不根据CAL图表临时放大残差到4/8、换网络或挑阈值**。缺可改性时保留负结果，预算去完成其他已规定任务。

### 8.7 零残差与评分不变

必须复用这一顺序：new segment logits加delta → low point2segment展开 → 严格logits>0 → inverse_map到full points → 原eval_on_segments的0/1均值>0.5多数决策。只替换原已保留候选的该扫描mask。

原top-k/filter/score在未修复prediction上完成；修复后不重新筛候选或重算score。禁用时原路径；启用零权重时经过真实映射仍等价。不能“delta=0则return旧bool”掩盖错误。

### 8.8 CAL评价共享上游，但部署成本独立测

两个head保存完整1500训练轨迹后，一次流式读取/生成CAL父soft证据，在同一sequence上评价0/500/1000/1500各checkpoint。为每个checkpoint独立维护publisher和official accumulator，不能共用已修改的归档状态。

两个head的update0同输出可去重。CAL每头按第10节选一个checkpoint，再各自只在SEL评价该固定checkpoint。原父输出同次前向生成并比较；大缓存超限按sequence流式而不是重新前向八遍，也不永久保存全部历史soft features。

---

## 9. 有限关联补测及有条件的新P修复

### 9.1 关联只用冻结R1新同源预测

复用 `preregistered_association_configs()` 的12个配置：

| 家族 | 配置 |
|---|---|
| A0-U | tau=0.60, 2/3, 0.73 |
| A1 | theta=0.50,0.65,0.80；tau=2/3，mutual_margin=.10 |
| A2 | lambda=.25,.50 × tau=.60,2/3,.73 |

输入是第5节新R1-live canonical frames（预测/observation/vertex都同源），不是旧不等价supplement。保持M-new、原始occurrence score和mean聚合。用原CrossWindow路由/publisher及D0桥接对照；不能把更差桥接版本当唯一baseline。[E16][E17]

全部配置在CAL上评价T2–T5；GPU前向仅一次，重放CPU。超过16 CPU-core-hours按固定返回顺序在配置边界停止，已完成配置保留，标 `SEARCH_BUDGET_PRUNED`，不称完整搜索失败。未完成单配置不参与排名。

CAL只取一个最高排名配置A*，然后在SEL检验相对R1-D0：mean≥.005、min≥−.001、long≥0。未通过回退D0，不根据SEL再挑第二名。identity错误作为解释指标，不另设未声明的否决门槛。没有完整CAL配置时只跳过此辅助线。

**本轮A*为R1上的独立候选，不与refiner叠加。** 两套publisher/配对支持不同，直接混插会改变训练—部署条件；本轮组合只允许“选定感知P + 在该P上重训的refiner”。不做关联×感知×修复笛卡尔积。

### 9.2 P≠R1时的可选组合实验

主线A独立完成并在SEL选出P后，如果P≠R1、R1修复对照已完成且至少一个R1修复头通过相对parent的基础门槛，并且预算允许：

1. 在**同一TRAIN reference/episode清单**上，用P重新生成soft数据；不能直接移植R1头。
2. 仍成对训练NEW和PAIR两个头，结构/步数/seed/抽样规则不变；最多新增这一组父模型，不在多个P上训练。
3. CAL各选checkpoint，SEL按同一第10节规则决定是否采用。
4. 资金条件为这组预计剩余成本×1.25能被修复额度/可转余额覆盖且不动最终确认预留；不满足标 `SKIPPED_BUDGET`。

因此最多两个父模型（R1和P）×两个修复模式。P=R1时完全去重。P线blocked不影响R1上的选模与确认。

---

## 10. 统一排名、组件选择与最终锁定

### 10.1 单位与排名

全部AP存[0,1]；0.001=0.1pp，0.005=0.5pp，0.01=1pp。每T是同population的official pooled AP；不把T间算术均值描述为新pooled AP。

相对指定parent/baseline，`d_T=AP_candidate(T)-AP_baseline(T)`：

```text
S_mean=(d2+d3+d4+d5)/4
S_long=(d4+d5)/2
S_min=min(d2,d3,d4,d5)
```

统一排名：S_mean降序→S_long降序→S_min降序→部署新增参数少→selected_update早→recipe_id字典序；逐项差绝对值≤1e-6视为排名平局。单组件selected_update为该组件选中step；组合tie_update=parent适配step+部署refiner step，R1记0，固定scorer500单列不计tie_update。所有非严格晋级门槛使用1e-6浮点容差（如x≥a解释为x≥a−1e-6）；第12节正式严格胜出仍固定为delta>1e-6。LR选择平局按第6节优先L是唯一额外规则。不能以短跑loss、单个T或ID指标私下改变排名。

### 10.2 修复器选择，区分普通修复与历史证据贡献

每个父模型上：

- NEW和PAIR分别从CAL规定点选一个，只有实际完成1500轨迹的头可以获得“本轮完成的修复臂”身份；0步可被选中但不叫有增益。
- 在SEL，相对该parent-D0须 mean≥.003、min≥−.001、long≥0，才是可采用修复头。
- PAIR要优先于NEW并声称历史soft证据有额外作用，还须相对**NEW-best-CAL** mean≥.001、min≥−.002、long≥0。
- PAIR同时通过两类门槛：选择PAIR；否则若NEW通过父门槛：选择NEW；否则不加修复。
- PAIR通过父门槛却不满足与NEW的差值、NEW又未通过父门槛时，保持parent，不通过口头解释放松已冻结规则。

保存三张差值：NEW-parent、PAIR-parent、PAIR-NEW；另列同步数PAIR-k/NEW-k帮助归因，不能只挑NEW最差点。重复观察价值、普通修复价值、整体超过R1是三个不同结论。

### 10.3 有限候选集合，不循环SEL调参

CAL/SEL各阶段完成的固定结果产生以下至多六种候选，按实际相同身份去重：

1. R1-D0（永远保留）；
2. 第7节P-D0；
3. R1-D0 + 已通过规则的修复模式；
4. P-D0 + 在P上重训并通过规则的修复模式；
5. R1 + 通过规则的A*；
6. 通过第7节基础门槛的同LR C0-best-D0（与P等同时去重）。

组件选择都使用开发集，不把SEL称untouched。禁止根据SEL再生成第三套超参。最终候选必须完整SEL覆盖；相对R1-D0 mean≥.003、min≥−.001、long≥0，且自身组件门槛已通过，才可进入最终排名。无候选通过，FINAL=R1-D0。

按统一SEL排名锁定唯一 `FINAL_LOCK.json`，同时记录所有失败/未运行候选。`development_result_size`在平均≥.005且long≥.01时为`TARGET_REACHED`，否则按实际为`MODEST_GAIN`或`NO_CONFIRMED_GAIN`；它不是PB胜出结论。

### 10.4 支线阻塞时仍能锁定可执行系统

某一支线真实blocked/按预算剪枝，其他支线可据已完成结果锁定；`execution_status`仍列出缺失，不能声称搜索完所有方案。缺公平控制的新机制不能借此晋级。原始R1-D0始终是可选择的已知fallback，避免所有训练失败就连正式基线与发布都无法做。

`FINAL_LOCK`包括 recipe_id、architecture、LR训练来源、parent/selected权重、refiner模式、scorer、关联配置、所有配置/数据hash、selected updates、seed、publisher、阈值以及固定确认方法清单。出现锁定后正确性bug，版本化invalidate受影响选择；不利用PB数值调参再声称未触碰确认。

---

## 11. 第二种子：固定配方复核，不重新选冠军

预算与数据允许时做seed46，种子不用于重新选择LR/超参/最佳step：

| 入选组件 | 复核内容 |
|---|---|
| 新感知P | 从同R1训练P46到s45锁定step；同LR C0-46到同step；比较P46-C046和P46-R1 |
| P=C0 | 只训练C0-46到锁定step，对R1比较，不复制一份相同对照 |
| R1 + refiner | 同一冻结R1/同训练清单，两头均训练到max(NEW45所选step,PAIR45所选step)，在各自s45固定点评价并补同step对照；同一scorer不重复训练 |
| P + refiner完整组合 | 先P46，再用P46重新生成soft数据训练两头；只完成R1-head复核不算组合复核 |
| 纯关联A* | 确定性规则没有新的训练种子；标NOT_APPLICABLE，不重复同预测制造seed稳定性 |
| P在0步选中 | 感知适配复核NOT_APPLICABLE；Q的scorer500固定条件明确说明 |

单组件SUPPORTED要求SEL上相对其规定配对对象mean>0且每T退步不超过.002；否则MIXED。只有所有需要复核的实际部署训练组件都完成并支持时，pipeline replication=SUPPORTED。预算不足只报告已完成scope；评价seed≠训练seed。

复核失败仍报告s45锁定系统及s46结果，不回到PB挑新冠军，不把负复核藏起来。

---

## 12. 正式确认：完整回答是否超过原生ReScene

在FINAL_LOCK后运行；从已合法缓存复用相同内容键，不重复物理前向。

### 12.1 必需比较方法

| 方法 | 作用 |
|---|---|
| FH-R1-native | 冻结R1，完整prefix，native官方输出，不改成FH-lag1 |
| R1-D0 | 同源真实pair producer，固定起点 |
| C0-best-D0 | 有完整对应full控制时必须报告；无该实验则N/A而非伪造 |
| P-D0 | 选定感知的系统，不带refiner |
| FH-P-native | P≠R1时必测，分离更强感知与短窗口机制 |
| FINAL | 唯一锁定系统 |
| FINAL的必要消融 | FINAL用PAIR时包括同父NEW及无修复parent；用NEW时包括parent；用A*时包括R1-D0 |

相同recipe与权重/政策完全相同的行去重。未被选中的探索head/关联配置保留开发表，不全都跑PB。

PB每方法129 logical units、516个T2–T5前缀；逐个记status。重排、同reference多个sequence不是独立场景。原生FH只用当前prefix扫描，不给任何方法未来扫描。

LOCAL-T2：原population测R1-native、完整选中C0-native（若有）、P-native，按原evaluation manifest规定seed分别和汇总。refiner属于额外两次观察/lag1，不混入“同native单次前向”主表。

ADDITIONAL：FH-R1、相应parent、FINAL在固定T2/T3/T4人口上比较；source manifest发现原固定列表实际上不可读时记缺失，不改分母，也不新增/复制T5。所有结果列出actual refs/units。

### 12.2 原生故障与覆盖

真实OOM可同设备/同输入/同精度清理本进程缓存后重试一次；不能裁点、降分辨率、少算难例后沿用相同方法名。缺任一必需prefix则对应全覆盖胜出为null，额外共同子集只能明确命名描述。

`tmap_all_T_vs_native_FH = all(FINAL(T)>FH_R1(T)+1e-6)`；
`tmap_all_T_vs_D0 = all(FINAL(T)>R1_D0(T)+1e-6)`。
只在各自两个方法完整覆盖时计算true/false。LOCAL的增益只用其人口，不能减论文34.8%或PB数值。[R01]

### 12.3 默认部署

建议替换D0的最低条件：完整PB下全T超过FH-R1，且FINAL对R1-D0四T平均>0、单T退步不超过.001。否则默认仍保留D0，同时发布锁定候选及真实结果。全T严格超过D0的字段仍独立判断，不因部署容差改成true。

成本、重复训练、额外reference结果独立报告。没有更快不取消精度事实；没有稳定复核不能宣称普遍稳定。

---

## 13. 真实成本及计算量归因

至少profile FH-R1-native、R1-D0、FINAL；P≠R1时加FH-P-native。每个PB reference取同一个首canonical master，共6个；一遍完整序列warmup、三遍实测，同一空闲A40，FP32。若最终无新增系统，相同方法去重。

每T记录：真实network_forward、scorer/refiner、association/state_update、materialize、end_to_end、T1→T cumulative；绝对peak_allocated/peak_reserved、CPU RSS、resident bytes、soft-buffer bytes、历史archive bytes。前后CUDA同步、重置peak、每次恢复同起始状态。

- 三条scope固定：网络输入准备/转移→网络；状态与修订；可用输出。端到端包含相同输入准备、H2D、网络、状态、物化，排除磁盘冷读和metric；另单列冷读IO诊断，不混方法。
- D0与refiner需使用最终实际producer；不得用读取已预测cache代替live forward。
- 研究期同时评价多个头共享parent的节省不是部署加速；profile只运行该方法需要的组件。
- 双窗口soft特征需释放旧窗口；binary历史输出可能随T增长，因此不宣称系统全部内存O(1)。
- mask修复改动只影响上一扫描，不给最后扫描使用不存在的未来窗口。
- 工程训练消耗、评测占卡、离线CPU metric、实际部署单次时延分表。样本小，只报median与范围，不报夸大稳定性的p99。

核心质量分析以reference block输出差值。可复用official sufficient state做固定seed的1000次reference-block bootstrap；没有完整实现就只给逐reference表，不用均值AP的CI冒充pooled AP的CI。

---

## 14. 可执行任务图与CLI

### 14.1 依赖不是一个遇错即break的串行列表

| task_id | 直接依赖 | 内容 | 局部失败后的动作 |
|---|---|---|---|
| BIND | 固定源码、旧manifest | 资产/预算/recipe/数据人口绑定 | 无权重则训练支线blocked，能发布的文件照常生成 |
| DATA | BIND | 各角色本地完整输入/已知缓存可读性 | 分角色阻塞，不把NFS状态套给所有任务 |
| BASELINE | DATA(CAL/SEL)、R1 | R1-D0与FH-R1完整开发基线 | 训练可运行；受影响选模等待或如实缺失 |
| HIGH_CONT | DATA(TRAIN)、旧resume | S-BAL-H350→750与C0-H复用 | 不阻塞LOW_PAIR/REPAIR_R1 |
| LOW_PAIR | DATA(TRAIN)、R1 | C0-L/S-BAL-L各750 | 某一失败不改另一配方；缺配对不宣称因果 |
| REPAIR_R1 | DATA(refiner TRAIN)、R1、映射smoke | 固定R1，NEW/PAIR缓存+1500+CAL | 不依赖HIGH/LOW/FULL/P锁定 |
| ASSOC | R1同源CAL预测 | 固定12配置重放、单一SEL候选 | 不阻塞任何训练 |
| AUX | probe终态、LR* | A-OPEN/Q-SEM/S-WORST短跑 | 各臂独立状态，scorer故障仅阻Q |
| PERCEPTION | 完整pilot和配对C0、BASELINE | 唯一机制full、CAL/SEL选P | 回退R1仍可继续后续 |
| REPAIR_P | P已选且≠R1、R1修复有基础正信号、预算 | P上两头重新训练 | budget skip不阻已完成系统 |
| LOCK | 可选组件均终态、所需SEL完整 | 按有限列表锁定FINAL | 缺某支线可锁定其他合格系统，记录搜索缺失 |
| REPLICATE | LOCK、预算 | 固定step/配方seed46 | 不重选冠军 |
| CONFIRM | LOCK、DATA确认角色 | PB/LOCAL/ADDITIONAL | 单元失败继续其它；比较null |
| PROFILE | LOCK、固定6序列可读 | 实际最终路径成本 | 不假造0ms，独立状态 |
| REPORT | 所有可运行任务到终态 | 从文件生成报告与交接 | 部分完成也运行 |
| PUBLISH | REPORT、Git写权限 | 真实push/tag/Release | Release缺权限仍push Git并列缺失assets |

只有外部/预算/数值错误影响依赖时阻塞后继；例如repair只需TRAIN可读即可开始生成，CAL暂缺只阻其评测。调度器每次运行启动ready任务，最多规定并行量；无ready且无本次正在运行的任务时，生成partial report并发布，不忙轮询、无限等待或假称后台将自动继续。

### 14.2 新CLI必须实现并实际调用

```bash
export PERSIST4D_GAIN_V2_ROOT="${PERSIST4D_GAIN_V2_ROOT:-$HOME/persist4d_runs/perception_gain_v2}"

conda run -n persist4d python -m scripts.perception_gain_v2 bootstrap \
  --config configs/perception_gain_v2.yaml \
  --external-root "$PERSIST4D_GAIN_V2_ROOT"

conda run -n persist4d python -m scripts.perception_gain_v2 run \
  --config configs/perception_gain_v2.yaml \
  --external-root "$PERSIST4D_GAIN_V2_ROOT" \
  --target all --resume

conda run -n persist4d python -m scripts.perception_gain_v2 status \
  --config configs/perception_gain_v2.yaml \
  --external-root "$PERSIST4D_GAIN_V2_ROOT"
```

`--target`接受all或表内task_id；执行其依赖闭包，不把用户选REPAIR_R1展开成全部感知训练。`report/publish`子命令也须可在partial状态执行。失败重试用显式 `--retry-blocked TASK_ID`，只有依赖签名变化或完成针对性修复才再次运行；不改run identity清零费用。

每次真实子进程日志包含argv（去除凭据）、cwd、task/recipe、输入输出键、PID、GPU、start/end/exit、optimizer范围、耗时与计量scope。不记录只有 `run --through scorer` 的宽泛命令作为完整训练证据。

调度状态按任务/recipe记录，模型checkpoint引用只读；多worker不并发覆盖同一RUN_STATE，主进程统一汇总原子写入即可，不建设分布式状态数据库。

---

## 15. 最小充分测试与一次实际交付审核

测试保护会影响分数的行为，不做渗透、fuzz、成百上千schema负例或重复全回归。目标测试CPU墙钟≤30分钟；GPU smoke≤1 GPU-hour计入余额。已有正确的用例优先复用；参数化数量可多于下列项，但语义范围不扩张。

| 类别 | 最小需要覆盖 |
|---|---|
| 运行身份 | H/L分别进入训练和评价config；低预算不被“必须192”拒绝；source_commit真实 |
| 恢复 | 350回350，optimizer/scheduler/draw/rank RNG延续；低LR不会误resume高LR |
| 数据 | 真实stage映射；本地副本不偷偷改人口；partial覆盖分母保留 |
| 训练 | raw-sum不重复计loss；2个真实计划内step有参数变化；scorer冻结 |
| 查询/attention | Q=100；只开放一次且padding保留；原开关关闭等价 |
| 修复模式 | 同候选/初始化/抽样；NEW输入对old数值扰动不敏感；mode贯穿checkpoint/eval/profile |
| 修复映射 | 非恒等vertex permutation、不同segment partition、zero/no-op经真实bool路径一致 |
| 状态隔离 | 修复前后D0关联事件/score/class不变；不同头不共享可变archive；末帧/无old回退 |
| 评价 | official tau/ignore/ambiguity沿用现有测试；pooled汇总与完整分母 |
| 调度 | HIGH或scorer失败仍能跑R1修复/低LR；REPAIR_R1不依赖P锁定 |
| 选模 | 同LR C0-best；PAIR不靠NEW差checkpoint胜出；PB缺失为null而非false |
| 发布 | 必需asset清单、远端B/tag、实际下载/metadata验证级别，不以空Release冒充完整 |

开发期间只跑改动对应的测试；结束时一次相关回归、一次定向ruff、一次git diff --check。既有缺外部checkpoint的测试可以明确跳过/标环境前提，但不能谎称全部passed，也不去重建无关旧实验。

发现数值/映射bug须实际修复并重跑受影响小用例，不能以节约测试为由放过；不因此开展第二套通用安全审计。所有费用写总账，测试次数不是研究成果。

---

## 16. 交接和结果文件：重现过程，不只保存最终摘要

轻量内容入Git，模型/大型预测包进入第17节发布路径：

```text
artifacts/perception_gain_v2/
  EXECUTION_INSTRUCTION.md
  CODE_BINDINGS.md
  RUN_CONFIG.json
  DATA_ROLES.json
  INPUT_MANIFEST.json
  RUN_STATE.json
  EXECUTION_LOG.jsonl
  budget/LEDGER.jsonl
  foundation/BASELINE_COVERAGE.json
  foundation/BASELINE_METRICS.csv
  training/<recipe>/resolved_config.yaml
  training/<recipe>/run_plan.json
  training/<recipe>/run_summary.json
  training/<recipe>/metrics.csv
  training/<recipe>/checkpoint_manifest.json
  training/<recipe>/interruptions.jsonl        # 只在真实中断时生成
  refinement/PAIR_COVERAGE.csv
  refinement/REPAIR_DIAGNOSTICS.csv
  association/CONFIG_RESULTS.csv
  selection/CAL_SELECTION.json
  selection/SEL_COMPONENTS.csv
  selection/FINAL_LOCK.json
  confirmation/COVERAGE.csv
  confirmation/METRICS.csv
  confirmation/BY_REFERENCE.csv
  confirmation/CONFIRMATION_SUMMARY.json
  resources/RAW_PROFILE.csv
  resources/PROFILE_SUMMARY.json
  FINAL_REPORT.md
  HANDOFF.md
  ARTIFACT_MANIFEST.json
  RELEASE_PLAN.json
```

不存在的结果不创建内容为“PASS”的占位文件；report允许读取missing并写NOT_RUN。未执行PB也写0/129单元、0/516前缀，不写0/0。真实LOCAL/ADDITIONAL分母从原manifest填。

训练metrics至少保留实际step/LR/原loss分项/数据源计数与恢复切分；每种算法的所有已评价checkpoint分数均上传，不只冠军。原始训练checkpoint继续本地保存，可发布部署bundle而非每个中间optimizer。

报告至少8张表：
1. 每任务已开发/实际训练/终点/状态/阻塞，分清代码与实测；
2. H/L×C0/S-BAL完整四格与同step、相对R1差值；
3. 辅助pilot/full/CAL/SEL选择与未选择原因；
4. 两父模型上NEW/PAIR/parent、配对覆盖与修正/误修数量；
5. 关联12配置已测范围、CAL冠军、SEL比较；
6. 正式PB/LOCAL/ADDITIONAL/每reference/每seed及各自分母；
7. 真正部署成本、研究GPU/CPU成本与估计值标识；
8. Git/Release/权重/预测包真实发布覆盖和缺失项。

HANDOFF固定回答：起点/实际代码SHA与实验提交；资产定位与hash；真实配置与数据角色/暴露；代码→任务映射；每臂steps/metrics/失效结果；锁定配方及fallback；完整确认/资源范围；必要复现命令；每个可恢复任务的checkpoint、step和next command；测试范围与未通过项；Git和Release实际链接/receipt位置。

命令从执行日志生成，使用可解析的环境变量；不留猜测路径或未实现CLI。不要只写“恢复NFS再跑”而不注明具体恢复任务/资产/失败记录。

---

## 17. 实际GitHub同步、模型包和Release

### 17.1 早检查权限，晚执行发布

bootstrap时分别检查Git读写所需的现有连接和Release现有授权，不打印token。上轮没有gh授权是历史事实，不是本轮永远不可用；可使用已有gh登录态或已授权GitHub API。两者都不可用则明确Release权限阻塞，**不阻塞实验或Git内容推送**。不要求把密钥粘贴到聊天，也不扫描其它应用凭据。

只stage本轮变更清单，不对整个用户工作树运行git add -A。原始数据、完整GT、凭据不入Git；模型自有新增参数与脱敏预测可发布。科学负结果也要推送代码与报告。

### 17.2 要上传什么

- Git：代码、配置、相关测试、上述真实小型结果/曲线/交接/manifest。新增的小型scorer/refiner权重若≤20MiB，可连同显式输入模式manifest直接进Git，作为额外可取得的交付，不只留本地路径。
- Release：最终recipe的所有新增必要训练参数、同LR C0部署对照（若有）、selected PAIR的NEW控制头、可复算的大预测包和其清单。无新模型也发布结果包；部分完成可发布已训scorer及最佳探索checkpoint，标 `EXPLORATORY_NOT_SELECTED`。
- 原始R1/Concerto只引用base hash与原获取方式；不要重复分发未授权基础权重或数据。
- 大模型bundle优先存相对R1的**完整替换参数张量及变化buffer**，不是浮点差分相加。提供可运行loader、recipe、base hash，在一个已有真实panel上证明重载输出一致。
- 单asset≤1GiB，需分片时有顺序、每片及整体SHA；不得通过大量小Git片绕开模型包发布。

部署包不得只有weights没有NEW_ONLY/OLD_NEW、scorer、publisher配置。必需asset列表由FINAL_LOCK或partial训练记录生成；有已训可发布权重却assets=[]时不标完全发布。

### 17.3 提交、tag、draft、验证的无循环顺序

1. 提交实验代码与科学结果为A。
2. HANDOFF记录A；生成交付manifest（不hash自己、不纳入不断变化的总账/receipt），创建发布快照B。B的publication_phase=READY，不预写已上传。
3. 实际push V2分支；核对local B和remote branch B。创建新tag `persist4d-perception-gain-v2`；若无关占用选最小-r2后缀，不移动V1tag，不force push。推送并核对tag目标B。
4. 使用已有授权创建**draft prerelease**，显式repo/tag，tag预先已存在；gh采用 `--verify-tag --target "$B" --draft --prerelease --latest=false --notes-file ...`。不能让CLI缺tag时默认从main创建。[R04]
5. 上传所有必需assets和manifest；读取远端名称/bytes/digest。无digest时对必要bundle下载回读一次，不对所有旧文件循环校验。
6. 在draft阶段可上传 `PREPUBLICATION_RECEIPT.json`，明确它仅验证B/tag/当时assets，排除它自身的id/hash，状态为DRAFT_ASSETS_VERIFIED；不先宣称release已公开。
7. 发布draft，再读Release及必需assets，写本地external `publication/PUBLICATION_RECEIPT.json`作为最终回执，并在最终答复给链接。若仓库开启immutable release，不依赖发布后再附加receipt；必要文件均在draft期上传。[R04]
8. 最终receipt记录B、remote branch/tag、release id/url、asset实际可用性/bytes/hash/验证级别/时间。它不写回B制造自引用，也不对自己的上传id/hash递归验证。

Release存在不代表完整发布。`VERIFIED`要求所有必需代码/报告/模型包/结果包都在对应Git/Release可访问且验证通过；仅Git/部分头已推送但必需大包未上传为 `CODE_ONLY`并列缺失assets；Git推送失败为BLOCKED。metadata验证不能冒充端到端hash验证。

上传失败只做一次针对性修复重试；不删除无关Release或覆盖资产。partial状态下完成同样的Git发布，并给每个恢复任务的具体命令。

---

## 18. 最终状态与禁止混淆

分别记录，不用一个PASS覆盖所有方面：

```text
execution_status: COMPLETE | PARTIAL_WITH_BLOCKERS
perception_selection: NEW_PERCEPTION | CONTINUATION_ONLY | KEEP_R1
repair_R1_selection / repair_P_selection: OLD_NEW | NEW_ONLY | KEEP_PARENT | BLOCKED | SKIPPED_BUDGET
association_selection: CONFIG_ID | KEEP_D0 | BLOCKED | SEARCH_BUDGET_PRUNED
replication_status: SUPPORTED | MIXED | NOT_APPLICABLE | NOT_RUN_BUDGET | BLOCKED
pb_all_t_vs_native_fh / pb_all_t_vs_d0 / local_gain: true | false | null
resource_status: IMPROVED | TRADEOFF | REGRESSED | UNCONFIRMED
publication_status: VERIFIED | CODE_ONLY | BLOCKED
```

任务状态另用PENDING/RUNNING/COMPLETE/BLOCKED/SKIPPED_BUDGET/EXCLUDED_NUMERICAL。缺失对照/覆盖时比较=null，不是0或false。按本协议明确允许的budget skip不等于方法失败；核心训练/确认因外部或硬预算未完成仍是PARTIAL。

一轮完成但无增益允许 `COMPLETE / KEEP_R1 / NO_CONFIRMED_SUPERIORITY`；不要无限加模块凑正结果。只达到代码测试、部分pilot则如实partial；需要真实图表而非state文字来说明效果。

**本轮最终答复必须给用户：** actual branch/B/tag、FINAL_REPORT、HANDOFF、Release/可下载模型包链接，核心质量差值与对应人口，所用训练步数，是否真的超过FH-R1，真实成本，未完成任务及最小恢复命令。不得只报测试数与提交成功。

---

## 19. 执行结束前的一次审核

用真实文件回答以下问题，将结论写入HANDOFF；不用再造大审计系统：

1. 是否从固定B出发，V1历史未覆盖？实际运行代码版本是否不再写死父SHA？
2. H/L、主模型与scorer/refiner的训练步数是否分别记录？350/379是否区分？
3. LOW_PAIR和R1两头是否独立执行，旧全五臂validator是否不再阻断？
4. 同LR/同runtime C0、NEW_ONLY控制是否存在且公平？
5. NEW/PAIR的支持范围与候选相同，模式是否传入训练/评价/profile/bundle？
6. 对齐和zero残差是否经过真实postprocess，state/score/class是否没变？
7. CAL/SEL基线是否真的完整，而不仅是smoke？PB是否在锁定之后？
8. 修复头换父模型后是否重新生成数据与训练？A*是否保持独立候选？
9. 剩余预算是否减掉V1与阻塞占卡，最终确认预留是否未被搜索挪用？
10. 质量是否由official pooled metric计算，缺失行分母和null是否正确？
11. 模型包是否实际上传、可按base hash恢复；是否避免空Release的虚假VERIFIED？
12. 最终失败、未训练、预算剪枝、外部阻塞、一般修复收益、历史证据收益是否分别表述？

本文是可执行研发规范；正确性与增益必须由Codex在实际环境执行上述步骤产生，不能把本次文档审核当作已完成训练或服务器验证。

---

## 20. 来源索引

源码均绑定B提交，少数V1事实使用同提交内保存的历史产物；外部只采用原论文/官方文档。E编号是本文的可回查来源，不是新实验结果。LaSSM仅支持语义—空间查询的设计动机，不把其静态数据集增益搬到本项目。[R02]

| 编号 | 来源 | 本文使用的事实/接口 |
|---|---|---|
| [E01] | V1最终报告：`artifacts/perception_gain_v1/FINAL_REPORT.md` | actual steps、CAL、未执行确认及profile |
| [E02] | V1运行状态：`artifacts/perception_gain_v1/RUN_STATE.json` | 11.803828054742961累计已记GPU-hours、scorer、partial状态 |
| [E03] | S-BAL实际中断：`artifacts/perception_gain_v1/training/S-BAL/INTERRUPTION.json` | 350 resume、379未保存、NFS/NCCL及估计费用 |
| [E04] | Scorer训练摘要：`artifacts/perception_gain_v1/training/scorer/RUN_SUMMARY.json` | 500 updates、64 refs、梯度/参数变化 |
| [E05] | 各checkpoint CAL结果目录：`artifacts/perception_gain_v1/training/pilot/evaluation/cal` | C0 0/250/750、S-BAL250、Q-SEM0；本目录下每个JSON是数值来源 |
| [E06] | 修正E1门槛：`artifacts/perception_gain_v1/foundation/e1/gate.json` | 94/337、ASSOCIATION_HEADROOM_SUPPORTED、ambiguity信息不完整 |
| [E07] | Live smoke及旧缓存比较：`artifacts/perception_gain_v1/foundation/LIVE_SMOKE.json` | cached/live FAIL，live repeat PASS；不能拼接跨producer结论 |
| [E08] | 感知/修复模型实现：`models/perception_gain.py` | SemanticQueryScorer、CausalMaskRefiner、pair_adjacent_soft_occurrences、阶段损失 |
| [E09] | 旧总编排：`scripts/perception_gain_campaign.py` | load_campaign_config固定192/runtime；pilot/full固定臂数与覆盖 |
| [E10] | 实际感知评价入口：`scripts/perception_gain_evaluation.py` | run_checkpoint_evaluation、_load_evaluation_weights、live producer与固定source/root |
| [E11] | V1代码绑定表：`artifacts/perception_gain_v1/CODE_BINDINGS.md` | 用于定位，但缺真实函数起止行；V2需补齐 |
| [E12] | Criterion实际接入：`models/criterion.py` | loss_masks/_loss_masks_by_stage、aux、stage_debug与legacy路径 |
| [E13] | 查询/注意力前向：`models/rescene.py` | initialize_queries、semantic_query_positioning、open_first_cross_attention |
| [E14] | 修复训练和运行回调：`scripts/train_perception_refiner.py` | align_old_probabilities_to_new_segments、CausalRefinerRevisionTransform、训练目标 |
| [E15] | Soft证据与物化：`scripts/rescene_task_postprocess.py` | extract_official_task_prediction、materialize_segment_logits；bool与heatmap路径 |
| [E16] | 有限关联配置：`models/overlap_entity_association.py` | preregistered_association_configs返回3+3+6，AssociationConfig |
| [E17] | 关联/输出重放：`scripts/replay_crosswindow_association.py` | canonical输入、独立publisher/HistoryBoundary、D0桥接 |
| [E18] | V1普通感知训练器：`trainer/perception_gain_trainer.py` | configure_optimizers、progress、checkpoint恢复、runtime合同 |
| [E19] | V1训练入口与确定性数据流：`scripts/train_perception_gain.py` | compose_variant_config、_build_loader、run、PerceptionDrawDataset、checkpointcallback |
| [E20] | V1发布器：`scripts/perception_gain_publish.py` | 固定roots/branch、stage清单、Receipt判定与assets |
| [E21] | 基础绑定与历史/实时资产区别：`scripts/perception_gain_foundation.py` | run_corrected_e1读取旧cache，_resolve_cache_assets强制旧缓存键 |
| [E22] | V1数值配置：`conf/perception_gain_v1/common.yaml` | LR5e-5、raw_sum、frozen_encoder_eval=false、batch32、3000调度 |
| [E23] | 父模型修复数据生成：`scripts/prepare_perception_refiner.py` | run显式parent、advance_d0_identity、32refs清单 |
| [E24] | 修复评价与旧schema：`scripts/perception_refiner_evaluation.py` | _load_refiner严格旧键；V2须贯穿mode/parent来源 |
| [E25] | Native FH评价入口：`scripts/perception_gain_native_evaluation.py` | 完整prefix producer与输出身份 |
| [E26] | LOCAL-T2入口：`scripts/perception_gain_local_evaluation.py` | 原生T2独立人口与manifest |
| [E27] | 官方metric adapter：`scripts/p6a_metrics.py` | OfficialMetricAccumulator、official_temporal_match_trace、阈值来源 |
| [E28] | 真实profile实现：`scripts/perception_gain_profile.py` | 真实网络/状态/物化及scope |
| [E29] | D0的修订接口：`scripts/task_memory_output.py` | RevisionMaskRequest、LagOnePublisher、logical/class/generation归档 |
| [E30] | 冻结数据角色：`artifacts/perception_gain_v1/DATA_ROLES.json` | TRAIN/CAL/SEL/PB/LOCAL/ADDITIONAL对应关系 |
| [E31] | V1已读执行规范：`artifacts/perception_gain_v1/EXECUTION_INSTRUCTION.md` | V1协议、运行约束与V2显式调整的来源 |
| [E32] | V1实际执行日志：`artifacts/perception_gain_v1/EXECUTION_LOG.jsonl` | 测试范围与耗时、RPC错误、partial发布 |

[E01]: https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/artifacts/perception_gain_v1/FINAL_REPORT.md
[E02]: https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/artifacts/perception_gain_v1/RUN_STATE.json
[E03]: https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/artifacts/perception_gain_v1/training/S-BAL/INTERRUPTION.json
[E04]: https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/artifacts/perception_gain_v1/training/scorer/RUN_SUMMARY.json
[E05]: https://github.com/Orangekostar/Persist4D/tree/8b5e93795817682fe70864daa545db219e1443c9/artifacts/perception_gain_v1/training/pilot/evaluation/cal
[E06]: https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/artifacts/perception_gain_v1/foundation/e1/gate.json
[E07]: https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/artifacts/perception_gain_v1/foundation/LIVE_SMOKE.json
[E08]: https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/models/perception_gain.py
[E09]: https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/scripts/perception_gain_campaign.py
[E10]: https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/scripts/perception_gain_evaluation.py
[E11]: https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/artifacts/perception_gain_v1/CODE_BINDINGS.md
[E12]: https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/models/criterion.py
[E13]: https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/models/rescene.py
[E14]: https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/scripts/train_perception_refiner.py
[E15]: https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/scripts/rescene_task_postprocess.py
[E16]: https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/models/overlap_entity_association.py
[E17]: https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/scripts/replay_crosswindow_association.py
[E18]: https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/trainer/perception_gain_trainer.py
[E19]: https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/scripts/train_perception_gain.py
[E20]: https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/scripts/perception_gain_publish.py
[E21]: https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/scripts/perception_gain_foundation.py
[E22]: https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/conf/perception_gain_v1/common.yaml
[E23]: https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/scripts/prepare_perception_refiner.py
[E24]: https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/scripts/perception_refiner_evaluation.py
[E25]: https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/scripts/perception_gain_native_evaluation.py
[E26]: https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/scripts/perception_gain_local_evaluation.py
[E27]: https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/scripts/p6a_metrics.py
[E28]: https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/scripts/perception_gain_profile.py
[E29]: https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/scripts/task_memory_output.py
[E30]: https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/artifacts/perception_gain_v1/DATA_ROLES.json
[E31]: https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/artifacts/perception_gain_v1/EXECUTION_INSTRUCTION.md
[E32]: https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/artifacts/perception_gain_v1/EXECUTION_LOG.jsonl

[R01]: https://arxiv.org/html/2601.11508v2
[R02]: https://arxiv.org/abs/2602.11007
[R03]: https://docs.pytorch.org/docs/2.14/data.html
[R04]: https://cli.github.com/manual/gh_release_create

外部来源用途：R01为原指标定义；R02为查询机制的邻近参考；R03仅核对DataLoader timeout/persistent_workers的接口语义，不要求升级到文档版本；R04核对显式tag、draft与immutable发布顺序。未进行新服务器实验，文档数值来自上述已发布产物，新增阈值与配方为本轮规定。
