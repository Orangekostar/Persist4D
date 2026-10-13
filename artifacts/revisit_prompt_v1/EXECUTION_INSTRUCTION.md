# Persist4D MPR-V1：Codex 最终执行指令

**任务**：实现、训练、选择、评价并发布“历史提示驱动的复访分割”第一轮有预算上限的实验。

**文档日期**：2026-10-13。**指令版本**：1.0。**语言**：代码遵循仓库风格，交接与实验结论使用中文。

**唯一历史起点**：`Orangekostar/Persist4D`，提交 `6b1b779172afdeb51e2719a737cd05e7eeeca982`，标签 `persist4d-cvpr-evidence-20261011-v1`。

**执行性质**：这是完整的开发—实验筛选—复验—发布任务，不是只写代码或只提出计划。完成必须有实际记录；条件不满足时，完成可完成的开发和发布，明确标记实验阻塞，不能冒充实验完成。没有任何条款要求新方法一定涨点，也没有条款授权新增 GPU 总预算。

---

## 0. 首先执行的总要求

你是负责本任务的 Codex 主执行者。按本文顺序工作，不再重新发散选题。除本地资产确实缺失、授权额度不足、运行出现实质性错误或远端访问失败外，不以泛化“安全检查”阻止执行；这些情况也必须生成交接文件并发布已经完成的工作。

1. 先建立基于固定提交的独立 worktree/分支，核实本文引用的关键源码和资产，再实现新入口。不要修改旧报告、旧权重、旧标签和旧成本账本。
2. 实现本文标为 **NEW** 的文件、函数和命令；它们是开发目标，不是已经存在的工具。不得直接运行尚未实现的命令，再把命令不存在当作仓库环境故障。
3. 使用原有点云与 Concerto 环境、官方指标和部署资产。此次不下载 SAM/VGGT 大模型、不采购算力、不安装全新 CUDA 栈、不移植完整外部基础模型。
4. 所有模型使用相同输入政策、100 个 query 和共同输出规则。历史状态只来自预测；GT 仅用于损失、训练侧标签账本和评价。
5. 所有执行结果都有代码版本、输入身份、权重、配置、步数、随机种子和成本对应关系。只保留对科学结论有用的检查，不建设新的审计平台。
6. 必须执行选模、生成 `REPORT.md` 和 `HANDOFF.md`，实际推送 GitHub，并核验远端内容。`git commit` 或上传 LFS 指针不等于完成发布。
7. 先严格遵守冻结的实验矩阵；不根据 CAL/SEL/PB 的偶然结果无限追加变体，不删负结果，不把短适配训练称为 29,700 步完整复现。

本文规范的顺序优先级：用户明确的预算/权限更新 > 本文固定实验契约 > 新配置中的默认值 > 历史脚本默认值。历史脚本不能反过来覆盖本任务。

---

## 1. 已确认的事实、研究判断和未核实条件

### 1.1 已确认事实

| 事实 | 依据 | 对任务的影响 |
|---|---|---|
| E0–E3 都完成 29,700 步；仅有训练种子 45 | R01、R02 | 本轮是从冻结父模型开始的新适配实验，不是恢复原四臂训练 |
| 可发布的最佳 E1 任务权重为 `E1-23760.pt`；编码器另供 | R09、R10 | 使用该父模型，完整加载全部任务参数及缓冲区 |
| 已有 `TaskMemoryRead`、proposal-anchored read 和多代表视觉记忆 | R04–R06 | 不把普通 memory attention 或多代表存储当作新实现 |
| 旧 M3-V-CORE 在多个强对照上失利 | R07 | 本轮不继续训练 M3-V-CORE；旧结果仅作取舍依据 |
| 原生新进程可能产生不同浮点预测，两个定向异常输入的 AP/REC 相同 | R01、R09 | 必须记录数值边界；不能把固定随机种子等同于逐张量可重复 |
| `NativeEpisodeMaster` 的旧扫描命名受正则限制，而新清单使用 UUID | R08、R11 | 不能直接把 UUID 送进旧 episode 构造器；新适配层保留真实 UUID |
| 旧加载器校验数值源码身份 | R10 | 模型改动后需要新加载契约，不能屏蔽旧检查 |
| 历史预算 1,800 GPUh，已耗 1,713.8830547291614 GPUh，预留 8 GPUh | R01、R02 | 历史可用余额 78.11694527083863 GPUh；执行前仍需核实最新余额 |
| CAL、SEL、PB 等已有研究暴露；PB 只有六个物理场景 | R01、R08 | 本轮结果最多是历史面板上的新实验确认，不自动成为全新外部证据 |

### 1.2 本轮提出、尚未验证的研究假设

**H1**：历史实体在首个掩码产生之前检索当前稠密特征，能补回部分当前感知器漏检/漏分的实体。

**H2**：从当前检索位置产生查询位置，比只传播历史描述与旧位置更适合搬动、长缺席后的复访。

**H3**：用当前观测的实体兼容证据学习拒绝错误恢复，可以减少相似对象造成的错误重激活。

这些假设不是已证实结论。没有足够对应数据时，不给它们填写“PASS”。

### 1.3 不能预先假定的事项

服务器当前空闲情况、SSH 可达性、本地数据路径、编码器路径、最新预算余额、是否有真正未暴露的外部数据，均由执行环境核实。不得把本文中的历史路径或聊天记忆当作已存在的文件。本文没有确认 RGB—点云投影资产，故不以它们作为执行前提。

---

## 2. 本轮研究边界和新颖性边界

方法工作名：**Memory-Prompted Revisit Segmentation，MPR-V1**。

要改变的不是骨干网络品牌，而是历史信息进入当前感知的时间：

```text
已有路线：当前初步掩码 → 当前 query 路由到历史槽 → 读取记忆 → 后续解码
本轮路线：历史槽 + 当前稠密特征 → 当前证据检索 → 提示 query → 首个掩码 → 后续解码
```

SAM-V 将几何条件用于掩码解码，而非仅用于事后匹配；MOTR 已有跨帧 query 传播；S4VY 已有共享视觉几何上的持久对象查询。它们是启发和近邻，不是我们已经复现的系统，也不证明本方法新颖。[P01–P03]

本轮争取的差异是：**无当前合格 proposal 前提的历史检索、由当前证据重新定位提示、以及预测驱动状态下的拒绝学习**。报告只能按实验实际支持的范围论述。不得写“首次持久查询”“首次长期记忆”“ReScene4D 只能 T=2”。

为保证本轮可完成，明确不做：全量外部基础模型复现、RGB 教师蒸馏、新数据采集、无限长度训练、容量淘汰策略搜索、E2/E3 续训、完整 CVPR 投稿包和大范围环境兼容性测试。这些不是遗漏，而是此次预先排除的独立项目。

---

## 3. 固定版本、工作区和资产解析

### 3.1 分支

新分支首选：`research/persist4d-mpr-v1-20261013`。
新标签首选：`persist4d-mpr-v1-20261013-v1`。

在现有 checkout 内先读取 `git status --short`、`git remote -v`、`git rev-parse --show-toplevel`、`git worktree list`。原目录有修改时不 stash、不 reset、不清理，直接另建 worktree。新分支必须基于固定 BASE；如果同名分支已经属于本任务，核验任务身份后恢复；如果属于别的任务，使用最小未占用整数后缀 `-r2`、`-r3`，记录实际名称。不要 force-push，不改主分支。

远端只允许确认后的 `Orangekostar/Persist4D`。不要将另一个远端设置为 origin 以绕过权限问题。

### 3.2 解析本地资产

NEW `scripts/revisit_prompt_campaign.py resolve --auto` 按以下优先级解析：

1. 本任务已存在且有效的 `assets.local.json`；
2. 当前进程已设置的相关环境变量；
3. 当前仓库历史交接/配置实际指向的资产解析文件和运行根；
4. 发布目录中的 Git LFS 任务权重与已有编码器缓存。

只搜索这些具体根目录，不扫描整个 `/home`、全磁盘或全部节点。多个候选匹配同一字节哈希时按解析后绝对路径字典序选第一个；不同内容不能凭文件名选。

必须绑定：`repo_root`、`run_root`、`data_root`、`encoder_path`、`parent_bundle`、`taxonomy_path`、`population_path`、可用设备和现有授权来源。`run_root` 默认设为新 worktree 同级的 `persist4d_mpr_v1_runs`；`data_root` 等无可靠候选时不发明路径。

父任务权重：`artifacts/native_long_postselection_v1/deployment/E1-23760.pt`。
权重清单：同目录 `MODEL_ASSET_MANIFEST.json`。
编码器 SHA-256：`845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07`。
分类规范：同目录 `rio.yaml`，哈希从清单读取。
训练/开发清单：`artifacts/native_long_retrain_v1/POPULATION.json`。
PB清单：`artifacts/native_long_postselection_v1/protocol/FORMAL_POPULATION.json`，仅读取其PB记录，原LOCAL/ADDITIONAL不加入本轮。
数据解析还必须绑定 `formal_population_path` 和训练/validation 两种索引空间。

只对本次选用的父权重、编码器、规范及清单做一次完整身份核验；后续加载复用已绑定身份。可以用已有 Git LFS 下载所需任务权重。不能把 LFS 指针当模型；不能声称部署包包含 optimizer 状态。

敏感路径和访问凭据仅在被忽略的 `assets.local.json`，公开文件用相对路径和逻辑名称。仅核查本次待提交文件是否夹带凭据，不扫描所有历史日志。

### 3.3 只读源码绑定

生成 `SOURCE_BINDINGS.json`，逐项保存 R01–R23 中本次用到的路径、BASE、实际 Git blob SHA、符号、用途和阅读范围。读取不足的相关函数，在实现前扩展阅读；不把“目录里有这个名字”当作理解了函数。

代码修改只覆盖 §4 的新文件和明确列出的最小插入点。第三方数值修复如确有必要，记录原版本、实际差异和作用范围；不得悄悄升级整个环境。

---

## 4. 任务与源码/新文件绑定

下面的 NEW 名称是必须实现的接口契约。可合并内部私有函数，但不能缺失命令或公开职责。

| 任务 | 既有源码/符号 | 处理方式 | NEW 产物 | 完成依据 |
|---|---|---|---|---|
| 父模型迁移 | R09–R10，`load_portable_bundle`、`portable_config` | 保留旧加载器；新增白名单迁移 | `scripts/revisit_prompt_assets.py` | 所有旧任务参数/缓冲区覆盖，新增参数单列 |
| 原始扫描到因果窗口 | R08、R11，`build_native_datasets`、`NativeLongCollator`、`StageMeta` | 重用 native 加载与索引；新 episode 描述接受 UUID | `datasets/revisit_prompt_episode.py` | 真实 UUID、局部 t、绝对阶段、顶点映射对齐 |
| 首掩码前提示 | R03，`initialize_queries` 和 `forward` 首次 `mask_module` 之间 | 最小新增默认空操作 hook | `models/revisit_prompt.py`；修改 `models/rescene.py` | 提示不读取当前初步掩码，backbone 只前向一次 |
| 检索及当前几何 | R03 的 `batched_features`、raw coordinates、point2segment | 当前段特征全局检索，再局部聚合 | `models/revisit_prompt_modules.py` | padding/旧扫描排除，检索和提示有真实梯度 |
| 历史附加几何 | R13 的 `TaskMemoryState`/commit 使用方式 | 小型并行 sidecar；不发明 GT 状态 | `models/revisit_prompt_state.py` | 只由已接受预测更新，重置与状态同步 |
| 共同关联/提交 | R13，`route_entities`、`commit_entities` | 所有变体都按最终输出重新路由；一次提交 | `scripts/revisit_prompt_runtime.py` | ID 不等于 query 槽；没有重复提交 |
| 旧读取对照 | R04–R05，`TaskMemoryRead`、`after_decoder_stage` | 迁移行为，不迁移旧训练结果 | 同上模型与配置 `M2` | 相同 E1 父权重、训练长度和最终发布规则 |
| 损失与训练标签账本 | R12、R14，`SetCriterion`、独立匹配、训练 ledger 思路 | 主任务损失重用，检索/拒绝监督新增 | `models/revisit_prompt_loss.py` | 标签只在损失侧，空集 loss 有定义 |
| 受限适配训练 | R12、R15，stage mean、预算与进度机制 | 新独立 trainer，不调用旧硬编码 R1 CLI | `trainer/revisit_prompt_trainer.py`、`scripts/train_revisit_prompt.py` | 真实 optimizer 更新、父参数冻结、可恢复 |
| 评价和发布语义 | R16–R18，官方后处理、轨迹累计、共同 GT | 小适配器复用，不重写 AP 算法 | `scripts/evaluate_revisit_prompt.py` | class/query 映射及所有分母可追溯 |
| 选模 | R19 的 CAL 选择思想 | 不复用旧 seed/arm 硬编码 | `scripts/select_revisit_prompt.py` | 有唯一排序、失败/并列/缺失规则 |
| 调度与成本 | R20，`remaining_budget`、`charged_process` | 明确 prior + 新事件；保留总上限 | `scripts/revisit_prompt_campaign.py` | 每个 GPU 子进程收费，预算不足不扩额 |
| 结果与 GitHub | R21，旧发布组织方式 | 发布新分支、新 tag、新模型和交接 | `scripts/publish_revisit_prompt.py` | 远端提交、LFS 实体字节、交接可回读 |
| 配置/必要测试 | 原配置 R22 | 父解析配置继承 + 本文覆盖 | `conf/revisit_prompt/default.yaml`、`tests/test_revisit_prompt.py` | 一次聚焦回归，修复后只重跑受影响测试 |

无需为每个输出单独建立“验收代理”、审计进程、签名服务或监控服务。编排器是一个主进程及受预算约束的工作进程。

---

## 5. 父权重迁移与冻结范围

### 5.1 必须保留的父模型内容

E1 任务包并非只有 Transformer 解码头。`ENCODER_PREFIXES` 仅排除了 `model.backbone.model.embedding.` 与 `model.backbone.model.enc.` 的冻结参数；Concerto 的任务解码部分以及其余任务权重必须加载。编码器相关可变缓冲区以部署包记录为准，不能误删。

在干净 BASE worktree 使用旧加载器核实父资产一次，然后在新代码中：

- 读取父包、核验父字节哈希及其原始数值源码身份。原身份与 BASE 下相关文件对应，新源码另外绑定。
- 从父包 **已解析 config** 初始化新系统，显式覆盖新模型 target、路径和新适配配置。不要用旧 YAML 的 batch 字段推断历史实际 batch。
- 所有原参数和缓冲区必须匹配名称、形状、dtype 并精确加载。仅允许 NEW 提示/拒绝模块命名空间缺失；不得宽泛 `strict=False` 后忽略返回项。
- 将两个 backbone 路径引用和 metric taxonomy 一并重定位，参考 `portable_config`。
- 不能把 `code_identity()` 改成恒真，不能修改旧权重 metadata 来伪造匹配。

### 5.2 本轮训练政策

冻结 **全部父模型参数**，包括原任务解码器；只训练新增提示/读取/拒绝参数。父模块保持 `eval()`，新增模块置 `train()`。这与旧 E1 的训练态冻结政策不同，报告必须说明。

冻结参数不等于整条前向 `no_grad()`：提示进入解码器后，必须允许梯度穿过冻结的解码运算回传到提示参数。只可在与提示无关的冻结 backbone 前向部分使用 `no_grad()`。不要用 `inference_mode()` 产生随后用于反向的中间张量，除非已复制为普通张量并验证可反向。

所有M0–M6都显式设 `return_query_features=True`，因为共同状态需要最终query特征；该输出开关不改变父预测数学路径。保持 `save_segment_info=False`，不启用旧native_long反馈/采样实验开关。输出开关、父模块冻结与新增模块训练三者分别记录。

本轮研究的是**固定感知器上的提示适配**，不是充分微调整个模型后的最佳可达性能。seed146/147 是新适配种子；父模型仍来自 seed45，不能称为两个独立预训练感知器。

---

## 6. 数据与因果 episode 契约

### 6.1 保持原物理场景分组

读取原 `POPULATION.json` 的完整结构。已知 TRAIN=361、CAL=8、SEL=8；CAL 的原计数 T1/T2/T3/T4/T5=31/19/14/14/6，SEL=39/21/21/15/15。[R08]

这些计数是原评价输入数，不是新流式 replay 的前向次数。新清单必须单列输入数、阶段前向数和独立物理场景数。不得按窗口随机划分导致同场景泄漏，不得重新调用旧分组函数挑一个更有利的新拆分。

不从 TRAIN 中“抽出”场景并称为父模型未见：E1 已在这些场景训练。本轮可用原 CAL 选点、原 SEL 做锁定候选检验、原 PB 做历史面板复测；它们都标记 `HISTORICALLY_EXPOSED`。

### 6.2 新 episode 描述

新 dataclass `RevisitEpisodeSpec` 接受真实扫描 UUID 元组和原清单索引，不使用以 `-` 拆分 UUID 的逻辑，不伪造 `scene####_##`。

每条 episode 长度 H=2、3、4 或5，包含同一 reference 的 H 次不同真实扫描。按已锁定清单的因果顺序处理：

```text
绝对阶段 0: [scan_0]，局部 t=0
绝对阶段 j>0: [scan_(j-1), scan_j]，局部 t=0,1
历史状态在进入阶段 j 时仅含 0..j-1 的已接受预测。
```

`StageMeta` 本身不要求旧扫描正则，可按其实际字段复用；新 spec/采样器不能调用旧的受限 `NativeEpisodeMaster` 校验。`absolute_stage_index`、`local_stage_ids`、`segment_stage_ids`、顶点原索引和 inverse map 必须分开。

### 6.3 固定训练 draw

每个 optimizer 更新累积 **4 条 episode**，顺序固定为 H=2、3、4、5，因此每次更新有14个阶段前向。每个长度从原 TRAIN 中 Tmax≥H 的 reference 均匀抽取，随后按 `stable_seed(seed, update, H)` 选择原 `canonical_orders` 中的一种顺序及连续 H 个位置；参考场景抽取和位置抽取都记录到 `DRAW_PLAN.jsonl`。

TRAIN/CAL/SEL通过原native训练数据库解析其索引；**PB改用 `native_long_confirmation_inference.formal_dataset` / `formal_samples` 的validation索引与aliases校验**，保留其原ambiguities。不能拿训练数据库的同号index读取PB。UUID是共同物理身份，索引是带dataset scope的定位值；公开清单同时保存两者。[R25]

各变体同一种子的 draw 文件完全相同。seed146 与147各自生成独立 draw。不要使用旧 `horizon_weights()`：旧课程初始10%只有T2，本轮短适配若照搬会没有真实长序列训练。

此次仅用 3RScan/rio 原 TRAIN 适配；不默认引入 ScanNet、RGB 或伪造的长时标签。父模型此前含混合训练，报告单列。主训练关闭随机几何/颜色增强，使跨窗口几何一致、避免引入未验证的增强调用差异；这不是全量训练的最优配方主张。

CAL、SEL 使用原固定 `evaluation_inputs`。对每条 H 输入从空状态 replay H 个阶段，只在其指定 H 端点记一次指标。不同输入不共享实体状态。允许完全相同、已经绑定的 CPU 数据缓存，不允许按 GT 合并输入或跳过困难输入。

### 6.4 标签语义和公共真值

官方评价必须使用完整**因果前缀**的共同真值，再投影到当前扫描/局部窗口；不能让 full-history 与 local window 分别从不同“首点类别”规则得到不同类别。参考 R18 的共同真值与顶点对齐逻辑。

构建真值是 CPU 评价/损失侧操作，不能输入模型、检索器、槽分配器或关联器。训练局部窗口的目标类别也使用同一因果前缀的类别语义投影；不能使用未来阶段修正当前标签。

`point2segment` 是现有几何超点映射，可作为输入结构；标签、GT 实例 ID、训练账本和 future scan 列表不可进入提示模块。保持 `save_segment_info=False`，不利用其 GT 导出支路。

---

## 7. 共同状态、历史查询分配和实际身份

### 7.1 共同状态

所有 M0–M6 都使用同一个 `TaskMemoryState` + `route_entities` + `commit_entities` 输出政策，容量 K=100，特征维 D=128。配置固定：class_weight=0.25、association_threshold=0.5、update_mode=confidence_ema、update_rate=max_update_rate=0.2；confidence_threshold=0.5、mask_threshold=0.5、minimum_mask_support=1。background 类索引从实际父配置解析并校验，不把数字18无条件塞给不匹配的 taxonomy。

这套共同政策使用最终输出重新路由，且采用 `commit_entities` 的出生排序，因此 M0 命名为 **E1+COMMON-STATE**，不能冒充历史报告原 `LOCAL-B4` 的逐实现复现。需要单列这个区别；旧表的数值不搬进新主表。

附加 `RevisitGeometryState` 只保存每槽的最近一次预测几何描述 `g[6]`、centroid_xyz[3] 和 valid[1]；与 logical_id/generation 对应。只在共同提交接受该槽当前观测之后更新，不改变旧关联打分。容量满时沿用旧拒绝出生政策，本轮不增加淘汰模块，不声称验证了长期容量饱和。

### 7.2 槽到提示预算

每阶段最多使用32个历史槽，至少保留68个新对象查询。原总 Q=100 不改变。

- 从进入阶段前的状态选槽，不能依赖本阶段掩码或 GT。
- 最近活跃集合：`last_seen == j-1`；休眠集合：`last_seen < j-1`，只含 occupied 槽。
- 先从两组各取最多16个，组内按 `(-confidence, -last_seen, logical_id, slot_index)` 排序；不足部分用未入选 occupied 槽按同一排序补至32。
- 去重后按 `(logical_id, slot_index)` 排列，映射到 query 索引 `0..m-1`；其余 query 保留父初始化。
- 空状态、无 occupied 槽时，新模块是严格空操作，不替换原 query/position，也不人为占用32个空槽。

此处的提示来源槽仅表示“这条查询受谁启发”，不是最终 tracking ID。最终身份必须由共同最终关联与提交产生，不能直接令 `track_id = query_index` 或强制继承提示来源身份。

### 7.3 共同提交顺序

每阶段必须依次执行：读取旧状态 → 构造提示/完成一次模型前向 → 计算训练损失（仅训练）→ 从最终输出构造 `PredictionObservation` → 最终 `route_entities` → `commit_entities` 一次 → 当前官方候选绑定 logical ID → 写入当期输出 → 更新几何 sidecar → detach 状态。

M2 的中间路由只是读取用；最后仍按最终输出重新路由。M2同样只允许§7.2选中的至多32个历史槽影响查询：正常计算中间route/read后，仅对来源槽在已选集合中的query施加read残差，其余query保持原值。不要改写route哈希来硬凑限制；最终route仍重新计算。这是**适配后的 proposal-read 对照**，不是旧 Q-TALA/TALA 训练协议原样复现。主训练所有变体都使用独立匹配，不偷偷只给 M2 增加 TALA。

## 8. 新模型的计算契约

以下维度、邻居数和损失权重是**此次固定实验设计**，不是从论文复制的已验证最优值。只有本文的预算准入和异常分支可自动调整；不能自行网格搜索它们。

### 8.1 插入点与数据形状

在 `ReScene.forward` 调用 `initialize_queries` 后、第一次 `mask_module` 前新增默认空操作 `before_first_mask(...)`，返回更新后的 `(queries, query_pos, sampled_coords)`。参数至少包括已初始化查询、稠密段特征、padding 信息、raw coordinates、point2segment 以及本次 label-free StageMeta。

- `queries/query_pos`：`[B,100,128]`。
- 当前稠密特征来自现有 `batched_features`，维度 `[B,S,128]`，不是已经筛出的 proposal。
- 使用 StageMeta 和 point2segment 构建与这些 S 行严格对应的局部阶段索引、段中心坐标。padding 和上一扫描段不能进入当前检索集合。
- 段中心用实际 raw XYZ 的 scatter mean；不能直接把稀疏整数网格坐标当成米。
- 空 hook 不增加参数、不改变旧执行顺序。新功能默认关闭。NEW 子类 `RevisitPromptReScene` 才启用。
- 新系统保留 `InstanceSegmentation` 所需的官方后处理方法；NEW trainer继承该基础类或显式复用其接口，不能用一个裸ReScene对象冒充含 `get_mask_and_scores` 等方法的system。
- 新模块参数命名空间固定为 `model.mpr_prompt.*`、`model.mpr_read.*`、`model.mpr_support.*`，分别用于提示、M2读取和M6拒绝；不存在的分支不要注册虚假可训练参数。
- 父/新模块的train/eval切换应有显式实现，保证外部 `system.train()` 不会重新打开冻结父模块的随机训练行为。
- 可增加只用于受控机制 replay 的 `backbone_outputs` 可选入口；默认实时路径仍只调用一次原 backbone。该入口必须保存必要的字段/索引并使用克隆，不能把它变成 GT 旁路或长期无限缓存。

### 8.2 检索：从整个当前扫描寻找，再形成局部证据

对已选槽 i 的历史描述 `e_i∈R^128`：

```text
a_is = dot(normalize(Wq(e_i)), normalize(Wk(f_s))) / 0.1
anchor_i = argmax_s a_is
N_i = 当前段中心到 anchor_i 中心最近的 min(32, S_current) 个段
w_is = softmax(a_is, s in N_i)
z_i = sum_s w_is * f_s
c_i = 当前 anchor 段的 XYZ 中心
```

`Wq/Wk` 为有偏置的 Linear(128,128)。全局相似度可按槽分块计算，不构建 `S×S` 注意力矩阵。最近邻和 argmax 的并列由原段索引升序打破。训练允许离散 anchor/邻居选择，但选中相似度和加权 z 不得 detach；M5/M6 的检索损失还直接监督全局相似度。

不以旧位置作硬门控；不从 GT 采样提示点；不先要求当前 mask IoU 或 proposal confidence 达标。当前 S=0 时保持父查询并记录空检索，不制造零坐标假对象。

### 8.3 低成本几何描述

定义统一 `geometry6(P)`：对 P 的段中心，计算质心、除以 max(n,1) 的协方差；取非负特征值升序的平方根3维，以及到质心距离的 mean/std(max用总体方差)/max 三维，对这6个非负数逐项 `log1p`。n=1 返回全零描述，centroid 为该点；n=0 返回 invalid。

当前几何 `g_c=geometry6(N_i)`；历史几何 `g_h` 来自该实体最近一次**被共同提交接受**的当前预测掩码支持段：按 sigmoid(mask) 降序选最多32段，段索引打破并列，计算同一描述。保存历史 centroid 只用于 M3 的旧位置对照，不用于 M4–M6 硬搜索。

这只是便宜的局部几何摘要，不是位姿估计、配准算法，也不保证跨视角不变。日志中应保留 valid 与采样段数，不能将缺失历史几何伪装成真实零大小物体；输入额外携带历史/当前两个 valid 标志。

### 8.4 提示残差

对 M1/M3/M4/M5/M6 使用同名公共提示子层：

```text
u_i = concat(e_source[128], z[128], g_source[6], g_current[6], valid_flags[2])  # 270维
h_i = GELU(Linear(270,128)(u_i))
delta_i = Linear(128,128)(h_i)
q_i = q_base_i + delta_i
```

最后一层 weight/bias 初始化为0；其他层用同一个公共初始化文件。不能因为输出零初始化，就要求所有上游参数第一个 backward 都有非零梯度；有历史上下文的第2–3次更新再检查实际参数变化。

M4–M6 的位置查询使用 `(c_i, 当前局部t)`，通过父 `pos_enc` 与 `query_projection` 的同一路径编码。输入范围和维度遵守 `initialize_queries` 的实际实现；不得把4维坐标误当3维。`sampled_coords` 同步记录实际使用的提示坐标。新生查询保持父初始化。

### 8.5 M6 的兼容性拒绝头

只对被历史提示占用的 query 计算：

```text
v_i = concat(e_i[128], z_i[128], g_h[6], g_c[6], valid_flags[2],
             max_global_similarity[1], top1_minus_top2_similarity[1])  # 272维
p_i = sigmoid(Linear(64,1)(GELU(Linear(272,64)(v_i))))
```

只有1个当前段时 margin=0。末层 weight=0、bias=2；其余初始化固定。未被历史提示占用的 query 系数为1。拒绝头学习的是“当前检索位置是否支持该历史实体”，不是物理存在/删除分类。

- 主任务 loss 使用原始 class/mask logits，不在 loss 前硬截断模型输出。
- 最终状态 observation 中历史提示 query 的 confidence 乘 p，再按共同0.5阈值计算 valid；class_prob 保持原分布。
- 原官方候选提取完成后，使用明确的 `source_query_ids` 将候选分数再相对原分数乘 p **一次**；不要在已缩放分数上重复乘，也不另删候选。官方 top-k 提取仍按原政策进行，报告这是后提取评分修正。
- 因状态使用了缩放置信度，后续状态轨迹可能变化。原始/缩放候选的单阶段诊断来自同一次前向，只能说明该阶段评分效应，不能称为两套自由运行系统的完整对照。

---

## 9. 冻结实验矩阵：一个固定基线、六个训练变体

| ID | 训练内容与前向定义 | 检索辅助损失 | 拒绝损失/评分 | 比较目的 |
|---|---|---:|---|---|
| M0 | 原 E1 无提示，E1+COMMON-STATE；不适配 | 无 | 无 | 原感知器的共同政策基线 |
| M1 | 当前内容控制：采用与 M4 同样提示/检索子层，但 `e_source` 来自各对应父 FPS query 最近的**当前**段特征，source 几何也由该当前段的32近邻构建；不读取历史 embedding/geometry | 0 | 无 | 匹配 M4 的新增参数及当前检索计算，检验历史内容价值 |
| M2 | 复用 TaskMemoryRead；在 `after_decoder_stage` 的首轮 hlevels 结束处由当前初步预测 route→read，最终统一重新关联 | 0 | 无 | 与已有 proposal 后读取方式比较 |
| M3 | 直接历史提示：`e_source=z=e_i`、`g_source=g_current=g_h`，位置用历史 centroid + 当前局部t；无当前特征检索，几何 invalid 时保留父位置 | 0 | 无 | 与简单历史 query/旧位置传播比较 |
| M4 | §8 的历史→当前特征检索→当前位置提示 | 0 | 无 | 隔离提示结构变化 |
| M5 | M4 + §10 检索监督 | 0.1 | 无 | 隔离检索训练目标的作用 |
| M6 | M5 + 兼容性拒绝头 | 0.1 | 0.1 BCE及§8.5 | 隔离拒绝学习与评分政策的联合变化 |

M1 与 M4 的实际可训练参数集合、形状和数量必须相同，不能加未执行的“占位参数”凑数。M3 不实例化不使用的 Wq/Wk；M2 独立读取模块不要求假装与 M4 等参数，实际数量单列。M6 多出来的拒绝头明确计数，不能称其收益已完全排除了额外参数/评分的影响。

M1 仍使用共同历史状态决定“本次改写多少个 query”，但它不读取历史内容。这是**内容匹配控制**，不能称完全无记忆系统；M0 才是无提示基线。自由运行时各变体状态可能不同，孤立内容因果效应另由同状态机制干预检查。

按本设计可推导的新增参数数：M1/M4/M5=84,224，M3=51,200，M6=101,761；直接复用所读TaskMemoryRead的M2=99,073。它们是结构推导值，不是运行测量。实际实例化后计算并记录；有额外未声明参数/遗漏参数必须解释并修正，不能只填这些数。

M1–M6 的共同父权重完全一致。共享提示子层从同一 seed 初始化；M2 只初始化其读取模块。M0 不做无意义的零参数训练。M0 的 step=0 是有效冻结基线；训练变体的 step=0 仅调试，不能参与正式候选选择。

M4/M5/M6 是候选组，M0/M1/M2/M3 是对照组。即使 M2 或 M3 最好，也应如实选为全表最佳，不改名为新方法成功。

---

## 10. 损失与训练侧历史标签

### 10.1 原任务损失

所有变体使用父 `SetCriterion` 的独立 Hungarian 匹配、原损失类型、aux 层和权重；参考实际 `weight_dict` 与 `_configured_objective_loss`，不得重新发明“官方 AP loss”。继承 `eos_coef` 等解析值，不偷改类别过滤、superpoint、体素、decoder层数。

每阶段任务损失记 `L_base,j`。每条 H episode 的总损失为各阶段平均；每次 optimizer 更新再对4条 episode 平均：

`L_update = (1/4) * Σ_episode (1/H) * Σ_j [L_base,j + λ_ret L_ret,j + λ_support L_support,j]`。

固定父模型、逐阶段 backward、阶段后 detach 状态，不保留跨阶段计算图。无可训练依赖的阶段（常见于首阶段）正常前向、记损失和更新预测状态，但不对无 grad_fn 的标量调用 backward。这不是跳过样本。报告 `truncation=1`，不声称进行了长时 BPTT。

### 10.2 独立训练账本

训练侧保存 `(logical_id,generation) -> (reference_uuid, original_gt_instance_id)` 的标签账本，模型状态中没有这个字段。

仅在预测出生已发生后，用当前扫描预测掩码与当前可评价 GT 做**类一致的一对一 IoU 匹配，IoU≥0.5**，为这些出生赋训练标签。忽略模糊身份、过滤类和未匹配出生。已经绑定的身份标签不因之后的错误关联被悄悄改写；未知身份不能由未来扫描倒填。

GT 不决定出生、路由、检索位置、提示槽数或记忆写入。账本在每条 episode 结束时清空。不用原始 query 槽序号当实体 ID。

### 10.3 检索监督，仅 M5/M6

历史槽有可靠训练标签且同一 GT 实例在当前有效目标中有支持时，以该实例占比≥0.5且身份不模糊的当前段为正集合 P_i：

`L_ret,i = logsumexp(a_i over all current valid segments) - logsumexp(a_i over P_i)`。

对满足条件的历史提示槽取平均；无有效槽时返回与检索分数连接的零损失。当前实例存在但没有纯度足够的段时记为 `NO_VALID_POSITIVE_SEGMENT` 并跳过这项辅助监督，不能删除该训练 episode 或主任务目标。

### 10.4 兼容性监督，仅 M6

根据**已预测的 anchor 段**的训练侧多数 GT 身份：

- anchor 段纯度≥0.5、类别有效、身份不模糊，且等于历史槽可靠标签：y=1；
- anchor 段满足相同可靠性条件，但属于另一个已知 GT 实体：y=0；
- 历史标签未知、当前段含糊/ignore、没有可评价身份：UNKNOWN，不做该项 BCE。

BCE 在全部已知标签提示上取平均；无标签时连接零损失。不会凭“当前 GT 没有该实例”就推断物理移除。负样本来自实际当前预测检索落到另一个可靠实体，不额外构造跨场景伪历史，不额外增加一次 backbone 前向。

正负/UNKNOWN 数量逐阶段汇总；没有有效负例时，M6 可以完成工程训练，但不能报告“已验证拒绝能力”。不因为负例不足而偷偷放宽纯度阈值。

---

## 11. 训练设置、成本准入和恢复

### 11.1 固定设置

- 父 E1 seed45；主适配 seed146；锁定后配对复验 seed147。
- 单 GPU，物理 episode batch=1，累积4条 episode；每更新按§6包含T2/T3/T4/T5。
- 新参数 AdamW：lr=1e-4，betas=(0.9,0.999)，eps=1e-8，weight_decay=0.01；梯度范数裁剪1.0。
- 5%线性 warmup + cosine 到0.1×lr，按实际计划 N 固定，不因恢复重置 scheduler。
- 使用原环境，FP32主训练、不启用 Trainer AMP、TF32关闭。内部原 FlashAttention 若仍使用 FP16，必须明确记录，不能称“全部算子FP32”。
- 父模块 eval，关闭训练增强；本轮前向 `is_eval=True`，保留其完整评价采样路径，以免短适配和最终测试偷偷更换采样政策。实际 OOM 不允许删大样本或改体素来规避；记录并按§15处理。
- 数据加载 workers=0 起步，不进行吞吐网格搜索。不要为了6个适配器建设多节点DDP队列。

### 11.2 新预算不是重新获得1,800 GPUh

先核实累计已用额度及占用，不把旧账本重复算两次，也不能将旧实验花费归零。默认本轮总支出上限：

`B = min(78.0, max(0, 当前已核实总余额 - 8.0)) GPUh`。

若当前余额仍为历史数值，则 B=78.0 GPUh，原总上限不变。未找到可靠余额时，仅做CPU开发与发布，状态 `BLOCKED_BUDGET_BINDING`；不得自行赋一个新的授权额度。

B=78时的**初始计划分配**如下；B较小时先按 B/78 等比例缩放。这不是要求花满的配额。P0/P1的上限固定；P2–P5在RUN_LOCK前按§11.3的实际成本公式一次性重分配，合计不得增加。RUN_LOCK之后不得自行挪用其他成本池或增加变体。

| 成本池 | 上限 GPUh | 内容 |
|---|---:|---|
| P0 | 4 | 必要真实 smoke、定位数值问题和一次针对性修复复核 |
| P1 | 4 | 基线及成本测量、数据实际接通、预算准入 |
| P2 | 48 | 六条主适配轨迹 + 最多两条锁定配对复验轨迹 |
| P3 | 8 | CAL四点选择及复验CAL评估 |
| P4 | 10 | 锁定SEL与PB的完整评价 |
| P5 | 4 | 最终机制面板、有限profile、模型回载检查 |
| **总计** | **78** | 所有失败、试跑和重复测量均计入 |

NEW campaign内实现薄包装 `charged_job`，复用 `write_json_atomic`、`merge_events` 和既有计费公式；不直接调用绑定旧 `root/selection/BUDGET_LOCK.json` 的 `charged_process`，也不修改旧函数。新包装只处理本任务RUN_LOCK、独占锁、子进程、预留与结算，不扩展成通用调度平台。

GPUh=分配GPU数量×进程墙钟时长/3600，不按利用率折扣。重复父/子进程不得重复收费。主编排器在启动每个工作进程前预留额度，进程结束结算；只终止本任务持有的子进程，不清理其他用户作业。复用现有 charged-process 思路，但新 root 的 prior 必须对应“旧累计已耗”，新 ledger 只记新事件。

### 11.3 根据真实成本选择唯一 N

候选 N 仅为 `{400,800,1200,1600}`，所有主训练和配对复验使用同一个 N。

在 P1 中，对 M1–M6 各做最多8个真实 optimizer 更新（相同小型 TRAIN draw，包含四种 H）；前2个预热，后6个计时。保存成本和梯度连通记录，然后丢弃试跑模型/optimizer，正式训练从共同初始化重新开始。试跑不用于选模，不把它计作正式 N。

计费预测使用每种变体后6更新的最大墙钟单步成本乘1.25；训练总预计为：六条主轨迹成本之和 + **最贵的两种变体**各一条 seed147 轨迹成本。评价成本由真实 CAL样本、完整一次episode端点、官方指标序列化的测量按计划输入/阶段数估计，乘1.5；启动开销也计入。

预算锁定前按下式一次性分配研究池，避免评价比初估贵就错误地判定“总预算不足”：

`research_pool = B - cap_P0 - cap_P1`；`cap_P3 = 全量CAL的实测上界预测`；`cap_P4 = 全量SEL/PB的实测上界预测`；`cap_P5 = 4*(B/78)`；`cap_P2 = research_pool - cap_P3 - cap_P4 - cap_P5`。

这里P3/P4使用上述1.5倍率，P2的轨迹预测使用1.25倍率，不重复乘两次。P2≤0即不能准入。选择六臂主训练+最贵两臂复验总预计不超过P2的最大N。CAL/SEL/PB每个锁定评价按3次独立进程预留；P3包含六臂×四点以及最多两条复验×四点，P4按M0、两条主种子端点和最多两条复验端点估计。P1已完成的M0-CAL记录只复用，不重复计费。N只改变训练成本，不能通过假称缩短N就缩短固定评价总体。

把重分配后的实际P0–P5上限和总和写进RUN_LOCK。此后只用锁内上限，不能运行中再次调整来掩盖成本超支。

若N=400也无法完整覆盖，状态 `BLOCKED_FULL_MATRIX_BUDGET`，不启动只跑新方法、少跑对照的偏置矩阵；完成代码、计划、成本测量与发布，并报告需要补充的实际预算估计。不能把阻塞称为方法NO_GO。

正式训练前将通过接通的实现提交为C（实现提交，区别于最终发布提交A/B），绑定C及实际数值文件哈希。训练中不修改已绑定的前向/损失/数据数值代码；实质bug修复需停止受影响运行、保留旧记录、生成修订锁并按共同协议重跑受影响对照，不能混用修复前后的数字。

在正式训练前写入 `RUN_LOCK.json`：N、全部变体、种子、初始化哈希、draw哈希、数据角色、配置差异、评价点、成本估计、阶段上限。此后不得按结果调N或lr。

### 11.4 保存与恢复

每条轨迹保存 N/4、N/2、3N/4、N 四个选点，另在每50更新及正常终止时保存 latest resume 文件。日志写入实际 optimizer更新数、累计episode/阶段数、loss分量、lr、梯度、已用GPUh。

resume 至少含：新模块状态、optimizer、scheduler、step、下一draw游标、Python/NumPy/Torch CPU和CUDA RNG、父权重/配置/代码/数据身份。checkpoint 可以依赖固定父资产，不需要重复保存冻结骨干；缺少父引用不构成可恢复文件。

同一轨迹进程重试最多2次。恢复到最近完整optimizer边界，重复运算收费，记录恢复差异，不承诺CUDA训练逐张量完全恢复。相同数据/代码错误重复一次仍在，不再盲目重启。

---

## 12. 数值重复性：有边界的测量，不做无限安全审计

### 12.1 不能省略的检查

输入UUID、扫描顺序、顶点/超点映射、类别映射、GT侧用途、父权重身份、有限loss和实际梯度必须正确。发现标签泄漏、输入变化、重复提交、NaN或错误分母，要修复后才能继续。

对 P0 的实际 TRAIN输入选择：按reference/input_id排序，覆盖T2和T5各2条，共4条，每条同协议新进程重复3次。保存输入结构、首掩码前特征摘要、最终候选和官方标量差异，不导出每层巨型张量。

只在出现具体差异时插入少量层级探针，定位第一次分叉；至多一次有依据的公共数值修复及一次复核，受P0上限约束。不能凭以前日志认定本次就是某CUDA内核；不能关闭所有哈希、增大容差后把未知原因写成已解决。

### 12.2 两条明确分支

**结构/数据/有限性错误**：停训练、修复；仍失败则 `BLOCKED_RUNTIME_CORRECTNESS`，发布已完成代码及阻塞证据。

**仅浮点非重复，结构稳定且loss有限**：保留 `NONBITWISE_RUNTIME_OBSERVED`。在P1对 M0 完整CAL做3次相同输入顺序、相同随机协议的新进程评价；令 `epsilon_run_pp = max(T1..T5 各AP的3次极差×100)`。

- `epsilon_run_pp > 0.2`：本轮小幅涨点无法可信判别，停止正式训练，状态 `BLOCKED_NUMERICAL_VARIANCE`。
- `epsilon_run_pp ≤ 0.2`：可开展明确标记的受限探索；所有CAL/SEL/PB关键结果继续3次评价并报告均值、极差和状态。**此分支不等于数值问题已解决**，也不能推翻历史报告的非重复观察。

0.2pp是本轮为控制小收益误判而预设的工程准入阈值，不是普遍科学标准或CVPR官方要求。每个最终比较自己的实际波动仍要进入§14的晋级判定。输入和状态身份检查不能被这个浮点阈值替代。

因此，此任务不会无限停留在“修复全世界CUDA确定性”，也不会用一次偶然一致的运行当作证明。未解决原生数值问题时，最终可得到条件性研究信号，不能自动成为完整投稿就绪结论。

### 12.3 仅做以下必要测试组

一次聚焦 `tests/test_revisit_prompt.py`，覆盖：

T01：关闭模块/空状态时，在同一冻结backbone上下文和RNG下保持父模型行为；新参数不影响M0。
T02：固定Q=100，最多32历史提示，当前/旧扫描/padding掩码正确；UUID不被旧正则误处理。
T03：检索不读取GT或初步mask；只交换标签而保持几何/超点不变，前向与状态不变。
T04：无槽、无有效检索、单段、无有效辅助标签时有定义且无NaN。
T05：有历史上下文的真实训练能改变被使用的新参数；冻结父参数不变；零初始化梯度预期合理。
T06：阶段时间、query→logical ID及一次提交正确；两条episode相同query索引不共享身份。
T07：官方候选`source_query_ids`对齐，M6分数只乘一次，类别展开不会错绑ID。
T08：共同前缀GT与当前顶点一致；空目标和窗口依赖类别例子不篡改分母。
T09：选模并列、缺行、负结果、未知gap指标、预算不足分支得到唯一结果。
T10：一次checkpoint保存/恢复与一次部署包回载可用，所有父/新增权重覆盖；远端校验由发布流程执行一次。

合成小样本主要跑CPU。实际GPU只用小面板和真实梯度连通检查，全部计入P0/P1。单轮聚焦CPU测试墙钟上限20分钟，修复后只重跑受影响项，最多两轮；超时或失败必须列明，不准把超时改成通过。

禁止：全仓库历史套件重复运行、fuzz/渗透/广泛恶意输入测试、百次重放、跨所有CUDA版本测试、反复哈希整个数据集、独立审计代理循环、监控刷屏或把新增测试数当作研究成果。一次新种子的训练是科学复验，不属于可被取消的“安全测试”。

## 13. 评价政策、指标和必须保留的结果

### 13.1 每次评价实际执行

每次CAL/SEL/PB评价是一个新进程，固定：evaluation seed=9001，输入排序 `(horizon,input_id)`，每条输入的随机种子由 `stable_seed(9001,input_id)` 产生，进入输入时状态为空。3次重复使用同一协议，不通过改变seed来掩盖非重复；重复编号只是记录身份。

每次重复独立计算一套官方指标，不把3次预测拼成一个更大的测试集。主表报告3次标量的算术均值和极差，全部原始行保留。预测不稳定时，不选择最接近其他方法的一次进行配对。

复用 `extract_official_task_prediction`、`OfficialCandidateTrajectoryAccumulator(score_reducer='mean')` 和 `OfficialMetricAccumulator`。实际函数签名、输出字段在实现前按固定版本读取，新增适配器不修改官方指标算法。

时间转换必须明确：给局部pair的 `extract_official_task_prediction` 传的是局部 `latest_stage_index=1`（首扫描为0），不是绝对episode阶段j；提取当前输出后，再以绝对j提交轨迹累计器。adapter提供 `stage_id=j` 和长度100的 `track_ids`，按 `CommitResult` 的query→logical_id/generation填入，未获ID为None；候选通过source_query_ids查这个数组。[R16–R17]

轨迹采用 **commit0**：每次只提交当前扫描，过去输出不回写；同一 logical ID 和类别的分数按已实际出现次数平均，没有出现的阶段不补零分数。未获持久ID的官方候选按原 accumulator 的 ephemeral 语义保留，不能因为记忆没关联上就删除检测。

使用候选携带的 `source_query_ids`，不能假定“候选第k列”就是“原始第k个query”。同一query的类别展开、类别改变和轨迹键遵守既有累计器政策；不擅自把不同类别串成同一类别的高分轨迹。

### 13.2 固定指标

主指标：`S_long = (pooled official t-mAP_T3 + t-mAP_T4 + t-mAP_T5)/3`。

辅助主指标：T1、T2官方AP，T3–T5的 pooled current-scan AP 均值 `S_current`。

身份指标：gap恢复召回、恢复精确率、身份切换率，以及各自事件分子/分母，按原 `system_comparison_metrics` 定义复用。若当前接口不能直接复用，先写出逐事件定义与旧实现的差异，不得用原始query槽变化替代ID切换。

zero denominator 一律 null/NA；不能填0。原 `gap_recovery_accuracy` 只有定义为“正确恢复/实际恢复尝试”时才能在报告解释为恢复精确率；名称不同也要同时列公式。事件次数不等于独立样本数。

总AP必须从官方充分状态池化，不能平均场景AP冒充 pooled AP。场景级表另存。单位：JSON/CSV原指标保存0–1，报告AP用百分数、差值用pp，转换明确。

### 13.3 物理场景统计

对锁定候选与锁定最强对照，在SEL、PB按物理reference配对 bootstrap 1000次，seed=9101；每次重采样相同的reference及其全部序列，按官方状态重新池化指标，再计算方法差。三个进程重复分别形成差值，bootstrap样本内对这三个差值取平均；同时单列重复波动，不能把它当成独立场景。

充分状态不能直接合并时，用本地已保存的必要预测/真值或官方中间状态在CPU重算；不得简化成场景AP均值。不得为bootstrap重跑1000次GPU推理。独立reference<10时注明区间不稳定、仅限该面板；不把95%区间当作全部科研结论。

### 13.4 困难例与机制，受P5预算约束

锁模后，从CAL按输入ID排序取最多6条有历史占用的长输入，最多12次成对机制前向：正常历史 vs 将所选槽的内容在槽间循环置换。共享同一份冻结backbone输出、旧状态和后续RNG；只有读取的历史描述/几何被置换，ID、GT和最终共同关联政策不改。置换只用于干预，不写回原记忆。

若少于2个有效历史槽则该样本不提供内容置换证据。受控上下文仍无法对齐时记 `CONFOUNDED`，保留原始结果，不反复试到对齐为止。该干预不是一个重新训练的消融，也不能单凭它证明几何分支独立有效。

正式报告还应从已经完成的评价输出分析：

- M0漏分而新方法恢复的当前GT对象，同时列新方法丢失对象，采用同类一对一IoU≥0.5诊断；该诊断不替代AP。
- 长缺席机会按既有评价身份与中间当前扫描支持记录定义，不把未进入GT的对象自动称为物理移除。
- 搬动子集仅在现成真实变换/变化标注有可靠对应时统计；没有就写 `NOT_AVAILABLE`，不能用局部可见点质心变化自动制造“真实搬动”标签。
- 同类干扰从同一当前可评价场景中不同实体的类别关系统计；未知/模糊身份排除该诊断但不删主评价输入。

### 13.5 资源测量

锁定候选、最强对照、M0在同一设备的同一CAL面板做1次warmup+3次测量，记录输入点数/段数、占用槽数、模型前向、关联、I/O、后处理、端到端延迟和GPU峰值 allocated/reserved。显式状态字节包括几何sidecar；离线评价输出档案另列，不能称整个系统恒定空间。

本轮冻结E1并未改变原骨干计算，不能预先写“只需原方法一半显存”或复用旧权重/输入的profile数字。完整前缀 FH-E1 可作为附加上下文参考，若进入计划须在RUN_LOCK前计入预算并遵守同一公共GT；默认本轮主比较只含M0–M6，不强制再启动完整FH研究线。

---

## 14. 选模、种子复验、是否继续研究的唯一规则

本节在训练前锁定。0.2pp、0.5pp等阈值都是本轮预先约定的研究决策规则，不是文献声称的定理或CVPR录用标准。

### 14.1 完整性优先

候选行必须拥有全部规定输入和3个进程重复、官方指标及分母、对应权重和真实step。缺失/NaN的行标记 `INCOMPLETE`，不能以可见部分均值参加排序。

某一变体缺少规定的四个CAL检查点时，该变体不可参选；关键矩阵缺行且不能在既定预算内补齐，则本轮 `EXPERIMENT_INCOMPLETE`，可以公布描述性最佳，但禁止宣布受控方法胜出。每个变体的step=0调试行不可替代缺失点。

### 14.2 CAL检查点排序

对每个训练变体，从 N/4、N/2、3N/4、N 中按以下顺序选择：

1. 最大 `S_long`；
2. 差在1e-6以内则最大 `S_current`；
3. 仍并列则最大 T2 AP；
4. 仍并列则较早step。

以上比较使用0–1比例及3次重复均值，容差仅用于排序，不改写原始数值。保存全部四点，不只保存赢家。

### 14.3 锁定对照与候选

各变体先独立选检查点。相对M0的CAL T1和T2都不能低于0.2pp，满足者为短时guard合格；M0始终可作为对照。跨变体排序按 `S_long → S_current → T2 → 实际可训练参数更少 → 变体ID字典序`。

- `best_overall`：全体完整变体按上述排序的描述性最佳，是否过guard单列。
- `control_star`：M0–M3中完整且过guard者的最佳。
- `candidate_star`：M4–M6中完整且过guard者的最佳；没有则为null，但另外保留该组描述性最佳。

保存 `CAL_SELECTION_LOCK.json`，含选择过程及淘汰原因。不能因M6名称最复杂而优先选择它。

### 14.4 是否执行seed147

candidate_star存在，且相对control_star满足：

- CAL `ΔS_long ≥ 0.5pp`，且 `ΔS_current ≥ 0`；
- T3/T4/T5每个长度的AP差都≥−0.2pp；
- 候选及对照 `S_long` 的进程重复极差均≤0.2pp；
- `ΔS_long > 2 × max(候选极差, 对照极差)`；

才执行两者的seed147配对适配。若control_star=M0，则只训练candidate_star seed147，并保留M0冻结参考，不为M0发明新训练权重。省下预算不用于新增候选。

seed147从同一个E1父模型重新初始化新增参数，完整跑同一N；不从seed146最佳点继续微调。变体家族不得根据seed147重选。各自seed147检查点仍只按CAL四点规则选择，记录为 `REPLICATION_LOCK.json`。

未满足seed147门槛：**不继续涨点搜索**；仍将锁定的描述性最佳候选与control_star送入SEL/PB，用于完成本轮诊断，结果状态 `SCREEN_NO_GO`。没有guard合格候选时，此时评价候选组描述性最佳，但不得称其为可晋级模型。

### 14.5 SEL和PB只评价锁定端点，不再改模型

SEL评价：M0、control_star、候选组锁定端点，已有seed147则加入对应端点；同一端点只计算一次计划条目。PB使用历史FORMAL_POPULATION中T2–T5的既有记录及身份绑定，全部锁定端点按同一新流式政策复测。CAL/SEL/PB名称保留，不改称全新测试集。若候选组没有任何完整端点，candidate字段为null，只评价实际存在的M0/control端点，状态保持EXPERIMENT_INCOMPLETE；不能把初始化权重代替不存在的最佳候选。

不根据SEL/PB修改参数、筛选阈值、发布规则或检查点；任何会改变预测的后续修改意味着新的实验协议，不能覆盖本轮结果。纯指标实现bug需要修复时，保留旧结果标记invalid，统一重算全部受影响锁定端点，不只重算赢家。

### 14.6 最终科学状态

- `PROMISING_REPLICATED_ADAPTER_SIGNAL`：完整配对适配（control为M0时允许冻结参考），两个适配种子在SEL和PB的ΔS_long均为正；跨种子平均ΔS_long在SEL和PB都≥0.5pp；两个总体的跨适配种子平均ΔS_current都≥0；SEL检查T1/T2和T3–T5，PB只有T2–T5则只检查这些实际存在的长度，不能补造PB-T1；逐长度guard同上；新进程波动低于上述阈值；没有未解决的输入/标签/指标正确性问题。
- `PROVISIONAL_SINGLE_ADAPTER_SEED_SIGNAL`：主种子结果为正，但未完成规定复验；不得写已重复验证。
- `SCREEN_NO_GO`：完整筛选但CAL晋级门槛不满足。
- `CONFIRMATION_NO_GO`：CAL通过，但完整锁定SEL/PB或第二适配种子没有支持收益。
- `INCONCLUSIVE`：关键结果不全、分母/事件不足以支持特定主张，或数值波动无法分辨差异。

若处于§12浮点非重复分支，另外保留 `NUMERICAL_ROOT_CAUSE_UNRESOLVED`，即使有正信号也不能宣称端到端逐张量复现。以上状态与 `CODE_COMPLETE`、`EXPERIMENT_COMPLETE`、`PUSH_VERIFIED` 分开。

**本轮从不自动输出 `READY_FOR_CVPR`**：因为父模型种子、外部独立场景、强外部系统、容量饱和与充分新颖性论证没有由本任务全部解决。正信号的意义是支持下一阶段投入，不是保证录用。

---

## 15. 逐阶段任务安排与停止规则

| 阶段 | 具体工作 | 进入下一阶段的证据 | 不满足时做什么 |
|---|---|---|---|
| A：定位与绑定 | worktree、资产、源码、数据角色、余额 | `START_STATE.json`、`SOURCE_BINDINGS.json`、资产可解析 | 不猜路径/权限；继续CPU代码与阻塞交接 |
| B：实现 | 新模型/数据/损失/训练/评价/选择/发布入口 | 全部NEW命令支持help/dry-run；聚焦CPU测试完成 | 修复受影响功能；不启动全量GPU |
| C：最小真实接通 | 真实父加载、因果T2/T5、梯度、数值面板 | 父覆盖、当前段对齐、loss有限、数值分支明确 | 在P0额度内修复；超过则阻塞发布 |
| D：成本准入 | M0 CAL重复、六变体计时、完整队列估价 | 唯一N、逐成本池可覆盖、RUN_LOCK | N=400也不够则不启动不公平矩阵 |
| E：六臂适配 | M1–M6 seed146完整N；定点CAL评价 | 所有轨迹/四点/官方状态完整 | 最多2次有原因重试；不足标INCOMPLETE |
| F：CAL锁定 | 候选/对照/晋级规则 | CAL_SELECTION_LOCK | 负结果也继续完成锁定诊断与交付 |
| G：可选配对复验 | 仅门槛通过时seed147 | 完整N与REPLICATION_LOCK | 如预算失配，明确未复验，不用短轨迹替代 |
| H：锁定确认 | SEL/PB、场景统计 | 无重选的完整端点结果 | 统一修正指标bug或注明不完整 |
| I：机制与资源 | 小型同上下文干预、profile、部署回载 | 机制边界和实测资源 | 缺少证据则删除相应主张，不无限补实验 |
| J：结果发布 | REPORT、HANDOFF、权重/指标、Git/LFS推送 | 远端回读及PUBLICATION_RECEIPT | 保留本地交付并明确PUSH_FAILED，不假报上传 |

运行时新出现以下情况的唯一处理：

- **新模型OOM**：检查是否误保留跨阶段graph或重复backbone；仅修复实现浪费。仍OOM则终止该轨迹并记录；不缩小输入、不降体素、不增加显存设备、不将已缩小输入结果混进主表。
- **个别真实输入损坏/缺失**：根据清单找到该输入并核实一次；不静默跳过。修复原映射/找回原文件或将对应评价标INCOMPLETE。
- **GPU被别的任务占用**：只使用已经授权且可用的设备。不能杀其他任务、绕过旧worker锁或擅自占用全部12张A40。
- **额度将不足**：在optimizer完整边界保存，结算失败/剩余计划，转交接与发布。不自动延长预算。
- **只有指标差，没有实现错误**：保留负结果，按规则选择，不以“调试”为名重跑到出现正收益。

---

## 16. NEW 命令接口与一键编排

以下CLI必须在新分支实现。所有路径默认由§3解析；不要在最终 `COMMANDS.md` 留下 `YOUR_PATH` 一类待填参数。

```bash
python -m scripts.revisit_prompt_campaign resolve --auto
python -m scripts.revisit_prompt_campaign preflight
python -m scripts.revisit_prompt_campaign plan
python -m scripts.revisit_prompt_campaign run --resume
python -m scripts.revisit_prompt_campaign summarize
python -m scripts.publish_revisit_prompt --execute
```

接口职责：

- `resolve --auto`：读取资产/权限/历史预算，生成本地resolver和公开START_STATE，不启动训练。
- `preflight`：CPU测试与限额真实接通；复用已完成且身份一致的结果，不重复花GPU。
- `plan`：执行尚未有记录的P1测量，锁定N、队列和成本；多次调用不改变已锁计划。
- `run --resume`：按阶段E–I执行，内部调用训练/评价/选模。只恢复身份一致的未完成项，不覆盖完成的负结果，不重复训练已经完成的轨迹。
- `summarize`：从全部实际结果生成表和判定，不能填入计划值代替实测值。
- `publish --execute`：§18中的真实Git/LFS发布及核验；不能只打印拟执行命令。

NEW训练入口支持 `--variant M1..M6 --seed 146|147 --resume`，N和配置只能来自RUN_LOCK。
NEW评价入口支持 `--role CAL|SEL|PB`、`--selection-lock`（默认取resolver绑定的已冻结锁文件路径）和 `--repeat-index 0|1|2`；训练过程CAL允许RUN_LOCK中的四个点，SEL/PB只允许CAL/REPLICATION锁中的端点。
NEW选模入口只读取结果文件并生成lock，不调用训练、不改checkpoint。
所有入口返回非零退出码表示真实失败，同时原子写入机器可读状态。`--help`、计划和dry-run不能谎报产生了模型。

主编排器可以在同一Codex任务执行期间自动推进，不需要每一阶段向用户重复请求“是否继续”。预算或资产无法解决时按定义分支交付，不自设权限。

---

## 17. 必须交付的文件与可复现模型

新公开目录：`artifacts/revisit_prompt_v1/`。运行大文件可以在run_root，最终公开资产统一同步到该目录或LFS。

```text
artifacts/revisit_prompt_v1/
  EXECUTION_INSTRUCTION.md         本最终指令原文
  PROMPT_AUDIT.md                  本指令审阅记录及实际执行中的接口修订
  START_STATE.json
  SOURCE_BINDINGS.json
  DATA_MANIFEST.json               物理身份、角色、顺序、计数与暴露状态
  PARAMETER_AND_LOAD_AUDIT.json    父覆盖、新参数、冻结范围、M1/M4参数匹配
  RUN_LOCK.json
  COST_PLAN.json
  COST_LEDGER.jsonl
  RUNTIME_SUMMARY.json
  TRAINING.csv
  CAL_ALL.csv
  CAL_SELECTION_LOCK.json
  REPLICATION_LOCK.json            未执行时明确status，不生成假选点
  SEL_LOCKED.csv
  PB_LOCKED.csv
  BY_REFERENCE.csv
  IDENTITY_EVENTS_SUMMARY.csv
  MECHANISM.csv
  PROFILE.csv
  SELECTION_DECISION.json
  STATUS.json
  REPORT.md
  HANDOFF.md
  COMMANDS.md
  MODEL_MANIFEST.json
  ARTIFACT_MANIFEST.json
  deployment/                     新增参数部署包及必要解析配置
  evidence/official_states.zip    可重新池化指标的官方充分状态
```

缺失阶段也应有状态/原因，而不是空文件假装完成。CSV保存变体、父身份、适配seed、step、role、repeat、单位、分母和状态；指标未知留null，不写0。单张汇总表不能代替原始四点与重复行。

### 17.1 模型交付格式

父E1和编码器已分别绑定，本轮可采用小型delta部署包，包含所有新增模块参数/缓冲区、变体、解析配置、schema、父E1哈希、编码器哈希、taxonomy哈希、新源码提交/文件哈希。至少上传M1–M6各自CAL最佳点；实际seed147候选/对照最佳点也上传，负结果同样保留。

NEW `load_revisit_bundle` 必须能由**固定父任务包 + 指定编码器 + 新delta包**重建完整预测模型，严格核验所有覆盖项；明确这是依赖父资产的部署形式，不称“仅靠delta文件即可推理”。没有修改的父包无需复制上传第二份，但MODEL_MANIFEST必须写清其Git/LFS取得位置。

resume文件另存新模块、optimizer/scheduler/RNG/进度并依赖父资产。至少为真正完成的最终候选/对照保留可恢复检查点或实际可取得的存储回执；若未上传resume文件，不称部署包可直接恢复optimizer。

部署回载只做一次完整参数覆盖与有限真实前向核验，不反复加载几十遍。公网上不上传原始点云、逐点GT、许可编码器、训练数据缓存、绝对运行路径或凭据。保留现有许可证及第三方来源，不擅自改变仓库许可证或公开性。

### 17.2 REPORT.md 必须回答

问题和候选新意是什么；已有proposal后读取与新方法的实际区别；采用/未采用哪些近邻机制；M0–M6到底训练了哪些参数；损失/输出政策有哪些差异；全部选点与负结果；CAL/SEL/PB及重复的分母；适配种子和父种子的区别；数值波动；资源；支持/不支持的H1–H3；当前最值得继续的唯一方向或停止判断。

H1“补回漏分割”需要当前掩码的一对一诊断也有支持。仅AP上升可能来自分数重排或身份关联，不能自动解释为掩码变好；仅长期AP提高但当前AP不提高时标记 `QUALITY_SCOPE=TEMPORAL_ONLY`，不宣布H1成立。

如果候选没有优于对照，直接写出。不能把“参数有梯度”“代码测试通过”“query发生变化”写成涨点机制已经有效。

---

## 18. GitHub实际发布：提交、交接、字节和回执

### 18.1 发布前仅做一次聚焦检查

检查本次diff和待提交文件，确认没有原始数据、编码器、凭据、意外大日志；检查新脚本import/help及聚焦测试结果；检查结果表能由已保存状态重算。不要扫描整个Git历史。

Git LFS只跟踪本次选定模型/checkpoint和必要指标压缩包。先确认LFS可用，再add与push；不要把几百MB权重当普通Git blob提交。记录真实文件字节与SHA-256，不只记录LFS文本指针。

### 18.2 避免提交SHA自引用循环

发布收尾使用明确的两次提交与一个外部远端回执（不包括前面的实现提交C）：

1. **提交A**：新代码、配置、结果、模型manifest、REPORT和指标资产。`ARTIFACT_MANIFEST.json` 仅绑定A中冻结的代码/配置/实验/模型资产，明确排除自身及 `HANDOFF.md`、`COMMANDS.md`、`PROMPT_AUDIT.md`、`PUBLICATION_RECEIPT.json`；后四者由B或外部回执负责。提交A后记录A的完整SHA。
2. **提交B**：增加最终HANDOFF/COMMANDS/PROMPT_AUDIT（及需要的交接清单），HANDOFF引用A的完整SHA，不把B自己的SHA硬写进B。创建新tag指向B。
3. 将本任务新LFS资产、分支和tag实际push；核验远端branch/tag都指向B，并核验LFS实体字节。
4. A/B内的发布状态写 `PENDING_EXTERNAL_VERIFICATION`，不在尚未push时预填成功。实验状态在A冻结；最终发布事实以外部回执和远端下载为准，不为了改写STATUS.json中的发布字段继续改动A/B。
5. **远端回执不再加入B**：在GitHub该tag的Release中上传 `PUBLICATION_RECEIPT.json`，同时保存在run_root的publication目录。回执记录A/B/tag、远端SHA、被核验资产和实际完成时间。这样没有“为了写回最终SHA而无限新建提交”的循环。

实现可用现有git/git-lfs/gh；若Release功能权限不足，但Git和LFS已验证成功，则明确 `CODE_AND_LFS_PUSH_VERIFIED_RELEASE_RECEIPT_LOCAL_ONLY`，保留本地回执和实际错误，不伪造Release链接。整体发布不能在此状态写成全部交付项已远端完成。

### 18.3 远端核验范围

一次新临时目录、空LFS缓存的下载核验足够：读取远端B、tag指向、HANDOFF、REPORT和所有本次部署delta文件，并比较bytes/SHA；官方状态压缩包核验一次。不要多次完整克隆巨大仓库，也不要拉取所有历史LFS对象。

`git ls-remote`只证明引用存在，不证明模型字节；`git lfs ls-files`只看到指针也不证明远端数据可下载。必须实际取回本次公开模型和指标包的实体文件。

如果push失败：保留本地A/B、模型、MD、错误和可重试命令，最终状态 `PUSH_FAILED`。禁止声称GitHub已同步。发布失败不反向改变实验科学判定。

### 18.4 HANDOFF.md 内容

必须写：任务目的；实际base/branch/tag/A；依赖父权重/编码器和数据取得方式；已完成/未完成范围；唯一选模规则和实际赢家；每个种子/检查点；所有限制与负结果；实际命令（无占位符）；GPU成本和剩余额度；是否仍有本任务活动进程；部署加载/指标重算/恢复方法；产物索引；远端回执在Release或本地的取得位置；唯一后续建议。

最终回复用户时提供B完整SHA、branch/tag、实验报告和交接的真实GitHub入口、模型/状态资产位置、总GPUh、CODE/EXPERIMENT/SCIENCE/PUBLICATION四种状态。不得用计划中的文件名构造不存在的成功链接。

---

## 19. 实现后必须完成的最后自审

这是一次内容审核，不是增加大量GPU测试。按如下顺序逐项写入 `PROMPT_AUDIT.md` 的执行部分：

1. 改动是否真的在首个掩码前使用历史检索，还是又写成proposal之后？
2. 新方法是否绕过了当前proposal置信度门槛，同时保留了新对象发现？
3. UUID、local-t、absolute-t、superpoint和当前段坐标是否对应实际源码？
4. 父包是否包含全部已训练任务参数和缓冲区；新加载器是否保留了原身份与新身份？
5. 冻结父参数时梯度是否仍可到提示；M1/M4是否真等参数且执行了这些参数？
6. M2是否明确为适配对照；M0是否没有借用旧B4结果；所有变体最终政策是否一致（M6的预定义差异除外）？
7. GT是否仅在训练损失/账本和评价；UNKNOWN有没有被当作负例？
8. N、lr、种子、四个选点、六臂和最多两臂复验是否遵守RUN_LOCK？
9. 是否完整报告重复、失败、分母、统计单位及历史暴露？
10. 最佳候选、方法胜出和投稿就绪是否被分开？
11. 总账是否未重置、未重复收费、未挪用8小时预留；有没有将小试当全量？
12. 是否真正生成了可加载资产、交接MD并验证GitHub字节；是否避免SHA自引用与后续交接改写前一manifest？

有失败项时按真实影响标记：代码bug修复；研究证据不足收缩论断；资产/预算/发布阻塞写明未完成。不能全部机械填PASS。

---

## 20. 对本指令的审阅边界

本指令基于固定仓库版本的相关调用链、部署契约、历史实验报告和论文一手页面制定。指令交付前做了静态一致性审阅和文件检查；**没有在用户GPU环境执行新训练，也没有预先验证新方法涨点**。

本文固定的是可执行任务、接口、实验选择和失败分支，不是保证外部资产/权限/预算一定可用。执行者需完成一次有范围的现场解析；遇到未知事实要记录未知，不能凭猜测填PASS。不要把这句话扩大成反复请求确认或开展无限审计的理由。

以下证据索引是指令本身的一部分。具体符号行为以固定提交中的实际源码为准；新增函数名和超参数是本任务设计，不伪装成已存在的API。


## 21. 固定源码与论文证据索引

源码均锚定BASE提交，Git blob SHA与阅读范围另见随包 `source_bindings.json`。只读取局部范围的文件没有被描述为已逐行审完全文。

### R01

- [artifacts/native_long_postselection_v1/REPORT.md](https://github.com/Orangekostar/Persist4D/blob/6b1b779172afdeb51e2719a737cd05e7eeeca982/artifacts/native_long_postselection_v1/REPORT.md)；阅读范围：报告主体；130行起末尾；符号：`实验、资源、投稿判断`。

**已观察**：本轮E0–E3结果、数值限制、记忆桥接和最终总成本。

**本轮决定**：保留正负结果；新研究不重用旧表来宣称新模块有效。

### R02

- [artifacts/native_long_postselection_v1/HANDOFF.md](https://github.com/Orangekostar/Persist4D/blob/6b1b779172afdeb51e2719a737cd05e7eeeca982/artifacts/native_long_postselection_v1/HANDOFF.md)；阅读范围：全文；符号：`交接、下载、后续科研边界`。

**已观察**：完成态交接、父权重交付与历史预算余额。

**本轮决定**：只继承可证实资产；实际额度执行前重新绑定。

### R03

- [models/rescene.py](https://github.com/Orangekostar/Persist4D/blob/6b1b779172afdeb51e2719a737cd05e7eeeca982/models/rescene.py)；阅读范围：1–230、230–530、530行起forward及相关尾部函数；符号：`ReScene.__init__; initialize_queries; forward; mask_module; aggregate_features; after_decoder_stage`。

**已观察**：原ReScene在初始化query后生成首次mask；其后才执行cross/self attention与after_decoder_stage。

**本轮决定**：新增默认空操作的首mask前hook，不把后续读取换名为新机制。

### R04

- [models/task_memory_read.py](https://github.com/Orangekostar/Persist4D/blob/6b1b779172afdeb51e2719a737cd05e7eeeca982/models/task_memory_read.py)；阅读范围：1行起至文件末尾；符号：`TaskMemoryRead.forward`。

**已观察**：单个路由历史槽与null分支竞争，再门控更新query。

**本轮决定**：M2重用已有读取模块；不能声称首次attention记忆。

### R05

- [models/persist4d_task_memory.py](https://github.com/Orangekostar/Persist4D/blob/6b1b779172afdeb51e2719a737cd05e7eeeca982/models/persist4d_task_memory.py)；阅读范围：1–220、210行起后续forward/hook实现；符号：`Persist4DTaskMemory; _build_pre_observation; after_decoder_stage; forward`。

**已观察**：首轮hlevels末尾用初步mask构造observation、route和task_read。

**本轮决定**：绑定M2的真实插入位置；注明最终重新关联的协议差异。

### R06

- [models/object_visual_memory.py](https://github.com/Orangekostar/Persist4D/blob/6b1b779172afdeb51e2719a737cd05e7eeeca982/models/object_visual_memory.py)；阅读范围：1–140；符号：`VisualCandidate; ObjectVisualState`。

**已观察**：已经有K100、r8、D128的有界多代表视觉记忆。

**本轮决定**：不重复开发普通多代表记忆并称创新。

### R07

- [artifacts/task_memory_retention_v2/FINAL_REPORT.md](https://github.com/Orangekostar/Persist4D/blob/6b1b779172afdeb51e2719a737cd05e7eeeca982/artifacts/task_memory_retention_v2/FINAL_REPORT.md)；阅读范围：全文；符号：`Protocol-B tMAP; Strict comparisons; Limitations`。

**已观察**：M3-V-CORE在旧研究中输给多个更强对照，且旧发布政策不同。

**本轮决定**：不续训该旧路线；不拼接跨协议数值。

### R08

- [artifacts/native_long_retrain_v1/POPULATION.json](https://github.com/Orangekostar/Persist4D/blob/6b1b779172afdeb51e2719a737cd05e7eeeca982/artifacts/native_long_retrain_v1/POPULATION.json)；阅读范围：1–100；符号：`counts; evaluation_counts; evaluation_inputs`。

**已观察**：原清单角色计数361/8/8、CAL/SEL输入计数，扫描为UUID。

**本轮决定**：保留物理角色和真实UUID；不要直接复用旧scene名正则。

### R09

- [artifacts/native_long_postselection_v1/deployment/README.md](https://github.com/Orangekostar/Persist4D/blob/6b1b779172afdeb51e2719a737cd05e7eeeca982/artifacts/native_long_postselection_v1/deployment/README.md)；阅读范围：全文；符号：`Trained native-long task bundles; portable loading`。

**已观察**：部署资产是非编码器任务参数和缓冲区；编码器独立；有源码与数值限制。

**本轮决定**：选择E1-23760，检查父资产而不假称optimizer恢复。

### R10

- [scripts/native_long_delivery.py](https://github.com/Orangekostar/Persist4D/blob/6b1b779172afdeb51e2719a737cd05e7eeeca982/scripts/native_long_delivery.py)；阅读范围：1行起完整加载及部署审查实现；符号：`portable_config; load_portable_bundle`。

**已观察**：portable loader校验旧数值code identity、父资产与两处backbone路径。

**本轮决定**：新迁移器核验父身份，同时绑定新代码；不删除旧检查。

### R11

- [datasets/native_long_dataset.py](https://github.com/Orangekostar/Persist4D/blob/6b1b779172afdeb51e2719a737cd05e7eeeca982/datasets/native_long_dataset.py)；阅读范围：1行起完整实现；符号：`stable_seed; isolated_rng; horizon_weights; NativeLongCollator; build_native_datasets`。
- [datasets/task_memory_episode.py](https://github.com/Orangekostar/Persist4D/blob/6b1b779172afdeb51e2719a737cd05e7eeeca982/datasets/task_memory_episode.py)；阅读范围：1–190、190–355；符号：`NativeEpisodeMaster; TaskMemoryEpisodeSpec; StageMeta`。

**已观察**：native数据有稳定种子、UUID清单和初始T2课程；旧episode spec校验scene正则，StageMeta保留几何映射。

**本轮决定**：新UUID episode + 原native loader；短适配从第一更新含四个长度。

### R12

- [trainer/task_memory_trainer.py](https://github.com/Orangekostar/Persist4D/blob/6b1b779172afdeb51e2719a737cd05e7eeeca982/trainer/task_memory_trainer.py)；阅读范围：1–260、380–670；符号：`stage_mean_coefficients; task_memory_lr_multiplier; commit_task_memory_stage; bind_training_births; TaskMemoryTrainer`。

**已观察**：旧trainer有stage平均、TBPTT、最终observation、commit与独立训练birth ledger。

**本轮决定**：复用必要思想/助手，不机械继承旧variant梯度审计和训练约束。

### R13

- [models/task_memory_routing.py](https://github.com/Orangekostar/Persist4D/blob/6b1b779172afdeb51e2719a737cd05e7eeeca982/models/task_memory_routing.py)；阅读范围：1–150、235–530；符号：`PredictionObservation; EntityRoute; route_entities; commit_entities`。
- [models/task_memory_state.py](https://github.com/Orangekostar/Persist4D/blob/6b1b779172afdeb51e2719a737cd05e7eeeca982/models/task_memory_state.py)；阅读范围：1–155；符号：`TaskMemoryConfig; TaskMemoryState.empty`。

**已观察**：实体状态有logical ID/generation；route关联旧状态，commit只接受当前有效观测并分配出生。

**本轮决定**：最终统一route/commit一次；query来源不是最终ID；几何sidecar只由预测更新。

### R14

- [models/task_memory_criterion.py](https://github.com/Orangekostar/Persist4D/blob/6b1b779172afdeb51e2719a737cd05e7eeeca982/models/task_memory_criterion.py)；阅读范围：1–220；符号：`TaskMemoryCriterion; _assignment; _safe_mask_losses; _empty_current_loss`。

**已观察**：旧criterion有独立匹配与TALA分支，空集mask loss与GT ledger要求不同。

**本轮决定**：主比较统一独立匹配；新增监督在loss侧，不把TALA route直接套给preproposal提示。

### R15

- [scripts/train_task_memory.py](https://github.com/Orangekostar/Persist4D/blob/6b1b779172afdeb51e2719a737cd05e7eeeca982/scripts/train_task_memory.py)；阅读范围：1–210；符号：`R1_SHA256; VARIANTS; validate_run_budget; compose_variant_config`。

**已观察**：旧训练入口硬绑定R1字节/哈希、旧变体、3000/1500步和120GPUh。

**本轮决定**：另写训练入口，不更换常数冒充原实验。

### R16

- [scripts/rescene_task_postprocess.py](https://github.com/Orangekostar/Persist4D/blob/6b1b779172afdeb51e2719a737cd05e7eeeca982/scripts/rescene_task_postprocess.py)；阅读范围：1–190；符号：`OfficialTaskPrediction; OfficialTaskSoftEvidence`。

**已观察**：官方候选携带source_query_ids/source_class_ids及局部latest-stage masks。

**本轮决定**：用明确候选谱系映射ID，分清局部t和绝对t。

### R17

- [scripts/system_comparison_v2_inference.py](https://github.com/Orangekostar/Persist4D/blob/6b1b779172afdeb51e2719a737cd05e7eeeca982/scripts/system_comparison_v2_inference.py)；阅读范围：1行起完整实现；符号：`CandidateTrajectoryKey; OfficialCandidateTrajectoryAccumulator.add_stage; snapshot`。

**已观察**：轨迹累计器使用persistent/ephemeral键、stage_id、source_query_ids和mean/latest/max reducer。

**本轮决定**：固定commit0/mean，保留未关联官方候选。

### R18

- [scripts/native_long_memory_bridge.py](https://github.com/Orangekostar/Persist4D/blob/6b1b779172afdeb51e2719a737cd05e7eeeca982/scripts/native_long_memory_bridge.py)；阅读范围：1–210；符号：`context_record; current_target; audit_current_alignment; project_common_truth; assert_same_temporal_truth`。

**已观察**：记忆桥接按局部/完整上下文构造记录，并核实公共GT和当前点索引。

**本轮决定**：复用公共真值对齐原则；不直接复用硬编码129条/E臂的工作进程。

### R19

- [scripts/native_long_execution.py](https://github.com/Orangekostar/Persist4D/blob/6b1b779172afdeb51e2719a737cd05e7eeeca982/scripts/native_long_execution.py)；阅读范围：1–175；符号：`select_cal_points; evaluate_checkpoint; charged_process入口`。

**已观察**：旧CAL选择要求完整官方总体并用pooled T3–T5平均；入口绑定旧E臂/seed45。

**本轮决定**：新selector保持正确指标语义，移除旧任务硬绑定但不重写官方AP。

### R20

- [scripts/native_long_budget.py](https://github.com/Orangekostar/Persist4D/blob/6b1b779172afdeb51e2719a737cd05e7eeeca982/scripts/native_long_budget.py)；阅读范围：全文；符号：`budget_cap; merge_events; remaining_budget; write_json_atomic`。

**已观察**：成本工具以prior+新事件计算余额，缺省cap可落回192。

**本轮决定**：必须显式绑定总上限和prior，避免重置旧消费或重复收费。

### R21

- [artifacts/native_long_postselection_v1/CODE_BINDINGS.md](https://github.com/Orangekostar/Persist4D/blob/6b1b779172afdeb51e2719a737cd05e7eeeca982/artifacts/native_long_postselection_v1/CODE_BINDINGS.md)；阅读范围：全文；符号：`代码与实测证据的对应`。

**已观察**：已有代码/结果绑定明确评价、交接、LFS和外部回执范围。

**本轮决定**：发布新结果沿用可追溯原则，但不建设新审计平台。

### R22

- [conf/config_native_long_retrain.yaml](https://github.com/Orangekostar/Persist4D/blob/6b1b779172afdeb51e2719a737cd05e7eeeca982/conf/config_native_long_retrain.yaml)；阅读范围：全文；符号：`general; data; model; optimizer; native_long`。

**已观察**：原配方设Q100、native flags和冻结政策；YAML默认batch不等于运行后实际fallback。

**本轮决定**：从导出解析配置迁移并单列本轮冻结全部父模型的改变。

### R23

- [scripts/native_long_assets.py](https://github.com/Orangekostar/Persist4D/blob/6b1b779172afdeb51e2719a737cd05e7eeeca982/scripts/native_long_assets.py)；阅读范围：全文；符号：`ENCODER_PREFIXES; export_task_bundle; load_task_bundle`。

**已观察**：任务包只省略编码器参数，保存任务权重及缓冲区，加载校验源码身份。

**本轮决定**：不能漏掉Concerto任务decoder；新delta必须依赖完整父包。

### R24

- [models/pointcept.py](https://github.com/Orangekostar/Persist4D/blob/6b1b779172afdeb51e2719a737cd05e7eeeca982/models/pointcept.py)；阅读范围：1–190、430–695；符号：`PointceptBackbone; _load_state_dict; forward; encoder; serialization helpers`。

**已观察**：PointceptBackbone分别编码和解码，encoder_only仅加载embedding/enc预训练。

**本轮决定**：冻结父模型仍包含已训练任务decoder；机制缓存不得假设只需一个裸特征矩阵。

### R25

- [scripts/native_long_confirmation_inference.py](https://github.com/Orangekostar/Persist4D/blob/6b1b779172afdeb51e2719a737cd05e7eeeca982/scripts/native_long_confirmation_inference.py)；阅读范围：1–130；符号：`formal_dataset; formal_samples`。
- [artifacts/native_long_postselection_v1/protocol/FORMAL_POPULATION.json](https://github.com/Orangekostar/Persist4D/blob/6b1b779172afdeb51e2719a737cd05e7eeeca982/artifacts/native_long_postselection_v1/protocol/FORMAL_POPULATION.json)；阅读范围：1–45；符号：`claim_boundary; coverage`。

**已观察**：正式数据loader使用validation模式及validation_aliases映射，PB有单独ambiguities。

**本轮决定**：明确分离train与validation索引空间；PB不复用训练index定位。

### R26

- [scripts/p6a_metrics.py](https://github.com/Orangekostar/Persist4D/blob/6b1b779172afdeb51e2719a737cd05e7eeeca982/scripts/p6a_metrics.py)；阅读范围：1–220；符号：`FrozenRawObservation; assert_shared_raw_predictions; adapt_raw_local_prediction`。

**已观察**：官方指标包装有FrozenRawObservation、候选指纹、raw_local和strict_online相关路径。

**本轮决定**：官方充分状态与运行预测需分别绑定；不平均场景AP冒充总体。

### 论文一手来源

- **P01** [SAM-V: Geometry-Aware Segment Anything for Multi-View Instance Segmentation](https://arxiv.org/abs/2609.25490)，2026-09-21。
- **P02** [MOTR: End-to-End Multiple-Object Tracking with Transformer](https://arxiv.org/abs/2105.03247)，2021-05-07 (first submission)。
- **P03** [S4VY: Segment Anything in Feed-Forward 4D Visual Geometry](https://arxiv.org/abs/2609.36875)，2026-09-29; v2 2026-10-03。

P01用于说明“条件进入掩码解码”的启发；P02、P03用于限定查询传播/持久query的新颖性边界。本文新几何摘要、矩阵、超参数、预算与晋级阈值都是预设实验设计，不是这些论文已经证明的结果。
