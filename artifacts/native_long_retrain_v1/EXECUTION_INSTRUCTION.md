# ReScene 原生长序列重训：最终 Codex 执行指令

版本：native-long-retrain-v1；编写日期：2026-09-30。

**全文是本轮唯一执行规范。先读完，再开发、实跑、选模、审核和发布。** `protocol.yaml` 是本文的结构化镜像；历史提示词、旧 campaign 的门槛和固定目录不叠加生效。发现镜像冲突先按本文修正镜像，记录一次，不分别执行两套计划。

本文作者已经读取下列源代码和历史产物，但没有执行用户服务器训练。拟新增文件、类、接口、阈值和训练策略均是本轮设计，不是声称仓库已有实现或论文已经验证会涨点。

---

## 0. 目标、范围与最终需要回答的问题

**目标：重新建立可靠的原生 ReScene 基线，研究真实长前缀训练及两个独立模块是否提高 T3–T5 的官方 t-mAP，同时保留 T1/T2 能力。** 先证明相对充分训练原生模型的收益，再考虑流式压缩。不要只打败旧 R1 或一个训练退化的控制。

现有默认模型/配置保持不动；新方法只通过本轮显式实验配置运行，正结果也不自动改默认部署。

本轮固定四臂：

| ID | 名称 | 结构和训练内容 | 主要归因比较 |
|---|---|---|---|
| E0 | B-REPRO | 原生 ReScene；固定预训练编码器；任务参数新初始化；3RScan T2＋ScanNet T1 | 新的受控原生短训练基线，不冒称作者逐项完全复现 |
| E1 | B-LONG | 与 E0 相同结构和任务初始化；3RScan 真实 T2–T5 长度课程 | E1−E0：受控长序列训练配方收益 |
| E2 | B-LONG＋S | E1＋训练时阶段分层注意力采样；总采样量不变 | E2−E1：采样机制收益 |
| E3 | B-LONG＋F | E1＋查询条件 mask 特征反馈；从任务训练开始联合优化 | E3−E1：内部特征更新模块收益 |

每条轨迹从共同的任务初始状态开始。E3 不从已训练 E1/E2 接着训练；E2/E3 不共享训练后的参数或 optimizer。可以共享原始输入、只读数据清单及初始公共权重。

**本轮不训练组合模型。** S/F 都出现收益，也只产出下一轮组合建议及兼容性记录。预算优先保证基线与独立模块，不把组合、关系评分、点级重建、LoRA、390/450尾段续训、D0/lag1、对象记忆、教师蒸馏、变化分类头或新骨干一并加入。

“完成”要求完成实测预算所授权的计划和明确的覆盖状态，不要求确认名单非空。预算允许且未遇到实质阻塞时必须运行，不能停在代码、单元测试或 smoke；预算不允许完成完整重训时按第10节处理，禁止缩短训练后仍标“450轮完整复现”。

## 1. 固定源版本及证据边界

### 1.1 源和目录

- 仓库：`Orangekostar/Persist4D`。
- 读取基准：`c6e9d01edfe2b3c832374a424fc45e44bdadfb68`。
- 已包含修复祖先：`465f37a46972e81584c1563bba57e1c05c911c1a`。
- 新分支：`research/rescene-native-long-retrain-v1`。
- 新 tag：`rescene-native-long-retrain-v1`。
- 新产物目录：`artifacts/native_long_retrain_v1/`。
- 运行根默认：`$HOME/persist4d_runs/native_long_retrain_v1`，允许启动时指定。

在现有仓库执行 `git worktree list --porcelain`，从上述完整基准新建独立 worktree。若用户已有同任务分支则按身份恢复；存在不相关同名分支时使用最小 `-r2` 后缀并记录。保留用户未提交文件，禁止 reset/clean、改旧 tag、force-push、merge main。若更新基准，应先记录 diff 和依赖迁移原因，不静默改成默认分支。

### 1.2 启动必须读取的已有材料

读取以下真实文件及其直接引用的运行配置、成本/数据清单。只补缺少的内容，不重做历史全仓库审计。

1. `artifacts/qp_mn_targeted_v1/{FINAL_REPORT.md,CODE_BINDINGS.md,BASE_BINDING.json}`。
2. `artifacts/qp_mn_targeted_v1/resources/NATIVE_RUNTIME_NOTE.md`、对应原生顺序复现记录。
3. `artifacts/rescene_task_learning_root_cause_v1/{HANDOFF.md,CODE_AUDIT.md}`。
4. `artifacts/rescene_task_learning_root_cause_v1/initialization/COMMON_INITIALIZATION.json`。
5. `artifacts/rescene_task_learning_root_cause_v1/full_candidate/{FULL_TRAINING_MANIFEST.json,learning_curve.csv,checkpoint_inventory.csv}`，存在不同名字时从同目录真实 listing 定位，不猜文件内容。
6. 最近正式确认的人口来源：perception_gain_v2 的 PB/LOCAL/ADDITIONAL 和旧 CAL/SEL reference 清单，沿实际报告/索引读取。
7. 本轮使用的已安装 Concerto、Sonata、stmetrics、PyTorch/Lightning 源位置与版本；不更新依赖来追求“最新版”。

已读事实和用途：

| 事实 | 当前直接依据 | 本轮决定 |
|---|---|---|
| 旧 R1 已完成29700更新/450轮，选择390轮 | full training manifest | 本轮不是旧模型欠训补跑，也不载入 R1 任务权重 |
| weighted/raw_sum 曾影响复现；配置缺省仍可能 weighted | rootcause config、trainer reducer | 显式绑定真实 `raw_sum` 及梯度，不只看权重字典 |
| 原混合数据权重1.0:0.8，按数据集大小归一 | mix.yaml、MultiDataset._setup_sampler | 所有新臂共同保留5/9与4/9抽样概率 |
| 原注意力训练采样合并时间点池；is_eval禁用限额 | ReScene.sample_and_batch_features | S只改训练抽样组成，不增加数量或改推理预算 |
| mask特征在查询循环前生成，原扩展钩子只返回query | ReScene.forward/after_decoder_stage | F必须更新后续mask计算实际读取的F |
| Q-P/M-N负结果，±2头有表达/覆盖限制 | 最新结果及诊断 | 不再重复同配方小头；不把F说成其涨点复现 |
| 原生冷启动有过程历史敏感性 | NATIVE_RUNTIME_NOTE | 新训练前定向处理/量化，不能靠94输入回放假装一般可复现 |

### 1.3 不把不同资产混为一谈

固定 Concerto 预训练文件：SHA256 `845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07`，433,987,358 bytes。

历史 R1 selected：SHA256 `629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd`，754,813,672 bytes；只作历史对照，不用于任务 warm-start。

**历史共同初始化的已发表清单**记录 SHA256 `d941b59ce95a8bb27bf5627f621cafe7c399a7a66b71ce05460782560fe98d4f`、540,886,910 bytes。它只作结构/来源参考。不得将另一份交接摘要中的相似路径或不同摘要视为同一文件；本轮重新生成公共初始化，使用本轮实测 hash。

所有本地路径通过已知资产索引与 worktree 定位，禁止扫全盘。权重各核对一次，不每步 hash。新模型性能不从历史文件名的三位小数读取，使用实测指标。

## 2. 代码—改动—实验证据绑定

下表的新增动作是待实现的最小方案。优先复用成熟函数，不复制整套旧 campaign 和历史授权规则。

| 已有位置 | 阅读重点 | 本轮动作/验证 |
|---|---|---|
| `models/pointcept.py::__init__/_load_state_dict/forward_override` | Concerto构造与权重加载、embedding/enc/dec、mixed序列化 | 新增仅加载预训练编码器的明确模式；不能让任务decoder意外继承训练权重 |
| `utils/rescene_rootcause_preflight.py::load_common_initialization/validate_common_tensor_state` | 初始公共参数完全对应 | 复用纯函数；不继承旧实验名、450轮授权文件和固定变体门槛 |
| `conf/config_rescene4d_concerto_rootcause.yaml` | 默认weighted、物理batch、OneCycle、冻结模式 | 新配置显式覆盖，移除旧preflight/callback输出身份 |
| `datasets/preprocessing/RScan_preprocessing.py::create_sequences/process_sequences` | 元数据、真实scan数、别名、ambiguity | 只生成必要元数据；不足T不得重复scan，禁止重分割整库 |
| `datasets/semseg.py` | `_load_scan_sequence`及实际scan索引入口、增强、编号、空输入 | 新长输入适配器以显式scan列表读取，保持GT实体与分区一致 |
| `datasets/pointcept_utils.py::voxelize/get_instance_masks` | 独立阶段体素化、两个batch命名空间、inverse和GT | 检查混合T2/T5批次；保持完整样本和时序标签，无假T5 |
| `datasets/multi_dataset.py` | 权重/长度与WeightedRandomSampler | 新可恢复计划保留域混合比；reference优先方案对四臂共同使用 |
| `models/rescene.py::sample_and_batch_features` | idx同时作用于feature/attention/position；padding | E2增加GT无关的stage配额抽样，默认路径不变 |
| `models/rescene.py::forward/mask_module` | 12次更新、辅助输出、F先生成 | E3增加一次F反馈，返回并使用更新F；不能只改query |
| `models/criterion.py`、`models/matcher.py` | legacy CE/BCE/Dice、InfoNCE及aux | 全臂保持；不加新worst loss，不偷偷去重raw_sum返回项 |
| `trainer/trainer.py::_configured_objective_loss/training_step/configure_optimizers` | 实际总loss、步数、冻结、调度 | 用新训练子类/薄入口复用；排除旧固定66步/epoch回调约束 |
| `scripts/rescene_task_postprocess.py`、`scripts/p6a_metrics.py` | 原生soft→bool、score/top-k、官方时序评价 | 新权重接原生路径，不经D0，不改官方算法 |
| `scripts/qp_mn_campaign.py`及其delivery接口 | 目录参数化、缓存依赖、A/B发布 | 仅复用窄工具，不能把旧head模型加载或旧人口硬编码带进来 |

建议新增不超过这些职责：`datasets/native_long_dataset.py`、`models/native_long_modules.py`、`trainer/native_long_trainer.py`、`scripts/native_long_campaign.py`、`conf/config_native_long_retrain.yaml`，以及必要少量测试。具体可合并，文件数量不作为成果。

## 3. 数据划分与真实长前缀

### 3.1 本轮选择人口必须先于训练固定

以官方元数据中 `type=train` 且至少有两次合法不同扫描的3RScan物理reference为候选；核对processed数据库与元数据对应，禁止根据当前模型AP选场景。排除已被定义为PB、LOCAL、ADDITIONAL及旧holdout的reference；若发现源清单有交叠，按物理reference去重并保存交叉表，不能仅按scan名区分。

从剩余reference建立新的内部 CAL/SEL，默认各8个，避免继续仅靠旧4个reference调参。**这是为了新任务模型的开发划分，不表示预训练编码器或历史研究从没看过这些场景。** 所有新增训练不得读取新CAL/SEL及正式确认reference的标签或特征用于优化。

确定性选法：按真实可用不重复扫描数分成Tmax=2、3、4、≥5四组；组内按 `SHA256('native-long-7001:'+reference_id)` 排序。先在≥5组为CAL、SEL各取1个，并至少给TRAIN留1个；≥5组≥6个时各取2个、至少留2个。随后按4、3、2、≥5轮转，CAL/SEL轮流取组内下一项，直到各8；普通组至少留1个TRAIN reference；≥5组原数量≥6时整个填充过程始终至少留2个TRAIN，否则留1个。空组略过，无可取项即停止，不无限循环。无法达到8则保留实际数量，但每侧必须≥4且都至少有一个真实T5 reference。

如果无法同时满足TRAIN、CAL、SEL的真实T5覆盖：标 `BLOCKED_LONG_POPULATION`，不伪造、不重复扫描、不偷偷借正式测试补训练。可继续完成E0及E1/S/F实现和已支持长度的只读检查，但不将它们标为已完成本轮T3–T5训练/选模。仅缺某个旧历史对照文件不构成此共同阻塞。

生成 `POPULATION.json`，列原始split来源、reference、scan_ids、Tmax、角色、数量和所有排除原因。保留原数据，不直接覆写train_database.yaml。

### 3.2 序列来源与身份

优先复用`sequence_database_sliding_2/3/4/5.yaml`的合法TRAIN记录及已验证原生长前缀读取路径。缺长序列索引时，用 `RScanPreprocessing.create_sequences` 的语义做元数据级索引生成，输出本轮运行目录；不调用会重新写全量点云/标签的整个preprocess构造流程。

实际组序列时必须用已核实的scene别名映射和显式scan列表，不用UUID字符串随意split('-')。序列内所有scan来自同一reference、互不重复。GT全序列身份使用该reference已有全局实例映射；不同scan相同的局部数字ID不是自动同实体，不把未经核验的两两对应传递成新GT。

所有输入统一提供xyz+t四维坐标；T1的t全0，不能在同一physical batch把一个样本做成D=3、另一个做成D=4并依赖循环最后的D推断格式。

T>2的`changes`在旧实现中不是每个transition独立分类目标。本轮变化头关闭，不伪造T−1标签；出现/消失由实例mask在各阶段是否为空和官方指标处理。保留可用ambiguity，未提供则记录缺失，不捏造清理过。

训练语义类别沿用18个前景类、ignore和label_offset处理；真实背景作为负点，unknown不改普通负点。检查真实长GT实例的类别一致性，异常按既有标注规则记录，不手工将模型预测类别写回GT。

### 3.3 域混合、reference采样与课程

每个逻辑输入以概率5/9选3RScan、4/9选ScanNet TRAIN，即1.0:0.8。ScanNet单扫描保持其官方训练split，禁止混入其validation/test。3RScan先均匀抽reference，再在reference内选真实扫描顺序与起点。

本轮**四臂共同采用reference优先**，不同于历史可能按扫描对频次采样。E0必须标为“原生架构/受控重训协议”，不是与作者所有数据采样完全一致。不得把这项共同协议变化归因于S/F。

每个reference预生成3种由元数据和固定seed决定的无重复scan排列，使用不同起点。没有可用真实日期时不声称这些排列是物理时间顺序；它们是稀疏重访输入次序，与作者允许扫描重排的任务一致。对应坐标t只为当前输入0…H−1。

用全局optimizer更新u、microstep、rank、input_slot和seed确定dataset/reference/order/start的draw。E0/E1共享这些公共随机决定；E0取该顺序中前2个scan，E1按课程取前H个，长度不足时仅从可用H集合归一化抽样。E2/E3与E1共享完整输入、长度和增强计划。

令p=u/U，U=29700：
- p<0.10：3RScan只取T2。
- 0.10≤p<0.25：a=(p−0.10)/0.15；权重w2=1−0.75a，w3=w4=w5=0.25a。
- p≥0.25：w2=w3=w4=w5=0.25。
- 对reference真正具有的H裁掉不可用项后归一化；禁止补重复scan。

E0总取T2；ScanNet始终T1。长度课程是预先规定训练配方，不用SEL改比例。每100更新记录实际各H、reference、scan和体素数；同reference全周期不断出现并不等于新增独立数据。按H另统计有效GT实例联合数量和超过Q=100的输入数；不因此按GT删对象/增Q，若有容量压力如实列为本轮结构限制。

E0/E1等optimizer更新、等逻辑draw，不等扫描数量、点数或GPUh。E1−E0只能先解释为长度训练配方效果，不能未经额外计算量对照就称同算力优势。E1/E2/E3才共享完全相同的输入。

### 3.4 输入/增强和loader

TRAIN保留核实过的原训练增强；四臂用独立于模型算子的per-draw RNG得到同一增强。只对空间和外观做既有增强，不歪曲t或跨扫描实例标签。CAL/SEL关闭 `apply_training_augmentation`，保持原split和target格式；不得直接切到会返回无标签/不同GridSample格式的`test`。

实际Pointcept collator从raw颜色/法向与坐标构建feat，而不一定使用dataset返回的features字段。按真实调用链记录，不顺带“修正颜色增强”形成另一个变量。

构造一个含不同点数、不同H的两样本批次，检查每个样本的full/low时序、GT列、point2segment、inverse及padding；再检查一个3RScan＋ScanNet批次。现有test分支中`original_temporal_stages`的外层变量用法只在确实进入且影响本轮时修，不能看到代码味道就重写全部collator。NOAUG带标签验证使用实际正确分支。

不丢弃困难/大点云评价输入，不以sample0替换原reference冒充覆盖。已知不可读或无合法监督输入在启动清单中明确排除；合法“某对象某阶段为空”必须保留。训练物理batch内空目标样本不得让整个batch返回空列表；需要时启用已有 `preserve_empty_targets` 并检查criterion。

## 4. 新初始化与共同原生模型

### 4.1 编码器预训练、任务从头训练

使用固定Concerto配置构造网络，**仅加载其实际embedding＋encoder及其所属必要buffer**。骨干decoder、mask_features_head、查询投影、query decoder、mask/class heads由其原构造器重新初始化。不要先加载全部checkpoint后对每一层粗暴调用reset_parameters：那可能破坏专用初始化或遗留嵌入/normalization。

推荐在PointceptBackbone加入显式`pretrained_scope='encoder_only'`选项，默认保持历史行为；新模式先原生构造，再按实际模块对象/严格前缀白名单过滤预训练字典。missing/unexpected必须逐类解释；不允许`strict=False`吞掉encoder未加载。constructor出现非encoder预训练键时本轮排除并记录。

seed45生成一个新的 `common_initial_state.pt`，在optimizer创建前保存公共model/criterion参数和buffer、键名类别、来源、大小/hash。E0/E1/E2复制全部公共张量；E3仅多F参数，公共键同shape同bytes。新F初始化使用fork_rng独立流，不能改变基线构造和随后数据抽样。第二seed46重新构造其自己的共同初始状态，不用seed45 checkpoint改文件名。

历史共同状态仅用于核对加载语义，不直接当作已确认的新初始状态；禁止用390/450任务权重和旧optimizer继续训练后称从头。

### 4.2 固定架构/训练行为

- Q=100、hidden=128、heads=8、FFN=1024、4尺度×3轮、共享query decoder、FPS位置＋零query内容。
- voxel=0.02、mean segment pooling、原始输出分区/score/top-k/filter规则。
- mixed序列化`standard+temporal_overlay`；temporal mask pooling关闭；changes head关闭。
- 预训练encoder权重冻结；沿旧R1 `frozen_encoder_eval=false` 的训练运行策略共同保留，验证全模型eval。冻结权重不等于其训练期DropPath/归一化buffer全部不变；记录实际模式/可变buffer，不宣称所有encoder输出恒定。
- 如第5节发现可变buffer或随机顺序是真实故障，必要修复先形成共同基准版本，再让四臂都从新初始状态重启；不能只给新方法使用更有利的模式。
- 只训练非encoder任务参数与新增F；不把GT-derived descriptor输入网络。
- mask loss为legacy CE/BCE/Dice与已验证InfoNCE；不加S-BAL/S-WORST、不改EOS0.2。

### 4.3 raw_sum必须是实际目标

显式设置 `general.rootcause_objective_mode=raw_sum`，关闭任何遗留weighted开关；用新实验身份，不继承旧rootcause授权manifest、历史save_dir或回调。

复用 `_configured_objective_loss` 的raw_sum语义，对一个真实batch列出criterion返回的每个键、实际参与系数、标量和参数梯度；与本次明确的sum逐项一致。**不要自行把aggregate/per-layer返回键“去重”后仍称R1原目标**。若认为原公式重复计数，应另立将来实验，本轮四臂一致保持。F不额外增加返回loss键。

旧P2/rootcause某些checkpoint与sampler代码固定了66更新/epoch、450轮、旧schema。新子类/薄入口必须用本轮真实draw位置与max_steps，不能伪造旧authorization或把新模块塞进旧固定参数列表。保留非有限值/输入错位的必要错误处理，不重写成熟训练算子。

## 5. 开训前的定向运行核验，不再全仓库审计

### 5.1 仅处理已经暴露的过程历史敏感性

在当前环境，用两个固定合法输入A(T2)、B(T5)，做独立进程中的A、B→A、A→B→A，各两次；固定input与RNG状态，比较输入/映射、encoder、decoder、maskF、query、最终输出，找第一处差异。最多0.5 GPUh计入开训预算，不重复94输入多轮重放。

优先检查调用中会改变的module/buffer/缓存、序列化随机顺序、显式RNG和稀疏算子状态。没有直接证据不要说“CUDA bug”。

共同评价使用per-input RNG隔离：seed=hash(9001,固定input_id)，保存并恢复Python/NumPy/Torch CPU/CUDA RNG；模型构造RNG独立。训练用per-draw RNG，不能每次重新设置相同seed而让增强恒定。mixed序列化仍保留，只让其随机性可追踪，不单独给E3关闭原生机制。

- 能定位并修复：保存最小补丁及真实反例，重新跑上述有限核验；四臂使用同一修复基线。
- 仍有差异但每种固定新进程协议内可重复：保留固定protocol，并量化输出/小panel AP差异，标 `RUNTIME_CONDITIONAL`；允许受控训练，但所有微小增益还必须过新进程成对核验，不能宣称一般部署复现。
- 同一固定protocol也不可重复、样本输出错位或有非有限值：只阻塞真实训练，继续完成可用代码/交接，标 `BLOCKED_NATIVE_CORRECTNESS`，不放宽到任意容差来PASS。

若确有残余差异，最终声明依赖条件；不把先跑94张其他输入当作无成本生产需求。

### 5.2 实际训练可行性与物理batch

优先2张同型空闲A40，每卡batch2、累积8，有效batch32。用最长且较大点数的真实样本、E3网络检查前向＋backward，而非只测一个小T2样本。

OOM时按固定顺序：释放无关张量/正确的非重入checkpoint和精确chunking；仍不行→全臂共同每卡batch1、累积16。只有单卡可用时共同batch2累积16或batch1累积32。**全臂物理batch统一，不能E0用2而E3用1后称完全同条件。** 不静默改AMP、体素、query、points限制或丢掉大reference。

输入批次/GT对比和F最大样本核验加上数值调查总预检预算1 GPUh。需要真实bug修复可在总预算内超出并说明，禁止为了达到测试数量扩展。

## 6. S：固定总数量的阶段分层采样

S只启用在E2，作用于 `sample_and_batch_features` 的**层级cross-attention输入采样**。不改变构造完整maskF的第一次调用，不改变GT/mask loss抽样，不改变query初始化。

### 6.1 精确索引算法

先沿原函数确定本batch实际 `curr_sample_size`，包括max长度、层级cap和is_eval逻辑。对样本b，N_b为该层位置数，k_b=min(N_b,curr_sample_size)。从与本层feature严格对齐的真实时序坐标得到stage_ids；不能从平均后非整数时间随意round猜阶段，应追踪空间层级对应。T1只有0。

若is_eval=True或N_b≤curr_sample_size，完全走原arange＋padding路径；真实H=1始终调用原始采样分支，需截断时采用与E1相同seed的randperm，不改成arange。S没有可选的推理加采样。

需要截断时，令n_t为各阶段可用位置数。配额a_t从0起，以stage升序循环，每次给仍有容量的stage加1，直到sum(a_t)=k_b。这等价于带容量约束的均衡配额和确定性余数分配；可向量化但结果要相同。

每阶段内无放回均匀取a_t个；使用本draw/层级/样本的专用generator。按stage升序拼接被选位置，构造padding后，与原函数一样**用同一个idx同时取features、attention mask、position和所有extra**。

原均匀采样与S都使用专用sampling RNG；S不能因为多调用随机数改变encoder DropPath、数据增强或其他未修改路径。无需两算法取完全相同点，但公共随机组件应不受影响。

### 6.2 记录与边界

每100更新累计各层cap触发数、分阶段配额、不同H的实际可见数量及padding，单次抽样不使用GT。T1及无需截断时与原路径逐元素一致；异长batch、某阶段小于配额、最后余数和extra映射各用一个简短例子检查。

若所有真实训练样本均不触发原采样cap，则S没有实际干预，标 `NO_INTERVENTION_SAMPLER_INACTIVE`，不通过更小cap人为激活。本轮E0/E1/F继续；S不占完整训练预算。

S不是“阶段均衡loss”，不是A-OPEN，不保证小目标或GT被采到；需要通过正式t-mAP检验。

## 7. F：在原生查询循环内部更新mask特征

F只启用E3，task decoder从开始共同学习。第一版使用segment-level F，不声称点级边界重建，也不引入外部记忆。

### 7.1 插入位置与必要代码改造

现有 `after_decoder_stage` 只返回queries。新增一个默认恒等的 `update_mask_features(queries, mask_features, padding_mask, execution_stage_idx)` 钩子，或等价的清楚返回接口。

在execution_stage_idx=7（第2轮第4尺度的FFN/query更新后）只调用一次F；把返回值写回真正的`batched_features`。后续第8…11阶段的mask、由mask构造的attention，以及最终mask必须读到更新后的F。此前辅助输出保持原历史值，最终辅助输出数量仍为12。

保持原 `segment_features=[agg_feat]` 的InfoNCE输入和loss键不变；F返回新张量，不in-place修改原agg_feat或其引用，本轮不增加对反馈F的第二份对比损失，以免把结构效果与loss变化混合。未来若增加新监督，另立实验。

不能只新建一个模块但从未接入最后prediction；也不能只改归档的可视化特征。用一次真实batch在F非零参数下验证返回F、后续logits及梯度确实变化。

### 7.2 唯一网络定义

F为[B,S,128]、Q为[B,100,128]。仅使用模型自己的query；no-object query也不由GT删去。

1. 对F与Q各用独立LayerNorm。
2. 8头reverse cross-attention：query来自LN(F)，key/value来自LN(Q)；q/k/v线性宽度128，dropout0。它让每个segment读取实例查询信息，不是让query再次读F。
3. attention输出经过128→128投影W_o，并残差加到F。
4. 对结果做pre-norm FFN：128→512→128、GELU、dropout0，再残差相加。

式子：

```
A = Attention(Wq LN_F(F), Wk LN_Q(Q), Wv LN_Q(Q))
F1 = F + Wo A
F2 = F1 + W2 GELU(W1 LN_2(F1))
```

W_o和W2的weight/bias全0初始化；其他层原生初始化。**不要再加一个也初始化为0的乘性门控**。这样初始输出等于F，但输出投影可以从第一次有效反向获得梯度，内部qkv在后续更新获得梯度。最后projection零梯度与第一次trunk零梯度要分别解释。

使用完整100个query，无GT mask或置信度硬门控；阶段信息由原查询/特征携带，第一版不额外加新位置编码或实例局部框。

S很多时按segment query维度chunk512；每个chunk仍访问完整Q，结果与非chunk数学一致；必要时使用保梯度的activation checkpoint，不detach F/Q。padding segment在反馈前后置回0，不能污染pool/输出。

### 7.3 F的科学范围

这是真正的“查询条件segment特征更新”，不同于旧M2冻结F后输出±2系数。它仍保留原超点边界，不会自动解决跨物体超点；也不是完整Relation3D或MoGA复现。

E3−E1证明的是新增反馈block的整体作用。额外参数与实例条件反馈的更细因果区分，需要将来的参数量控制；本轮不能仅凭一次query打乱诊断声称已完全隔离“关系建模”机制。

初始F=恒等与公共模型参数相同时，确认一次真实T2和T5前向的预测一致；不使用“检测0权重直接返回旧预测”的特殊分支跳过数学路径。F训练后读取本轮训练参数，严禁复用旧R1输出缓存代替它。

## 8. 训练计划、恢复与实际成本

### 8.1 完整训练定义

默认完整预算U=29,700 optimizer updates、有效batch32，即950,400个逻辑输入draw（不含只读评价）。这是为了与旧完整训练的更新预算对应，**不声称新loader也恰好450个epoch**。课程与OneCycle从第一步就按U定义。新Trainer以`max_steps=29700`控制完整训练，`max_epochs=-1`或足够大且不先触发；不可继承旧450-epoch提前终止。pilot停止仅改变执行终点，不改变scheduler总长；恢复到U时不重建新OneCycle。

AdamW峰值lr5e-4、betas(.9,.999)、eps1e-8、weight_decay.01；OneCycle pct_start.3、cos、div25、final_div1e4、cycle_momentum=True、momentum .85/.95。gradient clipping取实际原完整R1解析配置的值，不能从最近小头clip1直接搬过来；绑定时缺失则显式设置不裁剪并给四臂共同使用，记录与旧路径是否一致。FP32、TF32关闭、matmul precision为highest，共同禁用torch.compile。torch算子的确定性策略继承第5节已核验的共同运行条件，不因单个方法报错暗改；记录实际warn-only/strict状态，不仅记录seed。

每990更新保存恢复checkpoint；每2970更新完整CAL评价，即10%、20%、…100%。0步只做一致性与初始损失检查，不花完整CAL/SEL计算。pilot结束为11880（40%U），已过OneCycle峰值，仍不等于最终性能。

同一次训练允许按更新边界恢复，但必须保存model/criterion/optimizer/scheduler、下一全局draw位置、每rank必要RNG、配置/代码/数据身份。新loop不得受旧epoch-boundary-only断言误阻塞；prefetch读过但未完成的draw不算已训练。一个短的split-resume测试确认下几个draw、LR和参数恢复一致即可。

### 8.2 不能用R1特征缓存训练任务网络

R1编码器之后的骨干decoder和mask特征在本轮可训练，**不能复用旧R1完整前向特征**；S/F也必须参与每次真实forward/backward。第一版不做encoder缓存，避免随机训练运行策略、输入增强和可变buffer的身份混淆。

如果需要减少显存，只使用正确的精确chunk/checkpoint，不把训练变成离线小头。评估缓存按本次完整checkpoint、input/RNG/代码/物化/metric身份建立；每个新checkpoint需要新预测。

### 8.3 共用采样与DDP

四臂全局domain/reference计划相同，E1/E2/E3的长度和输入完全相同。shuffle、augmentation、stage sampler和F初始化各有独立RNG流。

固定global batch32按rank与microstep切分，不允许每个rank各自独立WeightedRandomSampler后造成重复样本或错误epoch长度。应记录真实draw IDs；无损分布不要求全部draw唯一，要求它们遵从规定有放回抽样且恢复不重计。

不同H可同physical batch，匹配按样本独立进行，t坐标与分区不跨逻辑样本聚合。长序列不读未来窗口缓存来补数据；一个native H前向只读取该列表中的H次扫描。

## 9. 固定评价人口、筛选与确认

### 9.1 内部开发输入

每个CAL/SEL reference生成3种固定元数据hash排列。对每种排列，只取存在的T2/3/4/5前缀各一个；相同ordered scan tuple去重。T1为这些输入所涉及的physical scan去重后独立前向。

没有T5的reference仍参加其可用长度，缺失H不填0。主分数：

`S_long = (pooled_tAP_T3 + pooled_tAP_T4 + pooled_tAP_T5)/3`。

三个长度必须各有人口和完整覆盖；不因一项缺失改成两项均值继续沿用名称。各H人口在训练前冻结，并额外报告同一T5 reference集合的嵌套前缀表，区分长度与人口变化。

### 9.2 基线、指标和score

使用原生 ReScene输出及锁定stmetrics，H1使用空间评价，H2–5使用完整prefix的temporal评价。保持官方ignore、最小区域、ambiguity和严格阈值；不得重用拼接IoU替代temporal匹配。标签repair接口仅在诊断需要时复用，不改变官方指标。

不启用Q评分头。正常训练后的query/class/raw score会变化，这属于任务模型训练，不能要求不同checkpoint输出的score/mask像小头试验一样冻结。相同后处理参数不等于相同候选集合；候选数与重复量分别报告。

验证先使用单GPU原生前向＋CPU累积官方充分统计，避免DDP padding重复输入。各方法/各H accumulator独立；缺一项输入就写INCOMPLETE与已完成/预期分母，不能继续用残缺AP选胜者。全部实测指标保留机器0–1单位，报告%=100AP、pp=100ΔAP。

### 9.3 开发选择规则

E0按CAL T2选检查点，T1打破平局，再取更早step。E1/E2/E3按CAL S_long选，依次按T2、T1、早step打破1e-6内平局。

在同step报告E1−E0、E2−E1、E3−E1；同时报告各自CAL选中点的工程比较，不能混淆。为真正模块对照，E1额外保留S/F所选step的checkpoint，并在固定SEL中作同step比较。

40%pilot只用于预算分配：模块ranking取30%与40%两个点相对同step E1的S_long差值均值。先比较两点T2下降均不超过0.002的候选；如果没有，按平均S_long差值、平均T2差值、预声明优先F、ID顺序排名，允许最有希望者继续完成，不能把一个早期负数当成方法定论。

所有当前预算计划里的seed45轨迹完成或明确停止后，一次CAL lock；随后评价所有已锁定臂的完整SEL。资源筛选停止的臂只以已完成pilot点评价，标 `RESOURCE_SCREENED_PILOT`，不能和完整训练混为同级证据。

### 9.4 正收益等级

对模块相对E1，主看S_long，同步报告相对E0：
- ΔS_long>1e-6：初步长时正信号；≤则不称正收益。
- T1、T2或任一T3–5下降超过0.002：记 `LONG_TRADEOFF`，保留真实数据，不当全能力胜出。
- ΔS_long≥0.005只是“0.5pp研发目标幅度”，不是显著性证明。
- 逐reference报告可用H的差值、正/零/负计数；不以“每个reference都正”作为强制筛除，也不把相关prefix当成独立样本。
- 被预算筛选而未跑完整的臂，不作“不可能涨点”的结论。

第二训练seed46只对至少一个完整训练、无上述严重取舍、SEL正收益的新增模块及其E1控制触发。默认只取排名最高的一个，按ΔS_long、最差H、T2、参数更少、ID打破平局。两臂从seed46新初始状态训练到U，**只评价seed45已锁定的对应step与完整U预声明端点，不在seed46重挑epoch**。预算不足时标NOT_REPLICATED。

E1本身胜E0而S/F不胜E1，也是一项真实训练配方结果；可按同预算规则优先复核E0/E1，明确不是新结构成功。只允许一组追加第二seed配对，不能把两种情形都自动追加。

### 9.5 正式确认

锁定结果后，按已存在真实manifest运行PB T2–T5、LOCAL-T2和可用ADDITIONAL；不同人口分表、重叠reference单列。默认确认E0、E1和最多一个SEL正向完整新增模块，负模块不再消耗整套正式评价。若无模块正收益仍确认两基线，回答“重新训练本身是否改善”。

正式确认主评价seed9001（per-input确定）。必要时再以9002/9003各作一遍推理重复，不是新增训练seed；预算允许才执行，并明确推理随机性与训练随机性不同。禁止按正式结果换epoch或换另一个模块。若seed46配对已完成，其固定点作为独立行确认；不与seed45挑较高分，也不把evaluation seeds当训练seeds。

旧R1需要时在同一新运行/输入路径下重新测，不将历史不同producer的数值直接减成训练收益。若正式人口中某reference误入新FIT，立即标污染并修正规划，不从正式表删除该reference伪造完整原协议；保留错误证据并仅对合法分离人口作新命名报告。

“超过ReScene”分为：E0较旧R1为复现/训练改进；E1较E0为长度训练改进；模块较E1为结构增益。作者34.8%只有严格匹配其协议才能比较，不能拿新内部CAL或PB相减。

## 10. 预算必须先实测，再决定完整训练数量

### 10.1 上限与必要预留

最近已披露累计80.7350387502 GPUh只是启动前参考值。读取最新event账本加入之后工作、去重prior小计，真实可用余额R=192−prior。不自动再授权192，不以本轮小头0.23 GPUh估计整网训练。

从R中预留8 GPUh用于固定SEL、正式确认、运行敏感性复核、真实profile与重载。预检最多1 GPUh已计在R内。若实际完整确认预测成本超过预留，应先增加预留、缩训练支线，不压缩正式人口。

### 10.2 有限实测和预算方案选择

使用初始化后的临时模型测真实T1/T2/T3/T4/T5 microbatch前向＋backward与一次optimizer/OneCycle更新；每种可用H最多两个稳定测量，E3额外测T2/T5，显存按大样本。恢复模型/RNG/optimizer后才正式开始，临时步数不混入训练计数。结合课程、实际n_t分布、物理batch/accumulation，估计每臂完整U的GPUh c0,c1,cS,cF；乘1.25裕量，评价费用另加。记录实际测量和预测误差，不给虚假精确工期。

在看到训练成绩之前固定预算模式：

1. **FULL_FOUR**：预计四臂完整U＋预留可容纳，则四臂都跑满，不因中途小负数砍掉方法。
2. **SCREEN_ONE**：容纳 `c0+c1+0.4(cS+cF)+0.6 max(cS,cF)`＋预留，则四臂跑到40%；E0/E1继续完整，S/F按第9节排名只续一个至U。预算足够时可续另一臂，但要在查看其新SEL前登记，不能拿SEL挑着续训。
3. **PRIORITY_F**：上述均不够，但容纳c0+c1+cF＋预留，则三臂完整，S仅开发/定向验证，标 `NOT_RUN_BUDGET`。这是预算取舍，不是S失败。
4. **BASELINE_RECOVERY_ONLY**：连两个基线＋一个完整模块都不够时，不自动缩短U。若c0＋预留可完成，先只把E0跑完整并评价，保存已实现S/F及长数据准备，整体状态 `PARTIAL_FULL_RETRAIN_BUDGET`；否则不启动注定无法完成的完整训练，完成开发/真实预检和交接。不得说全部任务完成。

c值已包含1.25裕量时不要重复乘。上述模式按顺序选择，且只能使用真实合法权重/数据；不能通过变小分辨率、删除场景、减少query凑预算。用户事先通过明确参数另授预算时可重算，但不得自行扩大。

一次任务运行中只在真实成本超预测时更新资源计划，不修改已定义的loss/数据/课程/U。触顶按合法checkpoint保存，其他可完成步骤继续，费用和状态真实记录。

### 10.3 时间与内存账目

GPUh按占用卡数×占用时间计，包含失败、预检、恢复与profile；CPU计时另列，不把时间窗重叠重复算总墙钟。最多2张空闲卡，CPU workers总≤8、RAM≤96GiB。训练与评价尽量用已合法复制的本地热数据，不在坏NFS上无限等待。

单个full checkpoint可能数百MB；本地保留last、规定评价点和锁定点。可将旧中间点压缩成inference-only任务权重，不能删除正在被选模或同step控制引用的资产。不得用损失日志代替模型文件。

## 11. 训练后必须做的机制与运行检查

这些不是新训练网格，尽量复用已产生的预测。

- S：原cap触发、阶段样本数及固定T5panel的采样均衡；无GT参与选择。覆盖提升但t-mAP不升照实报告。
- F：实际F更新的norm、后续四次查询mask及最终输出变化，新增参数/原task decoder梯度；训练初期零输出只查一两次。统计既有正确实例被破坏与高IoU候选新增，不以loss减少代替AP。
- 长时：固定T5reference嵌套前缀中首次失败阶段、错误出现/消失、候选缺失与身份分裂；只用TRAIN/CAL做方法诊断，不用SEL诊断结果设计新模块。
- 运行：被选模块与同条件E1在新进程A、B→A两种输入顺序成对核验。仅有字节差异不自动算精度错误，须给预测/AP的差异量。残余运行波动大于候选增益时标 `RUNTIME_UNRESOLVED_GAIN`，不当确认正结果。
- profile：同一空闲A40，固定CAL每长度一组合法输入，E0/E1及选中模块各warmup1、测量3；计完整网络、特征反馈/采样、后处理，IO单列；冷启动setup单列。不能把共享encoder/旧cache当部署免费。

本轮不强制新增可视化大工程；保存固定例子索引、必要prediction差异/统计即可。若需要图，GT只用于最终解释，不用于选推理区域。

## 12. 最小充分测试与审核

只覆盖本轮会改变结果的边界，复用已有测试：
1. encoder-only实际加载、task重新初始化、公共参数一致、F独立初始化。
2. 一个真实混合H批次的GT/时间/分区/inverse对应、无假扫描重复。
3. raw_sum实际标量与梯度、12aux及原对比项保持；无NaN被静默变正常结果。
4. S配额、总数、padding、extra同索引、无cap/T1恒等。
5. F初始恒等、两步梯度、反馈真正进入后续/最终mask、chunk等价。
6. 原生NOAUG和per-input RNG；不同进程顺序问题的实际结果。
7. 从optimizer边界恢复的draw/LR/参数；changed模型/标签/输入使缓存失效。
8. 完整人口、独立metric、选点/正式锁定、负结果/null、发布资产存在性。

CPU设计目标≤20分钟，GPU定向验证包含第5节总预检。不以凑几十上百测试为要求；开发时只跑受影响测试，最后一次相关回归、定向lint/compile和diff check即可。真实问题需要更多排查时留下原因；不跑全库安全、fuzz、无关checkpoint回归，不为历史symlink格式测试重造数据。

完成前阅读真实配置和结果，逐项写 `REQUIREMENT_REVIEW.md`。PASS只能对应实际证据；若预算只完成某模式，清楚列主计划未跑部分，不能用“模块已实现”填训练PASS。

## 13. 实际CLI、状态与恢复

下面接口是本轮要实现的薄入口，而不是声称当前已存在：

```bash
export RESCENE_NATIVE_LONG_ROOT="$HOME/persist4d_runs/native_long_retrain_v1"
python -m scripts.native_long_campaign run \
  --config conf/config_native_long_retrain.yaml \
  --root "$RESCENE_NATIVE_LONG_ROOT" \
  --base-commit c6e9d01edfe2b3c832374a424fc45e44bdadfb68 --resume
python -m scripts.native_long_campaign status --root "$RESCENE_NATIVE_LONG_ROOT"
python -m scripts.native_long_campaign report --root "$RESCENE_NATIVE_LONG_ROOT"
python -m scripts.native_long_campaign publish --root "$RESCENE_NATIVE_LONG_ROOT" --resume
```

复用现有Lightning，新增训练子类提供真实数据计划/验证及新checkpoint hooks，不建新训练框架。run内部：bind→population→runtime/init→cost-plan→train→CAL-lock→SEL→eligible-replication→formal-confirm→profile→review→publish。

stage可独立恢复，完整状态不等于当前身份仍有效；发布回执不能让run跳过计算校验。改变U/课程/loss/物理batch是新训练轨迹，不能沿旧optimizer假装不变。日志格式修正可复用数值资产，但必须证明没有改变训练调用。

某个新模块失败不阻断基线；长GT不可用不阻断合法E0；共同预训练/data/指标不可用才共同阻塞。超预算/权限失败等外部条件也要交付已经完成的代码和真实结果，而不是停在“等授权”。

## 14. 产物与实际GitHub发布

### 14.1 必需产物按内容，不按数量验收

```
artifacts/native_long_retrain_v1/
  EXECUTION_INSTRUCTION.md
  SOURCE_AND_INITIALIZATION.json
  CODE_BINDINGS.md
  POPULATION.json
  CONFIG_RESOLVED.yaml
  RUNTIME_NOTE.md
  RESOURCE_PLAN.json
  RUN_STATE.json
  COST_LEDGER.jsonl
  training/<arm>/seed<seed>/{metrics.csv,checkpoint_manifest.json,resolved_config.json}
  evaluation/{CAL_ALL.csv,SEL_LOCKED.csv,BY_REFERENCE.csv,PAIRED_DELTAS.csv,NESTED_PREFIX.csv}
  selection/{CAL_LOCK.json,SHORTLIST.json,FORMAL_LOCK.json}
  confirmation/{PB.csv,LOCAL_T2.csv,ADDITIONAL.csv}
  diagnostics/{SAMPLING.csv,FEATURE_UPDATE.csv,LONG_ERRORS.csv}
  resources/PROFILE.csv
  evidence/official_metric_states.zip
  REQUIREMENT_REVIEW.md
  FINAL_REPORT.md
  HANDOFF.md
  ARTIFACT_MANIFEST.json
```

可合理合并文件；有状态和分母时不为未执行项伪造空CSV PASS。完整模型、optimizer和rawcache在运行根保存，Git内manifest指向实际位置/远端附件和hash；不把用户机器路径当成可下载URL。

主表必须含E0/E1/E2/E3的完整/partial步数、真正初始化类型、数据/采样条件、各T官方分数、相对E0/E1与同step差值、逐reference情况、两种seed复核身份、GPUh和真实资源。只用新共同运行的旧R1结果做历史桥接。

状态分开：科学可以是 `COMPLETE_NO_MODULE_GAIN`、`COMPLETE_WITH_PROVISIONAL_GAIN`、`COMPLETE_REPLICATED_GAIN`、`PARTIAL_FULL_RETRAIN_BUDGET` 等；发布可以是 `GIT_COMPLETE`、`COMPLETE_WITH_MODEL_ASSETS` 或 `CODE_ONLY`。没有独立确认不能说超越作者官方性能。

### 14.2 权重与结果不能再次只留服务器路径

用户已授权本任务开发完成后同步GitHub。使用现有Git和Release/LFS授权，不索要或打印token，不调用无关账户凭证。权限检查在启动时做一次，以便知道完整模型的交付渠道，但Release失败不阻止训练和Git交付。

代码、所有轻量正负结果、配置、训练曲线、指标充分统计和小模块参数入新Git分支。默认单个小包≤20MiB、总新增二进制≤50MiB；模型大文件走已授权Release或真正已启用的LFS。不得把大模型切碎规避Git限制，也不得只提交LFS指针就声称权重已上传。

完整部署资产至少包含：E0/E1锁定的任务权重，以及获得评价的完整S/F锁定点（负结果可按实际规模保留终点与选中点）；若独立encoder可由已知预训练文件重建，可导出全部**非encoder任务权重＋必要更新buffer＋F参数**和明确加载器，减少体积。不能遗漏BN/统计buffer而使重载输出改变。

每个包必须有数据/配置/初始化/代码/step/父预训练SHA及一次真实重载检查。被裁剪的pilot保留其评价点。raw数据、原始GT和预训练原文件不重新分发；数据许可未知时只上传充分统计，不打包逐点GT。

发布顺序：
1. 实际stage本轮代码/配置/结果路径，提交实验A。
2. 写HANDOFF引用A、生成不含自身hash的manifest，提交B。B内容不写自身SHA。
3. 实际push新分支和新tag，核对remote full SHA。
4. 上传必需大权重/指标附件，核对bytes/SHA及真正远端可取得；小包直接Git验证。
5. 外部 `publication/PUBLICATION_RECEIPT.json`记录A/B/tag/资产可用性。缺必要权重时必须CODE_ONLY或明确PARTIAL_ASSETS；不能用空Release或补发命令代替已上传。

如果Git可写但Release/LFS不可用，仍完成代码/轻量结果push，列准确缺失资产和使用现有授权可执行的补发命令。不要反复尝试授权，不创建未经请求的公共第三方文件分享。

### 14.3 最终回复Codex必须提供

完整B、分支、tag、报告和交接真实GitHub位置；四臂各自实际训练步数与状态；净增益/机制差值；没有涨点则明确没有；预算和运行限制；已上传权重及缺失资产；禁止以测试数量作为精度成果。

---

## 15. 来源索引与本轮判断边界

固定源链接根为 `https://github.com/Orangekostar/Persist4D/blob/c6e9d01edfe2b3c832374a424fc45e44bdadfb68/`。下列源在编写中实际读取；部分服务器资产及实际长TRAIN可用数量需执行时核验，不预填PASS。

- E01 `artifacts/rescene_task_learning_root_cause_v1/full_candidate/FULL_TRAINING_MANIFEST.json`：29700/450、390选择及450权重身份。
- E02 `artifacts/rescene_task_learning_root_cause_v1/initialization/COMMON_INITIALIZATION.json`：共同初始权重和预训练encoder来源。
- E03 `utils/rescene_rootcause_preflight.py`：公共张量加载、旧固定日程与变体门槛，说明哪些纯函数可复用、哪些旧约束不能继承。
- E04 `main_instance_segmentation.py`：现有训练入口及P2固定计数/恢复处理。
- E05 `trainer/trainer.py`：raw_sum开关、真正目标、冻结train/eval、原生eval与optimizer接口。
- E06 `conf/config_rescene4d_concerto_rootcause.yaml`：R1式配置和weighted缺省风险。
- E07 `conf/data/datasets/mix.yaml`、`datasets/multi_dataset.py`：1.0:0.8与归一采样。
- E08 `datasets/semseg.py`：真实数据库、scan索引、增强开关、空扫描与分区。
- E09 `datasets/pointcept_utils.py`：阶段分开体素化、逻辑batch、inverse与监督格式。
- E10 `datasets/preprocessing/RScan_preprocessing.py`：元数据组序列、不足真实scan数不循环重复、ambiguity。
- E11 `models/rescene.py`、`conf/model/rescene.yaml`：cap/is_eval、12次更新、maskF固定和query-only钩子。
- E12 `models/pointcept.py`、`conf/serialization/mixed.yaml`：预训练加载、编码器/decoder与混合序列化。
- E13 `models/criterion.py`、`models/matcher.py`、`conf/loss/contrastive/infoNCE.yaml`：原损失与跨阶段实例监督。
- E14 `artifacts/qp_mn_targeted_v1/FINAL_REPORT.md`、`resources/NATIVE_RUNTIME_NOTE.md`：小头负结果、累计80.7350387502 GPUh与原生过程历史限制。
- E15 `scripts/rescene_task_postprocess.py`、`scripts/p6a_metrics.py`：原生输出及官方指标适配。
- P01 ReScene4D原文（2026，尤其3.2/3.3、C.2、D.2）：https://arxiv.org/html/2601.11508v2 。冻结预训练encoder、从头训练decoder与混合数据是文献依据；本轮长课程不是作者已经证实的方案。
- P02 Relation3D，CVPR2025：https://openaccess.thecvf.com/content/CVPR2025/html/Lu_Relation3D__Enhancing_Relation_Modeling_for_Point_Cloud_Instance_Segmentation_CVPR_2025_paper.html 。支持研究superpoint关系与特征更新，不等于本轮F是其原样实现。
- P03 Robust Promptable Video Object Segmentation / MoGA，CVPR2026：https://openaccess.thecvf.com/content/CVPR2026/html/Lee_Robust_Promptable_Video_Object_Segmentation_CVPR_2026_paper.html 。对象表示条件化适配的近期参考，视频提示条件不直接用于本轮无GT提示的3D实例网络。
- P04 PyTorch OneCycleLR：https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html 。需要完整总步数与按optimizer更新执行；不能把早停点伪装成跑完独立完整周期。

**S是根据实际代码提出的工程假设，F是受文献启发的新迁移设计；课程、插入位置、预算与晋级幅度都是前瞻选择，没有预先测得收益。** 本轮允许用真实负结果否定配方，不允许虚构性能或为了正名单随意换协议。
