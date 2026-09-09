# Persist4D 长时任务质量保持与低成本推理：Codex完整执行提示词

本文件合并八个执行阶段，附固定来源索引。主任务不是再写一轮计划，而是由Codex在现有环境实现、训练、评价并完成GitHub交接。新结构的收益尚未验证，成功条件与负结果边界详见正文。

仅使用此合并版也可以启动；文中I/X引用在来源附录解析。原36篇全文知识卡与机器可读清单位于配套ZIP，供需要时查阅。

# Persist4D：长时任务质量保持与有界视觉记忆 V2

## 给 Codex 的主执行指令

你在 `Orangekostar/Persist4D` 内进行研究开发。先阅读本包八个阶段文件，再按 I/X/P 标识查询 `knowledge/` 中对应条目；不要把本任务理解成重新检索 36 篇论文或安装 35 个仓库。单独使用合并版时，文中的证据表与附录来源清单已提供实施所需绑定，不因未挂载完整知识目录而阻塞。

**当前任务是实施并检验方法，不是只写计划、脚手架或测试后宣称完成。** 有 GPU、数据和权重时，运行下文授权的真实实验；无资产时完成可完成的实现和明确的阻塞交接，不能编造指标。科学目标失败仍须提交代码、负结果及交接文档。

### 目标优先级

1. 一个最终 checkpoint、一套固定推理规则，在相同数据与官方指标下，T2/T3/T4/T5 的绝对 t-mAP 均高于指定 ReScene 对照。
2. 保持有竞争力的 T2 起点，显著减少随 T 增长的任务质量下降；不得降低 T2 制造漂亮的 retention。
3. 相比完整历史方法，长时更新的实际延迟、工作显存与历史工作状态更低或增长更受控。统计读取、路由、写入、修订和输出成本；不得把 CPU offload 或历史归档隐藏成常数空间。
4. t-mAP50、t-mAP25、t-REC、prefix overall mAP 同时完整报告。恢复率只能解释机制，不能替代 t-mAP 失败。

这四点是研究目标，**不是对未运行结果的承诺**。绝对“不下降”、有限 T 的小幅下降和“比 ReScene 下降少”是不同结论。

### 基线快照与分支

```text
reviewed_parent: 32a51e11b51043ec5a5825215669ef7b0ea03bc2
reviewed_parent_branch: research/persist4d-allt-task-superiority-v1
new_branch: research/persist4d-task-memory-retention-v2
new_artifacts: artifacts/task_memory_retention_v2/
R1_SHA256: 629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd
R1_bytes: 754813672
Concerto_pretrained_SHA256: 845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07
```

运行时先确认工作树、远端和可用 worktree。不能假定 `paper5/.worktrees/...` 是 Git 内的真实路径；本包源码路径均相对于真正的仓库根。若当前分支多了用户提交，保存它们，从已核实 parent 新建隔离分支/worktree，不 reset 或覆盖用户工作。

### 当前事实，不得改写

| R1 / 同 Protocol-B，百分数 | T2 | T3 | T4 | T5 |
|---|---:|---:|---:|---:|
| FullHistory | 22.878 | 14.831 | 10.275 | 8.157 |
| B4 | 21.964 | 13.550 | 8.016 | 5.854 |
| 上轮 C2 | 22.503 | 14.106 | 7.600 | 5.132 |

这些是已提交结果 [I01,I02,KB-S]。它们不支持“当前已经保持长时 t-mAP”。上轮 C2/C1 是特定轻量实现；不将其失败上升成 MOTR、Cutie 或所有记忆方法失败。原 All-T FAIL 永远保留，不通过重定义目标改成 PASS。

### 本轮收敛的方法

**当前发现锚点 → 预测驱动的实体条件 query → 序列一致监督 → 有界对象视觉证据 → 条件性 prefix 教师。**

这是知识库 P26/P27/P29、P18/P19/P21/P22、P13 的机制适配，不是完整复现这些外部模型。保留 R1、Concerto、原 taxonomy 和官方后处理；不引入 2D GT mask、SAM 前端、语言模型或另一个 backbone。

本轮刻意不同时做：QCL/RRE/RCA、LaSSM、Sonata 重训、label255/EOS/physical-batch 复现、learned eviction、全历史点云归档解码。它们不进入本轮结果选择空间。

### 执行顺序

| 阶段 | 工作 | 必须交付 | 训练授权 |
|---|---|---|---|
| M0 | 快照、知识绑定、数据/输出合同、少量失效分析和长期 bank 对照 | `START_STATE`、`EVIDENCE_MAP`、`DATA_CONTRACT`、`BASELINES` | 无新训练 |
| M1 | 可变长度 episode、显式元信息、query 路由和状态更新接口 | 可运行的数据与模型原型 | 真实小样本 smoke |
| M2 | W-BASE、Q-INDEP、Q-TALA、FH-MATCH 受控训练 | 完整学习曲线与四 T 开发结果 | 第一组正式运行 |
| M3 | 基础延续、V-LAST、V-CORE 同起点视觉记忆对照 | 视觉证据贡献与预算对照 | 有数据/路由有效性时运行 |
| M4 | 同学生有/无 prefix 教师；必要时一个有界轨迹损失实验 | 独立归因表 | 按 05 中条件运行，不全扫 |
| M5 | 冻结单候选，统一全 T 评价、独立 references、资源/容量 | `FINAL_REPORT`、全部数值表 | 不再选参 |
| PUB | 分阶段 commit、交接、push、远端回读 | `HANDOFF`、`FINAL_MANIFEST`、发布回执 | 必须执行 |

这里的 M2/M3 是不同科学假设，不能因某个单因素没有全 T 领先就自动取消所有互补实验。也不能在明确无效时继续盲目扩展。具体技术停止条件在 05。

### 输出合同：不再把“零修订”当作低成本方法的必要条件

主任务定义为**当前 prefix 可用信息下的估计**；无方法可读取尚未到达的扫描。实现两种显式输出策略：

- `commit0`：原有不可修订输出，作为强制历史对照。
- `lag1`：本轮主候选策略；仅允许当前 W=2 前向修订前一个扫描，并立即提交当前扫描的暂定结果。更旧 mask 不再改变。使用已有窗口输出，不新增历史网络前向。

这是预先选定的工程方案，不是发现 lag1 分数好以后选它。训练方法比较必须在相同策略下进行；FH-native 仍作为重要对照，另外给出 FH-lag1 来区分输出权利。原 B4 commit0 的表不得覆盖。`lag1` 需要保留上一扫描的有界缓冲，其成本计入；全部历史逐点 mask 的物化是随输出量增长的评价/导出操作，不宣称 O(1)。

**先做低成本协议原型。若不能正确、可核验地完成 lag1，不得暗中退回 commit0 后宣称主目标完成；报告 `LAG1_BLOCKED`，保留可运行的 commit0 研究结果。** 具体跨窗口 ID 与修订细则见 02。

### 科研边界与资源纪律

只把标签用于损失、评估及事后诊断；部署状态、query 路由、视觉证据选择、修订接受规则都只能使用预测与输入元信息。标签随机替换不能改变同种子推理输出。

不重跑全仓上千 tests，不安装全部外部仓库，不做模糊测试、恶意输入、权限或网络压力测试。最多一次必要的修改文件秘密检查；不把正常路径名当安全风险反复清洗。只运行 07 的直接正确性检查。

真实失败必须记录：OOM、预算不足、模型退化、未运行的条件实验。400 步只可标为 pilot；不能再把预算缩成十分之一后仍声称完成充分正式检验。不要为通过人为“全部参数非零梯度”门禁而改模型；条件未使用的参数允许无梯度，真正测试相关正样本路径即可。

最终不要只回答“任务完成”。分别报告 `EXECUTION`、`TMAP_ALL_T`、`RETENTION`、`RESOURCE`、`MECHANISM`、`GENERALIZATION`、`PUBLICATION`。


---

# 证据、现有代码与任务的绑定

## 证据等级

- **已观察事实**：本轮实际读取的源文件、固定提交内结果、知识库原文。
- **推导**：由源码或数字直接推导，明确假设。
- **待检验设计**：本包建议的模型、预算、阈值、损失与协议。不得将“有文献依据”写成“已经证明有效”。

本包没有执行训练或检查服务器的大权重本体。运行端只需首次检查 R1 文件身份，并复用已有 manifest；不要每次评价重新读取 755 MB 文件做 hash。

## 内部来源与具体决策

源码统一固定到 `Orangekostar/Persist4D@32a51e11b51043ec5a5825215669ef7b0ea03bc2`。文件的 blob 与读取范围在 `evidence/source_index.json`。范围外不声称已经逐行审计。

| ID | 已读取路径/符号 | 已确认行为 | 本轮对应任务 |
|---|---|---|---|
| I01 | `artifacts/allt_task_superiority_v1/HANDOFF.md` | R1/C2/FH 权重身份；400 updates；旧开发集 R1-base exposed；C2 cache 约119GB | 新实验从 R1 开始；数据曝光分开；禁止膨胀缓存和预算偷换 |
| I02 | `artifacts/allt_task_superiority_v1/FINAL_REPORT.md`（知识库绑定） | 上轮 C2 四 T 结果与关闭读取诊断 | 负结果不可覆盖；不能预设 gate 越强越好 |
| I03 | `models/persist4d_allt.py::strict_load_r1_with_named_adapters` | 原树严格加载，仅接受指定新模块前缀缺失 | 复用加载思路，扩展新前缀白名单，不泛用 strict=False |
| I04 | `models/persist4d_allt.py::after_decoder_stage` | 只在 `len(hlevels)-1` 执行原 L/M；forward 状态暂存 finally 清理 | 新模型继承 ReScene 的扩展点，不同时挂旧 C2 |
| I05 | `models/rescene.py::forward/after_decoder_stage`，约399–500 | `execution_stage_idx` 与共享权重索引分离；先聚合mask特征再执行多层 query refinement | 保留真实执行索引；第一次完整尺度pass后一次性做实体条件化 |
| I06 | 同文件 `sample_and_batch_features/aggregate_features/mask_module` | padding=True 表示无效；mask head最终由 query 与mask features点积；`segment_features=[agg_feat]` 有层级外包 | 按真实 segment 行写视觉证据，不能把 `[layer][batch]` 当 `[batch]`；不能误写临时副本 |
| I07 | `trainer/persist4d_allt_trainer.py::training_step` | 每窗口 criterion；手工累积；backward 后预测写入；写入detach | 新 trainer 保留优化框架，添加监督账本，支持两阶段 TBPTT |
| I08 | 同文件 `prefix_balanced_stage_coefficients` | H=5约为[.3208,.3208,.1958,.1125,.05] | 不能称每更新阶段等权；新共同训练recipe明确损失归一化 |
| I09 | 同文件 `update_prediction_memory/segment_stages_from_target` | stages 从元信息构造；写入返回 state/read_state，丢弃 slot mapping | 新接口返回 `slot_for_query/births/generation`，不依赖GT标签 |
| I10 | `datasets/persist4d_sequence_dataset.py::EpisodeMaster/build_episode_draw_plan` | master只接受1或5，multihorizon plan要求5 | 新 sibling loader支持真实2–5；旧loader保持不变 |
| I11 | 同文件 `Persist4DEpisodeDataset.__getitem__` | 每个窗口相同随机seed重新load；显式change_file=None | 复用子窗口接口，但不误以为同seed保证相同空间变换 |
| I12 | `datasets/semseg.py::_load_scan_sequence`，约470–640 | 窗口均值/minmax参与变换；segments重编号；时间维最后恢复 | 显式 episode augmentation；原始点ID/变换追踪；不保存未对齐的绝对窗口坐标 |
| I13 | `datasets/pointcept_utils.py::voxelize/get_instance_masks` | group center shift随窗口范围变化；inverse maps；targets有原instance `ids`；原无target会返回空list | 教师/学生按原始scan vertex ID对齐；新loss处理当前阶段空目标；不假定跨stage target行号是ID |
| I14 | `models/criterion.py::SetCriterion.forward`，约1320–1404 | final和每个aux层都独立调用matcher | inherited约束需覆盖所有受影响层；不能只改final后被aux拉回 |
| I15 | 同文件 `loss_labels/loss_masks/loss_segment_contrastive` | no-object在最后一类；ignore253；loss_masks直接stack；对比返回aggregate及per-layer | 新empty branch返回可微零；不把background_class=0误当noobj；不无意重改R1 objective |
| I16 | `models/persistent_memory.py::associate_observations/step`，约615–910 | B4先完整assignment再阈值过滤；按confidence乘更新率；slot mapping在结果中；有watermark | 路由/commit两阶段复用数值规则；先读后写；每stage只提交一次 |
| I17 | `models/persistent_memory_read.py`，95–215，本轮复核 | L2 QK无scale，null value可学，有通道gate | 归一化读取必须明确scale与真零分支；是候选缺陷不是已证实root cause |
| I18 | `scripts/train_persist4d_allt.py`，1–230 | `FORMAL_OPTIMIZER_UPDATES=400`；variants和checkpoint quarters硬编码 | 新入口独立预算；不改旧常量篡改历史实验 |
| I19 | `scripts/evaluate_persist4d_allt.py`，1–265 | 四T已支持，但stage_requests要求恰好5扫描；cache schema绑定旧window modes | 新schema支持native length和输出policy；旧schema只读 |
| I20 | `scripts/rescene_task_postprocess.py::OfficialTaskPrediction/extract_official_task_prediction` | 同时持有完整窗口mask与latest mask；可`prediction(latest_only=False)`；CPU/NumPy/detach | lag1可复用完整窗口预测；训练不能经过该后处理求梯度 |
| I21 | `models/pointcept.py::encoder/forward_override`，约130–180、414–440 | 名为encoder的方法会运行model.enc以及model.dec | 真正编码缓存必须截在enc后、dec前；不能缓存整个encoder()并声称前缀无关 |
| I22 | `models/pointcept.py::decoder/change_hierarchical_serialization` | 使用pop和in-place修改父层链 | teacher/FH缓存必须重建非别名结构；不能复用被前向污染的Point对象 |
| I23 | `scripts/system_comparison_v2_inference.py`，80–240，本轮复核 | append-only；key含(track_id,class_id)；mean/latest/max | 新lag1 accumulator独立命名；不删除class_id伪造涨分 |

### 三项新任务边界

1. **route ≠ commit**：一次前向可做预测关联来读取状态，但不能调用 `memory.step` 两次。已确认旧step更新watermark和年龄。新writer接受预路由结果，对已继承query不再次重配到另一个slot。
2. **metadata ≠ supervision**：point2segment、stage索引、scan vertex索引来自输入预处理，可以用于推理；labels/GT ids/GT masks只进入criterion和评估。
3. **before/after hook特征空间必须一致**：旧B4状态存最终 `decoder_norm(query)`；新预路由在中间层用该空间只是初始化假设，应增加轻量projection并记录错路由，不能宣称天然兼容。

## 外部源码对照及引用边界

| ID /知识库 | 实际核对代码 | 应借的机制 | 不能照搬 |
|---|---|---|---|
| X01 / P26 MOTR | `megvii-research/MOTR:models/motr.py::match_for_single_frame`，165–285，blob `c358e3f0d958f4b607207d1c9c20632e343a7b0a` | 已有track继承训练关系，余下query匹配新目标，aux继承同关系 | 2D框loss、GT驱动的推理state；不得自称完整MOTR |
| X02 / P19 Cutie | `hkchengrex/Cutie:cutie/model/transformer/object_transformer.py`，30–185，blob `089b51e4eef29388c9fc4e75c391adb012ede4f3` | 对象摘要和局部视觉证据分工，双向读取 | 首帧GTmask与2D密集特征；第一版仅做有界对象视觉读取，不声称完整Cutie |
| X03 / P18 XMem | `hkchengrex/XMem:inference/memory_manager.py`，205–335，blob `bce2c00cc8aa15eb85166d0cd02b9ef3af79b180` | 工作记忆压缩为原型及value；按预算维护 | 全局usage独占预算会损害稀有重现对象，不能直接套 |
| X04 / P35 QKNorm | 论文与 `CyndxAI/QKNorm` | normalized QK配可学习scale | 越尖越好、任意对cosine再除sqrt(d) |
| X05 / P13 StreamVGGT | 论文 §训练、蒸馏对照；`wzzheng/StreamVGGT` | 有限状态学生接受prefix教师监督 | KV默认不一定有界；教师query index不天然对应学生 |
| X06 / P27/P29/P09 | KB中MOTRv2、MeMOTR、AutoSeg3D源码对照 | 保护发现、可靠状态更新、track/object交互 | 本轮不下载这些任务checkpoint替换R1，不加2D输入或硬运动门禁 |

本轮实际重新打开正文：MOTR、Cutie、StreamVGGT、QKNorm。其余按已给36篇知识库定位；不声称再次全文复审36篇或复现任何外部模型。概念适配原则上自己写小模块，若复制实现，保留对应license/NOTICE和来源commit；只核对真正使用的仓库一次。

## 可直接引用的来源

- R1和All-T源码：`https://github.com/Orangekostar/Persist4D/tree/32a51e11b51043ec5a5825215669ef7b0ea03bc2`
- MOTR：`https://arxiv.org/html/2105.03247`；`https://github.com/megvii-research/MOTR`
- Cutie：`https://arxiv.org/html/2310.12982v2`；`https://github.com/hkchengrex/Cutie`
- XMem：`https://arxiv.org/abs/2207.07115`；`https://github.com/hkchengrex/XMem`
- StreamVGGT：`https://arxiv.org/html/2507.11539v2`；`https://github.com/wzzheng/StreamVGGT`
- QKNorm：`https://aclanthology.org/2020.findings-emnlp.379/`；`https://github.com/CyndxAI/QKNorm`

公开网页日期和版本不等于作者内部训练recipe。外部源码blob是**文件对象hash**，不是仓库commit；不要把它填写为base commit。


---

# M0–M1：数据、输出合同与最小基线

本文引用的 I/X/KB 标识见 `01_EVIDENCE_AND_CODE_MAP.md`。所有新增接口和参数是本轮设计，不假装当前仓库已经实现。

## 1. 先绑定输入，不再从零审计整个项目

读取本包与旧 `HANDOFF.md`，只记录：实际 Git 根、parent SHA、现有环境、可用 GPU、R1/预训练文件定位、数据目录、旧缓存定位与剩余磁盘。每个大权重首次读取时算一次 SHA256；已有可信内容 manifest 的不可变缓存可复用，访问一个条目才校验一个条目，不反复扫描全盘。

生成 `START_STATE.json`、`DATA_CONTRACT.json`、`OUTPUT_CONTRACT.md`、`BUDGET_CONTRACT.json`。旧 source results 保持只读，不以机器路径替换旧文件中的历史 provenance。

停止于资产问题时：记录真正缺失项、尝试过的路径/工具和可恢复命令；不要求用户重做已经有证据的准备，也不伪造空结果表。

## 2. 数据必须支持真实的 native T2–T5

### 2.1 修复的是新实验入口，不覆盖旧协议

已知旧 `EpisodeMaster` 只接受1或5扫描，旧 draw plan 强制5扫描母序列 [I10]。新增 `datasets/task_memory_episode.py` 或等价 sibling，复用原始数据加载与 voxelization，不把只有2–4次真实扫描的场景从新训练中无故排除。

先清点实际可用元数据和已处理文件，记录：每个 reference 的 scan UUID、原 split、原始实例对应是否存在、可用长度、基础训练/旧选模/新开发曝光。不要把 metadata 列出的扫描当作本机一定可读；也不要把 scene 数、masters 数和排列数混在一起。

从真实同-reference扫描构造最大5阶段的训练 episodes。禁止跨房间拼成一条物体轨迹，禁止复制scan制造T5。对于超过5扫描的reference，使用事前固定的窗口/顺序策略，不能遍历所有排列再挑有利结果。

保留两套最终评价 population：

1. `protocol_b_common_129`：既有43 masters×3 orders、六reference；同一组序列形成T2–T5曲线。
2. `additional_native_refs`：未参与本轮选择的真实独立reference；每个T可用数量单列，不把不同population拼成同一retention曲线。

原基础模型见过、新适配没见过的reference，标 `base_exposed_adaptation_holdout`，不是unseen。若没有真正新场景，报告 `INDEPENDENT_GENERALIZATION_NOT_ESTABLISHED`；不要因此无限阻塞核心实现与受控实验。

### 2.2 训练计划在评价前冻结

先保留旧8个开发reference和六个Protocol-B reference作为本轮禁止训练集合，再在可用训练reference上建立native长度清单。若原始split明确不允许某reference参与训练，遵从原split。新增reference的开发/确认角色依据hash和元数据划分，在读取方法分数前锁定。

建议共同recipe：五个episode桶各20%：`single_scan`,`T2`,`T3`,`T4`,`T5`；single_scan桶优先使用原ScanNet训练数据，亦可包含明确标记的RIO训练单scan。每个时间桶先均匀选符合长度条件的reference，再选择真实扫描组合。若某桶为空，记录缺口，在正式运行前一次性调整并对所有候选一致，不复制数据补齐。

这个4:1序列/单scan配比是新适配实验的工程起点，不是作者ReScene训练的1:0.8复现。必须由 W-BASE 与 FH-MATCH 共同控制其影响。

DDP同一同步组内保证H一致、rank抽取不同draw位置；有放回采样出现相同scan本身不等于DDP错误。只做一次前64个global draws检查，不重新跑整个旧sampler审计。

### 2.3 输入元信息独立于监督

新增 `StageMeta` 至少有：

```text
reference_id / episode_id / scan_ids_in_window
absolute_stage_index / local_stage_ids
original_vertex_ids / scan_vertex_offsets
point2segment / segment_stage_ids
augmentation_transform_id / coordinate_frame_id
voxel_inverse / full_resolution_point2segment
```

这些来自扫描顺序和无标签预处理。`TrainingTargets` 单独携带 class、instance masks、qualified GT IDs、ambiguity groups。model/state/read/write 接口只接前者和预测。

GT身份用 `(reference_id, canonical_instance_id)`；必须用数据集正式对应关系确认raw instance ID跨scan是否稳定。不能把每窗口target行号、query行号或碰巧相同的整数当成跨时间GT ID。[I13]

### 2.4 增强与点对应

同seed不保证同一个空间变换：旧dataset依赖窗口mean/minmax，elastic field也依赖窗口坐标 [I11–I13]。第一版选择可明确追踪的episode级刚体/尺度与颜色增强；变换只依赖随机种子和已知输入规则，不用全T bounding box去变换早期prefix。

对所有共同训练候选保持相同recipe。暂不采用无法在重复scan间对齐的独立elastic distortion；这项变化由W-BASE/FH-MATCH控制，不声称它本身是新方法贡献。

保留每个scan原vertex ID。教师与学生的voxel/segment数量可以不同，但监督映射必须通过同一原始点索引。历史视觉证据的绝对窗口坐标不直接复用；第一版仅用特征和episode stage年龄，几何距离不作为跨重访硬拒绝。

校验一次：重复出现的同scan，原vertex ID与标签对应一致；改变未来scan不影响之前已执行的commit0输出。尽量分阶段搬到GPU，别把所有H的点云同时常驻GPU后宣称有界推理。

## 3. 固定输出政策：lag1主候选，commit0强制对照

### 3.1 定义，不用未来信息

时刻t仅访问已到达的 `X1..Xt`。主学生网络输入W=2，t=1输入单scan。

- `commit0`：t时仅提交当前scan；过去mask永不改。
- `lag1`：同一次 `(X[t-1],X[t])` 前向输出两个scan；固定用这个窗口的前一scan预测替换它在t-1的暂定版本，并冻结更旧部分；当前scan暂定，下一步可修订一次。
- `FH-native`：用全部prefix联合预测整个prefix；属于原ReScene重要对照。
- `FH-lag1`：相同FullHistory输入，但采用上述lag1提交规则，并明确它的跨前向ID发布器。

不得在不同T挑policy。lag1不等于看完整未来；也不等于原commit0协议。主表逐行标明。训练/模块比较以相同policy为准，FH-native依然保留，避免只比较被限制后的ReScene。

完整窗口输出已经由 `OfficialTaskPrediction.pred_masks/temporal_stages` 提供 [I20]；不要新增一遍网络前向来重新求它，也不能只拿 `latest_stage_masks` 冒充前一scan修订。

### 3.2 先做不训练的 policy-only baseline

在开发与小型historical diagnostic panel上比较 R1+B4 的commit0与lag1。若旧cache只存latest mask，不能从中发明prev mask；只为缺失窗口补一次前向。此试验归因于输出策略，不归因于新memory。

技术判定只检查合法性：point lineage、正确阶段、一次修订范围、无GT或未到达scan进入预测。不用“lag1先涨分”作为启用门槛。若两次前向使早期结果变差，也要保留；新策略不是oracle逐mask选优。

### 3.3 一次输出的实体ID与修订

严格区分：

```text
current visibility          当前scan上是否有有效mask
previous-window visibility 上一scan上是否有有效mask
persistent identity        既有对象的身份
```

只在前一scan出现的query不能被 `latest_valid=False` 全部删掉；它仍可修订前一scan，当前mask为空即可。不能将“当前没有有效点”误写成“该对象从历史删除”。

先用第03章的预测route绑定持续slot；对没有route的窗口candidate，可以与缓冲中**同一个scan的原vertex集合**做class-compatible mask IoU Hungarian匹配来继承前一暂定ID。初始阈值0.5、完全相同score时按source query index稳定决策，参数在正式开发前冻结。这里比较同scan的重复预测，不是假设物体跨时刻没移动。

若route与同scan重叠锚冲突：以明确route优先，记录冲突，不用GT裁决。一个logical identity在同stage每class最多一候选；重复者按官方预测score、再按query index确定唯一候选，其他作为独立ephemeral预测处理，不偷偷删掉FP以涨分。

整阶段替换，不执行GT-guided逐物体择优；若新窗口没有给某个原candidate，修订版本也应如实反映这一变化。所有规则对学生与FH-lag1相同。**如果比较中关联器和评分也发生改变，就报告系统policy对照，不能把全部差值写成纯修订收益。**

使用 `(logical_track_id, generation, class_id)` 作为内部键；是否复用slot都保留generation字段。已冻结阶段不可全局重新label；类别变化按官方候选生成新的class轨迹，不能删除class_id让AP虚高。

### 3.4 有界工作状态与输出归档

仅保留上一scan的可修订mask/点映射缓冲和当前window。更早输出写入只追加的紧凑归档，模型禁止读归档。prefix评分时为官方evaluator物化完整历史mask，其成本按output/evaluation单列。

末尾prefix当前scan是暂定估计，指标可在当前时刻计算；若应用需要最终确定输出，报告lag1的一扫描确认延迟。不读取下一个不存在的scan来补flush。

## 4. 最小强基线：只围绕本次科学问题

先搜索旧reviewer/final_evidence中已有长期bank/简单tracker实现，确认已具备什么，再决定是否新增。不因为旧表名 `TRACKER_REJECTED` 就认定它已等价覆盖本轮R1、policy和候选语义。

本轮在相同R1官方候选上至少比较：

| 名称 | 保留机制 | 更新 | policy |
|---|---|---|---|
| B2 | 相邻阶段 | 原实现 | commit0与lag1可比输出 |
| B4 | 原有持久状态 | 原confidence-EMA | 两种policy |
| D-LAST | 相同K、失联保留 | 最后一次可靠特征 | 两种policy |
| D-EMA | 相同K、失联保留 | 固定EMA，不额外learn | 两种policy |

做机制隔离时，D-LAST/D-EMA使用B4的完整assignment后阈值语义和相同类别概率处理。已知B2的threshold-aware matching与B4不是同一实现，不能把二者差异悄悄统一后仍使用旧名。

若某bank与B4代数等价，合并为等价实现检查，不列成独立强基线。先固定阈值；确有必要的竞争性调参最多共享3个预先列出的开发点，所有方法相同预算，只在dev上选择。总时间上限按05。

补充报告gap=0/1/≥2、首次出现、同类新生、长期未观察等事件。标注不可分辨的ambiguity group继续使用官方规则，不能强行赋予可辨别ID。条件恢复accuracy、attempt coverage、recall与t-mAP一起报告。

## 5. M0/M1结束时必须能交接什么

```text
START_STATE.json
DATA_CONTRACT.json + references_inventory.csv
OUTPUT_CONTRACT.md
BASELINE_CONTRACT.json
BUDGET_CONTRACT.json
baseline/policy_comparison.csv
baseline/long_memory_controls.csv
baseline/gap_event_strata.csv
implementation/preflight_real_sequence.json
```

无数据的格子写 `NOT_AVAILABLE/NOT_RUN` 与理由。M0的目的为模型开发消除具体混淆，不是建立新的大审计平台；达到这些条件立即进入模型实现。


---

# M1–M2：当前发现锚点、预测路由与序列一致实体监督

## 1. 本轮到底实现什么

新增 `models/persist4d_task_memory.py::Persist4DTaskMemory`，继承 `ReScene`，不要把旧C2和新read叠两次。新增小型 `models/task_memory_state.py`、`models/task_memory_routing.py`；原 `PersistentMemory` 和旧checkpoint接口仍用于B4复验。

本轮第一版是 **proposal-anchored entity-conditioned decoder**：让已有100个R1 discovery queries读取相应历史实体，再完成剩余解码；通过tracklet-aware监督保持实体目标一致。它不是原版MOTR的可变数量track queries，不声称完全端到端去掉了关联。

这一选择避免最初就把100 detection queries额外扩成200，同时尽量保留R1初始空间覆盖。它仍可能受R1 proposal覆盖限制；这一限制应通过newborn/low-IoU候选诊断检验，而不是承诺能够发现所有漏检对象。

## 2. 必须保持的接口与新文件

| 文件/接口 | 实际作用 |
|---|---|
| `ReScene.after_decoder_stage` | 保留参数无关hook，用真实execution_stage_idx |
| 新 `Persist4DTaskMemory.forward(..., task_state, stage_meta)` | 仅接受预测state与输入元信息；返回raw predictions及lineage |
| 新 `route_entities(pre_output, old_state, stage_meta)` | 只读，返回query→slot；不更新时间和年龄 |
| 新 `commit_entities(final_output, route, old_state, stage_meta)` | 每stage唯一一次写入；返回state、identity map、births |
| 新 `TrainingIdentityLedger` | 训练端GT对应，仅用于loss；不进forward/state/checkpoint部署数据 |
| 新 `TaskMemoryCriterion` | 按模式独立matching或继承matching，兼容aux与空目标 |
| 新 `TaskMemoryTrainer` | 调度stage、TBPTT、loss与预测写入；替代旧400-step入口 |

可少量扩展基类以显式导出hook所需raw tensors；关闭新开关必须保持原数值行为。不能因方便把全部旧文件复制成另一个失去维护的ReScene仓库。

## 3. 模型forward：一条数据流，不重复网络

### 3.1 第一pass仍由当前输入驱动

按R1原流程提取W2特征、FPS位置和100个query。到 `execution_stage_idx == len(hlevels)-1` 时，已经完成一次当前特征读取。该处获得：

```text
Q_pre            [B,100,128]
class_pre        [B,100,C_with_no_object]
mask_pre[b]      [S_window_b,100]
segment_features [layer][batch] -> [S_window_b,128]
```

padding True表示无效。直接使用在真实forward中的feature张量，禁止修改一个没有被下游使用的临时copy，然后宣称视觉证据改变了mask。

### 3.2 预测路由

将 `decoder_norm(Q_pre)` 与旧slot embedding放入原B4特征+类别scorer，先assignment再threshold，与原B4数值语义一致。**中间层特征与最后层状态不是天然同分布**，先记录跨层路由覆盖和错误。第一版route保持B4数值规则，read中的query/key投影另行学习；不声称已训练了离散关联器。若一个projection只用于detach后的Hungarian而没有连续损失路径，它不会被有效训练，不得写成learned routing。

默认路由候选valid规则基于预测类别confidence及窗口mask支持，明确区分current与previous support，metadata不含GT。保持新生发现query，不预先用所有slots占满query预算。

返回 `EntityRoute`：

```text
query_to_slot [B,Q], -1为未继承
slot_to_query [B,K], -1为无当前query
route_score [B,Q]
prior_logical_id / generation
current_supported / previous_supported
```

route本身离散、stop-gradient；不要把Hungarian当成可微操作。它将预测query绑定到一个持续实体，**不是GT决定读哪个实体**。

### 3.3 实体条件化

同一个模型结构用于Q-INDEP与Q-TALA；二者仅损失匹配不同。

初始版本对已路由query读取对应slot，未路由query维持discovery角色。使用归一化QK时有明确正scale，默认 `log_scale=log(sqrt(128))`，运行时scale限制[1,64]；这是工程初值，不是文献最佳值。单slot/null读取仍要有真正可学习的拒绝能力，不能只因占用就强制注入。

```text
read_t = attention(query_pre, matched slot evidence + null)
query_post = query_pre + gate(query_pre, read_t) * projected_read
```

输出投影weight/bias初始为0，使该路径在初始/关闭时退回R1；gate初始可为sigmoid(0)，不要gate与输出全部封死。null的value固定为0，投影无单独null bias；拒绝全部历史时delta必须严格为0。一个抽象learned null token不自动满足此条件。

第一版不同时加top-k搜索、年龄衰减、类别prior和质量网络。路由已经提供候选限制，先验证它能正确读取/拒绝。显式记录本轮模块不是旧C2 `all_occupied` 混合read的简单改名。

后续剩余ReScene decoder stages照常执行，最终class/mask由原head生成。新模块关闭时原raw输出及官方候选在相同RNG下对齐。

### 3.4 只提交一次状态

forward后 `commit_entities` 使用已冻结的route，而不是再次Hungarian后把同一query绑定到另一slot。每stage仅更新一次watermark/age。

- 有route且当前有效：更新相应slot特征、类别和confidence，保持logicalID。
- 有route但当前无有效mask：保留历史slot为dormant，不注入伪mask、不写低质量visual；前一scan可按lag1规则修订。
- 无route且当前有效：按预测分数顺序占用空slot；新logicalID，不用GT选择。
- 只有前一scan支持的candidate：可作为前一scan修订或短暂历史实体输出；不以“当前存在”写入active状态。
- 空间满：第一版采用原B4 reject-birth策略，记录拒绝；不新增复杂淘汰算法。明确这不能保证无限新增对象下精度。

输出 `CommitResult` 至少含state、query_to_logical_id、births、rejected_births、active、route承诺。对于slot复用的未来扩展必须更新generation，本版不复用也保留字段。

旧 `update_prediction_memory` 丢弃mapping [I09]；新实现需返回mapping，不能用近似二次匹配补回。部署state禁止包含任何GT ID、GT mask或目标质量字段。

## 4. 持续实体监督：借MOTR逻辑，改为3D mask任务

### 4.1 训练监督账本与运行状态完全分离

账本建立 `logical_slot_id/generation -> qualified_GT_id`，只有criterion可读写。最初来自正常训练Hungarian匹配结果与**实际预测birth**的交集，不能为了拿到更多正样本强行用GT插槽、激活实体或筛选视觉证据。

次时刻，query在prediction route中继承某slot，criterion可用账本查到目标GT。账本不能改变route；即使route错，也应产生相应的训练惩罚/诊断，而不是用GT修正推理。

若账本目标本次窗口完全不出现，该query是持续实体的“无观测”训练项。若仅前一scan出现，则监督前一scan mask、当前mask为空，并保持正确语义类别；不能把它整窗口都设为no-object。

ambiguity group内不强制一个无法从数据辨别的成员ID。空间分割监督保留；身份继承损失只对可靠、可区分对应启用，记录排除数量。

### 4.2 Q-INDEP 与 Q-TALA 对照

- `Q-INDEP`：相同query条件化、相同router和writer，但每阶段普通集合matching。
- `Q-TALA`：对已知账本实体的继承query固定GT对应；剩余queries只匹配未被这些关系占用的GT。

不要对每个窗口硬规定同一个query行号对应同个GT。行号可变，继承的是预测slot的监督关系。

若多个预测slot错误地绑定同个GT，账本不能偷偷合并预测ID；loss端只以固定的预测route confidence/稳定ID规则选一个正训练主项，其他记录duplicate并施加适当no-object/冲突监督，不能重复把全部设为同一正样本。此处理是监督分配，不改变部署预测。

### 4.3 必须覆盖辅助层

原 `SetCriterion.forward` 会为每个aux重新Hungarian [I14]。新criterion应支持明确的 `assigned_indices`/`matcher_mode`。

- 实体条件化之前的discovery辅助输出：原独立matching，保护当前发现能力。
- 条件化之后的辅助层和final：同一批继承关系；未占用部分可逐层重新match。
- 不得先在final施加TALA，又让后面aux函数无意重新匹配所有queries。

保留R1基础loss各分量与实际raw-sum multiplier。对比loss中已存在aggregate/perlayer的原训练行为先保持；新增visibility/identity loss在独立registry按显式权重加入。打印一次component表即可，不逐step审计所有key。

### 4.4 空目标、类别与可见性

`no-object` 是class head最后一类，不能与B4的 `background_class=0` 混淆；ignore253继续按原taxonomy处理。[I15]

窗口/当前阶段零正mask时返回可微0的mask/dice项，classification仍包含合法负项。不能索引target[-1]代表消失，也不能为避空集虚构一个GTmask。若同batch其他样本有GT，不得因为一个样本为空把整个batch target列表清空。

第一版不新增可见性head；用原class/no-object与当前stage mask支持表示 `visible`/`not observed`，不声称未观察就意味着物理移走。对已继承但当前无GT支持的实体，显式监督当前mask为空；若前一stage有GT，仍保留窗口类别和前一mask正监督。该负项属于TALA的继承/消失监督策略，须与匹配策略一起在loss表中披露。空mask是已知实体在该stage的负监督，不是虚构一个新GT物体。

## 5. 梯度与状态：不得“有序列”却没有有效学习路径

第一版用2-stage截断BPTT支持实体query投影/更新路径。route、阈值、候选选择保持detach；允许当前chunk内已选择的连续embedding值传梯度。跨chunk之后detach，不声称完整T反传。

旧B4 `step` 含in-place和detach，不直接把它的状态当可微长图。新runtime state保持预测数值；训练端可有一份数值相同的graph-valued embedding shadow，仅保留当前chunk，且不含GT选择。对影子分支做一次forward-value parity，不强求bitwise梯度相同。

每chunk完成后统一backward，不能在第一stage已经释放graph后再尝试通过第二stage回传。optimizer仍按完整episode/有效batch更新，不能每stagestep导致与对照学习率和样本曝光失配。所有M2候选共享相同TBPTT/优化边界；W-BASE/FH-MATCH无state梯度但同stage监督预算。

若稀疏算子无法支持该graph路线，在代码里明确实现 `tbptt=1` 的共同适配控制并将“跨step状态学习”判为未验证；不能悄悄降级后仍宣称实现了跨step学习。可继续当前read-module的有梯度研究，不把阻塞变成假阳性。

### 最小真实学习检查

一个含已有实体、一次缺失、再次出现和一个新实例的小训练episode面板，跑两次optimizer updates：

- 证明输入R1子树初始一致，空state/关闭module退回R1。
- 某个已启用adapter路径在第二步有非零梯度和权重变化；zero-init第一步上游0梯度正常。
- 当前损失反传到预定当前/前一stage连续状态；detach边界之外不反传。
- route/writer每stage调用次数、logicalID继承、birth结果来自预测。

仅这些与方法直接相关的检查足够。不要建立“每个参数每步都非零”的虚假门槛。

## 6. 预期产物

```text
implementation/query_state_contract.md
implementation/query_shape_trace.json
implementation/r1_load_report.json
implementation/sequence_loss_example.csv
implementation/real_gradient_smoke.json
implementation/router_diagnostics.csv
configs/W-BASE.yaml, Q-INDEP.yaml, Q-TALA.yaml, FH-MATCH.yaml
```

每份配置保存resolved diff：W-BASE→Q-INDEP的变化属于模型/路由，Q-INDEP→Q-TALA仅监督分配，不能模糊称“两边只改一行loss”却实际改不同writer或初始化。


---

# M3–M4：有界视觉证据、prefix教师与轨迹质量

## 1. 先区分三种记忆，不把名称当成机制

已有B4保存的 `embedding/class_prob/confidence/...` 是身份关联状态。C2让decoder读这个embedding，但没有保存与原mask对应的局部superpoint证据 [I09,I17]。

本轮新增视觉证据库借鉴Cutie/XMem/DeAOT的角色分离 [X02,X03,KB P21]，并不引入2D mask提示，不宣称完整复现原论文。第一版只给已路由对象的query提供多个视觉代表；Cutie式反向更新全部superpoints暂不启用。

拟新增：

```text
models/object_visual_memory.py
  ObjectVisualState
  extract_visual_candidates
  select_latest_representatives
  select_temporal_coreset
  read_object_visual_evidence
```

## 2. 状态schema与真实字节预算

```text
values       [B,K=100,r=8,D=128]   FP32 默认
valid        [B,K,r]               bool
quality      [B,K,r]               FP32，预测质量而非GT IoU
source_stage [B,K,r]               int64，来自已到达scan
source_key   [B,K,r]               稳定scan/代表索引或紧凑整数
slot_generation [B,K]             int64
```

不长期存全部点云、每个历史scan的dense feature、完整raw logits或autograd图。身份state仍另存，实际记录每个tensor的dtype/numel/bytes。仅value在此配置为409,600 bytes；加入key副本、metadata和身份状态后据实汇总。默认额外永久工作state预算不超过2 MiB/episode；这是工程上限，不是总显存上限，更不是已测性能。

优先存 **读取前特征**，在read时投影为key/value；不要保存由不断更新projection生成、无法追溯版本的旧key。训练episode之间state重置；推理模型权重固定。

lag1缓冲的当前/前一scan mask和映射是单独的有界窗口成本，不能拿2 MiB把它们省略；历史输出归档也单独统计。

## 3. 从哪里取真实视觉特征

使用 `ReScene.aggregate_features` 产生的mask feature空间：

```text
segment_features[layer_index=0][batch_index]
```

逐行与最终raw mask logits及segment_stage_id绑定。若为了效率对聚合特征又做了采样，必须同步使用返回的采样indices；不能假定原segment index仍成立。[I06]

只将**当前已到达scan**中的预测证据写入长期库。上一scan的重新解码可用于lag1输出，但默认不以第二次写入重复放大同一来源；同一source_key去重。

每个实际接受的query/slot：

1. 按预测mask sigmoid、类别confidence和有效support生成可靠性；不读取GT masks、GT classes或事后IoU。
2. 每实体最多先保留32个当前候选segment，按预测质量稳定排序；feature归一化仅用于选择相似性，value保留正确幅值/原特征。
3. 在候选内选择代表，具体分支如下。
4. 当前query无有效support/关联被拒绝/槽位generation已失效时，不写视觉state。

旧 `PointceptBackbone` 中命名为encoder的入口包含decoder [I21]。这里使用当前forward实际产生的特征，不“缓存整个encoder()”误当固定表示。

## 4. M3的三臂对照

三臂从同一个M2 checkpoint开始，相同优化预算、episode计划、read结构和新参数初值。`visual_enabled`与选择政策以外不改其他设置。

| 臂 | 保存信息 | 目的 |
|---|---|---|
| BASE-CONT | 仅原实体embedding，按同预算继续训练 | 排除额外训练产生的收益 |
| V-LAST | 每个实体8个代表，全部来自最近一次有效观察 | 视觉多代表是否有用，而非依赖遥远历史 |
| V-CORE | 相同8个代表，2个近期＋6个历史质量/覆盖代表 | 同容量跨时压缩是否优于只保留最新 |

另做一个不必完整训练的 `repeated_mean` 读取诊断：8个位置都重复同一均值，形状与token预算相同。它是内容控制，不是独立强模型，更不能用读取不同分布的失败代替训练对照。

### V-LAST

按当前候选质量选择第一个代表，再以feature距离的最远覆盖选择其余代表，全部只使用本次有效观察。实体当前没出现时保留上次库，不清空失联对象。

### V-CORE

新候选与最多8个旧代表合并，最多40个元素；同source去重。先锁定2个当前高质量/不同特征代表（不足时按实际数量），余下位置在其余候选中以：

```text
selection_score = predicted_quality * (0.5 + 0.5 * min_cosine_distance_to_selected)
```

逐次选到总数8。距离定义为 `(1-cos)/2` 截至[0,1]；平分时source_stage/source_key稳定排序。若当次没有可靠新候选，不改原库。保持实体独立配额，不用全局热门实体挤掉长期未访问对象。

该coreset是受文献启发的本地启发式，不称为XMem原算法；其质量与覆盖作用需由V-LAST及简单控制检验。不追加复杂可学习淘汰策略。

## 5. 读取与拒绝

对已路由query只读取绑定slot的有效visual代表，加一个严格零value的拒绝分支。先使用第03章同样的scaled-normalized注意力，再用零初始化输出残差；Q-INDEP/Q-TALA和visual分支的基础scale策略统一，避免将温度修复混为视觉记忆收益。

只给query加残差，后面的既有ReScene decoder会继续读取当前scene特征。新query仍有当前发现路径；无route或无visual时返回原query。

记录：有效token数、来源年龄分布、正确实体的历史权重（仅事后诊断）、null mass、残差范数、特征更新幅度。不以“attention变尖”“gate更大”作为成功目标。

若读取错误实体造成污染，先区分route问题与value问题，不通过GT纠正读对象。是否加入class/age/quality打分属于下轮单因素，不在本轮同时扫描。

若这一版得到明确visual收益但边界仍主导失败，才可提出object↔superpoint双向更新；本包不自动授权额外大模块。hook当前只返回queries，不能悄悄改一个feature副本后说实现了Cutie双向更新。

## 6. 训练基础损失与保持发现能力

基本损失使用03的继承/新生分配。与R1的共同窗口分割objective并存，保留条件化前的discovery辅助监督。不得使用target框、GT mask作为新query输入。

若T1/新生发现退步，先看独立discovery监督是否仍被执行、match是否被已继承GT重复占用、query预算是否真保持100，不立即靠额外再运行完整R1一次做ensemble救分。任何多前向ensemble要单独计成本和公平基线，不是当前默认方案。

在视觉M3内，先对长期视觉写入detach，保持当前read/decoder可学习；不要求每个旧token都跨多个stage保留梯度。知识保存是否有用，由同checkpoint读取干预和同预算训练对照共同验证。

## 7. 条件性prefix教师蒸馏

### 7.1 触发条件

只有出现以下证据才运行KD组：学生的读取确实影响正确对象的预测，训练loss/开发mask有可利用信号，但与FH在完整轨迹质量上仍有差距；或者全轨迹失败分析显示FH对学生弱阶段有可用预测。单纯“想继续涨分”不授权无限KD变体。

允许teacher：冻结FH-R1或本轮已选定FH-MATCH，其身份在训练前锁定。不根据最终Protocol-B结果选择教师。

### 7.2 教师可看什么

时刻t教师仅看 `X1..Xt`，学生看W2＋state。监督当前阶段及lag1允许修订的前一阶段。不得使用后续scan教前面不可访问的未来信息，除非另立离线训练变体并明确披露；本轮不做该变体。

教师和学生的voxel数量、superpoint划分可不同。通过 `original_vertex_ids` 对齐公共原始点，不能直接截前N个mask列或按queryindex匹配。

### 7.3 教师与学生的实例配对

训练期先各自对GT匹配，按qualified GT identity连接教师/学生；只为教师class正确且mask IoU≥0.5的配对提供KD，阈值0.5为本轮固定工程起点。若没有合格教师，保留GT训练，KD返回可微0并记录覆盖率。

默认新增：matched class KL（temperature2）和mask soft BCE，各自按有效配对/点规范化；基础GTloss不变；KD总权重0.5，class/mask内部等权。第一版不再同时加feature KD、uncertainty网络、pseudo-label再训练。

这些loss是本地拟议设计，不是声称原StreamVGGT或ReScene的精确loss。

### 7.4 数据增强与缓存

优先对同一原始episode和显式augmentation seed在线执行teacher（eval/no_grad），避免离线cache与随机增强错配。若缓存，key必须绑定teacher、scans、augmentation、预处理和原vertex映射；重新采样voxel时不能复用不相容teacher tensor。

不保存所有重复prefix的密集teacher masks；可以只缓存实际配对所需soft targets或一次原始scan映射，并设磁盘预算。teacher GPU-h全部计入训练，不用于推理。

### 7.5 对照

从同一个选定学生checkpoint分叉：`STUDENT-CONT` 与 `STUDENT-KD`，相同数据、初始化、更新数、LR，只有KD监督不同。允许一次这样的两臂实验，不扫描多个教师、多温度、多lambda。

需要报告额外阶段适配后的FH对照。若选择不继续训练FH，明确总训练量不匹配，不能宣称已经通过matched-training全T优越性门槛。

## 8. 最差阶段辅助目标：一个条件性实验，不直接称优化t-mAP

仅当已有序列匹配可靠、GT对应可用，且确有单一薄弱stage限制轨迹时才启用。为同一实体定义当前有效stage的 `1-softIoU`，用平滑最大值或选定最差stage的代理强调短板。不得对已经detach的历史boolmask算一个数就宣称它能训练过去。

低显存可实现方式：训练episode内先无梯度记录每stage预测状态与RNG，找出对应实例的薄弱stage；然后从预测快照重算该小窗口并对选中的raw logits求损失。快照不含GT输入，GT仅选择监督项；重算次数和扫描曝光计入训练预算。它不是完整端到端t-mAP，也不是部署期oracle。

这个实验与KD只能顺序隔离实施，默认先KD；最多增加一组有/无proxy的对照。若当前训练成本超过05上限或序列对应不稳定，则交付设计说明与 `NOT_RUN`，不要把无效梯度伪装成完整机制。

## 9. M3/M4最小产物

```text
visual/state_schema.json + memory_bytes.csv
visual/selection_diagnostics.csv
visual/paired_content_ablation.csv
training/M3_learning_curves.csv
training/M3_comparison.csv
teacher/teacher_contract.json
teacher/matched_target_coverage.csv
training/KD_pair_results.csv
```

不运行的阶段以状态记录占位，不生成看似实验结果的全零数值表。


---

# 执行预算、对照、晋级与停止规则

## 1. 本章数字是本轮工程起点，不是文献保证

过去400 updates的实验已经执行且科学目标失败；这不能被抹去，也不能当作充分训练验证所有结构方向。本轮使用新的固定预算，不改写旧 `FORMAL_OPTIMIZER_UPDATES=400`。

默认资源使用现有两张A40和已可用环境；无须为了本任务新增多机训练设施。作业不得终止无关用户进程，不根据旧GPU列表假定所有设备当前空闲。

| 阶段 | 实际工作 | 预算 |
|---|---|---|
| M0/M1 | 小样本、接口、内容诊断、简单bank | 无全模型训练；真实优化smoke 2–8步；诊断最多12个metadata预选dev masters |
| M2 | W-BASE、Q-INDEP、Q-TALA、FH-MATCH | 每臂3000 optimizer updates，一训练seed45 |
| M3 | BASE-CONT、V-LAST、V-CORE | 每臂1500 updates，从相同M2 checkpoint |
| M4 | STUDENT-CONT、STUDENT-KD；FH预算匹配 | 每臂1000 updates，条件运行；最多一组教师 |
| 额外weak-stage proxy | 一组有/无，对照预算匹配 | 非默认；必须在总资源cap内 |
| 确认 | 最终候选及最强同预算FH | 有希望后第二训练seed46；评价随机性单列 |

初始campaign上限：**120 GPU-hours训练（GPU数×实际训练墙钟），30 GPU-hours诊断/评价**。这是节制试验的建议上限，不是对完成时间的预测。资源不足时先保留完成的共同阶段，不压缩为400步再自称正式完成。进入运行前可依据实测单元成本一次性修改总cap并在contract中说明；不依据AP结果只给赢家加预算。

若需要超过已冻结cap，停止当前扩展，保存可恢复checkpoint并交接 `BUDGET_EXHAUSTED`；不自行无限刷点。用户没有明确授权的云租赁/付费服务不启动。

## 2. 正式训练recipe

### 2.1 从同一R1初始化

W-BASE/FH-MATCH严格加载全部R1树。Q-INDEP/Q-TALA共有新增结构必须从同一随机初始化文件加载；两个checkpoint第0步共同tensor逐项hash一致。新增prefix白名单显式记录，不用宽泛strict=False忽略主模型缺失。

M3/M4是相同父checkpoint下的等步数分叉，不拿“V-CORE多训练1500步”与“Q-TALA未继续训练”直接比较就归因视觉memory。保留BASE-CONT。

### 2.2 优化与精度

默认2GPU、physical1/GPU、accum4、effective8 episodes，32-true。各rank同H；同样本计划、相同优化边界、相同训练层。保留R1 encoder参数冻结与现有runtime mode；不要此轮顺便改变DropPath/EOS/filter255/分割总loss语义。

训练范围：原R1可训练的PTv3 decoder/ReScene decoder及heads＋本候选新模块。记录实际参数数，不照抄旧26.8M。额外训练预算相同指搜索与训练过程预算；不同dev选中步数的checkpoint曝光可能不同，必须额外给同terminal update的对照，不能把selected-best差值当作完全单点因果差值。已冻结网络可以传梯度到adapter输入，不把包含新模块梯度路径的整个后续decoder包进no_grad。

建议LR：既有trainable子树1e-5，新模块1e-4；AdamW weight_decay沿R1 resolved配置。warmup=该阶段总updates的5%，cosine衰减到初始LR的10%；gradient clip沿已有值。全阶段日程在开跑前固定。

在M2新共同recipe中，H长度损失首先按有效监督stage平均，每stage内部保持原criterion归一化；所有候选包括FH一致。前一window重叠监督/lag1发布不能重复算成两个独立样本，记录published-scan数量和loss-evaluated-scan数量。

该recipe不是严格ReScene复现，必须标 `adapted`。W-BASE与FH-MATCH是用来控制新数据/日程/初始化的效果。

### 2.3 Pilot不能换一条更短LR曲线

M2可先跑300updates观察运行/梯度/爆炸，scheduler仍是3000-step的前300步，保存优化器/RNG/state-contract。若继续，从这个trajectory精确恢复，不能重新设置300-stepcosine后拿早期最佳曲线判结构优越。

resume必须恢复optimizer/scheduler、stage/episode计数和下一draw index；同seed但忽略sampler状态不是精确恢复。无需追求跨硬件bitwise相同，声明可核验的恢复边界。

不依赖单个早期validation点淘汰互补路线。除nan、数据契约错误、持续OOM等技术失败外，M2比较都跑完固定预算。非finite不通过全局nan_to_num掩盖；定位新路径，不能重写旧训练objective。

## 3. 评估与模型选择

### 3.1 开发面板

正式dev以同一批具备T5的heldout参考构建四T曲线；native2–4更多场景作为额外dev覆盖分表，不混合分母。M2 checkpoints在0/750/1500/2250/3000评价；M3/M4在0及四等分阶段评价。缓存足够时mean/latest/max共享同一前向结果，不给每个reducer单独推理。

主要评价seed45，候选接近晋级时对两个bestdev checkpoints最多补46/47采样检查；这些是evaluation seeds，不是training seeds。不能为单候选试几十个eval seeds找高分。

### 3.2 主选择准则

对每个真实baseline checkpoint B：

```text
delta[T,B] = tmap_candidate[T] - tmap_B[T]
S_min = min over T=2..5 and prespecified baselines of delta[T,B]
```

开发selection可以使用多个baseline作为约束包络，但最终表必须逐个显示实际baseline，不能捏出一颗每个T取最优的虚构ReScene模型。

在已经注册的checkpoints中，先选S_min最高者，再选四T均值增益、再选低延迟，最后选较早update。primary reducer固定mean，policy固定lag1。不能按不同T选择不同epoch。

低T2基线不能制造retention优势：任何“质量保持”主张都同时列T2绝对值和相对FH的差距。没有全T领先的候选，仍选择一个最有信息量的研究候选去最终评价，但结果标FAIL，不降低原门槛。

## 4. M2→M3：不是一个全T门禁卡死所有路线

M3授权前必须确认：继承query在真实训练episode中存在、监督账本无GT泄漏、read可学习、当前/新生路径仍执行、状态预算正确。若dataset映射错或所有route都为空，应先修实现，不拿空机制训练结果判方向无效。

选择M3母体：

1. Q-TALA相对Q-INDEP的开发最小增益不低于-1.0pp，且任一长T提升≥0.5pp或身份/类别轨迹诊断给出对应改善时，选Q-TALA。
2. 否则若Q-INDEP相对W-BASE满足同样信号，选Q-INDEP。
3. 若两者均未产生任何可重复的mask/class/long-T改善，且同checkpoint关闭read不降，在技术诊断后只允许一个有依据的修复（如scale或route分布），用原对照和等预算重复；修复也无信号则停止扩展，给出 `MECHANISM_UNSUPPORTED`。不自动全跑视觉/KD。

以上0.5/1.0pp是内部工程晋级线，不是统计显著性或最终成功线。选择只用dev，规则在看到新实验前冻结。

**单纯没超过FH，不足以取消M3。** 视觉证据是不同假设，允许在有可学习实体路径但信息不足时检验；不能重复旧C3那种“两个短T必须都严格为正，否则所有互补组合都不许跑”的规则。

## 5. M3→M4：视觉贡献与蒸馏条件

V-CORE与V-LAST必须同时对比BASE-CONT。内容干预中真实长期visual优于只读latest/无关内容，或明确改善相同对象的薄弱mask，才称 `VISUAL_MECHANISM_SUPPORTED`。

如果V-LAST已经足够，V-CORE不优于它，则报告视觉证据有效但长期压缩策略无额外优势，不能为了叙事强选V-CORE。

M4只在04条件满足并且剩余预算可完成整个配对时运行。若没有教师可靠目标或学生看起来完全不使用memory，不先用KD掩盖失效。最多一个teacher、一个固定KD权重recipe。

## 6. FullHistory的公平训练

M2的FH-MATCH获得相同初始R1、episode计划、optimizer updates和GT监督源。其输入点数更高、训练GPU-h可以不同，分别报告同更新数和实际成本，不能声称两者算力完全相同。

若候选进入M3/M4并额外训练，将FH-MATCH从其选定M2 checkpoint继续同样stage日程/数据曝光，标为实际FH-CONT checkpoint。不要只让学生多训练后仍用早期弱FH标作matched。

若本轮新增的是通用局部分割结构，需给FH同样结构控制；本方案不默认加QCL/MAFT等通用模块，避免无限matrix。

FH编码缓存可作为最终部署优化，只有数值契约成立才启用。不能为了缓存而修改坐标、采样、精度或attention范围后仍称同一个baseline。真正编码边界是I21/I22指出的位置。

## 7. 哪些情况停止，怎样解释

| 情况 | 停止/继续 | 科学状态 |
|---|---|---|
| 数据ID无法跨scan正确对齐 | 停继承训练，交付实现和明确问题 | BLOCKED_DATA_ID |
| 当前模型与目标loss不相连 | 修直接梯度路径，最多聚焦迭代 | BLOCKED_IMPLEMENTATION |
| 300步pilot不涨分但运行正常 | 继续固定正式budget，不提前宣判 | PILOT_ONLY |
| 正式M2无信号且read被关闭不降 | 做一次有依据修复；否则停扩展 | MECHANISM_UNSUPPORTED |
| 部分T提升，部分T退步 | 保留最差T诊断，有互补证据才进入下一臂 | PARTIAL_NOT_ALL_T |
| 与FH全T领先但成本不达标 | 精度成功、联合目标未成 | ACCURACY_ONLY |
| 预算触顶/会话终止 | 保存完整可恢复状态、push部分产物 | BUDGET_EXHAUSTED/INTERRUPTED |
| 候选最终四T有一格不赢 | 不再在最终集合调参 | TMAP_ALL_T_FAIL |

不足预算不是“证明没希望”；运行完成不是“证明成功”。两者均要在handoff中独立标记。

## 8. 训练产物与节制

每臂保存：第0步初始化manifest、固定间隔checkpoint manifest、最终可恢复last checkpoint、dev选择的best checkpoint、learning_curves.csv、config diff、训练成本与曝光表。未选中大权重保留到完成比较后，按用户存储政策仅清理本轮可重建临时文件，不删除旧实验资产。

不要每step导出全部参数梯度、attention矩阵或raw点云。loss每20–50步记录；真实梯度检查在前两次有效update和中段一次；额外诊断抽样不超过预先固定的少量episodes。

最终全量Protocol-B只对预先冻结的候选和实际最强可比FH执行一次完整主评价，再做规定的采样/训练seed确认。知道它是历史曝光benchmark，别写untouched。


---

# M5：官方全T评价、机制验证、资源与结论

## 1. 新预测必须由新模型真实产生

新增 `scripts/evaluate_task_memory.py`，复用现有官方后处理、stmetrics与identity事件函数，不复制一套自定义AP。新模型的mask/class会变化，不能拿旧R1/C2 sidecar替代新输出，也不能要求新候选必须与旧候选相同。

正确的parity检查是：新模块关闭/空状态时，同一输入RNG下复原R1 raw输出与官方后处理；不是要求学习后仍保持129/129候选相同。

旧 `scripts/evaluate_persist4d_allt.py` 的请求构造只接受5scan [I19]。新入口支持native 1–5，并把统一common5曲线与native扩展分开。所有H评价都从空state执行T1..H，不能T2跳T4，也不能每个stage重新置空memory。

## 2. 三个独立的指标通道

### 2.1 主任务通道

按预测class、mask、persistent identity及固定mean分数构造prefix轨迹，调用原始stmetrics。

```text
t_mAP
t_mAP50
t_mAP25
t_REC
prefix_overall_mAP
```

另外报告官方direct latest/current AP。`prefix_overall_mAP` 必须调用Legacy AP对整个prefix的预测/GT，不能把最新scan AP改名。使用旧实现前核对其accumulator是否会自动裁到latest；不匹配时写一个薄包装，不改官方metric定义。

保留类ID和source query lineage。一个query可以产生多个class候选，继承的是query identity但class候选分别形成轨迹；不能只保留GTclass或删掉低分FP来提高AP。阈值及top-k设置统一、开发前固定。

主reducer为mean；latest/max全报为敏感性，只从同一候选计算。相同checkpoint、samepolicy acrossT。不能每个T分别挑reducer。

### 2.2 身份事件通道

重新计算：opportunities、attempts、correct、coverage、accuracy、recall，ID switch、fragmentation、merge、false birth、true rejected birth、false reactivation。分母0为N/A不是0或1。

FH每次独立前向的queryindex没有天然跨前向ID稳定性。FH-native t-mAP与FH的跨前向identity诊断是不同对象；若比较持续身份，必须使用公开的预测ID发布器，不能故意拿未经链接的raw indices生成巨量switch当作主要胜利。

### 2.3 内容/因果诊断通道

最终dev checkpoint上做：真实history、读取关闭、previous-only、相同形状/占用数的无关内容。不是只排列memory行。记录历史token来自哪个真实stage、当前query路由对错、mask薄弱stage改善与错误重激活。

这些事后干预是机制诊断，会改变运行分布；不单独称因果证明，也不依据最终benchmark关断结果再选一个更好的主模型。

## 3. 缓存不可再膨胀为每个prefix/reducer一整份点mask

上轮C2 cache约119GB [I01]。本轮默认紧凑缓存：

- 每次前向的官方候选mask按stage存一份（bitpack/RLE等无损），每scan原vertex映射单独存。
- 分数/class/lineage/版本以小表保存；三个reducer不重复mask。
- lag1只保存替换前后有必要核验的日志和最终stage版本，不能为每个T保存重叠的整个prefix dense矩阵。
- 评价时逐reference流式构造必需tensor，用完释放；不要所有model/seed同时常驻RAM。
- raw float mask/logits/全部attention只保留预选少量诊断样本，不保存全量作为默认产物。

缓存key至少包含：模型SHA、配置SHA、状态schema、数据/顺序/augmentation、eval seed、policy、后处理版本、完整到当前stage的扫描列表。状态依赖预测历史，所以不能只按“当前两scan相同”跨episode复用新模型输出。

每个文件写入时hash一次，首次读入校验；之后同次分析按manifest复用。不让所有reducer/指标都重复读权重或重hash119GB旧缓存。

默认本轮新evaluation工作缓存上限40GiB；超限先流式/无损压缩，不静默删结果或降低点数。可将必要归档转external，但真实读取与存储范围须披露。

## 4. 统一结果表，不拼版本

主表字段：

```text
population_id, reference_count, master_count, order_count
source_commit, checkpoint_sha256, config_sha256
training_seed, evaluation_seed, policy, reducer
window, K, r, state_bytes, T
all five task metrics, direct_current_AP
```

必须包括：

1. FH-R1 native，原B4 commit0，R1+B4 lag1。
2. 本轮W-BASE、实际母体与完整候选。
3. FH-MATCH及累计训练预算相当的FH-CONT实际checkpoint。
4. 必要的FH-lag1；若编码缓存等价成功，资源表额外列FH-cached。
5. D-LAST/D-EMA长期bank控制。

所有主结果同一population上比。不能从R0复制B3填入R1表；不能把35个文献仓库的原AP/HOTA表拼成此协议的t-mAP对照。

保存两种版本的学习比较：固定terminal update的对照，以及按照共同dev规则选出的best。不要只显示每种方法偶然最高的单点。

## 5. 全T胜出、质量保持与统计

### 5.1 主目标

对预注册的每个真实可比ReScene checkpoint：

```text
TMAP_ALL_T_PASS iff all(delta[T] > 1e-6 for T=2,3,4,5)
```

1e-6仅是数值平等容差，不是最小有意义增益。对原FH-R1、matched-training FH分开判定。四T均值更高但有一T为负，必须FAIL。

二级20格任务门槛：四T×五指标全部按相同数值规则正增益才标 `TASK_METRICS_ALL_T_PASS`。不通过时报告真实正格数，不淡化失败。

训练种子/评价种子原始结果与均值都提供。`single_run_pass` 与 `replicated_all_t_pass` 分开；一个训练seed不写稳定统计确认。

### 5.2 t-mAP保持

同时报告：

```text
A(T) = absolute t-mAP
R(T) = A(T)/A(2)             A(2)>0时
Dmax = max_T [A(2)-A(T)]
Delta_to_FH(T)
```

T2下降换来漂亮R(T)不算主目标。若要称“基本不变”，先注册实际可容忍的AP差值和不确定性；未注册时仅报告曲线，不下该强结论。可额外展示90% retention作为**aspirational target**，不能写成用户已同意的成功阈值。

t-IoU取最小stage但aggregate AP、GT集合和重估预测也会变，因此不得声称数学上任何实验的t-mAP必定严格随T单调下降。参考同一GT实体cohort做诊断，但完整官方GT评价仍是主表，不能只保留从始至终可见的简单对象。

### 5.3 独立单位

Protocol-B六个physical references，129 orders不是129独立环境。逐reference差值完整报告；bootstrap按reference重采样。要估计pooled AP区间，需每次从对应预测/GT重新算pooled AP，不能把macro AP差值区间换名。

节约成本可只对最终两个模型、T2/T5做至多1000次重采样；若调用太贵，使用equal-reference描述区间并明确它不是pooled AP置信区间。没有必要为每个中间配置做上万次bootstrap。

## 6. 资源目标：不同scope分开，所有对照真实计费

### 6.1 固定测量panel

沿用每reference预先选定一个canonical master；选择与GT/模型分数无关。对**同一panel**同时算task quality与成本，再画Pareto/retention曲线。

一张同型号A40上依次测候选和FH；5次warmup、10次measurement。测量前从真实previous-state快照克隆相同状态，每次回放一致；不能连续十次把同一scan写进memory导致状态增长或cache变热条件不一。

记录点数、voxels、superpoints、实际occupied、query数、precision和实现版本。旧B4在T5比T2更快可能只是局部scan更小；不解释为算法随着时间加速。

### 6.2 两个时间scope

1. `model_update`：H2D后，backbone、query路由/读取、mask/class、官方后处理、state/visual写入、lag1修订、紧凑输出记录。CPU/GPU同步计时包含这些步骤。
2. `end_to_end_update`：从已就绪文件读取至提交输出，包括I/O、collate、H2D；另报，不与scope1混。

若某模型输出序列化明显昂贵，单列 `output_materialization`，比较双方相同输出接口；不能只让FH计算/输出整个prefix而让学生只返回一帧就把差值全当网络提升。

从T1实际运行至每个T的**累计时间**另测一次；不能把独立各T的中位数机械相加当准确累计耗时。诊断preroll可排除在单步计时外，但累计运行必须包含真正前面更新。

### 6.3 显存/状态

同时记录：静态loaded-model allocated、绝对peak allocated、incremental peak、peak reserved、CPU历史工作状态、视觉库、lag1缓冲、输出归档字节。不同定义不要跨表混用。

目标是长T延迟和工作显存优于最强等价优化FH；T2也完整展示开销。默认 `RESOURCE_ADVANTAGE` 要求T4和T5的panel中位更新延迟与绝对peak allocated均低于相应FH，且状态预算通过；同时提供逐单元差值和端到端范围，不用仅一个最好单元判定。额外state上限2MiB不包括当前输入/有界lag1buffer，也不代表整个模型2MiB。

### 6.4 FH缓存是否合法

只允许做一轮小型等价检查：固定输入预处理与采样，拆分真正 `model.enc` 与跨时decoder；维护Point层级和不共享的in-place父链。`PointceptBackbone.encoder()` 本身包含decoder，直接缓存不合法 [I21,I22]。

窗口center shift会改变输入，不能将不同prefix中同一scan的编码当然复用。如果无法在合理工作量内证明等价，标 `FH_ENCODER_CACHE_NOT_ESTABLISHED`，保留未缓存FH基线；不为它重构整个外部PTv3，不暗改baseline预处理。

即使仅GPU驻留减小，CPU缓存仍增长也要报告。若存在等价FH缓存实现，它必须进入成本主对照，不能为了显得更快忽略它。

## 7. 容量与错误情形

B4/D-LAST等纯后端对照可复用冻结局部观察做容量曲线。**新memory-conditioned模型改变K/r会改变后续mask；不得只重放固定候选模拟新K模型的全部任务效果。** 真正的新模型容量实验至少对预注册小panel重新前向，并标清是否改变训练分布。

本轮最多选两个容量压力点加默认K=100，优先观察真实reject/误关联，不扫描几十个K后选最有利参数。全部没满就写“未进入饱和区”，不称淘汰策略有效。已有K64–200未满结论不覆盖新visualstate。

## 8. 最终状态必须拆开

```text
EXECUTION: COMPLETE / PARTIAL / BLOCKED
TMAP_ALL_T_VS_R1: PASS / FAIL / NOT_RUN
TMAP_ALL_T_VS_MATCHED_FH: PASS / FAIL / NOT_RUN
TASK_METRICS_ALL_T: PASS / FAIL / NOT_RUN
RETENTION: IMPROVED / NOT_IMPROVED / INCONCLUSIVE
RESOURCE: ADVANTAGE / TRADEOFF / NO_ADVANTAGE / NOT_MEASURED
MECHANISM: SUPPORTED / PARTIAL / UNSUPPORTED
GENERALIZATION: INDEPENDENT / BASE_EXPOSED_ONLY / NOT_ESTABLISHED
PUBLICATION: PUSH_VERIFIED / PUSH_FAILED / NOT_ATTEMPTED
```

不得把RESOURCE_PASS覆盖TMAP_FAIL，不得把一次t-mAP微小正值说成所有任务指标稳定改善，也不得把执行失败写成科学无效。


---

# 直接正确性检查、执行命令与GitHub交接

## 1. 只做与结论直接有关的检查

不用测试数量当成果。将检查集中在约8类，能复用旧测试就只补变更分支：

| 检查 | 最小内容 | 不做什么 |
|---|---|---|
| R1加载与关断parity | 真实T1/T2输入，named新增prefix，raw输出/官方候选一致 | 不全仓重训/重放129个parity单元 |
| 数据与监督隔离 | scan/vertex/ID对应、无未来、GT只进loss | 不做模糊/恶意输入大矩阵 |
| 路由与提交 | route只读、commit一次、birth/dormant/overflow | 不测试无关整数极值或网络攻击 |
| TALA监督 | inherited与newborn、缺失/ambiguity、aux不重配 | 不强迫每参数每步非零梯度 |
| 视觉库 | 真实currentstage token、固定r/K、相同来源去重、null零残差 | 不全量存attention和raw点云 |
| 训练路径 | 两次真实update，TBPTT边界、resume draw cursor | 不每step全参数hash/梯度扫描 |
| 输出与metric | lag1一次修订、class轨迹、fullprefix AP、N/A分母 | 不改官方metric让分数好看 |
| 发布 | 小manifest可解析、变更文件一次检查、push远端回读 | 不反复全盘/全历史隐私扫描 |

真实GPU gate只检查必要路径，有效后不重复多卡/多环境相同工作。偶发数值差记录tol和输入RNG，不为bitwise一致展开大规模兼容工程。Ruff只对本轮变更Python；`git diff --check`一次；相邻回归挑直接调用的路径，不默认143项全部复跑，更不跑几千全仓测试。

遇到错误按影响范围修复和重测；不重新跑所有研究阶段。保持01 source evidence和07 correctness evidence分开：测试通过不意味着方法主张成立。

## 2. 需要实现的CLI合同

下列是**新入口应支持的命令设计**，不是声称parent里已有这些文件。Codex实现后在真实环境验证并将最后使用的命令写入 `COMMANDS.md`。现有工具可复用，但不要伪造执行成功的stdout。

```bash
# 全部根目录由当前环境明确提供，绝不假定旧机器绝对路径还在。
export PERSIST4D_DATA_ROOT=/actual/data/root
export PERSIST4D_R1_CHECKPOINT=/actual/R1.ckpt
export PERSIST4D_CONCERTO_PRETRAINED=/actual/concerto_base.pth
export PERSIST4D_RUN_ROOT=/actual/new-run-root

python -m scripts.prepare_task_memory_v2 \
  --config conf/task_memory_v2/experiment.yaml \
  --data-root "$PERSIST4D_DATA_ROOT" \
  --r1-checkpoint "$PERSIST4D_R1_CHECKPOINT" \
  --output artifacts/task_memory_retention_v2

python -m scripts.train_task_memory \
  --config conf/task_memory_v2/Q-TALA.yaml \
  --run-contract artifacts/task_memory_retention_v2/BUDGET_CONTRACT.json \
  --data-contract artifacts/task_memory_retention_v2/DATA_CONTRACT.json \
  --init "$PERSIST4D_R1_CHECKPOINT" --seed 45 \
  --run-root "$PERSIST4D_RUN_ROOT"

python -m scripts.evaluate_task_memory \
  --checkpoint /actual/selected.ckpt \
  --evaluation-contract artifacts/task_memory_retention_v2/EVALUATION_CONTRACT.json \
  --population protocol_b_common_129 --policy lag1 \
  --horizons 2 3 4 5 --reducer mean \
  --cache-root "$PERSIST4D_RUN_ROOT/cache" \
  --output artifacts/task_memory_retention_v2/final

python -m scripts.profile_task_memory \
  --comparison-contract artifacts/task_memory_retention_v2/PROFILE_CONTRACT.json \
  --warmup 5 --repeats 10 \
  --output artifacts/task_memory_retention_v2/resources
```

所有真正执行的variant命令、resume命令、选定checkpoint必须在交接中给出。不得把shell占位符当作已验证的真实路径；公开命令使用环境变量，未公开实际路径在本机untracked解析文件保存。

## 3. 产物目录：少而完整，不要求几百个审计MD

```text
artifacts/task_memory_retention_v2/
  START_STATE.json
  EVIDENCE_MAP.md
  DATA_CONTRACT.json
  OUTPUT_CONTRACT.md
  BUDGET_CONTRACT.json
  EVALUATION_CONTRACT.json
  PROFILE_CONTRACT.json
  COMMANDS.md
  references_inventory.csv
  implementation/
    r1_load_report.json
    real_gradient_smoke.json
    query_state_contract.md
    sequence_loss_example.csv
  baseline/
    policy_comparison.csv
    long_memory_controls.csv
    gap_event_strata.csv
  training/
    variants.json
    resolved_configs/
    config_diffs/
    learning_curves.csv
    terminal_update_comparison.csv
    selected_checkpoint_comparison.csv
    selected_checkpoints.json
    costs_and_exposure.csv
  mechanism/
    read_content_interventions.csv
    visual_selection.csv
    teacher_coverage.csv
  final/
    all_t_metrics.csv
    paired_deltas.csv
    per_reference_metrics.csv
    identity_counts.csv
    retention.csv
    status.json
    independent_reference_results.csv
  resources/
    profile_units.csv
    per_update_measurements.csv
    cumulative_cost.csv
    state_bytes.csv
    paired_quality_cost.csv
  TEST_REPORT.md
  FINAL_REPORT.md
  FINAL_MANIFEST.json
  HANDOFF.md
```

未授权/未运行stage只在 `variants.json`/`status.json`记录state和reason，不创建假的空数值CSV充实文件数。可合并小产物，只要主张能定位到原始数字和来源。

## 4. MANIFEST与大文件

提交：代码、配置、必要测试、MD/CSV/JSON、少量解释性图、紧凑日志、input/checkpoint manifests。普通Git不提交大权重、数据集、119GB缓存、密钥token。

每个外部checkpoint记录：logical URI、SHA256、bytes、source code SHA、config SHA、parent checkpoint、训练seed、selected update、实际训练总updates、数据contract、selection metric及其dev population。

每个模型的`last`恢复点和`selected`任务点要区分；不能只保存selected模型参数而丢失optimizer，却声称可以精确续训。

存储manifest引用可实际定位的内容地址/本机alias解析说明，不写一个无人能解析的 `external:foo` 就算完成归档。共享Git只存必要逻辑引用，本地真实地址放untracked `external_assets.local.json`，交接说明由谁/哪种root解析，不暴露凭证。

**避免自引用hash：** `FINAL_MANIFEST.json`哈希清单不包括它自身及HANDOFF/发布回执。它可以包括已经冻结的FINAL_REPORT与结果。HANDOFF写manifest的真实hash；最终用户回执给HANDOFF真实hash。不要让两个文件互相引用对方最终hash，造成无限重写。

## 5. GitHub发布必须真正闭环

用户要求上传代码和结果，不只是给一个PR创建网址。

### 5.1 开发分支

```bash
git fetch origin
git worktree add -b research/persist4d-task-memory-retention-v2 \
  /chosen/isolated/worktree \
  32a51e11b51043ec5a5825215669ef7b0ea03bc2
```

目录已存在则核验后使用，不覆盖。按功能阶段提交，建议：

```text
M0: freeze task-memory contracts and evidence
M1: add prediction routing and sequence supervision
M2: complete controlled entity-query adaptation
M3: evaluate bounded visual evidence                 # only if executed
M4: evaluate prefix distillation                     # only if executed
RESULTS: freeze full-T evaluation and resource results
DOCS: finalize handoff and publication metadata
```

每阶段将可复验结果一起commit；不用每个函数一个commit，也不在一次小测试后push几十次。

### 5.2 两段式提交解决“最终SHA写进自身”的矛盾

1. 冻结代码与实验结果，提交得到 **E = results_commit**。
2. push E，`git ls-remote`确认远端E；这时已能证明实验内容被上传。
3. 根据E生成最终MANIFEST/HANDOFF，写 `results_commit=E`、实验是否通过、外部文件位置和已验证的E发布状态。不要写尚不存在的自身最终SHA。
4. 文档提交得到 **P = publication_commit**，push P，读取远端同分支SHA并确认P。
5. 通过GitHub在P上的原始HANDOFF、MANIFEST及一份主CSV回读，比较内容hash。最终回复报告P/E以及三文件hash。

公开HANDOFF的 `publication_commit_resolution` 可写“通过此文档所处commit/远端分支解析”；实际P与 `PUSH_VERIFIED` 记录在终端发布回执/最终回复中，不需要为了写P再生成新commit无限循环。

如果push失败，保留本地commit，报告错误原因与未上传部分；不宣称完成上传。PR可创建但不是必要条件；只有实际创建成功的PR URL才叫PR，不将`/pull/new/...`叫已创建。

```bash
git status --short
git diff --check
git push -u origin research/persist4d-task-memory-retention-v2
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git ls-remote origin \
  refs/heads/research/persist4d-task-memory-retention-v2 | cut -f1)
test "$LOCAL" = "$REMOTE"
```

不要force-push旧branch，不更改main，不更改旧FAIL表。最终worktree如有未提交本轮必要结果，应解释或提交；不能说clean但实际有结果遗漏。

## 6. HANDOFF.md模板

```markdown
# Persist4D Task-Memory Retention V2 Handoff

## 1. Goal and verdict
Execution / all-T vs R1 / all-T vs matched FH / retention / resources /
mechanism / generalization / publication，逐项列出。

## 2. Repository
parent SHA、branch、results_commit E、publication tip解析方式。

## 3. What was actually implemented
文件/符号、功能，哪些只是设计，哪些运行过。

## 4. Knowledge and upstream evidence
实际用的论文、repo、immutable revision/file blob、许可证适配边界。

## 5. Data and output policy
训练/开发/确认reference数与曝光；native长度；commit0/lag1定义。

## 6. Model state and training
Q/K/r/d、路由与写入、TALA/visual/KD设置、可训练参数、优化与总曝光。

## 7. Experiments completed and not run
每臂source/init/seed/updates、条件跳过和原因，不把gate-skipped写失败。

## 8. Primary table
同一checkpoint四T，逐个真实FH对照，五指标和实际差值。

## 9. Mechanism and failure evidence
正确历史依赖、read-off、视觉来源、gap/新生/冷启动等。

## 10. Cost and storage
同panel质量与模型/端到端更新、累计耗时、绝对/增量显存、状态与归档。

## 11. Selected and resume checkpoints
SHA/bytes/配置/父checkpoint/选择依据/实际可定位方式。

## 12. Reproduction commands
只列验证过的命令；新入口语法、环境变量、恢复cursor。

## 13. Tests and limitations
直接测试、真实GPU路径；未验证项及是否独立泛化。

## 14. Claims supported / not supported
不将有限T写无限期；不将policy收益写纯memory收益；不将评价seed写训练seed。

## 15. GitHub and next exact action
结果提交E已验证；文档tip回读步骤；下一步只给一个有依据的优先动作。
```

## 7. Codex最终回复格式

不要只回复“完成并通过审计”。给出：

1. branch、parent、E、P、本地/远端SHA、真实GitHub链接；
2. 实际实现与实际运行的candidate；
3. 最终checkpoint及其SHA、四T完整主表；
4. 对每个FH基线的最小t-mAP差值和20格结果；
5. T2起点、retention与成本是否同时达标；
6. 机制是否被read-off/同预算bank/visual控制支持；
7. GPU-hours、数据/扫描曝光、缓存与工作state范围；
8. 未完成/跳过/负结果，尤其没有独立数据时的限制；
9. `FINAL_REPORT.md`、`HANDOFF.md`、`FINAL_MANIFEST.json`路径和实际hash；
10. 简短的下一步建议，不能把科研FAIL藏在工程PASS后。

这是本轮任务的完成条件：**执行有依据的研究开发、实测预期目标、诚实判定并完成GitHub交接**，不是保证未做实验必定得到漂亮数字。


---

# 附录：可追溯来源索引

内部源码固定提交：`32a51e11b51043ec5a5825215669ef7b0ea03bc2`。Git blob是文件对象hash，不是仓库提交hash。以下条目只说明读取范围；未执行任何GPU实验。

| ID | 来源 | 源码读取范围/证据层级 | 文件 blob |
|---|---|---|---|
| I01 | [artifacts/allt_task_superiority_v1/HANDOFF.md](https://github.com/Orangekostar/Persist4D/blob/32a51e11b51043ec5a5825215669ef7b0ea03bc2/artifacts/allt_task_superiority_v1/HANDOFF.md) | source_read_this_audit | `未另取blob` |
| I02 | [artifacts/allt_task_superiority_v1/FINAL_REPORT.md](https://github.com/Orangekostar/Persist4D/blob/32a51e11b51043ec5a5825215669ef7b0ea03bc2/artifacts/allt_task_superiority_v1/FINAL_REPORT.md) | knowledge_base_and_conversation_source | `66099a3242149cd7beef5b6a3ab09691faa0644c` |
| I03 | [models/persist4d_allt.py](https://github.com/Orangekostar/Persist4D/blob/32a51e11b51043ec5a5825215669ef7b0ea03bc2/models/persist4d_allt.py) | source_read_this_audit | `3e4bafa07d33aba3c8b72b9c056f07f9af6172fd` |
| I04 | [models/persist4d_allt.py](https://github.com/Orangekostar/Persist4D/blob/32a51e11b51043ec5a5825215669ef7b0ea03bc2/models/persist4d_allt.py) | source_read_this_audit | `3e4bafa07d33aba3c8b72b9c056f07f9af6172fd` |
| I05 | [models/rescene.py](https://github.com/Orangekostar/Persist4D/blob/32a51e11b51043ec5a5825215669ef7b0ea03bc2/models/rescene.py) | 235–500 | `81f58619e7a67a4ef59e199c93950b067c5a0a38` |
| I06 | [models/rescene.py](https://github.com/Orangekostar/Persist4D/blob/32a51e11b51043ec5a5825215669ef7b0ea03bc2/models/rescene.py) | 235–500 | `81f58619e7a67a4ef59e199c93950b067c5a0a38` |
| I07 | [trainer/persist4d_allt_trainer.py](https://github.com/Orangekostar/Persist4D/blob/32a51e11b51043ec5a5825215669ef7b0ea03bc2/trainer/persist4d_allt_trainer.py) | 1–350 | `ef8b74267b54216221e71d4c3e1591821672af7c` |
| I08 | [trainer/persist4d_allt_trainer.py](https://github.com/Orangekostar/Persist4D/blob/32a51e11b51043ec5a5825215669ef7b0ea03bc2/trainer/persist4d_allt_trainer.py) | 1–350 | `ef8b74267b54216221e71d4c3e1591821672af7c` |
| I09 | [trainer/persist4d_allt_trainer.py](https://github.com/Orangekostar/Persist4D/blob/32a51e11b51043ec5a5825215669ef7b0ea03bc2/trainer/persist4d_allt_trainer.py) | 1–350 | `ef8b74267b54216221e71d4c3e1591821672af7c` |
| I10 | [datasets/persist4d_sequence_dataset.py](https://github.com/Orangekostar/Persist4D/blob/32a51e11b51043ec5a5825215669ef7b0ea03bc2/datasets/persist4d_sequence_dataset.py) | 1–440 | `fa5a28fdf79e346a388a6986b070a4363bb6d703` |
| I11 | [datasets/persist4d_sequence_dataset.py](https://github.com/Orangekostar/Persist4D/blob/32a51e11b51043ec5a5825215669ef7b0ea03bc2/datasets/persist4d_sequence_dataset.py) | 430–660 | `fa5a28fdf79e346a388a6986b070a4363bb6d703` |
| I12 | [datasets/semseg.py](https://github.com/Orangekostar/Persist4D/blob/32a51e11b51043ec5a5825215669ef7b0ea03bc2/datasets/semseg.py) | 470–710 | `9f396e2214dc5d4390092ce8268916e9e38062eb` |
| I13 | [datasets/pointcept_utils.py](https://github.com/Orangekostar/Persist4D/blob/32a51e11b51043ec5a5825215669ef7b0ea03bc2/datasets/pointcept_utils.py) | 1–190, 190–410 | `3ca75fc2e0706dde992adb4fd2ca52722517a1d5` |
| I14 | [models/criterion.py](https://github.com/Orangekostar/Persist4D/blob/32a51e11b51043ec5a5825215669ef7b0ea03bc2/models/criterion.py) | 1320–1480 | `acba8a0a6c2ab80a15cc0a329a83c6b4c13af05a` |
| I15 | [models/criterion.py](https://github.com/Orangekostar/Persist4D/blob/32a51e11b51043ec5a5825215669ef7b0ea03bc2/models/criterion.py) | 550–860, 1000–1330 | `acba8a0a6c2ab80a15cc0a329a83c6b4c13af05a` |
| I16 | [models/persistent_memory.py](https://github.com/Orangekostar/Persist4D/blob/32a51e11b51043ec5a5825215669ef7b0ea03bc2/models/persistent_memory.py) | 560–910 | `895683f7e9c77afbf5484de4268da672d6215ba6` |
| I17 | [models/persistent_memory_read.py](https://github.com/Orangekostar/Persist4D/blob/32a51e11b51043ec5a5825215669ef7b0ea03bc2/models/persistent_memory_read.py) | 95–215 | `f818e6be412bb7182abd724df76acdada6da0fb2` |
| I18 | [scripts/train_persist4d_allt.py](https://github.com/Orangekostar/Persist4D/blob/32a51e11b51043ec5a5825215669ef7b0ea03bc2/scripts/train_persist4d_allt.py) | 1–230 | `3b16bf8a953c6235fd627f8366b8e97f2e8bc1b2` |
| I19 | [scripts/evaluate_persist4d_allt.py](https://github.com/Orangekostar/Persist4D/blob/32a51e11b51043ec5a5825215669ef7b0ea03bc2/scripts/evaluate_persist4d_allt.py) | 1–265 | `760098a7cd7e95e74ba322650ed6dfb0f49ffe5d` |
| I20 | [scripts/rescene_task_postprocess.py](https://github.com/Orangekostar/Persist4D/blob/32a51e11b51043ec5a5825215669ef7b0ea03bc2/scripts/rescene_task_postprocess.py) | 1–275 | `2e068775f791e9684a78517bfb3aeb172f8b6c8b` |
| I21 | [models/pointcept.py](https://github.com/Orangekostar/Persist4D/blob/32a51e11b51043ec5a5825215669ef7b0ea03bc2/models/pointcept.py) | 1–230, 300–455 | `311c4e4d2573dd924664213132c3d6bc6a2c516f` |
| I22 | [models/pointcept.py](https://github.com/Orangekostar/Persist4D/blob/32a51e11b51043ec5a5825215669ef7b0ea03bc2/models/pointcept.py) | 1–230, 300–455 | `311c4e4d2573dd924664213132c3d6bc6a2c516f` |
| I23 | [scripts/system_comparison_v2_inference.py](https://github.com/Orangekostar/Persist4D/blob/32a51e11b51043ec5a5825215669ef7b0ea03bc2/scripts/system_comparison_v2_inference.py) | 80–240 | `ada48c0cabf5db0a8e98cbe2640aa689d993a61f` |

## 外部方法与核心文件

**X01** [megvii-research/MOTR](https://github.com/megvii-research/MOTR/blob/main/models/motr.py)；论文：<https://arxiv.org/html/2105.03247>。证据层级：`source_read_this_audit`。 文件blob：`c358e3f0d958f4b607207d1c9c20632e343a7b0a`；内容地址：<https://api.github.com/repos/megvii-research/MOTR/git/blobs/c358e3f0d958f4b607207d1c9c20632e343a7b0a>。

**X02** [hkchengrex/Cutie](https://github.com/hkchengrex/Cutie/blob/main/cutie/model/transformer/object_transformer.py)；论文：<https://arxiv.org/html/2310.12982v2>。证据层级：`source_read_this_audit`。 文件blob：`089b51e4eef29388c9fc4e75c391adb012ede4f3`；内容地址：<https://api.github.com/repos/hkchengrex/Cutie/git/blobs/089b51e4eef29388c9fc4e75c391adb012ede4f3>。

**X03** [hkchengrex/XMem](https://github.com/hkchengrex/XMem/blob/main/inference/memory_manager.py)；论文：<https://arxiv.org/abs/2207.07115>。证据层级：`source_read_this_audit`。 文件blob：`bce2c00cc8aa15eb85166d0cd02b9ef3af79b180`；内容地址：<https://api.github.com/repos/hkchengrex/XMem/git/blobs/bce2c00cc8aa15eb85166d0cd02b9ef3af79b180>。

**X04** [CyndxAI/QKNorm](https://github.com/CyndxAI/QKNorm)；论文：<https://aclanthology.org/2020.findings-emnlp.379/>。证据层级：`primary_paper_plus_knowledge_base`。本轮未独立固定整个外部仓库commit，实际复制代码前需锁定。

**X05** [wzzheng/StreamVGGT](https://github.com/wzzheng/StreamVGGT)；论文：<https://arxiv.org/html/2507.11539v2>。证据层级：`primary_paper_plus_knowledge_base`。本轮未独立固定整个外部仓库commit，实际复制代码前需锁定。

**X06**：知识库 P27 MOTRv2、P29 MeMOTR、P09 AutoSeg3D；详细作者仓库和此前核心源码记录在 `knowledge/`。

