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
