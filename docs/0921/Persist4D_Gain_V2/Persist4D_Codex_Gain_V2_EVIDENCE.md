# Persist4D Gain V2：证据与任务绑定补充

日期：2026-09-22。此文件解释证据和审核范围，不提供另一套竞争执行规则。执行以 `Persist4D_Codex_Gain_V2_FINAL.md` 为准。

## 1. 版本

- 核查到的远端V1分支仍指向 `8b5e93795817682fe70864daa545db219e1443c9`。
- 主指令以该提交为新开发起点；所有仓库来源均使用完整SHA链接。
- 已对照当前会话中完整的V1最终指令；未从文件名猜测内容。
- 检查范围为指定训练/评价/修复/关联/发布接口与已产生结果，不是宣称逐行审查整个仓库。

## 2. 本次重点补查的实现细节

| 已查看接口 | 源码中可确认的行为 | V2据此作出的安排 |
|---|---|---|
| `PerceptionGainTrainer.configure_optimizers` | 优化器从settings读取LR；恢复保存optimizer/scheduler/rank RNG | 参数化recipe即可测试低LR，不重写trainer；H350只能原调度恢复 |
| `_build_loader` | 固定sample plan、显式draw cursor、独立collator RNG；已有worker没有timeout参数 | 不改变样本流；补局部IO超时/本地数据策略，拒绝把超时当NFS修复 |
| `CausalRefinerRevisionTransform` | 恰有相邻old才处理；按真实vertex与new segment映射 | 两种头同一个配对范围，只消除NEW_ONLY的old数值信息 |
| `CausalMaskRefiner.forward` | 135维特征、2*tanh残差、基础输入detach | 保持容量与±2；明确可改性限制，不临时放大残差挑结果 |
| `_load_refiner` | frozen-v1 schema只允许固定键，没有input_mode | schema与train/eval/profile/bundle一起扩展，防止用错模式 |
| `preregistered_association_configs` | 已有12个固定配置 | 复用而非新增关联模型；由于publisher不同，不直接叠加refiner |
| `run_checkpoint_evaluation` | 通过compose加载配置，source_commit为固定父值 | V2传入真实H/L recipe与executed commit，不给低LR权重贴高LR配方 |
| `_resolve_cache_assets` | 原live调用也可能被旧dev cache键卡住 | V2按任务区分live资产与legacy-import资产 |
| `build_publication_receipt` | 旧判断主要以release存在与否决定状态 | 必需assets必须实齐，空Release不能算完整交付 |
| GitHub CLI官方说明 | 没有tag时可能默认从main创建；immutable发布后附件不可改 | 先推B/tag，draft传全部必需文件，再发布，最终receipt外置 |

## 3. 哪些是事实，哪些是实验假设

| 项目 | 类型 | 不应扩大成什么 |
|---|---|---|
| C0-750相对冻结R1平均−1.659443pp | 历史CAL结果 | 不等于已证明学习率太高或全部微调无效 |
| S-BAL-250相对同步数C0长时平均+0.455961pp | 历史CAL局部正信号 | 不等于新系统已经超过冻结R1 |
| 94/337=27.893175% | 修正口径后、旧缓存上的E1事件比例 | 不等于V2当前live关联错误率或可恢复AP上界 |
| 1e-5适配LR | 新实验候选值 | 不等于已发现最佳学习率 |
| R-NEW与R-PAIR对照 | 新的控制实验设计 | 不等于已经证明历史证据有效 |
| 0.5pp平均、1pp长时 | 研发目标 | 不等于准确率保证 |
| 双主线解耦 | 任务依赖修订 | 不等于没有真实数据/权限也能完成所有实验 |

## 4. 来源目录

- **E01** [V1最终报告](https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/artifacts/perception_gain_v1/FINAL_REPORT.md)：actual steps、CAL、未执行确认及profile。
- **E02** [V1运行状态](https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/artifacts/perception_gain_v1/RUN_STATE.json)：11.803828054742961累计已记GPU-hours、scorer、partial状态。
- **E03** [S-BAL实际中断](https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/artifacts/perception_gain_v1/training/S-BAL/INTERRUPTION.json)：350 resume、379未保存、NFS/NCCL及估计费用。
- **E04** [Scorer训练摘要](https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/artifacts/perception_gain_v1/training/scorer/RUN_SUMMARY.json)：500 updates、64 refs、梯度/参数变化。
- **E05** [各checkpoint CAL结果目录](https://github.com/Orangekostar/Persist4D/tree/8b5e93795817682fe70864daa545db219e1443c9/artifacts/perception_gain_v1/training/pilot/evaluation/cal)：C0 0/250/750、S-BAL250、Q-SEM0；本目录下每个JSON是数值来源。
- **E06** [修正E1门槛](https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/artifacts/perception_gain_v1/foundation/e1/gate.json)：94/337、ASSOCIATION_HEADROOM_SUPPORTED、ambiguity信息不完整。
- **E07** [Live smoke及旧缓存比较](https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/artifacts/perception_gain_v1/foundation/LIVE_SMOKE.json)：cached/live FAIL，live repeat PASS；不能拼接跨producer结论。
- **E08** [感知/修复模型实现](https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/models/perception_gain.py)：SemanticQueryScorer、CausalMaskRefiner、pair_adjacent_soft_occurrences、阶段损失。
- **E09** [旧总编排](https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/scripts/perception_gain_campaign.py)：load_campaign_config固定192/runtime；pilot/full固定臂数与覆盖。
- **E10** [实际感知评价入口](https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/scripts/perception_gain_evaluation.py)：run_checkpoint_evaluation、_load_evaluation_weights、live producer与固定source/root。
- **E11** [V1代码绑定表](https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/artifacts/perception_gain_v1/CODE_BINDINGS.md)：用于定位，但缺真实函数起止行；V2需补齐。
- **E12** [Criterion实际接入](https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/models/criterion.py)：loss_masks/_loss_masks_by_stage、aux、stage_debug与legacy路径。
- **E13** [查询/注意力前向](https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/models/rescene.py)：initialize_queries、semantic_query_positioning、open_first_cross_attention。
- **E14** [修复训练和运行回调](https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/scripts/train_perception_refiner.py)：align_old_probabilities_to_new_segments、CausalRefinerRevisionTransform、训练目标。
- **E15** [Soft证据与物化](https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/scripts/rescene_task_postprocess.py)：extract_official_task_prediction、materialize_segment_logits；bool与heatmap路径。
- **E16** [有限关联配置](https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/models/overlap_entity_association.py)：preregistered_association_configs返回3+3+6，AssociationConfig。
- **E17** [关联/输出重放](https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/scripts/replay_crosswindow_association.py)：canonical输入、独立publisher/HistoryBoundary、D0桥接。
- **E18** [V1普通感知训练器](https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/trainer/perception_gain_trainer.py)：configure_optimizers、progress、checkpoint恢复、runtime合同。
- **E19** [V1训练入口与确定性数据流](https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/scripts/train_perception_gain.py)：compose_variant_config、_build_loader、run、PerceptionDrawDataset、checkpointcallback。
- **E20** [V1发布器](https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/scripts/perception_gain_publish.py)：固定roots/branch、stage清单、Receipt判定与assets。
- **E21** [基础绑定与历史/实时资产区别](https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/scripts/perception_gain_foundation.py)：run_corrected_e1读取旧cache，_resolve_cache_assets强制旧缓存键。
- **E22** [V1数值配置](https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/conf/perception_gain_v1/common.yaml)：LR5e-5、raw_sum、frozen_encoder_eval=false、batch32、3000调度。
- **E23** [父模型修复数据生成](https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/scripts/prepare_perception_refiner.py)：run显式parent、advance_d0_identity、32refs清单。
- **E24** [修复评价与旧schema](https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/scripts/perception_refiner_evaluation.py)：_load_refiner严格旧键；V2须贯穿mode/parent来源。
- **E25** [Native FH评价入口](https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/scripts/perception_gain_native_evaluation.py)：完整prefix producer与输出身份。
- **E26** [LOCAL-T2入口](https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/scripts/perception_gain_local_evaluation.py)：原生T2独立人口与manifest。
- **E27** [官方metric adapter](https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/scripts/p6a_metrics.py)：OfficialMetricAccumulator、official_temporal_match_trace、阈值来源。
- **E28** [真实profile实现](https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/scripts/perception_gain_profile.py)：真实网络/状态/物化及scope。
- **E29** [D0的修订接口](https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/scripts/task_memory_output.py)：RevisionMaskRequest、LagOnePublisher、logical/class/generation归档。
- **E30** [冻结数据角色](https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/artifacts/perception_gain_v1/DATA_ROLES.json)：TRAIN/CAL/SEL/PB/LOCAL/ADDITIONAL对应关系。
- **E31** [V1已读执行规范](https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/artifacts/perception_gain_v1/EXECUTION_INSTRUCTION.md)：V1协议、运行约束与V2显式调整的来源。
- **E32** [V1实际执行日志](https://github.com/Orangekostar/Persist4D/blob/8b5e93795817682fe70864daa545db219e1443c9/artifacts/perception_gain_v1/EXECUTION_LOG.jsonl)：测试范围与耗时、RPC错误、partial发布。

外部一手资料：[ReScene4D](https://arxiv.org/html/2601.11508v2)、[LaSSM](https://arxiv.org/abs/2602.11007)、[PyTorch DataLoader](https://docs.pytorch.org/docs/2.14/data.html)、[GitHub release create](https://cli.github.com/manual/gh_release_create)。

这些来源支持接口与设计动机；没有把其它数据集的论文涨点幅度搬到本项目。
