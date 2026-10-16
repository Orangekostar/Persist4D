# MPR-V1 实际交接

本提交发布状态：PENDING_EXTERNAL_VERIFICATION。发布事实以该 tag 的 Release 外部回执为准；上传失败时回执保存在运行根 publication/PUBLICATION_RECEIPT.json，状态与错误如实记录。

任务：固定 E1-23760 seed45，比较首掩码前历史检索与 M0–M6 强对照。
固定 base：6b1b779172afdeb51e2719a737cd05e7eeeca982。
分支：`research/persist4d-mpr-v1-formal-20261014`；tag：`persist4d-mpr-v1-formal-20261014-v1`；冻结实验提交 A：`e655d21399acc65df8cacaf4849605f1a8f503e5`。

代码：COMPLETE_C3_TRAINING_WITH_LOCKED_V4_INFERENCE；实验：COMPLETE；科学：SCREEN_NO_GO。
正式 N：400；选模状态：COMPLETE；seed147：NOT_EXECUTED_CAL_GATE_FAILED。
复验未执行原因：["DELTA_AP_T3_BELOW_-0.002", "DELTA_AP_T4_BELOW_-0.002"]。
评测政策：{"policy_id": "ordered-sparse-reference-v4", "policy_sha256": "4f1f7b6d1d1efdfbf73cbb6c56296b8d00b39dd5c1f2f6a7ead5a2a15dc309c6"}。
实际逐变体最佳点：{"M0": {"adapter_seed": 146, "step": 0, "variant": "M0"}, "M1": {"adapter_seed": 146, "step": 200, "variant": "M1"}, "M2": {"adapter_seed": 146, "step": 300, "variant": "M2"}, "M3": {"adapter_seed": 146, "step": 400, "variant": "M3"}, "M4": {"adapter_seed": 146, "step": 200, "variant": "M4"}, "M5": {"adapter_seed": 146, "step": 200, "variant": "M5"}, "M6": {"adapter_seed": 146, "step": 200, "variant": "M6"}}。
实际锁定端点：{"candidate_descriptive_star": {"AP": {"T1": 0.43269217014312744, "T2": 0.2687074542045593, "T3": 0.13553912937641144, "T4": 0.10350719094276428, "T5": 0.18716569244861603}, "AP_ranges": {"T1": 0.0, "T2": 0.0, "T3": 0.0, "T4": 0.0, "T5": 0.0}, "S_current": 0.4079311589399974, "S_long": 0.14207067092259726, "S_long_range": 0.0, "adapter_seed": 146, "current_AP": {"T1": 0.43269217014312744, "T2": 0.4164098799228668, "T3": 0.38026824593544006, "T4": 0.376382976770401, "T5": 0.467142254114151}, "guard_pass": true, "parameters": 84224, "step": 200, "variant": "M4"}, "candidate_star": {"AP": {"T1": 0.43269217014312744, "T2": 0.2687074542045593, "T3": 0.13553912937641144, "T4": 0.10350719094276428, "T5": 0.18716569244861603}, "AP_ranges": {"T1": 0.0, "T2": 0.0, "T3": 0.0, "T4": 0.0, "T5": 0.0}, "S_current": 0.4079311589399974, "S_long": 0.14207067092259726, "S_long_range": 0.0, "adapter_seed": 146, "current_AP": {"T1": 0.43269217014312744, "T2": 0.4164098799228668, "T3": 0.38026824593544006, "T4": 0.376382976770401, "T5": 0.467142254114151}, "guard_pass": true, "parameters": 84224, "step": 200, "variant": "M4"}, "control_star": {"AP": {"T1": 0.43269217014312744, "T2": 0.27176153659820557, "T3": 0.14357934892177582, "T4": 0.11752919107675552, "T5": 0.1418112963438034}, "AP_ranges": {"T1": 0.0, "T2": 0.0, "T3": 0.0, "T4": 0.0, "T5": 0.0}, "S_current": 0.391400545835495, "S_long": 0.13430661211411157, "S_long_range": 0.0, "adapter_seed": 146, "current_AP": {"T1": 0.43269217014312744, "T2": 0.40336376428604126, "T3": 0.3761378824710846, "T4": 0.3838258981704712, "T5": 0.4142378568649292}, "guard_pass": true, "parameters": 51200, "step": 400, "variant": "M3"}}。

选点只使用完整 CAL 四点、三次独立进程，按 S_long、S_current、T2、较少参数和变体名排序；家族内最后以较早步数打破并列。SEL/PB 不改模型或阈值。seed146/147 适配同一个 seed45 父模型，不是两个独立父训练。

新增 GPUh：45.853692703；累计：1760.667754119；累计余额：62.332245881；保留预留：8.000000000。
发布准备要求本任务 ACTIVE_JOB.json 不存在；最终活动进程检查、各阶段完成范围和部署回载证据见 STATUS.json 与 PROMPT_AUDIT.md。

父包：artifacts/native_long_postselection_v1/deployment/E1-23760.pt，SHA256 `8c9d0d57044ce23c7964ddb1e7a2d8bd18e388e6ee657a7f4b02b1368b5b3a98`。
编码器 SHA256 `845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07`；按原 Concerto 许可单独取得，未重新发布。
3RScan 按原许可取得，使用仓库 README 的预处理和 resolver；固定父包 Git/LFS、编码器及数据来源见 [DEPENDENCY_ACQUISITION.json](DEPENDENCY_ACQUISITION.json)。原始点云、逐点 GT、编码器和本地绝对路径不在本次公开资产中。

实际模型及指标包数量：9；每项实体字节、SHA256、变体、种子、步数和 optimizer 标记见 MODEL_MANIFEST.json。
部署必须同时具备固定父包、指定编码器和 delta；严格加载、V4 推理上下文及离线复算命令见 [DEPLOYMENT_PROTOCOL.md](DEPLOYMENT_PROTOCOL.md)。optimizer 恢复使用 manifest 实际列出的 resume 文件与原 RUN_LOCK，其进度为原 N 边界。
evidence/official_states.zip 保存可重新池化的官方充分状态。TRAINING/CAL_ALL/SEL_LOCKED/PB_LOCKED/BY_REFERENCE/IDENTITY_EVENTS_SUMMARY 保留原始行；机制与资源见 MECHANISM/PROFILE，实际命令见 COMMANDS.md。

REPORT.md 与 PROMPT_AUDIT.md 记录所有失败、分母、历史面板暴露、数值根因限制以及 H1–H3 支持范围。有限重复一致不证明全局逐张量确定性；内部面板结果不等于 ReScene 原论文完整测试指标，也不自动表示 CVPR 投稿就绪。

ARTIFACT_MANIFEST.json 冻结 A 中的代码、配置和实验资产，排除自身、B 交接及外部回执。HANDOFF 只引用 A，不嵌入自己的提交 SHA。

唯一后续判断：按 SCREEN_NO_GO 停止将此候选作为已验证赢家继续涨点或投稿；保留可复现基线及负结果，不追加训练或修改门槛。
