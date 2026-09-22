# Persist4D Perception Gain V1 Handoff

## 1. 仓库、起点 SHA、工作分支、实验代码/结果提交 SHA

仓库 `Orangekostar/Persist4D`；起点 `6ef77620aa20926311eff3124a794a6ca2e32727`；分支 `research/persist4d-perception-gain-v1`；实验内容提交 `9e537358b4f78d48f037c75ea1aa040125cb12e5`。

## 2. 本轮目标与最终结论：哪项是实测、哪项仍未确认

执行状态 `PARTIAL_WITH_BLOCKERS`。已完成阶段：bootstrap, bind, foundation, scorer。确认状态 `NOT_RUN`；未完成项不作胜出结论。

## 3. 已改文件与函数、训练/推理数据流

唯一代码映射见 `CODE_BINDINGS.md`。训练为 ScanNet 单扫描与 TRAIN 两扫描；CAL/SEL 冻结选模；PB/LOCAL/ADDITIONAL 只用于锁定后确认；D0 保持 lag1/mean。

## 4. 确认的 R1 初始化、最终模型/头的哈希、所需基础权重

R1 `629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd`；Concerto `845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07`；最终 recipe `N/A`。基础权重由 `external:assets.local.json` 解析，不随 Git 分发。

## 5. 数据 population、物理 reference 数量与历史暴露

TRAIN/CAL/SEL/PB/LOCAL-T2/ADDITIONAL 数量分别为 `377/4/4/6/41/41`。PB 是历史暴露人口；LOCAL-T2 与 ADDITIONAL 分表报告。

## 6. 每个臂实际步数、checkpoint 选择和淘汰理由

以 `RUN_STATE.json` 的 `perception_pilot`、`perception_full`、`perception_select` 字段及 `selection/` 冻结文件为准。`NOT_RUN` 与外部阻塞不记为模型实验失败。

## 7. 最终锁定 recipe 与默认部署回退决定

锁定 recipe：`N/A`。默认部署由 `confirmation/CONFIRMATION_SUMMARY.json` 的严格 PB 规则决定；无完整确认时保持 `UNCONFIRMED`，不从 PB 反向选模。

## 8. LOCAL/PB/ADDITIONAL 结果、相对 R1 与 C0 的差值

确认状态 `NOT_RUN`。完整数值和逐 reference 行见 `FINAL_REPORT.md` 与 `confirmation/CONFIRMATION_SUMMARY.json`。

## 9. 当前 native FH 与真实 profile 完成程度

Profile 状态 `NOT_RUN`；方法、6 个 canonical masters、1 次整序列 warmup、3 次整序列测量及 scope 见 `resources/PROFILE_SUMMARY.json`。缓存重放不替代 live profile。

## 10. 测试命令与实际结果；没有运行的部分

目标测试、最终回归和 lint 结果以发布前最后一条执行日志及最终报告为准。未执行的训练、确认、profile 会在表 1 和阻塞列表保持显式。

## 11. 真实复现命令、环境、资产解析方法

环境命令前缀 `conda run -n persist4d`；公共执行入口为 `python -m scripts.perception_gain_campaign run --config configs/perception_gain_v1.yaml --external-root "$PERSIST4D_PERCEPTION_RUN_ROOT" --through publish --resume`。实际命令：

- `bootstrap`
- `run --through scorer`
- `conda run -n persist4d python -m pytest tests/test_perception_gain_*.py tests/test_crosswindow_metrics.py tests/test_p6a_metrics.py tests/test_rescene_query_features.py tests/test_rescene_task_postprocess.py tests/test_task_memory_output.py -q`
- `conda run -n persist4d python -m pytest tests/test_crosswindow_core.py tests/test_crosswindow_metrics.py tests/test_p6a_metrics.py tests/test_p6b_artifacts.py tests/test_p6b_runner.py tests/test_perception_gain_*.py tests/test_r1_downstream_cache.py tests/test_rescan_evaluator.py tests/test_rescene_query_features.py tests/test_rescene_task_postprocess.py tests/test_reviewer_closure_decomposition.py tests/test_reviewer_closure_sidecar.py tests/test_run_system_comparison.py tests/test_system_comparison_inference.py tests/test_system_comparison_v2_cache.py tests/test_system_comparison_v2_parity.py tests/test_task_memory_evaluation.py tests/test_task_memory_output.py -q`
- `timeout 12 rpcinfo -t 192.168.100.102 nfs 4`
- `finalize-partial --blocker repo:artifacts/perception_gain_v1/training/S-BAL/INTERRUPTION.json`

## 12. 大文件 release 链接、manifest、取得基础权重的方法

Git 小文件清单见 `ARTIFACT_MANIFEST.json`，拟上传大文件见 `RELEASE_PLAN.json`。基础 R1/Concerto 仅记录 SHA256 和资产键；不重新分发第三方权重或原始数据。

## 13. 剩余阻塞：已执行命令、异常、最小下一动作

- `perception_pilot`: NFS data workers blocked long enough for rank 0 to hit the 1800000 ms NCCL ALLREDUCE watchdog timeout

最小下一动作是恢复对应外部依赖后，原命令加 `--resume` 继续，不重置 schedule。

## 14. GitHub 发布验证方法与 release receipt 位置

发布阶段比较本地 HEAD、远端分支 SHA 与 tag 目标；再读取远端 `HANDOFF.md` 和确认 summary。外部 receipt 固定为 `external:publication/PUBLICATION_RECEIPT.json`。当前 publication phase 为 `READY`。
