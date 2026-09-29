# 0929 专项修复复现

工作目录：`/home/ww/paper5/.worktrees/rescene-code-first-audit-v1`。
历史基线为 `4bf00902c9428795f7f47547bf462540aa304cc6`，历史目录、锁点和 tag 不修改。
新运行目录为 `/home/ww/persist4d_runs/rescene_code_first_audit_v1`，
公开证据在 `artifacts/rescene_code_first_audit_v1`。

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHON=/home/ww/miniconda3/envs/persist4d/bin/python
$PYTHON -m scripts.rescene_code_first_diagnostics
$PYTHON -m scripts.rescene_code_first_audit relabel
$PYTHON -m scripts.rescene_code_first_audit train-q --device cuda:0
$PYTHON -m scripts.rescene_code_first_geometry
$PYTHON -m scripts.rescene_code_first_audit train-m --device cuda:0
$PYTHON -m scripts.rescene_code_first_audit export-noaug --device cuda:0
$PYTHON -m scripts.rescene_code_first_audit evaluate-all --device cuda:0
```

按顺序运行；不得并行启动 GPU 命令。运行器使用独占锁，并将成功、失败及中断的
GPU 保留时间计入新账本。继承成本 79.98230841015994 GPU 小时，本轮上限 4 GPU
小时，累计上限仍为 192 GPU 小时。
RUN_CONFIG 保留历史方法配方；本轮标签政策以 IMPLEMENTATION_PLAN 和代码为准，
本轮成本约束以 BUDGET_CONTRACT 为准，不继承旧配方文件中的 48 小时轮次额度。

`relabel` 校验并复用 481 份历史预测，只生成新标签，R1 前向次数为零。
不使用历史 `run --resume` 代替本次修复：旧的 COMPLETE 不证明当前源码有效；
历史发布恢复只能显式调用 publish，不能跳过计算检查。
训练恢复检查标签身份和当前训练依赖；评分缓存检查完整官方指标源码、头部、物化、
输入清单、权重及 dataset spec。依赖不一致时拒绝复用，不能手改摘要绕过。

Q1/Q2 的监督资格是各官方阈值资格的并集；Q3 仅对有效 candidate×threshold
位置平均 BCE。显式歧义、未知类别和无效候选不补成负例。几何质量独立计算。
Q 修复保留历史 M 资格、分配、segment target 和权重。

本次真实审计发现 TRAIN 扩展资格涉及 1,215 个候选、62 个 M 分配位置变化，
因此执行独立 M0/M1 对照；两臂共享扩展分配与原配方，不重训 M2。
所有新头仍用 seed45、原初始化与抽样、1,500 更新；评价点固定为历史
Q1=1500、Q2=1500、Q3=1000、M0=1500、M1=1000，不重新选择。

无增强核验只导出 94 个原 CAL/SEL 输入。保留 train 模式的数据发现逻辑，
仅关闭训练增强，并逐项检查原标签一致。评价使用原六个已锁定小头和该条件自己的
B0，不使用修复后小头替代这一核验，也不据此重新选点。

`evaluate-all` 最后调用 `rescene_code_first_report`，从保存的指标状态重算 AP，
核对当前依赖、历史结果复现及训练不变量。报告脚本不会在输入不全时给出完成状态。
项目最终完成仍须主代理逐项人工核对原提示词。

已完成后的证据包生成命令：`$PYTHON -m scripts.package_rescene_code_first_audit`。
EVIDENCE_BUNDLE.zip 包含公开报告、36 份结果、72 份指标状态及原/新固定检查点，
不包含原始扫描和特征缓存；DELIVERY_MANIFEST.json 提供每个成员及整个包的 SHA256。
