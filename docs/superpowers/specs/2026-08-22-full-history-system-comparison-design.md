# Full-History vs Persistent-State System Comparison Design

## Goal

在相同 ReScene4D backbone、checkpoint、Protocol-B master prefixes 和三种预注册 order 下，比较：

- ReScene4D Full-History (Frozen T2 Checkpoint)：每次更新重新处理 `S1:St`；
- Persist4D Persistent-State：每次只处理 `(S[t-1], St)`，并更新冻结的 P6-A single-state memory。

实验同时评价 causal-prefix task quality、deployment identity、gap identity recovery、per-update/cumulative compute、峰值显存和历史状态规模。完成 `SYSTEM_COMPARISON_GO_NOGO_REPORT.md` 后停止，不训练或接入新模块。

## Frozen Boundary

本阶段不修改 ReScene backbone、decoder、checkpoint、query feature、mask/class prediction、Persist4D memory representation、association/update/birth/dormant/reactivation rule 或 capacity。允许新增的代码仅限配置绑定、评估适配、full-history inference driver、指标、profiling、统计、可视化与报告。

Incumbent 固定为 P6-A `B4/frozen_p5_persist4d`：capacity 100、association threshold 0.5、class weight 0.25、background class 18、update rate 0.2、confidence threshold 0.5、mask threshold 0.5、minimum mask support 1。任何绑定或回归不一致均 fail closed，并阻止 full-history 大实验。

## Components

1. `configs/system_comparison/persist4d_incumbent.yaml`
   记录 P6-A incumbent 参数以及 config、checkpoint、source commit、P6-A reference artifact 的 SHA256 绑定。

2. `scripts/system_comparison_protocol.py`
   从已跟踪的 P6-A Protocol-B manifest 构造只读 system-comparison manifest，验证 43 masters、6 clusters、3 orders、T2-T5 exact prefixes，不重新采样。

3. `scripts/run_system_comparison.py`
   复用现有 dataset/collation/model checkpoint loader。Full-History 对每个 prefix 直接执行一个 `R(S1:St)`；Persist4D 对同一 master/order 依次执行局部 T2 observation 和冻结 B4 tracker。预测缓存采用内容寻址和 provenance 校验，不能覆盖不匹配缓存。

4. `scripts/system_comparison_metrics.py`
   将 task quality 与 deployment identity 分开。task evaluator 每个 horizon 只接收当前 prefix；identity evaluator 比较相邻更新已发布 ID；gap evaluator只统计 `visible -> absent >= 1 update -> visible`。ReScene 使用其实际 query namespace，不附加 Persist4D memory。

5. `scripts/profile_system_comparison.py`
   用 5 次 warmup、10 次测量、CUDA synchronize 和 reset peak memory 记录 forward/update latency、allocated/reserved VRAM、point/input volume 与 Persist4D tensor-storage bytes。数据加载和 host-to-device 不计入模型 latency。

6. `scripts/system_comparison_analysis.py` 与 `scripts/system_comparison_figures.py`
   生成 per-sequence/per-order/aggregate 结果、paired reference-scene cluster bootstrap、LOSO、累计成本、六张规定图和两张主表。未知或未运行结果不得填造。

7. `scripts/build_system_comparison_artifacts.py`
   校验完整 artifact schema、输入 hash、覆盖范围和统计单位，并只在全部必需证据齐备后发布最终报告。

## Data Flow And Gates

1. 校验 worktree source commit、checkpoint SHA256、P6-A config/artifact hash，重算 incumbent 指标并与冻结 P6-A reference 在数值容差内比较。
2. 审计 dataset temporal construction、collation、model forward、query initialization、temporal sharing、prediction head、ID namespace、evaluator、determinism 与 change-label 路径，写入代码级证据。
3. 先运行 T2 regression 和同一 S1:S5 三次 determinism smoke；失败则停止，不进入全量实验。
4. 运行 metric toy cases 和 no-future-leakage 测试；语义不明确则停止，不输出替代数字。
5. 在统一 smoke subset 上验证 full-history、persistent-state 和 profiler。
6. 运行 43 masters x 3 orders x T2-T5，随后统计、LOSO、图表和报告。
7. 仅当 Full-History 在 T4/T5 有实质 accuracy advantage 时运行已有 Oracle attribution；不得自动实现后续方法。

## Identity Semantics

Full-History 的一个 prefix forward 内，query index 可作为该次 forward 的 system-issued ID；跨 prefix 稳定性必须由实际 query namespace 的一致性来测量。若相同输入的 query IDs 不确定，先固定 RNG 与 deterministic settings；仍不稳定时，raw query index 不得作为 deployment identity，报告 metric semantic gap 并停止该指标。

Persist4D 使用 sequence-scoped persistent entity ID。系统级统一术语为 Gap Identity Recovery；Persist4D 的 dormant reactivation 仅作为额外机制诊断。

## Failure Handling

- 缺失或不匹配的 checkpoint/config/manifest/cache：fail closed，禁止继续。
- T>2 代码路径不受支持：记录准确代码证据，不修改模型语义。
- T2-trained checkpoint 在 T3-T5 的结果必须标为 zero-shot temporal-horizon extension。
- evaluator future leakage、identity namespace 不公平或 timing 污染：先以单元测试修复 adapter/profiler，再运行实验。
- 显存不足或单序列异常：保留失败记录，不删除 horizon/order；只调整评估执行资源，不调整方法。

## Verification

至少覆盖：common-prefix integrity、full-history T2 regression、三次 determinism、no-future-leakage、deployment ID-switch toy case、gap recovery toy case、CUDA timing synchronization、incumbent artifact regression。最终额外执行全量单元测试、ruff/compile checks、artifact schema/hash verifier，并逐项审计监督提示词的 19 步交付物。

## Decision Rule

最终只允许 `SYSTEM_LOCK`、`SYSTEM_PARETO_LOCK`、`ASSOCIATION_LIMITED` 或 `REPRESENTATION_LIMITED`。结论由真实 task/identity/gap/compute 证据和必要时的 Oracle attribution 决定，不以“Persist4D 必须全面获胜”为目标。
