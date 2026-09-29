# ReScene 单模块实验：先代码、后方法的专项审查

日期：2026-09-29  
审查对象：`Orangekostar/Persist4D@4bf00902c9428795f7f47547bf462540aa304cc6`。

## 1. 结论和执行范围

当前不能把空确认名单直接归因为论文思想或方法类别无效。审查发现：

1. **质量监督存在跨 IoU 阈值的有效性缺口**：训练用 `ignore > min(thresholds)` 排除整个候选；官方评价却在每个阈值按 `p_ignore <= threshold` 决定未匹配预测是否计入。候选在0.5被忽略，并不意味着它在0.75或0.9也被忽略。这是明确的监督—评价接口不一致；它也出现在上轮执行规范里，不应单方面归咎于执行者。
2. **恢复/缓存有确定的失效判据缺口**：`run --resume` 可凭旧COMPLETE记录跳过导出，而不检查当前源码；评分缓存只检查若干wrapper源码、旧cache identity和权重，存在辅助函数修后仍读取旧结果的可达路径。尚无证据证明这次正式结果已经因此错误。
3. **CAL/SEL的数据集处于train模式**，实际会进入训练增强分支；父模型eval并不会关闭独立数据集的增强。这是已记录的继承设置，不是偷偷改变协议，也不能单独解释新模块相对B0的退化；但它限制了“标准无增强原生验证”的解释。
4. **没有在所查主链发现**Q/M交叉修改输出、T1偷读第二扫描、二次softmax、M训练使用二值化物化导致断梯度、Q3硬cummin断梯度等错误。这里是源代码与最小计算检查的结论，不是完整服务器运行证明。

没有重新运行R1、读取真实服务器soft缓存或加载已训练的小头；没有修改/推送用户仓库。没有计算“修复后涨点”。本地执行的是CPU最小反例和头部方程检查；另附可在原服务器环境调用真实仓库/官方evaluator的探针，后者尚未执行。

## 2. 实际调用链

```
short_module_screen.build_datasets
  -> NativeSession.produce
  -> dataset.load_scan_indices / _load_scan_sequence
  -> Pointcept collator -> frozen R1
  -> extract_official_task_prediction(return_soft_evidence=True)
  -> 原top-k/filter一次 + source_query/class/retained_index
  -> quality_labels / geometry_assignment / low_segment_targets
  -> train_arm: QualityHead 或 MaskHead
  -> apply_module
  -> OfficialMetricAccumulator
  -> CAL lock -> SEL固定点 -> seed46固定点
```

主要源：`scripts/short_module_native.py`、`scripts/short_module_data.py`、`models/short_module_heads.py`、`scripts/short_module_training.py`、`scripts/short_module_evaluation.py`、`scripts/rescene_task_postprocess.py`、`datasets/semseg.py`、`datasets/pointcept_utils.py`；配置和历史修订见RUN_CONFIG、CODE_BINDINGS、REGULARIZATION_FIX。

## 3. 发现一：跨阈值ignore过滤错误地被压成单一有效性位

### 3.1 已确认代码行为

`short_module_data.py::quality_labels`：

```python
ignore[i] = evaluator._proportion_ignore(pred, *params)
if ignore[i] > min(thresholds):
    status[i] = "IGNORE"
    continue
```

Q1/Q2/Q3随后共同使用候选级 `valid`；Q3的全部阈值输出也由同一候选级valid控制。推理对所有保留候选输出新分数，并不保留被排除候选的原分数。

官方 `stmetrics/instances/evaluator.py::Evaluator._match` 在匹配失败时：

```python
pred_valid[pri] = p_ignore <= overlap_th
```

主AP阈值实际为0.50、0.55、…、0.90，严格大于判断。

### 3.2 最小反例

每阶段GT占100点；一个错误候选包含这100点再加150个void点。两阶段均这样构造，所有候选都满足100点最小区域条件。

- 每阶段IoU=100/250=0.4。
- ignore比例=150/250=0.6。
- 当前训练：0.6>0.5，整个候选退出Q监督。
- 官方阈值0.5：未匹配候选被忽略。
- 官方阈值0.75：未匹配候选计作FP。
- 将该候选从低分提升到正确候选之前，会改变高阈值的TP/FP排序。

CPU最小反例已执行，见`minimal_test_results.json`。这是对源码判据的复现，不是全数据集损失归因。

### 3.3 影响范围与不确定性

不能把17,111条TRAIN unknown/ignore全部当成本缺口的受影响项；该类别还混有完全void、空候选、无效类别、显式歧义等。需要从已有缓存的`ignore_proportion/status`字段计数 `(0.5,0.9]` 范围，再查看新头是否给这些候选升分。当前公开汇总不提供这一完整交叉表。

也不能据此宣称Q的约10–13.5pp退化全由此造成；需要固定输入、权重父模型和训练配方，修正标签后重跑受影响Q臂才能量化。

### 3.4 修复边界

- 分开保存：候选几何质量、每阈值监督有效性、显式unknown/ambiguity、全阈值均不计分的候选。
- Q3采用candidate×threshold有效掩码，而不是只用candidate valid。
- 几何标签应在有效、类别相容的GT上计算，与当前分数和score排序匹配无关。
- 不把真实unknown/明确歧义强行补成负例；按官方各阈值是否参与计分明确处理。
- Q1/Q2的标量监督需明确覆盖哪些候选，不能以“0.5会忽略”推出“所有阈值都可自由改分但不监督”。
- 测试至少包含r_ignore=0、0.6、1三类，以及正确出现/消失；r_ignore=0.6是现有全void测试没有覆盖的情况。

### 3.5 几何标签的连带依赖

`geometry_assignment` 直接取 `labels['valid']` 作为候选资格。因此修改Q有效性会连带改变M训练分配。质量监督资格和形状监督资格必须显式解耦。先核对受影响数量；在量化Q修复时，保持旧M分配不动，不能把两处改动一起算成单一修复收益。

## 4. 发现二：恢复和评测缓存不能充分感知依赖变化

### 4.1 直接证据

- `run_pipeline --resume` 看到EXPORT_INDEX COMPLETE可直接跳过export，不重新比较当前producer/label源码与旧cache身份。
- 若存在PUBLISH_STATE，run --resume直接退化为只跑publish。
- `evaluate_point` 的源码摘要只包含evaluate_point、load_head、materialization_system三项；旧结果命中条件检查旧index identity与head hash。更改apply_module或labelhelper而未使index重新生成，存在返回旧结果的路径。

### 4.2 定性

这是确认的恢复/失效风险，**不是已证明本轮缓存污染**。本轮正则修补有单独清理与重训记录，不能因为系统有风险就否定其已记录重跑。

### 4.3 修复

- 预测缓存身份和标签缓存身份拆开。只改标签无需重新执行全部R1前向。
- 恢复时读取当前直接依赖digest，与持久化身份比较；COMPLETE仅表示完成，不表示当前有效。
- 结果摘要涵盖真实apply_module、头部实现、materializer、metric版本、dataset spec和输入/头参数身份。
- 诊断修复使用新运行目录/新版本，不覆盖旧CAL_LOCK、tag或原报告；不将publish-only恢复误当成计算重跑。

## 5. 发现三：CAL/SEL仍进入训练增强

RUN_CONFIG.json明确记录dataset_horizon 2和5均为`train`。NativeSession调用load_scan_indices，没有关闭增强参数；semseg的`if 'train' in self.mode`包含随机中心/平移、翻转、弹性形变和volume/image augmentation。

父R1处于eval且固定seed，是另一层状态，不能关闭这个分支。固定seed只是固定一次增强。

这沿用了既有开发数据路径，CODE_BINDINGS已披露；所以不能当成Codex违背规范的独立错误，也不是当前模块差值必然无效的证明。可以保留旧结果为“固定增强开发筛选”。但不能把它直接等同于原生无训练增强验证、或与LOCAL论文结果直接比较。

下一步先在同一批固定扫描、同样的父权重与已选小头上，增加无训练增强的只读核验。只关闭增强，不换split/reference、标签或序列顺序；不要简单改成会改变数据发现和target格式的test模式。两种模式各自有自己的B0。先不重新挑选小头checkpoint。

## 6. 排查后未发现的几类故障

| 检查 | 已见证据 | 范围 |
|---|---|---|
| T1使用了第二扫描/T2使用了第三扫描 | NativeSession按scan_indices读取并记录真实np.load路径 | 源码及既有记录，不是本地重跑原始数据 |
| 双重softmax或候选列错位 | fresh_output独立字典；原postprocess一次；query/class/index谱系保留 | 源码确认 |
| mask物化切断训练梯度 | train_arm对float logits计算loss，materialize只在评价调用 | 源码确认 |
| M零初始化永远不学习 | 最小CPU计算：输出层首步有梯度，trunk第二步有梯度 | 仅头方程，不是训练后checkpoint重载 |
| Q3硬单调操作断梯度 | softplus累计负间隔，CPU检查27/27元素有非零有限梯度 | 仅数值机制 |
| Q改变mask/M改变score | apply_module与evaluate_point有独立检查 | 源码与报告一致 |
| 仍用错误full-partition给low-segment目标编号 | low_point2segment[voxel_inverse]构造soft target；已核对实现 | 源码确认 |
| 将拼接IoU当temporal IoU | Q1/Q2各自标签，官方_select_overlap用于Q2/Q3 | 主路径已区分；ignore问题另见上文 |

## 7. 属于方法/配方、不能冒充代码bug的部分

1. Q1/Q2直接替换原分数，而不是残差校准，是原规范的设计选择。不能在负结果之后说“Codex漏乘原分数”。
2. M残差固定在±2；强于2的同号logit不能被反号，是设计的表达范围，不是autograd错误。
3. M1在所有采样candidate位置平均loss，只有配对候选接受分割监督；日志约2223/24000位置有形状监督。这是配方选择，不是梯度丢失的证据。改变正例抽样/归一化属于另一个实验。
4. mean pooling特征与raw阶段差值的尺度选择需要诊断，但未看到真实张量统计前不能断言LayerNorm或尺度就是主要根因。
5. 旧正则缺漏已在锁定前修复重训，不能重复把旧bug当作当前失败原因。

## 8. 最小闭环，而不是重开大型实验

1. CPU读取现有缓存，计数ignore跨阈值缺口以及受其影响的M分配资格；同时确认实际安装的evaluator源码身份。
2. 补一个部分void跨阈值回归；修复label资格表达与缓存失效。不要扩成全库测试。
3. 保持父R1、原输入、训练步数与抽样范围，重新计算受影响Q标签并重训Q臂；分别报告修复前后同输入差值。M的分配不要被隐式改动。
4. 用原锁定的小头做无训练增强开发核验，作为独立输入条件比较，不用它再次大网格选点。
5. 若M资格确受影响，再对M0/M1同分配同配方补必要重跑；没有受影响则不为“全跑一遍”重训。
6. 只有这些核验后仍未改善，才把失败更可靠地归因于已测方法/表达/监督，而不是继续猜测修复能涨点。

## 9. 交付与可信边界

- `minimal_counterexamples.py`：已在本地torch 2.10.0 CPU运行，全部断言通过；是按已读源码抽出的判据/公式，不是完整原仓库导入测试。
- `minimal_test_results.json`：实际输出。
- `probe_repository_ignore_gap.py`：已语法检查；用于原服务器导入真实quality_labels与官方evaluator，并可读取现有缓存计数。**未在用户服务器执行**。
- 本次未完成真实影响人数统计、修复后训练或AP重算；这是尚待量化的效应，不是省略已可读取的Git事实。

## 10. 关键源码索引

根地址：`https://github.com/Orangekostar/Persist4D/blob/4bf00902c9428795f7f47547bf462540aa304cc6/`

- `scripts/short_module_data.py`：quality_labels、geometry_assignment、low_segment_targets、geometry_loss。
- `scripts/short_module_training.py`：valid掩码、可微loss与保存/恢复。
- `scripts/short_module_native.py`：NativeSession与读文件、soft输出。
- `models/short_module_heads.py`：QualityHead、MaskHead、apply_module。
- `scripts/short_module_screen.py`：build_datasets、export身份、run_pipeline恢复。
- `scripts/short_module_evaluation.py`：evaluate_point与缓存复用。
- `scripts/rescene_task_postprocess.py`：一次postprocess、soft谱系与零残差验证。
- `datasets/semseg.py`、`datasets/pointcept_utils.py`：增强和分区。
- `tests/test_short_module_data.py`：已存在测试范围。
- `artifacts/short_module_screen_v1/RUN_CONFIG.json`、`OFFICIAL_THRESHOLDS.json`、`CODE_BINDINGS.md`、`recovery/REGULARIZATION_FIX.md`。

官方evaluator已读取blob `6c558fad20ecc033c1763864732eb98f2902108e`，matcher blob `d08c1746c3d97f89ac5bebe657b3a17112d3737f`：
`https://github.com/GradientSpaces/stmetrics/blob/main/stmetrics/instances/evaluator.py`
`https://github.com/GradientSpaces/stmetrics/blob/main/stmetrics/instances/matcher.py`

探针会报告服务器实际安装文件的Git blob SHA；不会自动升级依赖。
