# 部署、推理政策与离线复算

本页说明实际实现的加载接口及固定依赖。最终模型副本、状态压缩包和部署前向是否已经完成，分别以 `MODEL_MANIFEST.json`、`PARAMETER_AND_LOAD_AUDIT.json` 和 `PROMPT_AUDIT.md` 的执行证据为准；接口存在不代表这些步骤已经完成。

`DEPENDENCY_ACQUISITION.json` 绑定固定父 E1、编码器、taxonomy、数据和环境来源。delta 只有新增模块参数，必须与固定父任务包和指定编码器共同使用。resume 是另存的 optimizer/scheduler/RNG/进度资产，不能用 delta 替代。

## 取得模型与依赖

在本轮发布的工作副本根目录，仅下载所需 LFS 资产：

```sh
git lfs pull --include='artifacts/revisit_prompt_v1/deployment/*.pt,artifacts/revisit_prompt_v1/resume/*.pt,artifacts/revisit_prompt_v1/evidence/official_states.zip,artifacts/native_long_postselection_v1/deployment/E1-23760.pt' --exclude=
```

编码器须按原 Concerto 分发条款独立取得，并由 `CONCERTO_CHECKPOINT` 环境变量绑定其本地文件；其 SHA256 固定为 `845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07`。不下载或重新分发训练扫描、逐点 GT、编码器或本地缓存。重新推理的数据准备依照仓库 `README.md`，实际面板和历史暴露范围以 `DATA_MANIFEST.json` 为准。

父任务包的配置由 `parent_config` 解析；变体、适配种子和新增权重来自 delta。加载器会严格核对父包在 BASE 的原源码身份、父任务参数和 buffer 覆盖，以及 delta 的父/编码器/taxonomy/新增张量和训练数值源码身份。

## 保持 V4 推理政策

正式训练保持原 C3 身份；本轮 CAL/SEL/PB/P5 使用单独冻结的 `ordered-sparse-reference-v4`。只调用裸 `load_revisit_bundle` 不会自动安装该评测政策。要重现本轮推理，应像 `scripts.evaluate_revisit_prompt_sparse` 一样，在构造模型和完整前向期间保持下面三个上下文。原有 CUDA、FlashAttention 及第三方包约束仍适用。

下面是无需旧运行目录的参数加载示例。它读取 manifest 中实际存在的候选 delta，验证其实体字节和 V4 源码/运行时身份，执行严格参数加载；没有数据前向，也不代表 optimizer 恢复或科学结论。

```sh
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

from scripts import revisit_prompt_assets as model_assets
from models.revisit_prompt_numerics import install_evaluation_policy
from models.revisit_prompt_native_numerics import evaluation_builders, runtime_identity
from models.revisit_prompt_centroid_numerics import evaluation_centroids
from models.revisit_prompt_sparse_reference import evaluation_sparse_reference

public = Path('artifacts/revisit_prompt_v1')
read = lambda path: json.loads(path.read_text())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
dependencies = read(public / 'DEPENDENCY_ACQUISITION.json')
policy_path = public / 'EVALUATION_POLICY_V4.json'
assert sha(policy_path) == dependencies['environment']['runtime_policy_sha256']
policy = read(policy_path)
for registry in ('training_numeric_sources', 'policy_sources'):
    for relative, expected in policy[registry].items():
        assert sha(Path(relative)) == expected, relative
assert runtime_identity() == policy['inference_runtime']

selection = read(public / 'CAL_SELECTION_LOCK.json')
endpoint = selection.get('candidate_star') or selection['candidate_descriptive_star']
manifest = read(public / 'MODEL_MANIFEST.json')
row = next(row for row in manifest['assets'] if row['kind'] == 'delta'
           and all(row[key] == endpoint[key] for key in ('variant', 'adapter_seed', 'step')))
delta = Path(row['path'])
assert delta.stat().st_size == row['bytes'] and sha(delta) == row['sha256']
assets = dict(parent_bundle=str(Path(dependencies['parent']['path']).resolve()),
              encoder_path=str(Path(os.environ['CONCERTO_CHECKPOINT']).resolve()),
              taxonomy_path=str(Path(dependencies['taxonomy']['path']).resolve()),
              hashes={name: dependencies[name]['sha256'] for name in ('parent', 'encoder', 'taxonomy')})
model_assets.configure_runtime(9001)
install_evaluation_policy()
with evaluation_builders(), evaluation_centroids(), evaluation_sparse_reference():
    system = model_assets.load_revisit_bundle(delta, assets=assets).eval()
    # 使用合法取得的扫描执行前向时，须继续留在这些上下文内。
    print(dict(variant=endpoint['variant'], step=endpoint['step'],
               status='PARAMETERS_LOADED_NO_DATA_FORWARD'))
PY
```

原实验环境中的队列恢复使用 `python -m scripts.continue_revisit_prompt_sparse`，其内部负责收费和原锁定任务去重。它不会重新训练已完成的主轨迹。不要手工伪造 `MPR_CHARGED_EVENT_ID` 绕过费用记录。既有候选/对照的 resume 位于 manifest 的 `kind=resume` 项，恢复到原 N=400 边界；不是为了本轮追加训练。

可选的 CPU 统计恢复入口是 `python -m scripts.continue_revisit_prompt_statistics --execute`，默认或 `--dry-run` 只打印 `DRY_RUN_NOT_STARTED`。它仍调用上述原控制器，沿用其互斥锁、收费、身份核查与完成任务去重；仅在原 `bootstrap_locked` 调用期间批量计算 CPU float32 PR 阈值，退出或异常时恢复函数绑定。原 AP 积分、reference 抽样、1000 次 bootstrap、种子、三次重复和 NA 规则均保留。当前 GPU 评测运行期间不得据此启动第二个控制器。

以下日期段落保留当时状态；最终完成证据见本页末尾。

2026-10-16 的实现验证见 `CPU_STATISTICS_ACCELERATION_VERIFICATION.json`：56 份已完成状态及各自 reference 池化指标精确相同，220 条实际 PR 曲线逐元素精确相同；23 项测试通过，包括 dry-run 不启动控制器、正常/异常退出恢复绑定及损坏状态被拒绝。该入口尚未用于接管正式控制器。SEL 两个锁定端点的三轮结果齐备后，主代理在独立 CPU 进程中调用原 `bootstrap_locked`，使用同一加速上下文及 V4 命名空间；保留 seed9101、原 1000 次抽样及 NA 规则，CUDA 未初始化。该进程于 09:34:57 UTC 实际完成全部 36000 次池化，耗时 7729.626 秒；1000 次抽样均有效。`SEL_BOOTSTRAP_EXECUTION_VERIFICATION.json` 记录实际输入/源码/输出哈希，以及独立的三进程差值、全部保存抽样百分位及字节核对。8 个 reference 的少场景警示保留，长期和当前差值的区间均跨零。原 GPU 控制器继续执行 PB，PB bootstrap 尚未执行；完整 SEL 耗时不代表 PB 耗时或整套统计加速比。

## 从官方充分状态复算

2026-10-16 CPU 缓存增补见 `CPU_POOL_CACHE_VERIFICATION.json`。独立模块 `scripts/revisit_prompt_pool_cache.py` 按完整且保序的状态内容、taxonomy 字节、源码和运行环境计算缓存键；缺项调用原官方池化，损坏输入仍由原校验拒绝，损坏缓存直接报错。实际 M3/M4 PB 各 T3–T5 的 long/current 共 12 组完整池化及 12 组固定抽样池化，六项指标均精确相同；33 项相关测试通过。10:18 UTC 启动的独立 CPU 预热只计算已有 repeat0 状态的缓存，不写正式 bootstrap 输出。此时最后一个 GPU PB 评测仍在运行，正式控制器没有接管。后续正式统计仍须读取全部六份实际进程结果并执行原 1000 次抽样，不能用缓存预热替代第三次进程确认。

2026-10-16 11:10 UTC：全部 93 个正式 CAL/SEL/PB 评测实际完成；M4 PB 三轮对应指标、轨迹及状态字节一致，见 `SPARSE_PB_M4_VERIFICATION.json` 与 `FORMAL_EVALUATION_COMPLETION_VERIFICATION.json`。独立 CPU 模块 `scripts/revisit_prompt_cooccurrence.py` 读取 18 份实际 SEL/PB 结果和同一 native loader 的最终当前 GT，保留原一对一匹配，仅在诊断中排除未知/歧义身份，按同类的不同实体计数；不运行模型、不改主 AP 或选点。`SAME_CLASS_DIAGNOSTICS.json` 保存 627 个输入、1881 行逐进程诊断与六组汇总，零分母为 null；对应 VERIFICATION 核对全部计数及两份实际 GT 抽查。它是条件召回描述，不是 AP、独立新场景统计或干扰因果证明。默认 `python -m scripts.revisit_prompt_cooccurrence --dry-run` 不加载数据；本轮 `--execute` 已真实完成，现有结果应保留，不重复执行覆盖。

11:12 UTC：预热已完成 1000 个原抽样序列的 12000 次逻辑池化，11916 项缓存，耗时 3181.765 秒，CUDA 未初始化。原控制器只有在全部 93 次 GPU 评测结束且无活动子进程/reservation 后才安全交接；`CPU_CONTROLLER_HANDOFF_VERIFICATION.json` 保留实际新旧 PID、预期中断、来源哈希和守卫范围。新控制器使用相同原环境，读取全部六份实际 PB 结果执行完整 1000 次原 bootstrap，之后由原控制器按原预算锁运行 P5。缓存只复用同一保序完整输入，不改变三重复、抽样、NA 或官方指标。此时 PB bootstrap 实际运行中，最终绑定恢复与区间核验尚未完成。

复算已保存的 AP 不需要模型、编码器、扫描或 GPU，但需要原 evaluator 及 taxonomy 环境。下例读取实际压缩包中的锁定候选 CAL T5 长期状态，调用与 bootstrap 相同的官方池化接口；AP 单位为 0–1。

```sh
CUDA_VISIBLE_DEVICES='' python - <<'PY'
import json
from pathlib import Path
from zipfile import ZipFile

from scripts.evaluate_revisit_prompt import pool_official_states

public = Path('artifacts/revisit_prompt_v1')
selection = json.loads((public / 'CAL_SELECTION_LOCK.json').read_text())
endpoint = selection.get('candidate_star') or selection['candidate_descriptive_star']
prefix = f"{endpoint['variant']}-s{endpoint['adapter_seed']}-u{endpoint['step']:06d}/CAL/repeat0"
with ZipFile(public / 'evidence/official_states.zip') as archive:
    state = json.loads(archive.read(prefix + '/official-long-T5.json'))
print(pool_official_states([state], dataset_spec='artifacts/native_long_postselection_v1/deployment/rio.yaml'))
PY
```

`OFFLINE_METRIC_RECOMPUTATION.json` 记录主代理已执行的 CPU 核查范围：两个锁定 CAL 端点的 repeat0 共 20 份 long/current 状态，以及 M0、M3 已完成的 SEL/PB repeat0 共 36 份状态，总计 56 份。所有官方指标与记录值完全相同，分别从 reference 状态重新池化也完全相同。`SPARSE_SEL_M4_VERIFICATION.json` 的 `offline_metric_recomputation` 另记录 M4-200 SEL repeat0 的 10 份总体状态及 68 份 reference 状态；使用原标量官方实现，指标也完全相同，CUDA 未初始化。`SPARSE_PB_M4_VERIFICATION.json` 的同名字段另记录 M4-200 PB repeat0 的 8 份总体状态及 48 份 reference 状态，全部六项指标与记录及 reference 池化精确相同；原标量官方实现实际耗时 166.375 秒，CUDA 未初始化。因此已有 74 份总体状态完成独立 CPU 核查，但尚未覆盖其他 repeat 的独立复算及最终压缩包，不能扩大表述为全部交付已经验收。

一次真实部署回载和有限前向可使用本轮 P5 profile 的 `EXACT_PARENT_ENCODER_DELTA_COVERAGE_AND_REAL_FORWARD` 完成证据，并核对所加载原 delta 与最终公开副本的 SHA256 相同。代码中的字符串、此处加载示例和 CPU 指标复算均不能代替这份实际前向证据。旧 Native 的数值异常根因仍未解决；有限 V4 重复一致不等于全局逐张量确定性。

## 2026-10-16 最终完成证据

PB 已实际完成原 1000 次成对 bootstrap，1000 有效/0 NA，全部 36000 次逻辑池化精确命中完整保序缓存；读取六份真实进程结果，函数绑定最终恢复，CPU 未初始化 CUDA。PB_BOOTSTRAP_EXECUTION_VERIFICATION 与 PB_BOOTSTRAP_INDEPENDENT_VERIFICATION 记录来源/输出身份、保存抽样及原百分位区间核对；后者不冒称第二轮 1000 次池化。SEL/PB 四个区间均跨零，少场景警示保留。

P5 实际四项作业均完成，六项内容干预 CONTROLLED，三个端点各一次 warmup/三次 measured pass。P5_EXECUTION_VERIFICATION 逐字段核对 MECHANISM/PROFILE 和实际收费；M3-400、M4-200 的严格父/编码器/delta 覆盖及真实前向与公开 delta 同 SHA，已满足本轮有限部署核验，不需重新加载。内容敏感性不能证明 AP 收益。正式控制器已正常退出，ACTIVE_JOB/reservation 均释放。

MODEL_MANIFEST 列出六份最佳 delta、两份 N=400 resume 及充分状态 ZIP；FINAL_DELIVERY_VERIFICATION 已核对所有实体字节，以及 ZIP 内全部 6156 份状态与原件一致。这里的字节核对不扩大原 74 份总体状态独立 CPU 指标复算的范围。原指令 §19 的最终内容审核见 PROMPT_AUDIT，科学状态仍 SCREEN_NO_GO；实际 Git/LFS/Release 发布事实由外部回执负责。
