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

2026-10-16 的实现验证见 `CPU_STATISTICS_ACCELERATION_VERIFICATION.json`：56 份已完成状态及各自 reference 池化指标精确相同，220 条实际 PR 曲线逐元素精确相同；23 项测试通过，包括 dry-run 不启动控制器、正常/异常退出恢复绑定及损坏状态被拒绝。该入口尚未用于接管正式控制器，完整加速 bootstrap 尚未执行。曲线局部计时不代表整套 bootstrap 的实测耗时。

## 从官方充分状态复算

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

`OFFLINE_METRIC_RECOMPUTATION.json` 记录主代理已执行的 CPU 核查范围：两个锁定 CAL 端点的 repeat0 共 20 份 long/current 状态，以及 M0、M3 已完成的 SEL/PB repeat0 共 36 份状态，总计 56 份。所有官方指标与记录值完全相同，分别从 reference 状态重新池化也完全相同。该范围不包括 M4 的 SEL/PB、其他 repeat 的独立复算及最终压缩包，不能扩大表述为全部交付已经验收。

一次真实部署回载和有限前向可使用本轮 P5 profile 的 `EXACT_PARENT_ENCODER_DELTA_COVERAGE_AND_REAL_FORWARD` 完成证据，并核对所加载原 delta 与最终公开副本的 SHA256 相同。代码中的字符串、此处加载示例和 CPU 指标复算均不能代替这份实际前向证据。旧 Native 的数值异常根因仍未解决；有限 V4 重复一致不等于全局逐张量确定性。
