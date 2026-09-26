# V2 交接

固定起点：`8b5e93795817682fe70864daa545db219e1443c9`；当前代码：`d58420568f46070a4b3a7dcee2b4e47fc42f4c7b`；分支：`research/persist4d-perception-gain-v2`。

实际实验代码见每个评价的 `execution_provenance` 与 `EXECUTION_LOG.jsonl`。资产 SHA/角色见 `INPUT_MANIFEST.json`、`DATA_ROLES.json`；本机绑定为 `$PERSIST4D_GAIN_V2_ROOT/assets.local.json`。源码映射见 `CODE_BINDINGS.md`。

当前执行：`COMPLETE`；候选：`R1-D0`；默认部署：`R1-D0`。正式结果、每臂步数和失效范围见 `FINAL_REPORT.md` 与各原始 JSON/CSV。

## 已执行命令

命令逐条来自真实日志：`tables/EXECUTED_COMMANDS.csv`。恢复前保留失败结果，并先解决对应 reason；控制器拒绝相同依赖下盲目重试。

```bash
export PERSIST4D_GAIN_V2_ROOT=/home/ww/persist4d_runs/perception_gain_v2
cd /home/ww/paper5/.worktrees/persist4d-perception-gain-v2
conda run -n persist4d python -m scripts.perception_gain_v2 status
conda run -n persist4d python -m scripts.perception_gain_v2 report
```

已有控制器运行时不启动另一控制器；待其退出后，按下表恢复。标为 CODE_HANDLER_PENDING 的任务暂不能运行。

## 检查与待完成证据

{
  "final_test_record": {
    "schema_version": "perception-gain-v2-final-tests-v1",
    "source_commit": "d58420568f46070a4b3a7dcee2b4e47fc42f4c7b",
    "status": "PASS_WITH_DOCUMENTED_WARNINGS",
    "full_related_regression": {
      "tested_source_commit": "b4d6a49",
      "command": "CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 1800 /home/ww/miniconda3/envs/persist4d/bin/python -m pytest tests/test_perception_gain*.py tests/test_local_diagnostic_loss.py tests/test_task_memory_evaluation.py -q",
      "exit_code": 0,
      "passed": 202,
      "skipped": 0,
      "warnings": 2,
      "reported_wall_seconds": 8.13,
      "log": "validation/full-related-regression.txt",
      "log_sha256": "0577988a47783edf2adc309be241b57913d26304730a9498c48b3cb2f74efa71"
    },
    "post_regression_fix": {
      "reason": "Final packaging exposed a source-unbound LOCAL prediction-cache collision; added a regression and source/input cache namespace. Report adds protocol-named selection summaries.",
      "command": "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/ww/miniconda3/envs/persist4d/bin/python -m pytest tests/test_perception_gain_v2_predictions.py tests/test_perception_gain_v2_confirmation.py tests/test_perception_gain_v2_report.py -q",
      "exit_code": 0,
      "passed": 22,
      "reported_wall_seconds": 2,
      "red_evidence": "New source/input-identity test failed with unexpected execution_binding argument before the implementation; passed after implementation.",
      "report_smoke": "Actual report CLI exit 0; external:tasks/report-delivery-smoke.log"
    },
    "runtime_namespace_followup": {
      "reason": "Recovered export environment must match the original confirmation; cache keys now distinguish numerical runtime environment as well as source/input identity.",
      "command": "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/ww/miniconda3/envs/persist4d/bin/python -m pytest tests/test_perception_gain_v2_predictions.py tests/test_perception_gain_v2_confirmation.py -q",
      "passed": 13,
      "reported_wall_seconds": 1.98,
      "exit_code": 0,
      "red_evidence": "Different cuBLAS workspace configurations mapped to the same cache directory before the fix; the regression failed before and passed after adding runtime_environment."
    },
    "partial_report_followup": {
      "reason": "Keep the new selection-summary table usable when a partial perception result has selected=null.",
      "command": "OMP_NUM_THREADS=1 /home/ww/miniconda3/envs/persist4d/bin/python -m pytest tests/test_perception_gain_v2_report.py -q",
      "passed": 9,
      "reported_wall_seconds": 0.06,
      "exit_code": 0
    },
    "lint": {
      "status": "PASS",
      "scope": "scripts/perception_gain_v2*.py tests/test_perception_gain_v2*.py plus train_perception_gain, local evaluator, task-memory evaluator and their related tests; all changed source files checked after latest edit"
    },
    "diff_check": "PASS",
    "warnings": [
      "Existing scipy.ndimage.filters deprecation in albumentations",
      "Existing torch.tensor(tensor) warning in models/criterion.py"
    ],
    "limitations": [
      "This is the task-related regression, not the entire repository test suite.",
      "Scientific execution identities precede delivery-only source cleanup; raw experimental provenance is preserved.",
      "Unit tests do not prove final prediction-export coverage or remote publication; those require the separate live recovery audit and publication receipt."
    ]
  },
  "requirement_by_requirement_audit": {
    "schema_version": "perception-gain-v2-requirement-audit-v1",
    "status": "SCIENTIFIC_AND_LOCAL_ASSET_REVIEW_COMPLETE_PUBLICATION_PENDING",
    "specification": "docs/0921/Persist4D_Gain_V2/Persist4D_Codex_Gain_V2_FINAL.md",
    "specification_sha256": "9b92e302d0af269614e86a76591cd324374f83fe64b097aa4c2ffdb330c3b688",
    "reviewed_source_commit": "d58420568f46070a4b3a7dcee2b4e47fc42f4c7b",
    "items": [
      {
        "question": 1,
        "status": "VERIFIED_WITH_LIMITATION",
        "conclusion": "固定父提交是当前分支祖先；相对父提交 artifacts/perception_gain_v1 无改动。实际执行 commit/source digest 保存在每次执行记录，交付源码版本另列。",
        "evidence": [
          "EXECUTION_LOG.jsonl",
          "RUN_STATE.json",
          "CODE_BINDINGS.md"
        ],
        "limitation": "历史预测漂移原因未查明；不把新的 baseline 数值冒充 V1 原数值。"
      },
      {
        "question": 2,
        "status": "VERIFIED",
        "conclusion": "H/L、主模型、scorer500、NEW/PAIR1500 分开。S-BAL-H 从保存的350恢复到750，379是未保存观察进度。C0-L/Q-SEM-L到3000，CAL选择250。",
        "evidence": [
          "tables/TRAINING.csv",
          "training/C0-L-s45/FULL_ENDPOINT_AUDIT.json",
          "training/Q-SEM-L-s45/FULL_ENDPOINT_AUDIT.json",
          "refiner/R1/NEW/TRAINING.json",
          "refiner/R1/PAIR/TRAINING.json"
        ],
        "limitation": "未选LR和seed46配置存在不表示实际训练。"
      },
      {
        "question": 3,
        "status": "VERIFIED",
        "conclusion": "LOW_PAIR与REPAIR_R1独立完成；依赖闭包回归保护HIGH/scorer失败不阻断这两条路径。",
        "evidence": [
          "RUN_STATE.json",
          "EXECUTION_LOG.jsonl",
          "repository:tests/test_perception_gain_v2.py"
        ],
        "limitation": "测试路径相对仓库根为 tests/test_perception_gain_v2.py。"
      },
      {
        "question": 4,
        "status": "VERIFIED_WITH_LIMITATION",
        "conclusion": "C0-L与新L臂同runtime/sample plan；NEW/PAIR同候选、相同初始化SHA、同1500步。完整CAL/SEL使用更正后的同一父cohort。",
        "evidence": [
          "refiner/R1/NEW/TRAINING.json",
          "refiner/R1/PAIR/TRAINING.json",
          "foundation/BASELINE_SUPERSESSION.json",
          "training/C0-L-s45/FULL_ENDPOINT_AUDIT.json",
          "training/Q-SEM-L-s45/FULL_ENDPOINT_AUDIT.json"
        ],
        "limitation": "不将历史不可复现数值作为当前因果控制。"
      },
      {
        "question": 5,
        "status": "VERIFIED",
        "conclusion": "两头共享5220条训练候选、32 references、64 episodes及source_shards_hash；input_mode进入checkpoint/eval/profile/bundle。最终未部署修复头，所以最终bundle的refiner=null正确。",
        "evidence": [
          "refinement/PAIR_COVERAGE.csv",
          "refiner/R1/NEW/TRAINING.json",
          "refiner/R1/PAIR/TRAINING.json",
          "publication/bundles/R1-D0.json",
          "validation/FINAL_TESTS.json"
        ],
        "limitation": "最终无修复器，不把无修复器的profile当成已部署修复收益。"
      },
      {
        "question": 6,
        "status": "VERIFIED",
        "conclusion": "真实postprocess zero NEW/PAIR与parent输出SHA一致；置换vertex/segment及state/score/class不变由相关回归覆盖，完整CAL的0步与parent一致。",
        "evidence": [
          "foundation/SMOKE.json",
          "refiner/R1/CAL.json",
          "refiner/R1/SEL.json",
          "validation/FINAL_TESTS.json"
        ],
        "limitation": "smoke不替代完整CAL/SEL；完整结果另存。"
      },
      {
        "question": 7,
        "status": "VERIFIED_WITH_LIMITATION",
        "conclusion": "CAL每T23/23、SEL每T24/24，D0与FH各四T。FINAL_LOCK在首轮PB之前；PB完整129/516，名称bug恢复未改锁。",
        "evidence": [
          "foundation/BASELINE_COVERAGE.json",
          "selection/FINAL_LOCK.json",
          "confirmation/PB_RECOVERY_AUDIT.json",
          "EXECUTION_LOG.jsonl"
        ],
        "limitation": "baseline漂移原因UNDETERMINED；原始结果保留于foundation/original-runtime。"
      },
      {
        "question": 8,
        "status": "NOT_APPLICABLE_AS_SPECIFIED",
        "conclusion": "P=R1，无换父模型，不重复训练P修复器；12配置关联候选独立CAL/SEL，未与感知/修复做笛卡尔积。",
        "evidence": [
          "refiner/P/SELECTION.json",
          "association/SELECTION.json",
          "selection/FINAL_LOCK.json"
        ],
        "limitation": "没有新P，因此不能宣称已实际执行新P重训；实现路径由回归覆盖。"
      },
      {
        "question": 9,
        "status": "VERIFIED_WITH_LIMITATION",
        "conclusion": "账本包含V1既有费用和失败/阻塞占卡；确认保护预算、正式分配和费用独立记录。两次LOCAL导出尝试均已结算，正式补导出0.512503 GPUh，先前中断0.122002 GPUh；已记录的两次CUDA可见回归按pytest报告墙钟另记一次。",
        "evidence": [
          "budget/LEDGER.jsonl",
          "budget/ALLOCATION_EVENTS.jsonl",
          "budget/CONFIRMATION_CURRENT_ALLOCATION.json",
          "budget/ASSOCIATION_CPU.jsonl"
        ],
        "limitation": "V1费用含明示估计；pytest计费不含解释器启动，未声称CUDA kernel计时。资源上限的运行配置/抽查不证明历史每一瞬间的RAM峰值。"
      },
      {
        "question": 10,
        "status": "VERIFIED",
        "conclusion": "质量取official pooled rows；PB/ADDITIONAL/LOCAL分开，按原始完整分母；LOCAL实际46 references更正在锁前。缺失对照比较为null，完整负结果为false。",
        "evidence": [
          "confirmation/COVERAGE.csv",
          "confirmation/METRICS.csv",
          "confirmation/BY_REFERENCE.csv",
          "foundation/LOCAL_POPULATION_AUDIT.json",
          "validation/FINAL_TESTS.json"
        ],
        "limitation": "没有实现bootstrap置信区间，不报告虚构区间。"
      },
      {
        "question": 11,
        "status": "LOCAL_ASSETS_VERIFIED_PUBLICATION_TRANSACTION_PENDING",
        "conclusion": "真实panel上FH/D0/C0-L bundle重载输出一致，所需模型包非空。六组LOCAL各154条已完整补导出，所有原指标精确复现；全部规定人口的预测包已准备，无未满足的本地资产要求。按协议先冻结A/B再实际推送，快照不预写VERIFIED。",
        "evidence": [
          "resources/PROFILE_SUMMARY.json",
          "publication/ASSET_PREPARATION.json",
          "publication/bundles/C0-L-s45.json",
          "confirmation/LOCAL_EXPORT_MATCHED_RUNTIME_AUDIT.json",
          "publication/PREDICTION_ASSETS.json"
        ],
        "limitation": "最终以external:publication/PUBLICATION_RECEIPT.json为准；无现有Release权限时允许CODE_ONLY并列出缺失资产。"
      },
      {
        "question": 12,
        "status": "VERIFIED",
        "conclusion": "报告区分未晋级、未训练、NOT_APPLICABLE、运行错误、一般修复收益和历史证据收益。科学结论为KEEP_R1/KEEP_PARENT/KEEP_D0，无全T超过FH。",
        "evidence": [
          "REPORT_STATUS.json",
          "validation/FINAL_INTERPRETATION.md",
          "selection/SEL_COMPONENTS.csv",
          "confirmation/CONFIRMATION_SUMMARY.json"
        ],
        "limitation": "科学评分和本地导出完成不等于外部发布完成；最终以单独发布回执判定实际可访问性。"
      }
    ],
    "pending": [
      "Actual Git branch/tag publication, existing-auth Release attempt and verified external receipt under the specified A/B transaction"
    ],
    "reproduction_commands": "validation/REPRODUCTION.md",
    "training_source_counts": "tables/TRAINING_SOURCE_COUNTS.csv",
    "instruction": "This record does not claim the user objective complete. Publication evidence belongs to the external final receipt to avoid a self-referential Git snapshot."
  },
  "publication_receipt": "NOT_PUBLISHED",
  "settled_cost": {
    "prior_gpu_hours": 12.124826178494708,
    "v2_gpu_hours": 66.81517986008913,
    "settled_gpu_hours": 78.94000603858385,
    "event_count": 41,
    "association_cpu_core_hours": 8.0876546187225,
    "unsettled_live_reservation_gpu_hours": 0.0
  }
}

## 可恢复训练

| recipe | completed_updates | last_checkpoint | status |
| --- | --- | --- | --- |
| A-OPEN-L-s45 | 750 | external:training/A-OPEN-L-s45/last.ckpt | COMPLETE |
| C0-L-s45 | 3000 | external:training/C0-L-s45/last.ckpt | COMPLETE |
| Q-SEM-L-s45 | 3000 | external:training/Q-SEM-L-s45/last.ckpt | COMPLETE |
| S-BAL-H-s45 | 750 | external:training/S-BAL-H-s45/last.ckpt | COMPLETE |
| S-BAL-L-s45 | 750 | external:training/S-BAL-L-s45/last.ckpt | COMPLETE |
| S-WORST-L-s45 | 750 | external:training/S-WORST-L-s45/last.ckpt | COMPLETE |

## 未完成任务

| task | status | reason | next_command |
| --- | --- | --- | --- |
| PUBLISH | RUNNING | — | conda run -n persist4d python -m scripts.perception_gain_v2 run --target PUBLISH --resume |


Publication snapshot: experiment A `705c39898d6b2d813b5cbaa592e19b2663144fb3`; phase `READY`; branch `research/persist4d-perception-gain-v2`; planned tag `persist4d-perception-gain-v2`.
Final receipt: `$PERSIST4D_GAIN_V2_ROOT/publication/PUBLICATION_RECEIPT.json` (external, written only after remote verification).
