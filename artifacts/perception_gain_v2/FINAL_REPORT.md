# Persist4D Perception Gain V2

生成时间：2026-09-26T17:23:06.993611+00:00；代码：`d58420568f46070a4b3a7dcee2b4e47fc42f4c7b`。这是实际文件的当前快照。

所有 AP 与差值使用 0–1 标度；缺失值为 —。只有完整人口才能支持全覆盖胜出结论。

```json
{
  "execution_status": "COMPLETE",
  "perception_selection": "KEEP_R1",
  "repair_R1_selection": "KEEP_PARENT",
  "repair_P_selection": "NOT_APPLICABLE",
  "association_selection": "KEEP_D0",
  "replication_status": "NOT_APPLICABLE",
  "pb_all_t_vs_native_fh": false,
  "pb_all_t_vs_d0": false,
  "resource_status": "IMPROVED",
  "publication_status": "NOT_PUBLISHED",
  "local_gain": false
}
```

## 1. 任务与训练范围

| task | status | reason | observed_from |
| --- | --- | --- | --- |
| BIND | COMPLETE | — | RUN_STATE.json |
| DATA | COMPLETE | — | RUN_STATE.json |
| BASELINE | COMPLETE | — | RUN_STATE.json |
| HIGH_CONT | COMPLETE | — | RUN_STATE.json |
| LOW_PAIR | COMPLETE | — | RUN_STATE.json |
| REPAIR_R1 | COMPLETE | — | RUN_STATE.json |
| ASSOC | COMPLETE | — | RUN_STATE.json |
| AUX | COMPLETE | — | RUN_STATE.json |
| PERCEPTION | COMPLETE | — | RUN_STATE.json |
| REPAIR_P | NOT_APPLICABLE | P is R1; reuse its existing paired repair | RUN_STATE.json |
| LOCK | COMPLETE | — | RUN_STATE.json |
| REPLICATE | COMPLETE | — | RUN_STATE.json |
| CONFIRM | COMPLETE | — | RUN_STATE.json |
| PROFILE | COMPLETE | — | RUN_STATE.json |
| REPORT | COMPLETE | — | RUN_STATE.json |
| PUBLISH | RUNNING | — | RUN_STATE.json |

## 训练终点（summary GPUh 仅归因，总账不重复相加）

| recipe | train_seed | completed_updates | status | last_checkpoint |
| --- | --- | --- | --- | --- |
| A-OPEN-H-s45 | 45 | 0 | NOT_RUN | — |
| A-OPEN-H-s46 | 46 | 0 | NOT_RUN | — |
| A-OPEN-L-s45 | 45 | 750 | COMPLETE | external:training/A-OPEN-L-s45/last.ckpt |
| A-OPEN-L-s46 | 46 | 0 | NOT_RUN | — |
| C0-H-s45 | 45 | 750 | IMPORTED | — |
| C0-H-s46 | 46 | 0 | NOT_RUN | — |
| C0-L-s45 | 45 | 3000 | COMPLETE | external:training/C0-L-s45/last.ckpt |
| C0-L-s46 | 46 | 0 | NOT_RUN | — |
| Q-SEM-H-s45 | 45 | 0 | NOT_RUN | — |
| Q-SEM-H-s46 | 46 | 0 | NOT_RUN | — |
| Q-SEM-L-s45 | 45 | 3000 | COMPLETE | external:training/Q-SEM-L-s45/last.ckpt |
| Q-SEM-L-s46 | 46 | 0 | NOT_RUN | — |
| S-BAL-H-s45 | 45 | 750 | COMPLETE | external:training/S-BAL-H-s45/last.ckpt |
| S-BAL-H-s46 | 46 | 0 | NOT_RUN | — |
| S-BAL-L-s45 | 45 | 750 | COMPLETE | external:training/S-BAL-L-s45/last.ckpt |
| S-BAL-L-s46 | 46 | 0 | NOT_RUN | — |
| S-WORST-H-s45 | 45 | 0 | NOT_RUN | — |
| S-WORST-H-s46 | 46 | 0 | NOT_RUN | — |
| S-WORST-L-s45 | 45 | 750 | COMPLETE | external:training/S-WORST-L-s45/last.ckpt |
| S-WORST-L-s46 | 46 | 0 | NOT_RUN | — |

## 2. H/L × C0/S-BAL 固定比较

| recipe | step | coverage | T2 | T3 | T4 | T5 | mean_delta_R1 | mean_same_step_delta_C0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0-H-s45 | 250 | COMPLETE | 0.353755 | 0.275942 | 0.235413 | 0.217662 | -0.010269 | 0.000000 |
| C0-H-s45 | 750 | COMPLETE | 0.334286 | 0.280689 | 0.225745 | 0.216750 | -0.016594 | 0.000000 |
| S-BAL-H-s45 | 250 | COMPLETE | 0.348642 | 0.277610 | 0.239545 | 0.222650 | -0.008850 | 0.001419 |
| S-BAL-H-s45 | 750 | COMPLETE | 0.327426 | 0.273793 | 0.235285 | 0.208094 | -0.019813 | -0.003218 |
| C0-L-s45 | 250 | COMPLETE | 0.353564 | 0.302494 | 0.244840 | 0.235544 | 0.003148 | 0.000000 |
| C0-L-s45 | 750 | COMPLETE | 0.357787 | 0.296449 | 0.245842 | 0.224350 | 0.000145 | 0.000000 |
| S-BAL-L-s45 | 250 | COMPLETE | 0.362998 | 0.290932 | 0.232792 | 0.221192 | -0.003983 | -0.007132 |
| S-BAL-L-s45 | 750 | COMPLETE | 0.358286 | 0.294519 | 0.241045 | 0.222473 | -0.001881 | -0.002026 |

## 3. 所有已评价感知检查点

| role | recipe | step | coverage | T2 | T3 | T4 | T5 | mean_delta_R1 | artifact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAL | A-OPEN-L-s45 | 0 | COMPLETE | 0.363079 | 0.292853 | 0.244949 | 0.221581 | -0.000346 | evaluation/CAL/A-OPEN-L-s45/update=0000.json |
| CAL | A-OPEN-L-s45 | 250 | COMPLETE | 0.364084 | 0.293312 | 0.243217 | 0.226326 | 0.000773 | evaluation/CAL/A-OPEN-L-s45/update=0250.json |
| CAL | A-OPEN-L-s45 | 750 | COMPLETE | 0.360537 | 0.293996 | 0.245000 | 0.230555 | 0.001560 | evaluation/CAL/A-OPEN-L-s45/update=0750.json |
| CAL | C0-H-s45 | 250 | COMPLETE | 0.353755 | 0.275942 | 0.235413 | 0.217662 | -0.010269 | evaluation/CAL/C0-H-s45/update=0250.json |
| CAL | C0-H-s45 | 750 | COMPLETE | 0.334286 | 0.280689 | 0.225745 | 0.216750 | -0.016594 | evaluation/CAL/C0-H-s45/update=0750.json |
| CAL | C0-L-s45 | 250 | COMPLETE | 0.353564 | 0.302494 | 0.244840 | 0.235544 | 0.003148 | evaluation/CAL/C0-L-s45/update=0250.json |
| CAL | C0-L-s45 | 750 | COMPLETE | 0.357787 | 0.296449 | 0.245842 | 0.224350 | 0.000145 | evaluation/CAL/C0-L-s45/update=0750.json |
| CAL | C0-L-s45 | 1500 | COMPLETE | 0.358611 | 0.291204 | 0.232958 | 0.222352 | -0.004681 | evaluation/CAL/C0-L-s45/update=1500.json |
| CAL | C0-L-s45 | 2250 | COMPLETE | 0.360576 | 0.286328 | 0.235274 | 0.223465 | -0.004551 | evaluation/CAL/C0-L-s45/update=2250.json |
| CAL | C0-L-s45 | 3000 | COMPLETE | 0.352118 | 0.289584 | 0.242043 | 0.219539 | -0.005141 | evaluation/CAL/C0-L-s45/update=3000.json |
| CAL | Q-SEM-L-s45 | 0 | COMPLETE | 0.360675 | 0.295631 | 0.240839 | 0.222269 | -0.001109 | evaluation/CAL/Q-SEM-L-s45/update=0000.json |
| CAL | Q-SEM-L-s45 | 250 | COMPLETE | 0.363144 | 0.299556 | 0.246988 | 0.233223 | 0.004766 | evaluation/CAL/Q-SEM-L-s45/update=0250.json |
| CAL | Q-SEM-L-s45 | 750 | COMPLETE | 0.362266 | 0.301670 | 0.246552 | 0.226507 | 0.003287 | evaluation/CAL/Q-SEM-L-s45/update=0750.json |
| CAL | Q-SEM-L-s45 | 1500 | COMPLETE | 0.366168 | 0.297662 | 0.240764 | 0.223430 | 0.001044 | evaluation/CAL/Q-SEM-L-s45/update=1500.json |
| CAL | Q-SEM-L-s45 | 2250 | COMPLETE | 0.362045 | 0.289257 | 0.236250 | 0.220421 | -0.003969 | evaluation/CAL/Q-SEM-L-s45/update=2250.json |
| CAL | Q-SEM-L-s45 | 3000 | COMPLETE | 0.358511 | 0.298014 | 0.241043 | 0.225074 | -0.000302 | evaluation/CAL/Q-SEM-L-s45/update=3000.json |
| CAL | S-BAL-H-s45 | 250 | COMPLETE | 0.348642 | 0.277610 | 0.239545 | 0.222650 | -0.008850 | evaluation/CAL/S-BAL-H-s45/update=0250.json |
| CAL | S-BAL-H-s45 | 750 | COMPLETE | 0.327426 | 0.273793 | 0.235285 | 0.208094 | -0.019813 | evaluation/CAL/S-BAL-H-s45/update=0750.json |
| CAL | S-BAL-L-s45 | 250 | COMPLETE | 0.362998 | 0.290932 | 0.232792 | 0.221192 | -0.003983 | evaluation/CAL/S-BAL-L-s45/update=0250.json |
| CAL | S-BAL-L-s45 | 750 | COMPLETE | 0.358286 | 0.294519 | 0.241045 | 0.222473 | -0.001881 | evaluation/CAL/S-BAL-L-s45/update=0750.json |
| CAL | S-WORST-L-s45 | 250 | COMPLETE | 0.360557 | 0.300178 | 0.243311 | 0.228993 | 0.002298 | evaluation/CAL/S-WORST-L-s45/update=0250.json |
| CAL | S-WORST-L-s45 | 750 | COMPLETE | 0.366004 | 0.298016 | 0.243803 | 0.218642 | 0.000654 | evaluation/CAL/S-WORST-L-s45/update=0750.json |
| SEL | C0-L-s45 | 250 | COMPLETE | 0.684552 | 0.645793 | 0.607534 | 0.572749 | -0.003017 | evaluation/SEL/C0-L-s45/update=0250.json |
| SEL | Q-SEM-L-s45 | 250 | COMPLETE | 0.687716 | 0.645871 | 0.605696 | 0.581248 | -0.000542 | evaluation/SEL/Q-SEM-L-s45/update=0250.json |

## 4. NEW/PAIR 与各自父模型

| parent | role | method | step | coverage | T2 | T3 | T4 | T5 | mean_delta_parent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | CAL | PARENT | 0 | COMPLETE | 0.362623 | 0.291813 | 0.246709 | 0.222703 | 0.000000 |
| R1 | CAL | R-NEW-R1@0000 | 0 | COMPLETE | 0.362623 | 0.291813 | 0.246709 | 0.222703 | 0.000000 |
| R1 | CAL | R-NEW-R1@0500 | 500 | COMPLETE | 0.355124 | 0.278525 | 0.232256 | 0.209977 | -0.011991 |
| R1 | CAL | R-NEW-R1@1000 | 1000 | COMPLETE | 0.355124 | 0.278525 | 0.232256 | 0.209977 | -0.011991 |
| R1 | CAL | R-NEW-R1@1500 | 1500 | COMPLETE | 0.355124 | 0.278525 | 0.232256 | 0.209977 | -0.011991 |
| R1 | CAL | R-PAIR-R1@0000 | 0 | COMPLETE | 0.362623 | 0.291813 | 0.246709 | 0.222703 | 0.000000 |
| R1 | CAL | R-PAIR-R1@0500 | 500 | COMPLETE | 0.355124 | 0.278525 | 0.232256 | 0.209326 | -0.012154 |
| R1 | CAL | R-PAIR-R1@1000 | 1000 | COMPLETE | 0.355124 | 0.278525 | 0.232256 | 0.209326 | -0.012154 |
| R1 | CAL | R-PAIR-R1@1500 | 1500 | COMPLETE | 0.355124 | 0.278525 | 0.232256 | 0.209326 | -0.012154 |
| R1 | SEL | PARENT | 0 | COMPLETE | 0.674208 | 0.649342 | 0.617273 | 0.581874 | 0.000000 |
| R1 | SEL | R-NEW-R1@0000 | 0 | COMPLETE | 0.674208 | 0.649342 | 0.617273 | 0.581874 | 0.000000 |
| R1 | SEL | R-PAIR-R1@0000 | 0 | COMPLETE | 0.674208 | 0.649342 | 0.617273 | 0.581874 | 0.000000 |
| P | CAL | PARENT | — | NOT_RUN | — | — | — | — | — |
| P | SEL | PARENT | — | NOT_RUN | — | — | — | — | — |

## 修复数据与诊断

| parent | mode | trained_updates | loss | post_diagnostic_status | corrected_original_errors | broken_original_correct |
| --- | --- | --- | --- | --- | --- | --- |
| R1 | NEW | 1500 | 0.994215 | PASS | 63618 | 6799 |
| R1 | PAIR | 1500 | 0.994194 | PASS | 63616 | 6795 |
| P | NEW | 0 | — | NOT_RUN | — | — |
| P | PAIR | 0 | — | NOT_RUN | — | — |

## 5. 固定 12 个关联配置

| config | attempt | status | completed_units | expected_units | T2 | T3 | T4 | T5 | cpu_core_hours |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0-U-tau-0.6 | CURRENT_PARENT | PASS | 23 | 23 | 0.362623 | 0.275234 | 0.226681 | 0.202842 | 0.462953 |
| A0-U-tau-0.666666666667 | CURRENT_PARENT | PASS | 23 | 23 | 0.362623 | 0.287443 | 0.239004 | 0.220029 | 0.451490 |
| A0-U-tau-0.73 | CURRENT_PARENT | PASS | 23 | 23 | 0.362623 | 0.290004 | 0.242931 | 0.221639 | 0.478022 |
| A1-theta-0.5 | CURRENT_PARENT | PASS | 23 | 23 | 0.362623 | 0.284334 | 0.238249 | 0.218732 | 0.458993 |
| A1-theta-0.65 | CURRENT_PARENT | PASS | 23 | 23 | 0.362623 | 0.284334 | 0.238241 | 0.218750 | 0.456305 |
| A1-theta-0.8 | CURRENT_PARENT | PASS | 23 | 23 | 0.362623 | 0.287177 | 0.238861 | 0.218789 | 0.468348 |
| A2-lambda-0.25-tau-0.6 | CURRENT_PARENT | PASS | 23 | 23 | 0.362623 | 0.269517 | 0.223256 | 0.193613 | 0.451979 |
| A2-lambda-0.25-tau-0.666666666667 | CURRENT_PARENT | PASS | 23 | 23 | 0.362623 | 0.276923 | 0.234715 | 0.212579 | 0.443104 |
| A2-lambda-0.25-tau-0.73 | CURRENT_PARENT | PASS | 23 | 23 | 0.362623 | 0.290176 | 0.243602 | 0.219679 | 0.458678 |
| A2-lambda-0.5-tau-0.6 | CURRENT_PARENT | PASS | 23 | 23 | 0.362623 | 0.264890 | 0.215067 | 0.183057 | 0.460793 |
| A2-lambda-0.5-tau-0.666666666667 | CURRENT_PARENT | PASS | 23 | 23 | 0.362623 | 0.265233 | 0.220935 | 0.190144 | 0.465624 |
| A2-lambda-0.5-tau-0.73 | CURRENT_PARENT | PASS | 23 | 23 | 0.362623 | 0.265981 | 0.225115 | 0.198573 | 0.476963 |
| A0-U-tau-0.6 | HISTORICAL_PARENT | INVALIDATED_PARENT_RUNTIME | 23 | 23 | 0.364479 | 0.282513 | 0.227254 | 0.204821 | 0.479109 |
| A0-U-tau-0.666666666667 | HISTORICAL_PARENT | INVALIDATED_PARENT_RUNTIME | 23 | 23 | 0.364479 | 0.291534 | 0.239950 | 0.220981 | 0.458368 |
| A0-U-tau-0.73 | HISTORICAL_PARENT | INVALIDATED_PARENT_RUNTIME | 23 | 23 | 0.364479 | 0.289980 | 0.242953 | 0.221589 | 0.469130 |
| A1-theta-0.5 | HISTORICAL_PARENT | INVALIDATED_PARENT_RUNTIME | 23 | 23 | 0.364479 | 0.291878 | 0.242922 | 0.223860 | 0.464712 |

## 6. 正式确认：各人口分别报告

| population | method | T | eval_seed | actual_units | expected_units | actual_references | expected_references | t_mAP | coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PB | FH-R1-native | 2 | 45 | 129 | 129 | 6 | 6 | 0.227096 | COMPLETE |
| PB | FH-R1-native | 3 | 45 | 129 | 129 | 6 | 6 | 0.149680 | COMPLETE |
| PB | FH-R1-native | 4 | 45 | 129 | 129 | 6 | 6 | 0.104179 | COMPLETE |
| PB | FH-R1-native | 5 | 45 | 129 | 129 | 6 | 6 | 0.080212 | COMPLETE |
| PB | R1-D0 | 2 | 45 | 129 | 129 | 6 | 6 | 0.249267 | COMPLETE |
| PB | R1-D0 | 3 | 45 | 129 | 129 | 6 | 6 | 0.149947 | COMPLETE |
| PB | R1-D0 | 4 | 45 | 129 | 129 | 6 | 6 | 0.082923 | COMPLETE |
| PB | R1-D0 | 5 | 45 | 129 | 129 | 6 | 6 | 0.058866 | COMPLETE |
| PB | C0-L-s45 | 2 | 45 | 129 | 129 | 6 | 6 | 0.239222 | COMPLETE |
| PB | C0-L-s45 | 3 | 45 | 129 | 129 | 6 | 6 | 0.147666 | COMPLETE |
| PB | C0-L-s45 | 4 | 45 | 129 | 129 | 6 | 6 | 0.082547 | COMPLETE |
| PB | C0-L-s45 | 5 | 45 | 129 | 129 | 6 | 6 | 0.055099 | COMPLETE |
| ADDITIONAL | FH-R1-native | 2 | 45 | 111 | 111 | 40 | 40 | 0.310104 | COMPLETE |
| ADDITIONAL | FH-R1-native | 3 | 45 | 77 | 77 | 23 | 23 | 0.277922 | COMPLETE |
| ADDITIONAL | FH-R1-native | 4 | 45 | 32 | 32 | 8 | 8 | 0.204651 | COMPLETE |
| ADDITIONAL | R1-D0 | 2 | 45 | 111 | 111 | 40 | 40 | 0.285252 | COMPLETE |
| ADDITIONAL | R1-D0 | 3 | 45 | 77 | 77 | 23 | 23 | 0.258518 | COMPLETE |
| ADDITIONAL | R1-D0 | 4 | 45 | 32 | 32 | 8 | 8 | 0.216075 | COMPLETE |
| LOCAL-T2 | FH-R1-native | 2 | 45 | 154 | 154 | 46 | 46 | 0.307081 | COMPLETE |
| LOCAL-T2 | FH-R1-native | 2 | 46 | 154 | 154 | 46 | 46 | 0.310732 | COMPLETE |
| LOCAL-T2 | FH-R1-native | 2 | 47 | 154 | 154 | 46 | 46 | 0.298566 | COMPLETE |
| LOCAL-T2 | FH-R1-native | 2 | mean(45,46,47) | 154 | 154 | 46 | 46 | 0.305460 | COMPLETE |
| LOCAL-T2 | C0-best-native | 2 | 45 | 154 | 154 | 46 | 46 | 0.300169 | COMPLETE |
| LOCAL-T2 | C0-best-native | 2 | 46 | 154 | 154 | 46 | 46 | 0.311938 | COMPLETE |
| LOCAL-T2 | C0-best-native | 2 | 47 | 154 | 154 | 46 | 46 | 0.296519 | COMPLETE |
| LOCAL-T2 | C0-best-native | 2 | mean(45,46,47) | 154 | 154 | 46 | 46 | 0.302875 | COMPLETE |

## 7. 工程成本（未结算的运行占用未加入本表）

| quantity | value |
| --- | --- |
| prior_gpu_hours | 12.124826 |
| v2_gpu_hours | 66.815180 |
| settled_gpu_hours | 78.940006 |
| event_count | 41 |
| association_cpu_core_hours | 8.087655 |
| unsettled_live_reservation_gpu_hours | 0.000000 |

## 8. 实际发布覆盖

| asset | status | bytes | sha256 | url |
| --- | --- | --- | --- | --- |
| Git branch/tag and required Release assets | NOT_PUBLISHED | — | — | — |

## 选择与证据边界

{
  "learning_rate": "L",
  "auxiliary_pilots": {
    "A-OPEN-L-s45": {
      "status": "COMPLETE",
      "reason": null,
      "completed_global_step": 750,
      "budget": null,
      "cal": [
        {
          "step": 0,
          "coverage": "COMPLETE"
        },
        {
          "step": 250,
          "coverage": "COMPLETE"
        },
        {
          "step": 750,
          "coverage": "COMPLETE"
        }
      ]
    },
    "Q-SEM-L-s45": {
      "status": "COMPLETE",
      "reason": null,
      "completed_global_step": 750,
      "budget": null,
      "cal": [
        {
          "step": 0,
          "coverage": "COMPLETE"
        },
        {
          "step": 250,
          "coverage": "COMPLETE"
        },
        {
          "step": 750,
          "coverage": "COMPLETE"
        }
      ]
    },
    "S-WORST-L-s45": {
      "status": "COMPLETE",
      "reason": null,
      "completed_global_step": 750,
      "budget": null,
      "cal": [
        {
          "step": 250,
          "coverage": "COMPLETE"
        },
        {
          "step": 750,
          "coverage": "COMPLETE"
        }
      ]
    }
  },
  "full_promotion": {
    "endpoint": 3000,
    "full_training_recipes": [
      "C0-L-s45",
      "Q-SEM-L-s45"
    ],
    "missing_controls": {},
    "pilot_best": {
      "A-OPEN-L-s45": {
        "S_long": 0.0030713602900505066,
        "S_mean": 0.0015601404011249542,
        "S_min": -0.0020852088928222656,
        "architecture_variant": "A-OPEN",
        "checkpoint": "external:training/A-OPEN-L-s45/update=0750.ckpt",
        "checkpoint_sha256": "314c8fd885636cae6c9cb1b18a6f5753cc6bca5c6898440ec287a8a7673679a1",
        "comparison_status": "COMPLETE",
        "completed_logical_units": 23,
        "coverage_status": "COMPLETE",
        "deltas": {
          "2": -0.0020852088928222656,
          "3": 0.0021830499172210693,
          "4": -0.0017093271017074585,
          "5": 0.007852047681808472
        },
        "expected_logical_units": 23,
        "method_id": "A-OPEN-L-s45",
        "metrics": {
          "2": 0.36053740978240967,
          "3": 0.2939961850643158,
          "4": 0.24499990046024323,
          "5": 0.2305554300546646
        },
        "new_parameter_count": 0,
        "optimizer_update": 750,
        "recipe_id": "A-OPEN-L-s45"
      },
      "C0-H-s45": {
        "S_long": -0.00816848874092102,
        "S_mean": -0.010268859565258026,
        "S_min": -0.01587069034576416,
        "architecture_variant": "C0",
        "checkpoint": "external:inputs/imported/C0-H-s45/update=0250.ckpt",
        "checkpoint_sha256": "362a8575fb0d011b079b60af4eb92f57fcb5734484230c76c468bb9718007b87",
        "comparison_status": "COMPLETE",
        "completed_logical_units": 23,
        "coverage_status": "COMPLETE",
        "deltas": {
          "2": -0.008867770433425903,
          "3": -0.01587069034576416,
          "4": -0.011295899748802185,
          "5": -0.005041077733039856
        },
        "expected_logical_units": 23,
        "method_id": "C0-H-s45",
        "metrics": {
          "2": 0.35375484824180603,
          "3": 0.27594244480133057,
          "4": 0.2354133278131485,
          "5": 0.21766230463981628
        },
        "new_parameter_count": 0,
        "optimizer_update": 250,
        "recipe_id": "C0-H-s45"
      },
      "C0-L-s45": {
        "S_long": 0.005485467612743378,
        "S_mean": 0.003148358315229416,
        "S_min": -0.009058594703674316,
        "architecture_variant": "C0",
        "checkpoint": "external:training/C0-L-s45/update=0250.ckpt",
        "checkpoint_sha256": "e85a2217c05d41fd2e82b996ff6676213cb050d5bc7dd48a0e37a33d26c11a1d",
        "comparison_status": "COMPLETE",
        "completed_logical_units": 23,
        "coverage_status": "COMPLETE",
        "deltas": {
          "2": -0.009058594703674316,
          "3": 0.010681092739105225,
          "4": -0.0018693506717681885,
          "5": 0.012840285897254944
        },
        "expected_logical_units": 23,
        "method_id": "C0-L-s45",
        "metrics": {
          "2": 0.3535640239715576,
          "3": 0.30249422788619995,
          "4": 0.2448398768901825,
          "5": 0.23554366827011108
        },
        "new_parameter_count": 0,
        "optimizer_update": 250,
        "recipe_id": "C0-L-s45"
      },
      "Q-SEM-L-s45": {
        "S_long": 0.005399301648139954,
        "S_mean": 0.004765920341014862,
        "S_min": 0.000279158353805542,
        "architecture_variant": "Q-SEM",
        "checkpoint": "external:training/Q-SEM-L-s45/update=0250.ckpt",
        "checkpoint_sha256": "84784c3656482a25d211dbc1cce6e3902dec595294699782d0dcc9d59c618e24",
        "comparison_status": "COMPLETE",
        "completed_logical_units": 23,
        "coverage_status": "COMPLETE",
        "deltas": {
          "2": 0.0005218684673309326,
          "3": 0.007743209600448608,
          "4": 0.000279158353805542,
          "5": 0.010519444942474365
        },
        "expected_logical_units": 23,
        "method_id": "Q-SEM-L-s45",
        "metrics": {
          "2": 0.36314448714256287,
          "3": 0.29955634474754333,
          "4": 0.24698838591575623,
          "5": 0.2332228273153305
        },
        "new_parameter_count": 8577,
        "optimizer_update": 250,
        "recipe_id": "Q-SEM-L-s45"
      },
      "S-BAL-H-s45": {
        "S_long": -0.0036088749766349792,
        "S_mean": -0.008850287646055222,
        "S_min": -0.014202684164047241,
        "architecture_variant": "S-BAL",
        "checkpoint": "external:inputs/imported/S-BAL-H-s45/update=0250.ckpt",
        "checkpoint_sha256": "9b47b174c1048167fc8156ea2a327692dabae947265adc9193f787d6c7c857f6",
        "comparison_status": "COMPLETE",
        "completed_logical_units": 23,
        "coverage_status": "COMPLETE",
        "deltas": {
          "2": -0.013980716466903687,
          "3": -0.014202684164047241,
          "4": -0.0071641504764556885,
          "5": -5.359947681427002e-05
        },
        "expected_logical_units": 23,
        "method_id": "S-BAL-H-s45",
        "metrics": {
          "2": 0.34864190220832825,
          "3": 0.2776104509830475,
          "4": 0.239545077085495,
          "5": 0.22264978289604187
        },
        "new_parameter_count": 0,
        "optimizer_update": 250,
        "recipe_id": "S-BAL-H-s45"
      },
      "S-BAL-L-s45": {
        "S_long": -0.0029473155736923218,
        "S_mean": -0.0018814951181411743,
        "S_min": -0.005664214491844177,
        "architecture_variant": "S-BAL",
        "checkpoint": "external:training/S-BAL-L-s45/update=0750.ckpt",
        "checkpoint_sha256": "f2322334abaa715b35440fec2eb9a70aa6aacd29d00fc9849f43806632082652",
        "comparison_status": "COMPLETE",
        "completed_logical_units": 23,
        "coverage_status": "COMPLETE",
        "deltas": {
          "2": -0.00433686375617981,
          "3": 0.002705514430999756,
          "4": -0.005664214491844177,
          "5": -0.0002304166555404663
        },
        "expected_logical_units": 23,
        "method_id": "S-BAL-L-s45",
        "metrics": {
          "2": 0.3582857549190521,
          "3": 0.2945186495780945,
          "4": 0.2410450130701065,
          "5": 0.22247296571731567
        },
        "new_parameter_count": 0,
        "optimizer_update": 750,
        "recipe_id": "S-BAL-L-s45"
      },
      "S-WORST-L-s45": {
        "S_long": 0.001445673406124115,
        "S_mean": 0.00229765847325325,
        "S_min": -0.0033986717462539673,
        "architecture_variant": "S-WORST",
        "checkpoint": "external:training/S-WORST-L-s45/update=0250.ckpt",
        "checkpoint_sha256": "8794dfde5d1b5b3a73fb476cc2a4caa54906a6d2cace69b6d452a718eb51b91e",
        "comparison_status": "COMPLETE",
        "completed_logical_units": 23,
        "coverage_status": "COMPLETE",
        "deltas": {
          "2": -0.002065688371658325,
          "3": 0.008364975452423096,
          "4": -0.0033986717462539673,
          "5": 0.006290018558502197
        },
        "expected_logical_units": 23,
        "method_id": "S-WORST-L-s45",
        "metrics": {
          "2": 0.3605569303035736,
          "3": 0.3001781105995178,
          "4": 0.24331055581569672,
          "5": 0.22899340093135834
        },
        "new_parameter_count": 0,
        "optimizer_update": 250,
        "recipe_id": "S-WORST-L-s45"
      }
    },
    "reason": "PILOT_GATE",
    "resume_update": 750
  },
  "perception_failures": {},
  "final_method": "R1-D0",
  "association_CAL": {
    "S_long": -0.0024211928248405457,
    "S_mean": -0.0016628392040729523,
    "S_min": -0.003778114914894104,
    "association_config": {
      "family": "A0-U",
      "lambda_": 0.0,
      "mutual_margin": 0.1,
      "overlap_theta": 0.65,
      "tau": 0.73
    },
    "checkpoint_sha256": "629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd",
    "comparison_status": "COMPLETE",
    "coverage_status": "COMPLETE",
    "deltas": {
      "2": 0.0,
      "3": -0.0018089711666107178,
      "4": -0.003778114914894104,
      "5": -0.0010642707347869873
    },
    "method_id": "A0-U-tau-0.73",
    "metrics": {
      "2": 0.36262261867523193,
      "3": 0.290004163980484,
      "4": 0.24293111264705658,
      "5": 0.22163911163806915
    },
    "new_parameter_count": 0,
    "optimizer_update": 0,
    "recipe_id": "C0-L-s45"
  },
  "association_SEL": {
    "S_long": -0.00011113286018371582,
    "S_mean": 0.0002415478229522705,
    "S_min": -0.0003787875175476074,
    "association_config": {
      "family": "A0-U",
      "lambda_": 0.0,
      "mutual_margin": 0.1,
      "overlap_theta": 0.65,
      "tau": 0.73
    },
    "checkpoint_sha256": "629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd",
    "comparison_status": "COMPLETE",
    "coverage_status": "COMPLETE",
    "deltas": {
      "2": 0.0,
      "3": 0.0011884570121765137,
      "4": 0.00015652179718017578,
      "5": -0.0003787875175476074
    },
    "method_id": "A0-U-tau-0.73",
    "metrics": {
      "2": 0.6742081642150879,
      "3": 0.6505308747291565,
      "4": 0.6174293160438538,
      "5": 0.5814954042434692
    },
    "new_parameter_count": 0,
    "optimizer_update": 0,
    "recipe_id": "C0-L-s45"
  },
  "replication": "NOT_APPLICABLE",
  "deployment": {
    "default_method": "R1-D0",
    "final_vs_D0": {
      "S_long": 0.0,
      "S_mean": 0.0,
      "S_min": 0.0,
      "artifact": "artifacts/perception_gain_v2/confirmation/PB/32fe7aa00d264cfa40ab287206d6a5d8ae7c1543581db0a89b4a116dc61d1772.json",
      "artifact_sha256": "9cb508a5f75fea8dbb8a40d7ac6ffcc833e70dea57595690433ebbc435e8acf2",
      "comparison_status": "COMPLETE",
      "coverage_by_horizon": {
        "2": {
          "actual_logical_units": 129,
          "actual_references": 6,
          "expected_logical_units": 129,
          "expected_references": 6,
          "status": "COMPLETE"
        },
        "3": {
          "actual_logical_units": 129,
          "actual_references": 6,
          "expected_logical_units": 129,
          "expected_references": 6,
          "status": "COMPLETE"
        },
        "4": {
          "actual_logical_units": 129,
          "actual_references": 6,
          "expected_logical_units": 129,
          "expected_references": 6,
          "status": "COMPLETE"
        },
        "5": {
          "actual_logical_units": 129,
          "actual_references": 6,
          "expected_logical_units": 129,
          "expected_references": 6,
          "status": "COMPLETE"
        }
      },
      "coverage_status": "COMPLETE",
      "data_role": "PB",
      "deltas": {
        "2": 0.0,
        "3": 0.0,
        "4": 0.0,
        "5": 0.0
      },
      "incomplete_units": [],
      "method_id": "R1-D0",
      "metric_rows": [
        {
          "T": 2,
          "current_stage_AP": 0.3823559582233429,
          "logical_unit_count": 129,
          "method": "R1-D0",
          "reference": "all",
          "reference_count": 6,
          "source_method": "PARENT",
          "t_REC": 0.41060909628868103,
          "t_mAP": 0.24926671385765076,
          "t_mAP25": 0.5738745927810669,
          "t_mAP50": 0.42159929871559143
        },
        {
          "T": 2,
          "current_stage_AP": 0.47927969694137573,
          "logical_unit_count": 30,
          "method": "R1-D0",
          "reference": "10b17940-3938-2467-8a7a-958300ba83d3",
          "reference_count": 1,
          "source_method": "PARENT",
          "t_REC": 0.4957005977630615,
          "t_mAP": 0.2965559959411621,
          "t_mAP25": 0.48683738708496094,
          "t_mAP50": 0.39211568236351013
        },
        {
          "T": 2,
          "current_stage_AP": 0.4591214954853058,
          "logical_unit_count": 36,
          "method": "R1-D0",
          "reference": "137a8158-1db5-2cc0-8003-31c12610471e",
          "reference_count": 1,
          "source_method": "PARENT",
          "t_REC": 0.4307054877281189,
          "t_mAP": 0.3218762278556824,
          "t_mAP25": 0.6137778759002686,
          "t_mAP50": 0.4656772315502167
        },
        {
          "T": 2,
          "current_stage_AP": 0.6327818632125854,
          "logical_unit_count": 15,
          "method": "R1-D0",
          "reference": "280d8ebb-6cc6-2788-9153-98959a2da801",
          "reference_count": 1,
          "source_method": "PARENT",
          "t_REC": 0.5045526027679443,
          "t_mAP": 0.41507476568222046,
          "t_mAP25": 0.7300218939781189,
          "t_mAP50": 0.6431086659431458
        },
        {
          "T": 2,
          "current_stage_AP": 0.41369953751564026,
          "logical_unit_count": 15,
          "method": "R1-D0",
          "reference": "5630cfcf-12bf-2860-8784-83d28a611a83",
          "reference_count": 1,
          "source_method": "PARENT",
          "t_REC": 0.3624247908592224,
          "t_mAP": 0.28938934206962585,
          "t_mAP25": 0.7040541172027588,
          "t_mAP50": 0.36594295501708984
        },
        {
          "T": 2,
          "current_stage_AP": 0.46544328331947327,
          "logical_unit_count": 15,
          "method": "R1-D0",
          "reference": "8eabc45f-5af7-2f32-8528-640861d2a135",
          "reference_count": 1,
          "source_method": "PARENT",
          "t_REC": 0.5319485664367676,
          "t_mAP": 0.3728611469268799,
          "t_mAP25": 0.7354336380958557,
          "t_mAP50": 0.5588504076004028
        },
        {
          "T": 2,
          "current_stage_AP": 0.31929993629455566,
          "logical_unit_count": 18,
          "method": "R1-D0",
          "reference": "ddc73797-765b-241a-9e2c-097c5989baf6",
          "reference_count": 1,
          "source_method": "PARENT",
          "t_REC": 0.3069281578063965,
          "t_mAP": 0.17916898429393768,
          "t_mAP25": 0.4233214557170868,
          "t_mAP50": 0.34871432185173035
        },
        {
          "T": 3,
          "current_stage_AP": 0.37156179547309875,
          "logical_unit_count": 129,
          "method": "R1-D0",
          "reference": "all",
          "reference_count": 6,
          "source_method": "PARENT",
          "t_REC": 0.31353816390037537,
          "t_mAP": 0.14994679391384125,
          "t_mAP25": 0.4341409504413605,
          "t_mAP50": 0.2733047604560852
        },
        {
          "T": 3,
          "current_stage_AP": 0.48055464029312134,
          "logical_unit_count": 30,
          "method": "R1-D0",
          "reference": "10b17940-3938-2467-8a7a-958300ba83d3",
          "reference_count": 1,
          "source_method": "PARENT",
          "t_REC": 0.36165496706962585,
          "t_mAP": 0.17695504426956177,
          "t_mAP25": 0.3135499656200409,
          "t_mAP50": 0.2499065399169922
        },
        {
          "T": 3,
          "current_stage_AP": 0.45642709732055664,
          "logical_unit_count": 36,
          "method": "R1-D0",
          "reference": "137a8158-1db5-2cc0-8003-31c12610471e",
          "reference_count": 1,
          "source_method": "PARENT",
          "t_REC": 0.33247724175453186,
          "t_mAP": 0.19910460710525513,
          "t_mAP25": 0.4316999614238739,
          "t_mAP50": 0.3180268108844757
        },
        {
          "T": 3,
          "current_stage_AP": 0.6096867322921753,
          "logical_unit_count": 15,
          "method": "R1-D0",
          "reference": "280d8ebb-6cc6-2788-9153-98959a2da801",
          "reference_count": 1,
          "source_method": "PARENT",
          "t_REC": 0.42408856749534607,
          "t_mAP": 0.3373963236808777,
          "t_mAP25": 0.6615177989006042,
          "t_mAP50": 0.5719261765480042
        },
        {
          "T": 3,
          "current_stage_AP": 0.4259970784187317,
          "logical_unit_count": 15,
          "method": "R1-D0",
          "reference": "5630cfcf-12bf-2860-8784-83d28a611a83",
          "reference_count": 1,
          "source_method": "PARENT",
          "t_REC": 0.29547327756881714,
          "t_mAP": 0.17541787028312683,
          "t_mAP25": 0.6319735050201416,
          "t_mAP50": 0.25958192348480225
        },
        {
          "T": 3,
          "current_stage_AP": 0.4988309144973755,
          "logical_unit_count": 15,
          "method": "R1-D0",
          "reference": "8eabc45f-5af7-2f32-8528-640861d2a135",
          "reference_count": 1,
          "source_method": "PARENT",
          "t_REC": 0.4373277425765991,
          "t_mAP": 0.25786519050598145,
          "t_mAP25": 0.5747599601745605,
          "t_mAP50": 0.3783983290195465
        },
        {
          "T": 3,
          "current_stage_AP": 0.35718217492103577,
          "logical_unit_count": 18,
          "method": "R1-D0",
          "reference": "ddc73797-765b-241a-9e2c-097c5989baf6",
          "reference_count": 1,
          "source_method": "PARENT",
          "t_REC": 0.18742500245571136,
          "t_mAP": 0.09067471325397491,
          "t_mAP25": 0.2812190353870392,
          "t_mAP50": 0.21585139632225037
        },
        {
          "T": 4,
          "current_stage_AP": 0.36008191108703613,
          "logical_unit_count": 129,
          "method": "R1-D0",
          "reference": "all",
          "reference_count": 6,
          "source_method": "PARENT",
          "t_REC": 0.2181292474269867,
          "t_mAP": 0.08292324095964432,
          "t_mAP25": 0.29680272936820984,
          "t_mAP50": 0.146309956908226
        },
        {
          "T": 4,
          "current_stage_AP": 0.4710144102573395,
          "logical_unit_count": 30,
          "method": "R1-D0",
          "reference": "10b17940-3938-2467-8a7a-958300ba83d3",
          "reference_count": 1,
          "source_method": "PARENT",
          "t_REC": 0.23026138544082642,
          "t_mAP": 0.09981263428926468,
          "t_mAP25": 0.2156699299812317,
          "t_mAP50": 0.14578798413276672
        },
        {
          "T": 4,
          "current_stage_AP": 0.42922526597976685,
          "logical_unit_count": 36,
          "method": "R1-D0",
          "reference": "137a8158-1db5-2cc0-8003-31c12610471e",
          "reference_count": 1,
          "source_method": "PARENT",
          "t_REC": 0.26283082365989685,
          "t_mAP": 0.1467166393995285,
          "t_mAP25": 0.35830289125442505,
          "t_mAP50": 0.23590126633644104
        },
        {
          "T": 4,
          "current_stage_AP": 0.596367359161377,
          "logical_unit_count": 15,
          "method": "R1-D0",
          "reference": "280d8ebb-6cc6-2788-9153-98959a2da801",
          "reference_count": 1,
          "source_method": "PARENT",
          "t_REC": 0.34685182571411133,
          "t_mAP": 0.26808953285217285,
          "t_mAP25": 0.5545441508293152,
          "t_mAP50": 0.4524463415145874
        },
        {
          "T": 4,
          "current_stage_AP": 0.362910658121109,
          "logical_unit_count": 15,
          "method": "R1-D0",
          "reference": "5630cfcf-12bf-2860-8784-83d28a611a83",
          "reference_count": 1,
          "source_method": "PARENT",
          "t_REC": 0.2522633671760559,
          "t_mAP": 0.18792462348937988,
          "t_mAP25": 0.6642701029777527,
          "t_mAP50": 0.22364290058612823
        },
        {
          "T": 4,
          "current_stage_AP": 0.45565325021743774,
          "logical_unit_count": 15,
          "method": "R1-D0",
          "reference": "8eabc45f-5af7-2f32-8528-640861d2a135",
          "reference_count": 1,
          "source_method": "PARENT",
          "t_REC": 0.29862359166145325,
          "t_mAP": 0.10316182672977448,
          "t_mAP25": 0.37758922576904297,
          "t_mAP50": 0.20070235431194305
        },
        {
          "T": 4,
          "current_stage_AP": 0.3901821970939636,
          "logical_unit_count": 18,
          "method": "R1-D0",
          "reference": "ddc73797-765b-241a-9e2c-097c5989baf6",
          "reference_count": 1,
          "source_method": "PARENT",
          "t_REC": 0.12145764380693436,
          "t_mAP": 0.06836818158626556,
          "t_mAP25": 0.20998281240463257,
          "t_mAP50": 0.1316554844379425
        },
        {
          "T": 5,
          "current_stage_AP": 0.3611888587474823,
          "logical_unit_count": 129,
          "method": "R1-D0",
          "reference": "all",
          "reference_count": 6,
          "source_method": "PARENT",
          "t_REC": 0.16761571168899536,
          "t_mAP": 0.05886579304933548,
          "t_mAP25": 0.24069426953792572,
          "t_mAP50": 0.10783883929252625
        },
        {
          "T": 5,
          "current_stage_AP": 0.5171102285385132,
          "logical_unit_count": 30,
          "method": "R1-D0",
          "reference": "10b17940-3938-2467-8a7a-958300ba83d3",
          "reference_count": 1,
          "source_method": "PARENT",
          "t_REC": 0.17051149904727936,
          "t_mAP": 0.07500172406435013,
          "t_mAP25": 0.17012116312980652,
          "t_mAP50": 0.11370131373405457
        },
        {
          "T": 5,
          "current_stage_AP": 0.42069509625434875,
          "logical_unit_count": 36,
          "method": "R1-D0",
          "reference": "137a8158-1db5-2cc0-8003-31c12610471e",
          "reference_count": 1,
          "source_method": "PARENT",
          "t_REC": 0.23192034661769867,
          "t_mAP": 0.12509414553642273,
          "t_mAP25": 0.3189285099506378,
          "t_mAP50": 0.224421426653862
        },
        {
          "T": 5,
          "current_stage_AP": 0.49656522274017334,
          "logical_unit_count": 15,
          "method": "R1-D0",
          "reference": "280d8ebb-6cc6-2788-9153-98959a2da801",
          "reference_count": 1,
          "source_method": "PARENT",
          "t_REC": 0.30638888478279114,
          "t_mAP": 0.2160715013742447,
          "t_mAP25": 0.5006078481674194,
          "t_mAP50": 0.38494595885276794
        },
        {
          "T": 5,
          "current_stage_AP": 0.4457601010799408,
          "logical_unit_count": 15,
          "method": "R1-D0",
          "reference": "5630cfcf-12bf-2860-8784-83d28a611a83",
          "reference_count": 1,
          "source_method": "PARENT",
          "t_REC": 0.21522632241249084,
          "t_mAP": 0.17537479102611542,
          "t_mAP25": 0.6370971202850342,
          "t_mAP50": 0.21246536076068878
        },
        {
          "T": 5,
          "current_stage_AP": 0.41476723551750183,
          "logical_unit_count": 15,
          "method": "R1-D0",
          "reference": "8eabc45f-5af7-2f32-8528-640861d2a135",
          "reference_count": 1,
          "source_method": "PARENT",
          "t_REC": 0.19068995118141174,
          "t_mAP": 0.06843206286430359,
          "t_mAP25": 0.2965795397758484,
          "t_mAP50": 0.1468885987997055
        },
        {
          "T": 5,
          "current_stage_AP": 0.3357519805431366,
          "logical_unit_count": 18,
          "method": "R1-D0",
          "reference": "ddc73797-765b-241a-9e2c-097c5989baf6",
          "reference_count": 1,
          "source_method": "PARENT",
          "t_REC": 0.07308368384838104,
          "t_mAP": 0.028750674799084663,
          "t_mAP25": 0.12768666446208954,
          "t_mAP50": 0.06001829355955124
        }
      ],
      "metrics": {
        "2": 0.24926671385765076,
        "3": 0.14994679391384125,
        "4": 0.08292324095964432,
        "5": 0.05886579304933548
      },
      "source_status": "PASS"
    },
    "scope": "PB only; all required prefixes and references must be covered",
    "tmap_all_T_vs_D0": false,
    "tmap_all_T_vs_native_FH": false
  },
  "profile": "COMPLETE"
}

逐 reference 原始行见 `confirmation/BY_REFERENCE.csv`；LOCAL 各 seed 的原始结果见 `confirmation/LOCAL-T2/METHODS.json`。未实现 pooled reference-block bootstrap 时不报告均值 AP 的置信区间。时延只接受真实 live profile；训练吞吐、CPU IoU microbenchmark 与多头共享前向不代表部署时延。

## R1 跨运行一致性

| role | status | published_output_equal | deltas |
| --- | --- | --- | --- |
| CAL | EXACT | True | {"2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0} |
| SEL | EXACT | True | {"2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0} |

差异诊断与原始运行证据保留在 `foundation/`；同一次前向中的零残差一致性不能替代跨运行一致性。

## LOCAL 人口元数据更正

继承的 DATA_ROLES 写为 41 refs；原始 native T2 的 154 条序列实际覆盖 46 refs。V1 bootstrap 将 ADDITIONAL reference 列表直接用作 LOCAL 列表。这里完整保留原始序列及 45/46/47 seeds，按实际 reference 数报告，并在 FINAL_LOCK 中绑定该元数据核对。核对只读取人口元数据，不含锁定前预测或评分；TRAIN overlap 为 0。详见 `foundation/LOCAL_POPULATION_AUDIT.json`；原始 DATA_ROLES 与 V1 历史文件保持不变。

## 父模型运行更正

固定 recipe 的 CAL/SEL 重跑状态为 COMPLETE；采用状态为 ADOPTED。原运行的差异原因仍为 UNDETERMINED，不能归因于已排除的线程数、零头或启动上下文。当前重跑已与修复器父模型逐字节一致；历史父模型上的关联结果保留但不参与当前排名，全部耗时仍计入预算。原始四项基线已保存在 `foundation/original-runtime/`，替换记录见 `foundation/BASELINE_SUPERSESSION.json`。

## 决策解释与训练边界

本轮未确认全面质量增益，最终锁定 R1-D0，不增加感知适配、修复器或关联模块。下表中的差值均为 0–1 AP，均值是不同 T 差值的算术平均，不是新的 pooled AP。

| 候选 | 实际主模型训练终点 | CAL 选择 | full / SEL 决定及原因 |
| --- | --- | --- | --- |
| C0-H | 导入 V1 750 | pilot 250 | 高 LR 控制；不再进行 full 训练 |
| S-BAL-H | 从保存的 350 恢复到 750 | pilot 250 | pilot 未超过 Q-SEM-L；379 仅为 V1 未保存进度，未用作恢复边界 |
| C0-L | 3000 | 250 | 同 LR full 控制；SEL mean −0.003017、min −0.009739、long −0.009432，未采纳 |
| S-BAL-L | 750 | 750 | pilot 排名未晋级，不称训练失败 |
| A-OPEN-L | 750 | 750 | pilot 排名未晋级，不称训练失败 |
| Q-SEM-L | 3000 | 250 | 唯一新机制 full 晋级；SEL mean −0.000542、min −0.011577、long −0.006102，未采纳 |
| S-WORST-L | 750 | 250 | pilot 排名未晋级，不称训练失败 |
| R1 + NEW_ONLY | 修复头 1500，父模型 0 | 修复头 0 | 与 parent 相同，未达到修复采纳门槛 |
| R1 + OLD_NEW | 修复头 1500，父模型 0 | 修复头 0 | 与 parent 相同，未证明相对 NEW-best 的历史证据收益 |
| A0-U τ=0.73 | 无训练参数 | 12 配置中的 CAL 冠军 | SEL mean +0.000242、min −0.000379、long −0.000111，保持 D0 |

H=5e−5、L=1e−5；新低 LR 配对从 R1 开始，不恢复高 LR optimizer。C0-L 与 Q-SEM-L 使用同一 3000 步计划，各完成 24,000 个 local batches / 96,000 个全局 draws；两个 rank 的 RNG、optimizer 和 scheduler 状态见各 `FULL_ENDPOINT_AUDIT.json`。Q-SEM 的 scorer 是固定的既有 500 步组件，不将它计入主模型 3000 步，也未在 seed46 重训。

R1 两修复头共享 32 个 TRAIN reference、64 个 episode、5,220 条训练候选与相同初始化 SHA；各完成 1500 步。NEW 修正 63,618 个原错误、误修 6,799 个原正确 segment；PAIR 分别为 63,616 和 6,795。训练段级纠错不等于完整 CAL/SEL pooled AP 增益；两头的训练点均未胜过 0 步。因此本轮既不声称一般修复收益，也不声称历史 soft 证据的额外收益。

P=R1，因此第二父模型的修复训练 NOT_APPLICABLE；表内 P/NOT_RUN 行仅表示未另造一份重复训练。最终未部署任何新增训练组件，seed46 适配复核 NOT_APPLICABLE；LOCAL 的评价 seeds45/46/47 不是三次训练。未选 LR 的辅助臂与全部 seed46 配方文件属于预先登记的可执行配置，不代表这些配方实际训练过。

## 确认结论与可复现性限制

FINAL_LOCK 于 2026-09-26T13:49:22.974496+00:00 固定，SHA256 为 `e474309f47e856332b8c49cee10d050fd91520d8572fa789ba317ad968504d26`。之后的正确性恢复未改变权重、阈值、人口、组件选择或最终配方。

PB 每个方法覆盖 129 个 logical units / 516 个 T2–T5 前缀、6 个 references。R1-D0 相对 FH-R1-native 的 T2/T3/T4/T5 差值依次为 +0.022170、+0.000267、−0.021255、−0.021346，因此没有全 T 超过 FH。ADDITIONAL 的 T2/T3/T4 分别覆盖 111/77/32 单元、40/23/8 references；它没有 T5，不与 PB 混为同一人口。

LOCAL 保留原始 154 个序列、三评价种子；实际覆盖 46 references（原元数据写 41，已在锁定前审计更正）。FH-R1-native 三种子平均 temporal mAP 为 0.305460；同 LR C0-best-native 为 0.302875。最终 P=R1，与自身原生基线没有新增感知收益。

真实 A40 profile 使用每个 PB reference 的首个 canonical master，完整序列 warmup 一次、测量三次。FINAL 与 R1-D0 身份相同，计时去重；两方法共 144 条测量。T4/T5 测量中位数的 D0/FH 时延比 0.462945、显存 allocated 峰值比 0.449286；这是描述性比较，不代表统计稳定性证明。冷输入 IO 单列，不把其等待混入部署前向时延。

原始 R1 历史预测无法复现，原因仍为 UNDETERMINED。旧结果保留在 `foundation/original-runtime/`，当前 CAL/SEL 对照来自更正后重复一致的 runtime；不能把历史数值直接作为当前因果比较。数据 staging 的 hardlink 元数据/大小检查不构成原始文件逐字节不可变证明。

750 步恢复时的 checkpoint 重写已修复；更正后的完整 CAL 750 重跑与此前输出 SHA、指标一致。原生 PB 顺序名称错误已修复并完整重跑；LOCAL CUDA 诊断 loss 的确定性兼容仅作用于 no-grad criterion。旧失败结果与费用均保留。LOCAL 预测导出另发生旧首条缓存冲突，新缓存现绑定执行源码、输入与数值环境。首次补导出的线程/cuBLAS 环境未匹配，已中断计费并归档排除，见 `confirmation/LOCAL_EXPORT_RECOVERY_AUDIT.json`；按原环境进行的正式补导出见 `confirmation/LOCAL_EXPORT_MATCHED_RUNTIME_AUDIT.json`。不能仅凭评分 COMPLETE 声称预测包完整。

正式补导出六组全部通过：每组154/154序列、46 references，所有六项指标与原确认逐项完全一致。此前环境不匹配的尝试被排除，未据其数值更改结论或选择。全部规定人口的预测资产现已完整准备；另保留六份明确标为 PARTIAL 的原失败导出，不能将这些诊断档案计作额外独立确认。

部署 bundle 使用完整替换张量及变化 buffer，不是浮点差分。FH、D0、C0-L 的真实 panel 重载输出 SHA 均与直接推理一致。基础 R1/Concerto 权重和 GT 不再分发。Git/Release 发布状态由最终 external receipt 确定；READY、模型本地可重载、Release 可下载是三个不同状态。

