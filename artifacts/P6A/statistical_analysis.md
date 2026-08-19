# P6-A Statistical Analysis

Unit: 43 master sequences x 3 deterministic orders; uncertainty is clustered by the six reference scenes. Thresholds are preregistered.

```json
{
  "G6A-1": {
    "checks": {
      "T4": {
        "ci_high": -0.061791608597164156,
        "clusters": [
          "10b17940-3938-2467-8a7a-958300ba83d3",
          "137a8158-1db5-2cc0-8003-31c12610471e",
          "280d8ebb-6cc6-2788-9153-98959a2da801",
          "5630cfcf-12bf-2860-8784-83d28a611a83",
          "8eabc45f-5af7-2f32-8528-640861d2a135",
          "ddc73797-765b-241a-9e2c-097c5989baf6"
        ],
        "n_clusters": 6,
        "n_pairs": 129,
        "passed": true,
        "relative_reduction": 0.45869710586465484
      },
      "T5": {
        "ci_high": -0.08088021944908838,
        "clusters": [
          "10b17940-3938-2467-8a7a-958300ba83d3",
          "137a8158-1db5-2cc0-8003-31c12610471e",
          "280d8ebb-6cc6-2788-9153-98959a2da801",
          "5630cfcf-12bf-2860-8784-83d28a611a83",
          "8eabc45f-5af7-2f32-8528-640861d2a135",
          "ddc73797-765b-241a-9e2c-097c5989baf6"
        ],
        "n_clusters": 6,
        "n_pairs": 129,
        "passed": true,
        "relative_reduction": 0.5549798533030731
      }
    },
    "passed": true,
    "threshold": {
      "ci_high_max": 0.0,
      "relative_reduction": 0.2
    }
  },
  "G6A-2": {
    "checks": {
      "T3": {
        "accuracy": 0.7480916030534351,
        "accuracy_improved": true,
        "baseline_accuracy": 0.5675675675675675,
        "baseline_recall": 0.2048780487804878,
        "improved": true,
        "passed": true,
        "recall": 0.47804878048780486,
        "recall_improved": true
      },
      "T4": {
        "accuracy": 0.7138728323699421,
        "accuracy_improved": true,
        "baseline_accuracy": 0.49714285714285716,
        "baseline_recall": 0.14695945945945946,
        "improved": true,
        "passed": true,
        "recall": 0.4172297297297297,
        "recall_improved": true
      },
      "T5": {
        "accuracy": 0.7201986754966887,
        "accuracy_improved": true,
        "baseline_accuracy": 0.445993031358885,
        "baseline_recall": 0.12586037364798427,
        "improved": true,
        "passed": true,
        "recall": 0.4277286135693215,
        "recall_improved": true
      }
    },
    "passed": true,
    "threshold": {
      "accuracy_and_recall_must_improve": true,
      "accuracy_min": 0.7,
      "recall_min": 0.25
    }
  },
  "G6A-3": {
    "checks": {
      "fingerprints_equal": true,
      "numeric_equal": true,
      "raw_metric_range": 0.40534985065460205,
      "raw_metric_tree_shape_equal": true
    },
    "passed": true,
    "threshold": {
      "absolute_tolerance": 1e-12
    }
  },
  "G6A-4": {
    "checks": {
      "T2_t_REC": {
        "drop": -0.004686564207077026,
        "passed": true
      },
      "T2_t_mAP": {
        "drop": -0.0018029212951660156,
        "passed": true
      },
      "T4_t_REC": {
        "delta": 0.009948894381523132
      },
      "T4_t_mAP": {
        "delta": 0.020602256059646606
      },
      "T5_t_REC": {
        "delta": 0.027252502739429474
      },
      "T5_t_mAP": {
        "delta": 0.027889017015695572
      },
      "positive_long_horizon_delta": true
    },
    "passed": true,
    "threshold": {
      "T2_drop_max": 0.05,
      "long_horizon_delta": ">0"
    }
  },
  "G6A-5": {
    "checks": {
      "explainability_share": 1.0
    },
    "passed": true,
    "threshold": {
      "minimum": 0.9
    }
  },
  "overall_passed": true
}
```
