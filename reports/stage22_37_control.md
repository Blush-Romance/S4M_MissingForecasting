# 步骤22.37 配对三seed复核

状态：PAIRED_THREE_SEED_COMPLETE；完整结束：True

仅validation；联合训练/掩码随机性；不访问test。

| seed | 方法 | MSE | MAE |
| --- | --- | ---: | ---: |
| 2027 | A_s4m | 0.534761380 | 0.326645003 |
| 2027 | B_pg95 | 0.533584491 | 0.321876765 |
| 2027 | C_s4m_cnm | 0.527417888 | 0.322658771 |
| 2027 | D_pg95_cnm | 0.525643405 | 0.318335866 |
| 2028 | A_s4m | 0.515788528 | 0.314668790 |
| 2028 | B_pg95 | 0.521511602 | 0.317144302 |
| 2028 | C_s4m_cnm | 0.511135801 | 0.312421980 |
| 2028 | D_pg95_cnm | 0.515863415 | 0.313442578 |
| 2029 | A_s4m | 0.531544676 | 0.325143267 |
| 2029 | B_pg95 | 0.538312534 | 0.331833910 |
| 2029 | C_s4m_cnm | 0.526687357 | 0.322312316 |
| 2029 | D_pg95_cnm | 0.532219122 | 0.328338935 |

```json
{
  "stage": "22.37",
  "queue_state": "SUCCESS",
  "completed": true,
  "decision": "PAIRED_THREE_SEED_COMPLETE",
  "test_accessed": false,
  "new_module": false,
  "randomness": "joint_training_and_missing_mask",
  "errors": [],
  "summary": {
    "cells": {
      "A_s4m": {
        "mse": {
          "mean": 0.5273648612912543,
          "sample_std": 0.010153591577390207
        },
        "mae": {
          "mean": 0.3221523531808534,
          "sample_std": 0.006524307845766577
        }
      },
      "B_pg95": {
        "mse": {
          "mean": 0.5311362088476542,
          "sample_std": 0.008663913327751769
        },
        "mae": {
          "mean": 0.32361832588597245,
          "sample_std": 0.0074980612035520575
        }
      },
      "C_s4m_cnm": {
        "mse": {
          "mean": 0.521747015212482,
          "sample_std": 0.00919683781037565
        },
        "mae": {
          "mean": 0.31913102235222796,
          "sample_std": 0.005812783061597949
        }
      },
      "D_pg95_cnm": {
        "mse": {
          "mean": 0.5245753140070559,
          "sample_std": 0.008230000336805774
        },
        "mae": {
          "mean": 0.320039126070745,
          "sample_std": 0.007592837713327912
        }
      }
    },
    "gains": {
      "pg95_gain_without_cnm_percent": {
        "mean": [
          -0.7209145424887854,
          -0.46156504775275636
        ],
        "sample_std": [
          0.8190214367268377,
          1.7811548442714993
        ],
        "dual_wins": 1
      },
      "pg95_gain_with_cnm_percent": {
        "mean": [
          -0.546256654643452,
          -0.28556794037189137
        ],
        "sample_std": [
          0.7670098950875632,
          1.6051864384181804
        ],
        "dual_wins": 1
      },
      "cnm_gain_on_s4m_percent": {
        "mean": [
          1.0630336069397124,
          0.9350192310395724
        ],
        "sample_std": [
          0.2687003201090914,
          0.2592256100526913
        ],
        "dual_wins": 3
      },
      "cnm_gain_on_pg95_percent": {
        "mean": [
          1.2344136884685233,
          1.1068382346959362
        ],
        "sample_std": [
          0.22118691300849003,
          0.05728718383461741
        ],
        "dual_wins": 3
      }
    },
    "metric_order": [
      "mse",
      "mae"
    ]
  },
  "scientific_conclusion": "PG95_INCREMENT_NOT_CONSISTENT"
}
```
