# 步骤22.41：最小距离修复三seed复核

状态：BUG_FIX_NOT_THREE_SEED_DUAL_GAIN_ANALYZE_NO_AUTO_EXPANSION

仅validation（验证集）；基线错误修复，不是创新模块。CRN（共同随机数）固定原始单路径为主要比较。
先固定2028/2029方案；不根据结果选seed、调整模型、训练预算或评估口径。

| Seed（随机种子） | 原始MSE | 修复MSE | MSE降低% | 原始MAE | 修复MAE | MAE降低% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2027 | 0.534761380 | 0.520650410 | 2.6387 | 0.326645003 | 0.316614357 | 3.0708 |
| 2028 | 0.515788528 | 0.516765068 | -0.1893 | 0.314668790 | 0.316654375 | -0.6310 |
| 2029 | 0.531544676 | 0.515420851 | 3.0334 | 0.325143267 | 0.314692281 | 3.2143 |

MSE：均方误差；MAE：平均绝对误差。均值与样本标准差仅作三seed描述，不作显著性或分区保证。

```json
{
  "stage": "22.41",
  "queue_state": "SUCCESS",
  "completed": true,
  "decision": "BUG_FIX_NOT_THREE_SEED_DUAL_GAIN_ANALYZE_NO_AUTO_EXPANSION",
  "new_research_innovation": false,
  "test_accessed": false,
  "original_paper_same_protocol": false,
  "original_paper_note": "Traffic official hyperparameters unavailable; fixed local baseline comparison only",
  "jobs": [
    {
      "id": "fixed_2028_eval_formal",
      "state": "SUCCESS",
      "run_dir": "/mnt/data/zza/code/S4M_MissingForecasting/runs/cnm_se_crn_v1_s4m__traffic__validation__seed2028__20261004_235635",
      "attempt_count": 1,
      "oom_retry_count": 0,
      "required_memory_estimate_mib": 4608,
      "failure_summary": null
    },
    {
      "id": "fixed_2028_eval_smoke",
      "state": "SUCCESS",
      "run_dir": "/mnt/data/zza/code/S4M_MissingForecasting/runs/cnm_se_crn_v1_s4m__traffic__validation__seed2028__smoke__20261004_231742",
      "attempt_count": 1,
      "oom_retry_count": 0,
      "required_memory_estimate_mib": 4608,
      "failure_summary": null
    },
    {
      "id": "fixed_2028_train_formal",
      "state": "SUCCESS",
      "run_dir": "/mnt/data/zza/code/S4M_MissingForecasting/runs/s4m_min_distance_fixed__traffic__point__mr0.06__seed2028__20261004_231820",
      "attempt_count": 1,
      "oom_retry_count": 0,
      "required_memory_estimate_mib": 6144,
      "failure_summary": null
    },
    {
      "id": "fixed_2028_train_smoke",
      "state": "SUCCESS",
      "run_dir": "/mnt/data/zza/code/S4M_MissingForecasting/runs/s4m_min_distance_fixed__traffic__point__mr0.06__seed2028__smoke__20261004_231539",
      "attempt_count": 1,
      "oom_retry_count": 0,
      "required_memory_estimate_mib": 6144,
      "failure_summary": null
    },
    {
      "id": "fixed_2029_eval_formal",
      "state": "SUCCESS",
      "run_dir": "/mnt/data/zza/code/S4M_MissingForecasting/runs/cnm_se_crn_v1_s4m__traffic__validation__seed2029__20261004_235029",
      "attempt_count": 1,
      "oom_retry_count": 0,
      "required_memory_estimate_mib": 4608,
      "failure_summary": null
    },
    {
      "id": "fixed_2029_eval_smoke",
      "state": "SUCCESS",
      "run_dir": "/mnt/data/zza/code/S4M_MissingForecasting/runs/cnm_se_crn_v1_s4m__traffic__validation__seed2029__smoke__20261004_231800",
      "attempt_count": 1,
      "oom_retry_count": 0,
      "required_memory_estimate_mib": 4608,
      "failure_summary": null
    },
    {
      "id": "fixed_2029_train_formal",
      "state": "SUCCESS",
      "run_dir": "/mnt/data/zza/code/S4M_MissingForecasting/runs/s4m_min_distance_fixed__traffic__point__mr0.06__seed2029__20261004_231820",
      "attempt_count": 1,
      "oom_retry_count": 0,
      "required_memory_estimate_mib": 6144,
      "failure_summary": null
    },
    {
      "id": "fixed_2029_train_smoke",
      "state": "SUCCESS",
      "run_dir": "/mnt/data/zza/code/S4M_MissingForecasting/runs/s4m_min_distance_fixed__traffic__point__mr0.06__seed2029__smoke__20261004_231539",
      "attempt_count": 1,
      "oom_retry_count": 0,
      "required_memory_estimate_mib": 6144,
      "failure_summary": null
    }
  ],
  "records": [
    {
      "seed": 2027,
      "comparison": {
        "fixed_evaluation_assets": "PASS",
        "metrics_roundtrip": "PASS",
        "sample_count": 1648,
        "baseline_eval": "/mnt/data/zza/code/S4M_MissingForecasting/runs/cnm_se_crn_v1_s4m__traffic__validation__seed2027__20260908_151505",
        "fixed_eval": "/mnt/data/zza/code/S4M_MissingForecasting/runs/cnm_se_crn_v1_s4m__traffic__validation__seed2027__20261004_230031",
        "primary_original_path": {
          "baseline": {
            "mse": 0.5347613803970451,
            "mae": 0.32664500314070355
          },
          "fixed": {
            "mse": 0.5206504102513323,
            "mae": 0.3166143573335947
          },
          "absolute_change": {
            "mse": -0.014110970145712831,
            "mae": -0.010030645807108829
          },
          "relative_gain_percent": {
            "mse": 2.638741439263216,
            "mae": 3.0708095059357423
          },
          "dual_win": true
        },
        "secondary_inherited_CNM": {
          "baseline": {
            "mse": 0.5274178883158254,
            "mae": 0.32265877134535237
          },
          "fixed": {
            "mse": 0.5149421325115996,
            "mae": 0.31441393318829625
          },
          "absolute_change": {
            "mse": -0.012475755804225774,
            "mae": -0.008244838157056122
          },
          "relative_gain_percent": {
            "mse": 2.36544039946463,
            "mae": 2.5552809621999706
          },
          "dual_win": true
        }
      },
      "pairing_checks": {
        "initialization": "PASS",
        "parameter_count": "PASS",
        "training_rng": "PASS",
        "sampler_prefix": "PASS",
        "original_epochs": 4,
        "fixed_epochs": 4,
        "identical_max_budget_and_early_stopping": true,
        "identical_actual_epochs_or_time_not_required": true
      },
      "cost": {
        "original": {
          "parameter_count": 28371290,
          "epochs_completed": 4,
          "optimizer_steps": 2992,
          "validation_elapsed_seconds": 18.096880878321826,
          "peak_gpu_memory_mb": 8007.2197265625,
          "peak_gpu_reserved_mb": 11870.0,
          "resource_usage": {
            "device": "cuda",
            "elapsed_seconds": 1913.2666697837412,
            "gpu_uuid": "GPU-c88b779d-77c7-92b9-e031-943bae990121",
            "peak_gpu_memory_mb": 8007.2197265625,
            "peak_gpu_reserved_mb": 11870.0,
            "return_code": 0
          }
        },
        "fixed": {
          "parameter_count": 28371290,
          "epochs_completed": 4,
          "optimizer_steps": 2992,
          "validation_elapsed_seconds": 16.33434213604778,
          "peak_gpu_memory_mb": 8005.2822265625,
          "peak_gpu_reserved_mb": 8670.0,
          "resource_usage": {
            "device": "cuda",
            "elapsed_seconds": 2089.97384607594,
            "gpu_uuid": "GPU-c88b779d-77c7-92b9-e031-943bae990121",
            "peak_gpu_memory_mb": 8005.2822265625,
            "peak_gpu_reserved_mb": 8670.0,
            "return_code": 0
          }
        }
      }
    },
    {
      "seed": 2028,
      "pairing_checks": {
        "initialization": "PASS",
        "parameter_count": "PASS",
        "training_rng": "PASS",
        "sampler_prefix": "PASS",
        "original_epochs": 4,
        "fixed_epochs": 4,
        "identical_max_budget_and_early_stopping": true,
        "identical_actual_epochs_or_time_not_required": true
      },
      "cost": {
        "original": {
          "parameter_count": 28371290,
          "epochs_completed": 4,
          "optimizer_steps": 2992,
          "validation_elapsed_seconds": 23.19135170802474,
          "peak_gpu_memory_mb": 8007.9462890625,
          "peak_gpu_reserved_mb": 11870.0,
          "resource_usage": {
            "device": "cuda",
            "elapsed_seconds": 3028.115309322253,
            "gpu_uuid": "GPU-a3af3617-5bcd-4c65-a68e-17e716b31edb",
            "peak_gpu_memory_mb": 8007.9462890625,
            "peak_gpu_reserved_mb": 11870.0,
            "return_code": 0
          }
        },
        "fixed": {
          "parameter_count": 28371290,
          "epochs_completed": 4,
          "optimizer_steps": 2992,
          "validation_elapsed_seconds": 16.368019751971588,
          "peak_gpu_memory_mb": 8006.95458984375,
          "peak_gpu_reserved_mb": 11868.0,
          "resource_usage": {
            "device": "cuda",
            "elapsed_seconds": 2284.694451135001,
            "gpu_uuid": "GPU-a3af3617-5bcd-4c65-a68e-17e716b31edb",
            "peak_gpu_memory_mb": 8006.95458984375,
            "peak_gpu_reserved_mb": 11868.0,
            "return_code": 0
          }
        }
      },
      "comparison": {
        "fixed_evaluation_assets": "PASS",
        "metrics_roundtrip": "PASS",
        "sample_count": 1648,
        "baseline_eval": "/mnt/data/zza/code/S4M_MissingForecasting/runs/cnm_se_crn_v1_s4m__traffic__validation__seed2028__20260919_204034",
        "fixed_eval": "/mnt/data/zza/code/S4M_MissingForecasting/runs/cnm_se_crn_v1_s4m__traffic__validation__seed2028__20261004_235635",
        "primary_original_path": {
          "baseline": {
            "mse": 0.5157885276869995,
            "mae": 0.3146687898905184
          },
          "fixed": {
            "mse": 0.5167650684180275,
            "mae": 0.3166543751345689
          },
          "absolute_change": {
            "mse": 0.0009765407310280416,
            "mae": 0.001985585244050514
          },
          "relative_gain_percent": {
            "mse": -0.18932967264844722,
            "mae": -0.6310080020142295
          },
          "dual_win": false
        },
        "secondary_inherited_CNM": {
          "baseline": {
            "mse": 0.5111358006036197,
            "mae": 0.3124219797895297
          },
          "fixed": {
            "mse": 0.510395580366765,
            "mae": 0.31394073515773524
          },
          "absolute_change": {
            "mse": -0.0007402202368547162,
            "mae": 0.00151875536820556
          },
          "relative_gain_percent": {
            "mse": 0.14481870297102295,
            "mae": -0.4861230855872255
          },
          "dual_win": false
        }
      }
    },
    {
      "seed": 2029,
      "pairing_checks": {
        "initialization": "PASS",
        "parameter_count": "PASS",
        "training_rng": "PASS",
        "sampler_prefix": "PASS",
        "original_epochs": 4,
        "fixed_epochs": 4,
        "identical_max_budget_and_early_stopping": true,
        "identical_actual_epochs_or_time_not_required": true
      },
      "cost": {
        "original": {
          "parameter_count": 28371290,
          "epochs_completed": 4,
          "optimizer_steps": 2992,
          "validation_elapsed_seconds": 40.156210917048156,
          "peak_gpu_memory_mb": 5559.1201171875,
          "peak_gpu_reserved_mb": 7458.0,
          "resource_usage": {
            "device": "cuda",
            "elapsed_seconds": 4755.2090320028365,
            "gpu_uuid": "GPU-a3af3617-5bcd-4c65-a68e-17e716b31edb",
            "peak_gpu_memory_mb": 5559.1201171875,
            "peak_gpu_reserved_mb": 7458.0,
            "return_code": 0
          }
        },
        "fixed": {
          "parameter_count": 28371290,
          "epochs_completed": 4,
          "optimizer_steps": 2992,
          "validation_elapsed_seconds": 16.576508742989972,
          "peak_gpu_memory_mb": 8005.28125,
          "peak_gpu_reserved_mb": 8670.0,
          "resource_usage": {
            "device": "cuda",
            "elapsed_seconds": 1920.1546473000199,
            "gpu_uuid": "GPU-c88b779d-77c7-92b9-e031-943bae990121",
            "peak_gpu_memory_mb": 8005.28125,
            "peak_gpu_reserved_mb": 8670.0,
            "return_code": 0
          }
        }
      },
      "comparison": {
        "fixed_evaluation_assets": "PASS",
        "metrics_roundtrip": "PASS",
        "sample_count": 1648,
        "baseline_eval": "/mnt/data/zza/code/S4M_MissingForecasting/runs/cnm_se_crn_v1_s4m__traffic__validation__seed2029__20260920_001842",
        "fixed_eval": "/mnt/data/zza/code/S4M_MissingForecasting/runs/cnm_se_crn_v1_s4m__traffic__validation__seed2029__20261004_235029",
        "primary_original_path": {
          "baseline": {
            "mse": 0.5315446757897183,
            "mae": 0.3251432665113381
          },
          "fixed": {
            "mse": 0.5154208514828585,
            "mae": 0.3146922805017688
          },
          "absolute_change": {
            "mse": -0.01612382430685977,
            "mae": -0.01045098600956934
          },
          "relative_gain_percent": {
            "mse": 3.0333902381591034,
            "mae": 3.2142710878513308
          },
          "dual_win": true
        },
        "secondary_inherited_CNM": {
          "baseline": {
            "mse": 0.5266873567180012,
            "mae": 0.32231231592180193
          },
          "fixed": {
            "mse": 0.5126118024412253,
            "mae": 0.3131234956964365
          },
          "absolute_change": {
            "mse": -0.01407555427677587,
            "mae": -0.009188820225365435
          },
          "relative_gain_percent": {
            "mse": 2.6724686091738103,
            "mae": 2.8509057120841725
          },
          "dual_win": true
        }
      }
    }
  ],
  "summary": {
    "seed_count": 3,
    "seeds": [
      2027,
      2028,
      2029
    ],
    "statistical_claim": "descriptive_three_seeds_not_significance_or_journal_guarantee",
    "primary_original_path": {
      "dual_win_count": 2,
      "baseline": {
        "mse": {
          "mean": 0.5273648612912543,
          "sample_std": 0.010153591577390207,
          "values": [
            0.5347613803970451,
            0.5157885276869995,
            0.5315446757897183
          ]
        },
        "mae": {
          "mean": 0.3221523531808534,
          "sample_std": 0.006524307845766577,
          "values": [
            0.32664500314070355,
            0.3146687898905184,
            0.3251432665113381
          ]
        }
      },
      "fixed": {
        "mse": {
          "mean": 0.5176121100507395,
          "sample_std": 0.002715728424178432,
          "values": [
            0.5206504102513323,
            0.5167650684180275,
            0.5154208514828585
          ]
        },
        "mae": {
          "mean": 0.3159870043233108,
          "sample_std": 0.0011214422350928183,
          "values": [
            0.3166143573335947,
            0.3166543751345689,
            0.3146922805017688
          ]
        }
      },
      "relative_gain_percent": {
        "mse": {
          "mean": 1.8276006682579575,
          "sample_std": 1.7578233468324145,
          "values": [
            2.638741439263216,
            -0.18932967264844722,
            3.0333902381591034
          ]
        },
        "mae": {
          "mean": 1.8846908639242812,
          "sample_std": 2.179839648789484,
          "values": [
            3.0708095059357423,
            -0.6310080020142295,
            3.2142710878513308
          ]
        }
      }
    },
    "secondary_inherited_CNM": {
      "dual_win_count": 2,
      "baseline": {
        "mse": {
          "mean": 0.521747015212482,
          "sample_std": 0.00919683781037565,
          "values": [
            0.5274178883158254,
            0.5111358006036197,
            0.5266873567180012
          ]
        },
        "mae": {
          "mean": 0.31913102235222796,
          "sample_std": 0.005812783061597949,
          "values": [
            0.32265877134535237,
            0.3124219797895297,
            0.32231231592180193
          ]
        }
      },
      "fixed": {
        "mse": {
          "mean": 0.5126498384398633,
          "sample_std": 0.0022735147138998797,
          "values": [
            0.5149421325115996,
            0.510395580366765,
            0.5126118024412253
          ]
        },
        "mae": {
          "mean": 0.31382605468082264,
          "sample_std": 0.0006528176919615422,
          "values": [
            0.31441393318829625,
            0.31394073515773524,
            0.3131234956964365
          ]
        }
      },
      "relative_gain_percent": {
        "mse": {
          "mean": 1.7275759038698209,
          "sample_std": 1.3792776544760865,
          "values": [
            2.36544039946463,
            0.14481870297102295,
            2.6724686091738103
          ]
        },
        "mae": {
          "mean": 1.640021196232306,
          "sample_std": 1.847218348902328,
          "values": [
            2.5552809621999706,
            -0.4861230855872255,
            2.8509057120841725
          ]
        }
      }
    }
  }
}
```
