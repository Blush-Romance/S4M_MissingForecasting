# 步骤22.21：Traffic原始结果

| job | 状态 | split | 原始MSE | 原始MAE | 多尺度MSE | 多尺度MAE | 样本 | batch | test访问 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| traffic_conditional_single_test_seed2026_formal | SUCCESS | test | 0.6720836875245719 | 0.3613795500442003 | 0.665930768052514 | 0.35818668122517666 | 3408 | 213 | True |
| traffic_pg95_multiscale_validation_seed2026_formal | SUCCESS | val | 0.5283183577465663 | 0.31501502000520865 | 0.522464090320265 | 0.31108324406668475 | 1648 | 103 | False |
| traffic_pg95_multiscale_validation_seed2026_smoke | SUCCESS | val | 0.5912915766008676 | 0.35474314373711635 | 0.588586080993196 | 0.35622934153105185 | 48 | 3 | False |
| traffic_pg95_seed2026_formal | SUCCESS | validation | 0.5283183610835206 | 0.3150150195182711 | None | None | 1648 | 103 | False |
| traffic_pg95_seed2026_smoke | SUCCESS | validation | 0.55747733690223 | 0.37543907077019284 | None | None | 48 | 3 | False |
| traffic_s4m_seed2026_formal | SUCCESS | validation | 0.5369708269505791 | 0.3169115486282794 | None | None | 1648 | 103 | False |
| traffic_s4m_seed2026_smoke | SUCCESS | validation | 0.5713774289443155 | 0.3733457647423426 | None | None | 48 | 3 | False |

## 唯一Traffic测试结论

- 决策：`TRAFFIC_TEST_MULTISCALE_CONFIRMED`
- 完整测试：213 batch / 3408 样本
- 多尺度相对PG95原始路径：MSE提升0.915499%、MAE提升0.883522%
- 多尺度相对原始+L5：MSE提升0.350510%、MAE提升0.444991%
