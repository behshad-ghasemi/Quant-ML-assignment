## Results

### Mean pinball loss across backtest tasks (lower is better)

| model                          |   mean pinball loss |   std across tasks |   n tasks |
|:-------------------------------|--------------------:|-------------------:|----------:|
| lightgbm__ens_baseline         |              8.1178 |             3.9093 |   15.0000 |
| xgboost__ens_baseline          |              8.1388 |             3.8261 |   15.0000 |
| xgboost__residual              |              8.1525 |             3.5703 |   15.0000 |
| lightgbm__residual             |              8.2226 |             3.8064 |   15.0000 |
| linear_qr__ens_baseline        |              8.2421 |             4.0959 |   15.0000 |
| baseline_empirical_climatology |              8.3273 |             4.5505 |   15.0000 |
| lightgbm                       |              8.4046 |             3.3334 |   15.0000 |
| linear_qr__residual            |              8.4309 |             3.5091 |   15.0000 |
| xgboost                        |              8.4373 |             3.1916 |   15.0000 |
| linear_qr                      |              8.5369 |             3.6329 |   15.0000 |
| benchmark_official             |             15.1433 |             7.5957 |   15.0000 |

### Calibration: empirical vs. nominal interval coverage

| model                          |   0.5 |   0.8 |   0.9 |   0.95 |   0.98 |
|:-------------------------------|------:|------:|------:|-------:|-------:|
| baseline_empirical_climatology | 0.521 | 0.778 | 0.863 |  0.897 |  0.933 |
| benchmark_official             | 0.003 | 0.003 | 0.003 |  0.003 |  0.003 |
| lightgbm                       | 0.435 | 0.729 | 0.826 |  0.889 |  0.937 |
| lightgbm__ens_baseline         | 0.475 | 0.775 | 0.858 |  0.902 |  0.946 |
| lightgbm__residual             | 0.494 | 0.750 | 0.843 |  0.897 |  0.952 |
| linear_qr                      | 0.434 | 0.692 | 0.793 |  0.866 |  0.936 |
| linear_qr__ens_baseline        | 0.480 | 0.759 | 0.854 |  0.908 |  0.952 |
| linear_qr__residual            | 0.444 | 0.699 | 0.809 |  0.881 |  0.943 |
| xgboost                        | 0.483 | 0.744 | 0.850 |  0.907 |  0.948 |
| xgboost__ens_baseline          | 0.505 | 0.782 | 0.870 |  0.912 |  0.951 |
| xgboost__residual              | 0.530 | 0.781 | 0.865 |  0.915 |  0.956 |

### Diebold-Mariano tests (model vs. baseline, on the hourly pinball-loss series)

| model                   | baseline                       |   dm_stat |   p_value |   mean_loss_diff |   n_obs | better                                      |
|:------------------------|:-------------------------------|----------:|----------:|-----------------:|--------:|:--------------------------------------------|
| linear_qr               | benchmark_official             |  -18.4456 |    0.0000 |          -6.5848 |   10968 | linear_qr (lower loss)                      |
| linear_qr               | baseline_empirical_climatology |    1.9467 |    0.0516 |           0.1975 |   10968 | baseline_empirical_climatology (lower loss) |
| linear_qr__residual     | benchmark_official             |  -18.8326 |    0.0000 |          -6.6910 |   10968 | linear_qr__residual (lower loss)            |
| linear_qr__residual     | baseline_empirical_climatology |    0.9357 |    0.3494 |           0.0913 |   10968 | baseline_empirical_climatology (lower loss) |
| lightgbm                | benchmark_official             |  -19.9834 |    0.0000 |          -6.7259 |   10968 | lightgbm (lower loss)                       |
| lightgbm                | baseline_empirical_climatology |    0.4544 |    0.6495 |           0.0565 |   10968 | baseline_empirical_climatology (lower loss) |
| lightgbm__residual      | benchmark_official             |  -20.8163 |    0.0000 |          -6.9041 |   10968 | lightgbm__residual (lower loss)             |
| lightgbm__residual      | baseline_empirical_climatology |   -1.1848 |    0.2361 |          -0.1217 |   10968 | lightgbm__residual (lower loss)             |
| xgboost                 | benchmark_official             |  -19.2443 |    0.0000 |          -6.6914 |   10968 | xgboost (lower loss)                        |
| xgboost                 | baseline_empirical_climatology |    0.7454 |    0.4561 |           0.0910 |   10968 | baseline_empirical_climatology (lower loss) |
| xgboost__residual       | benchmark_official             |  -20.5099 |    0.0000 |          -6.9794 |   10968 | xgboost__residual (lower loss)              |
| xgboost__residual       | baseline_empirical_climatology |   -1.8471 |    0.0648 |          -0.1971 |   10968 | xgboost__residual (lower loss)              |
| linear_qr__ens_baseline | benchmark_official             |  -18.8468 |    0.0000 |          -6.8737 |   10968 | linear_qr__ens_baseline (lower loss)        |
| linear_qr__ens_baseline | baseline_empirical_climatology |   -1.7949 |    0.0727 |          -0.0914 |   10968 | linear_qr__ens_baseline (lower loss)        |
| lightgbm__ens_baseline  | benchmark_official             |  -19.7896 |    0.0000 |          -7.0015 |   10968 | lightgbm__ens_baseline (lower loss)         |
| lightgbm__ens_baseline  | baseline_empirical_climatology |   -3.5016 |    0.0005 |          -0.2191 |   10968 | lightgbm__ens_baseline (lower loss)         |
| xgboost__ens_baseline   | benchmark_official             |  -19.4605 |    0.0000 |          -6.9803 |   10968 | xgboost__ens_baseline (lower loss)          |
| xgboost__ens_baseline   | baseline_empirical_climatology |   -3.1665 |    0.0015 |          -0.1980 |   10968 | xgboost__ens_baseline (lower loss)          |

### Mean pinball loss across internal (history-only) folds — no Kaggle Solution files required

| model                          |   mean pinball loss |   std across folds |   n folds |
|:-------------------------------|--------------------:|-------------------:|----------:|
| linear_qr                      |              8.3537 |             3.4980 |   90.0000 |
| lightgbm                       |              8.4003 |             3.2146 |   90.0000 |
| xgboost                        |              8.4178 |             3.1644 |   90.0000 |
| baseline_empirical_climatology |              8.7050 |             4.3981 |   90.0000 |

