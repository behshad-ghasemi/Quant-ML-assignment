## Results

_(run `python scripts/run_backtest.py` first -- no pinball_summary.csv found)_

### Mean pinball loss across internal (history-only) folds — no Kaggle Solution files required

| model                          |   mean pinball loss |   std across folds |   n folds |
|:-------------------------------|--------------------:|-------------------:|----------:|
| linear_qr                      |              8.3537 |             3.4980 |   90.0000 |
| lightgbm                       |              8.4003 |             3.2146 |   90.0000 |
| xgboost                        |              8.4178 |             3.1644 |   90.0000 |
| baseline_empirical_climatology |              8.7050 |             4.3981 |   90.0000 |

