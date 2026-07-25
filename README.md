# GEFCom2014 Probabilistic Load Forecasting

Behshad Ghaseminezhadabdolmaleki;
Preferred: Beth Gasemin

## 1. Problem

This repository addresses the load-forecasting track of the Global Energy Forecasting Competition 2014 (GEFCom2014-L). Given up to ~10 years of hourly electricity load and hourly temperature readings from 25 weather stations for a single utility zone, the task is to produce **one-month-ahead hourly probabilistic forecasts** — the 1st through 99th percentile of load for every hour of a held-out target month — evaluated with pinball loss. The dataset is organized as 15 sequential monthly "tasks," each revealing one more month of history and asking for a forecast of the following month.

The central modelling challenge is that **temperature is a strong predictor of load, but the real future temperature is not available at forecast time** in a genuine one-month-ahead setting. This repository treats that constraint as a first-class design requirement rather than an afterthought: every feature used by the default models is verifiably computable using only information available strictly before the target month, and the cost of that constraint is measured explicitly (see the oracle-weather comparison in Results).

## 2. Repository structure

```
src/gefcom/
  timestamps.py        -- robust reconstruction of the dataset's ambiguous raw TIMESTAMP format
  discovery.py          -- locates train/benchmark/solution files per task, tolerant of naming variants
  data_loading.py        -- builds each task's cumulative history + target month into a TaskBundle
  features.py            -- calendar + climatology feature engineering (leakage-safe by construction)
  baselines.py            -- empirical-quantile climatology baseline
  quantile_models.py      -- knot-based quantile regression (linear / LightGBM / XGBoost), one interface
  metrics.py              -- pinball loss, monotonicity enforcement, coverage/interval-width utilities
  calibration.py          -- reliability-curve and interval-coverage diagnostics
  stats_tests.py           -- Diebold-Mariano test with HAC (Newey-West) variance correction
  pipeline.py              -- orchestrates one task's full fit -> predict -> evaluate run
  lstm_model.py            -- optional PyTorch sequence-model comparison, off by default

scripts/
  run_backtest.py                  -- full pipeline on real GEFCom2014 tasks, scored where a Solution file exists
  internal_multi_fold_backtest.py  -- history-only multi-fold backtest, no Solution files required
  tune_hyperparams.py              -- Optuna hyperparameter search using leakage-safe internal folds
  diagnose_task_files.py           -- audits whether each task's own file is a full history or an increment
  make_results_section.py          -- renders outputs/*.csv as markdown tables for this README
  run_lstm_experiment.py           -- optional, single-task LSTM comparison (exploratory only)

tests/
  test_timestamps.py            -- ambiguous-date parsing and hourly-grid reconstruction
  test_no_leakage.py             -- history/target separation, feature-column consistency, trend continuity
  test_metrics.py                 -- pinball loss, monotonicity, coverage/interval-width correctness
  test_stats_tests.py             -- Diebold-Mariano correctness, including HAC vs. naive variance
  test_pipeline_integration.py    -- end-to-end run on a small synthetic fixture

configs/config.yaml   -- all paths, model hyperparameters, and experiment settings (no hard-coded params)
```

## 3. Setup and reproduction

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
`requirements.txt` is included in this repository with exact pinned versions for every package the code imports (pandas, numpy, scikit-learn, lightgbm, xgboost, scipy, PyYAML, optuna, pytest, tabulate). PyTorch is intentionally not pinned, since the optional LSTM comparison (`src/gefcom/lstm_model.py`) lazy-imports it only if that path is used.

Download the GEFCom2014 dataset from Kaggle (https://www.kaggle.com/datasets/cthngon/gefcom2014-dataset/data) and use only the `GEFCom2014-L_V2` load-track files. Point `configs/config.yaml`'s `paths.load_dir` at the `Load` folder, e.g.:
```yaml
paths:
  load_dir: "C:/Users/<you>/.../GEFCom2014-L_V2/Load"
  output_dir: outputs
```

Run the tests:
```bash
pytest -q
```

Reproduce the results below, in order:
```bash
# 1. (optional) audit the raw task files for the known Task 2/Task 14 data-quality issue
python scripts/diagnose_task_files.py --config configs/config.yaml

# 2. hyperparameter tuning (leakage-safe internal folds from Task 1's own history)
python scripts/tune_hyperparams.py --config configs/config.yaml --family lightgbm --n-trials 30
python scripts/tune_hyperparams.py --config configs/config.yaml --family xgboost --n-trials 30
# copy the resulting outputs/best_params_*.json values into configs/config.yaml under models.<family>

# 3. real, scored backtest on the one task with a locally available Solution file
python scripts/run_backtest.py --config configs/config.yaml --tasks 15

# 4. multi-fold backtest across many tasks, using only historical data (no Solution files needed)
python scripts/internal_multi_fold_backtest.py --config configs/config.yaml --tasks 1,3,5,9,13,15 --n-folds 3

# 5. render the results tables below
python scripts/make_results_section.py --output-dir outputs

# 6. (optional) generate predictions for every task in the dataset, not just Task 15 --
#    only Task 15 is scored (it's the only task with a local Solution file), but this
#    demonstrates the pipeline runs end-to-end across the whole load track
python scripts/run_backtest.py --config configs/config.yaml --tasks 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

## 4. Methodology summary

- **Leakage-safe features by construction**: every feature (calendar cyclic encodings, a trend term, and month×hour×day-of-week load/weather climatology estimated only from data strictly before the target month) is computable for any hour of the target month without needing that month's own data. Recursive/lag-based forecasting is deliberately avoided — the whole target month is forecast in one shot. `tests/test_no_leakage.py` enforces this as a regression-tested invariant, not just a design intention.
- **Two baselines**: the official GEFCom2014 naive benchmark, and a hand-built empirical-quantile climatology baseline (per month×hour×day-of-week group, with a fallback chain for sparse groups). All model families are compared against both.
- **Three model families**, one shared interface: linear quantile regression, LightGBM, and XGBoost, each fit at 23 quantile knots (including both tails) and linearly interpolated to the full 1st–99th percentile grid.
- **Two complementary backtests**: (a) a real, scored evaluation on Task 15 (December 2011), the only task for which a Kaggle Solution file was available locally; (b) an internal, history-only multi-fold backtest (expanding-window, trailing months held out from each task's own training history) that needs no Solution files and provides the multi-fold statistical comparison the single real task alone cannot.
- **Statistical comparison**: a Diebold-Mariano test with Newey-West (HAC, Bartlett-kernel) variance correction and a small-sample Student-t adjustment, applied to the hourly pinball-loss differential — chosen because hourly load-forecast errors are strongly autocorrelated and a naive i.i.d. variance estimate would be anti-conservative.
- **Explicit leakage-impact diagnostic**: an oracle-weather variant (using the real, future temperature — illegal in a genuine forecast) is run separately and reported only as a diagnostic, quantifying how much forecast accuracy is currently bottlenecked by the unavailability of real weather forecasts rather than by the models themselves.

## 5. Results

### 5a. Multi-fold comparison (leakage-safe, no oracle weather)

Internal, history-only backtest — 6 tasks × 3 held-out months each, spanning Jul 2010–Sep 2011, 18 real (task, fold) pairs, 13,197 pooled hourly observations:

| Model | Mean pinball loss | Std across folds | vs. baseline (Diebold-Mariano) |
|---|---|---|---|
| linear_qr | 8.3515 | 3.2937 | beats baseline, p = 0.0132 |
| xgboost | 8.3588 | 2.8745 | beats baseline, p = 0.0278 |
| lightgbm | 8.3652 | 3.1885 | beats baseline, p = 0.0279 |
| baseline_empirical_climatology | 8.5753 | 4.0656 | — |

All three model families beat the empirical-climatology baseline by a statistically significant margin, and all three also show lower variance across folds than the baseline — the baseline's accuracy swings more from month to month than any of the fitted models'. The three model families perform almost identically to each other, suggesting that once the calendar + climatology feature set is in place, the choice between a linear quantile model and a gradient-boosted tree adds little on top.

### 5b. Task 15 (December 2011) — the one real, officially-scored held-out month

| Model | Mean pinball loss (leakage-safe) | Mean pinball loss (oracle future temperature — illegal, diagnostic only) |
|---|---|---|
| baseline_empirical_climatology | 8.7844 | — |
| linear_qr | 10.9010 | 5.3434 |
| lightgbm | 11.4218 | 3.5138 |
| xgboost | 11.2283 | 3.5703 |

On this single real target month, the baseline was ahead of all three fitted models under leakage-safe weather — the opposite of the multi-fold finding above. Access to the real future temperature (illegal in a genuine forecast, included purely as a diagnostic) cuts model error by roughly 60–70%, quantifying how much of the models' potential is currently bottlenecked by the unavailability of real weather forecasts.

**Reconciling 5a and 5b:** December is a holiday-dense month (Christmas, New Year's) not represented in the multi-fold sample (which spans Jul 2010–Sep 2011). The most likely explanation is that December is a genuinely harder, less-typical month for a feature set built around ordinary calendar/climatology structure, rather than the multi-fold finding being wrong. This is exactly why the assignment calls for evaluation across multiple folds rather than a single month — a single-December read gives the opposite conclusion from what holds on a broader, more representative sample. All statistical comparisons above use the Diebold-Mariano test described in Section 4; the pooled multi-fold test in 5a is the stronger piece of evidence given its larger, more diverse sample (n=13,197 across 18 folds vs. n=744 across 1 fold).

### 5c. Full-dataset prediction coverage

`outputs/predictions.csv` contains raw 99-quantile forecasts for **all 15 tasks**, not only Task 15 — the pipeline was run end-to-end across the entire load track (`--tasks 1,2,...,15`) to confirm it generalizes across every task's history length and target month, not just the one with a local Solution file. Only Task 15 contributes a scored loss (Section 5b), since it is the only task for which a local ground-truth file was available; predictions for tasks 1–14 are included as evidence the pipeline produces valid, leakage-safe forecasts across the full dataset, not as additional scored evidence.

### 5d. Calibration

The baseline's empirical interval coverage tracks its nominal targets closely (e.g. 93.0% empirical vs. 90% nominal, 94.8% vs. 95% nominal on Task 15). Model calibration is broadly reasonable but less consistent — see `outputs/coverage_summary.csv` for the full table across all nominal intervals and models.

## 6. Limitations and things tried that didn't fully pan out

- **Solution files (ground truth) were only available locally for Task 15** among the 15 tasks. The official multi-task comparison the assignment describes was therefore supplemented with an internal, history-only backtest rather than scoring official held-out months for tasks 1–14.
- **Two source files (Task 2, Task 14) are mislabeled/duplicated** in the downloaded dataset — both are detected and skipped automatically (see `diagnose_task_files.py` and the warnings in `data_loading._build_cumulative_history`), leaving two genuine one-month gaps in the cumulative history. Re-downloading these two folders from Kaggle would remove the gaps; not done here due to time constraints.
- **Hyperparameter tuning used folds drawn only from Task 1's own history** (Jan 2005–Sep 2010), which never reaches a winter/holiday month — the tuned hyperparameters may be mildly biased toward warm-season load patterns. Widening the tuning window to reach a winter fold is noted as future work.
- **A data artifact was found and fixed during development**: every train file's final row is the hour-24-to-midnight rollover into the next calendar date, which naive month-grouping reads as a spurious one-row "next month." Left unfixed, this silently consumed one requested fold per task in the internal multi-fold backtest without being replaced. The fix filters out any calendar month with fewer than 100 rows before selecting the trailing N folds; the multi-fold numbers above already reflect the fix.
- **Feature curation (a reduced, less redundant feature set for the tree-based models) and a finer day-of-week climatology grouping were tested and improved model accuracy** on Task 15 specifically, but did not close the gap to the baseline on that single month.
- **A legitimate (non-oracle) weather-persistence feature and a full 15-task × 6-fold internal backtest** were scoped but not run, due to time constraints; both are natural next steps.
- **Deep learning (LSTM)** was implemented as an optional, off-by-default comparison per the assignment's allowance, but was not prioritized for tuning, since the evidence above indicates the bottleneck is feature/weather information rather than model capacity — the assignment notes unnecessary complexity does not receive additional credit.

## 7. What I'd do with more time

- Re-download Task 2 / Task 14 to close the two data gaps.
- Widen hyperparameter tuning to include a winter fold, and re-tune.
- Run the internal multi-fold backtest across all 15 tasks at 6 folds each, rather than the 6-task subset used here.
- Test a legitimate weather-persistence (trailing degree-day trend) feature as a middle ground between climatology and oracle weather.
- Investigate why December specifically favors the baseline — likely a holiday-density effect — with a per-hour loss breakdown.
