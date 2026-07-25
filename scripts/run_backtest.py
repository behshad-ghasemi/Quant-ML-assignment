#!/usr/bin/env python3
"""Run the full rolling-origin backtest across the configured GEFCom2014-L
tasks: for each task, fit the baselines + models, predict 99 quantiles for
the target month, and (where a solution file is available) evaluate with
pinball loss, calibration, and Diebold-Mariano tests against baselines.

Usage:
    python scripts/run_backtest.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gefcom.pipeline import GBM_MODEL_FEATURES, LINEAR_MODEL_FEATURES, ModelSpec, run_task  # noqa: E402
from gefcom.stats_tests import diebold_mariano_test  # noqa: E402


def build_model_specs(cfg: dict) -> list[ModelSpec]:
    knots = np.array(cfg.get("quantile_knots", None) or
                      [0.01,0.02,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,
                       0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.98,0.99])
    specs = []
    models_cfg = cfg.get("models", {})

    if models_cfg.get("linear_qr", {}).get("enabled", True):
        p = models_cfg.get("linear_qr", {})
        specs.append(ModelSpec(
            name="linear_qr", family="linear_qr", knots=knots,
            params={"l1_reg": p.get("l1_reg", 0.001)},
            feature_subset=LINEAR_MODEL_FEATURES,
            max_train_rows=p.get("max_train_rows", 4000),
        ))
    if models_cfg.get("lightgbm", {}).get("enabled", True):
        p = models_cfg.get("lightgbm", {})
        specs.append(ModelSpec(name="lightgbm", family="lightgbm", knots=knots, params=p,
                                feature_subset=GBM_MODEL_FEATURES))
    if models_cfg.get("xgboost", {}).get("enabled", True):
        p = models_cfg.get("xgboost", {})
        specs.append(ModelSpec(name="xgboost", family="xgboost", knots=knots, params=p,
                                feature_subset=GBM_MODEL_FEATURES))
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--tasks", type=str, default=None,
                     help="Comma-separated task ids to override the config, e.g. --tasks 1,2,3 for a fast dev run")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    load_dir = cfg["paths"]["load_dir"]
    output_dir = Path(cfg["paths"].get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    task_ids = [int(t) for t in args.tasks.split(",")] if args.tasks else cfg.get("tasks", list(range(1, 16)))
    val_fraction = cfg.get("validation", {}).get("val_fraction", 0.1)
    early_stopping_rounds = cfg.get("validation", {}).get("early_stopping_rounds", 50)
    run_oracle = cfg.get("leakage", {}).get("run_oracle_comparison", False)

    specs = build_model_specs(cfg)
    print(f"Models: {[s.name for s in specs]}")
    print(f"Tasks: {task_ids}")

    all_mean_pinball = []
    all_per_hour_loss: dict[str, list[np.ndarray]] = {}
    all_coverage = []
    predictions_log = []

    for task_id in task_ids:
        t0 = time.time()
        print(f"\n=== Task {task_id} ===")
        try:
            result = run_task(
                load_dir, task_id, specs,
                val_fraction=val_fraction,
                early_stopping_rounds=early_stopping_rounds,
                run_oracle_weather=run_oracle,
            )
        except FileNotFoundError as e:
            print(f"  SKIPPING task {task_id}: {e}")
            continue

        print(f"  fit+predict done in {time.time()-t0:.1f}s "
              f"(has_ground_truth={result.has_ground_truth})")

        for name, arr in result.predictions.items():
            df = pd.DataFrame(arr, index=result.target_index,
                               columns=[f"{q/100:.2f}" for q in range(1, 100)])
            df.insert(0, "task_id", task_id)
            df.insert(1, "model", name)
            predictions_log.append(df.reset_index().rename(columns={"index": "timestamp"}))

        if result.has_ground_truth:
            for name, loss in result.mean_pinball.items():
                all_mean_pinball.append({"task_id": task_id, "model": name, "mean_pinball_loss": loss})
                all_per_hour_loss.setdefault(name, []).append(result.per_hour_pinball[name])
                print(f"    {name:35s} mean pinball = {loss:.4f}")
            for name, tbl in result.coverage_tables.items():
                tbl = tbl.copy()
                tbl.insert(0, "task_id", task_id)
                tbl.insert(1, "model", name)
                all_coverage.append(tbl)


    # -- Save raw predictions -------------------------------------------------
    if predictions_log:
        pd.concat(predictions_log, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)
        print(f"\nSaved predictions -> {output_dir/'predictions.csv'}")

    if not all_mean_pinball:
        print("\nNo tasks had ground truth available (no Solution files found) -- "
              "predictions were generated but no evaluation metrics could be computed. "
              "Add the Kaggle 'Solution to Task N' files to the Load/Solution folder to enable scoring.")
        return

    # -- Aggregate pinball loss ------------------------------------------------
    df_pinball = pd.DataFrame(all_mean_pinball)
    df_pinball.to_csv(output_dir / "pinball_by_task.csv", index=False)

    summary = (
        df_pinball.groupby("model")["mean_pinball_loss"]
        .agg(["mean", "std", "count"])
        .sort_values("mean")
    )
    summary.to_csv(output_dir / "pinball_summary.csv")
    print("\n=== Mean pinball loss across tasks (lower is better) ===")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    # -- Calibration ------------------------------------------------------------
    if all_coverage:
        df_cov = pd.concat(all_coverage, ignore_index=True)
        df_cov.to_csv(output_dir / "coverage_by_task.csv", index=False)
        cov_summary = df_cov.groupby(["model", "nominal_coverage"])["empirical_coverage"].mean().reset_index()
        cov_summary.to_csv(output_dir / "coverage_summary.csv", index=False)
        print("\n=== Mean empirical coverage by nominal interval ===")
        print(cov_summary.pivot(index="model", columns="nominal_coverage", values="empirical_coverage")
              .to_string(float_format=lambda x: f"{x:.3f}"))

    # -- Diebold-Mariano tests: every model vs. every baseline -----------------
    baseline_names = [n for n in all_per_hour_loss if n.startswith("benchmark_") or n.startswith("baseline_")]
    model_names = [n for n in all_per_hour_loss if n not in baseline_names]

    dm_rows = []
    for model_name in model_names:
        loss_model = np.concatenate(all_per_hour_loss[model_name])
        for baseline_name in baseline_names:
            loss_base = np.concatenate(all_per_hour_loss[baseline_name])
            if len(loss_model) != len(loss_base):
                continue
            res = diebold_mariano_test(loss_model, loss_base, model_name, baseline_name)
            dm_rows.append({
                "model": model_name, "baseline": baseline_name,
                "dm_stat": res.dm_stat, "p_value": res.p_value,
                "mean_loss_diff": res.mean_loss_diff, "n_obs": res.n_obs,
                "better": res.better_model,
            })
            print(f"\nDM test: {model_name} vs {baseline_name}\n  {res}")

    if dm_rows:
        pd.DataFrame(dm_rows).to_csv(output_dir / "diebold_mariano_tests.csv", index=False)

    print(f"\nAll results saved under {output_dir}/")


if __name__ == "__main__":
    main()
