#!/usr/bin/env python3
"""Generate the plots referenced in the README's Results section from the
CSVs already produced by scripts/run_backtest.py -- no refitting needed.

Produces (into --output-dir, default "outputs"):
    calibration_reliability.png   -- nominal vs. empirical quantile coverage
                                      per model, pooled across all backtest
                                      tasks that have ground truth. Directly
                                      addresses assignment requirement 3
                                      ("assess calibration").
    pinball_by_task.png           -- per-task mean pinball loss, one line per
                                      model, so fold-to-fold variance is
                                      visible at a glance (requirement 5,
                                      "do not rely on a single performance
                                      number").
    coverage_width_tradeoff.png   -- (optional/nice-to-have) empirical
                                      coverage vs. mean interval width for a
                                      few nominal intervals, one point per
                                      model -- a compact view of the
                                      calibration/sharpness tradeoff.

Usage:
    python scripts/make_plots.py --config configs/config.yaml --output-dir outputs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe; no display needed to save PNGs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gefcom.calibration import reliability_curve  # noqa: E402
from gefcom.data_loading import load_task  # noqa: E402
from gefcom.metrics import QUANTILE_LEVELS  # noqa: E402

QUANTILE_COLUMNS = [f"{q/100:.2f}" for q in range(1, 100)]

# Keep plots readable: show baselines + the headline models, skip the
# residual/ensemble/oracle variants which are secondary comparisons already
# covered in the tables.
DEFAULT_MODELS_TO_PLOT = [
    "benchmark_official",
    "baseline_empirical_climatology",
    "linear_qr",
    "lightgbm",
    "xgboost",
]


def load_predictions_with_ground_truth(load_dir, output_dir: Path, task_ids: list[int]):
    """Merge outputs/predictions.csv (already-computed forecasts) with
    freshly-loaded ground truth (solution files) for tasks that have one.
    Returns a dict: model_name -> (y_true concatenated, y_pred concatenated)."""
    preds_path = output_dir / "predictions.csv"
    if not preds_path.exists():
        raise FileNotFoundError(
            f"{preds_path} not found -- run scripts/run_backtest.py first."
        )
    preds_df = pd.read_csv(preds_path, parse_dates=["timestamp"])

    per_model_true: dict[str, list[np.ndarray]] = {}
    per_model_pred: dict[str, list[np.ndarray]] = {}

    for task_id in task_ids:
        try:
            bundle = load_task(load_dir, task_id)
        except FileNotFoundError:
            continue
        if bundle.solution_load is None:
            continue  # no ground truth for this task -- skip for calibration

        y_true = bundle.solution_load  # indexed by timestamp

        task_preds = preds_df[preds_df["task_id"] == task_id]
        for model_name, group in task_preds.groupby("model"):
            group = group.set_index("timestamp").sort_index()
            common_idx = group.index.intersection(y_true.index)
            if len(common_idx) == 0:
                continue
            y_arr = y_true.loc[common_idx].to_numpy()
            pred_arr = group.loc[common_idx, QUANTILE_COLUMNS].to_numpy(dtype=float)
            per_model_true.setdefault(model_name, []).append(y_arr)
            per_model_pred.setdefault(model_name, []).append(pred_arr)

    out = {}
    for model_name in per_model_true:
        y_true_all = np.concatenate(per_model_true[model_name])
        y_pred_all = np.concatenate(per_model_pred[model_name], axis=0)
        out[model_name] = (y_true_all, y_pred_all)
    return out


def plot_calibration(pooled: dict, models_to_plot: list[str], output_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="perfect calibration")

    for model_name in models_to_plot:
        if model_name not in pooled:
            continue
        y_true, y_pred = pooled[model_name]
        curve = reliability_curve(y_true, y_pred, QUANTILE_LEVELS)
        ax.plot(curve["nominal"], curve["empirical"], marker=".", markersize=3, label=model_name)

    ax.set_xlabel("Nominal quantile level")
    ax.set_ylabel("Empirical coverage (fraction of y_true <= predicted quantile)")
    ax.set_title("Calibration: nominal vs. empirical quantile coverage\n(pooled across all backtest tasks with ground truth)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    out_path = output_dir / "calibration_reliability.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_pinball_by_task(output_dir: Path, models_to_plot: list[str]):
    path = output_dir / "pinball_by_task.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run scripts/run_backtest.py first.")
    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(10, 5))
    for model_name in models_to_plot:
        sub = df[df["model"] == model_name].sort_values("task_id")
        if sub.empty:
            continue
        ax.plot(sub["task_id"], sub["mean_pinball_loss"], marker="o", label=model_name)

    ax.set_xlabel("Task ID (backtest fold)")
    ax.set_ylabel("Mean pinball loss")
    ax.set_title("Mean pinball loss per backtest task, by model")
    ax.set_xticks(sorted(df["task_id"].unique()))
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = output_dir / "pinball_by_task.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_coverage_width_tradeoff(output_dir: Path, models_to_plot: list[str]):
    path = output_dir / "coverage_summary.csv"
    if not path.exists():
        print(f"Skipping coverage/width tradeoff plot -- {path} not found.")
        return
    df = pd.read_csv(path)
    if "mean_interval_width" not in df.columns:
        print("Skipping coverage/width tradeoff plot -- coverage_summary.csv has no width column "
              "(regenerate it from coverage_by_task.csv if you need this plot).")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    for model_name in models_to_plot:
        sub = df[df["model"] == model_name].sort_values("nominal_coverage")
        if sub.empty:
            continue
        ax.plot(sub["mean_interval_width"], sub["empirical_coverage"], marker="o", label=model_name)
        for _, row in sub.iterrows():
            ax.annotate(f"{row['nominal_coverage']:.0%}",
                        (row["mean_interval_width"], row["empirical_coverage"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=7)

    ax.set_xlabel("Mean interval width (MW)")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Coverage vs. interval width\n(labels show nominal coverage level)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = output_dir / "coverage_width_tradeoff.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--output-dir", default=None, help="Defaults to paths.output_dir in the config")
    ap.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS_TO_PLOT),
                     help="Comma-separated model names to include in the plots")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    load_dir = cfg["paths"]["load_dir"]
    output_dir = Path(args.output_dir or cfg["paths"].get("output_dir", "outputs"))
    task_ids = cfg.get("tasks", list(range(1, 16)))
    models_to_plot = [m.strip() for m in args.models.split(",")]

    print("Loading predictions + ground truth for calibration plot ...")
    pooled = load_predictions_with_ground_truth(load_dir, output_dir, task_ids)
    if not pooled:
        print("No tasks with ground truth found -- skipping calibration plot. "
              "(Add Solution files to enable this.)")
    else:
        plot_calibration(pooled, models_to_plot, output_dir)

    print("Plotting per-task pinball loss ...")
    plot_pinball_by_task(output_dir, models_to_plot)

    print("Plotting coverage/width tradeoff ...")
    plot_coverage_width_tradeoff(output_dir, models_to_plot)

    print(f"\nAll plots saved under {output_dir}/ -- embed them in README.md, e.g.:\n"
          f"  ![calibration](outputs/calibration_reliability.png)\n"
          f"  ![pinball by task](outputs/pinball_by_task.png)")


if __name__ == "__main__":
    main()
