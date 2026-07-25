#!/usr/bin/env python3
"""Render outputs/pinball_summary.csv, coverage_summary.csv and
diebold_mariano_tests.csv (produced by scripts/run_backtest.py) as
markdown tables, ready to paste into the README's Results section.

Usage:
    python scripts/make_results_section.py --output-dir outputs > results_section.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="outputs")
    args = ap.parse_args()
    out = Path(args.output_dir)

    print("## Results\n")

    pinball_path = out / "pinball_summary.csv"
    if pinball_path.exists():
        df = pd.read_csv(pinball_path, index_col=0)
        df.columns = ["mean pinball loss", "std across tasks", "n tasks"]
        print("### Mean pinball loss across backtest tasks (lower is better)\n")
        print(df.to_markdown(floatfmt=".4f"))
        print()
    else:
        print("_(run `python scripts/run_backtest.py` first -- no pinball_summary.csv found)_\n")

    cov_path = out / "coverage_summary.csv"
    if cov_path.exists():
        df = pd.read_csv(cov_path)
        piv = df.pivot(index="model", columns="nominal_coverage", values="empirical_coverage")
        print("### Calibration: empirical vs. nominal interval coverage\n")
        print(piv.to_markdown(floatfmt=".3f"))
        print()

    dm_path = out / "diebold_mariano_tests.csv"
    if dm_path.exists():
        df = pd.read_csv(dm_path)
        print("### Diebold-Mariano tests (model vs. baseline, on the hourly pinball-loss series)\n")
        print(df.to_markdown(index=False, floatfmt=".4f"))
        print()
        
    internal_path = out / "internal_backtest_summary.csv"
    if internal_path.exists():
        df = pd.read_csv(internal_path, index_col=0)
        df.columns = ["mean pinball loss", "std across folds", "n folds"]
        print("### Mean pinball loss across internal (history-only) folds — no Kaggle Solution files required\n")
        print(df.to_markdown(floatfmt=".4f"))
        print()


if __name__ == "__main__":
    main()
