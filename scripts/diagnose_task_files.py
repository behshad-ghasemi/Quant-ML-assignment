#!/usr/bin/env python3
"""Diagnostic: is each task's OWN train.csv a full standalone history back
to 2001, or just an incremental delta on top of the previous task? Reads
each file directly and independently -- no cumulative-history logic, no
assumptions -- so the answer is settled by direct observation.
Usage:
    python scripts/diagnose_task_files.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gefcom.data_loading import _read_train  # noqa: E402
from gefcom.discovery import discover_task  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--tasks", type=str, default=",".join(str(i) for i in range(1, 16)))
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    load_dir = cfg["paths"]["load_dir"]
    task_ids = [int(t) for t in args.tasks.split(",")]

    print(f"{'task':>4} | {'rows':>7} | {'file start':>19} | {'file end':>19} | verdict")
    print("-" * 90)

    prev_end = None
    for tid in task_ids:
        try:
            paths = discover_task(load_dir, tid)
            hist, idx = _read_train(paths.train_csv)
        except Exception as e:
            print(f"{tid:>4} | ERROR: {e}")
            continue

        n = len(hist)
        start, end = idx[0], idx[-1]
        if n > 20000:
            verdict = "FULL HISTORY (spans years)"
        else:
            verdict = "INCREMENTAL (spans ~weeks/months)"
            if prev_end is not None:
                gap = (start - prev_end).total_seconds() / 3600
                if gap != 1:
                    verdict += f"  <-- starts {gap:.0f}h after previous task's own file ended (expected 1h)"

        print(f"{tid:>4} | {n:>7} | {str(start):>19} | {str(end):>19} | {verdict}")
        prev_end = end

    print(
        "\nInterpretation:\n"
        "  - If Task 1 is FULL HISTORY and everything else is INCREMENTAL with no "
        "'starts Nh after' warnings: the dataset is incremental-by-design, and "
        "data_loading.py's _build_cumulative_history is the correct approach -- keep it.\n"
        "  - If a task shows 'starts Nh after' with N far from 1 (or 0): that task's own "
        "file is corrupted/mislabeled independent of anything downstream. Try "
        "re-downloading exactly that task's folder from Kaggle.\n"

    )


if __name__ == "__main__":
    main()
