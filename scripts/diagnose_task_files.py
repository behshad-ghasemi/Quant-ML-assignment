#!/usr/bin/env python3
"""Diagnostic: is each task's OWN train.csv a full standalone history back
to 2001, or just an incremental delta on top of the previous task? Reads
each file directly and independently -- no cumulative-history logic, no
assumptions -- so the answer is settled by direct observation.

If a task's file is a full history, its row count will be large (tens of
thousands) and its first reconstructed timestamp will be 2001-01-01
01:00. If it's an incremental delta, its row count will be small (roughly
one month, ~720-750, sometimes a bit more if it bundles several new
months) and its first timestamp will be somewhere in the middle of the
series.

Also flags any task whose OWN file's start date doesn't match what should
have been newly revealed at that point (i.e. roughly one month after the
previous task's own file ends) -- this is the Task 2 / Task 14 symptom,
and this script will tell you directly whether other tasks have the same
problem, without relying on the cumulative-history skip/gap warnings.

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
        "  - If most/all tasks show FULL HISTORY (tens of thousands of rows each): the "
        "original assumption (each task is independently complete) was right after all, "
        "and something else caused Task 2/14 specifically to be short -- worth "
        "re-downloading those two before trusting the cumulative-history codepath."
    )


if __name__ == "__main__":
    main()
