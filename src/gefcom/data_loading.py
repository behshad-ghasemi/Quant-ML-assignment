"""Load a task's raw CSVs into a clean, timestamp-indexed ``TaskBundle``.

Column conventions produced here:
    history        : DataFrame indexed by hourly DatetimeIndex, columns
                      ['ZONEID', 'LOAD', 'w1', ..., 'w25']. Covers every
                      hour strictly BEFORE the target month, in the exact
                      form available at forecast time (LOAD may be NaN for
                      the earliest years, see module note below).
    target_index   : DatetimeIndex of the target (forecast) month, taken
                      from the length of the benchmark template file.
    benchmark_q    : DataFrame (target_index x 99 quantile columns) -- the
                      official naive benchmark, for reference only.
    solution_load  : Series of true LOAD for the target month, or None if
                      no solution file is available for this task.
    solution_weather: DataFrame of true w1..w25 for the target month, or
                      None if no solution-temperature file is available.

Note on the leading NaN block: for every task, LOAD is NaN from the start
of the series (2001-01-01) until real metering data begins (empirically
2005-01-01 for the zone in this dataset). This is a genuine gap in the
*load* series, not a bug -- temperature stations were recorded from 2001,
but load metering only from 2005. Feature/label construction must treat
these as missing rows (drop from training targets), not impute them.

Note on cumulative history (`_build_cumulative_history`): empirically,
only Task 1's own train.csv is a full standalone history back to
2001-01-01. From Task 2 onward, each task's own train.csv appears to
contain only the newly-revealed increment since the previous task (about
one month), not the full history again -- confirmed by row counts (e.g.
Task 3's own file contributes ~720 rows, i.e. about one month, on top of
Task 1's ~85k-row base). `load_task` therefore reconstructs each task's
full history by chaining every earlier task's file onto Task 1's base,
rather than trusting any single Ln-train.csv (for n>1) to be complete on
its own. 
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .discovery import TaskPaths, discover_task
from .timestamps import hourly_index_from_start, reconstruct_hourly_index, verify_regular_hourly_grid

QUANTILE_COLUMNS = [f"{q/100:.2f}" for q in range(1, 100)]
WEATHER_COLUMNS = [f"w{i}" for i in range(1, 26)]


@dataclass
class TaskBundle:
    task_id: int
    zone_id: int
    history: pd.DataFrame
    target_index: pd.DatetimeIndex
    benchmark_q: pd.DataFrame
    solution_load: pd.Series | None
    solution_weather: pd.DataFrame | None



def _read_train(path, anchor_override: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    df = pd.read_csv(path)
    idx = reconstruct_hourly_index(df["TIMESTAMP"], anchor_override=anchor_override)
    verify_regular_hourly_grid(idx)
    df = df.drop(columns=["TIMESTAMP"]).set_index(idx)
    df.index.name = "timestamp"
    return df, idx


def _read_benchmark(path, target_start: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_csv(path)
    idx = hourly_index_from_start(target_start, len(df))
    q_cols = [c for c in df.columns if c not in ("ZONEID", "TIMESTAMP")]
    out = df[q_cols].copy()
    out.index = idx
    out.index.name = "timestamp"
    # normalise column labels to canonical "0.01".."0.99" strings
    out.columns = [f"{float(c):.2f}" for c in out.columns]
    out = out[QUANTILE_COLUMNS]
    return out


def _read_solution_plain(path, target_start: pd.Timestamp, expected_n: int) -> pd.Series:
    df = pd.read_csv(path)
    if len(df) != expected_n:
        raise ValueError(
            f"Solution file {path} has {len(df)} rows but benchmark template has {expected_n}; "
            "these must cover the same target month."
        )
    idx = hourly_index_from_start(target_start, len(df))
    s = pd.Series(df["LOAD"].to_numpy(), index=idx, name="LOAD")
    return s


def _read_solution_temperature(path, target_start: pd.Timestamp, expected_n: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    if len(df) != expected_n:
        raise ValueError(
            f"Solution-temperature file {path} has {len(df)} rows but benchmark template has "
            f"{expected_n}; these must cover the same target month."
        )
    idx = hourly_index_from_start(target_start, len(df))
    out = df[WEATHER_COLUMNS].copy()
    out.index = idx
    out.index.name = "timestamp"
    return out



def _build_cumulative_history(load_dir, task_id: int) -> tuple[pd.DataFrame, pd.DatetimeIndex | None, list[str]]:
    from .discovery import discover_task

    combined: pd.DataFrame | None = None
    own_task_index: pd.DatetimeIndex | None = None
    warnings_out: list[str] = []

    for tid in range(1, task_id + 1):
        paths = discover_task(load_dir, tid)

        anchor = None if combined is None else combined.index.max() + pd.Timedelta(hours=1)
        hist, idx = _read_train(paths.train_csv, anchor_override=anchor)

        if tid == task_id:
            own_task_index = idx

        if combined is None:
            combined = hist
            continue

        # A task file wildly longer than one month (e.g. the wrong file
        # dropped in that folder) is a different, real problem worth
        # flagging even though continuity-anchoring will still reconstruct
        # SOME index for it.
        if len(hist) > 2000:
            warnings_out.append(
                f"Task {tid}'s train.csv has {len(hist)} rows (expected ~720-745 for one "
                f"month) -- this looks like the wrong file was placed in Task {tid}'s folder. "
                f"Re-download/verify the Task {tid} folder."
            )

        combined = pd.concat([combined, hist])

    return combined, own_task_index, warnings_out


def load_task(load_dir, task_id: int, paths: TaskPaths | None = None) -> TaskBundle:
    if paths is None:
        paths = discover_task(load_dir, task_id)

    history, own_idx, warnings_out = _build_cumulative_history(load_dir, task_id)
    for w in warnings_out:
        warnings.warn(w, stacklevel=2)
    try:
        verify_regular_hourly_grid(history.index)
    except ValueError as e:
        warnings.warn(
            f"Task {task_id}: cumulative history has gap(s) -- {e}. Proceeding anyway; "
            f"features/models will treat this as a genuinely missing period, same as the "
            f"documented 2001-2005 LOAD gap.",
            stacklevel=2,
        )

    target_start = own_idx[-1] + pd.Timedelta(hours=1)

    zone_id = int(history["ZONEID"].iloc[0])
    if history["ZONEID"].nunique() != 1:
        raise ValueError(f"Task {task_id}: expected a single ZONEID, found {history['ZONEID'].unique()}")
    history = history.drop(columns=["ZONEID"])
    benchmark_q = _read_benchmark(paths.benchmark_csv, target_start)
    target_index = benchmark_q.index

    solution_load = None
    if paths.solution_csv is not None:
        solution_load = _read_solution_plain(paths.solution_csv, target_start, len(target_index))

    solution_weather = None
    if paths.solution_temperature_csv is not None:
        solution_weather = _read_solution_temperature(
            paths.solution_temperature_csv, target_start, len(target_index)
        )
        # Cross-check the solution-temperature file's own LOAD column against
        # the plain solution file when both exist -- cheap consistency check.
        df_check = pd.read_csv(paths.solution_temperature_csv)
        if solution_load is not None and not np.allclose(
            df_check["LOAD"].to_numpy(), solution_load.to_numpy(), equal_nan=True
        ):
            raise ValueError(
                f"Task {task_id}: LOAD in solution file and solution-temperature file disagree."
            )

    return TaskBundle(
        task_id=task_id,
        zone_id=zone_id,
        history=history,
        target_index=target_index,
        benchmark_q=benchmark_q,
        solution_load=solution_load,
        solution_weather=solution_weather,
    )
