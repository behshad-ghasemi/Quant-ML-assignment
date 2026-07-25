"""Locate the train/benchmark/solution files for each task inside the
``GEFCom2014-L_V2/Load`` directory, tolerant of minor naming variations
(e.g. "Task 1" vs "Task1", "solution1_L.csv" vs "Solution1_L.csv")."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskPaths:
    task_id: int
    train_csv: Path
    benchmark_csv: Path
    solution_csv: Path | None
    solution_temperature_csv: Path | None


def _find_task_dir(load_dir: Path, task_id: int) -> Path:
    candidates = [
        load_dir / f"Task {task_id}",
        load_dir / f"Task{task_id}",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    # Fall back to a case-insensitive, whitespace-insensitive scan.
    pattern = re.compile(rf"^task\s*{task_id}$", re.IGNORECASE)
    for child in load_dir.iterdir():
        if child.is_dir() and pattern.match(child.name.strip()):
            return child
    raise FileNotFoundError(f"Could not find a 'Task {task_id}' directory under {load_dir}")


def _find_file(task_dir: Path, task_id: int, kind: str) -> Path:
    """kind is 'train' or 'benchmark'."""
    candidates = [
        task_dir / f"L{task_id}-{kind}.csv",
        task_dir / f"L{task_id}_{kind}.csv",
    ]
    for c in candidates:
        if c.is_file():
            return c
    pattern = re.compile(rf"^L{task_id}[-_]{kind}\.csv$", re.IGNORECASE)
    for child in task_dir.iterdir():
        if child.is_file() and pattern.match(child.name):
            return child
    raise FileNotFoundError(f"Could not find the '{kind}' file for task {task_id} in {task_dir}")


def _find_solution_dir(load_dir: Path) -> Path | None:
    for child in load_dir.iterdir():
        if child.is_dir() and "solution" in child.name.lower():
            return child
    return None


def _find_solution_files(solution_dir: Path, task_id: int) -> tuple[Path | None, Path | None]:
    plain_pat = re.compile(rf"^solution0*{task_id}_L\.csv$", re.IGNORECASE)
    temp_pat = re.compile(rf"^solution0*{task_id}_L_temperature\.csv$", re.IGNORECASE)
    plain, temp = None, None
    for child in solution_dir.iterdir():
        if not child.is_file():
            continue
        if temp_pat.match(child.name):
            temp = child
        elif plain_pat.match(child.name):
            plain = child
    return plain, temp


def discover_task(load_dir: Path, task_id: int) -> TaskPaths:
    load_dir = Path(load_dir)
    task_dir = _find_task_dir(load_dir, task_id)
    train_csv = _find_file(task_dir, task_id, "train")
    benchmark_csv = _find_file(task_dir, task_id, "benchmark")

    solution_csv = solution_temp_csv = None
    solution_dir = _find_solution_dir(load_dir)
    if solution_dir is not None:
        solution_csv, solution_temp_csv = _find_solution_files(solution_dir, task_id)

    return TaskPaths(
        task_id=task_id,
        train_csv=train_csv,
        benchmark_csv=benchmark_csv,
        solution_csv=solution_csv,
        solution_temperature_csv=solution_temp_csv,
    )


def discover_all_tasks(load_dir: Path, task_ids: list[int]) -> list[TaskPaths]:
    out = []
    for tid in task_ids:
        try:
            out.append(discover_task(load_dir, tid))
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Task {tid}: {e}\nExpected layout: <load_dir>/Task {tid}/L{tid}-train.csv "
                f"and L{tid}-benchmark.csv, plus optionally <load_dir>/Solution/solution{tid}_L.csv"
            ) from e
    return out
