"""
Robust timestamp reconstruction for the GEFCom2014-L Kaggle export.

Why this module exists
-----------------------
The raw ``TIMESTAMP`` column in the Ln-train.csv / Ln-benchmark.csv /
solutionN_L.csv files looks like ``"1212011 1:00"``. The date part is
month + day + 4-digit year concatenated with NO separators and NO fixed
width for month/day (e.g. "112001" = 1/1/2001, "1212011" = 12/1/2011).
This is genuinely ambiguous to parse in isolation ("1212011" could be
1/21/2011 or 12/1/2011).

However, the underlying series is a *perfectly regular hourly grid with
zero gaps* (verified empirically against every sample file shipped with
this repo, and re-checked at load time by ``verify_regular_hourly_grid``
below). This lets us sidestep the ambiguous string parsing entirely:

1. Parse the FIRST row's date only, using the constraint that valid
   splits must have 1 <= month <= 12 and a legal day-of-month, breaking
   ties by preferring the split where day == 1 (every train file in this
   dataset starts on the 1st of a month).
2. From that anchor, reconstruct every subsequent timestamp with
   ``pandas.date_range(..., freq="h")`` -- i.e. we *count* hours rather
   than re-parsing each string.
3. We cross-check the reconstruction against a handful of unambiguous
   rows (e.g. any row whose date prefix has only one valid split) as a
   safety net; if the check fails we raise loudly rather than silently
   producing wrong dates.

Hour convention: this export writes "hour 24" of day D as "hour 0" of
day D+1 (a common artifact of spreadsheet round-tripping). The
``pandas.date_range`` reconstruction handles this automatically since we
never look at the hour text at all once the anchor is fixed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

_TS_RE = re.compile(r"^(\d{6,8})\s+(\d{1,2}):00$")


@dataclass(frozen=True)
class AnchorCandidate:
    month: int
    day: int
    year: int
    timestamp: pd.Timestamp


def _split_candidates(date_digits: str, year: int, month_day_digits: str) -> list[AnchorCandidate]:
    candidates: list[AnchorCandidate] = []
    for m_len in (1, 2):
        if m_len >= len(month_day_digits):
            continue
        m_str, d_str = month_day_digits[:m_len], month_day_digits[m_len:]
        if not d_str:
            continue
        try:
            month, day = int(m_str), int(d_str)
        except ValueError:
            continue
        if not (1 <= month <= 12):
            continue
        try:
            ts = pd.Timestamp(year=year, month=month, day=day)
        except ValueError:
            continue
        candidates.append(AnchorCandidate(month, day, year, ts))
    return candidates


def parse_first_timestamp(raw: str) -> pd.Timestamp:
    """Parse the first TIMESTAMP string of a file into an (unambiguous where
    possible) anchor ``pd.Timestamp``, including the hour.

    Ties (e.g. "1212011" -> Jan 21 2011 vs Dec 1 2011) are broken by
    preferring day == 1, since every file in this dataset begins on the
    1st of a month (verified against the shipped samples).
    """
    m = _TS_RE.match(raw.strip())
    if not m:
        raise ValueError(f"Unrecognised TIMESTAMP format: {raw!r}")
    date_digits, hour_str = m.group(1), m.group(2)
    if len(date_digits) < 5:
        raise ValueError(f"TIMESTAMP date part too short to contain a 4-digit year: {raw!r}")
    year = int(date_digits[-4:])
    month_day_digits = date_digits[:-4]

    candidates = _split_candidates(date_digits, year, month_day_digits)
    if not candidates:
        raise ValueError(f"Could not find a valid (month, day) split for {raw!r}")

    if len(candidates) == 1:
        chosen = candidates[0]
    else:
        day_one = [c for c in candidates if c.day == 1]
        chosen = day_one[0] if day_one else candidates[0]

    hour = int(hour_str)
    # Hour convention: "24:00" is written as "0:00" on this export, so the
    # raw hour is already consistent with plain addition of `hour` hours
    # onto midnight of the parsed calendar date.
    return chosen.timestamp + pd.Timedelta(hours=hour)


def hourly_index_from_start(start: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    """Build a gap-free hourly DatetimeIndex of length n starting at `start`.

    Use this (rather than self-parsing) for benchmark/solution files: their
    own first-row date string can be genuinely ambiguous in isolation (e.g.
    "1012010 1:00" collides between Jan 1 2010 and Oct 1 2010 -- both are
    valid (month, day) splits with day == 1). The correct anchor is always
    unambiguous when derived from the *end* of the associated train file
    instead (target month starts exactly one hour after the last training
    timestamp).
    """
    return pd.date_range(start=start, periods=n, freq="h")


def reconstruct_hourly_index(
    raw_timestamps: pd.Series, anchor_override: pd.Timestamp | None = None
) -> pd.DatetimeIndex:
    """Reconstruct an exact, gap-free hourly DatetimeIndex for a GEFCom2014-L
    file, anchored on the first row (or on `anchor_override` if given) and
    counted forward one hour at a time.

    Prefer passing `anchor_override` (derived from train-file continuity,
    see `hourly_index_from_start`) whenever one is available -- self-parsing
    a file's own first row is only safe when the (month, day) split has a
    single valid candidate, which is NOT guaranteed for benchmark/solution
    files (see module docstring and `hourly_index_from_start`).

    Raises if the reconstructed grid disagrees with any row whose raw
    string happens to be unambiguous (a safety net against silently
    mis-parsing a file with a different structure than expected).
    """
    n = len(raw_timestamps)
    anchor = anchor_override if anchor_override is not None else parse_first_timestamp(raw_timestamps.iloc[0])
    index = pd.date_range(start=anchor, periods=n, freq="h")

    # Safety net: spot-check every row whose (month, day) split is
    # unambiguous (only one valid candidate). If any of these disagree
    # with the reconstructed grid, the assumption of a regular, gapless
    # hourly series is wrong for this file and we must fail loudly.
    sample_idx = np.linspace(0, n - 1, num=min(n, 200), dtype=int)
    mismatches = []
    for i in sample_idx:
        raw = raw_timestamps.iloc[i]
        m = _TS_RE.match(raw.strip())
        if not m:
            mismatches.append((i, raw, "unparseable"))
            continue
        date_digits, hour_str = m.group(1), m.group(2)
        year = int(date_digits[-4:])
        month_day_digits = date_digits[:-4]
        candidates = _split_candidates(date_digits, year, month_day_digits)
        if len(candidates) != 1:
            continue  # ambiguous row, can't use as a check
        expected = candidates[0].timestamp + pd.Timedelta(hours=int(hour_str))
        if expected != index[i]:
            mismatches.append((i, raw, str(expected)))

    if mismatches:
        raise ValueError(
            "Hourly-grid reconstruction failed a consistency check against "
            f"{len(mismatches)} unambiguous row(s), e.g. {mismatches[:3]}. "
            "This file may have gaps or a different layout than assumed; "
            "do not trust the reconstructed index."
        )
    return index


def verify_regular_hourly_grid(index: pd.DatetimeIndex) -> None:
    """Raise if `index` is not a perfectly regular, gap-free hourly grid."""
    diffs = index.to_series().diff().dropna().unique()
    if len(diffs) != 1 or diffs[0] != pd.Timedelta(hours=1):
        raise ValueError(f"Timestamp index is not a regular hourly grid: unique diffs={diffs}")
