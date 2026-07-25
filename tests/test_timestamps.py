import pandas as pd
import pytest

from gefcom.timestamps import (
    hourly_index_from_start,
    parse_first_timestamp,
    reconstruct_hourly_index,
    verify_regular_hourly_grid,
)


def test_parse_first_timestamp_unambiguous_year_boundary():
    # "112001" -> only (month=1, day=1) is a valid split -> Jan 1 2001, hour 1
    assert parse_first_timestamp("112001 1:00") == pd.Timestamp("2001-01-01 01:00")


def test_parse_first_timestamp_prefers_day_one_on_tie():
    # "1212011" is ambiguous between Jan 21 2011 and Dec 1 2011; both splits
    # are valid calendar dates, but only Dec 1 has day == 1, which is the
    # tie-break rule (every file in this dataset starts on the 1st).
    assert parse_first_timestamp("1212011 1:00") == pd.Timestamp("2011-12-01 01:00")


def test_hour_zero_parses_as_midnight_of_that_date():
    # hour=0 in the raw string means midnight of the literal parsed date
    # (no rollover needed at this level -- the "hour 24 of previous day"
    # interpretation only matters when reconstructing a full sequence,
    # which is covered by test_reconstruct_hourly_index_* below).
    ts = parse_first_timestamp("112010 0:00")  # unambiguous: month=1, day=1, year=2010
    assert ts == pd.Timestamp("2010-01-01 00:00")


def test_reconstruct_hourly_index_is_gapless_and_matches_known_anchor():
    raw = pd.Series(["112001 1:00", "112001 2:00", "112001 3:00", "112001 4:00"])
    idx = reconstruct_hourly_index(raw)
    verify_regular_hourly_grid(idx)
    assert idx[0] == pd.Timestamp("2001-01-01 01:00")
    assert idx[-1] == pd.Timestamp("2001-01-01 04:00")


def test_reconstruct_hourly_index_detects_gap():
    # A file that skips an hour should fail the internal consistency check
    # rather than silently produce a wrong index. We construct a raw series
    # that, if treated as gapless, would disagree with its own (unambiguous)
    # later timestamps.
    raw = pd.Series(["112001 1:00", "112001 2:00", "3132001 14:00"])  # bogus jump
    with pytest.raises(ValueError):
        reconstruct_hourly_index(raw)


def test_hourly_index_from_start_matches_benchmark_continuity_case():
    # Regression test for the real ambiguous case found in L1-benchmark.csv:
    # its own first row ("1012010 1:00") is genuinely ambiguous between
    # Jan 1 2010 and Oct 1 2010 (both splits give day == 1). The correct
    # anchor must come from train-file continuity, not self-parsing.
    train_last = pd.Timestamp("2010-10-01 00:00")
    idx = hourly_index_from_start(train_last + pd.Timedelta(hours=1), n=744)
    assert idx[0] == pd.Timestamp("2010-10-01 01:00")
    assert idx[-1] == pd.Timestamp("2010-11-01 00:00")
