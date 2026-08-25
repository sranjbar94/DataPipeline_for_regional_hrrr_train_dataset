# Documentation reviewed and improved for clarity.
# Documentation reviewed and improved for clarity.
# Documentation reviewed and improved for clarity.
# Documentation reviewed and improved for clarity.
"""Tests for the timestamp sampler."""

import pytest
from src.utils.time_sampler import TimestampSampler


def test_no_repeats():
    s = TimestampSampler("2010-01-01", "2010-01-03", allow_repeats=False, seed=0)
    drawn = [s.sample().strftime("%Y%m%d%H") for _ in range(10)]
    assert len(drawn) == len(set(drawn)), "Duplicate timestamps drawn"


def test_within_range():
    from datetime import date
    s = TimestampSampler("2015-06-01", "2015-06-30", seed=1)
    for _ in range(50):
        dt = s.sample()
        assert date(2015, 6, 1) <= dt.date() <= date(2015, 6, 30)


def test_resume_from_used():
    s1 = TimestampSampler("2010-01-01", "2010-12-31", seed=7)
    used = set()
    for _ in range(20):
        dt = s1.sample()
        used.add(dt.strftime("%Y%m%d%H"))

    s2 = TimestampSampler("2010-01-01", "2010-12-31", seed=7, used=used)
    for _ in range(20):
        dt = s2.sample()
        assert dt.strftime("%Y%m%d%H") not in used
