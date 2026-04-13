"""Random timestamp sampler for the dataset build loop."""

from __future__ import annotations
import random
from datetime import date, datetime, timedelta


class TimestampSampler:
    """
    Draws random (day, hour) pairs from [date_start, date_end] without
    replacement until the pool is exhausted, then optionally wraps.

    Parameters
    ----------
    date_start, date_end : str  e.g. "2010-01-01"
    allow_repeats        : bool  allow reuse after pool exhausted
    seed                 : int
    used                 : set   pre-populated from checkpoint
    """

    def __init__(
        self,
        date_start: str,
        date_end: str,
        allow_repeats: bool = False,
        seed: int = 42,
        used: set | None = None,
    ):
        self.start = date.fromisoformat(date_start)
        self.end   = date.fromisoformat(date_end)
        self.allow_repeats = allow_repeats
        self.rng = random.Random(seed)
        self.used: set[str] = used or set()

        total_days = (self.end - self.start).days + 1
        self._total_pool = total_days * 24
        self._days = total_days

    def sample(self) -> datetime:
        """Return an unused random datetime. Raises StopIteration if pool exhausted."""
        for _ in range(self._total_pool * 2):
            rand_day  = self.rng.randint(0, self._days - 1)
            rand_hour = self.rng.randint(0, 23)
            dt = datetime.combine(
                self.start + timedelta(days=rand_day),
                datetime.min.time(),
            ).replace(hour=rand_hour)
            key = dt.strftime("%Y%m%d%H")
            if key not in self.used or self.allow_repeats:
                self.used.add(key)
                return dt

        if self.allow_repeats:
            raise StopIteration("Timestamp sampler pool exhausted.")
        raise StopIteration(
            f"All {self._total_pool:,} timestamps used. "
            "Set allow_ts_repeats: true in config to allow reuse."
        )

    @property
    def n_used(self) -> int:
        return len(self.used)

    @property
    def pool_size(self) -> int:
        return self._total_pool
