"""Statistics for reporting a pass rate honestly (lecture part 4 §4.5, part 8 §8.4).

YOU WRITE THESE. `numpy` is allowed; nothing else. Four functions, all small. Your week-1
`metrics.py` has most of the ingredients.
"""
from __future__ import annotations

import math


def wilson_interval(passed: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Returns (lo, hi) in [0, 1].
    n == 0 → (0.0, 1.0). Prefer this to the normal approximation: at n = 40 and p̂ near 0 or 1
    the normal interval leaves [0, 1]; Wilson does not."""
    raise NotImplementedError


def pass_rate(passed: int, n: int) -> dict:
    """{"passed": int, "n": int, "rate": float, "ci95": (lo, hi), "half_width": float}
    where half_width = (hi − lo) / 2 from wilson_interval. n == 0 → rate 0.0."""
    raise NotImplementedError


def paired_diff(baseline: dict[str, bool], candidate: dict[str, bool]) -> dict:
    """Per-case comparison of two runs over the SAME case ids (part 8 §8.4: compare per case,
    never by aggregate). Both dicts map case_id → passed.

    Returns {
      "n": number of ids present in both,
      "regressions": sorted ids that passed in baseline and failed in candidate,
      "improvements": sorted ids that failed in baseline and passed in candidate,
      "unchanged": count of ids with the same verdict,
      "only_in_baseline": sorted ids missing from candidate   (NODATA — never count as pass),
      "only_in_candidate": sorted ids missing from baseline,
      "net": len(improvements) − len(regressions),
    }"""
    raise NotImplementedError


def noise_floor(runs: list[dict[str, bool]]) -> dict:
    """Given k runs of the UNCHANGED system over the same cases (each run: case_id → passed),
    measure how much the suite moves on its own.

    Returns {
      "k": k,
      "flaky_cases": sorted ids that were not identical across all runs
                     (Google's definition: both a pass and a fail with the same code),
      "flaky_fraction": len(flaky_cases) / n_cases,
      "rates": [pass rate of each run],
      "rate_spread": max(rates) − min(rates),
    }
    The gate's threshold must exceed rate_spread, and flaky_cases are candidates for
    quarantine or for `trials > 1` (part 8 §8.4)."""
    raise NotImplementedError
