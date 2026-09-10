"""Generate the two synthetic score files used by the metrics assignment.

You do not need to run this — `data/` already contains the output. It is here so
you can see exactly how the data was made (and change the prevalence or the
over-confidence to see what happens to your curves).

Both files come from the SAME simulated classifier: the class-conditional score
distributions are identical. Only the class balance differs. Part of the
assignment (NOTES.md) is to run your metrics on both and explain what you see.

The simulated classifier applies a temperature to its logits. Whether that makes
it over- or under-confident, and on which file, is for your reliability diagram
to tell you.

Columns: y_true (0/1), y_score (float in (0, 1)).
"""
from __future__ import annotations

import csv
import pathlib

import numpy as np

SEED = 20260910
N = 20_000
TEMPERATURE = 0.6   # applied to the logits; try 1.0 and 1.5 and re-run your diagram
MU_POS, MU_NEG, SIGMA = 1.2, -1.2, 1.0


def simulate(n: int, prevalence: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    y = (rng.random(n) < prevalence).astype(int)
    z = np.where(y == 1, rng.normal(MU_POS, SIGMA, n), rng.normal(MU_NEG, SIGMA, n))
    # Log-likelihood ratio of the two Gaussians is (2*mu/sigma^2) * z; we use it as
    # the logit, scaled by TEMPERATURE. Notice what this formula does NOT include.
    logit = (2 * MU_POS / SIGMA**2) * z / TEMPERATURE
    p = 1.0 / (1.0 + np.exp(-logit))
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return y, np.round(p, 6)


def write(path: pathlib.Path, y: np.ndarray, p: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["y_true", "y_score"])
        for yi, pi in zip(y, p):
            w.writerow([int(yi), f"{pi:.6f}"])


if __name__ == "__main__":
    here = pathlib.Path(__file__).parent
    rng = np.random.default_rng(SEED)
    y, p = simulate(N, prevalence=0.01, rng=rng)
    write(here / "data" / "fraud_1pct.csv", y, p)
    y, p = simulate(N, prevalence=0.50, rng=rng)
    write(here / "data" / "balanced_50pct.csv", y, p)
    print("wrote data/fraud_1pct.csv and data/balanced_50pct.csv")
