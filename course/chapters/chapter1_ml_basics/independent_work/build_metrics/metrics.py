"""metrics.py — binary classification metrics from scratch.

Course: Applied AI Engineering, week 1 (Lecture 1 build).
Rules: numpy allowed; matplotlib allowed only inside plot_reliability_diagram;
       no sklearn, scipy or pandas.

Conventions (also in ASSIGNMENT.md — the tests assume these):
  * labels are 0/1 ints; scores are floats in [0, 1]
  * positive prediction  <=>  score >= threshold
  * undefined ratios return 0.0 (never NaN, never raise)
  * confusion matrix is a dict {"tp", "fp", "fn", "tn"}
  * threshold sweep visits every unique score, descending, plus one threshold
    above the maximum score
  * ROC AUC is trapezoidal with (0,0) and (1,1) included
  * average precision is the step sum  sum_n (R_n - R_{n-1}) * P_n,  R_0 = 0
  * reliability bins are equal-width on [0,1]; bin i = [i/n, (i+1)/n), last bin
    also includes 1.0; empty bins have count 0 and None for the means
  * expected cost = cost_fp * FP + cost_fn * FN

Every function body below is a stub. Replace `raise NotImplementedError` with
your implementation. Keep the signatures — test_metrics.py imports them.
"""
from __future__ import annotations

from typing import Sequence, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[int], Sequence[float]]

# --------------------------------------------------------------------------- #
# §1  Counting
# --------------------------------------------------------------------------- #


def binarize(y_score: ArrayLike, threshold: float) -> np.ndarray:
    """Turn scores into 0/1 predictions: 1 where score >= threshold, else 0.

    Returns an int array the same length as y_score.
    """
    raise NotImplementedError


def confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, int]:
    """Count tp, fp, fn, tn.

    tp: true 1, predicted 1      fp: true 0, predicted 1
    fn: true 1, predicted 0      tn: true 0, predicted 0
    Returns {"tp": int, "fp": int, "fn": int, "tn": int} with Python ints.
    Raise ValueError if lengths differ.
    """
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# §2  Point metrics — all take the dict returned by confusion_matrix
# --------------------------------------------------------------------------- #


def accuracy(cm: dict[str, int]) -> float:
    """(tp + tn) / (tp + tn + fp + fn). 0.0 if the matrix is empty."""
    raise NotImplementedError


def precision(cm: dict[str, int]) -> float:
    """tp / (tp + fp). 0.0 if nothing was predicted positive."""
    raise NotImplementedError


def recall(cm: dict[str, int]) -> float:
    """tp / (tp + fn). Also called sensitivity / true positive rate. 0.0 if no actual positives."""
    raise NotImplementedError


def false_positive_rate(cm: dict[str, int]) -> float:
    """fp / (fp + tn). 0.0 if no actual negatives."""
    raise NotImplementedError


def fbeta(cm: dict[str, int], beta: float) -> float:
    """(1 + beta^2) * P * R / (beta^2 * P + R). 0.0 if the denominator is 0.

    beta > 1 weights recall more; beta < 1 weights precision more.
    Note: tn does not appear anywhere in this formula. That is a feature of
    F-measures you should be able to explain (lecture part 4 §4.4).
    """
    raise NotImplementedError


def f1(cm: dict[str, int]) -> float:
    """fbeta with beta = 1."""
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# §3  Threshold sweep and curves
# --------------------------------------------------------------------------- #


def threshold_sweep(y_true: ArrayLike, y_score: ArrayLike) -> list[dict]:
    """Evaluate every distinct operating point.

    Thresholds: one extra threshold strictly greater than max(y_score) (the
    "flag nothing" row), then every unique value in y_score, all in DESCENDING
    order. One dict per threshold with keys:
        threshold, tp, fp, fn, tn, precision, recall, fpr, f1
    Because rows are in descending-threshold order, recall and fpr are both
    non-decreasing down the list.

    Aim for O(n log n) overall. Calling confusion_matrix once per threshold is
    O(n * distinct scores) and will not scale — see ASSIGNMENT.md §3.
    """
    raise NotImplementedError


def roc_curve(y_true: ArrayLike, y_score: ArrayLike) -> tuple[list[float], list[float]]:
    """Return (fpr, tpr) lists for plotting, starting at (0, 0) and ending at (1, 1).

    Built from threshold_sweep. Points must be in non-decreasing fpr order.
    """
    raise NotImplementedError


def roc_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Trapezoidal area under the ROC curve from roc_curve.

    Interpretation to remember: the probability that a random positive is
    scored above a random negative.
    """
    raise NotImplementedError


def pr_curve(y_true: ArrayLike, y_score: ArrayLike) -> tuple[list[float], list[float]]:
    """Return (recall, precision) lists in descending-threshold order.

    Do not append an artificial (recall=0, precision=1) endpoint; the sweep's
    "flag nothing" row already gives recall 0 (with precision 0.0 by the
    undefined-ratio convention). Draw this as steps, never as a smooth line.
    """
    raise NotImplementedError


def average_precision(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Step-sum area under the PR curve:  sum_n (R_n - R_{n-1}) * P_n,  R_0 = 0.

    Iterate over sweep rows in descending-threshold order (recall increasing).
    Not trapezoidal, not interpolated — see lecture part 4 §4.6.
    With random scores this tends to the positive prevalence; report prevalence
    next to it.
    """
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# §4  Costs and thresholds
# --------------------------------------------------------------------------- #


def expected_cost(cm: dict[str, int], cost_fp: float, cost_fn: float) -> float:
    """cost_fp * fp + cost_fn * fn. Correct decisions cost nothing."""
    raise NotImplementedError


def optimal_threshold(
    y_true: ArrayLike, y_score: ArrayLike, cost_fp: float, cost_fn: float
) -> tuple[float, float]:
    """Search the sweep for the threshold minimizing expected cost.

    Returns (threshold, cost). On ties, return the HIGHEST threshold (the one
    that flags the fewest items).
    """
    raise NotImplementedError


def bayes_threshold(cost_fp: float, cost_fn: float) -> float:
    """Closed-form optimal threshold for a CALIBRATED score: cost_fp / (cost_fp + cost_fn).

    Elkan (2001), eq. 2, with zero cost for correct decisions. Only optimal if
    the score is a calibrated probability at the deployment prevalence.
    Compare with optimal_threshold on the sample data and explain the gap.
    """
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# §5  Calibration
# --------------------------------------------------------------------------- #


def reliability_bins(y_true: ArrayLike, y_prob: ArrayLike, n_bins: int = 10) -> list[dict]:
    """Bin predictions into n_bins equal-width bins on [0, 1].

    Bin i covers [i/n_bins, (i+1)/n_bins); the last bin also includes 1.0.
    One dict per bin, in order, with keys:
        lower, upper, count, mean_pred, frac_pos
    where mean_pred is the mean predicted probability in the bin and frac_pos is
    the fraction of items in the bin whose label is 1. For an empty bin,
    count = 0 and mean_pred = frac_pos = None.
    """
    raise NotImplementedError


def expected_calibration_error(y_true: ArrayLike, y_prob: ArrayLike, n_bins: int = 10) -> float:
    """ECE = sum_over_bins (count_b / n) * |frac_pos_b - mean_pred_b|.

    Empty bins contribute nothing. Guo et al. (2017) §2.
    """
    raise NotImplementedError


def brier_score(y_true: ArrayLike, y_prob: ArrayLike) -> float:
    """Mean squared error between probability and 0/1 label: mean((y - p)^2).

    A strictly proper scoring rule: it rewards calibration AND resolution.
    """
    raise NotImplementedError


def plot_reliability_diagram(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    path: str,
    n_bins: int = 10,
    title: str = "",
) -> None:
    """Save a reliability diagram PNG to `path`.

    Required elements:
      * the diagonal y = x as a reference line
      * one marker (or bar) per NON-EMPTY bin at (mean_pred, frac_pos)
      * a second panel (or inset) with per-bin counts on a LOG y-axis
      * ECE in the title (adding n as well is part of the "Strong" rubric)
    matplotlib may be imported inside this function only.
    """
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------------- #


def load_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read a two-column CSV with header `y_true,y_score`. Provided for you."""
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    y_true = data[:, 0].astype(int)
    y_score = data[:, 1].astype(float)
    return y_true, y_score


if __name__ == "__main__":  # a place to poke at your functions; not graded
    import sys

    p = sys.argv[1] if len(sys.argv) > 1 else "data/fraud_1pct.csv"
    y, s = load_csv(p)
    print(f"{p}: n={len(y)} positives={int(y.sum())} prevalence={y.mean():.4f}")
    # After implementing, try e.g.:
    # print("AUC", roc_auc(y, s), "AP", average_precision(y, s), "ECE", expected_calibration_error(y, s))
