"""Tests for metrics.py.  Run:  pytest test_metrics.py -q

These tests check your implementation against (a) small cases you can verify by
hand and (b) reference values for the two CSVs in data/ computed once with a
standard library and frozen here. They are checks, not solutions: nothing in
this file tells you how to compute anything.

If a test fails, ASSIGNMENT.md → "If stuck" maps the symptom to the convention
you most likely missed.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pytest

import metrics as M

HERE = os.path.dirname(os.path.abspath(__file__))
FRAUD = os.path.join(HERE, "data", "fraud_1pct.csv")
BALANCED = os.path.join(HERE, "data", "balanced_50pct.csv")

# A small case you can work by hand (lecture part 4). ------------------------
Y8 = [1, 0, 1, 1, 0, 0, 1, 0]
S8 = [0.9, 0.8, 0.7, 0.6, 0.55, 0.4, 0.3, 0.1]


def approx(x, rel=1e-6, abs_=1e-9):
    return pytest.approx(x, rel=rel, abs=abs_)


# --------------------------------------------------------------------------- #
# §1 counting
# --------------------------------------------------------------------------- #


def test_binarize_uses_greater_or_equal():
    out = M.binarize([0.5, 0.49999, 0.9, 0.0], 0.5)
    assert list(out) == [1, 0, 1, 0]


def test_confusion_matrix_hand_case():
    cm = M.confusion_matrix(Y8, M.binarize(S8, 0.5))
    assert cm == {"tp": 3, "fp": 2, "fn": 1, "tn": 2}
    assert all(isinstance(v, int) for v in cm.values())


def test_confusion_matrix_rejects_length_mismatch():
    with pytest.raises(ValueError):
        M.confusion_matrix([1, 0], [1])


# --------------------------------------------------------------------------- #
# §2 point metrics
# --------------------------------------------------------------------------- #


def test_point_metrics_hand_case():
    cm = {"tp": 3, "fp": 2, "fn": 1, "tn": 2}
    assert M.accuracy(cm) == approx(5 / 8)
    assert M.precision(cm) == approx(0.6)
    assert M.recall(cm) == approx(0.75)
    assert M.false_positive_rate(cm) == approx(0.5)
    assert M.f1(cm) == approx(2 * 0.6 * 0.75 / (0.6 + 0.75))


def test_fbeta_direction():
    # precision 1.0, recall 0.5 -> F0.5 favours precision, F2 favours recall
    cm = {"tp": 1, "fp": 0, "fn": 1, "tn": 2}
    assert M.precision(cm) == 1.0 and M.recall(cm) == 0.5
    assert M.fbeta(cm, 1.0) == approx(2 / 3)
    assert M.fbeta(cm, 0.5) == approx(0.8333333333)
    assert M.fbeta(cm, 2.0) == approx(0.5555555556)
    assert M.fbeta(cm, 0.5) > M.f1(cm) > M.fbeta(cm, 2.0)


def test_undefined_ratios_return_zero_not_nan():
    nothing_flagged = {"tp": 0, "fp": 0, "fn": 3, "tn": 5}
    assert M.precision(nothing_flagged) == 0.0
    assert M.f1(nothing_flagged) == 0.0
    no_positives = {"tp": 0, "fp": 2, "fn": 0, "tn": 5}
    assert M.recall(no_positives) == 0.0
    no_negatives = {"tp": 4, "fp": 0, "fn": 1, "tn": 0}
    assert M.false_positive_rate(no_negatives) == 0.0
    empty = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    assert M.accuracy(empty) == 0.0


def test_f1_ignores_true_negatives():
    a = {"tp": 30, "fp": 10, "fn": 20, "tn": 5}
    b = {"tp": 30, "fp": 10, "fn": 20, "tn": 5_000_000}
    assert M.f1(a) == approx(M.f1(b))
    assert M.accuracy(a) != approx(M.accuracy(b))


# --------------------------------------------------------------------------- #
# §3 sweep and curves
# --------------------------------------------------------------------------- #


def test_sweep_shape_and_order():
    rows = M.threshold_sweep(Y8, S8)
    # 8 unique scores + 1 "flag nothing" row
    assert len(rows) == 9
    assert set(rows[0].keys()) >= {"threshold", "tp", "fp", "fn", "tn", "precision", "recall", "fpr", "f1"}
    thresholds = [r["threshold"] for r in rows]
    assert thresholds == sorted(thresholds, reverse=True)
    assert thresholds[0] > max(S8)                       # flag-nothing row first
    assert rows[0]["tp"] == 0 and rows[0]["fp"] == 0
    assert rows[-1]["threshold"] == approx(min(S8))
    assert rows[-1]["tp"] + rows[-1]["fp"] == len(S8)    # everything flagged
    recalls = [r["recall"] for r in rows]
    fprs = [r["fpr"] for r in rows]
    assert recalls == sorted(recalls) and fprs == sorted(fprs)


def test_sweep_row_matches_confusion_matrix():
    rows = {r["threshold"]: r for r in M.threshold_sweep(Y8, S8)}
    row = rows[0.5] if 0.5 in rows else None
    # 0.5 is not a score value, so it is not a sweep threshold; 0.55 is.
    assert row is None
    r = rows[0.55]
    cm = M.confusion_matrix(Y8, M.binarize(S8, 0.55))
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (cm["tp"], cm["fp"], cm["fn"], cm["tn"])


def test_sweep_handles_tied_scores_as_one_operating_point():
    s = [0.9, 0.8, 0.7, 0.7, 0.55, 0.4, 0.3, 0.1]
    rows = M.threshold_sweep(Y8, s)
    assert len(rows) == 8                                # 7 unique + 1
    r = {row["threshold"]: row for row in rows}[0.7]
    assert r["tp"] == 3 and r["fp"] == 1                 # both 0.7s flagged together


def test_roc_curve_endpoints():
    fpr, tpr = M.roc_curve(Y8, S8)
    assert (fpr[0], tpr[0]) == (0.0, 0.0)
    assert (fpr[-1], tpr[-1]) == (1.0, 1.0)
    assert fpr == sorted(fpr)


def test_roc_auc_hand_case():
    # 11 of the 16 (positive, negative) pairs have the positive scored higher
    assert M.roc_auc(Y8, S8) == approx(11 / 16)


def test_roc_auc_perfect_and_inverted():
    assert M.roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == approx(1.0)
    assert M.roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == approx(0.0)


def test_pr_curve_shape():
    rec, prec = M.pr_curve(Y8, S8)
    assert len(rec) == len(prec) == 9
    assert rec == sorted(rec)
    assert rec[0] == 0.0 and rec[-1] == 1.0
    assert all(0.0 <= p <= 1.0 for p in prec)


def test_average_precision_is_step_sum_not_trapezoid():
    assert M.average_precision(Y8, S8) == approx(0.7470238095238095)
    # trapezoid would give a different (higher) number; make sure you did not
    assert M.average_precision(Y8, S8) < 0.76


def test_average_precision_with_ties():
    s = [0.9, 0.8, 0.7, 0.7, 0.55, 0.4, 0.3, 0.1]
    assert M.average_precision(Y8, s) == approx(0.7678571428571428)


def test_average_precision_of_perfect_ranking_is_one():
    assert M.average_precision([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == approx(1.0)


# --------------------------------------------------------------------------- #
# §4 costs
# --------------------------------------------------------------------------- #


def test_expected_cost():
    cm = {"tp": 3, "fp": 2, "fn": 1, "tn": 2}
    assert M.expected_cost(cm, cost_fp=1.0, cost_fn=9.0) == approx(2 * 1 + 1 * 9)


def test_bayes_threshold():
    assert M.bayes_threshold(1, 9) == approx(0.1)
    assert M.bayes_threshold(9, 1) == approx(0.9)
    assert M.bayes_threshold(3, 3) == approx(0.5)


def test_optimal_threshold_picks_minimum():
    t, c = M.optimal_threshold([1, 0], [0.9, 0.1], cost_fp=1, cost_fn=1)
    assert t == approx(0.9) and c == approx(0.0)


def test_optimal_threshold_tie_breaks_to_highest_threshold():
    # costs: t>0.9 -> 2 ; t=0.9 -> 1 ; t=0.5 -> 2 ; t=0.1 -> 1   (tie at 1)
    t, c = M.optimal_threshold([1, 0, 1], [0.9, 0.5, 0.1], cost_fp=1, cost_fn=1)
    assert c == approx(1.0)
    assert t == approx(0.9)


# --------------------------------------------------------------------------- #
# §5 calibration
# --------------------------------------------------------------------------- #

YC = [0, 1, 0, 1, 1, 0, 1, 0]
PC = [0.05, 0.15, 0.15, 0.95, 0.95, 0.95, 0.55, 0.55]


def test_reliability_bins_hand_case():
    bins = M.reliability_bins(YC, PC, n_bins=10)
    assert len(bins) == 10
    assert sum(b["count"] for b in bins) == len(YC)
    assert bins[0]["lower"] == approx(0.0) and bins[0]["upper"] == approx(0.1)
    assert bins[9]["lower"] == approx(0.9) and bins[9]["upper"] == approx(1.0)
    assert bins[0]["count"] == 1 and bins[0]["mean_pred"] == approx(0.05) and bins[0]["frac_pos"] == approx(0.0)
    assert bins[1]["count"] == 2 and bins[1]["mean_pred"] == approx(0.15) and bins[1]["frac_pos"] == approx(0.5)
    assert bins[5]["count"] == 2 and bins[5]["frac_pos"] == approx(0.5)
    assert bins[9]["count"] == 3 and bins[9]["mean_pred"] == approx(0.95) and bins[9]["frac_pos"] == approx(2 / 3)
    empty = bins[2]
    assert empty["count"] == 0 and empty["mean_pred"] is None and empty["frac_pos"] is None


def test_reliability_bin_boundaries():
    bins = M.reliability_bins([0, 1, 1], [0.0, 0.3, 1.0], n_bins=10)
    assert bins[0]["count"] == 1          # 0.0 -> first bin
    assert bins[3]["count"] == 1          # 0.3 -> bin 3, not bin 2
    assert bins[9]["count"] == 1          # 1.0 -> last bin, not out of range


def test_ece_hand_case():
    # (1/8)*|0-0.05| + (2/8)*|0.5-0.15| + (2/8)*|0.5-0.55| + (3/8)*|2/3-0.95|
    assert M.expected_calibration_error(YC, PC, n_bins=10) == approx(0.2125, rel=1e-6)


def test_ece_of_perfectly_calibrated_bins_is_zero():
    y = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    p = [0.5] * 10
    assert M.expected_calibration_error(y, p, n_bins=10) == approx(0.0)


def test_brier_hand_case():
    assert M.brier_score(YC, PC) == approx(0.27)


def test_brier_bounds():
    assert M.brier_score([1, 0], [1.0, 0.0]) == approx(0.0)
    assert M.brier_score([1, 0], [0.0, 1.0]) == approx(1.0)


def test_plot_writes_png(tmp_path):
    out = tmp_path / "rd.png"
    M.plot_reliability_diagram(YC, PC, str(out), n_bins=10, title="hand case")
    assert out.exists() and out.stat().st_size > 1000
    with open(out, "rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------- #
# Reference values on the sample data (frozen; see make_data.py)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def fraud():
    return M.load_csv(FRAUD)


@pytest.fixture(scope="module")
def balanced():
    return M.load_csv(BALANCED)


def test_fraud_prevalence(fraud):
    y, _ = fraud
    assert len(y) == 20_000 and int(y.sum()) == 203


def test_fraud_confusion_at_0_5_and_0_9(fraud):
    y, s = fraud
    assert M.confusion_matrix(y, M.binarize(s, 0.5)) == {"tp": 172, "fp": 2256, "fn": 31, "tn": 17541}
    assert M.confusion_matrix(y, M.binarize(s, 0.9)) == {"tp": 142, "fp": 825, "fn": 61, "tn": 18972}


def test_fraud_point_metrics_at_0_9(fraud):
    y, s = fraud
    cm = M.confusion_matrix(y, M.binarize(s, 0.9))
    assert M.precision(cm) == approx(0.146846, rel=1e-4)
    assert M.recall(cm) == approx(0.699507, rel=1e-4)
    assert M.f1(cm) == approx(0.242735, rel=1e-4)
    assert M.accuracy(cm) > 0.95


def test_fraud_auc_and_ap(fraud):
    y, s = fraud
    assert M.roc_auc(y, s) == approx(0.945743, rel=1e-4)
    assert M.average_precision(y, s) == approx(0.427301, rel=1e-4)


def test_balanced_auc_and_ap(balanced):
    y, s = balanced
    assert M.roc_auc(y, s) == approx(0.953990, rel=1e-4)
    assert M.average_precision(y, s) == approx(0.954083, rel=1e-4)


def test_curves_across_prevalence(fraud, balanced):
    # Same simulated classifier on both files. Explain these two bounds in NOTES.md Q1.
    yf, sf = fraud
    yb, sb = balanced
    assert abs(M.roc_auc(yf, sf) - M.roc_auc(yb, sb)) < 0.02
    assert M.average_precision(yb, sb) - M.average_precision(yf, sf) > 0.4


def test_fraud_calibration_numbers(fraud):
    y, s = fraud
    assert M.expected_calibration_error(y, s, n_bins=10) == approx(0.132639, rel=1e-3)
    assert M.brier_score(y, s) == approx(0.086833, rel=1e-4)
    top = M.reliability_bins(y, s, n_bins=10)[9]
    assert top["count"] == 967
    assert top["mean_pred"] == approx(0.964, abs_=1e-3)
    assert top["frac_pos"] == approx(0.1468, abs_=1e-3)


def test_balanced_calibration_numbers(balanced):
    y, s = balanced
    assert M.expected_calibration_error(y, s, n_bins=10) == approx(0.047373, rel=1e-3)
    assert M.brier_score(y, s) == approx(0.087580, rel=1e-4)


def test_fraud_optimal_threshold_cost_1_9(fraud):
    y, s = fraud
    t, c = M.optimal_threshold(y, s, cost_fp=1, cost_fn=9)
    assert c == approx(1059.0)
    assert t == approx(0.979594, abs_=1e-6)
    # Compare with bayes_threshold(1, 9); NOTES.md Q4 asks you to explain the gap.
    assert abs(t - M.bayes_threshold(1, 9)) > 0.5


def test_sweep_is_fast_enough(fraud):
    y, s = fraud
    t0 = time.perf_counter()
    M.threshold_sweep(y, s)
    elapsed = time.perf_counter() - t0
    # An O(n log n) sweep on 20k rows takes well under a second. A per-threshold
    # recount takes tens of seconds. Generous bound so slow laptops pass.
    assert elapsed < 5.0, f"threshold_sweep took {elapsed:.1f}s on 20k rows — see ASSIGNMENT.md §3"
