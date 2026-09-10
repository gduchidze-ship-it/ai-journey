# Build — `metrics.py` (2 h)

**Deliverable.** A single file `metrics.py` that passes `pytest test_metrics.py`, one PNG of a reliability diagram for `data/fraud_1pct.csv`, and a short `NOTES.md` (five answers, see below). All three land in your course repo under `src/metrics/` (or wherever your template puts library code) with the tests under `tests/`.

**Rules.** From scratch. `numpy` is allowed for arrays and sorting. `matplotlib` is allowed *only* inside `plot_reliability_diagram`. **No `sklearn`, no `scipy`, no `pandas`** — not imported, not copied from. The point is that you will never again wonder what a library function is doing to your denominators. Type hints and docstrings are part of the deliverable; this file becomes half of Project 1.

**Time.** If the core functions (§1–§4 below) take more than 90 minutes, stop, commit what passes, and write down where the time went. Calibration (§5) is the last 30 minutes.

## Conventions — read these first, the tests assume them

1. **Labels** are 0/1 integers. **Scores** are floats in `[0, 1]` (a probability or something pretending to be one).
2. **Thresholding rule:** `y_pred = 1 if score >= threshold else 0`. Greater-or-equal, not greater.
3. **Undefined ratios return `0.0`** — precision with no predicted positives, recall with no actual positives, F1 when both precision and recall are 0. Do not raise; do not return NaN. (scikit-learn does the same, with a warning.)
4. **Confusion matrix layout** — return a dict with keys `tp, fp, fn, tn`, not a 2×2 array, so nobody has to remember which axis is which.
5. **Threshold sweep** evaluates every unique score value present in the data as a threshold, in *descending* order, plus one extra threshold strictly above the maximum score (so that the row with zero predicted positives exists — in descending order it is the *first* row). This gives one row per distinct operating point.
6. **ROC AUC** is the trapezoidal area under the (FPR, TPR) points from the sweep, with `(0, 0)` and `(1, 1)` included.
7. **Average precision** is the step-sum `AP = Σₙ (Rₙ − Rₙ₋₁) · Pₙ` over the sweep rows in descending-threshold order, with `R₀ = 0`. **Not** trapezoidal. Lecture part 4 §4.6 says why.
8. **Reliability bins:** `n_bins` equal-width bins on `[0, 1]`. Bin `i` covers `[i/n_bins, (i+1)/n_bins)`; the last bin also includes `1.0`. A score of exactly `0.3` with 10 bins goes in bin 3. Empty bins are reported with `count = 0` and `mean_pred = frac_pos = None`, and contribute nothing to ECE.
9. **Expected cost** is `cost_fp · FP + cost_fn · FN` (correct decisions cost nothing).

## What to implement

The skeleton `metrics.py` has every signature and docstring already; every body except the provided `load_csv` is `raise NotImplementedError`. Fill them in. Suggested order, with the lecture part that defines each (≈ 100 min of coding, leaving 15 min for the reflection in the next section and a few minutes for setup):

### §1 Counting (part 4 §4.1) — 10 min
- `binarize(y_score, threshold) -> np.ndarray`
- `confusion_matrix(y_true, y_pred) -> dict[str, int]`

### §2 Point metrics (part 4 §4.2) — 10 min
- `accuracy(cm)`, `precision(cm)`, `recall(cm)`, `false_positive_rate(cm)`, `f1(cm)`, `fbeta(cm, beta)`
  All take the dict from `confusion_matrix`. `f1` should call `fbeta`.

### §3 Threshold sweep and curves (part 4 §4.5–4.7) — 35 min
- `threshold_sweep(y_true, y_score) -> list[dict]` — one dict per row with keys `threshold, tp, fp, fn, tn, precision, recall, fpr, f1`. Aim for **O(n log n)** overall. Recomputing a confusion matrix from scratch for every threshold is O(n × distinct scores) — it will pass the tests on 20 000 rows but you will be running this on millions of rows by week 12, so design for that now. Before coding, ask: what does one pass over the data in some order buy you?
- `roc_curve(y_true, y_score) -> tuple[list[float], list[float]]` — `(fpr, tpr)`, from the sweep, `(0,0)` first and `(1,1)` last.
- `roc_auc(y_true, y_score) -> float` — trapezoid.
- `pr_curve(y_true, y_score) -> tuple[list[float], list[float]]` — `(recall, precision)` in descending-threshold order.
- `average_precision(y_true, y_score) -> float` — step-sum.

### §4 Costs and thresholds (part 5) — 15 min
- `expected_cost(cm, cost_fp, cost_fn) -> float`
- `optimal_threshold(y_true, y_score, cost_fp, cost_fn) -> tuple[float, float]` — `(threshold, cost)` minimizing expected cost over the sweep. On ties, return the **highest** threshold (fewest flags).
- `bayes_threshold(cost_fp, cost_fn) -> float` — the closed form `C_FP / (C_FP + C_FN)`. One line. It exists so that you can *compare* it with `optimal_threshold` on the sample data and notice they disagree; part 5 and part 6 explain why.

### §5 Calibration (part 6 §6.3) — 35 min
- `reliability_bins(y_true, y_prob, n_bins=10) -> list[dict]` — one dict per bin with keys `lower, upper, count, mean_pred, frac_pos`.
- `expected_calibration_error(y_true, y_prob, n_bins=10) -> float`
- `brier_score(y_true, y_prob) -> float`
- `plot_reliability_diagram(y_true, y_prob, path, n_bins=10, title="") -> None` — saves a PNG. Diagonal reference line; one point (or bar) per non-empty bin at `(mean_pred, frac_pos)`; a second panel or inset showing per-bin counts on a **log** scale (on the 1 % file the first bin has ~15 000 items and the middle bins a few hundred — linear scale hides everything). Put the ECE in the title.

## Then look at what you built (do not skip — 15 min of the 2 h)

Run your functions on both CSVs and write five short answers into `NOTES.md` next to your code. No word limit, but a sentence each is fine:

1. Compare `roc_auc` on the two files. Compare `average_precision`. Which one moved, and by how much? Which lecture claim did you just verify?
2. On `fraud_1pct.csv`, what is precision at threshold 0.9? What fraction of your flags are false alarms? Multiply the FPR at 0.9 by the number of negatives — that is your daily false-alarm count if this were one day of traffic. Now reread part 5 §5.4.
3. Look at the top reliability bin (0.9–1.0) on the fraud file: mean predicted probability versus fraction actually positive. Then the same bin on the balanced file. Same classifier, same scores. What changed and why? (Part 6 §6.6.)
4. For `cost_fp = 1, cost_fn = 9`: what threshold does `bayes_threshold` give, and what does `optimal_threshold` give on the fraud file? They should be far apart. Which assumption of the Bayes formula is violated by this data?
5. Report your `threshold_sweep` runtime on 20 000 rows. Estimate it for 20 000 000. If the estimate is over a minute, your loop is not O(n log n).

## Self-grading rubric

| | Pass | Strong |
|---|---|---|
| Tests | All of `test_metrics.py` green | Plus your own tests for at least two edge cases the file does not cover (all-negative labels, all scores identical, a single row) |
| Conventions | Followed | Docstrings state every convention explicitly so a reader never needs this file |
| Sweep | Correct | O(n log n), verified by timing at two sizes |
| Diagram | Saved, has diagonal and points | Log-scale count panel; ECE and n in the title; readable at thumbnail size |
| NOTES.md | Five answers present | Each answer names the lecture section it verifies or contradicts |

## If stuck

| Symptom | Likely cause | Where to look |
|---|---|---|
| `roc_auc` off by a little | Missing `(0,0)` or `(1,1)` endpoint, or `>` instead of `>=` | Convention 2 and 6 |
| `average_precision` too high | You used trapezoid or linear interpolation | Part 4 §4.6; Davis & Goadrich §4 |
| `average_precision` too low | You dropped the row where recall first becomes > 0, or you have `R₀` wrong | Convention 7 |
| Confusion counts at 0.9 off by a few | Floating-point compare on the CSV strings — parse to `float`, compare with `>=` | Convention 2 |
| ECE differs slightly | Bin assignment at the boundaries, or you weighted by `1/n_bins` instead of `count/n` | Convention 8; part 6 §6.3 |
| Sweep slow | Recomputing the confusion matrix from scratch per threshold | §3 note — one ordered pass is enough |
| `optimal_threshold` ties | You returned the first minimum rather than the highest threshold | §4 |

Sources for every formula in this assignment are listed under parts 4, 5 and 6 of `../../READING.md`.
