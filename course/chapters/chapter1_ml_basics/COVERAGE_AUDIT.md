# Coverage audit — Lecture 1 materials

*Author's self-reflection, written after the materials were drafted and reviewed. Date: 2026-09-10.*

This file answers one question: **does every bullet of the syllabus have a home in these materials, grounded in a real source, with no solutions leaked to the student?** It records where each bullet is covered, what an independent review found, what was changed in response, and what remains open. It is kept in the folder deliberately: the course teaches that a claim of coverage without evidence is worth nothing, so the materials should hold themselves to the same standard.

## Method

1. Drafted the eight lecture parts, the reading list, the build and the two drills from the syllabus.
2. Researched every bullet on the open web *before* writing — 90+ sources fetched and checked, section headings recorded; only sources that actually loaded are cited (four that could not be reached are listed at the bottom of `READING.md`).
3. Validated `test_metrics.py` against a private reference implementation (not shipped): 38/38 pass; the shipped skeleton fails 37/38 cleanly with `NotImplementedError` and passes only the length check on the data file.
4. Ran an independent review (fresh reader, no access to the drafting context) against the syllabus with strict criteria, a solution-leakage check, a cross-file consistency check, a factual red-flag pass, and a link check. Findings and fixes are below.

## A. Syllabus coverage

Rating scale: **Full** = every clause of the bullet has an explicit passage with a cited source; **Partial** = covered but a clause is thin; **Missing**.

| # | Syllabus bullet | Where | Rating | Evidence / note |
|---|---|---|---|---|
| 1 | ML as function fitting — data in, parameters fitted, behaviour generalized | `lecture/01` §1.1 | Full | Structured as those three clauses; ERM (Murphy eq. 1.2), population risk, generalization gap, i.i.d. + stationarity, train/val/test with the leak argument (sklearn §3.1). |
| 2 | Bias and variance, model capacity, overfitting and underfitting | `lecture/01` §1.2, §1.3 | Full | Goodfellow's two-part criterion; capacity U-curve; approximation vs estimation error; `MSE = Bias² + Var`; diagnostic table; double-descent caveat (Belkin; Nakkiran) so the U-curve is not taught as a law. |
| 3 | Regularization: L1, L2, early stopping — *what each actually constrains* | `lecture/02` §2.2–2.4, table §2.6 | Full | Each has a labelled "What it constrains" paragraph: L2 → low-curvature Hessian directions (`λᵢ/(λᵢ+α)`); L1 → exact zeros via the diamond's corners; early stopping → information transfer from data to parameters, `τ ≈ 1/(εα)`. |
| 4 | Weight decay vs L2 in the loss; why AdamW exists | `lecture/02` §2.5 | Full | Prop. 1 (SGD equivalence), Prop. 2 (no L2 coefficient recovers decoupled decay under Adam), Algorithm 2 line 6 vs line 12, PyTorch pseudocode for `SGD`/`Adam`/`AdamW`. |
| 5 | Why a training curve alone never tells you whether to ship | `lecture/03` | Full | What a curve *can* diagnose (MLCC), then four gaps: proxy objective (Rule #13), wrong data / three skews (Rule #37, Monitor 7), averages hide slices (Model 6), lower loss ≠ launch (Rules #24, #39; Model 2, Model 5). |
| 6 | Confusion matrix; precision, recall, F1 — and when each is the wrong summary | `lecture/04` §4.1–4.4 | Full | Labelled layout; formulas + worked example; F-beta; §4.4 gives a failure case for precision, recall, accuracy and three specific defects of F1 (TN-blind, equal-cost assumption, point metric). |
| 7 | ROC/AUC vs PR under class imbalance | `lecture/04` §4.5–4.7 | Full | AUC as ranking probability; step-sum AP and why not interpolated (Davis & Goadrich §4); prevalence-invariance of ROC vs PR with Saito & Rehmsmeier's 160/1,600 FP example; D&G's opposite-ranking example; PR baseline = prevalence; Theorems 3.1–3.2. |
| 8 | Threshold as a product decision: costing FP vs FN before choosing | `lecture/05` §5.1–5.3 | Full | Default 0.5 is the accuracy threshold (Elkan; sklearn §3.3); `Cost(t)`; Elkan eq. 2 closed form and its invariances; validation-only tuning; who owns the numbers. |
| 9 | Why accuracy lies on imbalanced data | `lecture/04` §4.3 | Full | Majority-class baseline; accuracy paradox with the 1,000,000 / 10 example; balanced accuracy and its limit. |
| 10 | Calibration: what a confidence score does and does not mean; reliability diagrams | `lecture/06` §6.1–6.4 | Full | Definition (Guo; DeGroot & Fienberg); calibration ≠ refinement; the base-rate caveat; bin construction, over/under-confidence direction, ECE/MCE, Brier decomposition; which model families are calibrated; modern nets and post-trained LLMs are over-confident (Guo; GPT-4 report; Tian). |
| 11 | Platt scaling and isotonic regression **[extension]** | `lecture/06` §6.5 | Full | Both mechanisms; temperature scaling; when to use which (~1,000-sample rule from NM&C and sklearn); held-out-data rule; `CalibratedClassifierCV` mechanics. Marked [extension]. |
| 12 | Why a guardrail at 0.9 fires far more than "10 % error" suggests | `lecture/05` §5.4 | Full | Bayes-rule precision formula; Gigerenzer's mammography numbers; breathalyzer/screening examples; a worked prompt-injection example (4,995 FP vs 800 TP); Llama Guard's use of AUPRC; OpenAI moderation docs on thresholds. Reinforced empirically in the build: the fraud file's top bin is 0.96 confident and 14.7 % correct. |
| 13 | Data quality: missing values (drop / impute / missingness as signal), class imbalance, distribution shift and how it is detected | `lecture/07` §7.1–7.3 | Full | MCAR/MAR/MNAR; drop, impute, indicator/MIA each with a paragraph and evidence (van Buuren; Perez-Lebel; Josse); imbalance options plus the 2022 evidence against resampling (van den Goorbergh; Elor); covariate/concept/label shift; detection: KS/χ², PSI with thresholds, adversarial validation, BBSD on model outputs (Rabanser), training/serving skew. |
| 14 | Awareness pass **[extension]**: LR · trees · RF · GB · NB · SVM · k-means · PCA — and where GB still beats a NN | `lecture/08` | Full | All eight named individually, one mechanism sentence + one "when" sentence + sklearn section each; §8.2 gives Grinsztajn's three findings and size regime, Shwartz-Ziv & Armon, McElfresh nuance, Stripe as the worked example. Marked [extension]. |

| Independent work | Where | Rating | Note |
|---|---|---|---|
| Build (2 h): `metrics.py` — confusion matrix, P, R, F1, threshold sweep, reliability diagram, from scratch, no sklearn | `independent_work/build_metrics/` | Full | All required functions plus ROC/AP, cost functions, ECE, Brier (needed by parts 5–6 and by Project 1). Skeleton with signatures and docstrings; 38 tests; two datasets; no-library rule stated; time-boxed sections. |
| System design drill (45 min): fraud under 100 ms; 10 recall / 30 aloud / write the delta | `independent_work/drills/system_design_fraud_100ms.md` | Full | Exact timing structure; recall questions tied to lecture parts; eight design layers; delta prompts against Stripe/PayPal sources read *after*; rubric; common misses hidden behind a disclosure. |
| Production drill (45 min): cost one shipped threshold in currency, or write who would know | `independent_work/drills/production_threshold_costing.md` | Full | Threshold-finding list for people who have not shipped ML; template with `C_FP`, `C_FN`, implied threshold, prevalence check, "who would know" table; requires actually sending the question. |
| Reading (30 min): precision/recall and ROC vs PR, marked sections | `READING.md` top section | Full | Three sources, exact sections, time estimates; full annotated list below it. |

**Verdict: 14/14 theory bullets and 4/4 independent-work items covered in full.** No bullet is covered by a single sentence; the thinnest is bullet 14, which the syllabus itself scopes to one sentence per model.

## B. Solution leakage — findings and fixes

The independent review found no code solutions in student-facing files. It flagged four places where *prose* gave away more than a specification should. All four were changed:

| Finding | Fix |
|---|---|
| `ASSIGNMENT.md` §3 and the "If stuck" row, and the `threshold_sweep` docstring, described the O(n log n) algorithm ("sort once, walk the sorted scores, running counts"). | Replaced with the complexity target and a question ("what does one pass over the data in some order buy you?"), leaving the design to the student. |
| `make_data.py` docstring stated the expected outcome of reflection questions Q1 and Q3 (ROC unchanged / PR changes; over-confident, below the diagonal). | Rewritten to say only that the two files share a classifier and that the diagram will tell you which way the temperature pushes it. The inline comment about the logit now says "notice what this formula does NOT include" instead of naming the missing prior term. |
| `test_metrics.py` comments narrated Q1 in words ("ROC AUC moves a little… AP moves a lot"), Q3 ("0.96 confident, 15 % right") and named the accuracy paradox. | Comments removed or reduced to a pointer to the NOTES question. Test names neutralized (`test_curves_across_prevalence`). The *numeric* expected values remain — they are checks, and the questions ask for the *why*, which no number gives away. |
| Docstrings for `bayes_threshold`, `expected_cost`, `brier_score`, `fbeta` state the formula. | Kept deliberately: these are one-line definitions the lecture teaches and the assignment labels them as such. The learning in the build is in the sweep, the curves, the step-sum AP and the binning, none of which is given. |

Accepted by design: the drill's "Common misses" sit behind a collapsed disclosure marked do-not-read-until-after; the "delta" section reveals Stripe/PayPal facts because the syllabus says "then write what you missed," which requires a reference to compare against.

## C. Cross-file consistency — findings and fixes

| Finding | Fix |
|---|---|
| Convention 5 said the "flag nothing" threshold comes *last*; the tests (correctly, given descending order) expect it *first*. | Wording fixed in `ASSIGNMENT.md` and the docstring: descending order puts it first. |
| Diagram title: assignment said ECE required, ECE + n "Strong"; docstring said both required. | Docstring aligned to the assignment. |
| "Each body is `raise NotImplementedError`" — `load_csv` is provided. | Wording fixed. |
| `NOTES.md` was a deliverable in the assignment but not in the README's table. | README updated. |
| Section time budget summed to 125 min for a 2 h build. | Rebalanced to ≈ 100 min coding + 15 min reflection. |
| `SELF_CHECK.md` said "20 of 24" over 29 questions. | Fixed to 24 of 29. |
| `ArrayLike` was a string constant used as a type; unused `Optional` import. | Proper `Union` alias; import cleaned. |
| `README.md` referenced this file before it existed. | This file. |
| Test cache and `__pycache__` were present from validation runs. | Deleted before delivery. |

All frozen reference values in `test_metrics.py` were independently recomputed by the reviewer from the CSVs and matched.

## D. Factual claims — findings and fixes

| Flag | Resolution |
|---|---|
| "Murphy's decomposition" of the Brier score was ambiguous with Kevin Murphy's textbook. | Attributed to Allan Murphy (1973) explicitly. |
| "Logistic regression is calibrated by construction" — overstated. | Softened to the empirical finding (NM&C) with the mechanism and the caveat about regularization/misspecification. Same fix in part 8. |
| sklearn's cost-sensitive example says "1/5 of the cost ratio" and "factor of 2"; Elkan's formula gives 1/6 ≈ 0.17 and −209 → −143 is ~30 %. | Verified the page really says that. The notes now quote the defensible sentence, do the arithmetic correctly, and flag the page's wording as loose. |
| Elkan's `C(i, j)` convention was transposed relative to his paper. | Fixed to Elkan's: predicting `i` when the truth is `j`; conditions and eq. 2 rewritten to match. The reduced form `C_FP/(C_FP+C_FN)` is unchanged. |
| `method='temperature'` in scikit-learn — is it real? | Verified on the current user guide (1.9.0). Version qualifier changed to "recent releases." |
| TabPFN "3,000-row limit" — the exact figure varies between TabPFN versions and McElfresh's setup. | Rephrased to "small training sets (a few thousand rows at most in that study)". |
| "AUC below 0.5 means your labels are inverted" — too strong. | Now "consistently below 0.5 usually means the labels or the score direction are inverted somewhere." |
| Temperature scaling described as "Platt with B = 0, A = 1/T" — sign convention loose. | Rephrased without the parameter mapping. |
| Specific numbers from papers (Loshchilov & Hutter's ~15 %; Llama Guard's AUPRC table; van den Goorbergh's intercept range; Perez-Lebel's 200×; Shwartz-Ziv & Armon's percentages; Gigerenzer's 21 %). | These were extracted from the fetched source text during research and are cited to their sections. Not re-verified a second time; a reader who finds a discrepancy should trust the paper and open an issue against these notes. |

## E. Known limitations and open items

- **Four sources could not be fetched from the authoring environment** (Hastie ESL free PDF; Quiñonero-Candela *Dataset Shift*; Platt 1999 full text; Saerens 2002 full text). They are named in `READING.md` with that caveat and are not load-bearing for any claim.
- **The "one sentence per model" pass** is one sentence of mechanism plus one of "when" — slightly over the syllabus's one sentence, on purpose, because the "when" is what an engineer will actually use.
- **Bullet 4 assumes the student knows what Adam does.** Part 2 gives a two-line description; Adam proper is week 3. A student who has never seen Adam should still be able to follow the argument (the decay term is or is not divided by the adaptive scale) without the derivation.
- **Double descent** is introduced as a caveat only; the mechanism is deferred to week 3 as the syllabus does not list it for week 1.
- **The build's expected values pin one set of conventions** (≥ thresholding, 0.0 for undefined ratios, `[i/n, (i+1)/n)` bins, step-sum AP). These match scikit-learn's numerical results for AUC/AP/Brier on the sample data but differ from `calibration_curve`'s bin-edge handling. This is stated in the assignment; a student who later compares with sklearn's reliability bins may see small differences at bin boundaries, which is itself worth noticing.
- **The system-design drill's rubric is self-graded.** It cannot check that the student actually spoke aloud or actually stopped at minute 40. The course's format relies on the student's honesty there, as the existing labs do ("verify can't see this — be honest").
- **Not covered, and not in the syllabus:** multiclass metrics beyond a note on averaging; MCC; log-loss as a metric in its own right; significance testing of metric deltas (week 1B); ranking metrics (week 1B); leakage (week 1B).

## F. What I would check next time

Have a second reviewer attempt the build cold with only `ASSIGNMENT.md`, `metrics.py` and `test_metrics.py`, and time it. The 2 h estimate is the least-evidenced number in the folder.
