# 7 · Data quality: missing values, class imbalance, distribution shift

*Lecture 1, part 7 of 8 · ~20 min reading*

Part 1 said a model is a function fitted to a dataset under the assumption that future data looks like that dataset. This part is about the three most common ways the dataset betrays you: some of it is not there, most of it is one class, and the world it describes has moved on.

## 7.1 Missing values

### Three kinds of missing

Rubin's taxonomy, stated by van Buuren (*Flexible Imputation of Missing Data*, §1.2) with a scale as the running example:

- **MCAR — missing completely at random.** "If the probability of being missing is the same for all cases." The scale's batteries died. Missingness carries no information.
- **MAR — missing at random.** "If the probability of being missing is the same only within groups defined by the observed data." The scale fails more on soft surfaces, and you recorded the surface. Missingness is predictable from columns you have.
- **MNAR — missing not at random.** "If neither MCAR nor MAR holds." The scale fails more for heavy objects — the *missing value itself* predicts its own absence. Missingness is predictable only from the thing you cannot see.

Real data is almost never MCAR. A customer who declines to state income is not a random customer. A sensor that drops readings under load is not dropping random readings. Before choosing what to *do* about missing values, ask which kind they are, because the right move differs.

### Three things to do

**Drop.** Delete rows (listwise deletion) or columns with missing values. Van Buuren §1.3.1: "If the data are MCAR, listwise deletion produces unbiased estimates" but is "potentially wasteful," and under anything other than MCAR it "can severely bias estimates of means, regression coefficients and correlations." Dropping rows is dropping the customers who did not state their income — a systematically different population.

**Impute.** Fill with a constant, the column mean or median, the mode, a nearest-neighbour value, or a value predicted from other columns (scikit-learn §8.4: `SimpleImputer`, `KNNImputer`, `IterativeImputer` — the last "inspired by the R MICE package … but differs from it by returning a single imputation instead of multiple imputations"). Van Buuren on the simplest version, §1.3.3: "Mean imputation distorts the distribution in several ways" — it underestimates variance, disturbs relations between variables, and biases "almost any estimate other than the mean." Regression imputation "artificially strengthens the relations in the data." These warnings are written for statisticians estimating parameters. For a *predictive* model the calculus is different — Josse et al. (2019) prove that imputing with a constant before learning "is consistent when missing values are not informative" — but the warning about MNAR stands: if the missingness is informative, filling it in *erases the information*.

**Treat missingness as signal.** Add a binary indicator column "was this value missing?" alongside the imputed value. scikit-learn: `SimpleImputer(add_indicator=True)`, or `MissingIndicator` directly. Van Buuren §1.3.7 notes that from the inferential viewpoint the indicator method "can yield severely biased regression estimates, even under MCAR" — but again, that is a statement about estimating coefficients. For prediction, the empirical evidence points the other way. Perez-Lebel et al. (GigaScience 2022), benchmarking on large health databases: "Adding an indicator to express which values have been imputed is important, suggesting that the data are missing not at random." And their headline: "Native support for missing values in supervised machine learning predicts better than state-of-the-art imputation with much less computational cost." The native method they mean is **missing incorporated in attribute (MIA)** — the tree learns, at each split, which branch missing values should take. It was the best overall performer in their benchmark and "up to 200 times faster" than iterative imputation.

**In practice.** Gradient-boosted trees handle missing values natively and you should let them: XGBoost — "supports missing values by default. In tree algorithms, branch directions for missing values are learned during training"; LightGBM — "enables the missing value handle by default," representing missing as NaN unless `zero_as_missing=true`; scikit-learn's `HistGradientBoosting*`, `DecisionTree*`, `RandomForest*` are all listed in §8.4.7 as NaN-capable. For linear models and neural networks, impute (median is a safer default than mean) *and* add the indicator. And in all cases: fit the imputer on the training split only, then apply it to validation and test — imputing on the full dataset before splitting is a leak (week 1B).

## 7.2 Class imbalance

Part 4 established that imbalance breaks accuracy and separates ROC from PR. This section is about what to *do*, and the answer has changed in the last few years.

### The options

- **Class weights / cost-sensitive loss.** Weight minority examples more in the loss. scikit-learn's `class_weight='balanced'` uses `n_samples / (n_classes · count(class))`; e.g. `y = [1,1,1,1,0,0]` → weights `[1.5, 0.75]`. Focal loss (Lin et al. 2017) is the deep-learning variant: it "down-weights the loss assigned to well-classified examples" so that "the vast number of easy negatives" does not "overwhelm the detector during training."
- **Oversampling.** Duplicate minority rows, or synthesize new ones. **SMOTE** (Chawla et al. 2002) interpolates: `x_new = xᵢ + λ(x_zi − xᵢ)`, a point on the segment between a minority example and one of its k nearest minority neighbours. Variants (Borderline-SMOTE, ADASYN, SMOTENC for mixed types) live in `imbalanced-learn` §2.
- **Undersampling.** Throw away majority rows (`RandomUnderSampler`, or smarter prototype selection such as NearMiss / Tomek links, `imbalanced-learn` §3).
- **Threshold moving.** Leave the data and the model alone; move the decision threshold (part 5).

### What the evidence now says

Two recent studies, one in medicine and one on 73 benchmark datasets, converge:

Van den Goorbergh et al. (JAMIA 2022), 24 simulation scenarios × 2,000 runs plus a clinical case study: "The use of random undersampling, random oversampling, or SMOTE yielded poorly calibrated models: the probability to belong to the minority class was strongly overestimated." Discrimination did not improve — AUROC stayed at 0.79–0.80 for every method — while calibration intercepts swung from near zero to between −0.7 and −4.5. And: "threshold adjustment on uncorrected data reached the same sensitivity/specificity balance." Conclusion: "correcting class imbalance did not result in better prediction models."

Elor & Averbuch-Elor (2022, "To SMOTE, or not to SMOTE?"), 73 imbalanced datasets, strong learners (CatBoost, XGBoost, LightGBM) against weak ones: "Balancing could improve prediction performance for weak classifiers but not for the SOTA classifiers." For metrics computed at a threshold (F1, balanced accuracy), "optimizing the decision threshold is recommended due to simplicity and lower compute cost."

The mechanism should be obvious after part 6: resampling *changes the base rate the model sees*, so its probabilities are calibrated to a fictional world where positives are common. Every downstream use of those probabilities — the optimal threshold, expected cost, a routing decision — inherits the distortion. Elkan's 2001 advice (part 5) was already this: fit on the data as given, then decide with costs.

**Default recipe for this course:** gradient boosting or a calibrated model on the data as it is → threshold from costs on a validation sweep → PR curve and prevalence reported side by side. Reach for resampling only with a weak learner you cannot replace, and if you do, resample *inside* the cross-validation fold — `imbalanced-learn` §9.1 warns that resampling "the entire dataset before splitting it into a train and a test partitions" leaks synthetic copies of test points into training.

## 7.3 Distribution shift

The stationarity assumption from part 1, failing.

### Three kinds of shift

Factor the joint distribution as `P(X, y) = P(X) · P(y | X)`. Lu et al.'s concept-drift survey (2018, §2.1) sorts every shift by which factor moved:

- **Covariate shift** — `P(X)` changes, `P(y|X)` does not. The inputs look different but the rule is the same. New users from a new region; a camera replaced by a sharper one. Lu et al. call it "virtual drift."
- **Concept drift** — `P(y|X)` changes. The rule itself moved. What counted as spam in 2019 does not in 2026; fraudsters adapted to your detector. Lu et al.: "actual/real drift."
- **Label shift** (prior probability shift) — `P(y)` changes, `P(X|y)` does not (Lipton et al. 2018: "the label marginal p(y) changes but the conditional p(x|y) does not"). Same kinds of fraud, more of it. This is the one that breaks calibration while leaving ROC intact (part 6, §6.6).

Lu et al. also name the *temporal patterns*: sudden, gradual, incremental, reoccurring (seasonal). Knowing the pattern tells you whether to retrain once or on a schedule.

### How shift is detected

You are comparing two samples — a reference (training, or last month) and a current window — and asking whether they came from the same distribution. The methods, from simplest to strongest:

**Per-feature statistical tests.** For a numeric feature, the two-sample **Kolmogorov–Smirnov** test: the statistic is "the maximum absolute difference between the empirical distribution functions of the samples" (scipy `ks_2samp`), null hypothesis that the two CDFs are identical. For a categorical feature, chi-squared on the category frequencies. Caveat from Evidently's benchmark: "the KS test tends to be pretty sensitive in larger datasets. It raises flags even for a minor change of 0.5 %, as soon as we have more than 100,000 objects." p-values shrink with sample size; with production volumes everything is "significant." Evidently's own defaults therefore switch at 1,000 rows from tests (KS / χ² / z-test, drift if `p ≤ 0.05`) to *distances* (Wasserstein normalized by standard deviation, Jensen–Shannon, drift if distance ≥ 0.1).

**Population Stability Index (PSI).** The finance industry's standard: bin the feature, compute `Σ (cur% − ref%) · ln(cur% / ref%)` over the bins. The rule of thumb, quoted in Yurdakul's 2018 dissertation on PSI: "PSI < 0.10 means 'little shift', .10–0.25 means 'significant shift, action required'" (> 0.25 major). Yurdakul's point is that "these benchmarks are being used without reference to Type I or Type II error rates" — they are conventions, not statistics; useful as an alarm threshold, not as a p-value.

**Adversarial validation / domain classifier.** Label reference rows 0 and current rows 1, shuffle, train a classifier to tell them apart. If its AUC is ≈ 0.5 the two windows are indistinguishable; if it is 0.9 something moved, and the classifier's feature importances tell you *what*. FastML's write-up is the standard introduction; Qian et al. (2021) use it to pick validation rows that resemble production. Rabanser et al. ("Failing Loudly," NeurIPS 2019) found that "domain-discriminating approaches tend to be helpful for characterizing shifts qualitatively and determining if they are harmful."

**Test on the model's outputs, not the inputs.** Rabanser et al.'s main empirical result, across many detection pipelines: "a two-sample-testing-based approach, using pre-trained classifiers for dimensionality reduction, performs best." Concretely: take the model's softmax outputs on reference and current data and run a KS test on them ("BBSDs"). The model has already compressed the inputs to the dimensions it cares about, so shifts *that matter to the model* show up and shifts it ignores do not. They found "large shifts can on average already be detected with better than chance accuracy at only 20 samples." Monitoring the prediction distribution is also what you can do when labels are delayed (fraud labels arrive with chargebacks, weeks later); Evidently: "monitoring input data distribution drift is a valuable proxy."

**Training–serving skew** is the special case that is an engineering bug rather than a world change: the feature computed at serving time differs from the one computed at training time (different code path, different timezone, a null handled differently). Google's guidance: "The Golden Rule: Ensure that training and production mimic each other as closely as possible"; monitor schema skew and feature skew, tracking "the number of detected skewed features, and the ratio of skewed examples per feature." Sculley et al. (2015) list "changes in the external world" and "hidden feedback loops" among the forms of technical debt specific to ML systems — the second being the case where *your model's decisions change the data it will next be trained on*, which is drift you caused.

### What to do about it

Detect (above), then diagnose which kind: label shift → reweight priors (part 6, §6.6) and recheck the threshold; covariate shift → retrain on recent data or reweight training rows; concept drift → the rule changed, you need new labels and a new model; skew → fix the pipeline. The *ML Test Score* (part 3) gives a point only for the automated version. A drift check you run by hand once is a demo.

## 7.4 What to carry forward

Ask why the data is missing before deciding what to do; for prediction, let trees handle NaN natively, otherwise impute *and* add an indicator, fitted on the training split only. Do not resample to fix imbalance with a strong learner — it wrecks calibration and threshold-moving does the same job; if you must, resample inside the fold. Factor `P(X,y) = P(X)P(y|X)` to name the shift; detect it with KS/PSI per feature, a domain classifier, or a KS test on the model's own outputs; and treat "significant at n = 10⁶" with suspicion.

---

**Sources for this part** (exact sections in `READING.md`): van Buuren, *Flexible Imputation of Missing Data* §1.2, §1.3 · scikit-learn User Guide §8.4 · Josse et al. 2019, arXiv 1902.06931 · Perez-Lebel et al., GigaScience 2022 · XGBoost FAQ, LightGBM Advanced Topics · Chawla et al., JAIR 2002 · imbalanced-learn §2, §3, §9.1 · scikit-learn `compute_class_weight`, §3.3 · van den Goorbergh et al., JAMIA 2022 · Elor & Averbuch-Elor 2022, arXiv 2201.08528 · Lin et al. 2017, arXiv 1708.02002 · Lu et al. 2018, arXiv 2004.05785, §2.1 · Lipton et al. 2018, arXiv 1802.03916 · Rabanser et al. 2019, arXiv 1810.11953, §3, §5, §6 · Yurdakul 2018 (PSI) · Evidently AI drift docs and blog · NannyML univariate drift docs · scipy `ks_2samp` · FastML adversarial validation; Qian et al. 2021 · Sculley et al., NIPS 2015 · Google, "Data and feature debugging."
