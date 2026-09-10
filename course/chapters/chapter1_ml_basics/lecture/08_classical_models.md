# 8 · One-sentence awareness pass: eight classical models, and where gradient boosting still beats a neural network **[extension]**

*Lecture 1, part 8 of 8 · ~10 min reading*

This course is about LLM systems. You will nonetheless spend a surprising share of your career next to tabular data — fraud features, user attributes, request logs — and the models below are what wins there. The goal of this part is *recognition*, not fluency: when a colleague says "we used a random forest for the router," you should know what that is, what its failure mode is, and whether to be surprised. One sentence of mechanism each, one sentence of when, with the scikit-learn user-guide section to read if you ever need more.

## 8.1 The eight

**Logistic regression** (sklearn §1.1.11). A linear model whose output is squashed through a sigmoid, `p̂ = 1 / (1 + exp(−(Xw + w₀)))`, fitted by minimizing log-loss plus an L1/L2 penalty. *When:* your baseline for every classification problem, because it is fast, interpretable, and usually well calibrated without post-processing (part 6); Google's Rule #14 says start here for exactly those reasons.

**Decision tree** (§1.10). A "non-parametric supervised learning method" that "predicts the value of a target variable by learning simple decision rules inferred from the data features" — a nested sequence of `if feature < threshold` splits chosen greedily by Gini impurity or entropy, yielding "a piecewise constant approximation." *When:* almost never alone in production — it overfits and "can be unstable because small variations in the data might result in a completely different tree" — but it is the unit every ensemble below is built from, and the natural home of the MIA missing-value trick from part 7.

**Random forest** (§1.11.2.1). Many decision trees, each "built from a sample drawn with replacement (i.e., a bootstrap sample)" and each considering "a random subset of the features" at every split, with predictions averaged; "the purpose of these two sources of randomness … is to decrease the variance." *When:* you want a strong tabular model with almost no tuning, embarrassingly parallel training, and reasonable calibration (Niculescu-Mizil & Caruana found bagged trees well-calibrated).

**Gradient boosting** — XGBoost, LightGBM, CatBoost, sklearn's `HistGradientBoosting*` (§1.11.1). An additive model `ŷ = Σ h_m(x)` where each new small tree `h_m` "is fitted to predict the negative gradients" of the loss on the current ensemble — each tree corrects the residual errors of everything before it. Histogram variants bin features into ≤ 255 buckets, "have built-in support for missing values (NaNs)," and are "orders of magnitude faster … when the number of samples is larger than tens of thousands." *When:* the default winner on tabular data (§8.2). Its probabilities are miscalibrated out of the box (part 6 §6.4) — calibrate before using them as probabilities.

**Naive Bayes** (§1.9). Bayes' theorem "with the 'naive' assumption of conditional independence between every pair of features given the value of the class variable" — multiply per-feature likelihoods, pick the biggest class. *When:* text with bag-of-words features, tiny data, or a need for a model that trains in one pass; never when you need the probabilities — sklearn: "a decent classifier … but a bad estimator, so the probability outputs from `predict_proba` are not to be taken too seriously."

**Support vector machine** (§1.4). Finds the separating boundary with the widest margin, defined by "a subset of training points in the decision function (called support vectors)," with kernels to make the boundary non-linear in the original space. *When:* small-to-medium datasets with many features and clean margins; fitting scales between O(n²) and O(n³) in the number of samples, it is "not scale invariant, so it is highly recommended to scale your data," and its scores are not probabilities without Platt scaling — which is literally what Platt scaling was invented for.

**k-means** (§2.3.2). Unsupervised; picks `k` centroids to minimize inertia `Σ minⱼ ‖xᵢ − μⱼ‖²` by alternating "assign each point to its nearest centroid" and "move each centroid to the mean of its points" (Lloyd's algorithm; k-means++ picks well-spread starting centroids). *When:* you need to group things and have no labels — user segments, embedding clusters for a retrieval index (weeks 13–17); it assumes clusters "are convex and isotropic," so it fails on elongated or nested shapes.

**Principal component analysis** (§2.5.1.1). Unsupervised; rotates the data onto "a set of successive orthogonal components that explain a maximum amount of the variance," computed by SVD, and keeps the top few. *When:* compressing high-dimensional features (embeddings included) for speed, visualization, or noise reduction; note that "PCA centers but does not scale the input data," so standardize first if your features have different units.

## 8.2 Where gradient boosting still beats a neural network

The one thing from this part to actually remember. On *typical tabular data* — rows of heterogeneous columns, thousands to a few hundred thousand of them — tree ensembles beat deep learning, and the reasons are understood.

**The benchmark.** Grinsztajn, Oyallon & Varoquaux (NeurIPS 2022 D&B) ran 45 datasets (each ≥ 3,000 rows, ≥ 4 features) in two regimes, train sets truncated to 10,000 and to 50,000 rows, comparing XGBoost, random forests and gradient boosting against MLPs, ResNets, FT-Transformer and SAINT, with ~400 random-search iterations per model per dataset (~20,000 compute-hours). Tree models won clearly at 10k; "increasing the train set size reduces the gap between neural networks and tree-based models," but trees still led at 50k.

**The three reasons**, from their §5:

1. *"NNs are biased to overly smooth solutions."* Real tabular targets are often irregular — a step at age 65, a cliff at a credit limit — and trees "easily learn irregular functions" while networks fight their own smoothness prior.
2. *"Uninformative features affect more MLP-like NNs."* Tabular data is full of junk columns; trees are "robust to uninformative features" because a split on a useless column is never chosen, while an MLP has to learn to ignore it.
3. *"Data are non invariant by rotation, so should be learning procedures."* Each tabular column has its own meaning; a learner that treats a rotation of the feature space as equivalent (MLPs do) throws away that structure, while trees split on one raw feature at a time and "preserve the orientation of the data."

**Confirmation.** Shwartz-Ziv & Armon (2021, "Tabular Data: Deep Learning is Not All You Need") tested four tabular deep models against XGBoost on 11 datasets: "XGBoost outperforms these deep models across the datasets, including the datasets used in the papers that proposed the deep models" — "each deep model was better only on the datasets that appeared in its own paper." Average degradation on unseen datasets: XGBoost 3.3 %, versus 7.6–14.2 % for the deep models. XGBoost also converged faster under hyperparameter search. The one honest point for deep learning: an ensemble of XGBoost *and* the deep models beat XGBoost alone.

**The nuance.** McElfresh et al. (NeurIPS 2023 D&B), 19 algorithms × 176 datasets: "for a surprisingly high number of datasets, either the performance difference between GBDTs and NNs is negligible, or light hyperparameter tuning on a GBDT is more important than choosing between NNs and GBDTs." Boosted trees win especially on "irregular" data — skewed or heavy-tailed feature distributions — and on larger datasets. Their strongest single model on average was TabPFN, a transformer pre-trained on synthetic tables — but it only works on small training sets (a few thousand rows at most in that study), a hint that the picture may change, and a reason to re-check this section in a year.

**The rule for now.** If the input is a table, start with gradient boosting. If the input is text, images, audio, or sequences — anything where the *representation* has to be learned — start with a neural network. Stripe's Radar, the system in this week's design drill, is the illustration: they began with logistic regression, moved to an XGBoost + DNN ensemble, and only in 2022 — with training data scaled 10× and a specific architecture — did a pure DNN beat the trees. It took a very large fraud-detection team a very large amount of data to earn the right to drop the tree model.

---

**Sources for this part** (exact sections in `READING.md`): scikit-learn User Guide §1.1.11, §1.10, §1.11.1, §1.11.2.1, §1.9, §1.4, §2.3.2, §2.5.1.1 · Grinsztajn, Oyallon & Varoquaux, NeurIPS 2022, arXiv 2207.08815, §3–5 · Shwartz-Ziv & Armon 2021, arXiv 2106.03253 · McElfresh et al. 2023, arXiv 2305.02997, §5 · Stripe, "How we built it: Stripe Radar" (2023), Lesson 1.
