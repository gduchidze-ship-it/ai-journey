# Reading — Lecture 1

Every URL below was fetched and checked on 2026-09-10. Section numbers refer to the online editions on that date (scikit-learn renumbers its user guide between releases; if a section number is off by one, search the heading text). Following the course rule, each topic has at most three sources marked as *primary*; everything else is reference for when you need to go deeper.

## The 30-minute assigned reading (Independent work → Reading)

Read these, in this order, marked sections only. Do not read the whole paper.

1. **Google ML Crash Course — "Classification: Accuracy, recall, precision, and related metrics."** All of it; it is short. Pay attention to the "Choice of metric and tradeoffs" table.
   https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall

2. **Saito & Rehmsmeier (2015), "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets," PLOS ONE.** Read *Results → Simulation* only (the balanced vs. imbalanced experiment, the "baseline of PRC is P/(P+N)" argument, and the figures). About 10 minutes.
   https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432

3. **Davis & Goadrich (2006), "The Relationship Between Precision-Recall and ROC Curves," ICML.** Read §1–2 (two pages) and the first page of §4 (why linear interpolation in PR space is wrong). Skip the proofs. About 10 minutes.
   https://ftp.cs.wisc.edu/machine-learning/shavlik-group/davis.icml06.pdf

If you have time left: scikit-learn User Guide §3.4, the "Precision-recall curve" subsection — specifically the average-precision formula and the sentence about why it is not interpolated.
https://scikit-learn.org/stable/modules/model_evaluation.html

---

## Full annotated source list by topic

### 1 · Function fitting, generalization, bias–variance, capacity

| Source | Read | Why |
|---|---|---|
| **Murphy, *Probabilistic Machine Learning: An Introduction* (MIT Press 2022, free PDF)** — https://github.com/probml/pml-book/releases/latest/download/book1.pdf | §1.2.1.4 Empirical risk minimization; §1.2.3 Overfitting and generalization; §4.7.6.3 The bias-variance tradeoff; §5.4.1 Empirical risk; §13.5.7 Over-parameterized models | *Primary.* The cleanest modern statement of ERM, population risk, generalization gap, and the bias–variance decomposition. |
| **Goodfellow, Bengio & Courville, *Deep Learning* (MIT Press 2016), ch. 5** — https://www.deeplearningbook.org/contents/ml.html | §5.2 Capacity, Overfitting and Underfitting; §5.3 Hyperparameters and Validation Sets; §5.4.4 Trading off Bias and Variance | *Primary.* Figures 5.3 and 5.6 are the pictures to memorize. |
| scikit-learn User Guide §3.1 Cross-validation — https://scikit-learn.org/stable/modules/cross_validation.html | Intro paragraphs; §3.1.2; §3.1.4 | Why the test set cannot be used for tuning, in library terms. |
| Google ML Crash Course — "Overfitting" and "Dividing the original dataset" — https://developers.google.com/machine-learning/crash-course/overfitting/overfitting · https://developers.google.com/machine-learning/crash-course/overfitting/dividing-datasets | Both pages | The stationarity assumption; the test set as a consumable. |
| Belkin, Hsu, Ma & Mandal (2019), "Reconciling modern machine learning practice and the bias-variance trade-off," PNAS — https://arxiv.org/abs/1812.11118 | Abstract and Fig. 1 | Double descent, the caveat to the U-curve. |
| Nakkiran et al. (2019), "Deep Double Descent" — https://arxiv.org/abs/1912.02292 · companion post https://openai.com/index/deep-double-descent/ | Abstract; the post | Double descent in deep nets, along model size, epochs and data size. |
| Hastie, Tibshirani & Friedman, *Elements of Statistical Learning*, ch. 7 — https://link.springer.com/chapter/10.1007/978-0-387-84858-7_7 | §7.2, §7.3 | The classical statistics treatment. The authors' free PDF at hastie.su.domains could not be verified from the authoring environment; the book is freely available there. |

### 2 · Regularization and AdamW

| Source | Read | Why |
|---|---|---|
| **Goodfellow et al., *Deep Learning*, ch. 7** — https://www.deeplearningbook.org/contents/regularization.html | §7.1.1 L² Parameter Regularization; §7.1.2 L¹ Regularization; §7.2 Norm Penalties as Constrained Optimization; §7.8 Early Stopping | *Primary.* The eigen-analysis of L2 and the τ ≈ 1/(εα) derivation for early stopping. |
| **Loshchilov & Hutter (2017/ICLR 2019), "Decoupled Weight Decay Regularization"** — https://arxiv.org/abs/1711.05101 (PDF https://arxiv.org/pdf/1711.05101) | §2, Propositions 1–2, Algorithm 2 | *Primary.* The AdamW paper. Algorithm 2 lines 6 vs 12 is the whole argument. |
| **PyTorch docs — `torch.optim.AdamW`, `torch.optim.Adam`, `torch.optim.SGD`** — https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html · https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html · https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html | The pseudocode block on each page | *Primary.* Read where `λθ` enters in each. |
| Murphy, *PML: An Introduction* | §11.3 Ridge; §11.4.2 Why does ℓ1 regularization yield sparse solutions? (Fig. 11.8); §13.5.1 Early stopping; §13.5.2 Weight decay | The diamond-vs-circle picture. |
| scikit-learn User Guide §1.1 Linear Models — https://scikit-learn.org/stable/modules/linear_model.html | §1.1.2 Ridge; §1.1.3 Lasso | The objectives as a library writes them. |
| Gugger & Howard (fast.ai, 2018), "AdamW and Super-convergence is now the fastest way to train neural nets" — https://www.fast.ai/posts/2018-07-02-adam-weight-decay.html | First half | The AdamW argument in code, for people who prefer code. |
| Goh (Distill, 2017), "Why Momentum Really Works" — https://distill.pub/2017/momentum/ | The polynomial-regression / early-stopping section | Early stopping ≈ Tikhonov, from the optimization side, with interactive figures. |
| Google ML Crash Course — "L2 regularization" — https://developers.google.com/machine-learning/crash-course/overfitting/regularization | Whole page incl. early stopping | Gentle version. |

### 3 · Training curves and shipping decisions

| Source | Read | Why |
|---|---|---|
| **Zinkevich, "Rules of Machine Learning: Best Practices for ML Engineering" (Google)** — https://developers.google.com/machine-learning/guides/rules-of-ml | Terminology; Rules #2, #13, #14, #24, #37, #39 | *Primary.* Metric vs objective; the three skews; launch decisions. |
| **Breck, Cai, Nielsen, Salib & Sculley (2017), "The ML Test Score"** — https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/ (PDF https://storage.googleapis.com/gweb-research2023-media/pubtools/4156.pdf) | §III Model 2, 5, 6; §V Monitor 7; §VI the rubric | *Primary.* The 28-test rubric; score = minimum over sections. |
| Google ML Crash Course — "Interpreting loss curves" — https://developers.google.com/machine-learning/crash-course/overfitting/interpreting-loss-curves | Whole page | Field guide to curve shapes. |

### 4 · Classification metrics; ROC vs PR

| Source | Read | Why |
|---|---|---|
| **scikit-learn User Guide §3.4 Metrics and scoring** — https://scikit-learn.org/stable/modules/model_evaluation.html | §3.4.4.4 Balanced accuracy; §3.4.4.6 Confusion matrix; §3.4.4.9 Precision, recall and F-measures; the ROC and Precision-recall subsections | *Primary.* Exact definitions your `metrics.py` must match. |
| **Davis & Goadrich (2006)** — https://ftp.cs.wisc.edu/machine-learning/shavlik-group/davis.icml06.pdf | §1–2; §3 Theorems 3.1–3.2 (statements only); §4 | *Primary.* Assigned. |
| **Saito & Rehmsmeier (2015)** — https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432 | Results → Simulation | *Primary.* Assigned. |
| Google ML Crash Course — "Thresholds and the confusion matrix"; "Accuracy, recall, precision"; "ROC and AUC" — https://developers.google.com/machine-learning/crash-course/classification/thresholding · https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall · https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc | All three pages | The AUC-as-probability interpretation; the metric-choice table. |
| "F-score," Wikipedia — https://en.wikipedia.org/wiki/F-score | Definition; Fβ; Criticism | F-beta as van Rijsbergen defined it; the Chicco & Jurman MCC critique. Fallback source. |
| "Accuracy paradox," Wikipedia — https://en.wikipedia.org/wiki/Accuracy_paradox | Whole article | The 1,000,000 / 10 worked example. Fallback source. |

### 5 · Thresholds, costs, base rates, guardrails

| Source | Read | Why |
|---|---|---|
| **Elkan (2001), "The Foundations of Cost-Sensitive Learning," IJCAI** — https://cseweb.ucsd.edu/~elkan/rescale.pdf | "Making decisions based on a cost matrix"; "Making optimal decisions" (eq. 2); "Effects of changing base rates" | *Primary.* The optimal threshold `p*` and its invariances. |
| **scikit-learn User Guide §3.3 Tuning the decision threshold** — https://scikit-learn.org/stable/modules/classification_threshold.html · example https://scikit-learn.org/stable/auto_examples/model_selection/plot_cost_sensitive_learning.html | §3.3.1; the whole example | *Primary.* `TunedThresholdClassifierCV`; the −209 → −143 credit example; per-transaction costs. |
| **Gigerenzer et al. (2007), "Helping Doctors and Patients Make Sense of Health Statistics," *Psychological Science in the Public Interest*** — https://www.stat.berkeley.edu/~aldous/157/Papers/health_stats.pdf | "Few Gynecologists Understand Positive Mammograms" | *Primary.* The 1 % / 90 % / 9 % example and the physicians who got it wrong. |
| "Base rate fallacy," Wikipedia — https://en.wikipedia.org/wiki/Base_rate_fallacy | "False positive paradox"; "Examples"; "Mathematical formalism" | Breathalyzer and screening examples. Fallback source. |
| Inan et al. (Meta, 2023), "Llama Guard" — https://arxiv.org/abs/2312.06674 | §4 Experiments, Table 2; Appendix B | A production guardrail evaluated by AUPRC. |
| OpenAI Moderation guide — https://developers.openai.com/api/docs/guides/moderation | Quickstart; notes on `category_scores` | The vendor telling you the threshold is your policy. |
| Stripe, "A primer on machine learning for fraud detection" — https://stripe.com/guides/primer-on-machine-learning-for-fraud-protection | "Evaluating machine learning models" | Precision/recall with an explicit `P(fraud) > 0.7` policy; the cost of a false decline. |

### 6 · Calibration; Platt scaling and isotonic regression

| Source | Read | Why |
|---|---|---|
| **Guo, Pleiss, Sun & Weinberger (ICML 2017), "On Calibration of Modern Neural Networks"** — https://arxiv.org/abs/1706.04599 | §2 Definitions; §3 (Fig. 1); §4.2 Temperature scaling; §5 Table 1 | *Primary.* Reliability diagrams, ECE, MCE, the over-confidence finding, temperature scaling. |
| **Niculescu-Mizil & Caruana (ICML 2005), "Predicting Good Probabilities With Supervised Learning"** — https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf | §2 Calibration Methods; §4 Qualitative Analysis; §5 Learning Curve Analysis | *Primary.* Which model families are calibrated; Platt vs isotonic by calibration-set size. |
| **scikit-learn User Guide §1.16 Probability calibration** — https://scikit-learn.org/stable/modules/calibration.html · https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html | §1.16.1–1.16.3 | *Primary.* The held-out rule; sigmoid vs isotonic vs temperature; the ~1000-sample rule of thumb. |
| DeGroot & Fienberg (1983), "The Comparison and Evaluation of Forecasters" — https://academic.oup.com/jrsssd/article-abstract/32/1-2/12/7121286 · full technical-report version https://apps.dtic.mil/sti/pdfs/ADA135966.pdf | Tech report §2 Calibration and Refinement; §4 Strictly Proper Scoring Rules | Calibration vs refinement; the origin of the reliability diagram. |
| Platt (1999), "Probabilistic Outputs for Support Vector Machines…" — https://transferlab.ai/refs/platt_probabilistic_1999/ (abstract; the original PDF hosts were not reachable) | Abstract | The original sigmoid method. |
| Zadrozny & Elkan (KDD 2002), "Transforming Classifier Scores into Accurate Multiclass Probability Estimates" — http://www.cs.columbia.edu/~djhsu/coms4771-f25/handouts/zadrozny2002kdd.pdf | §2–3 | Isotonic regression / PAV for calibration. |
| scikit-learn `brier_score_loss` — https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html · "Brier score," Wikipedia, §Decomposition — https://en.wikipedia.org/wiki/Brier_score | Both | Brier = reliability − resolution + uncertainty. |
| OpenAI (2023), GPT-4 Technical Report — https://cdn.openai.com/papers/gpt-4.pdf | §5 Limitations, Figure 8 | Post-training hurts calibration. |
| Kadavath et al. (Anthropic, 2022), "Language Models (Mostly) Know What They Know" — https://arxiv.org/abs/2207.05221 | Abstract; §1 | LLM calibration on multiple choice; P(True), P(IK). |
| Tian et al. (EMNLP 2023), "Just Ask for Calibration" — https://arxiv.org/abs/2305.14975 | Abstract | Verbalized confidence beats token probabilities for RLHF models. |
| Saerens, Latinne & Decaestecker (2002), *Neural Computation* — https://direct.mit.edu/neco/article/14/1/21/6577/ (abstract) · Lipton, Wang & Smola (ICML 2018), "Detecting and Correcting for Label Shift with Black Box Predictors" — https://arxiv.org/abs/1802.03916 · Moreo (2025) — https://arxiv.org/html/2505.11380 | Abstracts; Lipton §Problem Setup; Moreo Def. 3.2, §5.2.1 | Why calibration breaks under prior shift, and the prior-ratio fix. |

### 7 · Data quality

| Source | Read | Why |
|---|---|---|
| **van Buuren, *Flexible Imputation of Missing Data*, 2nd ed. (free online)** — https://stefvanbuuren.name/fimd/sec-MCAR.html · https://stefvanbuuren.name/fimd/sec-simplesolutions.html | §1.2 MCAR/MAR/MNAR; §1.3 Ad-hoc solutions (1.3.1, 1.3.3, 1.3.4, 1.3.7) | *Primary.* The taxonomy with the scale example; what each ad-hoc fix distorts. |
| **scikit-learn User Guide §8.4 Imputation of missing values** — https://scikit-learn.org/stable/modules/impute.html | §8.4.2–8.4.4; §8.4.6 Marking imputed values; §8.4.7 Estimators that handle NaN | *Primary.* `add_indicator`, `MissingIndicator`, native-NaN estimators. |
| **Rabanser, Günnemann & Lipton (NeurIPS 2019), "Failing Loudly"** — https://arxiv.org/abs/1810.11953 | §3 Shift detection techniques; §5 Results; §6 Discussion | *Primary.* Test on model outputs (BBSDs); 20-sample detection. |
| Perez-Lebel et al. (GigaScience 2022), "Benchmarking missing-values approaches for predictive models on health databases" — https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giac013/6568998 · https://arxiv.org/abs/2202.10580 | Abstract; Results | Indicators help; native tree handling wins and is 200× cheaper. |
| Josse, Chen, Prost, Scornet & Varoquaux (2019), "On the consistency of supervised learning with missing values" — https://arxiv.org/abs/1902.06931 | Abstract | Constant imputation is consistent for prediction when missingness is uninformative; MIA for trees. |
| XGBoost FAQ — https://xgboost.readthedocs.io/en/stable/faq.html · LightGBM Advanced Topics — https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html | "How to deal with missing values"; "Missing Value Handle" | Library behaviour. |
| Chawla et al. (JAIR 2002), "SMOTE" — https://www.jair.org/index.php/jair/article/view/10302 · imbalanced-learn user guide — https://imbalanced-learn.org/stable/over_sampling.html · https://imbalanced-learn.org/stable/under_sampling.html · https://imbalanced-learn.org/stable/common_pitfalls.html | SMOTE abstract; imblearn §2.1.2, §2.2.1, §3, §9.1 | The methods, and the leakage pitfall. |
| van den Goorbergh et al. (JAMIA 2022), "The harm of class imbalance corrections for risk prediction models" — https://academic.oup.com/jamia/article/29/9/1525/6605096 | Abstract; Results | Resampling wrecks calibration; threshold moving is equivalent. |
| Elor & Averbuch-Elor (2022), "To SMOTE, or not to SMOTE?" — https://arxiv.org/abs/2201.08528 | Abstract; Conclusions | Balancing helps weak learners, not strong ones. |
| Lin et al. (ICCV 2017), "Focal Loss for Dense Object Detection" — https://arxiv.org/abs/1708.02002 | Abstract | The one-line cost-sensitive loss for deep nets. |
| scikit-learn `compute_class_weight` — https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html | Whole page | The `balanced` formula. |
| Lu et al. (IEEE TKDE 2018), "Learning under Concept Drift: A Review" — https://arxiv.org/abs/2004.05785 | §2.1 Problem description | Covariate / concept / both; temporal patterns. |
| Yurdakul (2018), "Statistical Properties of Population Stability Index," PhD dissertation — https://scholarworks.wmich.edu/dissertations/3208/ | Abstract | The PSI thresholds and the fact they are conventions. |
| Evidently AI — "Which test is the best? We compared 5 methods to detect data drift on large datasets" — https://www.evidentlyai.com/blog/data-drift-detection-large-datasets · drift docs https://docs.evidentlyai.com/metrics/explainer_drift · guide https://www.evidentlyai.com/ml-in-production/data-drift | Blog whole; docs "defaults" section | KS over-sensitivity at scale; the 1,000-row switch to distances. |
| NannyML — Univariate drift detection — https://nannyml.readthedocs.io/en/stable/how_it_works/univariate_drift_detection.html | Continuous and categorical methods | Plain definitions of KS, JS, Wasserstein, Hellinger, χ², L∞. |
| scipy `ks_2samp` — https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html | Whole page | The statistic you will use in the drill. |
| FastML, "Adversarial validation, part one" — https://fastml.com/adversarial-validation-part-one/ · Qian et al. (2021) — https://arxiv.org/abs/2112.10078 | Blog whole; abstract | Domain classifier as drift detector. |
| Sculley et al. (NIPS 2015), "Hidden Technical Debt in Machine Learning Systems" — https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html | Abstract; §4 Feedback loops; §6 Changes in the External World | Drift you cause yourself. |
| Google, "Data and feature debugging" — https://developers.google.com/machine-learning/testing-debugging/common/data-errors | "Check for training-serving skew"; "Monitor model performance" | Skew monitoring. |

### 8 · Classical models; trees vs. neural networks on tabular data

| Source | Read | Why |
|---|---|---|
| **Grinsztajn, Oyallon & Varoquaux (NeurIPS 2022 D&B), "Why do tree-based models still outperform deep learning on typical tabular data?"** — https://arxiv.org/abs/2207.08815 | §3 Benchmark; §4 Results; §5 Findings 1–3 | *Primary.* The three explanations and the size regime. |
| **Shwartz-Ziv & Armon (2021), "Tabular Data: Deep Learning is Not All You Need"** — https://arxiv.org/abs/2106.03253 | Abstract; §4 Results | *Primary.* Deep tabular models only win on their own paper's datasets. |
| McElfresh et al. (NeurIPS 2023 D&B), "When Do Neural Nets Outperform Boosted Trees on Tabular Data?" — https://arxiv.org/abs/2305.02997 | Abstract; §5 | The nuance: often negligible; tuning matters more; irregular data favours trees. |
| scikit-learn User Guide — Logistic regression §1.1.11 https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression · Decision trees §1.10 https://scikit-learn.org/stable/modules/tree.html · Random forests §1.11.2.1 and Gradient boosting §1.11.1 https://scikit-learn.org/stable/modules/ensemble.html · Naive Bayes §1.9 https://scikit-learn.org/stable/modules/naive_bayes.html · SVM §1.4 https://scikit-learn.org/stable/modules/svm.html · k-means §2.3.2 https://scikit-learn.org/stable/modules/clustering.html#k-means · PCA §2.5.1.1 https://scikit-learn.org/stable/modules/decomposition.html#pca | First subsection of each | One-sentence mechanism and caveats, from the reference implementation's own docs. |

### System design drill — fraud under 100 ms

| Source | Read | Why |
|---|---|---|
| **Stripe, "How we built it: Stripe Radar" (Drapeau, 2023)** — https://stripe.dev/blog/how-we-built-it-stripe-radar | Lessons 1–3 | *Primary.* "More than 1,000 characteristics … in less than 100 milliseconds"; logistic regression → XGBoost+DNN → DNN-only; recall as the guardrail metric. Read *after* the drill, not before. |
| Stripe Radar docs, "Risk evaluation" — https://docs.stripe.com/radar/risk-evaluation | Risk levels; Risk score; Rules | Score 0–99; thresholds 65/75; `not_assessed` / `unknown` levels — what a fallback looks like in a product. |
| PayPal, "How PayPal Uses Real-time Graph Database and Graph Analysis to Fight Fraud" — https://medium.com/paypal-tech/how-paypal-uses-real-time-graph-database-and-graph-analysis-to-fight-fraud-96a2b918619a | Requirements list; Real-time Graph Stack | 100 ms SLA; 10 ms two-hop queries; seconds-level freshness. |
| Uber, "Palette Meta Store Journey" — https://www.uber.com/us/en/blog/palette-meta-store-journey/ | Whole post | Why a feature store exists: offline/online parity. |

---

*Sources that could not be verified from the authoring environment and are therefore not cited in the notes:* the Hastie ESL free PDF (hastie.su.domains, proxy-blocked; the Springer chapter page loads), Quiñonero-Candela et al. *Dataset Shift in Machine Learning* (MIT Press, 403), Platt 1999 full text (only the abstract page was reachable), and the Saerens et al. 2002 full text (abstract only). They are real and worth reading if you can reach them.
