# 6 · Calibration: what a confidence score does and does not mean

*Lecture 1, part 6 of 8 · ~20 min reading · §6.5 is an [extension]*

## 6.1 The claim a probability makes

When a model outputs 0.8, it is making a claim. The claim is not "this one is positive." The claim is: *among all the cases I score 0.8, eight in ten are positive.* A model whose claims of this kind are true is **calibrated**. Formally (Guo et al. 2017, §2):

```
P( Y = 1 | p̂ = p ) = p     for all p in [0, 1]
```

scikit-learn's plain-English version (§1.16): "a well calibrated (binary) classifier should classify the samples such that among the samples to which it gave a `predict_proba` value close to, say, 0.8, approximately 80 % actually belong to the positive class." The definition is older than machine learning — DeGroot & Fienberg (1983) wrote it for weather forecasters: a forecaster "is said to be well-calibrated if p(x) = x for all x" — among the days you said 30 %, it rained on 30 %.

Why you care: part 5's optimal threshold `t* = C_FP/(C_FP + C_FN)` is only optimal if the score is a probability. Every downstream use that *adds or compares* scores — expected cost, expected value of a transaction, "route to a human if below 0.6", an LLM router choosing a cheaper model when confident — silently assumes calibration. Ranking (AUC) does not.

## 6.2 What calibration does *not* mean

Two things a calibrated score does not tell you.

**It does not tell you the model is good.** DeGroot & Fienberg separate **calibration** from **refinement**. A forecaster who says "1 % chance of fraud" on every single transaction, in a world where 1 % of transactions are fraud, is *perfectly calibrated* and *perfectly useless* — it cannot separate anything from anything. Refinement (also called resolution, or sharpness) is the ability to push scores toward 0 and 1 *while staying calibrated*. You want both, and they are different axes. Part 4's AUC measures the second axis and is blind to the first; a reliability diagram measures the first and is blind to the second.

**It does not tell you how many flags you will get.** The base-rate argument of part 5 §5.4: "0.9" is a statement about the items scored 0.9, not about the size of the population that will score 0.9. Calibration fixes the *meaning* of the score; prevalence fixes the *count*.

## 6.3 Reliability diagrams

The standard picture, defined in Guo et al. §2. Take your validation set. Bin the predictions into `M` equal-width confidence bins (ten is conventional — Niculescu-Mizil & Caruana: "the prediction space is discretized into ten bins"). For each bin `B_m` compute

```
conf(B_m) = mean predicted probability of the items in the bin
acc(B_m)  = fraction of items in the bin that are actually positive
```

Plot `acc` against `conf`. A perfectly calibrated model lies on the diagonal. Points *below* the diagonal mean the model is **over-confident** — it said 0.9 and only 0.7 came true. Points *above* mean **under-confident**. Always draw the bin counts too (a histogram underneath is conventional); a bin with six items tells you nothing, and the high-confidence bins on rare-positive data are often nearly empty.

Two scalar summaries, both from Guo et al. §2:

```
ECE = Σ_m (|B_m| / n) · | acc(B_m) − conf(B_m) |        expected calibration error
MCE = max_m           | acc(B_m) − conf(B_m) |           maximum calibration error
```

ECE is the count-weighted average gap between the diagonal and your points; MCE is the worst bin. ECE depends on the binning, so state `M` when you report it.

**The Brier score** is the other number to know: `(1/N) Σ (yᵢ − p̂ᵢ)²`, mean squared error between the probability and the 0/1 outcome (scikit-learn `brier_score_loss`; range [0, 1] for binary, lower is better). It is a *strictly proper scoring rule*, which means the only way to minimize it is to report your true beliefs. Allan Murphy's 1973 decomposition (the meteorologist — not the textbook author cited elsewhere in these notes), `Brier = Reliability − Resolution + Uncertainty`, splits it into exactly the axes of §6.2: reliability (calibration error; 0 is perfect), resolution (refinement; higher is better, and it enters with a minus sign), and uncertainty (the base-rate variance `p̄(1−p̄)`, which is about the data, not the model). scikit-learn (§1.16) says the same in one sentence: proper scoring rules "assess calibration (reliability) and discriminative power (resolution) of a model, as well as the randomness of the data (uncertainty) at the same time." Log-loss is the other proper scoring rule you already know — it is cross-entropy.

## 6.4 Which models are calibrated out of the box

Niculescu-Mizil & Caruana (ICML 2005) drew reliability diagrams for ten model families and the shapes are worth knowing because they predict what you will see:

- **Logistic regression** came out well calibrated and needed no post-hoc fix. Its objective — the likelihood of the observed labels under its own probabilities — pushes it that way, though heavy regularization or a badly misspecified model can still miscalibrate it.
- **Naive Bayes** pushes scores toward 0 and 1 (its independence assumption double-counts evidence). scikit-learn's own docs (§1.9): naive Bayes "is a decent classifier … but a bad estimator, so the probability outputs from `predict_proba` are not to be taken too seriously."
- **SVMs and boosted trees** push scores *away* from 0 and 1, toward the middle — sigmoid-shaped miscalibration.
- **Bagged trees and (older) neural nets** were reasonably calibrated.

And then the modern finding. Guo et al. (2017) compared a 1998 LeNet with a 2016 ResNet-110 on CIFAR-100. The LeNet had 44.9 % error and was nearly calibrated. The ResNet had 30.6 % error and an ECE of 16.5 % — badly over-confident. Their abstract: "modern neural networks, unlike those from a decade ago, are poorly calibrated," and "depth, width, weight decay, and Batch Normalization are important factors influencing calibration." Better accuracy, worse honesty. This is the norm for deep models and you should assume it until a reliability diagram says otherwise.

**For LLMs specifically**, the pattern is worse and directly relevant to any guardrail built on one. The GPT-4 technical report (§5): "the pre-trained model is highly calibrated … However, after the post-training process, the calibration is reduced." Its Figure 8 caption: "The post-training hurts calibration significantly." Kadavath et al. (Anthropic, 2022) found that large models are "well-calibrated on diverse multiple choice and true/false questions when they are provided in the right format" — the format caveat matters. Tian et al. (2023) found that for RLHF-tuned models, "verbalized confidences emitted as output tokens are typically better-calibrated than the model's conditional probabilities," often halving ECE. The takeaway for this course: **a chat model's token log-probabilities are not a trustworthy confidence score without recalibration on your own validation data.** You will measure this yourself in the agent-eval weeks.

## 6.5 Fixing it: Platt scaling and isotonic regression **[extension]**

The good news: miscalibration is usually a *monotone* distortion — the model's ordering is fine, its numbers are wrong. So you can learn a monotone map from raw score to probability on held-out data and leave the ranking untouched. Two standard maps.

**Platt scaling** (Platt 1999) fits a two-parameter sigmoid to the raw score `f(x)`:

```
P(y = 1 | f) = 1 / (1 + exp(A·f(x) + B))
```

with `A, B` fitted by maximum likelihood — that is, logistic regression with one input feature. Platt introduced it to turn SVM margins into probabilities "while still retaining the sparseness of the SVM." One detail from the original that scikit-learn preserves: to avoid over-fitting the sigmoid, fit against smoothed targets `y₊ = (N₊+1)/(N₊+2)` and `y₋ = 1/(N₋+2)` rather than hard 0/1. In scikit-learn this is `CalibratedClassifierCV(method='sigmoid')`, described as "Platt's method (i.e. a binary logistic regression model)."

**Temperature scaling** (Guo et al. §4.2) is Platt scaling stripped to a single scale parameter and no offset, applied to the logits: `softmax(z / T)`, `T` fitted by minimizing NLL on validation data. Because dividing all logits by the same `T` does not change the argmax, *accuracy is exactly unchanged*. It took the ResNet-110's ECE from 16.5 % to 1.3 %. Guo et al.: "a single-parameter variant of Platt Scaling — is surprisingly effective." Recent scikit-learn releases expose it as `CalibratedClassifierCV(method='temperature')`, noting its "natural way to obtain (better) calibrated multi-class probabilities with just one free parameter."

**Isotonic regression** (Zadrozny & Elkan 2002) drops the parametric form: "a non-parametric form of regression in which we assume that the function is chosen from the class of all isotonic (i.e. non-decreasing) functions." The pair-adjacent-violators algorithm "finds the stepwise-constant isotonic function that best fits the data according to a mean-squared error criterion," in linear time. The result is a staircase that can correct *any* monotone distortion, not just sigmoid-shaped ones. In scikit-learn: `CalibratedClassifierCV(method='isotonic')`.

**Which to use.** The guidance is consistent across twenty years of sources:

- Niculescu-Mizil & Caruana §5: "When the calibration set is small (less than about 200–1000 cases), Platt Scaling outperforms Isotonic Regression … when there are 1000 or more points in the calibration set, Isotonic Regression always yields performance as good as, or better than, Platt Scaling."
- scikit-learn §1.16.3: sigmoid "is most effective for small sample sizes or when the un-calibrated model is under-confident and has similar calibration errors for both high and low outputs"; its symmetric-error assumption "can be a problem for highly imbalanced classification problems." Isotonic "can correct any monotonic distortion … However, it is more prone to overfitting, especially on small datasets." Rule of thumb: isotonic when you have "greater than ~1000 samples."
- Zadrozny & Elkan §2–3: "The sigmoidal shape does not appear to fit naive Bayes scores as well as it fits SVM scores" — if the distortion is not sigmoid-shaped, Platt is the wrong model of it.

One side-effect: isotonic "introduces ties in the predicted probabilities," so it can nudge AUC slightly; sigmoid and temperature are strictly monotone and leave AUC exactly alone.

**The rule that must not be broken: calibrate on held-out data.** scikit-learn §1.16.2: "Using the classifier output of training data to fit the calibrator would thus result in a biased calibrator that maps to probabilities closer to 0 and 1 than it should." The model is over-confident on its own training data by construction; a calibrator fitted there learns to trust that over-confidence. `CalibratedClassifierCV` handles this with an internal 5-fold split (`ensemble=True` averages five model+calibrator pairs; `ensemble=False` fits one calibrator on out-of-fold predictions); an already-trained model is calibrated by wrapping it in `FrozenEstimator` and passing fresh data. Same principle as the threshold in part 5, same principle as hyperparameters in part 1: anything fitted after the model is fitted on data the model did not see.

## 6.6 Calibration does not survive a change in prevalence

Last warning, and it connects to part 7. Calibration is a statement about `P(Y | p̂)`, and that depends on the base rate `P(Y)`. If the model was calibrated when 1 % of traffic was attacks and attacks rise to 10 %, the same score 0.9 now corresponds to a *higher* true positive fraction — the model is under-confident — and every threshold you chose from costs is now wrong. Moreo (2025), summarizing Ovadia et al.: "a classifier is unlikely to remain well-calibrated across different distributions." This is **label shift** (Lipton et al. 2018: "the label marginal p(y) changes but the conditional p(x|y) does not"), and under pure label shift the *ranking* is unaffected — ROC unchanged — while the *probabilities* are wrong. Saerens et al. (2002) give the fix once you know the new prior `q(y)`: reweight the posterior by `q(y)/p(y)` and renormalize; they also show the new prior can be estimated from *unlabeled* production data by EM. Lipton et al.'s BBSE does it from the confusion matrix and "works even when predictors are biased, inaccurate, or uncalibrated." You do not need to implement these this week. You need to know that a reliability diagram is a snapshot, and to monitor it.

## 6.7 What to carry forward

A score of 0.8 is a claim that eight in ten such items are positive. Check the claim with a reliability diagram; summarize it with ECE and the Brier score, and remember the Brier score also measures resolution, which calibration alone does not. Logistic regression is calibrated; modern deep nets and post-trained LLMs are over-confident. Fix with Platt/temperature scaling below ~1,000 calibration points and isotonic above, always on held-out data. Calibration is a snapshot at one base rate.

---

**Sources for this part** (exact sections in `READING.md`): Guo, Pleiss, Sun & Weinberger, ICML 2017, arXiv 1706.04599, §2, §3, §4, §5 · Niculescu-Mizil & Caruana, ICML 2005, §2, §4, §5 · scikit-learn User Guide §1.16 and `CalibratedClassifierCV`, `brier_score_loss` references · DeGroot & Fienberg 1983 (JRSS-D abstract; DTIC technical report §2, §4) · Platt 1999 · Zadrozny & Elkan, KDD 2002, §2–3 · Brier score decomposition (Wikipedia, "Decomposition") · GPT-4 Technical Report §5, Fig. 8 · Kadavath et al. 2022, arXiv 2207.05221 · Tian et al. 2023, arXiv 2305.14975 · Saerens, Latinne & Decaestecker 2002 · Lipton, Wang & Smola 2018, arXiv 1802.03916 · Moreo 2025, arXiv 2505.11380, Def. 3.2, §5.2.1.
