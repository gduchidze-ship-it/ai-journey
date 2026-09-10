# 5 · The threshold is a product decision — and why a guardrail at 0.9 fires far more than "10 % error" suggests

*Lecture 1, part 5 of 8 · ~15 min reading*

## 5.1 The default threshold is nobody's decision

Every classification library ships a default. scikit-learn's user guide (§3.3) is candid about what it is and what it is worth: "a positive class is predicted when the conditional probability P(y|X) is greater than 0.5 … or if the decision score is greater than 0," and "these hard-coded rules … are most certainly not ideal for most use cases." Elkan (2001) explains where 0.5 comes from: "standard learning algorithms are designed to yield classifiers that maximize accuracy. In the two-class case, these classifiers implicitly make decisions based on the probability threshold 0.5." Part 4 established that accuracy is the wrong objective on imbalanced data. Its threshold is therefore the wrong threshold, and it was chosen by the library author, not by anyone who knows what your errors cost.

## 5.2 Costing the two errors

Write down what each error costs. Elkan's notation: `C(i, j)` is the cost of *predicting* class `i` when the *true* class is `j`. Two sanity conditions must hold — `C(1,0) > C(0,0)` and `C(0,1) > C(1,1)` — "the cost of labeling an example incorrectly should always be greater than the cost of labeling it correctly." With correct decisions costing nothing, you have two numbers: **C_FP = C(1,0)**, the cost of a false positive, and **C_FN = C(0,1)**, the cost of a false negative.

The expected cost of a threshold `t` on a dataset is then simply

```
Cost(t) = C_FP · FP(t) + C_FN · FN(t)
```

and the right threshold is the one that minimizes it. That is a one-dimensional search over the threshold sweep your `metrics.py` will produce. There is no theory needed to *use* this; the theory tells you what the answer will be.

**The Bayes-optimal threshold.** If the model's score `p` is a calibrated probability (part 6 defines that), Elkan's eq. 2 gives the optimal rule in closed form: predict positive iff `p ≥ p*`, where

```
p* = (C(1,0) − C(0,0)) / ( (C(1,0) − C(0,0)) + (C(0,1) − C(1,1)) )
```

which with zero cost for correct decisions reduces to

```
t* = C_FP / (C_FP + C_FN)
```

Read it: if a false negative costs 9× a false positive, `t* = 1/(1+9) = 0.1` — flag anything the model thinks is more than 10 % likely to be positive. If a false positive costs 9× a false negative, `t* = 0.9`. The threshold *is* the cost ratio. Elkan also notes the invariances: the optimal decision is "unchanged if each entry in the matrix is multiplied by a positive constant" or if "a constant is added to each entry" — you only need the *ratio* of the two costs, which is often much easier to get from a business owner than the absolute numbers.

**The recommended procedure**, in Elkan's words: "learn a classifier from the training set as given, and then compute optimal decisions explicitly using the probability estimates given by the classifier." Do not rebalance the training data to move the threshold (part 7 explains what that breaks). Fit, calibrate, then choose the threshold from costs.

**What this looks like in a library.** scikit-learn's cost-sensitive example uses a German credit dataset with gain matrix `[[0, −1], [−5, 0]]`: "misclassifying a 'bad' credit as 'good' is five times more costly … than misclassifying a 'good' credit as 'bad'." The guide notes that, "given that our model is calibrated and our data set representative and large enough, we do not need to tune the threshold" — Elkan's eq. 2 gives it directly; for a 1 : 5 cost ratio that is `1/(1+5) ≈ 0.17`, not 0.5. (The page's own phrasing, "1/5 of the cost ratio", is loose; do the arithmetic yourself.) Tuning empirically with `TunedThresholdClassifierCV` found `best_threshold_ = 0.03` and moved the business metric from −209 to −143, a roughly 30 % reduction in loss from a threshold change alone, with no change to the model. The same example's second half is the one that matters for fraud: when "the business metric depends on the amount of each individual transaction," the cost matrix is per-row, and you optimize expected cost with those per-row weights.

One rule from the same guide: "you should never use the same data for training the classifier and tuning the decision threshold." The threshold is a hyperparameter; it is tuned on validation data, like every other hyperparameter in part 1.

## 5.3 Who owns the numbers

`C_FP` and `C_FN` are not in the dataset. They are in the business. A declined legitimate card payment costs a margin, a customer, and a support ticket; Stripe's fraud primer describes false declines as causing "both a gross profit and reputational hit." A missed fraud costs the transaction amount plus the chargeback fee. A blocked benign chat message costs a user's trust and an appeal; a harmful message that gets through costs whatever it costs — which is the hard case, and one you should still write down as a range rather than leave blank.

If you cannot write both costs in currency, you cannot choose the threshold; someone chose it for you, and it was the person who wrote the library default. The production drill this week is exactly this exercise: find one threshold in a system you have shipped and price its two errors. If you cannot, the deliverable is the name of the person who can.

## 5.4 Why a guardrail at threshold 0.9 fires far more often than "10 % error" suggests

Here is the trap that catches engineers who have understood everything above.

A classifier is calibrated (part 6). You set its threshold to 0.9. You reason: "we only act when the model is 90 % sure, so at most 10 % of our actions are mistakes." That reasoning is correct *for the flags*. It says nothing about *how many flags there are* — and the number of flags is dominated by the number of negatives, not the quality of the model.

**The mechanism is Bayes' rule.** Let `π` be the prevalence of positives, `TPR` the recall at your threshold, and `FPR` the false positive rate at your threshold. Then

```
precision = P(positive | flagged) = TPR · π / ( TPR · π + FPR · (1 − π) )
```

When `π` is small, the term `FPR · (1 − π)` is a small rate multiplied by a huge population, and it swamps `TPR · π`, which is a large rate multiplied by a tiny population. Your precision collapses even with an excellent classifier.

**The canonical numbers.** Gigerenzer et al. (2007) on mammography screening — prevalence 1 %, sensitivity 90 %, false positive rate 9 %: "Ten out of every 1,000 women have breast cancer. Of these 10 women with breast cancer, 9 test positive. Of the 990 women without cancer, about 89 nevertheless test positive." So among ~98 positives, 9 are real: precision ≈ 9 %. "Only about 1 out of every 10 women who test positive in screening actually has breast cancer." When Gigerenzer asked gynaecologists, the majority answered 90 % or 81 %; only 21 % got it right. These are professionals who do this for a living. Your product manager will make the same mistake, and so will you if you do not compute it.

The base-rate article's other examples land the same way. Breathalyzer: 1 in 1,000 drivers drunk, 5 % false positive rate, 100 % sensitivity — "the correct probability [that a flagged driver is drunk] is about 2 %," because 1 true positive stands against ≈ 50 false ones. Screening a population of 1,000,000 for 100 targets with 1 % FPR and 1 % FNR: "about 99 of the 100 [targets] will trigger the alarm — and so will about 9,999 of the 999,900 non-[targets]"; the probability a flagged person is a target "is only about 99 in 10,098, which is less than 1 %."

**Apply it to a guardrail.** Suppose a prompt-injection detector sees 1,000,000 requests a day, of which 0.1 % (1,000) are attacks. At threshold 0.9 it catches 80 % of attacks (800) and has a false positive rate of 0.5 % — which sounds excellent. That is 0.005 × 999,000 ≈ **4,995 false alarms** against **800 true ones**. Precision ≈ 14 %. Six out of seven users you block are innocent. The classifier did not lie; the phrase "90 % sure" was about a different quantity than the one you cared about. The score threshold controls precision *among items with that score*; the base rate controls how many such items exist on each side.

Two consequences:

1. **Ask for the FPR at your threshold, and multiply it by your traffic.** That product — false alarms per day — is the number that decides whether your review queue drowns. A "0.5 % FPR" is 5,000 tickets at a million requests.
2. **Calibration does not save you.** A perfectly calibrated 0.9 means 10 % of the *0.9-scored* items are wrong. It does not mean 10 % of your *blocks* are wrong unless every block scored exactly 0.9, and it does not mean anything at all about whether the prevalence in production matches the prevalence the model was calibrated on (part 6, §6.6).

**Real guardrails report it this way.** Meta's Llama Guard paper evaluates with the area under the precision–recall curve, "following [OpenAI's moderation paper]. AUPRC focuses on the trade-off between precision and recall" — that is, on the quantity that base rates attack. Its reported numbers on the OpenAI moderation set are 0.847 (Llama Guard), 0.856 (OpenAI API), 0.787 (Perspective API); on ToxicChat 0.626, 0.588, 0.532. OpenAI's own moderation docs say the vendor's `flagged` boolean is *not* your policy: "Treat moderation scores as signals for your application's policy, not as an automatic blocking decision," and warn that "custom policies that rely on `category_scores` may need recalibration over time" as the model changes. The threshold is yours. So is the base rate.

## 5.5 What to carry forward

The default threshold encodes the library author's cost assumptions, not yours. Write down `C_FP` and `C_FN` in currency; the optimal threshold on calibrated scores is `C_FP / (C_FP + C_FN)`, and on uncalibrated scores it is whatever minimizes `C_FP·FP(t) + C_FN·FN(t)` on the validation sweep. Then multiply the false positive *rate* at that threshold by the number of negatives you will actually see, because at a rare positive class that product is the cost you will pay every day, and a "90 % confident" threshold does nothing to shrink it.

---

**Sources for this part** (exact sections in `READING.md`): Elkan, "The Foundations of Cost-Sensitive Learning," IJCAI 2001, "Making optimal decisions," eq. 2 · scikit-learn User Guide §3.3 and the example "Post-tuning the decision threshold for cost-sensitive learning" · Gigerenzer et al., *Psychological Science in the Public Interest* 2007, "Few Gynecologists Understand Positive Mammograms" · "Base rate fallacy," Wikipedia (breathalyzer and screening examples) · Inan et al., "Llama Guard," arXiv 2312.06674, §4 Table 2 · OpenAI Moderation guide · Stripe, "A primer on machine learning for fraud detection," "Evaluating machine learning models."
