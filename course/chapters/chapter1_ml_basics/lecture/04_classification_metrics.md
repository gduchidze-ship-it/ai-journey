# 4 · Judging a classifier: confusion matrix, precision, recall, F1, ROC and PR

*Lecture 1, part 4 of 8 · ~25 min reading*

Everything in this part is built from four integers. Get the four integers right and every metric is arithmetic. Get them confused and every metric is wrong in a way that looks fine.

## 4.1 The confusion matrix

A binary classifier emits a score; a **threshold** turns the score into a yes/no. Compare that yes/no with the truth over a dataset and you get four counts. Scikit-learn's definition (§3.4.4.6): "entry *i, j* in a confusion matrix is the number of observations actually in group *i*, but predicted to be in group *j*."

```
                        predicted negative     predicted positive
actual negative          TN  (true negative)    FP  (false positive)
actual positive          FN  (false negative)   TP  (true positive)
```

Conventions differ on which axis is which, so always label yours. The names are read as *[correctness] [what was predicted]*: a **false positive** is a positive prediction that was false — the model said "fraud" and it wasn't. A **false negative** is a negative prediction that was false — the model said "fine" and it was fraud. Which of these hurts more is a business question, and part 5 is about pricing it. In fraud, FPs are declined legitimate customers; FNs are money gone. In a content guardrail, FPs are blocked legitimate users; FNs are harm that got through.

## 4.2 The summaries you will actually use

All from the four counts (scikit-learn §3.4.4.9.1, Google Crash Course "Accuracy, recall, precision"):

| Name | Formula | Reads as |
|---|---|---|
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) | Of everything, how much did I get right? |
| **Precision** | TP / (TP + FP) | Of what I flagged, how much was real? |
| **Recall** (sensitivity, true positive rate) | TP / (TP + FN) | Of what was real, how much did I flag? |
| **False positive rate** | FP / (FP + TN) | Of what was benign, how much did I flag anyway? |
| **F1** | 2·P·R / (P + R) = 2TP / (2TP + FP + FN) | Harmonic mean of precision and recall |

Worked example from the Crash Course: 5 TP, 6 TN, 3 FP, 2 FN → recall = 5/7 ≈ 0.714, precision = 5/8 = 0.625, accuracy = 11/16 ≈ 0.69, F1 ≈ 0.667.

**F-beta.** F1 is the special case β = 1 of

```
F_β = (1 + β²) · P · R / (β² · P + R)
```

scikit-learn describes it as the harmonic mean "with precision's contribution to the mean weighted by some parameter β." Van Rijsbergen's original phrasing (via the F-score article): β is "chosen such that recall is considered β times as important as precision." Same example with y_true = [0,1,0,1], y_pred = [0,1,0,0]: precision 1.0, recall 0.5, F1 0.667, F0.5 = 0.83 (favours precision), F2 = 0.55 (favours recall).

**Why the harmonic mean.** The arithmetic mean of P = 1.0 and R = 0.01 is 0.505 — flattering for a model that finds 1 % of the positives. The harmonic mean is 0.0198. The harmonic mean is dragged toward the smaller of the two, which is the point: F1 punishes you for being good at one and useless at the other.

## 4.3 Why accuracy lies on imbalanced data

Google's one sentence: for heavily imbalanced datasets, "a model that predicts negative 100% of the time would score 99% on accuracy, despite being useless." Its definition of the problem: "When the total of actual positives is not close to the total of actual negatives, the dataset is imbalanced."

This is called the **accuracy paradox** and the worked example is standard: a city of 1,000,000 people with 10 wanted individuals; a detector flags 1,000 people and catches all 10. Accuracy ≈ 99.9 %. Precision = 10/1000 = 1 %. Recall = 100 %. F1 ≈ 2 %. "990 out of the 1000 positive predictions are incorrect." Any real-world positive class — fraud, disease, abuse, an outage — is rare, so the **majority-class baseline** is the first number you compute: always predict the common class, and note the accuracy. Your model has to beat *that*, not zero.

Scikit-learn offers `balanced_accuracy_score`, "the macro-average of recall scores per class or, equivalently, raw accuracy where each sample is weighted according to the inverse prevalence of its true class." It "avoids inflated performance estimates on imbalanced datasets." Useful; but it is still one number and still hides the FP/FN split you will need in part 5.

## 4.4 When each summary is the wrong one

**Precision is wrong** when missing a positive is the expensive error. A cancer screen with 100 % precision and 30 % recall kills people politely.

**Recall is wrong** when the cost of a false alarm dominates. Recall 100 % is trivially achievable — flag everything — so it is meaningless without precision or FPR beside it. Google's table puts it as: use recall "when false negatives are more expensive than false positives," use FPR "when false positives are more expensive than false negatives."

**F1 is wrong more often than people think.** Three specific reasons:

1. **F1 ignores true negatives.** Look at the F-beta formula `(1+β²)·TP / ((1+β²)·TP + FP + β²·FN)`: TN appears nowhere. Two classifiers with identical TP, FP, FN but wildly different TN have identical F1. If your negatives matter — and for a guardrail, letting benign users through *is* the product — F1 cannot see that you are doing it well.
2. **F1 weights precision and recall equally**, which is a cost assumption almost never true. If a false negative costs 50× a false positive, F1 is the wrong summary and F-beta with β chosen from the cost ratio, or the expected cost directly (part 5), is right.
3. **F1 is a point metric at one threshold.** Change the threshold and F1 changes. A single F1 tells you about one operating point, not about the model. Two models with the same F1 at 0.5 can be very different at 0.3.

Chicco & Jurman argue that the Matthews correlation coefficient (which uses all four cells) is "more truthful and informative" than F1. You do not need MCC this week; you need to know that F1's blindness to TN is a real defect, not a nitpick.

**Accuracy is wrong** whenever the classes are imbalanced (§4.3), which is nearly always. Google grants it one honest use: "rough indicator of model training progress."

**Multiclass note.** scikit-learn's `average=` argument exists because F1 for many classes is ambiguous: `macro` = equal weight per class; `weighted` = by class support; `micro` = every sample-class pair equal, which in single-label multiclass collapses to accuracy. Pick deliberately.

## 4.5 Curves instead of points: ROC

Every metric so far depends on the threshold. To see the model rather than one operating point, sweep the threshold from 1 down to 0 and plot what happens.

The **ROC curve** (receiver operating characteristic) plots true positive rate (recall) against false positive rate, one point per threshold. Random guessing gives the diagonal. The **AUC** — area under the ROC curve — has a clean interpretation the Crash Course states exactly: it "represents the probability that the model, if given a randomly chosen positive and negative example, will rank the positive higher than the negative." 1.0 is perfect ranking; 0.5 is coin-flipping; consistently below 0.5 usually means the labels or the score direction are inverted somewhere in your pipeline.

AUC is a **ranking** metric. It cares only about the order of scores, not their values, not the threshold. That is its virtue when you have not chosen a threshold yet, and its blindness once you have.

## 4.6 Curves instead of points: precision–recall

The **PR curve** plots precision against recall, one point per threshold. Its area is **average precision**; scikit-learn computes it as `AP = Σₙ (Rₙ − Rₙ₋₁) · Pₙ` — a step function, deliberately *not* linearly interpolated, because (citing Davis & Goadrich) "linear interpolation of points on the precision-recall curve provides an overly-optimistic measure." Davis & Goadrich show the correct interpolation between two PR points must follow the local ratio of new false positives to new true positives; a straight line overstates achievable precision. Your `metrics.py` should therefore compute AP as the step-sum, and if you draw a curve, draw steps.

## 4.7 ROC versus PR under class imbalance

Now the question that decides which curve you show your team.

**ROC does not see the class balance. PR does.** Saito & Rehmsmeier (PLOS ONE 2015) ran the same classifiers on a balanced set (1,000 positives, 1,000 negatives) and an imbalanced one (1,000 positives, 10,000 negatives). Result: "The ROC plots are unchanged between balanced and imbalanced datasets" and "all AUC (ROC) scores are unchanged accordingly. In contrast … the PRC plots are changed." The mechanism: the ROC x-axis is FPR = FP/(FP+TN), a *rate over the negatives*. Multiply the negatives by ten and you multiply both FP and TN by ten; the rate is unchanged. Precision = TP/(TP+FP) has raw FP in the denominator; ten times as many negatives means ten times as many false positives against the same true positives. Saito & Rehmsmeier's own numbers: the same ROC point corresponds to 160 FP against 500 TP on the balanced set (precision 0.76) and 1,600 FP against 500 TP on the imbalanced one (precision 0.24). *Identical ROC coordinates, wildly different precision.*

Davis & Goadrich (ICML 2006) said the same thing a decade earlier: with many negatives, "a large change in the number of false positives can lead to a small change in the false positive rate," whereas precision "captures the effect of the large number of negative examples on the algorithm's performance." Their example: 20 positives, 2,000 negatives; one algorithm has AUC-ROC 0.875 but AUC-PR 0.038; a competitor has AUC-ROC 0.813 but AUC-PR 0.514. **ROC and PR can rank two models in opposite orders.** On imbalanced data the PR ranking is the one that corresponds to the experience of whoever reviews the flags.

**The baselines differ.** A random classifier's ROC is the diagonal regardless of data. Its PR curve is a horizontal line at the positive prevalence: Saito & Rehmsmeier, "the baseline of PRC is determined by the ratio of positives (P) and negatives (N) as y = P/(P+N)" — 0.5 for balanced data, 0.09 at 1:10. scikit-learn: "With random predictions, the AP is the fraction of positive samples." This is why an AP of 0.30 can be excellent (at 1 % prevalence it is 30× random) and an AUC of 0.90 can be useless (see the Davis & Goadrich example). Always report the prevalence next to the AP.

**The formal relationship.** Davis & Goadrich Theorem 3.1: for a fixed dataset "there exists a one-to-one correspondence between a curve in ROC space and a curve in PR space." Theorem 3.2: "one curve dominates a second curve in ROC space if and only if the first dominates the second in Precision-Recall space." So the two carry the same information *for a fixed class balance*; they differ in what they make visible, and in what happens when the class balance is not fixed. ROC is the right view when you want a summary that survives a change in prevalence, or when the classes are balanced. PR is the right view when positives are rare and the cost of reviewing a flag is real. Google's Crash Course agrees: for imbalanced data "precision-recall curves … may offer a better comparative visualization."

## 4.8 What to carry forward

Four integers. Precision answers "of my flags, how many were right"; recall answers "of the real ones, how many did I catch"; both depend on the threshold. Accuracy is a lie at 1 % prevalence. F1 cannot see true negatives and assumes equal costs. AUC is prevalence-blind — a virtue and a blindness. PR curves track what the people looking at your flags actually experience, and their floor is the prevalence. Report prevalence next to every metric.

---

**Sources for this part** (exact sections in `READING.md`): scikit-learn User Guide §3.4.4.4, §3.4.4.6, §3.4.4.9, ROC and PR subsections · Google ML Crash Course, "Thresholds and the confusion matrix," "Accuracy, recall, precision, and related metrics," "ROC and AUC" · Davis & Goadrich, ICML 2006, §2–4 · Saito & Rehmsmeier, PLOS ONE 2015, Results → Simulation · "Accuracy paradox" and "F-score" (Wikipedia, as fallbacks for the worked examples and the Chicco & Jurman critique).
