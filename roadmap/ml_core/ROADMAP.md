# Classical ML — Roadmap
**Part of:** [Master Syllabus](../SYLLABUS.md) — the order all five roadmaps run in.

**Time:** ~10 h (AAI) · ~8 h (INF). **+10 h** if you target Google, Meta or Amazon — see Lane B (§10–§11). **Prereq:** none. **Supersedes:** `README.md` in this directory.
**Feeds:** [llm_internals](../llm_internals/ROADMAP.md) **Part B §11 (Evals)** — the metrics and statistics there come from here.

> **Verdict on the original list: too big. Cut ~60%.** Decision trees, random forests, gradient boosting, naive Bayes and SVMs do not come up in Applied AI or AI Infra interviews. That was ~25% of the old plan buying ~5% of the value. What stays is the part that later becomes **eval design**: metrics, thresholds, leakage, significance. Two needed topics were missing: retrieval metrics (§4) and system comparison statistics (§6). Sections 1–9 are the whole topic for a frontier lab. If it feels small, that is correct.
>
> **Added later:** §10–§11 (Lane B). Google, Meta and Amazon put Applied AI engineers on feed, search and ads relevance, so their ML round tests recommenders, ranking and A/B testing — none of which was here. Gradient boosting is partly un-cut for the same reason. Skip Lane B if those three companies are not on your list.

**How to read the tables.** AAI = Applied AI Engineer. INF = AI Infrastructure Engineer. **Core** = learn it well. **Useful** = know it. **Skim** = one paragraph. **Skip** = ignore it.

## Path — do this in order

1. **Generalization vocabulary** — 1.5 h · 3 blocks · no code
2. **Classification metrics and thresholds** — 1.5 h · 3 blocks · ~30 lines
3. **Data leakage** — 1 h · 2 blocks · ~30 lines
4. **Ranking and retrieval metrics** — 2 h · 4 blocks · ~40 lines
5. **Embeddings and similarity** — 10 min · pointer only
6. **Comparing two systems** — 1.5 h · 3 blocks · ~30 lines
7. **Calibration** — 45 min · 2 blocks · ~15 lines
8. **Data quality basics** — 1 h · 2 blocks · ~20 lines
9. **Awareness pass** — 30 min · 1 block · notes only

**Lane B — Google / Meta / Amazon only, +10 h:**

10. **Recommender and ranking systems** — 7 h · 17 blocks · ~90 lines
11. **A/B testing and online metrics** — 3 h · 7 blocks · ~30 lines

One block = 25 minutes. Finish a section's **Build** before starting the next section. **INF path:** §7 is Skim and the nDCG part of §4 is optional — that is where the 6 h comes from.

## 1. Generalization vocabulary

| Sub-topic | AAI | INF |
|---|---|---|
| Bias/variance, capacity, over/underfitting | Useful | Useful |
| L1/L2 regularization, early stopping | Useful | Useful |
| Weight decay vs L2-in-the-loss under Adam | Useful | Core |

**Plain words.** *Overfitting* = the model learns the training data by heart and fails on new data. *Regularization* = a penalty that keeps the model simple. *Weight decay* = shrink every weight a little on each training step.

- **Start here:** StatQuest with Josh Starmer ([statquest.org](https://www.statquest.org/)) — 3 videos, in order: *"Machine Learning Fundamentals: Bias and Variance"*, *"Regularization Part 1: Ridge (L2) Regression"*, *"Regularization Part 2: Lasso (L1) Regression"*. Stop there. Skip Part 3 (Elastic Net).
- **Reference:** Loshchilov & Hutter, *Decoupled Weight Decay Regularization* — [arXiv:1711.05101](https://arxiv.org/abs/1711.05101). Read §2 and §3 only. This is the AdamW point, and it is the half of this section that gets asked.

- **Build:** no code. Write 3 sentences in your notes: what overfitting is, what regularization does, what AdamW changed. ~10 min.
- **Ready to move on when:** you close your notes and say out loud, in two sentences, why AdamW's weight decay is not the same as adding an L2 term to the loss, and what breaks if you use L2 instead.

## 2. Classification metrics and thresholds

| Sub-topic | AAI | INF |
|---|---|---|
| Confusion matrix, precision, recall, F1 | Core | Useful |
| ROC/AUC vs precision–recall curves under class imbalance | Core | Useful |
| Threshold selection as a product decision | Core | Skim |

**Plain words.** *Precision* = of the cases you flagged, how many were right. *Recall* = of the cases you should have flagged, how many you caught. *Threshold* = the score above which you call it positive.

- **Start here:** StatQuest — *"Machine Learning Fundamentals: The Confusion Matrix"*, *"Machine Learning Fundamentals: Sensitivity and Specificity"*, *"ROC and AUC, Clearly Explained!"*.
- **Optional:** Google ML Crash Course — [Classification: Accuracy, precision, recall](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall). Read the threshold part only.

- **Build:** ~30 lines, ~45 min. Take any small labeled dataset where one class is rare. Train anything simple. Plot an ROC curve and a PR curve for that same model. Pick one threshold and write two sentences defending it: what a false positive costs, what a false negative costs.
- **Ready to move on when:** you answer "when is a PR curve better than an ROC curve, and why?" with no hedging. Then, for a content-moderation classifier, name the cost of a false positive and of a false negative *before* you pick the threshold.

## 3. Data leakage

The reason a model scores 0.94 offline and fails in production. Interviewers hunt for this, and it is the same skill as spotting a contaminated eval set later.

| Sub-topic | AAI | INF |
|---|---|---|
| Train/test contamination — the same row on both sides of the split | Core | Useful |
| Temporal leakage — using the future to predict the past | Core | Useful |
| Group leakage — the same user or document split across train and test | Useful | Skim |
| Benchmark contamination in LLMs — the test set was in the pretraining data | Core | Useful |

**Plain words.** *Leakage* = the model sees something during training that will not exist at prediction time. The score looks great and the live system does not work. *Benchmark contamination* = the same thing for LLMs: the model already read the exam.

- **Start here:** scikit-learn — [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html). Read the **Data leakage** section. Short, with code showing the wrong way and the right way.
- scikit-learn — [Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html), the **`TimeSeriesSplit`** part. This is how you split data that has a time order.
- Kapoor & Narayanan, *Leakage and the Reproducibility Crisis in ML-based Science* — [arXiv:2207.07048](https://arxiv.org/abs/2207.07048). Read the leakage **taxonomy table** only. Eight named types; naming them is what makes you sound senior.

- **Build** (~30 lines, 1 h): take any dataset that has a date or timestamp column. Train the same model twice — once with a **random split**, once with a **time-based split** (train on the past, test on the future). Report both scores side by side.
- **Ready to move on when:** you name three kinds of leakage with one example each; you explain why a random split flatters the score on time-ordered data; and you say what benchmark contamination means for trusting a published LLM eval number.

## 4. Ranking and retrieval metrics — **missing from the original list**

| Sub-topic | AAI | INF |
|---|---|---|
| recall@k, precision@k | Core | Useful |
| MRR | Core | Useful |
| nDCG and what it penalizes that recall does not | Core | Skim |
| Choosing k for a retrieval stage feeding a generator | Core | Useful |

**Plain words.** These score a *ranked list*, not a yes/no answer. *recall@k* = did the right document land in the top k. *MRR* = how high the first right answer sits. *nDCG* = same idea, but rank 1 counts more than rank 10.

- **Start here:** Manning, Raghavan & Schütze, *Introduction to Information Retrieval* — [Ch. 8, "Evaluation in information retrieval"](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-in-information-retrieval-1.html). Read §8.3 and §8.4 only. Free HTML.

- **Build 3a:** ~20 lines, ~30 min. Write `recall@k` and `MRR` from scratch, no libraries.
- **Build 3b:** ~20 lines, ~30 min. Write `nDCG@k` from scratch. First make a 5-document example and compute every score **on paper**, then unit-test your code against those numbers.
- **Ready to move on when:** your paper numbers and your code numbers match. Then say in one sentence what nDCG@10 measures that recall@10 does not, and why a retriever with better recall@50 can still make a worse RAG system.

## 5. Embeddings and similarity — pointer only

Do **not** study this here. It is covered where you will actually use it: [llm_internals](../llm_internals/ROADMAP.md) **§10 (RAG)** — cosine vs dot product vs L2, and the normalization traps.

It is listed at this position because it belongs next to §4: retrieval metrics measure a ranking, and similarity is what produces that ranking. When you reach §10 there, do the 20-minute exercise: write all three similarity functions by hand, run them on the same vectors, and find the case where cosine and dot product disagree about the order.

- **Ready to move on when:** you can say in one sentence why cosine and dot product give the same ranking once vectors are normalized, and different rankings when they are not.

## 6. Comparing two systems — **missing from the original list**

| Sub-topic | AAI | INF |
|---|---|---|
| Bootstrap confidence intervals | Core | Useful |
| Paired tests on the same eval set | Core | Useful |
| Sample size: how many cases to detect a given delta | Core | Useful |
| Multiple comparisons when you test 12 prompt variants | Useful | Skim |

**Plain words.** Two systems score differently on your eval set. This section tells you if the difference is real or just noise. *Bootstrap* = resample your own results many times and watch how much the score wobbles.

- **Start here:** StatQuest — *"Bootstrapping Main Ideas!!!"* and *"p-values: What they are and how to interpret them"*.
- **Reference:** SciPy docs — [`scipy.stats.bootstrap`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html). Read the examples. Use `paired=True` for eval-set comparisons.

- **Build:** `compare.py`, ~30 lines, ~45 min. Input: two lists of per-case scores on the *same* eval set. Output: the mean difference plus a bootstrap 95% confidence interval. *Optional stretch:* also print how many cases you need to detect a 3-point delta at 80% power.
- **Ready to move on when:** someone says *"the new retriever scored 0.71 vs 0.68 on 200 queries."* In under a minute you answer three things: is it real, what test you would run, how many more queries you need.

## 7. Calibration

| Sub-topic | AAI | INF |
|---|---|---|
| What a confidence score does and does not mean | Useful | Skim |
| Reliability diagrams, Platt scaling, isotonic regression | Useful | Skip |

**Plain words.** A model that says 0.9 should be right about 90% of the time. Often it is not. *Calibration* is fixing that gap.

- **Start here:** scikit-learn User Guide — [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html). Read §1.16.1 and the reliability-diagram figures. Stop there.

- **Build:** ~15 lines, ~20 min. Reuse the model from §2. Put its predicted probabilities into 10 buckets. Plot predicted probability against the real hit rate per bucket. That plot is a reliability diagram.
- **Ready to move on when:** looking at your own plot, you can explain why a guardrail classifier at threshold 0.9 may fire far more often than "10% error" suggests.

## 8. Data quality basics — awareness level only

One sitting. You need to recognize these three problems and name the standard fix. Nothing deeper.

| Sub-topic | AAI | INF |
|---|---|---|
| Missing values: drop, impute, or treat missingness as a signal | Useful | Skim |
| Class imbalance: resampling, class weights, and why accuracy lies | Useful | Skim |
| Spotting distribution shift between training data and production data | Useful | Useful |

**Plain words.** *Imputation* = filling a missing value with a guess (the column mean, for example). *Distribution shift* = production data no longer looks like training data, so the model quietly gets worse.

- **Start here:** scikit-learn — [Imputation of missing values](https://scikit-learn.org/stable/modules/impute.html). Read the intro and `SimpleImputer`. Stop there.
- Evidently AI — [Data drift](https://www.evidentlyai.com/ml-in-production/data-drift). How shift is actually detected in production, in plain language.

- **Build** (~20 lines, 30 min): on one dataset print three things — missing count per column, the class balance, and the mean of each feature in the first half versus the second half of the data by date. Three prints, that is all.
- **Ready to move on when:** given a new dataset you say within two minutes whether it has a missingness problem, an imbalance problem, or a shift problem, and name one fix for each.

## 9. Awareness pass — one paragraph each, then stop

| Sub-topic | AAI | INF |
|---|---|---|
| Logistic regression | Skim | Skim |
| Decision trees, random forests | Skim | Skip |
| Gradient boosting (GBDT: XGBoost / LightGBM) | **Useful\*** | Skip |
| Naive Bayes, SVM, k-means, PCA | Skim | Skip |

- **Start here:** Géron, *Hands-On Machine Learning*, 3rd ed. — **Ch. 1 "The Machine Learning Landscape"**. 45 minutes, reading only. Do not watch the StatQuest tree/boosting/Bayes videos.

- **Build:** one sentence per algorithm in your notes. ~10 min. Then stop.
\* **Only if you target Google, Meta or Amazon.** GBDT still runs production ranking there, so it comes up. At Mistral, Anthropic or Lovable it does not — keep it at Skim. If it applies to you, §10.2 below covers it properly.

- **Ready to move on when:** you can say what gradient boosting is in one sentence, and name one place it still beats a neural network.

---

# Lane B — only if you target Google, Meta or Amazon

**Skip §10 and §11 entirely if you target Mistral, Anthropic, OpenAI or Lovable.** Their loops do not test this.

Google, Meta and Amazon hire Applied AI engineers onto **feed, search and ads relevance** teams. Their ML round is about that domain, not about LLM internals. This is ~10 h and it was missing from the plan, because the earlier version of this roadmap was aimed at frontier labs only.

## 10. Recommender and ranking systems

**Time:** ~7 h · 17 blocks.

| Sub-topic | AAI | INF |
|---|---|---|
| The funnel: candidate generation → ranking → re-ranking | Core | Skim |
| Why the funnel exists (millions of items, ~10 ms budget) | Core | Skim |
| Two-tower retrieval; user tower and item tower | Core | Skip |
| Negative sampling; why random negatives are too easy | Core | Skip |
| GBDT vs deep models for ranking; when each wins | Useful | Skip |
| CTR prediction, feature crosses, embedding tables | Useful | Skip |
| Cold start, popularity bias, position bias | Useful | Skip |

**Plain words.** *Candidate generation* = cheaply narrow millions of items to a few hundred. *Ranking* = score those few hundred carefully. *Two-tower* = one network turns the user into a vector, another turns the item into a vector, and you rank by dot product — the same trick as embedding search in RAG. *Negative sampling* = choosing which "wrong" items to train against.

### 10.1 The funnel and two-tower retrieval — ~4 h

- **Start here:** Google — [Recommendation Systems course](https://developers.google.com/machine-learning/recommendation). Free and short. Read *Retrieval*, *Scoring*, *Re-ranking*. This is the whole mental model.
- Eugene Yan — [System Design for Discovery](https://eugeneyan.com/writing/system-design-for-discovery/). The same funnel as it is actually built and served.
- Huang et al., *Embedding-based Retrieval in Facebook Search* — [arXiv:2006.11632](https://arxiv.org/abs/2006.11632). **§3 and §4 only** (model and negative mining). Meta's own system, and the negative-mining section is the part interviewers push on.

- **Build** (~50 lines, 1.5 h): a toy two-tower model on any small user-item dataset. Train with random negatives, measure recall@10. Then retrain with hard negatives (items the model scored highly but the user did not pick) and measure again. Write down both numbers.
- **Ready to move on when:** you draw the three-stage funnel and give a rough item count and latency budget for each stage; and you explain why random negatives make the offline number look good and the live system look bad.

### 10.2 Ranking models — ~3 h

- **Start here:** scikit-learn — [Ensembles user guide](https://scikit-learn.org/stable/modules/ensemble.html), the **Histogram-Based Gradient Boosting** section. This is the GBDT you were told to cut; it is back because Google, Meta and Amazon still rank with it.
- Cheng et al., *Wide & Deep Learning for Recommender Systems* — [arXiv:1606.07792](https://arxiv.org/abs/1606.07792). **§3 only.** Memorization vs generalization: the standard interview framing.
- Naumov et al., *DLRM* — [arXiv:1906.00091](https://arxiv.org/abs/1906.00091). **§2 only.** Meta's ranking model; note how big the embedding tables get and why that is a memory problem.

- **Build** (~40 lines, 1 h): train a GBDT and a small neural net on the same click dataset. Compare AUC and training time. Write two sentences on which you would ship and why.
- **Ready to move on when:** you say when GBDT beats a neural network for ranking (small data, tabular features, fast iteration) and when it does not (huge sparse embedding tables, multi-task).

## 11. A/B testing and online metrics

**Time:** ~3 h · 7 blocks.

| Sub-topic | AAI | INF |
|---|---|---|
| Offline metric vs online metric; why they disagree | Core | Useful |
| Guardrail metrics; the metric you must not hurt | Core | Useful |
| Novelty and primacy effects | Useful | Skip |
| Network interference (why social products break A/B assumptions) | Useful | Skip |
| Sample size and test duration | Core | Useful |

**Plain words.** *Guardrail metric* = a number that must not get worse even if your target metric improves — latency, or how often people leave. *Novelty effect* = a change looks good for a week only because it is new. *Network interference* = your control group is affected by the treatment group because they are friends.

- **Start here:** Ron Kohavi — [exp-platform.com](https://www.exp-platform.com/), the paper collection. Read *"Seven Rules of Thumb for Web Site Experimenters"*. Short, and the rules are exactly what gets asked.
- Kohavi, Tang & Xu, *Trustworthy Online Controlled Experiments* (Cambridge, 2020). Ch. 1–3 if you want depth. Optional.

- **Build** (~30 lines, 45 min): reuse `compare.py` from §6. Simulate an A/B test: generate two groups, apply a small true effect, and check how often a 1-week test detects it versus a 3-week test. This is sample size, felt rather than read.
- **Ready to move on when:** given "our new ranker raised clicks 2% but raised latency 40 ms," you say whether to ship it and what else you would look at first; and you name two reasons a good offline metric fails to move the online metric.

---

## Cut from the original README

`Naive Bayes` · `Decision Trees` · `Random Forests Part 1-2` · `Regularization Part 3: Elastic Net` · `Logistic Regression` (full series) · *Hands-On ML* **Ch. 3, 4, 5, 6, 7** · exercises in Ch. 1–4. Ch. 3 and Ch. 4 were cut in this pass: StatQuest covers the same ground faster, and you do not need both.

**Un-cut:** `Gradient Boost` — restored as §10.2, but only for the Google / Meta / Amazon lane.
