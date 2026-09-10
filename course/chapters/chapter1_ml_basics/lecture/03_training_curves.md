# 3 · Why a training curve alone never tells you whether to ship

*Lecture 1, part 3 of 8 · ~10 min reading*

A training curve is a plot of loss against steps. It is the first thing every ML tutorial shows and the last thing that should decide a launch. This part is short because the argument is short; it is a separate part because the failure is common.

## 3.1 What the curve *can* tell you

Google's Crash Course page "Interpreting loss curves" is a good field guide to the diagnostics a curve does support, and you should be able to recognize each shape:

- **Training loss falling while validation loss rises** → "The model is overfitting the training set." Fixes: simplify, fewer features, "increase the regularization rate," and — the one people skip — check that train and validation sets are statistically equivalent.
- **Oscillating loss** → bad examples in the data, or a learning rate too high.
- **A sharp jump** → NaNs, or a burst of outliers in a batch.
- **Repeating waves** → "The training set is not shuffled well."

Those are all statements about *the optimization*. None of them is a statement about *the product*.

## 3.2 Four gaps between the curve and the decision

**Gap 1 — the loss is a proxy.** Google's "Rules of Machine Learning" (Zinkevich) separates the words carefully. A *metric* is "a number that you care about. May or may not be directly optimized." An *objective* is "a metric that your algorithm is trying to optimize." Rule #13: "The ML objective should be something that is easy to measure and is a proxy for the 'true' objective … often there is no 'true' objective." Cross-entropy is going down; is fraud loss in dollars going down? Those are different questions, and the curve answers only the first.

**Gap 2 — the loss is on the wrong data.** Rule #37, "Measure Training/Serving Skew," names three distinct differences: (1) "the performance on the training data and the holdout data" — the only one the curve shows; (2) "the performance on the holdout data and the 'next-day' data" — time has passed; (3) "the performance on the 'next-day' data and the live data" — a discrepancy here "probably indicates an engineering error." Breck et al.'s *ML Test Score* (Monitor 7) states the underlying fact: "Validation data will always be older than real serving input data, so measuring a model's quality on that validation data before pushing it to serving is only an estimate."

**Gap 3 — the loss is an average.** A single number hides slices. *ML Test Score* Model 6: "Model quality is sufficient on all important data slices." A guardrail that is 99 % accurate overall and 60 % accurate on one language is a guardrail that fails for that language's users, and the curve is indifferent.

**Gap 4 — lower loss is not the launch criterion.** Rule #39: "Launch decisions are a proxy for long-term product goals … The only easy launch decisions are when all metrics get better (or at least do not get worse)." Rule #24: before any user sees a new model, "calculate just how different the new results are from production." *ML Test Score* Model 2 makes the requirement explicit: "Offline proxy metrics correlate with actual online impact metrics" — you have to have *shown* that your offline metric tracks engagement, revenue, or whatever the product is measured by. And Model 5, the humbling one: "A simpler model is not better" — "regularly testing against a very simple baseline model, such as a linear model with very few features, is an effective strategy." If a curve is your only evidence, you have not run that test.

## 3.3 What a shipping decision actually rests on

The rest of this lecture is the answer. A decision to ship a classifier rests on: a **metric that matches the cost structure** (part 4 — and F1 is often the wrong one); a **threshold chosen from those costs** (part 5); a **confidence score that means what it says** (part 6); evidence that the **data the model will see resembles the data it was fitted on** (part 7); and a monitoring plan, because *ML Test Score* awards a full point only when a test runs "automatically on a repeated basis," and the rubric's final score is "the minimum of the scores aggregated for each of the 4 sections." Your weakest section is your score.

The training curve is a debugging tool. It tells you whether the optimizer is healthy. It never tells you whether to ship.

---

**Sources for this part** (exact sections in `READING.md`): Google ML Crash Course, "Interpreting loss curves" · Zinkevich, "Rules of Machine Learning," Terminology and Rules #13, #24, #37, #39 · Breck, Cai, Nielsen, Salib & Sculley, "The ML Test Score" (IEEE Big Data 2017), §III Model 2/5/6, §V Monitor 7, §VI.
