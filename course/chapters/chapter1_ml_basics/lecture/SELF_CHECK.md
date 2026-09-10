# Self-check — Lecture 1

Answer from memory, on paper, after finishing the notes and the reading. No answers are provided; every question is answered somewhere in `lecture/01`–`08` and you should be able to find and verify your own. If you cannot answer one without looking, that is the section to reread before the build.

A good target: 24 of 29 without notes.

## Function fitting and generalization

1. Write the empirical risk in one line. What is the difference between it and the population risk, and what is the name of that difference?
2. Name the two assumptions that let a training score say anything about a future input, plus the third one production breaks most often.
3. Your training loss is 0.02 and your validation loss is 0.45. Which word describes this, and name three different moves that address it.
4. Your training loss and validation loss are both 0.40. Which word describes this, and why does adding L2 make it worse?
5. Complete: "It might be wise to use a biased estimator, so long as …". Why is this sentence the justification for every regularizer?
6. In what regime does the bias–variance U-curve stop being the whole story, and what is the phenomenon called?

## Regularization

7. Which parameters does L2 shrink the most — the ones the data pins down or the ones it does not? Explain in terms of Hessian eigenvalues.
8. Draw the 2-D constraint regions for L1 and L2. Which one produces exactly-zero weights, and why?
9. State the relationship between early-stopping steps `τ`, learning rate `ε`, and L2 strength `α`.
10. Write the one-line update for Adam + L2 and the one-line update for AdamW. Circle the difference.
11. Why does L2-in-the-loss regularize the *most active* parameters the *least* under Adam? Which proposition in Loshchilov & Hutter says no L2 coefficient can fix this?
12. In PyTorch, what does `Adam(weight_decay=0.01)` actually do, and what should you use instead?

## Training curves and shipping

13. Name Google's three training/serving skews (Rule #37). Which one does a training curve show?
14. State two of the four gaps between a loss curve and a launch decision without looking.

## Metrics

15. From TP = 40, FP = 60, FN = 10, TN = 9,890: compute accuracy, precision, recall, F1, FPR, and the majority-class baseline accuracy. Which of these numbers would you show a product manager, and which would you refuse to show alone?
16. Name three specific defects of F1 as a summary.
17. A colleague reports "AUC 0.93" on a dataset with 0.2 % positives. What one question do you ask, and what curve do you ask them to draw?
18. You double the number of negatives in a test set. What happens to the ROC curve? To the PR curve? To the PR baseline?
19. Why does scikit-learn refuse to linearly interpolate the PR curve when computing average precision?

## Thresholds and base rates

20. `C_FP = €3`, `C_FN = €27`. What is the Bayes-optimal threshold on a calibrated score, and what assumption makes it optimal?
21. A detector has 90 % recall and 2 % FPR at your threshold, on traffic that is 0.5 % positive. Compute precision. Then explain to a non-engineer why "we only act at 90 % confidence" did not give 90 % precision.

## Calibration

22. Define calibration in one equation. Then describe a model that satisfies it perfectly and is useless, and name the property it lacks.
23. Sketch a reliability diagram for an over-confident model. Where do the points sit relative to the diagonal? Write the ECE formula.
24. When do you choose isotonic regression over Platt scaling, and what is the one rule about *which data* you fit either on?

## Data quality and classical models

25. Give a one-line example each of MCAR, MAR, MNAR. For which one does imputation destroy information, and what do you add to preserve it?
26. Why does SMOTE break the threshold you computed in Q20? What does the 2022 evidence say to do instead?
27. Factor `P(X, y)` and name the shift for each factor moving. Which one leaves ROC unchanged but breaks calibration?
28. Why does a KS test at n = 1,000,000 flag everything, and what do drift tools switch to at scale?
29. State the three reasons Grinsztajn et al. give for trees beating neural networks on tabular data.
