# Chapter 1 — What a model is, and how you judge one

**Week 1 · Lecture A · fully theoretical · 2 h lecture + 4 h independent work**

This chapter is for engineers who already ship software and now need to reason about a component whose behaviour was *fitted* rather than *written*. Nothing here assumes prior ML. Everything here is used again every week of the course: the metrics you build this week are the ones you will point at your retrieval stage in week 1B, your fine-tuned model in week 10, and your eval gate in week 21.

## How to use this folder

Read the lecture notes in order (they build on each other), then do the independent work in the order listed. Every source cited in the notes is real, was checked on 2026-09-10, and is listed with the exact section to read in [`READING.md`](READING.md). Prefer those sources over memory when the notes and your intuition disagree.

There are **no solutions in this folder**. The `metrics.py` build ships with a test file so you can check yourself, and the drills ship with a rubric so you can grade yourself. If you are stuck, the notes tell you which source section unblocks you.

## File map

| Path | What it is | Time |
|---|---|---|
| `lecture/01_function_fitting.md` | ML as function fitting; ERM; generalization; train/val/test; bias–variance; capacity; over/underfitting; double descent caveat | 25 min |
| `lecture/02_regularization.md` | L1, L2, early stopping — what each constrains; weight decay vs L2 in the loss; why AdamW exists | 25 min |
| `lecture/03_training_curves.md` | Why a training curve alone never tells you whether to ship | 10 min |
| `lecture/04_classification_metrics.md` | Confusion matrix; precision, recall, F1; when each is the wrong summary; why accuracy lies; ROC/AUC vs PR under imbalance | 25 min |
| `lecture/05_thresholds_and_base_rates.md` | The threshold as a product decision; cost matrices; why a guardrail at 0.9 fires far more than "10% error" suggests | 15 min |
| `lecture/06_calibration.md` | What a confidence score means; reliability diagrams; ECE; Brier; Platt scaling and isotonic regression **[extension]** | 20 min |
| `lecture/07_data_quality.md` | Missing values (drop / impute / missingness as signal); class imbalance; distribution shift and how it is detected | 20 min |
| `lecture/08_classical_models.md` | One-sentence awareness pass over eight classical models; where gradient boosting still beats a neural network **[extension]** | 10 min |
| `lecture/SELF_CHECK.md` | Questions to answer from memory after the lecture. No answers given. | — |
| `READING.md` | The 30-minute marked reading, plus the full annotated source list | 30 min |
| `independent_work/build_metrics/` | **Build (2 h).** `metrics.py` specification, skeleton, sample data, and a test file | 2 h |
| `independent_work/drills/system_design_fraud_100ms.md` | **System design drill (45 min).** Fraud detection under a hard 100 ms budget | 45 min |
| `independent_work/drills/production_threshold_costing.md` | **Production drill (45 min).** Cost each error type of one shipped threshold, in currency | 45 min |
| `COVERAGE_AUDIT.md` | Author's self-reflection: every syllabus bullet mapped to where it is covered, and known gaps | — |

## Time budget for the 4 h independent block

| Slot | Time | Deliverable that lands in your repo |
|---|---|---|
| Build | 2 h | `metrics.py` passing `test_metrics.py`, one reliability diagram PNG, and `NOTES.md` with five short answers |
| System design drill | 45 min | `fraud_100ms_design.md` — your design, then your written delta |
| Production drill | 45 min | `threshold_costing.md` — one threshold, both error costs in currency, or the name of who knows |
| Reading | 30 min | Nothing to hand in; the self-check questions assume you did it |

## What you should be able to do by the end

Explain to another engineer, without notes, why accuracy is the wrong number on a 1 % positive rate; sketch a confusion matrix and derive precision, recall, F1 from it; draw a ROC and a PR curve for the same classifier and say which one moves when the class balance changes; state the Bayes-optimal threshold in terms of the two error costs; read a reliability diagram; name the three kinds of missingness and the three kinds of distribution shift; and say in one sentence what AdamW changes relative to Adam with L2.

This chapter feeds **Project 1 — Metrics and significance toolkit** (due after week 1). The `metrics.py` you build here is half of it; week 1B adds ranking metrics and the significance test.
