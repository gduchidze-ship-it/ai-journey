# System design drill — Fraud detection under a hard 100 ms budget (45 min)

**Format.** Ten minutes recall, thirty minutes designing aloud, five minutes writing what you missed. Timed. Speak out loud even if alone — the point is to practise producing a design under time pressure with an audience, and silence hides the gaps.

**Deliverable.** `fraud_100ms_design.md` in your repo, containing your design as you spoke it (a photo of the whiteboard is fine) and, separately headed, the *delta*: what you missed, in your own words, after reading the references. Do **not** read the references before the drill.

---

## The prompt

> You are the engineer responsible for the fraud-scoring step inside a card-payment authorization flow. When a payment request arrives, your service must return **allow / block / review** within **100 ms at the 99th percentile**, including any feature lookups and the model call. Above 100 ms the payment processor times out and the transaction is treated as approved without a decision. Volume is 5,000 requests per second at peak. Roughly 0.1 % of transactions are fraudulent. Fraud labels arrive as chargebacks 30–90 days later. Design the system.

Ask yourself the clarifying questions an interviewer would expect; when you cannot ask, state the assumption and continue.

## Minute 0–10: recall

Before designing, write from memory (no notes) the answers to these. They are the lecture material this drill exercises.

1. What does 0.1 % prevalence do to precision at any given threshold? Write the Bayes' rule formula for precision in terms of TPR, FPR and prevalence. (Part 5 §5.4)
2. Which two curves would you show, and which one changes when prevalence changes? (Part 4 §4.7)
3. What are the two error costs here, in words, and who in the company knows the numbers? (Part 5 §5.2–5.3)
4. What is the optimal threshold rule for calibrated scores, and what assumption does 30–90-day label delay threaten? (Part 5 §5.2; Part 6 §6.6; Part 7 §7.3)
5. Which model family would you reach for first on tabular payment features, and why? (Part 8 §8.2)

## Minute 10–40: design aloud

Work through these layers. The order is deliberate: latency budget first, because it constrains everything after it.

**1. The latency budget.** Split 100 ms into stages: network in, feature fetch, model inference, rule evaluation, network out. Write a number next to each. What is left for the model? Where do tail latencies come from and what is your p99 story for each stage? What happens at 101 ms — and is "treated as approved" the right fallback, or should the fallback be a rules-only decision you can compute in 5 ms?

**2. Features.** List ten features you would want. For each: can it be computed at request time from the request itself, or does it depend on history (velocity: "number of transactions on this card in the last hour"; "countries this card was approved in this month")? History features need a **precomputed store** with a latency SLA. How fresh must they be — seconds, minutes, hours? What is the consistency story between the feature values used in training and the ones served (part 3 §3.2, Gap 2 — training/serving skew)? Where do graph features (shared device, shared address across accounts) fit, and what do they cost in latency?

**3. The model.** Which family, why, what does it cost per inference on CPU, and does that fit the budget you left for it? How many features can it consume in that time? How would you know if a bigger model is worth it — what offline metric, evaluated how?

**4. Metrics and threshold.** Which curve do you show the risk team? What is the operating point, and *whose* costs set it? Sketch the cost of a false positive (declined good customer) and of a false negative (chargeback) as line items. What review capacity exists, and how does that cap the number of *review* decisions per day? Convert your FPR at the chosen threshold into false declines per day at 5,000 rps.

**5. Calibration and delayed labels.** Labels arrive in 30–90 days. What do you monitor *today* to know whether the model is degrading — inputs, outputs, or both (part 7 §7.3)? How do you detect that fraud prevalence has doubled before the chargebacks tell you? What happens to your threshold when it does (part 6 §6.6)?

**6. Rules and overrides.** Where do hand-written rules sit relative to the model — before, after, in parallel? Who can edit them and how fast do they deploy? How do you stop a rule from silently eating the model's recall?

**7. Rollout.** How does a new model get to production without betting the payment flow on it? Shadow scoring, canary percentage, what metric gates promotion, what triggers rollback, and how long you wait given the label delay.

**8. Explanation.** A merchant asks why a payment was declined. What can you say, and what did you build to be able to say it?

Stop at minute 40 regardless of where you are.

## Minute 40–45: the delta

Now read, in this order: Stripe, "How we built it: Stripe Radar" (Lessons 1–3, ~8 min); Stripe Radar docs, "Risk evaluation" (risk levels and the `not_assessed` / `unknown` states); PayPal's real-time graph post (the requirements list). Links in `../../READING.md` under "System design drill." Then write under a heading **Delta** what you missed or got wrong. Prompts:

- Did your budget leave room for ~1,000 features? Stripe's does, inside 100 ms.
- Did you start with the simplest model? Stripe started with logistic regression, moved to XGBoost + DNN, and only in 2022 — with 10× the training data — moved to DNN-only. Did you jump straight to the neural network?
- Did you name a *guardrail metric* for model changes? Stripe rejected a change over "a 1.5 % drop in recall — an unacceptably large regression." What is your equivalent, and did you say who owns it?
- Did you define what happens when the model *cannot* score? Stripe's product exposes `not_assessed` and `unknown` as first-class risk levels. Did your design have an explicit "no decision" path, or did it silently fall through?
- Did you separate the *score* (0–99 in Stripe's product) from the *thresholds* (65 elevated, 75 high by default) and let the merchant move the thresholds? Or did your model emit allow/block directly?
- Did your history features have a freshness number? PayPal's graph stack targets seconds-level freshness, 10 ms two-hop queries, and a 100 ms SLA for graph compute.
- Did you mention explanations at all? Stripe: "Explanation matters as much as detection."
- Did you do the base-rate arithmetic out loud? At 5,000 rps, 0.1 % fraud and a 0.5 % FPR, how many good customers do you decline per hour?

## Self-grading rubric

| | Pass | Strong |
|---|---|---|
| Recall (min 0–10) | 4 of 5 from memory | All 5, with the precision formula written correctly |
| Latency budget | Stages named with numbers that sum to ≤ 100 ms | Plus a p99 story per stage and an explicit fallback path with its own latency |
| Features | ≥ 8 features, split into request-time vs. precomputed | Plus freshness SLAs and a training/serving parity mechanism |
| Metrics & threshold | PR curve named; both error costs named in words; threshold tied to costs | FPR × traffic converted to declines per day; review-queue capacity as a constraint |
| Delayed labels | Output-distribution monitoring mentioned | Prevalence-shift detection and its effect on the threshold |
| Rollout | Shadow → canary → promote, with a gate metric | Plus a rollback trigger and an explicit wait tied to label delay |
| Delta | Written, ≥ 4 concrete misses | Each miss paired with the design change you would make |

## Common misses (do not read until after the delta)

<details>
<summary>Reveal</summary>

- Designing the model before the budget. The budget decides the model.
- No "no decision" path. Something *will* time out at p99.99; if the default is "approve," fraudsters will learn to cause the timeout.
- Precision quoted without prevalence. At 0.1 % positive, "95 % precision" is an extraordinary claim.
- Threshold set by the engineer. It is set by the cost ratio, which is owned by risk/finance; the engineer's job is to expose the sweep.
- Ignoring the review queue as a hard capacity. "Review" is not free; it is a team of people with a throughput.
- Retraining "weekly" without saying on which labels — the last week's labels do not exist yet.
- Treating rules as legacy. They are the fastest-deploying control you have and the fallback when the model is unavailable.

</details>
