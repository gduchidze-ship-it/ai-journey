# Production drill — Cost one shipped threshold, in currency (45 min)

**The task, verbatim from the syllabus.** Take one classifier threshold in any system you have shipped and write the cost of each error type in currency. If you cannot, write down who would know.

**Deliverable.** `threshold_costing.md` in your repo, filled in from the template below. It is short. The value is in the finding, not the writing.

## Why this drill exists

Part 5 argued that the threshold *is* the cost ratio, and that if you do not know the two costs, the library author chose your threshold for you. This drill finds out whether that is true of something you actually own. Most engineers who do it for the first time discover that (a) there is a threshold in their system they had not thought of as a threshold, and (b) nobody has ever written down what its errors cost. Both discoveries are the deliverable.

## Finding a threshold (5 min)

"Classifier threshold" is broader than "a number passed to an ML model." Any place a continuous signal becomes a yes/no decision counts. If you have never shipped an ML model, you have shipped one of these:

- an alerting rule (`error_rate > 2 % for 5 min` → page someone)
- a rate limiter or abuse detector (`requests/min > N` → block)
- a spam/abuse filter, a moderation score cutoff, a prompt-injection or jailbreak guardrail
- a fraud or risk score cutoff, a credit or eligibility rule
- a retry/circuit-breaker decision (`p99 latency > X` → open circuit)
- a search or recommendation relevance cutoff (`score < X` → don't show)
- an autoscaler trigger (`CPU > 70 %` → add a node)
- a deduplication or entity-matching similarity cutoff
- a health check (`3 failures` → restart the pod)
- an LLM router (`confidence < X` → send to the bigger model / a human)

Pick **one**. Prefer the one whose errors you have personally been paged for.

## The template (30 min)

Copy this into `threshold_costing.md` and fill every line. Where you do not know, write "unknown" and go to the last section — do not guess and do not leave it blank.

```markdown
# Threshold costing — <system name>

## The decision
- **What continuous signal** becomes a yes/no here?
- **What is the current threshold value**, and where does it live (config, code, a dashboard someone edits)?
- **Who set it, when, and on what basis?** (git blame is allowed. "Default" is an acceptable and alarming answer.)
- **How many decisions per day** does this threshold make? How many come out positive?

## The two errors
Define them in this system's own words before pricing them.
- **False positive** (flagged, but shouldn't have been): what concretely happens to a user / a customer / an on-call engineer / the business when this fires wrongly?
- **False negative** (not flagged, but should have been): what concretely happens?

## The costs, in currency
For each, show the arithmetic, not just the number. State the currency and the unit (per event).
- **C_FP = ___ per false positive.** Components: (e.g. lost margin on a declined order; support-ticket handling time × loaded hourly rate; churn probability × customer lifetime value; on-call minutes × rate; compute for the fallback path)
- **C_FN = ___ per false negative.** Components: (e.g. chargeback amount + fee; incident cost; regulatory exposure; harm to a user, priced at what you would pay to prevent it)
- **Cost ratio C_FN / C_FP = ___.**
- **Implied threshold** if the score were a calibrated probability: C_FP / (C_FP + C_FN) = ___. (Part 5 §5.2. If the score is not a probability, write that down — it means the closed form does not apply and the threshold must come from a sweep.)

## Comparison
- Current threshold: ___ .  Cost-implied threshold: ___ .
- Are they in the same neighbourhood? If not, in which direction is the current threshold wrong, and what is that costing per day (estimate: (# of decisions/day) × (rate of the error you are over-producing) × its unit cost)?
- Would you change it tomorrow? If not, what stops you? (Reasons like "we've never measured FPR" or "we don't know the positive rate" are the right kind of answer.)

## Prevalence check
- What fraction of the signal's inputs are truly positive (best estimate)? How do you know?
- At the current threshold, roughly what is the FPR? Multiply it by the number of negatives per day. Is that number visible to anyone today?

## If you could not fill a line
For every "unknown" above, one row:
| Unknown | Who would know (name or role) | How you would ask them (one sentence) |
|---|---|---|
| | | |
```

## Then (10 min)

Send one of the questions from your last table to the person who would know — a message, a ticket, a calendar hold. The drill is complete when the question has left your machine. Record in the file what you sent and to whom.

## Self-grading rubric

| | Pass | Strong |
|---|---|---|
| Threshold identified | One real threshold, with its current value and location | Plus who set it and on what basis |
| Errors defined | Both errors described concretely in the system's terms | Described from the affected person's point of view |
| Costs | Both costs in currency with arithmetic shown, **or** both marked unknown with a named owner | Costs in currency with a range and the assumption behind each component |
| Comparison | Current vs. implied threshold stated | Daily cost of the gap estimated |
| Prevalence | Positive rate and FPR estimated | FPR × negatives/day computed and compared with a real queue or pager volume |
| Follow-through | A question actually sent | A reply received and folded back into the file |

## What people usually find

Most first attempts end with the cost of a false negative reasonably priced and the cost of a false positive marked "unknown" — because false negatives cause incidents, which get post-mortems, which produce numbers, and false positives cause quiet friction, which nobody tallies. Quiet friction at scale is usually the bigger number. The base-rate argument of part 5 §5.4 says why: at a rare positive class the false positives outnumber the true ones, so the error you are not pricing is the one you are producing the most of.
