# System design drill — Eval-gated CI/CD: thresholds, noise, blocking policy, rollback (45 min)

**Format.** Ten minutes recall, thirty minutes designing aloud, five minutes writing what you missed.
Timed. Speak out loud even if alone — silence hides the gaps. Whiteboard or paper; a photo is fine.

**Deliverable.** `eval_gated_cicd_design.md` in your repo: your design as you spoke it, then, under a
separate heading **Delta**, what you missed — first from memory, then after reading the reference
solution and the sources. The timed drill is 45 minutes; the second delta is 15 minutes more,
taken from this week's reading slot (the solution is part of the reading).

**Do not open `system_design_eval_gated_cicd_SOLUTION.md` until the 45-minute timer has run.** It is a full worked
design with diagrams. Reading it first turns a drill into a reading assignment.

---

## The prompt

> You own the CI/CD pipeline for a customer-support agent (the one you have been building since
> Lecture 34). It has four things that change its behaviour and are edited by different people at
> different rates: the **system prompt** (edited several times a week by two engineers and a product
> manager), the **tool schemas and descriptions** (weekly), the **retrieval index** (rebuilt nightly
> from a documentation repo that other teams edit), and the **model ID** (changed every few months,
> sometimes by the vendor deprecating one). You have a 40-case deterministic harness from this week
> and, from next week, a validated LLM judge on the same cases. A full run of the forty cases costs
> about $2 and takes about six minutes. Roughly 15 PRs a week touch one of the four things.
>
> Design the pipeline that decides whether each change ships: what runs where, what the gate
> compares, how the threshold is set so it blocks real regressions without blocking on noise, what
> blocks versus what warns, and how a bad change is rolled back — including one that came from the
> vendor rather than from a PR. Design it so that a reviewer who sees "BLOCKED" can act without
> re-running anything.

Ask the clarifying questions an interviewer would expect; when you cannot ask, state the assumption
and continue.

## Minute 0–10: recall

From memory, no notes:

1. A test asserts a property; an eval estimates a rate. What is the correct pass rate for each, and
   what are the two failure modes of confusing them? (Part 8 §8.1)
2. Why compare two runs per case rather than by aggregate pass rate? Whose recommendation is that,
   and what does pairing buy? (Part 4 §4.5, part 8 §8.4)
3. At `n = 40`, what is the 95 % interval half-width around an 80 % pass rate? What can a gate on
   forty cases therefore detect, and what can it not? (Part 4 §4.5)
4. What is the noise floor, how do you measure it, and what is Google's operational definition of a
   case that contributes to it? (Part 8 §8.2, §8.4)
5. Name the things that must be version-pinned for two eval reports to be comparable (the notes list
   seven), and the vendor guarantee that makes model pinning meaningful. (Part 8 §8.4)
6. What is `NODATA`, where does the term come from, and why must it never count as a pass? (Part 8 §8.4)

## Minute 10–40: design aloud

Work through these layers in order. The order is deliberate: what you can afford to run per PR
decides everything downstream.

**1. Stages and their budgets.** Fifteen PRs a week, six minutes and $2 per full run. Which of the
following runs on every commit, on every PR, on merge to main, nightly, and on a vendor change:
pytest over the check code; deterministic checks over *recorded* trajectories; the forty cases
against the live pinned agent; repeated trials for `pass^k`; the judge; the held-out slice; the
capability set. Put a wall-clock and a dollar figure next to each stage. What does caching (replayed
recordings, cassette-style LLM call caches) let you move to an earlier stage, and what does a cached
run *not* tell you?

**2. What exactly is compared.** Define the *baseline*: the last run on `main` with the same pinned
model, index version and tool-schema version? A fresh run of `main` triggered alongside the PR? Why
does the SRE Workbook call before/after comparison "risky," and does that apply here? Define the
*candidate*. Define the *diff*: per-case flips, with check names, not two percentages.

**3. Thresholds and the noise floor.** How do you measure how much the suite moves on its own?
How often do you re-measure it (the model is pinned, but the index is rebuilt nightly)? Given a
measured spread, how do you set the aggregate threshold so that the gate blocks a real regression
and not a flake? What do you do with a case that is flaky by itself — quarantine, `trials: 3`, or
delete — and who decides? What is the Ladder policy and how does it apply?

**4. Blocking policy.** Write the list of things that block a merge outright, the list that warns,
and the list that is merely recorded. Where do these go: a forbidden-tool call on an injection case;
a flip on a case tagged `incident-*`; a schema-validity failure; a five-point drop in aggregate pass
rate on forty cases; a `NODATA` case; a cost p95 that doubled; a judge-score drop (next week)? Who
can override a block, how is the override recorded, and does the override create a case?

**5. Regression versus capability.** You have both kinds of case in the set. Which suite does the
gate read, and what happens to the other? How do you stop the regression suite from silently
becoming 100 %-and-meaningless?

**6. Pinning and provenance.** Every report carries: model ID, prompt sha, tool-schema sha, index
version, decoding parameters, harness version, case-set sha. Which of these is the trigger for a
rerun when it changes *without* a PR (the nightly index; a vendor alias moving)? How do you make a
vendor-side change visible as a diff in *your* pipeline rather than as a mystery drop in production?

**7. Rollback.** A bad change merged. What is the rollback for each of the four change types —
prompt, tool schema, index, model — in terms of *re-pinning a version*, not redeploying code? How
long does each take? What if the bad change is the vendor's (an alias now resolves to a new snapshot,
or the pinned snapshot is deprecated)? Where does the canary from Lecture 40 fit relative to this
gate: what does offline eval prove that a canary cannot, and vice versa?

**8. The reviewer's view.** Sketch the PR comment. A reviewer sees `BLOCKED`. What must be on the
screen so they can act without re-running: which case, which check, the reason string, the flip
direction, a link to the trajectory diff, the noise-floor context ("this case has been stable for 40
runs" versus "this case flaked twice last week"). What does `WARN` look like, and how do you keep it
from being ignored?

Stop at minute 40 regardless of where you are.

## Minute 40–45: the first delta

Timer still running, solution still closed: write under **Delta** the questions from the design
section you could not answer or knew you were hand-waving. That list is the honest measure of the
drill; the reference will only confirm it.

## After the timer — 15 min, the second delta

Now open `system_design_eval_gated_cicd_SOLUTION.md`. Read §3.1's table, §4's two lists, §7's table
and §9 "Decisions" — about ten minutes; the diagrams are there for when you want the mechanism, not
for now. Then the two sources, five minutes: SRE Workbook ch. 16 §"Before/After Evaluation Is Risky";
Kayenta §"Judgment". Extend the **Delta** with what you missed or got wrong. Prompts:

- Did you separate the *commit stage* (model-free, seconds) from the *merge gate* (live, minutes,
  dollars) from the *nightly* (judge, held-out, repeated trials)? Or did everything run on every PR?
- Did your baseline share the candidate's pinned versions, or were you comparing a PR against a
  week-old number from a different index?
- Did you measure the noise floor before setting a threshold, or did you pick 95 %?
- Did anything block on an aggregate percentage alone? On forty cases, should it?
- Did `NODATA` appear anywhere in your design? Did a missing trajectory count as a pass?
- Did you have a rollback for a *vendor* change, and did your pipeline notice it before production did?
- Did you name what blocks, what warns, and who can override — and does an override mint a case?
- Could your reviewer act on `BLOCKED` without re-running? What was on the screen?
- Did you keep the capability suite out of the gate?

## Self-grading rubric

| | Pass | Strong |
|---|---|---|
| Recall | 4 of 6 from memory | 6 of 6, with the numbers (±12 pp at n = 40; 3.05× clustering; 5 % × 20 % = 1 %) |
| Stages | Three stages with budgets | Cached replay in the commit stage; live run gated on which files changed |
| Comparison | Per-case diff named | Baseline re-run under identical pins; SRE "before/after" risk named and handled |
| Noise | Noise floor measured before threshold | Re-measured on index rebuild; flaky cases quarantined *and* tracked; Ladder step size stated |
| Policy | Block / warn / record lists written | Override path recorded and mints a case; capability suite excluded from the gate |
| Rollback | Per-change-type re-pin | Vendor-change detection and rollback; relation to Lecture 40's canary stated correctly |
| Delta | Written | Each miss tied to the lecture part or source that covers it |
