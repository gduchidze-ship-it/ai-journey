# Chapter 37 — Eval types and deterministic checks

**Week 19 · Lecture A · fully theoretical · 2 h lecture + 4 h independent work**

You have spent weeks 17–18 building an agent: a loop, tools, a graph with checkpoints, rails,
recovery, an injection defence. This chapter is the first of four (37–40) on the question none of that
answers: how do you *know* it works, how do you know it *still* works after Tuesday's prompt edit, and
what evidence is enough to ship a change. This week is the half of the answer that needs no model to
grade anything — the deterministic half — plus the case-set discipline that the judge (Lecture 38),
the experiment (39) and the shipping decision (40) all stand on.

## How to use this folder

Read the lecture notes in order; they build on each other and the later parts assume the vocabulary of
the earlier ones. Then do the independent work in the order listed. Every source cited in the notes is
real, was fetched and checked on 2026-09-10, and is listed with the exact section to read in
[`READING.md`](READING.md). Prefer those sources over memory when the notes and your intuition disagree.

There are **no solutions in this folder for the build or the production drill**. The harness ships
with a test file so you can check yourself, and the drills ship with rubrics. The **system design
drill is the exception**: it has a full worked reference solution with diagrams,
`system_design_eval_gated_cicd_SOLUTION.md`, and the drill tells you not to open it until the timer has run.

## File map

| Path | What it is | Time |
|---|---|---|
| `lecture/01_why_judgement_cannot_ship.md` | Non-determinism at temperature 0 and why; the sample-size argument; flakiness as a property, not a bug; pass@k vs pass^k | 15 min |
| `lecture/02_eval_types.md` | Unit / component / end-to-end — what each catches and cannot; offline vs online; reference-based vs reference-free; regression vs capability | 20 min |
| `lecture/03_golden_vs_synthetic.md` | Golden vs synthetic cases; the curated → production → synthetic order; how to build each; what synthetic gets wrong | 20 min |
| `lecture/04_contamination_and_held_out.md` | Prompt tuning as fitting; Blum–Hardt, Dwork, Recht, GSM1k; the steel-man; the held-out slice, the one-look rule, refresh; error bars and how big a slice must be | 25 min |
| `lecture/05_deterministic_checks.md` | Why deterministic first; exact/normalized match, schema validation, tool-call correctness (BFCL's four stages), citation presence/validity; reference-free property checks; writing a check that means something | 25 min |
| `lecture/06_agent_deterministic_checks.md` | Right tool, valid order (constraints not scripts), step budget, termination reason (the seven `stop_reason`s and the `end_turn` trap), end state; repeat and report the rate | 25 min |
| `lecture/07_cases_from_real_failures.md` | Criteria drift; error analysis (open → axial coding); failure → trace → check → one representative per cluster; every bug becomes a case; closing the loop from production | 20 min |
| `lecture/08_pytest_vs_evaluation_ci.md` | A test asserts a property, an eval estimates a rate; stages; the gate — per-case comparison, noise floor, block/warn, NODATA, re-pin rollback; a worked shape | 25 min |
| `lecture/SELF_CHECK.md` | 38 questions to answer from memory. No answers given. | — |
| `READING.md` | The 45-minute marked reading, plus the full annotated source list by topic | 45 min |
| `independent_work/build_harness/` | **Build (2 h).** The deterministic half of a 40-case harness: 18 checks + 4 stats functions (runner and gate provided); the case-set structure with a held-out slice, seeded with ≥ 15 cases about your agent and grown to 40 by Lecture 38; a 15-case test bed with recorded trajectories; 53 tests | 2 h |
| `independent_work/drills/system_design_eval_gated_cicd.md` | **System design drill (45 min).** Eval-gated CI/CD — thresholds, noise, blocking policy, rollback | 45 min |
| `independent_work/drills/system_design_eval_gated_cicd_SOLUTION.md` | Full reference design with four Mermaid diagrams, a decisions table, and the reviewer's PR view. **Read after the drill.** | 15 min, after the timer |
| `independent_work/drills/production_bugs_to_eval_cases.md` | **Production drill (45 min).** Every bug you fix becomes an eval case before the fix merges; start with the last three, and *prove* each fails pre-fix | 45 min |
| `COVERAGE_AUDIT.md` | Author's self-reflection: every syllabus bullet mapped to where it is covered, the independent review's findings, fixes, and known gaps | — |

## Time budget for the 4 h independent block

| Slot | Time | Deliverable that lands in your repo |
|---|---|---|
| Build | 2 h | The harness passing all 53 tests; ≥ 15 cases about *your* agent split working/held-out by a pre-written rule (40 by the end of Lecture 38); recorded trajectories; a replay report; `NOTES.md` with four answers |
| System design drill | 45 min + 15 | `eval_gated_cicd_design.md` — your design as spoken, a delta from memory, then a second delta against the reference solution (the 15 min comes out of the reading slot) |
| Production drill | 45 min | Three `incident-*` cases, each proven to fail on the pre-fix commit; `bugs_to_cases.md` |
| Reading | 45 min | Nothing to hand in; the self-check assumes you did it. (The four slots sum to 4 h 15, as the syllabus's own numbers do.) |

## What you should be able to do by the end

Explain to another engineer, without notes, why temperature 0 does not give you determinism and why
that rules out shipping on inspection; name the three eval levels and the one failure each alone
cannot see; state the difference between offline and online in one clause; give the curated →
production → synthetic ordering and the reason synthesis comes last; explain why forty accepted prompt
revisions make a case set's score a training score, quote the 50 %-reported-as-63 % result, and state
the one-look rule for a held-out slice and what a slice of forty can and cannot detect; list the four
families of deterministic check and what each cannot tell you; list the five agent-specific assertions
and the three termination traps; describe open and axial coding and SQLite's regression rule; and draw
the three stages of an eval-gated pipeline with what blocks, what warns, and what `NODATA` means.

This chapter is the first quarter of **Project 7 — Agent platform, eval gate, security and
observability** (due after week 21). The harness you build here is the deterministic half of the
gate; Lecture 38 adds the judge and wires it into CI.
