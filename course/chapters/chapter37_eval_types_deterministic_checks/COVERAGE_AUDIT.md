# Coverage audit — Lecture 37 materials

*Author's self-reflection, written after the materials were drafted and independently reviewed. Date: 2026-09-10.*

This file answers one question: **does every bullet of the syllabus have a home in these materials, grounded in a real source, with no solutions leaked to the student for the build or the production drill?** It records where each bullet is covered, what an independent review found, what was changed in response, and what remains open. It is kept in the folder deliberately: the lecture teaches that a claim of coverage without evidence is worth nothing, so the materials hold themselves to the same standard.

## Method

1. Researched every syllabus bullet on the open web *before* writing, in four parallel passes (eval basics and non-determinism; datasets and contamination; deterministic and agent checks; failure-driven cases and CI). About 55 primary sources were fetched and read; section headings and verbatim quotes were recorded; only sources that actually loaded are cited. Twelve that could not be reached are listed at the bottom of `READING.md` and are not load-bearing.
2. Drafted the eight lecture parts, the reading list, the build, the two drills and the system-design reference solution from the syllabus and the research notes.
3. Validated `test_checks.py` against a private reference implementation of `checks.py`, `stats.py` (not shipped): 53/53 pass. The shipped skeleton fails 51/53 cleanly with `NotImplementedError` and passes only the two fixture-loading tests. The provided `runner.py` (`run_suite`, `write_report`, `gate`) was exercised end-to-end against the reference over the fixtures, including a NODATA case, a tagged regression and an untagged regression within allowance.
4. Rendered all four Mermaid diagrams in the reference solution with mermaid-cli and inspected them.
5. Ran an independent review (fresh reader, no access to the drafting context) against the syllabus with strict criteria: coverage, solution leakage, cross-file consistency, factual red flags with recomputation, pedagogy and time budget, diagram semantics. Findings and fixes are below.

## A. Syllabus coverage

Rating scale: **Full** = every clause of the bullet has an explicit passage with a cited source; **Partial** = covered but a clause is thin; **Missing**.

| # | Syllabus bullet | Where | Rating | Evidence / note |
|---|---|---|---|---|
| 1 | Why a non-deterministic system cannot be shipped on judgement alone | `lecture/01` §1.1–1.5 | Full | Mechanism (Thinking Machines: batch-size variation, not "GPUs are random"; 80 unique completions from 1,000 at T=0); vendor contracts (`seed` best-effort, `system_fingerprint`; 0.114 → 0.045, not 0); the sample-size argument (`0.95^10 ≈ 0.60`, `n ≥ 59`) on Yan's 5–10 % residual defect rate; Google's 16 % flakiness as the calibration; pass@k vs pass^k; τ-bench pass^8 < 25 %. |
| 2 | Eval types: unit, component, end-to-end; offline vs online; what each catches and cannot | `lecture/02` §2.1–2.6 | Full | Three levels defined and mapped to Husain's Levels 1–3, LangChain's run/trace/thread and ADK's test-file/evalset; a "catches / cannot catch" table; localization vs emergence; offline/online defined by the one clause ("doesn't have a reference output"); reference-based vs reference-free; regression vs capability. |
| 3 | Golden vs synthetic datasets | `lecture/03` §3.1–3.6 | Full | Definitions; DeepEval's own curated → production → synthetic order and "complement — not replace"; the golden-set recipe with numbers (≥30 traces to saturation, ~100/failure mode, 60 floor, 50:50 balance, 10–20/40–45/40–45 split); dimension-grid synthesis (RAGAS scenario; FAQ's 20 tuples); five unreliable-synthetic scenarios; Anthropic's "volume over quality" reconciled as a different regime. |
| 4 | Contamination of your own eval set: forty rounds of prompt tuning turns it into a training set | `lecture/04` §4.1–4.3 | Full | Tuning described as fitting; Blum & Hardt's dependence argument and the Ladder; Dwork et al.'s 50 %-reported-as-63 % at k = 500; Recht et al. with its own nuance (drift, not adaptivity); GSM1k, SWE-bench Verified, Sainz; the steel-man (Akinwande: prompts overfit slowly); criteria drift as the human channel. The syllabus phrase itself has no verbatim source and the notes say so. |
| 5 | Keeping a held-out slice you look at rarely | `lecture/04` §4.4–4.6; `cases/heldout/README.md`; `LOOKS.md` | Full | The split; the one-look rule quoted from the OpenAI/Husain cookbook; look-to-ship-not-to-tune; refresh policy (LiveBench 1/6; GSM1k rebuild); sizing with both the normal table and Wilson intervals; clustering (DROP 3.05×) and pairing; operationalized in the build with a `heldout/` directory, an assignment rule written before assigning, and a looks log. |
| 6 | Deterministic checks first — free, instant, unarguable: exact match, schema validation, tool-call correctness, citation presence | `lecture/05` §5.1–5.5 | Full | Anthropic's three-grader ranking and why "reliable" is the word; each of the four families with what it cannot do and its citable form (OpenAI `string_check`; JSON Schema 2020-12 keywords; Pydantic "guarantees the output, not the input"; BFCL's AST stages; Anthropic Citations validity-by-construction vs presence); reference-free property checks; the reproducible-but-not-unarguable caveat on BLEU/ROUGE; three habits (assert the negative, return a reason, never repair). Every family is implemented in the build. |
| 7 | Agent-specific deterministic checks: right tool, valid order, within the step budget, terminated for the right reason | `lecture/06` §6.1–6.8 | Full | Required/forbidden tool sets (Anthropic's `required:` spec; DeepEval flags; Lecture 36's blast radius); order as constraints — exact/subsequence/set ladder (ADK default 1.0; LangSmith subsequence) plus precedence, idempotence, no-cycles, terminal-only; budgets (LangGraph 1000 since v1.0.6; `MaxTurnsExceeded`) recorded as numbers; termination — the seven `stop_reason` values, loop vs model reason, the `end_turn`-empty trap, budget stop as failure, paired refusals; end state (τ-bench, SWE-bench FAIL_TO_PASS). All implemented in the build with recorded trajectories exhibiting each failure. |
| 8 | Building the case set from real failures, not from imagination | `lecture/07` §7.1–7.6 | Full | Criteria drift as the reason a priori cases fail; Rules of ML #23 and #27; open → axial coding with the Nurture Boss numbers (3 modes = 60 %; 33 % → 95 %); saturation and cadence numbers; failure → trace → check → one representative per cluster (Braintrust); SQLite's rule and 590×; closing the loop from production with the balance and representativeness guards. |
| 9 | Where pytest ends and evaluation begins, and why both gate CI | `lecture/08` §8.1–8.6 | Full | "A test asserts a property; an eval estimates a rate"; Husain's "pass rate is a product decision"; DeepEval's "Handling Flaky Test Cases" as the tell; Google's flakiness and Luo et al.; Fowler's stages; commit / merge-gate / nightly with caching's role; the gate's principles (per-case pairing, noise floor, block/warn, NODATA, seven pins, re-pin rollback, SRE before/after warning, Kayenta); a worked shape. |

| Independent work | Where | Rating | Note |
|---|---|---|---|
| Build (2 h): the deterministic half of a 40-case harness, plus the case-set structure and a held-out slice | `independent_work/build_harness/` | Full, re-scoped | 18 check functions + `normalize_text` + `run_checks` and 4 stats functions as skeletons; runner and gate **provided**; case-set structure (`working/`, `heldout/`, schema, assignment rule, looks log) with a 15-case test bed and 15 recorded trajectories; 53 model-free tests. After review the case-count deliverable this week is ≥ 15 cases about the student's agent, growing to 40 by Lecture 38's build — see §E. |
| System design drill (45 min): eval-gated CI/CD — thresholds, noise, blocking policy, rollback | `drills/system_design_eval_gated_cicd.md` + `_SOLUTION.md` | Full | Timed drill in the course's 10/30/5 format with recall questions and eight design layers, the four title words each an explicit layer; a full reference solution with four rendered Mermaid diagrams (stages; comparison; gate decision; nightly diff sequence), a worked noise-floor table, block/warn/record tables, per-surface rollback, the reviewer's PR view, and an eleven-row decisions table. Solution is intended and allowed. |
| Production drill (45 min): every bug you fix becomes an eval case before the fix merges; start with the last three | `drills/production_bugs_to_eval_cases.md` | Full | Three bugs from the git log; case authoring rules; a *proof* step running the gate with post-fix as baseline and pre-fix as candidate that must BLOCK; the three reasons it might not; reflection table; a PR-template sentence. |
| Reading (45 min): eval design, marked sections | `READING.md` top section | Full | Four sources, exact sections, times summing to 45; full annotated list by topic below it; at most three primaries per topic (counted). |

**Verdict: 9/9 theory bullets and 4/4 independent-work items covered in full.** The independent reviewer's one Partial — that the *component* eval level is defined in the notes but no check in the build is a component eval in isolation — is recorded in §E as a limitation rather than fixed: the harness's citation and tool-call checks *are* component-level when run on a single-stage recording, but the build does not ask the student to isolate a stage, and adding that would break the time budget further.

## B. Solution leakage — findings and fixes

The independent review found **no code solutions** in student-facing files for the build or the production drill: all 20 bodies in `checks.py` and 4 in `stats.py` are `raise NotImplementedError`; the adapter is a stub. Three places were judged "at the line, not over it" and kept deliberately:

| Finding | Decision |
|---|---|
| `checks.py` docstring gives a worked reason-string example (`"schema: $.refund required 'amount'"`). | Kept — the tests assert `"amount" in r.reason`, so the format must be specified. Specifying an output format is not giving away the algorithm. |
| `ASSIGNMENT.md` §1 names the `jsonschema.Draft202012Validator(...).iter_errors` API. | Kept — one line, the library is pinned, and the learning is in the two-layer parse/schema design and the reason strings, not in finding the API. |
| `ASSIGNMENT.md` §2 tells the student to work out on paper what `window=1` and `window=2` catch on two named fixtures "the tests encode the answer." | Kept — it points at the test rather than stating the answer. |

One structural leak was flagged: lecture part 8 §8.4 gave the shape of the system-design answer in the notes the student reads before the drill. **Fixed**: the framing sentence now says §8.4 gives the principles and the drill asks for the mechanism; the specifics (numbers, per-surface rollback, reviewer view, decisions) live only in the reference solution.

## C. Cross-file consistency — findings and fixes

The reviewer checked ~40 `part N §N.M` cross-references and found them all correct. The inconsistencies found were structural, and all were fixed:

| Finding | Fix |
|---|---|
| **The build's headline deliverable was impossible.** `test_checks.py` keys 24 tests to the fifteen template cases, while `ASSIGNMENT.md` told the student to delete or move them and then demanded 51/51 green. | Re-framed: the fifteen cases and their trajectories are the **test bed** and stay in place; the student's own cases live in `evals/cases/` in their repo; the runner takes `--cases` and `--root`. Stated in `ASSIGNMENT.md`, `cases/README.md` and the test file's docstring. |
| Check-function count given as "20" in two places; 18 return a `CheckResult`, plus `normalize_text` and `run_checks`. | Fixed to "18 checks + `normalize_text` + `run_checks`" everywhere. |
| `stats.py` said "three functions"; there are four. | Fixed. |
| `checks.py` pointed to an `ASSIGNMENT.md §"Conventions"` that did not exist; only four of six conventions appeared in the assignment prose. | Added a `## Conventions` section restating C1–C6. |
| Lecture 05 said the build implements BFCL's "four stages"; `tool_call_valid` implements three (name, required, type). | Lecture 05, C4 and a new self-check question now say three, with the fourth (*values*) asserted through `expect` (end state, exact answer). |
| **The production drill's proof command could not block**: it passed `--block-tags incident` while cases carry `incident-<date>`, and the gate's tag semantics were unspecified. | The provided `gate` does prefix matching (`incident` matches `incident-2026-08-14`); the CLI default is `incident`; the drill's command is annotated. Tested. |
| `runner.py` usage examples omitted the required `run` subcommand. | Fixed; the runner is now provided code and its examples were re-run. |
| `README.md` listed `COVERAGE_AUDIT.md` before it existed. | This file. |
| `README.md` said all 15 template cases have recorded trajectories; 14 do (`cap_013` deliberately has none, to exercise NODATA). | Fixed. |
| The number of things to version-pin was four in one place, five in another, seven in the solution. | Seven everywhere (prompt, model ID, tool schemas, index version, decoding params, harness version, case-set sha); the drill's recall question says "the notes list seven." |
| `FinalAnswerExpectation.non_empty` was a dead field consumed by no check. | Removed from the model, the two cases that set it, the schema and the README anatomy; `termination_ok` already asserts non-empty content on done. |
| `slice` vs directory was documented but not enforced. | `load_cases` now raises on a slice/directory mismatch and on filename ≠ id. |
| `CASE_SCHEMA.json` could drift from `Case.model_json_schema()` silently. | Test added asserting equality. |
| Test-file section banners (§1–§8) did not match `ASSIGNMENT.md`'s sections (§1, §2, §2b, §3, §4, §5). | Banners renumbered to the assignment's sections. |
| Time budgets: the build's sections summed to 140 min for a "2 h" document; README's independent block summed to 255 min for "4 h." | Build re-scoped and re-budgeted to 120 min (§E). README states that the four slots sum to 4 h 15 as the syllabus's own numbers do, and that the system-design solution's 15 min comes out of the reading slot. |
| `adapter.py` had no `run` function, so `--agent harness.adapter:run` would `AttributeError`. | Stub added raising `NotImplementedError` with a pointer to the sketch. |
| `cases/heldout/README.md` lacked the "Assignment rule" heading the assignment told the student to write under; "a fifth of forty, so eight to ten." | Heading added; sizing sentence corrected (three at fifteen, eight at forty, with the 44 % and 68 % Wilson lower bounds). |
| Self-check was lopsided (8 questions on part 4, 3 on part 6 — the part the build exercises). | Two part-6 questions added; counts updated to 38 / aim for 31. |

## D. Factual claims — findings and fixes

The reviewer spot-verified five of the riskiest quantitative claims against primary sources (GSM1k's 8 % and r² = 0.36; Dwork's n, d, 50 %, > 63 % at k = 500; LangGraph's 1000-step default since v1.0.6; SWE-bench Verified's 59.4 % and date) and found all verbatim-correct, and recomputed every probability, standard error and Wilson interval in the notes, the tests and the solution (`0.95^10`, `n ≥ 59`, the five-row half-width table, Yan's ±2.4 pp, 1,785/200, 92,053/155.8, 32/40, 40/40, 0/40, 34/40, 35/40, 36/40) and found them correct. Attributions checked and correct: *CanaryRelease* to Danilo Sato, not Fowler; Miller at Anthropic; the reusable holdout's six authors; Luo, Hariri, Eloussi & Marinov.

| Flag | Resolution |
|---|---|
| "A drop from 80 % to 40 % is more than six standard errors" — true for the single-run SE, but "a drop" is a difference of two proportions, whose SE at n = 40 is 0.10 → 4σ. | Fixed to "about four standard errors on the difference of two proportions." |
| The §4.5 table used the normal approximation while the harness computes Wilson; a student would see 65–90 % where the notes said 68–92 %. | Table now carries both columns, labelled; the prose uses the Wilson figures; the caveat explains why the harness uses Wilson (40/40 gives ±0 under the normal). |
| "Test-design problems in 59.4 % of audited items" mischaracterized OpenAI's figure (59.4 % is *all* material issues; 35.5 % over-strict tests). | Fixed with the sub-figure and quote. |
| `pass^k`'s definition was quoted while `READING.md` admitted the τ-bench PDF §4 was not fetched. | The definition is now attributed to Anthropic's guide (fetched); τ-bench is credited with introducing the metric and with the pass^8 number from its abstract (fetched); `READING.md`'s note says so. |
| Two lecture footers cited sections not listed in `READING.md` (Husain's judge guide §"Step 5"; Braintrust §"Run regression evals in CI/CD"). | Added to `READING.md`. |
| The syllabus phrase "forty rounds … turns it into a training set" has no verbatim source. | Stated openly in §4.1; grounded via Blum & Hardt's dependence argument, Dwork's inflation result and Hamel's "never place dev or test examples in the judge prompt." Not attributed to anyone. |
| Specific numbers extracted from fetched sources during research (Thinking Machines' 80/78/103; OpenAI cookbook's 0.114/0.045; Husain's 60-example floor; DeepEval's synthesizer defaults; BFCL's 2,000 items and ±20 %; Google's 16 %/84 %/1.5 %; Luo's 45/20/12 %; Kayenta's 30 %/200; LiveBench's 1/6). | Extracted verbatim from the fetched text and cited to their sections; five were re-verified by the reviewer, the rest were not re-verified a second time. A reader who finds a discrepancy should trust the source and open an issue against these notes. |

## E. Pedagogy and time budget — findings and fixes

The reviewer's strongest finding: **the build as first drafted was a 4–6 hour task labelled 2 hours** — 18 checks with reason strings pinned by exact equality, a JSON-path formatter and a deep diff, four stats functions, three untested runner functions, forty hand-authored cases, and four live runs of the agent. Changes made:

- **Runner and gate are now provided**, tested (two tests exercise them over the fixtures), and documented as the thing Lecture 38's build wires into CI. The student writes checks, stats and the adapter.
- **Case count this week is ≥ 15**, including the production drill's three, with 40 as the target for the end of the week (Lecture 38's build) — which is what the production drill's rule and the solution's own §10 say is how the set grows anyway. The syllabus's "case-set structure and a held-out slice" is delivered in full; the "40" is the harness's design capacity, seeded this week.
- **Live runs cut to one** (record once, replay to confirm identity); the three-run noise-floor measurement moves to Lecture 38 where CI needs it. `stats.noise_floor` is still written now.
- **Priority order stated**: checks before cases; the three order checks (`tool_order`, `no_cycles`, `terminal_only`) named as the ones to leave for last; "if §1–§2b passes 85 minutes, stop and commit."
- **Exact reason-string formats** were already in every docstring; the Conventions section now says the tests compare several with `==`.
- **Budget**: §1 15 + §2 50 + §2b 5 + §3 20 + §4 20 + §5 10 = 120 min.
- **System-design drill**: the delta split into a 5-minute from-memory delta inside the timer and a 15-minute second delta after it, reading four named sections of the solution plus two sources; README says the 15 minutes comes from the reading slot.
- **Self-check** rebalanced toward part 6.

Accepted as the reviewer noted: the 2 h estimate remains the least-evidenced number in the folder even after re-scoping (see §F).

## F. Known limitations and open items

- **The component eval level is taught but not isolated in the build.** The checks are trajectory- and answer-level; nothing asks the student to hold the rest of the system fixed and evaluate one model-bearing stage. Lecture 38's RAG triad is where that happens in this course; a one-paragraph pointer in part 2 says so.
- **Twelve sources could not be fetched** (listed in `READING.md`): among them the *Science* publisher page for Dwork et al. (a hosted copy was read), the τ-bench PDF §4 (definition now attributed elsewhere), the OpenAI Responses API `incomplete_details.reason` enumeration (the notes say "verify against the live schema"), and the four-way `agentevals` trajectory taxonomy (only *subsequence* is claimed).
- **`stop_reason` values and framework defaults are dated.** The seven Anthropic values, LangGraph's 1000-step default and the 4.6-generation model-ID semantics were correct on 2026-09-10 and are flagged as version-dependent in the notes.
- **The build's fixtures are synthetic by construction** — hand-authored trajectories exhibiting each failure, over a five-tool support agent that is not the student's. The notes are explicit that they are a test bed for the check code, not evidence about any agent; the student's cases come from their own traces.
- **Exact-equality reason strings** make the tests unforgiving of spacing. This is deliberate (a reason string is an interface the PR comment depends on) but it costs time; the docstrings give every format.
- **The production drill's proof requires a runnable pre-fix commit** of the student's agent. The drill gives a fallback (revert the fix in a scratch branch), but if the agent's dependencies have moved the drill gets harder; that is real-world friction, not a materials defect.
- **Self-graded formats** (spoken drill, timer, one-look rule) rely on the student's honesty, as the course's other labs do.
- **Not covered, and not in the syllabus**: LLM-as-judge design and validation (38); statistical tests for A/B comparison beyond pairing (39); online/canary mechanics beyond their place in the pipeline (40); dashboards and drift tracking (42).

## G. What I would check next time

Have a second reviewer attempt the build cold with only `ASSIGNMENT.md`, the skeleton and the tests, and time it. The 2 h estimate — even after cutting the runner, the case count and the live runs — is the least-evidenced number in the folder. If it runs long, the next cut is to move `tool_order`, `no_cycles` and `terminal_only` into Lecture 38's build and reduce the test file accordingly.
