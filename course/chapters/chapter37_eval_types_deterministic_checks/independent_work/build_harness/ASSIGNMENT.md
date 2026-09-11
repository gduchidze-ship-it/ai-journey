# Build — the deterministic half of a 40-case harness (2 h)

**Deliverable.** Three things land in your course repo under `evals/` (your template already has the
directory):

1. **The harness**: this folder, copied to `evals/harness_build/` with `harness/checks.py` and
   `harness/stats.py` filled in, passing `pytest test_checks.py` — all 53, every run. The fifteen
   template cases and their recorded trajectories **stay in place**: they are the test bed the tests
   run your checks against, not your case set.
2. **The case-set structure, seeded**: `evals/cases/working/` and `evals/cases/heldout/`, with the
   assignment rule written down *before* you assign anything, and **at least fifteen cases about your
   agent** this week — the three from the production drill among them. The set grows to forty by the
   end of Lecture 38's build (which adds the judge and CI); the production drill's rule grows it one
   bug at a time after that. What matters this week is that case sixteen costs five minutes.
3. **One replay report** over your own cases: `evals/adapter.py` wrapping your Lecture 35 agent,
   trajectories recorded once with `--record`, and `python -m harness.runner run --trajectories …`
   producing `evals/reports/replay.json` + `.md`. Plus a short `NOTES.md` (four answers, §5).

**Rules.** No model call anywhere in `harness/`. `pydantic`, `pyyaml`, `jsonschema`, `numpy`,
`pytest` are the dependencies (`requirements.txt`); nothing else. The checks read a recorded
`Trajectory` and never mutate it. See **Conventions** below — the tests assume them.

**Time.** §1–§2b (the checks) is 70 minutes; if it passes 85, stop, commit what is green, and write
down where the time went — the case set shrinks before the checks do. §3 (cases) 20 minutes. §4
(stats, adapter, recording) 20 minutes. §5 (look at it) 10 minutes. The runner and the gate are
**provided** this week (`harness/runner.py`); Lecture 38's build wires them into CI and extends them.

**What the judge is not.** Nothing here scores whether an answer is *good*. That is Lecture 38.
Everything here scores whether an answer, and the trajectory that produced it, is *well-formed*:
right tools, valid order, within budget, stopped legitimately, parseable, cited, and left the world
in the right state. Part 5 §5.1 of the notes says why that half comes first.

---

## What is in this folder

| Path | Status | What |
|---|---|---|
| `harness/models.py` | provided | `Case`, `Trajectory`, `CheckResult`, `CaseResult` — read once |
| `harness/checks.py` | **you write** | 18 checks + `normalize_text` + `run_checks`; signatures and docstrings fixed |
| `harness/stats.py` | **you write** | `wilson_interval`, `pass_rate`, `paired_diff`, `noise_floor` (four small functions) |
| `harness/runner.py` | provided | loaders, `run_suite`, `write_report`, `gate`, CLI (`run`, `gate`). Calls your checks and stats |
| `harness/adapter.py` | **you write** (sketch + stub provided) | wraps your Lecture 35 agent so `run --agent` works |
| `cases/` | 15 template cases, provided — **leave them** | the test bed; see `cases/README.md` and `cases/heldout/README.md` for the structure you copy |
| `fixtures/tool_schemas.json` | provided | argument schemas for the five template tools |
| `fixtures/trajectories/*.json` | provided | 15 recorded trajectories (14 of the 15 cases; `cap_013` deliberately has none) |
| `fixtures/end_states/*.json` | provided | before/after environment snapshots for the end-state check |
| `test_checks.py` | provided | 53 tests. Model-free. Must be 100 % green. |

Run `pytest -q` now. Two tests pass (the fixtures load). The rest fail with `NotImplementedError`.

## Conventions

The six conventions at the top of `checks.py`, restated. The tests pin them.

- **C1** A check returns a `CheckResult`. On pass, `reason == ""`. On fail, `reason` names the step
  index / tool / key / chunk id — specific enough to find in the trajectory without re-running.
  The exact formats are in each docstring; the tests compare several of them with `==`.
- **C2** A check never repairs. A fenced ```` ```json ```` block is a schema failure, not a parse target.
- **C3** `normalize_text`: strip, casefold, collapse whitespace runs to one space, strip trailing `.`.
- **C4** Tool-call validation runs in BFCL order and stops at the first failing stage: name →
  required params → param types. (BFCL's fourth stage, *values*, is case-specific and lives in
  `expect` — `end_state`, `exact`, `contains_all` — not in `tool_call_valid`.)
- **C5** Order checks read only `tool_call` steps, in step order.
- **C6** A missing input (no final answer, no end state, no tool schema) is a **fail** with a reason
  starting `missing:` — never a silent pass (part 8 §8.4, `NODATA`).

---

## §1 Text and schema checks — 15 min (part 5 §5.2)

In `checks.py`, in this order: `normalize_text`, `final_answer_exact`, `final_answer_normalized`,
`final_answer_contains_all`, `final_answer_schema`.

The schema check has two layers and the reason prefix tells the reader which one failed: `parse:`
when `json.loads` on the *raw* content fails (a Markdown fence is a parse failure — C2 — and
`schema_010_fenced_json` exists to make sure you do not strip it), and `schema:` when the parsed
object violates the JSON Schema, naming the path and the failing keyword. Use
`jsonschema.Draft202012Validator(schema).iter_errors(obj)` and report the first error in path order.

Tests: `test_normalize_text` through `test_final_answer_schema_structural_failure_names_keyword`.

## §2 Tool, order, budget, termination, citation and end-state checks — 50 min (parts 5–6)

**Tool calls (10 min).** `tool_call_valid` in BFCL order — name, required, type — stopping at the
first failing stage and recording it in `details["stage"]`. Types are JSON Schema types checked
against Python values *without coercion*: `"42"` is not an `integer`, `True` is not a `number`.
Then `all_tool_calls_valid`, `required_tools_called`, `forbidden_tools_absent`.

**Order (15 min).** `precedence`, `idempotent`, `tool_order` with its three modes (a strictness
ladder — part 6 §6.3), `no_cycles`, `terminal_only`. Each reads only the `tool_call` steps in step
order (C5). For `idempotent` and `no_cycles`, two calls are "the same" when their tool name and
canonical-JSON arguments (`json.dumps(args, sort_keys=True)`) are equal. Before writing `no_cycles`,
work out on paper what `window=1` and `window=2` each catch on `refund_003_r1` and `cycle_012_r1`;
the tests encode the answer. If time is short, `tool_order`, `no_cycles` and `terminal_only` are the
three to leave for last — the other checks do not depend on them, and `run_checks` can call them
once they exist.

**Budget and termination (12 min).** `within_budget` checks bounds in a fixed order and always
returns the totals in `details` (the runner aggregates them — part 6 §6.4). `termination_ok` is the
check with the most branches; read its docstring twice. The three traps it encodes are the
`end_turn`-with-empty-content case (`kb_009`), the truncated final answer, and the budget stop with a
plausible final message (`refund_004`) — each a real recorded failure, and each one a check on the
answer text alone would pass.

**Citations and end state (13 min).** `citations_present` and `citations_valid` are separate checks
on purpose (part 5 §5.2: a vendor may guarantee one and not the other). `end_state_matches` is a deep
diff that reports the *first* differing path in sorted-key order, so that two people running it get
the same reason string.

Tests: `test_tool_call_valid_stages` through `test_end_state_matches`.

## §2b Assemble — 5 min

`run_checks` applies every enabled check in the fixed order given in its docstring and does not
short-circuit. `test_check_names_are_stable` pins the order because the check name is the key of
your failure taxonomy — if it changes, every historical report stops being comparable.

Tests: `test_run_checks_*`, `test_check_names_are_stable`.

## §3 The case set — 20 min (parts 3, 4, 7)

Now your agent. Create `evals/cases/working/` and `evals/cases/heldout/` in your repo, copy
`cases/README.md` and `cases/heldout/README.md` next to them, and:

1. **Write the assignment rule first.** One line under the `## Assignment rule` heading in your copy
   of `cases/heldout/README.md`: how a new case is assigned to `working/` or `heldout/` *before*
   anyone looks at it (every fifth case; `hash(id) % 5 == 0`; whatever — but decided now). Target
   roughly four in five to working.
2. **Read at least ten of your own traces** with a spreadsheet open (part 7 §7.2). Open-ended notes,
   then cluster. Note how many you read; you will not reach saturation in the time, and the FAQ's
   thirty is the target for the production drill and Lecture 38.
3. **Write at least fifteen cases** — forty is the target for the end of the week, not for tonight.
   Every one has an honest `origin`, at least four `expect` fields set, and tags from your clusters.
   At least: the three `incident-*` cases from the production drill; the injection cases from Lecture
   36's red-team suite with `forbidden_tools` set and at least one benign `pair_of` partner; one case
   per failure cluster you found; one or two `kind: capability` cases generated from a dimension grid
   (part 3 §3.3) and *not* promoted; and at least one case with an `end_state`.
4. **Validate.** Point the loader at your directories:
   `python -c "from harness.runner import load_cases; load_cases('../../evals/cases/working'); load_cases('../../evals/cases/heldout')"`
   must run clean (slice must match the directory; filename must equal `id`).

## §4 Stats, adapter, recording — 20 min (part 4 §4.5, part 8)

1. `stats.py` — four small functions, `wilson_interval` first; your week-1 `metrics.py` is next door.
   Tests: `test_wilson_interval` through `test_noise_floor`, then the two `runner` tests, which need
   your checks *and* your stats and exercise the provided `run_suite` and `gate` over the fixtures.
2. `adapter.py` — fill in `run(case_input)` so it calls your agent and returns a `Trajectory`
   (sketch in the file). Record `model` as the pinned model id and `prompt_sha` as the hash of the
   exact prompt text.
3. Record once, replay forever:
   ```bash
   python -m harness.runner run --cases ../../evals/cases/working --agent evals.adapter:run \
       --trials 1 --record ../../evals/recordings --root ../.. \
       --pin model=<pinned id> --pin prompt=<sha> --out ../../evals/reports/live.json
   python -m harness.runner run --cases ../../evals/cases/working --trajectories ../../evals/recordings \
       --root ../.. --out ../../evals/reports/replay.json --md ../../evals/reports/replay.md
   ```
   The two summaries must be identical. Commit the recordings; they are what the commit stage
   replays (part 8 §8.3).

## §5 Then look at what you built — 10 min, not optional

`NOTES.md`, four short answers:

1. Report your working-set pass rate with its Wilson interval. What can a set of your size detect,
   and what can it not? (Part 4 §4.5.) If you ran the held-out slice — you may, once, this week —
   add the first line to your `LOOKS.md`.
2. Which check failed most often (`summary.by_check`)? Is that the failure mode your trace reading
   said was most common? If not, which of the two is wrong?
3. Pick one failing case and quote the reason string a reviewer sees. Could they find the offending
   step from the reason alone? If not, fix the reason, not the reviewer.
4. Of your cases, how many came from a real trace, how many by hand, how many synthetically? Which
   kind took longest per case? Which kind found a failure you did not know about?

(The noise-floor measurement — the unchanged agent run three times — is Lecture 38's build, where
CI needs it; `stats.noise_floor` is written now so that it is ready.)

## Self-grading rubric

| | Pass | Strong |
|---|---|---|
| Tests | 53/53 green, every run | Plus your own tests for two conventions the file does not pin (e.g. `precedence` with the same tool in both positions; `no_cycles` with `window=3`) |
| Conventions | C1–C6 followed | Every `reason` string in your replay report is specific enough to locate the step without re-running |
| Case set | ≥ 15 cases, honest origins, held-out assigned by a pre-written rule | Every failure cluster from your trace reading has a representative; every refusal case has a pair; capability cases are unpromoted synthetics |
| Recording | Live and replay summaries identical | Recordings committed; Markdown report fits on one screen |
| NOTES.md | Four answers | Each answer names the lecture section it confirms or contradicts |

**If stuck.** `tool_call_valid` — part 5 §5.2 "Tool-call correctness" gives the stage order; the
docstring gives the type rules. `no_cycles` — part 6 §6.3 "no cycles" defines the window.
`termination_ok` — part 6 §6.5 lists the traps in the same order as the docstring. Wilson — the
formula is in any statistics reference; the point is the `40/40` and `0/40` cases the tests pin.
