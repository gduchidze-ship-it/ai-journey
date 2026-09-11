# The case set

One YAML file per case. `CASE_SCHEMA.json` is generated from `harness/models.py::Case` and is what
`test_case_files_validate_against_schema` checks every file against — a malformed case is a test
failure, not a skipped case.

## Layout

```
cases/
  working/     ← run on every change; used to decide what to fix
  heldout/     ← run on a schedule; NEVER used to choose between two prompts (see heldout/README.md)
  CASE_SCHEMA.json
```

A case's `slice` field must agree with the directory it lives in, and its filename must equal its `id` — `load_cases` raises otherwise. Moving a case between slices is
a deliberate act (part 4 §4.4): promote held-out cases into `working/` only when you mint fresh
held-out ones to replace them, and record the move in the case's `origin.note`.

## The fifteen shipped cases are the *test bed*, not the deliverable

Thirteen are in `working/`, two in `heldout/`. Every one has a recorded trajectory in
`fixtures/trajectories/` except `cap_013`, which is there to make the runner report `NODATA`
correctly. `test_checks.py` runs your checks against them, so **leave them in place**. Your own cases
are about your agent (Lectures 34–36), live in your repo under `evals/cases/`, and their
trajectories come from your adapter, not from `fixtures/`. Copy this file and `heldout/README.md`
there as the structure.

Read the shipped cases in this order: `refund_001_happy_path` (the reference path), then
`refund_002`–`refund_004` (three real-failure variations of it), then `inject_005` and `inject_006`
(a paired refusal / non-refusal), then `kb_007`–`kb_009` (citations and the empty `end_turn`),
then `schema_010`–`schema_011`, `cycle_012`, and finally `cap_013` and the two held-out ones.

## Anatomy of a case

```yaml
id: refund_002_precedence            # stable, unique, filename == id
title: Refund processed before identity was verified
slice: working                        # working | heldout — must match directory
kind: regression                      # regression (gate at ~100 %) | capability (track, don't gate)
tags: [refund, regression, tool_order, incident-2026-08-14]   # incident-* tags are what the gate blocks on
origin:
  type: production                    # production | handwritten | synthetic
  trace_ref: trace://prod/2026-08-14/7f3a
  date: '2026-08-14'
  note: one sentence on what happened and why this case exists
input:
  user_message: ...                   # what the adapter passes to your agent
  fixture: fixtures/end_states/acct_42_before.json     # starting environment state, if any
  injected_text: ...                  # optional, for injection cases
expect:                               # every field optional; unset = not checked
  required_tools: [...]
  forbidden_tools: [...]
  precedence: [[before, after], ...]
  idempotent_tools: [...]
  terminal_tool: final_tool_name
  order: subsequence                  # exact | subsequence | set
  expected_tool_sequence: [...]
  max_steps: 8                        # hard rail
  expected_steps: 4                   # soft: steps <= expected_steps + step_slack
  step_slack: 2
  max_cost_usd: 0.10
  max_wall_ms: 30000
  allowed_termination: [final_answer] # loop reasons that count as legitimate for THIS case
  require_content_on_done: true
  final_answer:
    exact: ...
    normalized: ...
    contains_all: [...]
    json_schema: {...}
  citations:
    required: true
    must_include: [chunk_7]
    all_valid: true
  end_state: fixtures/end_states/acct_42_after_refund.json
trials: 1                             # >1 for cases you have seen flake (pass^k)
pair_of: inject_006_benign_pair       # the near-neighbour that guards against over-correction
```

## Rules for adding a case

1. **It came from somewhere.** `origin.type` is honest. `production` cases carry a `trace_ref`.
   `synthetic` cases start with `kind: capability` until a human has read the input and vouched for
   the expectation — then, and only then, flip them to `regression` and note who promoted them.
2. **Write the check before you run the case** (part 4 §4.6). If you revise an expectation after
   seeing the output, say so in `origin.note`.
3. **One representative per failure cluster** (part 7 §7.3), not every instance. Put the other
   trace refs in the note.
4. **Refusal cases come in pairs.** Set `pair_of` both ways.
5. **Tags are the taxonomy.** Use the failure-mode names from your error analysis. `incident-<date>`
   for anything that was a production bug. The gate's `--block-tags` reads these.
6. **A fixed bug gets a case before the fix merges** (part 7 §7.4; the production drill). The case
   must fail on the pre-fix commit and pass on the post-fix one. Record both shas in the note.
