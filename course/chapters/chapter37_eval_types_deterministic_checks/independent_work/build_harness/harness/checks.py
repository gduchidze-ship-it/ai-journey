"""Deterministic checks. THIS IS THE FILE YOU WRITE.

Every function below has its signature, docstring and conventions fixed; every body is
`raise NotImplementedError`. `test_checks.py` exercises them against the recorded
trajectories in fixtures/. Nothing here may call a model, read the network, or mutate the
trajectory it is given.

Conventions (the tests assume them — read ASSIGNMENT.md → "Conventions" first):

  C1  A check returns a CheckResult. On pass, `reason == ""`. On fail, `reason` names the
      offending step index / tool / key / chunk id — specific enough that a reader could
      find it in the trajectory without re-running anything.
  C2  A check never repairs. A fenced ```json block is a schema failure, not a parse target.
  C3  `normalize_text`: strip, casefold, collapse internal whitespace runs to one space,
      strip trailing "." — nothing else.
  C4  Tool-call validation is done in the BFCL order and STOPS at the first failing stage:
      name → required params → param types → (values are not checked here).
      Stage names in `details["stage"]`: "name", "required", "type".
  C5  Order checks read only `tool_call` steps, in step order.
  C6  A missing input (no final answer, no end_state, no tool schema) is a FAIL of the check
      that needed it, with reason starting "missing:" — never a silent pass (part 8 §8.4, NODATA).
"""
from __future__ import annotations

import json
from typing import Any

from .models import Case, CaseResult, CheckResult, Trajectory

# --------------------------------------------------------------------------- text & schema


def normalize_text(s: str) -> str:
    """Convention C3. Used by `final_answer_normalized` and by nothing else."""
    raise NotImplementedError


def final_answer_exact(traj: Trajectory, expected: str) -> CheckResult:
    """check="final_answer_exact". Exact string equality on the final step's content.
    Fails with "missing: final answer" if there is no final step."""
    raise NotImplementedError


def final_answer_normalized(traj: Trajectory, expected: str) -> CheckResult:
    """check="final_answer_normalized". Equality after normalize_text on both sides."""
    raise NotImplementedError


def final_answer_contains_all(traj: Trajectory, needles: list[str]) -> CheckResult:
    """check="final_answer_contains_all". Case-sensitive substring test for every needle.
    reason lists the missing needles."""
    raise NotImplementedError


def final_answer_schema(traj: Trajectory, schema: dict[str, Any]) -> CheckResult:
    """check="final_answer_schema". Two layers (part 5 §5.2):
       1. `content` must parse as JSON (json.loads on the raw content — C2, no fence stripping).
          Fail reason starts "parse:".
       2. The parsed object must validate against `schema` (JSON Schema 2020-12; use the
          `jsonschema` package). Fail reason starts "schema:" and names the JSON path and
          the failing keyword (e.g. "schema: $.refund required 'amount'").
    """
    raise NotImplementedError


# --------------------------------------------------------------------------- tool calls


def tool_call_valid(tool: str, args: dict[str, Any], tool_schemas: dict[str, dict[str, Any]]) -> CheckResult:
    """check="tool_call_valid". BFCL-style, convention C4.

    `tool_schemas` maps tool name → JSON Schema for its arguments (see fixtures/tool_schemas.json).
      stage "name":     tool must be a key of tool_schemas.
      stage "required": every name in schema["required"] must be present in args.
      stage "type":     every present arg whose schema declares a `type` must have that JSON type
                        (string/number/integer/boolean/array/object). integer must not be a bool;
                        number accepts int or float but not bool. Do NOT coerce ("42" is not 42).
    details = {"stage": <first failing stage or "ok">}. reason names tool + param."""
    raise NotImplementedError


def all_tool_calls_valid(traj: Trajectory, tool_schemas: dict[str, dict[str, Any]]) -> CheckResult:
    """check="all_tool_calls_valid". Runs tool_call_valid on every tool_call step; fails on the
    FIRST invalid one with reason "step <i>: <inner reason>" and details from the inner result
    plus {"step": i}."""
    raise NotImplementedError


def required_tools_called(traj: Trajectory, required: list[str]) -> CheckResult:
    """check="required_tools_called". Set semantics: every required tool appears at least once.
    reason lists the missing tools, sorted."""
    raise NotImplementedError


def forbidden_tools_absent(traj: Trajectory, forbidden: list[str]) -> CheckResult:
    """check="forbidden_tools_absent". No forbidden tool appears. reason names the first
    offending step: "step <i>: <tool>". This is the blast-radius check from Lecture 36."""
    raise NotImplementedError


# --------------------------------------------------------------------------- order (part 6 §6.3)


def tool_order(traj: Trajectory, mode: str, expected_sequence: list[str]) -> CheckResult:
    """check="tool_order". Strictness ladder:
       mode="exact":       the list of called tool names equals expected_sequence.
       mode="subsequence": expected_sequence appears in order (not necessarily contiguous)
                           within the called names.
       mode="set":         every name in expected_sequence appears (order ignored) — same as
                           required_tools_called; included so a case can pick one mode.
    An empty expected_sequence passes in every mode."""
    raise NotImplementedError


def precedence(traj: Trajectory, pairs: list[tuple[str, str]]) -> CheckResult:
    """check="precedence". For each (before, after): if `after` is ever called, at least one call
    of `before` must occur at an EARLIER step than the FIRST call of `after`. If `after` is never
    called the pair is vacuously satisfied. reason: "<after>@step <i> before any <before>"."""
    raise NotImplementedError


def idempotent(traj: Trajectory, tools: list[str]) -> CheckResult:
    """check="idempotent". For each listed tool, no two calls share identical args (compare
    canonical JSON: json.dumps(args, sort_keys=True)). reason: "<tool> called twice with same
    args at steps <i>,<j>"."""
    raise NotImplementedError


def no_cycles(traj: Trajectory, window: int = 2) -> CheckResult:
    """check="no_cycles". Over the tool_call steps as a list of (tool, canonical_args), fail if
    any contiguous window of length `window` is immediately followed by an identical window
    (i.e. the same `window` calls repeated back-to-back). reason: "window of <window> repeats at
    step <i>". A window of 1 therefore catches any immediate identical repeat."""
    raise NotImplementedError


def terminal_only(traj: Trajectory, tool: str) -> CheckResult:
    """check="terminal_only". If `tool` is called at all: it is called exactly once and it is the
    LAST tool_call step. reason: "<tool> at step <i> is not last (last is step <j>)" or
    "<tool> called <n> times"."""
    raise NotImplementedError


# --------------------------------------------------------------------------- budget (part 6 §6.4)


def within_budget(traj: Trajectory, max_steps: int | None, expected_steps: int | None, step_slack: int,
                  max_cost_usd: float | None, max_wall_ms: int | None) -> CheckResult:
    """check="within_budget". Fails on the first violated bound, in this order:
       totals.steps > max_steps                      → "steps <n> > max_steps <m>"
       totals.steps > expected_steps + step_slack    → "steps <n> > expected <e>+<slack>"
       totals.cost_usd > max_cost_usd                → "cost <c> > <max>"
       totals.wall_ms > max_wall_ms                  → "wall_ms <w> > <max>"
    None bounds are skipped. details always carries the four totals so the runner can
    aggregate them into distributions."""
    raise NotImplementedError


# --------------------------------------------------------------------------- termination (part 6 §6.5)


def termination_ok(traj: Trajectory, allowed: list[str], require_content_on_done: bool) -> CheckResult:
    """check="termination_ok".
       1. termination.loop_reason must be in `allowed`; else reason "loop_reason <r> not in <allowed>".
       2. If loop_reason == "final_answer": there must be a final step (else "missing: final answer"),
          its model stop_reason must not be one of {"max_tokens", "model_context_window_exceeded"}
          (else "final answer truncated: <stop_reason>"), and if require_content_on_done its content
          must be non-empty after strip (else "empty content on done" — the end_turn trap).
       3. If loop_reason == "refusal": the final step's stop_reason must be "refusal" OR its content
          must be non-empty (a written decline); else "refusal without content"."""
    raise NotImplementedError


# --------------------------------------------------------------------------- citations (part 5 §5.2)


def citations_present(traj: Trajectory, must_include: list[str]) -> CheckResult:
    """check="citations_present". The final step has ≥1 citation, and every id in must_include
    is among them. reason: "no citations" or "missing: <ids>"."""
    raise NotImplementedError


def citations_valid(traj: Trajectory) -> CheckResult:
    """check="citations_valid". Every cited id on the final step is in traj.context_chunks.
    reason: "invalid: <ids>" (sorted). No final step or no citations → pass (presence is a
    different check)."""
    raise NotImplementedError


# --------------------------------------------------------------------------- end state (part 6 §6.6)


def end_state_matches(traj: Trajectory, expected_state: dict[str, Any] | None) -> CheckResult:
    """check="end_state_matches". Deep-equality of traj.end_state against expected_state.
    Missing observed state → "missing: end_state"; missing expected → "missing: expected end_state".
    On mismatch, reason names the FIRST differing path in sorted-key order, e.g.
    "$.accounts.42.balance: 120.0 != 60.0" (observed != expected) or "$.x: missing in observed"."""
    raise NotImplementedError


# --------------------------------------------------------------------------- assemble


def run_checks(case: Case, traj: Trajectory, tool_schemas: dict[str, dict[str, Any]],
               expected_end_state: dict[str, Any] | None) -> CaseResult:
    """Apply every check that the case's `expect` block enables, in this fixed order, and return
    a CaseResult. A check is enabled when its expectation is set / non-empty:

      all_tool_calls_valid      (always, when the trajectory has any tool calls)
      required_tools_called     (expect.required_tools non-empty)
      forbidden_tools_absent    (expect.forbidden_tools non-empty)
      tool_order                (expect.expected_tool_sequence non-empty)
      precedence                (expect.precedence non-empty)
      idempotent                (expect.idempotent_tools non-empty)
      no_cycles                 (always, window=2)
      terminal_only             (expect.terminal_tool set)
      within_budget             (always)
      termination_ok            (always)
      final_answer_exact / _normalized / _contains_all / _schema   (each when set)
      citations_present         (expect.citations.required or must_include non-empty)
      citations_valid           (expect.citations.all_valid)
      end_state_matches         (expect.end_state set)

    passed = all checks passed. Do not short-circuit: run every enabled check so the report
    shows the full failure set for the case."""
    raise NotImplementedError
