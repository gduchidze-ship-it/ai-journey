"""Tests for the deterministic harness. Run from build_harness/:  pytest -q

These are TESTS in the sense of lecture part 8 §8.1: they call no model, they read recorded
trajectories from fixtures/, and every one of them must pass — 100 %, every run. They check that
your CHECK CODE classifies known trajectories correctly. They say nothing about your agent.

The skeleton fails every test except the fixture-loading ones with NotImplementedError. Work
through checks.py in the order of ASSIGNMENT.md §1–§2b and watch sections go green.

The fifteen cases under cases/ and the fifteen trajectories under fixtures/ are the TEST BED for these
tests. Leave them in place; your own cases live in your repo's evals/cases/ (ASSIGNMENT §3).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness import checks, stats
from harness.models import Case, Trajectory
from harness.runner import load_cases, load_end_state, load_tool_schemas, load_trajectories

ROOT = Path(__file__).resolve().parent
CASES = {c.id: c for c in load_cases(ROOT / "cases" / "working") + load_cases(ROOT / "cases" / "heldout")}
TRAJS = load_trajectories(ROOT / "fixtures" / "trajectories")
SCHEMAS = load_tool_schemas(ROOT / "fixtures" / "tool_schemas.json")


def T(run_id: str) -> Trajectory:
    for lst in TRAJS.values():
        for t in lst:
            if t.run_id == run_id:
                return t
    raise KeyError(run_id)


# --------------------------------------------------------------------------- fixtures load (provided code)


def test_fixtures_load():
    assert len(CASES) == 15
    assert sum(len(v) for v in TRAJS.values()) == 15
    assert set(SCHEMAS) == {"verify_identity", "lookup_order", "process_refund", "search_kb", "send_email"}
    assert CASES["refund_001_happy_path"].trials == 2
    assert len(TRAJS["refund_001_happy_path"]) == 2


def test_case_files_validate_against_schema():
    # cases/CASE_SCHEMA.json is generated from harness.models.Case; it must not drift, and the YAML must satisfy it
    import jsonschema, yaml
    schema = json.loads((ROOT / "cases" / "CASE_SCHEMA.json").read_text())
    assert schema == Case.model_json_schema(), "regenerate cases/CASE_SCHEMA.json from Case.model_json_schema()"
    for p in list((ROOT / "cases" / "working").glob("*.yaml")) + list((ROOT / "cases" / "heldout").glob("*.yaml")):
        jsonschema.Draft202012Validator(schema).validate(yaml.safe_load(p.read_text()))


# --------------------------------------------------------------------------- §1 text & schema (ASSIGNMENT §1)


def test_normalize_text():
    assert checks.normalize_text("  Refunded   in FULL. ") == "refunded in full"
    assert checks.normalize_text("a\tb\n c") == "a b c"
    assert checks.normalize_text("x...") == "x"


def test_final_answer_exact_and_normalized():
    t = T("kb_007_r1")
    assert checks.final_answer_exact(t, "Delivered items can be returned within 30 days of delivery [chunk_7].").passed
    r = checks.final_answer_exact(t, "delivered items can be returned within 30 days of delivery [chunk_7]")
    assert not r.passed and r.check == "final_answer_exact"
    assert checks.final_answer_normalized(t, "  delivered items can be   returned within 30 days of delivery [chunk_7] ").passed


def test_final_answer_missing_is_a_fail_not_a_pass():
    t = T("cycle_012_r1")   # no final step
    for r in (checks.final_answer_exact(t, "x"), checks.final_answer_normalized(t, "x"),
              checks.final_answer_contains_all(t, ["x"]), checks.final_answer_schema(t, {"type": "object"})):
        assert not r.passed
        assert r.reason.startswith("missing:")


def test_final_answer_contains_all():
    t = T("inject_006_r1")
    assert checks.final_answer_contains_all(t, ["c42@example.com", "A1001"]).passed
    r = checks.final_answer_contains_all(t, ["c42@example.com", "refund"])
    assert not r.passed and "refund" in r.reason and "c42@example.com" not in r.reason


REFUND_SCHEMA = CASES["refund_001_happy_path"].expect.final_answer.json_schema


def test_final_answer_schema_pass():
    assert checks.final_answer_schema(T("refund_001_r1"), REFUND_SCHEMA).passed


def test_final_answer_schema_fenced_json_is_a_parse_failure():
    r = checks.final_answer_schema(T("schema_010_r1"), REFUND_SCHEMA)
    assert not r.passed and r.reason.startswith("parse:")


def test_final_answer_schema_structural_failure_names_keyword():
    t = T("inject_005_r1")   # final content is prose, so first make a JSON-but-wrong trajectory
    bad = t.model_copy(deep=True)
    bad.steps[-1].content = json.dumps({"status": "refunded", "order_id": "A1001", "amount": "60"})
    r = checks.final_answer_schema(bad, REFUND_SCHEMA)
    assert not r.passed and r.reason.startswith("schema:") and "amount" in r.reason
    bad.steps[-1].content = json.dumps({"status": "refunded", "order_id": "A1001", "amount": 60, "extra": 1})
    r = checks.final_answer_schema(bad, REFUND_SCHEMA)
    assert not r.passed and r.reason.startswith("schema:")
    bad.steps[-1].content = json.dumps({"status": "refunded", "order_id": "A1001"})
    r = checks.final_answer_schema(bad, REFUND_SCHEMA)
    assert not r.passed and "amount" in r.reason


# --------------------------------------------------------------------------- §2 tool calls (ASSIGNMENT §2)


def test_tool_call_valid_stages():
    ok = checks.tool_call_valid("lookup_order", {"order_id": "A1001"}, SCHEMAS)
    assert ok.passed and ok.details["stage"] == "ok"

    r = checks.tool_call_valid("cancel_order", {"order_id": "A1001"}, SCHEMAS)
    assert not r.passed and r.details["stage"] == "name"

    r = checks.tool_call_valid("process_refund", {"order_id": "A1001", "amount": 60.0}, SCHEMAS)
    assert not r.passed and r.details["stage"] == "required" and "reason" in r.reason

    r = checks.tool_call_valid("lookup_order", {"order_id": 1001}, SCHEMAS)
    assert not r.passed and r.details["stage"] == "type" and "order_id" in r.reason

    # no coercion, and bool is not a number
    assert not checks.tool_call_valid("search_kb", {"query": "q", "top_k": "3"}, SCHEMAS).passed
    assert not checks.tool_call_valid("search_kb", {"query": "q", "top_k": True}, SCHEMAS).passed
    assert not checks.tool_call_valid("process_refund", {"order_id": "A", "amount": True, "reason": "r"}, SCHEMAS).passed
    assert checks.tool_call_valid("process_refund", {"order_id": "A", "amount": 60, "reason": "r"}, SCHEMAS).passed

    # required is checked before type: a call missing a required param with a bad type elsewhere reports "required"
    r = checks.tool_call_valid("process_refund", {"order_id": 5, "amount": 60.0}, SCHEMAS)
    assert r.details["stage"] == "required"


def test_all_tool_calls_valid():
    assert checks.all_tool_calls_valid(T("refund_001_r1"), SCHEMAS).passed
    r = checks.all_tool_calls_valid(T("schema_011_r1"), SCHEMAS)
    assert not r.passed and r.reason.startswith("step 2:") and r.details["step"] == 2 and r.details["stage"] == "type"


def test_required_and_forbidden_tools():
    t = T("inject_005_r1")
    r = checks.required_tools_called(t, ["verify_identity", "lookup_order", "process_refund"])
    assert not r.passed and r.details["missing"] == ["process_refund"]
    r = checks.forbidden_tools_absent(t, ["send_email"])
    assert not r.passed and r.reason == "step 3: send_email"
    assert checks.forbidden_tools_absent(T("inject_006_r1"), ["process_refund"]).passed
    assert checks.required_tools_called(T("inject_006_r1"), ["send_email"]).passed


# --------------------------------------------------------------------------- §2 order


def test_tool_order_modes():
    t = T("heldout_h01_r1")   # verify, lookup, search, refund
    assert checks.tool_order(t, "subsequence", ["verify_identity", "process_refund"]).passed
    assert checks.tool_order(t, "subsequence", ["verify_identity", "lookup_order", "process_refund"]).passed
    assert not checks.tool_order(t, "subsequence", ["process_refund", "verify_identity"]).passed
    assert not checks.tool_order(t, "exact", ["verify_identity", "lookup_order", "process_refund"]).passed
    assert checks.tool_order(t, "exact", ["verify_identity", "lookup_order", "search_kb", "process_refund"]).passed
    assert checks.tool_order(t, "set", ["process_refund", "verify_identity"]).passed
    assert checks.tool_order(t, "exact", []).passed


def test_precedence():
    r = checks.precedence(T("refund_002_r1"), [["verify_identity", "process_refund"]])
    assert not r.passed and r.reason == "process_refund@step 2 before any verify_identity"
    assert checks.precedence(T("refund_001_r1"), [["verify_identity", "process_refund"], ["lookup_order", "process_refund"]]).passed
    # vacuous when `after` never called
    assert checks.precedence(T("kb_007_r1"), [["verify_identity", "process_refund"]]).passed


def test_idempotent():
    r = checks.idempotent(T("refund_003_r1"), ["process_refund"])
    assert not r.passed and r.details["steps"] == [3, 4]
    assert checks.idempotent(T("refund_001_r1"), ["process_refund"]).passed
    # same tool, different args → fine
    assert checks.idempotent(T("refund_004_r1"), ["lookup_order"]).passed


def test_no_cycles():
    r = checks.no_cycles(T("cycle_012_r1"), window=2)
    assert not r.passed and r.details["step"] == 3
    assert checks.no_cycles(T("cycle_012_r1"), window=1).passed          # no immediate identical repeat
    assert not checks.no_cycles(T("refund_003_r1"), window=1).passed     # steps 3,4 identical
    assert checks.no_cycles(T("refund_004_r1"), window=1).passed         # seven lookups, all different args
    assert checks.no_cycles(T("refund_001_r1"), window=2).passed


def test_terminal_only():
    assert checks.terminal_only(T("refund_001_r1"), "process_refund").passed
    assert checks.terminal_only(T("kb_007_r1"), "process_refund").passed          # never called → pass
    r = checks.terminal_only(T("refund_002_r1"), "process_refund")
    assert not r.passed and "step 2" in r.reason and "step 3" in r.reason
    r = checks.terminal_only(T("refund_003_r1"), "process_refund")
    assert not r.passed and "2 times" in r.reason


# --------------------------------------------------------------------------- §2 budget & termination


def test_within_budget():
    t = T("refund_001_r1")
    ok = checks.within_budget(t, 8, 4, 2, 0.10, None)
    assert ok.passed and ok.details["steps"] == 4
    r = checks.within_budget(T("refund_004_r1"), 8, 4, 2, None, None)
    assert not r.passed and r.reason == "steps 9 > max_steps 8"
    r = checks.within_budget(T("cycle_012_r1"), 8, 2, 2, None, None)
    assert not r.passed and r.reason == "steps 6 > expected 2+2"
    r = checks.within_budget(t, None, None, 0, 0.01, None)
    assert not r.passed and r.reason.startswith("cost")
    r = checks.within_budget(t, None, None, 0, None, 1000)
    assert not r.passed and r.reason.startswith("wall_ms")
    assert checks.within_budget(t, None, None, 0, None, None).passed


def test_termination_ok():
    assert checks.termination_ok(T("refund_001_r1"), ["final_answer"], True).passed
    r = checks.termination_ok(T("refund_004_r1"), ["final_answer"], True)
    assert not r.passed and "max_steps" in r.reason
    r = checks.termination_ok(T("kb_009_r1"), ["final_answer"], True)
    assert not r.passed and r.reason == "empty content on done"
    assert checks.termination_ok(T("kb_009_r1"), ["final_answer"], False).passed
    assert checks.termination_ok(T("heldout_h02_r1"), ["refusal", "final_answer"], True).passed
    assert not checks.termination_ok(T("heldout_h02_r1"), ["final_answer"], True).passed
    trunc = T("refund_001_r1").model_copy(deep=True)
    trunc.steps[-1].stop_reason = "max_tokens"
    r = checks.termination_ok(trunc, ["final_answer"], True)
    assert not r.passed and "truncated" in r.reason


# --------------------------------------------------------------------------- §2 citations


def test_citations():
    assert checks.citations_present(T("kb_007_r1"), ["chunk_7"]).passed
    assert checks.citations_valid(T("kb_007_r1")).passed
    r = checks.citations_valid(T("kb_008_r1"))
    assert not r.passed and r.details["invalid"] == ["chunk_12"]
    r = checks.citations_present(T("kb_008_r1"), ["chunk_7"])
    assert not r.passed and r.details["missing"] == ["chunk_7"]
    r = checks.citations_present(T("kb_009_r1"), [])
    assert not r.passed and r.reason == "no citations"
    assert checks.citations_valid(T("kb_009_r1")).passed        # nothing cited → validity vacuous
    assert checks.citations_valid(T("refund_001_r1")).passed


# --------------------------------------------------------------------------- §2 end state


def test_end_state_matches():
    after = load_end_state("fixtures/end_states/acct_42_after_refund.json")
    assert checks.end_state_matches(T("refund_001_r1"), after).passed
    r = checks.end_state_matches(T("refund_003_r1"), after)
    assert not r.passed and r.reason == "$.accounts.42.balance: 120.0 != 60.0"
    r = checks.end_state_matches(T("kb_007_r1"), after)
    assert not r.passed and r.reason == "missing: end_state"
    r = checks.end_state_matches(T("refund_001_r1"), None)
    assert not r.passed and r.reason == "missing: expected end_state"


# --------------------------------------------------------------------------- §2b assemble (ASSIGNMENT §2b)


EXPECTED_VERDICTS = {
    "refund_001_r1": True, "refund_001_r2": True,
    "refund_002_r1": False, "refund_003_r1": False, "refund_004_r1": False,
    "inject_005_r1": False, "inject_006_r1": True,
    "kb_007_r1": True, "kb_008_r1": False, "kb_009_r1": False,
    "schema_010_r1": False, "schema_011_r1": False, "cycle_012_r1": False,
    "heldout_h01_r1": True, "heldout_h02_r1": True,
}

EXPECTED_FAILED_CHECKS = {
    "refund_002_r1": {"tool_order", "precedence"},
    "refund_003_r1": {"idempotent", "end_state_matches"},
    "refund_004_r1": {"required_tools_called", "tool_order", "within_budget", "termination_ok", "end_state_matches"},
    "inject_005_r1": {"required_tools_called", "forbidden_tools_absent", "tool_order"},
    "kb_008_r1": {"citations_present", "citations_valid"},
    "kb_009_r1": {"termination_ok", "final_answer_contains_all", "citations_present"},
    "schema_010_r1": {"final_answer_schema"},
    "schema_011_r1": {"all_tool_calls_valid"},
    "cycle_012_r1": {"no_cycles", "within_budget", "termination_ok"},
}


def _run(run_id: str):
    t = T(run_id)
    c = CASES[t.case_id]
    return checks.run_checks(c, t, SCHEMAS, load_end_state(c.expect.end_state))


@pytest.mark.parametrize("run_id,expected", sorted(EXPECTED_VERDICTS.items()))
def test_run_checks_verdicts(run_id, expected):
    r = _run(run_id)
    assert r.passed is expected, [(c.check, c.reason) for c in r.failed_checks()]


@pytest.mark.parametrize("run_id,expected", sorted(EXPECTED_FAILED_CHECKS.items()))
def test_run_checks_failure_sets(run_id, expected):
    r = _run(run_id)
    assert {c.check for c in r.failed_checks()} == expected


def test_run_checks_does_not_short_circuit_and_every_fail_has_a_reason():
    r = _run("refund_004_r1")
    assert len(r.checks) >= 8
    for c in r.checks:
        assert c.reason == "" if c.passed else c.reason != ""


def test_check_names_are_stable():
    r = _run("refund_001_r1")
    assert [c.check for c in r.checks] == [
        "all_tool_calls_valid", "required_tools_called", "forbidden_tools_absent", "tool_order",
        "precedence", "idempotent", "no_cycles", "within_budget", "termination_ok",
        "final_answer_schema", "citations_valid", "end_state_matches",
    ]


# --------------------------------------------------------------------------- §4 stats (ASSIGNMENT §4)


def test_wilson_interval():
    lo, hi = stats.wilson_interval(32, 40)
    assert abs(lo - 0.6516) < 0.002 and abs(hi - 0.8945) < 0.002
    assert stats.wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = stats.wilson_interval(40, 40)
    assert hi == 1.0 and 0.90 < lo < 0.92          # normal approx would give exactly [1, 1]
    lo, hi = stats.wilson_interval(0, 40)
    assert lo == 0.0 and 0.08 < hi < 0.10


def test_pass_rate():
    p = stats.pass_rate(32, 40)
    assert p["rate"] == 0.8 and p["n"] == 40 and abs(p["half_width"] - 0.1215) < 0.002
    assert stats.pass_rate(0, 0)["rate"] == 0.0


def test_paired_diff():
    base = {"a": True, "b": True, "c": False, "d": True, "e": False}
    cand = {"a": True, "b": False, "c": True, "d": True, "f": True}
    d = stats.paired_diff(base, cand)
    assert d["n"] == 4
    assert d["regressions"] == ["b"] and d["improvements"] == ["c"]
    assert d["unchanged"] == 2 and d["net"] == 0
    assert d["only_in_baseline"] == ["e"] and d["only_in_candidate"] == ["f"]


def test_noise_floor():
    runs = [
        {"a": True, "b": True, "c": False, "d": True},
        {"a": True, "b": False, "c": False, "d": True},
        {"a": True, "b": True, "c": False, "d": False},
    ]
    nf = stats.noise_floor(runs)
    assert nf["k"] == 3 and nf["flaky_cases"] == ["b", "d"] and nf["flaky_fraction"] == 0.5
    assert nf["rates"] == [0.75, 0.5, 0.5] and abs(nf["rate_spread"] - 0.25) < 1e-9


# --------------------------------------------------------------------------- runner (provided code; needs your checks + stats)


def _replay_report():
    from harness.runner import run_suite
    cases = load_cases(ROOT / "cases" / "working")
    return run_suite(cases, lambda c: TRAJS.get(c.id, []), SCHEMAS, root=ROOT)


def test_run_suite_over_fixtures():
    rep = _replay_report()
    s = rep["summary"]
    assert s["nodata"] == ["cap_013_two_orders_one_message"]
    assert s["n_cases"] == 12 and s["passed_cases"] == 3
    assert rep["cases"]["refund_001_happy_path"]["trials"] == 2
    assert rep["cases"]["refund_001_happy_path"]["passed"] is True
    assert set(rep["cases"]) == set(CASES) - {"cap_013_two_orders_one_message", "heldout_h01_refund_variant", "heldout_h02_refusal"}
    assert s["by_check"]["termination_ok"] == 3
    assert abs(s["rate"] - 0.25) < 1e-9 and s["ci95"][0] < 0.25 < s["ci95"][1]


def test_gate_blocks_on_tagged_regression_and_nodata_only():
    import copy
    from harness.runner import gate
    base = _replay_report()
    assert gate(base, base, 0, ["incident"], None)[0] is True

    # an incident-tagged case that used to pass now fails → BLOCK naming it
    b2 = copy.deepcopy(base)
    b2["cases"]["refund_002_precedence"]["passed"] = True
    ok, reasons = gate(b2, base, 0, ["incident"], None)
    assert ok is False and any("refund_002_precedence" in r and r.startswith("BLOCK") for r in reasons)

    # a plain-regression case flips, within allowance → WARN not BLOCK
    c = copy.deepcopy(base)
    c["cases"]["kb_007_citations"]["passed"] = False
    c["cases"]["kb_007_citations"]["failed_checks"] = {"citations_valid": 1}
    ok, reasons = gate(base, c, 1, ["incident"], None)
    assert ok is True and any(r.startswith("warn:") and "kb_007_citations" in r for r in reasons)
    ok, reasons = gate(base, c, 0, ["incident"], None)
    assert ok is False

    # a case vanishes from the candidate → NODATA blocks
    c2 = copy.deepcopy(base)
    del c2["cases"]["inject_006_benign_pair"]
    c2["summary"]["nodata"].append("inject_006_benign_pair")
    ok, reasons = gate(base, c2, 5, ["incident"], None)
    assert ok is False and any("NODATA" in r and "inject_006_benign_pair" in r for r in reasons)
