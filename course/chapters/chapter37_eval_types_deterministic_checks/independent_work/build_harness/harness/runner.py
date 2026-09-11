"""Harness runner — PROVIDED. Loads cases and trajectories, applies your checks, writes a report,
and gates one report against another.

You do not need to edit this file this week. It exists so that the production drill can run today
and so that Lecture 38's build has something to wire into CI. It calls your `checks.run_checks` and
your `stats` functions, so it does nothing useful until those exist. Read `run_suite` and `gate`
once — the gate's block/warn rules are the ones from lecture part 8 §8.4, and the system design
drill asks you to design them before you read how this file does it.

    python -m harness.runner run --cases cases/working --trajectories fixtures/trajectories \
        --tool-schemas fixtures/tool_schemas.json --out reports/replay.json --md reports/replay.md

    python -m harness.runner run --cases ../../evals/cases/working --agent evals.adapter:run \
        --trials 1 --out reports/live.json

    python -m harness.runner gate --baseline reports/main.json --candidate reports/pr.json \
        --block-tags incident --max-regressions 0

Two ways of getting trajectories:
  * replay — read recorded trajectories from a directory. Calls no model. Commit stage (part 8 §8.3).
  * live   — import `module:function`, call it with case.input (plus case_id), get back a Trajectory.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

from . import checks, stats
from .models import Case, CaseResult, Trajectory

ROOT = Path(__file__).resolve().parent.parent   # build_harness/


# --------------------------------------------------------------------------- loaders


def load_cases(cases_dir: Path) -> list[Case]:
    """Load every *.yaml under cases_dir (non-recursive), validate against Case, sort by id.
    A file that fails validation raises — a malformed case is a test failure, not a skipped case.
    A case whose `slice` disagrees with the directory it lives in (working/ or heldout/) raises too."""
    cases_dir = Path(cases_dir)
    cases: list[Case] = []
    for p in sorted(cases_dir.glob("*.yaml")):
        with p.open() as f:
            data = yaml.safe_load(f)
        c = Case.model_validate(data)
        if cases_dir.name in ("working", "heldout") and c.slice != cases_dir.name:
            raise ValueError(f"{p.name}: slice={c.slice!r} but file is in {cases_dir.name}/")
        if p.stem != c.id:
            raise ValueError(f"{p.name}: filename must equal case id {c.id!r}")
        cases.append(c)
    ids = [c.id for c in cases]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"duplicate case ids: {dupes}")
    return sorted(cases, key=lambda c: c.id)


def load_trajectories(traj_dir: Path) -> dict[str, list[Trajectory]]:
    """Load every *.json under traj_dir. Returns case_id → list of trajectories (one per trial),
    ordered by run_id."""
    out: dict[str, list[Trajectory]] = {}
    for p in sorted(Path(traj_dir).glob("*.json")):
        with p.open() as f:
            t = Trajectory.model_validate(json.load(f))
        out.setdefault(t.case_id, []).append(t)
    for v in out.values():
        v.sort(key=lambda t: t.run_id)
    return out


def load_tool_schemas(path: Path) -> dict[str, dict[str, Any]]:
    with Path(path).open() as f:
        return json.load(f)


def load_end_state(rel_path: str | None, root: Path = ROOT) -> dict[str, Any] | None:
    if rel_path is None:
        return None
    p = root / rel_path
    if not p.exists():
        return None            # the check reports "missing: expected end_state" — convention C6
    with p.open() as f:
        return json.load(f)


def import_agent(spec: str) -> Callable[[dict[str, Any]], Trajectory]:
    """'package.module:function' → callable(case_input) -> Trajectory."""
    mod_name, _, fn_name = spec.partition(":")
    mod = __import__(mod_name, fromlist=[fn_name])
    return getattr(mod, fn_name)


# --------------------------------------------------------------------------- run


def _percentile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, round(q * (len(xs) - 1))))
    return xs[k]


def run_suite(cases: Iterable[Case], get_trajectories: Callable[[Case], list[Trajectory]],
              tool_schemas: dict[str, dict[str, Any]], root: Path = ROOT,
              pins: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run every case; pass^k semantics per case (ALL trials must pass). Cases with zero
    trajectories go to summary.nodata and are excluded from n_cases — missing is not passing."""
    report_cases: dict[str, Any] = {}
    nodata: list[str] = []
    by_check: dict[str, int] = {}
    by_tag: dict[str, dict[str, int]] = {}
    steps_all: list[float] = []
    cost_all: list[float] = []

    for case in cases:
        trajs = get_trajectories(case)
        if not trajs:
            nodata.append(case.id)
            continue
        expected_end = load_end_state(case.expect.end_state, root)
        results: list[CaseResult] = [checks.run_checks(case, t, tool_schemas, expected_end) for t in trajs]
        failed: dict[str, int] = {}
        for r in results:
            for c in r.failed_checks():
                failed[c.check] = failed.get(c.check, 0) + 1
                by_check[c.check] = by_check.get(c.check, 0) + 1
        passed_trials = sum(r.passed for r in results)
        passed = passed_trials == len(results)
        totals = {
            "steps": [t.totals.steps for t in trajs],
            "cost_usd": [t.totals.cost_usd for t in trajs],
            "wall_ms": [t.totals.wall_ms for t in trajs],
        }
        steps_all += totals["steps"]
        cost_all += totals["cost_usd"]
        for tag in case.tags:
            by_tag.setdefault(tag, {"n": 0, "passed": 0})
            by_tag[tag]["n"] += 1
            by_tag[tag]["passed"] += int(passed)
        report_cases[case.id] = {
            "kind": case.kind, "tags": case.tags, "slice": case.slice,
            "trials": len(results), "passed_trials": passed_trials,
            "passed": passed, "pass_at_1": passed_trials > 0,
            "failed_checks": failed, "totals": totals,
            "results": [r.model_dump() for r in results],
        }

    n = len(report_cases)
    n_passed = sum(1 for v in report_cases.values() if v["passed"])
    rate = stats.pass_rate(n_passed, n)
    summary = {
        "n_cases": n, "passed_cases": n_passed, "rate": rate["rate"], "ci95": list(rate["ci95"]),
        "half_width": rate["half_width"],
        "by_tag": by_tag, "by_check": dict(sorted(by_check.items(), key=lambda kv: -kv[1])),
        "steps_p50": _percentile(steps_all, 0.5), "steps_p95": _percentile(steps_all, 0.95),
        "cost_p50": _percentile(cost_all, 0.5), "cost_p95": _percentile(cost_all, 0.95),
        "nodata": nodata,
    }
    return {"pins": pins or {}, "cases": report_cases, "summary": summary}


def write_report(report: dict[str, Any], out_json: Path, out_md: Path | None = None) -> None:
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n")
    if out_md is None:
        return
    s = report["summary"]
    lo, hi = s["ci95"]
    lines = [
        "# Eval report",
        "",
        f"**{s['passed_cases']}/{s['n_cases']} cases pass ({100*s['rate']:.1f} %, 95 % CI {100*lo:.1f}–{100*hi:.1f})** · "
        f"steps p50/p95 {s['steps_p50']}/{s['steps_p95']} · cost p95 ${s['cost_p95']}",
        "",
    ]
    if report.get("pins"):
        lines += ["Pins: " + " · ".join(f"{k}={v}" for k, v in report["pins"].items()), ""]
    if s["nodata"]:
        lines += [f"**NODATA** (no trajectory — not counted as pass): {', '.join(s['nodata'])}", ""]
    if s["by_check"]:
        lines += ["Failures by check: " + ", ".join(f"{k} ×{v}" for k, v in s["by_check"].items()), ""]
    failing = [(cid, v) for cid, v in report["cases"].items() if not v["passed"]]
    if failing:
        lines += ["| case | tags | trials passed | failed checks | first reason |", "|---|---|---|---|---|"]
        for cid, v in failing:
            first = next((c["reason"] for r in v["results"] for c in r["checks"] if not c["passed"]), "")
            lines.append(f"| {cid} | {', '.join(v['tags'])} | {v['passed_trials']}/{v['trials']} | "
                         f"{', '.join(v['failed_checks'])} | {first} |")
    else:
        lines.append("No failing cases.")
    Path(out_md).write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- gate


def _has_block_tag(tags: list[str], block_tags: list[str]) -> bool:
    """Prefix match: `incident` blocks on `incident-2026-08-14`. Exact names match themselves."""
    return any(t == b or t.startswith(b + "-") or t.startswith(b + ":") for t in tags for b in block_tags)


def gate(baseline: dict[str, Any], candidate: dict[str, Any], max_regressions: int,
         block_tags: list[str], min_rate: float | None) -> tuple[bool, list[str]]:
    """Decide block/allow from two reports (part 8 §8.4). Returns (ok, reasons).

    BLOCK when any of:
      * NODATA: a case with a baseline result has no candidate result — missing is not passing
      * a pass→fail flip on a case carrying a block tag (prefix match, e.g. `incident`)
      * more than `max_regressions` pass→fail flips overall (regression-kind cases only)
      * candidate rate < min_rate (if set)
    Everything else is a WARN (prefixed "warn:") and leaves ok=True.
    Every blocking reason names a case id and the checks that failed."""
    base = {cid: v["passed"] for cid, v in baseline["cases"].items() if v["kind"] == "regression"}
    cand = {cid: v["passed"] for cid, v in candidate["cases"].items() if v["kind"] == "regression"}
    diff = stats.paired_diff(base, cand)
    reasons: list[str] = []
    ok = True

    def failed_checks(cid: str) -> str:
        return ", ".join(candidate["cases"][cid]["failed_checks"]) or "?"

    nodata = [cid for cid in diff["only_in_baseline"] if cid in candidate["summary"].get("nodata", []) or cid not in candidate["cases"]]
    if nodata:
        ok = False
        reasons.append(f"BLOCK NODATA: {', '.join(nodata)} had a baseline result but no candidate trajectory")

    tagged = [cid for cid in diff["regressions"] if _has_block_tag(candidate["cases"][cid]["tags"], block_tags)]
    for cid in tagged:
        ok = False
        reasons.append(f"BLOCK {cid} [{', '.join(candidate['cases'][cid]['tags'])}]: pass→fail on {failed_checks(cid)} — a tagged case regressed")

    untagged = [cid for cid in diff["regressions"] if cid not in tagged]
    if len(untagged) > max_regressions:
        ok = False
        reasons.append("BLOCK regressions > max_regressions=%d: " % max_regressions
                       + "; ".join(f"{cid} ({failed_checks(cid)})" for cid in untagged))
    elif untagged:
        reasons.append("warn: regressions within allowance: " + "; ".join(f"{cid} ({failed_checks(cid)})" for cid in untagged))

    if min_rate is not None and candidate["summary"]["rate"] < min_rate:
        ok = False
        reasons.append(f"BLOCK rate {candidate['summary']['rate']:.3f} < min_rate {min_rate}")

    b, c = baseline["summary"], candidate["summary"]
    reasons.append(f"info: rate {b['rate']:.3f} → {c['rate']:.3f} (candidate 95 % CI {c['ci95'][0]:.3f}–{c['ci95'][1]:.3f}); "
                   f"improvements: {diff['improvements'] or '—'}; unchanged: {diff['unchanged']}")
    if b.get("cost_p95") and c.get("cost_p95") and c["cost_p95"] > 1.5 * b["cost_p95"]:
        reasons.append(f"warn: cost p95 {b['cost_p95']} → {c['cost_p95']}")
    if b.get("steps_p95") and c.get("steps_p95") and c["steps_p95"] > 1.5 * b["steps_p95"]:
        reasons.append(f"warn: steps p95 {b['steps_p95']} → {c['steps_p95']}")
    cap_flips = [cid for cid, v in candidate["cases"].items() if v["kind"] == "capability"
                 and cid in baseline["cases"] and baseline["cases"][cid]["passed"] != v["passed"]]
    if cap_flips:
        reasons.append(f"info: capability cases flipped (never gated): {', '.join(cap_flips)}")
    return ok, reasons


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="harness")
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run the suite (replay or live)")
    run.add_argument("--cases", type=Path, required=True)
    run.add_argument("--tool-schemas", type=Path, default=ROOT / "fixtures" / "tool_schemas.json")
    src = run.add_mutually_exclusive_group(required=True)
    src.add_argument("--trajectories", type=Path, help="replay recorded trajectories from this dir")
    src.add_argument("--agent", type=str, help="live: 'module:function' returning a Trajectory")
    run.add_argument("--trials", type=int, default=None, help="override case.trials (live mode)")
    run.add_argument("--record", type=Path, default=None, help="live mode: also write each trajectory here")
    run.add_argument("--root", type=Path, default=ROOT, help="base dir for relative fixture paths in cases")
    run.add_argument("--pin", action="append", default=[], help="key=value, e.g. --pin model=claude-sonnet-4-6")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--md", type=Path, default=None)

    g = sub.add_parser("gate", help="compare two reports and exit 1 to block")
    g.add_argument("--baseline", type=Path, required=True)
    g.add_argument("--candidate", type=Path, required=True)
    g.add_argument("--max-regressions", type=int, default=0)
    g.add_argument("--block-tags", type=str, default="incident", help="comma-separated; prefix match")
    g.add_argument("--min-rate", type=float, default=None)

    args = ap.parse_args(argv)

    if args.cmd == "run":
        cases = load_cases(args.cases)
        tool_schemas = load_tool_schemas(args.tool_schemas)
        pins = dict(kv.split("=", 1) for kv in args.pin)
        if args.trajectories:
            recorded = load_trajectories(args.trajectories)
            get = lambda c: recorded.get(c.id, [])                       # noqa: E731
        else:
            agent = import_agent(args.agent)

            def get(c: Case) -> list[Trajectory]:                         # live mode
                n = args.trials or c.trials
                out = []
                for k in range(n):
                    t = agent({**c.input, "case_id": c.id})
                    out.append(t)
                    if args.record:
                        args.record.mkdir(parents=True, exist_ok=True)
                        (args.record / f"{t.run_id}.json").write_text(t.model_dump_json(indent=2) + "\n")
                return out
        report = run_suite(cases, get, tool_schemas, root=args.root, pins=pins)
        write_report(report, args.out, args.md)
        print(json.dumps(report["summary"], indent=2))
        return 0

    if args.cmd == "gate":
        with args.baseline.open() as f:
            base = json.load(f)
        with args.candidate.open() as f:
            cand = json.load(f)
        ok, reasons = gate(base, cand, args.max_regressions, args.block_tags.split(","), args.min_rate)
        for r in reasons:
            print(r)
        print("ALLOW" if ok else "BLOCK")
        return 0 if ok else 1

    return 2


if __name__ == "__main__":
    sys.exit(main())
