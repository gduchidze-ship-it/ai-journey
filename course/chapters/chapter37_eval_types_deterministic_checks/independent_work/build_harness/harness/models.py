"""Data structures for the deterministic eval harness.

These are PROVIDED. They define the three objects the harness moves around:

  Case        — what we ask the system to do and what we expect (from cases/*.yaml)
  Trajectory  — what the agent actually did, recorded per step (from your Lecture 35 log,
                or from fixtures/trajectories/*.json while you build the checks)
  CheckResult — one check's verdict on one trajectory: passed, plus a machine-readable
                reason. Never a bare bool (lecture part 5 §5.4).

Nothing in this file calls a model. Read it once; the checks in checks.py consume these.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- Case


class Origin(BaseModel):
    """Where a case came from. `production` cases carry a trace reference; `synthetic`
    cases stay in the capability set until a human promotes them (part 3 §3.4)."""

    type: Literal["production", "handwritten", "synthetic"]
    trace_ref: Optional[str] = None
    date: Optional[str] = None
    note: Optional[str] = None


class FinalAnswerExpectation(BaseModel):
    """Expectations on the final answer text / object. All optional; unset = not checked."""

    exact: Optional[str] = None                      # exact match after no normalization
    normalized: Optional[str] = None                 # match after normalize_text()
    contains_all: list[str] = Field(default_factory=list)
    json_schema: Optional[dict[str, Any]] = None     # JSON Schema the parsed answer must satisfy
    # (non-empty content on a "done" is asserted by termination_ok, not here)


class CitationExpectation(BaseModel):
    required: bool = False                # at least one citation present
    must_include: list[str] = Field(default_factory=list)   # chunk ids that must be cited
    all_valid: bool = True                # every cited id must be in trajectory.context_chunks


class Expect(BaseModel):
    """Everything deterministic we assert about a trajectory for this case."""

    model_config = ConfigDict(extra="forbid")

    required_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    precedence: list[tuple[str, str]] = Field(default_factory=list)   # (before, after)
    idempotent_tools: list[str] = Field(default_factory=list)          # same args at most once
    terminal_tool: Optional[str] = None      # if set, must be the last step and appear once
    order: Literal["exact", "subsequence", "set"] = "subsequence"
    expected_tool_sequence: list[str] = Field(default_factory=list)    # used by order=exact/subsequence

    max_steps: Optional[int] = None          # hard rail — must hold
    expected_steps: Optional[int] = None     # soft — steps <= expected_steps + step_slack
    step_slack: int = 2
    max_cost_usd: Optional[float] = None
    max_wall_ms: Optional[int] = None

    allowed_termination: list[str] = Field(default_factory=lambda: ["final_answer"])
    require_content_on_done: bool = True

    final_answer: FinalAnswerExpectation = Field(default_factory=FinalAnswerExpectation)
    citations: CitationExpectation = Field(default_factory=CitationExpectation)
    end_state: Optional[str] = None          # path to expected end-state JSON, relative to build_harness/


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    slice: Literal["working", "heldout"]
    kind: Literal["regression", "capability"] = "regression"
    tags: list[str] = Field(default_factory=list)
    origin: Origin
    input: dict[str, Any]                    # user_message, fixture, injected_text, ...
    expect: Expect
    trials: int = 1                          # how many times to run for pass^k
    pair_of: Optional[str] = None            # id of the paired near-neighbour case (part 5 §5.4)


# --------------------------------------------------------------------------- Trajectory


class ToolCallStep(BaseModel):
    i: int
    type: Literal["tool_call"] = "tool_call"
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    error: Optional[str] = None
    stop_reason: Optional[str] = None        # the MODEL's stop reason at this step (e.g. "tool_use")
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    ms: int = 0


class FinalAnswerStep(BaseModel):
    i: int
    type: Literal["final_answer"] = "final_answer"
    content: str = ""
    citations: list[str] = Field(default_factory=list)
    stop_reason: Optional[str] = None        # e.g. "end_turn", "max_tokens", "refusal"
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    ms: int = 0


Step = ToolCallStep | FinalAnswerStep


class Termination(BaseModel):
    """Two reasons, recorded separately (part 6 §6.5): why YOUR LOOP stopped, and the
    model's last stop_reason. They disagree in instructive ways."""

    loop_reason: str                          # final_answer | max_steps | cost_cap | timeout | cycle_detected | tool_error_terminal | refusal | ...
    model_stop_reason: Optional[str] = None


class Totals(BaseModel):
    steps: int
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    wall_ms: int = 0


class Trajectory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    run_id: str
    model: str                                # pinned model id
    prompt_sha: str
    steps: list[Step]
    termination: Termination
    totals: Totals
    context_chunks: list[str] = Field(default_factory=list)   # chunk ids that were in context
    end_state: Optional[dict[str, Any]] = None                 # observed environment state after the run

    def tool_calls(self) -> list[ToolCallStep]:
        return [s for s in self.steps if isinstance(s, ToolCallStep)]

    def final(self) -> Optional[FinalAnswerStep]:
        finals = [s for s in self.steps if isinstance(s, FinalAnswerStep)]
        return finals[-1] if finals else None


# --------------------------------------------------------------------------- Results


class CheckResult(BaseModel):
    """One check, one verdict. `check` is a stable snake_case name (it becomes the failure
    taxonomy key in the report). `reason` is empty on pass and specific on fail —
    "which tool, which step, which key" — never just "failed"."""

    check: str
    passed: bool
    reason: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class CaseResult(BaseModel):
    case_id: str
    run_id: str
    passed: bool                              # all checks passed
    checks: list[CheckResult]

    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]
