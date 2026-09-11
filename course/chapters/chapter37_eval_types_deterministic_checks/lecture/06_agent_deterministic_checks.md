# 6 · Agent-specific deterministic checks — the right tool, a valid order, within budget, stopped for the right reason

*Lecture 37, part 6 of 8 · ~25 min reading*

## 6.1 An agent leaves a trail, and the trail is data

A chain produces an output. An agent produces a *trajectory* — Anthropic's definition: "the complete record of a trial, including outputs, tool calls, reasoning, intermediate results" — and then an outcome, "the final state in the environment at the end of the trial." Lecture 35 made you log every iteration of the loop: which tool, which arguments, what came back, how many tokens, how much time, and why the loop stopped. That log is not just for debugging. Every field in it is something a deterministic check can assert.

This part is the four assertions the syllabus names, and one it implies. **Did it call the right tool?** **In a valid order?** **Within the step budget?** **Did it terminate for the right reason?** And underneath all four: **did the world end up in the right state?** Each is a fact about the log, each is free, and together they catch a class of agent failure that no check on the final text can see — the answer that is correct but cost thirty steps, the refund that was processed twice, the "done" that was actually a timeout.

## 6.2 Right tool, right arguments

Part 5 gave the four-stage BFCL decomposition for a single call: name, required parameters, types, values. For an agent, apply it per step and then ask a set-level question: over the whole trajectory, was every tool the task *required* called at least once, and was any tool the task *forbade* never called?

Anthropic's agent-eval guide shows the shape of a requirement spec — `required: [{tool: verify_identity}, {tool: process_refund}]` — a *set* of tools that must appear, not a script. DeepEval's `ToolCorrectnessMetric` computes `correctly used tools / total tools called` between `expected_tools` and `tools_called`, and with `should_exact_match=False` (the default) it is a coverage measure: did the expected tools appear, ignoring extras. Extras matter separately — an agent that calls `search` eleven times before `answer` may be correct and is certainly wasteful — so record the count and check it against a budget rather than forbidding it.

The **forbidden** set is the security half of the check and comes straight from Lecture 36. If a task's input contains injected text, the deterministic assertion is that no tool from the blast-radius list (`send_email`, `delete_*`, `transfer_funds`) appears anywhere in the trajectory. This is the twenty-case red-team suite you already built, restated as a trajectory check; it belongs in the same harness.

## 6.3 A valid order

"Valid order" is where Anthropic's warning bites: "There is a common instinct to check agents followed specific steps. We've found this approach too rigid and results in brittle tests." Two agents can both be right — one looks up the account then verifies identity, the other verifies then looks up — and a check that scripts the sequence fails one of them for nothing. The resolution is to assert *constraints on* the order, not the order.

Three levels of strictness, and the frameworks expose all three:

- **Exact sequence.** Google ADK's `tool_trajectory_avg_score` does "exact matching of tool-call sequences," and its default threshold is **1.0** — perfect match required. Use it when the task genuinely has one correct procedure (a fixed compliance workflow), and expect to loosen it.
- **Subsequence.** LangSmith's worked example implements `trajectory_subsequence`: "check how many of the desired steps the agent took," in order, tolerating extra steps between them. This is usually the right default: the required tools appear in the required relative order, and the agent may do other things in between.
- **Set (unordered).** DeepEval's `should_consider_ordering=False` (the default): the required tools appear, order ignored.

And then the constraints that are neither, which you write as predicates over the log: **precedence** (`verify_identity` occurs before any call to `process_refund`); **idempotence** (`process_refund` with the same arguments occurs at most once — Lecture 35's idempotent-step requirement, checked); **no cycles** (no window of `k` consecutive steps repeats with identical arguments — the repeated-state detection from Lecture 35, now asserted rather than only enforced); **terminal-only** (`final_answer` is the last step, and nothing follows it). Each is three lines of Python over a list of `(tool, args)` tuples. Each catches a bug that has shipped somewhere.

## 6.4 Within the step budget

Lecture 36 made you enforce a step limit, a cost cap and a wall-clock timeout in the loop. Those rails stop the invoice; they do not tell you whether the agent is *efficient*. The check here is that the trajectory finished with room to spare, and the harness metrics from Lecture 35 — steps per resolved task, cost per resolved task — are the numbers to record on every case.

The frameworks give you the budget primitives to assert against. LangGraph: "The recursion limit sets the maximum number of super-steps the graph can execute during a single execution," raising `GraphRecursionError` when hit — and note that "starting in version 1.0.6, the default recursion limit is set to 1000 steps," where older material says 25; read the version you run. OpenAI's Agents SDK raises `MaxTurnsExceeded` "when the agent's run exceeds the `max_turns` limit passed to the `Runner.run`" family. Anthropic's design guidance: "it's also common to include stopping conditions (such as a maximum number of iterations) to maintain control."

Two assertions per case, and a case-level constant for each: `steps_taken ≤ step_budget` (the hard rail held), and, more useful, `steps_taken ≤ expected_steps + slack` (the agent did not solve a three-step task in nineteen). Record `steps_taken`, `tokens_in`, `tokens_out`, `cost_usd`, `wall_ms` as numbers, not booleans, so that a regression from 4 steps to 9 is visible even when both are under the rail. Then set thresholds on the *distribution* across the suite — median steps, p95 cost — because a single case's step count is noisy (part 1) and a suite-wide shift is signal.

## 6.5 Terminated for the right reason

An agent stops for one of a small number of reasons, and only some of them mean "done." The check is that the recorded reason is one of the acceptable ones *for this case*, and that it is consistent with the rest of the log.

The API gives you the field. Anthropic's Messages API `stop_reason` takes seven values: `end_turn` ("Claude finished its response naturally"), `max_tokens` ("the response reached your max_tokens limit"), `stop_sequence`, `tool_use` ("Claude is calling a tool"), `pause_turn` ("a server-tool loop reached its iteration limit"), `refusal` ("Claude declined to respond"), and `model_context_window_exceeded` ("the response filled the model's context window"). The docs draw the line you need: stop reasons describe *successful responses*; errors are *failed requests*; they are different things and your log should record both. OpenAI's Responses API exposes `status` (`completed`, `incomplete`, `in_progress`, and others) with `incomplete_details` on truncation; the Agents SDK's rule for a final output is precise and fully checkable: "it produces text output with the desired type, and there are no tool calls."

Your loop adds its own reasons on top of the model's: `final_answer` (the agent said it was done), `max_steps`, `cost_cap`, `timeout`, `cycle_detected`, `tool_error_terminal`, `human_approval_required`. Record the *loop's* reason and the *model's* last `stop_reason` separately; they disagree in instructive ways.

Then the assertions:

- **The reason is in the case's allowed set.** For a task the agent should complete: `{final_answer}`. For an injected or forbidden task: `{refusal, final_answer_with_decline}`. For a deliberately impossible task: `{final_answer_with_cannot, max_steps}` — and *not* `cost_cap`, because giving up should be a decision, not an accident.
- **A "done" is really done.** Anthropic's docs flag a trap: "Empty responses with `end_turn`" exist. So `stop_reason == "end_turn"` is necessary, not sufficient: assert non-empty content, and for an agent, assert that `final_answer` was accompanied by whatever artefact the task demanded (a parsable answer object, a written file, a state change).
- **A budget stop is a failure even when the answer looks fine.** `max_steps` with a plausible final message is the "plausible garbage" case from Lecture 36: the loop was cut off and the model produced a summary anyway. It is a fail for a task that should complete, regardless of what the text says.
- **A refusal is right or wrong, never neutral.** Build the pair (part 5 §5.4): the injected case must end in refusal; the benign near-neighbour must not.

## 6.6 The world ended up in the right state

Everything above reads the log. The strongest agent check reads the *environment*. τ-bench, the Sierra benchmark for tool-agent-user interaction, "compares the database state at the end of a conversation with the annotated goal state" — the order was cancelled, the flight was rebooked, the balance is what it should be — independently of what the agent *said*. SWE-bench's resolution criterion is the same idea for code: an instance is resolved when "all tests across FAIL_TO_PASS and PASS_TO_PASS pass" — the tests that were failing now pass, and the ones that passed still do. Anthropic's list of code-based graders for agents is exactly this family: "String match checks (exact, regex, fuzzy, etc.) • Binary tests (fail-to-pass, pass-to-pass) • Static analysis."

For your agent, this means every case that has a side effect needs a *fixture* — a starting state — and an *expected end state*, and the check diffs them. It is more work per case than a log check, and it is the one that catches the refund that was described but never issued, and the double refund that was issued twice (the log check for idempotence and the state check for the balance should agree; when they do not, one of them has a bug).

And then run it more than once. τ-bench's `pass^k` — in Anthropic's gloss, "the probability that all k trials succeed" — exists precisely because "even state-of-the-art function calling agents … succeed on <50 % of the tasks" and are "quite inconsistent (pass^8 <25 % in retail)." A deterministic end-state check is deterministic *per run*; the agent is not. The check you want in CI is the deterministic check, repeated, reported as a rate (part 8).

## 6.7 Where these checks live

All of this reads a recorded trajectory. That means the harness does not need the live agent at check time — it needs the log format from Lecture 35, and a runner that produces logs. Record once, check many ways: the same trajectory JSON is scored by the tool-set check, the order predicates, the budget assertions, the termination check, and — where there is a fixture — the end-state diff. When the judge arrives in Lecture 38 it reads the same file. The build this week gives you a handful of pre-recorded trajectories to write checks against before you point the harness at your own agent, precisely so that the check code and the agent code are debugged separately.

## 6.8 What to carry forward

An agent's trajectory and end state are data, and five deterministic assertions cover most of what goes wrong: the required tools were called and the forbidden ones were not (a set, with BFCL's name/params/types/values per call); the order satisfies the task's constraints — precedence, idempotence, no cycles, terminal-only — asserted as predicates rather than a script, at the strictness the task warrants (exact / subsequence / set); the step, token, cost and time budgets held *and* the numbers are recorded so efficiency regressions show; the termination reason is in the case's allowed set, a `done` has content and an artefact, a budget stop is a failure even with a good-looking answer, and refusals are checked in pairs; and where there is a side effect, the environment's end state matches the goal state. Repeat the run; report the rate.

---

**Sources for this part** (exact sections in `READING.md`): Anthropic Engineering, "Demystifying evals for AI agents," §"The structure of an evaluation," §"Types of graders for agents," §"How to think about non-determinism" · DeepEval docs, "Tool Correctness," parameters `should_consider_ordering`, `should_exact_match` · Google ADK, "Evaluate," §"Evaluate trajectory and tool use," `tool_trajectory_avg_score` · LangSmith docs, "Evaluate a complex agent," §"Trajectory evaluator" · LangGraph docs, "Graph API," recursion limit · OpenAI Agents SDK, "Running agents," §"The agent loop," `MaxTurnsExceeded` · Anthropic, "Building effective agents," §"Agents" · Anthropic, "Handling stop reasons," §"Stop reason values," §"Stop reasons vs. errors," §"Empty responses with end_turn" · OpenAI, Responses API object reference, `status`, `incomplete_details` · Yao et al., "τ-bench," abstract · Jimenez et al., "SWE-bench," §A.4 Evaluation Procedure · Gorilla, "Berkeley Function Calling Leaderboard," §"Evaluation Metrics."
