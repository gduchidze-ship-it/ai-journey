"""Adapter: wrap YOUR agent (Lecture 35/36) so the runner can call it in live mode.

The runner imports `module:function` and calls it with `case.input` (a dict). It must return a
`Trajectory`. Your Lecture 35 per-iteration log already contains everything Trajectory needs;
this file is where you translate one into the other.

Sketch — fill in for your agent:

    from harness.models import Trajectory, ToolCallStep, FinalAnswerStep, Termination, Totals

    def run(case_input: dict) -> Trajectory:
        log = my_agent.run(case_input["user_message"], fixture=case_input.get("fixture"))
        steps = []
        for i, rec in enumerate(log.iterations, start=1):
            if rec.kind == "tool":
                steps.append(ToolCallStep(i=i, tool=rec.tool, args=rec.args, result=rec.result,
                                          stop_reason=rec.model_stop_reason, tokens_in=..., ...))
            else:
                steps.append(FinalAnswerStep(i=i, content=rec.text, citations=rec.citations,
                                             stop_reason=rec.model_stop_reason, ...))
        return Trajectory(case_id=case_input["case_id"], run_id=log.run_id, model=log.model_id,
                          prompt_sha=log.prompt_sha, steps=steps,
                          termination=Termination(loop_reason=log.stop, model_stop_reason=log.last_stop_reason),
                          totals=Totals(steps=len(steps), tokens_in=..., cost_usd=..., wall_ms=...),
                          context_chunks=log.retrieved_ids, end_state=my_env.snapshot())

Record `model` as the PINNED model id and `prompt_sha` as the hash of the exact prompt text —
without both, no report is comparable to any other (part 8 §8.4).
"""

from .models import Trajectory


def run(case_input: dict) -> Trajectory:
    """Called by `harness.runner run --agent harness.adapter:run`. `case_input` is the case's `input`
    dict plus `case_id`. Replace this body with the translation sketched above."""
    raise NotImplementedError("wrap your Lecture 35 agent here — see the docstring at the top of this file")
