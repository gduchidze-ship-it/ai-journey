# Production drill — Every bug you fix becomes an eval case before the fix merges. Start with the last three. (45 min)

**The rule, from this week on.** A change to your agent that fixes an observed failure does not merge
until a case exists in `evals/cases/` that **fails on the commit before the fix and passes on the
commit after it**. This is SQLite's rule — "that bug is not considered fixed until new test cases that
would exhibit the bug have been added" — applied to a system where, as Braintrust puts it, "a
hallucinated citation, a missed retrieval, an invalid tool argument, or a format violation may not
trigger an exception." Nothing crashes, so nothing but a case will ever tell you the bug came back.

**Deliverable.** Three case files in `evals/cases/working/`, each tagged `incident-<date>` and each
carrying, in `origin.note`, the pre-fix sha, the post-fix sha, and one sentence on what the fix was.
Plus `bugs_to_cases.md` in your repo with the table from §4 filled in. The cases count toward the
forty in the build.

**Timing.** Ten minutes finding the three bugs. Twenty-five minutes writing and *proving* the cases.
Ten minutes on the reflection table.

---

## §1 Find the last three bugs you fixed (10 min)

Go to your agent's git log since Lecture 34. You are looking for commits whose message or diff says
"fix" about *behaviour*: a prompt edit because it kept doing X, a tool-description change because it
picked the wrong tool, a retry-policy change because it duplicated a side effect, a truncation
change because it ran out of context, a rail added because it looped. Lecture 36's build report —
"what happened, what caught it, what you changed," three times — is the obvious source. If you
cannot find three, use two from the report and one from any trace in your logs where you thought
"that's wrong" and moved on.

For each one, write down before you touch a case file:

| | Bug 1 | Bug 2 | Bug 3 |
|---|---|---|---|
| What the user saw | | | |
| What the trajectory actually did | | | |
| The commit that fixed it (sha) | | | |
| The commit before it (sha) | | | |
| Which *deterministic* check would have caught it — or "none: needs a judge" | | | |

If the last row says "none" for all three, pick different bugs. The point of this drill is the
deterministic half; Lecture 38's production drill takes the judged half.

## §2 Write the three cases (15 min)

Follow `cases/README.md`. For each bug:

- `origin.type: production`, with the `trace_ref` if you have one, else `handwritten` with the note
  saying "reconstructed from memory of <date>" — honesty over tidiness.
- `input` reproduces the situation: the user message, the fixture state, any injected text.
- `expect` contains the check that catches *this* failure and — this is the part people skip — the
  checks that define the *correct* behaviour on the same input. A case whose only expectation is
  "did not call `send_email`" is passed by an agent that does nothing. Add `required_tools`,
  `allowed_termination`, and a `final_answer` expectation.
- `tags` include `regression`, `incident-<date>`, and the failure-mode name from your error analysis.
- If the fix was to a retry or a loop rule, the case almost certainly wants `idempotent_tools` or a
  `no_cycles` expectation; if it was to a stop condition, `allowed_termination` and `max_steps`; if
  it was to a tool description, `precedence` or `expected_tool_sequence`.

## §3 Prove them (10 min)

This is the step that makes the case a *regression* test rather than a hope.

```bash
git stash                                    # if you have uncommitted work
git checkout <pre-fix-sha>
python -m harness.runner run --cases evals/cases/working --agent evals.adapter:run \
    --root . --trials 1 --out /tmp/prefix.json
git checkout <post-fix-sha>
python -m harness.runner run --cases evals/cases/working --agent evals.adapter:run \
    --root . --trials 1 --out /tmp/postfix.json
python -m harness.runner gate --baseline /tmp/postfix.json --candidate /tmp/prefix.json \
    --block-tags incident      # prefix match: blocks on any incident-<date> tag
```

The gate, run with the *post-fix* report as the baseline and the *pre-fix* report as the candidate,
must say `BLOCK` and name your three case ids. If it says `ALLOW`, one of three things is true: the
case does not actually exercise the bug (the input is not the triggering input); the check does not
actually detect the failure (the expectation is on the wrong thing); or the bug was non-deterministic
and did not fire this time — in which case set `trials: 3` or `5` and re-run, and note the flakiness
in the case. Record which of the three it was in `bugs_to_cases.md`. Do not merge a case you could not
make fail.

If your pre-fix commit no longer runs (dependencies moved on), revert the fix in a scratch branch
instead and use that as the pre-fix state.

## §4 Reflection table (10 min)

In `bugs_to_cases.md`:

| | Bug 1 | Bug 2 | Bug 3 |
|---|---|---|---|
| Case id | | | |
| Check that catches it | | | |
| Failed on pre-fix? (y / n / flaky, k = ) | | | |
| Passes on post-fix? | | | |
| Minutes to write the case | | | |
| Was the fix *also* covered by a pytest test on code? (y / n / not applicable) | | | |
| Did writing the case change your mind about whether the fix was complete? | | | |

Then three sentences:

1. Of the three, which would have been caught by a check you *already had* in `checks.py` if the case
   had existed at the time? What does that say about where the gap was — checks or cases?
2. How long did the three cases take in total? Compare that to how long the three bugs took to find
   and fix the first time. (Rechat's hundreds of tests were built this way, one bug at a time.)
3. Write the sentence you will put in your team's PR template. Something like: *"If this PR fixes an
   observed behaviour, link the eval case that fails without it."*

## Self-grading rubric

| | Pass | Strong |
|---|---|---|
| Cases | Three, tagged, honest origins, pre/post shas in notes | Each case also asserts the correct behaviour, not only the absence of the bug |
| Proof | Gate says BLOCK on pre-fix with all three ids | You recorded *why* any case initially failed to fail, and fixed the case rather than lowering the bar |
| Table | Filled in | Row 6 answered honestly — most behaviour fixes have no code test, and that is the point |
| Rule | PR-template sentence written | Rule adopted by your team, or the reason it was not written down |
