# Held-out slice — read this before you open any file in this directory

This directory is the part of the case set you **do not tune against** (lecture part 4). Its only
job is to tell you, occasionally, whether the pass rate you see on `working/` is real or is the
result of forty rounds of fitting the prompt to those forty cases.

## The rules

1. **Do not run this slice to decide between two prompts.** Run it when you are about to ship, or on
   the nightly schedule, never in the edit–run–edit loop. The OpenAI/Husain cookbook's phrasing for
   the equivalent judge test set: "run the judge on this set one time."
2. **Do not read its outputs while writing or revising checks.** Criteria drift (Shankar et al.) is
   the leak that runs through you; if you have seen the output, the expectation you write matches
   the output rather than the requirement. Write expectations for held-out cases from the
   requirement alone, and freeze them.
3. **Count the looks.** Keep `LOOKS.md` in this directory: one line per time the slice was run with
   a human reading the per-case results, with the date, the working-set rate, the held-out rate,
   and what you did about it. When you *act* on a look — promote cases, fix something the slice
   revealed — the slice is a little less held out, and that is the moment to refresh it.
4. **Refresh, don't patch.** When you act on a look: move the cases you acted on into `working/`,
   mint at least as many new held-out cases from the *next* error-analysis cycle before anyone
   tunes on them, and note the swap in each case's `origin.note`. LiveBench's policy is the model —
   a fraction rotates every period; a fraction is always unseen.
5. **Size it honestly.** At the size you will have (roughly a fifth of the set — three cases at
   fifteen, eight at forty) the slice can catch a *collapse*, not a *drift*: eight cases at 100 % have a
   Wilson 95 % interval reaching down to about 68 %; three cases at 100 % reach down to 44 %. That is its job at this size. Say so in the report
   rather than quoting the point estimate alone. Your `stats.pass_rate` prints the interval for a
   reason.

## Assignment rule

*Write yours here, before assigning any case:* ……

## What goes here

Cases that are *representative* of the working set — same failure modes, same distribution of
kinds — not the hardest cases and not the easiest. If the held-out slice is systematically harder
than the working set, a lower rate tells you nothing about overfitting (Recht et al.'s finding on
the rebuilt ImageNet test set is precisely this confound). Assign cases to the slice by a rule
decided *before* looking at them: every fifth new case from error analysis, or a hash of the id.

## What is here now

Two template cases (`heldout_h01_refund_variant`, `heldout_h02_refusal`), each with one recorded
trajectory in `fixtures/`, so that `test_checks.py` can exercise the slice loader. They are
examples of the *shape*. Replace them with cases about your agent, assigned by the rule above.

`LOOKS.md` starts empty:

```
# Held-out looks
| date | who | working rate (n) | held-out rate (n) | action |
|---|---|---|---|---|
```
