# Self-check — Lecture 37

Answer from memory, after the lecture and the reading, before the build. No answers are given here; every question is answered in a specific part of the notes, and the part number is the only hint. If you cannot answer 31 of the 38 without opening the notes, reread the parts you missed before starting the harness — the build assumes them.

## Part 1 — Why judgement cannot ship

1. Temperature 0 does not make an LLM API deterministic. What is the actual mechanism, in one sentence, and why does it mean the output you saw depends on traffic you cannot observe?
2. A system has a 5 % defect rate. You inspect ten outputs. What is the probability you see zero defects? How many outputs would you need to inspect to have a 95 % chance of seeing at least one?
3. Google reports that 16 % of its tests are flaky. What did Google do about it, and why is that machinery the *starting point* for an LLM system rather than a remedy?
4. Define `pass@k` and `pass^k`. Which one is the reliability number, and which one would a vendor prefer to quote?

## Part 2 — Eval types

5. Name the three levels (unit / component / end-to-end). For each, name one failure it catches that the other two cannot.
6. What is the single most informative combination of results across the three levels, and what does it tell you?
7. Offline versus online: what is the one clause that defines the difference, and which family of checks does it rule out online?
8. Reference-based versus reference-free checks: which can run in production, and why does that matter for how you organize the harness?
9. Regression evals versus capability evals: what is the target pass rate for each, and which one gates a merge?

## Part 3 — Golden versus synthetic

10. State the three-source priority order for building an eval dataset, and the sentence from a synthetic-data vendor's own docs that justifies putting synthesis last.
11. How many traces should you read yourself before automating anything, and what is the stopping rule?
12. Why does a labelled set need to be roughly balanced between passes and fails? What does a 95 %-pass set fail to teach you?
13. Describe the dimension-grid method for synthetic generation in three steps, and name the step most teams skip.
14. Name three situations in which synthetic cases are unreliable, and explain why a synthetic case should start in the capability set rather than the regression set.

## Part 4 — Contamination and the held-out slice

15. Explain, using the vocabulary of fitting, why forty accepted prompt revisions on one case set make that set's score a training score.
16. Dwork et al.'s experiment: what were `n` and `d`, what was the true accuracy, and what did naive holdout reuse report after 500 selections?
17. What is the Ladder policy, and what does it become when applied to accepting a prompt change?
18. Recht et al. found 11–14 point drops on a rebuilt ImageNet test set. What did *they* conclude the cause was, and why does that give you two separate problems to guard against?
19. At `n = 40` and an observed pass rate of 80 %, what is the approximate 95 % half-width? What can a slice of that size detect, and what can it not?
20. State the one-look rule for the test slice, in the words of the OpenAI cookbook, and say what the *next action* is when the held-out slice disagrees with the working set.
21. What is clustering in Miller's sense, why does it matter for a RAG suite built over eight documents, and what was the DROP inflation factor?
22. What is criteria drift, and what habit prevents it from contaminating the held-out slice's checks?

## Part 5 — Deterministic checks

23. Anthropic ranks code-based, LLM-based and human grading. Quote or paraphrase the one-line trade-off for each. Why is "reliable" the key word for code-based?
24. Name the four families of deterministic check and, for each, one thing it *cannot* tell you.
25. BFCL's AST evaluation has four stages. Name them in order and say why they should be kept separate in your harness.
26. A vendor guarantees citation validity by construction. Which half of the citation check is still yours?
27. Why must a check never repair the output it is checking?

## Part 6 — Agent-specific checks

28. Anthropic warns that checking agents "followed specific steps" is brittle. Name three order *constraints* (not sequences) you can assert instead, and the strictness ladder the frameworks expose.
29. List Anthropic's seven `stop_reason` values. Why is `end_turn` necessary but not sufficient for "done"? Why is `max_steps` a failure even when the final message looks fine?
30. τ-bench compares end-of-conversation database state to a goal state. Why is that stronger than any check on the transcript, and what does `pass^8 < 25 %` say about how many times you should run it?
31. Your harness records `steps`, `cost_usd` and `wall_ms` as numbers even when the hard rail held. Why numbers rather than a boolean, and at what level — case or suite — should the threshold on them sit?
32. BFCL's fourth stage is *values*. Why is it not part of `tool_call_valid` in your harness, and where does a value assertion live instead?

## Parts 7–8 — Cases from failures; pytest versus evaluation

33. Open coding versus axial coding — what happens in each, and what shape did the failure distribution take in the Nurture Boss case?
34. State SQLite's regression rule in one sentence and the LLM-specific reason it is not optional ("nothing crashes").
35. "A test asserts a property; an eval estimates a rate." Give the correct pass rate for each, and the two failure modes that come from confusing them.
36. Which stage of the pipeline does each of these belong in, and why: pytest over the check code; deterministic checks over recorded trajectories; the forty cases against the live pinned agent; judge scores; the held-out slice?
37. The gate compares two runs. Why per case rather than by aggregate? What is the noise floor and how do you measure it? Name three things that should *block* and one that should only *warn*. What is `NODATA`?
38. What must be version-pinned for a baseline to be comparable at all, and what does rollback consist of?

*Thirty-eight questions; aim for 31 from memory.*
