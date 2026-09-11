# 2 · Eval types — unit, component, end-to-end; offline versus online

*Lecture 37, part 2 of 8 · ~20 min reading*

## 2.1 Three questions, three instruments

"Does it work?" is three different questions, and each is answered by a different instrument at a different cost.

**Does this piece do its job?** — a *unit* eval: one component, one input, one check, no model call in the check itself. Does the tool-call parser reject a malformed argument? Does the citation extractor find every `[3]` in a paragraph? Does the output validator reject JSON missing a required key? These are ordinary tests and they run in milliseconds.

**Does this stage of the pipeline produce what the next stage needs?** — a *component* eval: one stage that *contains* a model call, judged in isolation with the rest of the system held fixed. Given this query, does retrieval return the chunk that contains the answer? Given this context, does the generator's answer cite only chunks that were provided? Given this user turn, does the agent pick the right tool with schema-valid arguments? Each of these has an input you control, an output you can inspect, and a check that does not require running the whole system.

**Does the system, taken whole, produce the outcome the user wanted?** — an *end-to-end* eval: the real entry point, the real tool chain, the real termination, scored on the final state. Did the refund get issued? Did the answer match the reference? Did the task finish within the step budget and stop for the right reason?

Hamel Husain's taxonomy — the one most of the industry converged on — layers a fourth question on top. His **Level 1** is "unit tests": scoped assertions that "are crucial to getting feedback quickly when iterating on your AI system," run "on every code change." **Level 2** is "human & model eval": reading full traces and, once you have labels, a calibrated LLM judge. **Level 3** is A/B testing on real traffic, which "it's okay to put … off until you are sufficiently ready." This lecture lives at Level 1 and the deterministic part of Level 2; Lecture 38 builds the judge; Lectures 39–40 are Level 3. LangChain's agent-eval framing uses the same ladder with different words: **run** ("Did the agent call the right tool at this step? Did it pass the correct arguments?"), **trace** ("final response, trajectory, state changes"), **thread** (a whole multi-turn session). Google's Agent Development Kit draws the same line with a software-testing analogy: single-interaction *test files* run fast during development, while multi-turn *evalsets* are "well-suited for integration tests" and run less often.

## 2.2 What each catches, and what it cannot

The three levels are not redundant. Each has a blind spot the others cover.

| Level | Catches | Cannot catch | Cost per case | Typical volume |
|---|---|---|---|---|
| Unit | Regressions in code you wrote: parsers, validators, schema checks, routing logic | Anything the model does — a unit test never calls the model | ms, free | Hundreds to thousands |
| Component | A stage that stopped doing its job: retrieval miss, wrong tool, schema violation, missing citation — *localized* to the stage | Failures that only emerge from the interaction of stages (right chunk retrieved, right tool called, wrong answer anyway) | One model call, cents | Tens to low hundreds per stage |
| End-to-end | Whether the outcome is right; emergent failures; termination and budget behaviour | *Why* it failed — an end-to-end failure tells you the system is wrong, not which stage | Full run, many calls, dollars per case for agents | Twenty to a few hundred |

Two asymmetries matter.

**Localization.** Only component evals tell you *where*. Anthropic's agent-eval guide observes that a transcript "is the complete record of a trial, including outputs, tool calls, reasoning, intermediate results" — the end-to-end score reads the last line; the component checks read the middle. The RAG triad in Lecture 38 (faithfulness, answer relevance, context relevance) is a component decomposition designed precisely to say "the retriever missed" versus "the generator hallucinated." Without component evals, every end-to-end failure is a debugging session. (This week's build checks whole recorded trajectories; isolating one model-bearing stage — retrieval alone, generation alone with retrieval held fixed — is what Lecture 38's RAG triad and component-level judge do. The tool-call and citation checks you write this week are the deterministic half of that when pointed at a single-stage recording.)

**Emergence.** Only end-to-end evals see failures that no stage owns. Anthropic warns against over-prescribing intermediate steps: "There is a common instinct to check agents followed specific steps. We've found this approach too rigid and results in brittle tests." Part 6 reconciles this with agent trajectory checks — the resolution is to assert *outcome-level* properties of the trajectory (a required tool was called; the step budget held; the stop reason was legitimate) rather than a fixed script.

The practical consequence: **a component eval that passes and an end-to-end eval that fails is the most informative combination you can get**, because it says the stages are individually fine and the composition is not. A unit suite that passes and a component eval that fails says the model changed behaviour with no code change — which is the case part 1 says you must expect.

## 2.3 Offline versus online

Orthogonal to *what level* is *what data*.

**Offline** evaluation runs against a dataset you control, before deployment. LangChain's definition: "Offline evals run against datasets where you control the examples." You know the input, you usually know the reference output, and you can run it a hundred times. The LangSmith concepts page lists the uses: "pre-deployment testing: Benchmarking, Regression testing, Unit testing, Backtesting." Offline is where the CI gate lives (part 8), because a gate needs a fixed set to compare against.

**Online** evaluation runs against production traffic, after deployment. "Online evals run against production traces where users supply the inputs and the system doesn't have a reference output." That last clause is the whole difference. Without a reference you cannot do exact match; you can do reference-free checks — schema validity, citation presence, latency, cost, refusal detection, policy violations — and reference-free judges. LangSmith's uses: "production monitoring: Real-time monitoring, Anomaly detection, Production feedback."

The Hamel–Shankar FAQ puts the data consequence sharply: "Test datasets for CI are small (100+ examples) and purpose-built," whereas production monitoring samples live traffic asynchronously. They are not the same eval run in two places. They are *different data with different availability of ground truth*, and LangChain's heading for the section is the right slogan: "Test and monitor need different data."

What each cannot do: offline cannot see distribution shift — your golden set is frozen while users move (part 3). Online cannot tell you whether a *proposed* change is safe, because the change is not in production yet; and it cannot give you exact-match correctness, because nobody wrote the reference. Anthropic's agent guide names the hand-off: "Production monitoring kicks in post-launch to detect distribution drift and unanticipated real-world failures." The loop closes in part 7: online catches the new failure, you add it to the offline set, offline stops it recurring.

## 2.4 Reference-based versus reference-free

One more axis, because it decides which checks you can run where. A **reference-based** check compares the output to a known-correct answer: exact match, schema match against an expected object, expected tool sequence, expected final database state. A **reference-free** check asserts a property the output must have regardless of the specific answer: it parses as JSON; it contains no internal identifiers; every citation points to a provided chunk; the step count is under budget; the stop reason is `end_turn` with non-empty content.

Reference-based checks are stronger and only exist offline. Reference-free checks are weaker and are the *only* deterministic checks you can run online. The deterministic half of the harness you build this week should be split along that line deliberately, because the reference-free half is also what you will point at production in Lecture 42 — Hamel notes that "unlike typical unit tests, you want to organize these assertions for use in places beyond unit tests," such as automatic retries and data cleaning.

## 2.5 Capability evals versus regression evals

Anthropic's agent guide adds a distinction that matters for the gate. "Capability or 'quality' evals should start at a low pass rate, targeting tasks the agent struggles with." "Regression evals ask, 'Does the agent still handle all the tasks it used to?' and should have nearly 100 % pass rate." These are the same mechanics with opposite targets. A regression suite is a gate: a drop is a bug. A capability suite is a compass: a rise is progress, and a 40 % pass rate is fine as long as it was 35 % last month. The Hamel–Shankar FAQ warns from the other side that "a 70 % pass rate might indicate a more meaningful evaluation" than one at 100 % — a suite that always passes has stopped measuring. Your forty cases this week are the seed of the regression suite; keep a separate, harder list of things you *wish* it could do, and do not gate on that one.

## 2.6 What to carry forward

Unit evals test your code and never call the model; component evals test one model-bearing stage in isolation and are the only level that localizes a failure; end-to-end evals test the outcome and are the only level that sees emergent failures. Offline runs on data you control with references you wrote and is where the CI gate lives; online runs on traffic you do not control without references and can only use reference-free checks. Split your deterministic checks into reference-based (offline only) and reference-free (offline and online), and split your case set into regression (gate at ~100 %) and capability (track, do not gate).

---

**Sources for this part** (exact sections in `READING.md`): Hamel Husain, "Your AI Product Needs Evals," §"The Types Of Evaluation" (Levels 1–3) · LangChain, "Evaluating AI Agents at the Run, Trace, and Thread Level," §"Three levels of agent evaluation," §"Offline and online evaluation" · LangChain, "LLM Evals: The Feedback Loop Behind Reliable AI Agents," §"Test and monitor need different data" · LangSmith docs, "Evaluation concepts," §"Offline and online evaluations," §"Reference-free vs reference-based evaluators," §"Quick reference" · Google ADK, "Evaluate," §"Why evaluate agents" · Anthropic Engineering, "Demystifying evals for AI agents," §"The structure of an evaluation," §"Capability vs. regression evals," §"How evals fit with other methods" · Hamel Husain & Shreya Shankar, "AI Evals FAQ," §"Production & Deployment."
