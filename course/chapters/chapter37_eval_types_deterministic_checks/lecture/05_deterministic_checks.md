# 5 · Deterministic checks first — free, instant, unarguable

*Lecture 37, part 5 of 8 · ~25 min reading*

## 5.1 Why first

Anthropic's evaluation guide ranks the three ways to grade an output. **Code-based grading** is "fastest and most reliable, extremely scalable, but also lacks nuance for more complex judgments that require less rule-based rigidity." **LLM-based grading** is "fast and flexible, scalable and suitable for complex judgment. Test to ensure reliability first then scale." **Human grading** is "most flexible and high quality, but slow and expensive. Avoid if possible." Then the design principle that follows from it: "Prioritize volume over quality: More questions with slightly lower signal automated grading is better than fewer questions with high-quality human hand-graded evals."

The word to notice is *reliable*. A deterministic check is the only kind whose result does not itself need evaluating. When a judge says "faithful," Lecture 38 will make you prove the judge agrees with humans before you quote it. When `json.loads` raises, nobody asks whether `json.loads` is calibrated. That is what "unarguable" means: the check's verdict is not a claim about the output, it is a fact about it. It is also why deterministic checks are *free* — no tokens, no model call, no cost per case, so you can run them on every case, every commit, many times — and *instant*, so they can sit in the commit stage of the pipeline (part 8) instead of a nightly job.

The tooling has already voted. promptfoo's assertion catalogue is organised as "Deterministic eval metrics" first, then "Model-assisted eval metrics." DeepEval, a library built around LLM judges, advertises its two non-LLM metrics as the exception: tool correctness is computed "not … using any models or LLMs, and instead via exact matching," and schema validation is "free and deterministic. A model is only invoked — incurring token cost — when an output fails the schema." LangSmith's concepts page: "Code evaluators are deterministic, rule-based functions. They work well for checks such as verifying the structure of a chatbot's response is not empty, that generated code compiles, or that a classification matches exactly." Hamel Husain's Level 1 is the same thing under a different name — scoped assertions, hundreds of them at Rechat, run on every change.

The limit is stated in the same breath. Deterministic checks catch what you can *specify*. They cannot tell you an answer is wrong when it is well-formed, cites a real chunk, and calls the right tool — only that it is well-formed, cites a real chunk, and calls the right tool. So the design is layered: deterministic checks first, because anything they fail is a failure nobody will argue about and nothing further needs to run; a judge after, on the survivors, for the properties only judgement can assess.

## 5.2 The catalogue

Four families cover most of what a deterministic check can assert about an LLM output. For each: what it checks, what it cannot, and the citable form.

### Exact and normalized match

The output equals the reference. OpenAI's graders make the semantics precise: `string_check` with `eq` "returns 1 if input matches reference (case-sensitive), 0 otherwise," `like` if the input *contains* the reference, `ilike` case-insensitive. Their guidance on when: "good for scoring straightforward pass or fail answers — for example, the correct name of a city, a yes or no answer, or an answer containing or starting with the correct information." promptfoo's list adds `regex`, `starts-with`, `contains-any`, `contains-all`, `icontains`. Berkeley's function-calling leaderboard normalizes before matching — "All white space is removed" — and the same idea (strip, lowercase, collapse whitespace, canonicalize numbers) is what turns a brittle check into a fair one.

What it cannot do: judge paraphrase. An exact-match check on a free-text answer is a test of *phrasing*, and a system that is right in different words fails it. Use exact match for outputs that *should* have one form — a classification label, an extracted field, a number, an identifier, a canonical tool name — and never for prose.

A caution on "deterministic." promptfoo files BLEU, ROUGE and Levenshtein distance under deterministic metrics, with default thresholds (ROUGE-N 0.75, BLEU 0.5). They are *reproducible* — the same inputs give the same score — but they are not *unarguable*: a ROUGE of 0.6 is a number that needs interpreting, and Braintrust's own autoevals docs admit "Normalizing metrics between 0 and 1 is tough." Reproducible-and-graded is a third category between exact match and a judge; keep it out of the blocking gate unless you have measured what a 0.05 change means.

### Schema validation

The output parses, and it has the shape the next stage needs. Two layers. The **format** layer: `is-json` (promptfoo), or for a structured field, "output is valid json (optional json schema validation)." The **structural** layer: JSON Schema 2020-12's validation vocabulary gives you assertions with exact semantics — `required` ("an object instance is valid against this keyword if every item in the array is the name of a property in the instance"), `enum` ("its value is equal to one of the elements in this keyword's array value"), `const`, `maxLength`, `additionalProperties: false` ("no additional properties will be allowed"). In Python the same check is a Pydantic parse: `Model.model_validate_json(text)` either returns an instance or raises `ValidationError`, and Pydantic is explicit that this "guarantees the types and constraints of the output, not the input data" — which is why a parse is a *check*, not a *repair*, and why you should run it in strict mode ("no data conversion is performed") in an eval, so that `"42"` does not silently pass as `42`.

DeepEval's `JsonCorrectnessMetric` is the packaged form: it takes a Pydantic `BaseModel` as `expected_schema` and scores 1 if the output fits, else 0.

What it cannot do: tell you the *values* are right. `{"refund_amount": 0}` validates. And note what the vendors have done to this check: OpenAI's structured outputs with `strict: true` "ensures the model will always generate responses that adhere to your supplied JSON Schema," so "no need to validate or retry incorrectly formatted responses." When you use constrained decoding (Lecture 18), schema validity becomes a *precondition* rather than a check — keep the check anyway, as a tripwire for the day someone turns strict mode off, and move your attention to the values.

### Tool-call correctness

The model asked for the right function, with arguments that are valid against its schema. This is the check Berkeley's Function Calling Leaderboard (BFCL) decomposes most carefully. Their AST evaluation has four stages: parse the call into an abstract syntax tree; match the **function name**; match the **required parameters** ("it extracts the arguments from the AST and check if each parameter can be found and exact matched in possible answers"); match **type and value** — and "the evaluation process is strict on typing." For functions that can be run, BFCL also does *executable* evaluation: exact match of outputs, a ±20 % tolerance for real-time numeric results, and structural match on type and length; multi-call scenarios are scored "all-or-nothing."

The first three stages are what your harness's `tool_call_valid` implements (the build this week); the fourth, *values*, is task-specific and is asserted through the case's expectations — the end state, an exact answer — rather than the argument checker. The point of keeping the stages separate is *localization*: "wrong tool" and "right tool, argument of the wrong type" are different bugs with different fixes. DeepEval's `ToolCorrectnessMetric` packages the name-level check as `correctly used tools / total tools called` over `expected_tools` and `tools_called`, with two flags you will meet again in part 6 — `should_consider_ordering` and `should_exact_match`, both defaulting to `False`.

What it cannot do: tell you the tool call was *appropriate*. `get_weather("Paris")` is schema-valid and named correctly; whether the user's question needed the weather is a judgement. It also cannot see the *result* of the call — that is the job of end-state checks (part 6).

### Citation presence

For grounded generation, every claim should point at something that was actually retrieved. Two separable assertions. **Presence:** the answer contains at least one citation, or — stronger and reference-based — cites at least the chunk you know contains the answer. **Validity:** every cited identifier resolves to a chunk that was in the context window for this request. The second one is what stops the model from inventing `[7]` when only five chunks were provided.

Anthropic's Citations API shows what happens when a vendor takes over one half. Because "the API parses citations into the response formats … and extracts `cited_text` directly, citations are guaranteed to contain valid pointers to the provided documents." The structured fields are exactly what a check consumes: `type` (`char_location`, `page_location`, `content_block_location`), `document_index`, `start_char_index`/`end_char_index` or page or block ranges, plus `cited_text`. Validity is guaranteed by construction; *presence and coverage* — did each claim get a citation, did the answer cite the chunk that mattered — are still yours to assert. In a hand-rolled RAG pipeline, neither is guaranteed and both must be checked.

What it cannot do: tell you the cited chunk *supports* the claim. That is faithfulness, and it is Lecture 38's judge.

## 5.3 Reference-free property checks

Alongside those four, a class of checks that need no reference and therefore run online as well as offline (part 2 §2.4). promptfoo's catalogue is a good list to steal from: `latency` ("below a threshold (milliseconds)"), `cost` ("below a threshold"), `is-refusal` ("output indicates the model refused to perform the task"), `contains-sql`/`is-sql`, `contains-html`. Add the ones specific to your system: no internal identifiers in user-facing text (Rechat's regex check, in Hamel's case study), no email or phone patterns where PII is forbidden (Lecture 41), response length bounds, language detection, a banned-phrase list. Each of these is a one-line function, each is a fact, and each doubles as a production guardrail.

## 5.4 Writing a check that means something

Three habits separate a useful check from a green light.

**Assert the negative too.** A check that a refusal case *was* refused is half of the pair; the other half is that a benign near-neighbour was *not*. Without both, the trivially safe system — refuse everything — passes. Hamel's judge guide insists on it for judges ("report True Positive and True Negative Rate separately") and the same logic holds for a deterministic suite: build cases in pairs.

**Return a reason, not a boolean.** `False` tells you a case failed. `("schema", "required key 'amount' missing at $.refund")` tells you which check failed and where. The build this week specifies a result object with a `reason` field for exactly this; it is what makes the CI report readable and the failure taxonomy (part 7) possible.

**Do not repair in the check.** If the model returned JSON wrapped in a Markdown fence, the temptation is to strip the fence and parse. Do that in the *application*, if you choose to, and record that you had to; in the *check*, the fence is a schema failure. Otherwise the check silently drifts toward accepting whatever the model does, which is the criteria drift of part 4.6 implemented in code.

## 5.5 What to carry forward

Deterministic checks are the only grader whose verdict is a fact rather than a claim, which makes them free, instant, and unarguable — and therefore the first layer, run on every case on every change. Four families do most of the work: exact/normalized match (for outputs with one correct form); schema validation (parse plus JSON Schema or a strict Pydantic model, as a check and not a repair); tool-call correctness (BFCL's four stages — name, required params, types, values — kept separate for localization); and citation presence and validity (presence is yours even when a vendor guarantees validity). Reference-free property checks — latency, cost, refusal, PII, identifiers — run online too. Every one of them tells you the output is *well-formed*; none tells you it is *right*; that is the judge's job, on the survivors.

---

**Sources for this part** (exact sections in `READING.md`): Anthropic, "Define success criteria and build evaluations," §"Grade your evaluations," §"Eval design principles" · promptfoo docs, "Assertions & metrics," §"Deterministic eval metrics" · DeepEval docs, "Json Correctness," "Tool Correctness," §"How Is It Calculated?" · LangSmith docs, "Evaluation concepts," §"Evaluation techniques → Code" · Hamel Husain, "Your AI Product Needs Evals," §"Level 1: Unit Tests" · OpenAI, "Graders," §"String check grader"; "Structured Outputs," §"Some benefits" · JSON Schema Validation 2020-12, §6.1.2 `enum`, §6.1.3 `const`, §6.5.3 `required`; "Understanding JSON Schema → object," `additionalProperties` · Pydantic docs, "Models," §"Validation — a deliberate misnomer," §"Validating data," strict mode · Gorilla, "Berkeley Function Calling Leaderboard," §"Evaluation Metrics" (AST and executable) · Anthropic, "Citations," §"How citations work," §"Response structure" · Braintrust, "autoevals," §"Heuristic evaluations" · Hamel Husain, "Using LLM-as-a-Judge," §"Step 5."
