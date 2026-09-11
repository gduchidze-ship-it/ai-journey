# Reading — Lecture 37

Every URL below was fetched and checked on 2026-09-10; section headings are as they appeared on that date. Several vendor doc URLs redirect (Anthropic's `docs.anthropic.com` → `platform.claude.com`; OpenAI's `platform.openai.com/docs` → `developers.openai.com/api/docs`; Google ADK → `adk.dev`) — the redirect targets are given. Following the course rule, each topic has at most three sources marked *primary*; the rest are reference. Sources that could not be fetched from the authoring environment are listed at the bottom and are not load-bearing for any claim in the notes.

## The 45-minute assigned reading (Independent work → Reading: "Eval design, marked sections")

Read these, in this order, marked sections only.

1. **Anthropic Engineering — "Demystifying evals for AI agents" (Grace, Hadfield, Olivares, De Jonghe, 9 Jan 2026).** Read §"The structure of an evaluation," §"Types of graders for agents," §"Capability vs. regression evals," §"How to think about non-determinism in evaluations for agents," and §"Going from zero to one: a roadmap to great evals for agents" (all three sub-sections). Skip the per-agent-type sections. About 15 minutes.
   https://anthropic.com/engineering/demystifying-evals-for-ai-agents

2. **Hamel Husain — "Your AI Product Needs Evals" (29 Mar 2024).** Read §"Case Study: Lucy" through the end of §"Level 1: Unit Tests" (Steps 1–3), then §"Level 3: A/B Testing" (short). Skip Level 2 for now — it is Lecture 38's reading. About 12 minutes.
   https://hamel.dev/blog/posts/evals/

3. **Evan Miller (Anthropic) — "Adding Error Bars to Evals" (arXiv 2411.00640, Nov 2024).** Read §1 Introduction, §2.1 Independent questions, §2.2 Clustered questions (skim the maths, read the prose and Table 4), and §4.2 Paired analysis. Skip §3 and §5. About 12 minutes.
   https://arxiv.org/abs/2411.00640

4. **Blum & Hardt — "The Ladder: A Reliable Leaderboard for Machine Learning Competitions" (ICML 2015).** Read §1 Introduction only (two pages) and the boxed algorithm in §3. About 6 minutes.
   https://arxiv.org/pdf/1502.04585

If you have time left: SQLite, "How SQLite Is Tested," §5 Regression Testing — one paragraph, and it is the production drill's charter. https://www.sqlite.org/testing.html

---

## Full annotated source list by topic

### 1 · Non-determinism and why judgement cannot ship (part 1)

| Source | Read | Why |
|---|---|---|
| **Horace He et al., Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference" (10 Sep 2025)** — https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ | §"The original sin: floating-point non-associativity"; §"Batch invariance and 'determinism'"; §"How nondeterministic are completions?" | *Primary.* Temperature 0 is not deterministic, and the cause is batch-size variation under load, not "GPUs are random." 1,000 samples → 80 unique completions; divergence at token 103. |
| OpenAI API guide, "Advanced usage → Reproducible outputs" — https://developers.openai.com/api/docs/guides/advanced-usage | §"Reproducible outputs" | The vendor contract: `seed` is "best effort"; `system_fingerprint` changes when the backend does. |
| OpenAI Cookbook, "How to make your completions outputs consistent with the new seed parameter" (Anadkat, Nov 2023) — https://developers.openai.com/cookbook/examples/reproducible_outputs_with_the_seed_parameter | §"Model level controls"; §"Example" | Measured: distance 0.114 → 0.045 with a seed. Not zero. |
| **Eugene Yan, "Task-Specific LLM Evals that Do & Don't Work" (31 Mar 2024)** — https://eugeneyan.com/writing/evals/ | §"Nonetheless, we still need human evaluation"; §"Calibrate your evaluation bar to the level of risk" | *Primary.* The 5–10 % residual defect rate after RAG and prompt engineering — the number that makes the sample-size argument. |
| John Micco, Google Testing Blog, "Flaky Tests at Google and How We Mitigate Them" (27 May 2016) — https://testing.googleblog.com/2016/05/flaky-tests-at-google-and-how-we.html | All (short) | 16 % of tests flaky; 84 % of pass→fail transitions involve a flake; fail-3-in-a-row rule; quarantine. |
| Anthropic, "Model IDs and versions" — https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions | §"Dateless IDs are pinned snapshots"; §"Model weights versus serving infrastructure" | Why pinning a model ID makes a baseline comparable; note the 4.6-generation change in what a dateless ID means. |

### 2 · Eval types: unit / component / end-to-end; offline / online (part 2)

| Source | Read | Why |
|---|---|---|
| **Hamel Husain, "Your AI Product Needs Evals"** — https://hamel.dev/blog/posts/evals/ | §"The Types Of Evaluation": Level 1, Level 2 (intro only), Level 3 | *Primary.* The taxonomy the industry converged on. |
| **LangChain, "Evaluating AI Agents at the Run, Trace, and Thread Level" (23 Jun 2026)** — https://www.langchain.com/resources/agent-evals | §"Three levels of agent evaluation"; §"Offline and online evaluation" | *Primary.* Run → trace → thread; the cleanest agent version of the ladder. Vendor content, but substantive. |
| LangChain, "LLM Evals: The Feedback Loop Behind Reliable AI Agents" (10 Mar 2026) — https://www.langchain.com/resources/llm-evals | §"Test and monitor need different data"; §"Turn production failures into regression coverage" | The one-clause definition of offline vs online: production "doesn't have a reference output." |
| LangSmith docs, "Evaluation concepts" — https://docs.langchain.com/langsmith/evaluation-concepts | §"Offline and online evaluations"; §"Reference-free vs reference-based evaluators"; §"Quick reference: Offline vs online evaluation" | The quick-reference table at the bottom is the slide. |
| Google ADK, "Evaluate" — https://adk.dev/evaluate/ | §"Why evaluate agents"; test files vs evalsets | The unit-test vs integration-test analogy for agents. |
| **Anthropic Engineering, "Demystifying evals for AI agents"** — https://anthropic.com/engineering/demystifying-evals-for-ai-agents | §"Capability vs. regression evals"; §"How evals fit with other methods" | *Primary.* Regression ≈ 100 %, capability starts low; where production monitoring takes over. |
| Hamel Husain & Shreya Shankar, "AI Evals FAQ" (28 May 2025, modified 1 Sep 2026) — https://hamel.dev/blog/posts/evals-faq/ | §"Production & Deployment" | CI datasets are "small (100+ examples) and purpose-built"; monitoring samples live traffic. |

### 3 · Golden versus synthetic datasets (part 3)

| Source | Read | Why |
|---|---|---|
| **Hamel Husain & Shreya Shankar, "AI Evals FAQ"** — https://hamel.dev/blog/posts/evals-faq/ | Q "How many examples do I need for an eval?"; Q "What is the best approach for generating synthetic data?"; Q "Are there scenarios where synthetic data may not be reliable?" | *Primary.* Thirty traces to saturation; the dimension-tuple method; the five unreliable-synthetic scenarios. |
| **DeepEval docs, "Synthetic data generation — Introduction"** — https://deepeval.com/docs/synthetic-data-generation-introduction · "Generate from documents" — https://deepeval.com/docs/synthesizer-generate-from-docs | §"Recommended Priority"; §"Best Practices On Synthetic Data Quality"; Synthesizer defaults | *Primary.* A synthesis vendor saying "complement — not replace" and "use it sparingly"; the curated → production → synthetic priority order. |
| **Eugene Yan, "Product Evals in Three Simple Steps" (23 Nov 2025)** — https://eugeneyan.com/writing/product-evals/ | §"First, label some data"; §"Finally, run our eval harness with each change" | *Primary.* 50–100 failures out of 200+; synthetic defects are "out-of-distribution"; the n = 200 → ±2.4 pp interval. |
| Eugene Yan, "An LLM-as-Judge Won't Save The Product — Fixing Your Process Will" (20 Apr 2025) — https://eugeneyan.com/writing/eval-process/ | §"Building product evals is simply the scientific method in disguise" | 50:50 pass/fail target; evals as practice, not artefact. |
| Hamel Husain, "Using LLM-as-a-Judge For Evaluation: A Complete Guide" (29 Oct 2024) — https://hamel.dev/blog/posts/llm-judge/ | §"Step 2: Create a Dataset"; §"Step 5: Build Your LLM as A Judge, Iteratively" (TPR/TNR reporting) | ~100 per failure mode, 60 as the floor; the 10–20 / 40–45 / 40–45 split; "Never place dev or test examples in the judge prompt." |
| RAGAS docs, "Test data generation" — https://docs.ragas.io/en/stable/concepts/test_data_generation/ and …/rag/ | §"Characteristics of an Ideal Test Dataset"; §"Scenario Generation"; §"Query Synthesizer" | Scenario = nodes × query length × query style × persona; single- vs multi-hop. |
| LangSmith docs, "Evaluation concepts" — https://docs.langchain.com/langsmith/evaluation-concepts | §"Building datasets" (manually curated / historical traces / synthetic) | 10–20 examples to start; 5–10 per critical component. |
| OpenAI, "Evaluation best practices" — https://developers.openai.com/api/docs/guides/evaluation-best-practices | §"Design your eval process"; §"Handle edge cases" | Production data plus domain-expert references; "Biased design" as an anti-pattern. |
| Anthropic, "Define success criteria and build evaluations" — https://platform.claude.com/docs/en/test-and-evaluate/develop-tests | §"Eval design principles" | "Prioritize volume over quality" — and why it is about capability evals, not product evals. |
| Anthropic, "Evaluate prompts in the developer console" (9 Jul 2024) — https://claude.com/blog/evaluate-prompts | All (short) | Auto-generated test cases as a product feature; 5-point expert grading. |
| Shankar, Zamfirescu-Pereira, Hartmann, Parameswaran & Arawjo, "Who Validates the Validators?" (UIST 2024) — https://arxiv.org/abs/2404.12272 | Abstract | "LLM-generated evaluators simply inherit all the problems of the LLMs they evaluate." |

### 4 · Contamination and the held-out slice (part 4)

| Source | Read | Why |
|---|---|---|
| **Blum & Hardt, "The Ladder" (ICML 2015)** — https://arxiv.org/pdf/1502.04585 | §1; §3 (the algorithm); §6 (Kaggle data) | *Primary.* Repeated evaluation creates dependence; the holdout stops being unbiased; the Ladder policy. 1,785 submissions / 200 teams. |
| **Dwork, Feldman, Hardt, Pitassi, Reingold & Roth, "The reusable holdout," *Science* 349:636 (2015)** — hosted copy https://www.nematilab.info/bmijc/assets/091218_paper.pdf (publisher page not fetched) | §"The Problem of Adaptive Data Analysis"; §"The Thresholdout Algorithm"; §"Experimental Validation" | *Primary.* 50 % true accuracy reported as > 63 % after k = 500 adaptive selections. |
| **Evan Miller, "Adding Error Bars to Evals" (arXiv 2411.00640)** — https://arxiv.org/abs/2411.00640 · HTML https://arxiv.org/html/2411.00640v1 | §2.1–2.2; §3.3 "Don't touch the thermostat!"; §4.2; §5 | *Primary.* SE formulas; clustered SE (DROP 3.05×); paired comparison; n ≈ 1,000 for a 3-pp effect. |
| Recht, Roelofs, Schmidt & Shankar, "Do ImageNet Classifiers Generalize to ImageNet?" (ICML 2019) — https://arxiv.org/abs/1902.10811 | Abstract | 11–14 pp drops — and their own conclusion that adaptivity was *not* the cause. |
| Hardt, "Climbing a shaky ladder" (2017) — https://arxiv.org/abs/1706.02733 | Abstract | Follow-up to the Ladder. Reference only. |
| Zhang et al. (Scale AI), "A Careful Examination of LLM Performance on Grade School Arithmetic" (GSM1k, 2024) — https://arxiv.org/abs/2405.00332 | Abstract | Rebuild-from-recipe; up to 8 pp drops; Spearman r² = 0.36 with memorization. |
| OpenAI, "Why SWE-bench Verified no longer measures frontier coding capabilities" (23 Feb 2026) — https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/ | §"Contamination"; §"Too narrow and too wide tests" | Models reproducing the human fix verbatim — a contamination probe you can run yourself. |
| Sainz et al., "NLP Evaluation in trouble" (Findings of EMNLP 2023) — https://aclanthology.org/2023.findings-emnlp.722/ | Abstract | Definition of the worst case and its consequence. |
| Xu et al., "Benchmark Data Contamination of Large Language Models: A Survey" (2024) — https://arxiv.org/abs/2406.04244 | Abstract | Survey citation. |
| White et al., "LiveBench: A Challenging, Contamination-Limited LLM Benchmark" (ICLR 2025) — https://arxiv.org/pdf/2406.19314 | §on monthly refresh | Rotate ~1/6 per month; keep ~1/6 private. Note the title says *Limited*, not *Free*. |
| Akinwande, Jiang, Sam & Kolter, "Understanding prompt engineering may not require rethinking generalization" (2023) — https://arxiv.org/html/2310.03957 | Abstract, §1 | The steel-man: discrete prompts overfit slowly. |
| Kapse & Husain, OpenAI Cookbook, "Building resilient prompts using an evaluation flywheel" (6 Oct 2025) — https://developers.openai.com/cookbook/examples/evaluation/building_resilient_prompts_using_an_evaluation_flywheel | §"Aligning your LLM judge" (the 20/40/40 split and "run the judge on this set one time") | The one-look rule, first-party. |
| Shankar et al., "Who Validates the Validators?" — https://arxiv.org/abs/2404.12272 | Abstract | Criteria drift. |

### 5 · Deterministic checks (part 5)

| Source | Read | Why |
|---|---|---|
| **Anthropic, "Define success criteria and build evaluations"** — https://platform.claude.com/docs/en/test-and-evaluate/develop-tests | §"Grade your evaluations"; §"Eval design principles" | *Primary.* Code-based / LLM-based / human grading trade-offs; volume over quality. |
| **Gorilla / UC Berkeley, "Berkeley Function Calling Leaderboard"** — https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html | §"Evaluation Metrics" (AST evaluation and executable evaluation) | *Primary.* The four-stage decomposition: name → required params → types → values; ±20 % numeric tolerance; all-or-nothing on multi-call. |
| **promptfoo docs, "Assertions & metrics"** — https://www.promptfoo.dev/docs/configuration/expected-outputs/ | §"Deterministic eval metrics" (the whole table) | *Primary.* The catalogue to steal from; note BLEU/ROUGE filed as deterministic. |
| OpenAI, "Graders" — https://developers.openai.com/api/docs/guides/graders | §"String check grader" | `eq`/`neq`/`like`/`ilike`, returns 0 or 1; when to use. |
| OpenAI, "Structured Outputs" — https://developers.openai.com/api/docs/guides/structured-outputs | §"Some benefits"; §"Structured Outputs vs JSON mode" | Schema validity becomes a precondition under `strict: true`. |
| JSON Schema Validation 2020-12 — https://json-schema.org/draft/2020-12/json-schema-validation · "Understanding JSON Schema → object" — https://json-schema.org/understanding-json-schema/reference/object | §6.1.2 `enum`, §6.1.3 `const`, §6.3.1 `maxLength`, §6.5.3 `required`; `additionalProperties` | The normative semantics of each assertion. |
| Pydantic docs, "Models" — https://pydantic.dev/docs/validation/latest/concepts/models/ | §"Validation — a deliberate misnomer"; §"Validating data"; strict mode | "Guarantees the output, not the input" — a parse is a check, not a repair. |
| DeepEval docs, "Json Correctness" — https://deepeval.com/docs/metrics-json-correctness · "Tool Correctness" — https://docs.confident-ai.com/docs/metrics-tool-correctness | §"How Is It Calculated?" on both | The two non-LLM metrics in an LLM-judge library; `should_consider_ordering`, `should_exact_match`. |
| Anthropic, "Citations" — https://platform.claude.com/docs/en/build-with-claude/citations | §"How citations work"; §"Response structure" | Validity guaranteed by construction; presence and coverage still yours. Field names. |
| LangSmith docs, "Evaluation concepts" — https://docs.langchain.com/langsmith/evaluation-concepts | §"Evaluation techniques → Code" | One-paragraph definition of code evaluators. |
| Braintrust, "autoevals" — https://www.braintrust.dev/docs/reference/autoevals | §"Heuristic evaluations" | "Normalizing metrics between 0 and 1 is tough." |
| Hamel Husain, "Your AI Product Needs Evals" — https://hamel.dev/blog/posts/evals/ | §"Level 1: Unit Tests" | Assertions reused as guardrails; "you don't necessarily need a 100 % pass rate." |

### 6 · Agent-specific deterministic checks (part 6)

| Source | Read | Why |
|---|---|---|
| **Anthropic Engineering, "Demystifying evals for AI agents"** — https://anthropic.com/engineering/demystifying-evals-for-ai-agents | §"The structure of an evaluation"; §"Types of graders for agents"; §"How to think about non-determinism" | *Primary.* Trajectory and outcome defined; code-based graders for agents; the warning against step-scripting; pass@k / pass^k. |
| **Yao, Shinn, Razavi & Narasimhan (Sierra), "τ-bench" (2024)** — https://arxiv.org/abs/2406.12045 · repo https://github.com/sierra-research/tau-bench | Abstract; §4 for pass^k (PDF) | *Primary.* End-state comparison against a goal state; pass^8 < 25 % in retail. |
| **Anthropic, "Handling stop reasons"** — https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons | §"Stop reason values" (all seven); §"Stop reasons vs. errors"; §"Empty responses with end_turn" | *Primary.* The field your termination check reads, and the `end_turn`-with-empty-content trap. |
| Google ADK, "Evaluate" — https://adk.dev/evaluate/ | §"Evaluate trajectory and tool use" | `tool_trajectory_avg_score` exact-match, default threshold 1.0; evalset field names. |
| LangSmith docs, "Evaluate a complex agent" — https://docs.langchain.com/langsmith/evaluate-complex-agent | §"Trajectory evaluator" | The `trajectory_subsequence` pattern. |
| DeepEval docs, "Tool Correctness" — https://docs.confident-ai.com/docs/metrics-tool-correctness | Parameters | Set-level coverage; ordering and exact-match flags. |
| LangGraph docs, "Graph API" — https://docs.langchain.com/oss/python/langgraph/graph-api | Recursion limit | Default 1000 super-steps since v1.0.6 (older material says 25); `GraphRecursionError`. |
| OpenAI Agents SDK, "Running agents" — https://openai.github.io/openai-agents-python/running_agents/ | §"The agent loop" | The two-clause final-output rule; `MaxTurnsExceeded`. |
| OpenAI, Responses API object — https://developers.openai.com/api/docs/api-reference/responses/object | `status`, `incomplete_details` | The OpenAI-side termination fields. Reason enumeration not verified — check the live schema. |
| Anthropic, "Building effective agents" (19 Dec 2024) — https://www.anthropic.com/engineering/building-effective-agents | §"Agents" | Stopping conditions "such as a maximum number of iterations." |
| Jimenez et al., "SWE-bench" (ICLR 2024) — https://arxiv.org/abs/2310.06770 · PDF https://arxiv.org/pdf/2310.06770 | §A.4 Evaluation Procedure | FAIL_TO_PASS / PASS_TO_PASS as a deterministic end-state check. |
| Liu et al., "AgentBench" (ICLR 2024) — https://arxiv.org/abs/2308.03688 | Abstract | Reference only: eight environments; long-horizon failures. |
| Barres et al., "τ²-Bench" (2025) — https://arxiv.org/abs/2506.07982 | Abstract | Reference only: dual-control environments. |

### 7 · Building the case set from real failures (part 7)

| Source | Read | Why |
|---|---|---|
| **Hamel Husain, "A Field Guide to Rapidly Improving AI Products" (24 Mar 2025)** — https://hamel.dev/blog/posts/field-guide/ | §"The Most Common Mistake: Skipping Error Analysis"; §"The Error Analysis Process"; §"Bottom-Up vs. Top-Down Analysis" | *Primary.* Open coding → taxonomy; three modes = 60 % of problems; 33 % → 95 %. |
| **Hamel Husain & Shreya Shankar, "AI Evals FAQ"** — https://hamel.dev/blog/posts/evals-faq/ | §"Error Analysis & Data Collection"; §"Evaluation Design & Methodology" | *Primary.* Open/axial coding; trace counts and cadence; code-check vs judge decision rule. |
| **SQLite, "How SQLite Is Tested"** — https://www.sqlite.org/testing.html | §5 Regression Testing; §7 Test Coverage; §12 Summary | *Primary.* "Not considered fixed until new test cases … have been added." 590× test code. |
| Shankar et al., "Who Validates the Validators?" — https://arxiv.org/abs/2404.12272 | Abstract | Criteria drift — why the cases cannot be written a priori. |
| Zinkevich, Google, "Rules of Machine Learning" (updated 25 Aug 2025) — https://developers.google.com/machine-learning/guides/rules-of-ml | Rule #23; Rule #27; Rule #26 | "You are not a typical end user"; "measure first, optimize second." |
| Braintrust, "How to turn LLM production failures into regression tests" (29 May 2026) — https://www.braintrust.dev/articles/turn-llm-production-failures-into-regression-tests | §"Why LLM production failures keep recurring"; §"Capture and classify"; §"Run regression evals in CI/CD"; §"Common pitfalls" | Trace contents; one representative per cluster; schema scorer "may need a perfect score"; "may not trigger an exception." |
| Eugene Yan, "An LLM-as-Judge Won't Save The Product" — https://eugeneyan.com/writing/eval-process/ | §"scientific method" | Looking at data as the loop's first step. |
| Anthropic Engineering, "Demystifying evals for AI agents" — https://anthropic.com/engineering/demystifying-evals-for-ai-agents | §"Collect tasks for the initial eval dataset" | "20–50 simple tasks drawn from real failures." |

### 8 · pytest versus evaluation; the CI gate (part 8; system design drill)

| Source | Read | Why |
|---|---|---|
| **Warner & Davidovič, Google SRE Workbook ch. 16, "Canarying Releases"** — https://sre.google/workbook/canarying-releases/ | §"What Is Canarying?"; §"Choosing a Canary Population and Duration"; §"Selecting and Evaluating Metrics"; §"Before/After Evaluation Is Risky" | *Primary.* The production half of the gate: 5 % × 20 % = 1 %; ≤ 12 metrics; before/after is risky. |
| **Graff & Sanden, "Automated Canary Analysis at Netflix with Kayenta" (10 Apr 2018)** — https://netflixtechblog.com/automated-canary-analysis-at-netflix-with-kayenta-3260bc7acc69 | §"Judgment"; §"Reporting" | *Primary.* Per-metric Mann-Whitney U; Pass/High/Low; aggregate score; `NODATA`; automatic abort. |
| **Miller, "Adding Error Bars to Evals"** — https://arxiv.org/abs/2411.00640 | §4.2 Paired analysis; §5 Power analysis | *Primary.* Per-case paired comparison and whether your suite can detect the effect you gate on. |
| Fowler, "DeploymentPipeline" (30 May 2013) — https://www.martinfowler.com/bliki/DeploymentPipeline.html | All (short) | Stages of increasing confidence at increasing cost. No minute numbers on this page. |
| Sato, "CanaryRelease" (25 Jun 2014, on martinfowler.com) — https://martinfowler.com/bliki/CanaryRelease.html | All (short) | Rollback as re-routing to the old version. Note the author is Danilo Sato. |
| Micco, "Flaky Tests at Google" — https://testing.googleblog.com/2016/05/flaky-tests-at-google-and-how-we.html | All | The noise-floor definition: "both a passing and a failing result with the same code." |
| Luo, Hariri, Eloussi & Marinov, "An Empirical Analysis of Flaky Tests" (FSE 2014) — https://www.cs.cornell.edu/courses/cs5154/2021sp/resources/LuoETAL14FlakyTestsAnalysis.pdf | §4 root causes; §5 fixes | Causes distribution; 78 % flaky from first write. |
| DeepEval docs, "Unit Testing in CI/CD" — https://deepeval.com/docs/evaluation-unit-testing-in-ci-cd | §"How It Works"; §"Handling Flaky Test Cases"; §"YAML File For CI/CD Evals" | `assert_test` under pytest; the flaky-tests section as the tell. |
| LangSmith docs, "How to run evaluations with pytest" — https://docs.langchain.com/langsmith/pytest | §"Log feedback"; §"Caching" | Pass *rate* as feedback; cassette caching. |
| promptfoo docs, "CI/CD integration" — https://www.promptfoo.dev/docs/integrations/ci-cd/ · "GitHub Action" — https://www.promptfoo.dev/docs/integrations/github-action/ | §"Quality Gates"; §"Caching Strategies" | Exit code 1 on threshold; the 95 % pass-rate idiom; 24 h cache TTL. |
| Braintrust, "braintrust-eval" GitHub Action — https://github.com/marketplace/actions/braintrust-eval | Inputs; PR comment format | `terminate_on_failure` defaults to false; per-case improvements/regressions counts. |
| OpenAI, "Working with evals" — https://developers.openai.com/api/docs/guides/evals · "Evaluation best practices" — https://developers.openai.com/api/docs/guides/evaluation-best-practices | §"Analyze the results"; §"Use evals to improve performance" | `result_counts` as a count vector; "continuous evaluation (CE)." |
| Anthropic, "Model IDs and versions" — https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions | §"Dateless IDs are pinned snapshots" | Pinning as the precondition for a comparable baseline. |
| Blum & Hardt, "The Ladder" — https://arxiv.org/pdf/1502.04585 | §3 | The step-size policy as a gate threshold. |

### System design drill references (read *after* the drill)

Google SRE Workbook ch. 16 (§"Before/After Evaluation Is Risky," §"Selecting and Evaluating Metrics") · Kayenta blog (§"Judgment," §"Reporting") · Miller §4.2 and §5 · Braintrust Action PR-comment format · Anthropic, "Model IDs and versions." All linked above.

---

## Not fetched from the authoring environment

Named in the notes only where marked, and not load-bearing for any claim: the *Science* publisher page for Dwork et al. (a hosted copy of the paper was read instead); Dwork et al., "Preserving statistical validity in adaptive data analysis" (STOC 2015); the ACM full text and Berkeley PDF of Shankar et al. (the arXiv abstract was read); the τ-bench PDF §4 (the pass^k *formula* — the abstract and repo were read; the notes quote the metric's definition from Anthropic's guide, which was read, and attribute the numbers to τ-bench's abstract); the OpenAI Responses API `incomplete_details.reason` enumeration (not on the pages read — verify against the live schema); OpenAI's model-deprecations page; the OpenAI Agents SDK tracing page; the SWE-bench harness docs' `report.json` schema (the paper's §A.4 was used); the LM Contamination Index repo; LangSmith's `agentevals` four-way trajectory taxonomy (only *subsequence* was verified on the page read); Chip Huyen, *AI Engineering*; Kent Beck and Michael Feathers on regression tests (SQLite was used as the primary statement instead). A single-author preprint on prompt iteration (arXiv 2601.22025) returned only a summary and is deliberately not cited.
