# LLM Internals and Applied LLM Systems — Roadmap
**Part of:** [Master Syllabus](../SYLLABUS.md) — the order all five roadmaps run in.

**Scope:** 130 h (AAI) · 120 h (INF).
**Prereq:** [dl_core](../dl_core/ROADMAP.md), including **§7 (numerics, memory, performance)**.
**Supersedes:** `README.md` in this directory.

> **Verdict on the original list: good syllabus, wrong axis.** Raschka, Alammar, HF tokenizers, CS336, Lilian Weng, MoE, RLHF, sampling, LoRA — that is a solid **model internals** course, and it is Part A below. But the 2026 Applied AI loop is roughly **40% RAG / agents / evals, 30% production systems, 20% LLM internals, 10% behavioral**. The original list covered the 20% well and the 40% not at all: no RAG, no agents, no evals, no structured output, no KV-cache math, no quantization. **Part B is that missing 40%, and for an Applied AI job it matters more than Part A.**
>
> Also added: **§14, the timed Python round and the customer round.** Anthropic and OpenAI both run them, most people fail the customer one, and nothing else in this repo prepares for either.

**Legend:** **Core** = asked directly, must be fluent · **Useful** = follow-up question, working knowledge · **Skim** = one paragraph · **Skip** = don't.

**Your machine:** Apple M4, 16 GB RAM, no NVIDIA GPU. Every **Build** below runs on that machine or on a paid API. Anything needing a real GPU is marked **read-only for now** — learn it, don't run it.

---

## The path

Do these in order, one at a time. If you fail a section's checkpoint, stay there. If you pass it, move on even if you did not read everything.

| Step | Topic | AAI h | INF h |
|---|---|---|---|
| 1 | Tokenization | 8 | 8 |
| 2 | Attention | 12 | 12 |
| 3 | Architecture assembly | 10 | 10 |
| 4 | Decoding and structured output | 8 | 6 |
| 5 | Modern architecture variants | 5 | 7 |
| 6 | Pretraining | 4 | 14 |
| 7 | Post-training and alignment | 7 | 5 |
| 8 | Fine-tuning in practice | 8 | 8 |
| — | **END OF PART A** — 62 h (AAI) · 70 h (INF) | | |
| 9 | Inference systems | 6 | 14 |
| 10 | RAG | 20 | 12 |
| 11 | Agents | 16 | 8 |
| 12 | Evals | 12 | 8 |
| 13 | Production LLM operations | 6 | 4 |
| 14 | Interview prep — **runs in parallel from week 1** | 8 | 4 |

**Get these two books first.** Sebastian Raschka, *Build a Large Language Model (From Scratch)* — book + [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) — is your spine for §1–§8. Chip Huyen, *AI Engineering* (O'Reilly, 2025) — chapter map at [chiphuyen/aie-book](https://github.com/chiphuyen/aie-book) — carries §8–§13 and is **the closest book there is to the Applied AI job description**.

# PART A — Model internals

## 1. Tokenization

| Sub-topic | AAI | INF |
|---|---|---|
| BPE (byte-pair encoding): training and encoding | Core | Core |
| Special tokens, chat templates, BOS/EOS bugs | Core | Useful |
| Token counting, vocabulary size, and the multilingual and code penalty | Core | Useful |

**Resources**
- **Start here:** Raschka **Ch. 2 "Working with Text Data"** — type every line, do not just read.
- Andrej Karpathy — [minbpe](https://github.com/karpathy/minbpe) and the video *"Let's build the GPT Tokenizer"* ([Zero to Hero](https://karpathy.ai/zero-to-hero.html)). Only if BPE still feels unclear after Raschka.
- [tiktoken](https://github.com/openai/tiktoken) and [tiktokenizer.vercel.app](https://tiktokenizer.vercel.app/) — paste text, see the tokens.

**Build (~3 h):** train BPE on ~1 MB of text and encode with it. Then compare your token counts against `tiktoken` on the same 2 KB of English, of Python code, and of Georgian. Write the three ratios in a small table.

**Ready to move on when:** you take one paragraph and say, *before* checking, whether English or Georgian costs more tokens and roughly how much more — then say what that does to a context window and to a bill.

## 2. Attention

| Sub-topic | AAI | INF |
|---|---|---|
| Q/K/V, scaled dot-product, why we divide by √d; causal masking; multi-head | Core | Core |
| Cost: O(n²) time and O(n²) attention memory without Flash | Core | Core |
| MQA / GQA / MLA and what they do to KV-cache size | Useful | Core |

**Resources**
- **Start here:** Raschka **Ch. 3 "Coding Attention Mechanisms"** — the whole chapter, typed out.
- Jay Alammar — [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) and [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/). Read next to Raschka Ch. 3, same week.
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models* — [arXiv:2305.13245](https://arxiv.org/abs/2305.13245). Abstract and method section only. Kept because nothing else here explains why cache size drives the design.

**Build (~4 h):** finish Raschka Ch. 3. Then add an `n_kv_heads` setting (that is GQA) and print the KV-cache size in bytes for full MHA versus GQA at the same model size. Runs on CPU.

**Ready to move on when:** you open an empty file, write causal multi-head attention from memory, and it runs — and you explain GQA in *bytes moved per generated token*, not "it is faster".

## 3. Architecture assembly

| Sub-topic | AAI | INF |
|---|---|---|
| The block: attention + FFN + residuals + norms; pre-norm vs post-norm | Core | Core |
| Positional encoding: absolute → learned → RoPE → ALiBi | Core | Core |
| SwiGLU, where the parameters sit, and counting them from a config | Useful | Core |

**Resources**
- **Start here:** Raschka **Ch. 4 "Implementing a GPT Model from Scratch"** — the whole chapter. You finish with a working GPT-2-size model.
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* — [arXiv:2104.09864](https://arxiv.org/abs/2104.09864). Read **§3.2 and Fig. 1** only. **RoPE is the one thing here to know cold.**
- Andrej Karpathy — [nanoGPT](https://github.com/karpathy/nanoGPT). After Ch. 4, read `model.py` and list how it differs from your code.

**Build (~3 h):** finish Ch. 4. Then write `param_count(config)` that reproduces the published parameter count of GPT-2 124M exactly. Llama 3 8B is a stretch goal, not a requirement.

**Ready to move on when:** given a config (layers, hidden size, heads, vocab, FFN multiplier) you compute the parameter count on paper to within 2% and say which part is biggest.

## 4. Decoding and structured output

| Sub-topic | AAI | INF |
|---|---|---|
| Greedy and beam search; why chat models rarely use beam | Core | Useful |
| Temperature, top-k, top-p, min-p; stop sequences; why `temperature=0` is still not bit-identical | Core | Useful |
| Constrained decoding: JSON schema, grammars, logit masking | Core | Core |

**Resources**
- **Start here:** Hugging Face blog — [How to generate text: using different decoding methods](https://huggingface.co/blog/how-to-generate).
- OpenAI — [Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs). Table stakes in 2026 and missing from your original list.
- [dottxt-ai/outlines](https://github.com/dottxt-ai/outlines) and [guidance-ai/guidance](https://github.com/guidance-ai/guidance) — read each README to see how masking logits against a grammar works.

**Build (~3 h):** add top-k and top-p sampling to your §3 model. Then take one small task and force JSON two ways — prompt only, and schema-constrained. Run each 50 times and count the broken outputs.

**Ready to move on when:** someone says "the model gives us broken JSON 3% of the time" and your first answer is constrained decoding — and you can say why better prompts stop helping after a point.

## 5. Modern architecture variants

| Sub-topic | AAI | INF |
|---|---|---|
| MoE (mixture of experts): routing, top-k experts, load-balancing loss | Useful | Core |
| MoE serving: active vs total parameters, memory vs FLOPs | Skim | Core |
| Long context: RoPE scaling, position interpolation, YaRN | Useful | Core |

**Resources**
- **Start here:** Hugging Face blog — [Mixture of Experts Explained](https://huggingface.co/blog/moe). Still the best free primer.
- Jiang et al., *Mixtral of Experts* — [arXiv:2401.04088](https://arxiv.org/abs/2401.04088). **§2 and §3.** Critical if you target Mistral.
- Peng et al., *YaRN: Efficient Context Window Extension* — [arXiv:2309.00071](https://arxiv.org/abs/2309.00071). **§3.** Every lab ships long-context models now.

**Build (~2 h):** one page comparing a dense 7B with a similar-quality MoE — active vs total parameters, memory, and when you would pick each. **Read-only for now:** you cannot serve an MoE on 16 GB.

**Ready to move on when:** you explain, in memory terms, why an MoE that uses fewer FLOPs can still be harder to serve than a dense model.

## 6. Pretraining — split by target

| Sub-topic | AAI | INF |
|---|---|---|
| The pretraining objective, data pipeline, deduplication, contamination | Skim | Core |
| Scaling laws and Chinchilla-optimal compute budgets | Useful | Core |
| Checkpointing, resume, failure recovery | Skip | Core |

**Resources**
- **Start here:** Raschka **Ch. 5 "Pretraining on Unlabeled Data"** — **AAI: read it, don't type it. INF: type it and run it.**
- Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla) — [arXiv:2203.15556](https://arxiv.org/abs/2203.15556). **§3 and Table 3** only. This is the one people quote.
- Stanford CS336 Spring 2025 — [course site](https://stanford-cs336.github.io/spring2025/), **Lectures 9 and 11: "Scaling laws."** INF only, optional depth.

**Build:** AAI (~1 h) — run the Ch. 5 notebook once and watch the loss go down; that is all. INF (~6 h) — train the small GPT on a small text file on your Mac, with save/resume that survives a mid-run `kill -9`. Multi-node pretraining is **read-only for now**.

**Ready to move on when:** you say what a scaling law lets you decide *before* spending money, and where it stops being reliable.

## 7. Post-training and alignment

| Sub-topic | AAI | INF |
|---|---|---|
| SFT: dataset format, chat templates, masking the loss on the prompt | Core | Useful |
| RLHF: reward model → PPO; what each stage buys you; DPO | Core | Useful |
| Constitutional AI; reasoning models and test-time compute | Core | Useful |

**Resources**
- **Start here:** Chip Huyen — [RLHF: Reinforcement Learning from Human Feedback](https://huyenchip.com/2023/05/02/rlhf.html). High level, and deep enough for AAI.
- Raschka **Ch. 7 "Finetuning to Follow Instructions."** The loss-masking detail here is a common interview probe.
- Bai et al., *Constitutional AI: Harmlessness from AI Feedback* — [arXiv:2212.08073](https://arxiv.org/abs/2212.08073). **§2–3.** Read it properly if you interview at Anthropic.

**Build (~2 h):** one page tracing a single prompt through base → SFT → preference-tuned, with a concrete example of the behaviour each stage adds.

**Ready to move on when:** you explain why a reward model trains on *comparisons* instead of absolute scores, and name two ways an RLHF pipeline goes wrong in practice.

## 8. Fine-tuning in practice

| Sub-topic | AAI | INF |
|---|---|---|
| LoRA: low-rank decomposition, rank and alpha, which modules to target | Core | Core |
| QLoRA (4-bit, NF4) and multi-LoRA serving — **read-only, needs NVIDIA** | Useful | Core |
| **When to fine-tune vs prompt vs retrieve** | Core | Useful |

**Resources**
- **Start here:** Raschka **Appendix E "Parameter-efficient Finetuning with LoRA."**
- Sebastian Raschka — [Practical Tips for Finetuning LLMs Using LoRA](https://lightning.ai/pages/community/lora-insights/). Measured ablations on rank, alpha, and target modules.
- Chip Huyen, *AI Engineering* **Ch. 7 "Finetuning."** Read it for one thing: the prompt vs RAG vs fine-tune decision framework.

**Build (~5 h):** LoRA fine-tune a small model (0.5B or smaller, MPS on your Mac) on one task where prompting clearly fails. Report accuracy before and after on a held-out set, plus adapter size and training time.

**Ready to move on when:** given a business problem, you argue prompt vs RAG vs fine-tune in under two minutes, with cost, latency, and maintenance for each.

# PART B — Applied LLM systems

**None of this was in the original README. It is ~40% of the Applied AI loop.** Sections 10, 11 and 12 are build sections: reading them without shipping the deliverable gives you recognition, not recall — and these are exactly the topics where interviewers keep asking follow-ups until you run out of answers.

## 9. Inference systems

Mostly covered by the other pillars: `lessons/` 03–10, 15–16, 20 and `system-design/` L34–L40. **Do not re-learn that here.** This is the reading that makes it click.

| Sub-topic | AAI | INF |
|---|---|---|
| Prefill vs decode; TTFT, TPOT, ITL; throughput vs latency | Core | Core |
| KV cache sizing and eviction; PagedAttention block tables | Core | Core |
| Continuous batching (iteration-level scheduling); prefix caching | Core | Core |
| Quantization for serving (GPTQ, AWQ, FP8); speculative decoding | Useful | Core |

**Resources**
- **Start here:** vLLM blog — [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html).
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* — [arXiv:2309.06180](https://arxiv.org/abs/2309.06180). **§3–4.** Know the block table.
- Chip Huyen, *AI Engineering* **Ch. 9 "Inference Optimization."** Quantization and speculative decoding without the kernel details.

**Build (~2 h):** on your CPU vLLM from `lessons/`, measure TTFT and TPOT at 1, 4 and 16 concurrent requests. Plot the three points. Three sentences on where the knee is and why.

**Ready to move on when:** you draw the request path from HTTP to token and name the queue at every hop, say what happens to TTFT and TPOT as concurrency rises, and given a missed latency SLO name the four most likely causes in order.

## 10. RAG (retrieval-augmented generation) — build level

Find text, put it in the prompt, answer from it.

| Sub-topic | AAI | INF |
|---|---|---|
| Embeddings and similarity (cosine vs dot vs L2); chunking: size, overlap, structure-aware | Core | Useful |
| Index internals: flat, IVF, HNSW; recall vs latency vs memory | Core | Core |
| BM25 (keyword search), hybrid retrieval, reranking with a cross-encoder | Core | Useful |
| Context construction: ordering, dedup, budget, lost-in-the-middle; citations; permission-aware retrieval | Core | Core |
| Failure types: nothing retrieved, wrong chunk, right chunk but wrong answer | Core | Useful |

**Resources**
- **Start here:** Chip Huyen, *AI Engineering* **Ch. 6 "RAG and Agents."** Read the RAG half now, the agent half in §11.
- Anthropic — [Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval). A working recipe (context per chunk + BM25 + reranking) with measured recall numbers. Memorize the numbers.
- Barnett et al., *Seven Failure Points When Engineering a Retrieval Augmented Generation System* — [arXiv:2401.05856](https://arxiv.org/abs/2401.05856). Short. It is the failure list you will use in interviews.

**Build (~10 h):** a small RAG service over one corpus you actually care about, about **2,000 chunks**. Hybrid retrieval (BM25 + dense), a cross-encoder reranker, citations in the answer. Hand-write **30** question/answer pairs as a golden set. Measure recall@5 three times: dense only, hybrid, hybrid + reranker. One paragraph on what each step bought. Use a local index (FAISS or Qdrant both run on your Mac); when you set `M` and `efSearch`, read that index's own docs and try three values instead of copying defaults.

**Ready to move on when:** someone says "our RAG returns junk" and you give an ordered list of checks *before* proposing a fix; you say what `efSearch` trades off from having changed it yourself; and you justify your chunk size with your own measurement.

## 11. Agents — build level

An agent = a loop where the model calls tools, sees the results, and decides what to do next.

| Sub-topic | AAI | INF |
|---|---|---|
| Tool calling: schema design, good descriptions, parameter validation | Core | Useful |
| The loop: plan → act → observe → repeat; when to stop; when a plain chain is better | Core | Skim |
| Context management: compaction, summarization, what to drop and when | Core | Useful |
| Error recovery and rails: retries, backoff, step limits, cost caps, cycle detection | Core | Core |
| Observability: traces, per-step tokens and cost | Core | Core |
| Prompt injection through tool output and retrieved text | Core | Core |

**Resources**
- **Start here:** Anthropic — [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents). The workflow-vs-agent split and the pattern names are the exact vocabulary interviewers use.
- [Anthropic tool use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — the API reference you keep open while building.
- Simon Willison — [prompt injection archive](https://simonwillison.net/tags/prompt-injection/). Read the newest few posts; this area moves fast.

**Build (~8 h):** an agent with **3 tools**, a step cap, a cost cap, validated tool schemas, retry with backoff, and a trace log written to a file. Then break it on purpose: make one tool always fail, give it a task that loops forever, and hide an injection string inside a tool result. Half a page on what happened and what you changed. API model, no GPU needed.

**Ready to move on when:** you name five ways an agent fails that normal unit tests never catch with a fix for each; you defend a decision *not* to use an agent for some task; and your trace log answers "why did it do that on run 47" without new logging.

## 12. Evals

**The biggest gap in this repo and the highest-value section for these jobs.** Interviewers treat "how would you prove it works in production" as half the answer to every design question. Do it after §10 and §11 — you need something worth evaluating. Metrics and statistics come from [ml_core](../ml_core/ROADMAP.md) §4 and §6, and eval-set contamination from its §3.

| Sub-topic | AAI | INF |
|---|---|---|
| Eval types: unit / component / end-to-end, offline vs online; golden vs synthetic datasets; contamination | Core | Core |
| Deterministic checks: exact match, schema validation, tool-call correctness | Core | Core |
| LLM-as-judge: rubric, scale, calibration; biases (position, verbosity, self-preference); judge validation | Core | Useful |
| RAG triad: faithfulness, answer relevance, context relevance | Core | Useful |
| Regression suite in CI; is a delta real or noise | Core | Core |

**Resources**
- **Start here:** Hamel Husain — [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/). The most-cited practitioner piece on the topic.
- Hamel Husain — [Creating a LLM-as-a-Judge That Drives Business Results](https://hamel.dev/blog/posts/llm-judge/). The full procedure for building and checking a judge.
- Chip Huyen, *AI Engineering* **Ch. 3 "Evaluation Methodology"** and **Ch. 4 "Evaluate AI Systems."** The strongest part of the book.

**Build (~8 h):** an eval harness over your §10 RAG. **40 cases.** Deterministic checks (schema valid, citation present) plus one LLM judge. Validate the judge: label 20 cases yourself and report how often it agrees with you. One command runs everything, prints the score, and exits non-zero when the score drops below your threshold.

**Ready to move on when:** given an unfamiliar LLM feature you design an eval suite in ten minutes (what to measure, what dataset, what number blocks the deploy); you answer "how do you know your judge is right" with a procedure; and you say whether a 3-point gain on 40 cases is real or noise.

## 13. Production LLM operations

Partly covered by `lessons/` 12–13, 17, 18 and `system-design/` L15–L17. This is the application layer *above* the serving layer.

| Sub-topic | AAI | INF |
|---|---|---|
| Latency budget across retrieval → rerank → generation → post-processing | Core | Core |
| Prompt caching: what is cacheable, prefix stability, prompt layout | Core | Core |
| Streaming: SSE, partial parsing of structured output, cancellation | Core | Core |
| Cost modeling; model routing and fallback chains | Core | Useful |
| Guardrails and prompt-injection defense | Core | Core |
| Versioning prompts, models and indexes; rollback, shadow deploys, silent quality drops | Core | Core |

**Resources**
- **Start here:** Chip Huyen, *AI Engineering* **Ch. 5 "Prompt Engineering"** and **Ch. 10 "AI Engineering Architecture and User Feedback."**
- Anthropic — [Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching). The cache-breakpoint rules decide how you order a prompt.
- [LiteLLM](https://github.com/BerriAI/litellm) — the routing, fallback and budget layer. Read the docs, then wire it into your §11 agent.

**Build (~4 h):** a one-page runbook for your §10 + §11 stack: three alerts with their conditions, two incident playbooks (quality regression, cost spike), and the rollback steps.

**Ready to move on when:** you produce a latency budget for a 2-second p95 target and say what you cut first when you miss it; you explain how a prompt change ships safely (version → eval gate → shadow → rollback); and you say how you would notice quality dropping silently after a provider model update.

## 14. Frontier-lab interview prep

Not blocked by anything. **Start in week 1 and run it beside everything else.** This is reps, not building.

| Sub-topic | AAI | INF |
|---|---|---|
| Timed Python round: build, parse or debug a real component, with tests | Core | Core |
| Python concurrency: asyncio, thread pools, backpressure, the GIL | Core | Core |
| Safety and alignment literacy | Core | Useful |
| Customer conversation: getting requirements, scoping, pushing back | Core | Skip |

**Resources**
- **Start here:** Python — [asyncio documentation](https://docs.python.org/3/library/asyncio.html) and [pytest](https://docs.pytest.org/en/stable/). The coding round is Python and it expects tests.
- Anthropic — [Core Views on AI Safety](https://www.anthropic.com/news/core-views-on-ai-safety). Form your own opinion; quoting it back is not enough.
- Anthropic — [anthropic-quickstarts](https://github.com/anthropics/anthropic-quickstarts) for realistic small-app shapes.

**Coding reps — 2 per week, 35–45 minutes each, cold, from a blank file, with a timer and tests:** bounded LRU cache with TTL · token-bucket rate limiter · streaming JSON parser that handles partial chunks · retry wrapper with exponential backoff and jitter · semaphore-bounded async fetcher over 1000 URLs · a KV-cache class with eviction · a batching scheduler that groups requests up to a size or a deadline.

**Build (~8 h, spread out):** a 1-page design doc for one system you built here, good enough to send as a writing sample. Plus a log of **10** timed reps with one line on what went wrong in each.

**Ready to move on when:** you write a bounded LRU cache with TTL, tests included, in 35 minutes, cold; you state your own position on one alignment trade-off and defend it; and you run a 20-minute requirements conversation about a vague feature request without jumping to a solution.

## Cut from the original README

| Item | Action | Why |
|---|---|---|
| Raschka **Ch. 5** (pretraining) | AAI: read only · INF: do it fully | Weeks of work; split by target |
| CS336 **Lectures 1–6** as a block | Keep only 9 and 11, in §6; 2, 5, 6 live in `dl_core` §7 | The rest is resource accounting and GPUs, not model internals |
| Lilian Weng's surveys; HF LLM Course and NLP Course Ch. 6 | Cut | Raschka, Alammar and the LoRA tips post cover the same ground faster |
| ~40 papers (Vaswani, DPO, Switch, GPTQ, ReAct, HNSW, ColBERT, …) | Cut | Nobody finishes 40 papers. Eight stayed, each with the exact section to read |

**Added, in order of how much their absence would cost you:** §12 Evals · §10 RAG · §11 Agents · §4 constrained decoding · §14 timed coding and the customer round · §9 quantization and KV-cache math · §5 long context · §14 safety literacy.
