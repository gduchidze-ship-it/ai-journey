# Applied LLM Systems — Course Syllabus

**A 20-week cohort bootcamp for working software engineers.**
No machine learning background required. You finish with a production LLM system you built, deployed and can defend.

| | |
|---|---|
| **Format** | Cohort-based, part-time, live + independent work |
| **Duration** | 20 weeks |
| **Commitment** | ~15 h / week — 5 h live, ~10 h independent |
| **Delivery** | 2 live sessions per week + office hours + cohort channel |
| **Cohort size** | Capped, so every student gets reviewed work |
| **Language** | English (all materials, sessions and code) |
| **Prerequisite** | Professional software experience. Not ML experience |

---

## 1. What this course is

Most engineers learn LLMs from the outside in: call an API, paste a prompt, ship a demo. That stops working the first time someone asks *why is it slow*, *why did it answer that*, *how do you know it works*, or *can this run inside our network*.

This course goes the other way. You build a transformer from scratch so the failure modes stop being magic. Then you build the system around it — an API, a retrieval pipeline, an agent, an eval harness, a security layer, tracing — and deploy it on Kubernetes. Every week ends with something running on your machine, and a checkpoint you have to demonstrate out loud.

**It is a systems course, not a research course.** You will not train a frontier model, invent an architecture, or read forty papers. You will build, measure and defend one real system.

## 2. Who this is for

**A good fit if you:**

- have professional software engineering experience (roughly 2+ years) in any language, and write Python comfortably
- know git, HTTP, the command line, and what a container is
- can explain a system you built at work and why it is shaped that way
- want to move into an Applied AI or AI Infrastructure role, or own AI features on your current team

**Not a good fit if you:**

- are new to programming — this is not a first course
- want prompt engineering and no-code tooling; that is a different, shorter course
- cannot protect ~15 hours a week for five months. The work is cumulative and there is no way to catch up on four weeks at once

**Explicitly not required:** machine learning, data science, statistics, linear algebra beyond high-school level, a GPU, or a maths degree. Everything mathematical in this course is taught from zero, in the amount an engineer needs and no more.

## 3. What you will build

One system, grown over twenty weeks. Every module adds a layer to the same codebase.

```
                    ┌──────────────────────────────────────────┐
  documents ──────► │  ingestion pipeline (incremental, PII-   │
                    │  redacted, versioned index)              │
                    └───────────────┬──────────────────────────┘
                                    ▼
  user ──► HTTP API ──► retrieval ──► rerank ──► agent loop ──► LLM ──► SSE stream
              │             │                       │            │
              │             └── hybrid search       └── tools     └── vLLM on Kubernetes
              │
              ├── audit log · tenant isolation · secrets
              ├── OpenTelemetry traces → Langfuse
              └── eval harness gates every deploy in CI
```

By the final demo day this system: ingests a corpus incrementally and re-runs for free when nothing changed; answers over a streaming HTTP API with citations; runs an agent with tools, cost caps and retry logic; redacts personal data and writes an audit record for every query; emits traces you can debug a bad answer from; blocks its own deploy when quality drops; and runs on a Kubernetes cluster with vLLM, Prometheus and Grafana on your laptop.

You also leave with a from-scratch GPT implementation, a working tokenizer, a memory calculator you trust, and a design document good enough to send to a hiring manager.

## 4. Learning outcomes

By the end of this course you can:

1. **Implement a transformer from scratch** — tokenizer, attention, the block, sampling — from an empty file, and explain every dimension.
2. **Count bytes before you buy hardware** — parameters, gradients, optimizer state, activations, KV cache — and size a deployment on paper in under three minutes.
3. **Ship an LLM service like a backend engineer** — REST contract, SSE streaming with correct cancellation, async concurrency with timeouts and backpressure, a container, and a test suite that runs offline in seconds.
4. **Build retrieval that measurably works** — chunking, hybrid search, reranking, citations — and prove each step with recall numbers instead of vibes.
5. **Build an agent and break it on purpose** — tools, step and cost caps, retries, cycle detection, prompt-injection exposure — then debug it from traces.
6. **Evaluate non-deterministic systems** — golden sets, deterministic checks, an LLM judge you validated against your own labels, and a CI gate that says whether a change is real or noise.
7. **Handle the constraints real customers impose** — PII detection and redaction, GDPR and data residency, on-premise and air-gapped deployment, secrets, audit logging, tenant isolation.
8. **Operate the data layer** — incremental ingestion, index versioning, zero-downtime rebuild, backup and a restore you have actually performed.
9. **Deploy and observe the platform** — Kubernetes, vLLM, Helm, request routing, Prometheus and Grafana, load testing, autoscaling.
10. **Drive a 50-minute system design conversation** — clarify, estimate, draw, go deep, name trade-offs and failure modes.

## 5. Format and weekly rhythm

| When | What | Hours |
|---|---|---|
| Session A — midweek evening | Concept workshop. Live coding, whiteboard, questions. Recorded | 2 h |
| Session B — weekend | Lab clinic. You build; instructors circulate; ends with demos from two students | 3 h |
| Independent | The week's build, from the module brief | ~8 h |
| Office hours | Twice weekly, drop-in, optional | — |
| Cohort channel | Async help, code review swaps, the room where most learning happens | — |

**Pair rotation.** Every three weeks you are paired with a different cohort member for code review. You review their build, they review yours, both reviews are visible to the instructors. Reading someone else's design decision is half the curriculum.

**Gates are spoken, not submitted.** Each module ends with a checkpoint you demonstrate live in a 10-minute conversation — you run your build, then answer questions about it without notes. This is deliberate: the difference between recognizing an idea and owning it only shows up when you have to say it out loud. Fail a gate and you retake it the following week; nobody is removed from the cohort for it.

---

## 6. Curriculum

Seven modules plus a parallel platform track. Hours are the expected total, live plus independent.
Full topic-by-topic coverage for every module is in [MODULES.md](MODULES.md).

| Module | Weeks | Topic | Hours |
|---|---|---|---|
| 0 | Week 0 | Orientation, setup, the capstone brief | 4 |
| 1 | 1 | Measuring things: the ML judgement an engineer needs | 10 |
| 2 | 2–3 | Deep learning core: autograd to memory arithmetic | 28 |
| 3 | 4–5 | Production backend for AI services | 16 |
| 4 | 6–10 | LLM internals: build a GPT from scratch | 62 |
| 5 | 11–16 | Applied LLM systems: RAG, agents, evals, data, security, tracing | 88 |
| 6 | 17–19 | System design and technical communication | 40 |
| 7 | 20 | Capstone hardening and demo day | 14 |
| **P** | 4–16 | **Platform lab** — Kubernetes + vLLM, in parallel, ~2 h/week | 26 |

**Total: ~288 hours** over 20 weeks.

---

### Module 0 — Orientation · week 0 · 4 h

Environment setup verified before week 1, not during it: Python toolchain, Docker, a working local model, API keys and budget, repository and CI skeleton. Cohort norms, how gates work, how to ask a good question in the channel. You receive the capstone brief on day one, because every module after this is a piece of it.

**Deliverable:** a repository that builds, tests and runs a hello-world service in CI. **Gate:** environment check passes on your own machine.

### Module 1 — Measuring things · week 1 · 10 h

*The vocabulary for proving a system works. It is not a statistics course; it is the part of statistics you will be asked about in a design review.*

**Topics.** Overfitting, regularization and weight decay in plain terms · confusion matrix, precision, recall, F1 · ROC versus precision-recall under class imbalance · picking a threshold as a product decision, not a default · data leakage: contamination, temporal, group, and benchmark contamination in LLMs · ranking and retrieval metrics: recall@k, MRR, nDCG, and choosing k · comparing two systems: bootstrap confidence intervals, paired tests, sample size · calibration: what a confidence score does and does not mean · data quality: missingness, imbalance, distribution shift.

**You build.** `recall@k`, `MRR` and `nDCG@k` from scratch, unit-tested against numbers you computed on paper. Then `compare.py`: two lists of per-case scores in, mean difference and a bootstrap 95% interval out.

**Gate.** Someone says *"the new retriever scored 0.71 versus 0.68 on 200 queries."* Inside a minute you say whether it is real, what test you would run, and how many more queries you need.

### Module 2 — Deep learning core · weeks 2–3 · 28 h

*Enough depth to stop guessing. You implement backpropagation, then spend the second half learning to count bytes — the skill that shows up in every later module.*

**Topics.** Tensors, shapes, broadcasting, `einsum`, devices and dtypes · autograd: the graph, `backward()`, `detach` versus `no_grad` · backpropagation by hand and why activations must be kept · cross-entropy, softmax and the log-sum-exp trick; logits versus probabilities; temperature · MLPs, activations, initialization · LayerNorm, RMSNorm, pre-norm versus post-norm · **embedding layers**: lookup tables, vocabulary × hidden as a share of the model, weight tying · optimizers: SGD → Adam → AdamW, optimizer-state memory, warmup, clipping · a real training loop: `Dataset`, `DataLoader`, checkpoint and resume · **OOM debugging and gradient accumulation** · **producing a loss spike on purpose and recovering from it** · CNNs and RNNs at awareness level · **memory accounting**: parameters + gradients + optimizer states + activations; inference memory; KV-cache size and what GQA does to it · arithmetic intensity: why prefill is compute-bound and decode is memory-bound.

**You build.** A micrograd-style autograd engine with an operator you add yourself. A training loop you wrote, that survives being killed mid-run — then you push the batch size until it dies, recover the effective batch with gradient accumulation, and deliberately poison a batch to watch a loss spike and fix it. Finally `memory_math.py`, checked against a published model's real numbers.

**Gate.** On a whiteboard, in under three minutes: KV-cache bytes for a 7B model at batch 32, 4k context, GQA with 8 KV heads, bf16 — within 10%. And you diagnose an OOM, a corrupted batch and a too-high learning rate from a training log alone.

### Module 3 — Production backend for AI services · weeks 4–5 · 16 h

*The layer between "it works in a notebook" and "it serves users". Taught early, because everything after this is a service.*

**Topics.** REST design: resources, status codes, one error contract for every failure, cursor pagination, versioning and deprecation, idempotency keys, long-running work as 202 plus polling · **streaming**: the SSE wire format, client disconnect and cancellation, proxies that silently buffer your stream, errors after the 200 has been sent, parsing structured output while it streams · **async Python at production level**: the event loop, why one blocking call stalls every request on a worker, `TaskGroup`, timeouts at every layer, semaphores, connection pooling, bounded queues as backpressure, the GIL versus asyncio versus processes · containerization: multi-stage builds, layer caching, non-root, health checks, draining on `SIGTERM` · **testing LLM applications**: mocking the provider at the HTTP boundary, recorded fixtures including streamed chunks, a scripted fake model to drive agent branches, what you must never assert (exact model text), and where pytest stops and evaluation begins.

**You build.** An HTTP API with a streaming endpoint, correct cancellation and an error contract; an async fetcher that demonstrates the blocking-call collapse in measured numbers; a container; and a test suite that runs offline, without an API key, in under ten seconds.

**Gate.** You explain what happens to fifty in-flight requests when someone drops a synchronous HTTP call into an async endpoint, name the fix, and describe what a container must do when Kubernetes sends `SIGTERM` mid-stream.

### Module 4 — LLM internals: build a GPT from scratch · weeks 6–10 · 62 h

*The heaviest module. By the end you have typed every line of a working language model.*

**Topics.** **Tokenization**: byte-pair encoding, training and encoding, special tokens and chat templates, the multilingual and code cost penalty · **attention**: Q/K/V, scaled dot-product, causal masking, multi-head, the O(n²) cost, MQA/GQA/MLA and what they do to cache size · **architecture assembly**: the block, residuals, norms, positional encoding from absolute to RoPE, SwiGLU, counting parameters from a config · **decoding**: greedy and beam, temperature, top-k, top-p, stop sequences, why `temperature=0` is still not bit-identical, and constrained decoding against a JSON schema · **modern variants**: mixture of experts, active versus total parameters, long-context scaling · **pretraining**, at the level of reading a training run and understanding scaling laws · **post-training**: supervised fine-tuning, loss masking, RLHF and DPO, what each stage buys · **fine-tuning in practice**: LoRA, rank and alpha, which modules to target, and the decision framework for prompt versus retrieve versus fine-tune.

**You build.** A BPE tokenizer trained on your own corpus, with a token-cost comparison across English, code and one non-English language. Causal multi-head attention with a GQA switch that prints cache size in bytes. A full GPT, plus `param_count(config)` reproducing a published model exactly. Top-k and top-p sampling, then a JSON-reliability experiment run 50 times with and without schema constraints. A LoRA fine-tune on a task where prompting demonstrably fails, reported with before-and-after accuracy, adapter size and training time.

**Gate.** From an empty file, you write causal multi-head attention from memory and it runs. Given a model config you compute the parameter count on paper to within 2%. And given a business problem you argue prompt versus retrieval versus fine-tuning in two minutes, with cost, latency and maintenance for each.

### Module 5 — Applied LLM systems · weeks 11–16 · 88 h

*The largest module, and the one that produces your capstone. Six weeks of building a system that survives contact with reality.*

**11 — Inference systems.** Prefill versus decode; TTFT, TPOT and inter-token latency; KV-cache sizing and eviction; PagedAttention; continuous batching; prefix caching; quantization and speculative decoding. You measure your own stack at 1, 4 and 16 concurrent requests and explain where the knee is.

**12 — Retrieval (RAG).** Embeddings and similarity; chunking strategies; index internals — flat, IVF, HNSW — and the recall/latency/memory triangle; BM25 and hybrid retrieval; reranking with a cross-encoder; context construction, ordering, deduplication, budget, lost-in-the-middle; citations; permission-aware retrieval; the seven ways retrieval fails. You build a real retrieval service over a corpus you care about, write 30 golden question-answer pairs by hand, and report recall@5 three times: dense, hybrid, hybrid plus reranker.

**13 — The data layer.** SQL to a working level — joins, window functions, `EXPLAIN`, indexes, N+1 — because the customer's data is in a database. Then batch versus incremental ingestion, content-hash change detection, idempotent upserts, tombstones for deletes, dead-letter queues, resumable runs. Then versioning: pinning the embedding model next to the corpus, zero-downtime reindex by alias swap, snapshots, and a restore you actually perform.

**14 — Agents.** Tool schema design; the plan-act-observe loop and when a plain chain is better; context compaction; error recovery, retries, step limits, cost caps, cycle detection; prompt injection through tool output and retrieved text. You build an agent with three tools and full rails, then break it on purpose three ways and write up what happened.

**15 — Evaluation.** Eval types and golden versus synthetic data; deterministic checks; LLM-as-judge with a rubric, its biases, and how to validate a judge against your own labels; the RAG triad; a regression suite in CI; telling a real improvement from noise, using Module 1. You build a 40-case harness that runs on one command and exits non-zero when quality drops.

**16 — Security, privacy and operations.** Secrets management and scanning; PII detection and redaction, and the difference between redaction, pseudonymization and anonymization; GDPR in the parts that touch you — controller versus processor, subprocessors, international transfers, and what a right-to-erasure request means for a vector index, a cache and fine-tuned weights; on-premise and air-gapped deployment; audit logging and tenant isolation with a cross-tenant leak test. Then observability: OpenTelemetry with GenAI conventions, self-hosted Langfuse, and debugging a broken agent run from traces alone. Finally the operations layer: latency budgets, prompt caching, model routing and fallback, versioning prompts and indexes, shadow deploys, and a runbook.

**Gate.** Your system runs end to end: it ingests incrementally, answers over a stream with citations, redacts personal data, writes an audit record, emits a trace, and fails CI when the eval score drops. You then debug a deliberately broken run using only the trace UI, and you answer *"can you delete everything about this person"* with a real procedure, naming which parts are minutes, which wait for the next rebuild, and which you must design around from the start.

### Module 6 — System design and communication · weeks 17–19 · 40 h

*Being right is half of it. This module is about the other half.*

**Topics.** Driving the conversation: clarify, estimate, draw, go deep, state trade-offs, manage the clock · turning a product ask into a problem with a metric, and knowing when rules beat a model · data, labels and features; training/serving skew; feedback loops · serving and latency: batch versus online, splitting a p95 budget across stages, multi-stage funnels, caching, graceful degradation · monitoring, drift, silent failure, retraining triggers, shadow deploys and rollback.

**Then reps.** Four timed 50-minute designs, spoken aloud, from a menu that includes: an inference batching service at 100k requests per second · retrieval over 100 million documents · an evaluation pipeline as a batch system · an agent platform with sandboxing and durability · a ranking or recommendation system · fraud detection under a hard latency budget. Each rep is followed by a comparison against a reference solution and a written delta.

**Gate.** You drive one design end to end in 50 minutes: clarify, estimate memory and cache, draw the path from request to token, deep-dive one component the reviewer chooses, and name three failure modes with mitigations.

### Module 7 — Capstone and demo day · week 20 · 14 h

Hardening, documentation, and a public demo. You present your system to the cohort and invited guests in 15 minutes: what it does, one architectural decision you would defend, one thing that went wrong, and the numbers.

**Deliverables.** The running system · a design document (2–4 pages) · your eval report · a one-page runbook · a recorded demo.

### Track P — Platform lab · weeks 4–16 · ~2 h/week

*Runs in parallel, so your capstone has somewhere to live.*

Kubernetes fundamentals — pods, deployments, services, config, secrets, probes, resources · running vLLM with an OpenAI-compatible API on a CPU backend · packaging with Helm · a request router and multiple replicas · Prometheus and Grafana with real inference metrics: TTFT, queue depth, KV-cache usage · load testing with k6 · horizontal autoscaling. Later weeks add ingress and TLS, network policy and tenancy, and SRE practice — SLOs, burn-rate alerts, and cost and capacity planning.

Each session is a lesson plus a lab with an automated grader, so you know immediately whether it works. **Gate:** you deploy your own capstone service to the cluster and show it under load on a Grafana dashboard.

---

## 7. Assessment and certification

There are no exams and no grades. There is a bar.

| Requirement | To pass |
|---|---|
| **Module gates** | Pass at least 6 of the 7 module gates, demonstrated live |
| **Weekly builds** | Submit at least 16 of 20 weekly builds, each reviewed by a peer or instructor |
| **Platform lab** | Your service deployed to the cluster, with a working dashboard |
| **Capstone** | System demo, design document, eval report and runbook |
| **Attendance** | 80% of live sessions, or the recording plus a written catch-up note |

Meeting all five earns the certificate. Missing the capstone does not — the whole course points at it.

## 8. Materials and costs

**Books (two, both required).** Sebastian Raschka, *Build a Large Language Model (From Scratch)* — the spine of Module 4. Chip Huyen, *AI Engineering* (O'Reilly) — the spine of Module 5. Buy both before week 1.

**Everything else is free.** Course lesson pages, labs and graders are provided. All other reading is open — course sites, primary documentation, and a small, deliberate number of papers with the exact sections marked. Each topic caps at three sources; adding a fourth is procrastination with a good reputation.

**You need to provide:**

| Item | Requirement |
|---|---|
| Laptop | 16 GB RAM minimum, ~60 GB free disk. Apple Silicon, Linux, or Windows with WSL2 |
| GPU | **Not required.** Everything runs on CPU or a paid API. GPU-only topics are taught as reading, and marked as such |
| API credits | ~$40–70 across the whole course, for agent and eval work |
| Time | ~15 h/week for 20 weeks. This is the real cost |

## 9. Policies

**Attendance.** Live sessions are recorded. Missing one is fine; missing the lab clinic repeatedly is not, because that is where your work gets looked at.

**Late work.** One week of grace on any build, no explanation needed. Beyond that, talk to an instructor before the deadline rather than after.

**AI assistants.** Encouraged for the same things you would use them for at work: boilerplate, docs lookup, debugging, review. **Gates are oral and unassisted** — you run your own code and answer questions about it. If an assistant wrote something you cannot explain, you have not finished the build. That is the whole policy, and it is enforced by the format rather than by surveillance.

**Collaboration.** Pair on problems, discuss designs, review each other's code. Submit your own work. Copying a solution wholesale is the only thing that wastes your own money here.

**Conduct.** Questions are never stupid; the channel exists for them. Condescension toward another student is the one thing that gets you removed.

## 10. What this course does not cover

Stated plainly, so nobody is surprised in week 12:

| Not covered | Why |
|---|---|
| Algorithm and data-structure interview prep | A separate discipline. Budget your own time if you are targeting companies that test it |
| Training frontier models, multi-node clusters, CUDA kernels | Taught as reading only. It needs hardware nobody in the cohort has |
| Data engineering platforms — Airflow, dbt, Spark, warehouse modeling | Adjacent job. We cover the slice an AI engineer owns |
| Front-end development | Beyond consuming a streaming endpoint correctly |
| Prompt-engineering tricks as a subject | Covered where it belongs: as one option among prompting, retrieval and fine-tuning |
| Vendor certifications | This is not a certification prep course |

## 11. Frequently asked

**I have never trained a model. Is Module 2 going to lose me?**
No. It starts at tensors and builds up, and the mathematics is taught as code you type. The students who struggle are the ones short on time, not short on background.

**I don't have a GPU.**
Neither does the course. Every build runs on a laptop CPU or a paid API, and the topics that genuinely need a datacenter are taught as reading and marked.

**Can I do this alongside a full-time job?**
That is exactly who it is for — the schedule is built around evenings and a weekend block. Twenty weeks at fifteen hours is the honest cost, and the cumulative structure means falling four weeks behind is very hard to recover from.

**What if I fail a gate?**
You retake it the next week. Gates exist to tell you where you are, not to remove you.

**Do I finish with something to show?**
Yes, and it is the point: a deployed system, a design document, an eval report and a recorded demo — plus a from-scratch language model implementation and a memory calculator you trust.

---

## 12. Schedule at a glance

| Week | Module | Focus | What you produce |
|---|---|---|---|
| 0 | 0 | Setup and orientation | Repository with CI |
| 1 | 1 | Metrics, leakage, retrieval metrics, significance | `compare.py` + metric implementations |
| 2 | 2 | Tensors, autograd, backprop, losses, layers, embeddings | micrograd with your own operator |
| 3 | 2 | Optimizers, a real training loop, OOM, loss spikes, memory math | Training loop with resume · `memory_math.py` |
| 4 | 3 · P | API design, error contracts · Kubernetes core | HTTP API with a real error contract |
| 5 | 3 · P | Streaming, async Python, containers, testing | Streaming service + offline test suite |
| 6 | 4 · P | Tokenization | BPE tokenizer + token-cost table |
| 7 | 4 · P | Attention | Multi-head attention with a GQA switch |
| 8 | 4 · P | Architecture assembly | A working GPT + `param_count` |
| 9 | 4 · P | Decoding and structured output | Sampler + JSON reliability experiment |
| 10 | 4 · P | Pretraining, alignment, LoRA fine-tuning | Fine-tune report with before/after numbers |
| 11 | 5 · P | Inference systems | TTFT/TPOT measurements on your own stack |
| 12 | 5 · P | Retrieval and RAG | Retrieval service + 30 golden pairs |
| 13 | 5 · P | SQL, ingestion, index versioning | `ingest.py --since` + zero-downtime reindex |
| 14 | 5 · P | Agents | Agent with three tools, rails, and a break-it report |
| 15 | 5 · P | Evaluation | 40-case harness with a validated judge, gating CI |
| 16 | 5 · P | Security, privacy, observability, operations | Data map, audit log, traces, runbook |
| 17 | 6 | Design protocol, framing, data and labels | Design rep 1 |
| 18 | 6 | Serving, latency budgets, monitoring and drift | Design reps 2–3 |
| 19 | 6 | Reps and mock reviews | Design rep 4 + delta log |
| 20 | 7 | Capstone hardening | Demo day |

---

*Questions before enrolling are welcome — the honest answer to "is this right for me" is worth more to both of us than a filled seat.*
