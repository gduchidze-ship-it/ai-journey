# Applied AI Engineering — Lesson Plan

**41 core lessons — two per week across twenty weeks, plus a week-0 setup lesson — with 13 parallel platform labs and 10 optional electives.**
Written for working software engineers with no ML background. Every topic in [MODULES.md](MODULES.md) appears in exactly one lesson below; the mapping is in Appendix B. Topics from [ai-system-design-guide](https://github.com/ombharatiya/ai-system-design-guide) are folded into the relevant lessons or into Track E, and its case studies join the Anthropic-style interview cases in the Appendix A case bank.

---

## How a week works

| | Day 1 — *Concept* | Day 2 — *Lab* |
|---|---|---|
| Length | 2 h live + ~3 h independent | 3 h live + ~5 h independent |
| Shape | Mechanism first: the thing is opened, not described. Live coding and whiteboard | You build. Instructors circulate. Ends with two students demoing |
| Output | Notes you could re-derive from | Code in your repo that runs |

Weeks 4–16 also carry a **2 h platform lab** (Track P, lessons P1–P13) that runs in parallel — a short lesson plus an auto-graded lab.

## How a lesson is written

Each lesson has the same five parts:

- **Engineer's frame** — why this matters to someone who ships services, in one paragraph. If you cannot connect the topic to a production failure, incident or invoice, it does not belong in this course.
- **Intuition** — the mental model, before any notation. For ML/DL topics this is the load-bearing part.
- **Topics** — the full coverage. `[ext]` marks extension material: taught as reading, not examined at the gate.
- **You build** — what lands in your repo.
- **Prove it** — the check you must pass out loud, without notes.

Notation appears only after the intuition, and every formula is followed by the code that computes it.

---

## Index

| # | Week · Day | Module | Lesson |
|---|---|---|---|
| 0 | 0 | 3 (3.01–3.03) | Environment, local model, repository |
| 1 | 1 · 1 | 1 | Generalization, metrics and thresholds |
| 2 | 1 · 2 | 1 | Leakage, ranking metrics and proving a delta |
| 3 | 2 · 1 | 2 | Tensors, autograd, backpropagation |
| 4 | 2 · 2 | 2 | Losses, softmax, layers, embeddings |
| 5 | 3 · 1 | 2 | Optimizers, a real training loop, the failure lab |
| 6 | 3 · 2 | 2 | Memory accounting, roofline, precision |
| 7 | 4 · 1 | 3 | API design for AI services |
| 8 | 4 · 2 | 3 | Streaming endpoints |
| 9 | 5 · 1 | 3 | Async Python at production level |
| 10 | 5 · 2 | 3 | Containers and testing LLM applications |
| 11 | 6 · 1 | 4 | Byte-pair encoding, from merges to bytes |
| 12 | 6 · 2 | 4 | Special tokens, chat templates, token economics |
| 13 | 7 · 1 | 4 | Attention: Q, K, V and the causal mask |
| 14 | 7 · 2 | 4 | Attention cost: quadratic time, KV bytes, GQA |
| 15 | 8 · 1 | 4 | The transformer block |
| 16 | 8 · 2 | 4 | Positional encoding, RoPE, and counting parameters |
| 17 | 9 · 1 | 4 | Decoding and sampling |
| 18 | 9 · 2 | 4 | Constrained decoding and structured output |
| 19 | 10 · 1 | 4 | MoE, long context, pretraining and scaling laws |
| 20 | 10 · 2 | 4 | Alignment, LoRA, and prompt vs retrieve vs fine-tune |
| 21 | 11 · 1 | 5 | Prefill, decode and the KV cache |
| 22 | 11 · 2 | 5 | Batching, prefix caching, quantization, speculation |
| 23 | 12 · 1 | 5 | Embeddings, chunking and index internals |
| 24 | 12 · 2 | 5 | Hybrid search, reranking and context construction |
| 25 | 13 · 1 | 5 | SQL to a working level |
| 26 | 13 · 2 | 5 | Ingestion, versioning and zero-downtime reindex |
| 27 | 14 · 1 | 5 | Agents: the loop and the tool schema |
| 28 | 14 · 2 | 5 | Rails, recovery and prompt injection |
| 29 | 15 · 1 | 5 | Eval types and deterministic checks |
| 30 | 15 · 2 | 5 | LLM-as-judge, the RAG triad and the CI gate |
| 31 | 16 · 1 | 5 | Secrets, PII, compliance, audit and tenancy |
| 32 | 16 · 2 | 5 | Observability and operations |
| 33 | 17 · 1 | 6 | The design protocol and problem framing |
| 34 | 17 · 2 | 6 | Data, labels, features — and rep 1 |
| 35 | 18 · 1 | 6 | Serving and latency budgets — and rep 2 |
| 36 | 18 · 2 | 6 | Monitoring, drift, retraining — and rep 3 |
| 37 | 19 · 1 | 6 | Rep 4 and the reference delta |
| 38 | 19 · 2 | 6 | Redraw from memory, mock review, the design document |
| 39 | 20 · 1 | 7 | Hardening |
| 40 | 20 · 2 | 7 | Documentation and demo day |
| P1–P13 | 4–16 | P | Platform lab — Kubernetes, vLLM, observability, load |
| E1–E10 | optional | E | Extension electives — model landscape, prompting, advanced RAG, documents, agent protocols, memory, frameworks, infrastructure, safety, patterns |
| A | — | — | Appendix A — case bank for the Module 6 reps |
| B | — | — | Appendix B — coverage map, MODULES.md → lesson |

---

# Lesson 0 — Environment, local model, repository

**Week 0 · 4 h · Module 3 §3.01–3.03 · no prerequisite**

**Engineer's frame.** Week 1 assumes a machine that already works. Every hour spent on a broken toolchain in week 6 is an hour stolen from attention. You also set a spend cap today, because the first uncapped agent loop is how people discover what a retry storm costs.

**Intuition.** Treat the course repository as a product: it builds, it tests, it runs in CI, from day one. Everything you add for twenty weeks lands in the same tree.

**Topics**
- Python 3.11+ and a managed environment (`uv`, or `venv` + pip), with a pinned lockfile
- The git we rely on: branches, a rebase-free workflow, commit hygiene, `.gitignore`
- Docker Desktop or an equivalent runtime; verifying it can build and run a container
- An editor with a debugger you can actually set a breakpoint in — Module 2 needs it
- Downloading open weights; where the model cache lives on disk and how large it gets
- Running a small model locally on CPU (llama.cpp / Ollama / vLLM CPU) behind an OpenAI-compatible API
- Provider accounts and API keys, in environment variables from the first call
- Setting a hard spend cap before your first API call
- The repository template: `src/`, `tests/`, `evals/`, `docs/`, `Makefile`
- A CI workflow running lint and tests on every push
- How weekly builds are submitted and how peer review is exchanged

**You build.** A repository that builds, tests and runs a hello-world service in CI, plus a local model answering a prompt over HTTP.

**Prove it.** `make test` passes on your machine and in CI; `curl` gets a completion from a model running on your own laptop; your provider dashboard shows a hard cap.

---

# Module 1 — Measuring things

**Week 1 · 10 h · Lessons 1–2**

The vocabulary for proving a system works. Not a statistics course — the part of statistics that decides whether your change shipped.

**Module build.** `recall@k`, `MRR` and `nDCG@k` from scratch, unit-tested against numbers you computed on paper. Then `compare.py`: two lists of per-case scores in, mean difference and a bootstrap 95% interval out.
**Module gate.** Someone says *"the new retriever scored 0.71 versus 0.68 on 200 queries."* Inside a minute you say whether it is real, what test you would run, and how many more queries you need.

---

## Lesson 1 — Generalization, metrics and thresholds

**Week 1 · Day 1 · Concept · covers §1.1, §1.2, §1.7, §1.8, §1.9**

**Engineer's frame.** You already know that a test suite passing locally does not mean the code works in production. A training curve is exactly that: a local run. This lesson is the vocabulary you need in the review where someone asks *how do you know it got better* — and every metric here reappears in week 15 when your eval harness decides whether a deploy ships.

**Intuition.** A model is a function fitted to examples. It can fail two ways: too rigid to represent the pattern (bias — like a constant-returning stub), or so flexible it memorized the examples including their noise (variance — like a lookup table keyed on the test inputs). Capacity is the dial between them. Everything called *regularization* is a way of spending capacity more carefully. A metric, then, is not a score — it is a summary that throws information away, and your job is to know which information each one throws away.

**Topics**
- Bias and variance, model capacity, overfitting and underfitting
- Regularization: L1, L2, early stopping — what each actually constrains
- Weight decay versus L2 in the loss, and why AdamW exists
- Why a training curve alone never tells you whether to ship
- Confusion matrix; precision, recall, F1 — and when each is the wrong summary
- ROC and AUC versus precision–recall curves under class imbalance
- The threshold as a product decision, not a library default
- Costing a false positive against a false negative before choosing the threshold
- Why accuracy lies on imbalanced data
- Calibration: what a confidence score does and does not mean
- Reliability diagrams — predicted probability against observed hit rate
- Platt scaling and isotonic regression **[ext]**
- Why a guardrail at threshold 0.9 fires far more often than "10% error" suggests
- Data quality: missing values — drop, impute, or treat missingness as signal
- Class imbalance: resampling and class weights
- Distribution shift between training data and production traffic, and how it is detected
- Awareness pass, one sentence each **[ext]**: logistic regression · decision trees and random forests · gradient boosting · naive Bayes · SVM · k-means · PCA
- Where gradient boosting still beats a neural network **[ext]**

**You build.** `metrics.py` — confusion matrix, precision, recall, F1 and a threshold sweep, from scratch. A reliability diagram over a set of scored predictions, plotted from your own bins.

**Prove it.** Given a fraud model at 99.2% accuracy on 0.5%-positive traffic, you say in one sentence why the number is meaningless, name the metric you want instead, and pick a threshold by naming the cost of each error type.

---

## Lesson 2 — Leakage, ranking metrics and proving a delta

**Week 1 · Day 2 · Lab · covers §1.3, §1.4, §1.5, §1.6**

**Engineer's frame.** Two of the most expensive mistakes in applied AI live here. One is leakage: an offline number that cannot survive production because the evaluation cheated. The other is celebrating noise: shipping on a 0.03 difference that a second sample would have reversed. Both are engineering bugs, not statistics, and both are found by looking at how the split was made.

**Intuition.** Leakage is any path by which information from the evaluation set reached the model — the same row twice, tomorrow's data in today's features, one user on both sides, or a public benchmark the pretraining corpus already ate. Ranking metrics differ from classification metrics in one way: position matters, so being right at rank 1 is worth more than being right at rank 40. And a difference between two systems is a random variable — so you report it with an interval, or you do not report it.

**Topics**
- Train/test contamination: the same row on both sides of the split
- Temporal leakage — using the future to predict the past; time-ordered splits
- Group leakage — the same user or document split across train and test
- Benchmark contamination in LLMs: the model already read the exam
- Why a random split flatters the score on time-ordered data
- What changes when you score a ranked list instead of a yes/no answer
- `recall@k` and `precision@k`
- MRR — how high the first correct answer sits
- `nDCG@k` — position-weighted gain, and what it penalizes that recall does not
- Choosing k for a retrieval stage that feeds a generator
- Why a retriever with better `recall@50` can still make a worse system
- Similarity, pointer only: cosine, dot product and L2, and when their rankings agree
- Why normalization decides whether cosine and dot product disagree — full treatment in Lesson 23
- Bootstrap confidence intervals from your own results
- Paired tests on the same evaluation set, and why pairing matters
- Sample size: how many cases you need to detect a given delta
- Multiple comparisons when you test twelve prompt variants
- Reporting a result honestly: effect size with an interval, never a bare number

**You build.** `recall_at_k`, `mrr`, `ndcg_at_k` from scratch, unit-tested against values you computed by hand on paper. Then `compare.py`: two lists of per-case scores in, mean difference and a bootstrap 95% interval out. Then a leakage demo — the same dataset scored under a random split and a time-ordered split, with the gap between them printed.

**Prove it.** The module gate: 0.71 versus 0.68 on 200 queries. Real or noise, which test, how many more queries.

---

# Module 2 — Deep learning core

**Weeks 2–3 · 28 h · Lessons 3–6 · prerequisite: Module 1**

You implement backpropagation, then learn to count bytes. The second half is the part every later module leans on.

**Module build.** A micrograd-style autograd engine with an operator you add yourself. A training loop you wrote, that survives being killed mid-run — then pushed until it OOMs, recovered with gradient accumulation, then deliberately poisoned to produce a loss spike you diagnose and fix. Finally `memory_math.py`, checked against a published model's real numbers.
**Module gate.** On a whiteboard, in under three minutes: KV-cache bytes for a 7B model at batch 32, 4k context, GQA with 8 KV heads, bf16 — within 10%. And you diagnose an OOM, a corrupted batch and a too-high learning rate from a training log alone.

---

## Lesson 3 — Tensors, autograd, backpropagation

**Week 2 · Day 1 · Concept · covers §2.1, §2.2**

**Engineer's frame.** A tensor library is a small runtime: it records a graph, executes it on a device, and walks it backwards. You debug it the way you debug any runtime — by knowing what it allocated and when it frees. Almost every "CUDA out of memory" ticket is really a question about the backward pass, and you cannot answer it from the outside.

**Intuition.** A neural network is a composition of functions. Autograd records that composition as a graph while the forward pass runs, then applies the chain rule from the loss backwards, node by node. Two consequences fall straight out and they explain most of training: every intermediate activation must be *kept alive* until its gradient is computed, and the backward pass costs roughly twice the forward pass. Memory is not a mystery — it is the size of the graph you just built.

**Topics**
- Indexing, `reshape` / `view` / `permute`, broadcasting rules
- `einsum`, and reading a `(B, H, T, D)` shape without guessing
- Device placement, dtypes, `.to()` semantics
- Autograd: how the graph is recorded, `backward()`, leaf tensors
- `detach` versus `no_grad`, and when each is correct
- Debugging shape errors without printing shapes first
- The chain rule over a computation graph
- Cost and memory of the forward pass versus the backward pass
- Why every layer's activations must be kept until the backward pass
- Gradient checkpointing: recompute instead of store, and what it costs
- Why training memory grows with batch × sequence length × depth

**You build.** A micrograd-style scalar autograd engine — `Value`, forward ops, `backward()` — with one operator you add yourself and verify against a numerical gradient.

**Prove it.** You add `tanh` (or `exp`) to the engine from scratch, and your analytic gradient matches a finite-difference check to 1e-6. You state, without running anything, which tensors are still alive at the moment `backward()` is called.

---

## Lesson 4 — Losses, softmax, layers, embeddings

**Week 2 · Day 2 · Lab · covers §2.3, §2.4, §2.5**

**Engineer's frame.** Three of these details cause silent, non-crashing wrongness — the worst class of bug. Handing softmax output to `CrossEntropyLoss` trains a worse model without ever raising. Skipping initialization gives a loss that plateaus instead of erroring. And the embedding matrix is where a surprising share of a small model's parameters live, which decides what fits on your laptop.

**Intuition.** Cross-entropy asks *how surprised was the model by the correct answer*, and it takes logits — raw, unbounded scores — because the log and the exponential cancel into something numerically stable. Temperature is nothing but dividing those logits before the softmax; the same knob returns in Lesson 17 as a sampling control. Normalization exists because deep stacks amplify scale drift, and LayerNorm is chosen over BatchNorm because a language model's batch dimension is not a statistically valid population per position. An embedding layer is not a matrix multiply against a one-hot vector — it is a row lookup, which is why only the rows you touched get a gradient.

**Topics**
- MSE versus cross-entropy, and why cross-entropy for classification
- Softmax numerical stability and the log-sum-exp trick
- Logits versus probabilities, and why losses take logits
- Temperature as a rescaling of logits — the same knob you meet again in Lesson 17
- What quietly breaks when you hand `CrossEntropyLoss` softmax output
- Activations: ReLU, GELU, SiLU; dead ReLUs
- Initialization: Xavier and He, and what breaks without them
- BatchNorm versus LayerNorm versus RMSNorm
- Why transformers use LayerNorm and not BatchNorm — two concrete reasons
- Pre-norm versus post-norm, and why pre-norm won
- Dropout, and where it still appears
- `nn.Embedding` as a lookup table, not a one-hot matrix multiply
- Embedding parameters = vocabulary × hidden, and what share of a model that is
- Sparse gradients: only the rows you indexed receive one
- Weight tying between the input embedding and the output projection
- Why tying is a large win on a small model and nearly noise on a large one

**You build.** A numerically stable softmax and cross-entropy you wrote, matching the framework's to 1e-6 on adversarial inputs (logits at ±1e4). An MLP with initialization you chose, plus a run showing what happens without it. A parameter breakdown of a small model, embeddings called out as a percentage.

**Prove it.** Given a config with vocabulary 50k and hidden 768, you state the embedding parameter count and its share of the model, and say whether tying is worth it here and why.

---

## Lesson 5 — Optimizers, a real training loop, the failure lab

**Week 3 · Day 1 · Concept + lab · covers §2.6, §2.7, §2.8, §2.9**

**Engineer's frame.** This is the lesson that makes training logs readable. Every shape of loss curve you will ever be shown in a screenshot — the spike, the plateau, the NaN, the slow divergence — is produced deliberately here, by you, so that recognizing it later takes seconds instead of an afternoon.

**Intuition.** SGD steps downhill. Momentum remembers the direction so it stops zigzagging. RMSProp scales each parameter's step by how noisy that parameter's gradient has been. Adam is both at once, which is why it carries two extra tensors per parameter — and that is where a third of your training memory goes. Adam's memory of past gradients is also why a single corrupted batch keeps hurting for hundreds of steps after it is gone: the poison is in `m` and `v`, not in the batch.

**Topics**
- SGD → momentum → RMSProp → Adam → AdamW, and what each step added
- What `m`, `v`, `β1`, `β2` and `ε` do
- Optimizer-state memory: two extra tensors per parameter
- Learning-rate warmup and cosine decay; why warmup exists
- Gradient clipping, and how to read a loss spike
- `nn.Module`, parameter registration, `state_dict`
- Custom layers, initialization, weight sharing
- `Dataset`, `DataLoader`, collation, worker count
- Device management and pinned memory **[ext]**
- Checkpoint and resume that survives the process being killed
- Failure lab: raising batch size until the process dies; what an OOM looks like on MPS versus CPU
- Gradient accumulation — recovering the effective batch, and what it costs in wall-clock time
- Producing a loss spike on purpose: inputs scaled ×1000, randomized labels, a `NaN` sample
- Why a bad batch poisons Adam's `m` and `v` long after it is gone
- Guards that catch each case: clipping, skipping non-finite steps
- CNN awareness: convolution, pooling, receptive fields, the residual connection
- RNN awareness: the sequential bottleneck — what transformers replaced, and why

**You build.** A training loop with checkpoint and resume, proven by `kill -9` mid-run and a restart that continues at the right step and optimizer state. Then three deliberate failures — OOM, poisoned batch, learning rate too high — each with the log excerpt, the diagnosis and the guard that catches it.

**Prove it.** You kill your own training run and resume it with no loss discontinuity. Shown three unlabelled loss curves, you name each failure and its fix.

---

## Lesson 6 — Memory accounting, roofline, precision

**Week 3 · Day 2 · Lab · covers §2.10, §2.11, §2.12**

**Engineer's frame.** This is the most reused hour in the course. Every hardware conversation, every "will this fit", every capacity plan in Track P, and half of Module 6's system design reduces to counting parameters, bytes and bandwidth. Learn it as arithmetic you can do on a whiteboard, not as a table you look up.

**Intuition.** Training memory has exactly four tenants: parameters, gradients, optimizer states, activations. Three of them are fixed by the model; the fourth is the one you control with batch size and checkpointing. Inference has three: weights, KV cache, activations. Speed has a separate model entirely — a chip can do far more arithmetic per second than it can fetch bytes per second, so the question is always *am I waiting on math or on memory*. Prefill processes a whole prompt at once and saturates the math units. Decode produces one token at a time, re-reading the whole model to do it, and starves on bandwidth. Every serving decision in Module 5 descends from that single sentence.

**Topics**
- Training memory = parameters + gradients + optimizer states + activations
- The ~16 bytes-per-parameter figure for Adam in mixed precision, and where it comes from
- Activation memory as a function of batch × sequence × layers
- Activation checkpointing: spending compute to save memory
- Inference memory = weights + KV cache + activations
- The KV-cache formula, and what GQA does to it
- FLOPs versus bytes moved; arithmetic intensity
- The roofline model: compute-bound versus memory-bound
- The memory hierarchy — registers, shared memory, L2, HBM — bandwidth versus capacity
- **Why prefill is compute-bound and decode is memory-bandwidth-bound**
- Why batching raises decode throughput but barely helps decode latency
- fp32, tf32, fp16, bf16, fp8 — exponent bits versus mantissa bits **[ext]**
- Why bf16 beat fp16 for training; what loss scaling works around **[ext]**
- Mixed precision: autocast, master weights **[ext]**
- Reading a profiler trace: overhead-bound, compute-bound or memory-bound **[ext]**
- Data, tensor and pipeline parallelism — what each shards and what it costs in communication **[ext]**

**You build.** `memory_math.py`: given a config, it prints training memory split four ways, inference memory split three ways, and KV-cache bytes per sequence with an MHA/GQA/MQA switch. Validated against a published model's real numbers.

**Prove it.** The module gate: 7B, batch 32, 4k context, 8 KV heads, bf16 — KV-cache bytes on a whiteboard in under three minutes, within 10%.

---

# Module 3 — Production backend for AI services

**Weeks 4–5 · 16 h · Lessons 7–10 · Platform lab P1–P2 starts**

The layer between "it works in a notebook" and "it serves users". Taught before the model internals, because everything after this is a service.

**Module build.** An HTTP API with a streaming endpoint, correct cancellation and an error contract; an async fetcher that demonstrates the blocking-call collapse in measured numbers; a container; and a test suite that runs offline, without an API key, in under ten seconds.
**Module gate.** You explain what happens to fifty in-flight requests when someone drops a synchronous HTTP call into an async endpoint, name the fix, and describe what a container must do when Kubernetes sends `SIGTERM` mid-stream.

---

## Lesson 7 — API design for AI services

**Week 4 · Day 1 · Concept · covers §3.1**

**Engineer's frame.** LLM endpoints break the assumptions a normal REST API is designed around: responses take thirty seconds, cost money per call, are non-deterministic, and are frequently retried by clients. Every rule below exists because one of those four properties turned a normal design into an incident.

**Intuition.** The contract is the product. A caller needs to know, without reading your code: what a failure looks like, whether a retry is safe, how to page through results while writes are happening, and what will still be true after your next deploy. Idempotency keys matter far more here than in CRUD, because a retried generation is a duplicated bill.

**Topics**
- Resource naming, HTTP verbs, and status codes that mean what they say
- One error contract for every failure: machine-readable code plus human message (RFC 9457)
- Pagination: cursor versus offset, and why offset breaks under concurrent writes
- Versioning: URL versus header, deprecation windows, additive-change discipline
- Idempotency keys, so a retry does not run the job twice
- Long-running work: 202 plus polling or a webhook, never a four-minute request
- Request validation at the boundary; OpenAPI as the contract, checked into the repo

**You build.** An HTTP API with one error contract used by every failure path, cursor pagination, an idempotency key on the generation endpoint, and an OpenAPI document committed to the repo.

**Prove it.** You issue the same POST twice with the same idempotency key and show one job, one bill. You show what your API returns for a validation failure, an upstream 429 and an internal error — three different codes, one shape.

---

## Lesson 8 — Streaming endpoints

**Week 4 · Day 2 · Lab · covers §3.2**

**Engineer's frame.** Streaming is where AI services leak money and latency. A proxy silently buffering your stream destroys time-to-first-token without any error appearing anywhere. A disconnected client whose generation keeps running is pure spend. And once you have sent `200 OK`, you have no status code left for the error that happens at token 400 — you need an in-band channel you designed in advance.

**Intuition.** SSE is a text protocol over one long-lived HTTP response: named fields, blank-line framing, nothing more. It is chosen over WebSocket because the traffic is one-way and it survives ordinary HTTP infrastructure. The hard parts are not the format — they are cancellation, buffering, and errors after the headers are gone.

**Topics**
- The SSE wire format: `data:`, `event:`, `id:`, blank-line framing
- Why SSE rather than WebSocket for one-way token streams
- Client disconnect and cancellation — stop generating, stop paying
- Proxies that buffer your stream, and how time-to-first-token dies there
- Errors after the 200 has already been sent: in-band error events
- Parsing structured output progressively while it streams
- Heartbeats, idle timeouts, and a slow client applying backpressure

**You build.** A streaming endpoint with in-band error events, heartbeats and cancellation that actually stops upstream generation — proven by a log line, not by hope. Plus a progressive JSON parser that consumes partial output.

**Prove it.** You disconnect a client mid-stream and show the upstream call being cancelled within one token. You show measured TTFT with and without a buffering proxy in front.

---

## Lesson 9 — Async Python at production level

**Week 5 · Day 1 · Concept + lab · covers §3.3**

**Engineer's frame.** The single most common production failure in Python AI services: one synchronous call — a `requests.get`, a blocking SDK, a `time.sleep` — inside an async endpoint. It does not error. It does not appear in a trace. It just stalls every other request on that worker, and the graph looks like the upstream got slow.

**Intuition.** The event loop is a single thread running ready callbacks. `await` is a yield point: it hands control back so other work can run. Code that never yields owns the thread until it finishes — so one blocking call is a global pause, not a local one. Everything else here is bounding: timeouts at each layer, a semaphore for concurrency, a bounded queue as backpressure, a retry budget so failure does not amplify into a stampede.

**Topics**
- The event loop; coroutine versus task; what `await` actually yields
- **One blocking call stalls every request on that worker** — and the fix
- Structured concurrency: `TaskGroup`, cancellation semantics, `asyncio.timeout`
- Concurrency limits with a semaphore; bounded queues as backpressure
- Connection pooling and keep-alive: one shared client, not one per request
- Timeouts at every layer — connect, read, total — plus the retry budget on top
- Retry with exponential backoff and jitter
- GIL versus asyncio versus processes: which problem each solves
- Uvicorn workers, and why a `def` endpoint in FastAPI runs in a thread pool

**You build.** An async fetcher with a shared client, a semaphore, layered timeouts and jittered backoff. Then the demonstration: p50 and p99 across fifty concurrent requests, with and without one blocking call inside the handler — the collapse in numbers.

**Prove it.** The module gate question: fifty in-flight requests, one synchronous call added. What happens, why, and the two ways to fix it.

---

## Lesson 10 — Containers and testing LLM applications

**Week 5 · Day 2 · Lab · covers §3.4, §3.5**

**Engineer's frame.** Two things every later week depends on. The container is what Track P deploys from week 8 onward — it must handle `SIGTERM` mid-stream, because Kubernetes will send one. The test suite is what lets you refactor an agent in week 14 without paying an API bill or waiting a minute per run.

**Intuition.** Mock at the HTTP boundary, not at your own wrapper. Mocking your wrapper tests your mock; mocking the transport tests the code that parses the response, retries the 429 and handles the truncated stream — which is where the bugs are. And never assert exact model text: assert structure, invariants and behaviour on failure paths. Where assertions stop, evaluation begins — that is Lesson 29, and both run in CI.

**Topics**
- Multi-stage builds; a runtime image without the build toolchain
- Layer caching: dependencies before source, or every edit rebuilds the world
- Pinned dependencies and reproducible installs
- Non-root user, `HEALTHCHECK`, and signal handling so `SIGTERM` drains cleanly
- Configuration by environment; model weights as a volume, never a layer
- pytest in practice: fixtures, `parametrize`, `monkeypatch`, markers
- Testing async code and an ASGI app in-process, with no live server
- **Mocking the provider at the HTTP boundary**, not at your own wrapper — and why that difference matters
- Recorded fixtures for real responses, including streamed chunk sequences
- A scripted fake model to drive agent branches: tool error, infinite loop, injected instruction
- **What you must never assert:** exact model text. Assert structure and invariants
- Failure-path tests: timeout, 429, truncated stream, cancelled client
- Where pytest stops and evaluation begins — and why both run in CI

**You build.** A multi-stage image running as non-root with a healthcheck and a `SIGTERM` handler that drains in-flight streams. A test suite with HTTP-boundary mocks, recorded streamed fixtures, a scripted fake model, and failure-path tests — running offline, with no API key, in under ten seconds.

**Prove it.** `docker run`, then `docker stop` mid-stream: the in-flight request finishes or closes cleanly, and the process exits without being killed. `pytest` passes with the network disabled.

---

# Module 4 — LLM internals: build a GPT from scratch

**Weeks 6–10 · 62 h · Lessons 11–20 · book: Raschka, *Build a Large Language Model (From Scratch)***

The heaviest module. Type every line; do not read it. By the end there is nothing in a forward pass you have not implemented.

**Module build.** A BPE tokenizer trained on your own corpus with a cross-language token-cost table. Causal multi-head attention with a GQA switch that prints cache size in bytes. A full GPT, plus `param_count(config)` reproducing a published model exactly. Top-k and top-p sampling, then a JSON-reliability experiment over 50 runs with and without schema constraints. A LoRA fine-tune on a task where prompting demonstrably fails, reported with before/after accuracy, adapter size and training time.
**Module gate.** From an empty file you write causal multi-head attention from memory and it runs. Given a model config you compute the parameter count on paper to within 2%. Given a business problem you argue prompt versus retrieval versus fine-tuning in two minutes with cost, latency and maintenance for each.

---

## Lesson 11 — Byte-pair encoding, from merges to bytes

**Week 6 · Day 1 · Concept + lab · covers Module 4 week 6, part 1**

**Engineer's frame.** The tokenizer is the boundary between your data and the model, and it is the layer that decides your bill. It is also the source of a whole family of bugs that look like model bugs: a prompt that works in English and fails in Georgian, a context window that mysteriously holds half as much code as prose, an off-by-one at a sequence boundary.

**Intuition.** BPE is compression with a fixed vocabulary budget. Start from bytes, repeatedly find the most frequent adjacent pair, merge it, record the merge. Training produces an ordered merge list; encoding replays that list on new text. Starting from bytes rather than characters is what makes it total — any input encodes, including emoji, broken UTF-8 and languages absent from the training corpus. They just cost more tokens.

**Topics**
- Byte-pair encoding: training the merges, then encoding with them
- The merge list as an ordered program, and why encoding must replay it in order
- Why byte-level, and what happens to characters outside the vocabulary
- Vocabulary size as a trade: fewer tokens per document versus a larger embedding matrix (Lesson 4's arithmetic, applied)
- Pre-tokenization and the regex split pattern, and what it prevents
- Round-trip guarantees: `decode(encode(x)) == x`, and where naive implementations break it

**You build.** A BPE tokenizer trained on your own corpus — trainer, encoder, decoder — with a round-trip property test over adversarial inputs (emoji, mixed scripts, code, whitespace runs).

**Prove it.** Your tokenizer round-trips a file containing four scripts and a code block, byte-identical. You explain the merge list's ordering requirement in one sentence.

---

## Lesson 12 — Special tokens, chat templates, token economics

**Week 6 · Day 2 · Lab · covers Module 4 week 6, part 2**

**Engineer's frame.** Special tokens and chat templates are where working code silently produces worse output. A missing BOS, a doubled EOS, a template applied twice — none of these raise, all of them degrade quality, and none appear in a diff review unless you know to look. On the money side: the same paragraph in English, Georgian and Python costs three different amounts, and that ratio is a line item in your capstone's cost model.

**Intuition.** A chat model does not see roles — it sees one flat token sequence with delimiter tokens the model was trained to recognize. The template is the code that flattens your message list into that sequence. If your template disagrees with the one used in training by even one token, you are prompting a slightly different model than the one they aligned.

**Topics**
- Special tokens: what makes them special, and why they must be unsplittable
- Chat templates, and the BOS/EOS bugs everyone hits
- Applying a template twice, and how to detect it
- Token counting: vocabulary size, and the cost penalty on non-English text and code
- What tokenization does to a context window and to a bill
- Comparing your tokenizer against a production one on the same text
- Token boundaries as a source of truncation bugs at the context edge

**You build.** A token-cost table: the same content in English, one non-English language and code, measured with your tokenizer and with a production tokenizer — tokens, ratio, and cost per 1M at a real provider price.

**Prove it.** You state your corpus's cost penalty for non-English text as a multiple, and show the exact token sequence your chat template produces, delimiters included.

---

## Lesson 13 — Attention: Q, K, V and the causal mask

**Week 7 · Day 1 · Concept · covers Module 4 week 7, part 1**

**Engineer's frame.** This is the gate lesson of the course: from an empty file, from memory, running. Everything after week 7 — KV caches, GQA, PagedAttention, prefix caching, long-context scaling — is a modification of the thirty lines you write here.

**Intuition.** Every position emits a query (*what am I looking for*), a key (*what do I offer*) and a value (*what I would contribute*). Score every query against every key, normalize the scores into weights, take the weighted sum of values. That is the whole mechanism. The √d divisor keeps the dot products from growing with dimension and driving softmax into saturation, where gradients die. The causal mask sets future positions to −∞ before the softmax so position *t* cannot read *t+1* — remove it and your training loss looks wonderful and your generation is worthless, because the model learned to copy an answer it will not have at inference.

**Topics**
- Q, K and V; scaled dot-product attention and why we divide by √d
- The `(B, H, T, D)` shape walk, one operation at a time
- Causal masking, and what breaks without it — with the loss curve to prove it
- Multi-head attention: splitting, projecting, concatenating
- Why heads are a reshape, not a loop
- The output projection, and where it sits in the parameter count

**You build.** Causal multi-head attention from scratch, with an explicit shape assertion at every step. Then the ablation: train the same tiny model with and without the mask, and show both loss curves and both generations.

**Prove it.** Empty file, no references: causal MHA that runs and produces the right shapes. You explain √d and the mask without hedging.

---

## Lesson 14 — Attention cost: quadratic time, KV bytes, GQA

**Week 7 · Day 2 · Lab · covers Module 4 week 7, part 2**

**Engineer's frame.** "Attention is expensive" is not an engineering statement. "This configuration moves 640 KB per generated token, so at 40 tokens/sec I need 25 MB/s per stream and the cache for 32 concurrent users does not fit" is. Lesson 6's byte-counting meets attention here, and the result is the number that decides GQA, quantization and batch size in Module 5.

**Intuition.** Attention cost has two separate stories. Compute is quadratic in sequence length — every position attends to every earlier position. Memory during generation is different: the KV cache stores keys and values for every past position, every layer, every head, and decode re-reads that cache for each new token. So decode speed is a bandwidth problem, and GQA attacks it directly by having several query heads share one KV head — cutting cache bytes by the sharing ratio, with a small quality cost.

**Topics**
- Cost: quadratic time and quadratic attention memory without a fused kernel
- Where the quadratic term actually hurts, and where it does not
- MQA, GQA and MLA — and what each does to KV-cache size
- The sharing ratio as a direct divisor on cache bytes
- Reading attention cost in bytes moved per generated token, not in adjectives
- FlashAttention as an IO-aware algorithm that never materializes the full matrix **[ext]**
- Tiling and the memory hierarchy, connected back to Lesson 6 **[ext]**

**You build.** A GQA switch on your attention module (`n_kv_heads`), plus a `cache_bytes(config, batch, seq)` function that prints cache size for MHA, GQA and MQA side by side, and the per-token bytes moved during decode.

**Prove it.** For your own config you state the cache bytes at batch 32 / 4k context under MHA and GQA-8, and the ratio between them — matching your `memory_math.py` from Lesson 6.

---

## Lesson 15 — The transformer block

**Week 8 · Day 1 · Concept + lab · covers Module 4 week 8, part 1**

**Engineer's frame.** The block is the unit that repeats. Once you can draw it and count its parameters, a model config stops being a configuration file and becomes an arithmetic problem you can do in a design review.

**Intuition.** A block does two things in sequence: attention lets positions exchange information; the feed-forward network processes each position independently, and it is where most parameters and most compute live. Residual connections make the block a correction to the stream rather than a replacement of it — which is why depth trains at all. Pre-norm (normalize before the sublayer, add the residual after) keeps the residual path clean, so gradients reach the bottom of a deep stack without a warmup cliff. Post-norm sits on the residual path and needs careful warmup. That is why pre-norm won.

**Topics**
- The transformer block: attention, feed-forward, residuals, norms
- Pre-norm versus post-norm inside a real block, and the training behaviour of each
- The residual stream as the model's working memory
- SwiGLU and the feed-forward expansion factor
- Why the FFN holds roughly two-thirds of the non-embedding parameters
- Assembling blocks into a model: embedding, N blocks, final norm, output head

**You build.** A complete GPT — embedding, stacked blocks, final norm, tied output head — that trains on your Lesson 11 tokenizer's output and generates text.

**Prove it.** You draw the block from memory with the residuals in the right places, and say which line moves if you switch pre-norm to post-norm.

---

## Lesson 16 — Positional encoding, RoPE, and counting parameters

**Week 8 · Day 2 · Lab · covers Module 4 week 8, part 2**

**Engineer's frame.** Position is the one thing attention has no opinion about — the mechanism is order-invariant by construction. How position is injected determines whether a model can be extended past its training context, which is exactly the long-context question in Lesson 19 and the "why did quality fall off a cliff at 9k tokens" ticket in production.

**Intuition.** Absolute encodings add a position vector to the embedding — simple, and they run out at the trained length. Learned encodings are the same with a parameter table, and extend even worse. RoPE takes a different route: it rotates the query and key vectors by an angle proportional to position, so the dot product between them depends on their *relative* distance. That is why it generalizes better and why scaling tricks in Lesson 19 have something to act on — you can change the rotation frequency after training. ALiBi does a similar job with a distance-proportional bias on the scores.

**Topics**
- Positional encoding: absolute → learned → RoPE → ALiBi
- Why attention is permutation-invariant without it
- RoPE specifically — the rotation, the frequency schedule, and why it dominates current models
- Where RoPE is applied (Q and K only, not V) and what that implies
- ALiBi as the linear-bias alternative
- Where the parameters actually sit, and counting them from a config
- `param_count(config)`: embeddings, attention projections, FFN, norms, head
- Reading a reference implementation and listing how yours differs

**You build.** RoPE in your attention module. `param_count(config)` that reproduces a published model's parameter count exactly. A written diff between your implementation and a reference implementation, line by line, with a reason for each difference.

**Prove it.** Given an unseen config, you compute total parameters on paper to within 2%, and name which term dominates.

---

## Lesson 17 — Decoding and sampling

**Week 9 · Day 1 · Concept + lab · covers Module 4 week 9, part 1**

**Engineer's frame.** Decoding parameters are the cheapest quality lever you own and the one most often set by copy-paste. They are also the source of the *"but it worked yesterday"* class of bug — and of the recurring question from every stakeholder: why is `temperature=0` not deterministic?

**Intuition.** The model outputs a distribution over the vocabulary at every step. Decoding is the policy that turns a distribution into a token. Greedy takes the argmax. Temperature rescales logits before the softmax — the same operation from Lesson 4 — flattening or sharpening the distribution. Top-k keeps a fixed count of candidates; top-p keeps a variable count covering a probability mass, which adapts to how confident the model is; min-p thresholds relative to the peak. And `temperature=0` is still not bit-identical because the arithmetic underneath is not: batching changes reduction order, kernels change with shapes, floating-point addition is not associative, and ties break differently.

**Topics**
- Greedy decoding and beam search; why chat models rarely use beam
- Temperature, top-k, top-p, min-p — what each cuts from the distribution
- Combining them, and the order in which they are applied
- Stop sequences, maximum tokens, and repetition controls
- Why `temperature=0` is still not bit-identical across runs
- Sampling as a knob on a latency/quality/cost trade, not a personality setting

**You build.** Top-k, top-p and min-p sampling implemented on your own model, with a distribution visualization at each stage showing exactly what was cut. Plus a determinism experiment: the same prompt at `temperature=0`, 20 runs, differences counted.

**Prove it.** You explain, to a non-specialist, why `temperature=0` gave two different answers — in under a minute, correctly.

---

## Lesson 18 — Constrained decoding and structured output

**Week 9 · Day 2 · Lab · covers Module 4 week 9, part 2**

**Engineer's frame.** Your service needs JSON that parses, every time, because a downstream system consumes it. Prompt engineering gets you to about "usually". Constrained decoding gets you to "by construction". Knowing which one you are relying on is the difference between a retry loop and a guarantee.

**Intuition.** At each step you have a distribution over the whole vocabulary. A grammar or JSON schema tells you which tokens could legally come next given what has been emitted. Mask the rest to −∞ before sampling and illegal output becomes impossible — not unlikely. The cost is a state machine tracking the parse position, and some quality loss when the mask forbids the token the model actually wanted. Better prompting stops helping past a certain point because prompting shifts probability mass; it never removes it.

**Topics**
- Constrained decoding: JSON schema, grammars, logit masking
- The parser state machine, and what it must track between tokens
- Measuring JSON reliability: prompt-only versus schema-constrained, counted over many runs
- Why better prompting stops helping past a certain point
- Failure modes of constraints: schema-valid but semantically wrong, and truncation at the token limit
- Streaming structured output while constrained — connecting to Lesson 8's progressive parser

**You build.** The JSON reliability experiment: 50 runs, prompt-only versus schema-constrained, parse-success rate and schema-validity rate reported with the interval from Lesson 2.

**Prove it.** You report both rates with a confidence interval, and name one failure that constrained decoding does *not* prevent.

---

## Lesson 19 — MoE, long context, pretraining and scaling laws

**Week 10 · Day 1 · Concept · covers Module 4 week 10, parts 1–2**

**Engineer's frame.** You will not pretrain a model. You will constantly make decisions that depend on understanding what pretraining did: why an MoE model with fewer active parameters is harder to serve than a dense one of the same speed, why a "128k context" model degrades at 40k, and why a vendor's model card lets you predict its cost profile.

**Intuition.** A mixture of experts replaces one large FFN with many, and a router that sends each token to a few of them. FLOPs per token drop because only the selected experts run — but *all* experts must be in memory, so serving cost is set by total parameters while speed is set by active parameters. That gap is the whole operational story. For long context: RoPE's frequencies were fitted to a training length, so extending the window means changing those frequencies — position interpolation and YaRN are principled ways to do that, and the degradation you see past the trained length is what happens when it is not done.

**Topics**
- Mixture of experts: routing, top-k experts, the load-balancing loss
- Active versus total parameters — why an MoE with fewer FLOPs can be harder to serve
- Expert imbalance as a production failure mode
- Long context: RoPE scaling, position interpolation, YaRN
- Why a "128k model" can degrade long before 128k
- Pretraining (read, not run): the objective, the data pipeline, deduplication, contamination
- Scaling laws and compute-optimal budgets: what they let you decide before spending money
- Checkpointing, resume and failure recovery at scale **[ext]**

**You build.** A short written analysis of two published model cards — one dense, one MoE — comparing active parameters, total parameters, memory to serve, and the deployment consequence of each.

**Prove it.** You explain why an MoE that is cheaper per token can be more expensive to run, using memory numbers.

---

## Lesson 20 — Alignment, LoRA, and prompt vs retrieve vs fine-tune

**Week 10 · Day 2 · Lab · covers Module 4 week 10, parts 3–4**

**Engineer's frame.** The most consequential decision you will make on an AI feature, and the one asked in every interview: do you prompt it, retrieve for it, or fine-tune it. This lesson gives you the numbers to answer with — cost, latency, adapter size, training time and maintenance burden — instead of a preference.

**Intuition.** A pretrained model predicts text. SFT teaches it to follow the shape of an instruction — same objective, curated data, with the loss masked on the prompt so it learns to *answer*, not to reproduce questions. RLHF then optimizes against a reward model trained on human *comparisons*, because people rank reliably and score absolutely very badly. DPO removes the separate reward model and optimizes preferences directly — simpler, fewer moving parts, slightly different behaviour. LoRA is the practical layer: freeze the weights, train a low-rank update beside them, ship a small adapter. Fine-tuning teaches *behaviour and format*; retrieval supplies *facts*. Confusing those two is the most common and most expensive mistake in the field.

**Topics**
- Supervised fine-tuning: dataset format, chat templates, masking the loss on the prompt
- RLHF: reward model then policy optimization; what each stage buys
- Why a reward model trains on comparisons instead of absolute scores
- DPO as the simpler alternative, and where it differs
- Constitutional AI; reasoning models and test-time compute
- RLVR — reinforcement learning from verifiable rewards, and why code and maths got there first
- Distillation: a large model's outputs as a small model's training set, and what it costs in coverage
- LoRA: the low-rank decomposition, rank and alpha, which modules to target
- QLoRA and multi-LoRA serving **[ext]**
- Cost, adapter size and training time as decision inputs
- **The decision framework: prompt versus retrieve versus fine-tune**
- Maintenance cost as the tiebreaker — what happens on the next base-model upgrade

**You build.** A LoRA fine-tune on a task where prompting demonstrably fails first — reported with before/after accuracy, adapter size in MB, training time and total cost. Plus a one-page decision write-up for the same task covering all three options.

**Prove it.** The module gate: a business problem, two minutes, prompt versus retrieval versus fine-tune, with cost, latency and maintenance for each.

---

# Module 5 — Applied LLM systems

**Weeks 11–16 · 88 h · Lessons 21–32 · book: Huyen, *AI Engineering***

Six weeks that produce your capstone. Each week adds a layer to the same service.

**Module build.** A retrieval service over a corpus you care about with hybrid search, reranking, citations and thirty hand-written golden pairs — reported as `recall@5` three ways. An agent with three tools and full rails, broken on purpose three ways. `ingest.py --since` that costs nothing on an unchanged corpus, and `reindex.py` with an alias swap and a timed restore. A forty-case eval harness with a validated judge that gates CI. Redaction on three paths, an audit log with a leak test, a one-page data map, OpenTelemetry traces in self-hosted Langfuse, and a runbook.
**Module gate.** The system runs end to end — incremental ingest, streamed answer with citations, redaction, audit record, trace, CI gate. You debug a deliberately broken run using only the trace UI. And you answer *"delete everything about this person"* with a real procedure, naming what takes minutes, what waits for the next rebuild, and what must be designed around from the start.

---

## Lesson 21 — Prefill, decode and the KV cache

**Week 11 · Day 1 · Concept · covers week 11, part 1**

**Engineer's frame.** Your service has two latency numbers that behave nothing alike, and treating them as one is why capacity plans are wrong. Time-to-first-token is a compute problem that scales with prompt length. Time-per-output-token is a bandwidth problem that barely moves with prompt length. Every knob in the next lesson trades one against the other.

**Intuition.** Prefill runs the whole prompt through the model once, in parallel — big matrix multiplies, math-bound, and it fills the KV cache. Decode then produces one token at a time: tiny matrix multiplies, but every step re-reads the model weights and the entire cache from memory. So prefill is compute-bound and decode is memory-bandwidth-bound — Lesson 6's roofline, now with a bill attached. The KV cache is the state that makes decode cheap in compute and expensive in memory, and how you allocate it decides how many users fit on one GPU.

**Topics**
- Prefill versus decode, and why they have different bottlenecks
- TTFT, TPOT, inter-token latency; throughput versus latency as separate goals
- Which user-visible complaint maps to which metric
- KV-cache sizing and eviction policies
- Fragmentation from contiguous allocation, and the waste it causes
- PagedAttention and the block table — virtual memory, applied to the cache
- Copy-on-write blocks for shared prefixes

**You build.** Measurements of your own stack: TTFT, TPOT and throughput at 1, 4 and 16 concurrent requests, with the knee identified and explained.

**Prove it.** You point at your own saturation curve and say which resource ran out and why, using bytes and bandwidth.

---

## Lesson 22 — Batching, prefix caching, quantization, speculation

**Week 11 · Day 2 · Lab · covers week 11, part 2**

**Engineer's frame.** These are the four levers a serving engine gives you, and each one buys a different thing. Confusing them produces the classic mistake: raising batch size to fix a latency complaint, and making it worse.

**Intuition.** Continuous batching schedules at the iteration level — a finished sequence leaves and a queued one joins on the very next step, instead of the whole batch waiting for the slowest member. It raises throughput enormously and does nothing for a single request's latency. Prefix caching skips prefill for a shared prefix, so it attacks TTFT directly — but only if the prefix is byte-stable, which is a prompt-layout decision you make in code. Quantization shrinks weights, cutting the bytes decode must move; it buys speed and capacity for some quality. Speculative decoding drafts several tokens cheaply and verifies them in one pass — the only lever here that improves single-stream latency, which is why it exists where batching cannot help.

**Topics**
- Continuous batching — iteration-level scheduling
- Why batching raises throughput but not single-request latency
- Prefix caching and what makes a prefix cacheable
- Prompt layout for cache stability: what must never move to the front
- Quantization for serving: GPTQ, AWQ, FP8 — what is lost and what is gained
- Weight-only versus activation quantization, and where quality goes
- Speculative decoding: why it helps latency where batching does not
- Draft-model choice and the acceptance rate that decides whether it pays
- Reading the request path from HTTP to token, naming the queue at each hop
- Diffusion LLMs as a different decode regime, and what it would change about this list **[ext]**
- On-device and edge deployment: quantized weights, NPU constraints, and what moves off the server **[ext]**

**You build.** A measured comparison on your own stack: throughput and TTFT with and without continuous batching; TTFT with and without a stable prefix. Plus a diagram of your request path with every queue named.

**Prove it.** Someone reports "generation is slow". You ask two questions, name the metric, and pick the correct lever — with the reason batching is or is not it.

---

## Lesson 23 — Embeddings, chunking and index internals

**Week 12 · Day 1 · Concept + lab · covers week 12, part 1**

**Engineer's frame.** Retrieval quality caps system quality: the generator cannot answer from a chunk it never received. Most RAG failures blamed on the model are chunking or indexing failures, and they are diagnosable — Lesson 2's `recall@k`, applied to your own retriever.

**Intuition.** An embedding maps text to a point in a space where nearby means related-as-the-training-objective-defined-it. Cosine measures angle; dot product measures angle and magnitude; they rank identically only when vectors are normalized — which is exactly the trap, because half of all embedding bugs are a missing normalization. Chunking decides what a "unit of retrievable meaning" is: too small and context is lost, too large and the embedding averages several topics into a vector that is near nothing. Index structures are pure engineering trade: flat is exact and slow, IVF partitions and probes some partitions, HNSW builds a navigable graph and is fast at high recall — for more memory and a build cost.

**Topics**
- Embeddings and similarity: cosine, dot product, L2, and the normalization traps
- When cosine and dot product rank identically, and when they disagree
- Chunking: size, overlap, and structure-aware splitting
- Chunking against document structure — headings, tables, code blocks
- Index internals: flat, IVF, HNSW — and the recall / latency / memory triangle
- HNSW parameters (`M`, `efConstruction`, `efSearch`) and what each trades
- Why `efSearch` is a query-time knob and `M` is not
- Recall measured against exact search as ground truth

**You build.** An index built three ways over the same corpus — flat, IVF, HNSW — with `recall@10` against exact search, p95 latency and memory, in one table. Plus a chunking ablation at three sizes.

**Prove it.** You name the parameter you would change to raise recall at fixed memory, and what it costs in latency.

---

## Lesson 24 — Hybrid search, reranking and context construction

**Week 12 · Day 2 · Lab · covers week 12, part 2**

**Engineer's frame.** Dense retrieval alone fails on exactly the queries enterprises care about: error codes, part numbers, names, rare acronyms. Hybrid retrieval fixes that, reranking fixes ordering, and context construction decides whether the generator can use any of it. This lesson produces the capstone's retrieval layer.

**Intuition.** BM25 matches tokens and is unbeatable on exact identifiers; embeddings match meaning and handle paraphrase. Fusing them covers both failure modes — the scores are not comparable, so you fuse by rank, not by value. A cross-encoder reranker reads query and document *together*, which is far more accurate and far too slow to run over the corpus — so it runs over the top 50 the cheap stage found. Then context construction: models attend unevenly across a long context, so ordering matters, duplicates waste budget, and every included chunk displaces another. Permission filtering must happen *before* retrieval, not after, or your top-k is silently made of documents this user cannot see.

**Topics**
- BM25 and keyword search; hybrid retrieval and score fusion
- Why fusion is done on ranks, not raw scores
- Reranking with a cross-encoder, and what it costs per query
- Contextual retrieval: prepending document-level context to each chunk before embedding, and the recall it buys
- Late interaction (ColBERT): per-token vectors and MaxSim, and why it sits between bi-encoder and cross-encoder in cost **[ext]**
- Choosing the candidate depth the reranker sees
- Context construction: ordering, deduplication, budget, lost-in-the-middle
- Citations that point at real spans
- Permission-aware retrieval — filtering before, not after
- The failure taxonomy: nothing retrieved, wrong chunk, right chunk but wrong answer
- Building a golden question set by hand, and why thirty of your own beats a public benchmark

**You build.** The retrieval service: hybrid search, cross-encoder reranking, span-level citations, permission filter applied pre-search. Thirty hand-written golden pairs. `recall@5` reported three ways — dense, hybrid, hybrid + reranker — each with an interval.

**Prove it.** Given a wrong answer, you classify it by the failure taxonomy in under a minute, using the retrieved chunks as evidence.

---

## Lesson 25 — SQL to a working level

**Week 13 · Day 1 · Concept + lab · covers week 13, part 1**

**Engineer's frame.** The customer's data is in a database, and your ingestion reads from it, your metadata filters run against it, and your eval results land in it. An AI engineer who cannot read `EXPLAIN ANALYZE` will eventually ship an ingestion job that does one query per document.

**Intuition.** An index is a sorted structure the planner may use — and it will not use it if you wrap the column in a function, because the index stores the column, not the function of it. `EXPLAIN ANALYZE` shows you the plan the database actually chose plus real timings, and the difference between a sequential scan and an index scan is usually the whole ticket. Window functions let you compute per-row values relative to a group without collapsing rows, which is how you deduplicate to the latest version per document — a query you will write in the next lesson.

**Topics**
- `SELECT`, `JOIN`, `GROUP BY`, `HAVING`; inner versus left join and the row-count trap
- CTEs and window functions: `ROW_NUMBER`, `LAG`, running totals
- Indexes: what they speed up, what they cost on write, why a function on a column kills one
- `EXPLAIN (ANALYZE)`: sequential scan versus index scan
- Reading the plan top-down, and finding the row-estimate that was wrong
- Keyset pagination; N+1 queries from application code
- Transactions and isolation **[ext]**

**You build.** A latest-version-per-document query using `ROW_NUMBER`, an N+1 in your own ingestion code found and fixed, and a before/after `EXPLAIN ANALYZE` for one query you made an index for.

**Prove it.** You read an `EXPLAIN ANALYZE` output aloud and say which node is the problem and why.

---

## Lesson 26 — Ingestion, versioning and zero-downtime reindex

**Week 13 · Day 2 · Lab · covers week 13, parts 2–3**

**Engineer's frame.** This is the operational half of RAG, and the half that produces incidents. A re-run that reprocesses an unchanged corpus burns money on every deploy. A deleted document whose chunks survive is a compliance finding. And an embedding-model upgrade is not a config change — it is a full rebuild, because vectors from two models do not share a space.

**Intuition.** Incremental ingestion is a diff, and a diff needs a stable identity and a change signal. Content hashes are the honest signal; modified timestamps lie (touched files, rewrites with identical content, clock skew). Stable chunk ids make upserts idempotent, so a re-run over unchanged input costs nothing. Deletion needs a tombstone because the source no longer tells you what existed. And a reindex should never mutate the live index: build the new one beside it, then swap an alias — one atomic operation, with the old index still there to roll back to.

**Topics**
- Batch reprocessing versus incremental updates, and when each is right
- Change detection: content hash, ETag, mtime — and why "modified date" lies
- Stable document and chunk ids; idempotent upsert so a re-run costs nothing
- Deletes and tombstones: the document is gone, the chunks are not
- Partial failure: retries, dead-letter queue, resumable runs
- Backfill without taking the live index down
- Parsing reality: PDFs, tables, HTML boilerplate, encoding
- Cost and throughput accounting per run
- Versioning the corpus, and pinning the embedding model version beside it
- **Changing the embedding model means a full rebuild** — embedding spaces do not mix
- Zero-downtime reindex: build under a new name, swap by alias
- Snapshots, backup, and a restore you actually perform
- Index-parameter changes that force a rebuild
- Retention and deletion, feeding Lesson 31
- Managed versus self-hosted versus vector-in-Postgres: the operational trade

**You build.** `ingest.py --since` with content-hash change detection, stable ids, idempotent upsert, tombstones on delete and a dead-letter queue — costing ~zero on an unchanged corpus, with the number printed. `reindex.py` performing an alias swap with no dropped queries. And a restore from snapshot, timed.

**Prove it.** You run `ingest.py` twice and show the second run's cost. You swap an index alias under live traffic with zero errors, and state your measured restore time.

---

## Lesson 27 — Agents: the loop and the tool schema

**Week 14 · Day 1 · Concept · covers week 14, part 1**

**Engineer's frame.** Most things sold as agents should be chains, and knowing when to refuse the loop is the senior judgement here. A chain is deterministic, testable and cheap. An agent is a loop with a model deciding control flow — you take that cost only when the number of steps genuinely cannot be known in advance.

**Intuition.** The loop is: model proposes an action, you execute it, you feed the result back, repeat until a termination condition. Everything hard follows from two facts — the model sees only what is in the context, and the context has a budget. So tool descriptions are prompts (the model chooses tools by reading them), tool results are untrusted input (Lesson 28), and context management is a memory-allocation problem: what to keep, what to summarize, what to drop.

**Topics**
- What separates an agent from a chain, and when the chain is the better answer
- Tool schema design: names, descriptions, parameter validation
- The tool description as a prompt the model actually reads
- Validating parameters before execution, and returning errors the model can act on
- The loop: plan → act → observe → repeat; termination conditions
- Context management: compaction, summarization, what to drop and when
- Keeping the tool result the model needs, not the tool result the API returned
- MCP as a tool-transport standard: servers, clients, and what it buys over hand-rolled tool wiring
- Durable execution: a loop that survives a process restart, and why checkpointed state beats an in-memory loop

**You build.** An agent with three tools — one of them your retrieval service — with validated parameters, explicit termination conditions and a context-compaction strategy you chose and can defend.

**Prove it.** You state, for your own use case, why it is an agent and not a chain — or you convert it to a chain and explain what you gained.

---

## Lesson 28 — Rails, recovery and prompt injection

**Week 14 · Day 2 · Lab · covers week 14, part 2**

**Engineer's frame.** An unbounded agent loop is an unbounded invoice, and a tool result is attacker-controlled input reaching your model's instruction channel. Both are engineering problems with engineering fixes. This is the lesson where you break your own agent on purpose, because the alternative is production breaking it for you.

**Intuition.** Rails are bounds: step limit, cost cap, wall-clock timeout, cycle detection on repeated state. Recovery is harder than it looks — a tool that fails loudly is easy, a tool returning *plausible garbage* is the dangerous case, because the model will happily build on it. And prompt injection reaches you through any text the model reads: a retrieved chunk, a web page a tool fetched, a file name. The defence is architectural — separate instruction from data, constrain what the model can invoke, validate on the way out — not a phrase in the system prompt.

**Topics**
- Rails: retries with backoff, step limits, cost caps, cycle detection
- Error recovery when a tool fails, returns garbage, or returns plausible garbage
- Distinguishing a retryable failure from a terminal one, in the loop
- Observability inside the loop: per-step tokens, cost and latency
- Prompt injection through tool output and retrieved text
- Why instructions in the system prompt are not a defence
- Breaking your own agent on purpose: a failing tool, an unbounded task, an injected instruction
- Guardrails as code: input filters, output validators, and where each belongs in the loop
- Red-teaming your own agent: a written attack list, run as tests

**You build.** Full rails on your agent — step limit, cost cap, cycle detection, backoff — plus per-step token/cost/latency logging. Then the break-it report: three deliberate failures, what happened, what caught it, and what you changed.

**Prove it.** You inject an instruction through a retrieved document and show your agent's behaviour with and without your mitigation, honestly — including what it does not stop.

---

## Lesson 29 — Eval types and deterministic checks

**Week 15 · Day 1 · Concept + lab · covers week 15, part 1**

**Engineer's frame.** You cannot ship changes to a non-deterministic system without a way to tell improvement from noise. This is Lesson 2 becoming infrastructure — and the reason your CI can block a deploy on quality rather than on tests alone.

**Intuition.** Evaluate at three levels for three different reasons: unit (does this component do its job — retrieval recall, tool-call correctness), component, end-to-end (does the whole answer hold up). Offline evals run against a fixed dataset and are repeatable; online evals run against real traffic and are true. Start with everything you can check *deterministically*, because deterministic checks are free, instant and unarguable — schema validity, citation presence, tool-call correctness, exact match where it applies. Only what remains needs a judge. And your golden set can be contaminated too: if you tune prompts against it forty times, it has become a training set.

**Topics**
- Eval types: unit, component, end-to-end; offline versus online
- Golden datasets versus synthetic datasets; contamination of your own eval set
- Keeping a held-out slice you look at rarely
- Deterministic checks: exact match, schema validation, tool-call correctness, citation presence
- Building the case set from real failures, not from imagination
- What each level catches, and what it cannot

**You build.** The deterministic half of a 40-case harness: schema validation, citation presence, tool-call correctness, plus the case-set structure and a held-out slice.

**Prove it.** You name which of your failures a deterministic check catches and which genuinely need a judge, with counts.

---

## Lesson 30 — LLM-as-judge, the RAG triad and the CI gate

**Week 15 · Day 2 · Lab · covers week 15, part 2**

**Engineer's frame.** A judge is a measuring instrument. Nobody deploys an uncalibrated instrument and reports its readings as fact — but that is exactly what happens when a team adds an LLM judge and starts quoting its scores. You validate the judge against your own labels first, and report the agreement number alongside every result.

**Intuition.** A judge is a model scoring outputs against a rubric, and it has known biases: position (prefers whichever came first), verbosity (prefers longer), self-preference (prefers its own family's output). You measure agreement against your own hand labels on a sample. If agreement is poor, fix the rubric, not the conclusion. The RAG triad decomposes an answer's quality into three checkable questions — is the answer grounded in the context, does it address the question, was the retrieved context relevant — which turns "bad answer" into a diagnosis pointing at retrieval or generation.

**Topics**
- LLM-as-judge: writing a rubric, choosing a scale, calibrating it
- Judge biases: position, verbosity, self-preference — and the mitigations for each
- **Validating the judge against your own labels**, and reporting agreement
- The RAG triad: faithfulness, answer relevance, context relevance
- Using the triad to localize a failure to retrieval or generation
- A regression suite in CI with a threshold that blocks a deploy
- Choosing the threshold so it catches regressions without blocking on noise
- Deciding whether a delta is real or noise — Lesson 2, applied
- Eval frameworks in practice: RAGAS for the triad, LangSmith and Langfuse for scored traces — and what you still write yourself
- Public benchmarks and leaderboards: what they measure, contamination, and why they do not replace your thirty cases
- Quality drift detection: tracking eval scores and score distributions over time, not just at merge

**You build.** The full 40-case harness with a validated judge — agreement against your own labels reported — running on one command and exiting non-zero when quality drops. Wired into CI.

**Prove it.** You show your judge's agreement number, then push a deliberately worse prompt and show CI going red for the right reason.

---

## Lesson 31 — Secrets, PII, compliance, audit and tenancy

**Week 16 · Day 1 · Concept + lab · covers week 16, parts 1–4**

**Engineer's frame.** These are the requirements that decide whether an enterprise can buy your system at all, and they are architectural: you cannot add data residency, deletion or tenant isolation late. The right-to-erasure question — *"delete everything about this person"* — is the module gate precisely because answering it honestly requires understanding every place data landed: the index, the cache, the traces, the logs, the fine-tuned weights.

**Intuition.** Redaction removes; pseudonymization replaces with a reversible token (still personal data, because the mapping exists); anonymization is irreversible and is the only one that leaves the regulation's scope. Detection is imperfect — regex has good precision and poor recall, NER is the reverse — so you measure your own false-negative rate by hand-labelling a sample rather than trusting a library's claim. Multi-tenancy has exactly two shapes, and the failure mode of a metadata filter is a forgotten `WHERE` clause returning another tenant's documents, which is why the leak test runs on every commit.

**Topics**
- Secrets: never in code, in a prompt, in a log, or in a trace
- Environment variables versus a secret manager; what each protects against
- Rotation without downtime; per-tenant and per-environment keys
- Scanning: pre-commit hooks and history scanning
- PII classes and special-category data
- Detection: regex versus NER, and the recall you actually get
- Redaction versus pseudonymization versus anonymization — and which one exits the regulation
- Where to apply it: before the provider, before the index, before logs and traces
- PII inside embeddings; PII inside eval datasets and recorded fixtures
- Measuring your own false-negative rate by hand-labelling a sample
- Controller versus processor; who is which in a vendor relationship
- Data Processing Agreements and subprocessors — calling a hosted API adds one
- International transfers, and why "EU region only" appears in tenders
- **Right to erasure versus a vector index, a cache and fine-tuned weights**
- Retention limits applied to traces and logs
- On-premise and air-gapped deployment: mirrored weights, private registries, offline flags
- Open weights versus hosted API as a consequence of the deployment, not a preference
- Supporting a customer whose data you are not allowed to see
- What an audit record must contain: actor, tenant, time, action, resource, outcome
- Audit logs separated from debug logs, with different retention
- Tenant isolation: collection-per-tenant versus metadata filter, and the failure mode of each
- **The cross-tenant leak test**, asserted on every commit
- RBAC versus ABAC, and why document-level permissions usually force attribute-based checks
- Propagating the caller's identity all the way to the retrieval filter, not just to the API edge
- AI governance: model and data inventories, risk classification, and the obligations that follow from it

**You build.** Redaction on three paths — before the provider, before the index, before logs. A measured false-negative rate from your own hand-labelled sample. An audit log with the required fields. A cross-tenant leak test in CI. A one-page data map: every place data lands, its retention, and its deletion path.

**Prove it.** The module gate: *"delete everything about this person."* Real procedure, naming what takes minutes, what waits for the next rebuild, and what must be designed around from the start.

---

## Lesson 32 — Observability and operations

**Week 16 · Day 2 · Lab · covers week 16, parts 5–6**

**Engineer's frame.** A user says "the answer was wrong yesterday afternoon". Without tracing, that is unanswerable. With it, it is: find the trace, read the retrieved chunks, see the tool call that failed, replay it, add it to the eval set. This lesson closes the loop from a complaint to a regression test — and adds the silent-failure case, where a provider updates a model and quality drops with no error anywhere.

**Intuition.** A trace is one request's causal tree: spans for retrieval, rerank, generation, each tool call, with tokens and cost attached. Sampling keeps the cost down — but you always sample errors and slow requests, because those are the ones you will be asked about. Latency budgets work like any other budget: assign milliseconds per stage in advance, then measure against the plan instead of arguing about impressions. Prompt caching turns prompt layout into a performance decision — anything that changes near the front invalidates everything behind it.

**Topics**
- Traces, spans, attributes, context propagation across async boundaries
- OpenTelemetry GenAI semantic conventions: model, tokens, cost, tool calls
- One trace spanning HTTP → retrieval → rerank → generation → tool call
- Sampling, and always sampling errors and slow requests
- Langfuse and LangSmith: sessions, traces, generations, scores
- Attaching eval scores and user feedback to the trace that produced them
- Masking PII before it reaches the tracing backend
- Self-hosted versus SaaS tracing, decided by the constraints in Lesson 31
- Debugging a production agent: complaint → trace → failing step → replay → regression case
- Latency budget across retrieval → rerank → generation → post-processing
- Prompt caching: what is cacheable, prefix stability, prompt layout
- Cost modelling; model routing and fallback chains
- Versioning prompts, models and indexes; rollback and shadow deploys
- Noticing a silent quality drop after a provider updates a model
- The runbook: alerts with thresholds, two incident playbooks, rollback steps
- The AI gateway: one egress point for routing, keys, quotas, retries and cost attribution
- Model routing in practice: a cheap default, an escalation rule, and a fallback chain across providers
- FinOps and token economics: cost per request, cost per resolved task, and the unit you actually report
- The LLMOps lifecycle: what is versioned, what is promoted, and what a rollback restores

**You build.** OpenTelemetry traces with GenAI conventions into self-hosted Langfuse, PII masked before export, eval scores attached to traces. A latency budget table with measured actuals. A runbook: three alerts with thresholds, two incident playbooks, rollback steps.

**Prove it.** The module gate's debugging half: given a deliberately broken run, you find the failing step using only the trace UI, and turn it into a regression case in your harness.

---

# Module 6 — System design and technical communication

**Weeks 17–19 · 40 h · Lessons 33–38 · prerequisite: Module 5**

Being right is half of it. This is the other half.

**Module build.** Four completed reps with delta notes, plus a two-to-four-page design document for one system you built.
**Module gate.** Drive one design end to end in fifty minutes: clarify, estimate memory and cache, draw request to token, deep-dive one component on request, and name three failure modes with mitigations.

---

## Lesson 33 — The design protocol and problem framing

**Week 17 · Day 1 · Concept · covers §6.1, §6.2**

**Engineer's frame.** A design conversation is a driving exercise, not a quiz. The reviewer is measuring whether they would trust you with an ambiguous problem and a budget. Most failures are not knowledge failures — they are forty minutes spent on the part the reviewer did not care about.

**Intuition.** Requirements first, numbers second, boxes third, depth last. Estimating before designing is what separates a plan from a wish: requests per second, bytes stored, cache size, GPU count — you already have all of this from Lesson 6 and Lesson 21. Then name the trade you made out loud, because an unstated trade reads as an oversight.

**Topics**
- Clarifying: users, scale, latency budget, and what "good" means
- Estimating before designing: requests per second, storage, memory
- Drawing the boxes, then going deep on the one your reviewer picks
- Stating trade-offs out loud and naming what you gave up
- Time management across a fifty-minute conversation
- Saying "I don't know" well
- Business goal → task: classification, ranking, retrieval, generation
- When rules beat a model
- Choosing the label, and what to do when the label does not exist yet
- Offline metric versus online metric, and the gap between them
- Defining "good enough" as a number before designing anything
- The answer frameworks: requirements → estimates → high-level → deep dive → failure modes → trade-offs

**You build.** A one-page protocol card you will use in every rep, and a written framing — goal, task type, metric, "good enough" number — for one case from the bank in the appendix.

**Prove it.** Given a vague product ask, you produce five clarifying questions and one number that must be established before any design starts.

---

## Lesson 34 — Data, labels, features — and rep 1

**Week 17 · Day 2 · Rep · covers §6.3 + rep 1**

**Engineer's frame.** Training/serving skew is the defect that passes every test and fails in production, and it is a *systems* bug: two code paths computing the same feature differently, a feature available offline but not at request time, a join that used the whole history. Every one of those is a thing you can spot in a diagram.

**Intuition.** Logging is a design decision, not an afterthought — what you fail to log today is what you cannot train or evaluate on next quarter. And a deployed model changes the data it will next be trained on: it decides what users see, so the feedback loop is a control system, not a dataset.

**Topics**
- Where training data comes from; logging as a design decision
- Label delay and feedback loops — the model changing its own training data
- Feature computation at training time versus serving time
- **Training/serving skew**, three ways it appears and the fix for each
- Leakage at the system level, not just in a dataset
- Position bias and presentation bias in logged interactions

**You do.** **Rep 1** — ten minutes of recall, forty minutes solo and spoken aloud, then a comparison against a reference solution and a written delta.

**Prove it.** In your rep's design you name where a feature is computed twice and how you would guarantee the two paths agree.

---

## Lesson 35 — Serving and latency budgets — and rep 2

**Week 18 · Day 1 · Concept + rep · covers §6.4 + rep 2**

**Engineer's frame.** "It needs to be under 300 ms" is not a requirement until it is split across stages with a number on each. That split is the artefact that makes the rest of the design decidable — it tells you whether the reranker fits, whether the cache is mandatory, and what has to be precomputed.

**Intuition.** Multi-stage serving is a funnel: a cheap stage over everything, an expensive stage over the survivors — recall from the cheap stage, precision from the expensive one. Caching has two forms with different invalidation rules: cache the prediction, or cache the features it was computed from. And when the model is unavailable, degrading gracefully to a worse-but-working answer beats an error page in nearly every product.

**Topics**
- Batch, online and streaming prediction — how to choose
- Splitting a p95 target across stages, with numbers
- Multi-stage serving: cheap filter, expensive model
- Caching predictions and caching features
- Fallbacks and graceful degradation when the model is down
- Load shedding and admission control under saturation

**You do.** **Rep 2**, same format, on a different case class from the bank. Plus a latency budget table for the rep's design, defended stage by stage.

**Prove it.** You state your p95 split and say which stage you would cut first under pressure, and what quality you lose by doing it.

---

## Lesson 36 — Monitoring, drift, retraining — and rep 3

**Week 18 · Day 2 · Rep · covers §6.5 + rep 3**

**Engineer's frame.** The worst failure mode of an ML system is not being down. It is being up, fast and wrong — and nothing in your infrastructure monitoring will tell you. This lesson is about the instrumentation that does.

**Intuition.** Data drift is the inputs changing; concept drift is the relationship between inputs and correct answers changing. They are detected differently — distribution comparison for the first, outcome tracking for the second — and only the second is directly about quality. Retraining triggers should be chosen deliberately: on a schedule, on measured drift, or on a performance drop, each with different cost and different failure modes.

**Topics**
- What to log: inputs, predictions, outcomes
- Data drift versus concept drift, and how each is detected
- Silent failure: the model is up, fast and wrong
- Retraining triggers: schedule, drift, or performance
- Shadow deploys, canaries and rollback
- Choosing the alert that would have caught your last incident

**You do.** **Rep 3**, same format. Plus a monitoring plan for the rep's design: three signals, their thresholds, and what each one is a proxy for.

**Prove it.** For one of your own systems you name the silent-failure mode and the single signal that would surface it.

---

## Lesson 37 — Rep 4 and the reference delta

**Week 19 · Day 1 · Rep · covers §6.6**

**Engineer's frame.** Reps only teach if the delta is written down. The comparison against a reference solution is where the learning lives — not in the drawing.

**Topics**
- Rep 4, fifty minutes, spoken aloud, on a case class you have not attempted
- The written delta: what the reference had that you missed, and why you missed it
- Classifying your own misses: knowledge, process, or time management
- The estimation numbers you should now know cold — tokens, bytes, bandwidth, cost per million

**You do.** **Rep 4** plus a consolidated delta log across all four reps, with the pattern in your own misses named.

**Prove it.** You name your own recurring failure mode across four reps and the countermeasure you will use in the gate.

---

## Lesson 38 — Redraw from memory, mock review, the design document

**Week 19 · Day 2 · Lab · covers §6.6 (redraw) + design document**

**Engineer's frame.** Redrawing a design a week later, from memory, is the test of whether you understood it or merely produced it. And the written design document is the artefact that travels — it is what a hiring manager, a staff engineer or a customer's architect reads when you are not in the room.

**Topics**
- Redrawing one earlier rep from memory a week later, then diffing against the original
- The whiteboard exercises: estimate a KV cache, size an index, split a latency budget, cost a run — under time
- The design document: problem, architecture, one defended decision, trade-offs, results
- Writing for a reader who will not ask you a follow-up question
- Mock review: driving a full fifty minutes with a peer as reviewer, then swapping

**You build.** A two-to-four-page design document for one system you actually built, plus one rep redrawn from memory with the diff attached.

**Prove it.** The module gate: fifty minutes, one design, end to end — clarify, estimate, draw request to token, deep-dive one component your reviewer picks, three failure modes with mitigations.

---

# Module 7 — Capstone and demo day

**Week 20 · 14 h · Lessons 39–40**

**Deliverables.** Running system · design document · eval report · runbook · recorded demo.

---

## Lesson 39 — Hardening

**Week 20 · Day 1 · Lab · covers §7.1**

**Engineer's frame.** The gap between a demo and a system is entirely in the failure paths. This day is spent on the paths nobody enjoys testing, because that is the difference the audience will notice when something goes wrong live.

**Topics**
- Closing the gaps your own eval suite found
- Failure paths verified: provider down, index missing, tool timeout, cancelled client
- Secrets, redaction and audit verified once more against the data map
- Load: know your own numbers at 1, 4 and 16 concurrent requests
- Rehearsing the rollback you documented in Lesson 32

**You build.** A verified failure-path checklist — each path triggered on purpose, each behaviour recorded — and a load table at 1, 4 and 16 concurrent requests.

**Prove it.** You kill the provider, the index and a tool, one at a time, and the system degrades in a way you predicted in advance.

---

## Lesson 40 — Documentation and demo day

**Week 20 · Day 2 · Demo · covers §7.2, §7.3**

**Engineer's frame.** The demo is fifteen minutes and it is the only part most people will see. One decision defended well beats a tour of every feature.

**Topics**
- Design document, two to four pages: problem, architecture, one defended decision, trade-offs, results
- Eval report: what you measure, on what data, with what threshold
- Runbook: three alerts, two incident playbooks, rollback steps
- README that lets someone else run it
- The demo: what it does, shown live
- One architectural decision you would defend under pressure
- One thing that went wrong and what it taught you
- The numbers: latency, cost, eval scores
- Questions from the room

**You build.** The full deliverable set: running system, design document, eval report, runbook, README, recorded demo.

**Prove it.** Fifteen minutes, live, then questions. Someone else clones your repo and runs it from the README alone.

---

# Track P — Platform lab

**Weeks 4–16 · 2 h per week · 26 h · Lessons P1–P13 · runs in parallel**

Each session is a short lesson plus a lab with an automated grader, so you know immediately whether it works.

**Track gate.** Your own capstone service deployed to the cluster, serving under load, on a Grafana dashboard you can read out loud.

| # | Week | Alongside | Topics |
|---|---|---|---|
| **P1** | 4 | Lesson 7–8 | **Cluster anatomy** — control plane, nodes, the scheduler. `kubectl` for real work: `describe`, `logs`, `exec`, `port-forward`. Reading why a pod is not running, from `describe` alone |
| **P2** | 5 | Lesson 9–10 | **Workloads** — Pods, Deployments, ReplicaSets, Services. Namespaces, labels and selectors. Your Module 3 container running in the cluster |
| **P3** | 6 | Lesson 11–12 | **Configuration** — ConfigMaps and Secrets, and what a Secret is *not*. Configuration by environment, carried over from Lesson 10 |
| **P4** | 7 | Lesson 13–14 | **Making it survivable** — liveness, readiness and startup probes, and what each one actually controls. Resource requests and limits; what happens at the limit (throttled versus killed). A persistent volume for the model cache |
| **P5** | 8 | Lesson 15–16 | **The inference engine** — vLLM with an OpenAI-compatible API on a CPU backend. Engine flags that matter: KV-cache space, max sequence length, batching |
| **P6** | 9 | Lesson 17–18 | **Engine on the cluster** — deploying the engine with probes and a volume; first real inference served by your own cluster |
| **P7** | 10 | Lesson 19–20 | **Packaging** — Helm: templates, values, releases, upgrades and rollbacks. One chart, three environments |
| **P8** | 11 | Lesson 21–22 | **Scale-out** — a request router in front of multiple engine replicas. Service discovery and session affinity, and why affinity interacts with prefix caching |
| **P9** | 12 | Lesson 23–24 | **Metrics** — Prometheus, ServiceMonitor, and scraping engine metrics |
| **P10** | 13 | Lesson 25–26 | **Dashboards** — Grafana for TTFT, queue depth, KV-cache usage, tokens per second. What each metric tells you when latency rises |
| **P11** | 14 | Lesson 27–28 | **Load** — load testing with k6, and a saturation curve you can point at |
| **P12** | 15 | Lesson 29–30 | **Autoscaling and the edge** — horizontal autoscaling, and why queue depth beats CPU as a signal. Ingress and TLS tuned for streaming: timeouts and buffering (the Lesson 8 failure, at the cluster edge) |
| **P13** | 16 | Lesson 31–32 | **Security and operations** — default-deny network policy, quotas, RBAC. SLOs and burn-rate alerts. Cost and capacity planning for a fixed hardware budget |

---

# Track E — Extension electives

**Optional · 2–4 h each · run any time after the listed prerequisite lesson**

The twenty-week spine covers one system, built deeply. These ten electives cover the surrounding landscape — the topics that come up in interviews and in the next project, but that would dilute the spine if forced into it. Each is a lesson with the same five parts, sized for one evening. Sources: the topic areas in [ai-system-design-guide](https://github.com/ombharatiya/ai-system-design-guide), mapped onto this course's structure.

| # | After | Elective |
|---|---|---|
| E1 | Lesson 20 | Model landscape, pricing and selection |
| E2 | Lesson 20 | Prompting and context engineering |
| E3 | Lesson 24 | Advanced retrieval: multimodal, agentic, and RAG at scale |
| E4 | Lesson 26 | Document processing and vision-LLM parsing |
| E5 | Lesson 28 | Agent protocols, multi-agent systems and computer use |
| E6 | Lesson 28 | Memory and state |
| E7 | Lesson 30 | Frameworks, orchestration and framework churn |
| E8 | Lesson 32 | Infrastructure: GPU clusters, gateways and FinOps |
| E9 | Lesson 32 | Safety: guardrails, red-teaming and governance |
| E10 | Lesson 38 | AI design patterns and anti-patterns |

---

## E1 — Model landscape, pricing and selection

**After Lesson 20 · 2 h**

**Engineer's frame.** "Which model" is a recurring engineering decision with a cost, a latency profile and a migration risk attached — not a preference. The landscape changes every quarter; the *method* for choosing does not.

**Intuition.** Read a model the way you read a database: capability class, context window, throughput, price per million in and out, and the operational shape (hosted, open weights, on-device). Then pick per task, not per company — a routing layer means the cheap model handles the 80% that is easy.

**Topics**
- Model taxonomy: frontier, mid-tier, small, open-weights, on-device — and what each class is actually for
- Reading a model card: context window, training cutoff, modalities, tool-use support
- Reasoning and extended-thinking models: what the extra tokens buy and what they cost
- Pricing mechanics: input versus output tokens, cached input, batch discounts, and why output dominates a chat bill
- Benchmarking a model on *your* task with your Lesson 30 harness instead of trusting a leaderboard
- Migration risk: what breaks when a provider deprecates a model, and how to be ready

**You build.** A selection table for your capstone's two hottest calls: three candidate models, your own eval score, p95 latency and cost per 1k requests.

**Prove it.** You justify a model choice with three numbers, one of them from your own harness.

---

## E2 — Prompting and context engineering

**After Lesson 20 · 2 h**

**Engineer's frame.** Prompting is covered here as one option among prompting, retrieval and fine-tuning — not as a subject. What matters to an engineer is which prompt structures are *load-bearing* under version control, and how a prompt behaves as an API contract.

**Intuition.** The context window is a budget you allocate, and everything in it competes: instructions, tools, retrieved chunks, history. Chain-of-thought and extended thinking buy accuracy on multi-step problems by spending output tokens — a measurable trade, not a trick. DSPy's contribution is the idea that prompts are a compiled artefact optimized against a metric, which is exactly how you should treat them once you have a harness.

**Topics**
- Prompt structure as a contract: instructions, data, output format, and the boundary between them
- Chain-of-thought and its cost; when the extra tokens do not pay
- Extended thinking / reasoning budgets, and the latency they add
- Few-shot examples as data, versioned like data
- Context engineering: allocating a budget across system prompt, tools, retrieval and history
- Prompt optimization with DSPy: prompts compiled against a metric rather than hand-tuned
- Prompt injection, revisited from Lesson 28 — with the prompt author's mitigations

**You build.** Two versions of your capstone's main prompt scored on your Lesson 30 harness, with the context budget written out in tokens.

**Prove it.** You state where your prompt spends its context budget and which part you would cut first.

---

## E3 — Advanced retrieval: multimodal, agentic, and RAG at scale

**After Lesson 24 · 3 h**

**Engineer's frame.** Lessons 23–24 build one good retrieval pipeline. This elective covers what changes when the corpus has images and tables, when one query is not enough, and when the corpus is 100 million documents instead of 100 thousand — which is the shape of the retrieval question in a design interview.

**Topics**
- Multimodal RAG: image and table embeddings, unified versus separate indexes, and citing a figure
- Agentic RAG: query decomposition, multi-hop retrieval, and the loop's cost and termination
- Query rewriting and HyDE, and when they help
- Production RAG at scale: sharding, replication, tiered storage, incremental index build
- Freshness: streaming updates versus periodic rebuild, and the staleness you are agreeing to
- Cost per query at scale, and where reranking stops being affordable
- Data engineering for AI: pipeline ownership, schemas and contracts upstream of the index

**You build.** A written scaling plan for your own retrieval service at 1000×: what shards, what is replicated, what caches, and where the cost lands.

**Prove it.** You size an index for 100 M documents — memory, shards, p95 — using Lesson 23's numbers.

---

## E4 — Document processing and vision-LLM parsing

**After Lesson 26 · 2 h**

**Engineer's frame.** Real corpora are PDFs — scanned, multi-column, full of tables that mean nothing once linearized. Ingestion quality caps retrieval quality, and this is where most of it is lost, silently.

**Topics**
- The parsing stack: text extraction, layout analysis, OCR, and where each fails
- Vision-LLM OCR: reading a page as an image, and its failure modes (hallucinated cells, dropped rows)
- Tables: preserving structure so a chunk means something on its own
- Multimodal parsing: figures, charts and captions as retrievable units
- Cost and latency of the parsing stage, per thousand pages
- Measuring parse quality directly, instead of inferring it from answer quality

**You build.** A parse-quality check over twenty hard pages from your own corpus, with the failure classes counted.

**Prove it.** You show a table your pipeline got wrong, and what it did to the retrieved chunk.

---

## E5 — Agent protocols, multi-agent systems and computer use

**After Lesson 28 · 3 h**

**Engineer's frame.** Multi-agent architectures are the most over-applied pattern in the field. The engineering question is whether the coordination cost buys anything a single loop with good tools would not — usually it does not, and knowing when it does is the point of this elective.

**Topics**
- MCP in depth: servers, clients, transports, and versioning a tool surface
- Agent-to-agent protocols: what a standard interchange buys, and the failure modes it adds
- Multi-agent topologies: supervisor/worker, pipeline, debate — and the cost of each in tokens and latency
- When one agent with better tools beats three agents
- Computer-use agents: the screenshot-act loop, its latency, and its failure modes
- Sandboxing: what a computer-use agent must not reach, and how that is enforced
- Loop engineering: termination, progress detection, and recovering a stuck agent
- Durable execution revisited: checkpointed state, replay and idempotent steps

**You build.** A single-agent versus supervisor/worker comparison on one task from your capstone: tokens, latency, cost and success rate.

**Prove it.** You state, with numbers, whether multi-agent paid for itself on your task.

---

## E6 — Memory and state

**After Lesson 28 · 2 h**

**Engineer's frame.** "The assistant should remember me" is a storage and retrieval design, not a model capability. Getting it wrong produces both classic failures: an agent that forgets what it just did, and an agent that drags an irrelevant year of history into every prompt.

**Intuition.** Memory is tiered like a cache. L1 is the live context window. L2 is a session store — recent turns, summarized. L3 is long-term memory in a durable store, retrieved on relevance like any other document. What distinguishes memory from retrieval is writing: deciding what is worth keeping, and reconciling contradictions when a fact changes.

**Topics**
- The memory tiers: working context (L1), session state (L2), long-term store (L3)
- Write policy: what gets remembered, by what rule, and who can see it
- Conflict and staleness: the user changed their mind, and both facts are stored
- Summarization and compaction as lossy compression, with the loss chosen deliberately
- Off-the-shelf memory layers (e.g. Mem0) versus your own store, and the operational trade
- Memory as personal data — Lesson 31's deletion path applies here too
- Caching adjacent to memory: semantic cache hits, and the wrong-answer risk they carry

**You build.** A three-tier memory design for your capstone, with the write rule, the retention and the deletion path written down.

**Prove it.** You explain what your system remembers, for how long, and how a user gets it erased.

---

## E7 — Frameworks, orchestration and framework churn

**After Lesson 30 · 2 h**

**Engineer's frame.** You built everything from primitives on purpose, so you can now evaluate a framework instead of adopting one by default. The real question is not features — it is what a framework does to your debuggability and your migration cost when it changes shape next quarter.

**Topics**
- LangGraph and graph-based orchestration: explicit state machines, checkpointing, human-in-the-loop
- LlamaIndex and retrieval-first frameworks: what they give you, what they hide
- DSPy revisited as a compiler over prompts, not a runtime
- Agent SDKs and coding agents (Claude Code and the open-coder landscape) as tools *and* as reference architectures
- The adoption test: can you trace it, test it offline, and remove it in a week
- Navigating framework churn: which layer to keep yours, which to rent
- Anti-pattern — a framework adopted to avoid understanding the loop

**You build.** A one-page adoption review of one framework against your own agent: what it would replace, what it would hide, and the exit cost.

**Prove it.** You defend either adopting or refusing a framework using your own trace and test story.

---

## E8 — Infrastructure: GPU clusters, gateways and FinOps

**After Lesson 32 · 3 h · pairs with Track P**

**Engineer's frame.** Track P deploys one engine on your laptop's cluster. This elective is the shape of the real thing: multiple GPUs, multiple models, many tenants, and a finance team asking what a conversation costs.

**Topics**
- GPU cluster basics: node types, memory per card, interconnect, and what it means for a model that does not fit
- Scheduling large and small models on a shared pool; bin-packing and fragmentation
- Model-weight distribution to workers: pull storms, caching, and warm-start time
- Multi-model serving: co-location, adapters (multi-LoRA) and the isolation trade
- The AI gateway in depth: keys, quotas, routing, retries, cost attribution per tenant
- FinOps: unit economics — cost per request, per session, per resolved task — and where a budget alert belongs
- Capacity planning against a fixed hardware budget, with Lesson 6's arithmetic

**You build.** A capacity and cost plan for your capstone at 10× and 100× traffic, with the GPU count and monthly cost derived, not guessed.

**Prove it.** You size a GPU pool for a stated QPS and latency target, showing the arithmetic.

---

## E9 — Safety: guardrails, red-teaming and governance

**After Lesson 32 · 2 h**

**Engineer's frame.** Guardrails are a product surface: they have false positives that annoy real users and false negatives that make the news. They are engineered and measured like any other classifier — which is Lesson 1, applied to safety.

**Topics**
- Input and output guardrails: what each catches, and why both are needed
- Guardrails as classifiers: precision, recall and the threshold as a product decision
- Refusal behaviour and graceful degradation, versus a hard block
- Red-teaming: building an attack set, running it as tests, and tracking the pass rate over time
- Jailbreak classes and what actually mitigates each
- AI governance and compliance: model inventory, risk classification, documented evaluations, human oversight
- Incident response for an AI-specific failure: what you disclose, what you log, what you change

**You build.** A red-team suite for your capstone — twenty attacks, run in CI — with the pass rate reported and the false-positive rate on benign traffic measured too.

**Prove it.** You report both rates and say which one you would rather be wrong on, and why.

---

## E10 — AI design patterns and anti-patterns

**After Lesson 38 · 2 h**

**Engineer's frame.** After twenty weeks you have built most of these once. Naming them turns experience into vocabulary you can use in a design review, quickly.

**Topics**
- The pattern catalog: router, cheap-then-expensive cascade, retrieve-then-generate, plan-execute, reflect-and-retry, evaluator-optimizer, human-in-the-loop gate, fallback chain, semantic cache
- Choosing between a chain, a router and a loop — the cost and testability of each
- Anti-patterns: the agent that should have been a chain · retrieval used to teach behaviour · fine-tuning used to supply facts · an LLM judge nobody validated · a framework adopted to avoid understanding the loop · unbounded loops without a cost cap · evals written after the launch
- Naming the pattern in your own capstone, out loud, in one sentence per component

**You build.** A pattern annotation of your own architecture diagram, plus two anti-patterns you have personally shipped and fixed.

**Prove it.** Given someone else's design, you name the pattern and the anti-pattern in under a minute.

---

# Appendix A — Case bank

Cases for Module 6's four reps (Lessons 34–37), for the mock review (Lesson 38) and for self-study. Run each as ten minutes of recall, forty minutes solo and spoken aloud, then a written delta against a reference solution. Pick reps from **different rows** — the failure you are training against is being fluent in only one shape of problem.

## A.1 — Infrastructure and serving

*Sourced from recent Anthropic-style interview reports ([prachub](https://prachub.com/?sort=hot&company=Anthropic&category=ML+System+Design%2CSystem+Design)). These are the highest-signal cases for this course, because the whole spine points at them.*

| Case | What it tests | Lessons it draws on |
|---|---|---|
| Design a low-latency GPU inference service | TTFT/TPOT split, batching policy, queueing | 6, 21, 22, P5–P12 |
| Design an LLM request batching system | Continuous batching, scheduling, fairness, SLO per tier | 21, 22, 35 |
| Design GPU inference request batching | The same problem at the engine level: queue, batch former, preemption | 22, P8 |
| Design a dynamically batched inference API | The API contract over a batching engine: timeouts, cancellation, backpressure | 7, 8, 9, 22 |
| Schedule large and small model inference on an eight-GPU pool | Bin-packing, memory fragmentation, priority, starvation | 6, 22, E8 |
| Deploy a large model to GPU workers | Placement, warm start, health, rollout | 6, P4–P7, E8 |
| Design large model-weight distribution to GPU workers | Pull storms, caching layers, cold-start time, integrity | E8, P7 |
| Design model weight distribution | The same, generalized: registry, replication, versioning, rollback | 26, E8 |
| Design peer-to-peer model distribution under a shared link cap | Bandwidth as the hard constraint; fan-out topology; failure and resume | E8 |
| Design a distributed rate limiter | Shared state, clock skew, per-tenant quotas, graceful rejection | 7, 9, E8 |
| Design a concurrent image processing service | Bounded concurrency, queueing, backpressure, partial failure | 9, 10 |
| Design a crash-resilient LRU cache | Durability versus speed, recovery, correctness after a crash | 26, 32 |
| Scale duplicate file detection | Content hashing at scale, sharding, memory budget | 26 |
| Find a distributed mode efficiently | Distributed aggregation, approximation, network cost | 25, E8 |

## A.2 — Products and platforms

| Case | What it tests | Lessons it draws on |
|---|---|---|
| Design a prompt playground | Multi-tenant state, streaming, versioning, cost caps | 7, 8, 31, E2 |
| Design a prompt sharing product | Sharing model, permissions, versioning, abuse | 7, 31 |
| Design a resilient chat system | Streaming, reconnection, ordering, durable history | 8, 9, E6 |
| Design a one-on-one chat service | The classic: fan-out, delivery, presence, storage | 25, 35 |
| Design Instagram (feed, photos, friend recommendations) | Ranking funnel, position bias, feature stores, media storage | 34, 35, 36 |
| Review and improve a flawed design document | Reading a design critically, ranking findings by severity | 38 |

## A.3 — Applied AI systems

*Scenario set from the [ai-system-design-guide](https://github.com/ombharatiya/ai-system-design-guide) case studies, aligned to this course's modules.*

| Case | What it tests | Lessons it draws on |
|---|---|---|
| Retrieval over 100 million documents, with permissions and freshness | Sharding, hybrid search, pre-filtering, staleness | 23, 24, 26, E3 |
| Real-time search over a live corpus | Freshness, incremental indexing, cache invalidation | 26, E3 |
| An autonomous coding agent | Tool design, sandboxing, durable state, cost caps | 27, 28, E5 |
| A computer-use agent in production | Screenshot-act loop, latency, blast-radius control | E5, E9 |
| Multi-tenant SaaS with AI features | Isolation, per-tenant quotas, cost attribution, leak tests | 31, E8 |
| A multi-tenant fine-tuning platform | Data isolation, adapter serving, scheduling, lineage | 20, E8 |
| Customer support automation | Deflection metric, escalation path, quality gate, feedback loop | 24, 30, 36 |
| Document intelligence over PDFs at scale | Parsing quality, table structure, cost per thousand pages | E4, 26 |
| Knowledge management for an enterprise | Permission-aware retrieval, freshness, ownership, adoption | 24, 31 |
| An MCP knowledge agent | Tool surface design, versioning, permissions | 27, E5 |
| An evaluation pipeline as a batch system | Orchestration, judge fan-out, cost per run, flakiness | 29, 30 |
| Eval-gated CI/CD | Thresholds, noise, blocking policy, rollback | 30, 32 |
| A customer distillation pipeline | Teacher/student, data collection, drift, coverage | 20, 36 |
| An agent platform: tools, sandboxing, durable state, cost caps | Platform-shaped agent design | 27, 28, E5 |
| A recommendation or ranking system | Funnel, features, position bias, online/offline gap | 34, 35, 36 |
| Fraud detection under a hard 100 ms budget | Threshold economics, latency split, cascade, drift | 1, 35, 36 |
| Compliance automation | Audit trail, human oversight, evidence, false negatives | 31, E9 |
| Voice AI in healthcare | Latency budget, turn-taking, PHI handling, failure disclosure | 31, 35 |
| Real-time voice agents | VAD, turn-taking, barge-in, speech-to-speech latency budget | 35, E5 |
| Multimodal generation pipeline | Provenance, evaluation, cost, moderation | E9, E3 |

## A.4 — How to use the bank

- **Reps 1–4 (Lessons 34–37):** one from A.1, one from A.2 or A.3, one you find hardest, one at random.
- **Whiteboard drills (Lesson 38):** estimate a KV cache · size an index for 100 M documents · split a p95 budget across four stages · cost a run at 1 M requests. Five minutes each, from memory.
- **Self-study:** the guide's question bank is a recall check, not a design rep. Use it between reps, not instead of them.

---

# Appendix B — Coverage map

Every heading in [MODULES.md](MODULES.md), and the lesson that owns it. Nothing is unassigned.

| MODULES.md | Lesson |
|---|---|
| 1.1 Generalization vocabulary | 1 |
| 1.2 Classification metrics and thresholds | 1 |
| 1.3 Data leakage | 2 |
| 1.4 Ranking and retrieval metrics | 2 |
| 1.5 Similarity — pointer | 2 (full treatment: 23) |
| 1.6 Comparing two systems | 2 |
| 1.7 Calibration | 1 |
| 1.8 Data quality | 1 |
| 1.9 Awareness pass **[ext]** | 1 |
| 2.1 Tensors and autograd | 3 |
| 2.2 Backpropagation | 3 |
| 2.3 Losses and softmax | 4 |
| 2.4 Layers, initialization, normalization | 4 |
| 2.5 Embedding layers | 4 |
| 2.6 Optimizers | 5 |
| 2.7 A real training run | 5 |
| 2.8 Failure lab | 5 |
| 2.9 CNN and RNN awareness | 5 |
| 2.10 Memory accounting | 6 |
| 2.11 Arithmetic intensity and the roofline | 6 |
| 2.12 Precision and profiling **[ext]** | 6 |
| 3.01 Tooling · 3.02 Local model access · 3.03 Repository and CI | 0 |
| 3.1 API design | 7 |
| 3.2 Streaming endpoints | 8 |
| 3.3 Async Python at production level | 9 |
| 3.4 Containerization | 10 |
| 3.5 Testing LLM applications | 10 |
| 4 · week 6 Tokenization | 11, 12 |
| 4 · week 7 Attention | 13, 14 |
| 4 · week 8 Architecture assembly | 15, 16 |
| 4 · week 9 Decoding and structured output | 17, 18 |
| 4 · week 10 Modern variants · Pretraining | 19 |
| 4 · week 10 Post-training and alignment · Fine-tuning in practice | 20 |
| 5 · week 11 Inference systems | 21, 22 |
| 5 · week 12 Retrieval | 23, 24 |
| 5 · week 13 SQL | 25 |
| 5 · week 13 Ingestion · Versioning and vector-store operations | 26 |
| 5 · week 14 Agents | 27, 28 |
| 5 · week 15 Evaluation | 29, 30 |
| 5 · week 16 Secrets · PII · Compliance · Audit and tenancy | 31 |
| 5 · week 16 Observability · Operations | 32 |
| 6.1 The protocol · 6.2 Framing | 33 |
| 6.3 Data, labels and features | 34 |
| 6.4 Serving and latency | 35 |
| 6.5 Monitoring, drift and retraining | 36 |
| 6.6 Reps — four timed designs | 34, 35, 36, 37; redraw in 38 |
| 7.1 Hardening | 39 |
| 7.2 Documentation · 7.3 The demo | 40 |
| P.1 Kubernetes core | P1, P2 |
| P.2 Production-shaped manifests | P3, P4 |
| P.3 The inference engine | P5, P6 |
| P.4 Packaging and scale-out | P7, P8 |
| P.5 Observability | P9, P10 |
| P.6 Load, autoscaling and the edge | P11, P12 |
| P.7 Security and operations | P13 |

**Added beyond MODULES.md** — from [ai-system-design-guide](https://github.com/ombharatiya/ai-system-design-guide), folded into the lessons above or into Track E: RLVR and distillation (20) · diffusion LLMs, on-device and edge (22) · contextual retrieval and ColBERT (24) · MCP and durable execution (27) · guardrails and red-teaming (28) · RAGAS, LangSmith, leaderboards, quality drift (30) · RBAC/ABAC and AI governance (31) · AI gateway, model routing, FinOps, LLMOps (32) · model landscape and pricing (E1) · prompting, CoT, extended thinking, DSPy (E2) · multimodal, agentic and at-scale RAG (E3) · vision-LLM OCR and document processing (E4) · agent protocols, multi-agent, computer use (E5) · memory tiers and semantic caching (E6) · LangGraph, LlamaIndex, framework churn (E7) · GPU clusters and capacity (E8) · safety and governance (E9) · patterns and anti-patterns (E10) · case studies and question bank (Appendix A).
