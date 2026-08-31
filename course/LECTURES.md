# Applied AI Engineering — Lecture Plan

**55 lectures · 2 hours each · 2 lectures per week · 27 weeks, plus a closing session after demo day.**
For engineers with a solid software background who want to build and operate production LLM systems. No ML background assumed.

## Weekly rhythm

| | Lecture A                                                 | Lecture B |
|---|-----------------------------------------------------------|---|
| Format | **Fully theoretical** — mechanism, derivation, whiteboard | **Half theory, half practical** — 1 h theory, 1 h live lab |
| Length | 2 h                                                       | 2 h |
| Independent work | 4 h                                                       | 6 h |

**Weekly total:** 4 h lectures + 10 h independent + 2 h platform track (weeks 4–16) ≈ **14–16 h**.

Independent hours are structured, not open-ended. Every week they contain four fixed slots:

| Slot | Time | What |
|---|---|---|
| **Build** | 4–6 h | The week's code. It lands in your repo and it runs |
| **System design drill** | 45 min | One case from the bank, spoken aloud, timed, then a written delta |
| **Production drill** | 45 min | One operational task: measure, break, harden, or cost something you already built |
| **Derivation drill** | 30 min | One derivation by hand, on paper: cross-entropy gradient, Adam bias correction, ring all-reduce cost, KL, DPO gradient |
| **Reading** | 1–2 h | Marked sections only. Three sources per topic, never four |

## Course-long project spine

Eleven projects. Each builds on the last, and the last one is what you demo.

| # | After week | Project | Hours |
|---|---|---|---|
| 1 | 1 | Metrics and significance toolkit | 8 |
| 2 | 3 | Autograd, training loop, memory calculator | 16 |
| 3 | 5 | Streaming AI service, containerized and tested | 16 |
| 4 | 10 | GPT from scratch: tokenizer → model → sampler → LoRA | 40 |
| 5 | 12 | Kernel and scaling report: Triton kernel + FSDP run | 16 |
| 6 | 17 | Retrieval, ingestion and storage platform | 34 |
| 7 | 21 | Agent platform, eval gate, security and observability | 48 |
| 8 | 23 | SLM inference service on Kubernetes | 24 |
| 9 | 23 | Paper reproduction report | 12 |
| 10 | 26 | System design portfolio: four reps and a design document | 20 |
| 11 | 27 | Capstone: the whole system, deployed, defended, plus one merged OSS PR | 14 |

**Platform track (weeks 4–16, 2 h/week, in independent time).** Kubernetes → vLLM → Helm → Prometheus/Grafana → k6 → autoscaling → network policy. Listed inside each week's independent work, so your capstone has somewhere to run.

**Pre-work (before week 1, ~4 h).** Python 3.11+ with a pinned lockfile (`uv` or `venv`) · git workflow and `.gitignore` · Docker running and verified · an editor with a working debugger · open weights downloaded and a small model served locally over an OpenAI-compatible API · provider keys in environment variables with a hard spend cap set · the repo template (`src/`, `tests/`, `evals/`, `docs/`, `Makefile`) with CI running lint and tests on push.

---

## Lecture 1 — What a model is, and how you judge one

**Week 1 · Lecture A · fully theoretical · 2 h**

- Machine learning defined as function fitting, in terms an engineer already uses: data in, parameters fitted, behaviour generalized
- Bias and variance, model capacity, overfitting and underfitting
- Regularization: L1, L2, early stopping — what each actually constrains
- Weight decay versus L2 in the loss, and why AdamW exists
- Why a training curve alone never tells you whether to ship
- Confusion matrix; precision, recall, F1 — and when each is the wrong summary
- ROC and AUC versus precision–recall curves under class imbalance
- The threshold as a product decision: costing a false positive against a false negative before choosing it
- Why accuracy lies on imbalanced data
- Calibration: what a confidence score does and does not mean; reliability diagrams
- Platt scaling and isotonic regression **[extension]**
- Why a guardrail at threshold 0.9 fires far more often than "10% error" suggests
- Data quality: missing values (drop, impute, or treat missingness as signal), class imbalance, distribution shift and how it is detected
- One-sentence awareness pass **[extension]**: logistic regression · decision trees · random forests · gradient boosting · naive Bayes · SVM · k-means · PCA — and where gradient boosting still beats a neural network

**Independent work — 4 h**
- **Build (2 h).** `metrics.py`: confusion matrix, precision, recall, F1, threshold sweep, reliability diagram — from scratch, no sklearn.
- **System design drill (45 min).** *Fraud detection under a hard 100 ms budget.* Ten minutes recall, thirty minutes designing aloud, then write what you missed.
- **Production drill (45 min).** Take one classifier threshold in any system you have shipped and write the cost of each error type in currency. If you cannot, write down who would know.
- **Reading (30 min).** Precision/recall and ROC vs PR, marked sections.

---

## Lecture 2 — Leakage, ranking metrics, and proving a delta

**Week 1 · Lecture B · theory + lab · 2 h**

**Theory (1 h)**
- Train/test contamination: the same row on both sides of the split
- Temporal leakage — using the future to predict the past; time-ordered splits
- Group leakage — the same user or document split across train and test
- Benchmark contamination in LLMs: the model already read the exam
- Why a random split flatters the score on time-ordered data
- What changes when you score a ranked list instead of a yes/no answer
- `recall@k`, `precision@k`, MRR, `nDCG@k` — and what position weighting penalizes that recall does not
- Choosing k for a retrieval stage that feeds a generator; why better `recall@50` can still make a worse system
- Similarity pointer: cosine, dot product, L2, and why normalization decides whether cosine and dot product disagree (full treatment in Lecture 23)
- Bootstrap confidence intervals, paired tests and why pairing matters, sample size for a given delta, multiple comparisons across twelve prompt variants
- Reporting a result honestly: effect size with an interval, never a bare number

**Lab (1 h)**
- Implement `recall_at_k`, `mrr`, `ndcg_at_k` and check them against values you computed by hand on paper
- Build `compare.py`: two lists of per-case scores in, mean difference and a bootstrap 95% interval out

**Independent work — 6 h**
- **Build (4 h).** Finish Project 1. Add a leakage demo: the same dataset scored under a random split and a time-ordered split, gap printed.
- **System design drill (45 min).** *An evaluation pipeline as a batch system* — orchestration, fan-out, cost per run.
- **Production drill (45 min).** Find a real evaluation in your own work or repo with no interval reported. Add one.
- **Reading (30 min).** Bootstrap resampling; benchmark contamination.

### Project 1 — Metrics and significance toolkit *(8 h, due end of week 1)*

**Deliverable.** A tested `metrics/` package: classification metrics with a threshold sweep, ranking metrics, `compare.py` with bootstrap intervals, and a leakage demonstration.
**Acceptance.** Unit tests against hand-computed values · CI green · `compare.py` answers "0.71 vs 0.68 on 200 queries — real or noise?" with an interval and a required sample size.
**Why it exists.** Every eval gate in weeks 15, 16 and 20 imports this package.

---

## Lecture 3 — Tensors, autograd, backpropagation

**Week 2 · Lecture A · fully theoretical · 2 h**

- A tensor library as a small runtime: it records a graph, executes on a device, walks it backwards
- Indexing, `reshape` / `view` / `permute`, broadcasting rules
- `einsum`, and reading a `(B, H, T, D)` shape without guessing
- Device placement, dtypes, `.to()` semantics
- Autograd: how the graph is recorded, `backward()`, leaf tensors
- `detach` versus `no_grad`, and when each is correct
- Debugging shape errors without printing shapes first
- The chain rule over a computation graph
- Cost and memory of the forward pass versus the backward pass
- **Why every layer's activations must be kept until the backward pass** — the origin of most OOMs
- Gradient checkpointing: recompute instead of store, and what it costs
- Why training memory grows with batch × sequence length × depth

**Independent work — 4 h**
- **Build (2 h).** Scalar autograd engine: `Value`, forward ops, `backward()`, verified against finite differences.
- **System design drill (45 min).** *Design a concurrent image processing service* — bounded concurrency, queueing, partial failure.
- **Production drill (45 min).** Take a training or inference script you can run and record peak memory at three batch sizes. Predict the third from the first two before measuring.
- **Reading (30 min).** Autograd mechanics, marked sections.

---

## Lecture 4 — Losses, layers, embeddings

**Week 2 · Lecture B · theory + lab · 2 h**

**Theory (1 h)**
- MSE versus cross-entropy, and why cross-entropy for classification
- Softmax numerical stability and the log-sum-exp trick
- Logits versus probabilities, and why losses take logits
- Temperature as a rescaling of logits — the same knob returning in Lecture 17
- What quietly breaks when you hand `CrossEntropyLoss` softmax output
- Activations: ReLU, GELU, SiLU; dead ReLUs
- Initialization: Xavier and He, and what breaks without them
- BatchNorm versus LayerNorm versus RMSNorm; why transformers use LayerNorm — two concrete reasons
- Pre-norm versus post-norm, and why pre-norm won; dropout and where it still appears
- `nn.Embedding` as a lookup table, not a one-hot matrix multiply
- Embedding parameters = vocabulary × hidden, and what share of a model that is
- Sparse gradients: only the rows you indexed receive one
- Weight tying, and why it is a large win on a small model and nearly noise on a large one

**Lab (1 h)**
- Add one operator to your autograd engine and prove the gradient numerically
- Write a stable softmax and cross-entropy; match the framework to 1e-6 on logits at ±1e4

**Independent work — 6 h**
- **Build (4 h).** An MLP trained end to end with your own initialization; run it once without initialization and keep both curves. Print a parameter breakdown with embeddings as a percentage.
- **System design drill (45 min).** *Design a distributed rate limiter.*
- **Production drill (45 min).** Instrument your MLP training to log parameter count, activation memory estimate and step time. No guessing later.
- **Reading (30 min).** Normalization placement; weight tying.

---

## Lecture 5 — Optimizers and a real training loop

**Week 3 · Lecture A · fully theoretical · 2 h**

- SGD → momentum → RMSProp → Adam → AdamW, and what each step added
- What `m`, `v`, `β1`, `β2` and `ε` do
- Optimizer-state memory: two extra tensors per parameter — a third of your training memory
- Learning-rate warmup and cosine decay; why warmup exists
- Gradient clipping, and how to read a loss spike
- **Why a bad batch poisons Adam's `m` and `v` long after it is gone**
- `nn.Module`, parameter registration, `state_dict`; custom layers, initialization, weight sharing
- `Dataset`, `DataLoader`, collation, worker count; device management and pinned memory **[extension]**
- Checkpoint and resume that survives the process being killed
- CNN awareness: convolution, pooling, receptive fields, the residual connection
- RNN awareness: the sequential bottleneck — what transformers replaced, and why

**Independent work — 4 h**
- **Build (2 h).** A training loop with checkpoint and resume. Prove it with `kill -9` mid-run and a clean continuation.
- **System design drill (45 min).** *Design a crash-resilient LRU cache* — durability versus speed, correctness after a crash.
- **Production drill (45 min).** Make your checkpoint atomic (write-temp-then-rename) and prove a crash mid-write cannot corrupt it.
- **Reading (30 min).** AdamW; warmup schedules.

---

## Lecture 6 — Memory accounting, roofline, and the failure lab

**Week 3 · Lecture B · theory + lab · 2 h**

**Theory (1 h)**
- Training memory = parameters + gradients + optimizer states + activations
- The ~16 bytes-per-parameter figure for Adam in mixed precision, and where it comes from
- Activation memory as a function of batch × sequence × layers; activation checkpointing as compute-for-memory
- Inference memory = weights + KV cache + activations; the KV-cache formula and what GQA does to it
- FLOPs versus bytes moved; arithmetic intensity; the roofline model
- The memory hierarchy — registers, shared memory, L2, HBM — bandwidth versus capacity
- **Why prefill is compute-bound and decode is memory-bandwidth-bound**
- Why batching raises decode throughput but barely helps decode latency
- fp32, tf32, fp16, bf16, fp8; why bf16 beat fp16; loss scaling; autocast and master weights **[extension]**
- Reading a profiler trace: overhead-, compute- or memory-bound **[extension]**
- Data, tensor and pipeline parallelism — what each shards and what it costs in communication **[extension]**

**Lab (1 h) — the failure lab**
- Raise batch size until the process dies; read the OOM on MPS versus CPU
- Recover the effective batch with gradient accumulation, and measure the wall-clock cost
- Poison a batch on purpose (inputs ×1000, randomized labels, a `NaN` sample) and watch Adam stay poisoned
- Add the guards: clipping, skipping non-finite steps

**Independent work — 6 h**
- **Build (4 h).** Finish Project 2. `memory_math.py`: training memory four ways, inference memory three ways, KV-cache bytes with an MHA/GQA/MQA switch — validated against a published model's real numbers.
- **System design drill (45 min).** *Deploy a large model to GPU workers* — placement, warm start, health, rollout.
- **Production drill (45 min).** Size the hardware for a 7B model at batch 32, 4k context, bf16, GQA-8 — on paper in under three minutes, then check against your own tool.
- **Reading (30 min).** Roofline model; mixed precision.

### Project 2 — Autograd, training loop, memory calculator *(16 h, due end of week 3)*

**Deliverable.** A micrograd-style engine with an operator you added · a training loop with atomic checkpoint/resume · three documented failures (OOM, poisoned batch, LR too high) with the log excerpt, the diagnosis and the guard · `memory_math.py`.
**Acceptance.** `kill -9` mid-training resumes with no loss discontinuity · your KV-cache number lands within 10% of a published model's · three unlabelled loss curves, three correct diagnoses.
**Why it exists.** `memory_math.py` is used in weeks 7, 11, 18 and in every capacity question for the rest of the course.

---

## Lecture 7 — API design for AI services

**Week 4 · Lecture A · fully theoretical · 2 h**

- Why LLM endpoints break normal REST assumptions: thirty-second responses, per-call cost, non-determinism, aggressive client retries
- Resource naming, HTTP verbs, and status codes that mean what they say
- One error contract for every failure: machine-readable code plus human message (RFC 9457)
- Pagination: cursor versus offset, and why offset breaks under concurrent writes
- Versioning: URL versus header, deprecation windows, additive-change discipline
- **Idempotency keys**, so a retry does not run the job twice — or bill twice
- Long-running work: 202 plus polling or a webhook, never a four-minute request
- Request validation at the boundary; OpenAPI as the contract, checked into the repo
- Multi-tenancy in the contract: where the tenant comes from and where it must be enforced

**Independent work — 4 h**
- **Build (2 h).** The API skeleton: one error contract, cursor pagination, idempotency key on the generation endpoint, OpenAPI committed.
- **System design drill (45 min).** *Design a prompt playground* — multi-tenant state, streaming, versioning, cost caps.
- **Platform track P1 (2 h).** Cluster anatomy: control plane, nodes, scheduler. `kubectl describe`, `logs`, `exec`, `port-forward`. Diagnose a pod that will not start, from `describe` alone.
- **Reading (30 min).** RFC 9457; idempotency patterns.

---

## Lecture 8 — Networking primer and streaming endpoints

**Week 4 · Lecture B · theory + lab · 2 h**

**Networking primer (30 min)** — assumed knowledge everywhere else in this course, stated once here
- TCP: the handshake, slow start, and why the first byte of a new connection is expensive
- TLS: the handshake cost, session resumption, and where termination happens in your stack
- **Keep-alive and connection reuse** — the reason Lecture 9 insists on one shared client
- HTTP/1.1 versus HTTP/2 versus HTTP/3: multiplexing, head-of-line blocking, and which one your proxy actually speaks
- Load balancers at L4 versus L7, idle timeouts, and response buffering — the three settings that silently kill a token stream
- DNS and service discovery, and what a stale record costs you during a rollout
- gRPC versus HTTP for internal hops, and why the answer differs inside and outside the cluster
- Latency versus bandwidth: a token is tiny, a handshake is not — which is why TTFT is a connection problem before it is a model problem

**Theory (30 min)**
- The SSE wire format: `data:`, `event:`, `id:`, blank-line framing
- Why SSE rather than WebSocket for one-way token streams
- Client disconnect and cancellation — stop generating, stop paying
- **Proxies that buffer your stream, and how time-to-first-token dies there**
- Errors after the 200 has already been sent: in-band error events
- Parsing structured output progressively while it streams
- Heartbeats, idle timeouts, and a slow client applying backpressure

**Lab (1 h)**
- Ship a streaming endpoint with in-band error events and heartbeats
- Disconnect a client mid-stream and prove upstream generation actually stops

**Independent work — 6 h**
- **Build (4 h).** Cancellation propagated to the model call · a progressive JSON parser over the stream · TTFT measured with and without a buffering proxy in front.
- **System design drill (45 min).** *Design a resilient chat system* — streaming, reconnection, ordering, durable history.
- **Production drill (45 min).** Put nginx in front of your stream with default settings, measure the TTFT damage, then fix the config.
- **Platform track P2 (2 h).** Pods, Deployments, ReplicaSets, Services; namespaces, labels, selectors. Your service running in the cluster.

---

## Lecture 9 — Async Python at production level

**Week 5 · Lecture A · fully theoretical · 2 h**

- The event loop; coroutine versus task; what `await` actually yields
- **One blocking call stalls every request on that worker** — the most common Python AI-service outage, and the two fixes
- Structured concurrency: `TaskGroup`, cancellation semantics, `asyncio.timeout`
- Concurrency limits with a semaphore; bounded queues as backpressure
- Connection pooling and keep-alive: one shared client, not one per request
- Timeouts at every layer — connect, read, total — plus the retry budget on top
- Retry with exponential backoff and jitter; why an unbudgeted retry is an outage amplifier
- GIL versus asyncio versus processes: which problem each solves
- Uvicorn workers, and why a `def` endpoint in FastAPI runs in a thread pool

**Independent work — 4 h**
- **Build (2 h).** An async fetcher: shared client, semaphore, layered timeouts, jittered backoff.
- **System design drill (45 min).** *Design a dynamically batched inference API* — timeouts, cancellation, backpressure.
- **Production drill (45 min).** Measure p50/p99 across 50 concurrent requests with and without one blocking call in the handler. Keep both numbers.
- **Platform track P3 (2 h).** ConfigMaps and Secrets, and what a Secret is *not*. Configuration by environment.

---

## Lecture 10 — Containers and testing LLM applications

**Week 5 · Lecture B · theory + lab · 2 h**

**Theory (1 h)**
- Multi-stage builds; a runtime image without the build toolchain
- Layer caching: dependencies before source, or every edit rebuilds the world
- Pinned dependencies and reproducible installs
- Non-root user, `HEALTHCHECK`, and signal handling so `SIGTERM` drains cleanly mid-stream
- Configuration by environment; model weights as a volume, never a layer
- pytest in practice: fixtures, `parametrize`, `monkeypatch`, markers; testing async code and an ASGI app in-process
- **Mocking the provider at the HTTP boundary**, not at your own wrapper — and why that difference matters
- Recorded fixtures for real responses, including streamed chunk sequences
- A scripted fake model to drive agent branches: tool error, infinite loop, injected instruction
- **What you must never assert:** exact model text. Assert structure and invariants
- Failure-path tests: timeout, 429, truncated stream, cancelled client
- Where pytest stops and evaluation begins — and why both run in CI

**Lab (1 h)**
- Build the image, run as non-root, and drain a `SIGTERM` while a stream is in flight
- Write the HTTP-boundary mock and one streamed fixture

**Independent work — 6 h**
- **Build (4 h).** Finish Project 3: full offline test suite including failure paths, running with the network disabled in under ten seconds.
- **System design drill (45 min).** *Design a distributed rate limiter*, second attempt — compare against your week-2 answer.
- **Production drill (45 min).** Shrink your image: measure before and after, and record what you removed.
- **Platform track P4 (2 h).** Liveness, readiness and startup probes — what each controls. Requests and limits; throttled versus killed. A persistent volume for the model cache.

### Project 3 — Streaming AI service *(16 h, due end of week 5)*

**Deliverable.** An HTTP API with one error contract, cursor pagination and idempotency keys · a streaming endpoint with cancellation, heartbeats and in-band errors · an async layer with semaphores, layered timeouts and jittered backoff · a multi-stage non-root container that drains on `SIGTERM` · an offline test suite.
**Acceptance.** Client disconnect stops upstream generation within one token · `docker stop` mid-stream exits cleanly · `pytest` passes with no network and no API key in under 10 s · measured p50/p99 for the blocking-call collapse, documented.
**Why it exists.** Every later module ships inside this service.

---

## Lecture 11 — Byte-pair encoding

**Week 6 · Lecture A · fully theoretical · 2 h**

- The tokenizer as the boundary between your data and the model — and the layer that sets your bill
- Byte-pair encoding: training the merges, then encoding with them
- The merge list as an ordered program, and why encoding must replay it in order
- Why byte-level, and what happens to characters outside the vocabulary
- Pre-tokenization and the regex split pattern, and what it prevents
- Vocabulary size as a trade: fewer tokens per document versus a larger embedding matrix (Lecture 4's arithmetic, applied)
- Round-trip guarantees: `decode(encode(x)) == x`, and where naive implementations break it
- Failure modes that look like model bugs: a prompt that works in English and fails elsewhere, a context window that holds half as much code as prose

**Independent work — 4 h**
- **Build (2 h).** BPE trainer and encoder over your own corpus.
- **System design drill (45 min).** *Scale duplicate file detection* — content hashing at scale, sharding, memory budget.
- **Production drill (45 min).** Measure tokens per second of your encoder on 100 MB of text. Profile it. Report the bottleneck.
- **Platform track P5 (2 h).** vLLM with an OpenAI-compatible API on a CPU backend. Engine flags that matter: KV-cache space, max sequence length, batching.

---

## Lecture 12 — Special tokens, chat templates, token economics

**Week 6 · Lecture B · theory + lab · 2 h**

**Theory (1 h)**
- Special tokens: what makes them special, and why they must be unsplittable
- Chat templates as the code that flattens a message list into one token sequence
- The BOS/EOS bugs everyone hits; applying a template twice, and how to detect it
- Why a template that disagrees with training by one token means you are prompting a different model
- Token counting: vocabulary size, and the cost penalty on non-English text and code
- What tokenization does to a context window and to a bill
- Token boundaries as a source of truncation bugs at the context edge

**Lab (1 h)**
- Round-trip property test across four scripts, emoji, whitespace runs and a code block
- Print the exact token sequence your chat template produces, delimiters included

**Independent work — 6 h**
- **Build (4 h).** A token-cost table: the same content in English, one non-English language and code — your tokenizer versus a production tokenizer, with tokens, ratio and cost per 1M at a real provider price.
- **System design drill (45 min).** *Design a prompt sharing product* — sharing model, permissions, versioning, abuse.
- **Production drill (45 min).** Add a token-count and cost estimate to every request log line in your Project 3 service.
- **Platform track P6 (2 h).** Deploy the engine to the cluster with probes and a volume. First real inference served by your own cluster.

---

## Lecture 13 — Attention: Q, K, V and the causal mask

**Week 7 · Lecture A · fully theoretical · 2 h**

- Every position emits a query, a key and a value; score, normalize, weighted-sum — the whole mechanism
- Scaled dot-product attention and why we divide by √d: saturation and dead gradients without it
- The `(B, H, T, D)` shape walk, one operation at a time
- Causal masking, and exactly what breaks without it — a beautiful training loss and worthless generation
- Multi-head attention: splitting, projecting, concatenating; why heads are a reshape, not a loop
- The output projection, and where it sits in the parameter count
- What attention does *not* know: order. Deferred to Lecture 16

**Independent work — 4 h**
- **Build (2 h).** Causal multi-head attention from scratch with a shape assertion at every step.
- **System design drill (45 min).** *Design a low-latency GPU inference service* — first attempt, before you know the serving material.
- **Production drill (45 min).** Train a tiny model with and without the mask; keep both loss curves and both generations as evidence.
- **Platform track P7 (2 h).** Helm: templates, values, releases, upgrades and rollbacks. One chart, three environments.

---

## Lecture 14 — Attention cost: quadratic time, KV bytes, GQA

**Week 7 · Lecture B · theory + lab · 2 h**

**Theory (1 h)**
- Two separate stories: compute is quadratic in sequence length; decode memory is the KV cache
- Quadratic time and quadratic attention memory without a fused kernel — where it hurts and where it does not
- MQA, GQA and MLA: the sharing ratio as a direct divisor on cache bytes, and the quality cost
- Reading attention cost in bytes moved per generated token, not in adjectives
- FlashAttention as an IO-aware algorithm that never materializes the full matrix; tiling against the memory hierarchy **[extension]**

**Lab (1 h)**
- Add an `n_kv_heads` switch to your attention module
- Write `cache_bytes(config, batch, seq)` printing MHA, GQA and MQA side by side, plus bytes moved per decode step

**Independent work — 6 h**
- **Build (4 h).** Benchmark your attention at four sequence lengths and plot time against length; identify where the quadratic term takes over.
- **System design drill (45 min).** *Schedule large and small model inference on an eight-GPU pool* — bin-packing, fragmentation, starvation.
- **Production drill (45 min).** Reconcile `cache_bytes` with `memory_math.py` from Project 2. If they disagree, one of them is wrong — find out which.
- **Platform track P8 (2 h).** A request router in front of multiple engine replicas; service discovery and session affinity, and why affinity interacts with prefix caching.

---

## Lecture 15 — The transformer block

**Week 8 · Lecture A · fully theoretical · 2 h**

- The block as the unit that repeats: attention mixes positions, the FFN processes each position independently
- Residual connections as a correction to a stream, and why depth trains at all because of them
- The residual stream as the model's working memory
- Pre-norm versus post-norm inside a real block, and the training behaviour of each
- SwiGLU and the feed-forward expansion factor
- Why the FFN holds roughly two-thirds of the non-embedding parameters
- Assembling blocks into a model: embedding, N blocks, final norm, output head
- Reading a model config as an arithmetic problem, not a configuration file

**Independent work — 4 h**
- **Build (2 h).** The full GPT: embedding, stacked blocks, final norm, tied head — training on your own tokenizer's output.
- **System design drill (45 min).** *Design large model-weight distribution to GPU workers* — pull storms, caching, cold-start time.
- **Production drill (45 min).** Measure tokens/second of your model at three batch sizes on CPU; state whether you are compute- or memory-bound and why.
- **Platform track P9 (2 h).** Prometheus, ServiceMonitor, and scraping engine metrics.

---

## Lecture 16 — Positional encoding, RoPE, parameter counting

**Week 8 · Lecture B · theory + lab · 2 h**

**Theory (1 h)**
- Why attention is permutation-invariant without positional information
- Absolute → learned → RoPE → ALiBi, and what each does at the edge of the trained length
- RoPE specifically: the rotation, the frequency schedule, why the dot product ends up depending on *relative* distance, and why it dominates current models
- Where RoPE is applied (Q and K only, not V) and what that implies
- ALiBi as the linear-bias alternative
- Where the parameters actually sit: embeddings, attention projections, FFN, norms, head

**Lab (1 h)**
- Implement RoPE in your attention module
- Write `param_count(config)` and make it reproduce a published model exactly

**Independent work — 6 h**
- **Build (4 h).** Read a reference implementation line by line and write the diff against yours, with a reason for every difference.
- **System design drill (45 min).** *Design model weight distribution* — registry, replication, versioning, rollback.
- **Production drill (45 min).** Take an unseen config and compute total parameters on paper to within 2%, then check with your tool.
- **Platform track P10 (2 h).** Grafana dashboards for TTFT, queue depth, KV-cache usage, tokens per second. What each metric says when latency rises.

---

## Lecture 17 — Decoding and sampling

**Week 9 · Lecture A · fully theoretical · 2 h**

- The model emits a distribution; decoding is the policy that turns it into a token
- Greedy decoding and beam search; why chat models rarely use beam
- Temperature, top-k, top-p, min-p — what each cuts from the distribution, and the order they are applied in
- Stop sequences, maximum tokens, and repetition controls
- **Why `temperature=0` is still not bit-identical across runs** — reduction order, kernel selection, non-associative floating point, tie-breaking
- Sampling as a latency/quality/cost trade, not a personality setting
- What decoding parameters belong in a versioned config rather than in a caller's request

**Independent work — 4 h**
- **Build (2 h).** Top-k, top-p and min-p on your own model, with a visualization of exactly what each stage cut.
- **System design drill (45 min).** *Design an LLM request batching system* — scheduling, fairness, SLO per tier.
- **Production drill (45 min).** Run the same prompt at `temperature=0` twenty times, count differences, and write the one-paragraph explanation you would send a stakeholder.
- **Platform track P11 (2 h).** Load testing with k6; produce a saturation curve you can point at.

---

## Lecture 18 — Constrained decoding and structured output

**Week 9 · Lecture B · theory + lab · 2 h**

**Theory (1 h)**
- Prompting shifts probability mass; constraints remove it — the difference between "usually" and "by construction"
- Constrained decoding: JSON schema, grammars, logit masking
- The parser state machine, and what it must track between tokens
- The cost: quality loss when the mask forbids the token the model wanted, plus throughput overhead
- Failure modes constraints do *not* fix: schema-valid but semantically wrong, and truncation at the token limit
- Streaming structured output while constrained — back to Lecture 8's progressive parser
- Why better prompting stops helping past a certain point

**Lab (1 h)**
- Run the JSON reliability experiment: 50 runs, prompt-only versus schema-constrained
- Report parse-success and schema-validity rates with the interval from `compare.py`

**Independent work — 6 h**
- **Build (4 h).** Wire constrained output into your Project 3 service as a real endpoint, streamed and validated.
- **System design drill (45 min).** *Review and improve a flawed design document* — find and rank the defects.
- **Production drill (45 min).** Add a schema-violation alert to the service: what fires, at what rate, and what it pages.
- **Platform track P12 (2 h).** Horizontal autoscaling, and why queue depth beats CPU as a signal. Ingress and TLS tuned for streaming: timeouts and buffering.

---

## Lecture 19 — MoE, long context, pretraining, scaling laws

**Week 10 · Lecture A · fully theoretical · 2 h**

- Mixture of experts: routing, top-k experts, the load-balancing loss
- **Active versus total parameters** — why an MoE with fewer FLOPs can be harder to serve: speed is set by active parameters, memory by total
- Expert imbalance as a production failure mode
- Long context: RoPE scaling, position interpolation, YaRN
- Why a "128k model" can degrade long before 128k
- Pretraining, read not run: the objective, the data pipeline, deduplication, contamination
- Scaling laws and compute-optimal budgets: what they let you decide before spending money
- Checkpointing, resume and failure recovery at scale **[extension]**

**Independent work — 4 h**
- **Build (2 h).** A written comparison of two published model cards — one dense, one MoE: active parameters, total parameters, memory to serve, deployment consequence.
- **System design drill (45 min).** *Design peer-to-peer model distribution under a shared link cap.*
- **Production drill (45 min).** Price serving both models at your capstone's expected traffic. One table, real provider or hardware numbers.
- **Platform track P13 (2 h).** Default-deny network policy, quotas, RBAC. SLOs and burn-rate alerts. Cost and capacity planning for a fixed hardware budget.

---

## Lecture 20 — Alignment, LoRA, and prompt vs retrieve vs fine-tune

**Week 10 · Lecture B · theory + lab · 2 h**

**Theory (1 h)**
- Supervised fine-tuning: dataset format, chat templates, masking the loss on the prompt
- RLHF: reward model then policy optimization; what each stage buys
- Why a reward model trains on comparisons instead of absolute scores
- DPO as the simpler alternative, and where it differs
- RLVR — reinforcement learning from verifiable rewards, and why code and maths got there first
- Constitutional AI; reasoning models and test-time compute; distillation and what it costs in coverage
- LoRA: the low-rank decomposition, rank and alpha, which modules to target
- QLoRA and multi-LoRA serving **[extension]**
- **The decision framework: prompt versus retrieve versus fine-tune** — fine-tuning teaches behaviour and format, retrieval supplies facts
- Maintenance cost as the tiebreaker: what happens at the next base-model upgrade

**Lab (1 h)**
- Run a LoRA fine-tune on a task where you first demonstrated prompting fails
- Record before/after accuracy, adapter size in MB, training time and total cost

**Independent work — 6 h**
- **Build (4 h).** Finish Project 4. Write the one-page decision document for your task covering all three options with numbers.
- **System design drill (45 min).** *A multi-tenant fine-tuning platform* — data isolation, adapter serving, scheduling, lineage.
- **Production drill (45 min).** Serve your adapter behind the Project 3 API and measure the added latency.
- **Reading (1 h).** DPO; LoRA rank selection.

### Project 4 — GPT from scratch *(40 h, due end of week 10)*

**Deliverable.** A BPE tokenizer with a cross-language cost table · causal multi-head attention with a GQA switch printing cache bytes · a full GPT with RoPE and `param_count(config)` reproducing a published model · top-k/top-p/min-p sampling · the JSON reliability experiment · a LoRA fine-tune report · a written diff against a reference implementation.
**Acceptance.** From an empty file you write causal MHA from memory and it runs · `param_count` matches a published model within 2% · JSON reliability reported with an interval · the fine-tune report shows before/after, adapter size, time and cost.
**Why it exists.** After this, no part of a forward pass is a black box — which is what makes weeks 11–16 engineering instead of guesswork.

---

---

## Lecture 21 — GPU profiling in practice

**Week 11 · Lecture A · fully theoretical · 2 h**

- Lecture 6's roofline, now measured rather than drawn: every claim in this lecture comes off a trace
- The profiling stack: `torch.profiler` for framework-level spans, Nsight Systems for the timeline, Nsight Compute for one kernel, CUDA events for honest wall-clock
- Reading a timeline: the CPU launch queue against the GPU stream, gaps, implicit synchronization points, and where `.item()` quietly stalls everything
- **Kernel launch overhead**: why a small model is launch-bound and the GPU sits idle between kernels; CUDA graphs as the fix
- Occupancy: warps per SM, register pressure, shared-memory limits — and why higher occupancy is not the same as higher throughput
- Per-kernel classification: achieved bandwidth against peak, achieved FLOP/s against peak, arithmetic intensity computed from the counters
- Tensor-core utilization, and why an fp32 matmul leaves most of the machine unused
- H100 and A100 specifics: HBM bandwidth, L2 size, SM count — and what each one's roofline looks like next to the other
- Where fusion opportunities appear in a trace: elementwise chains, normalization, softmax, attention
- The bottleneck taxonomy: overhead-bound · memory-bound · compute-bound · synchronization-bound · dataloader-bound — with the signature each leaves on a timeline
- False conclusions people draw from profiles: averaging over warm-up, profiling with a batch size nobody uses, reading a synchronous timing as asynchronous work

**Independent work — 4 h**
- **Build (2 h).** Profile one training step of your GPT. Classify the top ten kernels by bound type, and annotate the trace with achieved bandwidth for each.
- **System design drill (45 min).** *Design a low-latency GPU inference service* — the third pass, now with kernel-level evidence.
- **Production drill (45 min).** Find the largest gap in your own timeline and explain it. Launch overhead, sync, or dataloader — name it with numbers.
- **Derivation drill (30 min).** Arithmetic intensity of a matmul and of a softmax, from first principles.
- **Reading (45 min).** Nsight metrics; profiler trace interpretation.

---

## Lecture 22 — Triton: one fused kernel

**Week 11 · Lecture B · theory + lab · 2 h**

**Theory (1 h)**
- Why fusion wins: the kernel chain moves the same tensor through HBM once per operation, and the fused version moves it once in total — bytes, not FLOPs
- The Triton programming model: programs and blocks, `tl.load` / `tl.store`, masks for ragged edges, `BLOCK_SIZE` as a tuning axis, autotuning
- What Triton hides that CUDA makes you write, and what it therefore cannot do
- Numerics in a fused reduction: two-pass versus online softmax, and keeping the max-subtraction stable at bf16
- Correctness before speed: testing a kernel against a reference at several dtypes and shapes, including the ragged tail
- **When a hand-written kernel loses**: cuBLAS-backed matmuls, `torch.compile` already fusing it, or a shape the autotuner never sees
- FlashAttention read again — no longer an extension bullet, but the same tiling idea you just implemented, applied to attention

**Lab (1 h)**
- Implement a fused softmax or RMSNorm in Triton
- Benchmark it against PyTorch eager and `torch.compile`; report achieved bandwidth against the card's peak

**Independent work — 6 h**
- **Build (4 h).** Wire your kernel into your GPT and prove correctness against the reference at fp32 and bf16. Then write the one-page verdict: bytes moved before and after, measured speedup, and why it wins or loses.
- **System design drill (45 min).** *Design a dynamically batched inference API* — revisited with kernel-level cost in hand.
- **Production drill (45 min).** Re-profile the training step with your kernel in place and show the change on the timeline, not just in a benchmark number.
- **Derivation drill (30 min).** Online softmax: derive the running max and running sum update.

---

## Lecture 23 — Distributed training mechanics

**Week 12 · Lecture A · fully theoretical · 2 h**

- What must be shared across devices: parameters, gradients, optimizer state, activations — every parallelism strategy is a choice of which one to shard
- Data parallelism: DDP, gradient all-reduce, bucketing, and overlapping communication with the backward pass
- ZeRO stages 1, 2 and 3, and FSDP as their implementation: shard optimizer state, then gradients, then parameters; all-gather on demand, reshard after use
- The collectives: all-reduce, reduce-scatter, all-gather, broadcast — and what each costs in bytes on the wire
- **Communication arithmetic**: ring all-reduce moves ~2×(N−1)/N × parameter bytes per step; compare that against your step time before choosing anything
- Interconnect decides strategy: NVLink versus PCIe versus InfiniBand, and why the same model shards differently on two clusters
- Tensor parallelism (splitting a matmul), pipeline parallelism (stages, microbatches, the bubble), sequence parallelism, expert parallelism
- 3D parallelism: how the combination follows from model size, memory per device and interconnect bandwidth
- Sharded checkpoints: saving state that a different world size can resume
- Failure modes: stragglers, NCCL timeouts, a rank that silently desyncs, a hang with no error
- Scaling efficiency as the metric — strong versus weak scaling, and why 70% at 8 GPUs can beat 95% at 2

**Independent work — 4 h**
- **Build (2 h).** Compute the communication bytes per step for your own model under DDP and under FSDP, then predict the scaling efficiency you expect at two GPUs.
- **System design drill (45 min).** *Deploy a large model to GPU workers* — now including how it was trained.
- **Production drill (45 min).** Write the memory equation for FSDP full-shard on your model and compare it with `memory_math.py`.
- **Derivation drill (30 min).** Ring all-reduce cost: derive the 2×(N−1)/N factor.
- **Reading (45 min).** ZeRO; FSDP sharding strategies.

---

## Lecture 24 — Multi-GPU run

**Week 12 · Lecture B · theory + lab · 2 h**

**Theory (1 h)**
- Choosing the config: model size against device memory, then the smallest strategy that fits — DDP before FSDP before anything exotic
- Mixed precision under sharding: where the master weights live, how clipping works on a sharded gradient
- Effective batch size, gradient accumulation and learning-rate scaling across ranks
- Measuring scaling efficiency honestly: same effective batch, same data, warm-up excluded, tokens per second as the unit
- What a profiler shows on two GPUs: the all-gather before a layer, the reduce-scatter after, and whether they overlap with compute
- Renting GPUs for a few hours, safely: spend cap first, a run that checkpoints, and results written somewhere durable

**Lab (1 h)**
- Run your GPT on two GPUs with FSDP
- Measure tokens per second at one and two GPUs, compute scaling efficiency, and locate the communication overhead in the trace

**Independent work — 6 h**
- **Build (4 h).** Finish Project 5. Save and resume a sharded checkpoint, then run one alternative configuration (DDP against FSDP full-shard) and report the difference in memory and throughput.
- **System design drill (45 min).** *Design large model-weight distribution to GPU workers.*
- **Production drill (45 min).** Compare predicted against measured: your Lecture 23 communication estimate versus what the trace shows. Explain the gap.
- **Derivation drill (30 min).** Adam bias correction: derive the 1/(1−β^t) terms.

### Project 5 — Kernel and scaling report *(16 h, due end of week 12)*

**Deliverable.** An annotated profiler trace of your training step with the top ten kernels classified by bound type · one fused Triton kernel, correctness-tested against the reference at two dtypes, benchmarked against eager and `torch.compile` with achieved bandwidth against peak · a two-GPU FSDP run with measured scaling efficiency · a predicted-versus-measured table for both memory and communication bytes.
**Acceptance.** Every performance claim points at a trace or a benchmark, never at an adjective · your fused kernel is correct at the ragged tail, not only at power-of-two shapes · you explain your scaling efficiency number, including the part you lost to communication.
**Why it exists.** After this you can answer "why is it slow" with a measurement instead of a theory — the difference between an applied AI engineer and someone who reads model cards.

## Lecture 25 — Prefill, decode and the KV cache

**Week 13 · Lecture A · fully theoretical · 2 h**

- Two latency numbers that behave nothing alike, and why treating them as one wrecks a capacity plan
- Prefill: the whole prompt in parallel, compute-bound, and it fills the cache
- Decode: one token at a time, re-reading weights and cache, memory-bandwidth-bound
- TTFT, TPOT, inter-token latency; throughput versus latency as separate goals
- Which user-visible complaint maps to which metric
- KV-cache sizing and eviction policies
- Fragmentation from contiguous allocation, and the waste it causes
- PagedAttention and the block table — virtual memory applied to the cache
- Copy-on-write blocks for shared prefixes
- Reading the request path from HTTP to token, naming the queue at each hop

**Independent work — 4 h**
- **Build (2 h).** Measure your own stack: TTFT, TPOT and throughput at 1, 4 and 16 concurrent requests.
- **System design drill (45 min).** *Design a low-latency GPU inference service* — second attempt. Diff against your week-7 answer.
- **Production drill (45 min).** Draw your request path and name every queue. Then find the one you did not know existed.
- **Reading (45 min).** PagedAttention, marked sections.

---

## Lecture 26 — Batching, prefix caching, quantization, speculation

**Week 13 · Lecture B · theory + lab · 2 h**

**Theory (1 h)**
- Continuous batching — iteration-level scheduling; why it raises throughput and does nothing for single-request latency
- Prefix caching and what makes a prefix cacheable; prompt layout as a performance decision
- What must never move to the front of a prompt
- Quantization for serving: GPTQ, AWQ, FP8 — what is lost and what is gained; weight-only versus activation quantization
- Speculative decoding: drafting and verification, the acceptance rate that decides whether it pays, and why it helps latency where batching cannot
- Diffusion LLMs as a different decode regime **[extension]** · on-device and edge deployment **[extension]**

**Lab (1 h)**
- Measure throughput and TTFT with and without continuous batching
- Measure TTFT with a stable versus an unstable prefix, and explain the delta

**Independent work — 6 h**
- **Build (4 h).** Restructure your service's prompts for prefix stability and prove the TTFT gain. Add a saturation curve to your dashboard.
- **System design drill (45 min).** *Design GPU inference request batching* — queue, batch former, preemption.
- **Production drill (45 min).** Someone reports "generation is slow". Write the two questions you ask and the decision tree from the answers to a lever.
- **Reading (45 min).** Continuous batching; speculative decoding.

---

## Lecture 27 — Data engineering fundamentals

**Week 14 · Lecture A · fully theoretical · 2 h**

- Why an AI engineer owns a pipeline whether they want to or not: the corpus, the eval sets, the traces, the feedback data and the labels all arrive through one
- Batch, streaming and micro-batch — how to choose, and the latency each actually buys
- ETL versus ELT; object storage as the substrate; warehouse, lake and lakehouse as layouts rather than products
- File formats: row versus columnar, Parquet, compression, predicate pushdown — and why a directory of JSON lines is a recurring bill
- Partitioning and layout: choosing a partition key, the small-file problem, compaction
- Schema management: schema-on-read versus schema-on-write, evolution, backward and forward compatibility
- **Data contracts** between producer and consumer: what a breaking change is and who pays for it
- Orchestration shapes: DAGs, scheduling, dependencies, retries, backfills — Airflow, Dagster and Prefect as three answers to the same problem
- Incremental processing: watermarks, late-arriving data, at-least-once versus effectively-once in practice
- **Idempotency as the property that makes a pipeline operable** — reruns, backfills and partial failure all depend on it
- Change data capture as the bridge from an OLTP database to your index
- Data quality as tests: freshness, volume, distribution and null-rate checks, run on every load
- Lineage and reproducibility: which data produced which index, which model and which eval number
- Cost: storage tiers, scan cost, and why a bad partition key shows up on the invoice before it shows up in latency

**Independent work — 4 h**
- **Build (2 h).** Stand up the corpus store you will index in week 16: partitioned Parquet, with scan time and bytes read measured against the same data as JSONL.
- **System design drill (45 min).** *A customer distillation pipeline* — data collection, lineage, drift, coverage.
- **Production drill (45 min).** Add four data-quality checks (freshness, volume, null rate, distribution) to the loader that fills that store, and make one of them fail on purpose.
- **Reading (45 min).** Data contracts; columnar formats.

---

## Lecture 28 — SQL to a working level

**Week 14 · Lecture B · theory + lab · 2 h**

**Theory (1 h)**
- Why an AI engineer needs this: the customer's data is in a database, and so is your metadata filter
- `SELECT`, `JOIN`, `GROUP BY`, `HAVING`; inner versus left join and the row-count trap
- CTEs and window functions: `ROW_NUMBER`, `LAG`, running totals — latest-version-per-document without collapsing rows
- Indexes: what they speed up, what they cost on write, and why a function on a column kills one
- `EXPLAIN (ANALYZE)`: sequential scan versus index scan; reading the plan top-down and finding the wrong row estimate
- Keyset pagination; N+1 queries from application code
- Transactions and isolation levels; what a long transaction does to a busy table

**Lab (1 h)**
- Write the latest-version-per-document query with `ROW_NUMBER`, then keyset pagination over your document table
- Add an index for one slow query and keep the before/after `EXPLAIN ANALYZE`

**Independent work — 6 h**
- **Build (4 h).** Design the schema the next four weeks will use: documents, chunks, versions and tenants, with real constraints. Then fix one N+1 in the service you already have.
- **System design drill (45 min).** *Find a distributed mode efficiently* — distributed aggregation, approximation, network cost.
- **Production drill (45 min).** Take your slowest query anywhere in the repo and cut its p95 in half, with the plan as evidence.
- **Reading (45 min).** Index selection; execution plans.

---

## Lecture 29 — Databases: SQL, NoSQL and vector stores

**Week 15 · Lecture A · fully theoretical · 2 h**

*The engine underneath*
- B-tree versus LSM-tree: read amplification against write amplification, and which workload each was built for
- Why the storage engine — not the marketing category — predicts the performance you will get

*Relational*
- Normalization, constraints and transactions; when the database should enforce an invariant instead of your service
- ACID in practice, and what each letter costs under load

*NoSQL families*
- Key-value, document, wide-column and graph — access pattern first, data model second
- What you give up: joins, constraints, ad-hoc queries, and a query planner that improves without you

*Distribution and consistency*
- Replication: leader/follower, quorum reads and writes, failover and the split-brain question
- Partitioning: hash versus range, hot partitions, resharding while live
- Consistency as a latency/staleness trade rather than a CAP slogan; linearizable, read-your-writes, eventual — and which one your feature actually needs
- Secondary indexes in a distributed store, and what they cost

*Vector stores*
- What a vector database actually is: an ANN index plus metadata filtering plus storage plus an operational surface
- Dedicated (Qdrant, Weaviate, Milvus) versus vector-in-Postgres (pgvector) versus a search engine with vectors (OpenSearch, Elasticsearch)
- **Pre-filter versus post-filter, and the recall trap** — filtering after the ANN search silently shrinks your top-k
- Operational reality: memory footprint, build cost, replication, snapshot and restore, multi-tenancy, rebuild on parameter change
- When the "vector database" you needed was BM25 with filters

*Choosing*
- The selection method: access patterns → consistency needs → scale → operational burden → team familiarity
- Polyglot persistence and its real cost: every extra store is another backup, another failover, another page at 03:00
- Design cases, worked in class: a chat history store · a per-tenant document index · a feature store for a ranker · an audit log · an agent transcript store · an embedding index at 100 M vectors

**Independent work — 4 h**
- **Build (2 h).** A store-selection table for your capstone: every piece of state you already have or will have — documents, chunks, vectors, traces, audit records, agent transcripts — the store it lives in, the consistency it needs, and the reason.
- **System design drill (45 min).** *Retrieval over 100 million documents, with permissions and freshness* — the storage layer defended before you have written a retriever.
- **Production drill (45 min).** Write down, now, what you predict pre-filter versus post-filter will do to `recall@5` under a restrictive metadata filter. You measure it in week 16 and check yourself against this note.
- **Reading (45 min).** LSM versus B-tree; pgvector versus a dedicated store.

---

## Lecture 30 — Message brokers and caching

**Week 15 · Lecture B · theory + lab · 2 h**

**Theory (1 h)**

*Brokers*
- Why async: decoupling, buffering, and absorbing a spike your GPU-bound backend cannot serve synchronously
- Queue versus log: RabbitMQ/SQS semantics against Kafka/Redpanda partitions, offsets, retention and replay
- Delivery semantics: at-most-once, at-least-once, effectively-once — and why the consumer's idempotency key is what actually delivers the guarantee
- Ordering guarantees and partition keys; head-of-line blocking as the price of ordering
- Consumer groups and **lag as the health metric** — the same signal that drives autoscaling in Lecture 44
- Backpressure, dead-letter queues, poison messages, retry topics with backoff
- The outbox pattern: one atomic database write plus a publish, without a distributed transaction
- Where a broker belongs in an LLM system: ingestion and embedding jobs, eval runs, long agent tasks, webhook delivery, batch scoring

*Caching*
- Cache-aside, write-through and write-behind; TTL and explicit invalidation
- Stampede and thundering herd; single-flight and jittered expiry
- Redis in practice: data structures, eviction policies, persistence, cluster mode and what it costs you
- The LLM cache stack: provider prefix cache · exact-response cache · **semantic cache and its wrong-answer risk** · embedding cache · retrieval-result cache
- **Cache keys must include model version, prompt version and index version** — or a deploy quietly serves yesterday's answers
- Measuring a cache honestly: hit rate, latency saved, cost saved, and the staleness you introduced

**Lab (1 h)**
- Put a response cache with versioned keys in front of the generation endpoint from Project 3; measure hit rate, latency and cost saved
- Stand up the work queue, dead-letter queue and idempotent consumer that week 17's ingestion will run on

**Independent work — 6 h**
- **Build (4 h).** Start Project 6: the queue and cache layer, running end to end on synthetic messages, with the store-selection table turned into real schemas and collections.
- **System design drill (45 min).** *An evaluation pipeline as a batch system* — orchestration, fan-out, queue depth, cost per run.
- **Production drill (45 min).** Poison your own queue: push a message that always fails, prove it lands in the DLQ, and prove the consumer keeps working.
- **Reading (45 min).** Delivery semantics; cache invalidation.

---

## Lecture 31 — Embeddings, chunking and index internals

**Week 16 · Lecture A · fully theoretical · 2 h**

- An embedding as a point in a space where "near" means what the training objective made it mean
- Cosine, dot product and L2: when they rank identically, and why a missing normalization is half of all embedding bugs
- Chunking: size, overlap, and structure-aware splitting against headings, tables and code blocks
- Why a chunk that averages several topics embeds near nothing
- Index internals: flat, IVF, HNSW — and the recall / latency / memory triangle
- HNSW parameters (`M`, `efConstruction`, `efSearch`); why `efSearch` is a query-time knob and `M` is not
- Recall measured against exact search as ground truth — Lecture 2's metrics, applied to your own retriever
- Why retrieval quality caps system quality: the generator cannot answer from a chunk it never received

**Independent work — 4 h**
- **Build (2 h).** Build the same corpus three ways — flat, IVF, HNSW — with `recall@10` against exact search, p95 latency and memory in one table.
- **System design drill (45 min).** *Retrieval over 100 million documents, with permissions and freshness.*
- **Production drill (45 min).** A chunking ablation at three sizes, scored with `recall@5`. Pick a size and write the sentence defending it.
- **Reading (45 min).** HNSW; chunking strategies.

---

## Lecture 32 — Hybrid search, reranking, context construction

**Week 16 · Lecture B · theory + lab · 2 h**

**Theory (1 h)**
- Why dense retrieval fails on exactly what enterprises search for: error codes, part numbers, names, rare acronyms
- BM25 and keyword search; hybrid retrieval and score fusion — why fusion is done on ranks, not raw scores
- Reranking with a cross-encoder, what it costs per query, and how deep a candidate list it should see
- Contextual retrieval: prepending document-level context to each chunk before embedding
- Late interaction (ColBERT) as the middle cost point **[extension]**
- Context construction: ordering, deduplication, budget, lost-in-the-middle
- Citations that point at real spans
- **Permission-aware retrieval — filtering before, not after**
- The failure taxonomy: nothing retrieved · wrong chunk · right chunk, wrong answer

**Lab (1 h)**
- Add BM25 and rank fusion to your retriever, then a cross-encoder reranker over the top 50
- Score `recall@5` three ways: dense, hybrid, hybrid + reranker

**Independent work — 6 h**
- **Build (4 h).** Write thirty golden question–answer pairs by hand. Add span-level citations and a pre-search permission filter.
- **System design drill (45 min).** *Knowledge management for an enterprise* — permission-aware retrieval, freshness, ownership.
- **Production drill (45 min).** Take five wrong answers and classify each by the failure taxonomy, using the retrieved chunks as evidence.
- **Reading (45 min).** Hybrid retrieval; lost-in-the-middle.

---
## Lecture 33 — Ingestion, versioning, zero-downtime reindex

**Week 17 · Lecture A · fully theoretical · 2 h**

- Incremental ingestion as a diff: it needs stable identity and an honest change signal
- Change detection: content hash, ETag, mtime — and why "modified date" lies
- Stable document and chunk ids; idempotent upsert so a re-run costs nothing
- Deletes and tombstones: the document is gone, the chunks are not
- Partial failure: retries, dead-letter queue, resumable runs — Lecture 30's broker, now carrying your corpus
- Backfill without taking the live index down
- Parsing reality: PDFs, tables, HTML boilerplate, encoding
- Cost and throughput accounting per run
- Versioning the corpus and pinning the embedding model version beside it
- **Changing the embedding model means a full rebuild** — embedding spaces do not mix
- Zero-downtime reindex: build under a new name, swap by alias, keep the old one for rollback
- Snapshots, backup, and a restore you actually perform; index-parameter changes that force a rebuild
- Retention and deletion, feeding Lecture 41 · managed versus self-hosted versus vector-in-Postgres, decided with Lecture 29's method

**Independent work — 4 h**
- **Build (2 h).** `ingest.py --since` with content hashes, stable ids and idempotent upsert. Run it twice; print the cost of the second run.
- **System design drill (45 min).** *Real-time search over a live corpus* — freshness, incremental indexing, cache invalidation.
- **Production drill (45 min).** Delete a source document and prove no chunk of it survives any index, cache, queue or backup you control.
- **Reading (45 min).** Vector store operations; alias-based deploys.

### Project 6 — Retrieval, ingestion and storage platform *(34 h, due end of week 17)*

**Deliverable.** A retrieval service over a corpus you care about: hybrid search, cross-encoder reranking, span citations, pre-search permission filtering · thirty hand-written golden pairs · `recall@5` reported three ways with intervals · a partitioned columnar corpus store with four data-quality checks · a store-selection table for every piece of state · queue-driven ingestion with a DLQ and idempotent consumers · a versioned cache layer · `ingest.py --since` costing ~nothing on an unchanged corpus · `reindex.py` with an alias swap · a performed restore.
**Acceptance.** Second ingest run cost printed and near zero · a poisoned message lands in the DLQ without stopping the consumer · cache keys carry model, prompt and index versions · alias swap under live traffic with zero errors · restore time measured, not estimated · every answer carries a citation that resolves to a real span.
**Why it exists.** This is the data half of your capstone, and the case study you will be asked about in interviews.

---

## Lecture 34 — Agents: the loop and the tool schema

**Week 17 · Lecture B · theory + lab · 2 h**

**Theory (1 h)**
- What separates an agent from a chain — and why most things sold as agents should be chains
- The loop: plan → act → observe → repeat; termination conditions
- Tool schema design: names, descriptions, parameter validation
- The tool description as a prompt the model actually reads
- Validating parameters before execution, and returning errors the model can act on
- Context management: compaction, summarization, what to drop and when
- Keeping the tool result the model needs, not the tool result the API returned
- MCP as a tool-transport standard: servers, clients, and what it buys over hand-rolled wiring
- Where the loop's state lives, and why that is the first architectural decision (opened up in Lecture 35)

**Lab (1 h)**
- Build an agent with three tools — one of them your retrieval service — with validated parameters and explicit termination
- Feed it a deliberately malformed tool result and watch what it does

**Independent work — 6 h**
- **Build (4 h).** Add context compaction with a trigger you chose, and per-step logging of tokens, cost and latency.
- **System design drill (45 min).** *An agent platform: tools, sandboxing, durable state, cost caps.*
- **Production drill (45 min).** Write the sentence justifying an agent over a chain for your task. If you cannot, convert it to a chain and record what you gained.
- **Reading (45 min).** Tool-use patterns; MCP.

---

## Lecture 35 — Loop engineering, graph engineering, dynamic workflows, harnesses

**Week 18 · Lecture A · fully theoretical · 2 h**

*Loop engineering — the loop is the product*
- The control loop as the thing you actually own: the model proposes, the loop decides what happens
- Loop state versus model state: what is reconstructible from a log and what is not
- Per-iteration context assembly: what is re-sent, what is pinned, what is summarized, and the compaction trigger
- Turn budgets and progress detection: no-progress heuristics, repeated-state detection, forced termination with a useful partial result
- Idempotent steps, so a retried or resumed iteration does not duplicate a side effect
- Why an in-memory loop is a prototype: the process will die mid-task

*Graph engineering — making control flow explicit*
- Representing the workflow as a graph: nodes as steps, edges as transitions, state as an explicit object passed along
- Checkpointing at node boundaries; resume, replay and deterministic re-execution
- Human-in-the-loop as a first-class node: pause, await approval, resume — days later if necessary
- Conditional edges, retry edges and error nodes, versus `if` statements buried in a loop body
- What a graph buys: testability, resumability, observability per node. What it costs: rigidity and ceremony
- Durable execution: the graph plus a durable store equals a workflow that survives a deploy

*Dynamic workflows*
- Static graph compiled ahead of time versus a workflow constructed at runtime by the model — the testability trade
- Plan-then-execute: the model emits a plan, the runtime executes it, and the plan is an artefact you can inspect, cache and diff
- Dynamic fan-out and fan-in: dependency DAGs, parallel step detection, joining partial results
- Budget allocation per branch: tokens, steps and money split before the fan-out, not after
- Failure containment: one branch fails, the join still produces something
- When a dynamic workflow beats a loop, and when it is a loop with extra machinery

*Agent harnesses — the layer between model and environment*
- What a harness owns: tool registry, permission model, sandbox, observation formatting, truncation policy, error surfaces, session state
- **Observation formatting as a design problem**: what the model sees after a tool call, how a 40 MB output is truncated, how an error is presented so the next step is recoverable
- The action space: fewer, better tools versus many narrow ones, and the effect on completion rate
- Permission gating and blast radius: what runs unattended, what requires approval, what is never available
- Session and state model: transcripts, resume, compaction, subagents and context isolation
- **Case study — the DeepSeek harness:** loop structure, tool surface, context assembly, and how the coding task shapes the design
- **Case study — the OpenCode harness:** session/state model, provider abstraction, tool surface, permission gating, and where it diverges from the above
- What to take from a harness and what to keep yours — the same adoption test as any framework
- Harness evaluation: pass rate per task class, steps per resolved task, cost per resolved task, and the failure taxonomy per harness

**Independent work — 4 h**
- **Build (2 h).** Add a per-iteration state dump to your agent, then convert the linear loop into an explicit graph with a checkpoint at every node boundary. Kill the process mid-run and resume from the last checkpoint.
- **System design drill (45 min).** *An autonomous coding agent* — harness design, tool surface, sandboxing, durable state, cost caps.
- **Production drill (45 min).** Read the loop and tool layer of DeepSeek's and OpenCode's harnesses and write a one-page diff: how each assembles context, formats observations, truncates and terminates. Name one thing you are stealing and one you are refusing.
- **Reading (45 min).** Durable execution; graph-based orchestration.

---

## Lecture 36 — Rails, recovery and prompt injection

**Week 18 · Lecture B · theory + lab · 2 h**

**Theory (1 h)**
- An unbounded loop is an unbounded invoice: step limits, cost caps, wall-clock timeouts, cycle detection
- Retries with backoff inside the loop; distinguishing a retryable failure from a terminal one
- Error recovery when a tool fails, returns garbage, or **returns plausible garbage** — the dangerous case, because the model will build on it
- Observability inside the loop: per-step tokens, cost and latency, joined to the node ids from Lecture 35
- Prompt injection through tool output and retrieved text; why instructions in the system prompt are not a defence
- The architectural defence: separating instruction from data, constraining what the model can invoke, validating on the way out
- Guardrails as code: input filters, output validators, and where each belongs in the loop or the graph
- Blast radius per tool, and which tools must never be reachable from injected text

**Lab (1 h)**
- Add full rails enforced at the graph level: step limit, cost cap, cycle detection, backoff
- Add a dynamic fan-out with a per-branch budget and a join that tolerates a failed branch

**Independent work — 6 h**
- **Build (4 h).** Break it three ways on purpose — a failing tool, an unbounded task, an injected instruction — and write the report: what happened, what caught it, what you changed. Then a twenty-case red-team suite running in CI.
- **System design drill (45 min).** *A computer-use agent in production* — screenshot-act loop, latency, blast-radius control.
- **Production drill (45 min).** Set a hard per-session cost cap and prove it trips before the spend, not after.
- **Reading (45 min).** Prompt injection; guardrail design.

---

## Lecture 37 — Eval types and deterministic checks

**Week 19 · Lecture A · fully theoretical · 2 h**

- Why a non-deterministic system cannot be shipped on judgement alone
- Eval types: unit, component, end-to-end; offline versus online; what each catches and what it cannot
- Golden datasets versus synthetic datasets
- Contamination of your own eval set: forty rounds of prompt tuning turns it into a training set
- Keeping a held-out slice you look at rarely
- Deterministic checks first — free, instant, unarguable: exact match, schema validation, tool-call correctness, citation presence
- Agent-specific deterministic checks: did it call the right tool, in a valid order, within the step budget, and terminate for the right reason
- Building the case set from real failures, not from imagination
- Where pytest ends and evaluation begins, and why both gate CI

**Independent work — 4 h**
- **Build (2 h).** The deterministic half of a 40-case harness, plus the case-set structure and a held-out slice.
- **System design drill (45 min).** *Eval-gated CI/CD* — thresholds, noise, blocking policy, rollback.
- **Production drill (45 min).** Every bug you fix from now on becomes an eval case before the fix merges. Start with the last three.
- **Reading (45 min).** Eval design, marked sections.

---

## Lecture 38 — LLM-as-judge, multi-agent evaluation, the CI gate

**Week 19 · Lecture B · theory + lab · 2 h**

**Theory (1 h)**

*The judge as an instrument*
- A judge is a measuring instrument, and nobody quotes an uncalibrated instrument
- Writing a rubric, choosing a scale, calibrating it; judge biases — position, verbosity, self-preference — and the mitigation for each
- **Validating the judge against your own labels**, and reporting agreement alongside every score
- The RAG triad: faithfulness, answer relevance, context relevance — and using it to localize a failure to retrieval or generation

*Evaluating agents and multi-agent systems*
- Why end-state scoring is not enough: the same answer in three steps or thirty is not the same system
- Trajectory evaluation: step-level correctness, tool-selection accuracy, argument correctness, redundant and looping steps
- Task completion, step efficiency, cost per resolved task — the harness metrics from Lecture 35, measured formally
- Multi-agent specifics: per-role scoring, hand-off correctness, message-passing overhead, and attributing a failure to the agent that caused it
- **DeepEval in practice:** metrics as code, `G-Eval` for rubric-based scoring, task-completion and tool-correctness metrics, conversational and multi-turn metrics, component-level evaluation of a traced run, and running it all under pytest so it gates CI like any other test
- Where DeepEval, RAGAS and your own deterministic checks each belong — and what you still write yourself
- Public benchmarks and leaderboards: what they measure, contamination, and why they do not replace your thirty cases

*The gate*
- A regression suite in CI with a threshold that blocks a deploy without blocking on noise
- Quality drift: tracking eval scores and score distributions over time, not just at merge
- Reporting an eval result the way Lecture 2 taught: effect size with an interval

**Lab (1 h)**
- Hand-label 30 outputs, run the judge over the same set, compute agreement
- Add DeepEval task-completion and tool-correctness metrics over ten recorded agent runs

**Independent work — 6 h**
- **Build (4 h).** The full 40-case harness in CI: deterministic checks, validated judge, DeepEval agent metrics, threshold derived from your own measured noise floor.
- **System design drill (45 min).** *An evaluation pipeline as a batch system* — orchestration, judge fan-out, cost per run.
- **Production drill (45 min).** Push a deliberately worse prompt and prove CI goes red for the right reason, not by accident.
- **Reading (45 min).** LLM-as-judge validation; DeepEval metric definitions.

---

## Lecture 39 — Experimentation and A/B testing

**Week 20 · Lecture A · fully theoretical · 2 h**

- Where this sits: Lecture 38 gave you an eval harness that *predicts* whether a change is better. This pair is how a change is *proven* on real traffic — and in a product company that is the only argument that ships anything
- The experiment as a hypothesis with a decision rule written **before** the data arrives
- Randomization units: user, session, request, account — and why the wrong unit invalidates the whole test
- Choosing the primary metric, and the discipline of exactly one; guardrail metrics that can only stop a launch, never justify one
- Proxy metrics and their failure: optimizing click-through while retention falls
- **Sample size and power**, computed before launch: minimum detectable effect, baseline variance, traffic per day → how long the test must run
- Statistical significance versus practical significance; a real effect too small to pay for
- **Peeking and sequential testing**: why checking daily inflates false positives, and the two correct answers — fixed horizon, or a sequential test designed for it
- Multiple comparisons across variants, metrics and segments — Lecture 2's problem at product scale
- Variance reduction: CUPED, stratification, and paired designs, so the test needs less traffic
- Interference: network effects, shared caches, marketplace two-sidedness, and agents that learn from each other
- Novelty and primacy effects; when a two-week result reverses in month two
- Segmentation done honestly: pre-registered segments, not the one segment where p < 0.05
- Switchback and interleaving designs, and where each beats a plain A/B
- **Experimentation on LLM systems specifically:** non-determinism as extra variance · cost and latency as guardrails, always · prompt, model and index versions as separate treatments · caches that leak one arm into the other · a judge score as a metric, only if it was validated
- Online against offline: your eval harness predicts, the experiment decides — and the gap between them is itself a metric worth tracking
- The rollout ladder: offline eval → shadow → canary by percentage → holdback → full launch, with the exit criterion for each rung
- Reading someone else's experiment report critically: unit, horizon, power, guardrails, and what was not reported

**Independent work — 4 h**
- **Build (2 h).** A power calculation for one real change to your capstone: baseline, variance, minimum detectable effect, traffic, and the run length that falls out of it.
- **System design drill (45 min).** *A recommendation or ranking system* — funnel, features, position bias, and now the experiment that would prove a change.
- **Production drill (45 min).** Take a change you already made and write the experiment that would have justified it. Include the guardrails and the stop rule.
- **Derivation drill (30 min).** Sample size for a two-proportion test, from the variance of a Bernoulli.
- **Reading (45 min).** Sequential testing; CUPED.

---

## Lecture 40 — Online evaluation and the shipping decision

**Week 20 · Lecture B · theory + lab · 2 h**

**Theory (1 h)**
- Shadow traffic: running the new system on real requests with the output thrown away — what it proves and what it cannot
- Canary by percentage, with automated rollback on a guardrail breach rather than on someone noticing
- Interleaving for ranked results: two systems, one user, far less traffic needed
- Holdbacks: a permanent small slice on the old behaviour, so long-term drift stays visible
- Feedback as data: thumbs, edits, retries, escalations, abandonment — what each one really measures, and the bias in every one of them
- Implicit signals for LLM products: copy events, regeneration, conversation length, hand-off to a human, task completion
- Attaching outcomes back to the request that produced them — and instrumenting for it now, so that Lecture 42's spans carry the label from the day you build them
- Turning production disagreements into eval cases automatically: the loop from complaint to golden set
- The shipping decision itself: what evidence is enough for a prompt change, a model swap, a retrieval change, an index rebuild — different rungs of the ladder for different blast radii
- Writing the launch memo: hypothesis, design, result with an interval, guardrails, decision, and what you would watch after launch
- When not to experiment: no traffic, an obvious defect, a compliance requirement, or a change too small to detect — and saying so instead of theatre

**Lab (1 h)**
- Split traffic on your own service by a hashed randomization unit, log the assignment beside every request, and verify the arms are balanced
- Run the two arms and compute the result with `compare.py` from Lecture 2 — effect size with an interval, plus your guardrails

**Independent work — 6 h**
- **Build (4 h).** Ship one real change through the full ladder on your capstone: offline eval → shadow → canary with an automatic rollback rule → decision. Write the launch memo.
- **System design drill (45 min).** *Customer support automation* — the deflection metric, the guardrail, and the experiment that proves it.
- **Production drill (45 min).** Wire one implicit feedback signal into your traces and check whether it correlates with your judge score. Report the correlation, whatever it is.
- **Derivation drill (30 min).** Why paired comparison has lower variance than two independent samples.

---

## Lecture 41 — Secrets, PII, compliance, audit, tenancy

**Week 21 · Lecture A · fully theoretical · 2 h**

- Secrets: never in code, in a prompt, in a log, or in a trace; environment variables versus a secret manager; rotation without downtime; per-tenant and per-environment keys; pre-commit and history scanning
- PII classes and special-category data; detection by regex versus NER and the recall you actually get
- **Redaction versus pseudonymization versus anonymization** — and which one exits the regulation
- Where to apply it: before the provider, before the index, before logs and traces
- PII inside embeddings; PII inside eval datasets, recorded fixtures and agent transcripts
- Measuring your own false-negative rate by hand-labelling a sample
- Controller versus processor; DPAs and subprocessors — calling a hosted API adds one
- International transfers, and why "EU region only" appears in tenders
- **Right to erasure versus a vector index, a cache, a queue, an agent transcript and fine-tuned weights**
- Retention limits on traces and logs; on-premise and air-gapped deployment: mirrored weights, private registries, offline flags
- Open weights versus hosted API as a consequence of the deployment, not a preference
- Supporting a customer whose data you are not allowed to see
- Audit records: actor, tenant, time, action, resource, outcome — separated from debug logs, with different retention
- Tenant isolation: collection-per-tenant versus metadata filter, and the failure mode of each
- RBAC versus ABAC, and propagating the caller's identity to the retrieval filter and to every tool the agent can call
- AI governance: model and data inventories, risk classification, and the obligations that follow

**Independent work — 4 h**
- **Build (2 h).** Redaction on three paths — before the provider, before the index, before logs — with your own measured false-negative rate. Plus the cross-tenant leak test in CI.
- **System design drill (45 min).** *Multi-tenant SaaS with AI features* — isolation, quotas, cost attribution, leak tests.
- **Production drill (45 min).** Write the one-page data map: every place data lands, its retention, its deletion path. Include the queue, the caches and the agent transcript store.
- **Reading (45 min).** GDPR roles and erasure; tenancy patterns.

---

## Lecture 42 — Observability and operations

**Week 21 · Lecture B · theory + lab · 2 h**

**Theory (1 h)**
- A trace as one request's causal tree: retrieval, rerank, generation, tool calls, graph nodes — with tokens and cost attached
- Traces, spans, attributes, context propagation across async boundaries and across a queue
- OpenTelemetry GenAI semantic conventions: model, tokens, cost, tool calls
- Sampling — and always sampling errors and slow requests
- Langfuse and LangSmith: sessions, traces, generations, scores; attaching eval scores and user feedback to the trace that produced them
- Masking PII before it reaches the tracing backend; self-hosted versus SaaS, decided by Lecture 41's constraints
- Debugging a production agent: complaint → trace → failing node → replay from checkpoint → regression case
- Latency budget across retrieval → rerank → generation → post-processing
- Prompt caching: what is cacheable, prefix stability, prompt layout
- The AI gateway: one egress point for routing, keys, quotas, retries and cost attribution
- Model routing and fallback chains; cost modelling and unit economics — cost per request, per session, per resolved task
- Versioning prompts, models, graphs and indexes; rollback and shadow deploys
- **Noticing a silent quality drop after a provider updates a model**
- The runbook: alerts with thresholds, two incident playbooks, rollback steps

**Lab (1 h)**
- Emit one trace spanning HTTP → retrieval → rerank → generation → tool call, with one span per graph node
- Attach an eval score to the trace that produced it

**Independent work — 6 h**
- **Build (4 h).** Finish Project 7: PII masked before export, the audit log, a latency budget table with measured actuals, and the runbook.
- **System design drill (45 min).** *Compliance automation* — audit trail, human oversight, evidence, false negatives.
- **Production drill (45 min).** Have someone break your system without telling you how. Find it from the trace UI alone, replay from the checkpoint, then add the regression case.
- **Reading (45 min).** OpenTelemetry GenAI conventions.

### Project 7 — Agent platform, eval gate, security and observability *(48 h, due end of week 21)*

**Deliverable.** An agent with three tools and full rails, broken on purpose three ways · the loop re-expressed as a checkpointed graph with resume and one dynamic fan-out with per-branch budgets · a written harness diff against DeepSeek's and OpenCode's designs · a 40-case eval harness with a validated judge and DeepEval agent metrics (task completion, tool correctness, trajectory) gating CI · a 20-case red-team suite · redaction on three paths with a measured false-negative rate · an audit log and a cross-tenant leak test · OpenTelemetry traces in self-hosted Langfuse, one span per node · a data map and a runbook.
**Acceptance.** The system runs end to end — incremental ingest, streamed answer with citations, redaction, audit record, trace, CI gate · you kill the process mid-task and resume with no duplicated side effects · you debug a deliberately broken run using only the trace UI · you answer *"delete everything about this person"* with a real procedure, naming what takes minutes, what waits for the next rebuild, and what must be designed around from the start.
**Why it exists.** This is the capstone. Weeks 22–27 harden and defend it.

---

## Lecture 43 — Agentic engineering: production code with Claude Code

**Week 22 · Lecture A · fully theoretical · 2 h**

*The premise*
- You have just built a harness by hand; this is the same problem seen from the user's side, applied to your own work
- Which task shapes pay with a coding agent and which do not: well-specified and verifiable pays, exploratory-and-unverifiable does not
- The one rule that decides quality: **the agent's output is only as trustworthy as the verification attached to it**

*Specification and context*
- Spec-first: a task brief an agent can execute cold — scope, acceptance criteria, files in play, what must not change
- Acceptance criteria as the contract, written before the work starts
- Context engineering for a repository: `CLAUDE.md` / `AGENTS.md` — conventions, commands, architecture notes; what belongs in the file versus the prompt versus a skill
- Repository shape that agents work well in: small focused files, real tests, a `Makefile` that runs everything, deterministic setup
- Why a codebase that is hard for an agent is usually hard for a new colleague

*The working loop*
- Plan mode and plan-then-execute: reviewing the plan before any file changes
- Approval gates for irreversible actions; tool permission allowlists; what is never auto-approved
- Verification loops: tests as the reward signal · TDD with an agent · never accepting an unverified diff
- **Writing clean code with an agent:** state the standard up front — file and function size limits, naming, error handling, no dead code, no speculative abstraction — put it in `CLAUDE.md` so it applies to every task, and enforce it with a review pass and a linter rather than by asking politely; make the agent delete what it replaced, and reject a diff that grows the codebase to solve a small problem
- Hooks as automation: format, lint and typecheck on write; a blocking gate on commit
- Subagents and parallelism: when to fan out, context isolation, what the parent keeps — Lecture 35's fan-out, applied to your own workflow
- Long sessions: context budget, compaction, resume, and when to start fresh instead

*Review and accountability*
- Reviewing agent output: diff discipline, reading for scope creep, and the three failure modes — plausible-but-wrong, silently weakened tests, and unrequested refactors
- The rule you cannot delegate: if you cannot explain the diff, it is not finished
- Commit hygiene and attribution when a machine wrote the first draft
- Security review of generated code: injection, secrets, unsafe defaults — the Lecture 41 checklist applied to your own repo
- Measuring the workflow: cycle time, rework rate, defects caught in review versus in production — this is an engineering practice, so it gets metrics

**Independent work — 4 h**
- **Build (2 h).** Write `CLAUDE.md` for your capstone repo: commands, conventions, code standards, architecture, and the things an agent must never do. Add hooks for format, lint and typecheck on write.
- **System design drill (45 min).** *An MCP knowledge agent* — tool surface design, versioning, permissions.
- **Production drill (45 min).** Take one real backlog task, write it as a spec with acceptance criteria, run it agentically end to end, and record cycle time and rework.
- **Reading (45 min).** Your own repository, read as an agent would: is every command discoverable, is every check runnable in one line?

---

## Lecture 44 — Production Kubernetes for applied AI

**Week 22 · Lecture B · theory + lab · 2 h**

**Theory (1 h)**

*What changes when it is not your laptop*
- The platform track gave you a cluster; this is what production adds — expensive pods, slow starts, stateful caches and a bill per node-hour
- Scheduling accelerators: the device plugin, resource requests for GPUs, node selectors, taints and tolerations, node pools per workload class
- Model weights on a cluster: image-baked versus PVC versus object storage with an init container — and the cold-start seconds each costs
- **Probes tuned for inference:** readiness that only passes after the engine is warm, liveness that does not kill a pod busy with a long generation, startup probes sized to the model load

*Rollout and lifecycle*
- Rollouts with expensive pods: surge versus unavailable, PodDisruptionBudgets, `terminationGracePeriodSeconds` long enough to drain a stream
- Draining a pod mid-stream — Lecture 10's `SIGTERM` handler meeting a real eviction
- Canary and shadow traffic for a model or prompt change; rollback as a Helm revision
- Priority classes and preemption: which workload dies when the cluster is full

*Scaling and cost*
- Autoscaling on the right signal: **queue depth and lag, not CPU** — HPA with the Prometheus adapter, or KEDA
- Scale-to-zero economics: what cold start costs against what an idle GPU costs
- Bin-packing and right-sizing: requests versus limits, and what over-requesting does to cluster cost
- Spot and preemptible nodes for batch work; GPU sharing — MIG and time-slicing — and when it is a false economy
- Capacity planning against a fixed budget, using Lecture 6's arithmetic

*Operating it*
- Namespaces, quotas and network policy for multi-tenant clusters
- Ingress for streaming: timeouts, buffering, keep-alive, and gRPC versus HTTP
- Secrets in a cluster: external/sealed secrets, per-environment values, and what a `Secret` still is not
- Engine metrics → Prometheus → Grafana → SLOs and burn-rate alerts
- Failure drills: node loss, `OOMKilled`, image pull failure, PVC unavailable, a stuck rollout

**Lab (1 h)**
- Deploy a small language model with vLLM behind an OpenAI-compatible API: PVC-cached weights, warm-up-aware readiness, a PDB and a graceful drain
- Add an HPA driven by queue depth and watch it scale under a k6 run

**Independent work — 6 h**
- **Build (4 h).** Project 8: Helm-package the SLM service, wire the Grafana dashboard, run the saturation test, and compute cost per million tokens on your own hardware.
- **System design drill (45 min).** *Schedule large and small model inference on an eight-GPU pool* — now with the scheduler details you actually know.
- **Production drill (45 min).** Kill a node while serving. Record time to recovery, dropped requests and what the dashboard showed.
- **Reading (45 min).** Probes and graceful termination; autoscaling on custom metrics.

### Project 8 — SLM inference service on Kubernetes *(24 h, due end of week 23)*

**Deliverable.** A small language model (1–3B, quantized) served by vLLM behind an OpenAI-compatible API on your cluster · a Helm chart with per-environment values · PVC-cached weights and warm-up-aware probes · a PodDisruptionBudget and a drain that does not cut a stream · an HPA on queue depth · a Grafana dashboard with TTFT, TPOT, queue depth, KV-cache usage and tokens per second · a k6 saturation run · default-deny network policy · one SLO with a burn-rate alert · a cost-per-million-tokens figure computed from your own numbers.
**Acceptance.** A rolling upgrade completes with zero dropped streams · a node kill recovers within a time you measured and documented · load drives the HPA from one replica to three on the queue-depth signal, and back down · you read the dashboard aloud and explain what each panel says when latency rises · your cost per million tokens is derived, not guessed.
**Timing.** Built in weeks 22–23, alongside the tail of Project 7 — the platform track from weeks 4–16 is the preparation for it.
**Why it exists.** Your capstone needs somewhere to run that is not a laptop, and "can this run inside our network" is the question that decides enterprise deals.

---

## Lecture 45 — Preference optimization by hand: DPO and GRPO

**Week 23 · Lecture A · fully theoretical · 2 h**

- Lecture 20 gave you the map of post-training; this lecture derives the two objectives you are most likely to implement
- The Bradley–Terry model: turning pairwise human comparisons into a probability, and why comparisons beat absolute scores
- The KL-constrained RLHF objective, stated precisely: maximize reward, stay near the reference policy
- **DPO derived from that objective**: the optimal policy in closed form, the implicit reward, and how the reward model disappears into the loss
- Reading the DPO loss term by term: chosen and rejected log-probabilities, the reference model, `β` as the strength of the KL constraint
- Why the reference model is frozen, and what happens when it drifts
- DPO failure modes: length bias, degenerate margins, both completions becoming less likely, over-optimization on a small preference set
- **GRPO**: group-relative advantage in place of a value network — sample a group, score it, normalize within the group
- Why removing the critic matters in practice, and what it costs in variance
- Verifiable rewards (RLVR) as GRPO's natural home: unit tests, exact answers, checkers — a reward that cannot be gamed by phrasing
- KL penalty, reward hacking, and the metrics that reveal it: win rate against the reference, KL drift, output-length drift
- Data requirements and cost: how many pairs, how noisy, and where the pairs come from
- Evaluating a preference run with your own harness — Lecture 34's judge, validated, not vibes

**Independent work — 4 h**
- **Build (2 h).** Implement the DPO loss from the derivation and run it on a small preference set; log the reward margin, the KL to the reference and the win rate.
- **System design drill (45 min).** *A multi-tenant fine-tuning platform* — data isolation, adapter serving, scheduling, lineage.
- **Production drill (45 min).** Run a toy GRPO loop with a verifiable reward (a checker your code can run) and show the reward curve plus one reward-hacked sample.
- **Derivation drill (30 min).** Derive the DPO gradient and say which sample the update pushes hardest.
- **Reading (45 min).** The DPO paper's objective section; a GRPO description.

---

## Lecture 46 — Paper reproduction

**Week 23 · Lecture B · theory + lab · 2 h**

**Theory (1 h)**
- Choosing a reproducible claim: one table or one figure, not a paper — and preferring a claim whose axis you can shrink
- Reading a paper for its experimental setup rather than its story: data, model, hyperparameters, metric definition, number of seeds
- What is always under-specified, and how to record the assumption you had to invent
- Scaling down honestly: what survives at 1% of the compute, what does not, and how to say which is which
- Baselines you must run yourself, because the paper's baseline is not your stack
- The reproduction report: claim · setup · deltas from the paper · results · where it matched · where it did not · your best explanation
- Why "it did not reproduce" is a publishable result inside a team, and how to write it without overclaiming
- Candidate claims, all reproducible on a laptop or a rented hour: speculative-decoding acceptance rate against draft-model size · LoRA rank ablation · quantization quality against perplexity · GQA cache size against quality · chunk size against `recall@5` · judge agreement against rubric wording

**Lab (1 h)**
- Pick the table, write the harness, and run the first cell of the grid live
- Write down every deviation from the paper before you have results, so the report is not retrofitted

**Independent work — 6 h**
- **Build (4 h).** Complete the grid and write the reproduction report with intervals from `compare.py` — Lecture 2, applied to someone else's claim.
- **System design drill (45 min).** *An evaluation pipeline as a batch system* — your grid is one.
- **Production drill (45 min).** Turn one row of your reproduction into a permanent regression case in the eval harness.
- **Derivation drill (30 min).** KL divergence between two categorical distributions, by hand.

### Project 9 — Paper reproduction report *(12 h, due end of week 23)*

**Deliverable.** One table or figure from a published paper, reproduced at reduced scale · the harness that produced it, committed and runnable in one command · a report: claim, setup, deviations, results with confidence intervals, where it matched, where it did not, and why.
**Acceptance.** Every deviation from the paper is written down before results, not after · your numbers carry intervals · you state plainly which conclusions survived your scale-down and which could not be tested at it.
**Why it exists.** Reading a claim and testing a claim are different skills, and only one of them makes you useful when a vendor or a colleague says a technique works.

## Lecture 47 — The design protocol and problem framing

**Week 24 · Lecture A · fully theoretical · 2 h**

- A design conversation is a driving exercise, not a quiz: the reviewer is measuring whether they would trust you with an ambiguous problem and a budget
- The protocol: clarify → estimate → draw → deep-dive → failure modes → trade-offs
- Clarifying: users, scale, latency budget, and what "good" means
- Estimating before designing: requests per second, storage, memory, GPU count — using Projects 2 and 4
- Drawing the boxes, then going deep on the one your reviewer picks
- Stating trade-offs out loud and naming what you gave up
- Time management across a fifty-minute conversation; saying "I don't know" well
- Business goal → task: classification, ranking, retrieval, generation; when rules beat a model
- Choosing the label, and what to do when the label does not exist yet
- Offline metric versus online metric, and the gap between them
- Defining "good enough" as a number before designing anything

**Independent work — 4 h**
- **Build (2 h).** Your protocol card, one page, used in every rep from here on. Plus a written framing — goal, task type, metric, "good enough" number — for one case.
- **System design drill (45 min).** *Design Instagram: feed, photos, friend recommendations* — a non-LLM rep on purpose.
- **Production drill (45 min).** Write the estimation numbers you should now know cold: bytes per parameter, KV bytes per token, cost per million tokens, embeddings per second, p95 per stage, cost per resolved agent task.
- **Reading (45 min).** Two reference designs, read for structure rather than content.

---

## Lecture 48 — Data, labels, features, and rep 1

**Week 24 · Lecture B · theory + rep · 2 h**

**Theory (1 h)**
- Where training data comes from; logging as a design decision — what you fail to log today you cannot train on next quarter
- Label delay and feedback loops: the model changing its own training data
- Feature computation at training time versus serving time
- **Training/serving skew**: three ways it appears and the fix for each
- Leakage at the system level, not just in a dataset
- Position bias and presentation bias in logged interactions

**Rep (1 h)**
- **Rep 1**, timed and spoken aloud, then compared against a reference solution

**Independent work — 6 h**
- **Build (3 h).** The written delta for rep 1: what the reference had, what you missed, and whether the miss was knowledge, process or time.
- **System design drill (2 × 45 min).** Two more cases from different rows of the bank.
- **Production drill (45 min).** Find a feature in your own system computed in two places. Prove the two paths agree, or make them.

---

## Lecture 49 — Serving and latency budgets, and rep 2

**Week 25 · Lecture A · fully theoretical · 2 h**

- Batch, online and streaming prediction — how to choose
- **Splitting a p95 target across stages, with numbers** — the artefact that makes the rest of the design decidable
- Multi-stage serving: cheap filter, expensive model; recall from the cheap stage, precision from the expensive one
- Caching predictions versus caching features, and their different invalidation rules
- Fallbacks and graceful degradation when the model is down
- Load shedding and admission control under saturation
- Capacity: GPU count from QPS, latency target and Lecture 6's arithmetic

**Independent work — 4 h**
- **Build (2 h).** A latency budget table for your capstone, stage by stage, with measured actuals beside the targets.
- **System design drill (45 min).** **Rep 2**, full fifty minutes, plus the written delta.
- **Production drill (45 min).** Decide which stage you cut first under pressure and what quality you lose. Write it into the runbook.
- **Reading (45 min).** Multi-stage serving; degradation patterns.

---

## Lecture 50 — Monitoring, drift, retraining, and rep 3

**Week 25 · Lecture B · theory + rep · 2 h**

**Theory (1 h)**
- The worst failure mode is not being down — it is being up, fast and wrong
- What to log: inputs, predictions, outcomes
- Data drift versus concept drift, and how each is detected
- Retraining triggers: schedule, drift, or performance — with the cost and failure mode of each
- Shadow deploys, canaries and rollback
- Choosing the alert that would have caught your last incident

**Rep (1 h)**
- **Rep 3**, timed and spoken aloud, on a case class you have not attempted

**Independent work — 6 h**
- **Build (3 h).** A monitoring plan for your capstone: three signals, thresholds, and what each is a proxy for. Wire at least one as a real alert.
- **System design drill (45 min).** Delta write-up for rep 3.
- **Production drill (2 × 45 min).** Simulate a silent quality drop (swap in a worse model without telling your own alerting) and see whether anything fires. Fix what did not.

---

## Lecture 51 — Estimation, references, and rep 4

**Week 26 · Lecture A · fully theoretical · 2 h**

- The estimation cookbook: tokens, bytes, bandwidth, QPS, storage, cost per million — derived, not memorized
- Working an estimate out loud so a reviewer can follow and correct it
- Reading a reference solution: what to take, what to reject, and how to tell an opinion from a constraint
- Classifying your own misses across four reps: knowledge, process, or time management
- Failure modes and mitigations as a first-class part of every design, not an afterthought
- The vocabulary check: router · cascade · retrieve-then-generate · plan-execute · graph with checkpoints · reflect-and-retry · human-in-the-loop gate · fallback chain · semantic cache — and the anti-patterns beside each
- Anti-patterns you can now name from experience: the agent that should have been a chain · the loop that should have been a graph · retrieval used to teach behaviour · fine-tuning used to supply facts · an unvalidated judge · unbounded loops without a cost cap · evals written after launch

**Independent work — 4 h**
- **Build (2 h).** **Rep 4** plus a consolidated delta log across all four reps, with your recurring failure mode named.
- **System design drill (45 min).** Redraw rep 1 from memory, a week later. Diff against the original.
- **Production drill (45 min).** Annotate your own capstone diagram with the pattern name for each component, and two anti-patterns you shipped and fixed.
- **Reading (45 min).** Two reference solutions in the case classes you scored worst on.

---

## Lecture 52 — Whiteboard drills, mock review, the design document

**Week 26 · Lecture B · theory + practice · 2 h**

**Theory (1 h)**
- The design document: problem, architecture, one defended decision, trade-offs, results — written for a reader who cannot ask a follow-up question
- What belongs in two-to-four pages and what does not
- Presenting numbers so they are checkable: source, method, date

**Practice (1 h) — whiteboard drills, five minutes each, from memory**
- Estimate a KV cache · size an index for 100 M documents · split a p95 budget across four stages · cost a run at 1 M requests · size a GPU pool for a stated QPS · budget an agent task in steps, tokens and money
- Then a mock review: drive fifty minutes with a peer as reviewer, and swap

**Independent work — 6 h**
- **Build (4 h).** Finish Project 10: the two-to-four-page design document for the system you built.
- **System design drill (45 min).** One final rep on your weakest case class.
- **Production drill (45 min).** Have a peer read your design document cold and write down every question it failed to answer. Fix those.

### Project 10 — System design portfolio *(20 h, due end of week 26)*

**Deliverable.** Four completed reps with written deltas · one rep redrawn from memory with the diff · a two-to-four-page design document for your own system · a pattern-annotated architecture diagram.
**Acceptance.** You drive one design end to end in fifty minutes: clarify, estimate memory and cache, draw request to token, deep-dive one component your reviewer picks, and name three failure modes with mitigations.
**Why it exists.** This is the artefact you send to a hiring manager, and the rehearsal for every design conversation you will have.

---

## Lecture 53 — Hardening

**Week 27 · Lecture A · fully theoretical · 2 h**

- The gap between a demo and a system is entirely in the failure paths
- Closing the gaps your own eval suite found
- Failure paths verified on purpose: provider down, index missing, tool timeout, cancelled client, disk full, expired key, agent stuck mid-graph
- Degradation you predicted in advance versus degradation you discovered live
- Secrets, redaction and audit verified once more against the data map
- Load: knowing your own numbers at 1, 4 and 16 concurrent requests, and where the knee is
- Rehearsing the rollback you documented — a rollback that has never been run is a hypothesis

**Independent work — 4 h**
- **Build (2 h).** Trigger every failure path on purpose; record the behaviour and whether you predicted it.
- **Production drill (45 min).** Run your documented rollback end to end, timed.
- **Production drill (45 min).** Final load table at 1, 4, 16 concurrent requests, with cost per request and cost per resolved agent task attached.
- **Reading (30 min).** Your own runbook, read as if you were on call at 03:00.

---

## Lecture 54 — Documentation and demo day

**Week 27 · Lecture B · theory + demo · 2 h**

**Theory (30 min)**
- The eval report: what you measure, on what data, with what threshold
- The runbook: three alerts, two incident playbooks, rollback steps
- A README that lets someone else run it — verified by someone else running it

**Demo (90 min)**
- Fifteen minutes each: what it does, shown live · one architectural decision you would defend under pressure · one thing that went wrong and what it taught you · the numbers: latency, cost, eval scores · questions from the room

**Independent work — 6 h**
- Final documentation pass, recorded demo, and a clean-clone run from the README by someone who has never seen the repo.

### Project 11 — Capstone *(14 h, week 27)*

**Deliverable.** The running system · the design document · the eval report · the runbook · a launch memo for one change shipped through the rollout ladder · a recorded demo · **one merged pull request to a real open-source inference or serving project** (vLLM, SGLang, Triton, llama.cpp or similar) — a doc fix does not count; a bug fix, a benchmark, a kernel or a test does.
**Acceptance.** A stranger clones the repo and runs it from the README alone · every number in the demo is reproducible from your own tooling · the failure paths behave as documented · your OSS PR is open with maintainer review requested, and you can explain the codebase around it.

---

## Lecture 55 — Behavioral interviews, and a practice system for DSA and system design

**Week 27 · Closing session (after demo day) · fully theoretical · 2 h**

*The one session that is not about the system you built. It is about converting twenty-seven weeks into an offer, and about the practice method that keeps working after the course ends.*

### Part 1 — Behavioral rounds

- What the behavioral round actually tests: judgement under constraint, scope of ownership, how you behave when you disagree, and what you did after you were wrong
- **The story inventory**: eight to ten stories, written down before you interview, drawn from this course and from your work — indexed by theme rather than by project
- The themes worth covering: conflict with a colleague · a failure with a real cost · ownership beyond your remit · ambiguity with no spec · influencing without authority · disagreeing with a manager · a deadline you missed · a decision you reversed
- STAR without sounding like a template: compress the situation, spend the time on the action, and end on a number
- **Quantify or it did not happen**: latency, cost, users, incidents, dates. "We improved reliability" is not an answer; "p99 from 4.2 s to 900 ms, incidents from six a month to one" is
- The failure story, which most people get wrong: pick a real failure, own the decision rather than the circumstances, and show what changed in your behaviour afterwards
- What the interviewer is writing down: scope, decision quality, collaboration, self-awareness, and whether the story is yours or your team's
- AI-lab specifics: mission fit, safety judgement, how you reason about a capability that could be misused, and why this company rather than the one paying more
- The questions you ask them — and what a weak answer to yours tells you about the team
- The offer conversation, briefly: never the first number, always a range with evidence behind it

### Part 2 — Running the loop well

- The shapes you will meet: coding, system design, ML and LLM depth, behavioral, take-home, bar-raiser — and what each one is really screening for
- The first five minutes of every round: confirm the goal, confirm the constraints, confirm the time
- Thinking aloud so a reviewer can follow you, and recovering out loud when you are stuck
- Saying "I don't know" well — the Lecture 47 skill, now under real pressure
- Take-homes: control the scope, ship a README and tests, and remember that reviewers read your commits
- **Your own debrief within thirty minutes of every round**: what was asked, what you missed, what you would say differently. This is the delta log from Module 6, applied to interviews
- Rejection: what it does and does not tell you, and the re-apply timeline

### Part 3 — A practice system for DSA and system design

*The point of this part: grinding produces recognition; interviews demand retrieval. Build a system that trains retrieval.*

- The failure of volume: four hundred problems solved once beats nothing, but two hundred solved on a spaced schedule beats it badly
- **The practice system, concretely:**
  - A problem log — one row per attempt: problem, date, attempt number, outcome, failure type, time taken
  - A review queue on spaced intervals: 1 day, 3 days, 7 days, 21 days
  - The rule that makes it work: a problem is *done* only when you solve it cold, from an empty file, at the last interval
- **A failure taxonomy for DSA**, because "I got it wrong" is not a diagnosis: could not model the problem · knew the pattern but botched the implementation · missed edge cases · wrong complexity analysis · ran out of time
- Pattern inventory over problem count — roughly fifteen carry most interviews: two pointers · sliding window · binary search on the answer · BFS and DFS · topological sort · union-find · heap and top-k · intervals · prefix sums · backtracking · 1D and 2D dynamic programming · shortest paths · tries · monotonic stack · bit manipulation
- Practice conditions that transfer: timed from the first day, spoken aloud, tests written before you call it finished, no editor autocomplete you will not have
- A cadence that works alongside a job: two or three problems a day for six weeks, reviewed on the queue — not eight problems on a Sunday
- **System design practice on the same machinery**: you already have the reps, the reference comparisons and the delta log from weeks 20 to 26 — put them on the same spaced schedule and redraw one from memory each week
- Grading yourself honestly against the six-part protocol from Lecture 47: clarify · estimate · draw · deep-dive · failure modes · trade-offs. Score each rep and keep the scores
- Building your own case bank: every system you read about, every incident at work, every design in this course's Appendix — each is a rep
- The mock cadence: one peer mock a week, alternating design and behavioral, recorded and rewatched once
- **Measuring progress by the right number**: cold-solve rate on re-attempts, delta-log shrinkage, mock scores — never problems attempted
- A six-week plan combining all three tracks: ~10 h/week — 5 h DSA on the queue, 3 h design reps and deltas, 2 h behavioral stories and mocks
- Anti-patterns: reading a solution and feeling productive · practising only your strong area · untimed practice · never speaking aloud · no log, so no evidence you improved

**Independent work — 6 h**
- **Build (3 h).** Stand up the practice system: the log, the review queue and the grading rubric. Seed it with ten problems and four design reps you have already done, scheduled on their intervals.
- **Build (2 h).** Write the story inventory: eight stories, each with a quantified result, at least three drawn from this course — the OOM you diagnosed, the leak test you wrote, the reproduction that did not reproduce.
- **Practice (1 h).** One recorded peer mock of each type, behavioral and design, with the debrief written within thirty minutes.

**Prove it.** Six weeks after the course ends, your log shows a cold-solve rate that moved and a delta log that shrank. If it does not, the system is not being run — and that is the only failure mode this lecture has.

---

# Topic ranking — what to protect when the schedule slips

Fifty-five lectures do not have equal weight. This is the honest ranking: by topic domain, by tier, and by what actually breaks if you skip it. Use it to triage a week you lost, to choose a reduced path, or to decide what to revise before an interview.

**Tiers**

| Tier | Meaning |
|---|---|
| **T1 — Load-bearing** | Later lectures import it directly. Skip it and the rest of the course degrades into recipes you cannot debug |
| **T2 — Job-critical** | What the day job and the interview actually ask for. Not a prerequisite for anything, but the reason you are here |
| **T3 — Differentiating** | Separates the engineer who ships from the one who decides. Needed to lead a call, not to write the code |
| **T4 — Depth** | Take it for the role you want or the curiosity you have. Nothing downstream waits on it |

---

## Ranked by domain

### 1. Measurement and judgement — the spine of everything else
| Lecture | Tier | Why it ranks here |
|---|---|---|
| **2** — Leakage, ranking metrics, proving a delta | **T1** | `compare.py` is imported by the eval gate (37–38), every experiment (39–40) and the reproduction (46). Without it you ship on noise |
| 1 — What a model is, and how you judge one | T2 | Vocabulary for the design review. Most engineers half-know it; the threshold-as-a-product-decision half is what they are missing |

### 2. Deep learning core — the byte-counting, not the mathematics
| Lecture | Tier | Why it ranks here |
|---|---|---|
| **6** — Memory accounting, roofline, failure lab | **T1** | The single most reused hour in the course. KV cache (14, 25), serving (26), capacity (44), distributed (23), every design rep. If you keep one lecture, keep this |
| 3 — Tensors, autograd, backpropagation | T2 | Makes memory explicable rather than magical |
| 4 — Losses, layers, embeddings | T2 | Softmax and temperature return in 17; embedding parameters return in 16 |
| 5 — Optimizers and a real training loop | T3 | You will rarely train. You will constantly read training logs |

### 3. Production backend — everything after this is a service
| Lecture | Tier | Why it ranks here |
|---|---|---|
| **9** — Async Python at production level | **T1** | One blocking call is the most common outage in a Python AI service, and it never raises |
| **8** — Networking primer and streaming endpoints | **T1** | Every LLM product streams, and TTFT dies in a proxy long before it dies in a model |
| 7 — API design for AI services | T2 | Idempotency and one error contract are what make a retried generation stop costing twice |
| 10 — Containers and testing LLM applications | T2 | The offline test suite is what lets you refactor an agent in week 15 without a bill |

### 4. LLM internals — built by hand so the failure modes stop being magic
| Lecture | Tier | Why it ranks here |
|---|---|---|
| **13** — Attention: Q, K, V, causal mask | **T1** | Everything from 14 to 26 is a modification of these thirty lines |
| **14** — Attention cost: KV bytes, GQA | **T1** | Turns "attention is expensive" into bytes per token — the number that decides GQA, quantization and batch size |
| **20** — Prompt vs retrieve vs fine-tune, LoRA | **T1** | The most consequential decision you will make on any AI feature, and the most asked interview question |
| **17** — Decoding and sampling | **T1** | The cheapest quality lever you own, and the one most often set by copy-paste |
| 18 — Constrained decoding, structured output | T2 | The difference between "usually parses" and "parses by construction" |
| 12 — Special tokens, chat templates, token economics | T2 | Silent quality loss, plus the multiplier on your bill |
| 11 — Byte-pair encoding | T2 | Explains the bill and the non-English penalty. Implementation depth is optional; the model of it is not |
| 15 — The transformer block | T2 | Where the parameters sit, and why depth trains |
| 16 — Positional encoding, RoPE, parameter counting | T2 | RoPE explains long-context degradation; `param_count` is a design-review skill |
| 19 — MoE, long context, pretraining, scaling laws | T3 | You will not pretrain. You will constantly read model cards and predict serving cost from them |

### 5. Inference systems — where the money and the latency live
| Lecture | Tier | Why it ranks here |
|---|---|---|
| **25** — Prefill, decode, the KV cache | **T1** | Two latency numbers that behave nothing alike. Treating them as one is why capacity plans are wrong |
| **26** — Batching, prefix caching, quantization, speculation | **T1** | The four levers, each buying a different thing. Confusing them is the classic mistake |

### 6. Retrieval — the quality ceiling of the whole system
| Lecture | Tier | Why it ranks here |
|---|---|---|
| **32** — Hybrid search, reranking, context construction | **T1** | Dense-only retrieval fails on exactly what enterprises search for. This lecture is most of your answer quality |
| **31** — Embeddings, chunking, index internals | **T1** | The generator cannot answer from a chunk it never received |

### 7. The data layer — where the incidents actually happen
| Lecture | Tier | Why it ranks here |
|---|---|---|
| **33** — Ingestion, versioning, zero-downtime reindex | **T1** | The operational half of RAG: free re-runs, tombstones, alias swaps, a restore you performed |
| 29 — Databases: SQL, NoSQL, vector stores | T2 | Heavily weighted in system design rounds, and the pre-filter/post-filter trap is a real recall bug |
| 30 — Message brokers and caching | T2 | Assumed knowledge in every design round; cache keys without a version field are a live incident |
| 27 — Data engineering fundamentals | T2 | Idempotency, contracts and partitioning are why a pipeline is operable |
| 28 — SQL to a working level | T2 | The customer's data is in a database. `EXPLAIN ANALYZE` is table stakes |

### 8. Agents — and knowing when to refuse the loop
| Lecture | Tier | Why it ranks here |
|---|---|---|
| **36** — Rails, recovery, prompt injection | **T1** | An unbounded loop is an unbounded invoice, and tool output is attacker-controlled input |
| **34** — Agents: the loop and the tool schema | **T1** | Includes the senior judgement: most things sold as agents should be chains |
| 35 — Loop, graph, dynamic workflows, harnesses | T2 | Differentiating for anyone building agent platforms; the resumability argument is the one seniors win with |

### 9. Evaluation — the only thing that lets you ship a non-deterministic system
| Lecture | Tier | Why it ranks here |
|---|---|---|
| **37** — Eval types and deterministic checks | **T1** | The cheapest, most unarguable half of evaluation. Start here or you will over-build a judge |
| **38** — Judge, multi-agent metrics, the CI gate | **T1** | An unvalidated judge is an uncalibrated instrument, and a CI gate is what makes quality a build failure |

### 10. Security, operations and the platform
| Lecture | Tier | Why it ranks here |
|---|---|---|
| **41** — Secrets, PII, compliance, audit, tenancy | **T1** | Decides whether an enterprise can buy at all. Erasure and tenancy cannot be added late |
| **42** — Observability and operations | **T1** | Without a trace, "the answer was wrong yesterday" is unanswerable |
| 44 — Production Kubernetes for applied AI | T2 | T1 if you are targeting inference or platform roles — probes, drains and queue-depth autoscaling are the interview |
| 43 — Agentic engineering with Claude Code | T2 | Compounding: it changes how fast you do everything else in this course |

### 11. Experimentation — how a change gets approved in a product company
| Lecture | Tier | Why it ranks here |
|---|---|---|
| 39 — Experimentation and A/B testing | T2 | **T1 in any product company** — this is the argument that ships things. T3 in a pure infrastructure role |
| 40 — Online evaluation and the shipping decision | T2 | The rollout ladder, and turning production disagreements into eval cases |

### 12. Performance and scale — the measurement skill
| Lecture | Tier | Why it ranks here |
|---|---|---|
| 21 — GPU profiling in practice | T3 | **T1 for inference and performance roles.** Turns "why is it slow" from a theory into a measurement |
| 23 — Distributed training mechanics | T3 | Interview-heavy and infra-critical; the communication arithmetic is the part people cannot fake |
| 22 — Triton: one fused kernel | T4 | The clearest signal of depth on a CV, and the lecture that makes FlashAttention readable |
| 24 — Multi-GPU run | T4 | Needs hardware or a rented hour. Skip last, not first |

### 13. Research depth
| Lecture | Tier | Why it ranks here |
|---|---|---|
| 45 — DPO and GRPO by hand | T3 | The two objectives you are most likely to implement, derived rather than imported |
| 46 — Paper reproduction | T3 | Reading a claim and testing a claim are different skills; only one makes you useful when a vendor says a technique works |

### 14. System design and communication
| Lecture | Tier | Why it ranks here |
|---|---|---|
| **47** — The design protocol and problem framing | **T1** | Being right is half of it. This is the other half, and it is graded in every loop |
| **49** — Serving and latency budgets | **T1** | A p95 target is not a requirement until it is split across stages with numbers |
| 51 — Estimation, references, rep 4 | T2 | The numbers you must know cold, and how to work one out loud |
| 48 — Data, labels, features | T2 | Training/serving skew is a systems bug you can see in a diagram |
| 50 — Monitoring, drift, retraining | T2 | Up, fast and wrong is the failure nothing in your infra monitoring reports |
| 52 — Whiteboard drills, mock review, design document | T2 | The artefact that travels when you are not in the room |

### 15. Capstone and career
| Lecture | Tier | Why it ranks here |
|---|---|---|
| 55 — Behavioral, interviews, the practice system | T2 | **T1 if you are job-hunting.** The practice system is the part that keeps working after week 27 |
| 53 — Hardening | T2 | The gap between a demo and a system is entirely in the failure paths |
| 54 — Documentation and demo day | T2 | The only part most people will ever see |

---

## The ten with the most leverage per hour

If you had ten lectures and no more, in this order:

| # | Lecture | What it buys |
|---|---|---|
| 1 | **6** — Memory accounting and roofline | Every capacity, hardware and "will this fit" answer for the rest of your career |
| 2 | **25** — Prefill, decode, KV cache | The two latency numbers, and why they need different fixes |
| 3 | **13** — Attention | The mechanism everything else modifies |
| 4 | **32** — Hybrid search, reranking, context | Most of your system's answer quality |
| 5 | **37** — Deterministic checks | Evaluation you can have running this afternoon |
| 6 | **2** — Proving a delta | Whether your change was real |
| 7 | **20** — Prompt vs retrieve vs fine-tune | The decision you will make on every feature |
| 8 | **9** — Async at production level | The outage you would otherwise ship |
| 9 | **33** — Ingestion and reindex | The incidents that come after launch |
| 10 | **47** — The design protocol | How you get to make any of these decisions |

---

## Reduced paths

| Path | Weeks | Lectures | For |
|---|---|---|---|
| **Ship-it** | 8 | 2 · 6 · 7 · 8 · 9 · 10 · 17 · 18 · 20 · 25 · 31 · 32 · 33 · 36 · 37 · 42 | You need one production LLM feature live and defensible, now |
| **Interview** | 13 | Ship-it + 1 · 13 · 14 · 26 · 29 · 30 · 34 · 38 · 41 · 47 · 49 · 51 · 52 · 55 | You are interviewing in three months for applied AI roles |
| **Infrastructure** | 16 | Interview minus 39–40, plus 19 · 21 · 22 · 23 · 24 · 44 | Inference, serving or platform teams |
| **Full** | 27 | All 55 | You want the whole thing, including the parts that only pay off in year two |

**Ordering constraints, and why the sequence is what it is.** 6 before 14, 23 and 25 · 13 before 14, 15 and 16 · 2 before 37, 38, 39 and 46 · **27–30 before 31**, so you know what a vector store *is* before you tune an HNSW index · 31 before 32 and 33 · 34 before 35 and 36 · **37–38 before 39–40**: offline evaluation predicts, online evaluation decides, and the two are one arc · **41 and 42 before 44**, because you deploy what you can already protect and observe · 8 and 9 before 44 · 47 before the reps in 48–52.

# Case bank — for the weekly system design drill

Two per week from week 1, one full timed rep in weeks 24–26. Pick from **different rows** each time: fluency in only one shape of problem is the failure this trains against.

## Infrastructure and serving
Design a low-latency GPU inference service · Design an LLM request batching system · Design GPU inference request batching · Design a dynamically batched inference API · Schedule large and small model inference on an eight-GPU pool · Deploy a large model to GPU workers · Design large model-weight distribution to GPU workers · Design model weight distribution · Design peer-to-peer model distribution under a shared link cap · Design a distributed rate limiter · Design a concurrent image processing service · Design a crash-resilient LRU cache · Scale duplicate file detection · Find a distributed mode efficiently

## Products and platforms
Design a prompt playground · Design a prompt sharing product · Design a resilient chat system · Design a one-on-one chat service · Design Instagram (feed, photos, friend recommendations) · Review and improve a flawed design document

## Applied AI and agent systems
Retrieval over 100 million documents with permissions and freshness · Real-time search over a live corpus · An autonomous coding agent · A coding-agent harness: tool surface, sandbox, session state · A durable workflow engine for long-running agents · A computer-use agent in production · Multi-tenant SaaS with AI features · A multi-tenant fine-tuning platform · Customer support automation · Document intelligence over PDFs at scale · Knowledge management for an enterprise · An MCP knowledge agent · An evaluation pipeline as a batch system · Eval-gated CI/CD · A customer distillation pipeline · An agent platform with sandboxing and durable state · A recommendation or ranking system · Fraud detection under a hard 100 ms budget · Compliance automation · Voice AI in healthcare · Real-time voice agents · A multimodal generation pipeline

**How to run one.** 10 min recall → 30–40 min designing aloud, alone, on paper → compare against a reference → write the delta. The delta is the part that teaches.

---

# Hour budget

| | Per week | Over 27 weeks |
|---|---|---|
| Lectures | 4 h | 110 h |
| Independent build | 8 h | 222 h |
| System design drill | 45–90 min | ~34 h |
| Production drill | 45–90 min | ~34 h |
| Derivation drill | 30 min | ~14 h |
| Platform track (weeks 4–16) | 2 h | 26 h |
| **Total** | **~15–16 h** | **~450 h** |

Projects are built inside the independent build hours, not on top of them.

# Coverage

Lectures 1–2 · metrics, leakage, significance
Lectures 3–6 · autograd through memory arithmetic
Lectures 7–10 · production backend: API, streaming, async, containers, testing
Lectures 11–20 · LLM internals, built by hand: tokenizer → attention → block → decoding → alignment → LoRA
Lectures 21–24 · performance and scale: GPU profiling, a Triton kernel, distributed training, a two-GPU run
Lectures 25–26 · inference systems: prefill and decode, batching, caching, quantization, speculation
Lectures 27–30 · the data layer: data engineering, SQL, database and vector-store trade-offs, brokers and caching
Lectures 31–33 · retrieval, and the corpus operations that keep it fresh
Lectures 34–36 · agents: the loop, loop and graph engineering, dynamic workflows, harnesses, rails
Lectures 37–40 · evaluation, offline then online: deterministic checks, judges, multi-agent metrics with DeepEval, the CI gate, then experimentation, A/B testing and the shipping decision
Lectures 41–44 · security and privacy, observability and operations, agentic engineering, production Kubernetes
Lectures 45–46 · preference optimization by hand, and a paper reproduction
Lectures 47–52 · system design and communication
Lectures 53–54 · hardening and demo
Lecture 55 · closing session — behavioral interviews, running the loop, and a practice system for DSA and system design that outlives the course
Platform track P1–P13 · weeks 4–16 · Kubernetes, vLLM, Helm, Prometheus, Grafana, k6, autoscaling, network policy — feeding Lecture 44 and Project 8

**Two standing drills, no lecture attached.** A 30-minute derivation by hand, on paper, every week — cross-entropy gradient, softmax Jacobian, Adam bias correction, attention's √d, ring all-reduce cost, online softmax, KL divergence, the DPO gradient, HNSW recall against `efSearch`. And from week 20, time reserved for the open-source pull request Project 11 requires: pick the project early, read it for two weeks, then fix something real.

Full topic-by-topic source: [MODULES.md](MODULES.md). Course terms, policies and assessment: [SYLLABUS.md](SYLLABUS.md).
