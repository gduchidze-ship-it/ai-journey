# Applied AI Engineering


**How to read a module.** Each one lists its **topics** as bullets — that is the full coverage, nothing hidden — then the **build** you produce, then the **gate** you demonstrate. Bullets marked **[ext]** are extension material: taught as reading, not examined at the gate, and there for students who want the depth.

**Books.** Module 4 follows Raschka, *Build a Large Language Model (From Scratch)*. Module 5 follows Huyen, *AI Engineering*. Everything else is taught from open sources supplied per week.

| Module | Weeks | Hours | Title |
|---|---|---|---|
| 1 | 1 | 10 | Measuring things |
| 2 | 2–3 | 28 | Deep learning core |
| 3 | 4–5 | 16 | Production backend for AI services |
| 4 | 6–10 | 62 | LLM internals: build a GPT from scratch |
| 5 | 11–16 | 88 | Applied LLM systems |
| 6 | 17–19 | 40 | System design and technical communication |
| 7 | 20 | 14 | Capstone and demo day |
| P | 4–16 | 26 | Platform lab (parallel track) |


# Module 1 — Measuring things

**Week 1 · 10 h · ML prerequisite: Module 0**

The vocabulary for proving a system works. Not a statistics course — the part of statistics that decides whether your change shipped.

### 1.1 Generalization vocabulary
- Bias and variance, model capacity, overfitting and underfitting
- Regularization: L1, L2, early stopping — what each actually constrains
- Weight decay versus L2 in the loss, and why AdamW exists
- Why a training curve alone never tells you whether to ship

### 1.2 Classification metrics and thresholds
- Confusion matrix; precision, recall, F1 — and when each is the wrong summary
- ROC and AUC versus precision–recall curves under class imbalance
- The threshold as a product decision, not a library default
- Costing a false positive against a false negative before choosing the threshold
- Why accuracy lies on imbalanced data

### 1.3 Data leakage
- Train/test contamination: the same row on both sides of the split
- Temporal leakage: using the future to predict the past; time-ordered splits
- Group leakage: the same user or document split across train and test
- Benchmark contamination in LLMs — the model already read the exam
- Why a random split flatters the score on time-ordered data

### 1.4 Ranking and retrieval metrics
- What changes when you score a ranked list instead of a yes/no answer
- recall@k and precision@k
- MRR — how high the first correct answer sits
- nDCG@k — position-weighted gain, and what it penalizes that recall does not
- Choosing k for a retrieval stage that feeds a generator
- Why a retriever with better recall@50 can still make a worse system

### 1.5 Similarity — pointer
- Cosine, dot product and L2, and when their rankings agree
- Why normalization decides whether cosine and dot product disagree
- Full treatment arrives in Module 5, week 12, where you use it

### 1.6 Comparing two systems
- Bootstrap confidence intervals from your own results
- Paired tests on the same evaluation set, and why pairing matters
- Sample size: how many cases you need to detect a given delta
- Multiple comparisons when you test twelve prompt variants
- Reporting a result honestly: effect size with an interval, never a bare number

### 1.7 Calibration
- What a confidence score does and does not mean
- Reliability diagrams: predicted probability against observed hit rate
- Platt scaling and isotonic regression **[ext]**
- Why a guardrail at threshold 0.9 fires far more often than "10% error" suggests

### 1.8 Data quality
- Missing values: drop, impute, or treat missingness as signal
- Class imbalance: resampling, class weights
- Distribution shift between training data and production traffic, and how it is detected

### 1.9 Awareness pass — one sentence each **[ext]**
- Logistic regression · decision trees and random forests · gradient boosting · naive Bayes · SVM · k-means · PCA
- Where gradient boosting still beats a neural network

---

# Module 2 — Deep learning core for Applied AI Engineer

**28h · prerequisite: Module 1**

You implement backpropagation, then learn to count bytes. The second half is the part every later module leans on.

## Week 2 — from tensors to a training loop

### 2.1 Tensors and autograd
- Indexing, `reshape` / `view` / `permute`, broadcasting rules
- `einsum`, and reading a `(B, H, T, D)` shape without guessing
- Device placement, dtypes, `.to()` semantics
- Autograd: how the graph is recorded, `backward()`, leaf tensors
- `detach` versus `no_grad`, and when each is correct
- Debugging shape errors without printing shapes first

### 2.2 Backpropagation
- The chain rule over a computation graph
- Cost and memory of the forward pass versus the backward pass
- Why every layer's activations must be kept until the backward pass
- Gradient checkpointing: recompute instead of store, and what it costs
- Why training memory grows with depth × batch × sequence length

### 2.3 Losses and softmax
- MSE versus cross-entropy, and why cross-entropy for classification
- Softmax numerical stability and the log-sum-exp trick
- Logits versus probabilities, and why losses take logits
- Temperature as a rescaling of logits — the same knob you meet again in Module 4
- What quietly breaks when you hand `CrossEntropyLoss` softmax output

### 2.4 Layers, initialization, normalization
- Activations: ReLU, GELU, SiLU; dead ReLUs
- Initialization: Xavier and He, and what breaks without them
- BatchNorm versus LayerNorm versus RMSNorm
- Why transformers use LayerNorm and not BatchNorm — two concrete reasons
- Pre-norm versus post-norm, and why pre-norm won
- Dropout, and where it still appears

### 2.5 Embedding layers
- `nn.Embedding` as a lookup table, not a one-hot matrix multiply
- Embedding parameters = vocabulary × hidden, and what share of a model that is
- Sparse gradients: only the rows you indexed receive one
- Weight tying between the input embedding and the output projection
- Why tying is a large win on a small model and nearly noise on a large one

### 2.6 Optimizers
- SGD → momentum → RMSProp → Adam → AdamW, and what each step added
- What `m`, `v`, `β1`, `β2` and `ε` do
- Optimizer-state memory: two extra tensors per parameter
- Learning-rate warmup and cosine decay; why warmup exists
- Gradient clipping, and how to read a loss spike

### 2.7 A real training run
- `nn.Module`, parameter registration, `state_dict`
- Custom layers, initialization, weight sharing
- `Dataset`, `DataLoader`, collation, worker count
- Device management and pinned memory **[ext]**
- Checkpoint and resume that survives the process being killed

### 2.8 Failure lab — deliberate breakage
- Raising batch size until the process dies; what an OOM looks like on MPS versus CPU
- Gradient accumulation: recovering the effective batch, and what it costs in wall-clock time
- Producing a loss spike on purpose: inputs scaled ×1000, randomized labels, a `NaN` sample
- Why a bad batch poisons Adam's `m` and `v` long after it is gone
- Guards that catch each case: clipping, skipping non-finite steps

### 2.9 CNN and RNN awareness
- Convolution, pooling, receptive fields, and the residual connection
- The RNN sequential bottleneck — what transformers replaced and why

## Week 3 — numerics, memory, speed

### 2.10 Memory accounting
- Training memory = parameters + gradients + optimizer states + activations
- The ~16 bytes-per-parameter figure for Adam in mixed precision, and where it comes from
- Activation memory as a function of batch × sequence × layers
- Activation checkpointing: spending compute to save memory
- Inference memory = weights + KV cache + activations
- The KV-cache formula, and what GQA does to it

### 2.11 Arithmetic intensity and the roofline
- FLOPs versus bytes moved; arithmetic intensity
- The roofline model: compute-bound versus memory-bound
- The memory hierarchy — registers, shared memory, L2, HBM — bandwidth versus capacity
- **Why prefill is compute-bound and decode is memory-bandwidth-bound**
- Why batching raises decode throughput but barely helps decode latency

### 2.12 Precision and profiling **[ext]**
- fp32, tf32, fp16, bf16, fp8 — exponent bits versus mantissa bits
- Why bf16 beat fp16 for training; what loss scaling works around
- Mixed precision: autocast, master weights
- Reading a profiler trace: overhead-bound, compute-bound or memory-bound
- Data, tensor and pipeline parallelism — what each shards and what it costs in communication

---

# Module 3 — Production backend for AI services

### 3.01 Tooling
- Python 3.11+ and a managed environment (`uv` or `venv` + pip), with a pinned lockfile
- git basics we rely on: branches, rebase-free workflow, commit hygiene, `.gitignore`
- Docker Desktop or an equivalent runtime; verifying it can build and run a container
- A code editor with a debugger you can set a breakpoint in — you will need it in Module 2

### 3.02 Local model access
- Downloading open weights, and where the model cache lives on disk
- Running a small model locally on CPU (llama.cpp / Ollama / vLLM CPU) and calling it over an OpenAI-compatible API
- Provider accounts and API keys, stored in environment variables from day one
- Setting a hard spend cap before your first API call

### 3.03 Repository and CI
- The course repository template: `src/`, `tests/`, `evals/`, `docs/`, `Makefile`
- A CI workflow that runs lint and tests on every push
- How weekly builds are submitted and how peer review is exchanged

### 3.1 API design
- Resource naming, HTTP verbs, and status codes that mean what they say
- One error contract for every failure: machine-readable code plus human message (RFC 9457)
- Pagination: cursor versus offset, and why offset breaks under concurrent writes
- Versioning: URL versus header, deprecation windows, additive-change discipline
- Idempotency keys, so a retry does not run the job twice
- Long-running work: 202 plus polling or a webhook, never a four-minute request
- Request validation at the boundary; OpenAPI as the contract, checked into the repo

### 3.2 Streaming endpoints
- The SSE wire format: `data:`, `event:`, `id:`, blank-line framing
- Why SSE rather than WebSocket for one-way token streams
- Client disconnect and cancellation — stop generating, stop paying
- Proxies that buffer your stream, and how time-to-first-token dies there
- Errors after the 200 has already been sent: in-band error events
- Parsing structured output progressively while it streams
- Heartbeats, idle timeouts, and a slow client applying backpressure

### 3.3 Async Python at production level
- The event loop; coroutine versus task; what `await` actually yields
- **One blocking call stalls every request on that worker** — and the fix
- Structured concurrency: `TaskGroup`, cancellation semantics, `asyncio.timeout`
- Concurrency limits with a semaphore; bounded queues as backpressure
- Connection pooling and keep-alive: one shared client, not one per request
- Timeouts at every layer — connect, read, total — plus the retry budget on top
- Retry with exponential backoff and jitter
- GIL versus asyncio versus processes: which problem each solves
- Uvicorn workers, and why a `def` endpoint in FastAPI runs in a thread pool

### 3.4 Containerization
- Multi-stage builds; a runtime image without the build toolchain
- Layer caching: dependencies before source, or every edit rebuilds the world
- Pinned dependencies and reproducible installs
- Non-root user, `HEALTHCHECK`, and signal handling so `SIGTERM` drains cleanly
- Configuration by environment; model weights as a volume, never a layer

### 3.5 Testing LLM applications
- pytest in practice: fixtures, `parametrize`, `monkeypatch`, markers
- Testing async code and an ASGI app in-process, with no live server
- **Mocking the provider at the HTTP boundary**, not at your own wrapper — and why that difference matters
- Recorded fixtures for real responses, including streamed chunk sequences
- A scripted fake model to drive agent branches: tool error, infinite loop, injected instruction
- **What you must never assert:** exact model text. Assert structure and invariants
- Failure-path tests: timeout, 429, truncated stream, cancelled client
- Where pytest stops and evaluation begins — and why both run in CI

---

# Module 4 — LLM internals: build a GPT from scratch 62 Hrs

The heaviest module. Type every line; do not read it.

## Week 6 — Tokenization

- Byte-pair encoding: training the merges, then encoding with them
- Why byte-level, and what happens to characters outside the vocabulary
- Special tokens, chat templates, and the BOS/EOS bugs everyone hits
- Token counting: vocabulary size, and the cost penalty on non-English text and code
- What tokenization does to a context window and to a bill
- Comparing your tokenizer against a production one on the same text

## Week 7 — Attention

- Q, K and V; scaled dot-product attention and why we divide by √d
- Causal masking, and what breaks without it
- Multi-head attention: splitting, projecting, concatenating
- Cost: quadratic time and quadratic attention memory without a fused kernel
- MQA, GQA and MLA — and what each does to KV-cache size
- Reading attention cost in bytes moved per generated token, not in adjectives
- FlashAttention as an IO-aware algorithm that never builds the full matrix **[ext]**

## Week 8 — Architecture assembly

- The transformer block: attention, feed-forward, residuals, norms
- Pre-norm versus post-norm inside a real block
- Positional encoding: absolute → learned → RoPE → ALiBi
- RoPE specifically, and why it dominates current models
- SwiGLU and the feed-forward expansion factor
- Where the parameters actually sit, and counting them from a config
- Reading a reference implementation and listing how yours differs

## Week 9 — Decoding and structured output

- Greedy decoding and beam search; why chat models rarely use beam
- Temperature, top-k, top-p, min-p — what each cuts from the distribution
- Stop sequences, maximum tokens, and repetition controls
- Why `temperature=0` is still not bit-identical across runs
- Constrained decoding: JSON schema, grammars, logit masking
- Measuring JSON reliability: prompt-only versus schema-constrained, counted over many runs
- Why better prompting stops helping past a certain point

## Week 10 — Variants, pretraining, alignment, fine-tuning

**Modern architecture variants**
- Mixture of experts: routing, top-k experts, the load-balancing loss
- Active versus total parameters — why an MoE with fewer FLOPs can be harder to serve
- Long context: RoPE scaling, position interpolation, YaRN

**Pretraining** — read, not run
- The objective, the data pipeline, deduplication, contamination
- Scaling laws and compute-optimal budgets: what they let you decide before spending money
- Checkpointing, resume and failure recovery at scale **[ext]**

**Post-training and alignment**
- Supervised fine-tuning: dataset format, chat templates, masking the loss on the prompt
- RLHF: reward model then policy optimization; what each stage buys
- Why a reward model trains on comparisons instead of absolute scores
- DPO as the simpler alternative, and where it differs
- Constitutional AI; reasoning models and test-time compute

**Fine-tuning in practice**
- LoRA: the low-rank decomposition, rank and alpha, which modules to target
- QLoRA and multi-LoRA serving **[ext]**
- Cost, adapter size and training time as decision inputs
- **The decision framework: prompt versus retrieve versus fine-tune**


Six weeks that produce your capstone. Each week adds a layer to the same service.

## Week 11 — Inference systems

- Prefill versus decode, and why they have different bottlenecks
- TTFT, TPOT, inter-token latency; throughput versus latency as separate goals
- KV-cache sizing and eviction policies
- PagedAttention and the block table
- Continuous batching — iteration-level scheduling
- Prefix caching and what makes a prefix cacheable
- Quantization for serving: GPTQ, AWQ, FP8 — what is lost and what is gained
- Speculative decoding: why it helps latency where batching does not
- Reading the request path from HTTP to token, naming the queue at each hop

## Week 12 — Retrieval

- Embeddings and similarity: cosine, dot product, L2, and the normalization traps
- Chunking: size, overlap, and structure-aware splitting
- Index internals: flat, IVF, HNSW — and the recall / latency / memory triangle
- HNSW parameters (`M`, `efConstruction`, `efSearch`) and what each trades
- BM25 and keyword search; hybrid retrieval and score fusion
- Reranking with a cross-encoder, and what it costs per query
- Context construction: ordering, deduplication, budget, lost-in-the-middle
- Citations that point at real spans
- Permission-aware retrieval — filtering before, not after
- The failure taxonomy: nothing retrieved, wrong chunk, right chunk but wrong answer
- Building a golden question set by hand, and why thirty of your own beats a public benchmark

## Week 13 — The data layer

**SQL to a working level**
- `SELECT`, `JOIN`, `GROUP BY`, `HAVING`; inner versus left join and the row-count trap
- CTEs and window functions: `ROW_NUMBER`, `LAG`, running totals
- Indexes: what they speed up, what they cost on write, why a function on a column kills one
- `EXPLAIN (ANALYZE)`: sequential scan versus index scan
- Keyset pagination; N+1 queries from application code
- Transactions and isolation **[ext]**

**Ingestion**
- Batch reprocessing versus incremental updates, and when each is right
- Change detection: content hash, ETag, mtime — and why "modified date" lies
- Stable document and chunk ids; idempotent upsert so a re-run costs nothing
- Deletes and tombstones: the document is gone, the chunks are not
- Partial failure: retries, dead-letter queue, resumable runs
- Backfill without taking the live index down
- Parsing reality: PDFs, tables, HTML boilerplate, encoding
- Cost and throughput accounting per run

**Versioning and vector-store operations**
- Versioning the corpus, and pinning the embedding model version beside it
- **Changing the embedding model means a full rebuild** — embedding spaces do not mix
- Zero-downtime reindex: build under a new name, swap by alias
- Snapshots, backup, and a restore you actually perform
- Index-parameter changes that force a rebuild
- Retention and deletion, feeding week 16
- Managed versus self-hosted versus vector-in-Postgres: the operational trade

## Week 14 — Agents

- What separates an agent from a chain, and when the chain is the better answer
- Tool schema design: names, descriptions, parameter validation
- The loop: plan → act → observe → repeat; termination conditions
- Context management: compaction, summarization, what to drop and when
- Rails: retries with backoff, step limits, cost caps, cycle detection
- Error recovery when a tool fails, returns garbage, or returns plausible garbage
- Observability inside the loop: per-step tokens, cost and latency
- Prompt injection through tool output and retrieved text
- Breaking your own agent on purpose: a failing tool, an unbounded task, an injected instruction

## Week 15 — Evaluation

- Eval types: unit, component, end-to-end; offline versus online
- Golden datasets versus synthetic datasets; contamination of your own eval set
- Deterministic checks: exact match, schema validation, tool-call correctness, citation presence
- LLM-as-judge: writing a rubric, choosing a scale, calibrating it
- Judge biases: position, verbosity, self-preference
- **Validating the judge against your own labels**, and reporting agreement
- The RAG triad: faithfulness, answer relevance, context relevance
- A regression suite in CI with a threshold that blocks a deploy
- Deciding whether a delta is real or noise — Module 1, applied

## Week 16 — Security, privacy, observability, operations

**Secrets**
- Never in code, in a prompt, in a log, or in a trace
- Environment variables versus a secret manager; what each protects against
- Rotation without downtime; per-tenant and per-environment keys
- Scanning: pre-commit hooks and history scanning

**PII**
- PII classes and special-category data
- Detection: regex versus NER, and the recall you actually get
- Redaction versus pseudonymization versus anonymization — and which one exits the regulation
- Where to apply it: before the provider, before the index, before logs and traces
- PII inside embeddings; PII inside eval datasets and recorded fixtures
- Measuring your own false-negative rate by hand-labelling a sample

**Compliance and deployment constraints**
- Controller versus processor; who is which in a vendor relationship
- Data Processing Agreements and subprocessors — calling a hosted API adds one
- International transfers, and why "EU region only" appears in tenders
- **Right to erasure versus a vector index, a cache and fine-tuned weights**
- Retention limits applied to traces and logs
- On-premise and air-gapped deployment: mirrored weights, private registries, offline flags
- Open weights versus hosted API as a consequence of the deployment, not a preference
- Supporting a customer whose data you are not allowed to see

**Audit and tenancy**
- What an audit record must contain: actor, tenant, time, action, resource, outcome
- Audit logs separated from debug logs, with different retention
- Tenant isolation: collection-per-tenant versus metadata filter, and the failure mode of each
- **The cross-tenant leak test**, asserted on every commit

**Observability**
- Traces, spans, attributes, context propagation across async boundaries
- OpenTelemetry GenAI semantic conventions: model, tokens, cost, tool calls
- One trace spanning HTTP → retrieval → rerank → generation → tool call
- Sampling, and always sampling errors and slow requests
- Langfuse and LangSmith: sessions, traces, generations, scores
- Attaching eval scores and user feedback to the trace that produced them
- Masking PII before it reaches the tracing backend
- Self-hosted versus SaaS tracing, decided by the constraints above
- Debugging a production agent: complaint → trace → failing step → replay → regression case

**Operations**
- Latency budget across retrieval → rerank → generation → post-processing
- Prompt caching: what is cacheable, prefix stability, prompt layout
- Cost modelling; model routing and fallback chains
- Versioning prompts, models and indexes; rollback and shadow deploys
- Noticing a silent quality drop after a provider updates a model
- The runbook: alerts with thresholds, two incident playbooks, rollback steps

**Build.** A retrieval service over a corpus you care about with hybrid search, reranking, citations and thirty hand-written golden pairs — reported as recall@5 three ways. An agent with three tools and full rails, broken on purpose three ways. `ingest.py --since` that costs nothing on an unchanged corpus, and `reindex.py` with an alias swap and a timed restore. A forty-case eval harness with a validated judge that gates CI. Redaction on three paths, an audit log with a leak test, a one-page data map, OpenTelemetry traces in self-hosted Langfuse, and a runbook.
**Gate.** The system runs end to end — incremental ingest, streamed answer with citations, redaction, audit record, trace, CI gate. You debug a deliberately broken run using only the trace UI. And you answer *"delete everything about this person"* with a real procedure, naming what takes minutes, what waits for the next rebuild, and what must be designed around from the start.

---

# Module 6 — System design and technical communication

**Weeks 17–19 · 40 h · prerequisite: Module 5**

Being right is half of it. This is the other half.

### 6.1 The protocol — driving the room
- Clarifying: users, scale, latency budget, and what "good" means
- Estimating before designing: requests per second, storage, memory
- Drawing the boxes, then going deep on the one your reviewer picks
- Stating trade-offs out loud and naming what you gave up
- Time management across a fifty-minute conversation
- Saying "I don't know" well

### 6.2 Framing — from a product ask to a problem with a metric
- Business goal → task: classification, ranking, retrieval, generation
- When rules beat a model
- Choosing the label, and what to do when the label does not exist yet
- Offline metric versus online metric, and the gap between them
- Defining "good enough" as a number before designing anything

### 6.3 Data, labels and features
- Where training data comes from; logging as a design decision
- Label delay and feedback loops — the model changing its own training data
- Feature computation at training time versus serving time
- **Training/serving skew**, three ways it appears and the fix for each
- Leakage at the system level, not just in a dataset

### 6.4 Serving and latency
- Batch, online and streaming prediction — how to choose
- Splitting a p95 target across stages, with numbers
- Multi-stage serving: cheap filter, expensive model
- Caching predictions and caching features
- Fallbacks and graceful degradation when the model is down

### 6.5 Monitoring, drift and retraining
- What to log: inputs, predictions, outcomes
- Data drift versus concept drift, and how each is detected
- Silent failure: the model is up, fast and wrong
- Retraining triggers: schedule, drift, or performance
- Shadow deploys, canaries and rollback

### 6.6 Reps — four timed fifty-minute designs
Chosen from:
- An inference batching service at 100k requests per second
- Retrieval over 100 million documents, with permissions and freshness
- An evaluation pipeline as a batch system: orchestration, judge fan-out, cost per run
- An agent platform: tools, sandboxing, durable state, cost caps
- A ranking or recommendation system: funnel, features, position bias
- Fraud detection under a hard 100 ms budget

Each rep: ten minutes of recall, forty minutes solo and spoken aloud, then a comparison against a reference solution and a written delta. One rep is redrawn from memory a week later.

**Build.** Four completed reps with delta notes, plus a two-to-four-page design document for one system you built.
**Gate.** Drive one design end to end in fifty minutes: clarify, estimate memory and cache, draw request to token, deep-dive one component on request, and name three failure modes with mitigations.

---

# Module 7 — Capstone and demo day

**Week 20 · 14 h**

### 7.1 Hardening
- Close the gaps your own eval suite found
- Failure paths verified: provider down, index missing, tool timeout, cancelled client
- Secrets, redaction and audit verified once more against the data map
- Load: know your own numbers at 1, 4 and 16 concurrent requests

### 7.2 Documentation
- Design document, two to four pages: problem, architecture, one defended decision, trade-offs, results
- Eval report: what you measure, on what data, with what threshold
- Runbook: three alerts, two incident playbooks, rollback steps
- README that lets someone else run it

### 7.3 The demo — fifteen minutes
- What it does, shown live
- One architectural decision you would defend under pressure
- One thing that went wrong and what it taught you
- The numbers: latency, cost, eval scores
- Questions from the room

**Deliverables.** Running system · design document · eval report · runbook · recorded demo.

---

# Track P — Platform lab

**Weeks 4–16 · ~2 h per week · 26 h · runs in parallel**

Each session is a short lesson plus a lab with an automated grader.

### P.1 Kubernetes core — weeks 4–5
- Cluster anatomy: control plane, nodes, the scheduler
- Pods, Deployments, ReplicaSets, Services
- `kubectl` for real work: describe, logs, exec, port-forward
- Namespaces, labels and selectors

### P.2 Production-shaped manifests — weeks 6–7
- ConfigMaps and Secrets, and what a Secret is not
- Liveness, readiness and startup probes — and what each one actually controls
- Resource requests and limits; what happens at the limit
- Persistent volumes for a model cache

### P.3 The inference engine — weeks 8–9
- vLLM with an OpenAI-compatible API on a CPU backend
- Engine flags that matter: KV-cache space, max sequence length, batching
- Deploying the engine to the cluster with probes and a volume
- First real inference served by your own cluster

### P.4 Packaging and scale-out — weeks 10–11
- Helm: templates, values, releases, upgrades and rollbacks
- A request router in front of multiple engine replicas
- Service discovery and session affinity

### P.5 Observability — weeks 12–13
- Prometheus, ServiceMonitor, and scraping engine metrics
- Grafana dashboards for TTFT, queue depth, KV-cache usage, tokens per second
- What each metric tells you when latency rises

### P.6 Load, autoscaling and the edge — weeks 14–15
- Load testing with k6 and a saturation curve
- Horizontal autoscaling, and why queue depth beats CPU as a signal
- Ingress and TLS tuned for streaming: timeouts and buffering

### P.7 Security and operations — week 16
- Default-deny network policy, quotas, RBAC
- SLOs and burn-rate alerts
- Cost and capacity planning for a fixed hardware budget

**Gate.** Your own capstone service deployed to the cluster, serving under load, on a Grafana dashboard you can read out loud.
