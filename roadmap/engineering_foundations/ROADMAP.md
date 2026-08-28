# Engineering Foundations — API, Data, Security, Observability — Roadmap
**Part of:** [Master Syllabus](../SYLLABUS.md) — the order all five roadmaps run in.

**Time:** ~44 h (AAI) · ~36 h (INF).
**Prereq:** Python at working level. No model material needed — this file is orthogonal to [dl_core](../dl_core/ROADMAP.md).
**Runs with:** [llm_internals](../llm_internals/ROADMAP.md) **Part B**. Do §1 *before* its §10, then wrap §2–§4 around the same builds.

> **Why this file exists.** DSA covers algorithms. [ai-ml-system-design](../ai-ml-system-design/ROADMAP.md) covers architecture. Nothing covered the layer in between — the code you are actually paid to write. Mistral's Applied AI JD asks for "APIs, back-end and front-end interfaces" in one line, and four things were missing across the whole repo:
>
> **(1) The API/back-end layer.** REST shape, streaming endpoints and SSE, error contracts, versioning. Async Python appeared only as a coding-round topic in `llm_internals` §14, never as a subject: event loop, backpressure, connection pooling, timeouts. Docker had no paragraph anywhere in the roadmaps. Testing had none either — and LLM applications have a specific testing problem (non-deterministic output, mocking the provider) that generic pytest advice does not solve.
>
> **(2) The data layer.** The RAG build in `llm_internals` §10 is one shot over 2,000 chunks. Production is a pipeline: incremental ingestion, data and index versioning, vector-DB operations (backup, migration, rebuild) — and SQL, because in Applied AI the customer's data is in a database and getting it out is your job.
>
> **(3) Security, scattered.** Prompt injection is in `llm_internals` §11, permission-aware retrieval in its §10, guardrails in its §13. Missing entirely: PII detection and redaction, data residency and GDPR, on-premise constraints, secrets management, audit logging. Mistral's customers are finance, defense, healthcare and public sector — for that role this is not a side topic, it is *why the customer chose a European vendor with an on-prem option*. Treat §3 as Core even where a table says Useful.
>
> **(4) Observability as a tool skill.** `llm_internals` §11 says "trace log written to a file" and §13 says monitoring. Neither makes you fluent in LangSmith, Langfuse or OpenTelemetry. "How do you debug a production agent?" is answered with tool names and a procedure, not with principles.

**Legend:** **Core** = asked directly, must be fluent · **Useful** = follow-up question, working knowledge · **Skim** = one paragraph · **Skip** = don't.
**Your machine:** Apple M4, 16 GB. Everything in this file runs locally. No GPU anywhere.

---

## The path

Not a sequential block. §1 is a prerequisite for building anything real; §2–§4 are layers you add to the systems you build in `llm_internals` Part B.

| Step | Topic | AAI h | INF h | When |
|---|---|---|---|---|
| 1 | Software engineering — API and back-end layer | 16 | 14 | before `llm_internals` §10 |
| 2 | Data engineering layer | 10 | 8 | with `llm_internals` §10 |
| 3 | Security, privacy and compliance | 12 | 10 | after `llm_internals` §11 |
| 4 | Observability and tracing | 6 | 4 | with `llm_internals` §11–§13 |

**One deliverable runs through all four.** Everything below upgrades the same artifact: the RAG service from `llm_internals` §10 and the agent from its §11. By the end it is an API with streaming, an incremental ingestion pipeline, a redaction and audit layer, and a trace you can debug from. That artifact is the portfolio piece; the sections are just the order you build it in.

---

# 1. Software engineering — the API and back-end layer

## 1.1 API design — 4 h (AAI) · 3 h (INF)

| Sub-topic | AAI | INF |
|---|---|---|
| Resource naming, HTTP verbs, status codes that mean what they say | Core | Core |
| Error contract: one shape for every failure, machine-readable code + human message | Core | Core |
| Pagination: cursor vs offset, and why offset breaks under writes | Core | Useful |
| Versioning: URL vs header, deprecation windows, additive-change discipline | Core | Useful |
| Idempotency keys for retryable writes | Core | Core |
| Long-running work: 202 + polling or webhook, not a 4-minute request | Core | Core |
| Request validation at the boundary (Pydantic), OpenAPI as the contract | Core | Useful |

**Plain words.** *Idempotency key* = a client-supplied id so that retrying the same request twice does not create two eval runs or charge twice. *Additive change* = you may add a field, you may never repurpose one.

**Read**
- **Start here.** [FastAPI docs](https://fastapi.tiangolo.com/) — *Tutorial* through *Request Body*, then *Handling Errors* and *Bigger Applications*. Type it. This is the framework the job uses.
- [RFC 9457 — Problem Details for HTTP APIs](https://www.rfc-editor.org/rfc/rfc9457.html). Short. It is the error shape you should copy instead of inventing one.
- [Microsoft REST API Guidelines](https://github.com/microsoft/api-guidelines/blob/vNext/azure/Guidelines.md) — the naming, versioning and pagination sections. Skim the rest. (Alternative with the same content: [Google AIP](https://google.aip.dev/), AIP-121, 132, 158, 193.)

- **Build** (~150 lines, 2 h): put a real HTTP API in front of your `llm_internals` §10 RAG. `POST /v1/answers` with a validated body; `GET /v1/documents` with **cursor** pagination; RFC 9457 errors for every failure path including a bad request, a missing document and a provider timeout; `Idempotency-Key` honoured on the write; the generated OpenAPI schema committed to the repo.
- **Done when:** you name the status code and the response body for six failures — bad input, unauthenticated, forbidden, not found, rate-limited, provider down — without hesitating, and you say why the retrying client needs each one to differ.

## 1.2 Streaming endpoints — 3 h · 3 h

| Sub-topic | AAI | INF |
|---|---|---|
| SSE wire format: `data:`, `event:`, `id:`, blank-line framing; why SSE and not WebSocket for one-way tokens | Core | Core |
| Client disconnect and cancellation — stop generating, stop paying | Core | Core |
| Proxies that buffer your stream (nginx, ingress) and how TTFT dies there | Useful | Core |
| Errors *after* the 200 has been sent: in-band error events | Core | Core |
| Partial parsing of structured output while it streams | Core | Useful |
| Heartbeats, timeouts, and a slow client applying backpressure | Useful | Core |

**Plain words.** *Backpressure* = the consumer is slower than the producer, and the system must slow down instead of buffering until it dies.

**Read**
- **Start here.** MDN — [Using server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events). The whole wire format in one page.
- [OpenAI streaming guide](https://platform.openai.com/docs/api-reference/responses-streaming) and [Anthropic streaming](https://docs.anthropic.com/en/docs/build-with-claude/streaming) — the event taxonomy you will be re-emitting. Note how both signal errors mid-stream.
- Starlette — [`StreamingResponse`](https://www.starlette.io/responses/#streamingresponse) and [request disconnection](https://www.starlette.io/requests/#request-disconnections). The cancellation half is what people get wrong.
- Repo cross-reference: `lessons/` 12 (ingress, TLS, LLM-tuned proxy timeouts and buffering) — do not re-derive it here.

- **Build** (~120 lines, 2 h): `POST /v1/answers/stream` over the same RAG. Stream tokens as SSE, emit a final `citations` event, emit an in-band `error` event when the provider fails mid-answer. Then kill the client at token 20 and prove from your own logs that generation stopped — if it did not, fix it. Finally, parse the streamed JSON progressively so a partial object never crashes the consumer.
- **Done when:** you explain what happens to a 30-second stream behind a proxy with a 10-second read timeout, what the user sees, and the two places you would fix it.

## 1.3 Async Python at production level — 5 h · 4 h

The single most common way an LLM back-end is slow for reasons that have nothing to do with the model.

| Sub-topic | AAI | INF |
|---|---|---|
| Event loop, coroutine vs task, what `await` actually yields | Core | Core |
| **One blocking call stalls every request on that worker** — `asyncio.to_thread`, executors | Core | Core |
| Structured concurrency: `TaskGroup`, cancellation, `asyncio.timeout` | Core | Core |
| Concurrency limits with a semaphore; bounded queues as backpressure | Core | Core |
| Connection pooling and keep-alive; one shared client, not one per request | Core | Core |
| Timeouts at every layer: connect, read, total, and the retry budget on top | Core | Core |
| GIL vs asyncio vs processes: which problem each solves | Useful | Core |
| Uvicorn workers, and why a `def` endpoint in FastAPI lands in a thread pool | Useful | Core |

**Read**
- **Start here.** Python docs — [asyncio](https://docs.python.org/3/library/asyncio.html): *Coroutines and Tasks* (including `TaskGroup`), *Synchronization Primitives* (`Semaphore`), *Queues*, and `asyncio.timeout`. Reference-grade, and the 3.11+ APIs are the ones to learn.
- Nathaniel J. Smith — [Notes on structured concurrency, or: Go statement considered harmful](https://vorpus.org/blog/notes-on-structured-concurrency-or-go-statement-considered-harmful/). Why `TaskGroup` exists and why loose tasks leak.
- [httpx docs](https://www.python-httpx.org/async/) — *Async support* and [*Resource limits / timeouts*](https://www.python-httpx.org/advanced/timeouts/). The pool settings are the numbers you will tune in production.

- **Build** (~120 lines, 2.5 h): an async fetcher over 500 URLs — shared `AsyncClient`, semaphore-bounded concurrency, per-request timeout, retry with backoff and jitter, results into a **bounded** queue consumed by a writer task. Measure four configurations and keep the table: (a) a fresh client per request vs one shared client, (b) with and without a `time.sleep(0.2)` deliberately dropped inside a coroutine. The blocking-call run is the lesson; write down the throughput collapse in numbers.
- **Done when:** someone shows you an async endpoint that calls `requests.get` and you say what happens to the other 50 in-flight requests, why, and the two-line fix. And you say what your bounded queue does when the consumer falls behind, and why that is better than an unbounded one.

## 1.4 Containerization — 1 h · 1 h

Short on purpose: `lessons/` 03–05 already builds, runs and deploys containers for real. This is the application-side hygiene those lessons do not stop to teach.

| Sub-topic | AAI | INF |
|---|---|---|
| Multi-stage builds; runtime image without the build toolchain | Useful | Core |
| Layer caching: dependencies before source, or every edit rebuilds the world | Core | Core |
| Pinned dependencies and reproducible installs | Core | Core |
| Non-root user, `HEALTHCHECK`, correct signal handling so `SIGTERM` drains | Useful | Core |
| Config by environment, never baked into the image; model weights are a volume, not a layer | Core | Core |

**Read**
- **Start here.** Docker — [Building best practices](https://docs.docker.com/build/building/best-practices/). One page, all of it.
- [uv in Docker](https://docs.astral.sh/uv/guides/integration/docker/) — the current way to get a fast, pinned, cache-friendly Python image.
- Repo cross-reference: `lessons/` 03–05 and `labs/solutions/charts/` for the same ideas at cluster scale.

- **Build** (~30 lines, 45 min): containerize the §1.1 API. Multi-stage, non-root, pinned lockfile, `HEALTHCHECK` hitting your readiness endpoint. Record the image size and the rebuild time after a one-line source change — then reorder the layers wrongly on purpose and record both again.
- **Done when:** you say why `COPY . .` before `pip install` makes every rebuild slow, and what a container must do when Kubernetes sends `SIGTERM` to a pod that is mid-stream.

## 1.5 Testing, including the LLM-specific part — 3 h · 3 h

| Sub-topic | AAI | INF |
|---|---|---|
| pytest: fixtures, `parametrize`, `monkeypatch`, markers, useful failure output | Core | Core |
| Testing async code and an ASGI app in-process, no live server | Core | Useful |
| **Mocking the provider at the HTTP boundary**, not at your own wrapper | Core | Useful |
| Recorded fixtures for real responses, including streamed chunk sequences | Core | Useful |
| A scripted fake model to drive agent branches: tool error, loop, injection | Core | Skim |
| **What you must not assert:** exact model text. Assert invariants and structure | Core | Core |
| Where pytest stops and evals start — and why both run in CI | Core | Useful |
| Failure-path tests: timeout, 429, truncated stream, cancelled client | Core | Core |

**Plain words.** The LLM part of your system is non-deterministic; the 90% around it is not. Test the deterministic shell hard — prompt assembly, schema validation, retries, citation extraction, tool routing — and send quality to the eval harness in `llm_internals` §12. Mocking at the HTTP boundary matters because that is the only place that also exercises your serialization, your timeout and your retry code.

**Read**
- **Start here.** [pytest docs](https://docs.pytest.org/en/stable/) — *Fixtures*, *Parametrizing*, *monkeypatch*. Then [FastAPI — Testing](https://fastapi.tiangolo.com/tutorial/testing/) and [Async Tests](https://fastapi.tiangolo.com/advanced/async-tests/).
- [respx](https://lundberg.github.io/respx/) (mock httpx at the transport layer) or [pytest-recording / VCR.py](https://github.com/kiwicom/pytest-recording) for record-and-replay cassettes. Pick one and use it everywhere.
- Cross-reference, not a re-read: `llm_internals` §12 (evals). Hamel Husain's eval posts there are the other half of this section.

- **Build** (~200 lines, 2 h): a test suite for your §1.1–§1.3 service that runs **offline, with no API key, in under 10 seconds**. At least: one streaming test that feeds a truncated chunk sequence, one 429-then-success retry test, one timeout test, one tool-error branch of the agent, one contract test that every error response validates against RFC 9457, and one test proving a cancelled client stops generation. Wire it to `pytest -q` in CI.
- **Done when:** asked "how do you test something non-deterministic", you split the answer in two — deterministic shell in pytest, quality in the eval suite — give one concrete example of each, and say exactly where the line falls in your own repo.

---

# 2. The data engineering layer

## 2.1 SQL to a working level — 4 h · 3 h

In Applied AI the customer's data is in Postgres and nobody will hand you a CSV.

| Sub-topic | AAI | INF |
|---|---|---|
| `SELECT` / `JOIN` / `GROUP BY` / `HAVING`; inner vs left join and the row-count trap | Core | Core |
| CTEs and window functions (`ROW_NUMBER`, `LAG`, running totals) | Core | Useful |
| Indexes: what they speed up, what they cost on write, why a function on a column kills one | Core | Core |
| `EXPLAIN (ANALYZE)`: reading a plan, seq scan vs index scan | Core | Core |
| Keyset pagination — the same problem as §1.1, at the query level | Useful | Useful |
| N+1 queries from application code | Core | Useful |
| Transactions and isolation, at awareness level | Skim | Useful |

**Read**
- **Start here.** [Mode SQL Tutorial](https://mode.com/sql-tutorial/) — *Basic* through *Advanced* (window functions). Free, hands-on, finishable in two sittings.
- [Use The Index, Luke](https://use-the-index-luke.com/) — chapters 1–3. The clearest free writing on indexes there is.
- PostgreSQL docs — [Using EXPLAIN](https://www.postgresql.org/docs/current/using-explain.html). Practice on [pgexercises](https://pgexercises.com/) if you want reps.

- **Build** (~1 SQL file, 2 h): run Postgres in Docker, load a real CSV of at least 100k rows. Write eight queries of rising difficulty, ending in one window function that ranks rows inside a group. Take the slowest, `EXPLAIN ANALYZE` it, add the right index, run it again, and record milliseconds before and after in a comment above the query.
- **Done when:** given an unfamiliar schema you write a three-table join with a group-by in a few minutes, and looking at a plan you say whether the index was used and why not when it was not.

## 2.2 Ingestion pipeline: batch vs incremental — 3 h · 3 h

| Sub-topic | AAI | INF |
|---|---|---|
| Batch reprocessing vs incremental updates; when each is right | Core | Core |
| Change detection: content hash, ETag, mtime — and why "modified date" lies | Core | Core |
| Stable document and chunk ids; idempotent upsert so a re-run costs nothing | Core | Core |
| Deletes and tombstones — the document is gone, the chunks are not | Core | Useful |
| Partial failure: retries, dead-letter queue, resumable runs | Core | Core |
| Backfill without taking the live index down | Useful | Core |
| Parsing reality: PDFs, tables, HTML boilerplate, encoding | Core | Skim |
| Cost and throughput accounting per run | Core | Useful |

**Plain words.** *Idempotent* = running it twice leaves the same state as running it once. For ingestion that means the second run over an unchanged corpus makes **zero** embedding calls. If it does not, you are burning money on every deploy.

**Read**
- **Start here.** LlamaIndex — [Ingestion Pipeline](https://developers.llamaindex.ai/python/framework/module_guides/loading/ingestion_pipeline/) and its docstore-backed [document management](https://developers.llamaindex.ai/python/framework/module_guides/loading/documents_and_nodes/). Read it for the upsert-and-dedup design, whether or not you use the library.
- [Unstructured](https://docs.unstructured.io/open-source/introduction/overview) — the parsing and chunking-by-title docs. This is the messy half of ingestion nobody teaches.
- Jay Kreps — [The Log](https://engineering.linkedin.com/distributed-systems/log-what-every-software-engineer-should-know-about-real-time-datas-unifying-abstraction). Read the first third. It is the mental model behind every incremental pipeline you will ever build.

- **Build** (~200 lines, 2 h): turn the one-shot §10 script into `ingest.py --since`. Content-hash each source document, derive stable chunk ids, upsert. Prove three things with printed numbers: a second run over an unchanged corpus makes **zero** embedding calls; after editing 3 documents and deleting 1, exactly those propagate and the deleted document's chunks are gone from the index; a run killed halfway resumes without duplicating work.
- **Done when:** you say what your pipeline does when the same document arrives twice, when it arrives changed, when it disappears, and when the embedding API fails on document 4,000 of 10,000.

## 2.3 Data versioning and vector-DB operations — 3 h · 2 h

| Sub-topic | AAI | INF |
|---|---|---|
| Versioning the corpus, and pinning the embedding model version next to it | Core | Core |
| **Changing the embedding model means a full rebuild** — you cannot mix embedding spaces | Core | Core |
| Zero-downtime reindex: build under a new name, swap by alias | Core | Core |
| Snapshots, backup and restore; an actual restore drill | Useful | Core |
| Index-parameter changes (`M`, `efConstruction`) that require a rebuild | Useful | Core |
| Retention and deletion, including erasure requests — feeds §3.3 | Core | Useful |
| Managed vs self-hosted vs pgvector-in-Postgres: the operational trade | Useful | Core |

**Read**
- **Start here.** Qdrant docs — [Snapshots](https://qdrant.tech/documentation/concepts/snapshots/) and [Collection aliases](https://qdrant.tech/documentation/concepts/collections/#collection-aliases). Aliases are the whole zero-downtime reindex trick, in one page.
- [pgvector](https://github.com/pgvector/pgvector) README — indexing (HNSW/IVFFlat) and the tuning notes. Read it as the on-prem answer: one database to back up instead of two systems.
- [DVC](https://dvc.org/doc/start/data-management/data-versioning) — data versioning in ten minutes. You need the idea, not the whole tool.

- **Build** (~100 lines, 2 h): version your §10 corpus. Write `reindex.py` that builds a new collection under a versioned name, verifies it with your golden queries, then swaps the alias — and prove with a loop of live queries during the swap that not one fails. Then take a snapshot, delete the collection, restore it, and time the restore. Write the five runbook lines you would want at 3 a.m.
- **Done when:** asked "you are switching to a better embedding model on Monday, what happens", you describe rebuild, dual-index verification, alias swap and rollback — with the cost and the downtime for each step.

---

# 3. Security, privacy and compliance

**Read the verdict at the top again.** For a European vendor selling to finance, defense, healthcare and public sector, this section is the product, not the paperwork. Everything here is Core for AAI even where the table softens it for INF.

## 3.1 Secrets management — 2 h · 2 h

| Sub-topic | AAI | INF |
|---|---|---|
| Never in code, never in a prompt, never in a log or a trace | Core | Core |
| Environment variables vs a secret manager; what each protects against | Core | Core |
| Kubernetes `Secret` is base64, not encryption — what actually protects it | Useful | Core |
| Rotation without downtime; per-tenant and per-environment keys | Useful | Core |
| Scanning: pre-commit hooks and history scanning | Core | Core |

**Read**
- **Start here.** OWASP — [Secrets Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html).
- [gitleaks](https://github.com/gitleaks/gitleaks) — install it, run it on this repo, wire the pre-commit hook.
- Repo cross-reference: `lessons/` 04 (ConfigMaps and Secrets) and 13 (security, tenancy, RBAC); [External Secrets Operator](https://external-secrets.io/) for where cluster secrets should come from.

- **Build** (~40 lines, 1 h): scan your own repo with gitleaks and fix anything it finds. Add the pre-commit hook. Then add a redaction filter to your logging and tracing path that masks anything matching a provider key pattern, and write a test that logs a fake key and asserts it never reaches the sink.
- **Done when:** you list every place a key exists in your stack — env, cluster secret, CI, your laptop, the trace backend — and say how you would rotate one in under an hour.

## 3.2 PII detection and redaction — 3 h · 2 h

| Sub-topic | AAI | INF |
|---|---|---|
| PII classes and what counts as special-category data | Core | Useful |
| Detection: regex vs NER, and the recall you actually get | Core | Useful |
| **Redaction vs pseudonymization vs tokenization** (reversible mapping) | Core | Useful |
| Where to apply it: before the provider, before the index, before the logs and traces | Core | Core |
| PII inside embeddings — vectors are derived data, not anonymized data | Useful | Skim |
| PII in eval datasets and in recorded test fixtures | Core | Useful |

**Plain words.** *Pseudonymization* = replace the name with a stable token you can reverse with a key you hold. Still personal data under GDPR — it lowers risk, it does not exit the regulation. *Anonymization* = irreversible; only that leaves the regulation.

**Read**
- **Start here.** [Microsoft Presidio](https://microsoft.github.io/presidio/) — analyzer and anonymizer docs. Runs locally, which is the point for on-prem.
- [OWASP Top 10 for LLM Applications](https://genai.owasp.org/llm-top-10/) — LLM02 (sensitive information disclosure) and LLM06. The current vocabulary for these risks.
- Carlini et al., *Extracting Training Data from Large Language Models* — [arXiv:2012.07805](https://arxiv.org/abs/2012.07805). **Abstract and §1 only.** The reason "we only fine-tuned on it" is not a privacy answer.

- **Build** (~120 lines, 2 h): run Presidio over 200 documents of your corpus, count entities by type, and put redaction on three paths — before text leaves for the provider, before it lands in a trace, and before it goes into an eval fixture. Then hand-label 30 documents yourself and report the false-negative rate. That number, not the library name, is the answer to "does your redaction work".
- **Done when:** you say which of redaction, pseudonymization and tokenization you would use for a support-ticket RAG and why, and what still leaks after you apply it.

## 3.3 Data residency, GDPR and the EU AI Act — 3 h · 2 h

| Sub-topic | AAI | INF |
|---|---|---|
| Controller vs processor; you are usually the processor, the customer is the controller | Core | Useful |
| Lawful basis and purpose limitation, in one paragraph each | Useful | Skim |
| DPAs and subprocessors — calling a US API makes that provider a subprocessor | Core | Useful |
| International transfers (Art. 44–49) and why "EU region only" appears in tenders | Core | Core |
| **Right to erasure vs a vector index, a cache, and fine-tuned weights** | Core | Useful |
| Retention limits and what they mean for traces and logs | Core | Core |
| Security of processing (Art. 32) as a concrete checklist | Useful | Core |
| EU AI Act risk tiers, awareness level only | Useful | Skim |

**Read**
- **Start here.** [gdpr-info.eu](https://gdpr-info.eu/) — read Articles **5, 17, 28, 32** and skim **44–49**. Article-by-article and short; this is a two-hour read, not a law degree.
- [artificialintelligenceact.eu](https://artificialintelligenceact.eu/) — the risk-tier overview page only. Know which tier a customer-facing RAG assistant lands in and why.
- [EDPB guidance index](https://www.edpb.europa.eu/our-work-tools/general-guidance/guidelines-recommendations-best-practices_en) — skim titles, read one that touches AI. Enough to know where the authoritative answers live.

- **Build** (2 h, one page): a **data map** for your own RAG service. Every place text lands — vector index, document store, prompt cache, application logs, trace backend, the provider — with, for each: region, retention, who can read it, and what happens on an erasure request. Then write the erasure procedure as steps, and be honest about the one that is hard.
- **Done when:** a customer asks "can you delete everything about this person", and you answer with the index, the cache, the logs, the traces and the fine-tune, saying which are minutes, which are the next rebuild, and which you must design around from the start.

## 3.4 On-premise and air-gapped deployment — 2 h · 2 h

| Sub-topic | AAI | INF |
|---|---|---|
| What breaks with no internet: weight downloads, package installs, telemetry, license checks | Core | Core |
| Mirroring: private registry, model cache, offline flags | Useful | Core |
| Open-weights vs API models, and how that decision follows from the deployment | Core | Core |
| Sizing for the customer's hardware, not yours | Useful | Core |
| Upgrades and patching without a network path | Skim | Core |
| **Support without seeing customer data** — debugging blind, redacted bug reports | Core | Core |

**Read**
- **Start here.** vLLM — [offline inference](https://docs.vllm.ai/en/latest/getting_started/quickstart.html#offline-batched-inference) plus Hugging Face [offline mode](https://huggingface.co/docs/transformers/en/installation#offline-mode) (`HF_HUB_OFFLINE`, pre-populated cache).
- Repo cross-reference: `lessons/` 13 (default-deny NetworkPolicy, quotas, RBAC), 14 (GitOps), 18 (cost and capacity). An air-gapped install is those three lessons with the internet removed.
- [Langfuse self-hosting](https://langfuse.com/self-hosting) and [Qdrant](https://qdrant.tech/documentation/guides/installation/) — proof that the observability and retrieval halves can run entirely inside a customer's network. Used again in §4.

- **Build** (~2 h): run your whole §10 stack with the network disabled — local embedding model, local LLM from the `lessons/` CPU vLLM stack, local vector store, local trace backend. List every point where it broke and what you replaced. That list is the answer to "can this run on-prem", and it is worth more than any slide.
- **Done when:** given "our data cannot leave our building", you sketch the deployment in five minutes, name what you lose (frontier model quality, hosted evals, managed scaling) and what you would offer instead.

## 3.5 Audit logging and tenant isolation — 2 h · 2 h

| Sub-topic | AAI | INF |
|---|---|---|
| What an audit record must contain: actor, tenant, time, action, resource, outcome | Core | Core |
| Append-only and tamper-evident; who is allowed to read the audit log | Useful | Core |
| Separating audit logs from debug logs, with different retention | Core | Core |
| Tenant isolation in a vector store: collection-per-tenant vs metadata filter | Core | Core |
| **The cross-tenant leak test** — the one test that must never be missing | Core | Core |
| Access reviews and least privilege | Skim | Useful |

**Read**
- **Start here.** OWASP — [Logging Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Logging_Cheat_Sheet.html). What to log, what never to log.
- Google — [Cloud Audit Logs types](https://cloud.google.com/logging/docs/audit) — admin vs data-access vs system events. Steal the taxonomy, ignore the product.
- Cross-reference: permission-aware retrieval in `llm_internals` §10; `lessons/` 13 for namespace and RBAC isolation.

- **Build** (~80 lines, 1.5 h): add an audit log to your service — actor, tenant, query hash, retrieved document ids, model, tokens, cost, timestamp, outcome — written separately from your debug logs. Answer one query from it: "what did tenant X see last Tuesday". Then write the leak test: a query from tenant A must never return a document id belonging to tenant B, asserted in pytest, run on every commit.
- **Done when:** you explain the trade between collection-per-tenant and a metadata filter — isolation strength against index count and memory — and you name the failure mode of the filter approach and the test that catches it.

---

# 4. Observability and tracing

Where "how do you debug a production agent" gets its answer. `llm_internals` §11's file-based trace log was the toy version; this is the tool version.

## 4.1 OpenTelemetry for LLM applications — 2 h · 2 h

| Sub-topic | AAI | INF |
|---|---|---|
| Traces, spans, attributes, context propagation across async boundaries | Core | Core |
| GenAI semantic conventions: model, tokens, cost, tool calls as standard attributes | Core | Core |
| One trace spanning HTTP → retrieval → rerank → generation → tool call | Core | Core |
| Sampling, and always sampling errors and slow requests | Useful | Core |
| Metrics and logs beside traces; when each answers the question faster | Useful | Core |
| Vendor-neutral instrumentation as an on-prem requirement | Core | Core |

**Read**
- **Start here.** OpenTelemetry — [Python instrumentation](https://opentelemetry.io/docs/languages/python/instrumentation/) and [concepts: traces](https://opentelemetry.io/docs/concepts/signals/traces/).
- [Semantic conventions for generative AI](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — the attribute names everyone is converging on. Use them instead of inventing your own.
- Repo cross-reference: `lessons/` 07 and 16 (Prometheus, Grafana, SLOs, burn-rate alerts) — metrics are already covered there; do not redo them.

- **Build** (~80 lines, 1.5 h): instrument the §1.1 service end to end. One trace per request with child spans for retrieve, rerank, generate and each tool call, carrying the GenAI attributes including token counts and cost. Verify the context survives your `TaskGroup` fan-out from §1.3 — if the spans come out orphaned, that is the lesson.
- **Done when:** looking at one trace you say where a 4-second p95 was spent, and you explain why a metric would not have told you that.

## 4.2 Langfuse and LangSmith — 2 h · 1 h

| Sub-topic | AAI | INF |
|---|---|---|
| Sessions, traces, generations, scores — the data model both share | Core | Useful |
| Attaching eval scores and user feedback to the trace that produced them | Core | Useful |
| Cost and latency dashboards per feature, per model, per tenant | Core | Core |
| Prompt versioning tied to trace history | Useful | Skim |
| Masking PII before it reaches the backend | Core | Core |
| **Self-hosted vs SaaS** — the decision that follows from §3.3 and §3.4 | Core | Core |

**Read**
- **Start here.** [Langfuse docs](https://langfuse.com/docs) — tracing data model, scores, and [self-hosting](https://langfuse.com/self-hosting). Open source and self-hostable, which is what makes it usable for an EU or on-prem customer.
- [LangSmith docs](https://docs.smith.langchain.com/) — tracing and evaluation sections. Know it because interviewers name it; know why you might not be allowed to use it.
- [OpenLLMetry](https://github.com/traceloop/openllmetry) — skim the README to see the OTel-native alternative and how the two worlds join up.

- **Build** (~60 lines, 1.5 h): run Langfuse locally in Docker, send your agent's traces to it with PII masked per §3.2, and attach the scores from your `llm_internals` §12 eval harness to the same traces. Build one view showing cost per run and one showing p95 latency by step.
- **Done when:** you argue self-hosted versus SaaS tracing for a healthcare customer in two minutes, using data residency, subprocessors and PII — and you say what you give up either way.

## 4.3 Debugging a production agent — 2 h · 1 h

| Sub-topic | AAI | INF |
|---|---|---|
| From a complaint to a trace: what identifier makes that possible | Core | Core |
| Isolating the failing step: retrieval, tool, or generation | Core | Core |
| Replaying one step with the exact recorded inputs | Core | Useful |
| Per-step token and cost attribution; finding the step that burns the budget | Core | Core |
| Detecting a silent quality drop by scoring a sample of live traffic | Core | Useful |
| Feeding a real failure back into the eval set so it cannot return | Core | Core |

**Read**
- **Start here.** Your own `llm_internals` §11 and §12 artifacts. This section is a procedure over systems you already have, not new reading.
- Anthropic — [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents), re-read only the section on measuring and iterating.
- Cross-reference: `system-design/` L16 (observability, SLOs, silent failure) for the general version of the same skill.

- **Build** (~1.5 h): break your agent on purpose in three ways — a tool that returns wrong-but-plausible data, a retrieval that returns the right document ranked eleventh, and a prompt change that quietly costs 3× more tokens. For each, find it **only** through the trace UI, write down the clicks, then add the case to your eval set. Three short incident notes at the end.
- **Done when:** given "a customer got a wrong answer yesterday at 14:32", you name the exact steps — find the trace by request id, read the retrieval span, check the tool result, compare against the golden case, add a regression case — and you say what you would have needed to log in advance to make that possible.

---

## What is deliberately not here

| Topic | Where it lives |
|---|---|
| Prompt injection, guardrails, permission-aware retrieval | [llm_internals](../llm_internals/ROADMAP.md) §10, §11, §13 |
| Eval design, judges, regression suites in CI | [llm_internals](../llm_internals/ROADMAP.md) §12 |
| Kubernetes, Helm, Prometheus, ingress, NetworkPolicy, GitOps | [`lessons/`](../../lessons/) 01–20 and [`labs/`](../../labs/) |
| Load balancing, queues, sharding, consistency, SLOs | [`system-design/`](../../system-design/) Phase 1–2 |
| Batching, KV cache, quantization — the serving layer under your API | [llm_internals](../llm_internals/ROADMAP.md) §9 |
| Front-end work beyond consuming an SSE stream | Nowhere, on purpose. The JD says "interfaces"; a working stream consumer is enough |
| Airflow, dbt, Spark, warehouse modeling | Nowhere. Real data engineering is a different job; §2 is the slice an Applied AI engineer owns |
