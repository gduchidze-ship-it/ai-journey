# Master Syllabus — the whole curriculum in one order

Five roadmaps, three built pillars, one order to do them in. Each roadmap is the authority on **what** to learn inside its topic; this file is the authority on **when**, **why now**, and **what you own when it is finished**.

**Read the roadmap file, not this file, when you are studying.** Read this file when you finish a section and need to know what comes next.

| | |
|---|---|
| **Total, Applied AI lane** | ~255 h core · ~277 h if Google / Meta / Amazon are on your list |
| **Total, AI Infra lane** | ~237 h core · ~259 h with the same add-ons |
| **At 15 h / week** | ~19 weeks (AAI) · ~18 weeks (INF) |
| **Machine** | Apple M4, 16 GB, no NVIDIA GPU. Every build runs locally or on a paid API |

---

## 1. The pieces

### The roadmaps — `roadmap/`

| File | Topic | AAI h | INF h | Owns |
|---|---|---|---|---|
| [ml_core](ml_core/ROADMAP.md) | Classical ML judgement | 10 | 8 | Metrics, leakage, retrieval metrics, significance. Becomes eval design later |
| [dl_core](dl_core/ROADMAP.md) | Deep learning core | 26 | 38 | Autograd, losses, layers, embeddings, a real training loop, numerics and memory |
| [engineering_foundations](engineering_foundations/ROADMAP.md) | API, data, security, observability | 44 | 36 | The code you are paid to write: HTTP APIs, async, ingestion, SQL, PII, GDPR, tracing |
| [llm_internals](llm_internals/ROADMAP.md) | LLM internals + applied LLM systems | 130 | 120 | Tokenizers → transformers → fine-tuning (Part A); RAG, agents, evals, ops (Part B) |
| [ai-ml-system-design](ai-ml-system-design/ROADMAP.md) | The design round | 45 | 35 | Driving a 50-minute design conversation; lane reps |

Optional lanes, priced separately: **ml_core §10–§11** (+10 h) and a **second set of design reps** (+12 h) — both only if you target Google, Meta or Amazon.

### The built pillars — the rest of the repo

These are not reading. They are courses with labs and graders that already exist, and the roadmaps lean on them instead of repeating them.

| Pillar | Where | State | Used by |
|---|---|---|---|
| **AI Infrastructure** — 20 lessons, 20 graded labs: kind → vLLM → Helm → router → Prometheus/Grafana → HPA, then GPU scheduling, TLS edge, NetworkPolicy, GitOps, perf, SRE, cost | [`lessons/`](../lessons/), [`labs/`](../labs/), [`reference/`](../reference/) | Written, 01–20, EN + KA | `llm_internals` §9, §13 · `engineering_foundations` §1.4, §3.1, §3.4, §4.1 |
| **System Design** — performance vocabulary, estimation, distributed systems | [`system-design/`](../system-design/) | Lessons 0001–0020 + Phase 0 written; Phase 4–6 still a plan | `ai-ml-system-design` (prerequisite, not repeated there) |
| **Behavioral & Culture** — story bank, values reasoning, lab culture | [`behaviour/`](../behaviour/) | Story-mining lesson only; the rest is yours to build | Runs continuously, see §4 |

---

## 2. Pick your lane before week 1

| Target | Lane | What changes |
|---|---|---|
| **Mistral, Anthropic, OpenAI** — Applied AI | AAI | The default path below. `engineering_foundations` §3 is Core, not optional. Design reps = Lane B only |
| **Mistral, Anthropic, OpenAI** — infra / serving | INF | `dl_core` §7 grows from 9 h to 21 h and is the highest-value block in the file. Lighter on RAG and agents, heavier on `lessons/` |
| **Google, Meta, Amazon** | AAI + big-tech add-ons | Add `ml_core` §10–§11 and the Lane A design reps. Also add DSA, which lives nowhere in this repo |
| **Lovable and similar** | AAI, trimmed | Skip the design reps. `engineering_foundations` §1 and `llm_internals` Part B carry the whole loop |

If your list mixes Meta with Mistral, you do both rep sets. Say the real number to yourself now — 277 h, not 255 h.

---

## 3. Dependencies

```
ml_core §1–§9 ──────────────────────────────► llm_internals §12 (evals)
                                                    ▲
dl_core §0–§6 ──► dl_core §7 ──► llm_internals Part A §1–§8 ──► Part B §9–§13
                       │                                              ▲
                       └──────────────► (memory math is reused by §9) │
                                                                      │
engineering_foundations §1 ───────────────────────────────────────────┤
   (API, async, docker, tests — needed before you build a service)    │
engineering_foundations §2–§4 ────────── wraps the same builds ───────┘
                                                                      │
lessons/ 01–20  +  system-design/ 0001–0020 ──────────────────────────┤
   (the platform and the distributed-systems half)                    │
                                                                      ▼
                                                    ai-ml-system-design §1–§8
```

Two hard rules. **`dl_core` §7 before `llm_internals`** — transformers make sense once you can count bytes. **`engineering_foundations` §1 before `llm_internals` §10** — otherwise you build the RAG system twice, once as a script and once as a service.

---

## 4. Three things run continuously, from week 1

They are not a phase. Book them into every week or they never happen.

| Thread | Source | Cadence |
|---|---|---|
| **Timed coding reps** | [llm_internals §14](llm_internals/ROADMAP.md) | 2 reps/week, 35–45 min each, cold, from a blank file, with tests |
| **Story bank** | [`behaviour/`](../behaviour/) | 1 story/week written to the competency map. 12 weeks = a full bank |
| **The infra platform** | [`lessons/`](../lessons/) 01–20 + labs | INF: 4–6 h/week, it is your spine project. AAI: 2 h/week, finish 01–10 and stop |

---

## 5. The plan

Weeks assume **15 h/week** plus the continuous threads. Slide the numbers, keep the order.

### Phase 0 — Set up · week 1 · 4 h

Pick the lane. Read the header and the verdict block of all five roadmaps — not the bodies, the verdicts; they tell you what was cut and why. Create a `learning-records/` entry format you will actually use: date, section, what you built, what you got wrong. Start the two continuous threads.

**Gate:** you can say in one paragraph which lane you are in, which of the five files you will skip sections of, and why.

### Phase 1 — ML judgement · week 1–2 · 10 h (AAI) / 8 h (INF)

[ml_core](ml_core/ROADMAP.md) §1–§9. Skip §10–§11 unless big tech is on your list.

**Why first:** it is small, it needs no prerequisites, and §4 (retrieval metrics) and §6 (comparing two systems) are the machinery that `llm_internals` §12 evals is built from. Doing it later means doing evals badly.

**Gate:** someone says *"the new retriever scored 0.71 vs 0.68 on 200 queries"* and inside a minute you say whether it is real, what test you would run, and how many more queries you need. Deliverable: `compare.py`.

### Phase 2 — Deep learning core · weeks 2–4 · 26 h (AAI) / 38 h (INF)

[dl_core](dl_core/ROADMAP.md) §0–§6, then §7 by lane — AAI does §7.2 plus the two Core rows of §7.3; INF does all of §7.

**Why:** §7.2 (memory accounting) is the single most reused block in the whole curriculum. It reappears in `llm_internals` §9, in every Lane B design rep, and in half the infra interview questions.

**Gate:** on a whiteboard in under three minutes, KV-cache bytes for a 7B model at batch 32, 4k context, GQA with 8 KV heads, bf16 — within 10%. Deliverables: micrograd with an extra operator, a training loop with resume, `memory_math.py`.

### Phase 3 — The engineering layer · weeks 4–5 · 16 h (AAI) / 14 h (INF)

[engineering_foundations](engineering_foundations/ROADMAP.md) §1 — API design, streaming and SSE, async Python, containers, testing.

**Why here:** everything from Phase 5 onward is a service. Learn to build one before you have something to put in it. The async block in particular is the most common way an LLM back-end is slow for reasons unrelated to the model.

**Gate:** you write an async endpoint with timeouts, a semaphore, a shared client and a bounded queue without looking anything up, and your offline test suite for it runs in under 10 seconds with no API key.

### Phase 4 — Model internals · weeks 5–9 · 62 h (AAI) / 70 h (INF)

[llm_internals](llm_internals/ROADMAP.md) Part A, §1–§8. Raschka is the spine; type it, do not read it.

**Why:** this is the 20% of the loop that is asked directly and cannot be bluffed. It is also the part that makes Part B's failure modes legible instead of magical.

**Gate:** from an empty file you write causal multi-head attention from memory and it runs; and given a config you compute parameter count on paper to within 2%. Deliverables: BPE tokenizer + token-cost table, a working GPT, a sampler with constrained JSON, a LoRA fine-tune with before/after numbers.

### Phase 5 — Applied LLM systems · weeks 10–15 · 88 h (AAI) / 68 h (INF)

The core of the Applied AI loop, and the phase where two files interleave. Do them together, not one after the other:

| Order | Do | Then immediately |
|---|---|---|
| 1 | `llm_internals` §9 — inference systems | — |
| 2 | `llm_internals` §10 — RAG, ~2,000 chunks | `engineering_foundations` §1.1–§1.2: put an API and an SSE stream in front of it |
| 3 | `engineering_foundations` §2 — SQL, incremental ingestion, reindex and restore | the one-shot script becomes a pipeline |
| 4 | `llm_internals` §11 — agents, 3 tools, caps, break it on purpose | `engineering_foundations` §4 — OTel + Langfuse, so "break it" produces traces, not print statements |
| 5 | `llm_internals` §12 — evals, 40 cases, validated judge, CI gate | attach the scores to the traces from step 4 |
| 6 | `engineering_foundations` §3 — secrets, PII, GDPR, on-prem, audit | the security layer over the whole thing |
| 7 | `llm_internals` §13 — production operations, runbook | — |

**Why interleaved:** each `engineering_foundations` section upgrades the artifact the `llm_internals` section just produced. Done in sequence instead, you write the RAG service twice and the security layer never gets built at all.

**Gate:** the artifact runs end to end — ingest incrementally, answer over HTTP with a stream and citations, redact PII, write an audit record, emit a trace, and fail CI when the eval score drops. And you can debug a deliberately broken run from the trace UI alone.

### Phase 6 — The design round · weeks 16–18 · 45 h (AAI) / 35 h (INF)

[ai-ml-system-design](ai-ml-system-design/ROADMAP.md) §1–§5 once, then reps: §7 (Lane B) for frontier labs, §6 (Lane A) for big tech, both if your list mixes.

**Prerequisite check:** `system-design/` lessons 0001–0020 must be done. That pillar owns load balancing, caching, sharding, queues and consistency, and this file does not repeat them.

**Gate:** you drive B1 — inference batching service at 100k RPS — end to end in 50 minutes: clarify, estimate GPU memory and KV cache, draw gateway → queue → scheduler → workers, deep-dive the batching policy, name three failure modes with mitigations.

### Phase 7 — Endgame · week 19 · ongoing

Two human mock interviews, booked. The rep log re-read until it gets boring. The behavioral bank finished. One design doc from Phase 5 polished into a writing sample.

**Gate:** across three consecutive reps you stop making the same class of mistake.

---

## 6. What you own at the end

One system, plus the artifacts that prove you built it.

**The portfolio artifact.** A retrieval-and-agent service with: an HTTP API with cursor pagination and RFC 9457 errors, an SSE streaming endpoint that cancels properly, an incremental ingestion pipeline that costs nothing on a no-op run, a versioned index with an alias swap and a tested restore, PII redaction on three paths, an audit log with a cross-tenant leak test, OpenTelemetry traces in a self-hosted Langfuse, an eval harness that gates CI, and a runbook — deployed on the Kubernetes + vLLM stack from `lessons/`.

**The supporting artifacts:**

| From | Artifact |
|---|---|
| ml_core §6 | `compare.py` — bootstrap CI on two systems' scores |
| dl_core §1, §5 | micrograd with an added operator · a training loop with resume, an OOM-then-accumulation run, and a deliberate loss spike |
| dl_core §7.2 | `memory_math.py`, checked against Llama 3 8B within 15% |
| llm_internals §1–§4 | BPE tokenizer + multilingual token-cost table · a working GPT + `param_count(config)` · constrained-JSON failure counts |
| llm_internals §8 | LoRA fine-tune with before/after accuracy, adapter size, training time |
| llm_internals §10–§13 | RAG with 30 golden pairs and three recall@5 numbers · agent break-it report · eval harness with a validated judge · ops runbook |
| engineering_foundations §1–§4 | Offline test suite under 10 s · `ingest.py --since` · `reindex.py` with a zero-downtime swap · a one-page data map · three incident notes written from traces |
| ai-ml-system-design §6–§8 | 8 logged design reps, 2 human mocks |
| lessons/ + labs/ | The platform itself, 20 graded labs, capstone design doc |
| behaviour/ | A story bank mapped to competencies |

---

## 7. Hour accounting

| File | AAI core | INF core | Optional |
|---|---|---|---|
| ml_core | 10 | 8 | +10 (§10–§11, big tech only) |
| dl_core | 26 | 38 | — |
| engineering_foundations | 44 | 36 | — |
| llm_internals | 130 | 120 | — |
| ai-ml-system-design | 45 | 35 | +12 (second rep set) |
| **Total** | **255** | **237** | **+22** |

Not counted, because they are separate pillars with their own pace: `lessons/` 01–20 (~40 h with labs), `system-design/` 0001–0020, `behaviour/`.

---

## 8. Honest gaps

| Missing | Who needs it | Where it would go |
|---|---|---|
| **DSA / LeetCode reps** | Google, Meta, Amazon — all three test it | Nowhere in this repo. Budget 60–100 h separately if they are on your list |
| **Amazon Leadership Principles** | Amazon, heavily weighted | `behaviour/`, not written yet |
| **Front-end beyond consuming a stream** | The JD says "interfaces" | Deliberately out of scope. A working SSE consumer is the bar this curriculum sets |
| **Real data engineering** — Airflow, dbt, Spark, warehouse modeling | Data engineers | Out of scope; `engineering_foundations` §2 is the slice an Applied AI engineer owns |
| **Multi-node training, CUDA kernels, FSDP, NCCL** | INF, at a lab | Read-only throughout — no GPU. `dl_core` §7.5 marks exactly what you can and cannot run |
| **`system-design/` Phase 4–6 lessons** | Lane B design reps | Planned, not written. `ai-ml-system-design` §7 is the substitute: you do the reps yourself |

---

## 9. Rules that keep this honest

1. **Gates are not optional.** Fail one, stay in the section. The gates are written as things you *do*, not things you have read.
2. **A section without its Build is not done.** Recognition is not recall, and the follow-up questions in an interview go exactly where the building would have been.
3. **Log every section** in `learning-records/`: what you built, what you got wrong, what you would do differently. Re-read the log before the next rep.
4. **Max three sources per section.** Every roadmap enforces this on purpose. Adding a fourth source is procrastination with a good reputation.
5. **When two files overlap, the roadmap that owns the topic wins.** The cross-reference tables at the bottom of each file say who owns what.
