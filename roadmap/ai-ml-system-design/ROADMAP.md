# AI / ML System Design — Roadmap
**Part of:** [Master Syllabus](../SYLLABUS.md) — the order all five roadmaps run in.

**Time:** ~45 h (AAI) · ~35 h (INF).
**Before:** [ml_core](../ml_core/ROADMAP.md) (including Lane B §10–§11) and [llm_internals](../llm_internals/ROADMAP.md) Part B. You cannot design a RAG system you have never built.
**Your machine:** irrelevant here. This round is paper, whiteboard and talking. The output is a drawing and an argument, not code.

> **Verdict: two different rounds hide under one name, and you must know which one you are in.**
>
> **Lane A — big-tech ML design (Google, Meta, Amazon).** "Design the news feed ranker." "Design ads relevance." The domain is recommenders, ranking, search and fraud. The hard parts are data, features, labels, offline-vs-online metrics and retraining. LLMs barely appear.
>
> **Lane B — frontier-lab AI-infra design (Mistral, Anthropic, OpenAI).** "Design an inference batching service at 100k RPS." "Design a RAG system over 100M documents." "Design an eval pipeline." The hard parts are latency, memory, batching, queues and cost.
>
> **Lovable** does neither at depth — they give you a small real feature to build live. §1 still helps; §6 and §7 do not.
>
> **What already exists in this repo.** The [`system-design/`](../../system-design/) pillar covers the generic distributed-systems half — load balancing, caching, sharding, queues, consistency. That half is a real prerequisite and it is **not** repeated here. Be honest about its state: lessons are written through 0020, and its Phase 4–6 (ML patterns, LLM inference, RAG/agents/eval design) is a plan, not written material. This file is the ML/AI-specific layer that sits on top, and the reps in §6–§7 are the part you have to actually do.

**Legend:** AAI = Applied AI Engineer · INF = AI Infrastructure Engineer. **Core** = must be fluent · **Useful** = know it · **Skim** = one paragraph · **Skip** = ignore.

## The path

Do §1–§5 once. Then pick your lane and grind reps. One rep = one timed 50-minute design, spoken out loud.

| Step | Topic | Hours | Lane |
|---|---|---|---|
| 1 | The protocol — how to drive the room | 3 | both |
| 2 | The ML framing — problem to metric | 5 | both |
| 3 | Data, labels and features | 6 | mostly A |
| 4 | Serving and latency | 5 | both |
| 5 | Monitoring, drift and retraining | 4 | both |
| 6 | **Lane A reps** — feed, search, ads, fraud | 12 | A |
| 7 | **Lane B reps** — inference, RAG, evals, agents | 12 | B |
| 8 | Mocks and the rep log | 4 | both |

Do **both** §6 and §7 if your list mixes Meta with Mistral. That is 57 h, not 45 h. Say so to yourself now rather than being surprised in week 6.

---

## 1. The protocol — how to drive the room

You are graded on whether you can run a 50-minute conversation without being led. Most people fail here, not on knowledge.

| Sub-topic | AAI | INF |
|---|---|---|
| Clarify: users, scale, latency budget, what "good" means | Core | Core |
| Estimate before designing: QPS, storage, memory | Core | Core |
| Draw the boxes, then go deep on **one** the interviewer picks | Core | Core |
| State trade-offs out loud; name what you gave up | Core | Core |
| Time management: never spend 20 minutes on requirements | Core | Core |

**Plain words.** *Drive* = you decide what to talk about next, not the interviewer. Silence is your problem to fill.

- **Start here:** [System Design Primer](https://github.com/donnemartin/system-design-primer) — the "How to approach a system design interview question" section only. Four steps. Learn them as a script.
- Exponent — [Meta Machine Learning Engineer interview guide](https://www.tryexponent.com/guides/meta-machine-learning-engineer-interview). Read what the ML design round is actually scored on.

- **Build** (1 h): write the four steps on one index card, plus your own list of 8 clarifying questions that work for any ML prompt. Keep it next to you for every rep in §6–§7.
- **Done when:** you can open any prompt with 5 minutes of clarifying questions and a rough number, without touching the whiteboard.

## 2. The ML framing — problem to metric

The step everyone skips. A product ask is not an ML problem until you turn it into one.

| Sub-topic | AAI | INF |
|---|---|---|
| Business goal → ML task (classification, ranking, retrieval, generation) | Core | Useful |
| Is ML even the right answer? When rules win | Core | Useful |
| Choosing the label; when the label does not exist yet | Core | Skim |
| Offline metric vs online metric, and the gap between them | Core | Useful |
| Defining "good enough" as a number before designing | Core | Core |

**Plain words.** *Label* = the answer you train on. Half of real ML design is arguing about where labels come from. *Offline metric* = measured on stored data. *Online metric* = measured on live users.

- **Start here:** Google — [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml). Rules 1–15 cover this section exactly. It is the best free document on ML system judgement and it is short.
- Chip Huyen — [Machine Learning Systems Design](https://huyenchip.com/machine-learning-systems-design/toc.html) (free booklet), the framing and case-study sections. Repo: [chiphuyen/machine-learning-systems-design](https://github.com/chiphuyen/machine-learning-systems-design).

- **Build** (1.5 h): take three prompts — "reduce spam", "recommend videos", "detect fraud" — and for each write in five lines: the ML task, the label and where it comes from, the offline metric, the online metric, and the number that means success.
- **Done when:** given any product ask you state the ML task, the label source and both metrics in under three minutes, and you can name one case where you would ship rules instead of a model.

## 3. Data, labels and features

| Sub-topic | AAI | INF |
|---|---|---|
| Where training data comes from; logging as a design decision | Core | Useful |
| Label delay and feedback loops (the model changes its own training data) | Core | Skim |
| Feature engineering; feature stores | Core | Useful |
| **Training/serving skew** — the classic interview trap | Core | Core |
| Data leakage in a system, not just in a dataset | Core | Useful |
| Class imbalance and sampling at scale | Useful | Skim |

**Plain words.** *Training/serving skew* = the features you compute during training are not identical to the ones you compute live, so the model quietly gets worse. *Feature store* = one place that computes a feature once and serves it to both training and production.

- **Start here:** Google — [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml), **Rules 29–37**. This is the training/serving skew material, written by the people it happened to.
- Sculley et al., *Hidden Technical Debt in Machine Learning Systems* — [NeurIPS 2015](https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html). Read §3 and §5. Feedback loops and entanglement; both are standard senior follow-ups.

- **Build** (2 h): for the feed-ranking prompt, write the data section alone — what you log, when the label arrives, three features and how each is computed at train time and at serve time. One page.
- **Done when:** you can name three ways training/serving skew appears and the fix for each; and you can explain how a ranking model poisons its own future training data.

## 4. Serving and latency

| Sub-topic | AAI | INF |
|---|---|---|
| Batch vs online vs streaming prediction; how to choose | Core | Core |
| The latency budget: split a p95 target across stages | Core | Core |
| Multi-stage serving (cheap filter → expensive model) | Core | Core |
| Caching predictions and caching features | Useful | Core |
| Model-in-image vs model-loaded-at-boot | Skim | Core |
| Fallbacks and graceful degradation when the model is down | Core | Core |

**Plain words.** *Multi-stage serving* = the same funnel idea as recommenders: something cheap narrows the field, something expensive decides. It shows up in ranking, in RAG (retrieve then rerank) and in moderation.

- **Start here:** Mercari — [ML System Design Patterns](https://mercari.github.io/ml-system-design-pattern/). Read the **Serving patterns** section and the **Anti-patterns** section. Named patterns give you vocabulary the interviewer recognizes.
- Eugene Yan — [Real-time Machine Learning: Challenges and Solutions](https://eugeneyan.com/writing/real-time-recommendations/). What "real time" actually costs.
- Repo cross-reference: [`lessons/`](../../lessons/) 03–10 for the serving stack you already built, and `system-design/` L15–L16 for reliability and observability.

- **Build** (1.5 h): take one 200 ms p95 target and split it across stages for a ranking system and for a RAG system. Two diagrams, one page of numbers. Mark which stage you cut first when you miss.
- **Done when:** given a latency target you allocate it across stages with numbers, and you say what degrades gracefully when the model server is down.

## 5. Monitoring, drift and retraining

| Sub-topic | AAI | INF |
|---|---|---|
| What to log: inputs, predictions, outcomes | Core | Core |
| Data drift vs concept drift; how each is detected | Core | Useful |
| Silent failure — the model is up and wrong | Core | Core |
| Retraining triggers: schedule vs drift vs performance | Core | Useful |
| Shadow deploys, canaries, rollback | Core | Core |

**Plain words.** *Data drift* = the inputs changed. *Concept drift* = the inputs look the same but the right answer changed. *Silent failure* = no alarm goes off because nothing crashed.

- **Start here:** Mercari — [ML System Design Patterns](https://mercari.github.io/ml-system-design-pattern/), the **Operation patterns** and **QA patterns** sections (prediction logs, monitoring, shadow A/B, load test).
- Evidently AI — [Data drift](https://www.evidentlyai.com/ml-in-production/data-drift). Practical detection methods, plainly written.
- Repo cross-reference: `system-design/` L16 (observability, SLOs, burn-rate alerts) — do not re-derive it here.

- **Build** (1.5 h): a one-page monitoring plan for one system from §6 or §7 — five metrics, three alerts with thresholds, one rollback trigger, and how you would notice a silent quality drop within 24 hours.
- **Done when:** you explain the difference between data drift and concept drift with one example each, and you name how you detect a model that is up, fast, and wrong.

---

# Lane A — big-tech reps (Google, Meta, Amazon)

## 6. Reps: ranking, search, ads, fraud

**Time:** ~12 h · four reps of ~3 h. Do them in this order; each reuses the last.

| Rep | Prompt | Why this one |
|---|---|---|
| A1 | **Design a news feed ranker** | The canonical Meta prompt. Funnel, features, labels, position bias |
| A2 | **Design search or ads relevance** | Query understanding, retrieval + ranking, CTR prediction, auctions |
| A3 | **Design a recommendation system (50M users)** | Cold start, popularity bias, freshness, candidate generation at scale |
| A4 | **Design fraud detection with a 100 ms budget** | Imbalance, rules-plus-model hybrid, hard latency, cost of each error |

**Plain words.** *Position bias* = users click the top item because it is at the top, not because it is best. Your logs are therefore lying to you, and you have to correct for it.

- **Start here:** [alirezadir/AIMLInterviews](https://github.com/alirezadir/AIMLInterviews) — the ML system design section and its case studies. Updated for 2026 and organized as interview reps.
- Aminian & Xu, *Machine Learning System Design Interview* (ByteByteGo, 2023). The standard book for this exact round. Work the chapters that match A1–A4.
- Model background you already have: [ml_core](../ml_core/ROADMAP.md) §10 covers the two-tower funnel, GBDT-vs-deep ranking, *Wide & Deep* and *DLRM*. Do not re-read them here; use them.

- **Rep protocol** (per rep, ~3 h): 10 min recall → **40 min solo timed design, spoken aloud, no notes** → 40 min compare against a reference solution → write the deltas into `learning-records/`. Redraw the whole thing from memory a week later.
- **Done when:** you complete A1 cold in 45 minutes, out loud, covering funnel, labels, features, offline and online metrics, latency budget and monitoring — and your week-later redraw is missing fewer than three boxes.

---

# Lane B — frontier-lab reps (Mistral, Anthropic, OpenAI)

## 7. Reps: inference, RAG, evals, agents

**Time:** ~12 h · four reps of ~3 h.

| Rep | Prompt | Why this one |
|---|---|---|
| B1 | **Inference batching service, 100k RPS** | The most-reported frontier-lab prompt, in several variants |
| B2 | **RAG over 100M documents** | Index choice, sharding, freshness, permissions, reranking cost |
| B3 | **Eval pipeline as a batch system** | Orchestration, judge fanout, result storage, cost per run |
| B4 | **Agent platform: tools, sandbox, durability** | Long-running state, retries, cost caps, isolation |

**Plain words.** *Batching service* = requests arrive one at a time from many users, you group them so the GPU does useful work, then you must return each answer to the right caller.

- **Start here:** your own builds. B2, B3 and B4 are design versions of what you already shipped in [llm_internals](../llm_internals/ROADMAP.md) §10, §12 and §11. Read your own repo before reading anything else.
- Kwon et al., *PagedAttention / vLLM* — [arXiv:2309.06180](https://arxiv.org/abs/2309.06180), §3–4, for B1. Anthropic — [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) for B4.
- Repo cross-reference: `system-design/` Phase 5 (L34–L40) is the written plan for B1; [`lessons/`](../../lessons/) 06, 09, 15–17 is the stack you built. Note again that Phase 5 lessons are **not written yet** — you are the one who has to do these reps.

- **Rep protocol:** same as §6. Timed, aloud, solo first.
- **Done when:** you drive B1 end to end in 50 minutes — clarify, estimate GPU memory and KV cache, draw gateway → queue → scheduler → workers, deep-dive the batching policy, and name three failure modes with mitigations.

---

## 8. Mocks and the rep log

| Sub-topic | AAI | INF |
|---|---|---|
| Speaking the design aloud, not drawing it silently | Core | Core |
| Handling "what if traffic is 100x" mid-answer | Core | Core |
| Saying "I don't know" well | Core | Core |
| Keeping a delta log across reps | Core | Core |

- **Start here:** no reading. Book **two human mock interviews** before any onsite. Every source surveyed puts this as the highest-return single step, and it is the one thing this repo cannot simulate for you.
- Keep a `learning-records/` entry per rep: prompt, what you missed, what you would say differently. Re-read the log before the next rep.

- **Build:** 8 reps logged (4 from your lane, minimum), and 2 human mocks completed.
- **Done when:** across three consecutive reps you stop making the same class of mistake, and your delta log gets boring.

---

## What is deliberately not here

| Topic | Where it lives |
|---|---|
| Load balancing, caching, sharding, queues, consistency, CAP | [`system-design/`](../../system-design/) Phase 1–2 — a real prerequisite, do not skip it |
| Classic non-ML designs (rate limiter, TinyURL, chat, payments) | `system-design/` Phase 3 |
| Kubernetes, vLLM, Helm, Prometheus, autoscaling — the built stack | [`lessons/`](../../lessons/) 01–20 |
| Recommender model internals (two-tower, GBDT, DLRM) | [ml_core](../ml_core/ROADMAP.md) §10 |
| RAG, agent and eval implementation | [llm_internals](../llm_internals/ROADMAP.md) §10–§12 |
| DSA coding rounds, Amazon Leadership Principles | Nowhere yet. Google, Meta and Amazon all test the first; Amazon weights the second heavily |
