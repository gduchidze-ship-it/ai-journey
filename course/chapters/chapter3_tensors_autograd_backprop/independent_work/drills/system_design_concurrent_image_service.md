# System design drill — A concurrent image processing service (45 min)

**Format.** Ten minutes recall, thirty minutes designing aloud, five minutes writing what you missed. Timed. Speak out loud even if alone — the point is to practise producing a design under time pressure with an audience, and silence hides the gaps.

**Deliverable.** `image_service_design.md` in your repo, containing your design as you spoke it (a photo of the whiteboard is fine) and, separately headed, the *delta*: what you missed, in your own words, after reading the references. Do **not** read the references before the drill.

**Why this case, this week.** It is not an ML system, and that is deliberate: the three properties in its title — bounded concurrency, queueing, partial failure — are the properties every inference service in weeks 23–27 will need, and this week's lecture gave you the one fact about the model step that makes "bounded" non-negotiable: memory is linear in the batch, and the batch is whatever concurrency lets through. A service with unbounded concurrency in front of a model is a service whose peak memory is set by its clients.

---

## The prompt

> You own a service that accepts image uploads (JPEG/PNG, 50 KB–20 MB, p50 2 MB) and returns, for each image, a set of derived outputs: three resized variants, a perceptual hash, and a label from a small classifier model that runs on the same host (≈ 40 ms per image on one core in batch 1, faster per image in larger batches, ≈ 600 MB of memory for the weights plus activation memory that grows with batch). Traffic is bursty: a steady 50 images/s with spikes to 500 images/s lasting one to two minutes a few times a day. Clients wait synchronously up to 10 s for a response; after that they time out and retry. You have a fleet of 8 hosts, each with 8 cores and 16 GB of RAM. Design the service so that it stays up and useful through the spikes, and so that one bad image or one slow dependency cannot take it down.

Ask yourself the clarifying questions an interviewer would expect; when you cannot ask, state the assumption and continue.

## Minute 0–10: recall

Before designing, write from memory (no notes) the answers to these. They are the lecture material this drill exercises.

1. Peak inference memory for the classifier at batch `b` is `M₀ + m·b`. What is in `M₀`, what is in `m`, and which of the two does `torch.inference_mode()` shrink relative to a training forward pass? (Part 4 §4.5; Part 7 §7.2, §7.5.)
2. Why does per-image latency *fall* as the batch grows, up to a point? (Part 8 §8.3, the paragraph on the trade paying for itself — arithmetic intensity; and Part 7 §7.1 for why the work is matmuls.)
3. Decoding a 20 MB JPEG into a `(H, W, 3)` uint8 tensor: how many bytes is a 6000 × 4000 image after decode? In fp32 after normalization? What does that say about where memory goes in this service? (Part 3 §3.2 — bytes per element.)
4. Little's law: `L = λW`. If 500 requests/s arrive and each spends 4 s in the system, how many are resident? What is that number in bytes if each holds one decoded image?
5. A Python service does CPU-bound image work. What does the GIL do to a thread pool here, and what is the standard library's answer? (Python docs, `concurrent.futures`.)

## Minute 10–40: design aloud

Work through these layers. The order is deliberate: the *memory budget per host* first, because it fixes the concurrency bound, and the concurrency bound fixes everything after it.

**1. The memory budget, hence the concurrency bound.** Per host: 16 GB. Subtract the OS, the runtime, and the classifier's weights. What is left for in-flight images? Cost one in-flight request: the encoded upload, the decoded pixels, the three resized outputs, the model's activation slice. Divide. That integer is your **maximum in-flight requests per host**, and it is a hard cap, not a tuning knob — above it the host OOMs and takes *every* in-flight request with it. Write the number down. Then split it: how many are allowed to be *decoding* concurrently (CPU-bound, one core each) versus *waiting for the model* (memory-bound, no core)?

**2. Bounded concurrency — the mechanism.** How do you enforce the cap? Name the primitive per stage (a semaphore with `N` permits in front of decode; a bounded pool of worker processes for CPU work; a single owner of the model). Why processes rather than threads for the decode/resize stage? Why exactly one process (or one per accelerator) for the model, and what does that process do with the requests that arrive while it is busy?

**3. Queueing — the right depth and what "full" means.** Between the acceptor and the workers there is a queue. What is its **maximum length**, and how did you derive it — from the memory budget, from the client's 10 s timeout and your service time via Little's law, or both? What happens when it is full: block the acceptor, or reject immediately with a 429/503 and a `Retry-After`? Argue for one. Then the subtle one: a request that has been in the queue for 9.5 s of the client's 10 s budget — should a worker still pick it up? What does processing it cost, and who benefits?

**4. Batching the model.** The model is faster per image in batches. Design the batcher: collect requests for up to `T_max` ms or until `B_max` images, whichever first. How do `T_max` and `B_max` trade latency for throughput? What sets `B_max` — the memory budget from layer 1 again. What happens to a batch when one image in it is malformed and makes the model raise?

**5. Partial failure — the bad image.** One upload is a 20 MB, 30 000 × 30 000 "decompression bomb" PNG; another is a truncated JPEG; a third is valid but the classifier returns NaN. For each: where is it detected, what is returned to *that* client, and — the design question — what guarantees it cannot affect the other 199 requests in flight on the same host? (Hint: which of your bounds is a per-request bound and which is a per-host bound? Is a per-request *memory* bound even possible in your runtime?)

**6. Partial failure — the slow dependency.** The perceptual hashes are stored in a remote index whose p99 latency has just gone from 5 ms to 4 s. What happens to your queue? To your memory? To the clients? Design the timeout, the fallback (return the labels and resizes without the hash? mark the response partial?), and the circuit breaker. Then: the *clients* are retrying after 10 s. Draw the load on your service during a two-minute dependency slowdown *with* and *without* their retries. Who owns the retry policy and what should it be?

**7. Overload — the spike.** 500 images/s for 90 s against 8 hosts is 62 images/s/host, versus a steady state of ~6. Your queue from layer 3 fills in how many seconds? Which requests do you shed, and in what order — newest, oldest, largest, lowest-priority client? What does "goodput" mean here as opposed to throughput, and how do you measure it? What do you *not* do (grow the pool, grow the queue)?

**8. Observability.** Name the five numbers you would graph to know whether layers 1–7 are working: in-flight count against the cap, queue depth against its max, queue wait time, rejections per second by reason, and per-host peak memory against the budget. Which of them is the *leading* indicator of an OOM, and how far ahead?

Stop at minute 40 regardless of where you are.

## Minute 40–45: the delta

Now read, in this order: Google SRE book, ch. 22 "Addressing Cascading Failures" — the sections *Queue Management*, *Load Shedding and Graceful Degradation*, and *Retries* (~8 min); ch. 21 "Handling Overload" — the *Client-Side Throttling* and *Criticality* sections (~5 min); Marc Brooker, "Exponential Backoff And Jitter" (~5 min); the Python docs for `concurrent.futures` (the `ProcessPoolExecutor` paragraph) and `asyncio.Queue` (the `maxsize` paragraph). Links in `../../READING.md` under "System design drill." Then write under a heading **Delta** what you missed or got wrong. Prompts:

- Did you derive the queue length from anything? The SRE book: "For a system with fairly steady traffic over time, it is usually better to have small queue lengths relative to the thread pool size (e.g., 50% or less)." Was your queue ten times your worker count "to absorb the spike"? What does a long queue do to the 10 s client budget?
- Did you distinguish load shedding from degradation? SRE: "Load shedding drops some proportion of load by dropping traffic as the server approaches overload conditions," while "Graceful degradation takes the concept of load shedding one step further by reducing the amount of work that needs to be performed" — e.g. skip the classifier during the spike and return resizes only. Did you have a *cheaper* response, or only a *rejected* one? Chapter 21 calls these "degraded responses: responses that are not as accurate as or that contain less data than normal responses, but that are easier to compute."
- Did you model the clients' retries as load? SRE: "The volume of retries grows: 100 QPS of retries in the first second leads to 200 QPS, then to 300 QPS, and so on." If your design only bounded *your* concurrency and left the client retry policy as "their problem," you designed half a system.
- Did you specify backoff *with jitter*? Brooker on plain exponential backoff: "there are still clusters of calls. Instead of reducing the number of clients competing in every round, we've just introduced times when no client is competing." His formula: `sleep = rand(0, min(cap, base * 2 ** attempt))`. His conclusion: "jittered backoff is huge, and it should be considered a standard approach for remote clients."
- Did you give clients a way to self-regulate? Chapter 21: "When a client detects that a significant portion of its recent requests have been rejected due to 'out of quota' errors, it starts self-regulating and caps the amount of outgoing traffic it generates." Did your 429 carry a `Retry-After`?
- Did you say *what to shed first*? Chapter 21 attaches a criticality to every request — "A request made to a backend is associated with one of four possible criticality values" — so that shedding is ordered rather than random. Did you shed the 20 MB uploads before the 50 KB ones? The requests already past 8 s of their 10 s budget?
- Did your concurrency bound come from *memory*, or from "number of cores"? The lecture's whole point: cores bound compute; the batch bounds memory; and it is memory that fails catastrophically. Chapter 21 warns that "modeling capacity as 'queries per second' … often makes for a poor metric."
- Did you use processes for the CPU work? The Python docs: `ProcessPoolExecutor` "allows it to side-step the Global Interpreter Lock but also means that only picklable objects can be executed and returned." What did that constraint do to your design — did you pass a 100 MB decoded image between processes by pickling it?
- Did you make the queue *bounded in code*, not just in a diagram? `asyncio.Queue(maxsize)`: "If *maxsize* is less than or equal to zero, the queue size is infinite." The default is infinite. `put_nowait` on a full queue raises `QueueFull` — that exception *is* your load shedder.
- Did you plan to test it? SRE ch. 22: "Load test components until they break." What is the load test that proves the memory cap holds at 500 images/s of 20 MB uploads?

## Self-grading rubric

| | Pass | Strong |
|---|---|---|
| Recall (min 0–10) | 4 of 5 from memory | All 5, with the decoded-image byte count and Little's-law number computed |
| Memory budget → cap | An in-flight cap per host, derived from bytes | Split into decode-stage and model-stage bounds, with `B_max` for the batcher derived from the same budget |
| Mechanism | Semaphore / bounded pool / single model owner named | Processes for CPU work with the pickling cost addressed (shared memory, or decode inside the worker) |
| Queue | Bounded, with a stated max and a stated full-behaviour | Max derived from the client timeout via Little's law; stale requests dropped before processing |
| Partial failure | Bad image cannot affect others; dependency has a timeout | Decode in an isolated process with a size pre-check; hash dependency has fallback + circuit breaker; partial responses are explicit |
| Overload | Shedding, not growing | Ordered shedding by criticality/size/age; a degraded mode; retries with jitter and `Retry-After` specified for clients |
| Observability | Five numbers named | The leading OOM indicator identified with a lead time |
| Delta | Written, ≥ 4 concrete misses | Each miss paired with the design change you would make |
