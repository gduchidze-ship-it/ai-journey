# Deep Learning Core — Roadmap
**Part of:** [Master Syllabus](../SYLLABUS.md) — the order all five roadmaps run in.

**Time:** ~**26 h** for Applied AI (AAI). ~**38 h** for AI Infra (INF) — §7 is much bigger for you.
**Before:** [ml_core](../ml_core/ROADMAP.md). **After:** [llm_internals](../llm_internals/ROADMAP.md) — do §7.2 first. **Supersedes:** `README.md` here.
**Your machine:** Apple M4, 16 GB, no NVIDIA GPU. Every build runs on it. What you can only read about is marked **read-only (no GPU)**.

> **Verdict on your original list: best of the three. Two fixes.**
> **(1) Ch. 13 is not optional for AI Infra.** You marked it "if I have time". It is closer to the job than Ch. 3 and Ch. 4 together. It is now §7, and it grew: float formats, memory math, roofline, profiling. None of that was in your list.
> **(2) Ch. 3 and Ch. 4 are a fast pass, not a deep read.** You need softmax and cross-entropy cold; you do not need to hand-write linear regression. Ch. 11 (attention, transformers) moved to [llm_internals](../llm_internals/ROADMAP.md), with the rest of the transformer material.

## The path — in this order

| # | Step | Hours |
|---|---|---|
| 1 | §0 Tensors and autograd — **gate, do not skip** | 2 h |
| 2 | §1 Backpropagation | 3 h |
| 3 | §2 Losses and softmax | 2 h |
| 4 | §3 MLPs, init, normalization, embeddings | 3 h |
| 5 | §4 Optimizers | 2 h |
| 6 | §5 A real training run | 4 h |
| 7 | §6 CNN / RNN awareness — reading only | 1 h |
| 8 | §7 Numerics, memory, speed — read its lane note first | 9 h AAI / 21 h INF |

**Table** = how much each item matters: `Core` = know it cold, `Useful` = know it exists and roughly how, `Skim`/`Skip` are literal. **Start here** = open this one first; max 3 sources per section, the rest were cut on purpose. **Done when** = self-test; fail it, redo the section.

## 0. Tensors and autograd — gate this

| Sub-topic | AAI | INF |
|---|---|---|
| Indexing, `reshape`/`view`/`permute`, broadcasting, `einsum` | Core | Core |
| Device placement, dtype, `.to()` semantics | Core | Core |
| Autograd (PyTorch records your ops and computes gradients for you): graph building, `backward()`, leaf tensors, `detach` vs `no_grad` | Useful | Core |

**Read**
- **Start here.** d2l.ai — [Ch. 2 "Preliminaries"](https://d2l.ai/chapter_preliminaries/index.html). Skim 2.1–2.4 fast. Read [**2.5 "Automatic Differentiation"**](https://d2l.ai/chapter_preliminaries/autograd.html) slowly — the only part of Ch. 2 worth slow reading.

- **Build** (~30 lines, 45 min): make a random tensor of shape `(B, H, T, D)` — batch, heads, time, features. Multiply the last two dims two ways: `matmul` + `transpose`, then `einsum`. Assert they match to 1e-5. Then compute one small gradient on paper and check it against `.backward()`.
- **Done when:** you write `(B, H, T, D)` code and can say what every dim means on every line. And you fix `RuntimeError: shapes cannot be multiplied` without printing shapes first.

## 1. Backpropagation

| Sub-topic | AAI | INF |
|---|---|---|
| The chain rule over a computation graph | Core | Core |
| Cost and memory of the forward vs backward pass | Useful | Core |
| Why activations (each layer's intermediate output) must be kept for the backward pass | Useful | Core |

**Read**
- **Start here.** 3Blue1Brown — [Neural networks series](https://www.3blue1brown.com/topics/neural-networks), all four: *"But what is a neural network?"*, *"Gradient descent, how neural networks learn"*, *"What is backpropagation really doing?"*, *"Backpropagation calculus"*. ~1.5 h, for intuition.
- Andrej Karpathy — [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html), video 1: *"The spelled-out intro to neural networks and backpropagation: building micrograd"*. **Type it, do not watch it.**

- **Build** (~120 lines typed with the video, then +30 min): finish micrograd. Add one operator the video does not add — it does `tanh`, you add `exp` or `**`. Check its gradient against `torch.autograd`.
- **Done when:** you can say why training memory grows with depth × batch × sequence length, and what gradient checkpointing gives up to shrink it (it recomputes activations instead of storing them).

## 2. Losses and softmax

| Sub-topic | AAI | INF |
|---|---|---|
| MSE vs cross-entropy, and why cross-entropy for classification | Core | Core |
| Softmax numerical stability, the log-sum-exp trick | Core | Core |
| Logits (raw scores before softmax) vs probabilities; why losses take logits | Core | Core |
| Temperature as a rescaling of logits | Core | Useful |

**Read**
- **Start here.** d2l.ai — [Ch. 4 "Linear Neural Networks for Classification"](https://d2l.ai/chapter_linear-classification/index.html). Read [**4.1 Softmax Regression**](https://d2l.ai/chapter_linear-classification/softmax-regression.html) and **4.4** (implementation, including numerical stability) properly.
- d2l.ai — [Ch. 3 "Linear Neural Networks for Regression"](https://d2l.ai/chapter_linear-regression/index.html). **Fast pass, 45 min, concepts only.** Do not hand-roll it.

- **Build** (~25 lines, 40 min): write `log_softmax` and cross-entropy yourself. Feed logits of size 1000 to a naive softmax and watch `inf`/`nan` appear. Subtract the max. Watch it work.
- **Done when:** with no notes, you explain why PyTorch's `CrossEntropyLoss` wants logits, not probabilities — and what quietly breaks if you hand it softmax output.

## 3. MLPs, initialization, normalization, embeddings

| Sub-topic | AAI | INF |
|---|---|---|
| Activations: ReLU, GELU, SiLU; dead ReLUs | Useful | Useful |
| Initialization: Xavier / He, and what breaks without it | Useful | Useful |
| BatchNorm vs LayerNorm vs RMSNorm | Core | Core |
| Pre-norm vs post-norm, and why pre-norm won | Core | Core |
| Dropout | Useful | Skim |

**Read**
- **Start here.** d2l.ai — [Ch. 5 "Multilayer Perceptrons"](https://d2l.ai/chapter_multilayer-perceptrons/index.html), all of 5.1–5.6, including **5.4 numerical stability and initialization** and **5.6 dropout**.
- Zhang & Sennrich, *Root Mean Square Layer Normalization* — [arXiv:1910.07467](https://arxiv.org/abs/1910.07467). **§3 only, skip the rest.** It writes out LayerNorm first, then shows what RMSNorm drops. That is why modern LLMs use RMSNorm.
- Xiong et al., *On Layer Normalization in the Transformer Architecture* — [arXiv:2002.04745](https://arxiv.org/abs/2002.04745). **§3 and Fig. 1 only.** Why modern LLMs train without a warmup babysitter.

- **Build** (~20 lines, 30 min): write LayerNorm by hand, then RMSNorm by hand. Compare your LayerNorm to `torch.nn.LayerNorm` on the same input. Print how many operations RMSNorm saved.
- **Done when:** you answer "why LayerNorm and not BatchNorm in a transformer?" with two facts — BatchNorm depends on the batch at inference, and sequences have different lengths — and say in one sentence what RMSNorm removes.

### 3.1 The embedding layer — 45 min

| Sub-topic | AAI | INF |
|---|---|---|
| `nn.Embedding` as a lookup table, not a one-hot matmul | Core | Core |
| Embedding parameters = vocab × hidden, and what share of the model that is | Core | Core |
| Sparse gradients — only the rows you indexed get one | Useful | Core |
| Weight tying: one matrix for the input embedding and the output projection | Core | Useful |

**Read**
- **Start here.** PyTorch — [`nn.Embedding`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html). Short. Note `padding_idx`, note `sparse`, and note that the forward pass is an index select and nothing else.
- Press & Wolf, *Using the Output Embedding to Improve Language Models* — [arXiv:1608.05859](https://arxiv.org/abs/1608.05859). **Abstract and §3 only.** Why the input embedding and the output projection can be the same matrix.
- Karpathy — [nanoGPT `model.py`](https://github.com/karpathy/nanoGPT/blob/master/model.py). Find the one line that ties `wte.weight` to `lm_head.weight`, and read the comment above it. That is weight tying in production code.

- **Build** (~30 lines, 45 min): do the same lookup twice — `one_hot(idx) @ W` and `nn.Embedding` — assert they match, then time both at vocab 50k and see how much the one-hot matmul wastes. Print `vocab × hidden` as a share of total parameters for GPT-2 small (50257 × 768 = 38.6 M out of 124 M). Tie the input and output matrices, print the parameter count before and after. Then run one backward pass on a batch of 4 tokens and print how many embedding rows came back with a nonzero gradient.
- **Done when:** from memory you give the embedding size for vocab 128k × hidden 4096 (~524 M parameters, ~1 GB in bf16), and you say why tying is a large win at 124 M parameters and close to noise at 70 B.

## 4. Optimizers

| Sub-topic | AAI | INF |
|---|---|---|
| SGD → momentum → RMSProp → Adam → AdamW | Core | Core |
| Optimizer state memory (Adam keeps 2 extra tensors per parameter) | Useful | Core |
| LR warmup, cosine decay, why warmup exists | Useful | Core |
| Gradient clipping; reading a loss spike | Useful | Core |

**Read**
- **Start here.** d2l.ai — [Ch. 12 "Optimization Algorithms"](https://d2l.ai/chapter_optimization/index.html): 12.4 (SGD), 12.6 (momentum), 12.10 (Adam), 12.11 (LR scheduling). **Skip 12.7–12.9.**
- Kingma & Ba, *Adam* — [arXiv:1412.6980](https://arxiv.org/abs/1412.6980). **Algorithm 1 only.** Know what `m`, `v`, `β1`, `β2`, `ε` each do.
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization* — [arXiv:1711.05101](https://arxiv.org/abs/1711.05101). Already read in `ml_core` §1. **Re-read Algorithm 2, 10 min**, now that you know Adam.

- **Build** (~40 lines, 45 min): train one tiny model twice on the same data, SGD then Adam, and plot both losses on one chart. Then print total bytes in `optimizer.state_dict()` and compare to total bytes of the parameters.
- **Done when:** on paper you compute Adam's optimizer-state memory for a 7B model in fp32, and you name three causes of a loss spike at step 40k with one check for each.

## 5. A real training run

| Sub-topic | AAI | INF |
|---|---|---|
| `nn.Module`, parameter registration, `state_dict` | Core | Core |
| Custom layers, init, weight sharing | Useful | Core |
| `Dataset`, `DataLoader`, collation, worker count | Core | Core |
| Device management, pinned memory | Useful | Core |
| Checkpoint and resume | Skim | Core |
| Batch size vs memory; gradient accumulation | Useful | Core |
| Reading a failure: OOM vs NaN loss vs plain divergence | Useful | Core |

**Read**
- **Start here.** d2l.ai — [Ch. 6 "Builders' Guide"](https://d2l.ai/chapter_builders-guide/index.html), all of 6.1–6.7. **The best hour-for-hour chapter in the book.** Your original list said this already — it was right.

- **Build 1** (~80 lines, 1.5 h): train a small MLP with a loop you wrote. No Lightning, no `Trainer`. Log train and val loss. Checkpoint every epoch, kill the process mid-run, resume from the checkpoint. Then set the learning rate 100× too high, watch it blow up, fix it with warmup + gradient clipping, and write down what each fix changed.
- **Build 2 — make it OOM** (+20 min): same loop, double the batch size until the process dies. On MPS you get `RuntimeError: MPS backend out of memory`; on CPU there is no clean error at all, the OS OOM-killer just takes the process. Write down the last batch size that fit. Now recover that effective batch with gradient accumulation: forward and `backward()` on every micro-batch with the loss divided by `accum_steps`, and `step()` + `zero_grad()` only every `accum_steps`. Confirm the loss curve tracks the big-batch run — and note that accumulation buys memory with wall-clock time, not for free.
- **Build 3 — cause a loss spike on purpose** (+20 min): §4 asks you to explain loss spikes; produce one first. Inject a corrupted batch at a known step, three variants, one run each — inputs scaled ×1000, 10% of labels randomized, one sample set to `NaN`. Log loss for 200 steps after the bad batch is gone. See which one recovers by itself, which one poisons Adam’s `m` and `v` and keeps hurting long after, and which one turns every later loss into `NaN`. Then re-run with `clip_grad_norm_` plus a skip-the-step-if-the-loss-is-not-finite guard, and record what each fix actually caught.
- **Done when:** from an empty file you write a correct loop — forward, loss, `zero_grad`, backward, step, `eval()`, `no_grad` — plus save and resume that survives being killed halfway. And, shown only a training log, you tell an OOM from a corrupted batch from a too-high learning rate, with the fix for each.

## 6. Awareness pass — CNNs and RNNs

| Sub-topic | AAI | INF |
|---|---|---|
| Convolution, pooling, receptive fields; ResNet's skip connection | Skim | Skim |
| The RNN sequential bottleneck; what transformers replaced | Skim | Skim |

**Read**
- **Start here.** d2l.ai — read only the chapter **summaries** of [Ch. 7](https://d2l.ai/chapter_convolutional-neural-networks/index.html), [Ch. 8](https://d2l.ai/chapter_modern-convolutional-neural-networks/index.html), [Ch. 9](https://d2l.ai/chapter_recurrent-neural-networks/index.html), [Ch. 10](https://d2l.ai/chapter_recurrent-modern/index.html). **50 min for all four. No build.**

- **Done when:** one sentence on why residual connections let networks get deep, and one on why an RNN cannot process a sequence in parallel during training but a transformer can.

---

## 7. Numerics, memory, and speed

d2l Ch. 13 plus the money material that was in none of your READMEs. Do it **before** `llm_internals` — transformers make far more sense once you can count bytes. **Pick your lane first:**

- **Applied AI:** do **7.2 only**, plus the **two Core rows of 7.3** (prefill vs decode, and batching). **Skip 7.1, 7.4, 7.5 for now.** ~9 h. Come back later.
- **AI Infra:** do all five, in order. ~21 h. **The highest-value section in this file for you.**

### 7.1 Float formats and mixed precision — INF now, AAI later

| Sub-topic | AAI | INF |
|---|---|---|
| fp32, tf32, fp16, bf16, fp8 — exponent bits vs mantissa bits | Useful | Core |
| Why bf16 beat fp16 for training | Useful | Core |
| Mixed precision: autocast, loss scaling, fp32 master weights | Skim | Core |
| Quantization vs low-precision training — different problems | Useful | Core |

**Read**
- **Start here.** Hugging Face — [Methods and tools for efficient training on a single GPU](https://huggingface.co/docs/transformers/en/perf_train_gpu_one). Just the bf16 / fp16 / tf32 comparison table.
- PyTorch — [AMP recipe](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html) and the [AMP package docs](https://docs.pytorch.org/docs/stable/amp.html). Goal: understand why `GradScaler` exists. That is the whole fp16 story. **Read-only (no GPU)** — autocast training needs CUDA.
- *INF only.* Stas Bekman — [ml-engineering](https://github.com/stas00/ml-engineering), directory `training/dtype/`. Written from real cluster pain, the most honest free source here.

- **Build** (~15 lines, 20 min): put a big number in a tensor, cast it to fp16 and to bf16, print both. fp16 becomes `inf`, bf16 survives. That one print is the whole argument.
- **Done when:** you answer "why bf16 and not fp16" using the words *dynamic range* and *precision*, and say what problem loss scaling works around.

### 7.2 Memory accounting — everyone does this one

| Sub-topic | AAI | INF |
|---|---|---|
| Training memory = parameters + gradients + optimizer states + activations | Useful | Core |
| The ~16 bytes-per-parameter figure for Adam in mixed precision, and where it comes from | Useful | Core |
| Activation memory grows with batch × sequence × layers | Useful | Core |
| Activation checkpointing: spend compute to save memory | Skim | Core |
| Inference memory = weights + KV cache + activations | Core | Core |
| KV cache size formula, and what GQA does to it | Core | Core |

**Read**
- **Start here.** Stanford CS336 Spring 2025 — [course site](https://stanford-cs336.github.io/spring2025/), **Lecture 2: "PyTorch, resource accounting."** Free on YouTube. Watch with a notebook open and redo every number yourself.
- kipply — [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/). KV-cache and FLOPs math worked end to end. Short and exact.
- Rajbhandari et al., *ZeRO* — [arXiv:1910.02054](https://arxiv.org/abs/1910.02054). **§3 and Table 1 only** — the parameter / gradient / optimizer-state split. Skip the rest for now; §5 comes back in 7.5.

- **Build** (~60 lines, 1.5 h): `memory_math.py`. In: layers, hidden size, attention heads, KV heads, vocab, dtype, batch, sequence length. Out: training memory split into the four terms, plus inference KV-cache size. Check it against Llama 3 8B's published numbers — target within 15%.
- **Done when:** on a whiteboard, under 3 minutes, you get KV-cache bytes for a 7B model at batch 32, 4k context, GQA with 8 KV heads, bf16 — within 10%.

### 7.3 Arithmetic intensity and the roofline

| Sub-topic | AAI | INF |
|---|---|---|
| FLOPs vs bytes moved; arithmetic intensity (math done per byte fetched) | Skim | Core |
| The roofline model; compute-bound vs memory-bound | Skim | Core |
| GPU memory hierarchy: registers → SMEM → L2 → HBM; bandwidth vs capacity | Skim | Core |
| **Why prefill is compute-bound and decode is memory-bandwidth-bound** | Core | Core |
| Why batching lifts decode throughput but barely helps decode latency | Core | Core |
| Kernel fusion, `torch.compile`, CUDA graphs — concept level | Skim | Core |

**Read**
- **Start here.** Horace He — [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html). Compute vs memory vs overhead. Read before anything else here.
- NVIDIA — [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/). Vendor blog, but its batching and memory-bound explanation is correct and short. **Covers the two Core rows** — if you are AAI, this plus Horace He is all of 7.3.
- *INF only.* Dao et al., *FlashAttention* — [arXiv:2205.14135](https://arxiv.org/abs/2205.14135), **§1–3 only.** Read it as an IO-aware algorithm that never builds the full N×N matrix. You do not need the CUDA.

- **Build** (~20 lines, 30 min): time a square matmul at growing sizes on your M4, print size vs GFLOP/s, and find where speed stops improving. That flat part is your own machine's memory-bound region.
- **Done when:** you draw a roofline, mark prefill and decode on it, and use that picture to say why continuous batching helps throughput while speculative decoding helps latency.

### 7.4 Profiling — INF

| Sub-topic | AAI | INF |
|---|---|---|
| `torch.profiler`: record a run, read a trace, find the stall | Skim | Core |
| Telling overhead-bound / compute-bound / memory-bound apart in a trace | Skim | Core |
| `nsys` / Nsight Systems, basic use | Skip | Core |

**Read**
- **Start here.** PyTorch — [Profiling your PyTorch Module (recipe)](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html).
- PyTorch blog — [Accelerating Generative AI with PyTorch II: GPT, Fast](https://pytorch.org/blog/accelerating-generative-ai-2/). A real optimization sequence with a measured number at each step. This is the shape of answer an infra interviewer wants.

- **Build** (~30 lines, 1 h): profile one training step of your §5 MLP, name the top 3 operations by time, apply `torch.compile`, re-profile, report the delta. **Partly read-only (no GPU)** — you get CPU traces, not CUDA kernels, and `nsys` you can only read about.
- **Done when:** given a trace you say whether the run is overhead-, compute-, or memory-bound, and name the next thing you would change.

### 7.5 Parallelism — INF, and last

| Sub-topic | AAI | INF |
|---|---|---|
| Data parallelism; DDP and the gradient all-reduce | Skim | Core |
| ZeRO stages 1 / 2 / 3 and what each shards | Skim | Core |
| FSDP in practice | Skip | Core |
| Tensor parallelism; where the collectives land in a transformer block | Skim | Core |
| Pipeline parallelism and the bubble | Skim | Core |
| Collectives: all-reduce, all-gather, reduce-scatter; NCCL | Skip | Core |
| Stragglers: why one slow node stalls 512 GPUs | Skip | Core |

**Read**
- **Start here.** CS336 — **Lectures 7 and 8: "Parallelism."** Both. The clearest free treatment there is. For ZeRO stages, reuse the paper from 7.2 — **§5** has the stage-by-stage table. No new reading.
- Shoeybi et al., *Megatron-LM* — [arXiv:1909.08053](https://arxiv.org/abs/1909.08053), **§3 only**: the tensor-parallel split of the MLP and attention blocks, with exact collective placement. Memorize that figure.
- d2l.ai — [Ch. 13 "Computational Performance"](https://d2l.ai/chapter_computational-performance/index.html), **13.5 (multi-GPU)** and **13.7 (parameter servers)**. The chapter your original list deferred.

- **Build** (~50 lines, 1.5 h): run DDP across 2 processes with the `gloo` backend on CPU — this works on your M4. Measure how much of each step is gradient all-reduce, at 3 model sizes, and note where communication starts to dominate. **Read-only (no GPU):** FSDP, NCCL, real tensor parallelism.
- **Done when:** you say what data, tensor, and pipeline parallelism each shard and what each costs in communication — and why one node at 5% packet loss can cost far more than 5% of cluster GPU time.

---

## What was cut

**From your original README:** StatQuest Neural Networks playlist (3B1B + Karpathy is strictly better) · d2l **Ch. 14** (Computer Vision) · **Ch. 15–16** (NLP pretraining and applications — Raschka covers this better) · **Ch. 17** (RL — you reach it through RLHF instead) · **Ch. 18–22** (Gaussian Processes, HPO, GANs, appendix) · Sebastian Raschka's blog archive (good, but consolidation, not a path) · deep reading of **Ch. 3**.

**Moved, not cut:** **Ch. 11** → `llm_internals`. **Ch. 13** → §7 here, expanded.
**Trimmed in this pass, to keep 3 sources max per section:** the LayerNorm paper (RMSNorm §3 covers both) · the scaling book · FlashAttention-2 · CS336 Lectures 5 and 6 · the FSDP tutorial · Stas Bekman's `model-parallelism/` and `network/` directories. All real, all good — but they are the *second* pass. Do not open them until you have finished this file once.
