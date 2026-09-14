# Reading — Lecture 3

Every URL below was fetched and checked on 2026-09-14. PyTorch's "stable" documentation resolved to version 2.14 on that date (a few pages to 2.13); section names refer to those pages, and where a page moved, the working URL is given. Following the course rule, each topic has at most three sources marked as *primary*; everything else is reference for when you need to go deeper.

## The 30-minute assigned reading (Independent work → Reading)

Read these, in this order, marked sections only. Do not read the whole page.

1. **PyTorch docs — "Autograd mechanics."** Read *How autograd encodes the history* (including *Saved tensors*), *Locally disabling gradient computation* (all five subsections: Setting `requires_grad`, Grad Modes, No-grad Mode, Inference Mode, Evaluation Mode), and *In-place operations with autograd*. Skip the multithreading and complex-numbers sections. About 12 minutes.
   https://docs.pytorch.org/docs/stable/notes/autograd.html

2. **PyTorch tutorial — "A Gentle Introduction to torch.autograd."** Read *Background*, *Differentiation in Autograd*, and *Computational Graph* including the note "DAGs are dynamic in PyTorch." Skip the code you already understand. About 8 minutes.
   https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

3. **Baydin, Pearlmutter, Radul & Siskind (2018), "Automatic Differentiation in Machine Learning: a Survey," JMLR 18(153).** Read §2 *What AD Is Not* (§2.1 numerical, §2.2 symbolic) and §3.2 *Reverse Mode* — specifically the cost paragraph (`n·c·ops(f)` versus `m·c·ops(f)`, `c < 6`) and the storage sentence. About 10 minutes.
   https://jmlr.org/papers/v18/17-468.html (PDF: https://www.jmlr.org/papers/volume18/17-468/17-468.pdf)

If you have time left: Korthikanti et al. (2022), §4.1 *Activation Memory Per Transformer Layer*, to see Equation 1 derived one saved tensor at a time — https://arxiv.org/abs/2205.05198.

**Before the build (optional, 2 h 25 min, not counted in the 30):** Karpathy, "The spelled-out intro to neural networks and backpropagation: building micrograd." Watch it *before* you start, then close it and build against `test_value.py`, which is stricter in three places the assignment names. https://www.youtube.com/watch?v=VMj-3S1tku0

---

## Full annotated source list by topic

### 1 · A tensor library as a runtime; indexing, view / reshape / permute, broadcasting

| Source | Read | Why |
|---|---|---|
| **Yang, "PyTorch internals" (blog, May 2019)** — http://blog.ezyang.com/2019/05/pytorch-internals/ | *The Tensor Data Structure*; *Strides and Views*; *Tensor-Storage Separation*; *Dispatch Mechanism*; the first paragraphs of *Automatic Differentiation* | *Primary.* The (storage, sizes, strides, offset) model, with pictures; the two-level dispatch; where `AutogradMeta` lives. |
| **PyTorch docs — "Tensor Views"** — https://docs.pytorch.org/docs/stable/tensor_view.html | Whole page (it is short) and the list of view ops | *Primary.* "View tensor shares the same underlying data with its base tensor"; `transpose` produces non-contiguous views; `reshape` may or may not be a view. |
| **PyTorch docs — "Broadcasting semantics"** — https://docs.pytorch.org/docs/stable/notes/broadcasting.html | *General semantics*; *In-place semantics* | *Primary.* The two rules, verbatim; the in-place shape restriction. |
| PyTorch docs — `torch.Tensor.view` — https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html | The contiguity-like condition `stride[i] = stride[i+1] * size[i+1]` and the recommendation to use `reshape` when unsure | Exactly when `view` can and cannot merge dimensions. |
| PyTorch docs — `torch.Tensor.permute`, `torch.Tensor.stride`, `torch.Tensor.is_contiguous`, `torch.reshape` — https://docs.pytorch.org/docs/stable/generated/torch.Tensor.permute.html · …/torch.Tensor.stride.html · …/torch.Tensor.is_contiguous.html · …/torch.reshape.html | One paragraph each | permute "Returns a view"; the definition of stride; reshape "you should not depend on the copying vs. viewing behavior." |
| PyTorch docs — `torch.cuda.memory.memory_allocated`, `memory_reserved` — https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.memory_allocated.html · …/torch.cuda.memory.memory_reserved.html | One paragraph each | Why allocated "is likely less than the amount shown in `nvidia-smi`"; what the caching allocator holds. |
| Paszke et al. (2017), "Automatic differentiation in PyTorch," NIPS Autodiff Workshop — https://openreview.net/pdf?id=BJJsrmfCZ | §1 Background; §3 Implementation; §4 in-place ops | The eager/recording design in four pages; views share storage via sizes and strides; the version counter. |
| Paszke et al. (2019), "PyTorch: An Imperative Style, High-Performance Deep Learning Library," NeurIPS — https://arxiv.org/abs/1912.01703 | §4.3 Automatic differentiation; §5.3 Custom caching tensor allocator | Why `cudaMalloc`/`cudaFree` are avoided; 512-byte rounding; why allocated ≠ reserved. |
| Rush, "Tensor Considered Harmful" (2019) — http://nlp.seas.harvard.edu/NamedTensor | *Tensor Traps* 1–3 | "Broadcasting by alignment" — why positional broadcasting has no semantics. |

### 2 · Reading `(B, H, T, D)`; `einsum`

| Source | Read | Why |
|---|---|---|
| **Vaswani et al. (2017), "Attention Is All You Need"** — https://arxiv.org/abs/1706.03762 (HTML: https://arxiv.org/html/1706.03762v7) | §3.2.2 Multi-Head Attention | *Primary.* `h = 8`, `d_k = d_v = d_model/h = 64`; the projection shapes. |
| **PyTorch docs — `torch.nn.functional.scaled_dot_product_attention`** — https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html | The shape legend and the shape table | *Primary.* Query `(N, ..., Hq, L, E)`, key `(N, ..., H, S, E)`, value `(N, ..., H, S, Ev)`, output `(N, ..., Hq, L, Ev)` — `L ≠ S` made official. |
| **PyTorch docs — `torch.einsum`** — https://docs.pytorch.org/docs/stable/generated/torch.einsum.html | Whole page, including the `opt_einsum` note | *Primary.* The grammar of the equation string; implicit output order; contraction-order optimization needs three operands and the package. |
| PyTorch docs — `torch.matmul` — https://docs.pytorch.org/docs/stable/generated/torch.matmul.html | The "batched matrix multiply" cases | Leading dimensions broadcast, the last two are the matrix — why `H` must precede `T, D`. |
| Shazeer, "Shape Suffixes — Good Coding Style" (Medium, 28 Feb 2024) — https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd | Whole post (short) | The naming convention the course adopts. |
| Rogozhnikov, "Einops: Clear and Reliable Tensor Manipulations with Einstein-like Notation," ICLR 2022 — https://openreview.net/pdf?id=oapKSVM2bcj | Abstract; §1 | Why `reshape` "easily breaks the tensor structure" and why such mistakes are undetectable by tests. |
| Rush, "Tensor Considered Harmful" | *Named Tensor: A Prototype*, Proposals 1–3; *Example: Neural Attention* | Broadcasting and contraction by name; attention written with named dims. |

### 3 · Devices, dtypes, `.to()`

| Source | Read | Why |
|---|---|---|
| **PyTorch docs — "Tensor Attributes"** — https://docs.pytorch.org/docs/stable/tensor_attributes.html | The `torch.dtype` table (S-E-M layouts); *Type promotion*; `torch.device` | *Primary.* bf16 = 1-8-7, fp16 = 1-5-10; the promotion rules including the zero-dim exception. |
| **PyTorch docs — `torch.Tensor.to`** — https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to.html | Whole page | *Primary.* "If the `self` Tensor already has the correct dtype and device, then `self` is returned"; `non_blocking` and pinned memory; `copy`. |
| PyTorch docs — `torch.mps.current_allocated_memory`, `torch.mps.driver_allocated_memory` — https://docs.pytorch.org/docs/stable/generated/torch.mps.current_allocated_memory.html · https://docs.pytorch.org/docs/stable/generated/torch.mps.driver_allocated_memory.html | One paragraph each | What each number includes; needed for the production drill on Apple silicon. |
| PyTorch docs — `torch.cuda.memory.max_memory_allocated`, `torch.cuda.memory.reset_peak_memory_stats` — https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.max_memory_allocated.html · https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.reset_peak_memory_stats.html | One paragraph each | The peak counter and its reset. (The older `torch.cuda.max_memory_allocated` URLs now 404; the functions are aliases.) |

### 4 · Autograd mechanics; `detach` vs `no_grad`

| Source | Read | Why |
|---|---|---|
| **PyTorch docs — "Autograd mechanics"** — https://docs.pytorch.org/docs/stable/notes/autograd.html | As in the assigned reading | *Primary.* Everything in part 4 §4.1–4.5 is quoted from here. |
| **PyTorch tutorial — "A Gentle Introduction to torch.autograd"** — https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html | As in the assigned reading | *Primary.* The three verbs of `backward()`; leaves and roots; "DAGs are dynamic." |
| **PyTorch docs — FAQ, "My model reports 'cuda runtime error(2): out of memory'"** — https://docs.pytorch.org/docs/stable/notes/faq.html | That section, all five sub-headings | *Primary.* `total_loss += loss`; holding references; RNN length; large linear layers; "Consider checkpointing." |
| PyTorch docs — `torch.no_grad`; `torch.autograd.grad_mode.inference_mode`; `torch.Tensor.detach` — https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html · https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html · https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html | Whole page each (short) | The exact contracts. (The `detach` and `reshape` web pages returned only navigation to the authoring tools; their docstrings were read from an installed PyTorch 2.14 instead and quoted verbatim.) |
| d2l.ai §2.5 "Automatic Differentiation" — https://d2l.ai/chapter_preliminaries/autograd.html | §2.5.1, §2.5.3 Detaching Computation, §2.5.4 Gradients and Python Control Flow | Gradient accumulation and `zero_()`; detach; dynamic control flow, with runnable code. |
| Karpathy, micrograd — https://github.com/karpathy/micrograd (`micrograd/engine.py`) | README example; `engine.py` (≈ 100 lines) | The whole autograd idea at scalar scale. The README example is frozen as a test in the build. Read *after* your own `backward()` works. |

### 5 · Debugging shape errors

| Source | Read | Why |
|---|---|---|
| Rogozhnikov, "Einops" (ICLR 2022) — https://openreview.net/pdf?id=oapKSVM2bcj | §1, the `reshape` example (`y1`/`y2`) | The silent class of shape bug, demonstrated. |
| Shazeer, "Shape Suffixes" — as above | — | Suffixes as pre-written derivations. |
| PyTorch docs — `torch.Tensor.view`; "Broadcasting semantics"; `torch.einsum` — as above | — | The three error signatures in part 5 §5.4 come from these three pages' rules. The exact error strings in §5.4 were reproduced on PyTorch 2.14 on 2026-09-14. |
| PyTorch docs — "Tensor Attributes" (`torch.device`) — as in topic 3; and the "Meta device" note — https://docs.pytorch.org/docs/stable/meta.html | `torch.device` types; the meta note's first section | `"meta"` as a device type; tensors with shape and dtype but no storage. (The meta page's title resolved on 2026-09-14 but its body did not render for the authoring tools; part 3 §3.5's description is from the Tensor Attributes device list and from running it.) |

### 6 · The chain rule over a graph; forward vs reverse mode; finite differences

| Source | Read | Why |
|---|---|---|
| **CS231n, "Backpropagation, Intuitions"** — https://cs231n.github.io/optimization-2/ | *Intuitive understanding of backpropagation*; *Backprop in practice: Staged computation*; *Patterns in backward flow*; the "Gradients add up at forks" paragraph | *Primary.* Local gradients; add distributes, max routes, multiply switches; gradients add at forks; cache forward variables. |
| **Baydin et al. (2018), "Automatic Differentiation in Machine Learning: a Survey"** — https://jmlr.org/papers/v18/17-468.html | §2, §2.1, §2.2, §3.1, §3.2 | *Primary.* AD vs numerical vs symbolic; the reverse-mode cost bound and storage cost; backprop as a special case. |
| **Karpathy, "Yes you should understand backprop" (Medium, Dec 2016)** — https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b | Whole post | *Primary.* The "leaky abstraction" argument: sigmoid saturation, dying ReLUs, exploding RNN gradients. |
| Griewank & Walther, *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation*, 2nd ed., SIAM 2008 — https://doi.org/10.1137/1.9780898717761 | Ch. 3 (the cheap gradient bound), via Baydin's citation | The origin of the reverse-mode cost bound. The SIAM page could not be fetched from the authoring environment; the bibliographic record was confirmed via Crossref. |
| Wikipedia, "Automatic differentiation" — https://en.wikipedia.org/wiki/Automatic_differentiation | *Reverse accumulation* | The adjoint notation and the Wengert list. |
| Wikipedia, "Numerical differentiation" — https://en.wikipedia.org/wiki/Numerical_differentiation | *Finite difference approximations*; *Practical considerations using floating-point arithmetic* | Central difference; why the first-order errors cancel; truncation vs round-off. (This page gives the `√ε` rule for the *forward* difference only.) |
| FiniteDiff.jl docs, "Step Size Selection" — https://docs.sciml.ai/FiniteDiff/dev/epsilons/ | Whole page (short) | `h* = ε^(1/3)` for central differences: "balances O(h²) truncation with O(ε/h) round-off." |
| PyTorch docs — `torch.autograd.gradcheck` — https://docs.pytorch.org/docs/stable/generated/torch.autograd.gradcheck.gradcheck.html | Parameters and the double-precision warning | `eps=1e-6, atol=1e-5, rtol=1e-3`; "designed for input of double precision." |

### 7 · Cost and memory of forward vs backward; activations

| Source | Read | Why |
|---|---|---|
| **Korthikanti, Casper, Lym, McAfee, Andersch, Shoeybi & Catanzaro (2022), "Reducing Activation Recomputation in Large Transformer Models"** — https://arxiv.org/abs/2205.05198 | §4 (the definition of "activations"); §4.1 Equation 1 and its derivation; §5 | *Primary.* `sbh(34 + 5as/h)`, one saved tensor at a time; the `s²` term; selective recomputation at 2–3 % overhead. |
| **Rajbhandari, Rasley, Ruwase & He (2020), "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models," SC '20** — https://arxiv.org/abs/1910.02054 | §3 *Where Did All the Memory Go?*, §3.1, §3.2 | *Primary.* 16 bytes per parameter under mixed-precision Adam, itemized; activations as "residual states"; 33 % recomputation overhead. |
| **Kaplan et al. (2020), "Scaling Laws for Neural Language Models"** — https://arxiv.org/abs/2001.08361 | §2.1 Notation, the paragraph after Table 1 | *Primary.* `C_forward ≈ 2N`, backward "approximately twice the compute as the forwards pass," `C ≈ 6N`. |
| Chen, Xu, Zhang & Guestrin (2016), "Training Deep Nets with Sublinear Memory Cost" — https://arxiv.org/abs/1604.06174 | §1 (the `O(n)` intermediate-results sentence) | Why memory is linear in depth. |
| Hugging Face Transformers docs, "GPU memory usage" — https://huggingface.co/docs/transformers/model_memory_anatomy | Whole page (short; the page was rewritten in 2025–26 and no longer has the old bullet list) | Bytes per parameter for weights, gradients, optimizer; activations "vary in size with batch size, sequence length, model depth, and hidden size"; temporary spikes. |
| Shi & DeVito, "Understanding GPU Memory 1: Visualizing All Allocations over Time" (PyTorch blog, Dec 2023) — https://pytorch.org/blog/understanding-gpu-memory-1/ | The first snapshot and its description | The sawtooth: "the rise of memory in the forward pass and the fall during the backward pass"; the optimizer-state floor. |
| PyTorch docs — "Understanding CUDA Memory Usage" — https://docs.pytorch.org/docs/stable/torch_cuda_memory.html | Whole page | How to take and read a memory snapshot; what the profiler can and cannot see. |
| PyTorch docs — FAQ, OOM section — as above | The RNN and linear-layer sub-headings | Memory linear in sequence length; weights vs activations of a linear layer. |

### 8 · Gradient checkpointing

| Source | Read | Why |
|---|---|---|
| **Chen, Xu, Zhang & Guestrin (2016), "Training Deep Nets with Sublinear Memory Cost"** — https://arxiv.org/abs/1604.06174 | Abstract; §4.1 General Methodology; §4.3 An O(√n) Memory Cost Algorithm; §4.4 (the recursion and `log n`); §5 (the 30 % figure) | *Primary.* The √n argument in two pages. |
| **PyTorch docs — `torch.utils.checkpoint`** — https://docs.pytorch.org/docs/stable/checkpoint.html | The `checkpoint` function's description, the two warnings, and the `use_reentrant` bullet list | *Primary.* "trades compute for memory"; what is kept and what is recomputed; the purity warning; `preserve_rng_state`; why `use_reentrant=False`. |
| **Korthikanti et al. (2022)** — as above | §1 (30–40 % overhead); §5 Selective Activation Recomputation, Equation 6 | *Primary.* Recompute only the attention internals: the `5as/h` term disappears for 2–3 % FLOPs. |
| Griewank (1992), "Achieving logarithmic growth of temporal and spatial complexity in reverse automatic differentiation," *Optimization Methods and Software* 1(1), 35–54 — https://doi.org/10.1080/10556789208805505 | Title and abstract, if you can reach it | The origin of checkpointing. Paywalled; confirmed via Crossref only. |
| Rajbhandari et al. (2020), "ZeRO" — as above | §3.2 | The 33 % figure and the √ activation reduction. |
| Hugging Face Transformers docs, "GPU" — https://huggingface.co/docs/transformers/main/en/perf_train_gpu_one | *Gradient checkpointing*; *Gradient accumulation* | The practitioner's knobs; "~20 %" for partial checkpointing. |

### System design drill

| Source | Read | Why |
|---|---|---|
| Beyer, Jones, Petoff & Murphy (eds.), *Site Reliability Engineering*, ch. 22 "Addressing Cascading Failures" — https://sre.google/sre-book/addressing-cascading-failures/ | *Queue Management*; *Load Shedding and Graceful Degradation*; *Retries*; *Testing for Cascading Failures* | Queue length ≤ 50 % of pool; shedding vs degradation; retry amplification; "load test components until they break." |
| *Site Reliability Engineering*, ch. 21 "Handling Overload" — https://sre.google/sre-book/handling-overload/ | *The Pitfalls of "Queries per Second"*; *Client-Side Throttling*; *Criticality*; *Utilization Signals* / degraded responses | Capacity is not QPS; adaptive throttling; ordered shedding by criticality. |
| Brooker, "Exponential Backoff And Jitter" (AWS Architecture Blog, 2015) — https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/ | Whole post | Why plain exponential backoff clusters; Full Jitter: `sleep = rand(0, min(cap, base * 2 ** attempt))`. |
| Python docs — `concurrent.futures` — https://docs.python.org/3/library/concurrent.futures.html | *ProcessPoolExecutor* first paragraph; `max_workers` defaults | Side-stepping the GIL, and its pickling cost. |
| Python docs — `asyncio.Queue`, `asyncio.Semaphore` — https://docs.python.org/3/library/asyncio-queue.html · https://docs.python.org/3/library/asyncio-sync.html | `maxsize`; `put_nowait` / `QueueFull`; the Semaphore counter | The default queue is *infinite*; the bounded primitives, verbatim. |
| Wikipedia, "Little's law" — https://en.wikipedia.org/wiki/Little%27s_law | Statement | `L = λW`, "not influenced by the arrival process distribution, the service distribution, the service order, or practically anything else." |

### Production drill

| Source | Read | Why |
|---|---|---|
| PyTorch docs — `torch.cuda.memory.max_memory_allocated`, `reset_peak_memory_stats`, `torch.mps.current_allocated_memory`, `torch.mps.driver_allocated_memory` — as in topic 3 | — | What each counter includes. |
| Python docs — `resource.getrusage` — https://docs.python.org/3/library/resource.html | `ru_maxrss` | Peak RSS for CPU runs; note the unit differs between macOS (bytes) and Linux (kilobytes). |
| Shi & DeVito, "Understanding GPU Memory 1" — as above | — | Where the peak sits in a step. |

---

## Sources that could not be reached from the authoring environment

Listed so you know they were not silently verified.

- AWS Builders' Library, "Using load shedding to avoid overload" (Yanacek) and "Timeouts, retries, and backoff with jitter" (Brooker). Both now redirect to builder.aws.com, which did not return article content. Both are worth reading for the system design drill if you can open them; the SRE-book chapters above cover the same ground.
- Griewank & Walther (SIAM) and Griewank (1992, Taylor & Francis): publisher pages blocked; bibliographic records confirmed via Crossref. Cited only for attribution.
- The `torch.Tensor.detach` and `torch.reshape` documentation pages returned navigation only; the same docstrings were read from an installed PyTorch 2.14 (`torch.Tensor.detach.__doc__`, `torch.reshape.__doc__`) and are quoted from there. All error-message strings in part 5 §5.4 and the behaviours in part 3 §3.3–3.4 were likewise executed and checked on 2.14.
- Numerical Recipes §5.7 (the classical source for the `ε^(1/3)` step-size rule) — the mirror found by search failed TLS verification; FiniteDiff.jl's documentation is cited instead for the same rule.
