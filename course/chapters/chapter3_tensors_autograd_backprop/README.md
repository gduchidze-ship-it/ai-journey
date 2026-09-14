# Chapter 3 — Tensors, autograd, backpropagation

**Week 2 · Lecture A · fully theoretical · 2 h lecture + 4 h independent work**

Week 1 was about judging a fitted function. This week opens the machine that does the fitting. Everything in it is a consequence of three jobs a tensor library performs — it records a graph of what you asked for, it executes each operation on a device, and it walks the graph backwards to produce gradients — and the chapter is organized so that each job is seen first as a data-structure question (what is a tensor, what does a node save) and then as a cost question (how many FLOPs, how many bytes, and why the bytes are the thing that fails). By the end you should be able to predict a training step's peak memory from a config and a batch shape, say which of four levers to pull when it does not fit, and explain why `detach`, `no_grad`, `inference_mode` and `eval()` are four different things.

## How to use this folder

Read the lecture notes in order (they build on each other; part 7 assumes the saved-tensors idea from parts 4 and 6, and part 8 assumes part 7's formula), then do the independent work in the order listed. Every source cited in the notes is real, was fetched and checked on 2026-09-14, and is listed with the exact section to read in [`READING.md`](READING.md). Prefer those sources over memory when the notes and your intuition disagree.

There are **no solutions in this folder**. The autograd build ships with a test file so you can check yourself — including an independent finite-difference oracle that does not depend on anything you write — and the drills ship with rubrics. If you are stuck, the notes tell you which source section unblocks you.

## File map

| Path | What it is | Time |
|---|---|---|
| `lecture/01_tensor_as_runtime.md` | A tensor as (storage, sizes, strides, offset, dtype, device); `view` / `reshape` / `permute` and when each copies; indexing views vs copies; the two broadcasting rules and the silent failure; dispatcher and caching allocator; the runtime's three jobs | 20 min |
| `lecture/02_shapes_and_einsum.md` | `B, H, T, S, D` and their names in Vaswani / PyTorch / Shazeer; the four questions for reading a shape; the attention forward with every shape and both copies annotated; shape suffixes; `einsum` grammar and costs | 20 min |
| `lecture/03_devices_and_dtypes.md` | `torch.device` rules; fp32 / fp16 / bf16 bit layouts and what each buys; type promotion; the `.to()` contract and five bugs it explains; the `meta` device | 15 min |
| `lecture/04_autograd_graph.md` | How the graph is recorded; leaf tensors and `requires_grad`; the three verbs of `backward()`; saved tensors and the version counter; `detach` vs `no_grad` vs `inference_mode` vs `eval()` with a decision table; the accumulating-history OOM | 20 min |
| `lecture/05_debugging_shapes.md` | Why printing shapes first is slow; predict → suspect → one assertion in named symbols → contents check; the three error signatures; a worked case | 15 min |
| `lecture/06_chain_rule_over_a_graph.md` | Local gradients and the backward message; gradients add at forks; reverse topological order; forward vs reverse mode and the cheap-gradient bound; what AD is not; finite differences, `h ≈ ε^(1/3)`, and the kink caveat | 20 min |
| `lecture/07_forward_vs_backward_cost_and_memory.md` | Backward ≈ 2× forward, `C ≈ 6N`; the four residents of memory and 16 bytes/param; **why every layer's activations must be kept** and where the peak is; `sbh(34 + 5as/h)` derived term by term; a worked GPT-2-small budget; `M(b) = M₀ + m·b` | 20 min |
| `lecture/08_gradient_checkpointing.md` | Recompute instead of store; the √n argument; +33 % FLOPs and the purity condition; `torch.utils.checkpoint` semantics; selective recomputation at 2–3 %; the four levers ranked; why memory grows with batch × sequence × depth | 20 min |
| `lecture/SELF_CHECK.md` | 38 questions to answer from memory. No answers given. | — |
| `READING.md` | The 30-minute marked reading, plus the full annotated source list by topic | 30 min |
| `independent_work/build_autograd/` | **Build (2 h).** Scalar autograd engine: `value.py` (the `Value` class, arithmetic, `exp`/`log`/`tanh`/`relu`, non-recursive `backward()`), `gradcheck.py` (central-difference oracle), 43 tests | 2 h |
| `independent_work/drills/system_design_concurrent_image_service.md` | **System design drill (45 min).** A concurrent image processing service — bounded concurrency from a memory budget, queueing, partial failure, overload | 45 min |
| `independent_work/drills/production_peak_memory_three_batch_sizes.md` | **Production drill (45 min).** Peak memory at three batch sizes; fit the line from two, predict the third before measuring, explain the miss | 45 min |
| `COVERAGE_AUDIT.md` | Author's self-reflection: every syllabus bullet mapped to where it is covered, review findings, and known gaps | — |

## Time budget for the 4 h independent block

| Slot | Time | Deliverable that lands in your repo |
|---|---|---|
| Build | 2 h | `value.py` and `gradcheck.py` passing `test_value.py`, and `NOTES.md` with five short answers (including the `h` sweep) |
| System design drill | 45 min | `image_service_design.md` — your design, then your written delta |
| Production drill | 45 min | `peak_memory.md` — three measurements, the prediction written before the third, the miss explained; plus the measuring script |
| Reading | 30 min | Nothing to hand in; the self-check questions assume you did it |

## What you should be able to do by the end

Explain to another engineer, without notes, why `x.transpose(0, 1).view(-1)` raises and `reshape` does not, and what the `reshape` costs; write every shape in an attention forward from `(B, T, Dm)` to the merged output and mark the one copy; state the two broadcasting rules and the case where they succeed wrongly; give the bit layouts of fp16 and bf16 and say which needs loss scaling; state the `.to()` contract in two sentences; fill the `detach` / `no_grad` / `inference_mode` / `eval()` table from memory and pick the right one for a validation loop, an EMA target, an optimizer step, and a serving path; trace why `total_loss += loss` grows memory; write the backward messages for `+`, `×`, `exp`, `tanh`, `relu`; explain why reverse mode is cheap in compute and expensive in memory, and where the memory goes; derive `sbh(34 + 5as/h)` for one transformer layer and compute it for GPT-2 small at `s = 1024, b = 8`; and say what full and selective activation recomputation each save and cost.

This chapter feeds **Project 2 — Autograd, training loop, memory calculator** (16 h, due end of week 3). The `Value` engine you build here is its first third; Lecture 4 adds the operator you will extend it with, Lectures 5–6 add the training loop, and the production drill's measurements are the first data `memory_math.py` must reproduce.
