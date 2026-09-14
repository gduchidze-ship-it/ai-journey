# Production drill — Peak memory at three batch sizes; predict the third before you measure it (45 min)

**The task, verbatim from the syllabus.** Take a training or inference script you can run and record peak memory at three batch sizes. Predict the third from the first two before measuring.

**Deliverable.** `peak_memory.md` in your repo, filled in from the template below, plus the script you measured with. It is short. The value is in the prediction, its miss, and your explanation of the miss — not in the numbers themselves.

## Why this drill exists

Part 7 claimed that peak training memory is `M(b) = M₀ + m·b`: a floor that does not depend on the batch (weights, gradients, optimizer state, the runtime's own allocations) plus a slope that does (activations, and anything else that is per-sample). If that is true, two measurements determine the line and a third measurement tests it. If the third point is *off* the line, either the claim is wrong for your script or something in your script is holding memory you did not account for. Both outcomes are the deliverable; the second is the more common and the more useful. This is also the first real data point for Project 2's `memory_math.py` — you will reconcile its prediction against these numbers in week 3.

## Choosing a script (5 min)

Any of these works. Prefer one you have already run:

- The MLP or small transformer you trained in the dl_core roadmap, or anything under `playground/`.
- A Hugging Face `transformers` fine-tuning script for a small model (`distilgpt2`, `bert-base`) with a fixed sequence length — set `max_length` and `padding="max_length"` so the batch is the only thing that varies.
- **Inference** on the small model you serve locally (the pre-work), with a fixed prompt length and `torch.inference_mode()`. This is allowed by the syllabus and is *less* informative — part 7 §7.3 says why: no activations are kept, so the slope is only the KV cache and logits. If you pick inference, say so, and predict a much smaller slope.

Fix everything except the batch size: sequence length, dtype, model, optimizer, number of steps. Four steps is enough, and **measure the fourth**: the peak of any single step sits at the end of its forward pass (part 7 §7.3), but the *first* step lacks the optimizer state (Adam allocates its moments on the first `step()`), and the first two or three steps are also when the caching allocator grows its pools and the accelerator compiles or caches kernels. Reset the peak counter just before step 4 (or, on MPS/CPU, take your samples during step 4 only) so the warm-up is excluded; if you skip this, allocator warm-up will be the largest term in your "miss" and you will have measured the runtime rather than the model. Use batch sizes in a ratio like `b, 2b, 4b` so the line has spread; if `4b` OOMs, that is data — write it down and use `b, 2b, 3b`.

## Measuring peak (10 min)

Measure *peak*, not current, and measure the number the runtime's allocator reports, not what the OS shows — part 7 §7.5 explained that the two differ by the allocator cache. Wrap the step:

**CUDA.** `torch.cuda.reset_peak_memory_stats()` before the step; `torch.cuda.max_memory_allocated()` after it — "the maximum GPU memory occupied by tensors in bytes for a given device." Also record `torch.cuda.max_memory_reserved()`; the gap between them is the allocator's cache and fragmentation.

**Apple silicon (`mps`).** There is no peak counter. `torch.mps.current_allocated_memory()` "Returns the current GPU memory occupied by tensors in bytes" and excludes the allocator's cached pools; `torch.mps.driver_allocated_memory()` includes them. Sample `current_allocated_memory()` at the three moments part 7 §7.3 identified — **immediately after the forward pass returns the loss (before `backward()`)**, after `backward()`, after `optimizer.step()` — and take the maximum. The first sample is the peak in almost every training script, because every layer's saved tensors are alive and nothing has been freed yet. Record `driver_allocated_memory()` at the same points as your "reserved" column.

**CPU.** `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` gives the process's peak resident set size — in bytes on macOS, in kilobytes on Linux; check which. It is a high-water mark for the whole process, so measure each batch size in a **fresh process** (a subprocess per batch size, or three separate runs), otherwise the largest run contaminates the smaller ones. `/usr/bin/time -l python train.py --batch 8` on macOS prints "maximum resident set size" and is the zero-code alternative.

Whichever device: **one batch size per process**, so nothing carries over, and report the unit (MB vs MiB) you are using.

## The template (25 min)

Copy this into `peak_memory.md` and fill every line. Fill the *Prediction* section **before** running the third measurement — that is the point of the drill, and it is on your honour.

```markdown
# Peak memory vs batch size — <script name>

## Setup
- **Script and commit:** <path>@<sha>
- **Mode:** training / inference. Optimizer: <AdamW / SGD / none>. Precision: <fp32 / bf16 autocast>.
- **Model:** <name>, parameter count Ψ = <n>. Sequence length s = <n>, fixed.
- **Device and the number measured:** <cuda max_memory_allocated / mps current_allocated_memory sampled at 3 points / cpu ru_maxrss>. Unit: <MiB>.
- **Fresh process per run:** yes / no (if no, why the results are still valid)

## Two measurements
| batch b | peak allocated | peak reserved (if available) |
|---|---|---|
| b₁ = <n> | <MiB> | <MiB> |
| b₂ = <n> | <MiB> | <MiB> |

## The line
- slope  m  = (M₂ − M₁) / (b₂ − b₁) = <MiB per sample>
- floor  M₀ = M₁ − m·b₁ = <MiB>

## Sanity-check the floor against part 7 §7.2
- Expected model-state floor = Ψ × <bytes per parameter for your optimizer and precision> = <MiB>
- Measured M₀ − expected floor = <MiB>. What is the difference? (runtime context, the allocator's first block, cached kernels, input pipeline, the logits' temporary copy — name your suspect)

## Sanity-check the slope against part 7 §7.4
- If a transformer: L·s·h·(34 + 5as/h) bytes per sample at 16-bit (halve if you counted in fp32 and doubled elsewhere; double if activations are fp32) = <MiB per sample>
- If an MLP: sum over layers of (layer input width × bytes) per sample, plus the logits.
- Measured m / predicted m = <ratio>. Above 1: something else is per-sample. Below 1: something is not being saved (fused attention? inference mode? checkpointing on by default?).

## Prediction (write BEFORE the third run)
- b₃ = <n>.  Predicted peak = M₀ + m·b₃ = <MiB>.
- I expect the miss to be: <under / over / within 5 %>, because <one sentence>.

## Third measurement
| batch b₃ | predicted | measured | miss (measured − predicted) | miss / predicted |
|---|---|---|---|---|
| <n> | <MiB> | <MiB> | <MiB> | <%> |

## Explanation of the miss
<Which of these is it? Pick and defend with one more measurement if you can:>
- **Allocator rounding / caching** — reserved grew in steps larger than allocated. Compare the reserved column.
- **A non-linear temporary** — the fp32 copy of the logits for the loss, an attention score matrix in fp32, a `torch.cat` of the whole batch. Which op, and what is its size at b₃?
- **A per-sample thing outside the model** — the data loader's prefetch (`prefetch_factor × num_workers` batches resident), tokenization buffers, a metric accumulating outputs.
- **Accumulating history** — `total_loss += loss`, a list of outputs kept for an epoch metric (part 4 §4.6). Fix: `.item()` / `.detach()`. If this was it, re-run all three.
- **The line was right** — miss under 5 %. Then say what you now know about b_max for this script on this machine: `b_max = (budget − M₀) / m` = <n>.

## What this changes
- One sentence: the largest batch this script can run on this machine is <n>, and the lever I would pull first if I needed more is <fused attention / bf16 / selective recompute / checkpointing / smaller micro-batch with accumulation> because <part 8 §8.5>.
```

## Self-grading rubric

| | Pass | Strong |
|---|---|---|
| Measurement | Three batch sizes, peak (not current), fresh process each, unit stated | Both allocated and reserved recorded; on MPS the three sample points are all reported and the peak's location (after forward) confirmed |
| Prediction | Written before the third run, with M₀ and m shown | Slope and floor each sanity-checked against the lecture's byte counts, with the ratio reported |
| Miss | Reported as a number and a percentage | Explained with a named cause *and* one extra measurement that confirms it |
| Consequence | `b_max` for this machine stated | Plus the first lever you would pull and why, in the lecture's terms |

## If stuck

| Symptom | Likely cause | Where to look |
|---|---|---|
| Peak barely changes with batch | Inference mode, or activations not saved (no `backward()`), or the model is dominated by its floor at these batch sizes — try `8b` | Part 7 §7.3; use larger spread |
| Peak grows with *step*, not batch | Accumulating history | Part 4 §4.6 |
| Third point far *above* the line, reserved ≫ allocated | Allocator cache/fragmentation; a temporary spike | Part 1 §1.5; Part 7 §7.5 |
| MPS numbers look too small | You sampled after `backward()` — the saved tensors were already freed | Part 7 §7.3: sample *before* `backward()` |
| CPU peak identical across runs | Not a fresh process; `ru_maxrss` is a process-lifetime high-water mark | "Measuring peak" above |
| Floor ≫ 16 bytes × Ψ | fp32 everything with Adam is also 16Ψ; more than that is the runtime context, the input pipeline, or a second copy of the model (EMA? `deepcopy` for eval?) | Part 7 §7.2 |
