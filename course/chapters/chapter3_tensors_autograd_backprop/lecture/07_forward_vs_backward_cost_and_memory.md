# 7 · Cost and memory of the forward pass versus the backward pass — and why activations must be kept

*Lecture 3, part 7 of 8 · ~20 min reading*

Part 6 ended with the price of reverse mode: storage "growing (in the worst case) in proportion to the number of operations in the evaluated function." This part converts that sentence into bytes you can predict before you run anything. By the end you should be able to look at a model config and a batch shape and say, within a factor of two, how much memory a training step will take and which term dominates — which is precisely what the production drill asks you to do.

## 7.1 Compute: backward costs about twice forward

Take a linear layer `y = x @ W` with `x: (N, d_in)`, `W: (d_in, d_out)`. Forward is one matmul: `2·N·d_in·d_out` FLOPs (a multiply and an add per term). Backward needs two matmuls of the same size: `x̄ = ȳ @ Wᵀ` for the gradient flowing to the previous layer, and `W̄ = xᵀ @ ȳ` for the parameter gradient. Two matmuls against one: **backward ≈ 2 × forward**, for every layer that is dominated by matmuls, which in a transformer is all of them.

Kaplan et al.'s scaling-laws paper states it as the standard accounting: forward compute is `C_forward ≈ 2N` FLOPs per token for `N` non-embedding parameters, and "Accounting for the backwards pass (approximately twice the compute as the forwards pass)" gives "the estimated non-embedding compute as `C ≈ 6N` floating point operators per training token." Memorize `6N`: a training step on `B·T` tokens costs about `6·N·B·T` FLOPs, of which `2N` per token is forward and `4N` is backward. This is consistent with Baydin's bound in part 6 §6.4 — reverse mode at `c ∼ 2–3` times the forward — and it is the number Lecture 7 uses to estimate training time and cost.

Two refinements you will need later and can ignore this week. Attention adds a term proportional to `T` per token (the `Q·Kᵀ` and `attn·V` matmuls are `T × T`), so `6N` undercounts at long context; Kaplan's own forward count carries this as an extra `2·n_layer·n_ctx·d_model` term. And — this follows from the two-matmul argument, not from a source — a layer whose *input* needs no gradient (the first layer of a network, or anything downstream of only frozen tensors) has no `x̄` to compute, so its backward is one matmul, not two; freezing early layers saves compute, not just optimizer memory. Part 4 §4.1's recording gate is what makes this automatic.

## 7.2 Memory: the four residents

The ZeRO paper (Rajbhandari et al. 2020, §3 "Where Did All the Memory Go?") divides training memory into **model states** — parameters, gradients, optimizer states — and "the rest of the memory is consumed by activations, temporary buffers and fragmented memory which we call residual states." Hugging Face's memory-anatomy doc uses the same partition. Take each in turn, in bytes per parameter for the model states.

**Parameters.** 4 bytes in fp32. In mixed precision, ZeRO §3.1: "an fp16 copy of the parameters … with memory requirements of 2Ψ" plus "an fp32 copy of the parameters" in the optimizer state, 4Ψ. Hugging Face: "two copies of the weights are required — one in fp16 for the forward/backward pass and one in fp32 as the 'main copy' for stable weight updates. This equates to 6 bytes per parameter."

**Gradients.** One per parameter, same dtype as the training copy: 2Ψ in mixed precision (ZeRO), 4Ψ in fp32. The PyTorch FAQ's reminder for the fp32 case: "you will need at least twice the size of the weights, since you also need to store the gradients."

**Optimizer state.** Adam keeps two moments per parameter. ZeRO: "momentum and variance, with memory requirements of … 4Ψ, and 4Ψ bytes, respectively." Hugging Face: "That's an additional 8 bytes per parameter." SGD with momentum keeps one (4Ψ); plain SGD keeps none.

ZeRO's total for mixed-precision Adam: "2Ψ + 2Ψ + KΨ = 16Ψ bytes of memory requirement" with `K = 12`, and the example: "For a model such as GPT-2 with 1.5 Billion parameters, this leads to a memory requirement of at least 24 GB". **Sixteen bytes per parameter** is the number to keep. It is fixed by the model and the optimizer, and it does not move when you change the batch size — which is the first thing to notice in the production drill. In pure fp32 with Adam the figure is also 16Ψ (4 + 4 + 8); in fp32 with plain SGD, 8Ψ; with 8-bit Adam the optimizer term drops from 8 to 2.

**Activations** are the resident that does move with the batch, and they are the subject of the rest of this part.

## 7.3 Why every layer's activations must be kept until the backward pass

Part 6's table had a column "what the node must have saved." Here is the same fact stated by the sources and then made unavoidable.

The autograd note: "Some operations need intermediary results to be saved during the forward pass in order to execute the backward pass." The checkpoint docs, describing the default behaviour: "By default, tensors computed during the forward pass are kept alive until they are used in gradient computations in the backward pass." Chen et al. (2016), on why memory is linear in depth: "Most gradient operators will depend on the intermediate results of the forward pass, we still need O(n) memory for intermediate results to train a n layer convolutional network or a recurrent neural networks with a sequence of length n." Korthikanti et al. define the term precisely: "'activations' in this paper refers to any tensor that is created in the forward pass and is necessary for gradient computation during back-propagation. As a result, this excludes the main parameters of the model and optimizer state, but, for example, includes the mask used by the dropout operation."

Now the reason it is *unavoidable*, from the local rules. Layer `ℓ` computes `y_ℓ = x_ℓ @ W_ℓ`. Its parameter gradient is `W̄_ℓ = x_ℓᵀ @ ȳ_ℓ`. It needs `x_ℓ` — the *input* to layer `ℓ`, which is the output of layer `ℓ−1`. And it needs `ȳ_ℓ`, which only exists once every layer *after* `ℓ` has run its backward. So `x_ℓ` must survive from the moment layer `ℓ` runs forward until the moment the backward pass reaches layer `ℓ` — which is after the forward pass has finished all `L` layers and the backward pass has undone `L − ℓ` of them. The first layer's input is held the longest: the entire forward pass and almost the entire backward pass. At the peak — the end of the forward pass, the instant before `backward()` starts — **every layer's saved tensors are alive at once**. That is the peak the production drill measures, and it is the moment most OOMs happen.

The PyTorch memory-snapshot blog shows it as a picture: "it is easy to see the rise of memory in the forward pass and the fall during the backward pass as the gradients are computed." Sawtooth up through forward, sawtooth down through backward, with the optimizer state as a flat floor underneath — "the optimizer state in yellow is allocated after the first iteration, and is kept constant for the rest of the job." The height of the tooth is the activation memory; the floor is the 16Ψ.

The FAQ's other two OOM rules are the same mechanism in different clothes. "The amount of memory required to backpropagate through an RNN scales linearly with the length of the RNN input" — each timestep is a layer whose activations are kept. And a linear layer `nn.Linear(m, n)` "uses O(nm) memory" — but that is the weight; the *activation* cost of the same layer is `O(N·m)` for its input, proportional to the batch, which is why halving the batch helps and shrinking the layer often does not.

## 7.4 Counting activation bytes: the transformer layer

Korthikanti et al. (2022, §4.1) do the count for one transformer layer, in bytes, assuming 16-bit activations (`s` = sequence length, `b` = micro-batch, `h` = hidden width, `a` = heads). It is worth walking through once because every term is a saved tensor from part 6's last column:

- Attention block. "The linear projection stores its input activations with size 2sbh and the attention dropout requires a mask with size sbh." "Query (Q), Key (K), and Value (V) matrix multiplies: We only need to store their shared input with size 2sbh." "QKᵀ matrix multiply: It requires storage of both Q and K with total size 4sbh." "Softmax: Softmax output with size 2as²b is required for back-propagation." "Softmax dropout: Only a mask with size as²b is needed." "Attention over Values (V): We need to store the dropout output (2as²b) and the Values (2sbh)". Subtotal `11sbh + 5as²b`.
- MLP. "The two linear layers store their inputs with size 2sbh and 8sbh. The GeLU non-linearity also needs its input with size 8sbh for back-propagation. Finally, dropout stores its mask with size sbh. In total, MLP block requires 19sbh bytes of storage."
- Layer norms. "Each layer norm stores its input with size 2sbh and therefore in total, we will need 4sbh of storage."

Total, their Equation 1:

```
Activations memory per layer = s·b·h·(34 + 5·a·s/h)          [bytes, 16-bit activations]
```

Multiply by `L` layers for the peak. Read the structure, because it is the whole of part 8's motivation:

- Every term has the factor **`s·b`** — tokens in the micro-batch. Activation memory is *linear in batch size* and *linear in sequence length*, at fixed model. Double the batch, double the activations; the 16Ψ floor does not move.
- The `34·s·b·h` term is the "everything else": linear layers, GeLU, norms, dropout masks. It is linear in `h`.
- The `5·a·s²·b` term is attention's score matrices — softmax output, dropout mask, dropout output, each `(b, a, s, s)`. It is **quadratic in `s`** and does not shrink with `h`; Korthikanti: "the term 5as/h in Equation 5 is due to the attention operations after the width of the network is increased … i.e., QKᵀ matrix multiply, softmax, softmax dropout, and attention over V." This term is why long-context training runs out of memory before it runs out of compute, and why fused attention kernels that never materialize `scores_BHTS` are a memory technique first.
- Multiplied by **`L`**: linear in depth, because every layer keeps its own.

So training memory grows with **batch × sequence length × depth** — the syllabus bullet — and the formula tells you the constants.

## 7.5 A worked budget

GPT-2 small: `L = 12`, `h = 768`, `a = 12`, `Ψ ≈ 124 M`. Train with `s = 1024`, `b = 8`, mixed precision, Adam.

Model states: `16 × 124 M ≈ 2.0 GB`. Fixed.

Activations per layer: `s·b·h = 1024 × 8 × 768 = 6.29 M`; `5·a·s/h = 5 × 12 × 1024 / 768 = 80`; so `6.29 M × (34 + 80) ≈ 717 MB` per layer, `× 12 ≈ 8.6 GB`. Of that, the attention `s²` term is `80/114 ≈ 70 %` — at `s = 1024` the score matrices already dominate. Plus the logits `(b, s, V) = 8 × 1024 × 50257 × 2 bytes ≈ 0.8 GB` in 16-bit, and, while the loss is being computed, an fp32 copy of them — another 1.65 GB that exists only briefly.

Peak ≈ 2.0 + 8.6 + 0.8 ≈ **11.4 GB** with every layer's activations alive, spiking to ≈ 13 GB while the fp32 logits are also alive; either way activations are about three quarters of it. Halve `b` and you save ~4.3 GB; the model states still cost 2.0. Double `s` to 2048 and the `34` term doubles while the `80` term quadruples: per layer `12.6 M × (34 + 160) ≈ 2.4 GB`, `× 12 ≈ 29 GB` — 3.4× the memory for 2× the tokens. That asymmetry is the `s²`.

The linear-in-`b` structure is what makes the production drill's prediction possible. Peak memory at batch `b` is `M(b) = M₀ + m·b`: a floor `M₀` (model states, code, allocator cache, the framework's own context) plus a slope `m` (activation bytes per sample, ≈ `L·s·h·(34 + 5as/h)` at 16-bit, plus logits). Two measurements give you `M₀` and `m`; the third is a prediction. When the third point misses the line, the residual is information: a non-linear temporary (the fp32 logits spike scales with `b` too, but the allocator's rounding and caching do not), a data loader holding batches, or a `total_loss += loss` from part 4 §4.6.

Two units to keep straight when measuring. The formula counts *bytes of saved tensors*; the runtime reports *allocated* (tensors) and *reserved* (allocator cache). On CUDA, `torch.cuda.max_memory_allocated()` returns "the maximum GPU memory occupied by tensors in bytes for a given device" — "By default, this returns the peak allocated memory since the beginning of this program", resettable with `reset_peak_memory_stats()`. On Apple silicon, `torch.mps.current_allocated_memory()` "Returns the current GPU memory occupied by tensors in bytes" and "does not include cached allocations", while `torch.mps.driver_allocated_memory()` "includes cached allocations in MPSAllocator pools as well as allocations from MPS/MPSGraph frameworks." There is no MPS peak counter; the drill tells you how to sample around the peak.

## 7.6 What to take into part 8

Backward is ~2× forward in FLOPs; a training step is ~`6N` per token. Memory has a fixed floor of ~16 bytes per parameter under mixed-precision Adam and a moving part — activations — that must be kept from the forward pass to the backward pass because each layer's parameter gradient needs that layer's *input*. Peak is at the end of the forward pass with every layer's saved tensors alive. Per transformer layer that is `s·b·h·(34 + 5as/h)` bytes, linear in batch and depth, quadratic in sequence length. Part 8 is the one trade available when that number does not fit: do not keep the activations — recompute them.

---

**Sources for this part** (exact sections in `READING.md`): Kaplan et al., "Scaling Laws for Neural Language Models" (2020), §2.1 · Rajbhandari et al., "ZeRO" (SC 2020), §3, §3.1, §3.2 · Hugging Face Transformers docs, "GPU memory usage" · Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models" (2022), §4, §4.1 Eq. 1, §5 · Chen et al., "Training Deep Nets with Sublinear Memory Cost" (2016), §1 · PyTorch docs: Autograd mechanics (Saved tensors); `torch.utils.checkpoint`; FAQ (OOM section); `torch.cuda.max_memory_allocated`, `torch.mps.current_allocated_memory`, `torch.mps.driver_allocated_memory` · Shi & DeVito, "Understanding GPU Memory 1" (PyTorch blog, 2023).
