# Self-check — Lecture 3

Answer from memory, on paper, after finishing the notes and the reading. No answers are provided; every question is answered somewhere in `lecture/01`–`08` and you should be able to find and verify your own. If you cannot answer one without looking, that is the section to reread before the build.

A good target: 31 of 38 without notes.

## Tensors as a runtime

1. Name the five things a tensor consists of. Which of them does `permute` change, and which does `view` change?
2. `x = torch.arange(12).view(3, 4)`. Write its strides. Now write the strides and shape of `x.t()`, and say whether `x.t().view(-1)` succeeds. Why?
3. State the condition under which `view` can merge two adjacent dimensions without copying. Give one operation that breaks it.
4. `reshape` versus `view` versus `contiguous`: which can copy, which always copies, and which never does?
5. Which of these is a view: `x[0]`, `x[:, 1]`, `x[[0, 1]]`, `x[x > 0]`, `x.expand(4, 3)`, `x + 0`? For `expand`, what is the stride of the new dimension?
6. Write the broadcasting procedure in four lines. Then: `(B, T, D) + (B, D)` — what happens if `T ≠ B`, and what happens if `T == B`? Which is the dangerous case, and what one-token change makes the intent explicit?
7. Why does `torch.cuda.memory_allocated()` disagree with `nvidia-smi`? Which one does an OOM care about?

## Shapes and `einsum`

8. In self-attention, write the shapes of `q`, `scores`, `attn`, `out` before the head merge, in `B, H, T, S, D` letters. Which axis does each matmul reduce, and which axis does the softmax normalize over?
9. Why does `H` have to be moved in front of `T` before `q @ kᵀ`? Which operation does the moving, and what does it do to contiguity?
10. In the standard attention forward, exactly one `reshape` copies data. Which one, and why that one?
11. Write `q @ kᵀ` and `attn @ v` as two `einsum` strings with explicit `->`. What would go wrong if you wrote the same letter for the query length and the key length?
12. When does `einsum`'s contraction order matter, and what does PyTorch do about it if `opt_einsum` is not installed?
13. What is a shape suffix? Rewrite `h = emb[ids]; logits = h @ emb.T` with suffixes and check the second line's shape by reading.

## Devices and dtypes

14. Write the S-E-M layouts of `float16` and `bfloat16`. Which one can represent 70 000, and which has more significant digits? Which one needs loss scaling for training, and why?
15. `x_bf16 * 0.5` — what dtype? `x_bf16 * torch.tensor([0.5])` — what dtype, and why the difference? `x_int64 * 0.5` — what dtype?
16. State the two sentences of `.to()`'s contract that decide whether `y is x`. Give one bug caused by each.
17. What two conditions are both required for a host-to-device copy to overlap with compute?
18. What is the `meta` device, and what class of bugs can it find in milliseconds?

## Autograd

19. When is an operation *not* recorded in the backward graph? What does "the graph is recreated from scratch at every iteration" allow you to write inside a model?
20. Define *leaf*. Which tensors get a `.grad` after `backward()`? Why is `torch.zeros(3, requires_grad=True).to("mps")` a trap?
21. State the three verbs of `backward()`. Which one makes `zero_grad()` necessary, and what does a loss curve look like if you forget it?
22. Why does `x = x * y` have to save both `x` and `y`, and why does a `MulBackward0` node refuse to run if `y` was modified in place after the multiply? Name the mechanism.
23. Fill the table from memory: for `detach`, `no_grad`, `inference_mode`, `eval()` — what happens to the graph, does the output require grad, can the output re-enter autograd later?
24. Which of the four is correct for: a validation loop; an EMA teacher's output used as a target; the optimizer step; a serving endpoint; freezing an encoder (trick question — which mechanism is *none of the four*)?
25. Explain why `total_loss += loss` grows memory every step, using the words *grad_fn*, *saved tensors*, and *reference*. Write the one-token fix.

## Chain rule and finite differences

26. For `z = x * y` with incoming gradient `g`, write the two messages sent backward. For `z = exp(x)` and `z = tanh(x)`, what does the node need to have saved, and why is it cheaper than saving `x`?
27. Why is `+=` rather than `=` required when accumulating a gradient into a node during a single backward pass? Write a three-node expression that exposes the bug.
28. Why must the backward pass visit nodes in reverse topological order? What goes wrong with a plain recursive traversal on `(a + b) * (a - b)`?
29. State the forward-mode and reverse-mode costs for `f: Rⁿ → Rᵐ` in terms of `ops(f)`, `n`, `m`, and the constant `c`. Why does that make reverse mode the right choice for training, and what is its price?
30. Write the central-difference formula and its error order. Derive why the optimal step is `h ≈ ε^(1/3)`. Why must the test run in `float64`, and what happens to a `relu` test at exactly `x = 0`?

## Cost and memory

31. Why is backward ≈ 2× forward for a linear layer? Write the `6N` rule and say what each of the `2N` and `4N` parts is. Which layer's backward is only 1× forward?
32. Name the four residents of training memory and give the bytes per parameter under mixed-precision Adam, with the breakdown. Which resident moves with batch size?
33. Explain in two sentences why layer ℓ's input must stay alive until the backward pass reaches layer ℓ. At what instant is memory at its peak?
34. Write the per-layer activation formula. Identify the term that is quadratic in sequence length and name the three tensors it consists of. Compute the per-layer bytes for `s = 2048, b = 4, h = 1024, a = 16` (not the lecture's numbers), and say what fraction is the attention term.
35. `M(b) = M₀ + m·b`: what is in `M₀`, what is in `m`, and what would make the third batch-size measurement fall *off* the line?
36. State what full activation checkpointing saves (in terms of `L` and the per-layer formula) and what it costs (in FLOPs, as a fraction of the step). Why is `preserve_rng_state` not optional in a model with dropout?
37. What does selective recomputation recompute, why is that the right choice, and what is its overhead? How does a fused attention kernel achieve the same memory saving?
38. Order the levers for an OOM as the course recommends, and say why "smaller micro-batch" is last.
