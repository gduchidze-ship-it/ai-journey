# 2 · Reading `(B, H, T, D)` without guessing, and `einsum`

*Lecture 3, part 2 of 8 · ~20 min reading*

Part 1 established that a shape is just a tuple of integers with no meaning attached. This part is about supplying the meaning yourself, consistently enough that you can read any tensor in a transformer and say what every axis is *before* you print it. The tool for reasoning is a naming convention; the tool for computing is `einsum`, which makes the convention executable.

## 2.1 The four letters

A transformer's attention block has a fixed cast of dimensions. The syllabus writes them `(B, H, T, D)`; the literature and every codebase spell them slightly differently, so learn the mapping once:

| Letter here | Meaning | Vaswani et al. 2017 | PyTorch `scaled_dot_product_attention` docs | Shazeer's suffixes |
|---|---|---|---|---|
| `B` | batch — independent sequences processed together | (implicit) | `N`: "Batch size" | `B` |
| `H` | heads — parallel attention functions | `h` ("we employ h=8 parallel attention layers, or heads") | `H` / `Hq`: "Number of heads of key and value" / "of query" | `H` |
| `T` | time / sequence position — tokens | (positions in the sequence) | `L`: "Target sequence length", `S`: "Source sequence length" | `L` |
| `D` | per-head feature width | `d_k = d_v = d_model / h = 64` | `E`: "Embedding dimension of the query and key", `Ev` for value | `D` (often the *model* width; `K` or `Dh` for per-head) |

Two things in that table are load-bearing. First, **`T` splits into two letters the moment you write attention**: the queries have their own length and the keys/values have theirs. In self-attention they coincide, which is exactly why people conflate them and then cannot read a cross-attention or KV-cache shape. The PyTorch docs keep them apart: query is `(N, ..., Hq, L, E)`, key is `(N, ..., H, S, E)`, value is `(N, ..., H, S, Ev)`, the mask must be broadcastable to `(N, ..., Hq, L, S)`, and the output is `(N, ..., Hq, L, Ev)`. Read that last line slowly: the output has the *query's* length and the *value's* width. Second, **`D` is per-head, not the model width**. Vaswani: "we found it beneficial to linearly project the queries, keys and values h times with different, learned linear projections to d_k, d_k and d_v dimensions, respectively," with `d_k = d_v = d_model/h`. The model width `d_model` is `H × D`. When you see `(B, T, 768)` and `(B, 12, T, 64)` in the same function, one is before the head split and one is after, and `768 = 12 × 64` is the check.

## 2.2 Reading a shape: the four questions

When a tensor lands in front of you, ask these in order, and refuse to guess:

1. **Which axis is the batch?** It is the one nothing in the layer mixes across. If two tensors of shape `(B, T, D)` are added, the batch survives untouched — a token in sequence 3 never sees sequence 5. If you cannot identify an axis that the operation treats as fully independent, the tensor probably has no batch axis (a weight matrix, for example).
2. **Which axis is being reduced?** Every contraction — matmul, sum, mean, softmax — names an axis that disappears (or, for softmax, is normalized over). In `Q @ Kᵀ` the reduced axis is `D`; in `softmax(scores, dim=-1)` the normalized axis is the key length `S`; in `attn @ V` the reduced axis is again `S`. If you know what is reduced, you know the output shape without running anything.
3. **Which axes are merely carried?** Batch and heads are carried through attention — they are "batch dimensions" for the matmul, in the sense the `torch.matmul` docs use: leading dimensions broadcast, the last two are the matrix. That is why `Q` is `(B, H, T, D)` and not `(B, T, H, D)`: `matmul` operates on the trailing two axes, so `T` and `D` must be last, and `H` has to be moved in front of them. The `permute` in every attention implementation exists for this one reason.
4. **What is the width of the last axis, and what does it multiply?** `D = 64` multiplies `Wᵀ` of shape `(64, ·)`. `768` multiplies `(768, ·)`. A mismatch here is the shape error you actually get; the previous three questions are how you find *which* upstream reshape produced it.

Run the four questions over the standard attention forward and you can write every shape without executing a line:

```
x        : (B, T, D_model)                       # tokens, model width
q,k,v    : (B, T, D_model)  → each is x @ W       # Wq, Wk, Wv are (D_model, D_model)
split    : (B, T, H, D)     → view, D_model = H·D  (a view: T and D_model are adjacent)
permute  : (B, H, T, D)     → move H in front of the matrix dims (a view; not contiguous)
scores   : (B, H, T, S)     → q @ kᵀ, reduce D; here S = T
attn     : (B, H, T, S)     → softmax over S (the last axis), scaled by 1/√D
out      : (B, H, T, D)     → attn @ v, reduce S
merge    : (B, T, H, D)     → permute back (a view; not contiguous)
         : (B, T, D_model)  → reshape: THIS one copies, because the permute left it non-contiguous
```

The two annotations about contiguity come straight from part 1 §1.2. In real code the last line is written `.transpose(1, 2).contiguous().view(B, T, -1)` — the `.contiguous()` is the copy, made explicit; or `.reshape(B, T, -1)`, the same copy, made implicit.

## 2.3 Make the convention part of the code: shape suffixes

Noam Shazeer's "Shape Suffixes — Good Coding Style" (2024) turns the reading discipline into a naming rule: "Designate a system of single-letter names for logical dimensions, e.g. `B` for batch size, `L` for sequence length, etc., and document it somewhere in your file/project/codebase", and then "When known, the name of a tensor should end in a dimension-suffix composed of those letters, e.g. `input_token_id_BL` for a two-dimensional tensor with batch and length dimensions." His justification is the whole of this part in one sentence: "for a tensor, nothing is more informative than how many dimensions it has, and what those dimensions represent". His own example:

```python
hidden_BLD = params.embedding_VD[input_token_id_BL]
logits_BLV = torch.matmul(hidden_BLD, params.embedding_VD.T)
```

Read the second line without knowing anything else: `BLD @ (VD)ᵀ = BLD @ DV → BLV`. The contraction is over `D`; the output is logits over the vocabulary for every token in every sequence. The suffix *is* the shape check, done by the reader at code-review time instead of by the interpreter at 3 a.m. Rush's diagnosis of why this is needed is "Trap 3: Access by Comments" — in ordinary tensor code the meaning of `dim=2` lives in a comment three functions away, and "this code will run fine for whatever value dim is given." Suffixes move the information from the comment into the identifier, where a mismatch is visible.

The course convention from here on: **`B, H, T, S, D` as in §2.1, with `Dm` for the model width and `V` for vocabulary**, and suffixes on any tensor whose shape is not obvious from the line that made it. It costs nothing and it is how the build in the independent work will be graded for readability.

## 2.4 `einsum`: the shape convention you can execute

`torch.einsum` "Sums the product of the elements of the input `operands` along dimensions specified using a notation based on the Einstein summation convention." The docstring's rules are the whole language:

- "The `equation` string specifies the subscripts (letters in `[a-zA-Z]`) for each dimension of the input `operands` in the same order as the dimensions, separating subscripts for each operand by a comma (','), e.g. `'ij,jk'` specify subscripts for two 2D operands."
- "The subscripts that appear exactly once in the `equation` will be part of the output, sorted in increasing alphabetical order." — so a letter that appears in the inputs but not the output is **summed over**.
- "Optionally, the output subscripts can be explicitly defined by adding an arrow ('->') at the end of the equation followed by the subscripts for the output." Always do this; implicit alphabetical output order is a trap.
- "Ellipsis ('...') can be used in place of subscripts to broadcast the dimensions covered by the ellipsis. Each input operand may contain at most one ellipsis".
- "The dimensions labeled with the same subscript must be broadcastable, that is, their size must either match or be 1." Note that this is *stricter* than general broadcasting — a mismatch that is not 1 is an error, which is what you want.
- "If a subscript is repeated for the same input operand … the operand will be replaced by its diagonal along these dimensions." (`'ii->i'` is the diagonal; `'ii->'` is the trace.)

Now write attention in it, with the letters from §2.1:

```python
scores_BHTS = torch.einsum('bhtd,bhsd->bhts', q_BHTD, k_BHSD) / math.sqrt(D)
attn_BHTS   = scores_BHTS.softmax(dim=-1)
out_BHTD    = torch.einsum('bhts,bhsd->bhtd', attn_BHTS, v_BHSD)
```

Three things to notice. The reduced axis is *named* — `d` in the first line, `s` in the second — which answers §2.2 question 2 by construction. The carried axes `b, h` are just letters that appear on both sides; there is no `permute` to get `H` "in front of the matrix dims," because `einsum` has no notion of "the last two axes are the matrix." And the transposition of `K` is gone: `kᵀ` was only ever a way to line up `d` for `matmul`. Rush's proposal 3 — "Broadcast should be by name matching" — and Rogozhnikov's einops paper make the same argument for the same reason: "Both input and output are described in the operation definition: tensor dimensionality and expected order of axes. This makes einops-based code more declarative and self-documenting." Einops (`rearrange('b t (h d) -> b h t d', h=H)`) is the same idea applied to reshapes; the paper's warning about the un-annotated alternative is worth quoting because it is the failure part 5 will debug: "reshape, a common operation in the DL code, easily breaks the tensor structure because a whole tensor is treated as a sequence and no connection is assumed between axes in input and output."

## 2.5 What `einsum` costs

Readable is not free, and you should know the two costs.

**Contraction order.** With two operands `einsum` is one matmul (after any needed permute), and the cost is the same as `matmul`. With three or more, the order of pairwise contractions can change the FLOP count by orders of magnitude — `(A·B)·C` versus `A·(B·C)`. The docstring: "This function will automatically speed up computation and/or consume less memory by optimizing contraction order. This optimization occurs when there are at least three inputs, since the order does not matter otherwise. … If opt-einsum is not available, the default order is to contract from left to right." So: install `opt_einsum`, or contract pairwise by hand in the order you have reasoned about. Do not write a four-operand `einsum` and assume the library will find the good order without the package.

**Kernel choice.** `matmul` on `(B, H, T, D)` dispatches to a batched GEMM. In the two-operand case `einsum` reduces to the same GEMM after rearranging operands (you can watch it do so with the profiler), so you pay the same matmul plus possibly a copy to make operands contiguous. That copy is the `reshape`-after-`permute` cost from part 1 again, now hidden inside a library call. Measure before assuming either is faster; in practice they are close, and fused attention kernels (`scaled_dot_product_attention`, FlashAttention) beat both by never materializing `scores_BHTS` at all — which, as part 7 will show, is a *memory* win before it is a speed win, because `scores` is the `5·a·s²·b` term in the activation formula.

## 2.6 What to take into part 3

Every axis has a name; `B` is what no layer mixes, `H` is carried, `T` and `S` are distinct the moment you write attention, `D` is per-head and `H·D = Dm`. Suffix your variables with the letters. Write contractions with `einsum` and name the summed axis explicitly; add `->`. Know that `permute` + `reshape` costs a copy and that three-operand `einsum` needs `opt_einsum`. Part 3 leaves the shape alone and asks where the bytes live and how wide each one is.

---

**Sources for this part** (exact sections in `READING.md`): Vaswani et al., "Attention Is All You Need" (2017), §3.2.2 · PyTorch docs: `torch.nn.functional.scaled_dot_product_attention` (shape table); `torch.einsum` · Shazeer, "Shape Suffixes — Good Coding Style" (2024) · Rush, "Tensor Considered Harmful" (2019), Traps 1–3, Proposals 1–3 · Rogozhnikov, "Einops: Clear and Reliable Tensor Manipulations with Einstein-like Notation" (ICLR 2022), abstract and §1.
