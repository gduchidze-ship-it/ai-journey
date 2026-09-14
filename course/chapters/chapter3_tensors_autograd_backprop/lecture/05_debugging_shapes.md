# 5 · Debugging shape errors without printing shapes first

*Lecture 3, part 5 of 8 · ~15 min reading*

The reflex when a shape error appears is to sprinkle `print(x.shape)` through the forward pass, run it again, and read the numbers. It works, eventually, and it teaches you nothing, so the next error takes just as long. This part replaces the reflex with a method that starts on paper and ends with one targeted assertion. It is short and it is the most immediately useful thing in the lecture.

## 5.1 Why printing first is the slow path

Three reasons, in increasing order of importance.

**The print tells you what *is*, not what *should be*.** A shape of `(32, 8, 128, 64)` is only informative if you already know it should have been `(32, 128, 8, 64)`. If you have not written down the expected shape, the print gives you a number to stare at and no way to know whether it is wrong.

**The error is usually not where the exception is.** A `matmul` complaining that `128` does not match `8` is the *consumer* of a bad tensor; the bug is in the `permute` or `view` two lines — or two modules — upstream that produced it. Printing at the crash site shows you a correctly-shaped-for-the-wrong-reason tensor and sends you backwards one print at a time.

**The worst shape errors do not raise.** Part 1 §1.4: `(B, T, D) + (B, D)` when `T == B` succeeds and adds the wrong thing. Part 2 §2.4 quoted the einops paper on `reshape`: it "easily breaks the tensor structure because a whole tensor is treated as a sequence and no connection is assumed between axes in input and output" — `x.view(B, H, T, D)` on a tensor laid out as `(B, T, H·D)` produces a tensor of exactly the right shape whose contents are scrambled across heads and positions. The loss goes down a little, the model is quietly broken, and no print of `.shape` will ever show it. Rogozhnikov: "These mistakes are elusive as no tools or tests can detect them." The method below is designed for this class first, because the raising class is the easy one.

## 5.2 The method: predict, then compare once

**Step 1 — write the expected shapes before reading any output.** Take the traceback's failing line and the function it lives in. On paper (or in a comment block), using the letters from part 2, write the shape of every tensor from the function's inputs to the failing line, using only the *definitions* of the ops: `matmul` reduces the last axis of the left against the second-to-last of the right and carries the rest; `view` keeps the element count and requires adjacency; `permute` reorders; `softmax(dim=k)` preserves shape; broadcasting right-aligns. This is part 2 §2.2's four questions applied line by line. It takes two to five minutes and it is the whole debugging step — everything after is confirmation.

**Step 2 — find the first line where you cannot derive the shape.** That is a line where you do not know the semantics of the op precisely, or where the op has an argument (`dim=`, a `-1` in a `view`, an implicit `einsum` output) that depends on the input layout. That line is the suspect, regardless of where the exception was raised. Common suspects: a `view(-1, D)` that assumed contiguity, a `dim=1` that was correct before a `permute` moved the axis, a `transpose(1, 2)` whose author thought of `(B, T, H, D)` while the tensor was `(B, H, T, D)`, an `einsum` without `->`.

**Step 3 — one assertion, at the suspect, in the letters.** Not a print. Write the expected shape as a tuple of named symbols and assert it:

```python
B, T, Dm = x_BTDm.shape
q_BTHD = q.view(B, T, H, Dm // H)             # suspect: was q laid out (B, T, Dm)?
assert q_BTHD.shape == (B, T, H, Dm // H), q_BTHD.shape
```

If the assertion passes, the suspect is cleared and your paper derivation was wrong somewhere *before* it — go back to the last shape you were sure of. If it fails, the message prints the actual shape next to the expected one, which is the first moment a number is useful: you now have a prediction to compare against. Note that the assertion is written in terms of `B, T, H, Dm`, never `32, 128, 8, 64`: a check with literals passes for one batch size and fails at inference with batch 1, which is a second bug you have just prevented.

**Step 4 — for the silent class, assert on *contents*, not shape.** When the shape is right and the model is wrong, the test is a round trip or an invariant. After a split-and-permute, the merge must reproduce the original: `assert torch.equal(merge(split(x)), x)`. After a softmax over the key axis, `attn.sum(-1)` must be ones with shape `(B, H, T)` — and if you accidentally normalized over `T` instead of `S`, the sum over the last axis is *not* ones, which is the assertion catching a bug that shapes cannot. After a causal mask, `attn[..., i, j] == 0` for `j > i`. These are two-line tests and they go into `tests/`, not into a debugging session.

## 5.3 Tools that make step 1 cheaper

**Shape suffixes** (part 2 §2.3) are step 1 done in advance by whoever wrote the code. `scores_BHTS = q_BHTD @ k_BHSD.transpose(-1, -2)` can be shape-checked by reading: `BHTD @ BHDS → BHTS`. Shazeer's point was exactly that the suffix is "nothing is more informative" — it is the derivation, stored in the identifier.

**`einsum` with an explicit `->`** turns a permute-transpose-matmul sequence whose shapes you would have to derive into a single line where the input and output shapes are written down: `'bhtd,bhsd->bhts'`. It also refuses silently-wrong alignments that `matmul` would accept, because "The dimensions labeled with the same subscript must be broadcastable, that is, their size must either match or be 1" — a `d` of 64 against a `d` of 128 is an error, where `matmul` would have raised too, but a `t` against an `s` that happen to be equal is *not confused* because you gave them different letters. Einops' `rearrange('b t (h d) -> b h t d', h=H)` does the same for reshapes: the split is spelled out and the library checks it.

**The `meta` device** (part 3 §3.5) runs the whole forward pass with no storage — every shape rule, no arithmetic, no memory, in milliseconds. `with torch.device("meta"): out = model(torch.empty(B, T, dtype=torch.long))` reproduces every *raising* shape error in a model that would take a minute to load. It is the right first move when the error is deep in a large model.

**`torch.Size` arithmetic in the assertion**, not literals — covered above, repeated because it is the step people skip under time pressure.

**Read the error message's numbers as a hint about *which axis*, not which value.** `mat1 and mat2 shapes cannot be multiplied (4096x64 and 128x512)`: `4096 = 32 × 128` tells you a `(B, T)` pair was already flattened into the row axis, and `64` versus `128` says the left operand is per-head width while the right expects the model width — the merge of heads back into `Dm` was skipped, or done as a `view` that silently interleaved (§5.1). The message carried the diagnosis; the print would have carried the same two numbers with less context.

## 5.4 A worked case

The exception (PyTorch 2.14 wording):

```
RuntimeError: view size is not compatible with input tensor's size and stride
(at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
```

at `out.view(B, T, Dm)`, the head-merge at the end of an attention block. Do not print, and do not take the message's advice yet — `reshape` will make the error go away, and you want to know what you would be paying for. `view` has two distinct failures. When the element counts differ it says `shape '[…]' is invalid for input of size N`; that is an arithmetic bug in your shape. This message is the *other* one: the counts agree, but the dimensions you asked it to merge are not laid out next to each other — part 1 §1.2's condition `stride[i] = stride[i+1]·size[i+1]` fails for the pair being merged. So something upstream produced a non-contiguous `out`, and the merge of `H` and `D` into `Dm` is straddling a stride jump.

Step 1, on paper: `attn @ v` gives `out_BHTD`, contiguous, strides `(H·T·D, T·D, D, 1)`. To merge `H` and `D` they have to be adjacent, so there must be a `transpose(1, 2)` giving `(B, T, H, D)` with strides `(H·T·D, D, T·D, 1)` — and now the pair to be merged, `H` then `D`, has strides `(T·D, 1)` while adjacency would need `H`'s stride to equal `D`'s stride × `D`'s size `= 1 × D`. It does not. That is the whole diagnosis, done without running anything; the transpose is the suspect. Step 3: `assert out_BTHD.is_contiguous()` fails — confirmed. The fix is a decision, not a search: `.contiguous().view(B, T, Dm)` or `.reshape(B, T, Dm)`, both of which copy `B·T·Dm` elements (part 2 §2.2 marked this as *the* copy in attention); or an `einsum`/einops formulation that never transposes; or a fused attention call that returns `(B, T, Dm)` directly. Then a step-4 test that stays: `merge(split(x))` equals `x`, so the merge is the inverse of the split and not a head-scramble.

The signatures, then. `view` has two messages: a *count* mismatch (`is invalid for input of size N`) means your shape arithmetic is wrong; a *stride* mismatch (`not compatible with input tensor's size and stride`) means a `transpose`/`permute`/slice upstream and a copy in your future. `matmul` fails on the *reduced* axis — `mat1 and mat2 shapes cannot be multiplied (4096x64 and 128x512)` — so its two numbers name the axis that was mis-merged (`64` is per-head width, `128` is not `Dm` either; someone merged the wrong pair). Broadcasting fails at a *right-aligned* position — `The size of tensor a (4) must match the size of tensor b (5) at non-singleton dimension 1` — so the reported dimension, counted after right-alignment, names the axis that a missing `unsqueeze` would have fixed. Each error type points at a different upstream operation. Learning those three signatures is most of the skill.

## 5.5 What to take into part 6

Derive first; the derivation is the debugging. Assert the suspect in named symbols, once. For the silent class — a `view` across non-adjacent axes, a softmax over the wrong axis, a broadcast that aligned by coincidence — assert on contents with a round trip or an invariant, and keep it as a test. Use suffixes and explicit `einsum` so the next person's step 1 is already written; use the `meta` device to reproduce raising errors in seconds. Part 6 stops treating `grad_fn` nodes as black boxes and derives what they compute.

---

**Sources for this part** (exact sections in `READING.md`): Rogozhnikov, "Einops" (ICLR 2022), §1 on `reshape` and undetectable mistakes · Shazeer, "Shape Suffixes" (2024) · Rush, "Tensor Considered Harmful" (2019), Traps 2–3 · PyTorch docs: `torch.Tensor.view` (the compatibility condition), `torch.einsum`, Broadcasting semantics, `torch.device` (`meta`).
