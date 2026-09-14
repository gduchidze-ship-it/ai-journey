# 1 · A tensor library is a small runtime

*Lecture 3, part 1 of 8 · ~20 min reading*

You already know what a runtime is. A JVM takes bytecode, decides where to execute it, manages memory, and keeps enough bookkeeping to unwind a stack when something throws. A tensor library does the same three things, and this lecture is organized around them: it **records** a graph of what you asked for, it **executes** each operation on a device, and, when you ask, it **walks the graph backwards** to produce gradients. Everything in the next seven parts — shapes, devices, autograd, memory — is a consequence of one of those three jobs. Part 1 is about the data structure that all three jobs share.

## 1.1 What a tensor actually is

The PyTorch paper (Paszke et al. 2019, §4.3) describes the autograd side; the tensor side is best laid out in Edward Yang's "PyTorch internals" (2019), which is the reading for this part. His opening description is the mental model to keep: "We can think of a tensor as consisting of some data, and then some metadata describing the size of the tensor, the type of the elements in contains (dtype), what device the tensor lives on (CPU memory? CUDA memory?)".

The part most engineers skip is the **stride**. Yang: "Strides are actually one of the distinctive features of PyTorch, so it's worth discussing them a little more … to find out where any element for a tensor lives, I multiply each index with the respective stride for that dimension, and sum them all together." The docs define it in one sentence: "Stride is the jump necessary to go from one element to the next one in the specified dimension." So a tensor is really five things:

```
storage   : a flat, contiguous buffer of N elements of one dtype, on one device
sizes     : the logical shape, e.g. (2, 3)
strides   : how many storage elements to skip per step in each dimension, e.g. (3, 1)
offset    : where in the storage element [0, 0, …] lives (Yang: "most of the time it's zero")
dtype, device
```

and `t[i, j]` reads `storage[offset + i·stride[0] + j·stride[1]]`. A fresh `torch.zeros(2, 3)` has strides `(3, 1)` — row-major, C order, and the docs call this **contiguous**. Yang again: "Storage defines the dtype and physical size of the tensor, while each tensor records the sizes, strides and offset, defining the logical interpretation of the physical memory." One buffer, many interpretations. That separation is why the next three operations are cheap, and why one of them sometimes is not.

## 1.2 `view`, `reshape`, `permute` — three ways to reinterpret one buffer

The Tensor Views note in the PyTorch docs states the rule that governs all three: "View tensor shares the same underlying data with its base tensor. Supporting `View` avoids explicit data copy, thus allows us to do fast and memory efficient reshaping, slicing and element-wise operations." And the consequence you must never forget: "Since views share underlying data with its base tensor, if you edit the data in the view, it will be reflected in the base tensor as well."

**`permute` (and `transpose`, `.T`, `movedim`)** reorders the *strides*, touching no data. The docs: permute "Returns a view of the tensor with its dimensions permuted." Take `x = torch.arange(6).view(2, 3)` with strides `(3, 1)`; `x.permute(1, 0)` has shape `(3, 2)` and strides `(1, 3)`. Same storage, same offset. Reading `[i, j]` of the permuted tensor now jumps 1 per row and 3 per column — which is exactly reading the original's `[j, i]`. No bytes moved. But the Tensor Views note warns: "Taking a view of contiguous tensor could potentially produce a non-contiguous tensor," and names `transpose` as "a common example." Strides `(1, 3)` are not the strides a fresh `(3, 2)` tensor would have; `is_contiguous()` is now `False`.

**`view`** changes sizes and strides together, without moving data, and *refuses* if it cannot. Its docstring: "Returns a new tensor with the same data as the `self` tensor but of a different `shape`. … For a tensor to be viewed, the new view size must be compatible with its original size and stride". The compatibility condition is that each new dimension either sits inside one old dimension or spans old dimensions `d, d+1, …, d+k` that satisfy `stride[i] = stride[i+1] * size[i+1]` — that is, the dimensions being merged must already be laid out next to each other in memory. "Otherwise, it will not be possible to view `self` tensor as `shape` without copying it (e.g., via `contiguous`)." This is why `x.transpose(0, 1).view(-1)` raises the error every PyTorch user has met once: the transposed strides `(1, 3)` cannot be flattened by arithmetic alone.

**`reshape`** is `view` with a fallback. Its own docstring: "When possible, the returned tensor will be a view of `input`. Otherwise, it will be a copy. Contiguous inputs and inputs with compatible strides can be reshaped without copying, but you should not depend on the copying vs. viewing behavior." The `view` docstring gives the advice from the other side: "When it is unclear whether a `view` can be performed, it is advisable to use `reshape`, which returns a view if the shapes are compatible, and copies (equivalent to calling `contiguous`) otherwise." And the Tensor Views note states the rule for code you write: "`reshape`, `reshape_as` and `flatten` can return either a view or new tensor, user code shouldn't rely on whether it's view or not."

So the decision is:

| You want | Use | Cost | Aliasing |
|---|---|---|---|
| Reorder dimensions | `permute` / `transpose` | O(1), metadata only | Always a view |
| Merge/split dimensions and you *require* zero-copy | `view` | O(1) or an error | Always a view |
| Merge/split dimensions and you do not care | `reshape` | O(1) or O(n) copy | Maybe a view |
| A guaranteed fresh row-major buffer | `.contiguous()` | O(n) copy, or free | "returns **itself** if input tensor is already contiguous" |

The engineering point: `reshape` hides a possible O(n) copy behind an O(1)-looking call. On a `(B, H, T, D)` attention tensor that is `B·H·T·D` elements copied per call, per layer, per step. When part 2 gets to attention you will see `permute` followed by `reshape` — and now you know the `reshape` after a `permute` is the one that *copies*, because the permute made the tensor non-contiguous.

## 1.3 Indexing — what is a view and what is a copy

Basic indexing follows the same storage/strides logic. `x[1]` picks a row: same storage, offset advanced by `stride[0]`, one fewer dimension — a view. `x[:, 1]` is a column: offset advanced by `stride[1]`, strides `(3,)` — a non-contiguous view. `x[::2]` doubles a stride. The Tensor Views note lists `select`, `narrow`, `squeeze`, `unsqueeze`, `expand`, `diagonal`, `unbind`, `split`, `chunk` and `as_strided` among the view operations; every one of them is a new `(sizes, strides, offset)` triple over the same storage.

Two things are **not** views and always allocate. **Advanced indexing** with a tensor or list of indices — `x[[0, 2]]`, `x[mask]` — must gather elements that have no regular stride pattern, so it copies. And any **arithmetic** result is a new tensor. The trap is mixing the two intuitions. `row = x[0]; row += 1` mutates `x` (a view, then an in-place op on it). `rows = x[[0]]; rows += 1` does *not* touch `x` (a copy, then an in-place op on the copy). Python's own `x[[0]] += 1` *does* write back, but only because augmented assignment on a subscript expands to `x.__setitem__(idx, x.__getitem__(idx) + 1)` — a gather, an add, and a scatter rather than one in-place op (checked on 2.14: `x` is updated). Rule: if you intend to mutate a slice of a big tensor in place, use basic indexing and, if you are unsure whether you hold a view, compare `data_ptr()` or check `._base is not None`.

`expand` deserves a sentence because part 1.4 depends on it: it produces a view with a **stride of 0** in the expanded dimension. `torch.tensor([1., 2., 3.]).expand(4, 3)` has shape `(4, 3)` and strides `(0, 1)`. Four rows, all reading the same three storage elements. That is how broadcasting is implemented without allocating.

## 1.4 Broadcasting — the two rules, and the one that bites

The Broadcasting semantics note gives the definition. Two tensors are broadcastable if "Each tensor has at least one dimension" and "When iterating over the dimension sizes, starting at the trailing dimension, the dimension sizes must either be equal, one of them is 1, or one of them does not exist." The result shape follows: "If the number of dimensions of x and y are not equal, prepend 1 to the dimensions of the tensor with fewer dimensions to make them equal length. Then, for each dimension size, the resulting dimension size is the max of the sizes of x and y along that dimension."

Write it as a procedure and it is four lines:

```
1. Right-align the two shapes.
2. Pad the shorter one on the LEFT with 1s.
3. For each position: equal → keep; one is 1 → take the other; else → error.
4. The 1-sized dimension becomes a stride-0 view (expand); no data is copied.
```

Worked: `(B, T, D) + (D,)` → pad to `(1, 1, D)` → result `(B, T, D)`; the bias is read `B·T` times from one row. `(B, T, D) + (T, D)` → pad to `(1, T, D)` → fine, a per-position bias. `(B, T, D) + (B, D)` → pad to `(1, B, D)` → position −2 compares `T` with `B`: an error unless `T == B` — and *if `T == B` it silently succeeds and adds the wrong thing*. This is the failure mode part 5 is about: broadcasting has no idea what your dimensions mean; it only compares integers from the right. Sasha Rush's "Tensor Considered Harmful" calls this **broadcasting by alignment**: "the rules of broadcasting do not have the correct semantics" — they are positional, not semantic. The fix is not to memorize more rules; it is to make the intended alignment explicit with `unsqueeze` / `[:, None]` so the 1 is *where you put it*, not where right-alignment happened to land it.

The in-place rule matters for training loops: "in-place operations do not allow the in-place tensor to change shape as a result of the broadcast." `a.add_(b)` requires the result shape to equal `a.shape`; `(D,).add_((B, D))` raises even though `(D,) + (B, D)` would work out of place. Good — that is a shape error surfacing early rather than a silent allocation.

## 1.5 Execute: the dispatcher and the allocator

The second job of the runtime is running the operation somewhere. Yang describes the two-level dispatch: "When you call `torch.mm`, two dispatches happen … The first dispatch is based on the device type and layout of a tensor … The second dispatch is a dispatch on the dtype in question." So `torch.mm(a, b)` with `a` on CUDA in bf16 resolves to a different kernel than the same call on CPU in fp32, and the Python you wrote is the same. Part 3 is about the consequences: which combinations exist, and what `.to()` really does.

Under the kernel sits an allocator. The PyTorch paper, §5.3: "calls to the CUDA memory management functions (`cudaMalloc` and `cudaFree`) slow down the execution quite dramatically by blocking the CPU thread for long periods of time", so "PyTorch implements a custom allocator which incrementally builds up a cache of CUDA memory and reassigns it to later allocations", and "it rounds up allocations to multiples of 512 bytes to avoid fragmentation issues." Two facts for part 7 and the production drill follow. First, memory you `del` is not returned to the driver; it is kept in PyTorch's cache, which is why `nvidia-smi` and `torch.cuda.memory_allocated()` disagree — the docs say the latter "is likely less than the amount shown in `nvidia-smi` since some unused memory can be held by the caching allocator". Second, "allocated" and "reserved" are different numbers, and OOM can happen while reserved memory still exists, through fragmentation.

## 1.6 Record and walk back: the third job, in one paragraph

Paszke et al. (2017) describe the recording job precisely: "An eager framework runs tensor computations as it encounters them; it avoids ever materializing a 'forward graph', recording only what is necessary to differentiate the computation." Each result tensor carries a pointer to the `Function` that made it and to the `Function`s that made its inputs — "every intermediate result records only the subset of the computation graph that was relevant to their computation." Yang locates the extra metadata: an `AutogradMeta` struct "which is needed for performing autograd when a user calls `loss.backward()`." Part 4 opens that struct. For now the point is architectural: the graph is not a separate object you build; it is a linked list of small records hanging off tensors, created as a side effect of executing ops, and freed when the last tensor referencing it dies. That is why holding a reference to one tensor can keep an entire iteration's worth of activations alive — the FAQ trap in part 4 §4.6 — and why part 7's memory story is really a story about which tensors the recorded graph refuses to let go of.

## 1.7 What to take into part 2

A tensor is a *view onto storage*, described by sizes, strides and an offset. `permute` edits strides; `view` re-derives strides or refuses; `reshape` copies when it must; basic indexing and `expand` are views, advanced indexing and arithmetic are not. Broadcasting is right-aligned integer comparison implemented with stride-0 views, and it has no semantics. The library records what it executes, executes through a two-level dispatch, and walks the record backwards on demand. Part 2 uses all of this to read a `(B, H, T, D)` tensor without guessing.

---

**Sources for this part** (exact sections in `READING.md`): Yang, "PyTorch internals" (2019) · PyTorch docs: Tensor Views note; `torch.Tensor.view`, `torch.Tensor.permute`, `torch.Tensor.stride`, `torch.Tensor.is_contiguous`; Broadcasting semantics note · Paszke et al., "Automatic differentiation in PyTorch" (NIPS-W 2017), §3 · Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library" (NeurIPS 2019), §4.3, §5.3 · Rush, "Tensor Considered Harmful" (2019), Trap 2.
