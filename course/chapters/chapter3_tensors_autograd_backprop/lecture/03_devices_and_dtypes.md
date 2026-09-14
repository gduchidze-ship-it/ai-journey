# 3 · Device placement, dtypes, and what `.to()` actually does

*Lecture 3, part 3 of 8 · ~15 min reading*

Part 1 said a tensor is metadata over a storage. Two pieces of that metadata decide where the bytes are and how wide each one is, and both are chosen by you, usually implicitly, usually wrong the first time. This part is short because the rules are short; it exists because the *semantics* of the one method that changes them, `.to()`, are the kind of thing people assume rather than read.

## 3.1 `torch.device`

"A `torch.device` is an object representing the device on which a `torch.Tensor` is or will be allocated." Its type is "most commonly `"cpu"` or `"cuda"`, but also potentially `"mps"`, `"xpu"`, `"xla"` or `"meta"`", plus an optional ordinal; "If the device ordinal is not present, this object will always represent the current device for the device type." On the machine this course assumes — an Apple M-series laptop — the accelerator device is `"mps"`, and the production drill this week measures memory on it.

Three rules cover almost everything:

**Every operand of a kernel must be on the same device.** The dispatcher from part 1 §1.5 picks the kernel by device *of the inputs*; there is no kernel for "one on CPU, one on GPU," so you get `RuntimeError: Expected all tensors to be on the same device`. The exception that confuses people is a **0-dimensional CPU tensor** or a Python scalar, which most ops accept alongside a GPU tensor — `x_cuda * 2.0` and `x_cuda * torch.tensor(2.0)` both work. `x_cuda * torch.tensor([2.0])` does not: one dimension is enough to be "a tensor on the wrong device."

**Placement is decided at creation and by explicit moves; nothing moves for you.** `torch.zeros(3)` is on CPU regardless of where everything else lives; pass `device=` at creation (`torch.zeros(3, device=x.device)`) rather than creating on CPU and moving. Module parameters move together with `model.to(device)`; tensors you create inside `forward` do not, which is the classic bug — a mask or positional table built with `torch.arange(T)` sits on CPU while the activations are on the accelerator.

**A host↔device copy is a synchronization point unless you say otherwise.** The `.to` docstring: "When `non_blocking`, tries to convert asynchronously with respect to the host if possible, e.g., converting a CPU Tensor with pinned memory to a CUDA Tensor." Two conditions in that sentence: `non_blocking=True` *and* the source is in pinned (page-locked) host memory. Without pinning the copy is synchronous no matter what you pass. A data loader with `pin_memory=True` followed by `.to(device, non_blocking=True)` is the standard way to overlap the next batch's transfer with the current step's compute; either half alone does nothing.

## 3.2 `torch.dtype` — width, layout, and what each buys

"A `torch.dtype` is an object that represents the data type of a `torch.Tensor`." The three floating-point types that matter this term, with the docs' bit layouts (sign–exponent–mantissa):

| dtype | bits | layout | Range | Precision | Use |
|---|---|---|---|---|---|
| `torch.float32` (`float`) | 32 | S-E-M 1-8-23, "as defined in IEEE 754" | ~1e±38 | ~7 decimal digits | default; master weights; optimizer state; anything you accumulate |
| `torch.float16` (`half`) | 16 | "S-E-M 1-5-10" | max 65 504 | ~3 decimal digits | inference on hardware without bf16; needs loss scaling for training |
| `torch.bfloat16` | 16 | "S-E-M 1-8-7", "sometimes referred to as Brain floating point" | same as fp32 | ~2 decimal digits | mixed-precision training and inference on modern accelerators |

Read the layouts, not the names. `bfloat16` has the **same eight exponent bits as fp32**, so anything representable in fp32 has a bf16 neighbour of the same magnitude — it will lose digits, never overflow to `inf`. `float16` spends those bits on mantissa instead: three more digits of precision, but a maximum of 65 504, which an attention score or a squared gradient exceeds without trying. That single difference is why bf16 training does not need loss scaling and fp16 training does; it is also the whole content of Lecture 5's mixed-precision section, so file it now. The integer and boolean types (`int64` is the default for integer literals and indices; `bool` for masks; `uint8`/`int8` for quantized weights) follow the usual C semantics.

Bytes per element is the number to carry into part 7: **4 for fp32, 2 for fp16/bf16, 8 for int64**. An index tensor of shape `(B, T)` in `int64` costs `8·B·T` bytes — four times the size of the same shape in bf16 — and it is almost always created in `int64` by default.

## 3.3 Type promotion — where mixed dtypes go

When two dtypes meet in an arithmetic op, the docs' rule is: "When the dtypes of inputs to an arithmetic operation (*add*, *sub*, *div*, *mul*) differ, we promote by finding the minimum dtype that satisfies the following rules", with the category order "complex > floating > integral > boolean", and one subtlety that saves you from a common surprise: "A floating point scalar operand has dtype `torch.get_default_dtype()` and an integral non-boolean scalar operand has dtype `torch.int64`", but "If a zero-dimension tensor operand has a higher category than dimensioned operands, we promote to a type with sufficient size and category to hold all zero-dim tensor operands of that category." Translated: multiplying a bf16 tensor by the Python float `0.125` keeps it bf16 (scalars do not upgrade *within* a category); multiplying an `int64` tensor by `0.5` makes it fp32 (a scalar *does* upgrade across categories). And a `float32` *tensor* of shape `(1,)` — one dimension, not zero — will promote your whole bf16 activation to fp32 silently, doubling its memory. This is the dtype version of the silent broadcast from part 1 §1.4: correct by the rules, wrong by intent, and invisible until you look at `.dtype` or at peak memory.

Matmul does not promote at all: `torch.mm(a_bf16, b_fp32)` raises `expected m1 and m2 to have the same dtype` (checked on 2.14). Cast explicitly.

## 3.4 `.to()` semantics — read the docstring once

"Performs Tensor dtype and/or device conversion. A `torch.dtype` and `torch.device` are inferred from the arguments of `self.to(*args, **kwargs)`." Then the two sentences that decide correctness:

> "If the `self` Tensor already has the correct `torch.dtype` and `torch.device`, then `self` is returned. Otherwise, the returned tensor is a copy of `self` with the desired `torch.dtype` and `torch.device`."

Consequences, each of which has produced a real bug:

1. **`.to()` may return the same object.** `y = x.to(x.device)` gives `y is x`; mutating `y` mutates `x`. If you wanted an independent copy, that is the `copy` flag: "When `copy` is set, a new Tensor is created even when the Tensor already matches the desired conversion." Or use `.clone()`. Do not rely on `.to()` to defensively copy.
2. **`.to()` may return a fresh copy, so it is not in-place.** `x.to("mps")` without assignment does nothing to `x`. `model.to(device)` *is* effectively in-place because `nn.Module.to` reassigns every parameter's data internally — the module method and the tensor method have different contracts, and mixing them up is the second-most-common device bug after §3.1's mask-on-CPU.
3. **A dtype conversion is a copy, and a copy is a kernel.** `x.to(torch.bfloat16)` allocates `x.numel()·2` bytes and runs an elementwise cast. In a hot loop that is real bandwidth; do it once at load time, or use autocast (Lecture 5) so the casts happen where the kernels want them.
4. **`.to()` preserves the autograd link.** The result is a differentiable function of the input — a device transfer or dtype cast has a gradient (the identity, cast back). It does not detach. If you want to break the graph *and* move, that is `.detach().to(...)`; part 4 §4.5 is about what `detach` does and when it is the right tool.
5. **`.to(other_tensor)`** is legal and means "same dtype and device as `other`" — the cleanest way to write "make this look like that."

The one-line habit that removes most of §3.1 and §3.4: **inside `forward`, create every new tensor with `device=x.device, dtype=x.dtype` from an input you were handed**, and never write a literal device string below the top-level script.

## 3.5 `meta` — a device with no bytes

One device type in the list above is not hardware. A tensor on `"meta"` has sizes, strides, dtype — everything from part 1 except the storage. Ops on meta tensors run the shape and dtype logic and allocate nothing. Two uses this term: instantiating a multi-billion-parameter model to inspect its parameter count and shapes on a laptop that cannot hold it (`with torch.device("meta"): model = Model(cfg)`), and checking that a forward pass *shape-checks* — every `matmul` lines up, every broadcast is legal — before spending a minute loading weights. Part 5 uses it as a debugging tool: it is the cheapest way to run a forward pass and get exactly the shape errors, and none of the numerical ones.

## 3.6 What to take into part 4

Devices are chosen at creation and by explicit moves; kernels need all inputs on one device; overlapped transfer needs pinning *and* `non_blocking`. bf16 keeps fp32's range with two digits of precision, fp16 keeps three digits and overflows at 65 504; index tensors are eight bytes each. Promotion is by category and dimensioned tensors win over scalars. `.to()` returns `self` when nothing changes and a copy otherwise, never mutates, never detaches. Part 4 opens the metadata part 1 skipped: the record autograd keeps on every tensor that requires a gradient.

---

**Sources for this part** (exact sections in `READING.md`): PyTorch docs: Tensor Attributes (`torch.dtype` table, type promotion rules, `torch.device`); `torch.Tensor.to` · Yang, "PyTorch internals" (2019), the dispatch paragraph · PyTorch docs: `torch.mps.current_allocated_memory`, `torch.mps.driver_allocated_memory` (for the production drill).
