# 4 · Autograd: how the graph is recorded, `backward()`, leaf tensors — and `detach` versus `no_grad`

*Lecture 3, part 4 of 8 · ~20 min reading*

This is the part of the runtime that has no analogue in the software you have shipped before, so it gets the most words. Everything is from the PyTorch "Autograd mechanics" note and the blitz tutorial, both of which you should read in full this week; the aim here is to arrange their sentences into a model you can run in your head.

## 4.1 What is recorded, and when

"Autograd is a reverse automatic differentiation system." The mechanism: "autograd records a graph recording all of the operations that created the data as you execute operations, giving you a directed acyclic graph whose leaves are the input tensors and roots are the output tensors." Internally "autograd represents this graph as a graph of `Function` objects (really expressions), which can be `apply()`ed to compute the result", and "the `.grad_fn` attribute of each `torch.Tensor` is an entry point into this graph."

So there is no separate graph object. When you write `z = x * y`, the multiply kernel runs (part 1's *execute* job) and, as a side effect, a `MulBackward0` node is created holding pointers to the nodes that produced `x` and `y`; `z.grad_fn` points at it. That is the *record* job, done lazily, per op, as a linked structure hanging off the outputs. Paszke et al. (2017): the framework "avoids ever materializing a 'forward graph', recording only what is necessary to differentiate the computation."

The recording has a gate: "During the forward pass, an operation is only recorded in the backward graph if at least one of its input tensors require grad." A `requires_grad=False` input contributes no node. Whole subgraphs that depend only on such tensors are never recorded at all — the cheapest possible way to not pay for autograd is to not ask for it.

And the recording is discarded after use: "An important thing to note is that the graph is recreated from scratch at every iteration, and this is exactly what allows for using arbitrary Python control flow statements, that can change the overall shape and size of the graph at every iteration. You don't have to encode all possible paths before you launch the training — what you run is what you differentiate." This is what "define-by-run" means, and the d2l chapter shows the consequence: "even if building the computational graph of a function required passing through a maze of Python control flow (e.g., conditionals, loops, and arbitrary function calls), we can still calculate the gradient". A Python `if` on a data value is legal inside a model. The graph you differentiate is the branch you took.

## 4.2 Leaf tensors and `requires_grad`

Two flags decide what happens to a tensor in the backward pass.

**`requires_grad`** "is a flag, defaulting to false *unless wrapped in a* `nn.Parameter`". It means "record operations on me." The note is emphatic that it is the primary control: "Setting `requires_grad` should be the main way you control which parts of the model are part of the gradient computation" — freezing a pretrained encoder is `for p in encoder.parameters(): p.requires_grad_(False)`, and the blitz tutorial's finetuning example does exactly that.

**Leaf** is a structural property: a tensor with no `grad_fn`. Parameters and inputs are leaves; anything computed from a tensor that requires grad is not. The rule that ties the two together: "During the backward pass (`.backward()`), only leaf tensors with `requires_grad=True` will have gradients accumulated into their `.grad` fields." Intermediate tensors participate in the backward pass — their `grad_fn` nodes are executed — but their `.grad` stays `None` unless you ask with `retain_grad()`. That is deliberate: the gradient with respect to an activation is a transient quantity used once and freed; the gradient with respect to a parameter is the thing the optimizer needs, so it is kept.

Leaf status is also why `x.to("mps")` or `x.float()` on a leaf produces a *non*-leaf — the conversion is a recorded op — and why `p = torch.zeros(3, requires_grad=True).to(device)` gives a tensor whose `.grad` will always be `None`, one of the oldest PyTorch pitfalls. Create it on the device, or call `.requires_grad_()` after the move.

## 4.3 What `backward()` does

The blitz tutorial gives the operational definition. When `.backward()` is called on the root, autograd "computes the gradients from each `.grad_fn`", "accumulates them in the respective tensor's `.grad` attribute", and, "using the chain rule, propagates all the way to the leaf tensors." Three verbs; take them separately.

**Computes from each `grad_fn`.** Each node knows one local rule: given the gradient of the loss with respect to *my output*, produce the gradient with respect to *each of my inputs*. `MulBackward0` for `z = x·y` returns `(g·y, g·x)`. The node has to have `y` and `x` available to do that — part 4.4 is about that fact. Part 6 derives these rules; here the point is that each node is independent and local.

**Propagates by the chain rule.** Autograd walks the DAG from root to leaves in reverse topological order, so that when a node runs, the gradient with respect to its output is complete. Karpathy's micrograd, which you build this week, does exactly this in a dozen lines: a depth-first topological sort of the graph, then "go one variable at a time and apply the chain rule to get its gradient" in reverse order.

**Accumulates.** This is the verb people miss. d2l: "Note that PyTorch does not automatically reset the gradient buffer when we record a new gradient. Instead, the new gradient is added to the already-stored gradient." Accumulation is the *correct* semantics when a leaf is used in several places — CS231n's rule that "if a variable branches out to different parts of the circuit, then the gradients that flow back to it will add" — and PyTorch applies it uniformly across calls too. Hence `optimizer.zero_grad()` (or `set_to_none=True`) at the top of every step. Forget it and every step trains on the sum of all previous gradients; the loss curve looks like a learning rate that is too high, which is why it belongs in Project 2's "three documented failures."

One more mechanical fact: after `backward()` runs, the graph's saved tensors are freed. A second `backward()` on the same graph raises unless you passed `retain_graph=True`. That freeing is the *point* — it is what returns the activation memory — so wanting `retain_graph=True` in a training loop is almost always a sign the loop is structured wrong.

## 4.4 Saved tensors, in-place ops, and the version counter

"Some operations need intermediary results to be saved during the forward pass in order to execute the backward pass." That sentence is the origin of every OOM in part 7, so pin it down now. `z = x * y` must keep `x` and `y`. `y = relu(x)` must keep the mask (or `x`). `y = x @ W` must keep `x` (for `∂/∂W`) and `W` (for `∂/∂x`). `softmax` keeps its *output*. The saved tensors are attached to the `grad_fn` node — you can see them: `z.grad_fn._saved_self`, `z.grad_fn._saved_other`. They live as long as the node lives, which is as long as anything downstream is referenced.

Saving by reference creates a hazard: an in-place op could overwrite a value a node saved. The note: "In-place operations can potentially overwrite values required to compute gradients." PyTorch detects this rather than silently computing a wrong gradient: "Every tensor keeps a version counter, that is incremented every time it is marked dirty in any operation. When a Function saves any tensors for backward, a version counter of their containing Tensor is saved as well." On access in backward, "if it is greater than the saved value an error is raised" — the `one of the variables needed for gradient computation has been modified by an inplace operation` message. The note's advice on in-place ops in general: "Unless you're operating under heavy memory pressure, you might never need to use them", because "Every in-place operation requires the implementation to rewrite the computational graph." So `x += 1` inside a model is not a free micro-optimization; it is a request to mutate a recorded graph, and it is only safe when nothing saved `x`.

## 4.5 Four ways to say "not this" — and which one you mean

There are four mechanisms that look alike and are not. The note builds the distinction; here it is as a decision table, then the reasons.

| You want to… | Use | What it does to the graph | Output `requires_grad`? | Can the output re-enter autograd later? |
|---|---|---|---|---|
| Cut *one tensor* out of the graph, keep using its value | `t.detach()` | Nothing to the existing graph; returns a new tensor with no `grad_fn` that shares storage | No | Yes — it is an ordinary tensor |
| Run a *block of code* without recording | `with torch.no_grad():` | No nodes are created inside the block | No ("even when the inputs have `requires_grad=True`") | Yes |
| Run inference as fast as possible | `with torch.inference_mode():` | No nodes, and "Skips additional autograd tracking overhead" | No | **No** — "tensors created in inference mode will not be able to be used in computations to be recorded by autograd after exiting inference mode" |
| Switch dropout / batch-norm to test behaviour | `model.eval()` | **Nothing.** "Evaluation mode is not a mechanism to locally disable gradient computation." | Unchanged | — |

**`detach`** is per-tensor and surgical. Its docstring: "Returns a new Tensor, detached from the current graph. The result will never require gradient." And the note beneath it: "Returned Tensor shares the same storage with the original one. In-place modifications on either of them will be seen, and may trigger errors in correctness checks." — the Tensor Views note lists `detach` among the view ops for the same reason, and the "correctness checks" are §4.4's version counter. Use it when you want a value to flow forward but not backward: a target computed from the model itself (a teacher output, a moving average, the "stop-gradient" in contrastive methods), a value you are logging, or the d2l case — "Sometimes, we wish to move some calculations outside of the recorded computational graph." The gradient stops *at that tensor*; everything upstream of it still gets gradients from any other path.

**`no_grad`** is per-region. "Computations in no-grad mode behave as if none of the inputs require grad" and "are never recorded in the backward graph even if there are inputs that have `require_grad=True`." Its docstring: "Disabling gradient calculation is useful for inference, when you are sure that you will not call `Tensor.backward()`." The note names the canonical non-inference use: no-grad is for when "you need to perform operations that should not be recorded by autograd, but you'd still like to use the outputs of these computations in grad mode later" — the optimizer's own update `p -= lr * p.grad` is done under no-grad, because updating a parameter is not a step of the model. Two footnotes from the docstring: it is thread-local, and "All factory functions, or functions that create a new Tensor and take a `requires_grad` kwarg, will NOT be affected by this mode" — `torch.zeros(3, requires_grad=True)` inside no-grad still requires grad.

**`inference_mode`** is "the extreme version of no-grad mode": "should be used when you are certain your operations will not interact with autograd", and "more restrictive, in that tensors created in this mode cannot be used in computations recorded by autograd." It buys speed by dropping the version counter and view-tracking bookkeeping from §4.4. The restriction is real: a KV cache allocated under `inference_mode` cannot later be fed into a training step. Serving code: yes. Evaluation inside a training script where the outputs feed a loss: no — use `no_grad`.

**`eval()`** is the odd one out and the most misused: "`module.eval()` (or equivalently `module.train(False)`) are completely orthogonal to no-grad mode and inference mode". It flips the behaviour of layers that behave differently at test time — dropout, batch norm. It records exactly as much graph as before. Validation code needs *both* `model.eval()` and `torch.no_grad()`; each alone is a bug (wrong numbers, or wasted memory).

**When each is correct** — the cases you will actually hit this term:

- Validation loop: `model.eval()` + `torch.no_grad()`. Then `model.train()` afterwards.
- Serving (Lecture 27 onward): `torch.inference_mode()`. Nothing leaves it into autograd.
- Logging a loss: `loss.item()` or `float(loss)` — see §4.6. `loss.detach()` if you need a tensor.
- A target that must not receive gradient (EMA teacher, bootstrapped value, stop-gradient): `.detach()` on that tensor, *not* `no_grad` around the whole block — you usually still want gradients into the *student* side of the same expression.
- Optimizer step, EMA update, weight clipping, manual parameter surgery: `torch.no_grad()`.
- Freezing part of a model: `requires_grad_(False)` on its parameters — not `detach`, not `no_grad`. The note is explicit that this is the primary mechanism, and it composes: gradients still flow *through* the frozen layers to earlier trainable ones, they are just not *accumulated* into the frozen parameters.

The wrong choices have signatures. `detach` where `requires_grad_(False)` was meant: layers before the detach point silently stop training. `no_grad` around a block that should have trained: the same, for the whole block. `inference_mode` in a training script: a `RuntimeError: Inference tensors cannot be saved for backward` far from the cause. `eval()` without `no_grad()`: validation memory equals training memory, and the first OOM of your career on a "read-only" pass.

## 4.6 The accumulating-history trap

The PyTorch FAQ's out-of-memory entry lists the mistake that combines everything above. Under "Don't accumulate history across your training loop":

> "By default, computations involving variables that require gradients will keep history. This means that you should avoid using such variables in computations which will live beyond your training loops, e.g., when tracking statistics. Instead, you should detach the variable or access its underlying data."

The example is a loop ending in `total_loss += loss`, of which the FAQ says: "Here, `total_loss` is accumulating history across your training loop, since `loss` is a differentiable variable with autograd history. You can fix this by writing `total_loss += float(loss)` instead."

Trace it with §4.1 and §4.4. `loss` has a `grad_fn`; that node holds saved tensors — the activations of the whole forward pass. `total_loss += loss` creates an `AddBackward0` node whose inputs include `loss`'s node. `total_loss` is a Python local that outlives the iteration. So every iteration's entire activation graph stays reachable from `total_loss`, and memory grows linearly with steps until the OOM. Nothing is "leaking"; the runtime is doing exactly what §4.1 said it does — keeping what it might need to differentiate — and you told it, by keeping a reference, that you might.

The FAQ's second rule is the same mechanism without arithmetic: "If you assign a Tensor or Variable to a local, Python will not deallocate until the local goes out of scope. You can free this reference by using `del x`." A list of per-batch `outputs` kept for a metric at the end of the epoch keeps every batch's graph alive; `outputs.append(out.detach())` keeps only the values.

## 4.7 What to take into part 5

Autograd records a node per op, only when an input requires grad, and throws the record away after each `backward()`. Leaves with `requires_grad` get `.grad` accumulated; everything else gets a `grad_fn`. Nodes save the tensors their local derivative needs, guarded by a version counter. `detach` cuts a tensor; `no_grad` silences a region; `inference_mode` silences it harder and permanently; `eval()` does none of that. Any reference to a tensor with history is a reference to its whole forward pass. Part 5 turns the shape rules of parts 1–2 into a debugging method, and part 6 goes inside the `grad_fn` nodes.

---

**Sources for this part** (exact sections in `READING.md`): PyTorch docs, "Autograd mechanics" note — How autograd encodes the history; Saved tensors; Locally disabling gradient computation (Setting `requires_grad`, Grad Modes, No-grad Mode, Inference Mode, Evaluation Mode); In-place operations with autograd · PyTorch tutorial, "A Gentle Introduction to torch.autograd" · PyTorch docs: `torch.no_grad`, `torch.autograd.grad_mode.inference_mode`, `torch.Tensor.detach`, Tensor Views note · PyTorch FAQ, "My model reports 'cuda runtime error(2): out of memory'" · d2l.ai §2.5 · Paszke et al. (2017) §1, §3–4 · Karpathy, micrograd `engine.py`.
