# Build — scalar autograd engine (2 h)

**Deliverable.** Two files, `value.py` and `gradcheck.py`, that pass `pytest test_value.py`, plus a short `NOTES.md` (five answers, see below). They land in your course repo under `src/autograd/` (or wherever your template puts library code) with the tests under `tests/`. This engine is the first third of **Project 2 — Autograd, training loop, memory calculator** (due end of week 3); next week's lecture adds the operator you will extend it with and the training loop that drives it.

**Rules.** From scratch, standard library only — `math` and nothing else. **No numpy, no torch**, not imported, not copied from. You may watch Karpathy's micrograd lecture *before* you start (it is in the reading list) but write the code with the video closed; the tests here are stricter than micrograd in three places (see §3) and copying its engine will not pass them. Type hints and docstrings are part of the deliverable.

**Time.** If §1–§3 take more than 75 minutes, stop, commit what passes, and write down where the time went. §4 (the finite-difference oracle) is the last 30 minutes and the most important part; do not skip it to polish §2.

## Conventions — read these first, the tests assume them

1. `data` and `grad` are Python `float`s. `grad` starts at `0.0` on every `Value`.
2. Every operation accepts a `Value` or a plain `int`/`float` on either side (`2 * x`, `x / 3`, `1 - x`). A plain number is wrapped in a fresh `Value` and becomes a parent of the result.
3. **Gradients accumulate within a pass.** A node's local rule adds (`+=`) into its inputs' `.grad`; it never assigns. A `Value` consumed on two paths receives the sum (part 6 §6.2, "gradients add up at forks").
4. **Leaves accumulate across passes; non-leaves do not.** A *leaf* is a `Value` with empty `_prev` — an input or a parameter. `backward()` first resets `.grad` to `0.0` on every **non-leaf** `Value` in the graph, then sets the *root's* grad to exactly `1.0` (assignment), then runs every node's rule exactly once, in an order where a node runs only after all of its consumers. Leaf grads are never touched by `backward()`, so two `backward()` calls without a `zero_grad()` between them exactly double every leaf grad — which is PyTorch's behaviour (part 4 §4.3) and the reason `optimizer.zero_grad()` exists. (PyTorch achieves the non-leaf half by not retaining intermediate grads at all — part 4 §4.2; here we keep them, for inspection, and reset them.) If you skip the reset, the *second* pass reads stale interior grads and the leaves come out wrong by more than 2× — `test_repeated_backward_doubles_leaf_grads_only` is the test.
5. `x ** k` takes a constant `int`/`float` exponent only. `Value ** Value` is out of scope.
6. `relu` has derivative `0.0` at exactly `0.0`. `log` is natural and raises `ValueError` on `data <= 0`.
7. `zero_grad()` on any `Value` zeroes it and everything reachable through `_prev`.
8. `backward()` must work on a graph 3 000 nodes deep. Python's default recursion limit is 1 000; a recursive traversal is therefore a bug, not a style choice.
9. `gradcheck.numerical_grad` uses the **central** difference and builds fresh `Value`s for every function evaluation. `check_gradients` compares with `|a − n| ≤ atol + rtol·|n|` per coordinate and returns `(ok, max_abs_diff, analytical, numerical)`.

## What to implement

The skeleton `value.py` has every signature and docstring; every body is `raise NotImplementedError`. Suggested order, with the lecture part that defines each (≈ 100 min of coding, then 15 min for the reflection; setup should be nothing — two files and `pytest`):

### §1 Arithmetic and the graph (part 4 §4.1, part 6 §6.2) — 25 min
- `__add__`, `__mul__`, `__pow__`, then `__neg__`, `__sub__`, `__truediv__` *in terms of the first three* — no new closures for the derived ops. Then the four reflected operands.
- Each primitive returns a new `Value` whose `_prev` is its inputs and whose `_backward` closure implements one row of the table in part 6 §6.2. Ask, for each op: what does the closure need to have captured? That is the "saved tensors" column, and it is the origin of part 7.

### §2 Non-linearities (part 6 §6.2) — 15 min
- `exp`, `log`, `tanh`, `relu`. Two of these can compute their derivative from their *own output* rather than their input. Notice which, and notice that it is cheaper.

### §3 `backward()` and `zero_grad()` (part 6 §6.3) — 30 min
- `backward()` must (a) visit each node once, (b) respect reverse topological order, (c) not recurse, (d) reset non-leaf grads first. Before you write it, design the traversal on paper for `(a + b) * (a − b)`, where `a` and `b` are each consumed twice, and for a 3 000-node chain. The three places this is stricter than micrograd: the depth requirement (convention 8), the leaf/non-leaf rule (convention 4 — a stale `grad` on the root or on any interior node must be overwritten, a leaf's must not), and `zero_grad()` on the graph.
- The tests that catch the classic bugs are named for them: `test_gradients_accumulate_when_a_value_is_used_twice`, `test_diamond_graph_needs_correct_order`, `test_each_node_backward_runs_exactly_once`, `test_deep_chain_does_not_hit_recursion_limit`. Run them first and often.

### §4 The finite-difference oracle (part 6 §6.5) — 30 min
- `gradcheck.numerical_grad(f, xs, h)` — central difference, one coordinate at a time, fresh `Value`s per evaluation.
- `gradcheck.analytical_grad(f, xs)` — fresh leaves, forward, `backward()`, collect `.grad`.
- `gradcheck.check_gradients(f, xs, h, atol, rtol)` — the comparison.
- Then **use it** on something the tests do not cover: write a five-parameter expression of your own with every op in it, check it, and then deliberately break one derivative in `value.py` (drop a factor, flip a sign) and confirm the check catches it. A gradient checker you have never seen fail is a gradient checker you cannot trust.

## Then look at what you built (do not skip — 15 min of the 2 h)

Write five short answers into `NOTES.md` next to your code. A sentence or two each:

1. For each of `exp`, `tanh`, `relu`, `mul`: what did the `_backward` closure capture, and how many floats is that? Now scale it: if each `Value` were a `(B, T, D)` tensor with `B·T·D = 10⁷` elements in bf16, how many bytes does one `mul` node hold until `backward()` runs? (Part 7 §7.3.)
2. Run `check_gradients` on `relu` at exactly `x = 0.0` and at `x = 1e-7`. Report both results and explain them using part 6 §6.5's last paragraph.
3. Sweep `h` over `{1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12}` for `f(x) = exp(3.7·x)` at `x = 0.9`. Tabulate `max_abs_diff`. Where is the minimum, and which of the two error terms dominates on each side of it?
4. Time `backward()` on the 3 000-node chain and on a 30 000-node chain. Is it linear? Where does the time go — the traversal or the closures?
5. How would this engine have to change to become PyTorch? Name three differences from parts 1 and 4 (think: what `data` is, what a closure must save, what happens to the graph after `backward()`), and say which of them is the reason `retain_graph=True` exists.

## Self-grading rubric

| | Pass | Strong |
|---|---|---|
| Tests | All of `test_value.py` green | Plus your own tests for a graph where a leaf feeds three consumers of different depths, and for `zero_grad()` on a diamond |
| Conventions | Followed | Docstrings on every method state what the closure captured |
| `backward()` | Correct on all shipped graphs, non-recursive | Also handles a `Value` that appears in its own `_prev` chain twice (`x * x * x`) with a single visit, and you can explain the invariant in one sentence |
| Oracle | `check_gradients` passes on the engine and fails on a broken one | The `h` sweep from question 3 is in `NOTES.md` with the minimum found and explained |
| `NOTES.md` | Five answers present | Each answer names the lecture section it verifies or contradicts |

## If stuck

| Symptom | Likely cause | Where to look |
|---|---|---|
| `a * a` gives grad `3.0` instead of `6.0` | Closure assigns (`=`) instead of accumulating (`+=`) into an input's grad | Convention 3; part 6 §6.2 "gradients add up at forks" |
| Diamond graph off, simple chains fine | A node ran before all its consumers finished; traversal is not reverse-topological | Convention 4; part 6 §6.3 |
| Some leaf grads doubled on a wide graph | A node's `_backward` ran twice — your traversal visits nodes more than once | Convention 4 — visited set |
| `RecursionError` on the deep chain | Recursive DFS | Convention 8; use an explicit stack |
| `2 * x` raises `TypeError` | Missing `__rmul__` (or `__radd__`, `__rsub__`, `__rtruediv__`) | Convention 2 |
| `1 - x` gives `x - 1` | `__rsub__` has the operands the wrong way round | Convention 2 |
| Root grad wrong after a second `backward()` | You `+=` into the root; convention 4 says assign `1.0` | Convention 4 |
| Second `backward()` gives leaves *more* than 2× | Interior grads were not reset before propagating; stale values were re-used | Convention 4; part 4 §4.2 on why PyTorch has no such grads to reset |
| FD test fails for `log` or `pow_half` only | Your derivative uses the output where it should use the input, or vice versa | Part 6 §6.2 table, last column |
| FD tests fail by ~1e-3 everywhere | `h` too small or too large, or a forward difference | Convention 9; part 6 §6.5 |
| `check_gradients` says `ok` on a gradient you know is wrong | Tolerance test uses `<` on the wrong side, or compares against the analytical instead of the numerical value | Convention 9 |
| FD on `relu` fails at `x = 0` | Not a bug; the central difference straddles the kink | Part 6 §6.5, last paragraph — test away from 0 |

Sources for every derivative rule and for the finite-difference analysis are under topic 6 of `../../READING.md`.
