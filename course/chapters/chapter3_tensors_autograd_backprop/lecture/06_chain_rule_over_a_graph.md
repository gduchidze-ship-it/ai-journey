# 6 · The chain rule over a computation graph

*Lecture 3, part 6 of 8 · ~20 min reading*

Part 4 treated each `grad_fn` node as a box that "knows its local rule." This part opens the box. The mathematics is the chain rule from first-year calculus; the engineering content is *which way you traverse the graph*, why that choice makes gradients cheap, and how to check that a rule you wrote is right. All three are what the build this week exercises.

## 6.1 A computation is a graph of local operations

CS231n's framing is the one to keep: "backpropagation, which is a way of computing gradients of expressions through recursive application of chain rule". Any expression a program evaluates — a loss as a function of parameters — is a DAG whose nodes are primitive ops (`+`, `×`, `exp`, `matmul`, `relu`, …) and whose edges carry intermediate values. Baydin et al.'s survey calls what autograd does "a non-standard interpretation of a given computer program by replacing the domain of the variables to incorporate derivative values and redefining the semantics of the operators to propagate derivatives per the chain rule of differential calculus." Same program, different interpretation of what each op means.

The key property of a node, from CS231n: "Every gate in a circuit diagram gets some inputs and can right away compute two things: 1. its output value and 2. the local gradient of its output with respect to its inputs." *Local* is the word. A multiply node with inputs `x, y` and output `z = xy` knows `∂z/∂x = y` and `∂z/∂y = x` without knowing anything about where `x` came from or where `z` goes. That ignorance is what makes the design compose: PyTorch has a few hundred such nodes and every model is a graph of them.

## 6.2 The chain rule as a message passed backwards

Write the loss as `L`, and for any intermediate `v` write `v̄ = ∂L/∂v` (Wikipedia's AD article: "the quantity of interest is the adjoint, denoted with a bar; it is a derivative of a chosen dependent variable with respect to a subexpression"). The chain rule says, for a node `z = f(x, y)`:

```
x̄ += z̄ · ∂z/∂x          ȳ += z̄ · ∂z/∂y
```

CS231n's description of the same line: once a gate "will eventually learn about the gradient of its output value on the final output of the entire circuit", the "Chain rule says that the gate should take that gradient and multiply it into every gradient" it computed locally. So each node receives one number (or tensor) from downstream — `z̄` — and emits one message per input: the incoming gradient times its local derivative.

The `+=` is not a typo; it is the third rule. CS231n: "if a variable branches out to different parts of the circuit, then the gradients that flow back to it will add" — the multivariable chain rule sums over all paths. This is the same accumulation part 4 §4.3 described at the level of `.grad`; here you see that it is required for correctness inside a single backward pass, not just across steps. In the build, the single most common bug is writing `x.grad = ...` where `x.grad += ...` was needed, and it only shows up when a `Value` is used twice — `a * a`, or `(a + b) * (a - b)`. Write that test first.

The local rules for the ops the build needs, written as the messages they send (`g` is the incoming `z̄`):

| `z =` | message to `x` | message to `y` | what the node must have saved |
|---|---|---|---|
| `x + y` | `g` | `g` | nothing |
| `x · y` | `g · y` | `g · x` | both inputs |
| `x ** n` (n constant) | `g · n · x**(n−1)` | — | `x` |
| `exp(x)` | `g · z` | — | its own output |
| `tanh(x)` | `g · (1 − z²)` | — | its own output |
| `relu(x)` | `g · (x > 0)` | — | the mask (or `x`) |
| `log(x)` | `g / x` | — | `x` |

The last column is the origin of part 7: a node that needs its inputs at backward time must keep them alive from forward to backward, and that is what activation memory *is*.

CS231n names the patterns so you can sanity-check a graph by eye: "The add gate always takes the gradient on its output and distributes it equally to all of its inputs"; "The max gate routes the gradient … to exactly one of its inputs" (`relu` is max with zero, hence the mask); and for multiply, "Its local gradients are the input values (except switched)" — the gradient into `x` is scaled by `y`. The last one is why "the scale of the data has an effect on the magnitude of the gradient": multiply the inputs by 1000 and the weight gradients grow by 1000, so normalization is not cosmetic. Karpathy's "Yes you should understand backprop" is a short catalogue of such consequences — a sigmoid gate shrinks the gradient "by one quarter (or more)" every time it is crossed, a ReLU that "never fires … will remain permanently dead," an RNN gradient "is always being multiplied by the same matrix" — and his conclusion is the reason the build exists: "The problem with Backpropagation is that it is a leaky abstraction."

## 6.3 Order matters: reverse topological

A node can only send its messages once `z̄` is complete — once every consumer of `z` has added its contribution. So the backward pass must visit nodes in an order where every node comes *after* all of its consumers: reverse topological order of the DAG. Karpathy's `engine.py` does exactly this — build a topological ordering by depth-first search from the output, seed the output's gradient with 1, and "go one variable at a time and apply the chain rule to get its gradient" in reverse order. PyTorch does the same with a scheduler over `grad_fn` nodes that counts pending dependencies. Either way the invariant is the same, and the build asks you to implement it yourself; the test for whether you got it right is again a `Value` used on two paths of different lengths, where a naive recursive traversal visits it before its gradient is complete.

## 6.4 Forward mode, reverse mode, and why backprop is the cheap one

There are two ways to run the chain rule over a graph, and the choice is the reason deep learning is feasible.

**Forward mode** carries derivatives *with respect to one input* along with the values, in the same direction as the computation. Baydin: "AD in forward accumulation mode is the conceptually most simple type." One pass gives you `∂(everything)/∂x_i` for a single `i`. For a gradient with respect to `n` parameters you need `n` passes.

**Reverse mode** carries derivatives *of one output* backwards. "AD in the reverse accumulation mode corresponds to a generalized backpropagation algorithm, in that it propagates derivatives backward from a given output." One pass gives you `∂L/∂(everything)`. For a scalar loss that is the whole gradient: "In the extreme case of `f: Rⁿ → R`, only one application of the reverse mode is sufficient to compute the full gradient."

Baydin states the cost bound, citing Griewank & Walther's *Evaluating Derivatives*: "for a function `f : Rⁿ → Rᵐ`, if we denote the operation count to evaluate the original function by `ops(f)`, the time it takes to calculate the `m × n` Jacobian by the forward mode is `n · c · ops(f)`, whereas the same computation can be done via reverse mode in `m · c · ops(f)`, where `c` is a constant guaranteed to be `c < 6` and typically `c ∼ [2, 3]`." For training, `m = 1` and `n` is in the billions. The gradient of a billion-parameter loss costs a small constant times the forward pass, independent of `n`. This is the fact Griewank & Walther call the *cheap gradient principle*, and it is the single reason gradient descent on neural networks is a practical algorithm at all. Baydin closes the loop: "the backpropagation algorithm is only a special case of AD: by applying reverse mode AD to an objective function evaluating a network's error as a function of its weights, we can readily compute the partial derivatives needed for performing weight updates."

Reverse mode's price is stated in the same section, and it is part 7's subject: "The advantages of reverse mode AD, however, come with the cost of increased storage requirements growing (in the worst case) in proportion to the number of operations in the evaluated function." Wikipedia's AD article says why: "Reverse accumulation requires the storage of the intermediate variables `wᵢ` as well as the instructions that produced them in a data structure known as a 'tape' or a Wengert list." Forward mode needs no such storage — it can discard each value as soon as the next op has consumed it — which is why forward mode is the right tool for *few inputs, many outputs* and reverse mode for *many inputs, one output*. Part 7 puts numbers on the price.

## 6.5 What AD is not: symbolic and numerical differentiation

Baydin's section 2 title is "What AD Is Not", and both alternatives matter for the build — one as a contrast, one as your test oracle.

**Symbolic differentiation** manipulates expressions to produce a derivative *expression*. It fails at scale because "Symbolic derivatives do not lend themselves to efficient runtime calculation of derivative values, as they can get exponentially larger than the expression whose derivative they represent" — expression swell — and because it cannot see through control flow. AD, by contrast, "can be applied to regular code with minimal change, allowing branching, loops, and recursion" — which is the dynamic-graph property of part 4 §4.1 seen from the mathematics side.

**Numerical differentiation** — finite differences — evaluates the function at nearby points: "the finite difference approximation of derivatives using values of the original function evaluated at some sample points." Forward difference: `∂f/∂xᵢ ≈ [f(x + h eᵢ) − f(x)] / h`, error `O(h)`. Central difference: `∂f/∂xᵢ ≈ [f(x + h eᵢ) − f(x − h eᵢ)] / (2h)`, error `O(h²)` because "the first-order errors cancel" (Wikipedia, "Numerical differentiation"). It is useless as a training algorithm — "the disadvantages of performing O(n) evaluations of f for a gradient in n dimensions" — but it is the *only* independent check on a gradient you derived by hand, because it uses no derivative rules at all. `torch.autograd.gradcheck` is exactly this: it will "Check gradients computed via small finite differences against analytical gradients". The build verifies your engine the same way.

Using it correctly requires understanding its two errors. Baydin: "Truncation error tends to zero as `h → 0`. However, as `h` is decreased, round-off error increases and becomes dominant". Truncation error is the `O(h²)` term from dropping the Taylor tail; round-off error is `O(ε/h)` from subtracting two nearly equal numbers in floating point and dividing by a tiny `h`. Balancing `h²` against `ε/h` gives the optimal step `h* ≈ ε^(1/3)` for the central difference (and `ε^(1/2)` for the forward difference), with a best-case relative error of order `ε^(2/3)`. In double precision, `ε ≈ 2.2 × 10⁻¹⁶`, so `h ≈ 6 × 10⁻⁶` and the best achievable agreement is about ten digits; `gradcheck`'s default `eps=1e-06` sits in that range, and the docs warn: "The default values are designed for `input` of double precision. This check will likely fail if `input` is of less precision, e.g., `FloatTensor`." In fp32, `ε ≈ 1.2 × 10⁻⁷`, `h* ≈ 5 × 10⁻³`, and the best case is around `2 × 10⁻⁵` relative — four or five digits, on a good day, with `h` chosen well; with `h = 1e-6` in fp32 you are deep in round-off and get nothing. Run your finite-difference tests in `float64`, use the central formula, use `h` near `1e-6`, and compare with a *relative* tolerance (`gradcheck` uses `allclose` with `atol=1e-05, rtol=0.001`). A test that uses `h = 1e-10` fails on a correct engine and passes on nothing; a test that uses `h = 0.1` passes on a wrong one.

There is a third caution specific to piecewise functions: the central difference straddles `x` by `±h`, so at a `relu` kink or a `max` switch it averages the two sides and disagrees with the one-sided analytical gradient. Test those ops away from their kinks, or expect a mismatch at exactly `x = 0` and understand why it is not a bug.

## 6.6 What to take into part 7

A computation is a DAG of local ops; each node sends `incoming × local derivative` to each input, and gradients at forks add. Nodes must run in reverse topological order. Reverse mode gives the full gradient of a scalar loss in a small constant times the forward cost, independent of parameter count — the cheap gradient principle — at the price of storing every intermediate the local rules need. Finite differences are the independent oracle: central formula, fp64, `h ≈ ε^(1/3)`, relative tolerance, away from kinks. Part 7 asks what the stored intermediates cost, in FLOPs and in bytes.

---

**Sources for this part** (exact sections in `READING.md`): CS231n, "Backpropagation, Intuitions" — Intuitive understanding; Patterns in backward flow; Gradients add up at forks · Baydin, Pearlmutter, Radul & Siskind, "Automatic Differentiation in Machine Learning: a Survey" (JMLR 2018), §2, §2.1, §2.2, §3.1, §3.2 · Griewank & Walther, *Evaluating Derivatives*, 2nd ed. (SIAM 2008) — the cheap gradient bound, as cited by Baydin · Wikipedia, "Automatic differentiation" (reverse accumulation, adjoint, Wengert list) and "Numerical differentiation" (central difference, truncation vs round-off) · FiniteDiff.jl docs, "Step Size Selection" (`h* = ε^(1/3)` for central differences) · PyTorch docs, `torch.autograd.gradcheck` · Karpathy, "Yes you should understand backprop" (2016) · Karpathy, micrograd `engine.py`.
