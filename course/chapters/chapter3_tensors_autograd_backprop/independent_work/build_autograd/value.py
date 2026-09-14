"""value.py — a scalar autograd engine, from scratch.

Course: Applied AI Engineering, week 2 (Lecture 3 build).
Rules: standard library only (`math`). No numpy, no torch — not imported, not
       copied from. The point is that after this you will know exactly what
       `loss.backward()` does, one node at a time.

The design mirrors what Lecture 3 part 4 described for PyTorch, collapsed to
scalars:

  * a Value holds one float (`data`) and the gradient of the final output with
    respect to it (`grad`), which starts at 0.0;
  * every operation returns a NEW Value that records which Values produced it
    (`_prev`) and a closure (`_backward`) that knows this node's LOCAL rule:
    "given grad on my output, add the right amount to each input's grad";
  * `backward()` on the root orders the graph so every node runs after all of
    its consumers, seeds the root's grad with 1.0, and runs each node's rule.

Conventions (also in ASSIGNMENT.md — the tests assume them):
  * `data` and `grad` are Python floats; `grad` starts at 0.0 on every Value.
  * Operations accept a Value or a plain int/float on either side.
  * Gradients ACCUMULATE: a node's rule must `+=` into its inputs' grads (a
    Value used on two paths receives the sum).
  * A LEAF is a Value with no `_prev` (an input or a parameter). `backward()`
    first resets `.grad` to 0.0 on every NON-leaf Value in the graph, then
    sets the root's own grad to 1.0 (assignment, not +=), then propagates.
    Leaf grads are never reset by `backward()` — so two `backward()` calls
    without `zero_grad()` exactly double every leaf grad, as in PyTorch, whose
    intermediate grads are not retained at all (lecture part 4 §4.2).
  * `x ** k` supports only a constant int/float exponent `k`.
  * `relu` has derivative 0 at exactly 0.0.
  * `log` is the natural log and may raise ValueError for data <= 0.

Every method body below is a stub. Replace `raise NotImplementedError` with
your implementation. Keep the names and signatures — test_value.py imports them.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, Union

Number = Union[int, float]


class Value:
    """A scalar with a gradient and a record of how it was computed."""

    __slots__ = ("data", "grad", "_prev", "_op", "_backward", "label")

    def __init__(self, data: Number, _children: Iterable["Value"] = (), _op: str = "", label: str = ""):
        self.data: float = float(data)
        self.grad: float = 0.0
        # Internal bookkeeping for the graph. `_prev` is the set of Values this
        # one was computed from; `_op` is a short name for the operation (for
        # debugging and drawing); `_backward` is the local chain-rule step.
        self._prev: set[Value] = set(_children)
        self._op: str = _op
        self._backward: Callable[[], None] = lambda: None
        self.label: str = label

    # ------------------------------------------------------------------ #
    # §1  Arithmetic — each returns a new Value with a `_backward` closure
    # ------------------------------------------------------------------ #

    def __add__(self, other: Union["Value", Number]) -> "Value":
        """self + other. Local rule: the add gate distributes the gradient."""
        raise NotImplementedError

    def __mul__(self, other: Union["Value", Number]) -> "Value":
        """self * other. Local rule: each input gets grad times the OTHER input."""
        raise NotImplementedError

    def __pow__(self, k: Number) -> "Value":
        """self ** k for a constant int/float k (not a Value)."""
        raise NotImplementedError

    def __neg__(self) -> "Value":
        """-self. Implement via __mul__; no new closure needed."""
        raise NotImplementedError

    def __sub__(self, other: Union["Value", Number]) -> "Value":
        """self - other. Implement via __add__ and __neg__."""
        raise NotImplementedError

    def __truediv__(self, other: Union["Value", Number]) -> "Value":
        """self / other. Implement via __mul__ and __pow__."""
        raise NotImplementedError

    # Reflected operands so that `2 * x`, `1 + x`, `1 - x`, `1 / x` work.
    def __radd__(self, other: Number) -> "Value":
        raise NotImplementedError

    def __rmul__(self, other: Number) -> "Value":
        raise NotImplementedError

    def __rsub__(self, other: Number) -> "Value":
        raise NotImplementedError

    def __rtruediv__(self, other: Number) -> "Value":
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # §2  Non-linearities — the ones a small MLP and a softmax need
    # ------------------------------------------------------------------ #

    def exp(self) -> "Value":
        """e ** self. Hint from lecture part 6: what does the node need to save?"""
        raise NotImplementedError

    def log(self) -> "Value":
        """Natural log. Raise ValueError if self.data <= 0."""
        raise NotImplementedError

    def tanh(self) -> "Value":
        raise NotImplementedError

    def relu(self) -> "Value":
        """max(0, self). Derivative is 0 at exactly 0.0."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # §3  Backward
    # ------------------------------------------------------------------ #

    def backward(self) -> None:
        """Populate `.grad` on every Value in this one's graph.

        Requirements (lecture part 6 §6.3; conventions in ASSIGNMENT.md):
          * first, every NON-leaf Value reachable from here has its `.grad`
            reset to 0.0 (leaves — Values with empty `_prev` — are left alone,
            so their grads accumulate across calls);
          * then this node's grad is set to 1.0;
          * then every node's `_backward` runs exactly once, and a node runs
            only after ALL of its consumers have run, so that its own `.grad`
            is complete when it propagates — reverse topological order over
            `_prev`.
        Must handle graphs where a Value is used more than once and graphs
        several thousand nodes deep (a Python recursion limit is a bug here).
        """
        raise NotImplementedError

    def zero_grad(self) -> None:
        """Set `.grad = 0.0` on this Value and every Value in its graph."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Helpers you may use freely
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g}{', ' + self.label if self.label else ''})"

    def __hash__(self) -> int:  # Values live in sets; identity semantics
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other


def graph_nodes(root: Value) -> list[Value]:
    """Every Value reachable from `root` through `_prev`, in any order.

    Provided. Iterative, so it works on deep graphs. Used by the tests to
    check that `zero_grad` reached everything.
    """
    seen: set[Value] = set()
    stack = [root]
    out: list[Value] = []
    while stack:
        v = stack.pop()
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
        stack.extend(v._prev)
    return out
