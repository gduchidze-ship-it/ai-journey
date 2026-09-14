"""gradcheck.py — verify an autograd engine against finite differences.

Course: Applied AI Engineering, week 2 (Lecture 3 build, §4).
Rules: standard library only.

This is the independent oracle from lecture part 6 §6.5: it knows no
derivative rules, only how to evaluate `f` at nearby points. If your engine's
gradient and this file's estimate disagree, one of them is wrong — and this
one is wrong only if `h` is badly chosen, which is why choosing `h` is part
of the assignment.

Conventions (the tests assume them):
  * `f` takes a list of Values and returns a single Value (the "loss").
  * `xs` is a list of plain floats: the point at which to differentiate.
  * `numerical_grad` uses the CENTRAL difference for every coordinate and
    perturbs one coordinate at a time, building fresh Values for every
    evaluation (never reuse a Value across evaluations — its graph is stale).
  * `analytical_grad` builds fresh Values from `xs`, calls `f`, calls
    `.backward()`, and returns the leaves' `.grad`.
  * `check_gradients` compares the two with the mixed tolerance
        |a - n| <= atol + rtol * |n|
    per coordinate (the same rule `torch.allclose` uses) and returns
    (ok: bool, max_abs_diff: float, analytical: list[float], numerical: list[float]).

Every function body is a stub. Keep the signatures.
"""
from __future__ import annotations

from typing import Callable, Sequence

from value import Value

Fn = Callable[[list[Value]], Value]


def numerical_grad(f: Fn, xs: Sequence[float], h: float = 1e-6) -> list[float]:
    """Central-difference estimate of df/dx_i at `xs`, one entry per coordinate.

    Lecture part 6 §6.5: the error is O(h^2) truncation + O(eps/h) round-off;
    with float64 the sweet spot is h ~ 1e-6 to 1e-5. Do not go smaller.
    """
    raise NotImplementedError


def analytical_grad(f: Fn, xs: Sequence[float]) -> list[float]:
    """Gradient from the engine: fresh leaves, forward, backward, read .grad."""
    raise NotImplementedError


def check_gradients(
    f: Fn,
    xs: Sequence[float],
    h: float = 1e-6,
    atol: float = 1e-6,
    rtol: float = 1e-4,
) -> tuple[bool, float, list[float], list[float]]:
    """Compare analytical and numerical gradients coordinate by coordinate."""
    raise NotImplementedError
