"""Tests for value.py and gradcheck.py.  Run:  pytest test_value.py -q

These tests check your engine against (a) small cases you can verify by hand,
(b) an independent numerical-derivative estimate computed inside this file
(it does not import your gradcheck.py, so it is a real second opinion), and
(c) three reference numbers frozen from a known-correct engine. They are
checks, not solutions: nothing here tells you how to build the graph or
order the backward pass, and the oracle below is deliberately NOT the
central-difference formula the assignment asks you to implement — it is a
five-point stencil with O(h^4) error, which is more than the build needs and
would fail convention 9 if you copied it.

If a test fails, ASSIGNMENT.md -> "If stuck" maps the symptom to the
convention or lecture section you most likely missed.
"""
from __future__ import annotations

import math
import random
import sys

import pytest

from value import Value, graph_nodes

# --------------------------------------------------------------------------- #
# Private numerical oracle (independent of the student's gradcheck.py).
# Five-point stencil: f'(x) ~ [-f(x+2h) + 8 f(x+h) - 8 f(x-h) + f(x-2h)] / 12h
# --------------------------------------------------------------------------- #

H = 1e-4  # a larger h is right for an O(h^4) formula; do not reuse for yours


def _fd(f, xs, h=H):
    out = []
    for i in range(len(xs)):
        def at(delta):
            pt = list(xs)
            pt[i] += delta
            return f([Value(x) for x in pt]).data
        out.append((-at(2 * h) + 8 * at(h) - 8 * at(-h) + at(-2 * h)) / (12 * h))
    return out


def _engine(f, xs):
    leaves = [Value(x) for x in xs]
    out = f(leaves)
    out.backward()
    return out, [v.grad for v in leaves]


def _agree(f, xs, rtol=1e-5, atol=1e-7):
    _, a = _engine(f, xs)
    n = _fd(f, xs)
    for ai, ni in zip(a, n):
        assert ai == pytest.approx(ni, rel=rtol, abs=atol), (a, n)


# --------------------------------------------------------------------------- #
# §1 forward values and basic structure
# --------------------------------------------------------------------------- #


def test_value_basics():
    v = Value(2)
    assert isinstance(v.data, float) and v.data == 2.0
    assert v.grad == 0.0
    assert v._prev == set() and v._op == ""


def test_forward_arithmetic_hand_case():
    a, b = Value(2.0), Value(-3.0)
    assert (a + b).data == -1.0
    assert (a * b).data == -6.0
    assert (a - b).data == 5.0
    assert (a / b).data == pytest.approx(-2.0 / 3.0)
    assert (a ** 3).data == 8.0
    assert (-a).data == -2.0


def test_forward_with_python_scalars_both_sides():
    a = Value(4.0)
    assert (a + 1).data == 5.0 and (1 + a).data == 5.0
    assert (a * 2).data == 8.0 and (2 * a).data == 8.0
    assert (a - 1).data == 3.0 and (1 - a).data == -3.0
    assert (a / 2).data == 2.0 and (2 / a).data == 0.5


def test_forward_nonlinearities():
    x = Value(0.5)
    assert x.exp().data == pytest.approx(math.exp(0.5))
    assert x.log().data == pytest.approx(math.log(0.5))
    assert x.tanh().data == pytest.approx(math.tanh(0.5))
    assert x.relu().data == 0.5
    assert Value(-0.5).relu().data == 0.0


def test_log_rejects_nonpositive():
    with pytest.raises(ValueError):
        Value(0.0).log()
    with pytest.raises(ValueError):
        Value(-1.0).log()


def test_operations_record_parents():
    a, b = Value(1.0), Value(2.0)
    c = a * b
    assert c._prev == {a, b}
    d = c + 3
    assert c in d._prev and len(d._prev) == 2  # the constant became a Value


# --------------------------------------------------------------------------- #
# §2 backward on cases you can do by hand (lecture part 6 §6.2)
# --------------------------------------------------------------------------- #


def test_backward_add_distributes():
    a, b = Value(3.0), Value(4.0)
    c = a + b
    c.backward()
    assert c.grad == 1.0
    assert a.grad == 1.0 and b.grad == 1.0


def test_backward_mul_switches():
    a, b = Value(3.0), Value(4.0)
    c = a * b
    c.backward()
    assert a.grad == 4.0 and b.grad == 3.0


def test_backward_pow_and_div():
    x = Value(2.0)
    y = x ** 3
    y.backward()
    assert x.grad == pytest.approx(12.0)  # 3 x^2

    x = Value(2.0)
    y = 1 / x
    y.backward()
    assert x.grad == pytest.approx(-0.25)  # -1/x^2


def test_backward_nonlinearities_hand_case():
    x = Value(0.0)
    x.tanh().backward()
    assert x.grad == pytest.approx(1.0)  # 1 - tanh(0)^2

    x = Value(1.0)
    x.log().backward()
    assert x.grad == pytest.approx(1.0)

    x = Value(2.0)
    y = x.exp()
    y.backward()
    assert x.grad == pytest.approx(y.data)

    x = Value(-1.0)
    x.relu().backward()
    assert x.grad == 0.0
    x = Value(1.0)
    x.relu().backward()
    assert x.grad == 1.0


def test_relu_derivative_is_zero_at_exactly_zero():
    x = Value(0.0)
    x.relu().backward()
    assert x.grad == 0.0


# --------------------------------------------------------------------------- #
# §3 the cases that break naive implementations (lecture part 6 §6.2–6.3)
# --------------------------------------------------------------------------- #


def test_gradients_accumulate_when_a_value_is_used_twice():
    a = Value(3.0)
    b = a * a  # d/da = 2a = 6, only if both paths add
    b.backward()
    assert a.grad == pytest.approx(6.0)


def test_diamond_graph_needs_correct_order():
    # (a + b) * (a - b) = a^2 - b^2 ; d/da = 2a, d/db = -2b
    a, b = Value(3.0), Value(2.0)
    c = (a + b) * (a - b)
    c.backward()
    assert a.grad == pytest.approx(6.0)
    assert b.grad == pytest.approx(-4.0)


def test_paths_of_different_length():
    # y = x * (x + (x * x)) = x^2 + x^3 ; dy/dx = 2x + 3x^2 at x=2 -> 16
    x = Value(2.0)
    y = x * (x + x * x)
    y.backward()
    assert x.grad == pytest.approx(16.0)


def test_each_node_backward_runs_exactly_once():
    # If a node's rule ran twice, its inputs' grads would double.
    a = Value(1.5)
    b = a.exp()
    c = b * b + b  # b consumed twice; b's own rule must still run once
    c.backward()
    expected = (2 * b.data + 1) * b.data  # dc/db * db/da
    assert a.grad == pytest.approx(expected)


def test_deep_chain_does_not_hit_recursion_limit():
    depth = 3000
    assert depth > sys.getrecursionlimit()
    x = Value(1.0)
    y = x
    for _ in range(depth):
        y = y * 1.0001 + 0.0
    y.backward()
    assert x.grad == pytest.approx(1.0001 ** depth, rel=1e-9)


def test_backward_sets_root_grad_to_one_not_plus_one():
    a = Value(2.0)
    c = a * 3
    c.grad = 5.0  # stale value from somewhere
    c.backward()
    assert c.grad == 1.0
    assert a.grad == pytest.approx(3.0)


def test_repeated_backward_doubles_leaf_grads_only():
    # Convention 4: leaves accumulate across passes, non-leaves are reset.
    # d = a*b ; c = 2*d.  dc/da = 2b = 10, dc/dd = 2.
    a, b = Value(2.0), Value(5.0)
    d = a * b
    c = d * 2
    c.backward()
    assert a.grad == pytest.approx(10.0) and d.grad == pytest.approx(2.0)
    c.backward()
    assert a.grad == pytest.approx(20.0)  # leaf: 10 + 10
    assert b.grad == pytest.approx(8.0)  # leaf: 4 + 4
    assert d.grad == pytest.approx(2.0)  # non-leaf: reset, not 4
    assert c.grad == 1.0


def test_zero_grad_resets_everything_and_backward_recomputes():
    a, b = Value(2.0), Value(5.0)
    c = a * b
    c.backward()
    c.zero_grad()
    assert all(v.grad == 0.0 for v in graph_nodes(c))
    c.backward()
    assert a.grad == pytest.approx(5.0)


def test_zero_grad_reaches_every_node_in_a_wide_graph():
    xs = [Value(float(i)) for i in range(20)]
    total = sum(xs, Value(0.0))
    total.backward()
    total.zero_grad()
    assert all(v.grad == 0.0 for v in graph_nodes(total))
    assert len(graph_nodes(total)) >= 21


# --------------------------------------------------------------------------- #
# §4 against finite differences — the independent oracle (part 6 §6.5)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name,f,xs",
    [
        ("add", lambda v: v[0] + v[1], [1.3, -0.7]),
        ("mul", lambda v: v[0] * v[1], [1.3, -0.7]),
        ("sub", lambda v: v[0] - v[1], [1.3, -0.7]),
        ("div", lambda v: v[0] / v[1], [1.3, -0.7]),
        ("pow3", lambda v: v[0] ** 3, [1.3]),
        ("pow_half", lambda v: v[0] ** 0.5, [1.3]),
        ("pow_neg", lambda v: v[0] ** -2, [1.3]),
        ("exp", lambda v: v[0].exp(), [0.9]),
        ("log", lambda v: v[0].log(), [0.9]),
        ("tanh", lambda v: v[0].tanh(), [0.4]),
        ("relu_pos", lambda v: v[0].relu(), [0.4]),
        ("relu_neg", lambda v: v[0].relu(), [-0.4]),
        ("scalar_left", lambda v: 3.0 - 2.0 * v[0], [0.4]),
        ("scalar_div", lambda v: 2.0 / v[0], [0.4]),
    ],
)
def test_single_op_matches_finite_differences(name, f, xs):
    _agree(f, xs)


def test_composite_expression_matches_finite_differences():
    def f(v):
        a, b, c = v
        return ((a * b + c).tanh() * (a - c).exp() + (b * b + 1.0).log()) / (1.0 + a * a)

    _agree(f, [0.7, -1.2, 0.3])


def test_softmax_cross_entropy_matches_finite_differences():
    # A 3-way softmax + negative log-likelihood of class 1, built from Values.
    def f(v):
        exps = [z.exp() for z in v]
        total = exps[0] + exps[1] + exps[2]
        p1 = exps[1] / total
        return -(p1.log())

    _agree(f, [0.5, -0.2, 1.1])
    # Also the closed form: dL/dz_i = p_i - [i == 1]
    _, grads = _engine(f, [0.5, -0.2, 1.1])
    zs = [0.5, -0.2, 1.1]
    denom = sum(math.exp(z) for z in zs)
    for i, g in enumerate(grads):
        assert g == pytest.approx(math.exp(zs[i]) / denom - (1 if i == 1 else 0), rel=1e-9)


def test_tiny_mlp_matches_finite_differences():
    # 2 -> 3 -> 1 MLP with tanh, squared error; all 13 parameters as leaves.
    rng = random.Random(0)
    x = [0.6, -0.9]
    target = 0.25

    def f(v):
        W1 = [v[0:2], v[2:4], v[4:6]]
        b1 = v[6:9]
        W2 = v[9:12]
        b2 = v[12]
        h = [(W1[j][0] * x[0] + W1[j][1] * x[1] + b1[j]).tanh() for j in range(3)]
        out = W2[0] * h[0] + W2[1] * h[1] + W2[2] * h[2] + b2
        return (out - target) ** 2

    params = [rng.uniform(-1, 1) for _ in range(13)]
    _agree(f, params, rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------- #
# §5 frozen reference: the micrograd README example
# --------------------------------------------------------------------------- #


def test_micrograd_readme_example_frozen_values():
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b ** 3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e ** 2
    g = f / 2.0
    g += 10.0 / f
    assert g.data == pytest.approx(24.7041, abs=1e-4)
    g.backward()
    assert a.grad == pytest.approx(138.8338, abs=1e-4)
    assert b.grad == pytest.approx(645.5773, abs=1e-4)


# --------------------------------------------------------------------------- #
# §6 the student's gradcheck.py
# --------------------------------------------------------------------------- #

gradcheck = pytest.importorskip("gradcheck")


def test_numerical_grad_on_known_function():
    f = lambda v: v[0] * v[0] * v[1] + v[1].exp()
    n = gradcheck.numerical_grad(f, [1.5, 0.3])
    assert n[0] == pytest.approx(2 * 1.5 * 0.3, rel=1e-6)
    assert n[1] == pytest.approx(1.5 ** 2 + math.exp(0.3), rel=1e-6)


def test_analytical_grad_returns_fresh_leaf_grads():
    f = lambda v: (v[0] * v[1]).tanh()
    a = gradcheck.analytical_grad(f, [0.3, 0.8])
    n = _fd(f, [0.3, 0.8])
    assert a == pytest.approx(n, rel=1e-6)


def test_check_gradients_passes_on_correct_engine():
    f = lambda v: (v[0] * v[1] + v[2] ** 2).tanh() / (1.0 + v[0].exp())
    ok, max_diff, a, n = gradcheck.check_gradients(f, [0.3, -0.8, 0.5])
    assert ok is True
    assert max_diff < 1e-6
    assert len(a) == len(n) == 3


def test_check_gradients_detects_a_wrong_gradient():
    # A function whose engine gradient is deliberately wrong: `.data` is used
    # inside f, which cuts the graph (the scalar version of `.detach()`), so
    # the analytical gradient w.r.t. v[1] is 0 while the numerical one is not.
    f = lambda v: v[0] * Value(v[1].data)
    ok, max_diff, a, n = gradcheck.check_gradients(f, [1.0, 2.0])
    assert ok is False
    assert a[1] == 0.0 and n[1] == pytest.approx(1.0, rel=1e-6)
    assert max_diff == pytest.approx(1.0, rel=1e-6)


def test_check_gradients_step_size_too_small_is_noisy():
    # This documents WHY h matters (part 6 §6.5). At h = 1e-13 in float64 the
    # round-off error dominates and a correct engine "fails" the check.
    f = lambda v: (v[0] * 3.7).exp()
    ok_good, _, _, _ = gradcheck.check_gradients(f, [0.9], h=1e-6)
    ok_bad, diff_bad, _, _ = gradcheck.check_gradients(f, [0.9], h=1e-13, atol=1e-8, rtol=1e-6)
    assert ok_good is True
    assert ok_bad is False and diff_bad > 1e-6
