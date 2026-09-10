# 2 · Regularization: L1, L2, early stopping — and why AdamW exists

*Lecture 1, part 2 of 8 · ~25 min reading*

## 2.1 What regularization is

Goodfellow, Bengio & Courville open chapter 7 with a definition worth memorizing because it is *operational*: regularization is "any modification we make to a learning algorithm that is intended to reduce its generalization error but not its training error."

Read that twice. A regularizer is expected to make training loss *worse*. Google's Crash Course says the same of early stopping: it "usually increases training loss, it can decrease test loss." If you add a regularizer and training loss does not go up, the regularizer is not doing anything. This is part 1's bias–variance trade in action: you are adding bias on purpose.

Three regularizers matter this week: L2, L1, and early stopping. For each, the question to answer is not "what is the formula" but **what does it constrain?** — which functions in the family does it make expensive, and why would those be the ones that generalize badly?

## 2.2 L2 — shrink everything, mostly the unimportant directions

Add a penalty proportional to the squared norm of the weights:

```
J̃(w) = J(w) + (α/2) · ‖w‖₂²          (Goodfellow §7.1.1)
```

Take the gradient and the update step becomes

```
w ← (1 − εα) · w − ε · ∇J(w)
```

Every step first *multiplies the weights by a number slightly less than one*, then takes the ordinary gradient step. That multiplicative shrink is why L2 is also called **weight decay** — hold on to that phrase; §2.5 is about the fact that the two names stopped meaning the same thing in 2017.

**What it constrains.** Goodfellow's eigen-analysis (§7.1.1) is the useful picture. Around the optimum, the loss is approximately a bowl whose curvature in each direction is an eigenvalue `λᵢ` of the Hessian. The L2-regularized solution is the unregularized one with each component rescaled by `λᵢ / (λᵢ + α)`. Directions where the loss is very curved (`λᵢ ≫ α`) — directions the data pins down firmly — are left almost alone. Directions where the loss is nearly flat (`λᵢ ≪ α`) — directions the data barely cares about — "will be shrunk to have nearly zero magnitude." L2 therefore constrains exactly the parameters the training data does not constrain. Those are the ones that would otherwise be set by noise, which is the variance you are trying to kill.

In scikit-learn this is `Ridge`: `min_w ‖Xw − y‖₂² + α‖w‖₂²` (§1.1.2). Equivalent constrained form (Goodfellow §7.2, Murphy eq. 11.80): minimize the loss subject to `‖w‖₂² ≤ B`. The weights must live inside a sphere; larger `α` means a smaller sphere.

## 2.3 L1 — force some weights to exactly zero

Swap the squared norm for the sum of absolute values:

```
J̃(w) = J(w) + α · ‖w‖₁ ,   ‖w‖₁ = Σᵢ |wᵢ|          (Goodfellow §7.1.2)
```

**What it constrains.** L1 "results in a solution that is more sparse" — some weights become *exactly* zero — whereas L2 "does not cause the parameters to become sparse." Murphy's §11.4.2 has the picture every textbook reuses (his Fig. 11.8): the constraint region `‖w‖₁ ≤ B` is a diamond (in 2-D) with corners on the axes, while the L2 region is a circle. The loss contours are ellipses expanding outward until they first touch the constraint region. "The corners of the ball are more likely to intersect the ellipse than one of the sides, especially in high dimensions, because the corners 'stick out' more. The corners correspond to sparse solutions." The circle has no corners, "so there is no preference for sparsity."

The consequence is that L1 does **feature selection** as a side-effect of fitting. In scikit-learn this is `Lasso`: `min_w (1/2n)‖Xw − y‖₂² + α‖w‖₁`, described as a model whose "tendency [is] to prefer solutions with fewer non-zero coefficients" (§1.1.3). If you have a thousand features and suspect thirty matter, L1 finds a model that *says which* thirty. L2 would give you a thousand small weights and no answer.

**When to use which.** L2 when you believe most features carry a little signal and you want stability. L1 when you believe most features carry no signal and you want the model to say so. Both when you are not sure (elastic net). In deep learning, L2/weight decay is near-universal and L1 on weights is rare; sparsity there is pursued by other means.

## 2.4 Early stopping — limit how much the data can move the parameters

Train, watch the validation loss, stop when it stops improving, and roll back to the parameters from the best validation step. Goodfellow §7.8 calls it "probably the most commonly used form of regularization in deep learning," and observes that it silently turns *the number of training steps* into a hyperparameter with a U-shaped validation curve.

**What it constrains.** Something surprising: for a quadratic loss trained by gradient descent, early stopping *is* L2 regularization in disguise. Goodfellow §7.8 works through the algebra — the shrink factor from τ steps of gradient descent, `(I − εΛ)^τ`, approximates the L2 shrink factor `(Λ + αI)⁻¹α` when

```
τ ≈ 1 / (ε · α)
```

That is: **steps × learning rate ≈ 1 / (L2 strength).** Training for fewer steps is the same as regularizing harder. Distill's "Why Momentum Really Works" (Goh 2017) shows the same thing from the optimization side — both early stopping and Tikhonov (L2) regularization suppress the small-eigenvalue components of the solution, the flat directions from §2.2. Murphy (§13.5.1) gives the intuition without the algebra: early stopping works "because we are restricting the ability of the optimization algorithm to transfer information from the training examples to the parameters."

The practical advantage over L2 is that you do not have to guess `α` up front. You run once and the validation curve tells you where to stop. The practical cost is that you need a validation set that is *not* the test set — the same leak as §1.1, in a different costume.

## 2.5 Weight decay versus L2 in the loss — and why AdamW exists

Here is the terminology trap. Goodfellow §7.1.1 and Murphy §13.5.2 both call L2 regularization "weight decay," and in 2016 that was fine. It stopped being fine when everyone switched to Adam.

### Two things that used to be the same

**L2 in the loss** means you add `(λ'/2)‖θ‖²` to the objective, so the *gradient* gets an extra term: `g ← ∇f(θ) + λ'θ`. Whatever the optimizer does with gradients, it now does to this one.

**Weight decay** means you leave the loss alone and, at update time, subtract a fraction of the weights directly: `θ ← θ − ηλθ − (the optimizer's step)`.

Loshchilov & Hutter (2017, "Decoupled Weight Decay Regularization") prove in their Proposition 1 that for plain SGD these are identical up to a rescaling: SGD with learning rate `α` and weight decay `λ` "executes the same steps … as it executes without weight decay on `f_reg(θ) = f(θ) + (λ'/2)‖θ‖²₂`, with `λ' = λ/α`." For SGD, the two names really did mean one thing.

### Why Adam breaks the equivalence

Adam does not apply the gradient directly. It keeps a running mean `m` of gradients and a running mean `v` of squared gradients, and its step is roughly `m̂ / (√v̂ + ε)` — it *divides* each parameter's step by a running estimate of that parameter's gradient magnitude. This is what makes it adaptive: parameters that see large gradients take smaller steps.

Now put L2 in the loss. The regularizer's gradient `λ'θ` goes into `m` and `v` along with everything else, so it too gets divided by `√v̂`. Loshchilov & Hutter state the consequence: "weights that tend to have large gradients in `f` do not get regularized as much as they would with decoupled weight decay, since the gradient of the regularizer gets scaled along with the gradient of `f`." The parameters that move the most — often the ones you most wanted to restrain — are regularized the *least*. Their Proposition 2 makes it formal: for any optimizer whose per-parameter scaling is not a constant, "there exists no L2 coefficient λ' such that running [it] on … `f_reg` without weight decay is equivalent to running [it] on `f` with decay λ." No choice of L2 strength recovers what weight decay does.

### The one-line difference

Their Algorithm 2 puts the two variants side by side. The only difference is *where* the `λθ` term enters:

```
Adam + L2 (line 6):    g_t ← ∇f_t(θ_{t−1}) + λ·θ_{t−1}         # then g_t feeds m_t and v_t
AdamW   (line 12):     θ_t ← θ_{t−1} − η_t · ( α·m̂_t / (√v̂_t + ε)  +  λ·θ_{t−1} )
```

In AdamW the decay term bypasses the moment estimates entirely. It is applied to the raw weights, at the same strength, every step, regardless of how big that parameter's gradients have been. That is **decoupled** weight decay, and it is what the W stands for.

The fast.ai write-up (Gugger & Howard 2018) says it in code for the people who prefer code: SGD weight decay is `w = w - lr*w.grad - lr*wd*w`; "in the case of L2 regularization we add this `wd*w` to the gradients then compute a moving average of the gradients and their squares before using both of them for the update. Whereas the weight decay method simply consists in doing the update, then subtract to each weight."

### What it bought

On CIFAR-10 with a 26-layer ResNet trained for 1800 epochs, Loshchilov & Hutter report AdamW giving roughly a 15 % relative improvement in test error over Adam with L2, and similar gains on ImageNet32×32 — closing most of the gap that had made practitioners keep SGD-with-momentum for vision. The paper's abstract is the sentence to remember: L2 and weight decay "are equivalent for standard stochastic gradient descent (when rescaled by the learning rate), but … this is *not* the case for adaptive gradient algorithms, such as Adam."

### What this means when you open PyTorch

The PyTorch docs encode exactly this split, and you should read the pseudocode blocks once so you never guess again:

- `torch.optim.SGD` and `torch.optim.Adam` both implement their `weight_decay` argument as **`g_t ← g_t + λθ_{t−1}`** — that is L2-in-the-gradient, default `weight_decay=0`.
- `torch.optim.AdamW` implements **`θ_t ← θ_{t−1} − γλθ_{t−1}`** as a separate step, default `weight_decay=1e-2`, so that (quoting the docs) "weight decay does not accumulate in the momentum nor variance."
- Recent PyTorch also exposes `Adam(..., decoupled_weight_decay=True)`, documented as: "this optimizer is equivalent to AdamW."

So the rule: **if you are using Adam and you want weight decay, use AdamW.** If someone hands you a training script using `Adam(weight_decay=…)`, you now know it is doing L2-in-the-loss, and you know why that regularizes the wrong parameters least. You will implement Adam by hand in week 3 and this distinction will be one line of your code.

## 2.6 What to carry forward

| Regularizer | What it constrains | Geometry | Cost |
|---|---|---|---|
| L2 / ridge | Shrinks weights, hardest along directions the data does not determine | Sphere `‖w‖₂² ≤ B` | Need to pick `α` |
| L1 / lasso | Drives some weights to exactly zero → feature selection | Diamond `‖w‖₁ ≤ B` with axis corners | Need to pick `α`; non-smooth |
| Early stopping | Limits information transferred from data to parameters; ≈ L2 with `α ≈ 1/(ετ)` | — | Need a validation set |
| Decoupled weight decay (AdamW) | Same shrink as L2, applied *uniformly* rather than divided by Adam's adaptive scale | — | None over Adam; just use it |

---

**Sources for this part** (exact sections in `READING.md`): Goodfellow et al. *Deep Learning* §7.1.1, §7.1.2, §7.2, §7.8 · Murphy *PML: An Introduction* §11.3, §11.4.2, §13.5.1, §13.5.2 · scikit-learn §1.1.2, §1.1.3 · Loshchilov & Hutter 2017, arXiv 1711.05101, §2, Props. 1–2, Alg. 2 · PyTorch docs for `torch.optim.SGD`, `Adam`, `AdamW` · Gugger & Howard, fast.ai 2018 · Goh, Distill 2017 · Google ML Crash Course, "L2 regularization".
