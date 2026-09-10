# 1 · What a model is: function fitting, generalization, and the bias–variance trade-off

*Lecture 1, part 1 of 8 · ~25 min reading*

## 1.1 Machine learning in terms you already use

You already write functions. A function takes an input, applies a rule, and returns an output. The rule is fixed at the moment you type it. Machine learning changes exactly one thing: **the rule is not typed, it is fitted.**

A model is a function `f(x; θ)` with two kinds of inputs. `x` is the ordinary argument — a transaction, an image, a token sequence. `θ` (theta) is the set of **parameters**: numbers that determine *which* function in a family you actually have. A linear model `f(x; θ) = θ₀ + θ₁x` is a family of every straight line; choosing `θ` picks one line. A transformer with 8 billion parameters is a family of an astronomically large number of functions; training picks one.

The whole discipline can be stated in three lines an engineer already recognizes:

1. **Data in.** A dataset `D_train = {(x₁, y₁), …, (x_N, y_N)}` of inputs paired with the outputs you want.
2. **Parameters fitted.** Choose `θ` to make `f(x_i; θ)` close to `y_i` on that data, where "close" is measured by a **loss function** `ℓ(y, ŷ)`. Formally you minimize the **empirical risk**:

   ```
   L(θ; D_train) = (1/N) · Σᵢ ℓ(yᵢ, f(xᵢ; θ))
   ```

   This is Murphy's eq. 1.2 / 1.30 and the object every optimizer in this course pushes downhill. Minimizing it is called **empirical risk minimization (ERM)**.

3. **Behaviour generalized.** You do not actually care about the training data — you already have its answers. You care about the **population risk**, the expected loss on new inputs drawn from the same distribution `p*(x, y)`:

   ```
   L(θ; p*) = E_(x,y)~p* [ ℓ(y, f(x; θ)) ]
   ```

   The gap between the two, `L(θ; p*) − L(θ; D_train)`, is the **generalization gap**. Murphy (§1.2.3) is blunt about it: a large gap — "low empirical risk but high population risk" — "is a sign that it is overfitting." You cannot compute the population risk, so you estimate it with data the fitting procedure never saw. Everything about model evaluation is a set of techniques for keeping that estimate honest.

So: **a model is a function whose rule was chosen by an optimizer to minimize a loss on a dataset, in the hope that the choice also minimizes the loss on data it has not seen.** Every failure mode in this course is a failure of one of those clauses — the dataset was wrong, the loss was the wrong proxy, or the hope was unfounded.

### The assumption that makes any of this work

Goodfellow, Bengio & Courville (§5.2) name the two assumptions that let a training score say anything about a future input: examples are **independent** of each other, and train and test are **identically distributed**, "drawn from the same probability distribution." Together: **i.i.d.** Google's ML Crash Course adds a third that engineers forget most: the data-generating process is **stationary** — it does not change over the life of the model. Part 7 of this lecture (distribution shift) is entirely about what happens when that third assumption fails in production.

### Train, validation, test

You need three disjoint splits, and the reason is the same one that makes you distrust a test suite that was written by looking at the implementation.

The **training set** is what the optimizer sees. The **test set** is what you report; it is touched once, at the end. The **validation set** exists because there are decisions the optimizer does not make — learning rate, model size, regularization strength, how many epochs, which threshold. Goodfellow §5.3: "The subset of data used to guide the selection of hyperparameters is called the validation set." If you tuned those on the test set, the scikit-learn user guide (§3.1) says what happens: "knowledge about the test set can 'leak' into the model and evaluation metrics no longer report on generalization performance."

Two engineering facts about the test set that Google's Crash Course states directly: it must contain "zero examples duplicated in the training set," and "the more you use the same test set, the more likely the model closely fits the test set." A test set is a consumable. Week 1B (leakage) is about how the first rule is broken accidentally; the second means you should budget how many times a test set can be looked at before it stops being a test set.

## 1.2 Underfitting, overfitting, and capacity

Goodfellow §5.2 gives the cleanest two-part statement of what a learning algorithm must do:

> 1. Make the training error small.
> 2. Make the gap between training and test error small.

Fail the first and you are **underfitting**: the model "is not able to obtain a sufficiently low error value on the training set." Fail the second and you are **overfitting**: "the gap between the training error and test error is too large."

The knob that trades one against the other is **capacity** — informally, per Goodfellow, "a model's ability to fit a wide variety of functions." Straight lines have low capacity; degree-9 polynomials have more; a deep network has vastly more. Goodfellow's Figure 5.3 tells the standard story: as capacity grows, training error falls monotonically toward zero while the generalization gap widens, so test error is U-shaped and the optimal capacity sits at the bottom of the U.

Murphy (§5.4.1) names the two components of that U. **Approximation error** is how far the best function in your family is from the truth — a straight line cannot represent a sine wave no matter how it is fitted. **Estimation error** is how far the function you actually found is from the best one in the family, because you only had N noisy examples to find it with. "We can decrease the approximation error by using a more expressive family of functions H. However, this usually increases overfitting, which increases the estimation error."

### Bias and variance

The same trade-off in statistical language. Suppose you could redraw the training set many times and refit each time. Your fitted function would wobble. **Variance** is how much it wobbles across redraws. **Bias** is how far its *average* over redraws sits from the truth. For a squared-error target the decomposition is exact (Goodfellow §5.4.4, Murphy eq. 4.241–4.242):

```
MSE = E[(θ̂ − θ)²] = Bias(θ̂)² + Var(θ̂)
```

Goodfellow's Figure 5.6 overlays this on the capacity axis: "As capacity increases, bias tends to decrease and variance tends to increase, yielding another U-shaped curve for generalization error." A low-capacity model is biased (it cannot bend to the data) but stable across redraws; a high-capacity model can bend to anything, including the noise in this particular sample, so it is unstable across redraws.

The practical consequence Murphy draws is the one to remember: "it might be wise to use a biased estimator, so long as it reduces our variance by more than the square of the bias." That sentence is the justification for every regularizer in part 2. Regularization *deliberately adds bias* to buy a larger reduction in variance.

### How to tell which one you have

You have two numbers: training loss and validation loss.

- Both high, close together → underfitting (high bias). More capacity, more features, train longer, less regularization.
- Training low, validation much higher → overfitting (high variance). More data, more regularization, less capacity, early stopping.
- Both low, close together → you may be done. Part 3 explains why "may" is doing heavy lifting.

## 1.3 A modern caveat: double descent

The U-shaped curve is a textbook truth that modern practice bends. Belkin, Hsu, Ma & Mandal (PNAS 2019) showed that if you keep increasing capacity *past* the point where the model can fit the training data perfectly (the **interpolation threshold**), test error often goes *back down*: a "double descent" curve that "subsumes the textbook U-shaped bias-variance trade-off curve by showing how increasing model capacity beyond the point of interpolation results in improved performance."

Nakkiran et al. (2019, "Deep Double Descent") found the same shape in deep networks, and found it along three axes: model size, training epochs, and — counter-intuitively — dataset size, with "certain regimes where increasing (even quadrupling) the number of train samples actually hurts test performance." The companion OpenAI post locates the danger: "the peak of test error appears systematically when models are just barely able to fit the train set."

What to take from this at week 1: the bias–variance U is correct for the regime most classical models live in (parameters ≪ samples), and Murphy (§13.5.7) states the boundary: "models where P < N show the usual U-shaped curve … However, when P > N, test error starts to decrease again." Large neural networks live on the far side. You will meet the mechanism properly in week 3. For now, know that "bigger model overfits more" is a heuristic with a known counterexample, not a law.

## 1.4 What to carry forward

A model is a fitted function. Training minimizes a loss on data you have; what you are paid for is the loss on data you do not have. The gap between them is controlled by capacity, and capacity trades bias against variance. Every technique in part 2 is a way of spending bias to buy variance. Every technique in parts 4–6 is a way of measuring the population loss honestly enough to make a decision.

---

**Sources for this part** (exact sections in `READING.md`): Murphy, *Probabilistic Machine Learning: An Introduction* §1.2.1.4, §1.2.3, §4.7.6.3, §5.4.1, §13.5.7 · Goodfellow, Bengio & Courville, *Deep Learning* §5.2, §5.3, §5.4.4 · scikit-learn User Guide §3.1 · Google ML Crash Course, "Overfitting" and "Dividing the original dataset" · Belkin et al. 2019 (arXiv 1812.11118) · Nakkiran et al. 2019 (arXiv 1912.02292).
