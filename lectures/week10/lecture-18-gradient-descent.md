# Lecture 18 - Gradient Descent for Linear Regression

In Lecture 17, we began with the familiar least-squares estimator for a straight line. Under an independent, common-variance Gaussian noise model, we found that the same weights are also maximum-likelihood estimates. We then generalized the prediction function with basis functions and used the normal equations to calculate its least-squares weights. For a modest design matrix, a numerical least-squares solver can find the fitted weights directly. But what if we have millions of daily measurements, observations arriving continuously, or a model for which setting the gradient to zero does not give a convenient equation to solve?

We can use the gradient for a second purpose. Instead of solving $\nabla J(\mathbf w)=\mathbf 0$ in one calculation, we can start with some weights and repeatedly move them in a direction that lowers the objective. This is **gradient descent**. It changes *how* we find the weights, not the conditional Gaussian model or the least-squares objective we derived. Lecture 2 gave us gradients, Lecture 7 connected likelihood to fitting, and Lecture 11 reminded us that the best training fit still needs validation on unseen data.

:::{admonition} Guiding question
:class: tip

Instead of solving for the best weights all at once, can we learn them gradually?
:::

## Learning objectives

After completing this lecture, you should be able to:

1. Obtain the gradient of the squared-error objective from the individual training examples.
2. Perform batch, stochastic, and mini-batch gradient-descent updates for linear regression.
3. Explain what the learning rate controls and recognize steps that are too small or too large.
4. Distinguish an update from an epoch and explain why stochastic updates can be noisy.
5. Choose an appropriate fitting method for a given data size and computational setting.

## The objective from Lecture 17

Retain the notation $\mathcal D=\{(\mathbf x_n,y_n)\}_{n=1}^N$, with fixed basis vector $\boldsymbol\phi(\mathbf x_n)$ and prediction

$$
\widehat y_n(\mathbf w)=\mathbf w^{\mathsf T}\boldsymbol\phi(\mathbf x_n).
$$

Lecture 17 used the *sum* of squared residuals, with a factor of $1/2$:

$$
E_D(\mathbf w)
=\frac12\sum_{n=1}^N
\left[y_n-\widehat y_n(\mathbf w)\right]^2.
$$

For this lecture, divide by the fixed sample size $N$ and call the result $J(\mathbf w)$:

$$
\boxed{
J(\mathbf w)
=\frac1N E_D(\mathbf w)
=\frac1N\sum_{n=1}^N
\underbrace{\frac12
\left[y_n-\mathbf w^{\mathsf T}\boldsymbol\phi(\mathbf x_n)\right]^2}_{\ell_n(\mathbf w)}
.}
$$

Here $\ell_n$ is the loss for one observation. Because $N$ is a positive constant, minimizing $J$ gives **exactly the same weights** as minimizing $E_D$. Averaging makes a batch gradient an average of individual gradients, so its scale does not grow merely because we collected more observations.

### Why this still maximizes Gaussian likelihood

Dividing the squared-error sum by $N$ changed its numerical value, but not which weights minimize it. We can also check directly that $J$ still fits the conditional Gaussian model from Lecture 17. For $N$ conditionally independent observations with the same fixed precision $\beta$, its **negative log-likelihood** is

$$
-\log p(\mathbf y\mid\boldsymbol\Phi,\mathbf w,\beta)
=
\underbrace{\frac N2\log(2\pi)-\frac N2\log\beta}_{\text{independent of }\mathbf w}
+
\frac{\beta}{2}\sum_{n=1}^{N}
\left[y_n-\widehat y_n(\mathbf w)\right]^2.
$$

The bracketed expression depends on $N$ and $\beta$, but **not on the weights**. In this comparison, we hold $N$ and $\beta$ fixed and try different values of $\mathbf w$. Therefore the bracketed expression has the same value for every candidate set of weights. Call that fixed value $C$; it does not have to equal zero.

Now multiply our definition of the average loss by $\beta N$:

$$
\begin{aligned}
\beta N J(\mathbf w)
&=\beta N\left\{
\frac{1}{2N}\sum_{n=1}^{N}
\left[y_n-\widehat y_n(\mathbf w)\right]^2
\right\}\\
&=\frac{\beta}{2}\sum_{n=1}^{N}
\left[y_n-\widehat y_n(\mathbf w)\right]^2.
\end{aligned}
$$

This matches the final term of the negative log-likelihood exactly. We can therefore rewrite the entire expression as

$$
-\log p(\mathbf y\mid\boldsymbol\Phi,\mathbf w,\beta)
=C+\beta N J(\mathbf w),
\qquad
C=\frac N2\log(2\pi)-\frac N2\log\beta.
$$

What does this say about choosing weights? Suppose two candidates, $\mathbf w_A$ and $\mathbf w_B$, have average losses $J(\mathbf w_A)$ and $J(\mathbf w_B)$. The same $C$ is added for both. If, for example, $N=2$, $\beta=1$, and their losses are $4$ and $1$, their negative log-likelihoods are $C+2(4)=C+8$ and $C+2(1)=C+2$. The second is smaller **regardless of the value of $C$**. More generally, subtracting the two negative log-likelihoods makes $C$ disappear:

$$
\bigl[C+\beta N J(\mathbf w_A)\bigr]
-\bigl[C+\beta N J(\mathbf w_B)\bigr]
=\beta N\bigl[J(\mathbf w_A)-J(\mathbf w_B)\bigr].
$$

Because $\beta N>0$, the candidate with smaller $J$ also has smaller negative log-likelihood. Minimizing negative log-likelihood is the same as maximizing likelihood, so minimizing $J$ finds the same Gaussian maximum-likelihood weights. We use $J$ because its average-of-individual-losses form is convenient for gradient descent. **This argument holds while fitting $\mathbf w$ with $\beta$ fixed**; if we also varied $\beta$, $C$ would change and could not be ignored when fitting that parameter.

To differentiate $\ell_n$, write the prediction error as $\widehat y_n-y_n$. For the $j$th weight, the chain rule gives

$$
\frac{\partial\ell_n}{\partial w_j}
=\frac{\partial}{\partial w_j}
\left[\frac12(\widehat y_n-y_n)^2\right]
=(\widehat y_n-y_n)
\underbrace{\frac{\partial\widehat y_n}{\partial w_j}}_{\phi_j(\mathbf x_n)}
=(\widehat y_n-y_n)\phi_j(\mathbf x_n).
$$

Collecting these partial derivatives in a vector, and then averaging over the data, gives

$$
\mathbf g_n(\mathbf w)
:=\nabla_{\mathbf w}\ell_n(\mathbf w)
=\left[\mathbf w^{\mathsf T}\boldsymbol\phi(\mathbf x_n)-y_n\right]
\boldsymbol\phi(\mathbf x_n),
\qquad
\boxed{
\nabla J(\mathbf w)=\frac1N\sum_{n=1}^N\mathbf g_n(\mathbf w)
=\frac1N\boldsymbol\Phi^{\mathsf T}
(\boldsymbol\Phi\mathbf w-\mathbf y).
}
$$

This is the gradient from Lecture 17 divided by $N$. If the prediction is too large, $\widehat y_n-y_n>0$ and that observation pushes against basis directions that increased its prediction. If it is too small, the sign reverses. The formulas below differ only in *which* of these individual gradients we use at each update.

## Moving downhill

For a differentiable function, the gradient points toward the direction of greatest *local* increase. Thus a small step in the opposite direction should lower the objective. Starting from chosen weights $\mathbf w^{(0)}$, gradient descent repeats

$$
\boxed{
\mathbf w^{(k+1)}
=\mathbf w^{(k)}-\alpha_k\,\mathbf g^{(k)}.
}
$$

Here $k$ counts **updates**, $\mathbf g^{(k)}$ is the gradient used for that update, and $\alpha_k>0$ is its **learning rate**. The negative sign is essential: adding the gradient would normally move uphill.

Why only a *small* step? Near the current weights, a first-order approximation says

$$
J(\mathbf w-\alpha\nabla J(\mathbf w))
\approx J(\mathbf w)-\alpha\|\nabla J(\mathbf w)\|_2^2.
$$

The right-hand side is smaller when the gradient is nonzero and $\alpha>0$. This is a **local approximation**, not a guarantee for an arbitrarily large $\alpha$: a large step can pass the minimum and increase the loss. We will see that explicitly below.

## Batch gradient descent: use every observation

In **batch gradient descent**, each update uses the full training set:

$$
\mathbf w^{(k+1)}
=\mathbf w^{(k)}
-\alpha_k\underbrace{
\frac1N\sum_{n=1}^N
\left[(\mathbf w^{(k)})^{\mathsf T}\boldsymbol\phi(\mathbf x_n)-y_n\right]
\boldsymbol\phi(\mathbf x_n)
}_{\nabla J(\mathbf w^{(k)})}.
$$

We calculate all $N$ contributions at the *same* current weights, average them, and then update the weights once. In matrix notation, the expression under the brace is $N^{-1}\boldsymbol\Phi^{\mathsf T}(\boldsymbol\Phi\mathbf w^{(k)}-\mathbf y)$. For fixed data and weights, this gradient is deterministic: repeating the calculation gives the same result.

For example, use Lecture 17's straight-line basis $\boldsymbol\phi(x)=(1,x)^{\mathsf T}$ and observations $(0,1)$, $(1,2)$, and $(2,2)$. At $\mathbf w^{(0)}=(0,0)^{\mathsf T}$, all predictions are zero, so

$$
\begin{aligned}
\mathbf g_1&=(-1)(1,0)^{\mathsf T}=(-1,0)^{\mathsf T},\\
\mathbf g_2&=(-2)(1,1)^{\mathsf T}=(-2,-2)^{\mathsf T},\\
\mathbf g_3&=(-2)(1,2)^{\mathsf T}=(-2,-4)^{\mathsf T}.
\end{aligned}
$$

Their average is $\nabla J(\mathbf w^{(0)})=(-5/3,-2)^{\mathsf T}$. With $\alpha_0=0.1$, one batch update gives

$$
\mathbf w^{(1)}
=(0,0)^{\mathsf T}-0.1(-5/3,-2)^{\mathsf T}
=(1/6,1/5)^{\mathsf T}.
$$

Each update considers all observations, but that can be expensive when the data set is very large. Also, one update is generally **not** the final least-squares solution: we repeat until a suitable stopping condition is met.

## Stochastic gradient descent: use one observation

Suppose new observations arrive one at a time, or we want to update the weights without scanning the entire data set first. **Stochastic gradient descent (SGD)** chooses one training example $i_k$ for update $k$ and uses only its loss gradient:

$$
\boxed{
\mathbf w^{(k+1)}
=\mathbf w^{(k)}
-\alpha_k
\left[(\mathbf w^{(k)})^{\mathsf T}\boldsymbol\phi(\mathbf x_{i_k})-y_{i_k}\right]
\boldsymbol\phi(\mathbf x_{i_k}).
}
$$

Equivalently, we *add* $\alpha_k(y_{i_k}-\widehat y_{i_k})\boldsymbol\phi(\mathbf x_{i_k})$. For squared-error linear regression, this is the **least-mean-squares (LMS) update** discussed in Bishop.

Using the same three observations as above, suppose the first chosen example is $(x_2,y_2)=(1,2)$. At zero weights its gradient is $(-2,-2)^{\mathsf T}$, so with learning rate $0.1$ the first SGD update gives $(0.2,0.2)^{\mathsf T}$. That differs from the batch update: SGD has listened to only one observation so far.

If we select $i_k$ uniformly from the $N$ examples *with replacement*, then at any fixed $\mathbf w$,

$$
\mathbb E_{i_k}[\mathbf g_{i_k}(\mathbf w)]
=\frac1N\sum_{n=1}^N\mathbf g_n(\mathbf w)
=\nabla J(\mathbf w).
$$

Thus a randomly selected example's gradient is an **unbiased estimate** of the full gradient. An individual update can nevertheless point somewhat away from the full gradient, and the full training loss need not decrease after *every* update. In practice, data are often shuffled and processed in order; the equation above states precisely what holds for uniform independent sampling.

One **epoch** means processing every training example once. In a shuffled one-example-at-a-time pass, an epoch contains $N$ updates. The weights change after each update, so the accumulated effect of an epoch is **not** the same as one batch update computed from the initial weights. SGD is useful for large data sets and streams, but its progress is noisier.

## Mini-batch gradient descent: use a small group

We can take a middle course. At update $k$, choose a subset $\mathcal B_k$ of $B$ examples and average their gradients, all evaluated at the same current weights:

$$
\boxed{
\mathbf w^{(k+1)}
=\mathbf w^{(k)}
-\alpha_k\frac1B\sum_{n\in\mathcal B_k}
\left[(\mathbf w^{(k)})^{\mathsf T}\boldsymbol\phi(\mathbf x_n)-y_n\right]
\boldsymbol\phi(\mathbf x_n).
}
$$

The number $B$ is the **mini-batch size**. If $B=1$, this is the single-example update; if $B=N$, it is the full-batch update. Intermediate values usually give a less noisy direction than one example while still allowing more frequent updates than a full pass through the data. Mini-batches can also exploit efficient matrix operations.

For the three-example data set, take the first two observations together at $\mathbf w^{(0)}=\mathbf0$. Their average gradient is $[(-1,0)+(-2,-2)]/2=(-3/2,-1)^{\mathsf T}$. With $\alpha_0=0.1$, the resulting weights are $(0.15,0.10)^{\mathsf T}$. Notice that this is neither the single-example SGD update nor the full-batch update.

If $N$ is divisible by $B$, a pass through all examples in non-overlapping mini-batches has $N/B$ updates. If it is not divisible, the final batch can be smaller; its gradient should be averaged using its *actual* size. The batch size is a choice about computation, not a new probability-model parameter.

## The learning rate controls the step

The learning rate is a **hyperparameter** since it is chosen as part of the fitting procedure rather than estimated as one of the regression weights. A rate that is too small makes progress slow. A rate that is too large can make the iterates jump across the minimum or diverge.

Consider the simple one-weight objective

$$
J(w)=\frac12(w-3)^2,
\qquad
\frac{dJ}{dw}=w-3.
$$

Its minimum is at $w=3$. Starting at $w^{(0)}=0$, gradient descent gives

$$
w^{(k+1)}
=w^{(k)}-\alpha(w^{(k)}-3).
$$

With $\alpha=0.5$, the first two steps are $w^{(1)}=1.5$ and $w^{(2)}=2.25$. Their losses are $J(0)=4.5$, $J(1.5)=1.125$, and $J(2.25)=0.28125$: each step gets closer. With $\alpha=2$, however, the weights alternate between $0$ and $6$, so the loss stays at $4.5$. A rate larger than $2$ would make the distance from $3$ grow for **this particular objective**. There is no universal numerical threshold: suitable rates depend on the curvature and scaling of the problem.

~~~{figure} images/gradient-descent-learning-rate.svg
:name: gradient-descent-learning-rate
:alt: Two plots of the same bowl-shaped loss. Steps with learning rate 0.5 approach its minimum; steps with learning rate 2 jump between equally poor weights on either side.
:width: 900px

Gradient descent on $J(w)=\tfrac12(w-3)^2$. The numbered points mark successive updates. With $\alpha=0.5$ (left), the weights and loss move toward the minimum. With $\alpha=2$ (right), the weights jump back and forth without reducing the loss.
~~~

Feature scale matters here. In the building example, temperature may be measured in tens of degrees, while a squared-temperature basis function can be in the hundreds or thousands. Their weight-gradient components can have very different scales, making one shared learning rate hard to choose. Scaling features often makes optimization easier. As in Lecture 11, compute any centering or scaling statistics on the **training data only**, then apply those same statistics to validation and test inputs.

For batch descent, we can watch the training objective to see whether updates are decreasing it: an increasing or wildly oscillating value suggests the learning rate may be too large. With stochastic or mini-batch descent, individual updates are noisy, so examine the trend over multiple updates or epochs rather than demanding a decrease at each step. A learning-rate schedule may reduce $\alpha_k$ as training proceeds, allowing coarse early progress and smaller later adjustments. As with all hyperparameters, validation performance is used when selecting the learning rate or an early-stopping point, and the test set is never used for tuning.

## Which fitting method should we use?

All four rows below address the **same** least-squares fitting problem. They differ in how the weights are computed, not in the Gaussian model or the meaning of the fitted weights.

| Method | How it uses the data | Good choice when | Main advantage | Main limitation |
| :--- | :--- | :--- | :--- | :--- |
| Direct least-squares solve | Solves using the design matrix rather than taking gradient steps | The matrix is manageable and a numerical solver is available | No learning rate or iteration schedule to tune, obtains a least-squares solution to numerical precision | Can become costly in memory or computation for very large problems |
| Batch gradient descent | Uses all $N$ examples for each update | The data fit the computation available, but an iterative method is desired | Uses the exact training-set gradient, relatively smooth progress | Every update requires a full pass over the training data |
| Stochastic gradient descent | Uses one example per update | Data are very large or arrive as a stream | Cheap, immediate updates, does not need a full pass before the first update | Noisy steps, sensitive to learning rate and data order |
| Mini-batch gradient descent | Uses $B$ examples per update, where $1<B<N$ | The data are large and batches can be processed efficiently | Balances update frequency with a less noisy gradient, supports vectorized computation | Still requires choosing batch size and learning rate, updates remain noisy |

Iterative methods become especially useful when a direct solve is expensive, observations arrive over time, or we are dealing with models without such a closed-form solution.

## Practice problems

### Problem 1: A batch update

Use $\boldsymbol\phi(x)=(1,x)^{\mathsf T}$ and the three observations $(0,1)$, $(1,2)$, and $(2,2)$. Start at $\mathbf w^{(0)}=(0,0)^{\mathsf T}$ and use $\alpha=0.1$.

1. Find the gradient $\mathbf g_n(\mathbf w^{(0)})$ of each individual loss.
2. Average them to obtain $\nabla J(\mathbf w^{(0)})$.
3. Perform one batch update. Does one update necessarily give the least-squares solution?

::::{admonition} Solution 1
:class: dropdown

The prediction at zero weights is $0$ for every example. Since $\mathbf g_n=(\widehat y_n-y_n)\boldsymbol\phi(x_n)$,

$$
\mathbf g_1=(-1,0)^{\mathsf T},\qquad
\mathbf g_2=(-2,-2)^{\mathsf T},\qquad
\mathbf g_3=(-2,-4)^{\mathsf T}.
$$

Their average is

$$
\nabla J(\mathbf w^{(0)})
=\frac13[(-1,0)+(-2,-2)+(-2,-4)]^{\mathsf T}
=(-5/3,-2)^{\mathsf T}.
$$

Therefore $\mathbf w^{(1)}=\mathbf0-0.1(-5/3,-2)^{\mathsf T}=(1/6,1/5)^{\mathsf T}$. This is one step toward a minimum, not necessarily the minimum itself. Lecture 17's exact solution for these observations is $(7/6,1/2)^{\mathsf T}$, so further steps are needed here.
::::

### Problem 2: One example versus a mini-batch

Use the same data and starting weights as Problem 1, with $\alpha=0.1$.

1. Process only the second observation $(1,2)$ with one SGD update. What are the new weights?
2. Instead, make one mini-batch update using the *first two* observations, both evaluated at the original zero weights. What are the new weights?
3. Why are these two updates different? Would processing the first two examples as **two consecutive SGD updates** necessarily equal the mini-batch update?

::::{admonition} Solution 2
:class: dropdown

For the second observation alone, $\mathbf g_2=(-2,-2)^{\mathsf T}$, so the SGD result is $(0.2,0.2)^{\mathsf T}$.

For the mini-batch $\{1,2\}$, average the two gradients at $\mathbf w^{(0)}=\mathbf0$:

$$
\frac{\mathbf g_1+\mathbf g_2}{2}
=\frac{(-1,0)+(-2,-2)}2
=(-1.5,-1)^{\mathsf T}.
$$

One mini-batch update gives $(0.15,0.10)^{\mathsf T}$. It differs from the single-example update because it includes the first observation too. Two consecutive SGD updates also differ because the second gradient is evaluated at weights already changed by the first update. For example, processing observation 1 first gives $\mathbf w=(0.1,0)^{\mathsf T}$; observation 2 then has prediction $0.1$ and gradient $(-1.9,-1.9)^{\mathsf T}$, leaving weights $(0.29,0.19)^{\mathsf T}$.
::::

### Problem 3: What the learning rate changes

For $J(w)=\frac12(w-3)^2$, start at $w^{(0)}=0$.

1. With $\alpha=0.5$, compute $w^{(1)}$, $w^{(2)}$, and the loss at all three weights.
2. Repeat the two updates with $\alpha=2$. What happens to the loss?
3. Explain why the result for $\alpha=2$ does not mean gradient descent itself is unsuitable.

::::{admonition} Solution 3
:class: dropdown

Because $J'(w)=w-3$, the update is $w^{(k+1)}=w^{(k)}-\alpha(w^{(k)}-3)$. With $\alpha=0.5$,

$$
w^{(1)}=0-0.5(0-3)=1.5,\qquad
w^{(2)}=1.5-0.5(1.5-3)=2.25.
$$

The losses are $J(0)=\frac12(9)=4.5$, $J(1.5)=\frac12(1.5^2)=1.125$, and $J(2.25)=\frac12(0.75^2)=0.28125$.

With $\alpha=2$, $w^{(1)}=6$ and $w^{(2)}=6-2(6-3)=0$. The loss is $4.5$ at $0$ and at $6$, so these steps simply bounce across the minimum at $3$. The issue is the chosen learning rate for this objective, not the idea of following a negative gradient: a smaller rate such as $0.5$ does decrease the loss.
::::

### Problem 4: Choosing a method

For each setting, choose a reasonable first approach from the comparison table and justify it briefly.

1. You have a moderate-sized regression data set whose design matrix fits comfortably in memory, and you want its least-squares weights without tuning a learning rate.
2. A sensor delivers labeled observations continuously, and you must update the predictor as each observation arrives.
3. You have a large static data set and can process groups of examples efficiently, but full-data updates are slow.
4. You plan to standardize the inputs before choosing a learning rate. Which data may you use to compute the standardization statistics?

::::{admonition} Solution 4
:class: dropdown

1. Use a direct numerical least-squares solve. It avoids selecting a learning rate and provides a least-squares solution to numerical precision.
2. Use SGD: each new example can produce an immediate update. Revisit the learning rate as the stream continues because the updates may be noisy.
3. Use mini-batch gradient descent. It makes updates more often than batch descent while allowing efficient computation on groups of examples.
4. Compute means and scales using the **training data only**. Apply those same values to validation and test inputs. Using their values to define preprocessing would leak information from the held-out sets into model development.
::::

## Reading

Christopher M. Bishop, *Pattern Recognition and Machine Learning*, Chapter 3, Sections 3.1.1 and 3.1.3, pages 140–144.

## References

- [Christopher M. Bishop, *Pattern Recognition and Machine Learning* (author-hosted PDF)](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf).
- [Stanford CS231n, *Optimization: Stochastic Gradient Descent*](https://cs231n.github.io/optimization-1/).
