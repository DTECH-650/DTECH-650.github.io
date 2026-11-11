# Lecture 21 - Logistic Regression and Generalized Linear Models

In Lecture 20, a linear discriminant assigned a high-demand day to one class and a normal-demand day to the other by checking the sign of a score. The score told the building operator **which** class the model chose, but not how likely a high-demand day was. A score of $2$ was not a probability of $2$, and rescaling the score could leave every decision unchanged. Yet the operator may need to make different plans when high demand has probability $0.51$ than when it has probability $0.95$.

We will keep the useful weighted combination of features, but give it a probabilistic interpretation. Lecture 17 used a Gaussian model for a continuous target, and Lecture 12 showed how the Bernoulli and Gaussian distributions fit a common exponential-family pattern. Here the target is binary, so a Bernoulli model is the natural starting point. We will learn its parameters directly from labeled inputs, without modeling how the inputs themselves were generated.

:::{admonition} Guiding question
:class: tip

How can a linear score become a valid probability for a binary class label, and how does that choice determine the loss used to fit the model?
:::

## Learning outcomes

After completing this lecture, you should be able to:

1. Distinguish a discriminative model for $p(t\mid\mathbf x)$ from a generative model that also describes the inputs.
2. Convert a linear score into a binary class probability using the logistic sigmoid, and recover the score from the log-odds.
3. Write the Bernoulli likelihood and derive its negative log-likelihood (binary cross-entropy) for labeled data.
4. Explain why a logistic-regression decision boundary is linear in its chosen feature space, even though its output is nonlinear in the score.
5. Recognize Gaussian linear regression, logistic regression, and Poisson regression as examples of generalized linear models with different response distributions and links.

## Model the label directly

Let $t=1$ denote a high-demand day and $t=0$ a normal-demand day. The input $\mathbf x$ might contain temperature and expected occupancy. We want a model for

$$
y(\mathbf x)\equiv p(t=1\mid\mathbf x,\mathbf w),
$$

so that $p(t=0\mid\mathbf x,\mathbf w)=1-y(\mathbf x)$. The notation $y(\mathbf x)$ is the model's predicted **probability**, not a measured electricity use. We write $\mathbf w$ explicitly to emphasize that the weights must be learned.

There are two broad routes to class probabilities:

- A **generative** approach models how inputs arise in each class, for example $p(\mathbf x\mid t=1)$ and $p(\mathbf x\mid t=0)$, along with class probabilities. Bayes' rule then gives $p(t\mid\mathbf x)$.
- A **discriminative** approach models $p(t\mid\mathbf x)$ directly. Logistic regression takes this route: it does not require a probability distribution for temperature or occupancy.

Both routes can make predictions; neither is inherently “more Bayesian.” *Discriminative* describes what distribution is modeled, whereas *Bayesian* describes how uncertainty about model parameters is handled. In this lecture the weights are fixed but unknown, as in the frequentist parameter-learning lectures.

## From a score to a probability

Start with the linear score from Lecture 20, now written using the feature vector from Lecture 17:

$$
a(\mathbf x)=\mathbf w^{\mathsf T}\boldsymbol\phi(\mathbf x).
$$

The first feature is usually $\phi_0(\mathbf x)=1$, so $w_0$ acts as a bias. If we use the original inputs, $\boldsymbol\phi(\mathbf x)=(1,x_1,\ldots,x_D)^{\mathsf T}$. Fixed nonlinear basis functions are also possible; as in Lecture 17, their values are determined from $\mathbf x$ before we fit $\mathbf w$.

The score $a$ ranges over all real numbers, but a probability must lie between zero and one. The **logistic sigmoid** makes this conversion:

$$
\boxed{\sigma(a)=\frac{1}{1+e^{-a}},\qquad
y(\mathbf x)=\sigma\!\left(\mathbf w^{\mathsf T}\boldsymbol\phi(\mathbf x)\right).}
$$

When $a=0$, $y=1/2$. Positive scores give $y>1/2$; negative scores give $y<1/2$. For example, scores $-1$, $0$, and $1$ give probabilities of approximately $0.269$, $0.500$, and $0.731$. As $a$ increases without bound, $y$ approaches $1$; as it decreases without bound, $y$ approaches $0$. For finite weights and features it never equals either endpoint exactly.

### Why this particular curve?

The **odds** of high demand are the probability of high demand divided by the probability of normal demand:

$$
\text{odds}(\mathbf x)=\frac{y(\mathbf x)}{1-y(\mathbf x)}.
$$

For the sigmoid, $1-y=1/(1+e^a)$ and $y=e^a/(1+e^a)$, so

$$
\frac{y}{1-y}=e^a,
\qquad
\boxed{\log\frac{y}{1-y}=a=\mathbf w^{\mathsf T}\boldsymbol\phi(\mathbf x).}
$$

Thus logistic regression assumes that the **log-odds**, rather than the probability itself, are linear in the chosen features. This also explains the name *logistic regression*: we regress a transformed probability on features, but the task is **classification**, not prediction of an unbounded continuous target.

If one feature $x_j$ increases by one unit while all others stay fixed, and $\phi_j(\mathbf x)=x_j$, the log-odds change by $w_j$. Equivalently, the odds are multiplied by $e^{w_j}$. This interpretation does not apply so simply if $x_j$ enters through nonlinear or interacting features.

### The decision boundary is still familiar

Suppose we assign high demand when $y(\mathbf x)\geq 1/2$. Because the sigmoid increases with $a$ and $\sigma(0)=1/2$, this rule is equivalent to $a(\mathbf x)\geq0$. The boundary is therefore

$$
\mathbf w^{\mathsf T}\boldsymbol\phi(\mathbf x)=0.
$$

With $\boldsymbol\phi(\mathbf x)=(1,\mathbf x^{\mathsf T})^{\mathsf T}$, it is a hyperplane in the original input space, just as in Lecture 20. With nonlinear basis functions it is still linear in **feature space**, but can be curved in the original input space. A probability threshold other than $1/2$ is possible when the costs of the two errors differ; then the boundary is a different level set of the score. We will use $1/2$ unless stated otherwise.

Unlike the arbitrary score scaling in Lecture 20, changing the size of the logistic-regression weights while keeping the boundary fixed changes the predicted probabilities. The likelihood will therefore care about more than merely classifying each training example correctly.

## A Bernoulli model for each observed label

Recall the Bernoulli PMF from Lecture 4. For an input $\mathbf x$ with predicted high-demand probability $y(\mathbf x)$, the model assigns

$$
p(t\mid\mathbf x,\mathbf w)
=y(\mathbf x)^t[1-y(\mathbf x)]^{1-t},
\qquad t\in\{0,1\}.
$$

If $t=1$, this expression is $y(\mathbf x)$; if $t=0$, it is $1-y(\mathbf x)$. That is exactly what a probability model for a binary label must do.

Let $\mathcal D=\{(\mathbf x_n,t_n)\}_{n=1}^{N}$ be the labeled training set, and abbreviate

$$
\boldsymbol\phi_n=\boldsymbol\phi(\mathbf x_n),
\qquad a_n=\mathbf w^{\mathsf T}\boldsymbol\phi_n,
\qquad y_n=\sigma(a_n).
$$

Assuming the labels are conditionally independent given their inputs and the shared weights, their conditional likelihood is the product of their Bernoulli probabilities:

$$
\boxed{p(\mathbf t\mid\mathbf X,\mathbf w)
=\prod_{n=1}^{N}y_n^{t_n}(1-y_n)^{1-t_n}.}
$$

The inputs $\mathbf X=(\mathbf x_1,\ldots,\mathbf x_N)$ are treated as given. The model does **not** claim that all labels have the same success probability: $y_n$ can differ from day to day because the inputs differ. Their probabilities are tied together by the same $\mathbf w$.

### From likelihood to cross-entropy

As in Lecture 7, maximizing the likelihood is equivalent to minimizing its negative logarithm. The logarithm converts the product above into a sum:

$$
\log p(\mathbf t\mid\mathbf X,\mathbf w)
=\sum_{n=1}^{N}\left[t_n\log y_n+(1-t_n)\log(1-y_n)\right].
$$

Hence the **negative log-likelihood**, also called **binary cross-entropy** for these observed labels, is

$$
\boxed{E(\mathbf w)
=-\sum_{n=1}^{N}\left[t_n\log y_n+(1-t_n)\log(1-y_n)\right].}
$$

For one observation, a correct label assigned a large probability has a small loss. If $t=1$, the loss is $-\log y$: predicting $y=0.8$ costs about $0.223$ nats, whereas predicting $y=0.2$ costs about $1.609$ nats. A confident wrong prediction is penalized strongly. Dividing $E$ by $N$ gives the **mean** cross-entropy; it changes the scale of the loss but not its minimizer.

The loss is not an arbitrary replacement for least squares. It follows from choosing a Bernoulli distribution for a binary target. In Lecture 17, a Gaussian distribution for a continuous target led instead to squared error. The model for the target determines the likelihood and, through it, the fitting objective.

### A small likelihood calculation

Suppose three days have observed labels $(1,0,1)$ and a particular choice of weights gives probabilities $(0.8,0.3,0.6)$. The likelihood of these labels, conditional on the inputs, is

$$
0.8\,(1-0.3)\,0.6=0.336.
$$

Its negative log-likelihood is $-\log(0.336)\approx1.091$ nats. The probability $0.3$ contributes $1-0.3=0.7$ because the second day's observed label is $0$. Fitting the weights means finding probabilities that make the *whole observed label sequence* likely, not fitting each day with an unrelated parameter.

## How the weights respond to errors

The sigmoid has a convenient derivative. Differentiating $\sigma(a)=(1+e^{-a})^{-1}$ gives

$$
\frac{d\sigma(a)}{da}
=\frac{e^{-a}}{(1+e^{-a})^2}
=\sigma(a)[1-\sigma(a)].
$$

For one training example, write its loss as $\ell_n=-t_n\log y_n-(1-t_n)\log(1-y_n)$. The chain rule gives

$$
\frac{d\ell_n}{da_n}
=\left(-\frac{t_n}{y_n}+\frac{1-t_n}{1-y_n}\right)
\frac{dy_n}{da_n}
=\left(-\frac{t_n}{y_n}+\frac{1-t_n}{1-y_n}\right)y_n(1-y_n)
=y_n-t_n.
$$

Because $a_n=\mathbf w^{\mathsf T}\boldsymbol\phi_n$, we have $\nabla_{\mathbf w}a_n=\boldsymbol\phi_n$. Adding the contributions from all observations yields

$$
\boxed{\nabla_{\mathbf w}E(\mathbf w)
=\sum_{n=1}^{N}(y_n-t_n)\boldsymbol\phi_n.}
$$

This resembles the residual-times-feature gradient for squared error, although the predicted $y_n$ and the loss are different. If a positive example ($t_n=1$) gets a low predicted probability, then $y_n-t_n<0$: a gradient-descent step tends to increase its score in the direction of its feature vector. The full gradient balances all examples, so an individual update need not improve every prediction.

Unlike Gaussian linear regression with a fixed design matrix, setting this gradient to zero does not produce a simple closed-form expression for $\mathbf w$, because each $y_n=\sigma(\mathbf w^{\mathsf T}\boldsymbol\phi_n)$ depends nonlinearly on the unknown weights. We can use gradient-based optimization, as in Lecture 18; the next lecture will develop Newton's method and iteratively reweighted least squares (IRLS).

### When the classes separate perfectly

There is a subtle limit to unregularized maximum likelihood. If some hyperplane separates all positive training labels from all negative ones, multiplying its weights by larger positive numbers pushes the predicted probabilities toward $1$ for positive examples and $0$ for negative examples. The training loss approaches zero, but no **finite** weight vector reaches that limit. A numerical solver may report very large coefficients rather than a stable finite maximum-likelihood estimate.

Regularization—equivalently, under appropriate conditions, a prior on the weights—can keep the fitted coefficients finite. This is a property of the *fitting problem*, not a reason to abandon the probabilistic interpretation of logistic regression.

## Logistic regression as a generalized linear model

At first, linear regression from Lecture 17 and logistic regression from this lecture may look like unrelated methods: one predicts a continuous quantity and the other predicts a class probability. **Both are generalized linear models (GLMs)** because both can be written using the same three-part recipe. First form a linear score from the features; next convert that score into the target's conditional mean; finally choose a probability distribution for the target with that mean:

$$
\boxed{
a(\mathbf x)=\mathbf w^{\mathsf T}\boldsymbol\phi(\mathbf x),
\qquad
\mu(\mathbf x)=f(a(\mathbf x)),
\qquad
T\mid\mathbf x,\mathbf w\sim\text{a distribution with mean }\mu(\mathbf x).
}
$$

Here $T$ is the random target and $t$ is its observed value; this $T$ is not the sufficient-statistic function $T(x)$ from Lecture 12. The function $f$ is the **inverse link**. Its inverse $g=f^{-1}$ is the **link function**, so $g(\mu(\mathbf x))=a(\mathbf x)$. “Linear” in GLM refers to the score $a$, not necessarily to $\mu$ as a function of the inputs.

Now write our two familiar models in that *same* template:

| Model | Linear score $a(\mathbf x)$ | Mean $\mu(\mathbf x)=f(a)$ | Conditional distribution of the target |
| --- | --- | --- | --- |
| **Linear regression** (Lecture 17) | $\mathbf w^{\mathsf T}\boldsymbol\phi(\mathbf x)$ | $a(\mathbf x)$ (identity inverse link) | $T\mid\mathbf x,\mathbf w\sim\mathcal N(a(\mathbf x),\beta^{-1})$ |
| **Logistic regression** (this lecture) | $\mathbf w^{\mathsf T}\boldsymbol\phi(\mathbf x)$ | $\sigma(a(\mathbf x))$ (sigmoid inverse link) | $T\mid\mathbf x,\mathbf w\sim\operatorname{Bernoulli}(\sigma(a(\mathbf x)))$ |

The notation $\beta^{-1}$ is the fixed Gaussian noise variance from Lecture 17. In either row, the weights enter through exactly the same kind of linear score. What changes is the **target distribution** and the function that turns the score into its mean. Gaussian linear regression uses $\mu=a$ and can predict any real mean. Logistic regression uses $\mu=\sigma(a)$, which is constrained to be a valid probability. That is why their likelihoods—and hence their losses, squared error and binary cross-entropy—are different, even though their scores look alike.

Lecture 12 introduced the exponential-family distributions behind both rows. Gaussian, Bernoulli, and Poisson targets take different kinds of values, but each can be used in the same GLM recipe. For Bernoulli data, $\mu(\mathbf x)=p(T=1\mid\mathbf x,\mathbf w)=y(\mathbf x)$. Logistic regression's link is the **logit**,

$$
g(\mu)=\log\frac{\mu}{1-\mu},
\qquad
g^{-1}(a)=\sigma(a).
$$

The link is necessary because the mean of a binary variable must be between $0$ and $1$, whereas the score can be any real number. In the linear-regression row, the identity link $g(\mu)=\mu$ needs no such transformation. A third example is **Poisson regression** for a count target: its log link $g(\mu)=\log\mu$ gives $\mu=e^a>0$, so the expected count stays positive. Gaussian, Bernoulli, and Poisson targets therefore use different inverse links—identity, sigmoid, and exponential—while retaining the same linear score.

These links are called **canonical** for the respective exponential-family models. To see what that means without introducing a new parameter convention, recall Lecture 12's exponential-family form and write a one-statistic version with an optional positive scale $s$:

$$
p(t\mid\eta,s)
=h(t,s)\exp\!\left(\frac{t\eta-A(\eta)}{s}\right).
$$

Here $\eta$ is the **natural parameter** and $A(\eta)$ is the log normalizer. A canonical link chooses the linear score itself as the natural parameter: $\eta=a(\mathbf x)$. Differentiating the normalization relation, as in Lecture 12, gives $\mathbb E[T\mid\eta]=A'(\eta)$, so the mean is $\mu=A'(a)$ and the link reverses that relationship. For Bernoulli targets, $s=1$, $\eta=\log[\mu/(1-\mu)]$, and $A(\eta)=\log(1+e^\eta)$; therefore

$$
A'(\eta)=\frac{e^\eta}{1+e^\eta}=\sigma(\eta)=\mu.
$$

Setting $\eta=a$ recovers logistic regression. For a Gaussian with fixed variance $s=\sigma^2$, we can use $\eta=\mu$ and $A(\eta)=\eta^2/2$; then $A'(\eta)=\eta$, giving the identity link. For a Poisson target, $s=1$ and $A(\eta)=e^\eta$; then $A'(\eta)=e^\eta$, giving the log link. The same architecture handles different target types, but **changing the target distribution also changes the likelihood and loss**. We do not apply squared error to a Bernoulli target merely because both models use a linear score.

We will not develop multiclass logistic regression or softmax in this course. Our classification model here has exactly two classes. The GLM viewpoint is useful even with that restriction: it connects the Gaussian regression of Lecture 17, the exponential families of Lecture 12, and the binary classifier of this lecture.

## Practice problems

### Problem 1: Scores, probabilities, and decisions

For $a(x)=-2+x$, let $y(x)=\sigma(a(x))$ and assign class $t=1$ when $y(x)\geq0.5$.

1. Calculate $y(1)$ and $y(3)$ to three decimal places, and classify each input.
2. Find the decision boundary.
3. If the decision threshold changes to $0.8$, what score threshold replaces zero?

::::{admonition} Solution 1
:class: dropdown

At $x=1$, $a=-1$ and $y=1/(1+e)\approx0.269$, so the model chooses $t=0$. At $x=3$, $a=1$ and $y=1/(1+e^{-1})\approx0.731$, so it chooses $t=1$ under the $0.5$ rule. The $0.5$ boundary occurs at $a=0$, or $x=2$. For a probability threshold of $0.8$, solve $\sigma(a)=0.8$: $a=\log[0.8/(1-0.8)]=\log4\approx1.386$. The new input threshold is $x=2+\log4\approx3.386$.
::::

### Problem 2: A Bernoulli likelihood

Three binary labels are $\mathbf t=(1,0,1)$. A model predicts $\mathbf y=(0.7,0.2,0.4)$ for the corresponding inputs.

1. Write the probability assigned to each observed label.
2. Find the conditional likelihood of the whole label sequence.
3. Find the total negative log-likelihood in nats, rounded to three decimals.

::::{admonition} Solution 2
:class: dropdown

The three observed-label probabilities are $0.7$, $1-0.2=0.8$, and $0.4$. Conditional independence gives $p(\mathbf t\mid\mathbf X,\mathbf w)=0.7(0.8)(0.4)=0.224$. Thus the total negative log-likelihood is $-\log(0.224)\approx1.496$ nats. We do not divide by $3$ because the question asks for the **total**, not the mean.
::::

### Problem 3: One observation's gradient

Let $\boldsymbol\phi=(1,2)^{\mathsf T}$, $\mathbf w=(0,0)^{\mathsf T}$, and the observed label be $t=1$.

1. Find the score $a$ and predicted probability $y$.
2. Calculate $\nabla_{\mathbf w}\ell=(y-t)\boldsymbol\phi$.
3. Take one gradient-descent step with learning rate $0.1$. What is the new probability for this same input, approximately?

::::{admonition} Solution 3
:class: dropdown

The initial score is $a=0$, so $y=\sigma(0)=0.5$. The gradient is $(0.5-1)(1,2)^{\mathsf T}=(-0.5,-1)^{\mathsf T}$. A gradient-descent step subtracts $0.1$ times this gradient, giving $\mathbf w_{\mathrm{new}}=(0.05,0.1)^{\mathsf T}$. For the same input the new score is $0.05+2(0.1)=0.25$, and the new probability is $\sigma(0.25)\approx0.562$. This one step raises the probability assigned to the observed positive label; it does not guarantee improvement for every other example.
::::

### Problem 4: Match the GLM to the target

A model must predict (i) a real-valued energy use, (ii) whether a machine fails during the next day, and (iii) the number of service calls received next week. Choose a Gaussian, Bernoulli, or Poisson conditional model for each target. Give its canonical link and explain briefly why applying the logistic sigmoid to all three scores would be inappropriate.

::::{admonition} Solution 4
:class: dropdown

Real-valued energy use can be modeled with a Gaussian conditional distribution and identity link, $\mu=a$. Failure/no failure is binary, so a Bernoulli model with logit link gives $\mu=\sigma(a)$, a valid probability of failure. Service-call counts are nonnegative integers, so a Poisson model with log link gives the positive expected count $\mu=e^a$. A sigmoid always outputs between zero and one; it cannot represent an unrestricted real-valued mean or an expected count greater than one. The distributions also assign different likelihoods to observations, so their fitting losses differ.
::::

## Reading

Christopher M. Bishop, *Pattern Recognition and Machine Learning*, Chapter 4, pages 203–206 and 212–213 (through Section 4.3.6).

## References

- [Christopher M. Bishop, *Pattern Recognition and Machine Learning* (author-hosted PDF)](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf).
