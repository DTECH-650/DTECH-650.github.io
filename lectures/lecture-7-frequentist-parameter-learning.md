# Lecture 7 - Frequentist Parameter Learning

The distributions introduced earlier contain parameters such as a Bernoulli success probability $\mu$, a Gaussian mean $\mu$, or a Gaussian variance $\sigma^2$. Until now, those parameters have usually been given. We now reverse the problem: the data are observed, but the parameter values that generated them are unknown.

This lecture develops the frequentist approach to parameter learning. We distinguish probability from likelihood, use the independent and identically distributed assumption to factor a data-set likelihood, introduce plate notation, and derive maximum-likelihood estimates for Bernoulli and Gaussian models.
<!-- 
## Learning objectives

After completing this lecture, you should be able to:

- explain what it means to treat a parameter as fixed but unknown;
- distinguish a probability distribution from a likelihood function;
- factor an IID data-set likelihood and interpret its plate diagram;
- explain why log-likelihood is usually easier to work with than likelihood;
- derive the Bernoulli maximum-likelihood estimate; and
- derive the Gaussian maximum-likelihood estimates of the mean and variance. -->

## From a probability model to parameter learning

A **parametric model** is a family of distributions indexed by one or more parameters:

$$
\left\{p(x\mid\boldsymbol{\theta}):
\boldsymbol{\theta}\in\Theta\right\}.
$$

The parameter vector $\boldsymbol{\theta}$ determines which member of the family is used. Examples include

$$
X\sim\operatorname{Bernoulli}(\mu),
\qquad 0\leq\mu\leq 1,
$$

and

$$
X\sim\mathcal{N}(\mu,\sigma^2),
\qquad \mu\in\mathbb{R},\quad \sigma^2>0.
$$

In **frequentist parameter learning**, the true parameter $\boldsymbol{\theta}$ is treated as a fixed constant. It has one value, but we do not know that value. The observations are random before data collection because another sample from the same process could contain different values.

Suppose

$$
X_1,X_2,\ldots,X_N
$$

are random variables and we observe the realized data

$$
\mathcal{D}=\{x_1,x_2,\ldots,x_N\}.
$$

An **estimator** is a rule applied to the random sample,

$$
\widehat{\boldsymbol{\theta}}
=g(X_1,\ldots,X_N).
$$

Before the sample is collected, the estimator is random because its inputs are random. After the observed values are inserted, it produces an **estimate**

$$
\widehat{\boldsymbol{\theta}}(\mathcal{D})
=g(x_1,\ldots,x_N),
$$

which is a fixed numerical value for that data set.

:::{admonition} Parameter, estimator, and estimate
:class: important

- $\boldsymbol{\theta}$ is the fixed but unknown parameter.
- $\widehat{\boldsymbol{\theta}}$ is a statistic whose value would vary across repeated samples.
- $\widehat{\boldsymbol{\theta}}(\mathcal{D})$ is the value calculated from the sample actually observed.
:::

Frequentist uncertainty about an estimate comes from asking how the estimator would behave across hypothetical repetitions of the data-generating process. We do not need to assign a probability distribution to $\boldsymbol{\theta}$ itself in order to study that sampling behavior.

## Probability versus likelihood

The same expression $p(x\mid\boldsymbol{\theta})$ can be viewed in two different ways.

| View | Held fixed | Allowed to vary | Question answered |
|---|---|---|---|
| **Probability distribution** | $\boldsymbol{\theta}$ | Possible data $x$ | What outcomes could occur, and how probable are they under this parameter value? |
| **Likelihood function** | Observed data $x$ | Candidate parameters $\boldsymbol{\theta}$ | Which parameter values make these observed data more or less plausible? |

For a complete observed data set, the **likelihood function** is

$$
\mathcal{L}(\boldsymbol{\theta};\mathcal{D})
=p(\mathcal{D}\mid\boldsymbol{\theta}),
$$

viewed as a function of $\boldsymbol{\theta}$. The semicolon is a reminder that the data are fixed rather than an additional conditioning operation.

### A one-observation example

For a Bernoulli variable,

$$
p(x\mid\mu)=\mu^x(1-\mu)^{1-x},
\qquad x\in\{0,1\}.
$$

If $\mu=0.7$ is fixed before the trial, then

$$
P(X=1\mid\mu=0.7)=0.7
$$

is a probability statement about a possible observation. If we instead observe $x=1$, then

$$
\mathcal{L}(\mu;x=1)=\mu,
\qquad 0\leq\mu\leq1,
$$

is a likelihood function comparing candidate values of $\mu$.

The likelihood is **not** a probability distribution over $\mu$. For example,

$$
\int_0^1 \mathcal{L}(\mu;x=1)\,d\mu
=\int_0^1\mu\,d\mu
=\frac{1}{2},
$$

not 1. A likelihood need not be normalized over the parameter space. It tells us how parameter values compare in their support for the observed data.

:::{admonition} A common error
:class: warning

$p(\mathcal{D}\mid\boldsymbol{\theta})$ is not the same quantity as $p(\boldsymbol{\theta}\mid\mathcal{D})$. Reversing the conditioning requires additional assumptions and, in a Bayesian analysis, a prior distribution. Maximum likelihood uses $p(\mathcal{D}\mid\boldsymbol{\theta})$ directly.
:::

For continuous data, $p(x\mid\boldsymbol{\theta})$ is a density value rather than the probability of observing exactly $x$. It still defines a valid likelihood after $x$ has been observed.

## IID observations and likelihood factorization

A common starting assumption is that, conditional on the parameter, the observations are **independent and identically distributed**, abbreviated **IID**:

$$
X_n\mid\boldsymbol{\theta}
\overset{\mathrm{iid}}{\sim}
p(x\mid\boldsymbol{\theta}),
\qquad n=1,\ldots,N.
$$

The two parts of this statement have distinct meanings:

- **Identically distributed:** every $X_n$ uses the same distributional form and the same parameter $\boldsymbol{\theta}$.
- **Independent:** after conditioning on $\boldsymbol{\theta}$, knowing one observation provides no information about the others.

Independence lets the joint distribution factor into a product:

$$
p(x_1,\ldots,x_N\mid\boldsymbol{\theta})
=\prod_{n=1}^{N}p(x_n\mid\boldsymbol{\theta}).
$$

Therefore, the IID likelihood is

$$
\mathcal{L}(\boldsymbol{\theta};\mathcal{D})
=\prod_{n=1}^{N}p(x_n\mid\boldsymbol{\theta}).
$$

### Plate notation

A **plate** is a compact graphical notation for repeated variables. Instead of drawing $N$ separate observation nodes, a rectangle encloses one representative node $x_n$, and the label $n=1,\ldots,N$ states how many times it is repeated.

```{figure} figures/iid-likelihood-plate.svg
---
name: iid-likelihood-plate
alt: An unknown parameter theta outside a plate points to a shaded observed variable x sub n inside a plate labeled n equals 1 through N.
width: 80%
---
The shared parameter $\boldsymbol{\theta}$ controls every observed variable $x_n$. The plate represents $N$ conditionally independent repetitions and therefore the product $\prod_{n=1}^{N}p(x_n\mid\boldsymbol{\theta})$.
```

In this diagram:

- the rectangle is the plate;
- the shaded node indicates an observed variable;
- the arrow means that the distribution of $X_n$ depends on $\boldsymbol{\theta}$; and
- $\boldsymbol{\theta}$ is outside the plate because one parameter value is shared by every observation.

The diagram records an assumption, not an empirical guarantee. Repeated measurements from the same person, measurements close together in time, or observations from connected systems may not be independent. Incorrectly treating dependent data as IID can make an analysis appear more certain than it should.

## Maximum likelihood estimation

The **maximum-likelihood estimate**, or MLE, is the allowed parameter value that maximizes the likelihood:

$$
\widehat{\boldsymbol{\theta}}_{\mathrm{ML}}
=\underset{\boldsymbol{\theta}\in\Theta}{\operatorname{arg\,max}}
\;\mathcal{L}(\boldsymbol{\theta};\mathcal{D}).
$$

It answers:

> Within the chosen model family, which parameter value makes the observed data most likely?

This does not mean the selected parameter is "the probability of being correct," nor does it prove that the model family itself is appropriate.

## The log-likelihood

An IID likelihood multiplies many probability terms, which can thus too small for finite-precision computation. We therefore take the logarithm:

$$
\ell(\boldsymbol{\theta};\mathcal{D})
=\log\mathcal{L}(\boldsymbol{\theta};\mathcal{D}).
$$

Because the logarithm converts products into sums,

$$
\ell(\boldsymbol{\theta};\mathcal{D})
=\log\prod_{n=1}^{N}p(x_n\mid\boldsymbol{\theta})
=\sum_{n=1}^{N}\log p(x_n\mid\boldsymbol{\theta}).
$$

The logarithm is strictly increasing, so it preserves the ordering of positive likelihood values:

$$
\underset{\boldsymbol{\theta}}{\operatorname{arg\,max}}
\;\mathcal{L}(\boldsymbol{\theta};\mathcal{D})
=
\underset{\boldsymbol{\theta}}{\operatorname{arg\,max}}
\;\ell(\boldsymbol{\theta};\mathcal{D}).
$$

Thus, maximizing log-likelihood gives the same MLE while providing three practical advantages:

- sums are easier to differentiate than products;
- sums are less vulnerable to numerical underflow; and
- terms that do not depend on the parameter are easy to identify and omit during optimization.

For a differentiable scalar parameter, an interior candidate often satisfies the **score equation**

$$
\frac{\partial}{\partial\theta}
\ell(\theta;\mathcal{D})=0.
$$

The parameter constraints and boundary values must also be checked. A zero derivative identifies a stationary point, not automatically a valid maximum.

## Bernoulli maximum likelihood

Let $X_1,\ldots,X_N$ be IID Bernoulli observations with an unknown success probability $\mu$:

$$
X_n\mid\mu\overset{\mathrm{iid}}{\sim}
\operatorname{Bernoulli}(\mu),
\qquad 0\leq\mu\leq1.
$$

The PMF of one observation is

$$
p(x_n\mid\mu)
=\mu^{x_n}(1-\mu)^{1-x_n}.
$$

Let

$$
S=\sum_{n=1}^{N}x_n
$$

be the number of observed successes. The IID likelihood factors as

$$
\begin{aligned}
\mathcal{L}(\mu;\mathcal{D})
&=\prod_{n=1}^{N}
\mu^{x_n}(1-\mu)^{1-x_n}\\
&=\mu^S(1-\mu)^{N-S}.
\end{aligned}
$$

For $0<\mu<1$, the log-likelihood is

$$
\ell(\mu;\mathcal{D})
=S\log\mu+(N-S)\log(1-\mu).
$$

Differentiate and set the result to zero:

$$
\frac{d\ell}{d\mu}
=\frac{S}{\mu}-\frac{N-S}{1-\mu}=0.
$$

Multiplying by $\mu(1-\mu)$ and rearranging gives

$$
S(1-\mu)-(N-S)\mu=0,
$$

so

$$
\boxed{
\widehat{\mu}_{\mathrm{ML}}
=\frac{S}{N}
=\frac{1}{N}\sum_{n=1}^{N}x_n
=\overline{x}
}.
$$

The Bernoulli MLE is the observed fraction of successes. For $0<S<N$, the second derivative

$$
\frac{d^2\ell}{d\mu^2}
=-\frac{S}{\mu^2}
-\frac{N-S}{(1-\mu)^2}
$$

is negative, confirming a maximum.

### Worked example: quality inspection

Suppose eight independently sampled components produce the outcomes

$$
1,1,0,1,0,1,1,0,
$$

where 1 means the component passes inspection. Here $S=5$ and $N=8$, so

$$
\widehat{\mu}_{\mathrm{ML}}=\frac{5}{8}=0.625.
$$

Within the IID Bernoulli model, the maximum-likelihood estimate of the pass probability is $62.5\%$.

If only the count $S$ is recorded, then $S$ has a Binomial distribution and its likelihood contains the additional factor

$$
\binom{N}{S}\mu^S(1-\mu)^{N-S}.
$$

The combinatorial factor does not depend on $\mu$, so it changes the likelihood's scale but not its maximizing value. The Bernoulli-sequence and Binomial-count views therefore produce the same MLE.

### Boundary cases

If every observation is 1, then $S=N$ and the likelihood is maximized at $\widehat{\mu}_{\mathrm{ML}}=1$. If every observation is 0, it is maximized at $\widehat{\mu}_{\mathrm{ML}}=0$. These estimates fit the observed sample perfectly but assign zero probability to the opposite outcome. With a small sample, this can produce overconfident predictions and illustrates a limitation of unregularized maximum likelihood.

## Gaussian maximum likelihood

Now suppose real-valued observations are modeled as IID Gaussian variables with both mean and variance unknown:

$$
X_n\mid\mu,\sigma^2
\overset{\mathrm{iid}}{\sim}
\mathcal{N}(\mu,\sigma^2),
\qquad \mu\in\mathbb{R},\quad\sigma^2>0.
$$

The likelihood is

$$
\mathcal{L}(\mu,\sigma^2;\mathcal{D})
=\prod_{n=1}^{N}
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left\{-\frac{(x_n-\mu)^2}{2\sigma^2}\right\}.
$$

Taking logarithms gives

$$
\ell(\mu,\sigma^2;\mathcal{D})
=-\frac{N}{2}\log(2\pi)
-\frac{N}{2}\log\sigma^2
-\frac{1}{2\sigma^2}
\sum_{n=1}^{N}(x_n-\mu)^2.
$$

### Estimating the mean

Differentiate with respect to $\mu$:

$$
\frac{\partial\ell}{\partial\mu}
=\frac{1}{\sigma^2}
\sum_{n=1}^{N}(x_n-\mu).
$$

Setting this derivative to zero gives

$$
\sum_{n=1}^{N}x_n-N\mu=0,
$$

and therefore

$$
\boxed{
\widehat{\mu}_{\mathrm{ML}}
=\frac{1}{N}\sum_{n=1}^{N}x_n
=\overline{x}
}.
$$

The Gaussian maximum-likelihood estimate of the mean is the sample mean.

### Estimating the variance

Let $v=\sigma^2$. Differentiating the log-likelihood with respect to $v$ gives

$$
\frac{\partial\ell}{\partial v}
=-\frac{N}{2v}
+\frac{1}{2v^2}
\sum_{n=1}^{N}(x_n-\mu)^2.
$$

Setting this derivative to zero and substituting $\widehat{\mu}_{\mathrm{ML}}=\overline{x}$ gives

$$
\boxed{
\widehat{\sigma}_{\mathrm{ML}}^2
=\frac{1}{N}
\sum_{n=1}^{N}(x_n-\overline{x})^2
}.
$$

The corresponding standard-deviation estimate is

$$
\widehat{\sigma}_{\mathrm{ML}}
=\sqrt{\widehat{\sigma}_{\mathrm{ML}}^2}.
$$

### Worked example: repeated measurements

Suppose four measurements are

$$
9.8,\quad10.1,\quad10.4,\quad9.7.
$$

Their Gaussian mean MLE is

$$
\widehat{\mu}_{\mathrm{ML}}
=\frac{9.8+10.1+10.4+9.7}{4}
=10.0.
$$

The squared deviations from 10.0 sum to

$$
(-0.2)^2+(0.1)^2+(0.4)^2+(-0.3)^2
=0.30.
$$

Thus,

$$
\widehat{\sigma}_{\mathrm{ML}}^2
=\frac{0.30}{4}=0.075,
$$

and

$$
\widehat{\sigma}_{\mathrm{ML}}
=\sqrt{0.075}\approx0.274.
$$

### Multivariate extension

For IID vectors $\mathbf{x}_n\in\mathbb{R}^D$ modeled by a multivariate Gaussian, the MLEs have the same form:

$$
\widehat{\boldsymbol{\mu}}_{\mathrm{ML}}
=\frac{1}{N}\sum_{n=1}^{N}\mathbf{x}_n,
$$

and

$$
\widehat{\boldsymbol{\Sigma}}_{\mathrm{ML}}
=\frac{1}{N}\sum_{n=1}^{N}
(\mathbf{x}_n-\widehat{\boldsymbol{\mu}}_{\mathrm{ML}})
(\mathbf{x}_n-\widehat{\boldsymbol{\mu}}_{\mathrm{ML}})^{\mathsf T}.
$$

The estimated mean is the sample centroid, and the estimated covariance records the spread and linear co-variation around it. When the sample is small relative to the dimension, the covariance estimate may be singular or unstable, again showing that a maximum-likelihood formula does not guarantee a useful estimate.

## What the two examples have in common

| Model | Unknown parameter | Data summary needed for the MLE | Maximum-likelihood estimate |
|---|---|---|---|
| Bernoulli | Success probability $\mu$ | Number of successes $S$ and sample size $N$ | $S/N=\overline{x}$ |
| Gaussian | Mean $\mu$ | Sum of observations | $\overline{x}$ |
| Gaussian | Variance $\sigma^2$ | Squared deviations from $\overline{x}$ | $N^{-1}\sum_n(x_n-\overline{x})^2$ |

In each case, the model connects an unknown population quantity to observable sample summaries. Maximum likelihood then chooses the parameter values under which those summaries are most compatible with the assumed distribution.

## Assumptions and limitations

Maximum likelihood is powerful, but its answer is conditional on the model and data-collection assumptions:

- **Distributional form:** Bernoulli data must genuinely represent binary trials; a Gaussian model may be unsuitable for strongly skewed, bounded, multimodal, or heavy-tailed measurements.
- **Independence:** dependence among observations invalidates the simple product factorization.
- **Identical distribution:** changes across time, groups, or environments may require different parameters or a richer model.
- **Finite samples:** an MLE can be unstable or located at a boundary when little data are available.
- **Point estimation:** an MLE provides a best-fitting parameter value but does not by itself quantify uncertainty about that estimate.

Model checking and knowledge of how the data were collected are therefore as important as differentiating the log-likelihood.

## A reusable maximum-likelihood workflow

1. Specify the observation model $p(x\mid\boldsymbol{\theta})$ and its parameter constraints.
2. State the sampling assumptions, including whether the observations are IID.
3. Write the joint likelihood $p(\mathcal{D}\mid\boldsymbol{\theta})$.
4. Take its logarithm and remove only terms that are constant with respect to $\boldsymbol{\theta}$.
5. Maximize over the allowed parameter space, checking stationary points and boundaries.
6. Interpret the estimate in the context of the model and examine whether the assumptions are plausible.

## Summary

- Frequentist parameter learning treats the parameter as fixed but unknown and the data as the outcome of a repeatable random process.
- Probability fixes the parameter and varies possible data; likelihood fixes the observed data and varies candidate parameter values.
- Under an IID assumption, the data-set likelihood factors into a product of one-observation distributions. Plate notation represents this repetition compactly.
- The log-likelihood converts the IID product into a sum without changing the maximizer.
- The Bernoulli MLE is the observed fraction of successes.
- The Gaussian MLEs are the sample mean and the average squared deviation from that mean.
- Every MLE depends on the selected model and sampling assumptions.

## Practice problems

### 1. Bernoulli likelihood

An online service independently records whether each of 12 requests finishes within its latency target. Nine requests meet the target.

1. Write the likelihood for the unknown success probability $\mu$.
2. Write the log-likelihood.
3. Find $\widehat{\mu}_{\mathrm{ML}}$.
4. Explain what would change if only the total number of successful requests, rather than the ordered sequence, were recorded.

```{admonition} Solution
:class: dropdown

With $S=9$ successes and $N-S=3$ failures, the likelihood of the observed sequence is

$$
\mathcal{L}(\mu;\mathcal{D})
=\mu^9(1-\mu)^3.
$$

The log-likelihood is

$$
\ell(\mu;\mathcal{D})
=9\log\mu+3\log(1-\mu).
$$

The Bernoulli MLE is the observed success fraction:

$$
\widehat{\mu}_{\mathrm{ML}}
=\frac{9}{12}=0.75.
$$

If only the count is observed, the Binomial likelihood includes $\binom{12}{9}$. This factor is constant with respect to $\mu$, so the MLE remains $0.75$.
```

### 2. Gaussian mean and variance

Assume the four observations

$$
2,\quad4,\quad4,\quad6
$$

are IID Gaussian with both $\mu$ and $\sigma^2$ unknown.

1. Find $\widehat{\mu}_{\mathrm{ML}}$.
2. Find $\widehat{\sigma}_{\mathrm{ML}}^2$.

```{admonition} Solution
:class: dropdown

The sample mean is

$$
\widehat{\mu}_{\mathrm{ML}}
=\overline{x}
=\frac{2+4+4+6}{4}=4.
$$

The squared deviations are $4,0,0,4$, whose sum is 8. Therefore,

$$
\widehat{\sigma}_{\mathrm{ML}}^2
=\frac{8}{4}=2.
$$

```

### 3. Diagnosing an IID assumption

A company records one measurement every second from the same machine and models 10,000 consecutive measurements as IID Gaussian observations.

1. What two distinct claims does the IID assumption make?
2. Give one reason the independence claim may fail.
3. Does drawing a plate diagram prove that the measurements are IID?

```{admonition} Solution
:class: dropdown

The assumption states that all measurements have the same Gaussian distribution with shared parameters and that, conditional on those parameters, the measurements are independent.

Independence may fail because measurements close together in time can share operating conditions or exhibit autocorrelation. A plate diagram records the assumed factorization; it does not establish that the assumption is true. The dependence structure must be justified and checked using knowledge of the process and appropriate diagnostics.
```

## Reading

Christopher M. Bishop, *Pattern Recognition and Machine Learning*, Chapter 1, pages 24–32, and Chapter 2, pages 68–70.

[Official book page and free PDF](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/)
