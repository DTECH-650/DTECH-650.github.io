# Lecture 13 - Bayesian Inference Foundations

Lecture 7 estimated an unknown parameter by choosing a single value that maximized the likelihood following the frequenciest approach. 

In lecture 9, we made an assumptions grounded in a frequentist perspective: We assumed that a sample's parameters can approximate the true populations' parameters (or the "true underlying model of the world"). Based on this assumption we were able to quantify uncertainty with a confidence interval using test-statistics like the Z-statistics and the t-statistics. 

In Lecture 12 we showed that, for many familiar models, the likelihood depends on the sample through a small set of sufficient statistics. 

Today, we go back to an earlier discussion in lecture 3, namely the probability foundations, where we pointed to a Baysian perspective and Bayes theorem that proposes that we cannot be always be sure about whether our underlying parameter changes as we see more data. Thus,  today, we are taking a Baysian view and ask a different but related question: **How should our uncertainty about an unknown parameter change when we observe data?**

Thus, the guiding question for this lecture is:

> How can we represent uncertainty about an unknown parameter and update
> that uncertainty after observing data?

## Learning objectives

After completing this lecture, you should be able to:

1. Distinguish frequentist parameter estimation from Bayesian inference.
2. Identify and interpret the prior, likelihood, evidence, and posterior.
3. Calculate a posterior and its evidence in a simple discrete-parameter example.
4. Explain posterior prediction and distinguish credible intervals from
   confidence intervals.

## 1. From parameter estimation to parameter uncertainty

### 1.1 Connection to Lecture 7: likelihood and maximum likelihood
Let $X$ denote a random observation, $x$ its realized value, and $\mathcal D=(x_1,\ldots,x_N)$ the observed dataset. A sampling model $p(x\mid\theta)$ describes observations conditional on a parameter value.

For observations that are conditionally independent and identically distributed given $\theta$,

$$
L(\theta;\mathcal D)=p(\mathcal D\mid\theta)
=\prod_{n=1}^{N}p(x_n\mid\theta).
$$

Once the data have been observed, the likelihood is a function of $\theta$. Lecture 7 used it to choose the maximum-likelihood estimate,

$$
\widehat\theta_{\mathrm{ML}}=\operatorname*{arg\,max}_{\theta}L(\theta;\mathcal D).
$$

This returns one value. It does not, by itself, provide a probability distribution over possible parameter values.

### 1.2 Connection to Lecture 9: uncertainty across repeated samples

Before observing data, an estimator $\widehat\theta=g(\mathbf X)$ is random because the sample $\mathbf X$ is random. Lecture 9 asked how that estimator would vary if the entire data-collection procedure were repeated under the same fixed parameter.

The sampling distribution describes the behavior of an estimation procedure across possible datasets.

### 1.3 The Bayesian perspective: a distribution over an unknown parameter

In Bayesian parameter learning, a probability distribution represents uncertainty about $\theta$. Before seeing the current data, it is the **prior**, $p(\theta)$. After observing the data, it is the **posterior**, $p(\theta\mid\mathcal D)$.

The parameter is not assumed to physically change when we collect data. What changes is our distribution describing which values are plausible. A random-variable representation of the parameter expresses uncertainty about its value.

:::{admonition} Frequentist and Bayesian uncertainty
:class: important

| Perspective | Parameter | What varies in the uncertainty calculation? | Principal question |
|---|---|---|---|
| Lecture 7: maximum likelihood | Fixed but unknown | Candidate parameter values in the likelihood calculation | Which value maximizes the likelihood of this dataset? |
| Lecture 9: sampling uncertainty | Fixed but unknown | Samples and estimators across hypothetical repetitions | How would the estimator vary across samples? |
| Lecture 13: Bayesian inference | Described by a distribution representing uncertainty | Possible parameter values conditional on the observed dataset | How plausible are different values after observing these data? |

Bayesian inference conditions on the dataset actually observed. It does not require collecting repeated datasets.
:::


## 2. The ingredients of a Bayesian model

Bayes' theorem (see lecture 3) provides the foundation. We had explained it when we introduced conditional probability. 

$$
P(A \mid B) = \frac{P(B \mid A)P(A)}{P(B)}
$$

The pieces have useful names:

- $P(A)$ is the **prior** probability of $A$.
- $P(B \mid A)$ is the **likelihood** of observing evidence $B$ if $A$ is true.
- $P(B)$ is the **evidence** or normalizing probability.
- $P(A \mid B)$ is the **posterior** probability of $A$ after observing $B$.
---

You may want to go back and refresh your memory on this theorem so that this lecture is easy to follow. 

We use a working example, assuming we have Bernoulli distribution (discussed in lecture 4 in the section on discrete random variables). 

### 2.1 Observed data and the sampling model

We use a working example, assuming we have Bernoulli distribution (discussed in lecture 4 in the section on discrete random variables) to explain the principle Baysian inference. 

$$
P(X=x\mid\theta)=\theta^x(1-\theta)^{1-x},
\qquad x\in\{0,1\}.
$$

For example, $X=1$ indicates that a drone completes a mission successfully,
such as landing at the designated airport, while $X=0$ indicates that it
does not meet the specified success criterion. The parameter $\theta$
is the unknown probability of mission success under the modeled operating
conditions.

We observe mission outcomes, not $\theta$ itself. Bayesian inference lets
us update our uncertainty about $\theta$ using those observed outcomes.

### 2.2 The prior: uncertainty before observing the data

The prior represents uncertainty before incorporating the current dataset. It can reflect previous studies, domain knowledge, or explicitly stated modeling assumptions. It is not automatically an objective description of the problem.

Parameters specifying a prior are called **hyperparameters**. In these lectures, we treat them as chosen, fixed quantities. Assigning them their own priors is a further modeling step.
### 2.3 The likelihood: how the parameter explains the observed data

The likelihood describes how well proposed parameter values explain the observed data. For a particular Bernoulli sequence with $S$ successes and $F$ failures,

$$
p(\mathcal D\mid\theta)=\theta^S(1-\theta)^F.
$$

This also connects to Lecture 12: given $N=S+F$, the success count is sufficient for $\theta$. (We will return to the importance of sufficient statistics in Lecture 14). 

:::{warning}
The likelihood $L(\theta;\mathcal D)$ is not a probability distribution over $\theta$. It need not sum or integrate to one across parameter values. A prior or posterior does.
:::

### 2.4 The joint distribution of parameters and data

The prior and sampling model together define

$$
p(\theta,\mathcal D)=p(\mathcal D\mid\theta)p(\theta).
$$

This joint model is the starting point for both parameter learning and prediction.

:::{warning}
[Distinguish a likelihood function from a probability distribution
over the parameter.]
:::

## 3. Bayes’ rule for parameter learning

### 3.1 The posterior: uncertainty after observing the data

Conditioning the joint distribution on the observed dataset gives

$$
\boxed{p(\theta\mid\mathcal D)
=\frac{p(\mathcal D\mid\theta)p(\theta)}{p(\mathcal D)}}.
$$

The numerator combines the prior with the likelihood. A value receives appreciable posterior probability only if their product is appreciable relative to the products for other values.
### 3.2 The evidence: normalizing the posterior

For a discrete parameter taking values $\theta_1,\ldots,\theta_K$,

$$
p(\mathcal D)=\sum_{k=1}^{K}p(\mathcal D\mid\theta_k)P(\theta=\theta_k).
$$

For a continuous parameter, the corresponding expression is

$$
p(\mathcal D)=\int p(\mathcal D\mid\theta)p(\theta)\,d\theta.
$$

The **evidence**, also called the marginal likelihood, averages the likelihood under the prior. It normalizes the posterior. For a fixed model and dataset, it is one number, not a function of $\theta$. With continuous observations it is a density value, which need not be less than one.

### 3.3 Posterior proportional to likelihood times prior

Because the evidence does not depend on $\theta$, we can write

$$
p(\theta\mid\mathcal D)\propto p(\mathcal D\mid\theta)p(\theta)
$$

and normalize afterward. The symbol $\propto$ means that the two expressions differ by a positive factor independent of $\theta$.

:::{admonition} Four quantities to distinguish
:class: important

| Quantity | Notation | Interpretation |
|---|---|---|
| Prior | $p(\theta)$ | Uncertainty about the parameter before the current data |
| Likelihood | $p(\mathcal D\mid\theta)$ | Fit of proposed parameter values to the observed data |
| Evidence | $p(\mathcal D)$ | Prior-averaged probability or density of the observed data |
| Posterior | $p(\theta\mid\mathcal D)$ | Uncertainty about the parameter after observing the data |
:::

## 4. Worked example: learning an unknown success probability

### 4.1 Three possible parameter values

For a deliberately simple model, suppose the success probability can take only three values:

$$
\theta\in\left\{\frac14,\frac12,\frac34\right\}.
$$

This is a discrete prior over a restricted parameter space. It is a teaching example, not an assumption that every success probability must have one of these values.

### 4.2 Assigning the prior

Assign each value prior probability $1/3$. The prior mean is $1/2$.

### 4.3 Observing one success

Observe $\mathcal D=(1)$. Since $P(X=1\mid\theta)=\theta$, the likelihood values are $1/4$, $1/2$, and $3/4$.

### 4.4 Evidence and normalized posterior

Multiply each likelihood by its prior probability and sum:

$$
P(X=1)=\frac14\frac13+\frac12\frac13+\frac34\frac13=\frac12.
$$

Divide each product by this evidence:

| Parameter value | Prior | Likelihood of success | Likelihood × prior | Posterior |
|---|---|---|---|---|
| $1/4$ | $1/3$ | $1/4$ | $1/12$ | $1/6$ |
| $1/2$ | $1/3$ | $1/2$ | $1/6$ | $1/3$ |
| $3/4$ | $1/3$ | $3/4$ | $1/4$ | $1/2$ |

The posterior probabilities add to one.

### 4.5 Interpretation

The success increases the probability assigned to $\theta=3/4$ and decreases the probability assigned to $\theta=1/4$. It does not determine the parameter with certainty.

*Figure instruction/caption: Use grouped bars at parameter values 0.25, 0.50, and 0.75. Show the prior probabilities (1/3 each) and posterior probabilities (1/6, 1/3, 1/2). Label the horizontal axis “Success probability parameter, θ” and the vertical axis “Probability mass.” These are probabilities over parameter values, not probabilities of binary outcomes.*

:::{admonition} Check your understanding
:class: tip

If the observed outcome were a failure, which parameter value would have the largest posterior probability? Explain using the likelihood $P(X=0\mid\theta)=1-\theta$.
:::

## 5. Interpreting posterior uncertainty

### 5.1 Credible intervals

For a continuous parameter, a $(1-\alpha)$ credible interval $[l,u]$ satisfies

$$
P(l\leq\theta\leq u\mid\mathcal D)
=\int_l^u p(\theta\mid\mathcal D)\,d\theta=1-\alpha.
$$

For example, a 95% credible interval contains 95% posterior probability under the specified model and prior. Different interval conventions can produce different intervals. For discrete parameters, we often use a credible set, whose mass may exceed the desired level because probabilities come in discrete increments.

### 5.2 Credible versus confidence intervals

| Interval | Interpretation |
|---|---|
| 95% confidence interval | Across repetitions, the interval procedure covers the fixed true parameter 95% of the time, under its assumptions. |
| 95% credible interval | Conditional on the observed data, the interval contains 95% posterior probability for the parameter, under the model and prior. |

:::{warning}
Repeated-sampling coverage and posterior probability are different statements. A credible interval does not automatically have matching frequentist coverage, and a confidence interval does not automatically assign posterior probability to its realized bounds.
:::

## 6. Summary

Bayesian parameter learning starts with a prior, combines it with the likelihood, and normalizes the product using the evidence. The resulting posterior represents uncertainty conditional on the observed dataset. Posterior prediction averages a future observation’s distribution over that posterior.

Our discrete example made every step visible through finite sums. Lecture 14 extends the same logic to continuous parameters, where conjugate priors allow exact updates. Beta–Bernoulli and scalar Gaussian–Gaussian models will show how sufficient statistics determine the update and how the posterior supports prediction.

## Conceptual practice

### 1. Evidence, posterior, and prediction

Reuse the two-value example: $\theta\in\{1/4,3/4\}$ with prior probability $1/2$ each. Observe one success. Calculate the evidence, posterior probabilities, and probability of success on the next trial.

::::{admonition} Solution 1
:class: dropdown

The evidence is $(1/4)(1/2)+(3/4)(1/2)=1/2$. Dividing the prior–likelihood products by it gives posterior probabilities $1/4$ and $3/4$. The next success probability is

$$
\frac14\frac14+\frac34\frac34=\frac58.
$$

Evidence averages under the prior; prediction averages under the posterior.
::::

### 2. A different prior

In the two-value model, change the prior probabilities for $1/4$ and $3/4$ to $3/4$ and $1/4$. Observe one success. Find the evidence and posterior. What does this illustrate?

::::{admonition} Solution 2
:class: dropdown

Both prior–likelihood products equal $3/16$. Their sum is $3/8$, so the posterior probabilities are $1/2$ each. The success offsets the prior preference for the lower success probability. The posterior depends on both the data and the prior.
::::

### 3. Interpreting uncertainty

Explain the difference between a 95% credible interval, a 95% confidence interval, and a predictive probability for the next observation.

::::{admonition} Solution 3
:class: dropdown

A credible interval contains 95% posterior probability for the parameter under the Bayesian model. A confidence interval is produced by a procedure with 95% repeated-sampling coverage under its assumptions. A predictive probability concerns a future observation, averaging its sampling distribution over the posterior parameter distribution.
::::

## Readings and references

- Christopher M. Bishop, *Pattern Recognition and Machine Learning* (2006), Sections 1.2.3 and 1.2.6: Bayesian probabilities and parameter inference.
- Lecture 7: likelihood and maximum-likelihood estimation.
- Lecture 9: sampling distributions and confidence intervals.
- Lecture 12: exponential families and sufficient statistics.