# Lecture 15 — Bayesian Hypotheses Testing

In prior lecture, we learned about conjugate priors for our drone testing scenario. We learned how the choice of your prior affects the posterior. 
In other words, as you engage in data-driven decision the prior belief impacts the uncertainty that you drive after you have seen more data. Or in other words, after you have gathered more evidence.
In this lecture, we now take these insights to the next level, and return to data-driven decision making and hypothesis testing. However, in this lecture, we learn about the Baysian way of hypotheses testing, and contrast it with the frequentist way that have covered in lecture 6. 

:::{admonition} Scenario: Engine-start testing

We repeatedly test an engine under a defined protocol, with comparable temperature, battery condition, and starting procedure. For each attempt, we record whether the engine starts in less than five seconds.

We code a successful start as $X_i=1$ and an unsuccessful start as $X_i=0$. Given the unknown parameter $\theta$, each outcome follows a Bernoulli distribution:

$$
X_i\mid\theta\sim\operatorname{Bernoulli}(\theta),
\qquad
P(X_i=1\mid\theta)=\theta.
$$

Thus, $\theta$ is the probability that the engine starts in less than five seconds under the specified testing conditions. Assume that test outcomes are conditionally independent given the same fixed $\theta$.

For this teaching example, we set the target at $\theta_c=0.8$; this is an illustrative requirement, not an externally established engineering standard. 

::: 

So this leads to the following practically motivated question: 
**Guiding question:** What do your test results tell you about the engine's unknown start-success probability, and how much support do they provide for the claim that it exceeds your threshold?

## Key learning outcomes

After completing Lecture 15, you should be able to: 

1. Calculate a crediable interval 
2. Contrast with confidence interval
3. Perform Hypothesis testing using Baysian inference 
4. Contrast it with frequenciest Hypotheses testing
5. Introduce the Bayes factor 
6. Learn about empirical Bayes priors

## 1. Credible intervals for the unknown parameter

### 1.1 Probability as area under the posterior

For our engine-start example, suppose prior knowledge is represented by

$$
\theta\sim\operatorname{Beta}(8,2).
$$

This prior has mean $8/(8+2)=0.8$. Its mean represents our prior
expectation, whereas the benchmark $\theta_c=0.8$ represents a
performance requirement. Their equality is a choice for this example;
a requirement does not determine the prior.

Now assume that we observe ten attempts: nine successful starts and one unsuccessful start. Using the conjugate update from Lecture 14 gives

$$
\theta\mid\mathcal D
\sim\operatorname{Beta}(8+9,2+1)
=\operatorname{Beta}(17,3).
$$

The posterior density is

$$
p(\theta\mid\mathcal D)
=\frac{\theta^{16}(1-\theta)^2}{B(17,3)},
\qquad 0<\theta<1.
$$

This density describes our uncertainty about the unknown start-success
probability. The probability assigned to a range of parameter values
is the area under the posterior density over that range:

$$
P(l\leq\theta\leq u\mid\mathcal D)
=\int_l^u p(\theta\mid\mathcal D)\,d\theta.
$$

Here $l$ and $u$ are the lower and upper interval boundaries.
The curve height is a density; the area between the boundaries
is a probability. We next choose these boundaries so that the interval contains 95% of the posterior probability.

### 1.2 Construct an equal-tailed 95% credible interval

A quantile identifies a value below which a specified amount of
probability lies. You may recognize this idea from percentiles in
standardized test scores. Here, we apply it to the posterior
distribution of the unknown parameter $\theta$.

The posterior quantile $q_r$ satisfies

$$
P(\theta\leq q_r\mid\mathcal D)=r.
$$

For example, $q_{0.025}$ is a parameter value with 2.5% of the
posterior probability below it. The number $0.025$ specifies an
area under the posterior curve; it does not mean that $\theta=0.025$.

To construct an **equal-tailed 95% credible interval**, we retain
the middle 95% of the posterior probability and leave 2.5% in
each tail.

We therefore choose

$$
l=q_{0.025},
\qquad
u=q_{0.975}.
$$

These boundaries satisfy

$$
P(\theta<l\mid\mathcal D)=0.025,
\qquad
P(\theta>u\mid\mathcal D)=0.025.
$$

The posterior probability between them is consequently

$$
P(l\leq\theta\leq u\mid\mathcal D)
=0.975-0.025
=0.95.
$$

For our Beta(17,3) posterior, we calculate the boundaries numerically.
SciPy's `ppf` function returns the parameter values corresponding
to the requested cumulative probabilities:

```python
from scipy import stats

lower, upper = stats.beta.ppf([0.025,0.975],a=17,b=3)
print(f"95% credible interval: [{lower:.3f},{upper:.3f}")
```
The input '0.025` and `0.975` are cumulative probabilities. The outputs `lower` and `upper` are possible values of the engines start-success probability $\theta$. 


### 1.3 Interpret the result

[State: conditional on the observed data, prior, and sampling model, the interval contains the unknown success probability with posterior probability 0.95. Explain that this concerns the parameter, not the fraction of future successful missions.]

**Figure 1** Shade the middle 95% of the Beta(5,3) posterior. Mark the quantile boundaries and label the two 2.5% tails. Retain the parameter and density axis labels.

### 1.4 Credible intervals versus confidence intervals

Both intervals summarize uncertainty about an unknown parameter,
but the probability statements have different meanings.

| Aspect | Bayesian credible interval | Frequentist confidence interval |
|---|---|---|
| Basis | Posterior distribution of $\theta$ given the observed data | Sampling distribution of a statistic under repeated sampling |
| Treatment of $\theta$ | Uncertainty about $\theta$ is represented by a probability distribution | $\theta$ is fixed but unknown |
| Role of the prior | Uses a specified prior together with the likelihood | Does not require a Bayesian prior |
| Meaning of 95% | Given the data and Bayesian model, the interval contains 95% of the posterior probability for $\theta$ | Over repeated samples, a procedure with 95% coverage produces intervals containing the fixed $\theta$ 95% of the time |
| Engine-start interpretation | “Given our observations, prior, and sampling model, we assign 95% probability to $\theta$ lying between these bounds.” | “If we repeatedly collected test data and constructed intervals using this procedure, 95% of those intervals would contain the true start-success probability.” |
| What the interval does not say | It does not mean that 95% of future engine starts will succeed | It does not, by itself, assign 95% probability to $\theta$ lying inside the particular observed interval |

For a 95% credible interval, the probability statement is

$$
P(l\leq\theta\leq u\mid\mathcal D)=0.95.
$$

For a confidence-interval procedure with exact 95% coverage,
the statement is

$$
P_\theta\!\left(
L(\mathcal D)\leq\theta\leq U(\mathcal D)
\right)=0.95.
$$

In the second expression, the probability refers to repeated datasets
$\mathcal D$. The endpoints $L(\mathcal D)$ and $U(\mathcal D)$ vary
from sample to sample, while $\theta$ remains fixed. As discussed in
Lecture 9, some confidence-interval procedures have approximate or
conservative coverage rather than exactly 95%.

:::{admonition} The key distinction
:class: important

A credible interval assigns posterior probability to a parameter
range after observing the data. A confidence level describes the
repeated-sampling coverage of the method used to construct the interval.

The two intervals can have similar numerical endpoints without
having the same interpretation.
:::


## 2. Bayesian hypothesis testing about a threshold

### 2.1 Stating complementary Hypotheses

For a threshold $0<\theta_0<1$ specified before examining the data,

$$
H_0:\theta\leq\theta_0,
\qquad H_1:\theta>\theta_0.
$$

[Use $\theta_0=1/2$ for transparent arithmetic in the worked example. Label this a teaching threshold, not an operational drone-reliability standard.]

### 2.2 Calculate posterior hypothesis probabilities

$$
P(H_1\mid\mathcal D)=\int_{\theta_0}^1p(\theta\mid\mathcal D)\,d\theta,
\qquad
P(H_0\mid\mathcal D)=1-P(H_1\mid\mathcal D).
$$

[Explain how this differs from constructing a credible interval: the hypothesis fixes the region first, then we calculate its probability. It is not a frequentist p-value.]

### 2.3 Compare prior and posterior odds

$$
O_{10}^{\mathrm{prior}}=\frac{P(H_1)}{P(H_0)},
\qquad
O_{10}^{\mathrm{post}}=\frac{P(H_1\mid\mathcal D)}{P(H_0\mid\mathcal D)}.
$$

### 2.4 Introduce the Bayes factor

$$
\mathrm{BF}_{10}
=\frac{p(\mathcal D\mid H_1)}{p(\mathcal D\mid H_0)}
=\frac{O_{10}^{\mathrm{post}}}{O_{10}^{\mathrm{prior}}}.
$$

[For these region hypotheses, the prior within each hypothesis is the original Beta prior restricted to its region and normalized. Hypothesis-specific evidence averages the likelihood using that conditional prior. Thus the odds-ratio identity uses a coherent single prior specification.]

[Distinguish posterior probability from change in odds. Interpret BF above, below, and equal to one. A large posterior probability may partly reflect strong prior support.]

### 2.5 Worked example using Beta(5,3)

With the original Beta(2,2) prior and $\theta_0=1/2$,

$$
P(H_1)=P(H_0)=\frac12,
\qquad
P(H_1\mid\mathcal D)=\frac{99}{128},
\qquad
P(H_0\mid\mathcal D)=\frac{29}{128}.
$$

Hence posterior odds are $99/29$ and $\mathrm{BF}_{10}=99/29\approx3.41$. Explain what the data changed and why four observations still leave substantial uncertainty.

**Figure instruction/caption:** Place prior and posterior density plots side by side; draw a vertical line at $\theta_0$ and shade the region for $H_1$ in both. Label its prior and posterior probabilities.

### 2.6 Interpretation, sensitivity, and decisions

[Repeat the hypothesis calculation under selected priors from Section 3. Distinguish the claim's posterior probability, the Bayes factor, and a decision rule informed by consequences. Do not use universal reject/accept cutoffs.]

:::{note}
A continuous Beta prior assigns probability zero to any exact point. Testing a point null such as $H_0:\theta=0.5$ with positive prior probability requires a separate model specification. Excluding a point from a credible interval does not itself supply a Bayes factor for that point null.
:::

**Textbook connection:** Bayes Rules!, Section 8.2.1, directly supports threshold tests. Bishop Section 3.4, especially pp. 161–162, supplies the general model-evidence and Bayes-factor framework; the regression-specific development is not required.

## 2.5. Summary about Baysian hypothesis testing

[Summary table: conjugate update, sequential update, prior sensitivity, credible interval, posterior hypothesis probability, and Bayes factor. For each, state the question answered and the assumptions used.]

[Closing question: What additional information would you report alongside the claim that the mission-success probability exceeds the specified threshold?]

Conjugate models let us calculate the posterior analytically. 

# 3. Empirical Bayes 

Next, we will discuss how to approximate a prior based on empirical data. 

## 3.1 Simple estimation based prior mean and variance data


## 3.2 Simple sampling-based approach 

## 3.3. Explain impact of coarse and fine grids


## 3.4. Overview of other sampling techniques 

## References

## Practice questions
