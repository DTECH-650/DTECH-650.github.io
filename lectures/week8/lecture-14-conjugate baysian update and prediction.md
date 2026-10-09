# Lecture 14 - Conjugate Bayesian Updating and Prediction

[Short introduction connecting the Bayesian framework from Lecture 13
with exponential-family structure and sufficient statistics from Lecture 12.]

Thus, the guiding question for this lecture is:

> How can we calculate posterior distributions and predictions exactly,
> and update them as new observations arrive?

## Learning objectives

After completing this lecture, you should be able to:

1. Explain conjugacy and use a posterior as the prior for a subsequent update.
2. Derive and apply the Beta–Bernoulli posterior, showing that batch and
   sequential updates agree.
3. Calculate evidence and posterior predictive probabilities for the
   Beta–Bernoulli model.
4. Calculate and interpret the posterior and predictive distribution for
   a Gaussian model with an unknown scalar mean and known observation variance.

## 1. From Bayes’ rule to exact update rules

### 1.1 Connection to Lecture 13: posterior proportional to likelihood times prior
### 1.2 What makes a prior conjugate?
### 1.3 Connection to Lecture 12: sufficient statistics and conjugacy

:::{admonition} From Bayesian objects to Bayesian calculations
:class: important

[Comparison table: what Lecture 13 defined and what Lecture 14 calculates.]
:::

## 2. Bayesian learning as a sequential process

### 2.1 The posterior becomes the next prior
### 2.2 Updating with one observation at a time
### 2.3 Why batch and sequential updates agree under conditional independence

:::{warning}
[Each observation enters the update once. Reusing previously incorporated
data would count the same information twice.]
:::

## 3. Beta–Bernoulli: learning a success probability

### 3.1 The Bernoulli sampling model and Beta prior
### 3.2 Deriving the posterior from success and failure counts
### 3.3 Interpreting the updated hyperparameters and posterior mean
### 3.4 Worked example: batch and sequential updates

[Table: observations, cumulative successes and failures,
posterior hyperparameters, and posterior mean.]

[Figure: prior and posterior densities at selected update stages.]

:::{admonition} Check your understanding
:class: tip

[Short question about how one additional success or failure
changes the posterior.]
:::

## 4. Evidence and prediction in the Beta–Bernoulli model

### 4.1 Calculating the evidence using the Beta normalizing constant
### 4.2 Distinguishing an observed sequence from a success count
### 4.3 Calculating the probability of success on the next trial
### 4.4 Worked example: evidence and posterior prediction

:::{admonition} Evidence versus posterior prediction
:class: important

[Comparison table: probability of the observed dataset versus
probability of a future observation given that dataset.]
:::

## 5. Gaussian–Gaussian: learning an unknown mean

### 5.1 A scalar Gaussian mean with known observation variance
### 5.2 Combining prior and data precision to obtain the posterior
### 5.3 Interpreting the posterior mean as a weighted average
### 5.4 Worked example: updating an estimated mean
### 5.5 The posterior predictive distribution

[Figure: prior, posterior, and posterior predictive distributions.]

:::{admonition} Two sources of predictive uncertainty
:class: important

[Interpret predictive variance as observation noise plus
remaining uncertainty about the mean.]
:::

## 6. Comparing the two conjugate models

[Summary table: unknown parameter, sampling model, prior,
sufficient statistics, posterior update, and predictive distribution.]

### 6.1 What the two examples have in common
### 6.2 How accumulating data changes posterior uncertainty

## 7. Summary and transition

### 7.1 Exact learning through updated sufficient statistics
### 7.2 Looking ahead to Lecture 15: Bayesian hypothesis testing and coding
### 7.3 Looking ahead to Lecture 18: from a constant mean to a conditional mean

[Brief transition identifying how scalar Gaussian learning prepares
students for regression.]

## Conceptual and calculation practice

[Three short exercises with dropdown solutions:
a sequential Beta update; evidence and prediction;
a Gaussian posterior and predictive-variance calculation.]

## Readings and references

[Assigned reading: Bishop, Chapter 2, pp. 113–117,
for the exponential-family and conjugacy connection.]

[Supporting readings for the Beta–Bernoulli and Gaussian–Gaussian examples.]