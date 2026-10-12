# Lecture 15 — Approximate Bayesian Inference and Prediction

This planning outline defines the proposed scope and learning outcomes for Lecture 15, building on the exact Bayesian updates and uncertainty summaries in Lecture 14.

**Guiding question:** How can we carry out Bayesian inference when the posterior cannot be calculated analytically, and use the resulting distribution to predict a new observation?

## Key learning outcomes

After completing Lecture 15, students should be able to:



1. Calculate a crediable interval 
2. Contrast with confidence interval
3. revisit sensitivity to Baysian priors and its impact on crediable interval
4. Perform Hypothesis testing using Baysian inference 
5. Contrast it with frequenciest Hypotheses testing
6. Introduce the Bayes factor 

## 4. Credible intervals for the unknown parameter

### 4.1 Probability as area under the posterior

Return to the original Beta(5,3) posterior so the data and prior are explicit:

$$
P(l\leq\theta\leq u\mid\mathcal D)
=\int_l^u p(\theta\mid\mathcal D)\,d\theta.
$$

### 4.2 Construct an equal-tailed 95% credible interval

[Define posterior quantiles and select $l=q_{0.025}$ and $u=q_{0.975}$. Each excluded tail has posterior probability 0.025. Compute quantiles numerically; a closed-form density does not require closed-form quantiles.]

### 4.3 Interpret the result

[State: conditional on the observed data, prior, and sampling model, the interval contains the unknown success probability with posterior probability 0.95. Explain that this concerns the parameter, not the fraction of future successful missions.]

**Figure instruction/caption:** Shade the middle 95% of the Beta(5,3) posterior. Mark the quantile boundaries and label the two 2.5% tails. Retain the parameter and density axis labels.

### 4.4 Credible intervals versus confidence intervals

[Comparison table: posterior probability given the observed data versus repeated-sampling coverage of an interval procedure. Recall Lecture 9. Avoid interpreting a frequentist confidence interval as posterior probability without specifying a Bayesian model.]

### 4.5 Revisit prior sensitivity

[Compute the same credible level under the alternative priors in Section 3. Compare bounds and explain the dependence on prior assumptions.]

**Textbook connection:** Bayes Rules!, Section 8.1 and the interval discussion in Section 8.5. Focus on interval estimation; the point summaries in that reading are not required here.

## Conceptual and calculation practice

[Develop four exercises with dropdown solutions:]

1. **Sequential update:** Begin with Beta(2,2), incorporate $(1,1,0,1)$, then $(1,0)$. Verify agreement with a batch update and identify the sufficient statistics.
2. **Prior sensitivity:** Use the same four outcomes with Beta(1,1), Beta(2,2), and Beta(10,10). Compare posteriors and numerically calculated 95% credible intervals.
3. **Interval interpretation:** Explain the equal-tailed interval and correct a statement confusing posterior probability with repeated-sampling coverage.
4. **Threshold hypotheses:** For the original Beta(5,3) posterior, calculate probabilities for $\theta>1/2$, prior and posterior odds, and BF. Explain why posterior probability and BF answer different questions.

[Provide required Beta quantiles or a short computational aid so numerical inversion does not obscure the conceptual objectives.]


## 1. Bayesian hypothesis testing about a threshold

### 1.1 Stating complementary Hypotheses

For a threshold $0<\theta_0<1$ specified before examining the data,

$$
H_0:\theta\leq\theta_0,
\qquad H_1:\theta>\theta_0.
$$

[Use $\theta_0=1/2$ for transparent arithmetic in the worked example. Label this a teaching threshold, not an operational drone-reliability standard.]

### 1.2 Calculate posterior hypothesis probabilities

$$
P(H_1\mid\mathcal D)=\int_{\theta_0}^1p(\theta\mid\mathcal D)\,d\theta,
\qquad
P(H_0\mid\mathcal D)=1-P(H_1\mid\mathcal D).
$$

[Explain how this differs from constructing a credible interval: the hypothesis fixes the region first, then we calculate its probability. It is not a frequentist p-value.]

### 1.3 Compare prior and posterior odds

$$
O_{10}^{\mathrm{prior}}=\frac{P(H_1)}{P(H_0)},
\qquad
O_{10}^{\mathrm{post}}=\frac{P(H_1\mid\mathcal D)}{P(H_0\mid\mathcal D)}.
$$

### 1.4 Introduce the Bayes factor

$$
\mathrm{BF}_{10}
=\frac{p(\mathcal D\mid H_1)}{p(\mathcal D\mid H_0)}
=\frac{O_{10}^{\mathrm{post}}}{O_{10}^{\mathrm{prior}}}.
$$

[For these region hypotheses, the prior within each hypothesis is the original Beta prior restricted to its region and normalized. Hypothesis-specific evidence averages the likelihood using that conditional prior. Thus the odds-ratio identity uses a coherent single prior specification.]

[Distinguish posterior probability from change in odds. Interpret BF above, below, and equal to one. A large posterior probability may partly reflect strong prior support.]

### 1.5 Worked example using Beta(5,3)

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

### 1.6 Interpretation, sensitivity, and decisions

[Repeat the hypothesis calculation under selected priors from Section 3. Distinguish the claim's posterior probability, the Bayes factor, and a decision rule informed by consequences. Do not use universal reject/accept cutoffs.]

:::{note}
A continuous Beta prior assigns probability zero to any exact point. Testing a point null such as $H_0:\theta=0.5$ with positive prior probability requires a separate model specification. Excluding a point from a credible interval does not itself supply a Bayes factor for that point null.
:::

**Textbook connection:** Bayes Rules!, Section 8.2.1, directly supports threshold tests. Bishop Section 3.4, especially pp. 161–162, supplies the general model-evidence and Bayes-factor framework; the regression-specific development is not required.

## 1.5. Summary about Baysian hypothesis testing

[Summary table: conjugate update, sequential update, prior sensitivity, credible interval, posterior hypothesis probability, and Bayes factor. For each, state the question answered and the assumptions used.]

[Closing question: What additional information would you report alongside the claim that the mission-success probability exceeds the specified threshold?]

Conjugate models let us calculate the posterior analytically. 

# 2. Empirical Bayes 

Next, we will discuss how to approximate a prior based on empirical data. 

## 2.1 Simple grid sampling


## 2.2 Compare with ground truth known Beta distribution


## 2.3. Explain impact of coarse and fine grids


## 2.4. Bootstrapping


2.5. outlook on advanced sampling techniques and how they can be used


## Terminology and scope

The quantity approximated here is the **posterior**, after the prior and likelihood have been specified. Estimating prior hyperparameters from data is a different procedure, often called empirical Bayes, and requires attention to which data are used and how hyperparameter uncertainty is handled. It is not required in these two lectures. MAP and regularization remain with the later Bayesian regression material. Ordinary bootstrap resampling remains with frequentist sampling uncertainty.

