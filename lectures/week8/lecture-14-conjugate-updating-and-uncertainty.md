# Lecture 14 — Conjugate Bayesian Updating, Credible Intervals, and Hypothesis Testing

## Introduction and guiding question

Lecture 13 used Bayes' rule to turn a Beta(2,2) prior into a Beta(5,3) posterior after observing the mission outcomes $\mathcal D=(1,1,0,1)$. Lecture 12 showed how sufficient statistics retain the information needed for parameter inference within an assumed model. We now generalize the update and use the posterior to answer questions about the unknown mission-success probability.

> How can we update our knowledge as data arrive, quantify uncertainty about a parameter, and evaluate a claim about its value?

[Opening discussion: Does observing three successful missions out of four establish that the underlying success probability exceeds a specified threshold? What can the whole posterior tell us?]

## Learning objectives

After completing this lecture, you should be able to:

1. Explain conjugacy and derive the general Beta–Bernoulli posterior update.
2. Apply sequential updating and explain why it agrees with batch updating under the assumed model.
3. Specify interpretable Beta priors and assess sensitivity of posterior conclusions to prior assumptions.
4. Construct and interpret an equal-tailed credible interval and distinguish its interpretation from a frequentist confidence interval.
5. Calculate posterior probabilities for complementary hypotheses about a success-probability threshold.
6. Distinguish prior odds, posterior odds, and a Bayes factor.

## 1. Conjugate priors and the general Beta–Bernoulli update

### 1.1 Recall the worked example from Lecture 13

[Reuse the original Beta(2,2) prior and Beta(5,3) posterior figure. Identify the prior, observed data, likelihood, and posterior.]

Maintain the likelihood notation from Lecture 7:

$$
\mathcal L(\theta;\mathcal D)=p(\mathcal D\mid\theta),
\qquad
p(\theta\mid\mathcal D)
\propto\mathcal L(\theta;\mathcal D)p(\theta).
$$

### 1.2 What makes a prior conjugate?

[Define conjugacy relative to a likelihood: the posterior belongs to the same distribution family as the prior. Explain its computational convenience and that scientific appropriateness still matters.]

### 1.3 Derive the general updating rule

For $\theta\sim\operatorname{Beta}(a,b)$ and conditionally IID Bernoulli outcomes with $S$ successes and $F$ failures,

$$
\begin{aligned}
p(\theta\mid\mathcal D)
&\propto\theta^S(1-\theta)^F\theta^{a-1}(1-\theta)^{b-1}\\
&=\theta^{a+S-1}(1-\theta)^{b+F-1}.
\end{aligned}
$$

Thus $\theta\mid\mathcal D\sim\operatorname{Beta}(a+S,b+F)$.

### 1.4 Connect the update to sufficient statistics

[Explain why the update uses $S$ and $F$, or $S$ and known $N$. Distinguish the unknown parameter $\theta$ from the fixed prior hyperparameters $a,b$.]

[Optional derivation: evidence for the particular ordered sequence is $B(a+S,b+F)/B(a,b)$. Explain that count evidence includes the binomial coefficient, although the posterior is unchanged.]

**Textbook connection:** Bishop, Sections 2.1.1 and 2.4.2; recall Section 2.4.1 from Lecture 12.

## 2. Bayesian learning as a sequential process

### 2.1 The posterior becomes the next prior

[State $p(\theta\mid\mathcal D_{\mathrm{old}},\mathcal D_{\mathrm{new}})\propto p(\mathcal D_{\mathrm{new}}\mid\theta)p(\theta\mid\mathcal D_{\mathrm{old}})$ under conditional independence of the two data batches given $\theta$. Explain what is held fixed.]

### 2.2 Continue the mission example

After the original four missions, observe an additional success followed by a failure:

$$
\operatorname{Beta}(5,3)
\xrightarrow{1}\operatorname{Beta}(6,3)
\xrightarrow{0}\operatorname{Beta}(6,4).
$$

### 2.3 Compare sequential and batch updates

[Table: original prior, all six observations, cumulative counts, and posterior shape parameters. Verify that Beta(2,2) plus four successes and two failures gives Beta(6,4). Explain order invariance under the shared-parameter IID model.]

**Figure instruction/caption:** Plot Beta(5,3), Beta(6,3), and Beta(6,4) densities on common axes. Label each by the newly incorporated outcome. Horizontal axis: “Mission-success probability, $\theta$”; vertical axis: “Probability density.”

:::{warning}
Each observation enters the update once. When using the current posterior as the next prior, multiply only by the likelihood of the new data.
:::

**Textbook connection:** Bishop, Section 2.1.1, especially the sequential discussion and Figure 2.3; Bayes Rules!, Section 4.4.

## 3. Choosing a prior and checking sensitivity

### 3.1 Translate knowledge into a prior distribution

[Use previous comparable experiments or domain knowledge. Explain that a prior represents uncertainty before the current data; its assumptions must match the operating conditions.]

### 3.2 Interpret the shape of a Beta prior

[Compare Beta(1,1), Beta(2,2), and Beta(10,10) to vary concentration around the same center. Compare Beta(2,6) and Beta(6,2) to vary the location with the same sum of shape parameters. Explain the curves visually; point-estimation rules are not required.]

### 3.3 Hold the data fixed and vary the prior

Use the original dataset $(1,1,0,1)$ for every comparison:

| Illustrative prior | Posterior |
|---|---|
| Beta(1,1) | Beta(4,2) |
| Beta(2,2) | Beta(5,3) |
| Beta(10,10) | Beta(13,11) |
| Beta(2,6) | Beta(5,7) |
| Beta(6,2) | Beta(9,3) |

[Explain that these are candidate assumptions for a sensitivity exercise, not equally justified choices for every application. Return to this table after Sections 4 and 5 to compare intervals and hypothesis probabilities.]

**Figure instruction/caption:** Use small panels with identical axes; show each prior and its posterior for the same four outcomes. Keep prior and posterior colors consistent with Lecture 13.

### 3.4 Report dependence on assumptions

[Explain how to report a conclusion that changes substantially across scientifically plausible priors. Do not choose a prior merely to obtain a desired conclusion. Distinguish representing prior knowledge from numerically approximating a posterior.]

**Textbook connection:** Bayes Rules!, Sections 4.1–4.3. Bishop Section 2.1.1 supplies the Beta family; Section 2.4.3 is optional background on why a uniform prior is not universally noninformative.

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

## 5. Bayesian hypothesis testing about a threshold

### 5.1 State complementary hypotheses

For a threshold $0<\theta_0<1$ specified before examining the data,

$$
H_0:\theta\leq\theta_0,
\qquad H_1:\theta>\theta_0.
$$

[Use $\theta_0=1/2$ for transparent arithmetic in the worked example. Label this a teaching threshold, not an operational drone-reliability standard.]

### 5.2 Calculate posterior hypothesis probabilities

$$
P(H_1\mid\mathcal D)=\int_{\theta_0}^1p(\theta\mid\mathcal D)\,d\theta,
\qquad
P(H_0\mid\mathcal D)=1-P(H_1\mid\mathcal D).
$$

[Explain how this differs from constructing a credible interval: the hypothesis fixes the region first, then we calculate its probability. It is not a frequentist p-value.]

### 5.3 Compare prior and posterior odds

$$
O_{10}^{\mathrm{prior}}=\frac{P(H_1)}{P(H_0)},
\qquad
O_{10}^{\mathrm{post}}=\frac{P(H_1\mid\mathcal D)}{P(H_0\mid\mathcal D)}.
$$

### 5.4 Introduce the Bayes factor

$$
\mathrm{BF}_{10}
=\frac{p(\mathcal D\mid H_1)}{p(\mathcal D\mid H_0)}
=\frac{O_{10}^{\mathrm{post}}}{O_{10}^{\mathrm{prior}}}.
$$

[For these region hypotheses, the prior within each hypothesis is the original Beta prior restricted to its region and normalized. Hypothesis-specific evidence averages the likelihood using that conditional prior. Thus the odds-ratio identity uses a coherent single prior specification.]

[Distinguish posterior probability from change in odds. Interpret BF above, below, and equal to one. A large posterior probability may partly reflect strong prior support.]

### 5.5 Worked example using Beta(5,3)

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

### 5.6 Interpretation, sensitivity, and decisions

[Repeat the hypothesis calculation under selected priors from Section 3. Distinguish the claim's posterior probability, the Bayes factor, and a decision rule informed by consequences. Do not use universal reject/accept cutoffs.]

:::{note}
A continuous Beta prior assigns probability zero to any exact point. Testing a point null such as $H_0:\theta=0.5$ with positive prior probability requires a separate model specification. Excluding a point from a credible interval does not itself supply a Bayes factor for that point null.
:::

**Textbook connection:** Bayes Rules!, Section 8.2.1, directly supports threshold tests. Bishop Section 3.4, especially pp. 161–162, supplies the general model-evidence and Bayes-factor framework; the regression-specific development is not required.

## 6. Summary and transition to Lecture 15

[Summary table: conjugate update, sequential update, prior sensitivity, credible interval, posterior hypothesis probability, and Bayes factor. For each, state the question answered and the assumptions used.]

[Closing question: What additional information would you report alongside the claim that the mission-success probability exceeds the specified threshold?]

Conjugate models let us calculate the posterior analytically. In Lecture 15, we will ask how to approximate a posterior when analytical updating is unavailable, how to assess the approximation, and how to use a posterior to predict a new observation. The numerical methods belong in Lecture 15; here this is only a transition.

## Conceptual and calculation practice

[Develop four exercises with dropdown solutions:]

1. **Sequential update:** Begin with Beta(2,2), incorporate $(1,1,0,1)$, then $(1,0)$. Verify agreement with a batch update and identify the sufficient statistics.
2. **Prior sensitivity:** Use the same four outcomes with Beta(1,1), Beta(2,2), and Beta(10,10). Compare posteriors and numerically calculated 95% credible intervals.
3. **Interval interpretation:** Explain the equal-tailed interval and correct a statement confusing posterior probability with repeated-sampling coverage.
4. **Threshold hypotheses:** For the original Beta(5,3) posterior, calculate probabilities for $\theta>1/2$, prior and posterior odds, and BF. Explain why posterior probability and BF answer different questions.

[Provide required Beta quantiles or a short computational aid so numerical inversion does not obscure the conceptual objectives.]

## Readings and references

### Primary course textbook

Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. [Official textbook PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf).

- **Section 2.1.1, pp. 71–74:** Beta priors, posterior updates, sequential learning, and prior influence. Focus on Equations (2.13)–(2.18) and Figure 2.3; defer predictive Equations (2.19)–(2.20).
- **Section 2.4.1, pp. 116–117:** sufficient-statistic connection; review from Lecture 12.
- **Section 2.4.2, p. 117:** general conjugate-prior structure. Assign the concept; the general exponential-family prior derivation is optional.
- **Section 3.4, especially pp. 161–162:** model probabilities, evidence, and Bayes factors. Use the opening general discussion without requiring Bayesian linear regression.
- **Section 2.4.3:** optional discussion of noninformative priors and parameterization, not a recipe for an automatically neutral prior.

### Supplemental textbook for intervals and hypothesis testing

Johnson, A. A., Ott, M. Q., and Dogucu, M. (2022). *Bayes Rules! An Introduction to Applied Bayesian Modeling*. Chapman & Hall/CRC. [Free author-maintained online edition](https://www.bayesrulesbook.com/).

- [Chapter 4](https://www.bayesrulesbook.com/chapter-4.html), **Sections 4.1–4.3:** prior influence and sensitivity; **Section 4.4:** sequential analysis.
- [Chapter 8](https://www.bayesrulesbook.com/chapter-8.html), **Section 8.1:** credible intervals; **Section 8.2.1:** one-sided hypotheses, posterior odds, and Bayes factors; **Section 8.5:** comparison with frequentist inference. Defer posterior prediction in Section 8.3 and simulation-based analysis in Section 8.4.

The supplemental book uses a Binomial count likelihood. Given the same trial count and successes, its Beta posterior matches the Bernoulli-sequence posterior used here: the binomial coefficient is constant in the parameter and cancels when normalizing. Its notation $\pi,\alpha,\beta$ corresponds to $\theta,a,b$ in these lectures. Its examples use R; students can read the mathematics and figures without learning R.

**Suggested minimum assignment:** Bishop Sections 2.1.1 and 2.4.2, plus Bayes Rules! Sections 4.1, 8.1, and 8.2.1. Treat the remaining sections as targeted review or optional extensions.


---

## Planning note: Lecture 15 — Approximate Bayesian Inference and Prediction

This is the proposed scope and learning-outcome handoff for the next lecture, rather than additional required material for Lecture 14.

**Guiding question:** How can we carry out Bayesian inference when the posterior cannot be calculated analytically, and use the resulting distribution to predict a new observation?

### Key learning outcomes

After completing Lecture 15, students should be able to:

1. Distinguish specifying a prior, learning a posterior from data, and numerically approximating a posterior.
2. Explain why a nonconjugate model can make the evidence integral and normalized posterior difficult to calculate analytically.
3. Construct a one-dimensional grid approximation from likelihood–prior products, accounting for grid spacing when normalizing and calculating probabilities.
4. Calculate approximate credible intervals and threshold-hypothesis probabilities, and assess their stability as the grid is refined.
5. Explain how posterior draws approximate expectations and probabilities; distinguish numerical approximation error from uncertainty represented by the posterior.
6. Construct a posterior predictive probability by averaging the observation model over the posterior, and distinguish it from a posterior probability about the parameter.

### Suggested teaching sequence

- **Benchmark:** Approximate the known Beta(5,3) posterior on a grid and compare numerical results with Lecture 14's exact results.
- **Nonconjugate example:** Retain the same Bernoulli likelihood and specify a continuous prior with an unfamiliar shape, such as a normalized truncated Gaussian density on $(0,1)$. Multiply, normalize numerically, and calculate posterior areas.
- **Accuracy:** Compare coarse and fine grids; explain probability weights versus density heights. A grid covering $(0,1)$ needs a suitable numerical quadrature rule and sufficient resolution.
- **Simulation:** Draw parameter values from the normalized grid approximation and estimate an event probability using the proportion of draws satisfying it. Explain Monte Carlo variability.
- **Prediction:** Introduce $P(X_{\mathrm{new}}=1\mid\mathcal D)=\int_0^1\theta p(\theta\mid\mathcal D)\,d\theta$ and its weighted-grid and sample-average approximations. This is a probability for a future observation, not the posterior probability of a parameter hypothesis.
- **Optional outlook:** Why grids become expensive in higher dimensions and how MCMC can help. A full MCMC algorithm and diagnostic treatment should be an extension, not required alongside every other outcome in one lecture.

### Terminology and scope

The quantity approximated here is the **posterior**, after the prior and likelihood have been specified. Estimating prior hyperparameters from data is a different procedure, often called empirical Bayes, and requires attention to which data are used and how hyperparameter uncertainty is handled. It is not required in these two lectures. MAP and regularization remain with the later Bayesian regression material. Ordinary bootstrap resampling remains with frequentist sampling uncertainty.

### Reading alignment for Lecture 15

- **Bayes Rules!, Section 6.1:** grid approximation and its limitations.
- **Bayes Rules!, Sections 8.3–8.4:** posterior prediction and simulation-based posterior analysis; choose the relevant portions and translate R examples to the course's computational language.
- **Bishop, Section 2.1.1, Equations (2.19)–(2.20):** Bernoulli posterior prediction, deferred from Lecture 14.
- **Optional:** Bayes Rules!, Sections 6.2–6.3, for MCMC and diagnostics if the course later expands the computational treatment.
