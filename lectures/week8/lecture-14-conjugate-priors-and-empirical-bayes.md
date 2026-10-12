# Lecture 14 — Conjugate Priors, Prior Choice, and Empirical Bayes

## Introduction and guiding question

In Lecture 13, we used Bayes' rule to update a distribution over an unknown mission-success probability. Starting with a Beta(2,2) prior and observing three successful missions and one failure, we obtained a Beta(5,3) posterior.

That calculation raises two practical questions. **Why did the posterior remain a Beta distribution? And where did the original values 2 and 2 come from?**

**Scenario**: Imagine that we are evaluating an engine of a drone (or a car or another physical system of your imagination) in a defined set of operating conditions. We observe engine health state over time (e.g. flagged by an onboard sensors) and want to update our knowledge after each new mission about the number of times the engine has been flagged as "unhealthy" (we can label this as true/false in this scenario). We may also have results from other comparable drones using the same engine. You have just been introduced to Baysian updating, and wonder: What prior should I use to calculate the posterior as I am collecting more data? 

So the guiding question for this lecture is: 
> How can we choose a prior, update it efficiently as data arrive, and how sensitive is the posterior and the quantification of uncertainty to the choice of my prior? 

This lecture develops conjugate updating, prior specification, and also helps you understand how sensitive a posterior is to the choice of a prior. We will also discuss how we can use empirical data to calculate an empirial prior (but we will not cover this in detail as this is an advanced topic requiring deeper understanding of sampling techniques like Monte-Carlo sampling). Lecture 15 will use posterior distributions to construct credible intervals and evaluate hypotheses.

## Learning objectives

After completing this lecture, you should be able to:

1. Explain conjugacy and derive the Beta–Bernoulli posterior update.
2. Apply sequential updating and show why it agrees with batch updating under the same model.
3. Interpret the mean and concentration of a Beta prior and assess prior sensitivity.
4. Recognize several common likelihood–conjugate-prior pairs.
5. Estimate a Beta prior's mean from related datasets in a simple empirical Bayes example with fixed concentration.

## 1. Conjugate priors and the Beta–Bernoulli update

### 1.1 Recap on Bayes' rule

As we had discussed in lecture 13, Bayes rule is important for Baysian updating. We will use the scenario above to develop our intutation on Bayesian updating. 

Let $X_i=1$ indicate a successful mission of the drone (meaning now flag of the engine health) and $X_i=0$ a flagged mission. The parameter $\theta$ is the success probability under the modeled operating conditions. Assume the mission outcomes are conditionally independent given the same $\theta$:

$$
p(X_i=x_i\mid\theta)=\theta^{x_i}(1-\theta)^{1-x_i},
\qquad x_i\in\{0,1\}.
$$

For the observed sequence $\mathcal D=(x_1,\ldots,x_N)$, we can define the success and failure counts as: 

$$
S=\sum_{i=1}^N x_i,
\qquad F=N-S.
$$

We can now write the likelihood as

$$
\mathcal L(\theta;\mathcal D)
=p(\mathcal D\mid\theta)
=\prod_{i=1}^N\theta^{x_i}(1-\theta)^{1-x_i}
=\theta^S(1-\theta)^F.
$$

For a candidate value of $\theta$, this is the probability of the observed sequence given that candidate. With the data fixed and $\theta$ varying, it is the likelihood function. The semicolon in $\mathcal L(\theta;\mathcal D)$ separates the parameter argument from the fixed data; it is not a conditional-probability symbol.

Bayes' rule gives

$$
p(\theta\mid\mathcal D)
=\frac{\mathcal L(\theta;\mathcal D)p(\theta)}{p(\mathcal D)},
\qquad
p(\mathcal D)=\int_0^1\mathcal L(\theta;\mathcal D)p(\theta)\,d\theta.
$$

| Quantity | Meaning |
|---|---|
| $p(\theta)$ | Prior density over candidate parameter values |
| $\mathcal L(\theta;\mathcal D)=p(\mathcal D\mid\theta)$ | Likelihood of the observed data as a function of the parameter |
| $p(\mathcal D)$ | Evidence, which normalizes the posterior |
| $p(\theta\mid\mathcal D)$ | Posterior density over parameter values after observing the data |

> Remember: The likelihood is not a probability distribution over $\theta$. The prior and posterior are distributions over $\theta$.

### 1.2 What makes a prior conjugate?

Now we turn to conjugacy: A prior family is **conjugate to a likelihood** when updating any prior in that family produces a posterior in the same family.

For a Bernoulli likelihood, a Beta prior produces a Beta posterior. The shape parameters change, but the distribution family stays the same. This lets us express the posterior using a small number of updated parameters, and makes computation very efficient. 

What is important: Conjugacy is a relationship between a prior family and a likelihood. It does not establish that the prior is appropriate for an application; that still depends on the assumptions it represents.

### 1.3 Derive the general update

Recall the Beta density from Lecture 13:

$$
p(\theta\mid a,b)
=\frac{1}{B(a,b)}\theta^{a-1}(1-\theta)^{b-1},
\qquad 0<\theta<1,\quad a,b>0,
$$

where

$$
B(a,b)=\int_0^1 t^{a-1}(1-t)^{b-1}\,dt
$$

is the normalizing constant. The positive numbers $a$ and $b$ are **hyperparameters**: parameters of the prior distribution for $\theta$.

Multiply the likelihood by the prior:

$$
\begin{aligned}
p(\theta\mid\mathcal D,a,b)
&\propto\theta^S(1-\theta)^F\theta^{a-1}(1-\theta)^{b-1}\\
&=\theta^{a+S-1}(1-\theta)^{b+F-1}.
\end{aligned}
$$

Compare the exponents with the Beta density. The new first shape parameter is $a+S$ and the new second shape parameter is $b+F$. Therefore,

$$
\boxed{\theta\mid\mathcal D,a,b\sim\operatorname{Beta}(a+S,b+F).}
$$

The normalized posterior density is

$$
p(\theta\mid\mathcal D,a,b)
=\frac{\theta^{a+S-1}(1-\theta)^{b+F-1}}{B(a+S,b+F)}.
$$

When $a,b$ are fixed and understood, we shorten the notation to $p(\theta\mid\mathcal D)$.

:::{admonition} The updating rule
:class: important

Each success adds one to the first Beta shape parameter. Each failure adds one to the second.
:::

For the Lecture 13 example,

$$
\mathcal D=(1,1,0,1),\qquad S=3,\quad F=1,
$$

so Beta(2,2) becomes Beta(5,3).

**Figure**  I will reuse the original Lecture 13 figure showing the Beta(2,2) prior and Beta(5,3) posterior for the four outcomes $(1,1,0,1)$. Horizontal axis “Mission-success probability, $\theta$” and the vertical axis “Probability density.” The two curves belong to the same family; the observations change their shape parameters.

### 1.4 Where did the evidence go?

Recognizing the Beta form gives us its normalizing constant directly. The evidence has not disappeared. For the particular observed sequence,

$$
\begin{aligned}
p(\mathcal D\mid a,b)
&=\frac{1}{B(a,b)}\int_0^1\theta^{a+S-1}(1-\theta)^{b+F-1}\,d\theta\\
&=\frac{B(a+S,b+F)}{B(a,b)}.
\end{aligned}
$$

Dividing the likelihood–prior product by this evidence gives the normalized posterior above. In the worked example, $B(2,2)=1/6$ and $B(5,3)=1/105$, so the evidence is $2/35$ and the posterior density is $105\theta^4(1-\theta)^2$.

If the recorded data are only a count of $S$ successes in $N$ trials, the likelihood includes $\binom NS$. The evidence includes that same factor, so it cancels in the posterior. A count and one particular sequence are different observed events, although they produce the same posterior for $\theta$ under this model.

### 1.5 Connection to sufficient statistics

The likelihood depends on the sequence through $S$ and $F$. For known $N$, knowing $S$ also determines $F$. Thus $S$ is sufficient for $\theta$ in this Bernoulli model, connects back to Lecture 12 where we introduced them. 

For example, $(1,1,0,1)$ and $(0,1,1,1)$ produce the same posterior from the same prior. Order does not change the likelihood under the shared-parameter, conditionally independent model. If reliability changes over time or outcomes remain dependent given $\theta$, we need a different model.

**Textbook reference for more details** Bishop, §§2.1.1 and 2.4.2; sufficient statistics in §2.4.1.

## 2. Bayesian learning as a sequential process

### 2.1 The posterior becomes the next prior

Now lets assume we take the Bayesian perspective forward, as we are collecting more data over time. Suppose we have already analyzed $\mathcal D_{\mathrm{old}}$ and then observe $\mathcal D_{\mathrm{new}}$. Under conditional independence of the batches given $\theta$,

$$
p(\theta\mid\mathcal D_{\mathrm{old}},\mathcal D_{\mathrm{new}})
\propto p(\mathcal D_{\mathrm{new}}\mid\theta)
\,p(\theta\mid\mathcal D_{\mathrm{old}}).
$$

The current posterior summarizes what we have learned about the same parameter. It supplies the prior distribution for the next update. The new likelihood contains only the new observations.

### 2.2 Continue the mission example

After the first four missions, the posterior is Beta(5,3). A fifth mission succeeds, and a sixth mission fails:

$$
\operatorname{Beta}(5,3)
\xrightarrow{X_5=1}\operatorname{Beta}(6,3)
\xrightarrow{X_6=0}\operatorname{Beta}(6,4).
$$

| Information incorporated | Cumulative successes | Cumulative failures | Distribution for $\theta$ |
|---|---:|---:|---|
| Before any mission | 0 | 0 | Beta(2,2) |
| First four outcomes $(1,1,0,1)$ | 3 | 1 | Beta(5,3) |
| Additional success $(1,1,0,1,1)$| 4 | 1 | Beta(6,3) |
| Additional failure $(1,1,0,1,1,0)$| 4 | 2 | Beta(6,4) |

![Sequential learning](images/prior_distribution.png)

**Figure caption:** Starting from a Beta(2,2) prior, three periods without a flag and one period with a flag yield a Beta(5,3) posterior. An additional period without a flag gives Beta(6,3); a subsequent period with a flag gives Beta(6,4). Each update multiplies the current density by the likelihood of the new observation and normalizes the result.

### 2.3 Why sequential and batch updates agree

Updating once with all six observations gives

$$
\operatorname{Beta}(2+4,2+2)=\operatorname{Beta}(6,4).
$$

More generally, batches with counts $(S_1,F_1)$ and $(S_2,F_2)$ give

$$
\operatorname{Beta}(a+S_1+S_2,b+F_1+F_2).
$$

The additions give the same result regardless of the grouping or order of observations. 
**Important**: This only holds if we assumes the same initial prior and sampling model throughout.

:::{warning}
When the current posterior becomes the next prior, do not multiply by the likelihood of the old data again. Their information is already included in that posterior.
:::

## 3. Choosing a Beta prior and checking sensitivity

### 3.1 What should the prior represent?

A prior describes uncertainty about $\theta$ before incorporating the data used in the current likelihood. It may reflect engineering knowledge, previous comparable experiments, or a deliberately broad starting assumption.

Comparability matters. Results obtained with different hardware, weather, or definitions of mission success may not describe the parameter we are studying. A convenient mathematical prior still needs a substantive justification.

### 3.2 Mean and concentration

For a Beta distribution,

$$
\mathbb E[\theta]=\frac{a}{a+b},
\qquad
\operatorname{Var}(\theta)=\frac{ab}{(a+b)^2(a+b+1)}.
$$

A useful way to specify it is through its mean $m$ and concentration $\kappa$:

$$
m=\frac{a}{a+b},\qquad \kappa=a+b.
$$

Then

$$
a=m\kappa,\qquad b=(1-m)\kappa,
\qquad
\operatorname{Var}(\theta)=\frac{m(1-m)}{\kappa+1}.
$$

The mean locates the prior's center of mass. At fixed mean, a larger concentration gives a smaller variance. Neither quantity is an observed mission outcome.

| Prior | Mean $m$ | Concentration $\kappa$ | Interpretation |
|---|---:|---:|---|
| Beta(1,1) | 0.50 | 2 | Uniform density on $(0,1)$ |
| Beta(2,2) | 0.50 | 4 | Symmetric, with more density near the center |
| Beta(10,10) | 0.50 | 20 | More concentrated around 0.50 |
| Beta(2,6) | 0.25 | 8 | Center of mass toward lower success probabilities |
| Beta(6,2) | 0.75 | 8 | Center of mass toward higher success probabilities |

For example, a chosen mean of 0.7 and concentration of 10 give $a=7$, $b=3$. A mean alone does not specify a Beta prior; we also need its concentration or another measure of spread.

Although the update adds counts to $a$ and $b$, prior shape parameters need not be integers or represent literal historical observations. Likewise, a uniform density in $\theta$ does not imply a uniform density after a nonlinear change of parameter.

### 3.3 Hold the observations fixed and vary the prior

Return to the original four outcomes $(1,1,0,1)$. Do not include the two additional missions from Section 2 in this comparison.

| Illustrative prior | Posterior after three successes and one failure |
|---|---|
| Beta(1,1) | Beta(4,2) |
| Beta(2,2) | Beta(5,3) |
| Beta(10,10) | Beta(13,11) |
| Beta(2,6) | Beta(5,7) |
| Beta(6,2) | Beta(9,3) |

Every row uses the same likelihood. Differences between posterior curves therefore arise from the prior assumptions. With four observations, those assumptions can have a substantial effect.

![Sensitivity analysis](images/sensitivity_prior.png)
**Figure caption:** Each panel shows one prior and its posterior from the table above. It illustrates the sensitivity to the prior. 

### 3.4 Explain the relative influence of prior and data

The posterior mean provides one simple way to describe the center of each curve:

$$
\begin{aligned}
\mathbb E[\theta\mid\mathcal D]
&=\frac{a+S}{a+b+N}\\
&=\frac{\kappa}{\kappa+N}m
+\frac{N}{\kappa+N}\frac{S}{N},\qquad N>0.
\end{aligned}
$$

It is a weighted average of the prior mean and the observed success proportion. For the Beta(2,2) example,

$$
\mathbb E[\theta\mid\mathcal D]
=\frac48\left(\frac12\right)+\frac48\left(\frac34\right)
=\frac58.
$$

For fixed $\kappa$, the prior's weight decreases as $N$ increases. This describes the center of the posterior; the full posterior also represents uncertainty around it. We are not introducing an optimization rule or replacing the distribution with a single estimate.

A prior-sensitivity analysis compares plausible priors using the same data. It is recommended that when you engage in data-driven decision making that, if the resulting distributions differ substantially, report that dependence rather than selecting a prior to obtain a preferred answer. Lecture 15 will extend the comparison to intervals and hypothesis probabilities.

## 4. Other conjugate families, and benefits and limits of conjugacy

### 4.1 Overview
Conjugacy also occurs in other familiar sampling models. The table gives an overview; Above, we have used the Beta–Bernoulli model as an example. We encourage you to explore other priors as well as you advance your statistical knowledge. However, for this class, we will primary use Beta-Bernoulli, and Normal-Normal. 

| Sampling model | Unknown parameter | Conjugate prior | Posterior update |
|---|---|---|---|
| Bernoulli or Binomial | Success probability $\theta$ | Beta$(a,b)$ | Beta$(a+S,b+F)$ |
| Categorical or Multinomial | Category-probability vector $\boldsymbol\theta$ | Dirichlet$(\alpha_1,\ldots,\alpha_K)$ | Dirichlet$(\alpha_1+n_1,\ldots,\alpha_K+n_K)$ |
| Poisson, equal unit observation periods | Event rate $\lambda$ | Gamma$(a,b)$, shape–rate convention | Gamma$(a+\sum_i x_i,b+N)$ |
| Normal, known observation variance $\sigma^2$ | Mean $\mu$ | Normal$(m_0,\tau_0^2)$ | Normal with updated mean and variance |

Here $n_k$ is the observed count in category $k$. The Dirichlet distribution extends the Beta distribution to a vector of probabilities that sum to one. For the Gamma row, the density is proportional to $\lambda^{a-1}e^{-b\lambda}$; $b$ is a rate, not a scale.

For the Normal row, the sampling model is $X_i\mid\mu\sim\mathcal N(\mu,\sigma^2)$ with conditionally independent observations. The posterior parameters are

$$
\tau_N^2=\left(\frac{1}{\tau_0^2}+\frac{N}{\sigma^2}\right)^{-1},
\qquad
m_N=\tau_N^2\left(\frac{m_0}{\tau_0^2}+\frac{\sum_i x_i}{\sigma^2}\right).
$$

> this is still not clear and requires more explanation! 

### 4.2 What conjugacy does and does not provide

Conjugacy makes Bayesian updating easier to calculate and repeat: the posterior belongs to the same distribution family as the prior.
Its main benefits are:

1. An explicit posterior formula. 
2. Straightforward normalization. (e.g. once we recognize the posterior as a Beta distribution, its known normalizing constant gives the normalized density. We do not need to calculate a new integral from scratch for every update.) 
3. Clear interpretation. The updating formula makes it easy to see how prior assumptions and observed data combine, and to compare results under different priors.

However, there are few things keep in mind: 
1. Conjugacy provides computational convenience, not a guarantee that the prior is appropriate. The prior must still represent reasonable assumptions about the unknown parameter. 
2. Also, sequential updating is possible without conjugacy; conjugacy makes the calculations particularly simple
3. Conjugacy provides a convenient analytic update. It does not determine the correct likelihood, select prior hyperparameters, or check whether observations satisfy the model assumptions.

> Keep in mind that Bayes' rule also applies to nonconjugate priors. The posterior may then require numerical integration or another approximation. Choosing a appropriate prior and computing its posterior are separate tasks.

**Textbook reference for more details:** Bishop §2.1.1 for Beta priors, §2.2 for the Dirichlet family, §2.3.6 for inference on Gaussian parameters, and §2.4.2 for general conjugacy.

## 5. Simple empirical Bayes 

### 5.1 Why use own data? 

Until now, we have specified $a$ and $b$ ourselves (assuming we have expert knowledge). But you may also rely on data to define your prior. Suppose we have mission records from several comparable drones and want a prior for a redesigned drone and its mission success. Those records can help us learn where a plausible prior should be centered.

This introduces a new assumption: since the different drones' mission have different success probabilities, but their probabilities are modeled as draws from one shared distribution. Write

$$
\theta_j\mid a,b\sim\operatorname{Beta}(a,b),
\qquad
S_j\mid\theta_j\sim\operatorname{Binomial}(n_j,\theta_j),
$$

for historical drones $j=1,\ldots,J$. Assume independent drones under this model and conditionally independent missions within each drone. Each drone's $\theta_j$ stays fixed during its recorded missions.

The shared Beta distribution describes variation in success probabilities across comparable drones. It supplies a prior for the unknown success probability of a new drone from the same population.

**Empirical Bayes** estimates prior hyperparameters from observed data and then uses those estimates in a Bayesian update. We will estimate only the prior mean and specify the concentration explicitly.

### 5.2 Historical data as a small teaching example

Consider these illustrative records, collected separately from the new drone's missions:

| Historical drone $j$ | Missions $n_j$ | Successes $S_j$ | Observed proportion $S_j/n_j$ |
|---|---:|---:|---:|
| A | 20 | 12 | 0.60 |
| B | 20 | 14 | 0.70 |
| C | 20 | 16 | 0.80 |
| D | 20 | 14 | 0.70 |

Each observed proportion is a noisy estimate of that drone's success probability. The proportions are not direct observations of the latent parameters $\theta_j$.

### 5.3 Estimate the shared prior mean

Under the model, with fixed mission counts,

$$
\mathbb E\!\left[\frac{S_j}{n_j}\,\middle|\,\theta_j\right]=\theta_j.
$$

Averaging also over the population of drones gives

$$
\mathbb E\!\left[\frac{S_j}{n_j}\right]
=\mathbb E[\theta_j]=m.
$$

We can therefore estimate $m$ by matching this expectation to the average observed proportion:

$$
\widehat m=\frac1J\sum_{j=1}^J\frac{S_j}{n_j}
=\frac{0.60+0.70+0.80+0.70}{4}=0.70.
$$

This is a simple **method-of-moments** estimate. Because every historical drone has the same number of missions, it also equals the pooled proportion $56/80$. For unequal mission counts, an unweighted average of drone proportions and a pooled proportion generally differ; they weight the drones differently.

### 5.4 Specify concentration and construct the fitted prior

The estimated mean does not determine the prior's spread. For this introductory example, choose $\kappa=10$ and state that choice explicitly. Then

$$
\widehat a=\widehat m\kappa=7,
\qquad
\widehat b=(1-\widehat m)\kappa=3.
$$

The fitted prior for the new drone is

$$
\theta_{\mathrm{new}}\mid\widehat m,\kappa
\sim\operatorname{Beta}(7,3).
$$

Its variance is $0.7(0.3)/11\approx0.0191$. We estimated its center from historical data; we did not estimate this concentration from those data. The historical total of 80 missions is not the concentration assigned to the new drone's prior.

:::{admonition} What was learned and what was chosen?
:class: important

The Beta family was chosen as a model. Its mean was estimated from historical drone records. Its concentration was fixed at 10 for the teaching example. All three choices should be visible in the analysis.
:::

### 5.5 Update using the new drone's observations

Suppose the new drone completes four missions with outcomes

$$
\mathcal D_{\mathrm{new}}=(1,1,0,1).
$$

These are this new drone's data in a separate worked example. They are not additional observations appended to the six missions in Section 2.

Its likelihood is

$$
\mathcal L(\theta_{\mathrm{new}};\mathcal D_{\mathrm{new}})
=\theta_{\mathrm{new}}^3(1-\theta_{\mathrm{new}}).
$$

Using the fitted prior gives

$$
\begin{aligned}
p_{\mathrm{EB}}(\theta_{\mathrm{new}}\mid\mathcal D_{\mathrm{new}},\widehat m,\kappa)
&\propto\theta_{\mathrm{new}}^3(1-\theta_{\mathrm{new}})
\theta_{\mathrm{new}}^{7-1}(1-\theta_{\mathrm{new}})^{3-1}\\
&=\theta_{\mathrm{new}}^{10-1}(1-\theta_{\mathrm{new}})^{4-1}.
\end{aligned}
$$

Thus the empirical Bayes posterior, conditional on the fitted hyperparameters, is Beta(10,4). Its mean is $10/14\approx0.714$, between the fitted prior mean 0.70 and the new drone's observed proportion 0.75.

**Figure instruction/caption:** Use two panels. Left: show the historical proportions 0.60, 0.70, 0.80, and 0.70, with a horizontal line at their mean 0.70; label them as observed proportions, not true parameter values. Right: show the fitted Beta(7,3) prior and Beta(10,4) posterior on the parameter axis. State “Prior mean estimated from historical drones; concentration fixed at 10.”

### 5.7 Limitations of this simple approach and looking ahead. 

The method relies on historical drones being relevant to the new drone. It also treats the estimated hyperparameters as fixed during the update. The Beta(10,4) posterior therefore represents uncertainty conditional on those fitted values; it does not propagate uncertainty in estimating the shared prior mean from only four historical drones.


## Summary: Benefits and downside of conjugate priors

Conjugacy lets us express the posterior in the same family as the prior. For Beta–Bernoulli learning, successes and failures update the two shape parameters by addition. 

Mean and concentration provide an interpretable way to specify a Beta prior and examine sensitivity. 


In Lecture 15, we will use these posterior distributions to construct credible intervals, calculate probabilities for parameter hypotheses, and interpret Bayes factors.

## Practice examples 

### 1. Sequential versus batch updating

Start with Beta(2,2). Observe $(1,1,0,1)$, followed by $(1,0)$.

1. Find the posterior after each batch.
2. Calculate the posterior using all six observations at once.
3. Explain why the answers agree.

::::{admonition} Solution 1
:class: dropdown

The first batch contains three successes and one failure, giving Beta(5,3). The second batch adds one success and one failure, giving Beta(6,4).

All six observations contain four successes and two failures. A batch update gives Beta$(2+4,2+2)=$ Beta(6,4). The conditionally independent likelihood factors across batches, and both approaches include every observation once.
::::

### 2. Specify a Beta prior

Choose a prior mean of 0.8 and concentration 5. Then observe two successes and one failure.

1. Find $a,b$ and the posterior parameters.
2. Repeat with concentration 20 and the same prior mean.
3. Explain which prior gives its mean greater weight in the posterior mean.

::::{admonition} Solution 2
:class: dropdown

With $\kappa=5$, $a=4$ and $b=1$. The posterior is Beta(6,2).

With $\kappa=20$, $a=16$ and $b=4$. The posterior is Beta(18,5).

The prior-mean weights are $5/(5+3)$ and $20/(20+3)$, respectively. The higher concentration gives the prior mean more weight. The posterior means are $6/8=0.75$ and $18/23\approx0.783$.
::::

### 3. Sequence likelihood and count likelihood

With a Beta(1,1) prior, compare observing the particular sequence $(1,0,1)$ with observing only that two of three missions succeeded.

1. Write both likelihoods.
2. Find the evidence for each observation.
3. Explain why the posteriors agree.

::::{admonition} Solution 3
:class: dropdown

The sequence likelihood is $\theta^2(1-\theta)$. The count likelihood is $\binom32\theta^2(1-\theta)=3\theta^2(1-\theta)$.

The sequence evidence is

$$
\int_0^1\theta^2(1-\theta)\,d\theta=\frac1{12}.
$$

The count evidence is $3/12=1/4$. In both cases, normalization gives $12\theta^2(1-\theta)$, the Beta(3,2) density. The extra factor is constant in $\theta$ and cancels.
::::

### 4. A simple empirical Bayes update

Three comparable historical drones each flew 10 missions, with 6, 8, and 7 successes. Fix the shared prior concentration at 10. A new drone then records one success and one failure.

1. Estimate the shared prior mean.
2. Construct the fitted prior and update it for the new drone.
3. Identify one modeling assumption and one source of uncertainty omitted by the plug-in calculation.

::::{admonition} Solution 4
:class: dropdown

The estimated mean is $(0.6+0.8+0.7)/3=0.7$. With concentration 10, the fitted prior is Beta(7,3). The new drone's two observations give Beta(8,4).

A modeling assumption is that the historical drones and the new drone have success probabilities drawn from the same relevant population distribution. The plug-in posterior does not propagate uncertainty in the mean estimated from the three historical drones. The concentration was chosen rather than estimated.
::::

## Readings and references

### Primary textbook

Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. [Official textbook PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf).

- **§2.1.1:** Beta priors and Bernoulli updating. Focus on Equations (2.13)–(2.18) and Figure 2.3; prediction is outside this lecture's scope.
- **§2.4.1:** sufficient statistics, reviewing Lecture 12.
- **§2.4.2:** the general idea of conjugate priors.
- **Optional §2.2 and §2.3.6:** other conjugate families represented in Section 4.
- **Optional §2.4.3:** noninformative priors and parameterization.

Bishop uses $\mu$ for the Bernoulli parameter in §2.1.1; this lecture retains $\theta$ from Lecture 13. The underlying Beta update is the same. The simple empirical Bayes example here is a teaching extension, not a worked example attributed to Bishop's Beta section.

### Supplemental reading

Johnson, A. A., Ott, M. Q., and Dogucu, M. (2022). *Bayes Rules! An Introduction to Applied Bayesian Modeling*. Chapman & Hall/CRC. [Chapter 4: Balance and Sequentiality in Bayesian Analyses](https://www.bayesrulesbook.com/chapter-4.html). Use §§4.1–4.3 for prior influence and §4.4 for sequential updating.

For further background on learning Beta prior hyperparameters from related Binomial datasets, see the empirical Bayes example in [Chapter 2 coding materials for *Bayesian Statistics: A Primer for Data Scientists, with R and Python*](https://statistique.cuso.ch/fileadmin/statistique/user_upload/CODE_CH2.html), by Agresti, Kateri, and Mira. That example estimates both Beta parameters through a marginal Beta–Binomial likelihood. Our fixed-concentration example estimates only the mean to keep the calculation introductory.

### Accompanying video

Steve Brunton, [*Bayesian Updates and Conjugate Priors*](https://www.youtube.com/watch?v=GqSX-8AQL90).

