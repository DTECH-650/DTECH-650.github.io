# Lecture 14 — Conjugate Priors, Prior Choice, and Empirical Bayes

## Introduction and guiding question

In Lecture 13, we used Bayes' rule with a Beta(2,2) prior and three observations coded 1 and one coded 0 to obtain a Beta(5,3) posterior. We now use the same binary model to study engine-health flags.

That calculation raises two practical questions. **Why did the posterior remain a Beta distribution? And where did the original values 2 and 2 come from?**

**Scenario:** We monitor an engine using a fixed sensor-based rule for raising an engine-health flag. For each monitoring period of the same duration, we record whether at least one flag was raised. We restrict attention to comparable operating conditions and assume the engine's condition and the flagging rule remain stable during the observations.

We code a period with no flag as $X_i=1$ and a period with a flag as $X_i=0$. Our unknown parameter $\theta$ is the probability of no flag during one such period. We want to update our knowledge of this probability as observations arrive. Records from other comparable engines may also help us specify a prior.

A sensor flag is an observed indicator, not a direct observation of true engine health. This model describes flag occurrence; it does not estimate the probability of a fault given sensor measurements.

> How can we choose a prior, update it efficiently as data arrive, and assess how sensitive the posterior is to that choice?

This lecture develops conjugate updating, prior specification, and prior sensitivity. We also introduce a simple empirical Bayes example that estimates a prior mean from historical engine records while fixing its concentration. This calculation does not require Monte Carlo sampling. Lecture 15 will use posterior distributions to construct credible intervals and evaluate hypotheses.

## Learning objectives

After completing this lecture, you should be able to:

1. Explain conjugacy and derive the Beta–Bernoulli posterior update.
2. Apply sequential updating and show why it agrees with batch updating under the same model.
3. Interpret the mean and concentration of a Beta prior and assess prior sensitivity.
4. Recognize several common likelihood–conjugate-prior pairs.
5. Estimate a Beta prior's mean from related datasets in a simple empirical Bayes example with fixed concentration.

## 1. Conjugate priors and the Beta–Bernoulli update

### 1.1 Recap on Bayes' rule

We use the engine-monitoring scenario to connect each step of Bayesian updating to the observed data.

Let $X_i=1$ indicate that no engine-health flag was recorded during monitoring period $i$, and $X_i=0$ indicate that at least one flag was recorded. The parameter $\theta$ is the probability of a period with no flag under the specified conditions. Assume the period outcomes are conditionally independent given the same fixed $\theta$:

$$
p(X_i=x_i\mid\theta)=\theta^{x_i}(1-\theta)^{1-x_i},
\qquad x_i\in\{0,1\}.
$$

For the observed sequence $\mathcal D=(x_1,\ldots,x_N)$, define $S$ as the number of periods with no flag and $F$ as the number with a flag: 

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

Each period with no flag adds one to the first Beta shape parameter. Each period with a flag adds one to the second.
:::

Using the same numerical sequence as Lecture 13, now interpreted as engine-monitoring outcomes,

$$
\mathcal D=(1,1,0,1),\qquad S=3,\quad F=1,
$$

so Beta(2,2) becomes Beta(5,3).

**Figure**  I will reuse the original Lecture 13 figure showing the Beta(2,2) prior and Beta(5,3) posterior, relabeled for the engine-monitoring scenario, for the four outcomes $(1,1,0,1)$. Horizontal axis “Probability of no engine-health flag, $\theta$” and the vertical axis “Probability density.” The two curves belong to the same family; the observations change their shape parameters.

### 1.4 Where did the evidence go?

In Section 1.3, we recognized the shape of the posterior using proportionality. We now keep all the constants and substitute the evidence into Bayes' rule to obtain the normalized density.

**Step 1: Write Bayes' rule with all four components.**

$$
\underbrace{p(\theta\mid\mathcal D,a,b)}_{\text{posterior}}
=\frac{
\underbrace{p(\mathcal D\mid\theta)}_{\text{likelihood}}
\underbrace{p(\theta\mid a,b)}_{\text{prior}}
}{
\underbrace{p(\mathcal D\mid a,b)}_{\text{evidence}}
}.
$$

Here $a,b$ are fixed prior hyperparameters. For the particular observed sequence, the likelihood–prior product is

$$
\begin{aligned}
p(\mathcal D\mid\theta)p(\theta\mid a,b)
&=\theta^S(1-\theta)^F
\frac{\theta^{a-1}(1-\theta)^{b-1}}{B(a,b)}\\
&=\frac{\theta^{a+S-1}(1-\theta)^{b+F-1}}{B(a,b)}.
\end{aligned}
$$

**Step 2: Calculate the evidence by integrating this product.**

$$
\begin{aligned}
p(\mathcal D\mid a,b)
&=\int_0^1 p(\mathcal D\mid t)p(t\mid a,b)\,dt\\
&=\frac{1}{B(a,b)}\int_0^1 t^{a+S-1}(1-t)^{b+F-1}\,dt\\
&=\frac{B(a+S,b+F)}{B(a,b)}.
\end{aligned}
$$

The last step uses the definition of the Beta function from Section 1.3. We use $t$ as the integration variable to distinguish it from the candidate parameter value $\theta$ in the posterior density. The evidence is a number for the fixed data and prior hyperparameters; it does not depend on $\theta$.

**Step 3: Substitute the evidence and simplify.**

$$
\begin{aligned}
p(\theta\mid\mathcal D,a,b)
&=\frac{
\displaystyle \frac{\theta^{a+S-1}(1-\theta)^{b+F-1}}{B(a,b)}
}{
\displaystyle \frac{B(a+S,b+F)}{B(a,b)}
}\\[6pt]
&=\frac{\theta^{a+S-1}(1-\theta)^{b+F-1}}{B(a,b)}
\frac{B(a,b)}{B(a+S,b+F)}\\[6pt]
&=\frac{\theta^{a+S-1}(1-\theta)^{b+F-1}}{B(a+S,b+F)}.
\end{aligned}
$$

Dividing by the evidence means multiplying by its reciprocal. The factor $B(a,b)$ then cancels, leaving the normalizing factor $1/B(a+S,b+F)$. This is exactly the Beta$(a+S,b+F)$ density.

:::{admonition} What cancels, and what remains?
:class: important

The **prior's normalizing factor** cancels with the matching factor in the evidence. The evidence itself is not simply discarded: dividing by it produces the normalized posterior. The likelihood remains incorporated in the updated exponents through $S$ and $F$.
:::

**Step 4: Check the numerical example.**

For $a=b=2$, $S=3$, and $F=1$, the likelihood and prior are

$$
p(\mathcal D\mid\theta)=\theta^3(1-\theta),
\qquad
p(\theta\mid 2,2)=6\theta(1-\theta).
$$

Since $B(2,2)=1/6$ and $B(5,3)=1/105$, the evidence is

$$
p(\mathcal D\mid 2,2)
=\frac{1/105}{1/6}=\frac{2}{35}.
$$

Substituting these three quantities into Bayes' rule gives

$$
\begin{aligned}
p(\theta\mid\mathcal D,2,2)
&=\frac{
\underbrace{\theta^3(1-\theta)}_{\text{likelihood}}
\underbrace{6\theta(1-\theta)}_{\text{prior}}
}{\underbrace{2/35}_{\text{evidence}}}\\[6pt]
&=6\theta^4(1-\theta)^2\frac{35}{2}\\
&=105\theta^4(1-\theta)^2.
\end{aligned}
$$

Its integral is $105B(5,3)=1$, confirming that it is normalized.

**If the data are a count rather than a particular sequence.**

Let $S$ denote the observed count of periods with no flag in $N$ trials, with $F=N-S$. The count likelihood and its evidence both contain the binomial coefficient:

$$
p(S\mid N,\theta)=\binom NS\theta^S(1-\theta)^F,
\qquad
p(S\mid N,a,b)=\binom NS\frac{B(a+S,b+F)}{B(a,b)}.
$$

Consequently,

$$
\begin{aligned}
p(\theta\mid S,N,a,b)
&=\frac{
\displaystyle \binom NS\frac{\theta^{a+S-1}(1-\theta)^{b+F-1}}{B(a,b)}
}{
\displaystyle \binom NS\frac{B(a+S,b+F)}{B(a,b)}
}\\[6pt]
&=\frac{\theta^{a+S-1}(1-\theta)^{b+F-1}}{B(a+S,b+F)}.
\end{aligned}
$$

Both the binomial coefficient and the prior normalizing factor cancel. A count and one particular sequence have different evidence values, but they produce the same posterior for $\theta$ under this model.

### 1.5 Connection to sufficient statistics

The likelihood depends on the sequence through $S$ and $F$. For known $N$, knowing $S$ also determines $F$. Thus $S$ is sufficient for $\theta$ in this Bernoulli model, connecting this update to Lecture 12. 

For example, $(1,1,0,1)$ and $(0,1,1,1)$ produce the same posterior from the same prior. Order does not change the likelihood under the shared-parameter, conditionally independent model. If the flag probability changes over time or outcomes remain dependent given $\theta$, we need a different model.

**Textbook reference for more details** Bishop, §§2.1.1 and 2.4.2; sufficient statistics in §2.4.1.

## 2. Bayesian learning as a sequential process

### 2.1 The posterior becomes the next prior

We now collect additional engine-monitoring observations over time. Suppose we have already analyzed $\mathcal D_{\mathrm{old}}$ and then observe $\mathcal D_{\mathrm{new}}$. Under conditional independence of the batches given $\theta$,

$$
p(\theta\mid\mathcal D_{\mathrm{old}},\mathcal D_{\mathrm{new}})
\propto p(\mathcal D_{\mathrm{new}}\mid\theta)
\,p(\theta\mid\mathcal D_{\mathrm{old}}).
$$

The current posterior summarizes what we have learned about the same parameter. It supplies the prior distribution for the next update. The new likelihood contains only the new observations.

### 2.2 Continue the engine-monitoring example

After the first four monitoring periods, the posterior is Beta(5,3). No flag is recorded in the fifth period, and a flag is recorded in the sixth:

$$
\operatorname{Beta}(5,3)
\xrightarrow{X_5=1}\operatorname{Beta}(6,3)
\xrightarrow{X_6=0}\operatorname{Beta}(6,4).
$$

| Information incorporated | Cumulative periods without a flag | Cumulative periods with a flag | Distribution for $\theta$ |
|---|---:|---:|---|
| Before monitoring begins | 0 | 0 | Beta(2,2) |
| First four outcomes $(1,1,0,1)$ | 3 | 1 | Beta(5,3) |
| Additional period without a flag | 4 | 1 | Beta(6,3) |
| Additional period with a flag | 4 | 2 | Beta(6,4) |

**Figure caption:** I need to Plot Beta(5,3), Beta(6,3), and Beta(6,4) on common axes. Labels - “After four monitoring periods,” “After one additional period without a flag,” and “After one additional period with a flag.” 

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
**Important**: This only holds both calculation routes start from the same original prior and use the same model with a fixed parameter $\theta$.

:::{warning}
When the current posterior becomes the next prior, do not multiply by the likelihood of the old data again. Their information is already included in that posterior.
:::

## 3. Choosing a Beta prior and checking sensitivity

### 3.1 What should the prior represent?

A prior describes uncertainty about $\theta$ before incorporating the data used in the current likelihood. It may reflect engineering knowledge, previous comparable experiments, or a deliberately broad starting assumption.

Comparability matters. Results obtained with different hardware, weather, or definitions of the flagging rule may not describe the parameter we are studying. A convenient mathematical prior still needs a substantive justification.

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

The mean locates the prior's center of mass. At fixed mean, a larger concentration gives a smaller variance. Neither quantity is an observed monitoring-period outcome.

| Prior | Mean $m$ | Concentration $\kappa$ | Interpretation |
|---|---:|---:|---|
| Beta(1,1) | 0.50 | 2 | Uniform density on $(0,1)$ |
| Beta(2,2) | 0.50 | 4 | Symmetric, with more density near the center |
| Beta(10,10) | 0.50 | 20 | More concentrated around 0.50 |
| Beta(2,6) | 0.25 | 8 | Center of mass toward lower probabilities of no flag |
| Beta(6,2) | 0.75 | 8 | Center of mass toward higher probabilities of no flag |

For example, a chosen mean of 0.7 and concentration of 10 give $a=7$, $b=3$. A mean alone does not specify a Beta prior; we also need its concentration or another measure of spread.

Although the update adds counts to $a$ and $b$, prior shape parameters need not be integers or represent literal historical observations. Likewise, a uniform density in $\theta$ does not imply a uniform density after a nonlinear change of parameter.

### 3.3 Hold the observations fixed and vary the prior

Return to the original four outcomes $(1,1,0,1)$. Do not include the two additional monitoring periods from Section 2 in this comparison.

| Illustrative prior | Posterior after three periods with no flag and one with a flag |
|---|---|
| Beta(1,1) | Beta(4,2) |
| Beta(2,2) | Beta(5,3) |
| Beta(10,10) | Beta(13,11) |
| Beta(2,6) | Beta(5,7) |
| Beta(6,2) | Beta(9,3) |

Every row uses the same likelihood. Differences between posterior curves therefore arise from the prior assumptions. With four observations, those assumptions can have a substantial effect.

**Figure caption:** Five small panels with common horizontal and vertical scales. Each panel shows one prior and its posterior from the table. Also I will add a code snippet underneath to reproduce the figure. 

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

It is a weighted average of the prior mean and the observed proportion of periods with no flag. For the Beta(2,2) example,

$$
\mathbb E[\theta\mid\mathcal D]
=\frac48\left(\frac12\right)+\frac48\left(\frac34\right)
=\frac58.
$$

For fixed $\kappa$, the prior's weight decreases as $N$ increases. This describes the center of the posterior; the full posterior also represents uncertainty around it. We are not introducing an optimization rule or replacing the distribution with a single estimate.

A prior-sensitivity analysis compares scientifically plausible priors using the same data. If the resulting distributions differ substantially, report that dependence rather than selecting a prior to obtain a preferred answer. Lecture 15 will extend the comparison to intervals and hypothesis probabilities.

## 4. Other conjugate families, and benefits and limits of conjugacy

### 4.1 Overview
Conjugacy also occurs in other familiar sampling models. The table gives an overview. Above, we have used the Beta–Bernoulli model as an example. We encourage you to explore other priors as well as you advance your statistical knowledge. However, for this class, we will primarily use Beta–Bernoulli and Normal–Normal models. 

| Sampling model | Unknown parameter | Conjugate prior | Posterior update |
|---|---|---|---|
| Bernoulli or Binomial | Probability of $X=1$ $\theta$ | Beta$(a,b)$ | Beta$(a+S,b+F)$ |
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


### 4.2 What conjugacy does and does not provide

Conjugacy makes Bayesian updating easier to calculate and repeat: the posterior belongs to the same distribution family as the prior.
Its main benefits are:

1. An explicit posterior formula. 
2. Straightforward normalization. (e.g. once we recognize the posterior as a Beta distribution, its known normalizing constant gives the normalized density. We do not need to calculate a new integral from scratch for every update.) 
3. Clear interpretation. The updating formula makes it easy to see how prior assumptions and observed data combine, and to compare results under different priors.

However, there are a few things to keep in mind: 
1. Conjugacy provides computational convenience, not a guarantee that the prior is appropriate. The prior must still represent reasonable assumptions about the unknown parameter. 
2. Also, sequential updating is possible without conjugacy; conjugacy makes the calculations particularly simple
3. Conjugacy provides a convenient analytic update. It does not determine the correct likelihood, select prior hyperparameters, or check whether observations satisfy the model assumptions.

> Keep in mind that Bayes' rule also applies to nonconjugate priors. The posterior may then require numerical integration or another approximation. Choosing an appropriate prior and computing its posterior are separate tasks.

**Textbook reference for more details:** Bishop §2.1.1 for Beta priors, §2.2 for the Dirichlet family, §2.3.6 for inference on Gaussian parameters, and §2.4.2 for general conjugacy.

## 5. Simple empirical Bayes 

### 5.1 Learning a prior from related engine records 

Until now, we have specified $a$ and $b$ ourselves (assuming we have expert knowledge). But you may also rely on data to define your prior. Suppose we have engine-monitoring records from several comparable engines and want a prior for another engine monitored under the same conditions. Those records can help us learn where a plausible prior should be centered.

This introduces a new assumption: different engines may have different probabilities of no flag, but these probabilities are modeled as draws from one shared distribution. Write

$$
\theta_j\mid a,b\sim\operatorname{Beta}(a,b),
\qquad
S_j\mid\theta_j\sim\operatorname{Binomial}(n_j,\theta_j),
$$

for historical engines $j=1,\ldots,J$. Assume independent engines under this model and conditionally independent period outcomes within each engine. All records use the same monitoring duration and flagging rule under comparable operating conditions. Each engine's $\theta_j$ stays fixed during its recorded monitoring periods.

The shared Beta distribution describes variation in probabilities of no flag across comparable engines. It supplies a prior for the unknown probability of no flag for a new engine from the same population.

**Empirical Bayes** estimates prior hyperparameters from observed data and then uses those estimates in a Bayesian update. We will estimate only the prior mean and specify the concentration explicitly.

### 5.2 Historical data as a small teaching example

Consider these illustrative records, collected separately from the new engine's monitoring periods:

| Historical engine $j$ | Monitoring periods $n_j$ | Periods without a flag $S_j$ | Observed proportion $S_j/n_j$ |
|---|---:|---:|---:|
| A | 20 | 12 | 0.60 |
| B | 20 | 14 | 0.70 |
| C | 20 | 16 | 0.80 |
| D | 20 | 14 | 0.70 |

Each observed proportion is a noisy estimate of that engine's probability of no flag. These proportions measure the frequency of periods without flags, not true engine health. They are not direct observations of the latent parameters $\theta_j$.

### 5.3 Estimate the shared prior mean

Under the model, with fixed numbers of monitoring periods,

$$
\mathbb E\!\left[\frac{S_j}{n_j}\,\middle|\,\theta_j\right]=\theta_j.
$$

Averaging also over the population of engines gives

$$
\mathbb E\!\left[\frac{S_j}{n_j}\right]
=\mathbb E[\theta_j]=m.
$$

We can therefore estimate $m$ by matching this expectation to the average observed proportion:

$$
\widehat m=\frac1J\sum_{j=1}^J\frac{S_j}{n_j}
=\frac{0.60+0.70+0.80+0.70}{4}=0.70.
$$

This is a simple **method-of-moments** estimate. Because every historical engine has the same number of monitoring periods, it also equals the pooled proportion $56/80$. For unequal numbers of monitoring periods, an unweighted average of engine proportions and a pooled proportion generally differ; they weight the engines differently.

### 5.4 Specify concentration and construct the fitted prior

The estimated mean does not determine the prior's spread. For this introductory example, choose $\kappa=10$ and state that choice explicitly. Then

$$
\widehat a=\widehat m\kappa=7,
\qquad
\widehat b=(1-\widehat m)\kappa=3.
$$

The fitted prior for the new engine is

$$
\theta_{\mathrm{new}}\mid\widehat m,\kappa
\sim\operatorname{Beta}(7,3).
$$

Its variance is $0.7(0.3)/11\approx0.0191$. We estimated its center from historical data; we did not estimate this concentration from those data. The historical total of 80 monitoring periods is not the concentration assigned to the new engine's prior.

:::{admonition} What was learned and what was chosen?
:class: important

The Beta family was chosen as a model. Its mean was estimated from historical engine records. Its concentration was fixed at 10 for the teaching example. All three choices should be visible in the analysis.
:::

### 5.5 Update using the new engine's observations

Suppose we observe the new engine for four monitoring periods with outcomes

$$
\mathcal D_{\mathrm{new}}=(1,1,0,1).
$$

These are this new engine's data in a separate worked example. They are not additional observations appended to the six monitoring periods in Section 2.

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

Thus the empirical Bayes posterior, conditional on the fitted hyperparameters, is Beta(10,4). Its mean is $10/14\approx0.714$, between the fitted prior mean 0.70 and the new engine's observed proportion 0.75.

**Figure instruction/caption:** Use two panels. Left: show the historical proportions 0.60, 0.70, 0.80, and 0.70, with a horizontal line at their mean 0.70; label them as observed proportions, not true parameter values. Right: show the fitted Beta(7,3) prior and Beta(10,4) posterior on the parameter axis. State “Prior mean estimated from historical engines; concentration fixed at 10.”

### Limitations of this simple approach 

The method relies on historical engines being relevant to the new engine. It also treats the estimated hyperparameters as fixed during the update. The Beta(10,4) posterior therefore represents uncertainty conditional on those fitted values; it does not propagate uncertainty in estimating the shared prior mean from only four historical engines.


## Summary: Benefits and downside of conjugate priors

Conjugacy lets us express the posterior in the same family as the prior. For Beta–Bernoulli learning, counts of periods without and with flags update the two shape parameters by addition. 

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

The first batch contains three periods with no flag and one with a flag, giving Beta(5,3). The second batch adds one period with no flag and one with a flag, giving Beta(6,4).

All six observations contain four periods with no flag and two with a flag. A batch update gives Beta$(2+4,2+2)=$ Beta(6,4). The conditionally independent likelihood factors across batches, and both approaches include every observation once.
::::

### 2. Specify a Beta prior

Choose a prior mean of 0.8 and concentration 5. Then observe two periods with no flag and one with a flag.

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

With a Beta(1,1) prior, compare observing the particular sequence $(1,0,1)$ with observing only that two of three monitoring periods had no flag.

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

Three comparable historical engines were each observed for 10 monitoring periods, with 6, 8, and 7 periods with no flag. Fix the shared prior concentration at 10. A new engine then records one period with no flag and one with a flag.

1. Estimate the shared prior mean.
2. Construct the fitted prior and update it for the new engine.
3. Identify one modeling assumption and one source of uncertainty omitted by the plug-in calculation.

::::{admonition} Solution 4
:class: dropdown

The estimated mean is $(0.6+0.8+0.7)/3=0.7$. With concentration 10, the fitted prior is Beta(7,3). The new engine's two observations give Beta(8,4).

A modeling assumption is that the historical engines and the new engine have probabilities of no flag drawn from the same relevant population distribution. The plug-in posterior does not propagate uncertainty in the mean estimated from the three historical engines. The concentration was chosen rather than estimated.
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

