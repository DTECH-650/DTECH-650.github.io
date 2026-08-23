# Lecture 9 - Frequentist Uncertainty and Hypotheses Testing (I/II)

In prior lectures, we have established a frequentist perspective towards paramater learning. In this lecturer, we deepen the classical frequentist perspective, with a focus on sampling statistics and hypotheses testing. Specifically, we focus on statistical decision theory for a special case where we might want to reject or accept a hypotheses for a population, given a finite number of sample data at hand. 

Thus, before we discuss the details of hypotheses testing, it is important to understand the mathematical foundations of **finite** sampling through the frequentist lens: 

Thus, the guiding question for this lecture is: 

> How can a **finite** random sample provide a reliable estimate of an unknown population quantity?

Thus, the concept of a **finite** random sample makes this lecturer distinct from the prior lectures in week4. 

This lecture will be guided by the following learning objectives. 

## Learning objectives

After completing this lecture, you should be able to:

1. Distinguish between a population, a sample, a population parameter, an estimator, and an observed estimate.
2. Explain why an estimator such as the sample mean is a random variable. 
3. Distinguish among population distribution, empirical distribution of one sample, the sampling distribution of an estimator. 
4. Derive the standard error of the sample mean. 
5. Explain the central limit theorem
6. Construct and interpret a confidence interval using repeated-sampling coverage (rather than as a posterior probability) 
7. Use Python simulation to examine sampling variability, the CLT, standard error, and confidence-interval coverage.

<!-- 
## Learning objectives

After completing this lecture, you should be able to:

- Distinguish between a population, a sample, a population parameter, an estimator, and an observed estimate.
- Explain why an estimator such as the sample mean is a random variable. 
- Connect the maximum-likelihood estimates introduced in Week 4 to their sampling distributions.
- Distinguish among population distribution, empirical distribution of one sample, the sampling distribution of an estimator. 
- Derive the standard error of the sample mean. 
- Explain the central limit theorem
- Analyze how sample size and population variability affect estimation uncertainty.
- Construct a confidence interval for a population mean under stated assumptions.
- Interpret a confidence interval using repeated-sampling coverage rather than as a posterior probability.
- Use Python simulation to examine sampling variability, the CLT, standard error, and confidence-interval coverage.-->

## From point estimation to uncertainty

In prior lectures, we treated model parameters as fixed but unknown and derived estimators from the likelihood. For example, the Gaussian maximum-likelihood estimator of the population mean is the sample mean:

$$
\widehat{\mu}_{\mathrm{ML}}=\overline{X}.
$$

Before the sample is observed, $\overline{X}$ is a random variable. After observing the particular data values $x_1,\ldots,x_N$, we obtain the numerical estimate

$$
\widehat{\mu}_{\mathrm{ML}}=\overline{x}
=\frac{1}{N}\sum_{i=1}^{N}x_i.
$$

A different random sample would generally produce a different estimate. If repeated samples produced

$$
\overline{X}^{(1)},\overline{X}^{(2)},\ldots,\overline{X}^{(R)},
$$

our central question would be:

> How widely do these estimates vary around the true population mean $\mu$?

This question moves us from **point estimation** to **quantifying uncertainty**.

## 2. Population and random sampling
In frequentist statistics, the term *population* is a very widely used, and it is important to understand uncertainty from a frequentist perspective. 

The word *population* is used in two related ways in statistics, namely in a finite-population setting and a model-based formulation. We begin with the finite-population setting because it makes random sampling concrete. We then introduce the model-based formulation used in most machine-learning theory.

### 2.1 Finite population

Suppose a fleet contains $M=1{,}000$ engines. Let

$$
\mathcal{P}_M=\{z_1,z_2,\ldots,z_M\}
$$

denote a fixed measurement associated with each engine—for example, its operating temperature during a specified test. The finite-population mean is

$$
\mu_M=\frac{1}{M}\sum_{j=1}^{M}z_j.
$$

The $M$ population values are fixed. Randomness enters because we select only some of the engines for inspection.

If we sample $N=50$ engines, then

$$
M=1{,}000 \quad \text{and} \quad N=50.
$$

Thus:

- $M$ is the number of units in the finite population;
- $N$ is the number of units selected for the sample.

:::{important}
There are only two quantities here. Some statistics texts, including Rice, use $N$ for population size and $n$ for sample size. In this course, we use $M$ for finite-population size and retain $N$ for sample size to remain consistent with Lecture 7.
:::

| Concept | Notation in this course  |
|---|---:|---:|
| Finite-population size | $M$ |
| Sample size | $N$|
| Example | $M=1{,}000$, $N=50$ |

> We use N for sample size throughout the course. When discussing a finite population, we use M for the total number of population units. Some statistics texts, including Rice (2007), instead use N for population size and n for sample size. The difference is purely notational.

A finite-population sample selects N units from M existing units. By contrast, the model-based formulation used in machine learning more broadly treats observations as draws from a probability distribution. We introduce finite sampling above to clarify this special case that is important for this lectures on sampling and hypotheses testing. 

### 2.2 Simple random sampling without replacement

A **simple random sample without replacement** of size $N$ is selected so that every subset of $N$ distinct population units has the same probability of being selected.

There are

$$
\binom{M}{N}
$$

possible subsets, so the probability of selecting any particular subset is

$$
P(\text{a particular sample})
=\frac{1}{\binom{M}{N}}.
$$

The phrase *without replacement* means that a selected unit cannot be selected again. Consequently, the selected observations are dependent: learning which unit was selected changes which units remain available.

### 2.3 Model-based population

In mathematical statistics and machine learning, a population is usually represented by a probability distribution:

$$
X\sim p(x\mid\theta).
$$

A random sample of size $N$ is modeled as

$$
X_1,\ldots,X_N
\overset{\mathrm{iid}}{\sim}p(x\mid\theta).
$$

Here, randomness enters through the data-generating process. There need not be a known finite collection of $M$ units, so $M$ does not appear in this formulation.

| Finite-population sampling | Model-based sampling |
|---|---|
| $M$ fixed units exist | The population is represented by $p(x\mid\theta)$ |
| Select $N$ of those units | Observe or generate $N$ data points |
| Often sample without replacement | Usually assume IID observations |
| Infer properties of the finite collection | Infer properties of the distribution or process |

The model-based IID formulation aligns with the broader problem of machine learning and statistical inference. We will return to finite sampling in this lecture when needed (e.g. when introducing the finite-population correction) to ensure that you build some intitution about the limitations of certain assumptions we make when dealing with small samples and finite sampling. 

## 3. Statistics and sampling distributions

Let

$$
T=T(X_1,\ldots,X_N)
$$

be a statistic. A statistic is any function of the observed data that does not depend on unknown parameters. Before observing the data, $T$ is a random variable.

The **sampling distribution** of $T$ is the probability distribution induced by repeatedly applying $T$ to samples generated by the same sampling process.

For the sample mean,

$$
\overline{X}=\frac{1}{N}\sum_{i=1}^{N}X_i.
$$

Hypothetical repeated sampling produces

$$
\overline{X}^{(1)},\overline{X}^{(2)},\ldots .
$$

The distribution of these possible values is the sampling distribution of $\overline{X}$.

### Three distributions to distinguish

| Distribution | What is distributed? | Example |
|---|---|---|
| Population distribution | Possible individual observations $X$ | Engine temperatures generated by $p(x\mid\theta)$ |
| Empirical sample distribution | The observed values $x_1,\ldots,x_N$ | A histogram of the measured temperatures |
| Sampling distribution | Values of a statistic across repeated samples | A histogram of repeated sample means $\overline{X}^{(r)}$ |

:::{warning}
The sampling distribution of $\overline{X}$ is not the same as the distribution of the individual observations $X_i$. It describes the behavior of an estimator over hypothetical repetitions of the sampling process.
:::

## 4. Expected value of the sample mean

Assume that

$$
\mathbb{E}[X_i]=\mu.
$$

Using linearity of expectation,

$$
\begin{aligned}
\mathbb{E}[\overline{X}]
&=\mathbb{E}\left[\frac{1}{N}\sum_{i=1}^{N}X_i\right]\\
&=\frac{1}{N}\sum_{i=1}^{N}\mathbb{E}[X_i]\\
&=\frac{1}{N}\sum_{i=1}^{N}\mu\\
&=\mu.
\end{aligned}
$$

Therefore,

$$
\mathbb{E}[\overline{X}]=\mu,
$$

which means that $\overline{X}$ is an **unbiased estimator** of $\mu$.

:::{note}
Unbiasedness does not mean that the estimate from one sample equals the population parameter. It means that the estimator is correct *on average over repeated samples*.
:::

## 5. Variance and standard error of the sample mean

Assume the observations are independent and

$$
\operatorname{Var}(X_i)=\sigma^2.
$$

Then

$$
\begin{aligned}
\operatorname{Var}(\overline{X})
&=\operatorname{Var}\left(\frac{1}{N}\sum_{i=1}^{N}X_i\right)\\
&=\frac{1}{N^2}\sum_{i=1}^{N}\operatorname{Var}(X_i)\\
&=\frac{1}{N^2}N\sigma^2\\
&=\frac{\sigma^2}{N}.
\end{aligned}
$$

Thus,

$$
\operatorname{Var}(\overline{X})=\frac{\sigma^2}{N}
$$

and

$$
\operatorname{SD}(\overline{X})=\frac{\sigma}{\sqrt{N}}.
$$

The standard deviation of an estimator's sampling distribution is called its **standard error**. Therefore, when $\sigma$ is known,

$$
\operatorname{SE}(\overline{X})=\frac{\sigma}{\sqrt{N}}.
$$

### Sample-size implication

Uncertainty decreases at the rate $1/\sqrt{N}$, not $1/N$. Therefore, halving the standard error requires four times as many independent observations.

This is an important practical result when designing experiments or deciding how much data to collect.

## 6. Sample variance and the connection to Lecture 7

Lecture 7 derived the Gaussian maximum-likelihood estimator of the variance:

$$
\widehat{\sigma}_{\mathrm{ML}}^2
=\frac{1}{N}\sum_{i=1}^{N}(X_i-\overline{X})^2.
$$

In classical sampling theory, the usual unbiased estimator of $\sigma^2$ is

$$
S^2
=\frac{1}{N-1}\sum_{i=1}^{N}(X_i-\overline{X})^2.
$$

These formulas are not contradictory. They are motivated by different statistical properties.

| Estimator | Denominator | Property |
|---|---:|---|
| Gaussian MLE | $N$ | Maximizes the Gaussian likelihood |
| Unbiased sample variance | $N-1$ | Has expectation $\sigma^2$ |

For IID Gaussian observations, the MLE is biased downward at finite $N$:

$$
\mathbb{E}\left[\widehat{\sigma}_{\mathrm{ML}}^2\right]
=\frac{N-1}{N}\sigma^2.
$$

By contrast,

$$
\mathbb{E}[S^2]=\sigma^2.
$$

### Why does the unbiased estimator use $N-1$?

The deviations from the sample mean satisfy

$$
\sum_{i=1}^{N}(X_i-\overline{X})=0.
$$

Once $N-1$ deviations are known, the final deviation is determined by this constraint. Only $N-1$ deviations are free to vary; we say that the residuals have $N-1$ **degrees of freedom**.

## 7. Estimated standard error

Because the population standard deviation $\sigma$ is usually unknown, we estimate it using $S$. The estimated standard error of the sample mean is

$$
\widehat{\operatorname{SE}}(\overline{X})
=\frac{S}{\sqrt{N}}.
$$

It is important to distinguish

$$
S
\quad \text{from} \quad
\frac{S}{\sqrt{N}}.
$$

- $S$ estimates the variability of individual observations.
- $S/\sqrt{N}$ estimates the variability of the sample mean across repeated samples.

The latter should not be called the standard deviation of the data. It is the estimated standard deviation of the estimator's sampling distribution.

## 8. Finite-population correction

The IID formula $\sigma/\sqrt{N}$ applies naturally to independent sampling, including sampling with replacement. When sampling without replacement from a finite population, the observations are dependent and uncertainty decreases slightly faster. 

We will derive this next. 

Let us define the finite-population variance using denominator $M$:

$$
\sigma_M^2
=\frac{1}{M}\sum_{j=1}^{M}(z_j-\mu_M)^2.
$$

For a simple random sample of size $N$ drawn without replacement from $M$ units,

$$
\operatorname{Var}(\overline{X})
=\frac{\sigma_M^2}{N}\frac{M-N}{M-1},
$$

and hence

$$
\operatorname{SE}(\overline{X})
=\frac{\sigma_M}{\sqrt{N}}
\sqrt{\frac{M-N}{M-1}}.
$$

The term

$$
\sqrt{\frac{M-N}{M-1}}
$$

is the **finite-population correction** under this definition of $\sigma_M^2$.

The correction has an intuitive interpretation:

- if $N=1$, the correction equals $1$;
- if $N=M$, the correction equals $0$ because observing the entire population eliminates sampling uncertainty;
- if $N/M$ is very small, the correction is close to $1$ and the IID standard-error formula is a good approximation.

### Example

If $N=50$ engines are selected without replacement from a fleet of $M=1{,}000$, then

$$
\sqrt{\frac{M-N}{M-1}}
=\sqrt{\frac{950}{999}}
\approx 0.975.
$$

The finite-population standard error is therefore about $2.5\%$ smaller than the independent-sampling approximation.

## 9. Normal approximation and the central limit theorem

Suppose

$$
X_1,\ldots,X_N\overset{\mathrm{iid}}{\sim}p(x),
$$

with

$$
\mathbb{E}[X_i]=\mu,
\qquad
\operatorname{Var}(X_i)=\sigma^2<\infty.
$$

The central limit theorem states that

$$
\frac{\overline{X}-\mu}{\sigma/\sqrt{N}}
\xrightarrow{d}\mathcal{N}(0,1)
\qquad \text{as } N\to\infty.
$$

Consequently, for sufficiently large $N$,

$$
\overline{X}
\overset{\cdot}{\sim}
\mathcal{N}\left(\mu,\frac{\sigma^2}{N}\right),
$$

where $\overset{\cdot}{\sim}$ denotes an approximate distribution.

### 9.1 Exact Gaussian case

If

$$
X_i\sim\mathcal{N}(\mu,\sigma^2),
$$

then

$$
\overline{X}\sim
\mathcal{N}\left(\mu,\frac{\sigma^2}{N}\right)
$$

exactly for every sample size $N$.

### 9.2 Non-Gaussian case

If the population distribution is not Gaussian, normality of the sample mean is generally an approximation. How large $N$ must be depends on features such as skewness, heavy tails, outliers, and dependence.

### 9.3 What the CLT does not guarantee

The CLT does not imply that:

- individual observations are Gaussian;
- every sample of size $30$ is sufficiently large;
- dependence can be ignored;
- biased sampling becomes valid as $N$ grows; or
- strong time-series correlation disappears merely because many observations were recorded.

## 10. Confidence interval when the variance is known

If $\sigma$ is known and either the population is Gaussian or the normal approximation is adequate, then

$$
Z
=\frac{\overline{X}-\mu}{\sigma/\sqrt{N}}
\approx\mathcal{N}(0,1).
$$

Let $z_{1-\alpha/2}$ denote the $(1-\alpha/2)$ quantile of the standard normal distribution. Then

$$
P\left(
-z_{1-\alpha/2}
\leq
\frac{\overline{X}-\mu}{\sigma/\sqrt{N}}
\leq
z_{1-\alpha/2}
\right)
\approx 1-\alpha.
$$

Rearranging gives the random interval

$$
\overline{X}
\pm
z_{1-\alpha/2}\frac{\sigma}{\sqrt{N}}.
$$

After observing the data, the corresponding interval is

$$
\overline{x}
\pm
z_{1-\alpha/2}\frac{\sigma}{\sqrt{N}}.
$$

For a $95\%$ confidence interval,

$$
z_{0.975}\approx 1.96.
$$

## 11. Confidence interval when the variance is unknown

When $\sigma$ is unknown, replace it with the sample standard deviation $S$ and define

$$
T
=\frac{\overline{X}-\mu}{S/\sqrt{N}}.
$$

For IID Gaussian observations,

$$
T\sim t_{N-1}.
$$

The exact confidence interval is then

$$
\overline{x}
\pm
t_{N-1,\,1-\alpha/2}\frac{s}{\sqrt{N}},
$$

where $t_{N-1,\,1-\alpha/2}$ is the appropriate quantile of the Student $t$ distribution with $N-1$ degrees of freedom.

Three ideas are important:

1. replacing $\sigma$ with $S$ introduces additional uncertainty;
2. the $t$ distribution has heavier tails than the standard normal distribution; and
3. the $t$ distribution approaches the standard normal distribution as $N$ increases.

## 12. Correct interpretation of a confidence interval

Before observing the sample,

$$
[L(X),U(X)]
$$

is a random interval. A $100(1-\alpha)\%$ confidence procedure satisfies

$$
P_{\theta}\bigl(L(X)\leq\theta\leq U(X)\bigr)
=1-\alpha,
$$

or approximately $1-\alpha$ when the procedure relies on an approximation.

After observing the sample,

$$
[L(x),U(x)]
$$

is fixed.

:::{important}
A correct frequentist interpretation is:

> If the sampling and interval-construction procedure were repeated many times, approximately $100(1-\alpha)\%$ of the resulting intervals would contain the true parameter, provided the assumptions hold.
:::

It is not technically correct in the frequentist framework to say that the fixed parameter has a $95\%$ probability of lying in the particular interval already observed.

## 13. Assumptions and limitations

The formulas in this lecture assume, explicitly or implicitly:

- representative random sampling;
- independence or suitably weak dependence;
- a stable data-generating distribution;
- finite variance for the standard central limit theorem;
- an adequate normal approximation; and
- correct identification of the sampling unit.

### Physical-system example: dependence in sensor data

Suppose a system records one engine-temperature value every second for $10{,}000$ seconds. This does not necessarily provide $10{,}000$ independent observations. Adjacent temperatures may be strongly autocorrelated.

If positive correlation is ignored,

$$
\widehat{\operatorname{SE}}(\overline{X})
=\frac{S}{\sqrt{N}}
$$

may substantially underestimate uncertainty. More data points do not automatically provide more independent information.

This is particularly important for time-series, robotics, autonomous-system, and industrial sensor data.

## 14. Python tutorial: making repeated sampling visible

### Experiment 1: Sampling distribution of the mean

For a known population distribution:

1. draw $R$ independent samples of size $N$;
2. compute the mean of each sample;
3. plot the resulting sample means;
4. compare their empirical mean with $\mu$; and
5. compare their empirical standard deviation with $\sigma/\sqrt{N}$.

### Experiment 2: Effect of sample size

Use

$$
N\in\{5,30,100,500\}.
$$

Verify empirically that standard error decreases approximately as

$$
\frac{1}{\sqrt{N}}.
$$

### Experiment 3: Central limit theorem

Repeat the simulation using a skewed exponential population. Show that the population distribution remains skewed while the sampling distribution of $\overline{X}$ becomes increasingly Gaussian as $N$ grows.

### Experiment 4: Confidence-interval coverage

Construct a $95\%$ confidence interval for each simulated sample and compute the proportion of intervals that contain $\mu$. The empirical coverage should be close to

$$
0.95.
$$

### Experiment 5: Finite sampling without replacement

Create a fixed finite population of size $M$. Repeatedly draw simple random samples of size $N$ without replacement. Compare:

1. the empirical standard deviation of the sample means;
2. the IID approximation $\sigma_M/\sqrt{N}$; and
3. the finite-population result

$$
\frac{\sigma_M}{\sqrt{N}}
\sqrt{\frac{M-N}{M-1}}.
$$

Repeat for several sampling fractions $N/M$.

### Experiment 6: Dependence

Generate an autocorrelated sequence and incorrectly treat its observations as IID. Compare nominal confidence-interval coverage with empirical coverage. This experiment illustrates why independence assumptions matter for industrial and physical-system data.

## References and supplementary resources

- John A. Rice, *Mathematical Statistics and Data Analysis*, especially Sections 5.3 and 7.2–7.3.
- Steven L. Brunton, [Population Statistics and Random Sampling](https://www.youtube.com/watch?v=OlkL1YatyHI).
- Steven L. Brunton, [Expected Value and Variance of the Sample Mean](https://www.youtube.com/watch?v=Gg3d-rn9eEU).
- Steven L. Brunton, [Random Sampling Without Replacement](https://www.youtube.com/watch?v=IDvp3pMm16k).
- Steven L. Brunton, [Sample Variance in Random Population Sampling](https://www.youtube.com/watch?v=yNnUVHfX5yQ).
- Steven L. Brunton, [Normal Approximation to the Sample Mean](https://www.youtube.com/watch?v=Arbj9SoU9Cs).
- Steven L. Brunton, [Confidence Intervals](https://www.youtube.com/watch?v=qTVdV8ITZfk).

## 15. Bridge to Lecture 10: hypothesis testing

Suppose someone proposes that

$$
\mu=\mu_0.
$$

Under this proposal and an appropriate normal approximation,

$$
\overline{X}
\approx
\mathcal{N}\left(\mu_0,\frac{\sigma^2}{N}\right).
$$

We can therefore ask:

> Is the observed sample mean reasonably compatible with this proposed sampling distribution, or is it unusually far into its tails?

We will discuss this in the next lecture. 

