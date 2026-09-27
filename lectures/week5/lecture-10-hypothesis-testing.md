# Lecture 10 - Hypothesis Testing
Lecture 9 developed the frequentist account of **sampling uncertainty**. We distinguished populations from samples, treated the sample mean as a random estimator, derived its standard error, introduced the central limit theorem, and constructed confidence intervals. This lecture continues directly from that foundation.

In Lecture 7, we treated model parameters as **fixed but unknown** and learned them from observed data using likelihood and maximum-likelihood estimation. 

In Lecture 9, we then asked how a statistic such as the sample mean varies across hypothetical repeated samples. We now use that sampling distribution to evaluate claims about the unknown population parameter.

Suppose someone proposes that

$$
\mu=\mu_0.
$$

If this proposal is correct, then the sampling distribution of the sample mean is centered at $\mu_0$. We can therefore ask:

> Is the observed sample mean reasonably compatible with the sampling distribution implied by $\mu=\mu_0$, or is it unusually far into its tails?

Thus, the guiding question for this lecture is:

> How can we use sampling distributions to evaluate a statistical hypothesis, quantify the strength of evidence against it, and understand the errors associated with a testing procedure?

The prior lectures 7,9, help you to perform the the hypotheses testing in focus for this lecture. 

:::{admonition} How Lectures 7, 9, and 10 connect
:class: important

The three lectures use the same statistical model but ask different questions.

| Lecture | Main mathematical object | Principal question |
|---|---|---|
| **Lecture 7: Parameter estimation** | Likelihood \(L(\theta;\mathbf{x})\) and estimator \(\widehat{\theta}_{\mathrm{ML}}\) | Given one observed sample, which parameter value best explains the data? |
| **Lecture 9: Sampling uncertainty** | Sampling distribution of \(\widehat{\theta}\) | How would the estimator vary if the complete sampling procedure were hypothetically repeated? |
| **Lecture 10: Hypothesis testing** | Null distribution of \(T(\mathbf{X})\) under \(H_0\) | Is the observed statistic compatible with a proposed parameter value \(\theta_0\)? |

The progression is

$$
\boxed{
\text{estimate the parameter}
\;\longrightarrow\;
\text{quantify the estimator's uncertainty}
\;\longrightarrow\;
\text{evaluate a claim about the parameter}
}
$$

For the population mean, this becomes

$$
\widehat{\mu}_{\mathrm{ML}}=\overline{X}
\quad\longrightarrow\quad
\operatorname{SE}(\overline{X})
=\frac{\sigma}{\sqrt{N}}
\quad\longrightarrow\quad
Z=\frac{\overline{X}-\mu_0}{\sigma/\sqrt{N}}.
$$

In short, Lecture 7 derives the estimator, Lecture 9 characterizes its repeated-sampling uncertainty, and Lecture 10 uses that uncertainty to test a claim about the population parameter.

:::


# Learning objectives
After completing the lecture and tutorial, students should be able to:
1. Formulate appropriate null and alternative hypotheses for a population parameter and distinguish between one-sided and two-sided alternatives.
2. Explain how the sampling distribution under the null hypothesis provides the reference distribution for a statistical test.
3. Select and compute a test statistic for a simple one-sample mean problem.
4. Define and correctly interpret a p-value as a probability calculated under the null model, and compare it with a prespecified significance level $\alpha$.
5. Identify Type I and Type II errors in an applied scenario.
6. Explain the relationships among \(\alpha\), \(\beta\), power, effect size, sample size, and variability.
7. Relate a two-sided hypothesis test to the corresponding confidence interval.
8. Use simulation to investigate p-values, Type I error, Type II error, and power under repeated sampling (notebook) 
9. Recognize limitations involving multiple testing, assumption violations, and data-dependent hypotheses.

## 1. From parameter estimation to statistical inference

It is useful to place this lecture in the sequence developed so far.

In Lecture 7, we started with a parametric model

$$
p(x\mid\theta)
$$

and used observed data to estimate a fixed but unknown parameter $\theta$.

For a Gaussian model with unknown mean, the maximum-likelihood estimator is the sample mean:

$$
\widehat{\mu}_{\mathrm{ML}}
=
\overline{X}.
$$

Before the data are observed, $\overline{X}$ is a random variable. After observing the data, we obtain the numerical estimate

$$
\overline{x}.
$$

Lecture 9 then studied the **sampling distribution** of $\overline{X}$. Under IID sampling,

$$
\mathbb{E}[\overline{X}]
=
\mu
$$

and

$$
\operatorname{SE}(\overline{X})
=
\frac{\sigma}{\sqrt{N}}
$$

when $\sigma$ is known.

For sufficiently large $N$, the central limit theorem gives

$$
\overline{X}
\overset{\cdot}{\sim}
\mathcal{N}
\left(
\mu,
\frac{\sigma^2}{N}
\right).
$$

This sampling distribution is the mathematical bridge from **point estimation** to **uncertainty quantification** and **hypothesis testing**.

A useful conceptual sequence is therefore

$$
\text{data}
\longrightarrow
\text{estimate}
\longrightarrow
\text{sampling distribution}
\longrightarrow
\text{uncertainty}
\longrightarrow
\text{statistical decision}.
$$

---

## 2. Review: confidence intervals

Lecture 9 used the sampling distribution of $\overline{X}$ to construct confidence intervals.

When $\sigma$ is known,

$$
Z
=
\frac{\overline{X}-\mu}
{\sigma/\sqrt{N}}
\approx
\mathcal{N}(0,1).
$$

Therefore,

$$
P
\left(
-z_{1-\alpha/2}
\le
\frac{\overline{X}-\mu}
{\sigma/\sqrt{N}}
\le
z_{1-\alpha/2}
\right)
\approx
1-\alpha.
$$

Rearranging gives

$$
\overline{X}
-
z_{1-\alpha/2}
\frac{\sigma}{\sqrt{N}}
\le
\mu
\le
\overline{X}
+
z_{1-\alpha/2}
\frac{\sigma}{\sqrt{N}}.
$$

After observing the data,

$$
\boxed{
\overline{x}
\pm
z_{1-\alpha/2}
\frac{\sigma}{\sqrt{N}}
}
$$

is the corresponding confidence interval.

For a 95% confidence interval,

$$
z_{0.975}\approx1.96.
$$

The confidence interval asks:

> Which values of $\mu$ are reasonably compatible with the observed estimate and its sampling uncertainty?

Hypothesis testing asks a closely related question:

> Is one particular proposed value $\mu_0$ reasonably compatible with the observed data?

---

## 3. Null and alternative hypotheses

A **statistical hypothesis** is a claim about a population parameter or probability model.

The **null hypothesis**, denoted by $H_0$, defines the reference claim against which the observed data are evaluated.

For a population mean,

$$
H_0:\mu=\mu_0.
$$

The **alternative hypothesis**, denoted by $H_1$ or $H_A$, describes the competing claim.

### 3.1 Two-sided alternative

If departures in either direction are scientifically relevant,

$$
H_1:\mu\neq\mu_0.
$$

This is a **two-sided test**.

### 3.2 One-sided alternatives

If only values larger than $\mu_0$ are relevant,

$$
H_1:\mu>\mu_0.
$$

If only values smaller than $\mu_0$ are relevant,

$$
H_1:\mu<\mu_0.
$$

The direction of the alternative hypothesis should be determined by the scientific question **before examining the observed result**.

:::{important}
The hypotheses concern an unknown population parameter or model. They are not claims that the observed sample mean must equal a particular value.
:::

---