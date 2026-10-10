# Lecture 12 - Exponential Families

Lectures 6 and 7 treated the Bernoulli, Poisson, and Gaussian distributions separately. They describe different kinds of observations: binary outcomes, counts, and continuous measurements. Yet each can be written in the same mathematical form. Recognizing that form gives us a common way to find moments, identify what a sample tells us about unknown parameters, and derive maximum-likelihood estimates. It is very important for Baysian inference as well, as we will see in in Lecture 13. 

This lecture answers the following question: **What common structure makes these probability models useful for statistical inference?**

We will use the notation from earlier lectures: $X$ is a random variable, $x$ is one realized value, and $\mathcal D=(x_1,\ldots,x_N)$ is an observed IID sample. Parameters are fixed but unknown in the frequentist calculations below. Natural parameters describe the same distributions using a different coordinate system - they are not additional random variables.



# Learning objectives: 

After completing this lecture, you should be able to: 
1. Explain sufficiency: describe what a sufficient statistic retains about an unknown parameter under an assumed probability model.
2. Recognize exponential-family structure: express Bernoulli, Poisson, and Gaussian models in exponential-family form and identify their components.
3. Calculate moments: use derivatives of the log normalizer to obtain expectations and variances of sufficient-statistic functions.
4. Derive maximum-likelihood estimates: explain and apply the moment-matching equations for exponential-family models.
5. Connect likelihood and KL divergence: explain how maximizing sample log-likelihood relates to minimizing a population-level KL divergence.


## A common form for probability models

An **exponential-family** model can be written as

$$
p(x\mid\boldsymbol\eta)
=h(x)\exp\!\left\{
\boldsymbol\eta^{\mathsf T}\mathbf T(x)-A(\boldsymbol\eta)
\right\}.
$$

Here $p$ can mean a probability mass function for discrete $X$ or a probability density for continuous $X$. The pieces have distinct jobs:

- $\boldsymbol\eta$ is the **natural parameter** vector. For a one-parameter model it is a scalar $\eta$.
- $\mathbf T(x)$ is the vector of **sufficient-statistic functions** for one observation. It does not depend on the parameter. For a scalar model, write $T(x)$. For an IID sample, the sum $\sum_n\mathbf T(x_n)$ is the sufficient sample statistic, as shown below.
- $h(x)$ is a nonnegative **base factor** that does not depend on the parameter. It may include constants or factors such as $1/x!$.
- $A(\boldsymbol\eta)$ is the **log normalizer**. It makes the PMF sum to one or the density integrate to one.

For this lecture, the set of possible values of $x$ is fixed as the parameter varies. The natural parameters must be chosen so that the normalizing sum or integral is finite. We do not need a more abstract description of the parameter space to use the examples below.

For a discrete variable, normalization means

$$
1=\sum_x h(x)
\exp\!\left\{\boldsymbol\eta^{\mathsf T}\mathbf T(x)
-A(\boldsymbol\eta)\right\}.
$$

Move $e^{-A(\boldsymbol\eta)}$ outside the sum and solve for $A$:

$$
A(\boldsymbol\eta)
=\log\sum_x h(x)
\exp\!\left\{\boldsymbol\eta^{\mathsf T}\mathbf T(x)\right\}.
$$

For a continuous variable, replace the sum with an integral over the fixed support of $X$. Thus $A$ is determined by the other ingredients and we do not choose it independently. The logarithm is a natural logarithm throughout this lecture unless another base is stated.

The notation $\boldsymbol\eta^{\mathsf T}\mathbf T(x)$ means a dot product. If there are two components, it is $\eta_1T_1(x)+\eta_2T_2(x)$. The same ordinary parameter can look quite different after conversion to natural parameters, but it still specifies the same probability distribution. The following sections give examples of distributions that belong in the exponential family.

## Bernoulli outcomes

Write $\theta$ for the Bernoulli probability of success throughout this lecture. For $x\in\{0,1\}$ and $0<\theta<1$, start with its usual PMF. Writing it as the exponential of its logarithm lets us collect the terms involving $x$:

$$
\begin{aligned}
p(x\mid\theta)
&=\theta^x(1-\theta)^{1-x}\\
&=\exp\!\left\{\log\!\left[\theta^x(1-\theta)^{1-x}\right]\right\}\\
&=\exp\!\left\{x\log\theta+(1-x)\log(1-\theta)\right\}\\
&=\exp\!\left\{x\log\frac{\theta}{1-\theta}+\log(1-\theta)\right\}\\
&=1\cdot\exp\!\left\{
\underbrace{\log\frac{\theta}{1-\theta}}_{\eta}
\underbrace{x}_{T(x)}
-\underbrace{\log\frac{1}{1-\theta}}_{A(\eta)}
\right\}.
\end{aligned}
$$

Compare the last line with $h(x)\exp\{\eta T(x)-A(\eta)\}$. It identifies each piece directly:

$$
\eta=\log\frac{\theta}{1-\theta},
\qquad T(x)=x,
\qquad h(x)=1,
\qquad A(\eta)=-\log(1-\theta).
$$

To write the log normalizer only in terms of the natural parameter, invert the relationship between $\eta$ and $\theta$:

$$
e^\eta=\frac{\theta}{1-\theta}
\quad\Longrightarrow\quad
\theta=\frac{e^\eta}{1+e^\eta}.
$$

Therefore $1-\theta=1/(1+e^\eta)$ and

$$
A(\eta)=-\log(1-\theta)=\log(1+e^\eta).
$$

The natural parameter $\eta$ can be any finite real number. The allowed endpoint values $\theta=0$ and $\theta=1$ arise as $\eta$ tends to $-\infty$ and $+\infty$, respectively - neither has a finite $\eta$.

## Poisson counts

As in the earlier distribution lecture, let $\lambda>0$ be the Poisson rate or expected count. For $x=0,1,2,\ldots$, take the exponential of the log of its PMF, then move the part that does not involve $\lambda$ outside the exponential:

$$
\begin{aligned}
p(x\mid\lambda)
&=\frac{\lambda^x e^{-\lambda}}{x!}\\
&=\exp\!\left\{\log\!\left[\frac{\lambda^x e^{-\lambda}}{x!}\right]\right\}\\
&=\exp\!\left\{x\log\lambda-\lambda-\log(x!)\right\}\\
&=\frac1{x!}\exp\!\left\{
\underbrace{\log\lambda}_{\eta}
\underbrace{x}_{T(x)}
-\underbrace{\lambda}_{A(\eta)}
\right\}.
\end{aligned}
$$

Comparing the final line with $h(x)\exp\{\eta T(x)-A(\eta)\}$ gives

$$
\eta=\log\lambda,
\qquad T(x)=x,
\qquad h(x)=\frac1{x!},
\qquad A(\eta)=e^\eta=\lambda.
$$

The factorial is part of the base factor because it depends on the observed count but not on $\lambda$. Again, $\eta$ can be any finite real number. The rate $\lambda=0$ appears only as the limit $\eta\to-\infty$.

## Gaussian measurements

The one-dimensional Gaussian model from Lecture 7 has mean $\mu$ and variance $\sigma^2>0$. When both are unknown, its log density has terms involving both $x$ and $x^2$. Start with the usual density, take the exponential of its logarithm, and expand the square:

$$
\begin{aligned}
p(x\mid\mu,\sigma^2)
&=\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\!\left\{-\frac{(x-\mu)^2}{2\sigma^2}\right\}\\
&=\exp\!\left\{\log\!\left[
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\!\left\{-\frac{(x-\mu)^2}{2\sigma^2}\right\}
\right]\right\}\\
&=\exp\!\left\{
-\frac12\log(2\pi)-\log\sigma
-\frac{x^2-2\mu x+\mu^2}{2\sigma^2}
\right\}\\
&=\underbrace{\frac1{\sqrt{2\pi}}}_{h(x)}
\exp\!\left\{
\underbrace{\frac{\mu}{\sigma^2}}_{\eta_1}
\underbrace{x}_{T_1(x)}
+\underbrace{\left(-\frac{1}{2\sigma^2}\right)}_{\eta_2}
\underbrace{x^2}_{T_2(x)}
-\underbrace{\left(\frac{\mu^2}{2\sigma^2}+\log\sigma\right)}_{A(\boldsymbol\eta)}
\right\}.
\end{aligned}
$$

The last line can be compared term by term with $h(x)\exp\{\boldsymbol\eta^{\mathsf T}\mathbf T(x)-A(\boldsymbol\eta)\}$. In vector notation, its components are

$$
\boldsymbol\eta=
\begin{bmatrix}
\eta_1\\[2pt]\eta_2
\end{bmatrix}
=\begin{bmatrix}
\mu/\sigma^2\\[2pt]-1/(2\sigma^2)
\end{bmatrix},
\qquad
\mathbf T(x)=
\begin{bmatrix}x\\[2pt]x^2\end{bmatrix},
\qquad
h(x)=\frac{1}{\sqrt{2\pi}},
$$

with

$$
A(\boldsymbol\eta)
=\frac{\mu^2}{2\sigma^2}+\log\sigma.
$$

Because $\sigma^2>0$, we require $\eta_2<0$. We can also express $A$ entirely in natural parameters. Since

$$
\sigma^2=-\frac{1}{2\eta_2},
\qquad
\mu=-\frac{\eta_1}{2\eta_2},
$$

substitution gives

$$
A(\eta_1,\eta_2)
=-\frac{\eta_1^2}{4\eta_2}
-\frac12\log(-2\eta_2).
$$

The $x^2$ term must remain in $\mathbf T(x)$ when the variance is unknown: its coefficient changes with $\sigma^2$. If $\sigma^2$ is **known** and only $\mu$ is unknown, that same $x^2$ term has a fixed coefficient. Starting again from the Gaussian density and regrouping its terms gives

$$
\begin{aligned}
p(x\mid\mu,\sigma^2)
&=\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\!\left\{-\frac{(x-\mu)^2}{2\sigma^2}\right\}\\
&=\exp\!\left\{\log p(x\mid\mu,\sigma^2)\right\}\\
&=\exp\!\left\{
-\frac12\log(2\pi\sigma^2)
-\frac{x^2-2\mu x+\mu^2}{2\sigma^2}
\right\}\\
&=\underbrace{\left[
\frac1{\sqrt{2\pi\sigma^2}}
\exp\!\left\{-\frac{x^2}{2\sigma^2}\right\}
\right]}_{h(x)}
\exp\!\left\{
\underbrace{\frac{\mu}{\sigma^2}}_{\eta}
\underbrace{x}_{T(x)}
-\underbrace{\frac{\mu^2}{2\sigma^2}}_{A(\eta)}
\right\}.
\end{aligned}
$$

The corresponding components are $\eta=\mu/\sigma^2$, $T(x)=x$,

$$
h(x)=\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\!\left\{-\frac{x^2}{2\sigma^2}\right\},
\qquad
A(\eta)=\frac{\mu^2}{2\sigma^2}
=\frac{\sigma^2\eta^2}{2}.
$$

Here $h$ depends on the **known, fixed** variance but not on the unknown mean. These are two descriptions of a Gaussian model with different choices about what is being estimated.

| Model and unknown parameters | Natural parameter | Sufficient-statistic function | Base factor $h(x)$ | Log normalizer $A$ |
|---|---|---|---|---|
| Bernoulli, $\theta$ | $\eta=\log\frac{\theta}{1-\theta}$ | $T(x)=x$ | $1$ | $\log(1+e^\eta)$ |
| Poisson, $\lambda$ | $\eta=\log\lambda$ | $T(x)=x$ | $1/x!$ | $e^\eta$ |
| Gaussian, $\mu$ and $\sigma^2$ | $\eta_1=\mu/\sigma^2$, $\eta_2=-1/(2\sigma^2)$ | $\mathbf T(x)=(x,x^2)^{\mathsf T}$ | $1/\sqrt{2\pi}$ | $-\eta_1^2/(4\eta_2)-\tfrac12\log(-2\eta_2)$ |

These rows are not separate tricks. In each case we took the logarithm of the familiar PMF or density, grouped terms involving $x$, and identified the remaining normalizing term.

## Means and variances from the log normalizer

Usually, computing an expectation requires summing or integrating $T(x)p(x\mid\eta)$. The exponential-family form lets us obtain it by differentiating $A$, making computations significantly faster. To see why, begin with a **scalar** natural parameter and define

$$
Z(\eta)=\sum_x h(x)e^{\eta T(x)},
\qquad A(\eta)=\log Z(\eta).
$$

For a continuous variable, replace the sum by an integral. In the examples here, for finite natural parameters inside the allowed range, we can differentiate the normalizing sum or integral with respect to $\eta$. First,

$$
Z'(\eta)=\sum_x T(x)h(x)e^{\eta T(x)}.
$$

Divide by $Z=e^A$:

$$
\begin{aligned}
A'(\eta)
&=\frac{Z'(\eta)}{Z(\eta)}\\
&=\sum_x T(x)h(x)e^{\eta T(x)-A(\eta)}\\
&=\sum_x T(x)p(x\mid\eta)\\
&=\mathbb E_{\eta}[T(X)].
\end{aligned}
$$

The subscript $\eta$ says that the expectation is taken under the distribution specified by $\eta$. Differentiate once more, using $Z''=\sum_x T(x)^2h(x)e^{\eta T(x)}$:

$$
\begin{aligned}
A''(\eta)
&=\frac{Z''(\eta)}{Z(\eta)}
-\left(\frac{Z'(\eta)}{Z(\eta)}\right)^2\\
&=\mathbb E_{\eta}[T(X)^2]
-\mathbb E_{\eta}[T(X)]^2\\
&=\operatorname{Var}_{\eta}(T(X)).
\end{aligned}
$$

Thus the **first derivative gives the mean of the statistic**, and the **second derivative gives its variance**. If $T(X)=X$, these are directly the mean and variance of $X$. If the statistic has several components, the gradient $\nabla_{\boldsymbol\eta}A$ gives their expectations, and the matrix of second derivatives gives their covariance matrix. We only need its first-derivative part for the two-parameter Gaussian calculation below.

### Bernoulli moments

For $A(\eta)=\log(1+e^\eta)$ and $T(X)=X$,

$$
\mathbb E[X]=A'(\eta)
=\frac{e^\eta}{1+e^\eta}=\theta.
$$

Differentiate the quotient once more:

$$
\operatorname{Var}(X)=A''(\eta)
=\frac{e^\eta}{(1+e^\eta)^2}
=\theta(1-\theta).
$$

The last equality uses $\theta=e^\eta/(1+e^\eta)$ and $1-\theta=1/(1+e^\eta)$. We recovered both familiar moments by differentiating one function.

### Poisson moments

Here $A(\eta)=e^\eta$ and $T(X)=X$. Both derivatives are $e^\eta$, so

$$
\mathbb E[X]=A'(\eta)=e^\eta=\lambda,
\qquad
\operatorname{Var}(X)=A''(\eta)=e^\eta=\lambda.
$$

This explains immediately why the Poisson mean equals its variance.

### Gaussian moments

For the **known-variance** Gaussian form, $A(\eta)=\sigma^2\eta^2/2$ and $T(X)=X$. Differentiation gives

$$
\mathbb E[X]=A'(\eta)=\sigma^2\eta=\mu,
\qquad
\operatorname{Var}(X)=A''(\eta)=\sigma^2.
$$

For the Gaussian with **both** $\mu$ and $\sigma^2$ unknown, $\mathbf T(X)=(X,X^2)^{\mathsf T}$. Differentiate its two-parameter normalizer:

$$
\begin{aligned}
\mathbb E[X]
&=\frac{\partial A}{\partial\eta_1}
=-\frac{\eta_1}{2\eta_2}=\mu,\\[4pt]
\mathbb E[X^2]
&=\frac{\partial A}{\partial\eta_2}
=\frac{\eta_1^2}{4\eta_2^2}
-\frac{1}{2\eta_2}
=\mu^2+\sigma^2.
\end{aligned}
$$

Finally use the variance identity from Lecture 5:

$$
\operatorname{Var}(X)
=\mathbb E[X^2]-\mathbb E[X]^2
=(\mu^2+\sigma^2)-\mu^2
=\sigma^2.
$$

The natural parameter's derivatives gave us the first and second raw moments; one subtraction recovered the variance. In a vector model, the derivatives refer to the chosen statistic $\mathbf T(X)$, so we must check whether a component is $X$, $X^2$, or something else before interpreting a derivative as a variance.

## What a sufficient statistic keeps

A statistic is a quantity computed from the data without knowing the model parameter. A statistic $S(\mathcal D)$ is **sufficient** for a parameter if, after $S$ is known, the remaining details of the sample provide no further information about that parameter **within the assumed model**. Sufficiency does not say the model is true, that the statistic answers every scientific question, or that no other summary of the data can be useful.

### A simple coin-flip example

Suppose we flip the same coin $N$ times to learn its unknown probability $\theta$ of landing heads. Record 1 for heads and 0 for tails. A sample of four flips might be heads, tails, heads, heads, or $(1,0,1,1)$.

The full sequence records *when* each head occurred. But under the assumption that the flips are independent and all have the same heads probability $\theta$, the likelihood of this particular sequence is $\theta^3(1-\theta)$. The sequence heads, heads, tails, heads has the same likelihood: both contain three heads and one tail. For estimating one shared $\theta$, all sequences with the same number of heads provide the same information about its value. The total number of heads, $S=\sum_{n=1}^{N}X_n$, is therefore a sufficient statistic for $\theta$ when $N$ is known. The order could still matter if we wanted to check whether the coin's behavior changed between flips. There is also a conditional way to verify the coin-flip claim. Suppose $S=s$. Every particular sequence with $s$ heads has probability $\theta^s(1-\theta)^{N-s}$. There are $\binom Ns$ such sequences, so

$$
P(\text{one particular sequence}\mid S=s,\theta)
=\frac{\theta^s(1-\theta)^{N-s}}
{\binom Ns\theta^s(1-\theta)^{N-s}}
=\frac{1}{\binom Ns}.
$$

Once the count $s$ is given, the conditional probability no longer depends on $\theta$. The order of heads gives no extra information about a **single shared** $\theta$ under this IID model.


In general, for IID exponential-family observations, multiply the one-observation expressions:

$$
\begin{aligned}
\mathcal L(\boldsymbol\eta;\mathcal D)
&=\prod_{n=1}^{N}p(x_n\mid\boldsymbol\eta)\\
&=\left[\prod_{n=1}^{N}h(x_n)\right]
\exp\!\left\{
\boldsymbol\eta^{\mathsf T}\sum_{n=1}^{N}\mathbf T(x_n)
-NA(\boldsymbol\eta)
\right\}.
\end{aligned}
$$

For fixed $N$, every part of this likelihood that changes with $\boldsymbol\eta$ uses the sample only through

$$
S(\mathcal D)=\sum_{n=1}^{N}\mathbf T(x_n).
$$

The factor $\prod_n h(x_n)$ may depend on all the observations, but it does **not** depend on $\boldsymbol\eta$. It cannot change which parameter values the data favor. This factorization establishes that $S$ is sufficient under the model. If sample size is not fixed in advance, retain $N$ as well.

The following table shows the sufficient sample summary for each model introduced above.

| IID model | Sufficient sample summary when $N$ is fixed | Why |
|---|---|---|
| Bernoulli, unknown $\theta$ | $\sum_n x_n$ | The likelihood depends on the number of successes. |
| Poisson, unknown $\lambda$ | $\sum_n x_n$ | The likelihood depends on the total event count. |
| Gaussian, unknown $\mu$, known $\sigma^2$ | $\sum_n x_n$ | Only the coefficient of $x$ varies with $\mu$. |
| Gaussian, unknown $\mu$ and $\sigma^2$ | $\left(\sum_n x_n,\sum_n x_n^2\right)$ | Both the $x$ and $x^2$ coefficients vary. |

For the last row, keeping only the sample mean would lose information about the variance. The pair of sums is equivalent to keeping the sample mean and its average squared deviation, given $N$.

## Maximum likelihood uses the same summary

Lecture 7 introduced maximum likelihood: choose the parameter under which the observed data have the largest likelihood. Taking the logarithm of the factorized expression gives

$$
\ell(\boldsymbol\eta;\mathcal D)
=\sum_{n=1}^{N}\log h(x_n)
+\boldsymbol\eta^{\mathsf T}\sum_{n=1}^{N}\mathbf T(x_n)
-NA(\boldsymbol\eta).
$$

The first term is constant with respect to $\boldsymbol\eta$. Differentiating the remaining terms gives the **score equation**

$$
\nabla_{\boldsymbol\eta}\ell
=\sum_{n=1}^{N}\mathbf T(x_n)
-N\nabla_{\boldsymbol\eta}A(\boldsymbol\eta).
$$

To look for the parameter that makes the observed data fit best, set this derivative to zero. This step works when the best fit occurs at a finite value of $\boldsymbol\eta$. Rearranging the score equation gives

$$
\nabla_{\boldsymbol\eta}A(\widehat{\boldsymbol\eta}_{\mathrm{ML}})
=\frac{1}{N}\sum_{n=1}^{N}\mathbf T(x_n).
$$

The previous section showed that $\nabla_{\boldsymbol\eta}A(\boldsymbol\eta)=\mathbb E_{\boldsymbol\eta}[\mathbf T(X)]$. The subscript on $\mathbb E_{\boldsymbol\eta}$ tells us **which distribution to average over**: use $p(x\mid\boldsymbol\eta)$, the model specified by that particular parameter value. For a discrete random variable, this means summing the value of the statistic at *every possible outcome* $x$, weighted by that outcome's model probability. At the fitted parameter, the equation above therefore expands to

$$
\begin{aligned}
\mathbb E_{\widehat{\boldsymbol\eta}_{\mathrm{ML}}}[\mathbf T(X)]
&=\sum_{x\in\mathcal X}\mathbf T(x)
   p(x\mid\widehat{\boldsymbol\eta}_{\mathrm{ML}})\\
&=\sum_{x\in\mathcal X}\mathbf T(x)h(x)
   \exp\!\left\{
      \widehat{\boldsymbol\eta}_{\mathrm{ML}}^{\mathsf T}\mathbf T(x)
      -A(\widehat{\boldsymbol\eta}_{\mathrm{ML}})
   \right\}\\
&=\boxed{\frac{1}{N}\sum_{n=1}^{N}\mathbf T(x_n)}.
\end{aligned}
$$

Here $\mathcal X$ is the set of possible outcomes, such as $\{0,1\}$ for a Bernoulli variable or $\{0,1,2,\ldots\}$ for a Poisson variable. The second line substitutes the exponential-family PMF from the beginning of this lecture. If $X$ is continuous, the same statement uses an integral over its possible values instead of a sum:

$$
\begin{aligned}
\mathbb E_{\widehat{\boldsymbol\eta}_{\mathrm{ML}}}[\mathbf T(X)]
&=\int_{\mathcal X}\mathbf T(x)
   p(x\mid\widehat{\boldsymbol\eta}_{\mathrm{ML}})\,dx\\
&=\int_{\mathcal X}\mathbf T(x)h(x)
   \exp\!\left\{
      \widehat{\boldsymbol\eta}_{\mathrm{ML}}^{\mathsf T}\mathbf T(x)
      -A(\widehat{\boldsymbol\eta}_{\mathrm{ML}})
   \right\}\,dx\\
&=\frac{1}{N}\sum_{n=1}^{N}\mathbf T(x_n).
\end{aligned}
$$

The left side is a **model expectation**: it considers outcomes that *could* occur under the fitted distribution, not just the outcomes we observed. The right side is an **observed sample average**: it uses only $x_1,\ldots,x_N$. Maximum likelihood sets these two quantities equal when a finite solution exists. If $\mathbf T$ has several components, as it does for the Gaussian model with unknown mean and variance, the equality must hold separately for each component.

For a Bernoulli example, $T(x)=x$ and the only possible outcomes are 0 and 1. Thus the model expectation is $0\,P(X=0\mid\widehat\theta)+1\,P(X=1\mid\widehat\theta)=\widehat\theta$, while the observed average is $(x_1+\cdots+x_N)/N$. Equating them gives $\widehat\theta=S/N$, where $S$ is the number of successes.

Sometimes no finite natural parameter can make the two sides equal. If every Bernoulli observation is 1, the observed average is 1, but $\mathbb E_\eta[X]=\theta<1$ for every finite $\eta$. The best-fitting Bernoulli probability is then $\widehat\theta_{\mathrm{ML}}=1$, reached as $\eta$ grows without bound. So the equation is a useful way to find an estimate when it has a finite solution; otherwise, check the edge of the original parameter's allowed values.

For the Bernoulli model, $T(x)=x$ and $\mathbb E_\eta[X]=\theta$. Moment matching yields

$$
\widehat\theta_{\mathrm{ML}}=\overline x.
$$

For the Poisson model, the same statistic has expectation $\lambda$, so

$$
\widehat\lambda_{\mathrm{ML}}=\overline x.
$$

For the Gaussian with both parameters unknown, match both components of $(X,X^2)$:

$$
\widehat\mu_{\mathrm{ML}}=\overline x,
\qquad
\widehat\mu_{\mathrm{ML}}^2+\widehat\sigma^2_{\mathrm{ML}}
=\frac1N\sum_{n=1}^{N}x_n^2.
$$

Rearranging the second equation recovers the Gaussian MLE from Lecture 7:

$$
\widehat\sigma^2_{\mathrm{ML}}
=\frac1N\sum_{n=1}^{N}x_n^2-(\overline x)^2
=\frac1N\sum_{n=1}^{N}(x_n-\overline x)^2.
$$

The denominator here is $N$ because this is the Gaussian **maximum-likelihood** estimate. If all observed Gaussian values are identical, the formula gives zero, but this model requires $\sigma^2>0$. In that case, the likelihood grows without bound as $\sigma^2$ approaches zero, so no allowed positive variance maximizes it.

## Connection to KL divergence

Lecture 8 introduced KL divergence as a way to compare probability distributions. Suppose two choices of parameters describe the same kind of observation: one gives a distribution $p_{\boldsymbol\eta_1}$ and the other gives $p_{\boldsymbol\eta_2}$. For a possible observation $x$, the ratio $p(x\mid\boldsymbol\eta_1)/p(x\mid\boldsymbol\eta_2)$ compares how much probability mass or density each model assigns to that *same* observation. KL divergence takes the logarithm of this ratio and averages it over observations from the first distribution:

$$
D_{\mathrm{KL}}\!\left(p_{\boldsymbol\eta_1}\parallel p_{\boldsymbol\eta_2}\right)
=\mathbb E_{\boldsymbol\eta_1}\!\left[
\log\frac{p(X\mid\boldsymbol\eta_1)}{p(X\mid\boldsymbol\eta_2)}
\right].
$$

We use natural logarithms here, so KL is measured in **nats**. To calculate this expectation, first simplify the expression *inside* it. Because both distributions belong to the **same** exponential family, they use the same $h(x)$ and $\mathbf T(x)$ and only the natural parameter and its log normalizer change. At a value $x$ in their common support,

$$
\begin{aligned}
\frac{p(x\mid\boldsymbol\eta_1)}{p(x\mid\boldsymbol\eta_2)}
&=\frac{h(x)\exp\!\left\{
\boldsymbol\eta_1^{\mathsf T}\mathbf T(x)-A(\boldsymbol\eta_1)
\right\}}
{h(x)\exp\!\left\{
\boldsymbol\eta_2^{\mathsf T}\mathbf T(x)-A(\boldsymbol\eta_2)
\right\}}\\
&=\exp\!\left\{
\boldsymbol\eta_1^{\mathsf T}\mathbf T(x)-A(\boldsymbol\eta_1)
-\boldsymbol\eta_2^{\mathsf T}\mathbf T(x)+A(\boldsymbol\eta_2)
\right\}.
\end{aligned}
$$

The two factors $h(x)$ cancel, and dividing exponentials subtracts their exponents. Now take the logarithm; it removes the remaining exponential. Collecting the two terms containing $\mathbf T(x)$ gives

$$
\log\frac{p(x\mid\boldsymbol\eta_1)}{p(x\mid\boldsymbol\eta_2)}
=(\boldsymbol\eta_1-\boldsymbol\eta_2)^{\mathsf T}\mathbf T(x)
-A(\boldsymbol\eta_1)+A(\boldsymbol\eta_2).
$$

We can now perform the average from the definition of KL. The natural parameters and the two $A$ terms are fixed numbers during this average: only $\mathbf T(X)$ varies. Provided the expectations are finite,

$$
\begin{aligned}
D_{\mathrm{KL}}\!\left(
p_{\boldsymbol\eta_1}\parallel p_{\boldsymbol\eta_2}
\right)
&=\mathbb E_{\boldsymbol\eta_1}\!\left[
(\boldsymbol\eta_1-\boldsymbol\eta_2)^{\mathsf T}\mathbf T(X)
-A(\boldsymbol\eta_1)+A(\boldsymbol\eta_2)
\right]\\
&=(\boldsymbol\eta_1-\boldsymbol\eta_2)^{\mathsf T}
\mathbb E_{\boldsymbol\eta_1}[\mathbf T(X)]
-A(\boldsymbol\eta_1)+A(\boldsymbol\eta_2)\\
&=(\boldsymbol\eta_1-\boldsymbol\eta_2)^{\mathsf T}
\left[\sum_{x\in\mathcal X}\mathbf T(x)
p(x\mid\boldsymbol\eta_1)\right]
-A(\boldsymbol\eta_1)+A(\boldsymbol\eta_2)\\
&=(\boldsymbol\eta_1-\boldsymbol\eta_2)^{\mathsf T}
\left[\sum_{x\in\mathcal X}\mathbf T(x)h(x)
\exp\!\left\{\boldsymbol\eta_1^{\mathsf T}\mathbf T(x)
-A(\boldsymbol\eta_1)\right\}\right]
-A(\boldsymbol\eta_1)+A(\boldsymbol\eta_2)\\
&=(\boldsymbol\eta_1-\boldsymbol\eta_2)^{\mathsf T}
\nabla A(\boldsymbol\eta_1)
-A(\boldsymbol\eta_1)+A(\boldsymbol\eta_2).
\end{aligned}
$$

The expanded sum applies when $X$ is discrete, with $\mathcal X$ denoting its possible outcomes. Each $\mathbf T(x)$ is weighted by $p(x\mid\boldsymbol\eta_1)$ because **the first distribution in the KL expression supplies the average**. We then substitute the exponential-family formula for that probability. Finally, the earlier identity $\mathbb E_{\boldsymbol\eta_1}[\mathbf T(X)]=\nabla A(\boldsymbol\eta_1)$ gives the compact last line. The terms $A(\boldsymbol\eta_1)$ and $A(\boldsymbol\eta_2)$ do not depend on $x$, so taking their expectation leaves them unchanged.

For a continuous $X$, replace the sum in square brackets with $\int_{\mathcal X}\mathbf T(x)p(x\mid\boldsymbol\eta_1)\,dx$, where $p$ is now a density. This integral also equals $\nabla A(\boldsymbol\eta_1)$. We can therefore compare two members of the family using their natural parameters and normalizers, without separately evaluating the full log-ratio average. The **order matters**: exchanging $\boldsymbol\eta_1$ and $\boldsymbol\eta_2$ changes which distribution supplies the expectation.

### From KL divergence to maximum likelihood

The calculation above compared **two members of one exponential family**. For estimation, the comparison is different: we have an unknown distribution $p_{\mathrm{data}}$ that actually generates observations, and we choose a candidate $q_{\boldsymbol\eta}$ from an exponential family. The data-generating distribution need not itself belong to that family. Which natural parameter makes the candidate as close as possible to the distribution producing the data?

KL divergence gives one answer. The first argument is $p_{\mathrm{data}}$ because it supplies the observations over which we average:

$$
D_{\mathrm{KL}}(p_{\mathrm{data}}\parallel q_{\boldsymbol\eta})
=\mathbb E_{p_{\mathrm{data}}}\!\left[
\log\frac{p_{\mathrm{data}}(X)}{q_{\boldsymbol\eta}(X)}
\right].
$$

Here $\mathbb E_{p_{\mathrm{data}}}$ means an average using $p_{\mathrm{data}}(x)$, not using the candidate $q_{\boldsymbol\eta}(x)$. For a discrete variable it is a sum over $x$; for a continuous variable it is an integral. Assume the distributions have compatible support and the expectations below are finite. Splitting the logarithm of the ratio gives

$$
D_{\mathrm{KL}}(p_{\mathrm{data}}\parallel q_{\boldsymbol\eta})
=\mathbb E_{p_{\mathrm{data}}}[\log p_{\mathrm{data}}(X)]
-\mathbb E_{p_{\mathrm{data}}}[\log q_{\boldsymbol\eta}(X)].
$$

Now imagine changing $\boldsymbol\eta$. The first expectation stays exactly the same: neither $p_{\mathrm{data}}$ nor the averaging rule depends on the candidate parameter. We may therefore omit that constant when finding the best parameter. The remaining minus sign changes a minimization into a maximization:

$$
\begin{aligned}
\underset{\boldsymbol\eta}{\operatorname{arg\,min}}\;
D_{\mathrm{KL}}(p_{\mathrm{data}}\parallel q_{\boldsymbol\eta})
&=\underset{\boldsymbol\eta}{\operatorname{arg\,min}}\;
\bigl\{-\mathbb E_{p_{\mathrm{data}}}[\log q_{\boldsymbol\eta}(X)]\bigr\}\\
&=\underset{\boldsymbol\eta}{\operatorname{arg\,max}}\;
\mathbb E_{p_{\mathrm{data}}}[\log q_{\boldsymbol\eta}(X)].
\end{aligned}
$$

Thus, **at the population level**, minimizing KL means maximizing the average log probability or log density that the candidate assigns to an observation drawn from the true distribution. This is not yet the likelihood of an observed sample: $p_{\mathrm{data}}$ is unknown, so we cannot evaluate that expectation directly.

### Replacing the unknown average with data

Suppose $\mathcal D=(x_1,\ldots,x_N)$ is an IID sample from $p_{\mathrm{data}}$. The observed average

$$
\frac1N\sum_{n=1}^{N}\log q_{\boldsymbol\eta}(x_n)
$$

is a sample-based estimate of $\mathbb E_{p_{\mathrm{data}}}[\log q_{\boldsymbol\eta}(X)]$. These two quantities are **not equal for a finite sample**; the first uses the observations we have, while the second averages over all possible observations. For a fixed candidate model, the sample average approaches the population average as more IID data are collected under the usual conditions.

The sample average leads directly to maximum likelihood. By IID factorization, the likelihood of the observed sample under the candidate model is $\mathcal L(\boldsymbol\eta;\mathcal D)=\prod_{n=1}^{N}q_{\boldsymbol\eta}(x_n)$. Therefore,

$$
\begin{aligned}
\underset{\boldsymbol\eta}{\operatorname{arg\,max}}\;
\frac1N\sum_{n=1}^{N}\log q_{\boldsymbol\eta}(x_n)
&=\underset{\boldsymbol\eta}{\operatorname{arg\,max}}\;
\frac1N\log\!\left[\prod_{n=1}^{N}q_{\boldsymbol\eta}(x_n)\right]\\
&=\underset{\boldsymbol\eta}{\operatorname{arg\,max}}\;
\mathcal L(\boldsymbol\eta;\mathcal D)
=\widehat{\boldsymbol\eta}_{\mathrm{ML}}.
\end{aligned}
$$

The first equality uses $\log\prod_n q(x_n)=\sum_n\log q(x_n)$. In the second, multiplying by the positive number $N$ and undoing the strictly increasing logarithm leave the maximizing parameter unchanged. For continuous data, $q_{\boldsymbol\eta}(x_n)$ is a *density* and the product is a likelihood, not the probability of observing those exact real numbers.

### What the exponential-family form adds

For our candidate family,

$$
q_{\boldsymbol\eta}(x)
=h(x)\exp\!\left\{
\boldsymbol\eta^{\mathsf T}\mathbf T(x)-A(\boldsymbol\eta)
\right\}.
$$

Taking its logarithm turns the three factors into a sum. Averaging over the observations then gives

$$
\begin{aligned}
\frac1N\log\mathcal L(\boldsymbol\eta;\mathcal D)
&=\frac1N\sum_{n=1}^{N}\log q_{\boldsymbol\eta}(x_n)\\
&=\underbrace{\frac1N\sum_{n=1}^{N}\log h(x_n)}_{\text{fixed once data are observed}}
+\boldsymbol\eta^{\mathsf T}
\underbrace{\left(\frac1N\sum_{n=1}^{N}\mathbf T(x_n)\right)}_{\overline{\mathbf T}}
-A(\boldsymbol\eta).
\end{aligned}
$$

The $h(x_n)$ terms are fixed with respect to $\boldsymbol\eta$. The observed sample affects the parameter-dependent part only through the **average sufficient statistic** $\overline{\mathbf T}$. Consequently,

$$
\widehat{\boldsymbol\eta}_{\mathrm{ML}}
=\underset{\boldsymbol\eta}{\operatorname{arg\,max}}\;
\left\{\boldsymbol\eta^{\mathsf T}\overline{\mathbf T}-A(\boldsymbol\eta)\right\}.
$$

Differentiating the expression in braces gives $\overline{\mathbf T}-\nabla_{\boldsymbol\eta}A(\boldsymbol\eta)$. If a finite maximizing parameter exists, setting this derivative to zero recovers the result from the previous section:

$$
\overline{\mathbf T}
=\nabla_{\boldsymbol\eta}A(\widehat{\boldsymbol\eta}_{\mathrm{ML}})
=\mathbb E_{\widehat{\boldsymbol\eta}_{\mathrm{ML}}}[\mathbf T(X)].
$$

For Bernoulli data, $T(x)=x$, so this says the fitted success probability equals the observed fraction of successes. For a Gaussian with both mean and variance unknown, $\mathbf T(x)=(x,x^2)^{\mathsf T}$, so both the observed average $x$ and the observed average $x^2$ enter the fit. The KL connection therefore leads back to the same moment-matching equations we derived directly from likelihood.

<!-- As noted above, a best fit can also occur only at a limiting parameter value, so a finite derivative-zero solution is not guaranteed. -->

<!-- ### An exact finite-sample statement for discrete data

The population argument explains *why* likelihood fitting is related to KL, but it did not say the finite-sample MLE exactly minimizes KL from the unknown $p_{\mathrm{data}}$. For discrete observations, we can make a different, exact statement using the **empirical distribution** $\widehat p$: $\widehat p(x)$ is the fraction of the $N$ observations equal to $x$. For example, three successes in four Bernoulli trials give $\widehat p(1)=3/4$ and $\widehat p(0)=1/4$.

Starting from the KL definition, expand the logarithm and group identical observations in the second sum. If $x$ occurs $N\widehat p(x)$ times, its $\log q_{\boldsymbol\eta}(x)$ term appears that many times in the sample log-likelihood:

$$
\begin{aligned}
D_{\mathrm{KL}}(\widehat p\parallel q_{\boldsymbol\eta})
&=\sum_x\widehat p(x)
\log\frac{\widehat p(x)}{q_{\boldsymbol\eta}(x)}\\
&=\sum_x\widehat p(x)\log\widehat p(x)
-\sum_x\widehat p(x)\log q_{\boldsymbol\eta}(x)\\
&=-H(\widehat p)
-\frac1N\log\mathcal L(\boldsymbol\eta;\mathcal D).
\end{aligned}
$$

As in Lecture 8, a term with $\widehat p(x)=0$ contributes zero. We also require the candidate to assign positive probability to each observed outcome. The empirical entropy $H(\widehat p)$ is fixed once the data are given. Minimizing this KL divergence therefore **exactly** maximizes the sample log-likelihood. For an exponential-family candidate, substitute the sample log-likelihood above to see what is being minimized:

$$
D_{\mathrm{KL}}(\widehat p\parallel q_{\boldsymbol\eta})
=\underbrace{-H(\widehat p)
-\frac1N\sum_{n=1}^{N}\log h(x_n)}_{\text{independent of }\boldsymbol\eta}
-\boldsymbol\eta^{\mathsf T}\overline{\mathbf T}
+A(\boldsymbol\eta).
$$

Only $-\boldsymbol\eta^{\mathsf T}\overline{\mathbf T}+A(\boldsymbol\eta)$ changes with the natural parameter. Minimizing it is the same calculation as maximizing $\boldsymbol\eta^{\mathsf T}\overline{\mathbf T}-A(\boldsymbol\eta)$. Practice problem 4 checks the equality numerically for Bernoulli data.

For **continuous** observations, the empirical distribution consists of point masses, whereas $q_{\boldsymbol\eta}$ is a density. Their KL divergence is generally not finite, so we do **not** use the exact empirical-KL identity in that case. The MLE still maximizes the observed average log *density*; the earlier population-level KL argument explains its connection to fitting the data-generating distribution. -->

## Practice problems and solutions

### 1. Bernoulli natural parameter and moments

Six IID Bernoulli observations are $1,0,1,1,0,1$.

1. Find the sample's sufficient statistic and the MLE of $\theta$.
2. Find the corresponding finite natural-parameter estimate $\widehat\eta$.
3. Differentiate $A(\eta)$ to obtain the mean and variance at that estimate.

::::{admonition} Solution 1
:class: dropdown

The sum of successes is $S=4$ and $N=6$, so the Bernoulli MLE is $\widehat\theta=S/N=2/3$. Since it lies strictly between zero and one, its natural parameter is finite:

$$
\widehat\eta
=\log\frac{\widehat\theta}{1-\widehat\theta}
=\log\frac{2/3}{1/3}
=\log 2.
$$

For $A(\eta)=\log(1+e^\eta)$,

$$
A'(\eta)=\frac{e^\eta}{1+e^\eta},
\qquad
A''(\eta)=\frac{e^\eta}{(1+e^\eta)^2}.
$$

At $\eta=\log2$, $e^\eta=2$. The fitted model has $\mathbb E[X]=A'(\log2)=2/3$ and $\operatorname{Var}(X)=A''(\log2)=2/9$. These are the model's moments at the estimated parameter, not the sampling variance of $\widehat\theta$.
::::

### 2. Poisson counts and sufficiency

Six IID Poisson counts are $0,1,2,1,0,2$.

1. Write the parameter-dependent part of their likelihood and identify a sufficient summary.
2. Find the MLE of $\lambda$ and its natural parameter.
3. Use derivatives of $A$ to find the fitted mean and variance of a future count.

::::{admonition} Solution 2
:class: dropdown

The counts sum to $S=6$. Their full likelihood is

$$
\mathcal L(\lambda;\mathcal D)
=\left(\prod_{n=1}^{6}\frac{1}{x_n!}\right)
\lambda^{S}e^{-6\lambda}.
$$

The factorial product is fixed once the data are observed, so the parameter-dependent part is $\lambda^6e^{-6\lambda}$ and $S=6$ is sufficient when $N$ is fixed. Moment matching gives $\widehat\lambda=S/N=1$, hence $\widehat\eta=\log1=0$.

Since $A(\eta)=e^\eta$, we have $A'(\eta)=A''(\eta)=e^\eta$. At $\widehat\eta=0$, a future count has fitted mean 1 and fitted variance 1.
::::

### 3. Gaussian with two unknown parameters

Suppose $x_1=1$, $x_2=2$, and $x_3=3$ are modeled as IID Gaussian observations with both $\mu$ and $\sigma^2$ unknown.

1. Compute the two components of the sufficient summary $S(\mathcal D)$.
2. Use the moment-matching equations to find $\widehat\mu_{\mathrm{ML}}$ and $\widehat\sigma^2_{\mathrm{ML}}$.
3. Convert these estimates to $\widehat\eta_1$ and $\widehat\eta_2$.

::::{admonition} Solution 3
:class: dropdown

The two sums are

$$
S(\mathcal D)
=\begin{bmatrix}
1+2+3\\[2pt]1^2+2^2+3^2
\end{bmatrix}
=\begin{bmatrix}6\\[2pt]14\end{bmatrix}.
$$

The average observed statistics are $(6/3,14/3)=(2,14/3)$. Matching $\mathbb E[X]=\mu$ and $\mathbb E[X^2]=\mu^2+\sigma^2$ gives

$$
\widehat\mu_{\mathrm{ML}}=2,
\qquad
\widehat\sigma^2_{\mathrm{ML}}
=\frac{14}{3}-2^2=\frac23.
$$

Therefore,

$$
\widehat\eta_1
=\frac{\widehat\mu}{\widehat\sigma^2}=3,
\qquad
\widehat\eta_2
=-\frac{1}{2\widehat\sigma^2}=-\frac34.
$$

The negative second natural parameter satisfies the Gaussian parameter restriction.
::::

### 4. Empirical KL and maximum likelihood

Four IID Bernoulli observations contain three successes and one failure. Compare two candidate models: $q_{0.75}=\operatorname{Bernoulli}(0.75)$ and $q_{0.5}=\operatorname{Bernoulli}(0.5)$.

1. Write the empirical distribution $\widehat p$ over outcomes 0 and 1, and identify the Bernoulli MLE.
2. Compute the average negative log-likelihood of each candidate in nats.
3. Compute $D_{\mathrm{KL}}(\widehat p\parallel q_{0.5})$. Show that it equals the difference between the two average negative log-likelihoods.

::::{admonition} Solution 4
:class: dropdown

The empirical probabilities are $\widehat p(1)=3/4$ and $\widehat p(0)=1/4$. The Bernoulli MLE is the success fraction, $\widehat\theta_{\mathrm{ML}}=3/4$, so $q_{0.75}=\widehat p$.

For $q_{0.75}$ the average negative log-likelihood is

$$
-\frac14\ell(0.75;\mathcal D)
=-\frac34\log\frac34-\frac14\log\frac14
\approx0.562\text{ nats}.
$$

For $q_{0.5}$ it is

$$
-\frac14\ell(0.5;\mathcal D)
=-\frac34\log\frac12-\frac14\log\frac12
=\log2\approx0.693\text{ nats}.
$$

Taking $\widehat p$ as the first argument of KL,

$$
\begin{aligned}
D_{\mathrm{KL}}(\widehat p\parallel q_{0.5})
&=\frac34\log\frac{3/4}{1/2}
+\frac14\log\frac{1/4}{1/2}\\
&=\frac34\log\frac32+\frac14\log\frac12\\
&\approx0.131\text{ nats}.
\end{aligned}
$$

Because $q_{0.75}=\widehat p$, its KL divergence from $\widehat p$ is zero. The difference between the average negative log-likelihoods is $0.693-0.562\approx0.131$ nats, exactly the KL difference before rounding. This example shows the MLE–KL connection using discrete outcomes, where the empirical distribution and both candidates are directly comparable.
::::

## Reading

Christopher M. Bishop, *Pattern Recognition and Machine Learning*, Section 2.4, especially Section 2.4.1 on maximum likelihood and sufficient statistics.

## References

- Michael I. Jordan, [*The Exponential Family: Basics*, Chapter 8](https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter8.pdf), especially Sections 8.1, 8.3, and 8.5–8.8.
