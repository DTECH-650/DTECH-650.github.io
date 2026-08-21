# Lecture 8 - Information Theory

Probability describes uncertainty about possible outcomes. Information theory asks a closely related question:

> How much do we learn when an outcome is observed, and how well does a probability model describe the outcomes that actually occur?


<!--
## Learning objectives

After completing this lecture, you should be able to:

- compute the information associated with an outcome;
- compute and interpret entropy and conditional entropy;
- distinguish entropy from cross-entropy;
- compute and interpret KL divergence and mutual information;
- explain how cross-entropy and KL divergence are related; and
- connect maximum likelihood with empirical cross-entropy.
-->

## Information as surprise

Suppose an outcome $x$ has probability $p(x)$. An outcome that was nearly certain tells us little when it occurs. A rare outcome is more surprising and therefore carries more information.

A useful measure of the information in observing $x$ should have three properties:

1. It should decrease as $p(x)$ increases.
2. An event with probability 1 should provide no new information.
3. The information from two independent outcomes should add.

The logarithm gives exactly these properties. The **self-information**, also called **surprisal**, of an outcome is

$$
I(x)=-\log p(x).
$$

The base of the logarithm determines the unit:

- $\log_2$ measures information in **bits**;
- $\ln$ measures information in **nats**.

Unless otherwise stated, this lecture uses natural logarithms in formulas because they are standard in statistical learning. Examples about binary codes use base-2 logarithms.

### Interpreting surprisal

| Probability of the outcome | Surprisal in bits |
|---:|---:|
| $1$ | $-\log_2 1=0$ |
| $1/2$ | $-\log_2(1/2)=1$ |
| $1/4$ | $-\log_2(1/4)=2$ |
| $1/8$ | $-\log_2(1/8)=3$ |

Halving an outcome's probability adds one bit of surprisal. An impossible event has infinite surprisal in the ideal mathematical model because

$$
-\log 0=+\infty.
$$

This is one reason probabilistic models are penalized so strongly when they assign zero probability to an event that then occurs.

If $X$ and $Y$ are independent, then $p(x,y)=p(x)p(y)$, and

$$
\begin{aligned}
I(x,y)
&=-\log p(x,y)\\
&=-\log\big[p(x)p(y)\big]\\
&=-\log p(x)-\log p(y)\\
&=I(x)+I(y).
\end{aligned}
$$

Information adds across independent observations because probabilities multiply and logarithms turn products into sums. The same algebra will later turn an IID likelihood into a sum of log-likelihood terms.

:::{admonition} Statistical information is not importance
:class: note

Surprisal measures how unexpected an outcome is under a specified probability model. It does not measure whether the outcome is meaningful, useful, or important to a person.
:::

## Entropy: average uncertainty

Surprisal describes one observed outcome. Before observing a discrete random variable $X$, we can average the surprisal over all possible outcomes. The resulting quantity is the **entropy**

$$
H(X)
=-\sum_x p(x)\log p(x)
=\mathbb{E}_{p}\left[-\log p(X)\right].
$$

We use the convention $0\log 0=0$. Entropy is the expected amount of surprise in one draw from the distribution.

For a discrete variable:

- $H(X)\geq0$;
- $H(X)=0$ when one outcome occurs with probability 1;
- spreading probability more evenly generally increases entropy; and
- among $K$ possible outcomes, entropy is maximized by the Uniform distribution.

For a Uniform distribution over $K$ outcomes,

$$
p(x)=\frac{1}{K}
$$

and

$$
H(X)
=-\sum_{x=1}^{K}\frac{1}{K}\log\frac{1}{K}
=\log K.
$$

In base 2, $\log_2K$ is the average number of bits required by an ideal code for one of $K$ equally likely outcomes.

### Example: entropy of a Bernoulli variable

Let

$$
X\sim\operatorname{Bernoulli}(\mu).
$$

Its entropy in bits is

$$
H(X)
=-\mu\log_2\mu
-(1-\mu)\log_2(1-\mu).
$$

| $\mu$ | Interpretation | Entropy |
|---:|---|---:|
| $0$ or $1$ | Outcome is certain | $0$ bits |
| $0.5$ | Both outcomes are equally likely | $1$ bit |
| $0.9$ | One outcome is much more likely | Approximately $0.469$ bits |

Entropy is largest at $\mu=0.5$, where the next outcome is hardest to predict. It approaches zero as $\mu$ approaches 0 or 1.

## Joint and conditional entropy

When two random variables are considered together, their **joint entropy** is

$$
H(X,Y)
=-\sum_x\sum_y p(x,y)\log p(x,y).
$$

The **conditional entropy** of $Y$ given $X$ is

$$
H(Y\mid X)
=-\sum_x\sum_y p(x,y)\log p(y\mid x).
$$

It measures the uncertainty that remains about $Y$ after $X$ is known. Entropy obeys the **chain rule**

$$
\boxed{
H(X,Y)=H(X)+H(Y\mid X)
}.
$$

The joint uncertainty equals the uncertainty about $X$ plus the uncertainty still remaining about $Y$ after learning $X$.

For discrete variables, conditioning cannot increase entropy on average:

$$
H(Y\mid X)\leq H(Y).
$$

Two limiting examples make the meaning clear.

### Perfectly informative variable

Let $X$ be a fair binary variable and let $Y=X$. Then

$$
H(Y)=1\text{ bit},
\qquad
H(Y\mid X)=0.
$$

Once $X$ is known, there is no uncertainty left about $Y$.

### Uninformative variable

Let $X$ and $Y$ be independent fair binary variables. Then

$$
H(Y)=1\text{ bit},
\qquad
H(Y\mid X)=1\text{ bit}.
$$

Knowing $X$ does not reduce uncertainty about $Y$.

## A caution about continuous variables

For a continuous random variable with density $p(x)$, the analogous quantity is **differential entropy**

$$
h(X)=-\int p(x)\log p(x)\,dx.
$$

Differential entropy resembles discrete entropy, but it behaves differently:

- it can be negative;
- its value changes when the units or coordinate scale change; and
- it is not, by itself, the expected number of bits needed to identify an exact real number.

For a Gaussian random variable,

$$
X\sim\mathcal{N}(\mu,\sigma^2),
$$

the differential entropy is

$$
h(X)
=\frac{1}{2}\log\left(2\pi e\sigma^2\right).
$$

It depends on the variance but not the mean. Among continuous distributions with a specified mean and variance, the Gaussian has the largest differential entropy. In this sense, it makes the fewest additional assumptions about the distribution's shape beyond those two moments.

Because differential entropy depends on the coordinate system, KL divergence and mutual information are often more reliable quantities for comparing continuous distributions.

## Cross-entropy: evaluating one distribution with another

Entropy evaluates a distribution using its own probabilities. In modeling, the true data-generating distribution $p(x)$ is generally unknown, and we use a model distribution $q(x)$ instead.

The **cross-entropy** from $p$ to $q$ is

$$
H(p,q)
=-\sum_x p(x)\log q(x)
=\mathbb{E}_{X\sim p}\left[-\log q(X)\right].
$$

For continuous variables, the corresponding expression replaces the sum with an integral when it is well defined.

Cross-entropy answers:

> If outcomes are generated according to $p$, how surprised will a model using $q$ be on average?

It is generally not symmetric:

$$
H(p,q)\neq H(q,p).
$$

The first distribution determines how frequently outcomes occur; the second supplies the probabilities used to score those outcomes.

### Binary cross-entropy

Suppose the true probability of a binary outcome is $p$, while a model predicts probability $q$. The cross-entropy is

$$
H(p,q)
=-p\log q-(1-p)\log(1-q).
$$

For observed binary labels $y_n\in\{0,1\}$ and predicted probabilities $q_n$, the empirical average is

$$
\widehat{H}
=-\frac{1}{N}\sum_{n=1}^{N}
\left[
y_n\log q_n
+(1-y_n)\log(1-q_n)
\right].
$$

This is the **binary cross-entropy loss** used in probabilistic binary classification.

### Multiclass cross-entropy

For $K$ classes, let $y_{nk}$ be a one-hot target and let $q_{nk}$ be the predicted probability of class $k$. The empirical multiclass cross-entropy is

$$
\widehat{H}
=-\frac{1}{N}
\sum_{n=1}^{N}\sum_{k=1}^{K}
y_{nk}\log q_{nk}.
$$

Because only the true class has $y_{nk}=1$, the loss for observation $n$ reduces to

$$
-\log q_{n,\text{true class}}.
$$

A confident correct prediction has low loss. A confident incorrect prediction has very high loss.

:::{admonition} Why calibrated probabilities matter
:class: tip

Accuracy only records whether the most probable class was correct. Cross-entropy also evaluates the probability assigned to that class. Two models can have the same accuracy while receiving different cross-entropies because one expresses better-calibrated confidence.
:::

## KL divergence: distributional mismatch

The **Kullback–Leibler divergence** that uses $p$ as the reference distribution and $q$ as the approximating distribution is

$$
D_{\mathrm{KL}}(p\parallel q)
=\sum_x p(x)
\log\frac{p(x)}{q(x)}.
$$

It can be rewritten as

$$
\begin{aligned}
D_{\mathrm{KL}}(p\parallel q)
&=\sum_xp(x)\log p(x)
-\sum_xp(x)\log q(x)\\
&=H(p,q)-H(p).
\end{aligned}
$$

Therefore,

$$
\boxed{
H(p,q)=H(p)+D_{\mathrm{KL}}(p\parallel q)
}.
$$

Cross-entropy contains two parts:

1. the irreducible uncertainty $H(p)$ in the data-generating distribution; and
2. the additional cost $D_{\mathrm{KL}}(p\parallel q)$ caused by using $q$ instead of $p$.

KL divergence has several important properties:

- $D_{\mathrm{KL}}(p\parallel q)\geq0$;
- it equals zero exactly when $p=q$ wherever $p$ has probability mass;
- it is asymmetric, so $D_{\mathrm{KL}}(p\parallel q)$ generally differs from $D_{\mathrm{KL}}(q\parallel p)$; and
- it can be infinite if $q(x)=0$ for an outcome with $p(x)>0$.

KL divergence is not a distance metric because it is asymmetric and does not satisfy the triangle inequality.

### Why minimizing cross-entropy minimizes KL divergence

When the data-generating distribution $p$ is fixed, its entropy $H(p)$ does not depend on the model $q$. Consequently,

$$
\underset{q}{\operatorname{arg\,min}}\ H(p,q)
=
\underset{q}{\operatorname{arg\,min}}\ 
D_{\mathrm{KL}}(p\parallel q).
$$

Optimizing cross-entropy is therefore a way to bring the model distribution closer to the data distribution in the KL-divergence sense.

## Mutual information: information shared by variables

**Mutual information** measures how much knowing one random variable reduces uncertainty about another:

$$
\begin{aligned}
I(X;Y)
&=D_{\mathrm{KL}}\left(
p(x,y)\parallel p(x)p(y)
\right)\\
&=H(X)-H(X\mid Y)\\
&=H(Y)-H(Y\mid X).
\end{aligned}
$$

It compares the true joint distribution $p(x,y)$ with the distribution $p(x)p(y)$ that would hold if $X$ and $Y$ were independent.

Its key properties are

$$
I(X;Y)\geq0,
\qquad
I(X;Y)=I(Y;X),
$$

and

$$
I(X;Y)=0
\quad\Longleftrightarrow\quad
X\text{ and }Y\text{ are independent}.
$$

For the earlier examples:

- If $Y=X$ and $X$ is a fair bit, then $I(X;Y)=1$ bit.
- If $X$ and $Y$ are independent fair bits, then $I(X;Y)=0$.

Unlike covariance, mutual information can detect nonlinear statistical dependence. It does not tell us that one variable causes the other.

## The connection to maximum likelihood

Let the unknown data-generating distribution be $p_{\mathrm{data}}(x)$ and let a parametric model be $q_{\boldsymbol{\theta}}(x)$. The population cross-entropy is

$$
H\left(p_{\mathrm{data}},q_{\boldsymbol{\theta}}\right)
=-\mathbb{E}_{X\sim p_{\mathrm{data}}}
\left[\log q_{\boldsymbol{\theta}}(X)\right].
$$

We do not know $p_{\mathrm{data}}$, but a sample

$$
\mathcal{D}=\{x_1,\ldots,x_N\}
$$

provides the empirical approximation

$$
\widehat{H}\left(q_{\boldsymbol{\theta}}\right)
=-\frac{1}{N}\sum_{n=1}^{N}
\log q_{\boldsymbol{\theta}}(x_n).
$$

For IID observations, the likelihood is

$$
\mathcal{L}(\boldsymbol{\theta};\mathcal{D})
=\prod_{n=1}^{N}q_{\boldsymbol{\theta}}(x_n),
$$

so

$$
\boxed{
\widehat{H}\left(q_{\boldsymbol{\theta}}\right)
=-\frac{1}{N}
\log\mathcal{L}(\boldsymbol{\theta};\mathcal{D})
}.
$$

It follows that

$$
\begin{aligned}
\underset{\boldsymbol{\theta}}{\operatorname{arg\,max}}\ 
\mathcal{L}(\boldsymbol{\theta};\mathcal{D})
&=
\underset{\boldsymbol{\theta}}{\operatorname{arg\,min}}\ 
\left[-\log\mathcal{L}(\boldsymbol{\theta};\mathcal{D})\right]\\
&=
\underset{\boldsymbol{\theta}}{\operatorname{arg\,min}}\ 
\widehat{H}\left(q_{\boldsymbol{\theta}}\right).
\end{aligned}
$$

Maximum likelihood can therefore be understood as selecting the model in the chosen family that minimizes empirical cross-entropy. At the population level, this corresponds to minimizing

$$
D_{\mathrm{KL}}
\left(
p_{\mathrm{data}}\parallel q_{\boldsymbol{\theta}}
\right).
$$

<!-- Lecture 7 developed likelihood and maximum-likelihood estimation formally. Lecture 9 will turn negative log-likelihood into practical loss functions. -->

<!-- ### A preview of later loss functions

| Observation model | Negative log-likelihood becomes |
|---|---|
| Bernoulli | Binary cross-entropy |
| Categorical | Multiclass cross-entropy |
| Gaussian with fixed variance | Squared error, up to scaling and an additive constant |

The correct loss follows from the assumed probability model for the observations. This is why selecting a loss function is also a modeling decision.

## Where these ideas return later

| Later course topic | Information-theoretic connection |
|---|---|
| Maximum-likelihood estimation | Maximizing likelihood minimizes empirical cross-entropy |
| Regression and classification | Bernoulli, categorical, and Gaussian likelihoods produce familiar losses |
| Model evaluation | Log loss rewards calibrated probabilities and heavily penalizes confident errors |
| Probabilistic graphical models | Conditional entropy and mutual information quantify remaining dependence |
| Bayesian inference | KL divergence compares posterior, prior, and approximate distributions |
| Neural networks | Sigmoid and softmax outputs are commonly trained with cross-entropy |
| Feature and representation learning | Mutual information quantifies retained information about a target or input | -->

## Common misconceptions

- **High entropy is not automatically good or bad.** It means the modeled outcome is uncertain.
- **A rare event is not necessarily an error.** It simply has high surprisal under the model.
- **Cross-entropy is not symmetric.** The data distribution and model distribution play different roles.
- **KL divergence is not a geometric distance.** Its direction matters.
- **Differential entropy is not discrete entropy with a sum replaced mechanically by an integral.** It can be negative and depends on scale.
- **Mutual information does not establish causation.** It measures statistical dependence.

## Summary

- Self-information, $-\log p(x)$, measures how surprising an observed outcome is.
- Entropy is expected surprisal and quantifies uncertainty in a distribution.
- Conditional entropy measures the uncertainty remaining after another variable is known.
- Cross-entropy evaluates predictions from one distribution on outcomes generated by another.
- KL divergence is the extra cross-entropy caused by using an imperfect model.
- Mutual information measures the reduction in uncertainty about one variable obtained from another.
- Average negative log-likelihood is empirical cross-entropy, connecting maximum likelihood directly to common machine-learning losses.

## Practice problems

### 1. Surprisal and entropy

A source produces four symbols with probabilities

$$
p(a)=\frac{1}{2},
\qquad
p(b)=\frac{1}{4},
\qquad
p(c)=p(d)=\frac{1}{8}.
$$

1. Find the surprisal of each symbol in bits.
2. Find the entropy of the source.
3. Is the entropy smaller than, equal to, or larger than that of a Uniform distribution over four symbols?

:::{admonition} Solution
:class: dropdown

The surprisals are

$$
I(a)=1,\qquad I(b)=2,\qquad I(c)=I(d)=3
$$

bits. Therefore,

$$
\begin{aligned}
H(X)
&=\frac{1}{2}(1)
+\frac{1}{4}(2)
+\frac{1}{8}(3)
+\frac{1}{8}(3)\\
&=1.75\text{ bits}.
\end{aligned}
$$

A Uniform distribution over four symbols has entropy

$$
\log_2 4=2\text{ bits}.
$$

The given source is less uncertain because one symbol occurs more frequently than the others.
:::

### 2. Cross-entropy and KL divergence

The true probability of success is $p=0.75$, but a model uses $q=0.5$. Use base-2 logarithms.

1. Compute the entropy $H(p)$.
2. Compute the cross-entropy $H(p,q)$.
3. Compute $D_{\mathrm{KL}}(p\parallel q)$ using the relationship between entropy and cross-entropy.

:::{admonition} Solution
:class: dropdown

The Bernoulli entropy is

$$
\begin{aligned}
H(p)
&=-0.75\log_2(0.75)-0.25\log_2(0.25)\\
&\approx0.8113\text{ bits}.
\end{aligned}
$$

Because the model assigns probability $0.5$ to either outcome,

$$
\begin{aligned}
H(p,q)
&=-0.75\log_2(0.5)-0.25\log_2(0.5)\\
&=1\text{ bit}.
\end{aligned}
$$

Therefore,

$$
D_{\mathrm{KL}}(p\parallel q)
=H(p,q)-H(p)
\approx1-0.8113
=0.1887\text{ bits}.
$$

The additional $0.1887$ bits quantify the cost of using the model probability $0.5$ instead of the true probability $0.75$.
:::

### 3. Conditional entropy and mutual information

Let $X$ be a fair bit. Let $Y=X$ with probability $0.9$ and $Y=1-X$ with probability $0.1$.

1. Find $H(Y)$.
2. Find $H(Y\mid X)$.
3. Find $I(X;Y)$.

:::{admonition} Solution
:class: dropdown

First write the conditional probabilities implied by the problem:

$$
\begin{aligned}
P(Y=0\mid X=0)&=0.9,
&
P(Y=1\mid X=0)&=0.1,\\
P(Y=1\mid X=1)&=0.9,
&
P(Y=0\mid X=1)&=0.1.
\end{aligned}
$$

Because $X$ is fair,

$$
P(X=0)=P(X=1)=0.5.
$$

Multiplying each conditional probability by the corresponding probability of $X$ gives the joint distribution:

| | $Y=0$ | $Y=1$ | $P(X=x)$ |
|---|---:|---:|---:|
| $X=0$ | $(0.5)(0.9)=0.45$ | $(0.5)(0.1)=0.05$ | $0.50$ |
| $X=1$ | $(0.5)(0.1)=0.05$ | $(0.5)(0.9)=0.45$ | $0.50$ |
| $P(Y=y)$ | $0.50$ | $0.50$ | $1.00$ |

**Step 1: Find $H(Y)$**

Sum each column of the joint table to obtain the marginal distribution of $Y$:

$$
\begin{aligned}
P(Y=0)&=0.45+0.05=0.5,\\
P(Y=1)&=0.05+0.45=0.5.
\end{aligned}
$$

Thus, $Y$ is also a fair bit. Its entropy is

$$
\begin{aligned}
H(Y)
&=-\sum_y P(Y=y)\log_2P(Y=y)\\
&=-0.5\log_2(0.5)-0.5\log_2(0.5)\\
&=1\text{ bit}.
\end{aligned}
$$

Before learning $X$, there is one bit of uncertainty about $Y$.

**Step 2: Find $H(Y\mid X)$**

If $X=0$, then $Y$ equals 0 with probability $0.9$ and 1 with probability $0.1$. Therefore,

$$
\begin{aligned}
H(Y\mid X=0)
&=-0.9\log_2(0.9)-0.1\log_2(0.1)\\
&\approx0.4690\text{ bits}.
\end{aligned}
$$

If $X=1$, the labels of the two outcomes are reversed, but their probabilities remain $0.9$ and $0.1$. Entropy depends on the probabilities rather than the outcome names, so

$$
H(Y\mid X=1)\approx0.4690\text{ bits}.
$$

Conditional entropy averages these two quantities over the possible values of $X$:

$$
\begin{aligned}
H(Y\mid X)
&=\sum_xP(X=x)H(Y\mid X=x)\\
&=(0.5)(0.4690)+(0.5)(0.4690)\\
&=0.4690\text{ bits}.
\end{aligned}
$$

Even after $X$ is known, some uncertainty about $Y$ remains because there is a $10\%$ probability that the bit was flipped.

**Step 3: Find $I(X;Y)$**

Mutual information is the reduction in uncertainty about $Y$ obtained by observing $X$:

$$
\begin{aligned}
I(X;Y)
&=H(Y)-H(Y\mid X)\\
&=1-0.4690\\
&=0.5310\text{ bits}.
\end{aligned}
$$

We can verify the result directly from the joint distribution:

$$
I(X;Y)
=\sum_x\sum_y p(x,y)
\log_2\frac{p(x,y)}{p(x)p(y)}.
$$

All four products of the marginals equal $(0.5)(0.5)=0.25$. The two matching outcomes have joint probability $0.45$, and the two mismatching outcomes have joint probability $0.05$. Hence,

$$
\begin{aligned}
I(X;Y)
&=2(0.45)\log_2\left(\frac{0.45}{0.25}\right)
+2(0.05)\log_2\left(\frac{0.05}{0.25}\right)\\
&=0.9\log_2(1.8)+0.1\log_2(0.2)\\
&\approx0.5310\text{ bits}.
\end{aligned}
$$

Therefore, knowing $X$ removes about $0.531$ of the original 1 bit of uncertainty about $Y$, while the noisy flip leaves about $0.469$ bits unresolved.
:::

### 4. Empirical cross-entropy

A binary data set contains six successes and two failures. Compare two constant-probability models:

$$
q_A=0.75,
\qquad
q_B=0.50.
$$

1. Compute the average negative log-likelihood of each model in nats.
2. Which model has the larger likelihood?
3. Explain the result using the empirical success frequency.

:::{admonition} Solution
:class: dropdown

For model $A$,

$$
\begin{aligned}
\widehat{H}_A
&=-\frac{1}{8}
\left[
6\ln(0.75)+2\ln(0.25)
\right]\\
&\approx0.5623\text{ nats}.
\end{aligned}
$$

For model $B$,

$$
\begin{aligned}
\widehat{H}_B
&=-\frac{1}{8}
\left[
6\ln(0.5)+2\ln(0.5)
\right]\\
&=\ln 2\\
&\approx0.6931\text{ nats}.
\end{aligned}
$$

Model $A$ has the lower average negative log-likelihood and therefore the larger likelihood. Its predicted probability $0.75$ matches the empirical success frequency $6/8=0.75$.
:::

## Reading

Christopher M. Bishop, *Pattern Recognition and Machine Learning*, Chapter 1, pages 48–58.

[Official book page and free PDF](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/)
