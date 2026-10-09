# Lecture 13 - Bayesian Inference and Conjugate Priors

Lecture 7 estimated an unknown parameter by choosing a single value that maximized the likelihood. Lecture 12 showed that, for many familiar models, the likelihood depends on the sample through a small set of sufficient statistics. We now ask a different but related question: **how should our uncertainty about an unknown parameter change when we observe data?**

The answer begins with a prior distribution and Bayes' rule. Updating a probability distribution over an unknown quantity after observing data is called **Bayesian inference**. When that quantity is a model parameter, as it is here, we are doing **Bayesian parameter learning**. We will first develop the general prior–likelihood–evidence–posterior calculation, including prediction. Then we will see why certain prior choices make that calculation especially simple. The Bernoulli model and its beta prior provide our main example; a Gaussian example shows that the same pattern is not limited to binary data.

We retain the notation of earlier lectures. $X$ is a random variable, $x$ is a realized value, and $\mathcal D=(x_1,\ldots,x_N)$ is the observed dataset. For the Bernoulli model, $\theta$ is the probability of success, as in Lecture 12. We write $p$ for a PMF or density according to the quantity being modeled, and $P$ for the probability of a stated event.

## From a parameter estimate to uncertainty about a parameter

In a frequentist likelihood calculation, we hold the observed data fixed and compare possible **fixed** parameter values. For example, the Bernoulli likelihood tells us how well each proposed $\theta$ explains the observed successes and failures. Maximum likelihood returns one estimate, $\widehat\theta_{\mathrm{ML}}$.

In Bayesian parameter learning, we instead represent uncertainty about $\theta$ with a probability distribution. Before seeing the current data, that distribution is the **prior**, $p(\theta)$. After seeing the data, it is the **posterior**, $p(\theta\mid\mathcal D)$. The underlying parameter is not assumed to change when we collect data; what changes is our distribution describing which values are plausible. This distinction also applies if the unknown parameter is a vector $\boldsymbol\theta$ rather than a scalar.

Parameters that specify the prior are called **hyperparameters**. We will treat them as chosen, fixed quantities in this lecture. Giving the hyperparameters their own priors would be a further modeling step, not a requirement for Bayesian parameter learning.

## Prior, likelihood, evidence, and posterior

Bayes' rule applied to the unknown parameter gives

$$
\boxed{
p(\theta\mid\mathcal D)
=\frac{p(\mathcal D\mid\theta)p(\theta)}{p(\mathcal D)}
}.
$$

Here $p(\mathcal D\mid\theta)$ describes the observed data **if** the parameter has a proposed value. Once $\mathcal D$ has been observed, the same expression viewed as a function of $\theta$ is the **likelihood**. It is not, by itself, a probability distribution over $\theta$: it need not integrate to one across parameter values. The prior $p(\theta)$ *is* a distribution over the parameter.

The numerator, $p(\mathcal D\mid\theta)p(\theta)$, favors parameter values that both explain the data and had appreciable prior density. To make the result a normalized posterior distribution, divide by the **evidence**:

$$
p(\mathcal D)
=\int p(\mathcal D\mid\theta)p(\theta)\,d\theta
\qquad\text{for a continuous parameter }\theta.
$$

If $\theta$ can take only a discrete set of values, replace the integral with a sum. The evidence averages the likelihood over possible parameter values, weighted by their prior probabilities. It is also called the **marginal likelihood** or the **prior predictive probability of the observed data**. For a fixed model and fixed dataset it is one number, not a function of $\theta$; hence we may first work with

$$
p(\theta\mid\mathcal D)
\propto p(\mathcal D\mid\theta)p(\theta)
$$

and normalize afterward. The symbol $\propto$ means that the two sides differ by a positive factor that does not depend on $\theta$. Evidence can also be used to compare different models, but that is a separate task: here its immediate job is to normalize the parameter posterior.

### A small example of the evidence calculation

Suppose we consider only two possible values, $\theta=1/4$ and $\theta=3/4$, and initially assign each probability $1/2$. We then observe one Bernoulli success, $\mathcal D=(1)$. The likelihood of that observation is $1/4$ under the first value and $3/4$ under the second. The evidence is the prior-weighted sum

$$
\begin{aligned}
P(X=1)
&=P(X=1\mid\theta=1/4)P(\theta=1/4)
 +P(X=1\mid\theta=3/4)P(\theta=3/4)\\
&=\frac14\cdot\frac12+\frac34\cdot\frac12
=\frac12.
\end{aligned}
$$

Bayes' rule therefore gives $P(\theta=1/4\mid X=1)=(1/4)(1/2)/(1/2)=1/4$ and $P(\theta=3/4\mid X=1)=3/4$. These posterior probabilities add to one. The success made the larger value of $\theta$ more plausible, but did not establish it with certainty. The continuous-parameter calculations below follow exactly the same logic, with an integral in place of this two-term sum.

### Repeated observations and posterior prediction

If $X_1,\ldots,X_N$ are **conditionally IID given** $\theta$, their likelihood factors as

$$
p(\mathcal D\mid\theta)=\prod_{n=1}^{N}p(x_n\mid\theta).
$$

The phrase *given $\theta$* matters. Before we know $\theta$, observations that share it need not be independent after we average over our uncertainty about it. Lecture 16 will give a graphical way to represent this assumption; no graph is needed to perform the update here.

After learning from $\mathcal D$, we often want to predict a new observation $X_{\mathrm{new}}$. If the new observation follows the same sampling model and is conditionally independent of $\mathcal D$ given $\theta$, we average its probability over the **posterior**:

$$
\boxed{
p(x_{\mathrm{new}}\mid\mathcal D)
=\int p(x_{\mathrm{new}}\mid\theta)
p(\theta\mid\mathcal D)\,d\theta
}.
$$

For a discrete parameter, this is a sum. This **posterior predictive distribution** is different from substituting just one point estimate into $p(x_{\mathrm{new}}\mid\theta)$: it accounts for the remaining uncertainty about $\theta$. The evidence averages a likelihood over the *prior* to assign probability to the already observed data; posterior prediction averages a new observation's distribution over the *posterior*.

## Why conjugate priors help

Bayes' rule works with any valid prior for which the required calculations can be performed. But the posterior may have a complicated form, and its evidence may require a difficult integral. A prior is **conjugate** to a likelihood when the resulting posterior belongs to the **same distribution family as the prior**. This gives us a familiar posterior form whose parameters can be updated directly from the data.

Conjugacy is a convenience, not the definition of Bayesian inference and not a guarantee that the prior is scientifically appropriate. The prior still needs to reflect reasonable assumptions for the problem. The benefit is clearest when observations arrive one at a time: after each observation, the posterior has the same form and can become the next prior.

There is a connection to Lecture 12. For IID observations from an exponential-family likelihood,

$$
p(x\mid\boldsymbol\eta)
=h(x)\exp\!\left\{
\boldsymbol\eta^{\mathsf T}\mathbf T(x)-A(\boldsymbol\eta)
\right\},
$$

the parameter-dependent part of the likelihood is

$$
p(\mathcal D\mid\boldsymbol\eta)
\propto
\exp\!\left\{
\boldsymbol\eta^{\mathsf T}\underbrace{\sum_{n=1}^{N}\mathbf T(x_n)}_{S(\mathcal D)}
-NA(\boldsymbol\eta)
\right\}.
$$

The $h(x_n)$ factors are omitted in the second line only because they do not depend on $\boldsymbol\eta$. The data enter through the same sufficient statistic $S(\mathcal D)$ that we used for maximum likelihood. A suitably chosen conjugate prior has parameter-dependent terms that combine with these likelihood terms, so updating its hyperparameters requires $N$ and $S(\mathcal D)$ rather than the full ordered sample. We will see this explicitly, without introducing a general conjugate-prior formula, in the Bernoulli calculation.

## Beta prior and Bernoulli observations

Suppose $X_n\mid\theta\sim\operatorname{Bernoulli}(\theta)$ independently for $n=1,\ldots,N$. The notation $X_n\mid\theta$ says that, once a value of $\theta$ is specified, each trial has success probability $\theta$. We do **not** observe $\theta$ directly. We observe successes and failures and use them to learn about it.

For $0<\theta<1$, choose a **beta prior** with fixed hyperparameters $a>0$ and $b>0$:

$$
p(\theta)
=\operatorname{Beta}(\theta\mid a,b)
=\frac{\theta^{a-1}(1-\theta)^{b-1}}{B(a,b)},
\qquad 0<\theta<1.
$$

The beta function in the denominator is the normalizing constant

$$
B(a,b)=\int_0^1 u^{a-1}(1-u)^{b-1}\,du.
$$

It makes the prior density integrate to one. For $a=b=1$, the prior is uniform on $(0,1)$. Its mean is $a/(a+b)$; increasing both $a$ and $b$ while keeping their ratio fixed concentrates the prior more tightly around that mean. The prior describes uncertainty about $\theta$. It is **not** the Bernoulli distribution of an individual outcome $X_n$.

### Updating the prior

Let $S=\sum_{n=1}^{N}x_n$ be the number of successes, and let $F=N-S$ be the number of failures. From the Bernoulli PMF in Lectures 6 and 12, the likelihood of the **particular observed sequence** is

$$
\begin{aligned}
p(\mathcal D\mid\theta)
&=\prod_{n=1}^{N}\theta^{x_n}(1-\theta)^{1-x_n}\\
&=\theta^{S}(1-\theta)^{F}.
\end{aligned}
$$

Multiply this likelihood by the prior. For the moment, omit factors that do not depend on $\theta$:

$$
\begin{aligned}
p(\theta\mid\mathcal D)
&\propto p(\mathcal D\mid\theta)p(\theta)\\
&\propto
\theta^S(1-\theta)^F
\theta^{a-1}(1-\theta)^{b-1}\\
&=\theta^{a+S-1}(1-\theta)^{b+F-1}.
\end{aligned}
$$

Compare the final expression with the beta prior density: it has exactly the same powers of $\theta$ and $1-\theta$, with $a$ changed to $a+S$ and $b$ changed to $b+F$. Normalizing gives

$$
\boxed{
p(\theta\mid\mathcal D)
=\operatorname{Beta}(\theta\mid a+S,b+F)
}.
$$

That is **Beta–Bernoulli conjugacy**. Each observed success adds one to the first beta parameter; each failure adds one to the second. The sample order does not appear in the answer because, under the assumed IID model, $S$ is sufficient for $\theta$ when $N$ is known.

### Where did the evidence go?

In the proportional calculation we temporarily omitted the denominator in Bayes' rule. For this model, we can calculate it exactly by integrating likelihood times prior:

$$
\begin{aligned}
p(\mathcal D)
&=\int_0^1 p(\mathcal D\mid\theta)p(\theta)\,d\theta\\
&=\frac{1}{B(a,b)}
\int_0^1\theta^{a+S-1}(1-\theta)^{b+F-1}\,d\theta\\
&=\boxed{\frac{B(a+S,b+F)}{B(a,b)}}.
\end{aligned}
$$

The last integral is precisely the definition of $B(a+S,b+F)$. Dividing the numerator $p(\mathcal D\mid\theta)p(\theta)$ by this evidence gives the normalized beta posterior above.

There is one detail to keep straight. This $p(\mathcal D)$ is for the **particular ordered sequence** $x_1,\ldots,x_N$. If the only observation reported were the *count* $S$, then its sampling distribution would be binomial and the evidence for that count would have the additional factor $\binom NS$. That factor does not depend on $\theta$, so either description produces the same posterior, but the two evidence values are not equal.

### Learning and predicting from a small sample

Take $\theta\sim\operatorname{Beta}(2,2)$. Its mean is $2/(2+2)=1/2$, so before collecting data it is centered on balanced success and failure rates. Suppose the four observed trials are $(1,1,0,1)$. There are $S=3$ successes and $F=1$ failure, giving

$$
p(\theta\mid\mathcal D)=\operatorname{Beta}(\theta\mid5,3).
$$

The figure shows how the data shift and concentrate the distribution over plausible values of $\theta$. Both curves are **densities over the unknown parameter**, not PMFs of the binary observations.

~~~{figure} images/beta-bernoulli-update.svg
:name: beta-bernoulli-update
:alt: Density plot over theta from zero to one. The Beta(2,2) prior peaks at one half. After three successes and one failure, the Beta(5,3) posterior is taller and shifted toward larger theta.
:width: 760px

The prior and posterior distributions for $\theta$ after observing three successes and one failure.
~~~

The maximum-likelihood estimate from Lecture 7 would be $\widehat\theta_{\mathrm{ML}}=S/N=3/4$. The posterior is a distribution, not a single estimate. Its **mean** is

$$
\mathbb E[\theta\mid\mathcal D]
=\frac{a+S}{a+b+N}
=\frac{5}{8}.
$$

The posterior mean lies between the prior mean $1/2$ and the sample success fraction $3/4$. Indeed, the formula can be rewritten as

$$
\mathbb E[\theta\mid\mathcal D]
=\frac{a+b}{a+b+N}\underbrace{\frac{a}{a+b}}_{\text{prior mean}}
+\frac{N}{a+b+N}\underbrace{\frac{S}{N}}_{\text{sample fraction}}.
$$

The two coefficients add to one. This helps interpret how the chosen prior and the observed sample combine; it does not turn the prior into literal observations. In this example, the evidence for the ordered sequence is $B(5,3)/B(2,2)=2/35$. We do not need that value to identify the posterior family, but it is the number that makes the posterior integrate to one.

Finally, predict one more trial. Conditional on $\theta$, the probability of success is $\theta$. Averaging over the posterior gives

$$
\begin{aligned}
P(X_{\mathrm{new}}=1\mid\mathcal D)
&=\int_0^1P(X_{\mathrm{new}}=1\mid\theta)
p(\theta\mid\mathcal D)\,d\theta\\
&=\int_0^1\theta\,p(\theta\mid\mathcal D)\,d\theta\\
&=\mathbb E[\theta\mid\mathcal D]
=\frac58.
\end{aligned}
$$

For a *single* future Bernoulli outcome, the posterior predictive success probability happens to equal the posterior mean of $\theta$. That is a result of this particular sampling model, not a general instruction to replace every Bayesian predictive calculation with a posterior mean.

## A brief Gaussian comparison

Conjugacy is also available for continuous measurements. Suppose $X_n\mid\mu\sim\mathcal N(\mu,\sigma^2)$ independently, where the observation variance $\sigma^2$ is **known** and only the mean $\mu$ is unknown. Give the unknown mean a Gaussian prior,

$$
\mu\sim\mathcal N(\mu_0,\tau_0^2).
$$

Here $\mu_0$ and $\tau_0^2$ are fixed prior hyperparameters; $\tau_0^2$ describes uncertainty about the *mean*, whereas $\sigma^2$ describes variation of an *observation around that mean*. The likelihood as a function of $\mu$ and the prior are both exponentials of quadratic expressions in $\mu$. Multiplying them and collecting terms therefore produces another Gaussian density:

$$
p(\mu\mid\mathcal D)=\mathcal N(\mu\mid\mu_N,\tau_N^2),
$$

where $\overline x=N^{-1}\sum_nx_n$ and

$$
\frac{1}{\tau_N^2}
=\frac{1}{\tau_0^2}+\frac{N}{\sigma^2},
\qquad
\mu_N
=\tau_N^2\left(
\frac{\mu_0}{\tau_0^2}+\frac{N\overline x}{\sigma^2}
\right).
$$

The first equation says that **precisions** (inverse variances) add: prior precision plus data precision. The second weights the prior mean and sample mean by those precisions. More observations make the posterior variance $\tau_N^2$ smaller. For a new observation from the same model, averaging over the uncertain mean gives $X_{\mathrm{new}}\mid\mathcal D\sim\mathcal N(\mu_N,\sigma^2+\tau_N^2)$. The predictive variance includes both observation noise and remaining uncertainty about $\mu$.

The condition that $\sigma^2$ is known matters. If the **precision** $1/\sigma^2$ is unknown while $\mu$ is known, a gamma prior is conjugate in the one-dimensional case. If both $\mu$ and the precision are unknown, a joint normal–gamma prior is a conjugate choice. We will not derive those updates here. The choice of which Gaussian quantities are unknown determines which conjugate prior is appropriate; there is no single “Gaussian conjugate prior” for every version of the model.

## What to take forward

Bayesian inference keeps a distribution over an unknown quantity, updates it using observed data, and can average over the resulting posterior when predicting. In this lecture the unknown quantity is a **parameter**. Evidence is the normalizing, prior-averaged likelihood of the data. Conjugacy makes some of these calculations exact and compact; for Beta–Bernoulli data, the update is simply $(a,b)\mapsto(a+S,b+F)$. The same sufficient statistics that simplified likelihood calculations in Lecture 12 also simplify this posterior update.

In Lecture 14 we will learn how to represent models involving several related random variables with directed graphs. Such a graph can depict an unknown parameter and its observations, but the prior–posterior calculation itself does not require graphical notation.

## Practice problems and solutions

### 1. Prior, evidence, posterior, and prediction

A Bernoulli success probability is believed to be either $\theta=1/4$ or $\theta=3/4$, with prior probability $1/2$ for each. One success is observed.

1. Compute the evidence for the observed success.
2. Compute the posterior probability of each possible $\theta$.
3. Find the posterior predictive probability that the next trial is a success.

::::{admonition} Solution 1
:class: dropdown

The two likelihoods of $X_1=1$ are $1/4$ and $3/4$. Averaging under the prior gives

$$
P(X_1=1)=\frac12\cdot\frac14+\frac12\cdot\frac34=\frac12.
$$

Divide each likelihood-times-prior value by this evidence:

$$
P(\theta=1/4\mid X_1=1)=\frac{(1/4)(1/2)}{1/2}=\frac14,
\qquad
P(\theta=3/4\mid X_1=1)=\frac{(3/4)(1/2)}{1/2}=\frac34.
$$

The next success probability is an average under these **posterior** probabilities:

$$
P(X_{\mathrm{new}}=1\mid X_1=1)
=\frac14\cdot\frac14+\frac34\cdot\frac34
=\frac{10}{16}=\frac58.
$$

The observed success raised the predictive probability above its prior value of $1/2$.
::::

### 2. Beta–Bernoulli update and evidence

Let $\theta\sim\operatorname{Beta}(1,1)$. Three IID Bernoulli trials produce the ordered sequence $(1,0,1)$.

1. Find the posterior distribution of $\theta$ and its mean.
2. Find the evidence for this **ordered sequence**. You may use $B(r,s)=(r-1)!(s-1)!/(r+s-1)!$ for positive integers $r,s$.
3. If only the count “two successes out of three” were reported, what would its evidence be? Explain why the posterior is unchanged.

::::{admonition} Solution 2
:class: dropdown

Here $N=3$, $S=2$, and $F=1$. Add successes and failures to the prior hyperparameters to obtain $\theta\mid\mathcal D\sim\operatorname{Beta}(3,2)$. Its mean is $3/(3+2)=3/5$.

For the exact sequence $(1,0,1)$, the likelihood is $\theta^2(1-\theta)$, so

$$
p(\mathcal D)
=\frac{B(3,2)}{B(1,1)}
=\frac{2!1!/4!}{1}
=\frac1{12}.
$$

There are $\binom32=3$ distinct sequences with two successes. Their likelihoods have the same $\theta$-dependent factor. Therefore the evidence for the *count* is $3(1/12)=1/4$. The factor of three does not depend on $\theta$ and cancels when the posterior is normalized, leaving the same $\operatorname{Beta}(3,2)$ posterior.
::::

### 3. Updating a prior sequentially

Begin with $\theta\sim\operatorname{Beta}(3,1)$. Observe the Bernoulli results $0,1,1,0$ in that order.

1. Give the beta posterior after each result.
2. Check that updating once with the entire sample gives the same final posterior.
3. Find the posterior predictive probability of success on the next trial.

::::{admonition} Solution 3
:class: dropdown

A failure increases the second beta parameter by one; a success increases the first. The successive distributions are

$$
\operatorname{Beta}(3,1)
\xrightarrow{0}\operatorname{Beta}(3,2)
\xrightarrow{1}\operatorname{Beta}(4,2)
\xrightarrow{1}\operatorname{Beta}(5,2)
\xrightarrow{0}\operatorname{Beta}(5,3).
$$

In the complete sample $S=2$ and $F=2$, so the batch update gives $\operatorname{Beta}(3+2,1+2)=\operatorname{Beta}(5,3)$ as well. Under the IID model, the order does not affect the final posterior. Averaging the next trial's success probability $\theta$ under this posterior gives $P(X_{\mathrm{new}}=1\mid\mathcal D)=5/(5+3)=5/8$.
::::

### 4. Gaussian mean with known observation variance

Suppose $X_n\mid\mu\sim\mathcal N(\mu,1)$ independently, with prior $\mu\sim\mathcal N(0,4)$. The observations are $1,2,3$.

1. Find the posterior variance $\tau_N^2$ and mean $\mu_N$.
2. Give the posterior predictive distribution of one new observation.
3. Compare $\mu_N$ with the maximum-likelihood estimate of $\mu$.

::::{admonition} Solution 4
:class: dropdown

Here $N=3$, $\overline x=2$, $\mu_0=0$, $\tau_0^2=4$, and $\sigma^2=1$. The posterior precision is

$$
\frac{1}{\tau_N^2}
=\frac14+\frac31
=\frac{13}{4},
\qquad\text{so}\qquad
\tau_N^2=\frac4{13}.
$$

The posterior mean is

$$
\mu_N
=\frac4{13}\left(\frac04+\frac{3\cdot2}{1}\right)
=\frac{24}{13}.
$$

The new observation has both observation variance $1$ and posterior uncertainty $4/13$ in the mean. Thus

$$
X_{\mathrm{new}}\mid\mathcal D
\sim\mathcal N\!\left(\frac{24}{13},\frac{17}{13}\right).
$$

The maximum-likelihood estimate is the sample mean, $\widehat\mu_{\mathrm{ML}}=2$. The posterior mean $24/13\approx1.85$ lies between the prior mean $0$ and the sample mean $2$.
::::

## Reading

Christopher M. Bishop, *Pattern Recognition and Machine Learning*, Sections 1.2.3 and 1.2.6 on Bayesian probabilities and prediction; Section 2.1.1 on the beta distribution; Section 2.3.6 on Bayesian inference for the Gaussian; and Section 2.4.2 on conjugate priors.

## References

- Christopher M. Bishop, [*Pattern Recognition and Machine Learning*](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf), 2006.
- Steve Brunton, [*Bayesian Inference: Overview*](https://www.youtube.com/watch?v=XCEpIBqKogo&list=PLMrJAkhIeNNT14qn1c5qdL29A1UaHamjx&index=28) (video).
- Steve Brunton, [*Conjugate Priors Example: Normal Distribution and the Exponential Family of Distributions*](https://www.youtube.com/watch?v=q8ypTSQotXQ) (video).
