# Lecture 6 - Probability Distributions

This short lecture introduces the Uniform, Gaussian, and Exponential distributions and shows how their parameters control their behavior. These distributions model quantities spread evenly over an interval, real-valued measurements concentrated around a mean, and nonnegative waiting times between events.

## The Uniform Distribution

A continuous **Uniform distribution** assigns constant density across an interval. If

$$
X\sim\operatorname{Uniform}(a,b),
\qquad a<b,
$$

then its density is

$$
p(x\mid a,b)=
\begin{cases}
\dfrac{1}{b-a}, & a\leq x\leq b,\\
0, & \text{otherwise}.
\end{cases}
$$

Its CDF is

$$
F(x)=
\begin{cases}
0, & x<a,\\
\dfrac{x-a}{b-a}, & a\leq x\leq b,\\
1, & x>b.
\end{cases}
$$

Every pair of subintervals with the same length has the same probability. Individual points still have probability zero because this is a continuous distribution. For $a\leq c\leq d\leq b$,

$$
P(c\leq X\leq d)=\frac{d-c}{b-a}.
$$

The expectation and variance are

$$
\mathbb{E}[X]=\frac{a+b}{2},
\qquad
\operatorname{Var}(X)=\frac{(b-a)^2}{12}.
$$

### Example: A Uniform Waiting Time

Suppose a waiting time $T$ is modeled as Uniform between 0 and 20 minutes:

$$
T\sim\operatorname{Uniform}(0,20).
$$

The probability of waiting between 5 and 12 minutes is

$$
P(5\leq T\leq12)=\frac{12-5}{20-0}=\frac{7}{20}=0.35.
$$

The expected waiting time is

$$
\mathbb{E}[T]=\frac{0+20}{2}=10\text{ minutes}.
$$

## The Gaussian Distribution

The **Gaussian distribution**, also called the normal distribution, is central in pattern recognition and deep learning because it is mathematically convenient, often approximates aggregate noise, and extends naturally to high-dimensional vectors.

For a scalar random variable:

$$
x\sim\mathcal{N}(\mu,\sigma^2)
$$

has density:

$$
p(x\mid\mu,\sigma^2)
=
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left\{-\frac{(x-\mu)^2}{2\sigma^2}\right\}.
$$

The mean is $\mu$, and the variance is $\sigma^2$:

$$
\mathbb{E}[X]=\mu,
\qquad
\operatorname{Var}(X)=\sigma^2.
$$

For a vector $\mathbf{x}\in\mathbb{R}^D$, the multivariate Gaussian is:

$$
p(\mathbf{x}\mid\boldsymbol{\mu},\boldsymbol{\Sigma})
=
\frac{1}{(2\pi)^{D/2}|\boldsymbol{\Sigma}|^{1/2}}
\exp\left\{
-\frac{1}{2}
(\mathbf{x}-\boldsymbol{\mu})^\mathsf{T}
\boldsymbol{\Sigma}^{-1}
(\mathbf{x}-\boldsymbol{\mu})
\right\}.
$$

Here $\boldsymbol{\mu}$ is the mean vector and $\boldsymbol{\Sigma}$ is the covariance matrix.

The quadratic term:

$$
(\mathbf{x}-\boldsymbol{\mu})^\mathsf{T}
\boldsymbol{\Sigma}^{-1}
(\mathbf{x}-\boldsymbol{\mu})
$$

is the squared Mahalanobis distance. It measures distance from the mean while accounting for scale and correlation. Points in directions of high variance are less surprising than points the same Euclidean distance away in directions of low variance.

Common uses include:

- modeling sensor noise,
- modeling regression residuals,
- defining squared-error losses through Gaussian likelihoods,
- representing latent variables,
- approximating posterior distributions.

## The Exponential Distribution

The **exponential distribution** models waiting times between events in a memoryless process. If:

$$
T\sim\operatorname{Exponential}(\lambda),
$$

then:

$$
p(t\mid\lambda)=\lambda e^{-\lambda t},
\qquad t\geq 0,
\qquad \lambda>0.
$$

Its CDF is:

$$
F(t)=P(T\leq t)=1-e^{-\lambda t},
\qquad t\geq 0.
$$

Its expectation and variance are:

$$
\mathbb{E}[T]=\frac{1}{\lambda},
\qquad
\operatorname{Var}(T)=\frac{1}{\lambda^2}.
$$

The exponential distribution has the **memoryless property**:

$$
P(T>s+t\mid T>s)=P(T>t).
$$

This means that, under the model, the remaining waiting time does not depend on how long we have already waited.

### UAV Example: Time Until a Communication Dropout

If communication dropouts occur at an average rate of $\lambda=0.2$ per minute, then the waiting time $T$ until the next dropout can be modeled as:

$$
T\sim\operatorname{Exponential}(0.2).
$$

The expected waiting time is:

$$
\mathbb{E}[T]=\frac{1}{0.2}=5 \text{ minutes}.
$$

The probability of at least one dropout within the next 3 minutes is:

$$
P(T\leq 3)=1-e^{-0.2\cdot 3}
\approx 0.451.
$$

So this model predicts about a 45.1% chance of a dropout within 3 minutes.

## Summary

The Uniform distribution assigns constant density over a bounded interval, so probabilities depend only on interval length. The Gaussian distribution models real-valued quantities concentrated around a mean and extends naturally to random vectors through a covariance matrix. The Exponential distribution models nonnegative waiting times and has the memoryless property.

## Practice Questions

### 1. Exponential Distribution: Time Until Link Dropout

During a UAV mission, the time $T$ until the next communication dropout is modeled as:

$$
T\sim\operatorname{Exponential}(0.15),
$$

where time is measured in minutes.

1. What is the expected time until the next dropout?
2. What is the probability that no dropout occurs during the first 4 minutes?
3. What is the probability that a dropout occurs within 10 minutes?
4. Given that no dropout has occurred during the first 6 minutes, what is the probability that the UAV goes at least 4 more minutes without a dropout?


```{admonition} Solution
:class: dropdown
For an exponential random variable with rate $\lambda=0.15$:

$$
\mathbb{E}[T]=\frac{1}{\lambda}=\frac{1}{0.15}\approx 6.67 \text{ minutes}.
$$

The probability of no dropout during the first 4 minutes is:

$$
P(T>4)=e^{-0.15(4)}=e^{-0.6}\approx 0.5488.
$$

The probability of a dropout within 10 minutes is:

$$
P(T\leq 10)=1-e^{-0.15(10)}
=1-e^{-1.5}
\approx 0.7769.
$$

Using the memoryless property:

$$
P(T>10\mid T>6)=P(T>4)=e^{-0.6}\approx 0.5488.
$$

So, even after 6 dropout-free minutes, the probability of going at least 4 more minutes without a dropout is still about 54.9%.
```

### 2. Gaussian Distribution: Cross-Track Error

A UAV's cross-track error $X$, measured in meters, is modeled as:

$$
X\sim\mathcal{N}(0,1.5^2).
$$

1. What is the probability that the UAV stays within 3 meters of the planned path?
2. What is the probability that the UAV is more than 2 meters to the right of the planned path?
3. Find the interval centered at zero that contains approximately 95% of the cross-track error.
4. If the safety corridor is $[-2.5,2.5]$ meters, what is the probability that the UAV leaves the corridor?


```{admonition} Solution
:class: dropdown
Standardize using:

$$
Z=\frac{X-\mu}{\sigma}=\frac{X}{1.5}.
$$

For the probability of staying within 3 meters:

$$
P(|X|\leq 3)=P\left(|Z|\leq \frac{3}{1.5}\right)
=P(|Z|\leq 2).
$$

Using the standard normal CDF $\Phi$:

$$
P(|Z|\leq 2)=\Phi(2)-\Phi(-2)
=2\Phi(2)-1
\approx 0.9545.
$$

The probability of being more than 2 meters to the right is:

$$
P(X>2)=P\left(Z>\frac{2}{1.5}\right)
=P(Z>1.333)
=1-\Phi(1.333)
\approx 0.0912.
$$

An interval centered at zero containing approximately 95% of the error is:

$$
0\pm 1.96(1.5),
$$

so:

$$
[-2.94,2.94]\text{ meters}.
$$

For the safety corridor:

$$
P(|X|>2.5)
=2P(X>2.5)
=2P\left(Z>\frac{2.5}{1.5}\right).
$$

Since $2.5/1.5\approx 1.667$:

$$
P(|X|>2.5)
=2(1-\Phi(1.667))
\approx 0.0956.
$$

So the UAV leaves the corridor with probability about 9.6%.
```

## References

- [1] Christopher M. Bishop, *Pattern Recognition and Machine Learning*, Springer, 2006. Book site: <https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/>
- [2] Christopher M. Bishop and Hugh Bishop, *Deep Learning: Foundations and Concepts*, Springer, 2023. Book site: <https://www.bishopbook.com/>
- [3] Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction*, MIT Press, 2022. Book site: <https://probml.github.io/pml-book/book1.html>
