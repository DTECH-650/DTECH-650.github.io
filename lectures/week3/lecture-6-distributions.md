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

### From Gaussian Integrals to $\Phi$, $\operatorname{erf}$, and $\operatorname{erfc}$

The Gaussian PDF tells us how density is distributed along the real line. To obtain a probability, we must integrate that density over an interval. For example, if $X\sim\mathcal{N}(\mu,\sigma^2)$, then its CDF is

$$
F_X(x)
=P(X\leq x)
=\int_{-\infty}^{x}
\frac{1}{\sigma\sqrt{2\pi}}
\exp\left(-\frac{(u-\mu)^2}{2\sigma^2}\right)du.
$$

It is natural to ask why we do not simply evaluate this integral directly. The difficulty is that the function $e^{-u^2}$, and therefore the Gaussian density, does not have an antiderivative that can be written using elementary functions such as polynomials, exponentials, logarithms, and trigonometric functions. The definite integral still exists and can be approximated numerically, but ordinary integration rules do not produce a convenient closed-form CDF.

There is also no need to solve a different numerical integration problem for every possible $\mu$, $\sigma$, and $x$. Every Gaussian curve has the same basic shape; changing $\mu$ shifts that shape and changing $\sigma$ rescales it. We can remove those two changes of location and scale through an exact change of variables.

#### Why We Standardize

Inside the CDF integral, define

$$
s=\frac{u-\mu}{\sigma}.
$$

Equivalently,

$$
u=\mu+\sigma s,
\qquad
du=\sigma\,ds.
$$

As $u$ moves from $-\infty$ to $x$, the new variable $s$ moves from $-\infty$ to $(x-\mu)/\sigma$. Substituting into the original integral gives

$$
\begin{aligned}
F_X(x)
&=\int_{-\infty}^{(x-\mu)/\sigma}
\frac{1}{\sigma\sqrt{2\pi}}
\exp\left(-\frac{(\sigma s)^2}{2\sigma^2}\right)
\sigma\,ds\\
&=\frac{1}{\sqrt{2\pi}}
\int_{-\infty}^{(x-\mu)/\sigma}e^{-s^2/2}\,ds.
\end{aligned}
$$

The factors of $\sigma$ cancel, and neither $\mu$ nor $\sigma$ remains inside the density. This is the reason for standardizing: it converts a probability under **any** univariate Gaussian into a probability under one reference distribution, the standard Gaussian.

At the random-variable level, the same transformation is written as

$$
Z=\frac{X-\mu}{\sigma},
\qquad
Z\sim\mathcal{N}(0,1).
$$

Subtracting $\mu$ centers the variable at zero. Dividing by $\sigma$ expresses distance in units of standard deviations. Thus the standardized threshold

$$
z=\frac{x-\mu}{\sigma}
$$

tells us how many standard deviations $x$ lies above or below the mean.

#### The Standard Gaussian CDF

Because every Gaussian probability can now be reduced to the same integral, that integral is given its own name:

$$
\Phi(z)
=P(Z\leq z)
=\frac{1}{\sqrt{2\pi}}
\int_{-\infty}^{z}e^{-s^2/2}\,ds.
$$

The original CDF can therefore be written compactly as

$$
\boxed{
F_X(x)=\Phi\left(\frac{x-\mu}{\sigma}\right)
}.
$$

Historically, values of $\Phi$ were looked up in standard Gaussian tables. Software now evaluates the same integral numerically. The symbols $\operatorname{erf}$ and $\operatorname{erfc}$ provide another standard way to represent and compute it.

#### Where the Error Function Comes From

The standard Gaussian density is symmetric about zero, so $\Phi(0)=1/2$. We can separate the CDF at zero:

$$
\Phi(z)
=\frac12+\frac{1}{\sqrt{2\pi}}
\int_0^z e^{-s^2/2}\,ds.
$$

Now set $t=s/\sqrt{2}$, so that $s=\sqrt{2}t$ and $ds=\sqrt{2}\,dt$. Then

$$
\begin{aligned}
\Phi(z)
&=\frac12+\frac{1}{\sqrt{\pi}}
\int_0^{z/\sqrt{2}}e^{-t^2}\,dt.
\end{aligned}
$$

The **error function** is defined by

$$
\operatorname{erf}(a)
=\frac{2}{\sqrt{\pi}}
\int_0^a e^{-t^2}\,dt.
$$

This definition is chosen so that the preceding Gaussian integral becomes

$$
\boxed{
\Phi(z)
=\frac12\left[
1+\operatorname{erf}\left(\frac{z}{\sqrt{2}}\right)
\right]
}.
$$

The unusual name comes from the function's historical use in studying measurement error. For our purposes, it is simply a named special function that software can evaluate. It packages an integral that cannot be expressed using elementary functions.

#### Why the Complementary Error Function Describes a Tail

For an unusually large observation, we usually want the probability to the **right** of a threshold:

$$
P(Z>z)=1-\Phi(z).
$$

The **complementary error function** is defined as

$$
\begin{aligned}
\operatorname{erfc}(a)
&=1-\operatorname{erf}(a)\\
&=\frac{2}{\sqrt{\pi}}
\int_a^\infty e^{-t^2}\,dt.
\end{aligned}
$$

Substituting the error-function expression for $\Phi$ shows why $\operatorname{erfc}$ appears naturally:

$$
\begin{aligned}
P(Z>z)
&=1-\frac12\left[
1+\operatorname{erf}\left(\frac{z}{\sqrt{2}}\right)
\right]\\
&=\frac12\left[
1-\operatorname{erf}\left(\frac{z}{\sqrt{2}}\right)
\right]\\
&=\frac12\operatorname{erfc}\left(\frac{z}{\sqrt{2}}\right).
\end{aligned}
$$

Returning to a general Gaussian variable, standardize the threshold first:

$$
\begin{aligned}
P(X>x)
&=P\left(
\frac{X-\mu}{\sigma}>
\frac{x-\mu}{\sigma}
\right)\\
&=\frac12\operatorname{erfc}\left(
\frac{x-\mu}{\sigma\sqrt{2}}
\right).
\end{aligned}
$$

Thus

$$
\boxed{
P(X>x)
=\frac12\operatorname{erfc}\left(
\frac{x-\mu}{\sigma\sqrt{2}}
\right)
}.
$$

The factor of $1/2$ and the $\sqrt{2}$ in the argument are not arbitrary constants. They arise from matching the standard Gaussian integral, which contains $e^{-s^2/2}$, to the definition of $\operatorname{erfc}$, which contains $e^{-t^2}$.

#### Example: An Unusually High Vibration Reading

Suppose a machine's vibration amplitude in millimeters per second is modeled as

$$
X\sim\mathcal{N}(10,2^2).
$$

We want the probability that a reading exceeds $13$ mm/s. First standardize the threshold:

$$
z=\frac{13-10}{2}=1.5.
$$

This says that $13$ is $1.5$ standard deviations above the mean. Because the question asks for a right-tail probability,

$$
\begin{aligned}
P(X>13)
&=P(Z>1.5)\\
&=\frac12\operatorname{erfc}\left(\frac{1.5}{\sqrt{2}}\right)\\
&=\frac12\operatorname{erfc}(1.0607)\\
&\approx0.0668.
\end{aligned}
$$

Under this model, approximately $6.68\%$ of normal-operation readings exceed the inspection threshold.

Python's standard `math` module provides $\operatorname{erfc}$ directly:

```python
import math

mu = 10.0
sigma = 2.0
threshold = 13.0

z = (threshold - mu) / sigma
upper_tail = 0.5 * math.erfc(z / math.sqrt(2.0))

print(upper_tail)  # 0.06680720126885809
```

#### Other Gaussian Probabilities

The same standardization applies to a left-tail probability. For a threshold $x$, define

$$
z_x=\frac{x-\mu}{\sigma}.
$$

Then

$$
P(X\leq x)
=P\left(\frac{X-\mu}{\sigma}\leq z_x\right)
=P(Z\leq z_x)
=\Phi(z_x).
$$

To write this probability using $\operatorname{erfc}$, recall that $\operatorname{erf}$ is an odd function:

$$
\operatorname{erf}(-a)=-\operatorname{erf}(a).
$$

This follows because its integrand $e^{-t^2}$ is symmetric about zero, while reversing the limits of integration changes the sign. Consequently,

$$
\begin{aligned}
\operatorname{erfc}\left(-\frac{z_x}{\sqrt{2}}\right)
&=1-\operatorname{erf}\left(-\frac{z_x}{\sqrt{2}}\right)\\
&=1+\operatorname{erf}\left(\frac{z_x}{\sqrt{2}}\right).
\end{aligned}
$$

Comparing this with the earlier expression for $\Phi$ gives

$$
\Phi(z_x)
=\frac12\operatorname{erfc}\left(-\frac{z_x}{\sqrt{2}}\right).
$$

Finally, because $-z_x=(\mu-x)/\sigma$,

$$
P(X\leq x)
=\Phi\left(\frac{x-\mu}{\sigma}\right)
=\frac12\operatorname{erfc}\left(
\frac{\mu-x}{\sigma\sqrt{2}}
\right).
$$

The sign is worth checking. If $x<\mu$, then $(\mu-x)/(\sigma\sqrt{2})$ is positive. For a positive argument, $\operatorname{erfc}$ is less than 1, so the left-tail probability is less than $1/2$, exactly as it should be for a threshold below the mean.

For example, if $X\sim\mathcal{N}(10,2^2)$ and $x=8$, then

$$
z_x=\frac{8-10}{2}=-1
$$

and

$$
P(X\leq8)
=\Phi(-1)
=\frac12\operatorname{erfc}\left(\frac{1}{\sqrt{2}}\right)
\approx0.1587.
$$

An interval probability is the difference between two CDF values:

$$
P(a<X\leq b)
=\Phi\left(\frac{b-\mu}{\sigma}\right)
-\Phi\left(\frac{a-\mu}{\sigma}\right).
$$

For a symmetric two-sided tail $k$ standard deviations from the mean,

$$
P(|X-\mu|>k\sigma)
=\operatorname{erfc}\left(\frac{k}{\sqrt{2}}\right).
$$

Because a continuous random variable assigns probability zero to any single point, using $<$ instead of $\leq$, or $>$ instead of $\geq$, does not change these probabilities.

### The Multivariate Gaussian

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

The Uniform distribution assigns constant density over a bounded interval, so probabilities depend only on interval length. The Gaussian distribution models real-valued quantities concentrated around a mean; its probabilities can be evaluated after standardization using $\Phi$, $\operatorname{erf}$, or $\operatorname{erfc}$, and it extends naturally to random vectors through a covariance matrix. The Exponential distribution models nonnegative waiting times and has the memoryless property.

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
