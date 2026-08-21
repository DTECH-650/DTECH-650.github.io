# Lecture 4 - Discrete Random Variables

Lecture 3 described probabilities of events and showed how to combine and condition them. We now use random variables to assign numerical values to outcomes and probability mass functions to describe how probability is distributed across those values.

This lecture develops discrete random variables, common discrete distributions, independence and conditional independence, expectation, and variance. It also distinguishes the probability of data under a fixed model from the likelihood of competing models after data have been observed.

## Discrete Random Variables

A **random variable** is a function that assigns a numerical value to each outcome of a random experiment. A **discrete random variable** takes values in a finite or countably infinite set.

Examples:

- $X =$ number of heads in 3 coin flips, so $X \in \{0,1,2,3\}$.
- $Y =$ result of a die roll, so $Y \in \{1,2,3,4,5,6\}$.
- $N =$ number of emails received in an hour, so $N \in \{0,1,2,\ldots\}$.

For a discrete random variable $X$, the **probability mass function** or **PMF** is:

$$
p(x) = P(X=x)
$$

The PMF must satisfy:

$$
p(x) \geq 0
\qquad \text{and} \qquad
\sum_x p(x) = 1
$$

```{figure} figures/binomial-pmf.png
---
name: binomial-pmf
alt: Bar chart of a binomial probability mass function for n equals 5 and p equals 0.4.
---
A PMF assigns probability to each possible value of a discrete random variable.
```

For discrete random variables, conditional probability is given by:

$$
P(X=x \mid Y=y)=\frac{P(X=x, Y=y)}{P(Y=y)}
\qquad \text{if } P(Y=y)>0
$$

Example: roll two fair dice. Let:

- $X =$ value of the first die.
- $S =$ sum of both dice.

What is $P(X=4 \mid S=7)$?

The outcomes with sum 7 are:

$$
(1,6),(2,5),(3,4),(4,3),(5,2),(6,1)
$$

There are 6 equally likely outcomes under the condition $S=7$. Only one has $X=4$, namely $(4,3)$. Therefore:

$$
P(X=4 \mid S=7)=\frac16
$$

Now ask a different question: what is $P(X \geq 4 \mid S=7)$?

The favorable outcomes are:

$$
(4,3),(5,2),(6,1)
$$

So:

$$
P(X \geq 4 \mid S=7)=\frac{3}{6}=\frac12
$$

## Common Types of Discrete Random Variables

Several discrete distributions appear so often that they are worth recognizing.

| Distribution | Values | Typical use | PMF |
| --- | --- | --- | --- |
| Bernoulli$(p)$ | $0,1$ | One yes/no trial | $P(X=1)=p$, $P(X=0)=1-p$ |
| Binomial$(n,p)$ | $0,1,\ldots,n$ | Number of successes in $n$ independent Bernoulli trials | $P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}$ |
| Geometric$(p)$ | $1,2,\ldots$ | Trial number of the first success | $P(X=k)=(1-p)^{k-1}p$ |
| Poisson$(\lambda)$ | $0,1,2,\ldots$ | Counts in a fixed time or space interval | $P(X=k)=e^{-\lambda}\lambda^k/k!$ |
| Discrete uniform | finite set | Equally likely outcomes | $P(X=x)=1/n$ for $n$ outcomes |

The distribution you choose should reflect the data-generating process. For example, a Binomial model assumes a fixed number of trials, two outcomes per trial, constant success probability, and independence.

## Example 1: Defective Parts

A factory produces parts. Each part is defective with probability $p=0.02$, independently of the others. If we inspect $n=10$ parts, let:

$$
X = \text{number of defective parts}
$$

Then:

$$
X \sim \text{Binomial}(10, 0.02)
$$

What is the probability of finding exactly one defective part?

$$
P(X=1)=\binom{10}{1}(0.02)^1(0.98)^9
$$

$$
P(X=1)=10(0.02)(0.98)^9 \approx 0.1667
$$

So there is about a 16.7% chance of exactly one defective part.

What is the probability of at least one defective part?

It is easier to use the complement:

$$
P(X \geq 1) = 1 - P(X=0)
$$

$$
P(X \geq 1) = 1 - \binom{10}{0}(0.02)^0(0.98)^{10}
$$

$$
P(X \geq 1) = 1 - 0.98^{10} \approx 0.1829
$$

So there is about an 18.3% chance of at least one defective part.

## Example 2: Help Desk Tickets

Suppose the number of support tickets arriving in an hour is modeled as:

$$
X \sim \text{Poisson}(3)
$$

Here $\lambda=3$ means the expected number of tickets per hour is 3.

What is the probability of exactly 5 tickets?

$$
P(X=5)=e^{-3}\frac{3^5}{5!}
$$

$$
P(X=5)=e^{-3}\frac{243}{120}\approx 0.1008
$$

What is the probability of at most 2 tickets?

$$
P(X \leq 2)=P(X=0)+P(X=1)+P(X=2)
$$

$$
P(X \leq 2)=e^{-3}\frac{3^0}{0!}+e^{-3}\frac{3^1}{1!}+e^{-3}\frac{3^2}{2!}
$$

$$
P(X \leq 2)=e^{-3}(1+3+4.5)\approx 0.4232
$$

So the probability of at most 2 tickets in an hour is about 42.3%.

## Independence and Conditional Independence

Independence describes when knowing one outcome provides no information about another. It is an assumption behind many probability models, including the independent Bernoulli trials used to construct a Binomial random variable.

### Independence

Two events $A$ and $B$ are **independent** if

$$
P(A \cap B)=P(A)P(B).
$$

When the relevant conditional probabilities are defined, this is equivalent to

$$
P(A \mid B)=P(A)
\qquad \text{and} \qquad
P(B \mid A)=P(B).
$$

In words, learning that one event occurred does not change the probability of the other.

Two discrete random variables $X$ and $Y$ are independent, written

$$
X \perp\!\!\!\perp Y,
$$

 if their joint PMF factors into the product of their individual PMFs for **every** pair of values $x$ and $y$:

$$
P(X=x,Y=y)=P(X=x)P(Y=y).
$$

For example, let $X$ and $Y$ be the results of two independent rolls of a fair six-sided die. For any $x,y \in \{1,2,3,4,5,6\}$,

$$
P(X=x,Y=y)=\frac{1}{36}
=\frac{1}{6}\cdot\frac{1}{6}
=P(X=x)P(Y=y).
$$

Knowing the result of the first roll therefore does not change the distribution of the second roll.

```{admonition} Independence is not mutual exclusivity
:class: important
Mutually exclusive events cannot occur together. If two mutually exclusive events both have positive probability, then they are **not** independent: observing one tells us that the other did not occur. Independence instead means that observing one event does not change the probability of the other.
```

### Conditional Independence

Sometimes two random variables are dependent overall but become independent once a third variable is known. Discrete random variables $X$ and $Y$ are **conditionally independent given** $Z$, written

$$
X \perp\!\!\!\perp Y \mid Z,
$$

if, for every $x$, $y$, and $z$ with $P(Z=z)>0$,

$$
P(X=x,Y=y \mid Z=z)
=P(X=x \mid Z=z)P(Y=y \mid Z=z).
$$

Under conditional independence, once $Z$ is known, learning $X$ provides no additional information about $Y$, and vice versa. This does **not** necessarily mean that $X$ and $Y$ are independent when $Z$ is unknown.

#### Example: Two UAV Sensors with a Common Cause

Let three binary random variables represent an obstacle-detection problem:

- $O=1$ if an obstacle is present and $O=0$ otherwise.
- $C=1$ if the camera raises an alert.
- $L=1$ if the lidar raises an alert.

Suppose the camera and lidar have separate measurement noise. **Once we know whether an obstacle is present, assume their alerts are independent**, with

| Condition | Camera alert probability | Lidar alert probability |
| --- | ---: | ---: |
| Obstacle present ($O=1$) | $P(C=1\mid O=1)=0.90$ | $P(L=1\mid O=1)=0.80$ |
| No obstacle ($O=0$) | $P(C=1\mid O=0)=0.05$ | $P(L=1\mid O=0)=0.10$ |

Conditional independence allows us to multiply within each condition:

$$
P(C=1,L=1\mid O=1)=(0.90)(0.80)=0.72,
$$

and

$$
P(C=1,L=1\mid O=0)=(0.05)(0.10)=0.005.
$$

Now suppose $P(O=1)=0.10$. Using the law of total probability, without knowing $O$, the marginal alert probabilities are

$$
P(C=1)=(0.10)(0.90)+(0.90)(0.05)=0.135,
$$

$$
P(L=1)=(0.10)(0.80)+(0.90)(0.10)=0.17,
$$

while the probability that both sensors alert is

$$
P(C=1,L=1)=(0.10)(0.72)+(0.90)(0.005)=0.0765.
$$

Because

$$
0.0765 \neq (0.135)(0.17)=0.02295,
$$

$C$ and $L$ are not independent when the obstacle state is unknown. They are associated through their common cause, $O$, but they are conditionally independent once $O$ is known.

## Expectation and Variance

The **expectation** of a discrete random variable is its probability-weighted average:

$$
E[X] = \sum_x xP(X=x)
$$

Expectation is not necessarily a value that $X$ can actually take. A fair die has:

$$
E[X] = 1\cdot\frac16 + 2\cdot\frac16 + \cdots + 6\cdot\frac16 = 3.5
$$

No die roll is 3.5, but 3.5 is the long-run average roll.

The **variance** measures spread around the expectation:

$$
\operatorname{Var}(X) = E[(X-E[X])^2]
$$

For a discrete random variable:

$$
\operatorname{Var}(X) = \sum_x (x-\mu)^2P(X=x)
\qquad \text{where } \mu=E[X]
$$

A useful equivalent formula is:

$$
\operatorname{Var}(X)=E[X^2]-(E[X])^2
$$

where:

$$
E[X^2]=\sum_x x^2P(X=x)
$$

For common distributions:

| Distribution | $E[X]$ | $\operatorname{Var}(X)$ |
| --- | --- | --- |
| Bernoulli$(p)$ | $p$ | $p(1-p)$ |
| Binomial$(n,p)$ | $np$ | $np(1-p)$ |
| Geometric$(p)$ | $1/p$ | $(1-p)/p^2$ |
| Poisson$(\lambda)$ | $\lambda$ | $\lambda$ |


## Probability and Likelihood

Probability and likelihood use the same mathematical model but answer different questions. Suppose a PMF is written as $p(x\mid\theta)$, where $x$ is a possible observation and $\theta$ is a model parameter for a given distribution.

| Concept | Held fixed | Allowed to vary | Question |
| --- | --- | --- | --- |
| **Probability** $p(x\mid\theta)$ | Model parameter $\theta$ | Possible data $x$ | If this model is fixed, how probable is each possible outcome? |
| **Likelihood** $L(\theta;x)=p(x\mid\theta)$ | Observed data $x$ | Candidate parameter $\theta$ | Given the data we observed, which parameter values are better supported? |

As a probability distribution over $x$, the PMF must satisfy

$$
\sum_x p(x\mid\theta)=1
$$

for each fixed value of $\theta$. A likelihood function is viewed as a function of $\theta$ and does not generally sum or integrate to 1 over the possible parameter values.

For example, suppose $X$ is the number of successful telemetry transmissions in 10 independent attempts, each with success probability $p$. Before observing $X$, if $p=0.8$ is fixed, the probability of exactly 8 successes is

$$
P(X=8\mid p=0.8)
=\binom{10}{8}(0.8)^8(0.2)^2
\approx 0.302.
$$

After observing $X=8$, the same Binomial expression becomes a likelihood function for the unknown success probability:

$$
L(p;X=8)=\binom{10}{8}p^8(1-p)^2.
$$

Now the observation 8 is fixed and different values of $p$ are compared. The likelihood is largest at $p=0.8$, which is the maximum likelihood estimate in this example.

```{admonition} Likelihood is not a posterior probability
:class: important
$L(p;X=8)$ is not $P(p\mid X=8)$. To construct a probability distribution over the unknown parameter $p$, we must specify a prior distribution and apply Bayes' theorem. More on this later.
```

## Summary

A discrete random variable maps outcomes to a finite or countably infinite set of numbers, and its PMF assigns probability to those values. Bernoulli, Binomial, Geometric, Poisson, and discrete uniform distributions model different data-generating processes. Probability treats the model as fixed and varies the possible data; likelihood treats the observed data as fixed and compares model parameters. Independence means that one variable supplies no information about another, while conditional independence means this holds after a third variable is known. Expectation and variance summarize the center and spread of a distribution.

## Practice Questions and Solutions

### 1. Discrete Random Variable: Telemetry Packets

A UAV transmits five telemetry packets during a maneuver. Each packet is received successfully with probability $0.80$, independently of the other packets. Let $X$ be the number of packets received successfully.

1. What distribution does $X$ follow?
2. What is the probability that at least four packets are received successfully?
3. Find $E[X]$ and $\operatorname{Var}(X)$.

```{admonition} Solution
:class: dropdown

There is a fixed number of independent trials, and each trial has the same success probability. Therefore:

$$
X\sim\operatorname{Binomial}(5,0.80).
$$

The probability of receiving at least four packets is:

$$
P(X\geq 4)=P(X=4)+P(X=5).
$$

Using the Binomial PMF:

$$
\begin{aligned}
P(X\geq 4)
&=\binom{5}{4}(0.8)^4(0.2)+\binom{5}{5}(0.8)^5 \\
&=5(0.4096)(0.2)+0.32768 \\
&=0.73728.
\end{aligned}
$$

Thus, the probability of receiving at least four packets is approximately 73.7%.

For a Binomial random variable:

$$
E[X]=np=(5)(0.8)=4
$$

and:

$$
\operatorname{Var}(X)=np(1-p)=(5)(0.8)(0.2)=0.8.
$$ -->
```

### 2. Discrete Random Variable: Wind Gusts

The number of unexpected strong wind gusts encountered during a UAV flight is modeled as a Poisson random variable with an average of two gusts per flight. Let $Y$ be the number of gusts.

1. What is the probability that the UAV encounters at most one gust?
2. What are the expected value and variance of $Y$?

```{admonition} Solution
:class: dropdown
The model is:

$$
Y\sim\operatorname{Poisson}(2).
$$

The probability of at most one gust is:

$$
\begin{aligned}
P(Y\leq 1)
&=P(Y=0)+P(Y=1) \\
&=e^{-2}\frac{2^0}{0!}+e^{-2}\frac{2^1}{1!} \\
&=3e^{-2}\approx 0.4060.
\end{aligned}
$$

Therefore, the probability of encountering at most one gust is approximately 40.6%.

For a Poisson random variable, both the expectation and variance equal $\lambda$:

$$
E[Y]=2, \qquad \operatorname{Var}(Y)=2.
$$ -->
```

### 3. Mean and Variance: GPS Lock Acquisition

Before beginning a mission, a UAV repeatedly scans for a reliable GPS lock. Each scan succeeds with probability $0.75$, independently of previous scans. Let $Z$ be the number of scans required to obtain the first reliable GPS lock, including the successful scan.

1. What distribution does $Z$ follow?
2. Find $E[Z]$ and $\operatorname{Var}(Z)$.
3. Briefly interpret the expected value in the context of the UAV mission.


```{admonition} Solution
:class: dropdown
The UAV repeats independent trials with a constant success probability until the first success. Therefore:

$$
Z\sim\operatorname{Geometric}(0.75).
$$

For a Geometric random variable that counts the number of trials up to and including the first success:

$$
E[Z]=\frac{1}{p}
$$

and:

$$
\operatorname{Var}(Z)=\frac{1-p}{p^2}.
$$

Substituting $p=0.75$ gives:

$$
E[Z]=\frac{1}{0.75}=\frac{4}{3}\approx 1.33
$$

and:

$$
\operatorname{Var}(Z)
=\frac{1-0.75}{(0.75)^2}
=\frac{0.25}{0.5625}
=\frac{4}{9}
\approx 0.44.
$$

Over many mission starts, the UAV will require an average of approximately 1.33 scans to obtain its first reliable GPS lock. This does not mean that a single mission can take 1.33 scans; an individual mission always requires a whole number of scans. -->
```

## References

- Joseph K. Blitzstein and Jessica Hwang, *Introduction to Probability*, 2nd ed., CRC Press, 2019. Book site: <https://projects.iq.harvard.edu/stat110/home>
- OpenStax, *Introductory Statistics*, probability and discrete random variable chapters: <https://openstax.org/details/books/introductory-statistics>
- An exceptional set of lectures on probability and statistics by Steven Brunton of UWash: <https://youtube.com/playlist?list=PLMrJAkhIeNNR3sNYvfgiKgcStwuPSts9V&si=sFO1lSGlWPbBZHzg/>
- Companion code notebook for plotting the discrete probability distributions: [discrete probability plotting notebook](https://github.com/VIP-Team-AIDA3/MiniTutorialCodes/blob/master/mini-tutorial%201%20-%20discrete%20probability.ipynb)
