# Lecture 3 - Probability Foundations

Probability is the language we use when outcomes are uncertain but not completely arbitrary. It lets us say more than "maybe." We can quantify how likely an event is, combine pieces of evidence, compare competing explanations, and make decisions before we know exactly what will happen.

In this tutorial, we will focus on discrete probability: settings where the outcomes can be listed, such as coin flips, dice rolls, survey categories, counts of events, or whether a test result is positive or negative.

## Why We Need Probability

Most real-world questions involve uncertainty. Measurements contain error, experiments produce different results, demand changes over time, components can fail, and we often must make decisions without complete information. Probability gives us a consistent language for describing this uncertainty, learning from data, and comparing possible decisions.

Probability is used across many fields:

- **Science and statistics:** Researchers use probability to quantify experimental uncertainty and determine what conclusions are supported by a sample of data.
- **Medicine and public health:** Diagnostic tests are imperfect, treatment outcomes vary, and risks must be compared when making clinical or policy decisions.
- **Engineering and reliability:** Engineers model measurement noise, manufacturing variation, component failures, communication errors, and other sources of uncertainty.
- **Machine learning:** A model can assign probabilities to possible predictions instead of returning only a yes-or-no answer, allowing uncertainty in the prediction to be considered.
- **Optimization and decision-making:** A decision-maker can balance cost, reward, and risk by accounting for several possible future outcomes rather than planning for only one.

### Example: Choosing a Travel Route

Suppose a traveler must choose between two routes to a destination:

| Route | Normal travel time | Probability of a major delay |
| --- | ---: | ---: |
| A: short route | 10 minutes | 0.20 |
| B: reliable route | 14 minutes | 0.05 |

If a major delay adds 30 minutes, a simple expected-time model is:

$$
E[\text{time}] = \text{normal travel time} + P(\text{delay})(30).
$$

For Route A:

$$
E[T_A] = 10 + (0.20)(30) = 16 \text{ minutes}.
$$

For Route B:

$$
E[T_B] = 14 + (0.05)(30) = 15.5 \text{ minutes}.
$$

Route A is faster when no delay occurs, but Route B has the lower expected travel time after accounting for uncertainty. A more complete decision could also consider cost, safety, deadlines, or how strongly the traveler wishes to avoid being late. Probability turns uncertain possibilities into quantities that can be compared systematically.

## Events, Sample Spaces, and Venn Diagrams

A **sample space**, usually written $\Omega$, is the set of all possible outcomes of an experiment. An **event** is a subset of the sample space.

For example, consider one roll of a fair six-sided die. The sample space is

$$
\Omega = \{1,2,3,4,5,6\},
$$

because these are all the possible outcomes. We can define events by grouping outcomes of interest:

- $A =$ "the roll is even" $= \{2,4,6\}$
- $B =$ "the roll is at least 4" $= \{4,5,6\}$

The individual numbers are outcomes, while $A$ and $B$ are events because each is a subset of $\Omega$. For instance, rolling a 2 means that event $A$ occurs, while rolling a 5 means that event $B$ occurs.

### Union and Intersection of Events

We can combine two events, $A$ and $B$, in two fundamental ways:

- The **intersection**, written $A \cap B$, contains the outcomes that belong to both $A$ **and** $B$:

  $$
  A \cap B = \{\omega \in \Omega : \omega \in A \text{ and } \omega \in B\}.
  $$

  In a Venn diagram, the intersection is the overlapping region of the two circles.

- The **union**, written $A \cup B$, contains the outcomes that belong to $A$, to $B$, or to both:

  $$
  A \cup B = \{\omega \in \Omega : \omega \in A \text{ or } \omega \in B\}.
  $$

  Here, **or is inclusive**: an outcome in both events is still part of the union. In a Venn diagram, the union is the entire region covered by either circle, including their overlap.

Using the die-roll events defined above:

- $A \cap B = \{4,6\}$ because 4 and 6 are both even and at least 4.
- $A \cup B = \{2,4,5,6\}$ because each of these outcomes is even, at least 4, or both.

```{figure} figures/venn-events.png
---
name: venn-events
alt: Venn diagram showing events A and B as overlapping subsets inside a sample space.
---
Events are subsets of the sample space. The overlap is the intersection $A \cap B$; the full area covered by the two circles is the union $A \cup B$.
```

For any event $A$, its probability satisfies:

$$
0 \leq P(A) \leq 1
$$

The impossible event has probability 0, the entire sample space has probability 1, and larger probabilities mean greater likelihood.

## The Frequentist Approach

The **frequentist** interpretation treats probability as a long-run relative frequency. If we repeat the same experiment many times under the same conditions, the probability of an event is the proportion of times that event occurs in the long run.

For example, if a coin is fair, the frequentist interpretation says:

$$
P(\text{heads}) = 0.5
$$

because the fraction of heads should approach $0.5$ as the number of tosses becomes very large.

This interpretation is useful for repeatable processes: coin flips, manufacturing defects, randomized experiments, survey sampling, A/B tests, and simulations. It is less straightforward for one-time events, such as "this exact bridge will fail this year" or "this candidate will win this election," because the event is not literally repeated under identical conditions.

<!-- ## Bertrand's Paradox: Why "Random" Must Be Precise

Bertrand's paradox shows that probability questions can be ambiguous when the random experiment is not specified carefully. The classic question asks:

> If a chord is chosen at random in a circle, what is the probability that the chord is longer than a side of an inscribed equilateral triangle?

Different reasonable methods for choosing a "random chord" produce different answers. For example, choosing two random endpoints on the circumference is not the same procedure as choosing a random midpoint inside the circle.

```{figure} figures/bertrand-paradox.png
---
name: bertrand-paradox
alt: Circle with an inscribed equilateral triangle and example chords.
---
Bertrand's paradox is a warning: before calculating a probability, define the random experiment precisely.
```

The lesson is not that probability is broken. The lesson is that a probability model must state what is random and how the random outcome is generated. -->

## Discrete Probability Basics

A probability model is **discrete** when its possible outcomes form a finite or countably infinite set. Examples include the number of GPS satellites visible to a UAV, the number of obstacles detected during a flight, or a flight-mode status selected from a fixed list.

Suppose a UAV classifies the wind condition during the next control interval as one of three outcomes:

$$
\Omega = \{\text{calm},\text{moderate},\text{strong}\}.
$$

Based on recent measurements, assume the probabilities are:

| Outcome $x$ | $P(X=x)$ |
| --- | ---: |
| Calm | $0.50$ |
| Moderate | $0.35$ |
| Strong | $0.15$ |

Each probability must be nonnegative, and the probabilities of all possible outcomes must sum to 1:

$$
0.50+0.35+0.15=1.
$$

For a discrete outcome, the probability of an event is found by adding the probabilities of the outcomes belonging to that event. Let $A$ be the event that the wind is not calm:

$$
A=\{\text{moderate},\text{strong}\}.
$$

Then:

$$
P(A)=P(X=\text{moderate})+P(X=\text{strong})
=0.35+0.15=0.50.
$$

The **complement** $A^c$ contains every outcome not in $A$. Therefore:

$$
P(A^c)=1-P(A).
$$

In this example, $A^c$ is the event that the wind is calm, so $P(A^c)=1-0.50=0.50$.

When all outcomes are equally likely, probability can also be computed by counting:

$$
P(A)=\frac{\text{number of outcomes in }A}
{\text{total number of outcomes}}.
$$

This counting shortcut applies to a fair die or a well-shuffled deck, but it would not apply to the wind example because its three outcomes have different probabilities. Once additional information is observed, such as a strong-wind warning from a nearby weather station, we use conditional probability to update these probabilities.


## Joint, Marginal, and Conditional Probability

A **joint probability** describes the probability that two events occur together. For events $A$ and $B$, their joint probability is

$$
P(A,B)=P(A\cap B).
$$

As an example, let $O$ be the event that an obstacle is present in a UAV's path and let $A$ be the event that its detector raises an alert. A joint probability table might be

| | Alert ($A$) | No alert ($A^c$) | Marginal probability |
| --- | ---: | ---: | ---: |
| Obstacle ($O$) | $0.090$ | $0.010$ | $0.100$ |
| No obstacle ($O^c$) | $0.045$ | $0.855$ | $0.900$ |
| Marginal probability | $0.135$ | $0.865$ | $1.000$ |

Each interior cell is a joint probability. For example,

$$
P(A\cap O)=0.090.
$$

A **marginal probability** describes one event without fixing the value of the other. We obtain it by summing over the possibilities for the other event, a process called **marginalization**. For example,

$$
P(A)=P(A\cap O)+P(A\cap O^c)
=0.090+0.045=0.135.
$$

More generally, if $B_1,\ldots,B_k$ are mutually exclusive and cover the sample space, then

$$
P(A)=\sum_{i=1}^{k}P(A\cap B_i).
$$

For two discrete quantities $X$ and $Y$, the same marginalization idea is written

$$
P(X=x)=\sum_y P(X=x,Y=y).
$$

A **conditional probability** asks how a probability changes after we restrict attention to a condition:

$$
P(A \mid B)=\frac{P(A \cap B)}{P(B)}
\qquad \text{if } P(B)>0.
$$

In the obstacle example, conditioning on an obstacle restricts attention to the first row of the table:

$$
P(A\mid O)=\frac{P(A\cap O)}{P(O)}
=\frac{0.090}{0.100}=0.90.
$$

Thus, joint probability asks about events together, marginal probability sums out information we are not specifying, and conditional probability updates after information is known.

## Rules of Probability

Two rules appear constantly: the sum rule and the product rule.

### Sum Rule

For any two events $A$ and $B$:

$$
P(A \cup B) = P(A) + P(B) - P(A \cap B)
$$

We subtract $P(A \cap B)$ because the overlap gets counted twice when we add $P(A)$ and $P(B)$.

If $A$ and $B$ are **mutually exclusive**, meaning they cannot both happen, then $P(A \cap B)=0$ and:

$$
P(A \cup B) = P(A) + P(B)
$$

### Product Rule

The product rule connects joint probability and conditional probability:

$$
P(A \cap B) = P(A \mid B)P(B)
$$

Equivalently:

$$
P(A \cap B) = P(B \mid A)P(A)
$$

## Bayes' Theorem

Bayes' theorem is a way to reverse a conditional probability. It follows directly from the product rule:

$$
P(A \mid B) = \frac{P(B \mid A)P(A)}{P(B)}
$$

The pieces have useful names:

- $P(A)$ is the **prior** probability of $A$.
- $P(B \mid A)$ is the **likelihood** of observing evidence $B$ if $A$ is true.
- $P(B)$ is the **evidence** or normalizing probability.
- $P(A \mid B)$ is the **posterior** probability of $A$ after observing $B$.

```{figure} figures/bayes-tree.png
---
name: bayes-tree
alt: Tree diagram showing disease status and positive or negative test results.
---
A probability tree helps track joint probabilities. Each full path multiplies the probabilities along its branches.
```

Example: suppose a disease affects 1% of a population. A test is positive 95% of the time when a person has the disease and positive 5% of the time when a person does not have it. What is $P(D \mid +)$, the probability a person has the disease given a positive test?

$$
P(D \mid +) =
\frac{P(+ \mid D)P(D)}
{P(+ \mid D)P(D) + P(+ \mid D^c)P(D^c)}
$$

Substitute:

$$
P(D \mid +) =
\frac{0.95 \cdot 0.01}
{0.95 \cdot 0.01 + 0.05 \cdot 0.99}
= \frac{0.0095}{0.059}
\approx 0.161
$$

Even with a positive test, the probability is about 16.1%, because the disease is rare and false positives matter.

## Law of Total Probability

The **law of total probability** combines conditional probabilities from all possible cases to find an unconditional, or marginal, probability. If $A_1,A_2,\ldots,A_k$ are mutually exclusive events that cover the sample space, then for any event $B$,

$$
P(B)=\sum_{i=1}^{k}P(B\cap A_i)
=\sum_{i=1}^{k}P(B\mid A_i)P(A_i).
$$

The first equality marginalizes over the possible cases $A_i$. The second uses the product rule on each joint probability. For the common situation in which $A$ either occurs or does not occur, the formula becomes

$$
P(B)=P(B\mid A)P(A)+P(B\mid A^c)P(A^c).
$$

In the medical-test example above, a positive result can occur either when the person has the disease or when the person does not have it. Therefore,

$$
\begin{aligned}
P(+)
&=P(+\mid D)P(D)+P(+\mid D^c)P(D^c)\\
&=(0.95)(0.01)+(0.05)(0.99)\\
&=0.059.
\end{aligned}
$$

This value is the evidence in the denominator of Bayes' theorem. The law of total probability first combines all ways the observed evidence could occur; Bayes' theorem then uses that total to determine how much of the evidence is attributable to a particular case.

## Summary

Probability starts with a sample space and events. Venn diagrams help visualize unions and intersections, while joint probabilities describe events occurring together. Marginalization sums out unspecified outcomes, and conditional probability restricts the sample space after information is known. The sum and product rules connect compound probabilities. Bayes' theorem updates a prior belief using evidence, while the law of total probability computes the overall probability of that evidence across all possible cases.

## Practice Problems and Solutions

### 1. Probability and Conditional Probability: Obstacle Detection

During a UAV flight, the probability that an obstacle is present in the planned path is $0.10$. The onboard detector raises an alert with probability $0.90$ when an obstacle is present. When no obstacle is present, it still raises a false alert with probability $0.05$.

1. What is the probability that the detector raises an alert?
2. If the detector raises an alert, what is the probability that an obstacle is actually present?

```{admonition} Solution
:class: dropdown
Let $O$ be the event that an obstacle is present and $A$ be the event that the detector raises an alert. We are given:

$$
P(O)=0.10, \qquad P(A\mid O)=0.90, \qquad P(A\mid O^c)=0.05.
$$

The probability of an alert is found using the law of total probability:

$$
\begin{aligned}
P(A)
&=P(A\mid O)P(O)+P(A\mid O^c)P(O^c) \\
&=(0.90)(0.10)+(0.05)(0.90) \\
&=0.09+0.045=0.135.
\end{aligned}
$$

Therefore, the detector raises an alert on 13.5% of flights under these assumptions.

Using Bayes' rule:

$$
\begin{aligned}
P(O\mid A)
&=\frac{P(A\mid O)P(O)}{P(A)} \\
&=\frac{(0.90)(0.10)}{0.135} \\
&=\frac{2}{3}\approx 0.667.
\end{aligned}
$$

Given an alert, the probability that an obstacle is actually present is approximately 66.7%.
```

### 2. Union, Intersection, and Complements

In a group of students, 60% have taken a Python course, 45% have taken a statistics course, and 25% have taken both. Let $A$ denote the event that a randomly selected student has taken Python and $B$ the event that the student has taken statistics.

1. What is the probability that the student has taken at least one of the two courses?
2. What is the probability that the student has taken neither course?
3. What is the probability that the student has taken exactly one of the courses?

```{admonition} Solution
:class: dropdown
Use the sum rule for the union:

$$
\begin{aligned}
P(A\cup B)
&=P(A)+P(B)-P(A\cap B)\\
&=0.60+0.45-0.25\\
&=0.80.
\end{aligned}
$$

Thus, 80% have taken at least one course. The event that a student has taken neither course is the complement of the union:

$$
P((A\cup B)^c)=1-P(A\cup B)=1-0.80=0.20.
$$

For exactly one course, first remove the overlap from each event:

$$
P(A\text{ only})=0.60-0.25=0.35,
$$

$$
P(B\text{ only})=0.45-0.25=0.20.
$$

Therefore,

$$
P(\text{exactly one})=0.35+0.20=0.55.
$$
```

### 3. Marginal and Conditional Probabilities from a Joint Table

The following joint probability table describes whether it rains during a commute and whether a traveler arrives late:

| | Late ($L$) | On time ($L^c$) | Total |
| --- | ---: | ---: | ---: |
| Rain ($R$) | $0.18$ | $0.12$ | $0.30$ |
| No rain ($R^c$) | $0.14$ | $0.56$ | $0.70$ |
| Total | $0.32$ | $0.68$ | $1.00$ |

Find $P(L)$, $P(L\mid R)$, and $P(R\mid L)$.

```{admonition} Solution
:class: dropdown
The marginal probability of arriving late is found by summing the late column:

$$
P(L)=P(L\cap R)+P(L\cap R^c)=0.18+0.14=0.32.
$$

Conditioning on rain restricts attention to the rain row:

$$
P(L\mid R)
=\frac{P(L\cap R)}{P(R)}
=\frac{0.18}{0.30}
=0.60.
$$

Conditioning on being late instead restricts attention to the late column:

$$
P(R\mid L)
=\frac{P(R\cap L)}{P(L)}
=\frac{0.18}{0.32}
=0.5625.
$$

The two conditional probabilities are different because they condition on different subsets of the sample space.
```

### 4. Total Probability and Bayes' Theorem

A factory uses three machines to produce components:

| Machine | Share of production | Defect probability |
| --- | ---: | ---: |
| $M_1$ | $0.50$ | $0.01$ |
| $M_2$ | $0.30$ | $0.02$ |
| $M_3$ | $0.20$ | $0.04$ |

1. What is the probability that a randomly selected component is defective?
2. If a component is defective, what is the probability that it was produced by $M_3$?

```{admonition} Solution
:class: dropdown
Let $D$ denote the event that a component is defective. The machines form a partition of the possible sources, so the law of total probability gives

$$
\begin{aligned}
P(D)
&=\sum_{i=1}^{3}P(D\mid M_i)P(M_i)\\
&=(0.01)(0.50)+(0.02)(0.30)+(0.04)(0.20)\\
&=0.005+0.006+0.008\\
&=0.019.
\end{aligned}
$$

Thus, 1.9% of all components are defective. The joint probability that a component is both defective and produced by $M_3$ is

$$
P(D\cap M_3)=P(D\mid M_3)P(M_3)=(0.04)(0.20)=0.008.
$$

Bayes' theorem then gives

$$
\begin{aligned}
P(M_3\mid D)
&=\frac{P(D\mid M_3)P(M_3)}{P(D)}\\
&=\frac{0.008}{0.019}\\
&\approx 0.421.
\end{aligned}
$$

Although $M_3$ produces only 20% of the components, it accounts for about 42.1% of the defective components because its defect rate is higher.
```

## References

- Joseph K. Blitzstein and Jessica Hwang, *Introduction to Probability*, 2nd ed., CRC Press, 2019. Book site: <https://projects.iq.harvard.edu/stat110/home>
- OpenStax, *Introductory Statistics*, probability and discrete random variable chapters: <https://openstax.org/details/books/introductory-statistics>
- Joseph Bertrand, *Calcul des probabilités*, 1889. See overview of Bertrand's chord paradox: <https://en.wikipedia.org/wiki/Bertrand_paradox_(probability)>
- Richard von Mises, frequentist interpretation overview: <https://plato.stanford.edu/entries/probability-interpret/>
- An exceptional set of lectures on probability and statistics by Steven Brunton of UWash: <https://youtube.com/playlist?list=PLMrJAkhIeNNR3sNYvfgiKgcStwuPSts9V&si=sFO1lSGlWPbBZHzg/>
