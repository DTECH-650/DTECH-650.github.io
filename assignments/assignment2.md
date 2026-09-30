# Assignment 2 — Learning, Information, and Uncertainty

This draft covers **Lectures 7, 8, 9, 10, and 11**. It contains **five questions worth 100 points**.

## Purdue Honor Pledge

> “As a Boilermaker pursuing academic excellence, I pledge to be honest and true in all that I do. Accountable together—We are Purdue.”

**When submitting this assignment via Gradescope, we assume that you commit to this honor pledge and submit your own work considering the instructions below.**

## Instructions

- Show the main steps in your reasoning. A correct numerical answer without supporting work may receive only partial credit.
- Unless explicitly stated otherwise, **compute** means compute step by step by hand. Write out algebra, differentiation, and probability calculations rather than using a program or calculator that performs those steps for you. An arithmetic calculator may be used to evaluate numerical expressions, including logarithms and square roots.
- **Only Question 3, part 4 requires code.** Solve all other parts by hand. For the coding part, submit readable, reproducible Python code, the requested figures and numerical results, and your interpretation. Include these in your submission PDF.
- Use natural logarithms for likelihood calculations in Question 1. Use logarithms to base 2 for Question 2, so information quantities are measured in bits.
- You may write your answers on paper or a tablet, or type them using a word processor or LaTeX. Make sure your work is legible and equations are clearly presented.
- Submit the assignment on Gradescope as a PDF, using the link on the course website or in the syllabus.
- **Please link your answers to the questions on Gradescope when submitting your PDF.**

## Questions

### Question 1: Learning a Bernoulli success probability (16 points)

A wireless device transmits packets to a receiver. Record $X_n=1$ if packet $n$ is received successfully and $X_n=0$ otherwise. Under fixed operating conditions, model the packet outcomes as IID Bernoulli random variables with an unknown success probability $p\in[0,1]$. The parameter $p$ is fixed throughout the experiment.

Two independent batches are collected under these same conditions. Batch A has 16 successes in 20 transmissions; Batch B has 18 successes in 30 transmissions. The individual packet outcomes are available, not just the two counts.

1. Likelihood of the observations: Let $N$ be the total number of transmissions and $S=\sum_{n=1}^{N}x_n$ the total number of successes. Starting from the PMF of one observation, write the likelihood of the observed sequence and its log-likelihood for $0<p<1$. Identify the assumption that permits the product factorization. (4 points)

2. Estimation: For $0<S<N$, derive the maximum-likelihood estimator $\widehat p_{\mathrm{ML}}$ by differentiating the log-likelihood. Show that the stationary point is a maximum, and explain why neither boundary gives a larger likelihood in this case. (4 points)

3. Combining the batches: Compute the separate estimates from Batches A and B and the estimate using all 50 transmissions. A colleague proposes taking the unweighted average of the two batch estimates. Compute that average and explain why it differs from the pooled MLE. Express the pooled estimate as a weighted average. (5 points)

4. A boundary estimate: In a separate experiment, all six transmitted packets are received. Find the MLE directly from the likelihood. Does the observed sample establish that failures are impossible? (3 points)

### Question 2: Information theory (15 points)

Use logarithms to base 2, so all information quantities are measured in bits.

1. Let $X\sim\operatorname{Bernoulli}(1/2)$. Compute the surprisal of observing either outcome and the entropy $H(X)$. (4 points)
2. Let $Y=X$. Compute $H(X,Y)$, $H(Y\mid X)$, and $I(X;Y)$. (6 points)
3. The true distribution over two classes is $P=(1/2,1/2)$, while a model reports $Q=(3/4,1/4)$. Compute the cross-entropy $H(P,Q)$ and the KL divergence $D_{\mathrm{KL}}(P\Vert Q)$. You may use $\log_2(3/4)\approx-0.415$ and $\log_2(1/4)=-2$. (5 points)


### Question 3: Sampling distributions and confidence intervals (25 points)

A packaging facility wants to understand uncertainty in estimated average processing times. First consider a small, fixed population to make repeated sampling concrete. Then consider observations from its ongoing operation.

1. **A finite population (6 points).** Four completed jobs have these processing times, in minutes:

   $$
   z_1=2,\qquad z_2=4,\qquad z_3=6,\qquad z_4=8
   $$

   Draw a simple random sample of $N=2$ distinct jobs **without replacement** from this population of size $M=4$.

   - **(a)** Compute the population mean $\mu_M$ and population variance $\sigma_M^2$, using denominator $M$ for the variance.
   - **(b)** List the six equally likely unordered samples and their sample means. Use them to give the probability distribution of $\overline X$, combining samples that have the same mean.
   - **(c)** Compute $\mathbb E[\overline X]$ and $\operatorname{Var}(\overline X)$ directly from this distribution. Verify the variance using the finite-population correction from Lecture 9. Compare it with the variance obtained from two independent draws **with replacement**.

2. **An ongoing process (4 points).** For this part, leave the four-job population behind. Assume processing times from an ongoing operation are IID, with unknown mean $\mu$ and known population standard deviation $\sigma=2$ minutes. A sample of $N=100$ jobs has mean $\overline x=2.30$ minutes. Assume the normal approximation for the sample mean is adequate.

   Compute the standard error of $\overline X$ and an approximate 95% confidence interval for $\mu$. Use $z_{0.975}=1.96$. Explain why the standard error is different from the standard deviation of individual processing times.

3. **Sample size and interpretation (3 points).** How many IID observations would be needed to halve the standard error in part 2? Give the repeated-sampling interpretation of its 95% confidence interval. Explain why it is not an interval intended to contain 95% of individual job times.

4. **Coding: repeated sampling from a skewed population (10 points).** For a simulation of the ongoing process, suppose processing times follow an exponential distribution with rate $\lambda=0.5$ per minute. Its true mean is $\mu=2$ minutes and its variance is $\sigma^2=4$ minutes squared. These values are known in the simulation so that you can check the behavior of the procedure.

   Initialize a NumPy random-number generator with seed `65002`. For each $N\in\{5,30,100\}$, generate $R=10{,}000$ independent datasets of size $N$. Each dataset represents a fresh sample of jobs from the same population.

   - **(a)** Compute the mean of every dataset. Report the empirical mean and standard deviation of the 10,000 sample means, alongside their theoretical values $\mu$ and $\sigma/\sqrt N$ in a table. Use `ddof=0` for the empirical standard deviation. (5 points)
   - **(b)** Plot a normalized histogram of the sample means for each $N$. Overlay the normal density with mean $\mu$ and variance $\sigma^2/N$. Label each plot with $N$ and the units of the horizontal axis. (2 points)
   - **(c)** For each dataset, construct the interval $\overline X\pm1.96\sigma/\sqrt N$, using the known value $\sigma=2$. Report the fraction of these intervals that contain the true mean $\mu=2$. This fraction is the **empirical coverage**. (3 points)
   <!-- - **(d)** In four to six sentences, discuss the shape and spread of the sampling distributions and the observed coverage. Explain why a normal approximation is less convincing for $N=5$, why coverage need not be exactly 0.95, and whether increasing $N$ makes the individual job times Gaussian. (2 points) -->

5. **Sampling assumptions (2 points).** Suppose the facility records only its fastest production line, or records consecutive jobs that experience the same temporary slowdown. Identify the problem in each case. Does collecting many more observations automatically justify the IID confidence interval for the mean across the entire facility?

### Question 4: Developing and evaluating a predictor (25 points)

A team wants to predict hourly energy use in **previously unseen buildings** from temperature and occupancy. Its dataset contains 10 observations from each of 120 buildings, for a total of 1,200 rows. Measurements from the same building may share characteristics. Assume the buildings are representative of the intended population and are independent of one another.

The team compares polynomial regression models with maximum total degree $M\in\{1,2,5\}$. Each model has a fitted coefficient vector $\mathbf w$. You do not need to fit a regression model or write code for this question.

1. Data roles and model choices: Identify the parameters and the hyperparameter. The team reserves 24 complete buildings for the final test set and uses four-fold cross-validation on the remaining 96 buildings, with 24 buildings in each fold. How many buildings and rows are used for training and validation in each fold? Explain why randomly splitting individual rows would not match the intended prediction task. (4 points)

2. Model selection: The following mean squared errors (MSEs) are obtained using the building-based folds. The training column is the average training MSE across the four fits.

| Degree $M$ | Training MSE | Validation fold 1 | Validation fold 2 | Validation fold 3 | Validation fold 4 |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.80 | 6.0 | 6.4 | 5.8 | 6.2 |
| 2 | 1.20 | 2.2 | 2.6 | 2.4 | 2.8 |
| 5 | 0.08 | 2.0 | 4.8 | 5.0 | 4.2 |

Compute the cross-validation MSE for each degree and select a degree. Explain why the training column alone should not determine that choice. Describe what should be refitted after selection, which data should be used, and when the test set should be evaluated. Does its test MSE equal the population error exactly? (4 points)

3. Preprocessing and leakage: Consider two proposed changes to the procedure:

- Standardize each feature using the mean and standard deviation of all 1,200 rows before creating any folds.
- Compare all three degrees on the reserved test set and use whichever gives the smallest test MSE.

Explain the problem with each proposal and describe the correct procedure. As a small numerical example, suppose the training values of one feature in a particular fold are $1,3,5,7$. Compute their mean and preprocessing standard deviation, using denominator 4. Use these quantities to transform a held-out value of 9. Should that held-out value influence the transformation? (6 points)

4. Dimensionality: Each cross-validation fit uses 720 training rows. For an illustration, suppose each feature is divided into 10 intervals and the rows are uniformly distributed among the resulting grid regions. Compute the number of regions and expected rows per region for $D=2$ features and for $D=6$ features. Explain the consequence for methods that need several nearby examples to make a prediction, and state one limitation of this illustration. (4 points)

5. Bias and variance: Consider a separate thought experiment at one fixed input $x$: a particular temperature and occupancy. The true conditional mean energy use is $m(x)=10$ kWh and the variance of a new target around this mean is $\sigma_\varepsilon^2(x)=1$ kWh$^2$.

Across possible training datasets, two learning procedures produce the following predictions at $x$. For this simplified example, **each row has probability $1/4$**, and the table describes the entire distribution of fitted predictions, not just four sampled fits. A new target is independent of the training dataset conditional on $x$.

| Training-dataset outcome | Procedure A prediction (kWh) | Procedure B prediction (kWh) |
|---|---:|---:|
| 1 | 8 | 7 |
| 2 | 9 | 10 |
| 3 | 10 | 13 |
| 4 | 9 | 10 |

For each procedure, compute the average prediction, bias, squared bias, prediction variance, and expected squared error on a new target at $x$. Use the bias–variance decomposition from Lecture 11. Which procedure has lower expected error here, and why is choosing the procedure with the smaller bias alone insufficient? Explain whether this comparison at one input establishes which procedure has lower population error over all inputs. (7 points)

### Question 5: Hypothesis testing, Type I and Type II errors, and power (19 points)

A flight-test team is evaluating an airspeed sensor for systematic positive bias. Let $X_n$ denote the measurement error, in m/s, on test $n$, where a positive value means that the sensor reports an airspeed that is too high.

Assume the measurement errors are IID with unknown population mean $\mu$ and known population standard deviation

$$
\sigma=1.2\text{ m/s}.
$$

The team collects

$$
N=36
$$

independent measurements and wishes to test whether the sensor has a positive mean bias.

Use significance level

$$
\alpha=0.05.
$$

You may use

$$
z_{0.95}\approx1.645
$$

and

$$
\Phi(-0.855)\approx0.196.
$$

1. **Hypotheses and rejection region (5 points).**
   State the null and alternative hypotheses. Explain why this is a one-sided test. Compute the standard error of $\overline X$ and determine the critical value $c$ such that the rejection region can be written as

   $$
   \overline X>c.
   $$

   Explain how the choice $\alpha=0.05$ determines this rejection region.

2. **Type I and Type II errors (4 points).**
   Describe, in the context of the airspeed sensor, what a Type I error and a Type II error mean. Which error probability is controlled directly by the chosen significance level $\alpha$?

3. **Type II error and power (6 points).**
   Suppose the true mean bias is actually

   $$
   \mu_1=0.5\text{ m/s}.
   $$

   Under this alternative, compute

   $$
   \beta(\mu_1)
   =
   P_{\mu_1}(\overline X\le c),
   $$

   and then compute the statistical power

   $$
   1-\beta(\mu_1).
   $$

   Interpret both quantities in the context of the sensor test.

4. **Effect of sample size (4 points).**
   Suppose the team increases the number of independent measurements while keeping $\alpha$, $\sigma$, and the true bias $\mu_1=0.5$ m/s fixed. Explain what happens to

   $$
   \operatorname{SE}(\overline X),
   $$

   the Type II error probability $\beta(\mu_1)$, and the statistical power. Explain why.
