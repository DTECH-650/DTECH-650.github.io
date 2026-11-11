# Lecture 19 - Bayesian Linear Regression

In this lecture we return to our scenario from Lecture 17: predicting electricity use for a building.


:::{admonition} Scenario: Predicting building electricity use from temperature

On a mild day, a building might use around 25 kWh; on a hot day, air conditioning might raise that to 60 kWh. Actual use can vary even on days with the same temperature.

In Lecture 17, we fitted the weights by least squares. Here we ask how much uncertainty remains about the intercept and slope after observing the data.
:::

The guiding question for this lecture is: 

:class: tip 
How can we incorporate prior knowledge in a linear regression model when we estimate the regression weights? 
:::

## Learning outcomes

After completing this lecture, you should be able to:

1. Specify a Bayesian linear regression model using a Gaussian likelihood and Gaussian prior.
2. Explain why this prior is conjugate.
3. Calculate and interpret the posterior mean and covariance. 
4. Explain the connection between a broad prior and the least-squares estimate.
5. Use the marginal posterior of a regression weight to calculate and interpret a credible interval and a threshold probability.

## Notation

As in Lectures 17 and 18, we use $Y$ for the random target and $y$ for an observed value. In statistics, the random target variable $Y$ is also sometimes called the dependent variable. An input is $\mathbf x\in\mathbb R^D$ (also called an independent variable), and the training data are $\mathcal D=\{(\mathbf x_n,y_n)\}_{n=1}^N$. Throughout this lecture, we treat the noise precision $\beta$ as known and fixed. The regression weights $\mathbf w$ are unknown.

## 1. Recap: Regression model from Lecture 17

To build intuition about the Bayesian regression model, let us return to the regression model we had built.

### 1.1 Formulating the regression model


As in the previous model, we specify temperature as the input $X$, and electricity use is our target $Y$.
We write $x_n$ for the observed temperature on day $n$ and $y_n$ for the electricity use measured that day.

We begin with a straight-line first-order regression model because we assume a linear relationship between input and target:

$$
f(x;\mathbf w)=w_0+w_1x,
$$

where $w_0$ is the intercept and $w_1$ is the slope. Under our zero-mean noise model, this function describes the expected electricity use at temperature $x$, given the weights $\mathbf w$.

Just like in Lecture 17, we can also formulate this using **general basis function notation**: Recall, that a basis function is a fixed function that transforms an input into a feature used by the regression model. The model multiplies each feature by a weight and adds the results.

Lets' take a simple straight line model, and choose the basis functions as: 

$$
\phi_0(x)=1,
\qquad
\phi_1(x)=x.
$$

Then, we can write the same straight-line model in vector notation:

$$
\boldsymbol\phi(x)=
\begin{pmatrix}1\\x\end{pmatrix}.
$$

Then

$$
f(x;\mathbf w)
=
\mathbf w^{\mathsf T}\boldsymbol\phi(x)
=
w_0+w_1x.
$$

### 1.2 The Gaussian observation model

In Lecture 17, we fitted a line to predict electricity use from temperature. At a given temperature, the line gives the **expected electricity use**. Actual meter readings can differ even on days with the same temperature.

Suppose day $n$ has temperature $x_n$. For a candidate line with intercept $w_0$ and slope $w_1$, we model that day's electricity use as

$$
Y_n \mid x_n,\mathbf w,\beta
\sim \mathcal N\!\left(
\underbrace{w_0+w_1x_n}_{\text{expected electricity use}},
\underbrace{\beta^{-1}}_{\text{variance}}
\right).
$$

For example, if the line gives an expected electricity use of 40 kWh at $20^\circ\mathrm{C}$, the Gaussian distribution at that temperature is centered on 40 kWh. We assume the same variance, $\beta^{-1}=\sigma^2$, at every temperature.

![Gaussian observation model for electricity use](images/lecture19-conditional-densities.png)

**Figure:** Each sideways curve shows the distribution of meter readings at one temperature for the chosen line. The vertical axis gives electricity use. The horizontal width of each curve shows the density, which is highest at the line.

Now suppose we have recorded the temperatures and meter readings for several days. **With these observations fixed, how well does each candidate line account for them?** For each candidate line, we evaluate the Gaussian density at each observed reading. Assuming independent errors, the product of those densities gives the **likelihood** as a function of the weights. Under this Gaussian model, maximizing the likelihood gives the same intercept and slope as minimizing the sum of squared residuals in Lecture 17.

### 1.3 What changes in the Bayesian approach?

We now combine this likelihood with a prior over the intercept and slope to find which lines remain plausible after observing the data.

The MLE fit in  section 1.2. figives us one intercept and one slope by maximaizing the log-likelihood. Given the same observations, which other intercepts and slopes remain plausible?

Here the unknown is a pair of weights, $\mathbf w=(w_0,w_1)^{\mathsf T}$. Each pair defines one line. We keep the Gaussian observation model from Lecture 17: with the weights fixed, it gives a density for the targets; with the observed targets fixed, the same expression gives the likelihood as a function of the weights. We now add a prior $p(\mathbf w)$ describing which intercepts and slopes we consider plausible together.

Let $\mathbf y$ collect the observed targets and let $\boldsymbol\Phi$ be the design matrix with one row $(1,x_n)$ per observation. Bayes’ rule gives

$$
\boxed{
p(\mathbf w\mid\mathbf y,\boldsymbol\Phi,\beta)
=
\frac{
p(\mathbf y\mid\boldsymbol\Phi,\mathbf w,\beta)\,p(\mathbf w)
}{
p(\mathbf y\mid\boldsymbol\Phi,\beta)
}.
}
$$

The terms have the same roles as in our earlier Bayesian examples:

| Term | Meaning in regression |
|---|---|
| Prior $p(\mathbf w)$ | Initial uncertainty about the intercept and slope |
| Likelihood $p(\mathbf y\mid\boldsymbol\Phi,\mathbf w,\beta)$ | How well candidate weights account for the observed electricity use |
| Posterior $p(\mathbf w\mid\mathbf y,\boldsymbol\Phi,\beta)$ | Updated uncertainty about the weights after observing the data |
| Evidence $p(\mathbf y\mid\boldsymbol\Phi,\beta)$ | Normalizing factor that makes the posterior density integrate to one |

The evidence is obtained by integrating over the possible weight
vectors:

$$
p(\mathbf y\mid\boldsymbol\Phi,\beta)
=
\int
p(\mathbf y\mid\boldsymbol\Phi,\mathbf w,\beta)
p(\mathbf w)\,d\mathbf w.
$$

For our straight-line model, this integration is over both
$w_0$ and $w_1$.

The result of Bayesian inference is therefore a posterior distribution over possible intercepts and slopes. Each weight
vector defines a regression line, so uncertainty about the weights also represents uncertainty about the regression line.



Please note that our model assumes that the same unknown weights describe every observation:

$$
y_n=w_0+w_1x_n+\epsilon_n,
\qquad n=1,\ldots,N.
$$

We do not draw a new intercept and slope for each day. Instead, the temperatures and errors vary across days, while the same underlying regression relationship applies throughout the dataset.



:::{admonition} Summary of sources of uncertainty
:class: tip There are two distinct sources of uncertainty:

- **Observation noise:** electricity use can vary even when the
  temperature and regression weights are specified. This is
  represented by $\epsilon_n$ and its variance $\beta^{-1}$.
- **Uncertainty about the weights:** we do not know the intercept
  and slope exactly. This is represented by the prior and posterior
  distributions over $\mathbf w$.

We will focus on the uncertainty coming from the weights in this tutorial as that is fundamental for the Bayesian regression modeling. 
:::

## 2. Priors on the weights for Bayesian regression

Recall from Lecture 14 our discussion on priors. Our next step is to choose a prior over the weights and calculate
how the observations update it. We make use conjugate priors for our weights. 

### 2.1 From plausible weights to plausible curves

The intercept $w_0$ and slope $w_1$ control the shape of the regression model. If $w_0$ increases, the whole line shifts upward. If $w_1$ increases, the line becomes steeper.
You can think of a prior over the weights as uncertainty about the shape of the curve.

For our straight-line model,

$$
f(x;\mathbf w)=w_0+w_1x,
$$

the intercept $w_0$ determines the conditional mean at $x=0$,
while the slope $w_1$ determines how that mean changes when
temperature increases by one unit.

Different weight vectors therefore describe different relationships.
For example,

$$
\mathbf w^{(1)}
=
\begin{pmatrix}10\\1.5\end{pmatrix},
\qquad
\mathbf w^{(2)}
=
\begin{pmatrix}20\\1.0\end{pmatrix}
$$

give the lines

$$
f(x;\mathbf w^{(1)})=10+1.5x,
\qquad
f(x;\mathbf w^{(2)})=20+x.
$$

Both predict a mean electricity use of 40 kWh at $20^\circ$C,
but they predict different changes as temperature increases.

The figure below shows this: 

![Figure with two regression lines](images/lecture19-regression-lines.png)

Before observing the data, we may consider several such lines
plausible. A **prior distribution over the weights** represents
these assumptions. Each draw of a weight vector from the prior gives one complete
regression line. Drawing several weight vectors therefore produces
several possible lines.

This represents uncertainty about the underlying relationship.
It is separate from the observation noise that causes electricity
use to vary around any one of these lines.


### 2.2 Choosing a Gaussian prior

In this lecture, we choose a multivariate Gaussian prior:

$$
p(\mathbf w)
=
\mathcal N(\mathbf w\mid\mathbf m_0,\mathbf S_0).
$$

Here, $\mathbf m_0$ is the prior mean vector and $\mathbf S_0$
is the prior covariance matrix. The subscript $0$ indicates
that these describe our assumptions before incorporating the
observations in this update.

For the two weights in our straight-line model,

$$
\mathbf m_0
=
\begin{pmatrix}
m_{0,0}\\
m_{0,1}
\end{pmatrix},
\qquad
\mathbf S_0
=
\begin{pmatrix}
s_{00} & s_{01}\\
s_{01} & s_{11}
\end{pmatrix}.
$$

The mean vector specifies the prior expected intercept and slope:

$$
\mathbb E[w_0]=m_{0,0},
\qquad
\mathbb E[w_1]=m_{0,1}.
$$

The diagonal entries of the covariance matrix describe uncertainty
about each weight:

$$
\operatorname{Var}(w_0)=s_{00},
\qquad
\operatorname{Var}(w_1)=s_{11}.
$$

Larger variances allow a wider range of plausible intercepts or
slopes. The off-diagonal entry describes how they vary together:

$$
\operatorname{Cov}(w_0,w_1)=s_{01}.
$$

A positive covariance associates larger intercepts with larger
slopes. A negative covariance associates larger intercepts with
smaller slopes. For a joint Gaussian prior, zero covariance means
the two weights are independent.

For example, an illustrative prior could be

$$
\mathbf m_0
=
\begin{pmatrix}10\\1.5\end{pmatrix},
\qquad
\mathbf S_0
=
\begin{pmatrix}
25 & 0\\
0 & 0.25
\end{pmatrix}.
$$

This gives a prior mean intercept of 10 kWh with standard deviation
5 kWh, and a prior mean slope of 1.5 kWh per degree Celsius with
standard deviation 0.5 kWh per degree Celsius.

The mean of the regression function under this prior is

$$
\mathbb E_{\mathbf w}[f(x;\mathbf w)]
=
\mathbf m_0^{\mathsf T}\boldsymbol\phi(x)
=
10+1.5x.
$$

Individual weight vectors drawn from the prior produce lines
around this prior mean function.

A prior may reflect information from previous studies, engineering
knowledge, or deliberately broad assumptions. Its parameters
should describe plausible weights in the chosen units.

We use a Gaussian prior because it is mathematically convenient:
combined with our Gaussian likelihood, it gives a Gaussian
posterior. We will derive this conjugate update in Section 3.

### 2.3 A simple prior for the worked example

In Section 2.2, we used building units (kWh and kWh per degree Celsius), so it was natural to choose different prior variances for intercept and slope. To make the Bayesian update easy to compute by hand, we now return to the small numerical example from Lecture 17:

$$
(x_1,y_1)=(0,1),\qquad
(x_2,y_2)=(1,2),\qquad
(x_3,y_3)=(2,2).
$$

We treat these as illustrative numerical values rather than temperatures
and electricity bills. With the straight-line model
$f(x;\mathbf w)=w_0+w_1x$, the least-squares fit from Lecture 17 is:

$$
\widehat{\mathbf w}_{\mathrm{LS}}=(7/6,\,1/2)^{\mathsf T},
\qquad \widehat f_{\mathrm{LS}}(x)=7/6+0.5x.
$$

We use this MLE estimation with Gaussian likelihood observation as a reference. For the Bayesian fit, we keep the same observations and Gaussian likelihood, fix the noise precision at $\beta=1$, and choose the prior

$$
\mathbf w \sim \mathcal N(\mathbf 0,\alpha^{-1}\mathbf I_2).
$$

For this worked example, set $\alpha=1$.

Before seeing the three observations, both the intercept and the slope have prior mean
zero and variance $1/\alpha$. They are independent under this prior.
A larger $\alpha$ concentrates the plausible weights closer to zero.

For this example, set $\alpha=1$. Our starting point is therefore

$$
\mathbf m_0=
\begin{pmatrix}0\\0\end{pmatrix},
\qquad
\mathbf S_0=
\begin{pmatrix}1&0\\0&1\end{pmatrix}.
$$

The prior mean line is $f(x;\mathbf m_0)=0$. This does not mean that
every plausible line is horizontal: a draw from the prior generally
has a nonzero intercept and slope.

We also set the observation-noise precision to $\beta=1$. Although
we chose the same numerical value for both precisions, they describe
different things:

- **Prior precision $\alpha$** controls the spread of the weights
  before we observe the data. Setting $\alpha=1$ gives the intercept
  and slope a prior variance of $1/\alpha=1$ each.
- **Noise precision $\beta$**, introduced in Section 1.2, controls
  the spread of observations around a given line. Setting $\beta=1$
  gives the observation error a variance of $1/\beta=1$.

#### Relating different draws to different regression lines 

With $\alpha=1$, our prior is
$\mathbf w\sim\mathcal N(\mathbf 0,\mathbf I_2)$.
Our first draw gives an intercept of approximately $-0.370$
and a slope of $0.994$. Together, they define the line

$$
f(x;\mathbf w^{(1)})\approx -0.370+0.994x.
$$

We can now evaluate this line at the three inputs from our worked example:

$$
\begin{aligned}
f(0;\mathbf w^{(1)}) &\approx -0.370,\\
f(1;\mathbf w^{(1)}) &\approx -0.370+0.994=0.624,\\
f(2;\mathbf w^{(1)}) &\approx -0.370+2(0.994)=1.618.
\end{aligned}
$$

A second draw gives a different intercept and slope, and hence a
different line. Here are three draws from the same prior:

| Draw | Intercept $w_0$ | Slope $w_1$ | $f(0)$ | $f(1)$ | $f(2)$ |
|---|---:|---:|---:|---:|---:|
| 1 | −0.370 | 0.994 | −0.370 | 0.624 | 1.618 |
| 2 | 0.416 | −0.618 | 0.416 | −0.202 | −0.820 |
| 3 | 0.672 | −1.450 | 0.672 | −0.778 | −2.228 |

These are conditional means calculated from the sampled weights.
We have used the input values $0,1,2$, but we have not yet used the
observed targets $1,2,2$. Those enter in Section 3, when we update
the weights.

The following figure illustrates how draws from the prior affect both intercept and slope in this example.


![Bayesian regression](images/lecture19-prior-lines.png)

**Figure:** The left panel shows the joint Gaussian prior over the
intercept and slope for the worked example. Each colored point is
one draw $(w_0,w_1)$; the matching line on the right is
$f(x;\mathbf w)=w_0+w_1x$. The black cross at $(0,0)$ corresponds
to the dashed prior mean line. The three observed target values
have not yet been used to update the weights.


## 3. Updating the distribution over the weights

In Section 2.3, we drew three intercept–slope pairs from the prior. Each pair defines one line. We now compare those same lines with the observations $(0,1)$, $(1,2)$, and $(2,2)$.

![Three lines drawn from the prior with the observed data](images/lecture19-prior-lines-with-data.png)

**Figure:** The colored lines are the same prior draws as in Section 2.3. The black dots show the observed targets.

Consider the observation $(2,2)$. At $x=2$, the first line gives a mean of $1.618$, while the third gives a mean of $-2.228$. Both assume the same noise variance. The observed value $y=2$ is closer to the first line’s mean, so this observation has a higher Gaussian density under the first line.

The **likelihood** makes this comparison using all three observations. To obtain the **posterior**, we multiply the likelihood by the prior density at each candidate weight pair and normalize:

$$
p(\mathbf w\mid\mathbf y,\boldsymbol\Phi,\beta)
\propto
\underbrace{p(\mathbf y\mid\boldsymbol\Phi,\mathbf w,\beta)}_{\text{likelihood}}
\underbrace{p(\mathbf w)}_{\text{prior}}.
$$

The three lines illustrate how the likelihood distinguishes between candidates. The update applies to **every** intercept–slope pair. A given pair still defines the same line; what changes is the density assigned to that pair.

### 3.1 Why the posterior is Gaussian

Recall conjugacy from Lecture 14: a Beta prior updated with a Bernoulli likelihood gave a Beta posterior. Here, a Gaussian prior over the weights gives a **Gaussian posterior** because the regression mean is linear in the weights and the observation noise is Gaussian. The log-likelihood and log-prior are both quadratic in the weights, so their sum has the form of a Gaussian log-density.

After $N$ observations, the posterior is

$$
p(\mathbf w\mid\mathbf y,\boldsymbol\Phi,\beta)
=
\mathcal N(\mathbf w\mid\mathbf m_N,\mathbf S_N).
$$

Just as two shape parameters specified the Beta posterior, the mean vector $\mathbf m_N$ and covariance matrix $\mathbf S_N$ specify the entire Gaussian posterior. We can therefore obtain the normalized posterior without calculating the evidence integral separately.

**Posterior covariance.** It is easiest to write the update in terms of **precision**, the inverse of covariance:

$$
\mathbf S_N^{-1}
=
\underbrace{\mathbf S_0^{-1}}_{\text{prior precision}}
+
\underbrace{\beta\boldsymbol\Phi^{\mathsf T}\boldsymbol\Phi}_{\text{precision contributed by the data}}.
$$

The inverse of this matrix is the posterior covariance $\mathbf S_N$. For a given prior and fixed noise precision $\beta$, it depends on **where** we observed the targets. For example, if we measured only at $x=0$, every observation would tell us about the intercept without directly measuring the slope.

**Posterior mean.** The observed target values $\mathbf y$ determine how the center of the distribution shifts from its prior mean $\mathbf m_0$:

$$
\mathbf m_N
=
\mathbf S_N
\left(
\mathbf S_0^{-1}\mathbf m_0
+
\beta\boldsymbol\Phi^{\mathsf T}\mathbf y
\right).
$$

### 3.2 Fitting and interpreting the three-point example

We now apply the update to our three observations using the prior $\mathbf w\sim\mathcal N(\mathbf 0,\mathbf I_2)$ and noise precision $\beta=1$. Each row of the design matrix is $(1,x_n)$:

$$
\boldsymbol\Phi=
\begin{pmatrix}
1&0\\
1&1\\
1&2
\end{pmatrix},
\qquad
\mathbf y=
\begin{pmatrix}
1\\
2\\
2
\end{pmatrix}.
$$

The following code evaluates the two update equations. Because the prior mean is zero, its contribution to the posterior mean equation is zero; the prior precision still enters the calculation.

```python
import numpy as np

x = np.array([0., 1., 2.])
y = np.array([1., 2., 2.])
Phi = np.column_stack((np.ones_like(x), x))

alpha = 1.0  # prior precision
beta = 1.0   # observation-noise precision
I = np.eye(2)

precision = alpha * I + beta * (Phi.T @ Phi)
S_N = np.linalg.solve(precision, I)
m_N = np.linalg.solve(precision, beta * (Phi.T @ y))

print("Posterior mean:", m_N)
print("Posterior covariance:\n", S_N.round(3))
```

| Quantity | Prior | Posterior |
|---|---:|---:|
| Mean intercept | 0 | 0.800 |
| Mean slope | 0 | 0.600 |
| Intercept variance | 1 | 0.400 |
| Slope variance | 1 | 0.267 |
| Intercept–slope covariance | 0 | −0.200 |

The first two rows give the **posterior mean line**, $f(x;\mathbf m_N)=0.8+0.6x$. We can now compare it with the least-squares line from Lecture 17.

![Least-squares and posterior mean lines for the same three observations](images/LQ-versus-Bayesian-posterior.png)

**Figure:** The dashed orange line is the least-squares fit, $\widehat f_{\mathrm{LS}}(x)=7/6+0.5x$. The blue line is the posterior mean, $f(x;\mathbf m_N)=0.8+0.6x$.

The remaining rows tell us something neither line shows. Both weight variances have fallen below their prior value of 1. The posterior covariance is negative: among lines that fit these observations reasonably well, a higher intercept tends to go with a lower slope.

For this Gaussian posterior, the mean is also the mode, so $\mathbf m_N$ is the **maximum a posteriori (MAP)** estimate. If we broaden the prior by letting $\alpha$ approach zero while keeping $\beta$ fixed, the posterior mean approaches the least-squares estimate. In this example, the two columns of $\boldsymbol\Phi$ are linearly independent, so that estimate is unique.

#### How certain are we that the slope is positive?

The blue line rises. How much posterior probability is assigned to rising lines? As in Lecture 15, we specify a threshold and calculate the posterior probability above it. Here the threshold is zero: a line rises when $w_1>0$.

We use the **marginal posterior** of the slope, which accounts for uncertainty about the intercept. Because the joint posterior is Gaussian, we can read the slope’s mean and variance directly from the table. Using the unrounded variance gives

$$
w_1\mid\mathcal D,\beta=1
\sim \mathcal N\!\left(0.6,\frac{4}{15}\right).
$$

```python
from scipy.stats import norm

# SciPy uses the standard deviation, so take the square root of the variance.
slope = norm(loc=m_N[1], scale=np.sqrt(S_N[1, 1]))
print("95% credible interval:", slope.ppf([0.025, 0.975]).round(2))
print(f"Probability of a positive slope: {slope.sf(0):.1%}")
```

Given the data, prior, and observation model, the posterior assigns **87.7% probability** to a positive slope. The equal-tailed **95% credible interval** is approximately $[-0.41,1.61]$. These results agree: rising lines receive most of the posterior probability, but about 12% remains on falling lines—enough for the middle 95% interval to include zero.




## Practice exercises

Use the same three observations throughout: $(0,1)$, $(1,2)$, and $(2,2)$. The model is $f(x;\mathbf w)=w_0+w_1x$, with independent Gaussian observation errors and fixed noise precision $\beta=1$.

### 1. Which line does the likelihood favor?

Consider two candidate lines: $f_A(x)=1+x$ and $f_B(x)=1$. Both give the same mean at $x=0$. Before calculating, which do you expect the observations to favor? For each line, calculate its three fitted means and its sum of squared residuals. Does the line with the higher likelihood need to pass through every point?

:::{admonition} Solution 1
:class: dropdown

Line A gives means $(1,2,3)$ and residuals $(0,0,-1)$, so its sum of squared residuals is 1. Line B gives means $(1,1,1)$ and residuals $(0,1,1)$, so its sum is 2. At the same noise precision, the smaller sum gives line A the higher Gaussian likelihood. It misses the last point, but fits the three observations better overall. This comparison uses only the likelihood.
:::

### 2. What does the posterior tell us beyond a fitted line?

With the prior $\mathbf w\sim\mathcal N(\mathbf 0,\mathbf I_2)$, the posterior mean is $(0.8,0.6)^{\mathsf T}$. Use it to find the posterior mean line's value at $x=1$. The observed target there is 2. Does the difference mean the update was calculated incorrectly? The posterior covariance between intercept and slope is $-0.2$: explain how an increase in the intercept could be offset by a change in slope at this input.

:::{admonition} Solution 2
:class: dropdown

At $x=1$, the posterior mean line gives $0.8+0.6=1.4$. The observed target is 0.6 above it. The line summarizes our estimate of the conditional mean; an individual observation can differ from it even if the weights were known. At this input the line height is $w_0+w_1$, so raising the intercept by 0.1 and lowering the slope by 0.1 leaves the height unchanged. The negative covariance tells us that higher intercepts tend to accompany lower slopes among plausible lines. It does not require an exact offset in every draw.
:::

### 3. What happens if we broaden the prior?

In the lecture's Python example, change only the prior precision from `alpha = 1.0` to `alpha = 0.01`. Before running the code, predict whether the posterior mean line will move closer to the prior mean line or to the least-squares line. Rerun the update and compare the new weights with the least-squares weights $(7/6,1/2)^{\mathsf T}$. Why is it important to leave `beta` unchanged for this comparison?

:::{admonition} Solution 3
:class: dropdown

The prior variance of each weight increases from 1 to 100. The prior now gives more similar density to candidate weights near the least-squares solution, so the likelihood has more influence on the fit. The posterior mean changes from $(0.800,0.600)^{\mathsf T}$ to approximately $(1.160,0.503)^{\mathsf T}$, close to the least-squares weights $(1.167,0.500)^{\mathsf T}$. Keeping $\beta$ fixed keeps the assumed observation noise and the likelihood unchanged, so we can attribute the difference to the prior.
:::



## References

Christopher M. Bishop, *Pattern Recognition and Machine Learning*, Chapter 3, pages 136–142 (especially Section 3.1 and Section 3.1.1).
