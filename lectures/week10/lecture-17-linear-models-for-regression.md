# Lecture 17 - Linear Models for Regression

Imagine managing a building and predicting its electricity use for tomorrow. On a mild day, when little heating or cooling is needed, the building might use around 25 kWh. On a hot day, air conditioning might raise that to around 60 kWh. These are illustrative averages and actual use will vary from day to day even at the same temperature.

To use temperature in the prediction, we could first try fitting a straight line to past measurements. We need a way to decide which line fits better: one candidate might be close on mild days but far off on hot days, while another might make smaller errors overall. **Least squares** gives us a rule for making that choice. We will build the rule from individual prediction errors before introducing any probability model.

After defining the least-squares estimator, we will ask why squared errors are useful here. A conditional Gaussian model provides a probabilistic answer: under independent, common-variance Gaussian noise, the weights found by maximum likelihood are the **same weights** found by least squares. We will then allow the predicted mean to curve with temperature using basis functions, while keeping the same fitting principle.

:::{admonition} Guiding question
:class: tip

What does a least-squares estimator actually do, why does Gaussian maximum likelihood give the same answer under certain assumptions, and how can we calculate that answer for a flexible regression model?
:::

## Learning outcomes

After completing this lecture, you should be able to:

1. Build the least-squares estimator from predictions, residuals, and a total squared-error score.
2. Calculate the straight-line least-squares slope and intercept, and distinguish the estimator from a way of computing it.
3. State the assumptions under which the least-squares and Gaussian maximum-likelihood weights are the same.
4. Connect the conditional Gaussian to Lecture 12's exponential-family form, then construct nonlinear basis functions and a design matrix.
5. Derive the normal equations for the least-squares weights and state when a unique closed-form solution exists.

## Notation

We will use $Y$ for the random target and $y$ for an observed value. Earlier lectures sometimes called the same target $T$ and its observed value $t$. An input is $\mathbf x\in\mathbb R^D$, and the training data are $\mathcal D=\{(\mathbf x_n,y_n)\}_{n=1}^N$. The inputs and observed targets are given and the weights $\mathbf w$ and noise precision $\beta$ are unknown parameters, not random variables in the frequentist fit below.

## What is the least-squares estimator?

Suppose our data consist of observed input–target pairs $(x_n,y_n)$. For now, let $x_n$ be one input, such as temperature, and choose a straight-line prediction rule

$$
\widehat y_n=f(x_n;\mathbf w)=w_0+w_1x_n,
$$

where $w_0$ is the intercept and $w_1$ is the slope. At this point we have chosen the **shape** of the model, but not the values of its weights. Different choices of $(w_0,w_1)$ give different lines and different predictions.

### Turn prediction errors into one score

For each observation, compare what happened with what the line predicted. The difference

$$
r_n(\mathbf w)=y_n-\widehat y_n
=y_n-(w_0+w_1x_n)
$$

is its **residual**. A positive residual means the line predicted too little; a negative residual means it predicted too much. Adding the raw residuals is a poor measure of fit because positive and negative errors can cancel. For example, errors of $+1$ and $-1$ add to zero even though neither prediction was correct.

Least squares **squares each residual and adds the results**. Squaring makes every contribution nonnegative and makes a large error count more than a small one. We write the total as

$$
E_D(\mathbf w)
=\frac12\sum_{n=1}^{N}r_n(\mathbf w)^2
=\frac12\sum_{n=1}^{N}
\left[y_n-(w_0+w_1x_n)\right]^2.
$$

The factor $1/2$ is only for convenience when we differentiate later: multiplying every candidate's score by the same positive number cannot change which candidate has the smallest score. Squared error is a **choice of fitting criterion**, not yet a consequence of assuming a probability distribution.

Consider three observations: $(0,1)$, $(1,2)$, and $(2,2)$. The candidate line $f(x)=1+x$ predicts $(1,2,3)$, so its residuals are $(0,0,-1)$ and its *sum* of squared residuals is $1$. The line $f(x)=1+x/2$ predicts $(1,1.5,2)$, giving residuals $(0,0.5,0)$ and a smaller squared-error sum of $0.25$. With the factor $1/2$, their $E_D$ scores are $0.5$ and $0.125$, respectively. Either way, the second line fits these three points better **by the least-squares criterion**. This comparison does not yet establish that it is the *best* line; we must consider all possible intercepts and slopes.

### Define the estimator

The **least-squares estimator (LSE)** is the rule that selects the weights with the smallest total squared-error score:

$$
\boxed{
\widehat{\mathbf w}_{\mathrm{LS}}
=\mathop{\arg\min}_{\mathbf w}E_D(\mathbf w).
}
$$

The notation $\arg\min_{\mathbf w}E_D(\mathbf w)$ means “find the *values of the weights* at which $E_D$ is smallest.” The result $\widehat{\mathbf w}_{\mathrm{LS}}$ is an **estimate** because it is calculated from the observed sample; another sample may give different fitted weights. For the straight-line model, this is called **ordinary least squares (OLS)**: each observation contributes one unweighted squared residual.

The definition tells us *which weights we want*. It does not specify *how* to find them. A straight-line formula, the matrix normal equations later in this lecture, and gradient descent in Lecture 18 are different calculation methods for the same least-squares goal. We can define this estimator without a Gaussian assumption.

### Find the best straight line

For the straight-line model, we can calculate the LSE by setting the derivatives of $E_D$ with respect to $w_0$ and $w_1$ to zero. The first derivative gives

$$
\frac{\partial E_D}{\partial w_0}
=-\sum_{n=1}^{N}(y_n-w_0-w_1x_n)=0
\quad\Longrightarrow\quad
w_0=\bar y-w_1\bar x,
$$

where $\bar x=N^{-1}\sum_n x_n$ and $\bar y=N^{-1}\sum_n y_n$ are the sample means. The fitted line therefore passes through $(\bar x,\bar y)$. The second derivative gives

$$
\frac{\partial E_D}{\partial w_1}
=-\sum_{n=1}^{N}x_n(y_n-w_0-w_1x_n)=0.
$$

Substitute $w_0=\bar y-w_1\bar x$ into this second equation:

$$
0=\sum_{n=1}^{N}x_n
\left[(y_n-\bar y)-w_1(x_n-\bar x)\right].
$$

The centered values satisfy $\sum_n(y_n-\bar y)=0$ and $\sum_n(x_n-\bar x)=0$. We may therefore replace $x_n$ in this sum by $x_n-\bar x$ without changing its value. Moving the term containing $w_1$ to the other side gives

$$
w_1\sum_{n=1}^{N}(x_n-\bar x)^2
=\sum_{n=1}^{N}(x_n-\bar x)(y_n-\bar y).
$$

If the $x_n$ values are not all equal, the sum on the left multiplying $w_1$ is positive, and we obtain the familiar straight-line solution

$$
\boxed{
\widehat w_1
=\frac{\sum_{n=1}^{N}(x_n-\bar x)(y_n-\bar y)}
{\sum_{n=1}^{N}(x_n-\bar x)^2},
\qquad
\widehat w_0=\bar y-\widehat w_1\bar x.
}
$$

For the three observations above, $\bar x=1$ and $\bar y=5/3$. The slope's numerator is $(-1)(-2/3)+0+(1)(1/3)=1$, and its denominator is $(-1)^2+0+1^2=2$. Thus $\widehat w_1=1/2$ and $\widehat w_0=5/3-1/2=7/6$. The best-fitting line by this criterion is $\widehat f(x)=7/6+x/2$. Notice that the earlier candidate $1+x/2$ had the same slope but a different intercept: it was *better* than the first candidate, but not the least-squares optimum.

~~~{figure} images/least-squares-residuals.svg
:name: least-squares-residuals
:alt: Three observed targets marked as black circles, their predictions on a blue least-squares line marked as squares, and signed residuals shown as orange vertical dashed segments.
:width: 900px

The black points are the observed target values—the “ground truth” for these three observations, **not** the unknown noise-free relationship. Blue squares show what the least-squares line predicts at the same inputs. The orange segments are the signed residuals $r_n=y_n-\widehat y_n$: $-1/6$, $+1/3$, and $-1/6$. Their squared sum is $1/6$, so the objective with its factor $1/2$ is $E_D(\widehat{\mathbf w}_{\mathrm{LS}})=1/12$.
~~~

We have now defined the estimator and found it for a straight line. Next, we ask why a Gaussian probability model selects these same weights.

## From a conditional Gaussian to the least-squares estimator

Recall that a Gaussian with mean $m$ and variance $\sigma^2=\beta^{-1}$ has density

$$
p(y\mid m,\beta)
=\left(\frac{\beta}{2\pi}\right)^{1/2}
\exp\!\left\{-\frac{\beta}{2}(y-m)^2\right\},
\qquad \beta>0.
$$

The **precision** $\beta=1/\sigma^2$ is the inverse variance. High precision means little spread around the mean. In regression, we let the mean depend on the input:

$$
Y\mid\mathbf x,\mathbf w,\beta
\sim\mathcal N\!\left(f(\mathbf x;\mathbf w),\beta^{-1}\right),
\qquad
p(y\mid\mathbf x,\mathbf w,\beta)
=\left(\frac{\beta}{2\pi}\right)^{1/2}
\exp\!\left\{-\frac{\beta}{2}
[y-f(\mathbf x;\mathbf w)]^2\right\}.
$$

Thus $\mathbb E[Y\mid\mathbf x,\mathbf w,\beta]=f(\mathbf x;\mathbf w)$ and $\operatorname{Var}(Y\mid\mathbf x,\mathbf w,\beta)=\beta^{-1}$. The model says that, at a given input, observations fluctuate around the predicted mean with the same variance. This **constant-variance** assumption is useful but may be inappropriate when the spread changes substantially with $\mathbf x$.

This is a conditional density for $Y$ given $\mathbf x$. It does not assign a probability distribution to the inputs themselves. We can therefore fit $p(y\mid\mathbf x,\mathbf w,\beta)$ using observed input-target pairs without specifying $p(\mathbf x)$. The function $f$ could be the straight line above or a more flexible prediction function we choose later.

### When Gaussian MLE equals LSE

For the data-set likelihood, assume that the observations have **independent Gaussian errors with the same variance** $1/\beta$. In other words, each target can be written as $y_n=f(\mathbf x_n;\mathbf w)+\epsilon_n$, where the $\epsilon_n$ are independent draws from $\mathcal N(0,\beta^{-1})$. The inputs can differ, so the predicted means $f(\mathbf x_n;\mathbf w)$ can differ too; the same weights $\mathbf w$ determine them all.

Independence lets us multiply the conditional densities:

$$
\begin{aligned}
p(\mathbf y\mid\mathbf x_1,\ldots,\mathbf x_N,\mathbf w,\beta)
&=\prod_{n=1}^{N}p(y_n\mid\mathbf x_n,\mathbf w,\beta)\\
&=\left(\frac{\beta}{2\pi}\right)^{N/2}
\exp\!\left\{-\frac{\beta}{2}\sum_{n=1}^{N}
\left[y_n-f(\mathbf x_n;\mathbf w)\right]^2\right\}.
\end{aligned}
$$

The inputs do not have to be identical for this multiplication. Take the logarithm to make the part involving the weights easier to see:

$$
\log p(\mathbf y\mid\mathbf x_1,\ldots,\mathbf x_N,\mathbf w,\beta)
=\underbrace{\frac N2\log\beta-\frac N2\log(2\pi)}_{\text{same for every choice of }\mathbf w\text{ if }\beta\text{ is fixed}}
-\frac{\beta}{2}\sum_{n=1}^{N}
\left[y_n-f(\mathbf x_n;\mathbf w)\right]^2.
$$

Compare two candidate weight vectors while holding $\beta>0$ fixed. The bracketed term is the same for both. Only the squared-error sum changes. A *smaller* sum is multiplied by the negative number $-\beta/2$, giving a *larger* log-likelihood. Therefore maximizing this Gaussian likelihood selects exactly the least-squares weights we defined earlier:

$$
\boxed{
\widehat{\mathbf w}_{\mathrm{ML}}
=\mathop{\arg\max}_{\mathbf w}
p(\mathbf y\mid\mathbf x_1,\ldots,\mathbf x_N,\mathbf w,\beta)
=\mathop{\arg\min}_{\mathbf w}E_D(\mathbf w)
=\widehat{\mathbf w}_{\mathrm{LS}}.
}
$$

This is an **equality under the stated assumptions**, not a new definition of least squares. We were able to define the LSE without a probability model; the Gaussian model explains why it is also an MLE in this case. Another noise model need not have squared-error MLE weights. This equality identifies the desired weights, but we still need a way to calculate them for more flexible prediction functions.

### The connection to Lecture 12

For fixed $\beta$, expand the square in the Gaussian log density. Writing $m=f(\mathbf x;\mathbf w)$ gives

$$
\begin{aligned}
\log p(y\mid m,\beta)
&=\frac12\log\frac{\beta}{2\pi}
-\frac{\beta}{2}(y-m)^2\\
&=\underbrace{\left[\frac12\log\frac{\beta}{2\pi}
-\frac{\beta y^2}{2}\right]}_{\log h_\beta(y)}
+\underbrace{\beta m}_{\eta}\underbrace{y}_{T(y)}
-\underbrace{\frac{\beta m^2}{2}}_{A_\beta(\eta)}.
\end{aligned}
$$

Because $\eta=\beta m$, the log normalizer is $A_\beta(\eta)=\eta^2/(2\beta)$. We have the exponential-family form

$$
p(y\mid\eta,\beta)
=h_\beta(y)\exp\!\left\{\eta T(y)-A_\beta(\eta)\right\},
\quad
T(y)=y,
\quad
\eta(\mathbf x)=\beta f(\mathbf x;\mathbf w).
$$

Here $T(y)$ is the sufficient-statistic *function* from Lecture 12. As a check, derivatives of the log normalizer give the conditional mean and variance:

$$
\mathbb E[Y\mid\mathbf x]
=\frac{dA_\beta}{d\eta}=\frac{\eta(\mathbf x)}{\beta}=f(\mathbf x;\mathbf w),
\qquad
\operatorname{Var}(Y\mid\mathbf x)
=\frac{d^2A_\beta}{d\eta^2}=\frac1\beta.
$$

The new idea is not a new probability distribution. It is that the Gaussian's natural parameter, and therefore its mean, varies systematically with $\mathbf x$. We hold $\beta$ fixed in this exponential-family comparison, but fitting $\beta$ can be considered separately (not shown in this lecture).

## Linear and nonlinear basis functions

Least squares tells us how to score a chosen prediction function, and Gaussian likelihood gives the same fitted weights under the assumptions above. Neither rule decides what *shape* the prediction function should have. Let us build one gradually, using the building-temperature example.

### Start with a straight line

Our first least-squares example used the straight-line model

$$
f(x;\mathbf w)=w_0+w_1x.
$$

The parameter $w_0$ sets the value at $x=0$, while $w_1$ describes how much the predicted mean changes when temperature rises by one unit. With several measured inputs, such as temperature and occupancy, we could add one weighted term for each input. In either case, the mean is a straight line (or, in several dimensions, a plane) in the original inputs.

That shape may be too restrictive. A building can use more electricity on *both* very cold days (heating) and very hot days (cooling) than on mild days. A single straight line cannot decrease toward mild temperatures and then increase again: its slope $w_1$ stays the same everywhere. We need a more flexible mean without giving every observed temperature an unrelated parameter.

### Transform the input before weighting it

Choose a reference temperature of $20^\circ$C and write $z=x-20$. A quadratic mean is

$$
f(x;\mathbf w)=w_0+w_1z+w_2z^2.
$$

At $20^\circ$C, $z=0$, so the predicted mean is $w_0$. At $5^\circ$C and $35^\circ$C, $z^2=225$ in both cases. If $w_2>0$, the squared term raises the prediction at both extremes relative to the center; the $w_1z$ term allows the cold and hot sides to differ. The data will determine the weights. Choosing the transformation $z^2$ simply gives the model a shape that a straight line could not express.

The functions $1$, $z$, and $z^2$ are **basis functions**: fixed ways of turning an input into numbers that the model can weight and add. For this example, define

$$
\phi_0(x)=1,\qquad
\phi_1(x)=x-20,\qquad
\phi_2(x)=(x-20)^2.
$$

Then the same model can be written $f(x;\mathbf w)=w_0\phi_0(x)+w_1\phi_1(x)+w_2\phi_2(x)$. The basis functions are chosen *before* fitting the weights. They describe which kinds of input dependence are available; the fitted weights decide how much each one contributes. The constant function $\phi_0(x)=1$ gives an intercept, even if the other functions are zero at some input.

Quadratics are only one possibility. A **Gaussian-shaped basis function**

$$
\phi_j(x)=\exp\!\left\{-\frac{(x-c_j)^2}{2s_j^2}\right\}
$$

is largest near its chosen center $c_j$ and becomes smaller farther away; $s_j>0$ controls how broad that region is. Several such functions centered at different temperatures can let the predicted mean respond differently in different temperature ranges. Unlike a power such as $x^2$, which changes across the entire input range, each Gaussian-shaped function emphasizes a local region. These are functions of the *input*, not claims that the input itself has a Gaussian distribution. For the linear-weight fitting problem in this lecture, their centers and widths are selected beforehand. If we learned $c_j$ or $s_j$ as well, the model would no longer be linear in all its unknown parameters.

~~~{figure} images/gaussian-basis-functions.svg
:name: gaussian-basis-functions-regression
:alt: Five fixed Gaussian-shaped basis functions centered at different temperatures above an example data set and a curve fitted as their weighted sum.
:width: 900px

The top panel shows five fixed Gaussian-shaped basis functions of temperature. In the bottom panel, fitting their weights to the example observations produces a curved predicted mean. The individual bells are **not warped by the data**: their centers and widths stay fixed, while their weights determine the combined curve.
~~~

### Why this is still a linear model

Both $z^2$ and the Gaussian-shaped function are nonlinear in $x$. Nevertheless, once the basis functions are fixed, each unknown weight appears only as $w_j$ times a known number. The model is **linear in its weights**, even when its graph as a function of $x$ is curved. This distinction matters because the closed-form least-squares result below relies on linearity in the weights, not linearity in the original input.

For a general input vector $\mathbf x\in\mathbb R^D$, choose $M$ basis functions $\phi_0(\mathbf x),\ldots,\phi_{M-1}(\mathbf x)$ and collect their values in

$$
\boldsymbol\phi(\mathbf x)
=\begin{pmatrix}\phi_0(\mathbf x)\\\vdots\\\phi_{M-1}(\mathbf x)\end{pmatrix},
\qquad
f(\mathbf x;\mathbf w)
=\mathbf w^{\mathsf T}\boldsymbol\phi(\mathbf x)
=\sum_{j=0}^{M-1}w_j\phi_j(\mathbf x),
\qquad
\mathbf w=\begin{pmatrix}w_0\\\vdots\\w_{M-1}\end{pmatrix}.
$$

Here $D$ counts components of the original input, whereas $M$ counts basis functions (including the constant one, if used). They need not be equal: one temperature measurement can produce many basis-function values. Ordinary straight-line regression is the special case $\boldsymbol\phi(x)=(1,x)^{\mathsf T}$; quadratic regression is another choice of $\boldsymbol\phi(x)$, but both use the same weighted-sum form.

The basis choice controls which shapes the model can express. Too few may miss a real pattern; too many may let it follow noise in the training data. As in Lecture 11, use validation data to assess that choice rather than judging it only by training error.

## The design matrix

Each training input produces one basis vector. Stack their transposes as rows of the **design matrix**:

$$
\boldsymbol\Phi
=\begin{pmatrix}
\boldsymbol\phi(\mathbf x_1)^{\mathsf T}\\
\boldsymbol\phi(\mathbf x_2)^{\mathsf T}\\
\vdots\\
\boldsymbol\phi(\mathbf x_N)^{\mathsf T}
\end{pmatrix},
\qquad
\Phi_{nj}=\phi_j(\mathbf x_n).
$$

Thus $\boldsymbol\Phi$ is $N\times M$, $\mathbf w$ is $M\times1$, and $\mathbf y=(y_1,\ldots,y_N)^{\mathsf T}$ is $N\times1$. The predicted values for all training inputs are $\boldsymbol\Phi\mathbf w$; its $n$th component is $\mathbf w^{\mathsf T}\boldsymbol\phi(\mathbf x_n)$.

For example, with $\boldsymbol\phi(x)=(1,x)^{\mathsf T}$ and inputs $x_1=0$, $x_2=1$, $x_3=2$,

$$
\boldsymbol\Phi=
\begin{pmatrix}1&0\\1&1\\1&2\end{pmatrix},
\qquad
\boldsymbol\Phi\mathbf w=
\begin{pmatrix}w_0\\w_0+w_1\\w_0+2w_1\end{pmatrix}.
$$

The first column of ones comes from the intercept basis function. The second column lists the three observed input values. If we added a quadratic basis function, the design matrix would gain a third column $(0,1,4)^{\mathsf T}$.

## Normal equations and the closed-form solution

We have defined the least-squares estimator, seen why it is also a Gaussian MLE under the stated assumptions, and constructed a design matrix for flexible predictions. Now we calculate the estimator from that matrix. Write $\widehat{\mathbf w}$ for the shared least-squares and Gaussian maximum-likelihood weights. The objective from the start of this lecture becomes

$$
E_D(\mathbf w)
=\frac12\sum_{n=1}^{N}
\left[y_n-\mathbf w^{\mathsf T}\boldsymbol\phi(\mathbf x_n)\right]^2
=\frac12\|\mathbf y-\boldsymbol\Phi\mathbf w\|_2^2.
$$

To find the minimum, expand the squared norm before differentiating:

$$
\begin{aligned}
E_D(\mathbf w)
&=\frac12(\mathbf y-\boldsymbol\Phi\mathbf w)^{\mathsf T}
(\mathbf y-\boldsymbol\Phi\mathbf w)\\
&=\frac12\mathbf y^{\mathsf T}\mathbf y
-\mathbf w^{\mathsf T}\boldsymbol\Phi^{\mathsf T}\mathbf y
+\frac12\mathbf w^{\mathsf T}\boldsymbol\Phi^{\mathsf T}
\boldsymbol\Phi\mathbf w.
\end{aligned}
$$

The two cross terms are equal scalars. Differentiating with respect to the $M$ weights gives

$$
\nabla_{\mathbf w}E_D(\mathbf w)
=-\boldsymbol\Phi^{\mathsf T}\mathbf y
+\boldsymbol\Phi^{\mathsf T}\boldsymbol\Phi\mathbf w
=-\boldsymbol\Phi^{\mathsf T}
(\mathbf y-\boldsymbol\Phi\mathbf w).
$$

At a minimum, each partial derivative is zero. Therefore the fitted weights satisfy the **normal equations**

$$
\boxed{\boldsymbol\Phi^{\mathsf T}\boldsymbol\Phi
\widehat{\mathbf w}
=\boldsymbol\Phi^{\mathsf T}\mathbf y.}
$$

Equivalently, $\boldsymbol\Phi^{\mathsf T}(\mathbf y-\boldsymbol\Phi\widehat{\mathbf w})=\mathbf0$. The vector in parentheses contains the fitted **residuals**. Its dot product with every column of $\boldsymbol\Phi$ is zero: no remaining residual pattern points along any chosen basis-function column. This is also the geometric picture of projecting $\mathbf y$ onto the space of predictions that the model can make.

If the $M$ columns of $\boldsymbol\Phi$ are linearly independent, $\boldsymbol\Phi^{\mathsf T}\boldsymbol\Phi$ is invertible. In that case the unique solution is

$$
\boxed{\widehat{\mathbf w}
=\left(\boldsymbol\Phi^{\mathsf T}\boldsymbol\Phi\right)^{-1}
\boldsymbol\Phi^{\mathsf T}\mathbf y.}
$$

Full column rank requires $N\geq M$, but $N\geq M$ alone does not guarantee it: two basis functions may produce identical columns on the observed inputs. If the columns are dependent, the inverse does not exist and the fitted weights need not be unique. The normal equations still describe least-squares solutions.

### A small complete fit

Return to the same three observations used in the elementary LSE example: $(0,1)$, $(1,2)$, and $(2,2)$. With $\mathbf y=(1,2,2)^{\mathsf T}$ and the straight-line basis,

$$
\boldsymbol\Phi^{\mathsf T}\boldsymbol\Phi
=\begin{pmatrix}3&3\\3&5\end{pmatrix},
\qquad
\boldsymbol\Phi^{\mathsf T}\mathbf y
=\begin{pmatrix}5\\6\end{pmatrix}.
$$

The normal equations are $3w_0+3w_1=5$ and $3w_0+5w_1=6$. Subtracting them gives $2w_1=1$, so $w_1=1/2$ and $w_0=7/6$—**the same LSE** we calculated with the straight-line formulas. Hence

$$
\widehat f(x)=\frac76+\frac12x,
\qquad
\mathbf y-\boldsymbol\Phi\widehat{\mathbf w}
=\begin{pmatrix}-1/6\\1/3\\-1/6\end{pmatrix}.
$$

The residuals sum to zero, as expected from the column of ones, and their input-weighted sum is also zero: $0(-1/6)+1(1/3)+2(-1/6)=0$. Thus they satisfy both normal equations. The line need not pass through every observation; it is the line with the smallest total squared residual for these data.

## Practice problems

### Problem 1: The Gaussian connection

Let $Y\mid x,w_0,w_1,\beta\sim\mathcal N(w_0+w_1x,\beta^{-1})$, with $\beta=4$. At $x=3$, take $w_0=1$ and $w_1=2$.

1. Find the conditional mean, variance, and standard deviation of $Y$.
2. With $\beta$ held fixed, identify $\eta(x)$, $T(y)$, and $A_\beta(\eta)$ in the exponential-family representation.
3. Is this model linear in the input? Is it linear in the weights?

::::{admonition} Solution 1
:class: dropdown

At $x=3$, the mean is $w_0+w_1x=1+2(3)=7$. Since $\beta=4$, the variance is $1/\beta=1/4$ and the standard deviation is $\sqrt{1/4}=1/2$.

For fixed $\beta$, $\eta(x)=\beta(w_0+w_1x)$, $T(y)=y$, and $A_\beta(\eta)=\eta^2/(2\beta)$. At this particular input, $\eta(3)=4(7)=28$ and $A_4(28)=28^2/8=98$. The mean is a straight line in $x$, and it is linear in $w_0,w_1$ as well.
::::

### Problem 2: Building a design matrix

Use $\boldsymbol\phi(x)=(1,x,x^2)^{\mathsf T}$ for inputs $x_1=-1$, $x_2=0$, and $x_3=2$.

1. Write the design matrix and its dimensions.
2. For $\mathbf w=(1,-2,1)^{\mathsf T}$, calculate the predicted-value vector.
3. Explain why this is still called a linear model.

::::{admonition} Solution 2
:class: dropdown

Each row is $(1,x_n,x_n^2)$, so

$$
\boldsymbol\Phi
=\begin{pmatrix}1&-1&1\\1&0&0\\1&2&4\end{pmatrix}
\quad(3\times3),
\qquad
\boldsymbol\Phi\mathbf w
=\begin{pmatrix}1+2+1\\1\\1-4+4\end{pmatrix}
=\begin{pmatrix}4\\1\\1\end{pmatrix}.
$$

The prediction contains $x^2$, so it is nonlinear in $x$. But $1$, $x$, and $x^2$ are fixed basis functions, each multiplied by one weight. The model remains linear in the unknown weights.
::::

### Problem 3: From likelihood to an objective

For $N$ conditionally independent observations with common precision $\beta$, write the negative log-likelihood in terms of $\mathbf y$, $\boldsymbol\Phi$, and $\mathbf w$. Explain which terms can be ignored when finding the maximum-likelihood estimate of $\mathbf w$ with $\beta$ fixed. Then derive the normal equations by differentiating the remaining objective.

::::{admonition} Solution 3
:class: dropdown

Multiplying the $N$ Gaussian densities and taking minus the logarithm gives

$$
-\log p(\mathbf y\mid\boldsymbol\Phi,\mathbf w,\beta)
=-\frac N2\log\beta+\frac N2\log(2\pi)
+\frac\beta2\|\mathbf y-\boldsymbol\Phi\mathbf w\|_2^2.
$$

At fixed positive $\beta$, the first two terms are constant in $\mathbf w$; the factor $\beta$ does not change the minimizer either. We minimize $E_D(\mathbf w)=\frac12\|\mathbf y-\boldsymbol\Phi\mathbf w\|_2^2$. Expanding and differentiating gives

$$
E_D(\mathbf w)=\frac12\mathbf y^{\mathsf T}\mathbf y
-\mathbf w^{\mathsf T}\boldsymbol\Phi^{\mathsf T}\mathbf y
+\frac12\mathbf w^{\mathsf T}\boldsymbol\Phi^{\mathsf T}\boldsymbol\Phi\mathbf w,
\qquad
\nabla_{\mathbf w}E_D=-\boldsymbol\Phi^{\mathsf T}\mathbf y
+\boldsymbol\Phi^{\mathsf T}\boldsymbol\Phi\mathbf w.
$$

Setting the gradient to zero yields $\boldsymbol\Phi^{\mathsf T}\boldsymbol\Phi\widehat{\mathbf w}=\boldsymbol\Phi^{\mathsf T}\mathbf y$.
::::

### Problem 4: Fitting and recognizing a failure of uniqueness

Use $\boldsymbol\phi(x)=(1,x)^{\mathsf T}$ with $(x_1,y_1)=(0,1)$, $(x_2,y_2)=(1,2)$, and $(x_3,y_3)=(2,2)$.

1. Use the straight-line least-squares formulas to find the slope and intercept.
2. Compute $\boldsymbol\Phi^{\mathsf T}\boldsymbol\Phi$ and $\boldsymbol\Phi^{\mathsf T}\mathbf y$, then solve the normal equations. Are the fitted weights the same as in part 1?
3. Find the three residuals and verify that they are orthogonal to both columns of $\boldsymbol\Phi$.
4. Suppose every training input had instead been $x_n=1$. Why could you no longer determine $w_0$ and $w_1$ separately from these observations?

::::{admonition} Solution 4
:class: dropdown

The sample means are $\bar x=1$ and $\bar y=5/3$. The straight-line formulas give

$$
\widehat w_1
=\frac{(-1)(-2/3)+0+(1)(1/3)}{(-1)^2+0+1^2}
=\frac12,
\qquad
\widehat w_0=\bar y-\widehat w_1\bar x
=\frac53-\frac12=\frac76.
$$

Now use the matrix method. Here $\boldsymbol\Phi=\begin{pmatrix}1&0\\1&1\\1&2\end{pmatrix}$ and $\mathbf y=(1,2,2)^{\mathsf T}$. Matrix multiplication gives

$$
\boldsymbol\Phi^{\mathsf T}\boldsymbol\Phi
=\begin{pmatrix}3&3\\3&5\end{pmatrix},
\qquad
\boldsymbol\Phi^{\mathsf T}\mathbf y
=\begin{pmatrix}5\\6\end{pmatrix}.
$$

Solving $3w_0+3w_1=5$ and $3w_0+5w_1=6$ gives $\widehat{\mathbf w}=(7/6,1/2)^{\mathsf T}$, the same LSE as in part 1. The predictions are $(7/6,5/3,13/6)^{\mathsf T}$, so the residuals are $(-1/6,1/3,-1/6)^{\mathsf T}$. Their dot product with the first column, $(1,1,1)^{\mathsf T}$, is $-1/6+1/3-1/6=0$. Their dot product with the second column, $(0,1,2)^{\mathsf T}$, is $0+1/3-2/6=0$.

If all $x_n=1$, both columns of $\boldsymbol\Phi$ would be columns of ones. Every prediction would equal $w_0+w_1$, so the data could constrain that sum but not the two weights separately. The design matrix would lack full column rank, and $\boldsymbol\Phi^{\mathsf T}\boldsymbol\Phi$ would not be invertible.
::::

## Reading

Christopher M. Bishop, *Pattern Recognition and Machine Learning*, Chapter 3, pages 136–142 (especially Section 3.1 and Section 3.1.1).

## References

- [Christopher M. Bishop, *Pattern Recognition and Machine Learning* (author-hosted PDF)](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf).
