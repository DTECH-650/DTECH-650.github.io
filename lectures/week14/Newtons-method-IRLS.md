# Lecture 22 - Iterative reweighted least square (Newton's method and IRLS)

In Lecture 20 and 21 we were learned about Discriminant functions and Logistic Regression, where the goal was to classify observed data. Specifically we asked how an operator of a building could answer the following question: *Will tomorrow be a high-demand day that requires extra capacity, or a normal day?* This questions implies that an exact kWh prediction is no longer the immediate goal of pattern recognition. Instead we want to assign one of two **class labels**. In lecture 21, we advised a probablistic perspective towards the classification problem, and introduced the logistic regression. 



:::{admonition} Guiding question
:class: tip

How can we solve a logistic regression - despite the absence of a closed form solution - iteratively using second-order gradient descent and why can each update be computed using weighted least squares?
:::

## Learning objectives

After completing this lecture, you should be able to:

Guiding question
How can curvature help us fit logistic regression, and why can each update be computed using weighted least squares?

Learning outcomes

After completing this lecture, students should be able to:
1. Explain why ordinary least squares has a direct solution, while logistic maximum-likelihood estimation generally requires iteration.
2. Distinguish gradient descent from Newton’s method by explaining the role of curvature.
3. Compute one Newton update and interpret the change in predicted probability.
4. Explain and implement an IRLS iteration, distinguishing the observed labels from the working targets and weights.

## Recap from lecture 20/21: From scores to probabilities 

In lecture 21, we had started with the score $a$ and defined it as function of weighted features: 

$$
a(\mathbf x)=\mathbf w^{\mathsf T}\boldsymbol\phi(\mathbf x).
$$

The first feature is usually $\phi_0(\mathbf x)=1$, so $w_0$ acts as a bias (it is also called intercept sometimes). If we use the original inputs plus the $\phi_0(\mathbf x)=1$, our feature vector is $\boldsymbol\phi(\mathbf x)=(1,x_1,\ldots,x_D)^{\mathsf T}$. 
We than converted $a$ to real values that lie between zero and one using the **logistic sigmoid**: 

$$
\boxed{\sigma(a)=\frac{1}{1+e^{-a}},\qquad
y(\mathbf x)=\sigma\!\left(\mathbf w^{\mathsf T}\boldsymbol\phi(\mathbf x)\right).}
$$

Here $\sigma$ is also called a "map" or a "linear operator". 

> Note: Add a plot on the sigmoid function so that it is easier for the students to understand! 

When $a=0$, $y=1/2$. Positive scores give $y>1/2$, while negative scores give $y<1/2$. For example, scores $-1$, $0$, and $1$ give probabilities of approximately $0.269$, $0.500$, and $0.731$. As $a$ increases without bound, $y$ approaches $1$; as it decreases without bound, $y$ approaches $0$. For finite weights and features it never equals either endpoint exactly.

## Recap from lecture 20/21: The Bernoulli model and cross-entry minimization

We defined the probability model for classification as following: 

Let $\mathcal D=\{(\mathbf x_n,t_n)\}_{n=1}^{N}$ be the labeled training set, and abbreviate

$$
\boldsymbol\phi_n=\boldsymbol\phi(\mathbf x_n),
\qquad a_n=\mathbf w^{\mathsf T}\boldsymbol\phi_n,
\qquad y_n=\sigma(a_n).
$$

Assuming the labels are conditionally independent given their inputs and the shared weights, their conditional likelihood is the product of their Bernoulli probabilities:

$$
\boxed{p(\mathbf t\mid\mathbf X,\mathbf w)
=\prod_{n=1}^{N}y_n^{t_n}(1-y_n)^{1-t_n}.}
$$

The inputs $\mathbf X=(\mathbf x_1,\ldots,\mathbf x_N)$ are treated as given. The model does **not** claim that all labels have the same success probability: $y_n$ can differ from day to day because the inputs differ. Their probabilities are tied together by the same $\mathbf w$.

Further, we learned that the **negative log-likelihood**, also called **binary cross-entropy** for these observed labels, is 

$$
\boxed{E(\mathbf w)
=-\sum_{n=1}^{N}\left[t_n\log y_n+(1-t_n)\log(1-y_n)\right].}
$$

## Finding a solution for the binary cross-entry by minimimizing cross-entry likelihood. 

To find a solution we have to minimize the negative binary cross-entry. To do so, we take the derivate $\nabla{E(w)}_w$. 

In the prior lecture we had derived this, using the chain rule: 

$$
\boxed{\nabla_{\mathbf w}E(\mathbf w)
=\sum_{n=1}^{N}(y_n-t_n)\boldsymbol\phi_n.}
$$

As discussed before, there is no closed form solution: Setting this gradient to zero does not produce a simple closed-form expression for $\mathbf w$, because each $y_n=\sigma(\mathbf w^{\mathsf T}\boldsymbol\phi_n)$ depends nonlinearly on the unknown weights. Instead of using gradient descent from lecture 18, we will now move to a second-order gradient-based method to solve this equation for $w$. 

## Newton's method

Gradient descent, introduced in Lecture 18, uses first derivatives
to determine an update. Newton's method also uses second derivatives,
which describe the curvature of the objective.

Newton's method can be used to find roots of a function. For
optimization, we apply it to the derivative of the objective:
we seek a stationary point where $E'(w)=0$.

### A quadratic approximation

First, consider an objective with one unknown weight $w$.
Near the current value $w$, a second-order Taylor expansion gives

$$
E(w+\Delta w)
\approx
E(w)+E'(w)\Delta w
+\frac12 E''(w)(\Delta w)^2.
$$

Here:

- $E'(w)$ is the slope of the objective at the current weight.
- $E''(w)$ is its curvature.
- $\Delta w$ is the proposed change in the weight.

We choose $\Delta w$ by minimizing this quadratic approximation.
Differentiating the approximation with respect to $\Delta w$
and setting the result to zero gives

$$
E'(w)+E''(w)\Delta w=0.
$$

When $E''(w)>0$, the approximation has a minimum, located at

$$
\Delta w=-\frac{E'(w)}{E''(w)}.
$$

The Newton update is therefore

$$
\boxed{
w_{\mathrm{new}}
=
w-\frac{E'(w)}{E''(w)}.
}
$$

Unlike gradient descent, Newton's method uses curvature to scale
the update. For the same slope, greater positive curvature produces
a smaller step.

### From one weight to several weights

With several weights, the gradient $\nabla E(\mathbf w)$ contains
the first derivatives. The Hessian $\mathbf H(\mathbf w)$ contains
the second derivatives:

$$
H_{ij}(\mathbf w)
=
\frac{\partial^2 E(\mathbf w)}
{\partial w_i\,\partial w_j}.
$$

The quadratic approximation becomes

$$
E(\mathbf w+\Delta\mathbf w)
\approx
E(\mathbf w)
+
\nabla E(\mathbf w)^{\mathsf T}\Delta\mathbf w
+
\frac12
\Delta\mathbf w^{\mathsf T}
\mathbf H(\mathbf w)
\Delta\mathbf w.
$$

When the Hessian is positive definite [[^positive-definite]], this approximation has
a unique minimum. Its location satisfies

$$
\boxed{
\mathbf H(\mathbf w)\Delta\mathbf w
=
-\nabla E(\mathbf w).
}
$$

We solve this linear system for $\Delta\mathbf w$ and then update:

$$
\mathbf w_{\mathrm{new}}
=
\mathbf w+\Delta\mathbf w.
$$

The two methods therefore use different updates:

$$
\begin{aligned}
\text{Gradient descent:}\qquad
\mathbf w_{\mathrm{new}}
&=
\mathbf w-\eta\nabla E(\mathbf w),
\\[2mm]
\text{Newton's method:}\qquad
\mathbf w_{\mathrm{new}}
&=
\mathbf w-\mathbf H(\mathbf w)^{-1}\nabla E(\mathbf w).
\end{aligned}
$$

The inverse makes the mathematical comparison clear. In Python,
we solve the linear system rather than explicitly calculate
the matrix inverse.

### Why a shorter step may be needed

The quadratic approximation describes the objective locally.
Its minimum need not be the minimum of the actual objective,
and a full Newton step can sometimes increase the loss.

We can therefore shorten the step:

$$
\mathbf w_{\mathrm{new}}
=
\mathbf w+\lambda\Delta\mathbf w,
\qquad 0<\lambda\leq1.
$$

Here $\lambda=1$ gives the full Newton step. If that step increases
the objective, we can try a smaller value of $\lambda$. This is
called a damped Newton update.

Next, we calculate the gradient and Hessian for logistic regression.

## 3. One Newton update for a simple classification model

### 3.1 Start with an intercept-only model

We first simplify our building example by temporarily leaving out
temperature and occupancy. Suppose we observe four days, of which
three are high-demand days:

$$
\mathbf t=(1,1,1,0)^{\mathsf T}.
$$

Without input features, the model contains only an intercept $w_0$.
It assigns the same high-demand probability to every day:

$$
y=\sigma(w_0)=\frac{1}{1+e^{-w_0}}.
$$

The observed labels differ, but their modeled probability is shared.
Our task is to estimate that probability by learning $w_0$.

The cross-entropy objective is

$$
E(w_0)=-3\log y-\log(1-y),
\qquad y=\sigma(w_0).
$$

### 3.2 Calculate one update

Recall that each observation contributes $(y_n-t_n)\boldsymbol\phi_n$
to the gradient. Here the only feature is the constant $1$, and every
predicted probability equals $y$. Therefore,

$$
E'(w_0)
=
\sum_{n=1}^{4}(y-t_n)
=
4y-3.
$$

Using the sigmoid derivative gives the second derivative:

$$
E''(w_0)
=
4\frac{dy}{dw_0}
=
4y(1-y).
$$

Start at $w_0=0$. This gives

$$
y=\sigma(0)=0.5.
$$

At this starting point, the gradient and curvature are

$$
E'(0)=4(0.5)-3=-1,
$$

$$
E''(0)=4(0.5)(1-0.5)=1.
$$

The Newton update is therefore

$$
w_{0,\mathrm{new}}
=
w_0-\frac{E'(w_0)}{E''(w_0)}
=
0-\frac{-1}{1}
=
1.
$$

The updated high-demand probability is

$$
\boxed{
y_{\mathrm{new}}
=
\sigma(1)
\approx0.731.
}
$$

### 3.3 Interpret the change

The estimated high-demand probability increased from $0.5$ to
approximately $0.731$. This moves it toward the observed proportion
of high-demand days, $3/4=0.75$.

For this particularly simple model, we can also find the optimum
directly by setting the gradient to zero:

$$
4y-3=0
\quad\Longrightarrow\quad
y=0.75.
$$

The corresponding intercept is

$$
\widehat w_0
=
\log\left(\frac{0.75}{1-0.75}\right)
=
\log 3
\approx1.099.
$$

Our first update has moved toward this solution but has not reached
it. We would now recompute the probability, gradient, and curvature
at $w_0=1$ and perform another update.

This intercept-only example has a direct solution. Logistic regression
with several input features generally requires iteration.


## 4. Newton's method for logistic regression

We now return to a model that uses features such as temperature
and occupancy. Each day can have a different high-demand probability:

$$
y_n
=
\sigma(\mathbf w^{\mathsf T}\boldsymbol\phi_n).
$$

### 4.1 Write the gradient using the design matrix

Let the design matrix contain one row of features per observation:

$$
\boldsymbol\Phi
=
\begin{pmatrix}
\boldsymbol\phi_1^{\mathsf T}\\
\vdots\\
\boldsymbol\phi_N^{\mathsf T}
\end{pmatrix}.
$$

With $M$ features, including the intercept feature,
$\boldsymbol\Phi$ has shape $N\times M$.

Collect the predicted probabilities and observed labels into

$$
\mathbf y=
\begin{pmatrix}
y_1\\
\vdots\\
y_N
\end{pmatrix},
\qquad
\mathbf t=
\begin{pmatrix}
t_1\\
\vdots\\
t_N
\end{pmatrix}.
$$

The gradient can then be written as

$$
\boxed{
\nabla E(\mathbf w)
=
\boldsymbol\Phi^{\mathsf T}(\mathbf y-\mathbf t).
}
$$

This is the same gradient as before, expressed using matrix notation.
The vector $\mathbf y-\mathbf t$ contains the differences between
predicted probabilities and observed labels.

### 4.2 Obtain the Hessian

Recall that

$$
a_n=\mathbf w^{\mathsf T}\boldsymbol\phi_n,
\qquad
\frac{dy_n}{da_n}=y_n(1-y_n).
$$

Applying the chain rule gives

$$
\nabla_{\mathbf w}y_n
=
y_n(1-y_n)\boldsymbol\phi_n.
$$

Differentiating the gradient once more therefore gives

$$
\mathbf H
=
\sum_{n=1}^{N}
y_n(1-y_n)
\boldsymbol\phi_n\boldsymbol\phi_n^{\mathsf T}.
$$

Define the diagonal matrix

$$
\mathbf R
=
\operatorname{diag}(r_1,\ldots,r_N),
\qquad
r_n=y_n(1-y_n).
$$

The Hessian becomes

$$
\boxed{
\mathbf H
=
\boldsymbol\Phi^{\mathsf T}\mathbf R\boldsymbol\Phi.
}
$$

The entries of $\mathbf R$ change as the current probabilities change.
They describe how observations contribute to the local curvature
of the objective.

For example,

$$
y_n=0.5
\quad\Longrightarrow\quad
r_n=0.25,
$$

whereas

$$
y_n=0.9
\quad\Longrightarrow\quad
r_n=0.09.
$$

These weights are largest when the current probability is $0.5$
and become smaller as it approaches $0$ or $1$. They are not ratings
of how trustworthy the observations are.

For finite scores, each $r_n>0$. If the columns of
$\boldsymbol\Phi$ are linearly independent, the Hessian is positive
definite and the Newton linear system has a unique solution.

### 4.3 Assemble the update

Substitute the gradient and Hessian into the Newton equation:

$$
\left(
\boldsymbol\Phi^{\mathsf T}\mathbf R\boldsymbol\Phi
\right)
\Delta\mathbf w
=
\boldsymbol\Phi^{\mathsf T}(\mathbf t-\mathbf y).
$$

After solving for $\Delta\mathbf w$, update the weights:

$$
\boxed{
\mathbf w_{\mathrm{new}}
=
\mathbf w+\Delta\mathbf w.
}
$$

Each iteration follows the same sequence:

1. Use the current weights to calculate the probabilities.
2. Calculate the gradient and Hessian.
3. Solve for the change in weights.
4. Update the weights and repeat.

The design matrix and labels remain fixed. The probabilities,
gradient, and Hessian are recalculated.


## 5. Why this is called iteratively reweighted least squares

Newton's update for logistic regression can also be written as
a weighted least-squares calculation. This equivalent form is
called **iteratively reweighted least squares**, or **IRLS**.

### 5.1 Using working targets as illustration: 

The current scores are

$$
\mathbf a=\boldsymbol\Phi\mathbf w.
$$

Define a temporary vector of working targets:

$$
\boxed{
\mathbf z
=
\mathbf a+\mathbf R^{-1}(\mathbf t-\mathbf y).
}
$$

For an individual observation,

$$
z_n
=
a_n+\frac{t_n-y_n}{y_n(1-y_n)}.
$$

The working target starts with the current score and adds a correction
based on the difference between the label and predicted probability.

The quantities have different roles:

| Quantity | Meaning |
|---|---|
| $t_n$ | Observed class label, either $0$ or $1$ |
| $y_n$ | Current predicted high-demand probability |
| $a_n$ | Current score before applying the sigmoid |
| $z_n$ | Temporary target on the score scale |

Working targets are not probabilities and need not lie between
zero and one. They are computational quantities, not new observations.

For our four-day example, the starting intercept was $w_0=0$.
Thus $a_n=0$, $y_n=0.5$, and $r_n=0.25$ for every day.

For a high-demand day,

$$
z_n
=
0+\frac{1-0.5}{0.25}
=
2.
$$

For a normal day,

$$
z_n
=
0+\frac{0-0.5}{0.25}
=
-2.
$$

The working-target vector is therefore

$$
\mathbf z=(2,2,2,-2)^{\mathsf T}.
$$

### 5.2 Expressing the Newton update as weighted least squares

Start with the Newton equation:

$$
\boldsymbol\Phi^{\mathsf T}\mathbf R\boldsymbol\Phi
\Delta\mathbf w
=
\boldsymbol\Phi^{\mathsf T}(\mathbf t-\mathbf y).
$$

Since $\Delta\mathbf w=\mathbf w_{\mathrm{new}}-\mathbf w$,
we can rearrange it:

$$
\begin{aligned}
\boldsymbol\Phi^{\mathsf T}\mathbf R\boldsymbol\Phi
\mathbf w_{\mathrm{new}}
&=
\boldsymbol\Phi^{\mathsf T}\mathbf R\boldsymbol\Phi\mathbf w
+
\boldsymbol\Phi^{\mathsf T}(\mathbf t-\mathbf y)\\
&=
\boldsymbol\Phi^{\mathsf T}\mathbf R
\left[
\boldsymbol\Phi\mathbf w+
\mathbf R^{-1}(\mathbf t-\mathbf y)
\right].
\end{aligned}
$$

The bracketed expression is $\mathbf z$. Consequently,

$$
\boxed{
\left(
\boldsymbol\Phi^{\mathsf T}\mathbf R\boldsymbol\Phi
\right)
\mathbf w_{\mathrm{new}}
=
\boldsymbol\Phi^{\mathsf T}\mathbf R\mathbf z.
}
$$

This is the normal equation for the weighted least-squares problem

$$
\min_{\mathbf v}
\frac12
(\mathbf z-\boldsymbol\Phi\mathbf v)^{\mathsf T}
\mathbf R
(\mathbf z-\boldsymbol\Phi\mathbf v).
$$

Equivalently, we minimize

$$
\frac12\sum_{n=1}^{N}
r_n\left(z_n-\boldsymbol\phi_n^{\mathsf T}\mathbf v\right)^2.
$$

Here $\mathbf v$ denotes the candidate weights in this temporary
least-squares problem. Its solution is the next weight vector.

For the intercept-only example, all weights $r_n=0.25$ are equal.
The fitted intercept is therefore the mean of the working targets:

$$
w_{0,\mathrm{new}}
=
\frac{2+2+2-2}{4}
=
1.
$$

This is exactly the Newton update we calculated in Section 3.

### 5.3 What gets repeated?

At each IRLS iteration:

1. Calculate the current scores and probabilities.
2. Recalculate the weights $r_n=y_n(1-y_n)$.
3. Recalculate the working targets $z_n$.
4. Solve the weighted least-squares problem.
5. Repeat until the convergence criterion is satisfied.

The original labels and input features stay fixed. Both the
working targets and the least-squares weights change.

:::{admonition} What objective are we minimizing?
:class: important

The original objective remains Bernoulli cross-entropy.

Weighted least squares provides a way to calculate each Newton
update iteratively. 
:::

This derivation gives the full Newton step.

## 6. Comparing gradient descent and Newton's method in Python

We now return to the building classification problem with temperature
and occupancy as features.

The purpose of this exercise is to inspect the updates. Use NumPy
for the matrix calculations, SciPy's sigmoid function, and Matplotlib
for the figures.

### 6.1 The optimization problem revisited: 

Use the same observations, feature scaling, and initial weights
for both methods.

For each day, let us construct

$$
\boldsymbol\phi_n=
\begin{pmatrix}
1\\
\widetilde x_{n,1}\\
\widetilde x_{n,2}
\end{pmatrix},
$$

where the two input features are scaled temperature and occupancy.

We can start both methods at

$$
\mathbf w^{(0)}=\mathbf 0.
$$

Every initial score is then zero, so every initial probability
is $0.5$.

> Note: If we assume we have a dataset whose design matrix has linearly independent columns
and whose classes cannot be completely or quasi-completely separated, we can ensures that the unregularized fit has a finite solution.
If the classes are perfectly separable, the likelihood can improve
as the weights grow without reaching a finite maximum.

Both methods minimize the same summed cross-entropy:

$$
E(\mathbf w)
=
-\sum_{n=1}^{N}
\left[
t_n\log y_n+(1-t_n)\log(1-y_n)
\right].
$$

### 6.2 Recording ◊what changes across iterations

At the starting point and after each update, record:

- The cross-entropy $E(\mathbf w)$.
- The gradient norm $\|\nabla E(\mathbf w)\|_2$.
- The high-demand probability for one selected day.

For the selected day's feature vector $\boldsymbol\phi_*$,
the probability is

$$
y_*^{(k)}
=
\sigma\left(
(\mathbf w^{(k)})^{\mathsf T}\boldsymbol\phi_*
\right).
$$

Keep this day fixed so that changes in its probability reflect
changes in the fitted weights.

For gradient descent, choose a learning rate and apply

$$
\mathbf w^{(k+1)}
=
\mathbf w^{(k)}-\eta\nabla E(\mathbf w^{(k)}).
$$

For Newton's method, calculate the current Hessian, solve for
the Newton direction, and update the weights. Reduce the step
if the full step increases cross-entropy.

Stop when the gradient norm is sufficiently small, or when a
specified maximum number of iterations is reached.

**Figure:** On the left, the plot shows
cross-entropy against iteration for both Netwon and Gradient descent. On the right,
the plot shows the selected day's high-demand probability against iteration.


### 6.3 Let us interpret the comparison

> Figure 

As you can see in the figure, for this unregularized problem with a unique finite optimum,
both methods approach the same fit when they converge.

Newton's method requires fewer iterations because it uses curvature. However, each iteration requires constructing a Hessian
and solving a linear system. Fewer iterations therefore do not automatically mean less computing time.


## 7. Summary and practice

Gradient descent uses the slope of the objective. Newton's method
also uses curvature to construct a local quadratic approximation.

For logistic regression,

$$
\nabla E
=
\boldsymbol\Phi^{\mathsf T}(\mathbf y-\mathbf t),
\qquad
\mathbf H
=
\boldsymbol\Phi^{\mathsf T}\mathbf R\boldsymbol\Phi.
$$

Rearranging the Newton update gives IRLS: repeatedly solve a
weighted least-squares problem with updated working targets
and weights.

### Exercise 1: Calculate one Newton update

Suppose five days have labels

$$
\mathbf t=(1,1,1,1,0)^{\mathsf T}.
$$

Use an intercept-only model and start at $w_0=0$.

1. Calculate the starting probability.
2. Calculate the gradient and second derivative.
3. Calculate the updated intercept and probability.
4. Compare the updated probability with the observed proportion.

:::{admonition} Solution
:class: dropdown

The starting probability is $y=0.5$. The derivatives are

$$
E'(w_0)=5y-4,
\qquad
E''(w_0)=5y(1-y).
$$

At the starting point,

$$
E'(0)=-1.5,
\qquad
E''(0)=1.25.
$$

Thus,

$$
w_{0,\mathrm{new}}
=
0-\frac{-1.5}{1.25}
=
1.2,
$$

and

$$
y_{\mathrm{new}}
=
\sigma(1.2)
\approx0.769.
$$

The probability moves toward the observed proportion $4/5=0.8$
but does not reach it in one update.
:::

### Exercise 2: Interpret the IRLS quantities

For the original four-day example, start at $w_0=0$.

1. Calculate the working target for a high-demand day and a
   normal day.
2. Explain why a working target can be negative.
3. Identify which quantities remain fixed across iterations.
4. State the objective that IRLS is minimizing.

:::{admonition} Solution
:class: dropdown

The working targets are $2$ for a high-demand day and $-2$ for
a normal day.

They can be negative because they are temporary targets on the
score scale, not probabilities.

The observed labels and input features remain fixed. The weights
of the regression model, predicted probabilities, working targets,
and diagonal entries of $\mathbf R$ change.

The original objective is Bernoulli cross-entropy.
:::

### Exercise 3: Compare the algorithms

Using the Python example:

1. Run gradient descent with three different learning rates.
2. Compare its loss curves with the Newton loss curve.
3. Report the final probability for the same selected day.
4. Explain why comparing iteration counts alone does not establish
   which algorithm is computationally faster.

Discuss whether each run appears to have converged. A learning
rate that is too small can give slow progress; a large learning
rate can produce oscillation or increasing loss.


[^positive-definite]: A Hessian is **positive definite** if
    $\mathbf v^{\mathsf T}\mathbf H\mathbf v>0$ for every nonzero
    direction $\mathbf v$. This means the quadratic approximation
    curves upward in every direction, giving it a unique minimum.
    **Positive semidefinite** allows
    $\mathbf v^{\mathsf T}\mathbf H\mathbf v=0$ in some directions:
    the approximation can be flat along those directions, so a
    unique minimum is not guaranteed. With one weight, these
    conditions reduce to $E''(w)>0$ and $E''(w)\geq0$, respectively.