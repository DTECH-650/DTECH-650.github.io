# Assignment 4 — Fitting Linear and Logistic Regression Models

This assignment connects linear regression and gradient descent (Lectures 17–18) with linear discriminants, logistic regression, and generalized linear models (Lectures 20–21). It contains **two questions worth 100 points total**. The IRLS update needed for Question 2 is provided below.

## Purdue Honor Pledge

> “As a Boilermaker pursuing academic excellence, I pledge to be honest and true in all that I do. Accountable together—We are Purdue.”

**When submitting this assignment via Gradescope, we assume that you commit to this honor pledge and submit your own work considering the instructions below.**

## Instructions

- This is an individual assignment. Show your setup, main calculations, and interpretation; a numerical answer alone may receive only partial credit.
- **Compute means work through the steps by hand.** An arithmetic calculator may be used for exponentials, logarithms, and numerical arithmetic. There is no coding question: do not replace the requested iterations with a software model fit or optimization routine.
- The two tables below are fixed, **simulated datasets** supplied for this exercise. Use these values directly; you do not need to generate new samples. Six observations make the arithmetic manageable, not sufficient to validate either model for deployment.
- Include an intercept in both models. Use the feature vector $\boldsymbol\phi(x)=(1,x)^{\mathsf T}$ and coefficient vector $\mathbf w=(w_0,w_1)^{\mathsf T}$. Do not standardize the supplied inputs again.
- Use the exact objectives, initial weights, and update rules specified. In particular, Question 1 uses the **mean half-squared error**, whereas Question 2 uses the **total** negative log-likelihood. Keep at least six decimal places during iterative calculations and report final coefficients to four decimal places.
- Questions 1(b) and 2(b) each ask for only one update. Stop after reporting the updated weights; convergence is not expected in either part.
- Submit your work as a PDF on Gradescope and link the pages to the corresponding questions. No submission deadline is specified in this draft.
- Please do your best not to rely on an LLM (for example, ChatGPT) to solve these questions; the purpose is to practice the fitting steps and explain what they mean.

## Questions

### Question 1: A conveyor's electricity use - linear regression (50 points)

An engineering team runs six simulated bench tests of a conveyor. Let $v$ be its speed in metres per minute and define a scaled input $x=(v-10)/2$. Thus $x=0$ corresponds to $10$ m/min, and increasing $x$ by one means increasing the speed by $2$ m/min. The continuous target $y$ is the measured electrical power in kW.

| Test $n$ | Scaled speed $x_n$ | Power $y_n$ (kW) |
| --- | ---: | ---: |
| 1 | $-2$ | $0.2$ |
| 2 | $-1$ | $0.8$ |
| 3 | $0$ | $1.9$ |
| 4 | $0$ | $2.1$ |
| 5 | $1$ | $2.8$ |
| 6 | $2$ | $4.2$ |

Use the model $\widehat y(x)=w_0+w_1x$ and the objective

$$
J(\mathbf w)=\frac{1}{2N}\sum_{n=1}^{N}(w_0+w_1x_n-y_n)^2,
\qquad N=6.
$$

**(a) Closed-form least squares (25 points).** Write the design matrix $\boldsymbol\Phi$, with row $n$ equal to $(1,x_n)$, and the target vector $\mathbf y$. Use the supplied least-squares formula

$$
\widehat{\mathbf w}_{\mathrm{LS}}
=(\boldsymbol\Phi^{\mathsf T}\boldsymbol\Phi)^{-1}\boldsymbol\Phi^{\mathsf T}\mathbf y.
$$

Compute $\boldsymbol\Phi^{\mathsf T}\boldsymbol\Phi$ and $\boldsymbol\Phi^{\mathsf T}\mathbf y$, then calculate the two fitted coefficients and predict the power at $v=11$ m/min. You may equivalently solve $(\boldsymbol\Phi^{\mathsf T}\boldsymbol\Phi)\widehat{\mathbf w}=\boldsymbol\Phi^{\mathsf T}\mathbf y$ directly rather than writing a matrix inverse. **No derivation of the formula or normal equations is required.**

**(b) One step of batch gradient descent (25 points).** Differentiate $J$ with respect to $w_0$ and $w_1$. Starting from $\mathbf w^{(0)}=(0,0)^{\mathsf T}$, use learning rate $\alpha=3/4$ and perform **one batch update**:

$$
\mathbf w^{(1)}=\mathbf w^{(0)}-\alpha\nabla J(\mathbf w^{(0)}).
$$

Use all six observations to evaluate the gradient at the initial weights, and compute both new coefficients from the **same old coefficient vector**. Show the gradient calculation and report the updated weights $\mathbf w^{(1)}$. **Stop after this single update: convergence is not expected.**

### Question 2: Maintenance alerts - logistic regression, discriminants, and GLMs (50 points)

A maintenance team wants to predict whether a machine will need a repair after a vibration check. The scaled vibration reading is $x$, and an inspection provides the binary target $t$: $t=1$ means repair is needed and $t=0$ means it is not. The six simulated inspections are

| Inspection $n$ | Scaled vibration $x_n$ | Repair label $t_n$ |
| --- | ---: | ---: |
| 1 | $-1$ | $0$ |
| 2 | $-1$ | $0$ |
| 3 | $-1$ | $1$ |
| 4 | $1$ | $0$ |
| 5 | $1$ | $1$ |
| 6 | $1$ | $1$ |

Repeated input values represent separate inspections with the same recorded vibration level. Vibration alone need not determine the outcome: other unrecorded factors can affect whether repair is needed. Model the label directly using

$$
a(x)=w_0+w_1x,
\qquad
y(x)=\sigma(a(x))=\frac{1}{1+e^{-a(x)}},
\qquad
T\mid x,\mathbf w\sim\operatorname{Bernoulli}(y(x)).
$$

Here $y(x)$ is a probability, not the continuous power target from Question 1. Assume the six labels are conditionally independent given their inputs and the shared weights. Fit the weights without a prior or regularization penalty.

**(a) Discriminant and probability (8 points).** Before fitting, explain the difference between the raw discriminant score $a(x)$, the model probability $y(x)$, and the observed label $t$. Why should the raw score not be reported as a probability?

**(b) One step of IRLS (30 points).** Starting from $\mathbf w^{(0)}=(0,0)^{\mathsf T}$, perform **one IRLS update**, using the supplied recipe. Calculate the initial probabilities $\mathbf y^{(0)}$, weight matrix $\mathbf R^{(0)}$, working responses $\mathbf z^{(0)}$, and the two sides of the weighted normal equations. Solve for and report the updated weights $\mathbf w^{(1)}$. **No derivations are required. Stop after this single update: convergence is not expected.**

:::{admonition} IRLS recipe - supplied for this question
:class: tip

For this single update, use the **initial** weights to calculate

$$
a_n=\mathbf w^{\mathsf T}\boldsymbol\phi_n,
\qquad y_n=\sigma(a_n),
\qquad r_n=y_n(1-y_n),
\qquad z_n=a_n+\frac{t_n-y_n}{r_n}.
$$

Here $\boldsymbol\phi_n=(1,x_n)^{\mathsf T}$, $r_n$ is the observation's positive IRLS weight, and $z_n$ is a **working response**, not a probability. Set $\mathbf R=\operatorname{diag}(r_1,\ldots,r_6)$ and collect the $z_n$ into $\mathbf z$. Find the new coefficients by solving the weighted least-squares normal equations

$$
(\boldsymbol\Phi^{\mathsf T}\mathbf R\boldsymbol\Phi)\mathbf w^{\mathrm{new}}
=\boldsymbol\Phi^{\mathsf T}\mathbf R\mathbf z.
$$

Do not add a learning-rate multiplier to this IRLS step. For this exercise, solve the weighted normal equations once and stop.
:::

**(c) Why are both models GLMs? (12 points).** Write the models in Questions 1 and 2 in the common form

$$
a(x)=\mathbf w^{\mathsf T}\boldsymbol\phi(x),
\qquad
\mu(x)=f(a(x)),
\qquad
\mu(x)=\mathbb E[\text{target}\mid x,\mathbf w].
$$

For each model, identify its conditional target distribution, mean $\mu(x)$, inverse link $f$, and link $g=f^{-1}$. For Question 1, use the independent Gaussian observation model from Lecture 17 with a common fixed variance $\beta^{-1}>0$.

## Instructor solutions and grading guide

**Private grading copy:** this section is intended to remain on the local `assign_4` branch. Do not publish it or merge it into a student-facing branch. A future question-only release should omit this entire section. Suggested points below total 100; allow small numerical differences caused by rounding.

### Question 1: A conveyor's electricity use (50 points)

#### (a) Closed-form least squares (25 points)

With one constant feature and one scaled-speed feature, the design matrix and target vector are

$$
\boldsymbol\Phi=
\begin{pmatrix}
1&-2\\
1&-1\\
1&0\\
1&0\\
1&1\\
1&2
\end{pmatrix},
\qquad
\mathbf y=\begin{pmatrix}0.2\\0.8\\1.9\\2.1\\2.8\\4.2\end{pmatrix}.
$$

The dataset has $\sum x_n=0$, $\sum x_n^2=10$, $\sum y_n=12$, and

$$
\sum x_ny_n=(-2)(0.2)+(-1)(0.8)+(0)(1.9)+(0)(2.1)+(1)(2.8)+(2)(4.2)=10.
$$

Therefore

$$
\boldsymbol\Phi^{\mathsf T}\boldsymbol\Phi
=\begin{pmatrix}6&0\\0&10\end{pmatrix},
\qquad
\boldsymbol\Phi^{\mathsf T}\mathbf y=\begin{pmatrix}12\\10\end{pmatrix}.
$$

The two equations are $6\widehat w_0=12$ and $10\widehat w_1=10$, giving

$$
\boxed{\widehat{\mathbf w}_{\mathrm{LS}}=(2,1)^{\mathsf T},
\qquad \widehat y(x)=2+x.}
$$

At $v=11$ m/min, the scaled input is $x=(11-10)/2=0.5$. The predicted power is $\widehat y=2+1(0.5)=2.5$ kW.

**Grading:** design matrix and targets, 6; two matrix products, 10; fitted coefficients, 6; prediction, 3. No derivation or conceptual explanation is required for this part.

#### (b) One step of batch gradient descent (25 points)

Let $e_n=w_0+w_1x_n-y_n$. Differentiating each squared term yields

$$
\frac{\partial J}{\partial w_0}=\frac16\sum_{n=1}^{6}e_n,
\qquad
\frac{\partial J}{\partial w_1}=\frac16\sum_{n=1}^{6}e_nx_n.
$$

Using the four sums from part (a), these expressions simplify to

$$
\boxed{\frac{\partial J}{\partial w_0}=w_0-2,
\qquad
\frac{\partial J}{\partial w_1}=\frac53(w_1-1).}
$$

The mean in the objective matters: omitting $1/6$ would multiply the gradient by $6$ and make the supplied learning rate inappropriate.

At $\mathbf w^{(0)}=(0,0)^{\mathsf T}$, the gradient is $(-2,-5/3)^{\mathsf T}$. Thus

$$
\mathbf w^{(1)}
=\begin{pmatrix}0\\0\end{pmatrix}
-\frac34\begin{pmatrix}-2\\-5/3\end{pmatrix}
=\begin{pmatrix}1.5\\1.25\end{pmatrix}.
$$

The requested answer is therefore $\boxed{\mathbf w^{(1)}=(1.5000,1.2500)^{\mathsf T}}$. These weights are not the closed-form solution $(2,1)^{\mathsf T}$, and that is expected after only one update. No further iterations or convergence check are required.

**Grading:** partial derivatives, 8; evaluating both gradient components at the initial weights using the full batch, 9; applying the learning rate and reporting both updated weights, 8. The simplified general gradient above is convenient but is not required if the student evaluates the two sums correctly.

### Question 2: Maintenance alerts (50 points)

#### (a) From a discriminant to a probability (8 points)

The raw score $a=w_0+w_1x$ can take any real value and indicates which side of a decision boundary the input lies on. It need not lie between zero and one, so it is not itself a probability. The sigmoid transforms it to $y=\sigma(a)\in(0,1)$, the modeled probability of repair. The observed target $t$ is the actual inspection outcome, either $0$ or $1$, not a fractional prediction.

#### (b) One step of IRLS (30 points)

**First IRLS update.** All scores are initially $0$, so

$$
\mathbf y^{(0)}=\begin{pmatrix}1/2\\1/2\\1/2\\1/2\\1/2\\1/2\end{pmatrix},
\qquad
\mathbf R^{(0)}=\frac14\mathbf I_6,
\qquad
\mathbf z^{(0)}=\begin{pmatrix}-2\\-2\\2\\-2\\2\\2\end{pmatrix}.
$$

Here $\mathbf I_6$ is the $6\times6$ identity matrix. For example, inspection 1 has $z_1=0+(0-1/2)/(1/4)=-2$, whereas inspection 3 has $z_3=0+(1-1/2)/(1/4)=2$. Working responses can be negative or greater than one because they are not probabilities.

For this question the design matrix is

$$
\boldsymbol\Phi=
\begin{pmatrix}1&-1\\1&-1\\1&-1\\1&1\\1&1\\1&1\end{pmatrix}.
$$

The weighted normal equations at the initial iterate are

$$
\underbrace{\begin{pmatrix}1.5&0\\0&1.5\end{pmatrix}}_{\boldsymbol\Phi^{\mathsf T}\mathbf R^{(0)}\boldsymbol\Phi}
\mathbf w^{(1)}
=\underbrace{\begin{pmatrix}0\\1\end{pmatrix}}_{\boldsymbol\Phi^{\mathsf T}\mathbf R^{(0)}\mathbf z^{(0)}}.
$$

Solving the two equations gives $w_0^{(1)}=0/1.5=0$ and $w_1^{(1)}=1/1.5=2/3$. The requested answer is therefore

$$
\boxed{\mathbf w^{(1)}=(0,2/3)^{\mathsf T}\approx(0.0000,0.6667)^{\mathsf T}.}
$$

These are the weights after **one update**, not a claim of convergence. No further iterations, predictions, or convergence check are required.

**Grading:** initial probabilities and IRLS weight matrix, 8; working responses, 8; weighted normal-equation calculations, 8; solving for the updated coefficients, 6. No likelihood, loss, or update-rule derivation is required. Accept tiny nonzero intercepts caused by rounding.

#### (c) The shared GLM form (12 points)

Both models use the same kind of linear predictor,

$$
a(x)=\mathbf w^{\mathsf T}\boldsymbol\phi(x)=w_0+w_1x,
\qquad \boldsymbol\phi(x)=(1,x)^{\mathsf T}.
$$

They differ in the distribution of the target and the inverse link that turns this score into a conditional mean:

| Item | Question 1: Gaussian linear regression | Question 2: Logistic regression |
| --- | --- | --- |
| Conditional target model | $Y\mid x,\mathbf w,\beta\sim\mathcal N(a(x),\beta^{-1})$ | $T\mid x,\mathbf w\sim\operatorname{Bernoulli}(\sigma(a(x)))$ |
| Conditional mean $\mu(x)$ | $\mathbb E[Y\mid x,\mathbf w,\beta]=a(x)$ | $\mathbb E[T\mid x,\mathbf w]=\sigma(a(x))$ |
| Inverse link $f(a)$ | $a$ (identity) | $\sigma(a)=1/(1+e^{-a})$ |
| Link $g(\mu)$ | $\mu$ (identity) | $\log[\mu/(1-\mu)]$ (logit) |

The Gaussian target is continuous, and the model permits any real mean. The Bernoulli target is $0$ or $1$, and its conditional mean is the probability of $1$, so it must lie between zero and one. Both conditional distributions belong to the exponential family. They share a linear score, but need different transformations and probability models for their different targets.

For independent Gaussian observations with common fixed precision $\beta$, the negative log-likelihood is

$$
-\log p(\mathbf y\mid\mathbf X,\mathbf w,\beta)
=\frac N2\log\!\left(\frac{2\pi}{\beta}\right)
+\frac\beta2\sum_{n=1}^{N}(y_n-a(x_n))^2
=\frac N2\log\!\left(\frac{2\pi}{\beta}\right)+\beta N J(\mathbf w).
$$

The first term is independent of $\mathbf w$ and $\beta N$ is a positive constant, so minimizing this negative log-likelihood gives the same coefficients as minimizing $J$. For Bernoulli targets, the PMF instead gives $-\sum_n[t_n\log y_n+(1-t_n)\log(1-y_n)]$, the binary cross-entropy. The **choice of target distribution**, not merely the appearance of $w_0+w_1x$, determines the loss.

**Grading:** target distributions and means, 4; links and inverse links, 4; shared structure and explanation of the two likelihood-based losses, 4.
