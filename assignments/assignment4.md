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

**(b) One step of IRLS (30 points).** Starting from $\mathbf w^{(0)}=(0,0)^{\mathsf T}$, perform **one IRLS update**, using the supplied recipe. Calculate the initial probabilities $\mathbf y^{(0)}$, weight matrix $\mathbf R^{(0)}$, effective targets (working responses) $\mathbf z^{(0)}$, and the two sides of the weighted normal equations. Solve for and report the updated weights $\mathbf w^{(1)}$. **No derivations are required. Stop after this single update: convergence is not expected.**

:::{admonition} IRLS recipe - supplied for this question
:class: tip

For this single update, use the **initial** weights to calculate

$$
a_n=\mathbf w^{\mathsf T}\boldsymbol\phi_n,
\qquad y_n=\sigma(a_n),
\qquad r_n=y_n(1-y_n),
\qquad z_n=a_n+\frac{t_n-y_n}{r_n}.
$$

Here $\boldsymbol\phi_n=(1,x_n)^{\mathsf T}$, $r_n$ is the observation's positive IRLS weight, and $z_n$ is an **effective target (working response)**, not a probability. Set $\mathbf R=\operatorname{diag}(r_1,\ldots,r_6)$ and collect the $z_n$ into $\mathbf z$. Find the new coefficients by solving the weighted least-squares normal equations

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
