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

When the Hessian is positive definite, this approximation has
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


