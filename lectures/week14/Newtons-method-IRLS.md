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

Here $\sigma\$ is also called a "map". 

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

## Finding a solution for the binary cross-entry? 

1) To find minimize the negative binary cross-entry, we take the derivate $\nabla{E(w)}$: 
$$
\boxed{\nabla{E(\mathbf w)} 
=-\sum_{n=1}^{N}\left[t_n\log y_n+(1-t_n)\log(1-y_n)\right].}
$$


