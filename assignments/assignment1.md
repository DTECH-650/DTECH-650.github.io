# Assignment 1 — Mathematical and Statistical Foundations

This assignment covers material from Weeks 1–3. It contains **eight questions worth 100 points total**.

## Purdue Honor Pledge

> “As a Boilermaker pursuing academic excellence, I pledge to be honest and true in all that I do. Accountable together—We are Purdue.” 

**When submitting this assignment via Gradescope, we assume that you commit to this honor pledge and submit your own work considering the instructions below.**  

## Instructions

- Show the main steps in your reasoning. A correct numerical answer without supporting work may receive only partial credit.
- Unless explicitly stated otherwise, **compute** means compute step-by step meaning by hand (for example do not use a scientific calculator and computer program that implicitly does integration or differentation but write out the steps). Your solution should document your steps and your logical reasoning. We would like to evaluate your understanding rather than the final solution. 
- Only Question 8 is a coding question; do not use code to solve Questions 1–7. Please submit readable, reproducible code along with the requested figure and interpretation.
- You may use an arithmetic calculator when solving Questions 1–7 but not a computer program that does it all at once.
- Unless stated otherwise, use natural logarithms for likelihood calculations.
- For a geometric random variable, let $X$ denote the trial number of the first success, so $X\in\{1,2,\ldots\}$.
- If you write your answers on paper or with an ipad and handwriting notetaking software, that is appreciated (You can also type up your final answer neatly with a word processor and use the build in symbols and equation functions or if you are a pro use latex or overleaf). Please make sure that your handwriting is legible, so that we can properly evaluate your effort.  
- All assignments are to be submitted on Gradescope (see website for link or syllabus) as a PDF. 

## Questions

### Question 1: Vectors, matrices, and eigenvectors (10 points)

Let

$$
\mathbf{u}=
\begin{bmatrix}
1\\
-2\\
2
\end{bmatrix},
\qquad
\mathbf{v}=
\begin{bmatrix}
2\\
1\\
0
\end{bmatrix},
\qquad
B=
\begin{bmatrix}
1 & 0 & 2\\
-1 & 3 & 1
\end{bmatrix},
$$

and

$$
A=
\begin{bmatrix}
2 & 1\\
1 & 2
\end{bmatrix}.
$$

1. State the dimensions of $\mathbf{u}$, $\mathbf{v}$, and $B$. (1 point)
2. Compute $\mathbf{u}^{\mathsf T}\mathbf{v}$. What does the result tell you geometrically? (2 points)
3. Compute $\lVert\mathbf{u}\rVert_1$, $\lVert\mathbf{u}\rVert_2$, and $\lVert\mathbf{v}\rVert_2$. (2 points)
4. Compute $B\mathbf{u}$. (1 point)
5. Find the eigenvalues of $A$ and one eigenvector corresponding to each eigenvalue. Verify one eigenvector by direct multiplication. (4 points)

### Question 2: Partial derivatives and gradients (10 points)

Consider the function

$$
L(w_1,w_2)=(w_1+2w_2-3)^2+w_1^2.
$$

1. Compute $\frac{\partial L}{\partial w_1}$ and $\frac{\partial L}{\partial w_2}$. (3 points)
2. Write the gradient $\nabla L(w_1,w_2)$ as a column vector. (2 points)
3. Evaluate the gradient at $(w_1,w_2)=(1,1)$. (2 points)
4. Starting at $(1,1)$, take one gradient-descent step with learning rate $\eta=0.25$. Give the new point and verify that the value of $L$ decreases. (3 points)

### Question 3: Events, independence, and Bayes' theorem (15 points)

A monitoring system uses two alarms, $A$ and $B$, to detect a fault $F$. The fault prevalence and alarm behavior are

$$
P(F)=0.05,
$$

$$
P(A\mid F)=0.90,
\qquad
P(B\mid F)=0.80,
$$

and

$$
P(A\mid F^c)=0.10,
\qquad
P(B\mid F^c)=0.20.
$$

Assume that $A$ and $B$ are conditionally independent given $F$, and also conditionally independent given $F^c$.

1. Describe a sample space whose outcomes record the fault status and the states of both alarms. Define the events $F$, $A$, and $B$ in words. (2 points)
2. Draw a two-event Venn diagram for $A$ and $B$. Shade the event that **exactly one** alarm activates and write that event using set notation. (2 points)
3. Compute $P(A\cap B\mid F)$ and $P(A\cap B\mid F^c)$. (2 points)
4. Use the law of total probability to compute $P(A\cap B)$. (2 points)
5. Use Bayes' theorem to compute $P(F\mid A\cap B)$. (2 points)
6. Compute $P(A)$ and $P(B)$. Are $A$ and $B$ marginally independent? Explain briefly. (3 points)
7. Compute $P(A\cup B)$. (2 points)

### Question 4: Discrete random variables (12 points)

Answer all three parts. Clearly identify the distribution and parameters you use.

#### 4(a) Binomial model (4 points)

Each of 10 independently inspected components is defective with probability $0.20$. Let $X$ be the number of defective components.

1. Write the PMF of $X$.
2. Compute $P(X=2)$.
3. Compute $E[X]$ and $\operatorname{Var}(X)$.

#### 4(b) Poisson model and marginalization (4 points)

Let $Z$ describe a server's operating mode. The server is in normal mode with probability $0.75$ and busy mode with probability $0.25$. Conditional on the mode,

$$
X\mid Z=\text{normal}\sim\operatorname{Poisson}(2),
\qquad
X\mid Z=\text{busy}\sim\operatorname{Poisson}(5),
$$

where $X$ is the number of requests received in one minute.

1. Marginalize over $Z$ to compute $P(X=0)$.
2. Compute $E[X]$ by averaging the two conditional expectations.

#### 4(c) Geometric model (4 points)

Independent transmission attempts succeed with probability $p=0.25$. Let $X$ be the trial number of the first successful transmission.

1. Write the PMF of $X$.
2. Compute $P(X>4)$.
3. Compute $E[X]$ and $\operatorname{Var}(X)$.

### Question 5: Continuous random variables (10 points)

#### 5(a) Gaussian measurements (4 points)

A calibrated sensor measurement is modeled as

$$
X\sim\mathcal{N}(50,4),
$$

where the second parameter is the variance.

1. Write the PDF of $X$.
2. State $E[X]$ and $\operatorname{Var}(X)$.

#### 5(b) Exponential waiting time (6 points)

The time $T$ in hours until a component fails is modeled as

$$
T\sim\operatorname{Exponential}(\lambda=0.5).
$$

1. Write the PDF of $T$. Then find its CDF by setting up and evaluating

   $$
   F_T(t)=\int_{-\infty}^{t}f_T(s)\,ds.
   $$
2. Compute $P(T\leq3)$.
3. State $E[T]$ and $\operatorname{Var}(T)$.
4. Use memorylessness to compute $P(T>5\mid T>2)$.

### Question 6: Joint distributions and covariance (18 points)

#### 6(a) Marginalization and conditional independence (10 points)

Let $Z\in\{0,1\}$ be an unobserved operating mode with

$$
P(Z=0)=P(Z=1)=\frac12.
$$

Conditional on $Z$, the continuous variables $X$ and $Y$ are independent. Their conditional distributions are

$$
X\mid Z=0\sim\operatorname{Uniform}(0,1),
\qquad
Y\mid Z=0\sim\operatorname{Uniform}(0,1),
$$

and

$$
X\mid Z=1\sim\operatorname{Uniform}(1,2),
\qquad
Y\mid Z=1\sim\operatorname{Uniform}(1,2).
$$

1. Write $f_{X,Y\mid Z}(x,y\mid z)$ for each value of $z$ and show the factorization expressing conditional independence. (2 points)
2. Marginalize over $Z$ to find the joint density $f_{X,Y}(x,y)$. (3 points)
3. Find the marginal densities $f_X(x)$ and $f_Y(y)$. (3 points)
4. Are $X$ and $Y$ marginally independent? Demonstrate your answer mathematically and explain the role of $Z$. (2 points)

#### 6(b) Jointly Gaussian variables and covariance (8 points)

Let $U$ and $V$ be independent standard Gaussian random variables, and define

$$
X=1+2U,
\qquad
Y=-1+U+V.
$$

The pair $(X,Y)$ is jointly Gaussian.

1. Compute $E[X]$, $E[Y]$, $\operatorname{Var}(X)$, and $\operatorname{Var}(Y)$. (3 points)
2. Compute $\operatorname{Cov}(X,Y)$. (2 points)
3. Write the covariance matrix of $(X,Y)$. (1 point)
4. Are $X$ and $Y$ independent? Justify your answer and briefly interpret the dependence between them. (2 points)

### Question 7: Gaussian maximum likelihood (15 points)

Suppose $x_1,\ldots,x_N$ are IID observations from a one-dimensional Gaussian distribution with unknown mean $\mu$ and unknown variance $\sigma^2$:

$$
x_n\sim\mathcal{N}(\mu,\sigma^2).
$$

1. Write the likelihood $L(\mu,\sigma^2)$ and log-likelihood $\ell(\mu,\sigma^2)$. (4 points)
2. Differentiate the log-likelihood with respect to $\mu$ and derive the maximum-likelihood estimator $\hat\mu_{\mathrm{ML}}$. (3 points)
3. Differentiate with respect to $\sigma^2$ and derive $\hat\sigma^2_{\mathrm{ML}}$. (4 points)
4. Compute both estimates for the observations $2,4,4,6$. (2 points)
5. In one or two sentences, explain the distinction between the Gaussian probability density and the Gaussian likelihood. (2 points)

### Question 8: Coding investigation (10 points)

An industrial vibration sensor records the root-mean-square vibration amplitude of a machine in millimeters per second. Under normal operating conditions, model one reading as

$$
X\sim\mathcal{N}(10,2^2).
$$

Thus the expected vibration amplitude is $10$ mm/s and the standard deviation is $2$ mm/s. A reading above $13$ mm/s is considered unusually high and triggers further inspection.

Complete the following investigation:

1. Explain what $\mu=10$ and $\sigma=2$ represent in this application. (1 point)
2. Initialize a NumPy random-number generator with seed `65000` and generate exactly 10,000 observations from the specified Gaussian distribution. (1 point)
3. Compute the empirical mean and variance and compare them with the theoretical mean and variance. (2 points)
4. Estimate $P(X>13)$ from the simulated observations and compute its theoretical value. (2 points)
5. Plot a normalized histogram of the simulated observations and overlay the theoretical Gaussian PDF. Mark the inspection threshold $x=13$ on the plot. (2 points)
6. In three to five sentences, discuss how closely the simulation agrees with theory and give one reason a Gaussian distribution might fail to describe real vibration measurements. (2 points)
---

<!-- ## Solutions

::::{admonition} Solution 1
:class: dropdown

1. Both $\mathbf{u}$ and $\mathbf{v}$ are $3\times1$ column vectors. The matrix $B$ is $2\times3$.

2. The dot product is

   $$
   \mathbf{u}^{\mathsf T}\mathbf{v}
   =(1)(2)+(-2)(1)+(2)(0)=0.
   $$

   Because both vectors are nonzero and their dot product is zero, they are orthogonal.

3. The requested norms are

   $$
   \lVert\mathbf{u}\rVert_1=|1|+|-2|+|2|=5,
   $$

   $$
   \lVert\mathbf{u}\rVert_2
   =\sqrt{1^2+(-2)^2+2^2}=3,
   $$

   and

   $$
   \lVert\mathbf{v}\rVert_2
   =\sqrt{2^2+1^2+0^2}=\sqrt{5}.
   $$

4. Matrix–vector multiplication gives

   $$
   B\mathbf{u}
   =
   \begin{bmatrix}
   1 & 0 & 2\\
   -1 & 3 & 1
   \end{bmatrix}
   \begin{bmatrix}
   1\\-2\\2
   \end{bmatrix}
   =
   \begin{bmatrix}
   5\\-5
   \end{bmatrix}.
   $$

5. The characteristic equation is

   $$
   \det(A-\lambda I)
   =
   \begin{vmatrix}
   2-\lambda & 1\\
   1 & 2-\lambda
   \end{vmatrix}
   =(2-\lambda)^2-1=0.
   $$

   Thus the eigenvalues are $\lambda_1=3$ and $\lambda_2=1$. Corresponding eigenvectors can be chosen as

   $$
   \mathbf{q}_1=
   \begin{bmatrix}1\\1\end{bmatrix},
   \qquad
   \mathbf{q}_2=
   \begin{bmatrix}1\\-1\end{bmatrix}.
   $$

   For example,

   $$
   A\mathbf{q}_1
   =
   \begin{bmatrix}3\\3\end{bmatrix}
   =3\mathbf{q}_1,
   $$

   which verifies the first eigenpair. Any nonzero scalar multiple of either eigenvector is also valid.
::::

::::{admonition} Solution 2
:class: dropdown

Let $r=w_1+2w_2-3$. Then $L=r^2+w_1^2$.

1. Applying the chain rule,

   $$
   \frac{\partial L}{\partial w_1}
   =2r+2w_1
   =4w_1+4w_2-6,
   $$

   and

   $$
   \frac{\partial L}{\partial w_2}
   =4r
   =4w_1+8w_2-12.
   $$

2. Therefore,

   $$
   \nabla L(w_1,w_2)
   =
   \begin{bmatrix}
   4w_1+4w_2-6\\
   4w_1+8w_2-12
   \end{bmatrix}.
   $$

3. At $(1,1)$,

   $$
   \nabla L(1,1)
   =
   \begin{bmatrix}2\\0\end{bmatrix}.
   $$

4. A gradient-descent step is

   $$
   \begin{bmatrix}w_1^{\mathrm{new}}\\w_2^{\mathrm{new}}\end{bmatrix}
   =
   \begin{bmatrix}1\\1\end{bmatrix}
   -0.25
   \begin{bmatrix}2\\0\end{bmatrix}
   =
   \begin{bmatrix}0.5\\1\end{bmatrix}.
   $$

   The old and new objective values are

   $$
   L(1,1)=1
   $$

   and

   $$
   L(0.5,1)=(0.5+2-3)^2+0.5^2=0.5.
   $$

   Thus this step decreases the objective from $1$ to $0.5$.
::::

::::{admonition} Solution 3
:class: dropdown

1. One suitable sample space is

   $$
   \Omega=\{F,F^c\}\times\{A,A^c\}\times\{B,B^c\},
   $$

   which contains eight possible combinations. Here $F$ is the event that a fault is present, $A$ is the event that alarm $A$ activates, and $B$ is the event that alarm $B$ activates.

2. Exactly one alarm activates when either $A$ occurs without $B$, or $B$ occurs without $A$:

   $$
   (A\cap B^c)\cup(A^c\cap B).
   $$

   In a two-circle Venn diagram, shade the $A$-only and $B$-only regions, leaving the intersection unshaded.

3. Conditional independence gives

   $$
   P(A\cap B\mid F)
   =P(A\mid F)P(B\mid F)
   =(0.90)(0.80)=0.72,
   $$

   and

   $$
   P(A\cap B\mid F^c)
   =P(A\mid F^c)P(B\mid F^c)
   =(0.10)(0.20)=0.02.
   $$

4. Marginalizing over the two possible fault states,

   $$
   \begin{aligned}
   P(A\cap B)
   &=P(A\cap B\mid F)P(F)
     +P(A\cap B\mid F^c)P(F^c)\\
   &=(0.72)(0.05)+(0.02)(0.95)\\
   &=0.036+0.019=0.055.
   \end{aligned}
   $$

5. Bayes' theorem gives

   $$
   P(F\mid A\cap B)
   =\frac{P(A\cap B\mid F)P(F)}{P(A\cap B)}
   =\frac{0.72(0.05)}{0.055}
   =\frac{36}{55}
   \approx0.655.
   $$

6. By the law of total probability,

   $$
   P(A)=(0.90)(0.05)+(0.10)(0.95)=0.14,
   $$

   and

   $$
   P(B)=(0.80)(0.05)+(0.20)(0.95)=0.23.
   $$

   If the alarms were marginally independent, then $P(A\cap B)$ would equal

   $$
   P(A)P(B)=(0.14)(0.23)=0.0322.
   $$

   This is not equal to $0.055$, so the alarms are not marginally independent. They are conditionally independent within each fault state, but the shared fault state creates marginal dependence.

7. Using the sum rule,

   $$
   P(A\cup B)
   =P(A)+P(B)-P(A\cap B)
   =0.14+0.23-0.055
   =0.315.
   $$
::::

::::{admonition} Solution 4
:class: dropdown

#### 4(a) Binomial model

Here $X\sim\operatorname{Binomial}(n=10,p=0.20)$.

1. Its PMF is

   $$
   P(X=k)=\binom{10}{k}(0.20)^k(0.80)^{10-k},
   \qquad k=0,1,\ldots,10.
   $$

2. Therefore,

   $$
   P(X=2)
   =\binom{10}{2}(0.20)^2(0.80)^8
   \approx0.3020.
   $$

3. The expectation and variance are

   $$
   E[X]=np=(10)(0.20)=2
   $$

   and

   $$
   \operatorname{Var}(X)=np(1-p)
   =(10)(0.20)(0.80)=1.6.
   $$

#### 4(b) Poisson model and marginalization

1. Marginalizing over $Z$,

   $$
   \begin{aligned}
   P(X=0)
   &=P(X=0\mid Z=\text{normal})P(Z=\text{normal})\\
   &\quad+P(X=0\mid Z=\text{busy})P(Z=\text{busy})\\
   &=e^{-2}(0.75)+e^{-5}(0.25)\\
   &\approx0.1032.
   \end{aligned}
   $$

2. Since a Poisson variable has expectation equal to its rate,

   $$
   E[X]=(0.75)(2)+(0.25)(5)=2.75.
   $$

   Notice that the unconditional distribution is a mixture of two Poisson distributions; it is not generally a single Poisson distribution.

#### 4(c) Geometric model

Here $X\sim\operatorname{Geometric}(p=0.25)$ with support $1,2,\ldots$.

1. The PMF is

   $$
   P(X=k)=(1-p)^{k-1}p=(0.75)^{k-1}(0.25),
   \qquad k=1,2,\ldots.
   $$

2. Use the complement of the CDF:

   $$
   P(X>4)=1-P(X\leq4).
   $$

   For the geometric PMF,

   $$
   \begin{aligned}
   P(X\leq4)
   &=\sum_{k=1}^{4}(0.75)^{k-1}(0.25)\\
   &=1-(0.75)^4.
   \end{aligned}
   $$

   Therefore,

   $$
   \begin{aligned}
   P(X>4)
   &=1-\left[1-(0.75)^4\right]\\
   &=(0.75)^4\\
   &\approx0.3164.
   \end{aligned}
   $$

3. The expectation and variance are

   $$
   E[X]=\frac1p=4
   $$

   and

   $$
   \operatorname{Var}(X)=\frac{1-p}{p^2}
   =\frac{0.75}{0.25^2}=12.
   $$
::::

::::{admonition} Solution 5
:class: dropdown

#### 5(a) Gaussian measurements

The standard deviation is $\sigma=\sqrt{4}=2$.

1. The PDF is

   $$
   f_X(x)
   =\frac{1}{2\sqrt{2\pi}}
   \exp\left(-\frac{(x-50)^2}{8}\right),
   \qquad -\infty<x<\infty.
   $$

2. The moments are

   $$
   E[X]=50,
   \qquad
   \operatorname{Var}(X)=4.
   $$

#### 5(b) Exponential waiting time

1. The PDF is

   $$
   f_T(t)=
   \begin{cases}
   0.5e^{-0.5t}, & t\geq0,\\
   0, & t<0.
   \end{cases}
   $$

   For $t\geq0$, integrating the PDF gives

   $$
   F_T(t)
   =\int_0^t0.5e^{-0.5s}\,ds
   =1-e^{-0.5t}.
   $$

   Thus

   $$
   F_T(t)=
   \begin{cases}
   0, & t<0,\\
   1-e^{-0.5t}, & t\geq0.
   \end{cases}
   $$

2. The probability of failure within three hours is

   $$
   P(T\leq3)=F_T(3)=1-e^{-1.5}\approx0.7769.
   $$

3. For an exponential variable,

   $$
   E[T]=\frac1\lambda=2
   $$

   and

   $$
   \operatorname{Var}(T)=\frac1{\lambda^2}=4.
   $$

4. By memorylessness,

   $$
   P(T>5\mid T>2)
   =P(T>5-2)
   =P(T>3)
   =e^{-1.5}
   \approx0.2231.
   $$
::::

::::{admonition} Solution 6
:class: dropdown

#### 6(a) Marginalization and conditional independence

1. Recall that a $\operatorname{Uniform}(a,b)$ random variable has density $1/(b-a)$ on $[a,b]$. Both intervals in this problem have width 1, so every conditional marginal density has height 1 on its support.

   Conditional on $Z=0$, independence therefore gives

   $$
   f_{X,Y\mid Z}(x,y\mid0)
   =f_{X\mid Z}(x\mid0)f_{Y\mid Z}(y\mid0)
   =
   \begin{cases}
   1, & 0\leq x\leq1,\ 0\leq y\leq1,\\
   0, & \text{otherwise}.
   \end{cases}
   $$

   This conditional joint density integrates to 1:

   $$
   \int_0^1\int_0^1
   f_{X,Y\mid Z}(x,y\mid0)\,dy\,dx
   =\int_0^1\int_0^1 1\,dy\,dx
   =1.
   $$

   Similarly,

   $$
   f_{X,Y\mid Z}(x,y\mid1)
   =f_{X\mid Z}(x\mid1)f_{Y\mid Z}(y\mid1)
   =
   \begin{cases}
   1, & 1\leq x\leq2,\ 1\leq y\leq2,\\
   0, & \text{otherwise}.
   \end{cases}
   $$

   Its normalization is

   $$
   \int_1^2\int_1^2
   f_{X,Y\mid Z}(x,y\mid1)\,dy\,dx
   =\int_1^2\int_1^2 1\,dy\,dx
   =1.
   $$

   These factorizations express $X\perp Y\mid Z$. The mode probability $P(Z=z)=1/2$ is not included in a density conditioned on $Z=z$; it is introduced when the modes are combined in the next part.

2. Marginalizing the discrete variable $Z$ gives

   $$
   \begin{aligned}
   f_{X,Y}(x,y)
   &=\sum_{z\in\{0,1\}}f_{X,Y\mid Z}(x,y\mid z)P(Z=z)\\
   &=\frac12f_{X,Y\mid Z}(x,y\mid0)
     +\frac12f_{X,Y\mid Z}(x,y\mid1).
   \end{aligned}
   $$

   Endpoint values do not affect continuous probabilities. To keep the two regions from overlapping at 1, use the equivalent half-open representation

   $$
   f_{X,Y}(x,y)=
   \begin{cases}
   \frac12, & 0\leq x<1,\ 0\leq y<1,\\
   \frac12, & 1\leq x\leq2,\ 1\leq y\leq2,\\
   0, & \text{otherwise}.
   \end{cases}
   $$

   This marginal joint density is also normalized: the two occupied squares each have area 1 and height $1/2$, so their total probability is

   $$
   \left(\frac12\right)(1)+\left(\frac12\right)(1)=1.
   $$

3. A marginal density is obtained by integrating the joint density over every possible value of the other variable. For $X$,

   $$
   f_X(x)=\int_{-\infty}^{\infty}f_{X,Y}(x,y)\,dy.
   $$

   Consider the possible values of $x$ separately.

   - If $0\leq x<1$, the point $(x,y)$ can lie only in the lower square. The joint density is $1/2$ for $0\leq y<1$, so

     $$
     f_X(x)
     =\int_0^1\frac12\,dy
     =\frac12.
     $$

   - If $1\leq x\leq2$, the point $(x,y)$ can lie only in the upper square. The joint density is $1/2$ for $1\leq y\leq2$, so

     $$
     f_X(x)
     =\int_1^2\frac12\,dy
     =\frac12.
     $$

   - If $x<0$ or $x>2$, the joint density is zero for every $y$, so $f_X(x)=0$.

   Combining these cases gives

   $$
   f_X(x)=
   \begin{cases}
   \frac12, & 0\leq x\leq2,\\
   0, & \text{otherwise},
   \end{cases}
   $$

   The calculation for $Y$ is analogous:

   $$
   f_Y(y)=\int_{-\infty}^{\infty}f_{X,Y}(x,y)\,dx.
   $$

   For $0\leq y<1$, integrate across the lower square:

   $$
   f_Y(y)=\int_0^1\frac12\,dx=\frac12.
   $$

   For $1\leq y\leq2$, integrate across the upper square:

   $$
   f_Y(y)=\int_1^2\frac12\,dx=\frac12.
   $$

   Outside $[0,2]$, the result is zero. Therefore,

   $$
   f_Y(y)=
   \begin{cases}
   \frac12, & 0\leq y\leq2,\\
   0, & \text{otherwise}.
   \end{cases}
   $$

   Both marginals integrate to 1; for example,

   $$
   \int_{-\infty}^{\infty}f_X(x)\,dx
   =\int_0^2\frac12\,dx
   =1.
   $$

   Thus

   $$
   X\sim\operatorname{Uniform}(0,2),
   \qquad
   Y\sim\operatorname{Uniform}(0,2).
   $$

   Although each marginal is uniform over $[0,2]$, the joint density is not uniform over the full square $[0,2]^2$: it assigns density only to the lower-left and upper-right unit squares. This distinction is what produces the marginal dependence examined next.

4. The variables are not marginally independent. For example,

   $$
   P(X<1,Y<1)=P(Z=0)=\frac12,
   $$

   whereas

   $$
   P(X<1)P(Y<1)=\left(\frac12\right)\left(\frac12\right)=\frac14.
   $$

   Equivalently, on the two occupied squares $f_{X,Y}(x,y)=1/2$, while $f_X(x)f_Y(y)=1/4$. The shared, unobserved mode $Z$ makes small values of $X$ occur with small values of $Y$, and large values with large values. Thus conditional independence given $Z$ does not imply marginal independence after $Z$ is hidden.

#### 6(b) Jointly Gaussian variables and covariance

Because $E[U]=E[V]=0$, $\operatorname{Var}(U)=\operatorname{Var}(V)=1$, and $U$ and $V$ are independent:

1. The means are

   $$
   E[X]=1+2E[U]=1
   $$

   and

   $$
   E[Y]=-1+E[U]+E[V]=-1.
   $$

   The variances are

   $$
   \operatorname{Var}(X)=\operatorname{Var}(2U)=4
   $$

   and

   $$
   \operatorname{Var}(Y)
   =\operatorname{Var}(U+V)
   =1+1=2.
   $$

2. Start from the definition of covariance:

   $$
   \operatorname{Cov}(X,Y)
   =E\left[(X-E[X])(Y-E[Y])\right].
   $$

   From part 1, $E[X]=1$ and $E[Y]=-1$. Therefore, the centered random variables are

   $$
   X-E[X]=(1+2U)-1=2U
   $$

   and

   $$
   Y-E[Y]=(-1+U+V)-(-1)=U+V.
   $$

   Notice that the constants $1$ and $-1$ disappear after centering. Substituting the centered expressions into the covariance definition gives

   $$
   \begin{aligned}
   \operatorname{Cov}(X,Y)
   &=E\left[(2U)(U+V)\right]\\
   &=E\left[2U^2+2UV\right]\\
   &=2E[U^2]+2E[UV].
   \end{aligned}
   $$

   Because $U$ is standard Gaussian, $E[U]=0$ and $\operatorname{Var}(U)=1$. Hence

   $$
   E[U^2]
   =\operatorname{Var}(U)+(E[U])^2
   =1.
   $$

   Also, $U$ and $V$ are independent, so the expectation of their product factors:

   $$
   E[UV]=E[U]E[V]=(0)(0)=0.
   $$

   It follows that

   $$
   \boxed{
   \operatorname{Cov}(X,Y)
   =2(1)+2(0)=2
   }.
   $$

   The positive covariance comes from the shared $U$ term: an increase in $U$ raises both $X$ and $Y$. The independent term $V$ appears only in $Y$, so it contributes nothing to the covariance between them.

3. The covariance matrix is

   $$
   \Sigma=
   \begin{bmatrix}
   4 & 2\\
   2 & 2
   \end{bmatrix}.
   $$

4. The variables are not independent. In particular, their covariance is nonzero. The positive covariance arises because both variables contain the common random component $U$; larger values of $U$ tend to make both $X$ and $Y$ larger.
::::

::::{admonition} Solution 7
:class: dropdown

1. The IID likelihood is

   $$
   \begin{aligned}
   L(\mu,\sigma^2)
   &=\prod_{n=1}^N
   \frac{1}{\sqrt{2\pi\sigma^2}}
   \exp\left[-\frac{(x_n-\mu)^2}{2\sigma^2}\right]\\
   &=(2\pi\sigma^2)^{-N/2}
   \exp\left[-\frac{1}{2\sigma^2}
   \sum_{n=1}^N(x_n-\mu)^2\right].
   \end{aligned}
   $$

   Taking the logarithm gives

   $$
   \ell(\mu,\sigma^2)
   =-\frac{N}{2}\log(2\pi)
    -\frac{N}{2}\log(\sigma^2)
    -\frac{1}{2\sigma^2}\sum_{n=1}^N(x_n-\mu)^2.
   $$

2. Differentiating with respect to $\mu$,

   $$
   \frac{\partial\ell}{\partial\mu}
   =\frac{1}{\sigma^2}\sum_{n=1}^N(x_n-\mu).
   $$

   Setting the derivative to zero gives

   $$
   \sum_{n=1}^Nx_n-N\mu=0,
   $$

   and therefore

   $$
   \boxed{\hat\mu_{\mathrm{ML}}
   =\frac1N\sum_{n=1}^Nx_n}.
   $$

3. Treat $s=\sigma^2$ as the parameter. Then

   $$
   \frac{\partial\ell}{\partial s}
   =-\frac{N}{2s}
   +\frac{1}{2s^2}\sum_{n=1}^N(x_n-\mu)^2.
   $$

   Setting this derivative to zero and multiplying by $2s^2$ yields

   $$
   -Ns+\sum_{n=1}^N(x_n-\mu)^2=0.
   $$

   Evaluating at $\mu=\hat\mu_{\mathrm{ML}}$ gives

   $$
   \boxed{\hat\sigma^2_{\mathrm{ML}}
   =\frac1N\sum_{n=1}^N
   (x_n-\hat\mu_{\mathrm{ML}})^2}.
   $$

4. For $2,4,4,6$,

   $$
   \hat\mu_{\mathrm{ML}}
   =\frac{2+4+4+6}{4}=4.
   $$

   The sum of squared residuals is

   $$
   (2-4)^2+(4-4)^2+(4-4)^2+(6-4)^2=8,
   $$

   so

   $$
   \hat\sigma^2_{\mathrm{ML}}=\frac84=2.
   $$

5. As a probability density, the parameters are fixed and the observation is the variable. As a likelihood, the observed data are fixed and the parameters $\mu$ and $\sigma^2$ are varied to determine which parameter values make those data most plausible.
::::

::::{admonition} Solution 8
:class: dropdown

Here $\mu=10$ mm/s is the expected vibration amplitude under normal operation, while $\sigma=2$ mm/s describes the typical spread of readings around that mean. The event of interest is $X>13$, corresponding to a reading that triggers further inspection.

```python
import math

import matplotlib.pyplot as plt
import numpy as np

# Model and simulation settings
mu = 10.0
sigma = 2.0
n = 10_000
threshold = 13.0

# A fixed seed makes the result reproducible.
rng = np.random.default_rng(65000)
x = rng.normal(loc=mu, scale=sigma, size=n)

# Empirical and theoretical moments
empirical_mean = x.mean()
empirical_variance = x.var(ddof=0)
theoretical_mean = mu
theoretical_variance = sigma**2

# P(X > threshold), estimated empirically and calculated theoretically.
# For a Gaussian variable, P(X > a) can be written using erfc.
empirical_probability = np.mean(x > threshold)
z = (threshold - mu) / sigma
theoretical_probability = 0.5 * math.erfc(z / math.sqrt(2.0))

print(f"Empirical mean:       {empirical_mean:.4f}")
print(f"Theoretical mean:     {theoretical_mean:.4f}")
print(f"Empirical variance:   {empirical_variance:.4f}")
print(f"Theoretical variance: {theoretical_variance:.4f}")
print(f"Empirical P(X > 13):  {empirical_probability:.4f}")
print(f"Theoretical P(X > 13): {theoretical_probability:.4f}")

# Normalized histogram and theoretical Gaussian PDF
grid = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 400)
pdf = (
    1.0 / (sigma * np.sqrt(2.0 * np.pi))
    * np.exp(-0.5 * ((grid - mu) / sigma) ** 2)
)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(
    x,
    bins=40,
    density=True,
    alpha=0.65,
    color="steelblue",
    edgecolor="white",
    label="Simulated measurements",
)
ax.plot(grid, pdf, color="darkred", linewidth=2.5, label="Theoretical PDF")
ax.axvline(threshold, color="black", linestyle="--", label="Event threshold")
ax.set_xlabel("Vibration amplitude (mm/s)")
ax.set_ylabel("Density")
ax.set_title("Simulated vibration readings and Gaussian model")
ax.legend()
plt.show()
```

With the fixed seed above, representative results are

```text
Empirical mean:       9.9821
Theoretical mean:     10.0000
Empirical variance:   3.9725
Theoretical variance: 4.0000
Empirical P(X > 13):  0.0656
Theoretical P(X > 13): 0.0668
```

The empirical mean, variance, and tail probability are close to their theoretical values, as expected for a sample of 10,000 observations. Small discrepancies remain because the sample is finite. A Gaussian model is plausible when many small additive effects contribute to the vibration reading, but real vibration data may be skewed, contain large transient spikes, or change as the machine wears. In addition, a Gaussian distribution assigns some probability to physically impossible negative vibration amplitudes, although that probability is extremely small for these parameters.
:::: -->
