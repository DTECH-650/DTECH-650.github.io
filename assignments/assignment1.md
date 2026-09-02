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


