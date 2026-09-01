# Assignment 2

## Questions

### Question 1: Information theory (9 points)

Use logarithms to base 2, so all information quantities are measured in bits.

1. Let $X\sim\operatorname{Bernoulli}(1/2)$. Compute the surprisal of observing either outcome and the entropy $H(X)$. (2 points)
2. Let $Y=X$. Compute $H(X,Y)$, $H(Y\mid X)$, and $I(X;Y)$. (3 points)
3. The true distribution over two classes is $P=(1/2,1/2)$, while a model reports $Q=(3/4,1/4)$. Compute the cross-entropy $H(P,Q)$ and the KL divergence $D_{\mathrm{KL}}(P\Vert Q)$. You may use $\log_2(3/4)\approx-0.415$ and $\log_2(1/4)=-2$. (3 points)
4. State the connection between cross-entropy and negative log-likelihood for labeled classification data. (1 point)

---

## Solutions

::::{admonition} Solution 1
:class: dropdown

1. Since each outcome has probability $1/2$, its surprisal is

   $$
   I(x)=-\log_2\left(\frac12\right)=1\text{ bit}.
   $$

   The entropy is

   $$
   H(X)
   =-\sum_xp(x)\log_2p(x)
   =-2\left(\frac12\log_2\frac12\right)
   =1\text{ bit}.
   $$

2. Because $Y=X$, the only possible pairs are $(0,0)$ and $(1,1)$, each with probability $1/2$. Consequently,

   $$
   H(X,Y)=1\text{ bit}.
   $$

   Knowing $X$ determines $Y$ exactly, so

   $$
   H(Y\mid X)=0.
   $$

   Therefore,

   $$
   I(X;Y)=H(Y)-H(Y\mid X)=1-0=1\text{ bit}.
   $$

3. The cross-entropy is

   $$
   \begin{aligned}
   H(P,Q)
   &=-\sum_xP(x)\log_2Q(x)\\
   &=-\frac12\log_2\left(\frac34\right)
     -\frac12\log_2\left(\frac14\right)\\
   &\approx-\frac12(-0.415)-\frac12(-2)\\
   &\approx1.208\text{ bits}.
   \end{aligned}
   $$

   Because $H(P)=1$ bit,

   $$
   D_{\mathrm{KL}}(P\Vert Q)
   =H(P,Q)-H(P)
   \approx1.208-1
   =0.208\text{ bits}.
   $$

4. For a labeled observation, the cross-entropy loss is the negative logarithm of the probability assigned to the observed class. Summing this loss over IID observations gives the negative log-likelihood, so minimizing cross-entropy is equivalent to maximizing the likelihood.
::::
