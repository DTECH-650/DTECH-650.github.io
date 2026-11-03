# Lecture 19 - Linear Discriminant Functions

## Introduction

In Lecture 17, we used a weighted combination of input features to predict a continuous quantity, such as tomorrow's electricity use in kWh. Lecture 18 showed how to fit those weights using gradient descent when we do not want to solve the least-squares problem directly. Now imagine that the building operator needs a different answer: *Will tomorrow be a high-demand day that requires extra capacity, or a normal day?* An exact kWh prediction is no longer the immediate goal. We need to assign one of two **class labels**.

The familiar weighted combination of features can still help. We will interpret its output as a **score** and use a threshold to make a decision. That simple change leads to a new geometric object: a **decision boundary** separating regions assigned to different classes. However, changing what the output *means* does not automatically make the least-squares method from Lecture 17 the best way to learn the weights. We will examine that distinction before briefly meeting a different approach, Fisher's linear discriminant, which looks for a projection that separates the labeled groups.

:::{admonition} Guiding question
:class: tip

How can a weighted sum of features separate classes, and what does its score tell us—and *not* tell us—about a prediction?
:::

## Learning outcomes

After completing this lecture, you should be able to:

1. Distinguish a continuous regression target, a class label, and a discriminant score.
2. Write and interpret a binary linear decision boundary, including the roles of the weight vector and bias.
3. Extend linear discriminants to more than two classes and identify where two class scores tie.
4. Explain how least squares could be used for classification and why its outputs are not automatically probabilities.
5. Describe, conceptually, how Fisher's linear discriminant chooses a useful projection of labeled data.

## From predicting a number to assigning a class

Let the input vector be $\mathbf x=(x_1,\ldots,x_D)^{\mathsf T}\in\mathbb R^D$. Its components could be scaled temperature, expected occupancy, and other measurements available before the building operator must make a decision. In regression, the target $y$ was a real number, such as $42$ kWh, and a model's output was a prediction of that number. In binary classification, the target is instead a label: $C_1$ for high demand or $C_2$ for normal demand. We can encode these labels as $t=1$ and $t=0$ when we need numbers for calculations, but the labels are still *categories*, not arbitrary quantities along a continuous scale.

A **discriminant function** takes $\mathbf x$ and supplies a score used to choose a class. It is a *direct* route from input to decision. Later, we will instead model a conditional class probability such as $p(C_1\mid\mathbf x)$ and use that probability to make a decision. These are related approaches, but a discriminant score by itself need not be a probability.

For now, consider scores built from a weighted sum of the original input features. We will return to the basis functions from Lecture 17 after understanding the geometry in the original input space.

## A linear discriminant for two classes

The simplest binary linear discriminant is

$$
\boxed{a(\mathbf x)=w_0+\mathbf w^{\mathsf T}\mathbf x
=w_0+\sum_{j=1}^{D}w_jx_j,}
$$

where $w_0$ is the **bias** or intercept and $\mathbf w=(w_1,\ldots,w_D)^{\mathsf T}$ is the weight vector. The score $a(\mathbf x)$ is a real number. We define the decision rule

$$
\widehat C(\mathbf x)=
\begin{cases}
C_1, & a(\mathbf x)\geq 0,\\
C_2, & a(\mathbf x)<0.
\end{cases}
$$

Assigning a point with score exactly zero to $C_1$ is simply our chosen tie rule. A different tie rule would affect only points lying exactly on the boundary.

### Where is the boundary?

The **decision boundary** is the set of inputs where the score equals the threshold:

$$
a(\mathbf x)=w_0+\mathbf w^{\mathsf T}\mathbf x=0.
$$

On one side the score is positive and we predict $C_1$; on the other it is negative and we predict $C_2$. With two input features, this equation describes a line. With three, it describes a plane. In $D$ dimensions it describes a $(D-1)$-dimensional **hyperplane**, provided $\mathbf w\ne\mathbf0$. If some hyperplane puts all training examples in their correct class regions, we call the training data **linearly separable**. A linear classifier can still be used when the classes overlap; it then makes some errors.

For a small example, let $x_1$ and $x_2$ be scaled temperature and occupancy, and set

$$
a(x_1,x_2)=-3+x_1+x_2.
$$

The input $(1,1)$ has score $-1$ and is assigned to normal demand, $C_2$. The input $(2,2)$ has score $1$ and is assigned to high demand, $C_1$. The boundary is $x_1+x_2=3$; the input $(2,1)$ lies on it. These numbers illustrate the geometry, not a fitted building-demand model.

~~~{figure} images/linear-discriminant-boundary.svg
:name: linear-discriminant-boundary
:alt: Two-dimensional plot with the line x1 plus x2 equals 3 separating a normal-demand point at (1,1) from a high-demand point at (2,2). A vector labeled w is perpendicular to the boundary.
:width: 780px

A binary linear discriminant. The line $a(\mathbf x)=0$ divides the input plane into two decision regions. The weight vector $\mathbf w=(1,1)^{\mathsf T}$ points perpendicular to the line, toward increasing scores.
~~~

### What do the weights and bias do?

The weight vector controls the boundary's orientation. To see why, take any two points $\mathbf x_A$ and $\mathbf x_B$ on the boundary. Both have score zero, so

$$
w_0+\mathbf w^{\mathsf T}\mathbf x_A=0,
\qquad
w_0+\mathbf w^{\mathsf T}\mathbf x_B=0.
$$

Subtracting the equations cancels $w_0$ and gives

$$
\mathbf w^{\mathsf T}(\mathbf x_A-\mathbf x_B)=0.
$$

The difference $\mathbf x_A-\mathbf x_B$ points *along* the boundary, and its dot product with $\mathbf w$ is zero. Thus $\mathbf w$ is perpendicular, or **normal**, to the boundary. Holding $\mathbf w$ fixed while changing $w_0$ shifts the boundary without rotating it.

The score's sign tells us the assigned side. Its magnitude is related to distance from the boundary, but is not itself a probability. In fact, the signed perpendicular distance is

$$
r(\mathbf x)=\frac{a(\mathbf x)}{\|\mathbf w\|},
$$

where $\|\mathbf w\|=\sqrt{w_1^2+\cdots+w_D^2}$ is the length of the weight vector. For the example above, $(2,2)$ has score $1$ and signed distance $1/\sqrt2$ from the line.

Multiplying both $w_0$ and $\mathbf w$ by the same positive constant changes every raw score but leaves the assigned classes and boundary unchanged. For example, $-3+x_1+x_2$ and $-30+10x_1+10x_2$ make identical decisions. Consequently, a score of $0.8$ is **not** an $80\%$ chance of $C_1$, and a score of $2$ is not automatically twice as certain as a score of $1$.

### The connection to basis functions

Lecture 17 sometimes replaced the original input with a fixed feature vector $\boldsymbol\phi(\mathbf x)$, whose first component was $1$ to include the intercept. We can do the same here:

$$
a(\mathbf x)=\widetilde{\mathbf w}^{\mathsf T}\boldsymbol\phi(\mathbf x).
$$

If $\boldsymbol\phi(\mathbf x)=(1,x_1,\ldots,x_D)^{\mathsf T}$, this is exactly $w_0+\mathbf w^{\mathsf T}\mathbf x$, and the boundary is linear in the original input space. If the basis functions themselves are nonlinear, the boundary may curve in the original input space even though the score remains linear in the weights. For example, with $\boldsymbol\phi(\mathbf x)=(1,x_1^2,x_2^2)^{\mathsf T}$ and $\widetilde{\mathbf w}=(1,-1,-1)^{\mathsf T}$, the boundary is $1-x_1^2-x_2^2=0$, a circle. It is a linear boundary **in feature space**, but not a straight line in the original $(x_1,x_2)$ plane. In the rest of this lecture, “linear decision boundary” refers to the original inputs unless we explicitly say otherwise.

## More than two classes

Suppose demand is categorized as low, normal, or high, giving three classes rather than two. One possible strategy is to train a separate binary classifier for each class versus all the others. But the classifiers could all reject a new input, or more than one could accept it. Another strategy compares every pair of classes; separate pairwise decisions can also create ambiguous votes. We would prefer one rule that compares all classes together.

For $K$ classes, give each class $C_k$ its own linear score,

$$
a_k(\mathbf x)=w_{k0}+\mathbf w_k^{\mathsf T}\mathbf x,
\qquad k=1,\ldots,K,
$$

and assign $\mathbf x$ to the class with the largest score:

$$
\widehat C(\mathbf x)=C_{\arg\max_k a_k(\mathbf x)}.
$$

Here $\arg\max$ returns the index of the largest score. If the scores are $(1.2,2.4,-0.5)$, for example, we choose $C_2$. An exact tie for the largest score needs a tie rule, but the prediction is otherwise unambiguous. These scores do **not** have to lie between zero and one or add to one.

Two class scores tie wherever

$$
a_i(\mathbf x)=a_j(\mathbf x)
\quad\Longleftrightarrow\quad
(\mathbf w_i-\mathbf w_j)^{\mathsf T}\mathbf x
+(w_{i0}-w_{j0})=0.
$$

This is again a hyperplane in the original input space. It is part of the *actual* boundary between $C_i$ and $C_j$ only where their tied scores are at least as large as every other class score. Each class's decision region is the set where its score beats all the others.

When recording the observed class labels for calculations, we often use a **one-hot** target vector. With three classes, examples from $C_1$, $C_2$, and $C_3$ have targets $(1,0,0)^{\mathsf T}$, $(0,1,0)^{\mathsf T}$, and $(0,0,1)^{\mathsf T}$, respectively. The vector is a *coding of the observed label*; it does not mean that an arbitrary vector of discriminant scores is already a vector of class probabilities.

## Could we reuse least squares?

Lecture 17 showed how to fit linear scores to numerical targets by minimizing squared errors. With one-hot class targets $\mathbf t_n$ and a vector of class scores $\mathbf a(\mathbf x_n)$, we could choose weights to minimize

$$
E_{\mathrm{LS}}
=\frac12\sum_{n=1}^{N}
\left\|\mathbf t_n-\mathbf a(\mathbf x_n)\right\|^2.
$$

This is the same least-squares idea applied to coded class labels: find scores numerically close to each target vector, then select the class with the largest fitted score. A design matrix and the normal-equation machinery from Lecture 17 can compute such a fit. This is a legitimate way to build a classifier, but the fitting objective is asking the scores to imitate the **numbers used for the coding**, rather than directly asking for well-separated classes.

For two classes, we could instead encode $C_1$ as $t=1$ and $C_2$ as $t=0$, fit a scalar regression output $r(\mathbf x)$, and predict $C_1$ when $r(\mathbf x)\geq0.5$. This is the same zero-threshold rule as before if we define the discriminant score $a(\mathbf x)=r(\mathbf x)-0.5$. The threshold changes only because the numerical label coding has changed.

The limitation becomes clear in a tiny example. Suppose two training cases are $(x,t)=(0,0)$ and $(1,1)$. The least-squares line $r(x)=x$ fits both exactly. At $x=2$, however, it returns $r(2)=2$. We can still classify that input as $C_1$, but $2$ cannot be interpreted as the probability of $C_1$. Least squares does not constrain binary outputs to $[0,1]$, nor multiclass outputs to valid probability vectors. Moreover, Lecture 17's Gaussian-likelihood explanation for squared error is natural for a continuous target, not for a label restricted to a finite set of categories. These observations do not mean least squares *never* classifies well; they tell us why it does not by itself solve the probabilistic classification problem.

## Fisher's linear discriminant: choose a separating direction

Least squares chooses weights by comparing numerical predictions with coded targets. Fisher's idea asks a different question: *Along which direction should we view the labeled data so that the two classes look well separated?*

Choose a direction $\mathbf w$ and project each $D$-dimensional input onto one number,

$$
z_n=\mathbf w^{\mathsf T}\mathbf x_n.
$$

Imagine placing all the projected $C_1$ points and all the projected $C_2$ points on a number line. We would like the two groups' centers to be far apart, but we also want each group to have a small spread. Large separation of centers alone is not enough: two very broad groups may still overlap substantially.

Write $C_n$ for the observed class of training input $\mathbf x_n$, and let $N_k$ be the number of training points in class $C_k$. For the chosen direction, define the projected class mean and a measure of within-class spread by

$$
m_k=\frac{1}{N_k}\sum_{n:\,C_n=C_k}z_n,
\qquad
S_k=\sum_{n:\,C_n=C_k}(z_n-m_k)^2.
$$

$S_k$ is the *sum* of squared deviations of class $k$ around its projected mean; it is not divided by $N_k$ or $N_k-1$. Fisher's separation criterion is

$$
\boxed{J_{\mathrm F}(\mathbf w)
=\frac{(m_1-m_2)^2}{S_1+S_2}.}
$$

The numerator rewards means that are far apart; the denominator penalizes spread within the classes. Fisher chooses a direction that makes this ratio large. Multiplying $\mathbf w$ by a nonzero constant scales both numerator and denominator by the constant squared, so the criterion evaluates the *direction*, not the arbitrary length, of $\mathbf w$.

Choosing that direction gives a one-dimensional projection, **not yet a complete classifier**. We must also choose a threshold on the projected line; equivalently, the threshold supplies a bias in a linear discriminant. Fisher's construction does not, by itself, assign class probabilities. The matrix derivation of its optimal direction and the multiclass extension are beyond the scope of this introductory treatment.

## Looking ahead

We now have the geometry of linear classification and two contrasting ways to learn from labeled data: least squares tries to match coded target values, while Fisher seeks separation after projection. Neither turns a raw discriminant score into a class probability. The next lecture will add a probabilistic model for the class label. In particular, logistic regression keeps a linear score but maps it to a valid binary probability, then learns the weights using a likelihood appropriate for binary outcomes.

## Practice problems

### Problem 1: A boundary and its geometry

Consider the binary score $a(x_1,x_2)=2x_1-x_2-3$, with $C_1$ assigned when $a\geq0$.

1. Classify $(3,1)$, $(1,1)$, and $(2,1)$.
2. Write the equation of the decision boundary and give a vector perpendicular to it.
3. Find the signed perpendicular distance of $(3,1)$ from the boundary.

::::{admonition} Solution 1
:class: dropdown

The scores are $a(3,1)=6-1-3=2$, $a(1,1)=2-1-3=-2$, and $a(2,1)=4-1-3=0$. Therefore $(3,1)$ is assigned to $C_1$, $(1,1)$ to $C_2$, and $(2,1)$ lies on the boundary and goes to $C_1$ under the stated tie rule.

Setting the score to zero gives $2x_1-x_2-3=0$, or $x_2=2x_1-3$. The weight vector $\mathbf w=(2,-1)^{\mathsf T}$ is perpendicular to this line. Its length is $\sqrt{2^2+(-1)^2}=\sqrt5$, so the signed distance of $(3,1)$ is $a(3,1)/\|\mathbf w\|=2/\sqrt5$.
::::

### Problem 2: Three class scores

Let $a_1(x_1,x_2)=x_1$, $a_2(x_1,x_2)=x_2+1$, and $a_3(x_1,x_2)=2-x_1$. Predict the class with the largest score.

1. Classify $(2,0)$ and $(0,2)$.
2. Find the line where $a_1=a_3$.
3. On that line, for which values of $x_2$ are $a_1$ and $a_3$ tied for the *largest* score?

::::{admonition} Solution 2
:class: dropdown

At $(2,0)$, the scores are $(2,1,0)$, so we choose $C_1$. At $(0,2)$, they are $(0,3,2)$, so we choose $C_2$.

Solving $x_1=2-x_1$ gives $x_1=1$. Along this line, $a_1=a_3=1$, while $a_2=x_2+1$. The tied pair are at least as large as $a_2$ when $1\geq x_2+1$, or $x_2\leq0$. For $x_2>0$, $C_2$ has the larger score, so that part of the line is *not* the visible boundary between $C_1$ and $C_3$. At $x_2=0$, all three scores tie.
::::

### Problem 3: A least-squares classification attempt

Encode $C_1$ as $t=1$ and $C_2$ as $t=0$. The observations $(x,t)=(0,0)$ and $(1,1)$ are fitted exactly by $r(x)=x$.

1. Using a threshold of $0.5$, classify $x=0.25$ and $x=2$.
2. Write a zero-threshold discriminant that gives the same classifications.
3. Why should $r(2)$ not be reported as a class probability?

::::{admonition} Solution 3
:class: dropdown

$r(0.25)=0.25<0.5$, so $x=0.25$ is assigned to $C_2$. $r(2)=2\geq0.5$, so $x=2$ is assigned to $C_1$. Define $a(x)=r(x)-0.5=x-0.5$. Then $a\geq0$ gives exactly the same decisions.

$r(2)=2$ is outside the interval $[0,1]$, so it cannot be a probability. Least squares fitted numerical codes; it did not impose a valid Bernoulli probability model.
::::

### Problem 4: Comparing two Fisher projections

Two candidate directions project the *same* four labeled observations onto a number line:

- Direction A gives $C_1$ values $(0,1)$ and $C_2$ values $(3,4)$.
- Direction B gives $C_1$ values $(-1,2)$ and $C_2$ values $(2,5)$.

For each direction, calculate the projected class means, $S_1+S_2$, and $J_{\mathrm F}$. Which direction does Fisher's criterion prefer, and why?

::::{admonition} Solution 4
:class: dropdown

For A, the means are $m_1=(0+1)/2=0.5$ and $m_2=(3+4)/2=3.5$. The squared deviations sum to $S_1=(0-0.5)^2+(1-0.5)^2=0.5$ and $S_2=(3-3.5)^2+(4-3.5)^2=0.5$. Thus

$$
J_{\mathrm F}(A)=\frac{(0.5-3.5)^2}{0.5+0.5}=9.
$$

For B, the means are again $0.5$ and $3.5$, but $S_1=(-1-0.5)^2+(2-0.5)^2=4.5$ and $S_2=(2-3.5)^2+(5-3.5)^2=4.5$. Hence

$$
J_{\mathrm F}(B)=\frac{(0.5-3.5)^2}{4.5+4.5}=1.
$$

Both directions separate the class *means* by $3$, but A leaves much less within-class spread. Fisher therefore prefers A.
::::

## Reading

Christopher M. Bishop, *Pattern Recognition and Machine Learning*, Chapter 4, pages 179–184; for the brief Fisher overview, pages 186–189.

## References

- [Christopher M. Bishop, *Pattern Recognition and Machine Learning* (author-hosted PDF)](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf).
