# Lecture 19 - Linear Discriminant Functions

In Lecture 17, we used a weighted combination of input features to predict a continuous quantity, such as tomorrow's electricity use in kWh. Lecture 18 showed how to fit those weights using gradient descent when we do not want to solve the least-squares problem directly. Now imagine that the building operator needs a different answer: *Will tomorrow be a high-demand day that requires extra capacity, or a normal day?* An exact kWh prediction is no longer the immediate goal. We need to assign one of two **class labels**.

The familiar weighted combination of features can still help. We will interpret its output as a **score** and use a threshold to make a decision. That simple change leads to a new geometric object: a **decision boundary** separating regions assigned to different classes. However, changing what the output *means* does not automatically make the least-squares method from Lecture 17 the best way to learn the weights. We will examine that distinction before briefly meeting a different approach, Fisher's linear discriminant, which looks for a projection that separates the labeled groups.

:::{admonition} Guiding question
:class: tip

How can a weighted sum of features separate classes, and what does its score tell us - and *not* tell us - about a prediction?
:::

## Learning outcomes

After completing this lecture, you should be able to:

1. Distinguish a continuous regression target, a class label, and a discriminant score.
2. Write and interpret a binary linear decision boundary, including the roles of the weight vector and bias.
3. Extend linear discriminants to more than two classes and identify where two class scores tie.
4. Explain how least squares could be used for classification and why its outputs are not automatically probabilities.
5. Describe, conceptually, how Fisher's linear discriminant chooses a useful projection of labeled data.

## From predicting a number to assigning a class

Let the input vector for the day be $\mathbf x=(x_1,\ldots,x_D)^{\mathsf T}\in\mathbb R^D$. Its components could be scaled temperature, expected occupancy, and other measurements available before the building operator must make a decision. In regression, the target $y$ was a real number, such as $42$ kWh, and a model's output during testing/inference was a prediction of that number. In binary classification, the target is instead a label: $C_1$ for high demand or $C_2$ for normal demand. We can encode these labels as $t=1$ and $t=0$ when we need numbers for calculations, but the labels are still *categories*, not arbitrary quantities along a continuous scale.

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

The weight vector $\mathbf w$ controls the boundary's orientation. To see why, take any two points $\mathbf x_A$ and $\mathbf x_B$ on the boundary. Both have score zero, so

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

If $\boldsymbol\phi(\mathbf x)=(1,x_1,\ldots,x_D)^{\mathsf T}$, this is exactly $w_0+\mathbf w^{\mathsf T}\mathbf x$, and the boundary is linear in the original input space (the original data itself, without it being transformed by arbitrary functions). If the basis functions themselves are nonlinear, the boundary may curve in the original input space even though the score remains linear in the weights. For example, with $\boldsymbol\phi(\mathbf x)=(1,x_1^2,x_2^2)^{\mathsf T}$ and $\widetilde{\mathbf w}=(1,-1,-1)^{\mathsf T}$, the boundary is $1-x_1^2-x_2^2=0$, a circle. It is a linear boundary **in feature space**, but not a straight line in the original $(x_1,x_2)$ plane. In general, a “linear decision boundary” refers to the original inputs unless explicitly stated otherwise.

## More than two classes

Suppose demand is categorized as low, normal, or high, giving three classes rather than two. One possible strategy is to train a separate binary classifier for each class versus all the others. But the classifiers could all reject a new input, or more than one could accept it. Another strategy compares every pair of classes, but again, separate pairwise decisions can also create ambiguous votes. We would prefer one rule that compares all classes together as follows.

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

### A three-class example

Keep $C_1$ as **high** demand and $C_2$ as **normal** demand from the binary example, and add $C_3$ for **low** demand. Suppose $x_1$ and $x_2$ are two nonnegative, scaled inputs, such as temperature and occupancy, and write $s=x_1+x_2$. To make the decision rule easy to see, choose these simple scores:

$$
a_1(\mathbf x)=s-1,\qquad
a_2(\mathbf x)=0.3,\qquad
a_3(\mathbf x)=1-s.
$$

Each score has the form $w_{k0}+\mathbf w_k^{\mathsf T}\mathbf x$. The normal-demand score happens to be constant because its feature weights are zero. These scores are **illustrative choices, not weights learned from building data**. We evaluate all three for a new day and choose the largest:

| Input $\mathbf x=(x_1,x_2)$ | $s$ | Scores $(a_1,a_2,a_3)$ | Largest score; assigned class |
| :--- | :--- | :--- | :--- |
| $(0.2,0.2)$ | $0.4$ | $(-0.6,0.3,0.6)$ | $a_3$; low demand |
| $(0.55,0.45)$ | $1.0$ | $(0,0.3,0)$ | $a_2$; normal demand |
| $(1.0,0.8)$ | $1.8$ | $(0.8,0.3,-0.8)$ | $a_1$; high demand |

The boundaries are where the *winning* score changes. Low and normal demand tie when $1-s=0.3$, or $s=0.7$. At that value, the high-demand score is $-0.3$, so the tie really is a boundary between low and normal. Normal and high demand tie when $0.3=s-1$, or $s=1.3$; the low-demand score is then $-0.3$, so this is another real boundary.

There is one more pairwise tie: low and high demand have equal scores when $1-s=s-1$, giving $s=1$. But both scores are then $0$, while the normal-demand score is $0.3$. **Normal demand wins at that tie and for nearby inputs on either side**, so $s=1$ is *not* a decision boundary. This is why we must compare *all* class scores, not just solve every pairwise equality and call each resulting line a boundary.

Putting this together, we assign low demand when $s<0.7$, normal demand when $0.7<s<1.3$, and high demand when $s>1.3$. At $s=0.7$ and $s=1.3$, the two largest scores tie, so we would use the chosen tie rule.

~~~{figure} images/three-class-linear-discriminant.svg
:name: three-class-linear-discriminant
:alt: A two-dimensional plot of x1 and x2 with low, normal, and high demand regions separated by solid parallel lines x1 plus x2 equals 0.7 and 1.3. A dashed line at x1 plus x2 equals 1 shows where low and high scores tie but normal has the larger score. Three example inputs are marked in the corresponding regions.
:width: 900px

The three colored regions show the class selected by the largest score. Solid lines are actual decision boundaries; the dashed line is only a tie between two *nonwinning* scores. The boundaries happen to be parallel in this particular example because all three scores depend on $x_1+x_2$; multiclass linear boundaries need not be parallel in general.
~~~

When recording the observed class labels for calculations, we often use a **one-hot** target vector. With three classes, examples from $C_1$, $C_2$, and $C_3$ have targets $(1,0,0)^{\mathsf T}$, $(0,1,0)^{\mathsf T}$, and $(0,0,1)^{\mathsf T}$, respectively. The vector is a *coding of the observed label*; it does not mean that an arbitrary vector of discriminant scores is already a vector of class probabilities.

## Could we reuse least squares?

The scores above tell us *how to classify once the weights are known*. How should we learn those weights from labeled examples? We already have one tool: Lecture 17 fitted a line to numerical targets by least squares. Since we can encode class labels as numbers, it is reasonable to ask whether the same method will work here.

### Start with two classes

Let $t_n=1$ when training input $\mathbf x_n$ belongs to $C_1$, and $t_n=0$ when it belongs to $C_2$. For the moment, treat those zeros and ones as if they were ordinary regression targets. Using the same feature vector as in Lecture 17, fit the linear output

$$
r(\mathbf x)=\widetilde{\mathbf w}^{\mathsf T}\boldsymbol\phi(\mathbf x)
$$

by choosing weights that minimize

$$
\boxed{
E_{\mathrm{LS}}(\widetilde{\mathbf w})
=\frac12\sum_{n=1}^{N}
\left[t_n-r(\mathbf x_n)\right]^2.
}
$$

Each term compares a *number produced by the model* with the numerical code of the observed class. For instance, if $t_n=1$ and the model gives $r(\mathbf x_n)=0.7$, the residual is $1-0.7=0.3$ and this example contributes $\tfrac12(0.3)^2=0.045$ to $E_{\mathrm{LS}}$. For a class-$C_2$ example with $t_n=0$ and $r(\mathbf x_n)=0.2$, the contribution is $\tfrac12(0-0.2)^2=0.02$. The factor $\tfrac12$ has the same calculus convenience as it did in Lecture 17.

Nothing new is required to *compute* the least-squares weights: the design-matrix method from Lecture 17 or gradient descent from Lecture 18 can minimize this objective. What is new is the **interpretation** of the fitted output. A building's electricity use really can be $42$ kWh; a class label cannot be $0.7$ of a class. The value $r(\mathbf x)=0.7$ is, at this stage, a fitted numerical score.

### Turn the fitted output into a decision

Because $0.5$ lies halfway between the codes $0$ and $1$, a natural decision rule is to choose the *closer code*. For a fitted output $r$, the squared distance to code $1$ is no larger than the squared distance to code $0$ precisely when

$$
(r-1)^2\leq(r-0)^2
\quad\Longleftrightarrow\quad
r^2-2r+1\leq r^2
\quad\Longleftrightarrow\quad
r\geq0.5.
$$

Thus the classification rule is

$$
\widehat C(\mathbf x)=
\begin{cases}
C_1, & r(\mathbf x)\geq0.5,\\
C_2, & r(\mathbf x)<0.5.
\end{cases}
$$

This threshold is a convention based on the chosen codes; it does **not** prove that $r(\mathbf x)$ is a probability. It also matches the zero-threshold discriminant from earlier in the lecture. Define $a(\mathbf x)=r(\mathbf x)-0.5$. Then $r(\mathbf x)\geq0.5$ is exactly the same condition as $a(\mathbf x)\geq0$, and the boundary is the set where $r(\mathbf x)=0.5$.

Consider just two training examples: $(x,t)=(0,0)$ and $(1,1)$. The line $r(x)=x$ fits both exactly, so it has zero training squared error. It places the class boundary at $x=0.5$. But at the new input $x=2$, it outputs $r(2)=2$. The decision rule still assigns $C_1$, yet a probability of $2$ is impossible.

~~~{figure} images/least-squares-binary-classification.svg
:name: least-squares-binary-classification
:alt: Plot of the fitted line r(x)=x through binary labels at x=0 and x=1. A horizontal threshold at 0.5 makes a decision boundary at x=0.5. At x=2, the fitted value is 2, outside the valid probability range from 0 to 1.
:width: 820px

The threshold converts a fitted number into a class decision. The line fits the two coded labels perfectly, but its output at $x=2$ cannot be read as a class probability.
~~~

### Why this first attempt is tempting, and where it falls short

There is a genuine reason to try squared error, beyond having learned how to minimize it. Let $T$ be a random binary label, with $T=1$ for $C_1$, and suppose the *true* conditional probability at a particular input is $q(\mathbf x)=P(T=1\mid\mathbf x)$. At that input, the conditional mean of the label is

$$
\mathbb E[T\mid\mathbf x]
=1\cdot q(\mathbf x)+0\cdot[1-q(\mathbf x)]
=q(\mathbf x).
$$

If we could choose a separate prediction $r$ for this one input using the entire population, its expected squared error would be

$$
\begin{aligned}
\mathbb E[(T-r)^2\mid\mathbf x]
&=q(\mathbf x)(1-r)^2+[1-q(\mathbf x)](0-r)^2\\
&=[r-q(\mathbf x)]^2+q(\mathbf x)[1-q(\mathbf x)].
\end{aligned}
$$

The second term does not depend on $r$, and the squared first term is smallest when $r=q(\mathbf x)$. So, **in this idealized population calculation**, squared error favors the true class probability. That explains why regression on zero-one labels can sometimes be useful.

Our fitted line is not that idealized calculation. We have only a finite sample, and one set of linear weights must serve *all* inputs. A straight line is also free to go below $0$ or above $1$, as the figure shows. Thus least squares does not guarantee that its scores are valid probabilities, even though the population-level squared-error target is a conditional probability.

There is another mismatch: least squares measures how close the scores are to the *numerical codes*, whereas classification accuracy asks only whether each score falls on the correct side of the threshold. The two criteria can prefer different weights. Consider three labeled examples $(x,t)=(0,0),(1,1),(2,1)$ and two candidate lines:

| Candidate | Outputs at $x=0,1,2$ | Sum of squared errors | Correct classes using $0.5$ |
| :--- | :--- | :--- | :--- |
| $r_A(x)=x$ | $(0,1,2)$ | $0^2+0^2+(-1)^2=1$ | $3$ out of $3$ |
| $r_B(x)=0.2+0.4x$ | $(0.2,0.6,1.0)$ | $(-0.2)^2+(0.4)^2+0^2=0.20$ | $3$ out of $3$ |

The table reports the *sum* of squared errors, before the factor $\tfrac12$ in $E_{\mathrm{LS}}$. Including that factor would give $0.5$ for A and $0.10$ for B, without changing which is smaller. Both lines classify every training example correctly. Among these two candidates, least squares prefers $r_B$ because its *numerical predictions* are closer to the codes. In particular, $r_A(2)=2$ is penalized for being far above the target $1$, even though it is on the correct side of the decision boundary. This does not make squared error useless; it shows that it is solving a different optimization problem from “minimize the number of wrong class decisions.”

~~~{figure} images/least-squares-classification-objectives.svg
:name: least-squares-classification-objectives
:alt: Side-by-side plots of two linear score functions for the same three binary-labeled examples. Both achieve three correct classifications, but the longer residual for candidate A makes its sum of squared errors larger than candidate B's.
:width: 900px

Two lines can make the same class decisions yet receive different least-squares scores. The dashed vertical segments show their numerical residuals from the coded targets.
~~~

### Extending the idea to more classes

For $K$ classes, use the one-hot target vector $\mathbf t_n=(t_{n1},\ldots,t_{nK})^{\mathsf T}$ introduced above. Here $t_{nk}=1$ for the observed class of example $n$, and $0$ for its other classes. Fit one linear output $a_k(\mathbf x)$ for each class and collect them into $\mathbf a(\mathbf x)=(a_1(\mathbf x),\ldots,a_K(\mathbf x))^{\mathsf T}$. The least-squares objective becomes

$$
E_{\mathrm{LS}}
=\frac12\sum_{n=1}^{N}
\left\|\mathbf t_n-\mathbf a(\mathbf x_n)\right\|^2
=\frac12\sum_{n=1}^{N}\sum_{k=1}^{K}
\left[t_{nk}-a_k(\mathbf x_n)\right]^2.
$$

The double sum simply squares the error in **every class score for every training example**. At prediction time we choose the class with the largest fitted score, as in the preceding section. For example, the score vector $(-0.2,1.3,0.4)^{\mathsf T}$ chooses $C_2$, but it is not a valid probability vector: it contains a negative component, another component greater than $1$, and its components do not add to $1$.

We have therefore learned a workable *decision procedure*, not a probabilistic model for the labels. Recall also that Lecture 17 connected squared error to a Gaussian model for a **continuous** target. A binary or one-hot label is not Gaussian. This motivates the next lecture's Bernoulli-based approach: keep the useful linear score, but use a model and fitting objective designed for class probabilities.

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

~~~{figure} images/fisher-projection-number-line.svg
:width: 100%
:name: fig-fisher-projection-number-line

The direction $\mathbf w$ turns each two-dimensional observation into a position $z_n$ on a number line. Fisher favors directions for which the projected class means $m_1$ and $m_2$ are far apart while points in each class remain close to their own mean. The shaded ranges illustrate spread; $S_1$ and $S_2$ are **sums of squared deviations**, not the widths of those ranges.
~~~

The numerator rewards means that are far apart; the denominator penalizes spread within the classes. Fisher chooses a direction that makes this ratio large. Multiplying $\mathbf w$ by a nonzero constant scales both numerator and denominator by the constant squared, so the criterion evaluates the *direction*, not the arbitrary length, of $\mathbf w$.

Choosing that direction gives a one-dimensional projection, **not yet a complete classifier**. We must also choose a threshold on the projected line, or put equivalenty, the threshold supplies a bias in a linear discriminant. Fisher's construction does not, by itself, assign class probabilities. The matrix derivation of its optimal direction and the multiclass extension are beyond the scope of this introductory treatment.

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
