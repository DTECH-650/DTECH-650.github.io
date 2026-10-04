# Lecture 14 - Probabilistic Graphical Models

Imagine looking out of your window in the morning and noticing that the lawn is wet. Did it rain overnight, or did the sprinkler turn on? Wet grass is evidence for rain, but it is not enough to settle the question: both explanations could produce what you see.

Now you check the sprinkler timer and discover that it ran during the night. The wet lawn becomes less convincing evidence for rain, because you have found another explanation. But suppose you also notice that the street is wet, well beyond the sprinkler's reach. That gives you additional evidence for rain. The observation that started the investigation has not changed - the lawn is still wet - but what you believe about its explanation changes as you learn more.

To reason about this situation, separate probabilities for rain, sprinkler activity, and wet grass are not enough. We need to describe **how these events are related** using joint probability distributions. Rain could make both the lawn and the street wet, while the sprinkler could make the lawn wet without making the street wet. A useful model should represent those relationships and tell us how to combine the evidence, including when one observation changes what another observation tells us.

A **probabilistic graphical model** combines a graph with probability distributions to answer questions like these. The graph organizes the model's assumptions, while the distributions supply the numerical probabilities. In this lecture, we study **directed acyclic graphs**, also called **Bayesian networks**.

The main idea is that a complicated joint distribution can often be built from a collection of smaller conditional distributions. The arrows tell us which variables enter each conditional distribution. They also let us reason about independence without calculating every probability.

## Directed graphs and the notation we will use

A directed graph consists of **nodes** and **directed edges**, drawn as arrows. In a probabilistic graphical model, each node represents a random variable. An arrow $X\to Y$ makes $X$ a **parent** of $Y$, and $Y$ a **child** of $X$.

We will use the notation from earlier lectures:

- $X_i$ is a random variable, and $x_i$ is one possible value of that variable.
- $\mathbf X=(X_1,\ldots,X_K)$ collects the variables in the graph; $\mathbf x=(x_1,\ldots,x_K)$ collects their values. Here $K$ is the number of nodes, not the sample size $N$.
- $\operatorname{pa}(i)$ is the set of indices of the parents of node $i$. The expression $\mathbf x_{\operatorname{pa}(i)}$ means the values of those parents.
- $p(\mathbf x)$ denotes a joint PMF for discrete variables or a joint density for continuous variables. We use $P(\cdot)$ when explicitly writing the probability of an event, such as $P(X=1)$.

For example, if $X_1$ and $X_3$ are the parents of $X_4$, then

$$
\operatorname{pa}(4)=\{1,3\},
\qquad
p(x_4\mid\mathbf x_{\operatorname{pa}(4)})
=p(x_4\mid x_1,x_3).
$$

A node with no parents is a **root**. Its probability factor is an unconditional distribution, such as $p(x_1)$.

Following arrows repeatedly defines an **ancestor** or a **descendant**. If $X\to Z\to Y$, then $X$ is an ancestor of $Y$, and $Y$ is a descendant of $X$. Parents and children are the immediate relatives; ancestors and descendants can be several arrows away.

### Why the graph must be acyclic

A **directed cycle** would let us start at a node, follow arrows, and return to the same node. A directed acyclic graph, abbreviated **DAG**, contains no such cycle. For example, $X\to Y\to Z\to X$ is not a DAG.

Without directed cycles, we can place the nodes in a **topological order**: an order in which every parent appears before its children. This is what allows us to build the distribution one variable at a time.

A Bayesian network consists of both a DAG and a conditional distribution for each node given its parents. The name does not require us to put a Bayesian prior on the model's parameters. We can also estimate those parameters using maximum likelihood.

<!-- An arrow is not automatically a causal claim. In this lecture it specifies a probability factorization. Interpreting an arrow as a cause-and-effect relationship requires additional assumptions. -->

## Joint probability and factorization

### Starting from the probability chain rule

For any collection of random variables, the probability chain rule gives

$$
p(x_1,\ldots,x_K)
=p(x_1)p(x_2\mid x_1)
p(x_3\mid x_1,x_2)\cdots
p(x_K\mid x_1,\ldots,x_{K-1}).
$$

This identity does not assume independence. Each new factor can depend on every variable appearing earlier in the chosen order.

A directed graphical model makes a more structured assumption: once a node's parents are known, the other variables earlier in a topological order do not provide additional information about that node. Its joint distribution therefore takes the form

$$
\boxed{
p(\mathbf x)
=\prod_{i=1}^{K}
p(x_i\mid\mathbf x_{\operatorname{pa}(i)})
}.
$$

The product symbol means to multiply one factor for each node. Every variable appears once as the variable being predicted, on the left side of its conditional distribution. It may also appear in the conditioning arguments of its children's factors.

The graph does not replace the joint distribution. It gives a compact recipe for constructing it.

### Example: faults, an alarm, and a support ticket

Let the following variables take values in $\{0,1\}$:

- $B=1$: a software bug is present.
- $F=1$: a hardware fault is present.
- $A=1$: the monitoring system raises an alarm.
- $T=1$: a support ticket is opened.

Suppose we use this graph:

~~~{figure} images/pgm-joint-factorization.svg
:name: pgm-joint-factorization
:alt: A software bug B and hardware fault F both point to alarm A, which points to support ticket T.
:width: 650px

The model has two root variables, an alarm depending on both roots, and a ticket depending on the alarm.
~~~

The general chain rule, in the order $B,F,A,T$, would be

$$
p(b,f,a,t)
=p(b)p(f\mid b)p(a\mid b,f)p(t\mid b,f,a).
$$

Our graph makes two simplifying assumptions. First, $B$ and $F$ are independent before we observe the alarm, so $p(f\mid b)=p(f)$. Second, once the alarm is known, the faults provide no additional information about the ticket, so $p(t\mid b,f,a)=p(t\mid a)$. Thus

$$
\boxed{
p(b,f,a,t)=p(b)p(f)p(a\mid b,f)p(t\mid a)
}.
$$

For illustration, suppose

$$
P(B=1)=0.2,\quad P(F=0)=0.9,\quad
P(A=1\mid B=1,F=0)=0.8,\quad
P(T=1\mid A=1)=0.7.
$$

The probability of this particular combination is

$$
\begin{aligned}
P(B=1,F=0,A=1,T=1)
&=0.2\times0.9\times0.8\times0.7\\
&=0.1008.
\end{aligned}
$$

This is one entry of the joint table. To find the probability of a ticket without knowing the faults or alarm, we marginalize:

$$
P(T=1)
=\sum_{b=0}^{1}\sum_{f=0}^{1}\sum_{a=0}^{1}
p(b)p(f)p(a\mid b,f)p(T=1\mid a).
$$

The sums account for every possible configuration of the variables we have not observed. For continuous variables, the corresponding marginalization uses integrals rather than sums.

### Why the product is a valid probability distribution

Each local conditional distribution is normalized. For example, $\sum_t p(t\mid a)=1$ for each fixed $a$. Summing the example's joint distribution over all variables gives

$$
\begin{aligned}
\sum_{b,f,a,t}p(b)p(f)p(a\mid b,f)p(t\mid a)
&=\sum_{b,f,a}p(b)p(f)p(a\mid b,f)\\
&=\sum_{b,f}p(b)p(f)\\
&=1.
\end{aligned}
$$

We first sum over the leaf $T$, then over $A$, and finally over the roots. A DAG always allows this reverse-topological summation or integration. Thus normalized local distributions produce a normalized joint distribution.

For four binary variables, an unrestricted joint table has $2^4=16$ entries and 15 free probabilities, because the entries must sum to one. This graph needs only eight free probabilities: one each for the two roots, four for the alarm's four parent configurations, and two for the ticket's two parent configurations. The reduction comes from modeling assumptions, not from a trick that works for every joint distribution.

## Independence and conditional independence

Recall that $X$ and $Y$ are independent, written $X\perp\!\!\!\perp Y$, if

$$
p(x,y)=p(x)p(y)
$$

for every possible pair of values. Independence means that learning one variable does not change the distribution of the other.

They are **conditionally independent given $Z$**, written

$$
X\perp\!\!\!\perp Y\mid Z,
$$

if

$$
p(x,y\mid z)=p(x\mid z)p(y\mid z).
$$

For discrete variables, this statement applies to every $z$ with $P(Z=z)>0$. For continuous variables, we use conditional densities where they are defined. Equivalently, when the relevant conditional distribution is defined,

$$
p(y\mid x,z)=p(y\mid z).
$$

Here we are comparing two situations: we know $Z$ alone, or we know both $Z$ and $X$. Conditional independence says that adding $X$ does not change the distribution of $Y$.

It is important not to treat unconditional and conditional independence as the same statement. Variables can be dependent until we observe a third variable, or independent until we observe a third variable. The next three graphs explain both possibilities. In the figures below, a blue-filled node is **observed**: we condition on its value.

## The three canonical structures

Every internal node along a path through a directed graph has one of three local arrow patterns. These are called a **chain**, a **fork**, and a **collider**. Understanding these small structures is the key to understanding larger graphs.

For the moment, assume that each displayed three-variable graph is the complete model, with no additional path connecting its endpoints.

### Chain: information passes through an intermediate variable

A chain, also called a serial or sequential connection, has the form

$$
X\to Z\to Y.
$$

~~~{figure} images/pgm-chain.svg
:name: pgm-chain
:alt: Two chains X to Z to Y. The first has no observed nodes and is active. The second has observed Z and is blocked.
:width: 850px

A chain is active when its middle variable is unobserved. Conditioning on the middle variable blocks this path.
~~~

The joint factorization is

$$
p(x,z,y)=p(x)p(z\mid x)p(y\mid z).
$$

Imagine that $X$ indicates a software bug, $Z$ indicates a service failure, and $Y$ indicates a customer complaint. A bug changes the chance of a failure, and a failure changes the chance of a complaint.

Suppose

$$
\begin{aligned}
P(Z=1\mid X=1)&=0.8,&
P(Z=1\mid X=0)&=0.1,\\
P(Y=1\mid Z=1)&=0.9,&
P(Y=1\mid Z=0)&=0.05.
\end{aligned}
$$

If we know about the bug but not the service failure, we must average over both possible values of $Z$. The law of total probability gives

$$
\begin{aligned}
P(Y=1\mid X=1)
&=0.9(0.8)+0.05(0.2)=0.73,\\
P(Y=1\mid X=0)
&=0.9(0.1)+0.05(0.9)=0.135.
\end{aligned}
$$

The different answers show that $X$ provides information about $Y$ when $Z$ is unknown.

Now suppose we know that a failure occurred. Under this model,

$$
P(Y=1\mid Z=1,X=1)
=P(Y=1\mid Z=1,X=0)
=0.9.
$$

Once the service failure is known, knowing about the bug adds no information about the complaint. Algebraically,

$$
\begin{aligned}
p(x,y\mid z)
&=\frac{p(x)p(z\mid x)p(y\mid z)}{p(z)}\\
&=\underbrace{\frac{p(x)p(z\mid x)}{p(z)}}_{p(x\mid z)}
p(y\mid z)\\
&=p(x\mid z)p(y\mid z).
\end{aligned}
$$

Therefore $X\perp\!\!\!\perp Y\mid Z$. The reversed chain $X\leftarrow Z\leftarrow Y$ has the same endpoint conditional-independence property.

### Fork: a shared parent links two variables

A fork, also called a diverging connection or common-cause structure, has the form

$$
X\leftarrow Z\to Y.
$$

~~~{figure} images/pgm-fork.svg
:name: pgm-fork
:alt: Two forks with Z pointing to X and Y. The first is active with Z unobserved; the second is blocked with Z observed.
:width: 850px

An unknown shared parent can create dependence between its children. Knowing that parent blocks the fork path.
~~~

The joint distribution factors as

$$
p(x,z,y)=p(z)p(x\mid z)p(y\mid z).
$$

Let $Z=1$ mean that a system is operating during a peak-demand period. Let $X=1$ mean that a processing queue is long, and $Y=1$ mean that network delay is high. Suppose queue length and network delay have independent fluctuations once the demand period is known:

$$
\begin{aligned}
P(Z=1)&=0.3,\\
P(X=1\mid Z=1)&=0.8,&P(X=1\mid Z=0)&=0.1,\\
P(Y=1\mid Z=1)&=0.7,&P(Y=1\mid Z=0)&=0.2.
\end{aligned}
$$

Without knowing the period, the marginal probabilities are

$$
\begin{aligned}
P(X=1)&=0.3(0.8)+0.7(0.1)=0.31,\\
P(Y=1)&=0.3(0.7)+0.7(0.2)=0.35.
\end{aligned}
$$

To find the probability that both occur, first condition on $Z$. Within each period, the two probabilities multiply:

$$
\begin{aligned}
P(X=1,Y=1)
&=0.3(0.8)(0.7)+0.7(0.1)(0.2)\\
&=0.182.
\end{aligned}
$$

This does not equal $P(X=1)P(Y=1)=0.31(0.35)=0.1085$. Thus the two variables are dependent overall. A long queue is evidence for a peak-demand period, which in turn makes high network delay more likely.

However, once we know $Z$, its shared influence has already been accounted for. Dividing the joint distribution by $p(z)$ gives

$$
p(x,y\mid z)=p(x\mid z)p(y\mid z),
$$

so $X\perp\!\!\!\perp Y\mid Z$. For example, during a known peak period the probability of both problems is $0.8(0.7)=0.56$.

### Collider: observing a shared effect can link independent variables

A collider, also called a converging connection, common-effect, or explaining-away structure, has the form

$$
X\to Z\leftarrow Y.
$$

The two arrowheads meet at $Z$. In this isolated three-node graph, the structure is also called a **v-structure**. More generally, that term refers to a collider whose two parents are not directly joined by an edge.

~~~{figure} images/pgm-collider.svg
:name: pgm-collider
:alt: X and Y both point to Z, and Z points to W. The path through Z is blocked when nothing is observed, active when Z is observed, and active when its descendant W is observed.
:width: 1000px

A collider blocks its path unless the collider or one of its descendants is observed. This is the opposite of the rule for a chain or fork.
~~~

First ignore $W$ in the figure. The factorization for the three-variable model is

$$
p(x,y,z)=p(x)p(y)p(z\mid x,y).
$$

When we do not observe $Z$, sum it out:

$$
\begin{aligned}
p(x,y)
&=p(x)p(y)\sum_z p(z\mid x,y)\\
&=p(x)p(y).
\end{aligned}
$$

The conditional distribution sums to one for every $x,y$, so $X$ and $Y$ are independent. But if we observe $Z=z$, then

$$
p(x,y\mid z)
=\frac{p(x)p(y)p(z\mid x,y)}{p(z)}.
$$

The factor $p(z\mid x,y)$ depends on both variables. It reweights their combinations according to how well they explain the observation, and the result generally no longer factors into separate distributions for $X$ and $Y$.

#### Example: explaining away an alarm

Let $X=1$ indicate an intrusion, $Y=1$ indicate a sensor fault, and $Z=1$ indicate an alarm. Assume the intrusion and fault are initially independent:

$$
P(X=1)=0.1,\qquad P(Y=1)=0.2.
$$

For a simple numerical example, suppose an alarm occurs whenever either an intrusion or a sensor fault is present. It does not occur when both are absent. Then

$$
P(Z=1)
=1-P(X=0,Y=0)
=1-(0.9)(0.8)
=0.28.
$$

An intrusion always produces an alarm in this example, so

$$
P(X=1\mid Z=1)
=\frac{P(X=1,Z=1)}{P(Z=1)}
=\frac{0.1}{0.28}
\approx0.357.
$$

The alarm raises the intrusion probability from 0.1 to about 0.357. Now also learn that the sensor is faulty. The fault already explains the alarm, which is guaranteed when $Y=1$. Thus

$$
P(X=1\mid Z=1,Y=1)
=P(X=1\mid Y=1)
=0.1.
$$

On the other hand, if the sensor is not faulty, an observed alarm must have come from an intrusion:

$$
P(X=1\mid Z=1,Y=0)=1.
$$

The distribution of $X$ now depends strongly on the value of $Y$. This is **explaining away**: evidence for one explanation reduces the need for another explanation of the same observation.

This deterministic alarm rule makes the arithmetic simple. The same phenomenon can occur with noisy alarm probabilities. It does not mean that every collider produces dependence for every parameter choice or every observed value. In this example, observing $Z=0$ fixes both parents at zero.

#### Observing a descendant can also open the collider

Return to the fourth node $W$ in the figure. Suppose $W=1$ means that an alarm notification is received, with

$$
P(W=1\mid Z=1)=0.9,
\qquad
P(W=1\mid Z=0)=0.05.
$$

A notification is imperfect evidence about the alarm. Even without directly observing $Z$, observing $W$ can therefore couple the two explanations for $Z$.

Using the preceding alarm model,

$$
P(W=1)=0.9(0.28)+0.05(0.72)=0.288.
$$

Whenever $X=1$, the alarm is present and the notification probability is 0.9. Hence

$$
P(X=1\mid W=1)
=\frac{0.1(0.9)}{0.288}
=0.3125.
$$

If we also know $Y=1$, the alarm is already guaranteed by the fault. The notification provides no further information about $X$, so

$$
P(X=1\mid W=1,Y=1)=0.1.
$$

These different values show that observing the descendant $W$ can make $X$ and $Y$ dependent.

### Comparing the three structures

| Structure along the path | With no relevant observation | What observation does to this path |
| --- | --- | --- |
| Chain: $X\to Z\to Y$ | Active; dependence between endpoints is possible | Observing $Z$ blocks it |
| Fork: $X\leftarrow Z\to Y$ | Active; dependence between endpoints is possible | Observing $Z$ blocks it |
| Collider: $X\to Z\leftarrow Y$ | Blocked if neither $Z$ nor any descendant is observed | Observing $Z$ or a descendant opens it |

Here **active** means that this path can transmit dependence; **blocked** means that it cannot. In the isolated graphs, blocking the only path guarantees endpoint independence, conditional on any observations. In a larger graph, another path may still be active.

## Graph separation and d-separation

### A path need not follow the arrows

To find a path between two nodes, we may traverse an edge in either direction, but we keep its arrowhead when classifying the path. For example, $X\leftarrow Z\to Y$ is a path from $X$ to $Y$, even though we cannot follow arrows all the way from $X$ to $Y$.

A **directed path** is different: all its arrows point in the direction of travel. Directed paths define descendants.

For separation, we consider paths with no repeated nodes. An internal node is a **collider on that path** when the two edges used by the path both point into it. Otherwise it is a **non-collider on that path**. This classification depends on the path, not just on the node. In the graph $X\to Z\leftarrow Y$ with $Z\to W$, $Z$ is a collider along $X,Z,Y$, but a non-collider along $X,Z,W$.

### Why ordinary connectivity is not enough

For chains and forks, conditioning on a middle node blocks the path. It might therefore seem that separation can be checked by removing all observed nodes and looking for remaining connections.

Colliders show why that rule fails. An unobserved collider can block a path even though the nodes are still connected. An observed collider can open a path rather than block it. We need a separation rule that respects arrow directions and observed descendants. That rule is **d-separation**, where the “d” refers to directed graphs.

### The active-path rule

Let $\mathcal O$ be the set of observed nodes. We condition on their values, written $\mathbf X_{\mathcal O}=\mathbf x_{\mathcal O}$. The query nodes are not members of $\mathcal O$.

A path is **active given $\mathcal O$** if both conditions hold:

1. Every internal non-collider on the path is outside $\mathcal O$.
2. Every internal collider is either in $\mathcal O$ itself or has at least one descendant in $\mathcal O$.

Both conditions must hold at every internal node. One blocking node is enough to block the entire path.

Two nodes are **d-separated given $\mathcal O$** if there is no active path between them. For sets of nodes, there must be no active path between any node in one set and any node in the other.

If the joint distribution factorizes according to the DAG, then

$$
\boxed{
\text{$X$ and $Y$ are d-separated given $\mathcal O$}
\quad\Longrightarrow\quad
X\perp\!\!\!\perp Y\mid\mathbf X_{\mathcal O}
}.
$$

This is a guarantee that comes from the graph. Conversely, an active path means the graph does **not** guarantee independence. It does not prove that a particular numerical model has dependence. For example, a conditional distribution might ignore one of its displayed parents, creating an additional independence that the graph does not show.

### Example: checking every path matters

Consider this graph:

~~~{figure} images/pgm-d-separation.svg
:name: pgm-d-separation
:alt: X points to M, M points to Y, X and Y both point to C, and C points to W.
:width: 650px

There are two paths between $X$ and $Y$: the chain $X,M,Y$ and the collider path $X,C,Y$. Node $W$ is a descendant of the collider.
~~~

Its joint factorization is

$$
p(x,m,y,c,w)
=p(x)p(m\mid x)p(y\mid m)p(c\mid x,y)p(w\mid c).
$$

We can check the paths separately:

| Observed set $\mathcal O$ | Chain $X\to M\to Y$ | Collider $X\to C\leftarrow Y$ | Are $X$ and $Y$ d-separated? |
| --- | --- | --- | --- |
| $\varnothing$ (nothing observed) | Active | Blocked | No |
| $\{M\}$ | Blocked | Blocked | Yes |
| $\{C\}$ | Active | Active | No |
| $\{M,C\}$ | Blocked | Active | No |
| $\{M,W\}$ | Blocked | Active: $W$ is an observed descendant of $C$ | No |

For example, observing $M$ gives $X\perp\!\!\!\perp Y\mid M$. But observing $M$ **and** $C$ removes that guarantee, because observing $C$ opens the second path. Adding information does not always preserve conditional independence.

## Local Markov properties and the Markov blanket

The word **Markov** here refers to a screening-off property: after we know an appropriate set of variables, some other variables provide no additional information. It does not require a time series.

### Parents screen off non-descendants

The **local Markov property** of a DAG says that a node is conditionally independent of its non-descendants, excluding its parents, given its parents.

Let $\operatorname{nd}(i)$ denote the indices of the other nodes that are not descendants of $X_i$. Then

$$
X_i\perp\!\!\!\perp
\mathbf X_{\operatorname{nd}(i)\setminus\operatorname{pa}(i)}
\mid
\mathbf X_{\operatorname{pa}(i)}.
$$

The set difference $\setminus$ means that we remove the parents from the non-descendant set. We do not include $X_i$ itself. A root's parent set is empty, so no conditioning is needed for its corresponding local independence statement.

In words, once we know the parents, the remaining non-descendants do not help us predict the node. This is the property used to simplify the chain-rule factorization.

Parents do not generally screen off a node from its **descendants**. If we learn that an alarm occurred, that observation can still change our belief about its parent, even when the parent's own parents are known.

### The Markov blanket screens off the rest of the graph

To isolate a node from **all** other nodes, we need its **Markov blanket**. In a DAG, the graphical Markov blanket of $X_i$ consists of:

- its parents;
- its children;
- the other parents of its children, sometimes called its **co-parents** or **spouses**.

Take the union of these sets, count each node only once, and exclude $X_i$ itself. Write this set as $\operatorname{MB}(i)$. Let $\mathcal R$ contain every node outside the blanket except $i$. Then

$$
X_i\perp\!\!\!\perp
\mathbf X_{\mathcal R}
\mid
\mathbf X_{\operatorname{MB}(i)}.
$$

Knowing the blanket means that learning any of the remaining variables gives no additional information about $X_i$.

~~~{figure} images/pgm-markov-blanket.svg
:name: pgm-markov-blanket
:alt: U points to P, P points to target X, X points to C1 and C2, Q also points to C1, and C2 points to V. P, Q, C1, and C2 form X's Markov blanket.
:width: 800px

The amber node is the target $X$. Its blue blanket contains parent $P$, children $C_1,C_2$, and co-parent $Q$. Nodes $U,V$ lie outside the blanket.
~~~

For this graph,

$$
\operatorname{MB}(X)=\{P,Q,C_1,C_2\}.
$$

The co-parent $Q$ is easy to overlook because it is not directly connected to $X$. But observing their shared child $C_1$ opens the collider $X\to C_1\leftarrow Q$. To screen off this path as well, the blanket includes $Q$.

We can also see the blanket directly in the probability factors. The full joint is

$$
\begin{aligned}
p(u,p,x,q,c_1,c_2,v)
={}&p(u)p(p\mid u)p(q)p(x\mid p)\\
&{}\times p(c_1\mid x,q)p(c_2\mid x)p(v\mid c_2).
\end{aligned}
$$

Here lowercase $p$ inside a probability argument is the realized value of the node $P$; $p(\cdot)$ remains our notation for a PMF or density.

Now condition on every variable except $X$. Every factor that does not contain $x$ is constant as we vary $x$, so those factors cancel between the numerator and the normalizing denominator:

$$
p(x\mid u,p,q,c_1,c_2,v)
\ \propto\
p(x\mid p)p(c_1\mid x,q)p(c_2\mid x).
$$

The symbol $\propto$ means “proportional as a function of $x$.” For discrete $X$, divide the right side by its sum over possible $x$ values to obtain a normalized conditional PMF. For continuous $X$, divide by the corresponding integral.

Only the blanket values $p,q,c_1,c_2$ remain. The outside values $u,v$ do not affect this conditional distribution.

A **Markov boundary** is a minimal Markov blanket: no member can be removed while retaining the screening-off property. The graphical blanket is always a valid blanket for a factorizing distribution, but a particular model can have a smaller boundary if some of its displayed dependencies are redundant.

### Different arrow directions can imply the same independences

The graphs

$$
X\to Z\to Y,
\qquad
X\leftarrow Z\leftarrow Y,
\qquad
X\leftarrow Z\to Y
$$

all imply $X\perp\!\!\!\perp Y\mid Z$ and do not generally imply $X\perp\!\!\!\perp Y$. They are examples of **Markov-equivalent** DAGs: they encode the same conditional-independence statements.

The collider $X\to Z\leftarrow Y$ is different. It implies unconditional endpoint independence in the isolated graph, but not independence after conditioning on $Z$.

This is another reason that independence information alone does not always determine an arrow's direction or establish causality.

## The Bayes-ball algorithm

For a small graph, we can list the paths and apply the d-separation rules by hand. For a large graph, explicitly listing every path can be expensive. **Bayes-ball** checks reachability using local rules instead.

Imagine a ball traveling between neighboring nodes. It may move along an arrow or against it. What it does at a node depends on:

- whether that node is observed;
- whether the ball arrived from a parent or from a child.

That second piece is essential. Reaching the same node from the two directions can have different consequences.

### The propagation rules

We use the basic version for probabilistic nodes. Specialized rules for deterministic relationships and the additional tasks handled by the original Bayes-ball algorithm are beyond this lecture.

| Arrival at a node | Is the node observed? | Where the ball may go next |
| --- | --- | --- |
| From a child | No | To all parents and all children |
| From a child | Yes | Nowhere: stop |
| From a parent | No | To all children |
| From a parent | Yes | Back to all parents |

“To all parents” includes any parent that the ball just came from. Repeated visits are handled by keeping track of visited states.

These rules reproduce the three canonical structures:

- In a chain, a ball arriving from a parent passes through an unobserved middle node to its children. At an observed middle node, it cannot continue down the chain.
- In a fork, a ball traveling up from one child can pass through an unobserved parent to another child. Observing that parent stops this move.
- In a collider, a ball arriving from one parent cannot turn toward the other parent while the collider is unobserved. If the collider is observed, it bounces back toward the parents, opening the connection.

If a collider has an observed descendant, the ball can travel down to that descendant and then bounce upward. On returning to the unobserved collider **from a child**, it may travel to the collider's other parent. This is how the rules account for observed descendants without first listing them.

### Keeping track of states

To ask whether unobserved query nodes $X$ and $Y$ are d-separated given $\mathcal O$:

1. Start a ball at $X$ as though it arrived from a child. This is an initialization convention; $X$ does not need to have a child.
2. Apply the table at each visited node and queue the allowed next moves.
3. Record the pair **(node, arrival direction)**, not just the node. Never process the same pair twice.
4. If a ball reaches $Y$, an active connection exists, so the graph does not guarantee conditional independence.
5. If all possible moves are exhausted without reaching $Y$, the two nodes are d-separated.

For multiple starting nodes, initialize each one in the same way. The following Python-style pseudocode records all reachable unobserved nodes:

~~~python
def bayes_ball(starts, observed, parents, children):
    # A state contains a node and the direction from which we arrived.
    agenda = [(node, "from_child") for node in starts]
    seen = set()
    reachable = set()

    while agenda:
        node, arrival = agenda.pop()
        state = (node, arrival)
        if state in seen:
            continue
        seen.add(state)

        if node not in observed:
            reachable.add(node)

        if arrival == "from_child":
            if node not in observed:
                agenda.extend(
                    (parent, "from_child") for parent in parents[node]
                )
                agenda.extend(
                    (child, "from_parent") for child in children[node]
                )
            # An observed node stops a ball arriving from a child.
        else:  # Arrived from a parent.
            if node in observed:
                agenda.extend(
                    (parent, "from_child") for parent in parents[node]
                )
            else:
                agenda.extend(
                    (child, "from_parent") for child in children[node]
                )

    return reachable
~~~

The dictionaries named parents and children store each node's immediate neighbors, including empty lists for nodes without parents or children. A move upward arrives at the next node “from a child”; a move downward arrives “from a parent.” The input observed is the set $\mathcal O$, and starts contains unobserved query nodes.

To test the query, check whether $Y$ belongs to the returned set. Reachability is a structural test, not a calculation of a posterior probability.

### Worked traversal: an observed descendant opens a path

Use the earlier graph with $X\to M\to Y$, $X\to C\leftarrow Y$, and $C\to W$.

First observe only $M$:

- From $X$, the ball can go down to $M$ or $C$.
- At observed $M$, it arrives from a parent. It can bounce back to $X$, but it cannot continue to $Y$.
- At unobserved $C$, it also arrives from a parent. It can go down to $W$, but cannot turn upward to $Y$.
- Unobserved $W$ has no children, so that route stops.

No ball reaches $Y$. This confirms $X\perp\!\!\!\perp Y\mid M$.

Now observe both $M$ and $W$. The chain remains blocked at $M$, but the other route changes:

$$
X
\ \longrightarrow\ C
\ \longrightarrow\ W
\ \longrightarrow\ C
\ \longrightarrow\ Y.
$$

These arrows describe the **ball's travel**, not the orientations of the graph's edges. The ball arrives at observed $W$ from its parent, so it bounces up to $C$. It now arrives at unobserved $C$ from a child, which permits a move to either parent, including $Y$.

The traversal revisits $C$, but in a different arrival state. This explains why remembering only “already visited $C$” would give the wrong result. The ball's walk corresponds to the now-active path $X\to C\leftarrow Y$, opened by the observed descendant $W$.

There are at most two arrival states per node. With adjacency lists and constant-time set membership, this traversal takes time proportional to $K+L$, where $K$ is the number of nodes and $L$ is the number of directed edges. It avoids enumerating all possible paths.

Bayes-ball answers which independences are guaranteed by the graph. Numerical inference—such as computing $P(X=1\mid W=1)$—still requires the local probability distributions and sums or integrals.

## Practice problems

### 1. Building and using a joint distribution

Consider $B\to A\leftarrow F$ and $A\to T$, where all four variables are binary.

1. Write the joint factorization.
2. If $P(B=1)=0.2$, $P(F=0)=0.9$, $P(A=1\mid B=1,F=0)=0.8$, and $P(T=1\mid A=1)=0.7$, compute $P(B=1,F=0,A=1,T=1)$.
3. Write a marginalization expression for $P(T=1)$.
4. Does the graph guarantee $B\perp\!\!\!\perp F$? Does it guarantee $B\perp\!\!\!\perp F\mid A$? Does it guarantee $T\perp\!\!\!\perp(B,F)\mid A$?

~~~{admonition} Solution 1
:class: dropdown

Each node contributes one factor conditioned on its parents:

$$
p(b,f,a,t)=p(b)p(f)p(a\mid b,f)p(t\mid a).
$$

The requested joint probability is

$$
0.2(0.9)(0.8)(0.7)=0.1008.
$$

To find the ticket probability, sum over the eight possible triples $(b,f,a)$:

$$
P(T=1)
=\sum_{b=0}^{1}\sum_{f=0}^{1}\sum_{a=0}^{1}
p(b)p(f)p(a\mid b,f)p(T=1\mid a).
$$

The only path between $B$ and $F$ has the collider $A$. With no observations, it is blocked, so $B\perp\!\!\!\perp F$ is guaranteed. Observing $A$ opens it, so $B\perp\!\!\!\perp F\mid A$ is not guaranteed.

The paths from either fault to $T$ pass through $A$ as a non-collider. Observing $A$ blocks those paths, so $T\perp\!\!\!\perp(B,F)\mid A$ is guaranteed.
~~~

### 2. Dependence from a shared parent

Let $Z\to X$ and $Z\to Y$, with no other edges. Assume

$$
\begin{aligned}
P(Z=1)&=0.4,\\
P(X=1\mid Z=1)&=0.75,&P(X=1\mid Z=0)&=0.25,\\
P(Y=1\mid Z=1)&=0.8,&P(Y=1\mid Z=0)&=0.2.
\end{aligned}
$$

1. Find $P(X=1)$, $P(Y=1)$, and $P(X=1,Y=1)$.
2. Are $X$ and $Y$ independent?
3. Find $P(X=1,Y=1\mid Z=1)$, and explain why conditional independence holds given $Z$.

~~~{admonition} Solution 2
:class: dropdown

Average over the two possible values of $Z$:

$$
\begin{aligned}
P(X=1)&=0.4(0.75)+0.6(0.25)=0.45,\\
P(Y=1)&=0.4(0.8)+0.6(0.2)=0.44.
\end{aligned}
$$

Within each value of $Z$, the joint conditional probability is the product of the two local probabilities. Thus

$$
\begin{aligned}
P(X=1,Y=1)
&=0.4(0.75)(0.8)+0.6(0.25)(0.2)\\
&=0.27.
\end{aligned}
$$

Independence would require $P(X=1,Y=1)=0.45(0.44)=0.198$. Because $0.27\ne0.198$, independence fails.

Given $Z=1$,

$$
P(X=1,Y=1\mid Z=1)=0.75(0.8)=0.6.
$$

The graph gives $p(x,y\mid z)=p(x\mid z)p(y\mid z)$ for all possible values, not just for this one pair. Observing $Z$ blocks the fork, so $X\perp\!\!\!\perp Y\mid Z$.
~~~

### 3. D-separation and Bayes-ball

Use the graph $X\to M\to Y$, $X\to C\leftarrow Y$, and $C\to W$.

1. Are $X$ and $Y$ d-separated when nothing is observed?
2. Are they d-separated given $M$?
3. Are they d-separated given both $M$ and $C$?
4. Are they d-separated given both $M$ and $W$? Trace the Bayes-ball moves that justify your answer.

~~~{admonition} Solution 3
:class: dropdown

There are two paths to check: $X,M,Y$ and $X,C,Y$.

1. With no observations, the chain through $M$ is active. The collider through $C$ is blocked, but one active path is enough. The nodes are **not d-separated**.
2. Given $M$, the chain is blocked by an observed non-collider. The collider $C$ and its descendant $W$ remain unobserved, so the second path is also blocked. The nodes **are d-separated**.
3. Given $M,C$, the chain stays blocked, but observing the collider $C$ opens the second path. The nodes are **not d-separated**.
4. Given $M,W$, the chain is blocked. Although $C$ itself is unobserved, its descendant $W$ is observed, so the collider path is active. The nodes are **not d-separated**.

For the last query, start at $X$ in the from-child state. The ball moves to $C$ in the from-parent state, then to $W$ in the from-parent state. Observed $W$ sends it back to $C$ in the from-child state. Unobserved $C$ now allows travel to its parents, including $Y$.

The two visits to $C$ use different states, so the second must not be discarded. Reaching $Y$ confirms an active connection; it does not by itself quantify the dependence for any particular choice of probability tables.
~~~

### 4. Finding a Markov blanket

Consider the graph $U\to P\to X$, $X\to C_1\leftarrow Q$, and $X\to C_2\to V$.

1. Identify the parents, children, and co-parents needed for the Markov blanket of $X$.
2. State the conditional-independence relationship between $X$ and the nodes outside its blanket.
3. Write the conditional distribution of $X$ given all other variables, up to a normalizing constant.
4. Explain why the blanket needs $Q$, even though $Q$ is not directly adjacent to $X$.

~~~{admonition} Solution 4
:class: dropdown

The parent is $P$, the children are $C_1,C_2$, and the other parent of child $C_1$ is $Q$. Therefore

$$
\operatorname{MB}(X)=\{P,Q,C_1,C_2\}.
$$

The outside nodes are $U,V$, giving

$$
X\perp\!\!\!\perp(U,V)\mid(P,Q,C_1,C_2).
$$

To obtain the full conditional, keep only the joint factors that contain $x$:

$$
p(x\mid u,p,q,c_1,c_2,v)
\propto p(x\mid p)p(c_1\mid x,q)p(c_2\mid x).
$$

If $X$ is discrete, the normalized expression is

$$
p(x\mid u,p,q,c_1,c_2,v)
=
\frac{p(x\mid p)p(c_1\mid x,q)p(c_2\mid x)}
{\displaystyle\sum_{x'}
p(x'\mid p)p(c_1\mid x',q)p(c_2\mid x')}.
$$

Here $x'$ is a dummy variable running over all possible values of $X$; the observed values in the conditioning set are fixed. For continuous $X$, replace the sum by an integral.

Observing $C_1$ opens the collider $X\to C_1\leftarrow Q$. Consequently, $Q$ can provide information about $X$ after their shared child is known. Including $Q$ in the blanket accounts for this dependence; the factor $p(c_1\mid x,q)$ also makes its role visible algebraically.
~~~

## Reading

Christopher M. Bishop, *Pattern Recognition and Machine Learning*, Chapter 8, Sections 8.1–8.2, especially Sections 8.2.1–8.2.2 on the three example graphs and d-separation.

## References

- Stanford CS228, [*Directed Graphical Models*](https://ermongroup.github.io/cs228-notes/representation/directed/), for additional explanations of Bayesian-network factorization and d-separation.
- Ross D. Shachter, [*Bayes-Ball: The Rational Pastime*](https://web.stanford.edu/~shachter/pubs/bayesbl.pdf), UAI 1998, for the original algorithm and its extensions.
