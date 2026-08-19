# Lecture 1 - Mathematical Foundations

This lecture reviews the linear algebra and multivariable calculus used throughout the course. An additional introduction to python notebook is available here: .

## Learning objectives

After completing this lecture, you should be able to:

- identify the dimensions of vectors and matrices;
- compute dot products, matrix products, and common norms;
- interpret eigenvectors and eigenvalues; and
- compute derivatives, partial derivatives, and gradients.

## Scalars, vectors, and matrices

A **scalar** is a single number, such as $3$, $-1.5$, or $\pi$. We usually write scalars with lowercase italic letters, such as $a$ or $x$.

A **vector** is an ordered list of numbers. A vector with $n$ entries belongs to $\mathbb{R}^n$.

$$
\mathbf{x}
=
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{bmatrix}
\in \mathbb{R}^n.
$$

Two vectors can be added when they have the same number of entries, and a vector can be multiplied by a scalar:

$$
\begin{bmatrix}1\\2\end{bmatrix}
+
\begin{bmatrix}3\\-1\end{bmatrix}
=
\begin{bmatrix}4\\1\end{bmatrix},
\qquad
2\begin{bmatrix}1\\2\end{bmatrix}
=
\begin{bmatrix}2\\4\end{bmatrix}.
$$

A **matrix** is a rectangular array of numbers. If $A$ has $m$ rows and $n$ columns, then $A\in\mathbb{R}^{m\times n}$:

$$
A=
\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}.
$$

The entry $a_{ij}$ is in row $i$ and column $j$. The **transpose** $A^T$ exchanges rows and columns. For example,

$$
A=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}
\quad\Longrightarrow\quad
A^T=\begin{bmatrix}1&4\\2&5\\3&6\end{bmatrix}.
$$

The $n\times n$ **identity matrix** $I_n$ has ones on its main diagonal and zeros elsewhere. It satisfies $I_n\mathbf{x}=\mathbf{x}$ and $AI_n=A$ whenever the dimensions are compatible.

## Dot products

The **dot product** of two vectors $\mathbf{x},\mathbf{y}\in\mathbb{R}^n$ is

$$
\mathbf{x}\cdot\mathbf{y}
=\mathbf{x}^T\mathbf{y}
=\sum_{i=1}^{n}x_i y_i.
$$

For example,

$$
\begin{bmatrix}1\\2\\-1\end{bmatrix}
\cdot
\begin{bmatrix}3\\0\\4\end{bmatrix}
=1(3)+2(0)+(-1)(4)=-1.
$$

The dot product measures alignment. If $\theta$ is the angle between two nonzero vectors, then

$$
\mathbf{x}\cdot\mathbf{y}
=\lVert\mathbf{x}\rVert_2\lVert\mathbf{y}\rVert_2\cos\theta.
$$

Vectors are **orthogonal** when their dot product is zero.

## Matrix multiplication

The product $AB$ is defined when the number of columns of $A$ equals the number of rows of $B$. If

$$
A\in\mathbb{R}^{m\times n}
\quad\text{and}\quad
B\in\mathbb{R}^{n\times p},
$$

then $AB\in\mathbb{R}^{m\times p}$ and

$$
(AB)_{ij}=\sum_{k=1}^{n}a_{ik}b_{kj}.
$$

Each entry is the dot product of a row of $A$ with a column of $B$. For example,

$$
\begin{bmatrix}1&2\\3&4\end{bmatrix}
\begin{bmatrix}5\\6\end{bmatrix}
=
\begin{bmatrix}
1(5)+2(6)\\
3(5)+4(6)
\end{bmatrix}
=
\begin{bmatrix}17\\39\end{bmatrix}.
$$

Matrix multiplication is associative and distributive, but it is generally **not commutative**: $AB$ does not usually equal $BA$.

## Norms

A **norm** measures the size of a vector. Three common vector norms are

$$
\begin{aligned}
\lVert\mathbf{x}\rVert_1
&=\sum_{i=1}^{n}|x_i|,\\
\lVert\mathbf{x}\rVert_2
&=\sqrt{\sum_{i=1}^{n}x_i^2},\\
\lVert\mathbf{x}\rVert_\infty
&=\max_i |x_i|.
\end{aligned}
$$

For $\mathbf{x}=[3,-4]^T$,

$$
\lVert\mathbf{x}\rVert_1=7,
\qquad
\lVert\mathbf{x}\rVert_2=5,
\qquad
\lVert\mathbf{x}\rVert_\infty=4.
$$

The distance between vectors $\mathbf{x}$ and $\mathbf{y}$ is often measured by $\lVert\mathbf{x}-\mathbf{y}\rVert_2$.

For a matrix $A\in\mathbb{R}^{m\times n}$, the **Frobenius norm** treats all entries as one long vector:

$$
\lVert A\rVert_F
=\sqrt{\sum_{i=1}^{m}\sum_{j=1}^{n}a_{ij}^2}.
$$

## Eigenvectors and eigenvalues

Let $A\in\mathbb{R}^{n\times n}$ be a square matrix. A nonzero vector $\mathbf{v}$ is an **eigenvector** of $A$ if

$$
A\mathbf{v}=\lambda\mathbf{v},
$$

where the scalar $\lambda$ is the corresponding **eigenvalue**. Multiplying by $A$ changes the length or direction of an eigenvector only by the factor $\lambda$.

For example,

$$
A=\begin{bmatrix}2&0\\0&3\end{bmatrix}.
$$

The coordinate vectors are eigenvectors because

$$
A\begin{bmatrix}1\\0\end{bmatrix}
=2\begin{bmatrix}1\\0\end{bmatrix},
\qquad
A\begin{bmatrix}0\\1\end{bmatrix}
=3\begin{bmatrix}0\\1\end{bmatrix}.
$$

Thus, the eigenvalues are $2$ and $3$. In general, eigenvalues are found by solving the characteristic equation

$$
\det(A-\lambda I)=0
$$

and then solving $(A-\lambda I)\mathbf{v}=\mathbf{0}$ for each eigenvalue. Eigenvectors are not unique: any nonzero scalar multiple of an eigenvector points in the same eigendirection.

## Derivatives

For a single-variable function $f(x)$, the **derivative** is the instantaneous rate of change of the output with respect to the input:

$$
f'(x)
=\frac{df}{dx}
=\lim_{h\to 0}\frac{f(x+h)-f(x)}{h}.
$$

Geometrically, $f'(x)$ is the slope of the tangent line to the graph of $f$ at $x$. For example, if

$$
f(x)=x^3-2x,
$$

then

$$
f'(x)=3x^2-2.
$$

Useful rules include

$$
\frac{d}{dx}x^n=nx^{n-1},
\qquad
\frac{d}{dx}(fg)=f'g+fg',
$$

and the chain rule

$$
\frac{d}{dx}f(g(x))=f'(g(x))g'(x).
$$

## Partial derivatives

A function of several variables can change in several independent directions. A **partial derivative** measures the change with respect to one variable while holding the others fixed.

For

$$
f(x,y)=x^2y+\sin y,
$$

the partial derivatives are

$$
\frac{\partial f}{\partial x}=2xy,
\qquad
\frac{\partial f}{\partial y}=x^2+\cos y.
$$

The symbol $\partial$ distinguishes a partial derivative from the ordinary derivative $d/dx$ of a single-variable function.

## Gradients

The **gradient** collects all first-order partial derivatives of a scalar-valued function into a vector. For $f:\mathbb{R}^n\to\mathbb{R}$,

$$
\nabla f(\mathbf{x})
=
\begin{bmatrix}
\dfrac{\partial f}{\partial x_1}\\
\dfrac{\partial f}{\partial x_2}\\
\vdots\\
\dfrac{\partial f}{\partial x_n}
\end{bmatrix}.
$$

For the previous example,

$$
\nabla f(x,y)
=
\begin{bmatrix}
2xy\\
x^2+\cos y
\end{bmatrix}.
$$

At a given point, the gradient points in the direction of steepest increase of $f$. Its negative, $-\nabla f$, points in the direction of steepest decrease, which is why gradient-based optimization updates often take the form

$$
\mathbf{x}_{k+1}
=\mathbf{x}_k-\alpha\nabla f(\mathbf{x}_k),
$$

where $\alpha>0$ is the step size.

## Practice problems

1. Compute the dot product of $[2,-1,3]^T$ and $[4,2,0]^T$.
2. Multiply $\begin{bmatrix}1&0\\-2&3\end{bmatrix}$ by $[4,5]^T$.
3. Find the $1$-, $2$-, and $\infty$-norms of $[-2,1,2]^T$.
4. Find the eigenvalues and eigenvectors of $\begin{bmatrix}4&0\\0&-1\end{bmatrix}$.
5. For $f(x,y)=x^2+3xy+y^2$, compute both partial derivatives and the gradient.
