# Lecture 11 - Machine Learning Basics

A model can describe the data we have already collected extremely well and still make poor predictions tomorrow due to overfitting. Therefore, high training score alone cannot tell us if "generalizable" learning has taken place. The central question of this lecture is:

> How do we develop and evaluate a model so that its performance on available data tells us something useful about its performance on new data?

Lecture 7 introduced parameter estimation through likelihood, and lecture 9 explained how an estimate changes across random samples. We now use those ideas to study **generalization**: the ability of a learned model to perform well on previously unseen observations from the population of interest.

## From parameter estimation to prediction

In supervised learning, each observation consists of an input and a target:

$$
\mathcal{D}=\{(\mathbf{x}_n,t_n)\}_{n=1}^{N},
\qquad
\mathbf{x}_n\in\mathbb{R}^{D}.
$$

For example, $\mathbf{x}_n$ could contain operating temperature, load, and vibration measurements, while $t_n$ is the energy consumed during the next hour for a machine. As described briefly in earlier lectures, predicting a real-valued target is **regression**. Predicting a category, such as whether a component will fail, is **classification**.

A prediction function $f(\mathbf{x};\mathbf{w})$ contains parameters $\mathbf{w}$ learned from data. For a single input, the polynomial model from Lecture 1 is

$$
f(x;\mathbf{w})=\sum_{j=0}^{M}w_jx^j.
$$

Once we choose the degree $M$, fitting the model determines the coefficients $w_0,\ldots,w_M$. However, fitting coefficients does not answer another important question: which degree should we choose?

This separates two tasks that will recur throughout the lecture: fitting a particular model, and selecting the modeling choices under which it is fitted.

## Training error and population error

### A loss measures the cost of a prediction

A loss function compares a prediction with its target. In regression, a common choice is squared error:

$$
L(t,f(\mathbf{x}))=(t-f(\mathbf{x}))^2.
$$

<!-- For a classifier that predicts a class label, the zero–one loss is 0 for a correct prediction and 1 otherwise. Averaging this loss gives the misclassification rate. While different applications can require different losses - missing a dangerous fault may matter more than raising a false alarm, for the purposes of this lecture, we consider all all misclassifications to have the same weight. -->

For clarity, most calculations below use squared error to illustrate some important topics in machine learning.

### Training error is measured on the fitting data

Suppose we have $N_{\mathrm{tr}}$ training examples. For any proposed parameter values $\mathbf{w}$, we can make a prediction for each training input, compare it with the observed target using the loss $L$, and average those losses. This average is the **empirical risk** on the training data:

$$
\widehat{R}_{\mathrm{tr}}(\mathbf{w})
=\frac{1}{N_{\mathrm{tr}}}
\sum_{n=1}^{N_{\mathrm{tr}}}
L(t_n,f(\mathbf{x}_n;\mathbf{w})).
$$

Here $f(\mathbf{x}_n;\mathbf{w})$ is the prediction for example $n$, $t_n$ is its observed target, and $L$ tells us how costly the prediction error is. The hat on $\widehat{R}$ indicates that the average is calculated from a finite sample. At this stage, $\mathbf{w}$ can be *any* proposed parameter values: the formula gives a different average loss for each choice.

For example, take two training pairs $(x,t)=(1,2)$ and $(2,4)$, the model $f(x;w)=wx$, and squared-error loss. Trying $w=1$ gives predictions 1 and 2, so its empirical risk is

$$
\widehat{R}_{\mathrm{tr}}(1)
=\frac{(2-1)^2+(4-2)^2}{2}
=2.5.
$$

Trying $w=2$ gives predictions 2 and 4, so $\widehat{R}_{\mathrm{tr}}(2)=0$. A learning procedure uses the training pairs to choose a fitted value, written $\widehat{w}$. In this simple example, minimizing the average squared loss gives $\widehat{w}=2$. The **training error** is the empirical risk *evaluated at the fitted value*:

$$
\text{training error}
=\widehat{R}_{\mathrm{tr}}(\widehat{w})
=\widehat{R}_{\mathrm{tr}}(2)
=0.
$$

With many parameters, we write $\widehat{\mathbf{w}}$ instead of $\widehat{w}$. The distinction is the same: $\widehat{R}_{\mathrm{tr}}(\mathbf{w})$ describes the loss for a proposed parameter choice, while $\widehat{R}_{\mathrm{tr}}(\widehat{\mathbf{w}})$ is the loss after fitting. A regularized learning procedure may choose $\widehat{\mathbf{w}}$ by minimizing a loss *plus a penalty*. We still report its training error using the average loss itself, unless we explicitly say that the penalty is included.

Because the parameters were chosen using the training observations, those observations give the model an advantage. A flexible polynomial may pass through every training target, including its measurement noise, as seen in the overfitting example from lecture 1. Its training error can then be zero without its future prediction error being zero.

<!-- For nested model classes, minimizing the same unregularized training loss exactly cannot increase the best achievable training error as the class expands. This says nothing by itself about performance on new observations. -->

### Population error is the quantity we want

Let $p(\mathbf{x},t)$ represent the joint distribution of inputs and targets that we expect to encounter in use. For a fixed fitted predictor $f$, its **population risk** is

$$
R(f)
=\mathbb{E}_{(\mathbf{X},T)\sim p}
\left[L(T,f(\mathbf{X}))\right].
$$

For continuous inputs and targets, this can be written as

$$
R(f)=\iint L(t,f(\mathbf{x}))p(\mathbf{x},t)\,dt\,d\mathbf{x}.
$$

The expectation averages over new examples, including examples absent from the training sample. We generally cannot evaluate it exactly because the data-generating distribution is unknown.

A held-out test set provides an empirical approximation:

$$
\widehat{R}_{\mathrm{test}}(f)
=\frac{1}{N_{\mathrm{test}}}
\sum_{n=1}^{N_{\mathrm{test}}}
L(t_n^{\mathrm{test}},f(\mathbf{x}_n^{\mathrm{test}})).
$$

If test observations are independent draws from the target population and independent of the procedure that produced $f$, then, conditional on the fitted predictor,

$$
\mathbb{E}_{\mathcal{D}_{\mathrm{test}}}
[\widehat{R}_{\mathrm{test}}(f)\mid f]
=R(f).
$$

The realized test error is still an estimate: another test sample would give a different value. This is a direct application of Lecture 9's discussion of sampling variability.

The **generalization gap** is the difference $R(f)-\widehat{R}_{\mathrm{tr}}(f)$. In practice we often examine test error minus training error as an estimate of this gap. Training error is often optimistic, but a particular finite test set need not have a larger error.

:::{important}
A test set estimates performance for the population it represents.  A motivating example for this statement - randomly holding out records from one factory does not establish performance in another factory with different sensors or operating conditions.
:::

## Parameters and hyperparameters

We have now seen how to measure the training loss of a model once its parameters have been fitted. But we still have to decide *which model to fit*. Consider polynomial regression: a straight line and a ninth-degree polynomial both have coefficients that can be learned from the same data, yet they give the learner very different freedom to bend around those observations.

The coefficients $\mathbf{w}$ are **parameters**. They are the unknown values adjusted while fitting a particular polynomial. The degree $M$ is a **hyperparameter**: choosing it determines which polynomial family we will fit. A regularization strength $\lambda$ is another hyperparameter because it controls how strongly large coefficients are discouraged during fitting.

| Choice | Example | How it is determined |
|---|---|---|
| Parameter | Polynomial coefficient $w_j$ | Fitted using training data |
| Hyperparameter | Polynomial degree $M$ | Selected using validation or cross-validation |
| Hyperparameter | Regularization strength $\lambda$ | Selected using validation or cross-validation |
| Hyperparameter | Learning rate | Chosen for the optimization procedure and assessed during development |

To see the two decisions in one expression, consider regularized polynomial regression:

$$
\widehat{\mathbf{w}}_{M,\lambda}
=\underset{\mathbf{w}}{\operatorname{argmin}}
\left[
\frac{1}{N_{\mathrm{tr}}}
\sum_{n=1}^{N_{\mathrm{tr}}}
\left(t_n-f_M(x_n;\mathbf{w})\right)^2
+\lambda\sum_{j=1}^{M}w_j^2
\right].
$$

We first choose $M$ and $\lambda$. For those choices, the minimization finds coefficients $\widehat{\mathbf{w}}_{M,\lambda}$ that balance squared training error against the penalty. If we change $M$ or $\lambda$, we solve a new fitting problem and obtain another set of coefficients.

The penalty regularization term discourages large non-intercept coefficients, but its effect depends on the scales of the input features. Note that hyperparameters may be selected automatically by a search algorithm (the simplest one is just iterating over different combinations of hyperparameters and choosing the best one - meaning we run the same experiment multiple times with the different hyperparameter combinations). Therefore, the name describes their role, not whether a person chose them by hand.

How should we compare these candidates? The training loss alone is unreliable for this purpose because a more flexible model has more opportunities to fit the peculiarities of the training sample. Thus, the model would perform exceptionally well on training data, but has not learned to generalize. We need observations that did not determine its coefficients.

## Training, validation, and test data

That leads to three distinct roles for data. The **training set** supplies the data used to fit parameters. A **validation set** supplies data used for feedback on user choices such as polynomial degree and regularization strength. A **test set** is reserved until those choices have been made so that we can assess the completed modeling procedure.

| Dataset | Role | Example |
|---|---|---|
| Training | Fit parameters and learned preprocessing | Estimate polynomial coefficients and feature means |
| Validation | Compare development choices | Choose degree and regularization strength |
| Test | Assess the completed procedure | Report error after the choices are fixed |

Suppose the following mean squared errors are obtained:

| Polynomial degree | Training MSE | Validation MSE |
|---|---:|---:|
| 1 | 8.0 | 8.4 |
| 3 | 2.0 | 2.8 |
| 9 | 0.1 | 7.5 |

Degree 9 has the lowest training MSE, which is unsurprising: it has much more flexibility than the other candidates. But its validation MSE is much worse. Degree 3 gives the lowest error on observations that did not determine its coefficients, so it is the better choice on this evidence. The difference suggests that degree 9 is fitting some accidental features of this particular training sample - it is clearly overfitting.

It may be tempting to inspect test performance to make this choice more confidently. However, doing so would make the test set another source of development feedback similar to the validation set. Even without changing the fitted coefficients directly, repeated choices based on test scores can favor a model that happens to perform well on that particular test sample. Its reported test error would then tend to be optimistic. Best practice is to use the test set once at the very end of the whole training-validation cycle after learning the best parameter-hyperparameter combination.

A typical workflow is:

1. Set aside a representative test set.
2. Divide the remaining development data into training and validation sets.
3. Fit candidate procedures on training data and compare validation performance.
4. Freeze the selected choices.
5. Optionally refit the selected procedure on the combined training and validation data.
6. Evaluate the resulting fitted procedure on the test set.

After selecting the degree, we often refit it using all available development data to make fuller use of the observations. Any learned preprocessing is refitted at the same time. Only then do we evaluate the resulting model on the test set.

The validation set is not an unlimited source of independent evidence either. If we repeatedly invent new models in response to its scores, we can gradually adapt to its random quirks. Keeping the test set separate preserves one final check on the entire development process.

### Split according to the prediction task

The word *unseen* must match the intended use of the model. If deployment means predicting new, independent observations from the same population, a random split may be reasonable. If deployment means predicting a new machine or forecasting next month, a random split of all rows can answer the wrong question.

For instance, suppose we have hundreds of sensor readings from each machine. If readings from one machine appear in both training and test sets, the model may benefit from characteristics of that familiar machine. To estimate performance on *new* machines, all readings from each machine should stay in a single split.

For forecasting, the ordering matters. Train on earlier observations and evaluate on later ones. Randomly mixing past and future records can let the training procedure learn from conditions that had not occurred at the time of the forecast.

For classification with rare classes, stratification can help preserve class proportions in the splits. It cannot repair a split that mixes observations from the same machine or lets future observations influence a past prediction.

## Cross-validation and model selection

The earlier comparison used one validation set. What if degree 3 won only because that particular data split into train-validation-test happened to contain unusually easy examples in validation? When development data are limited, this possibility matters: a different split can yield a different choice.

**Cross-validation** gives each development observation a turn in validation. Rather than relying on a single division, we repeatedly fit a fresh model on part of the development data and score it on the part left out. We then compare candidates using their scores across all these divisions. The final test set still remains separate.

### How $K$-fold cross-validation works

In $K$-fold cross-validation, partition the development data into $K$ disjoint groups called *folds*, $\mathcal{I}_1,\ldots,\mathcal{I}_K$ after keeping the final test set aside. If $K=5$, for example, each fit uses four folds for training and the fifth for validation; repeating this five times gives every fold one turn as the held-out fold.

For each candidate hyperparameter setting $h$ and each fold $k$:

1. Fit a fresh version of the entire procedure using all development folds except $\mathcal{I}_k$.
2. Predict the targets in $\mathcal{I}_k$.
3. Record the validation loss on that fold.

The superscript $(-k)$ in $f_h^{(-k)}$ reminds us that this predictor was fitted *without* fold $k$. Its validation loss on that fold is

$$
E_k(h)
=\frac{1}{|\mathcal{I}_k|}
\sum_{n\in\mathcal{I}_k}
L(t_n,f_h^{(-k)}(\mathbf{x}_n)).
$$

If the folds have the same number of observations, average their losses to obtain one score for setting $h$:

$$
\operatorname{CV}(h)=\frac{1}{K}\sum_{k=1}^{K}E_k(h).
$$

If fold sizes differ, a simple mean would give a small fold the same influence as a large one. To obtain the average held-out loss *per observation*, weight each fold by its size:

$$
\operatorname{CV}(h)
=\frac{\sum_{k=1}^{K}|\mathcal{I}_k|E_k(h)}
{\sum_{k=1}^{K}|\mathcal{I}_k|}.
$$

This produces a development score for each candidate. After selecting one, fit it again using all development data and assess that final fit on the reserved test set. The $K$ intermediate models served to evaluate the candidate; ordinary cross-validation does not combine their predictions into the final model.

### Example

Suppose two degrees give these three-fold validation MSEs:

| Degree | Fold 1 | Fold 2 | Fold 3 | Mean |
|---|---:|---:|---:|---:|
| 2 | 3.0 | 3.6 | 3.3 | 3.3 |
| 5 | 2.4 | 4.2 | 4.8 | 3.8 |

On the first fold, degree 5 looks preferable. The remaining folds tell a different story, and degree 2 has the lower average error. This is why repeated validation can be more informative than a single held-out result. We would select degree 2 under this criterion, then fit its coefficients again using all development data.

If two candidates have similar predictive performance, computational cost, interpretability, or stability may also matter. Such criteria belong to development decisions and they should not be invented after seeing the final test scores. Occam's razor dictates that, more often than not, in cases where two candidates have similar predictive performance, it is usually better to select the "simpler" hyperparameter set.

### Limits of cross-validation

It might seem that using more folds must always give a better assessment. In **leave-one-out cross-validation**, $K$ equals the number of development observations, so each fit holds out one observation and trains on almost everything else. This uses data efficiently for each fit, but requires many fits and is not automatically preferable to a smaller $K$.

Nor should we treat the $K$ fold scores as $K$ independent random samples. Their training sets overlap, so their scores are dependent. Dividing the standard deviation of the fold scores by $\sqrt{K}$ does not automatically give the ordinary independent-sample standard error from Lecture 9.

Cross-validation also does not eliminate the effect of searching. If we try many settings and select the lowest score, part of that winning result may reflect favorable chance variation across folds. The reserved test set assesses the chosen procedure once the search is complete. Whatever the number of folds, the split must respect the prediction task. If the task concerns new machines, folds should keep each machine's readings together. If the task is forecasting, validation observations must come after the observations used for fitting. Repeating an inappropriate split does not make the result more trustworthy.

<!-- When data are too limited for a simple final holdout, **nested cross-validation** uses an inner loop to select hyperparameters and an outer loop to evaluate the whole selection procedure. -->

For implementation examples and split strategies, see the [scikit-learn cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html).

## Data leakage and preprocessing

So far, we have treated the boundaries between training, validation, and test data as if they were easy to maintain. In practice, information can cross those boundaries through a seemingly harmless step in data preparation. This is **data leakage**: information used during fitting or model selection would not have been available at the stage being evaluated, or would not be available when a real prediction must be made.

Consider a machine-failure task. Several quite different mistakes have the same underlying effect:

- a feature recording a repair performed after failure reveals information from after the prediction time;
- copies of the same measurement in training and test data make the test examples partially familiar;
- choosing features by their association with targets across the complete dataset lets held-out targets influence development;
- changing the model after inspecting its test errors uses the test set for model selection.

In each case, the model or its development process gains access to information that a future prediction would lack. Leakage need not produce an obvious warning or an impossibly high score. It can quietly make the evaluation look better than performance on genuinely new data.

### Fit a transformation on training data

Preprocessing is a common place for leakage because it often looks like a preliminary calculation rather than part of learning. Suppose one numerical feature has a different scale from the others, so we decide to standardize the data by centering each feature using its mean and dividing by itd standard deviation. Those two numbers must be learned from the training observations:

$$
\widehat{\mu}_{\mathrm{tr}}
=\frac{1}{N_{\mathrm{tr}}}\sum_{n=1}^{N_{\mathrm{tr}}}x_n,
\qquad
\widehat{s}_{\mathrm{tr}}
=\sqrt{\frac{1}{N_{\mathrm{tr}}}
\sum_{n=1}^{N_{\mathrm{tr}}}(x_n-\widehat{\mu}_{\mathrm{tr}})^2}.
$$

Here the denominator $N_{\mathrm{tr}}$ defines a useful preprocessing scale; we are not trying to construct an unbiased estimator of population variance. Once these quantities have been calculated for a nonconstant feature, transform any value by

$$
z=\frac{x-\widehat{\mu}_{\mathrm{tr}}}{\widehat{s}_{\mathrm{tr}}}.
$$

The word *fitted* matters: $\widehat{\mu}_{\mathrm{tr}}$ and $\widehat{s}_{\mathrm{tr}}$ are now part of the trained procedure. We must apply that same transformation to training, validation, test, and future inputs. Computing a separate validation or test mean would change what the model receives and would let held-out data determine part of the procedure.

As a concrete example, for training values $2,4,6$,

$$
\widehat{\mu}_{\mathrm{tr}}=4,
\qquad
\widehat{s}_{\mathrm{tr}}=\sqrt{\frac{8}{3}}.
$$

A held-out value $8$ is therefore transformed to

$$
z=\frac{8-4}{\sqrt{8/3}}\approx2.45.
$$

If we had included 8 while computing the mean, the mean would instead be 5. Even though 8 is an input value rather than a target, it would have influenced the procedure used during fitting. The held-out evaluation would no longer reproduce what happens when the fitted procedure meets a truly new value.

The same reasoning applies when we learn values for missing-data imputation, choose features using labels, or learn a lower-dimensional representation. These operations have a *fit* stage that must use only permitted data. A fixed conversion specified independently of the dataset, such as Celsius to kelvin, has no such fit stage.

### Preprocessing must be inside each cross-validation fit

Cross-validation makes the training boundary move from one fold to the next. In fold $k$, fit the preprocessing quantities using only the other $K-1$ folds, then apply that fitted transformation to fold $k$. When a different fold is held out, start over and calculate new preprocessing quantities from its corresponding training folds.

If we scale the entire development dataset once *before* cross-validation, each fold contributes to the mean and scale used in its own validation. Reserving a final test set does not repair this error in the development scores.

A **pipeline** treats preprocessing and prediction as one procedure. Each fold fits the whole sequence on that fold's training subset, reducing the chance that a transformation is fitted on the wrong data. After model selection, refit the complete pipeline on all development data. The final test inputs then pass through the transformations learned from those development data.

For concrete examples, see [scikit-learn's common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html).

## The curse of dimensionality

Suppose we want to predict a machine's energy use from its temperature. We have many observations near any particular temperature, so we can compare machines operating under similar conditions. Now suppose we also want to match on vibration, load, pressure, age, and humidity. Two observations count as similar only if they are close on *all six* measurements. Even with the same number of training examples, it becomes harder to find examples resembling a new machine.

The number of input features is called the **dimension** of the input space. Temperature alone gives one dimension; temperature and vibration give two. As the dimension grows, the available observations spread across many more possible combinations of feature values. This difficulty is called the **curse of dimensionality**. The following simplified calculations show where it comes from.

### The number of regions grows rapidly

Imagine splitting the range of each feature into 10 equal intervals: 10 temperature ranges, 10 vibration ranges, and so on. With temperature alone, there are 10 intervals. With temperature *and* vibration, each temperature interval can occur with any of the 10 vibration intervals, giving $10\times10=100$ combinations. Every new feature multiplies the number of combinations by another 10.

If there are $D$ features, this grid has

$$
10^D
$$

regions. To see what this means for a fixed dataset, suppose we have $N=1{,}000$ observations and, for this illustration, observations are spread uniformly across the regions:

| Number of features $D$ | Number of regions $10^D$ | Expected observations per region $N/10^D$ |
|---:|---:|---:|
| 1 | 10 | 100 |
| 2 | 100 | 10 |
| 3 | 1,000 | 1 |
| 6 | 1,000,000 | 0.001 |

The last number is an *average*: it does not mean a region contains a fraction of an observation. It means most regions would be empty. Adding features did not add training examples, but it did make combinations of feature values much more numerous. A method that needs several examples in each small region will therefore need far more data, or a way to share information across regions.

### A neighborhood must become surprisingly wide

One response to empty regions is to look farther from the input we want to predict. But how far must we look to include enough training examples?

Scale each feature to the interval $[0,1]$ and consider a region that spans a fraction $\ell$ of the range of *each* feature. Under a uniform distribution, that region occupies a fraction $\ell^D$ of the whole input space. If we want it to contain an expected fraction $q$ of the observations, we must choose $\ell$ so that $\ell^D=q$, or

$$
\ell=q^{1/D}.
$$

Suppose we want roughly 1% of the observations, so $q=0.01$. Then

$$
\ell=
\begin{cases}
0.10, & D=2,\\
0.63\text{ approximately}, & D=10.
\end{cases}
$$

With two features, spanning 10% of each feature's range gives about 1% of the observations. With ten features, the region must span about 63% of **each** range to capture the same fraction. To collect enough examples, we may have to include observations that differ substantially from the new input. That weakens the idea of learning from genuinely nearby examples.

These percentages rely on uniform data and simple box-shaped regions. Real datasets need not follow either assumption. The calculation illustrates the central problem: “nearby” becomes harder to maintain when there are many dimensions and a fixed amount of data.

### Flexible representations also grow

The same issue appears in a different form when a model uses many coefficients. A polynomial with one input can include terms such as $1,x,x^2,x^3$. With two inputs, we can also include interactions such as $x_1x_2$ and $x_1^2x_2$. As we add inputs, the number of possible terms grows.

A term's **total degree** is the sum of its exponents: $x_1^2x_2$ has total degree $2+1=3$, while the intercept $1$ has degree 0. If we include *every* term whose total degree is at most $M$ across $D$ inputs, the number of coefficients is

$$
\binom{D+M}{M}
=\frac{(D+M)!}{D!\,M!}.
$$

<!-- The symbol $n!$, read “$n$ factorial,” means the product of all positive integers from 1 through $n$:

$$
n!=n(n-1)(n-2)\cdots2\cdot1,
\qquad 0!=1.
$$

For example, $5!=5\cdot4\cdot3\cdot2\cdot1=120$. The parentheses in $\binom{D+M}{M}$ denote a **binomial coefficient**; the factorial expression on the right shows how to calculate it. -->

For two inputs and degree at most 3, we can see the count directly. The terms are

$$
\underbrace{1}_{\text{degree 0}},\quad
\underbrace{x_1,x_2}_{\text{degree 1}},\quad
\underbrace{x_1^2,x_1x_2,x_2^2}_{\text{degree 2}},\quad
\underbrace{x_1^3,x_1^2x_2,x_1x_2^2,x_2^3}_{\text{degree 3}}.
$$

That is $1+2+3+4=10$ terms, matching the formula:

$$
\binom{2+3}{3}=\frac{5!}{2!\,3!}=10.
$$

With ten inputs and the same maximum degree, the count becomes

$$
\binom{10+3}{3}=\frac{13!}{10!\,3!}
=\frac{13\cdot12\cdot11}{3\cdot2\cdot1}
=286.
$$

Each term has its own coefficient to fit. If the dataset is small relative to the number of coefficients, many different fitted polynomials may explain the observed examples, and predictions can become sensitive to which examples happened to be collected.

The grid and polynomial examples describe two consequences of the same problem: the amount of data stays fixed while the possible input combinations or fitted terms multiply. High-dimensional learning can still work when the data have useful structure. Perhaps only a few of the measured features matter, or perhaps observations share a simpler pattern across many regions.

<!-- Feature selection, regularization, and dimensionality reduction can help a model use that structure. If any of those steps are learned from data, they must be fitted within the training boundaries discussed earlier. -->

## The bias–variance tradeoff

Suppose a transit agency wants to predict how long an 8:30 a.m. bus trip will take. For this thought experiment, imagine that the true average travel time for such trips is 30 minutes. Individual trips still vary: traffic lights, passengers, and weather make some faster and some slower.

Now imagine that we repeatedly collect a new training dataset and fit a prediction method to each dataset. One method smooths its predictions heavily. At 8:30, it predicts about 28 minutes almost every time. Its predictions are **stable**, but they consistently fall short of the true average. Another method reacts strongly to the trips in its particular training dataset. Across repeated datasets, its 8:30 predictions average 30 minutes, but individual fitted versions might predict substantially less or more.

The first method has a systematic error; the second is sensitive to the training sample. Neither property alone tells us which method predicts a *new* trip better. The new trip's own unpredictable variation matters too. This is the question addressed by the **bias–variance tradeoff**: how do systematic error, variation across fitted models, and irreducible variation in new outcomes combine to determine prediction error?

Lecture 9 treated the sample mean as a random quantity before data were observed. We now apply the same repeated-sampling idea to the prediction made by an entire fitted model. For regression with squared-error loss, the three sources of error can be separated mathematically.

### Separate the predictable target from noise

Let $X$ denote the input of a future observation and $T$ its target. We will study one particular input value $x$; in the bus example, $x$ could mean departure at 8:30 a.m. Even when $X=x$ is known, different trips can have different travel times. Their average target is called the **conditional mean** or true regression function:

$$
m(x)=\mathbb{E}[T\mid X=x].
$$

The notation $\mathbb{E}[T\mid X=x]$ means “average $T$ over new observations whose input is $x$.” For squared-error loss, predicting this average gives the smallest possible expected squared error among predictions that use only $x$.

Define the part of a new target that remains after subtracting this average as $\varepsilon=T-m(x)$. Rearranging this definition gives $T=m(x)+\varepsilon$. Because $m(x)$ is the conditional mean, the average residual at this input is zero:

$$
T=m(x)+\varepsilon,
\qquad
\mathbb{E}[\varepsilon\mid X=x]
=\mathbb{E}[T\mid X=x]-m(x)=0.
$$

We denote the residual's conditional variance by $\sigma_\varepsilon^2(x)$. Variance is the expected squared distance from the mean. Since the mean of $\varepsilon$ is zero here,

$$
\sigma_\varepsilon^2(x)
=\operatorname{Var}(\varepsilon\mid X=x)
=\mathbb{E}[\varepsilon^2\mid X=x].
$$

This is variation in a new trip's duration that remains even after its departure time is known. It may have a different value at another input.

Next let $\mathcal{D}$ denote a training dataset. Before we collect it, we do not know which observations it will contain. Fitting the same learning procedure to different possible datasets can therefore produce different predictions at $x$. We write $\widehat{f}_{\mathcal{D}}(x)$ for the prediction fitted from dataset $\mathcal{D}$. Its average over hypothetical training datasets is

$$
\overline{f}(x)=\mathbb{E}_{\mathcal{D}}
[\widehat{f}_{\mathcal{D}}(x)].
$$

The subscript $\mathcal{D}$ on $\mathbb{E}_{\mathcal{D}}$ tells us *what is being averaged over*. The value $\overline{f}(x)$ is an imagined average across repeated training studies, not an average of predictions for different points inside one dataset.

We assume a new target $T$ is independent of the training dataset once its input $X=x$ is specified. Thus knowing which training dataset we collected does not change the conditional mean or variance of the *new target's residual*. We also assume the squared quantities below have finite expectations so the averages are defined.

### Deriving the decomposition

Fix the input at $X=x$, such as an 8:30 departure. We will make two averages in sequence. First, hold one fitted model fixed and average over possible *new trips*. Then average the result over the different models that could have been fitted from different training datasets.

**First average: variation in a new target.** Once a particular training dataset $\mathcal{D}$ has been collected, its prediction $\widehat{f}_{\mathcal{D}}(x)$ is a fixed number. Substitute $T=m(x)+\varepsilon$ into the prediction error:

$$
T-\widehat{f}_{\mathcal{D}}(x)
=\bigl(m(x)-\widehat{f}_{\mathcal{D}}(x)\bigr)+\varepsilon.
$$

The expression in parentheses is the difference between the true average target and this particular model's prediction. The residual $\varepsilon$ is the additional variation in the new target. To compute *squared* error, use $(a+b)^2=a^2+2ab+b^2$ with $a=m(x)-\widehat{f}_{\mathcal{D}}(x)$ and $b=\varepsilon$:

$$
\begin{aligned}
(T-\widehat{f}_{\mathcal{D}}(x))^2
&=(m(x)-\widehat{f}_{\mathcal{D}}(x))^2\\
&\quad+2(m(x)-\widehat{f}_{\mathcal{D}}(x))\varepsilon
+\varepsilon^2.
\end{aligned}
$$

Now average this equation over possible *new targets*, keeping $x$ and $\mathcal{D}$ fixed. We write this average as $\mathbb{E}_{T\mid X=x,\mathcal{D}}$. Because $m(x)-\widehat{f}_{\mathcal{D}}(x)$ is fixed during this average, it can be taken outside the expectation:

$$
\begin{aligned}
\mathbb{E}_{T\mid X=x,\mathcal{D}}
[(T-\widehat{f}_{\mathcal{D}}(x))^2]
&=(m(x)-\widehat{f}_{\mathcal{D}}(x))^2\\
&\quad+2(m(x)-\widehat{f}_{\mathcal{D}}(x))
\mathbb{E}[\varepsilon\mid X=x,\mathcal{D}]\\
&\quad+\mathbb{E}[\varepsilon^2\mid X=x,\mathcal{D}].
\end{aligned}
$$

The new trip and the training dataset are independent once $X=x$ is specified. Therefore, the conditional averages of $\varepsilon$ are the ones we defined above: the first moment is 0, and the second moment is $\sigma_\varepsilon^2(x)$. Substituting those values removes the middle term:

$$
\mathbb{E}_{T\mid X=x,\mathcal{D}}
[(T-\widehat{f}_{\mathcal{D}}(x))^2]
=(m(x)-\widehat{f}_{\mathcal{D}}(x))^2
+\sigma_\varepsilon^2(x).
$$

At this point, we have **the fitted model's squared error relative to the true average**, plus **the variation in a new trip**. We still need to understand how the first part changes when we collect a different training dataset.

**Second average: variation across training datasets.** Recall that $\overline{f}(x)$ is the average fitted prediction across hypothetical datasets. Add and subtract it inside the first squared term:

$$
m(x)-\widehat{f}_{\mathcal{D}}(x)
=\underbrace{\bigl(m(x)-\overline{f}(x)\bigr)}_{\text{same for every training dataset}}
+\underbrace{\bigl(\overline{f}(x)-\widehat{f}_{\mathcal{D}}(x)\bigr)}_{\text{changes with the training dataset}}.
$$

The first difference is fixed at this $x$: it is how far the *average* fitted prediction is from the true average target. The second difference measures how far one fitted model is from the average fitted model. Square the sum using the same algebra as before:

$$
\begin{aligned}
(m(x)-\widehat{f}_{\mathcal{D}}(x))^2
&=(m(x)-\overline{f}(x))^2\\
&\quad+2(m(x)-\overline{f}(x))
(\overline{f}(x)-\widehat{f}_{\mathcal{D}}(x))\\
&\quad+(\overline{f}(x)-\widehat{f}_{\mathcal{D}}(x))^2.
\end{aligned}
$$

Average over training datasets. The only potentially puzzling term is the middle one. Its average is zero because $\overline{f}(x)$ was *defined* as the average of $\widehat{f}_{\mathcal{D}}(x)$:

$$
\begin{aligned}
\mathbb{E}_{\mathcal{D}}
[\overline{f}(x)-\widehat{f}_{\mathcal{D}}(x)]
&=\overline{f}(x)
-\mathbb{E}_{\mathcal{D}}[\widehat{f}_{\mathcal{D}}(x)]\\
&=\overline{f}(x)-\overline{f}(x)=0.
\end{aligned}
$$

Thus the averaged squared error of fitted predictions has two terms:

$$
\begin{aligned}
\mathbb{E}_{\mathcal{D}}
[(m(x)-\widehat{f}_{\mathcal{D}}(x))^2]
&=(m(x)-\overline{f}(x))^2\\
&\quad+\mathbb{E}_{\mathcal{D}}
[(\widehat{f}_{\mathcal{D}}(x)-\overline{f}(x))^2].
\end{aligned}
$$

The first is **squared bias**: the squared difference between the average prediction and the true conditional mean. The second is **prediction variance**: the average squared distance between a fitted prediction and the average fitted prediction. The sign inside a square does not matter, which is why the last term above uses $\widehat{f}_{\mathcal{D}}(x)-\overline{f}(x)$.

Finally, average the result of the *first* step over training datasets and substitute the result of the *second* step. The noise term $\sigma_\varepsilon^2(x)$ is unchanged across datasets, so it simply carries through:

$$
\begin{aligned}
\mathbb{E}_{\mathcal{D}}
\left[\mathbb{E}_{T\mid X=x,\mathcal{D}}
[(T-\widehat{f}_{\mathcal{D}}(x))^2]\right]
&=
\underbrace{(\overline{f}(x)-m(x))^2}_{\text{squared bias at }x}\\
&\quad+
\underbrace{\mathbb{E}_{\mathcal{D}}
[(\widehat{f}_{\mathcal{D}}(x)-\overline{f}(x))^2]}_{\text{prediction variance at }x}\\
&\quad+
\underbrace{\sigma_\varepsilon^2(x)}_{\text{noise variance at }x}.
\end{aligned}
$$

The left side means: repeatedly fit the model on a new training dataset, evaluate it on a new target whose input is $x$, square the prediction error, and average those errors. The identity holds at this fixed $x$. Averaging both sides over the population distribution of $X$ gives the corresponding expected population squared error, averaged over training datasets. It does not claim that the error of one fitted model on one test set can be split into three known numbers.

### What the terms mean

The **bias** term measures how far the average learned prediction lies from $m(x)$. For example, straight lines fitted to many samples may all miss the same bend in an underlying curved relationship. That systematic miss produces bias even if the lines are stable across samples.

The **variance** term measures how much the prediction changes across training datasets. A high-degree polynomial fitted to a small noisy sample can rise or fall sharply as individual training points change. It can have low bias across repeated samples while still producing unstable individual fits.

The **noise** term is the target variation left after the available input $x$ is known. No prediction rule using only $x$ can remove it. Better measurements or additional informative inputs might lower that conditional uncertainty, but fitting the same data more closely does not.

This is the tradeoff: greater flexibility often lets a model follow the underlying relationship more closely, reducing bias, but can make it react more strongly to the sampled data, increasing variance. Stronger regularization often moves in the opposite direction. We care about their *sum* in prediction error, along with irreducible noise, rather than minimizing either bias or variance alone.

<!-- These are common tendencies, not universal laws about every algorithm or dataset. The decomposition is an exact statement under the assumptions above; it does not promise a perfectly U-shaped test-error curve whenever we change a hyperparameter. -->

### A numerical comparison

Return to the 8:30 bus-trip example. Suppose that, across repeated training datasets, the stable method predicts 28 minutes on average and its predictions have variance 1 minute squared. The more responsive method predicts 30 minutes on average, but its predictions have variance 9 minutes squared. In this illustration, the true average is 30 minutes and the variation of a new trip around that average is 4 minutes squared. The decomposition gives:

| Procedure | Squared bias (min$^2$) | Prediction variance (min$^2$) | Trip noise (min$^2$) | Expected squared error (min$^2$) |
|---|---:|---:|---:|---:|
| Stable method | $(28-30)^2=4$ | 1 | 4 | $4+1+4=9$ |
| Responsive method | $(30-30)^2=0$ | 9 | 4 | $0+9+4=13$ |

The responsive method has no bias at 8:30: averaged across training datasets, it predicts the correct 30 minutes. Yet its prediction from any *one* fitted model varies enough that its expected squared error on a new trip is higher. The stable method's consistent two-minute shortfall is less costly here than the responsive method's large swings. The noise contribution is the same for both methods because the bus trip itself remains unpredictable to the same degree.

In a real study, we rarely know $m(x)$ or have access to the distribution of fitted models, so we cannot simply calculate the three components from one training sample. The decomposition explains why different procedures behave as they do. Validation and cross-validation provide the practical evidence used to choose among them.

## Putting the ideas together

These ideas fit together as a single procedure. First decide what a future prediction looks like and which mistakes matter, so the loss and the data split reflect the actual task. Set aside the test data. Then fit candidate pipelines on development data, using validation or cross-validation to decide among model classes and hyperparameters. After those choices are fixed, refit the selected pipeline and evaluate it on the untouched test set.

If performance is disappointing, the ideas above guide the diagnosis. High-dimensional inputs may require more observations or stronger structure. A rigid model may miss a real pattern, while a very flexible one may respond too strongly to sample noise. And even a promising test score is meaningful only if the data boundaries were respected throughout development.


## Practice problems and solutions

### 1. Selecting a predictor and estimating population error

A regression study fits three polynomial models:

| Degree | Training MSE | Validation MSE |
|---|---:|---:|
| 1 | 9.0 | 9.5 |
| 3 | 2.2 | 3.1 |
| 7 | 0.2 | 5.6 |

**(a)** Identify the parameters and one hyperparameter.

**(b)** Which degree would you select using these results? Explain why the training errors do not determine the choice.

**(c)** After selection and refitting on the development data, the model is evaluated on four independent test observations. Targets are $(2,4,6,8)$ and predictions are $(3,3,5,9)$. Compute the test MSE.

**(d)** Does that test MSE equal the population error? May we now compare ten more degrees on these same test observations and still call the winning score an independent test result?

::::{admonition} Solution 1
:class: dropdown

**(a)** The polynomial coefficients are parameters. The degree is a hyperparameter because it determines the model class before the coefficients are fitted.

**(b)** Select degree 3, whose validation MSE is the lowest at 3.1. Degree 7 fits the training observations more closely, but its larger validation error suggests poorer generalization under this split.

**(c)** The residuals are $(-1,1,1,-1)$, so

$$
\widehat{R}_{\mathrm{test}}
=\frac{(-1)^2+1^2+1^2+(-1)^2}{4}
=1.
$$

**(d)** This is a finite-sample estimate of population risk. Independence and representative sampling justify it as an estimate; they do not make it exactly equal to the unknown expectation.

Using these test observations to select among ten more degrees would make them part of model development. An independent final assessment would then require fresh held-out data or a properly designed outer evaluation procedure.
::::

### 2. Cross-validation and a leaking preprocessing step

Two regularization strengths produce the following four-fold validation MSEs. All folds have equal size.

| Strength | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---:|---:|---:|---:|
| $\lambda=0.01$ | 2 | 8 | 4 | 6 |
| $\lambda=1$ | 4 | 4 | 5 | 3 |

**(a)** Compute both cross-validation scores and select a strength.

**(b)** A student standardized all development inputs before constructing the folds. Explain the problem and describe the corrected procedure.

**(c)** In one corrected fold, the training feature values are $2,4,6$, and a held-out value is $8$. Compute the training mean, the training scale using denominator 3, and the transformed held-out value.

**(d)** After selecting $\lambda$, what is fitted again, and which data are used?

::::{admonition} Solution 2
:class: dropdown

**(a)** For equal fold sizes,

$$
\operatorname{CV}(0.01)=\frac{2+8+4+6}{4}=5,
\qquad
\operatorname{CV}(1)=\frac{4+4+5+3}{4}=4.
$$

Select $\lambda=1$ under the mean-validation-error criterion.

**(b)** The held-out fold helped determine the scaling used during its own evaluation. Within each fold, fit the scaler using only that fold's training subset, transform both subsets with that fitted scaler, and then fit and evaluate the predictor. A fresh pipeline is fitted for every fold and candidate.

**(c)** The training mean is

$$
\widehat{\mu}_{\mathrm{tr}}=\frac{2+4+6}{3}=4.
$$

The scale is

$$
\widehat{s}_{\mathrm{tr}}
=\sqrt{\frac{(2-4)^2+(4-4)^2+(6-4)^2}{3}}
=\sqrt{\frac{8}{3}}.
$$

Thus the held-out value becomes

$$
z=\frac{8-4}{\sqrt{8/3}}=\sqrt{6}\approx2.45.
$$

**(d)** Freeze $\lambda=1$ and refit both preprocessing and model parameters on all development data. Apply the resulting pipeline to the reserved test set. Do not estimate a new scaler from the test set.
::::

### 3. Dimensionality and the evaluation population

A team represents each observation with $D$ features in $[0,1]$. It divides each feature axis into five intervals and has 10,000 observations.

**(a)** Compute the number of grid cells for $D=2$ and $D=6$. Under a uniform distribution, compute the expected observations per cell.

**(b)** How many observations would be needed in six dimensions to match the two-dimensional average occupancy?

**(c)** The records come from 50 machines, with 200 measurements per machine. The intended task is prediction on new machines. Is randomly splitting individual rows an appropriate test design? Explain.

**(d)** Does the grid calculation prove that every model requires that many observations?

::::{admonition} Solution 3
:class: dropdown

**(a)** There are $5^D$ cells. Thus

$$
D=2:\quad 5^2=25,\qquad \frac{10{,}000}{25}=400
$$

observations per cell on average, while

$$
D=6:\quad 5^6=15{,}625,\qquad
\frac{10{,}000}{15{,}625}=0.64.
$$

Many six-dimensional cells will be empty even though the dataset size has not changed.

**(b)** To retain an average occupancy of 400,

$$
N=400(15{,}625)=6{,}250{,}000.
$$

This is $5^4=625$ times the original sample size.

**(c)** Keep machines together and hold out entire machines. Row-level splitting would generally put measurements from the same machine in both subsets, potentially rewarding recognition of familiar machines. It would assess a different task from transfer to unseen machines. Development cross-validation should use the same group-based principle.

**(d)** No. The calculation illustrates the burden of covering a full grid without exploiting structure. A model with suitable assumptions or informative low-dimensional structure can share information across regions and need substantially fewer observations.
::::

### 4. Bias, variance, and prediction error

At a fixed input $x_0$, the true conditional mean is $m(x_0)=5$, and the conditional noise variance is 2. For this simplified example, predictions across random training datasets take the following values, each with probability $1/3$:

- Procedure A: $2,3,4$;
- Procedure B: $1,5,9$.

**(a)** Compute the average prediction and bias of each procedure.

**(b)** Compute each procedure's prediction variance. Use the given probability distribution, not the unbiased sample-variance formula.

**(c)** Compute the expected squared prediction error for each procedure, including noise.

**(d)** Which procedure is preferable at $x_0$ under squared-error loss? Explain why zero bias alone does not settle the choice.

::::{admonition} Solution 4
:class: dropdown

**(a)** The average predictions are

$$
\overline{f}_A=\frac{2+3+4}{3}=3,
\qquad
\overline{f}_B=\frac{1+5+9}{3}=5.
$$

The biases are $3-5=-2$ and $5-5=0$. Their squared biases are therefore 4 and 0.

**(b)** Taking expectations over the specified three-outcome distributions,

$$
\operatorname{Var}(\widehat{f}_A)
=\frac{(2-3)^2+(3-3)^2+(4-3)^2}{3}
=\frac23,
$$

and

$$
\operatorname{Var}(\widehat{f}_B)
=\frac{(1-5)^2+(5-5)^2+(9-5)^2}{3}
=\frac{32}{3}.
$$

The denominator is 3 because we are computing the exact variance of a known discrete distribution over fitted predictions.

**(c)** Applying the decomposition,

$$
E_A=4+\frac23+2=\frac{20}{3}\approx6.67,
$$

$$
E_B=0+\frac{32}{3}+2=\frac{38}{3}\approx12.67.
$$

**(d)** Procedure A has lower expected squared error at this input. Procedure B is unbiased across training datasets, but its predictions fluctuate much more. Model selection should target predictive performance rather than unbiasedness alone. This comparison is at $x_0$; a population-wide comparison would also average over inputs.
::::


## Reading

Christopher M. Bishop, *Pattern Recognition and Machine Learning*, Chapter 1, pages 4–12, 32–38, 41–42, and 46–48, and Chapter 3, pages 147–151.

## References

- scikit-learn, [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html).
- scikit-learn, [Common pitfalls and recommended practicese](https://scikit-learn.org/stable/common_pitfalls.html).
