# Project - Airfoil Noise: Evidence, Prediction, and Decisions

An engineering team is studying noise from an airfoil. It has two related tasks: predict the measured sound-pressure level for a set of test conditions, and flag conditions likely to produce **high noise**. Before using a new configuration, the team also wants to assess a small, separate quality-assurance audit.

For this project, **125 dB is an illustrative design-screening limit chosen for the exercise**. It is not presented as a legal, medical, or universal engineering safety standard. Define a high-noise result as a measured sound-pressure level **at or above 125 dB**; a result below 125 dB passes the screen.

Questions 2 and 3 use the same [UCI Airfoil Self-Noise dataset](https://archive.ics.uci.edu/dataset/291/airfoil%2Bself-noise), which contains 1,503 wind-tunnel observations, five inputs, and a measured sound-pressure level in decibels. Question 1 uses a separate, hypothetical audit under one fixed test condition. Do **not** treat all rows of the UCI dataset as repeated identical-condition trials for Question 1: its operating conditions vary.

## Purdue Honor Pledge

> “As a Boilermaker pursuing academic excellence, I pledge to be honest and true in all that I do. Accountable together—We are Purdue.”

**When submitting this project, we assume that you commit to this honor pledge and submit your own work considering the instructions below.**

## Instructions

- Use the accompanying [starter notebook](project1-starter.ipynb). Its setup cells download the public dataset and make one shared train/validation/test split. Complete the blank experiment and interpretation cells.
- **Use SciPy and scikit-learn functions.** You are not asked to implement a statistical test, a regression solver, gradient descent, or logistic regression from scratch, but **you are expected to read up on the documentation for the code and use the necessary packages, functions, and classes**, just like in the real world. Show the function calls, their relevant outputs, and your interpretation.
- Use the **same five inputs and the same data split** for Questions 2 and 3. Keep the starter notebook's seed, split code, model settings, and final reporting threshold unchanged so numerical results are comparable across submissions. The measured sound-pressure level is the target; never include it among the model inputs. It may be used to create the binary high-noise target.
- Fit preprocessing steps, such as feature scaling, on the training set only. Use validation results to interpret each model before reporting its test results; do not use the test set to tune a model or threshold.
- Label numerical results and figures with units where appropriate. A small table and a short explanation are usually more useful than a large block of unannotated output.
- Submit an executed notebook with your code, requested results, figures, and written answers. **Use the exact labeled outputs and table columns specified in the starter notebook** so results can be graded consistently. Submission location and deadline will be announced separately.

## Question 1: Does a fixed airfoil configuration reliably pass the noise screen?

A separate quality-assurance audit tests **one fixed airfoil configuration at one fixed operating condition**. In this teaching scenario, 60 independent runs are performed using the same measurement procedure. In 57 runs the measured sound-pressure level is below 125 dB; in three it is at or above 125 dB. These summary counts are **hypothetical project data**, not a subset of the UCI dataset.

Let $\theta$ be the probability that a new run under these same conditions passes the screen. Before observing these 60 runs, the team set a minimum target of $\theta_c=0.90$ and assigned the prior $\theta\sim\operatorname{Beta}(9,1)$. Assume the runs are conditionally independent given $\theta$.

- **(a) Frequentist test.** Test $H_0:\theta\leq0.90$ against $H_1:\theta>0.90$ at significance level $\alpha=0.05$ using SciPy's exact one-sided binomial test. Report the observed pass fraction and the p-value. State the decision and interpret the p-value.
- **(b) Bayesian update.** Use SciPy's Beta-distribution functions to find the posterior distribution of $\theta$ and calculate $P(\theta>0.90\mid\mathcal D)$. Report the prior probability $P(\theta>0.90)$ as well. You may use the Beta–Bernoulli update from Lecture 14; no numerical integration code is required.
- **(c) Bayes factor and sensitivity.** Calculate prior odds, posterior odds, and $\mathrm{BF}_{10}$ for $H_1$ relative to $H_0$. Then repeat the posterior probability calculation using the alternative prior $\operatorname{Beta}(1,1)$, **without reusing the first posterior as a prior**. Explain briefly what changed and what did not.
- **(d) Operational interpretation.** In four to six sentences, compare what the p-value, posterior probability, and Bayes factor say. Would the available evidence make you comfortable claiming that this configuration exceeds the 90% target? Identify one assumption about the audit that you would want to verify in practice.

## Question 2: Predict the sound-pressure level

The UCI data record frequency, angle of attack, chord length, free-stream velocity, and suction-side displacement thickness. The continuous target is **scaled sound-pressure level in dB**. Here the team wants a numerical prediction, so this is a regression problem.

- **(a) Inspect the data.** Load the dataset using the starter notebook. Report its number of rows, the five inputs, the target's units, and whether any values are missing. **After the shared split**, make a histogram of the *training-set* measured sound-pressure levels.
- **(b) Direct least-squares fit.** Fit a linear regression model using scikit-learn's `LinearRegression`. This is the direct numerical least-squares solution corresponding to the least-squares estimator from Lecture 17, and you do **not** need to form or invert $\boldsymbol\Phi^{\mathsf T}\boldsymbol\Phi$ yourself. Use a `StandardScaler` pipeline so preprocessing is learned from the training data. Report validation mean absolute error (MAE) and root mean squared error (RMSE), both in dB.
- **(c) Gradient-based fit.** Fit `SGDRegressor` on the **same training rows and inputs**, using the fixed `SGD_SETTINGS` in the starter notebook (squared-error loss, no regularization penalty, and a fixed random seed) so it targets the same least-squares objective. This function uses **stochastic** gradient descent, and you do not need to write an optimizer from scratch. Report the settings, epochs used, and validation MAE and RMSE. If a convergence warning appears, report it, but keep the settings unchanged for the graded comparison.
- **(d) Compare the fits.** Plot measured versus predicted dB on the validation set for both models. Compare their errors and predictions. Explain why the two fitted weight vectors need not be numerically identical after a finite number of stochastic updates, even though the unregularized objectives agree. Choose which regression fit you would report **using validation results**, then report test MAE and RMSE for both regression models.

## Question 3: Predict whether noise will be high

Use the **same UCI observations, five inputs, and train/validation/test rows** as in Question 2. Define a binary target

$$
Z=
\begin{cases}
1,&\text{measured sound-pressure level}\geq125\text{ dB (high noise)},\\
0,&\text{measured sound-pressure level}<125\text{ dB (passes screen)}.
\end{cases}
$$

The measured dB value is used to form this label, but it must **not** be supplied as a predictor.

- **(a) Check the new target.** Report the number or fraction of high-noise examples in the training and validation splits; save classification test-set reporting for part (d). Explain why a continuous-target regression problem has now become a binary classification problem.
- **(b) Fit and inspect probabilities.** Fit a `StandardScaler`-`LogisticRegression` pipeline using only the training rows and the fixed `LOGISTIC_SETTINGS` in the starter notebook. Use `predict_proba` to obtain the estimated probability of high noise for validation rows. Show a few examples with their measured dB, true class, and predicted probability. Scikit-learn applies regularization by default.
- **(c) Make a decision on validation data.** At a probability threshold of $0.5$, report the logistic model's validation confusion matrix and identify the **false negatives**: cases that actually have high noise but are predicted to pass. Compare logistic thresholds $0.3$, $0.5$, and $0.7$ on the **validation set**. Use the fixed $0.3$ threshold for the graded final test, but say whether you would recommend a different threshold in practice. Explain the tradeoff between missed high-noise cases and unnecessary warnings.
- **(d) Final logistic-model test.** Report the high-noise count in the held-out test set. Apply the fixed $0.3$ probability threshold to the logistic model's test-set probabilities, then report TN, FP, FN, TP (the entries of its confusion matrix) and accuracy. Interpret the false negatives and false positives in the context of an initial noise-screening tool. Do not use either regression model as a classifier in this question.

## Data and tool references

- [Brooks, Pope, and Marcolini, *Airfoil Self-Noise*, UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/291/airfoil%2Bself-noise).
- [SciPy exact binomial test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html) and [Beta distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html).
- [scikit-learn linear regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html), [stochastic-gradient regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html), and [logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
