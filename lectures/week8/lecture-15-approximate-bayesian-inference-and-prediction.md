# Lecture 15 — Approximate Bayesian Inference and Prediction

This planning outline defines the proposed scope and learning outcomes for Lecture 15, building on the exact Bayesian updates and uncertainty summaries in Lecture 14.

**Guiding question:** How can we carry out Bayesian inference when the posterior cannot be calculated analytically, and use the resulting distribution to predict a new observation?

## Key learning outcomes

After completing Lecture 15, students should be able to:

1. Distinguish specifying a prior, learning a posterior from data, and numerically approximating a posterior.
2. Explain why a nonconjugate model can make the evidence integral and normalized posterior difficult to calculate analytically.
3. Construct a one-dimensional grid approximation from likelihood–prior products, accounting for grid spacing when normalizing and calculating probabilities.
4. Calculate approximate credible intervals and threshold-hypothesis probabilities, and assess their stability as the grid is refined.
5. Explain how posterior draws approximate expectations and probabilities; distinguish numerical approximation error from uncertainty represented by the posterior.
6. Construct a posterior predictive probability by averaging the observation model over the posterior, and distinguish it from a posterior probability about the parameter.

## Suggested teaching sequence

- **Benchmark:** Approximate the known Beta(5,3) posterior on a grid and compare numerical results with Lecture 14's exact results.
- **Nonconjugate example:** Retain the same Bernoulli likelihood and specify a continuous prior with an unfamiliar shape, such as a normalized truncated Gaussian density on $(0,1)$. Multiply, normalize numerically, and calculate posterior areas.
- **Accuracy:** Compare coarse and fine grids; explain probability weights versus density heights. A grid covering $(0,1)$ needs a suitable numerical quadrature rule and sufficient resolution.
- **Simulation:** Draw parameter values from the normalized grid approximation and estimate an event probability using the proportion of draws satisfying it. Explain Monte Carlo variability.
- **Prediction:** Introduce $P(X_{\mathrm{new}}=1\mid\mathcal D)=\int_0^1\theta p(\theta\mid\mathcal D)\,d\theta$ and its weighted-grid and sample-average approximations. This is a probability for a future observation, not the posterior probability of a parameter hypothesis.
- **Optional outlook:** Why grids become expensive in higher dimensions and how MCMC can help. A full MCMC algorithm and diagnostic treatment should be an extension, not required alongside every other outcome in one lecture.

## Terminology and scope

The quantity approximated here is the **posterior**, after the prior and likelihood have been specified. Estimating prior hyperparameters from data is a different procedure, often called empirical Bayes, and requires attention to which data are used and how hyperparameter uncertainty is handled. It is not required in these two lectures. MAP and regularization remain with the later Bayesian regression material. Ordinary bootstrap resampling remains with frequentist sampling uncertainty.

## Reading alignment for Lecture 15

- **Bayes Rules!, Section 6.1:** grid approximation and its limitations.
- **Bayes Rules!, Sections 8.3–8.4:** posterior prediction and simulation-based posterior analysis; choose the relevant portions and translate R examples to the course's computational language.
- **Bishop, Section 2.1.1, Equations (2.19)–(2.20):** Bernoulli posterior prediction, deferred from Lecture 14.
- **Optional:** Bayes Rules!, Sections 6.2–6.3, for MCMC and diagnostics if the course later expands the computational treatment.
