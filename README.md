# BayesKNN

[![Build Status](https://github.com/stanlazic/BayesKNN.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/stanlazic/BayesKNN.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/stanlazic/BayesKNN.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/stanlazic/BayesKNN.jl)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://stanlazic.github.io/BayesKNN.jl/dev/)

BayesKNN.jl implements the probabilistic nearest-neighbour classifier of Holmes and
Adams (2002), and also extends their approach to multiclass outcomes. The
model treats the neighbourhood size `k` and neighbour-strength parameter `beta` as
unknown, samples their posterior distribution, and returns posterior predictive
class probabilities for new observations.

## Why use BayesKNN?

Standard KNN classifiers require choosing a fixed `k`, typically by
cross-validation, and then treat that choice as known. BayesKNN has three advantages
over this approach:

1. **No need to select a single `k`.** You supply a set of candidate values and
   the model learns from the data which are plausible, without requiring a
   separate tuning step.

2. **Uncertainty over `k` is propagated into predictions.** Because `k` is
   sampled from its posterior rather than fixed, predictions reflect the
   uncertainty about the right neighbourhood size. The posterior PMF over `k`
   (available via `posterior_k_pmf`) shows which values the data support.

3. **Predicted probabilities have continuous support.** Standard KNN produces
   class probabilities that are multiples of `1/k` — for example, with `k = 5`
   the only possible values are 0, 0.2, 0.4, 0.6, 0.8, and 1. By averaging
   over posterior draws of `k` and `beta`, BayesKNN produces probabilities that vary
   continuously in `(0, 1)`.

## Installation

```julia
using Pkg
Pkg.add("BayesKNN")
```

Until the package is registered, install it from the repository:

```julia
using Pkg
Pkg.add(url = "https://github.com/stanlazic/BayesKNN.jl")
```

## Quick Start

Rows are observations and columns are predictors. Predictors should be
standardized or otherwise put on a common scale before fitting.

```julia
using BayesKNN

X = [0.0; 0.2; 3.8; 4.0;;]
y = ["a", "a", "b", "b"]

fit = fit_bayesknn(
    X,
    y;
    k_values = [1, 2],
    nsamples = 1_000
)

pred = predict_proba(fit, [0.1; 3.9;;])
cls = predict_class(fit, [0.1; 3.9;;])
```

`pred.p_mean` contains posterior mean class probabilities. `pred.classes`
contains the original labels corresponding to the probability columns.

## API

- `fit_bayesknn(Xtrain, ytrain; k_values, beta_prior, beta_step, nsamples, discard_initial, nchains, rng)`
  fits the model.
- `predict_proba(fit, Xtest)` returns posterior predictive class probabilities.
- `predict_class(fit, Xtest)` returns class predictions from posterior mean
  probabilities.
- `posterior_k_pmf(fit)` summarizes the posterior distribution of `k`.
- `posterior_beta_summary(fit)` summarizes the posterior distribution of
  `beta`.
- `anomaly_score(fit, Xtest; threshold)` returns distance-based anomaly scores
  (see below).

By default, `beta` has the prior `truncated(Normal(0.0, 5.0), 0.0, Inf)`.
Pass `beta_prior` to use another prior distribution with nonnegative support,
for example `fit_bayesknn(X, y; beta_prior = Gamma(2.0, 1.0))` after
`using Distributions`.

## MCMC diagnostics

Every fitted model stores a `diagnostics` named tuple. For a single chain it
contains:

- `ess_beta`, `ess_k`: effective sample size for `beta` and `k`, computed via
  the initial positive sequence estimator. Values much below the requested
  `nsamples` indicate high autocorrelation; consider increasing `nsamples` or
  adjusting `beta_step`.

When `nchains > 1` the tuple additionally contains:

- `rhat_beta`, `rhat_k`: split-R-hat (Vehtari et al. 2021). Values near 1.0
  indicate the chains have mixed; values above 1.1 suggest they have not
  converged.

```julia
fit = fit_bayesknn(X, y; nchains = 4)
fit.diagnostics   # (ess_beta, ess_k, rhat_beta, rhat_k)
```

## Anomaly detection

`anomaly_score(fit, Xtest)` can be used to assess whether a test point is
unusual relative to the training data. For each posterior draw of `k`, the
score is the Euclidean distance from the test point to its `k`th nearest
training neighbour. Averaging over draws gives a posterior mean distance, along
with a credible interval. Because `k` is itself uncertain, the score reflects
both the distance to the neighbourhood boundary and the uncertainty about where
that boundary lies.

An optional `threshold` argument returns the posterior probability that the
`k`th neighbour distance exceeds the threshold, which can serve as a
calibrated flag for unusual observations. A Threshold can be determined from
the training data; for example, the 95th or 99th percentile of the training
data `mean_distance` scores.

## Notes

The method uses Euclidean nearest neighbours through NearestNeighbors.jl. The
implementation stores cumulative neighbour class counts only for the requested
candidate neighbourhood sizes, with size
`n_train * length(k_values) * n_classes`.

### Scaling

The dominant cost is MCMC sampling. Each iteration evaluates the log-posterior
once per candidate `k` value, and each evaluation iterates over all training
observations, so runtime scales roughly as
`nsamples × length(k_values) × n_train`. The method works well with up to a
few thousand observations.

If fitting is slow, the most effective remedy is to reduce the upper limit of
`k_values`. The default caps at 50, but for large datasets something like
`k_values = 1:15` can substantially cut runtime.

## Citation

Holmes, C. C. and Adams, N. M. (2002). A probabilistic nearest neighbour method
for statistical pattern recognition. *Journal of the Royal Statistical Society:
Series B*, 64(2), 295-306.
