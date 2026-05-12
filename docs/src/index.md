```@meta
CurrentModule = BayesKNN
```

# BayesKNN.jl

BayesKNN.jl implements the probabilistic nearest-neighbour classifier of Holmes and
Adams (2002). It samples the posterior distribution of the neighbourhood size
`k` and neighbour-strength parameter `beta`, then averages over posterior draws
to produce class probabilities for new observations.

## Quick Start

Rows are observations and columns are predictors. Put predictors on a common
scale before fitting.

```julia
using BayesKNN, Random

X = [0.0; 0.2; 3.8; 4.0;;]
y = ["a", "a", "b", "b"]

fit = fit_bayesknn(X, y; k_values = [1, 2], nsamples = 1_000, rng = MersenneTwister(1))
pred = predict_proba(fit, [0.1; 3.9;;])
```

## Input Convention

`Xtrain` and `Xtest` must be real-valued matrices with observations in rows and
predictors in columns. Labels may be strings, numbers, or other sortable values.
Missing labels and non-finite predictor values are rejected.

## Beta Prior

The neighbour-strength parameter `beta` is constrained to be nonnegative. By
default, `fit_bayesknn` uses `truncated(Normal(0.0, 5.0), 0.0, Inf)`. You can pass a
different prior with nonnegative support:

```julia
using Distributions

fit = fit_bayesknn(X, y; beta_prior = Gamma(2.0, 1.0))
```

## API Reference

```@docs
BayesKNNFit
fit_bayesknn
predict_proba
predict_class
posterior_k_pmf
posterior_beta_summary
anomaly_score
```

## Citation

Holmes, C. C. and Adams, N. M. (2002). A probabilistic nearest neighbour method
for statistical pattern recognition. *Journal of the Royal Statistical Society:
Series B*, 64(2), 295-306.
