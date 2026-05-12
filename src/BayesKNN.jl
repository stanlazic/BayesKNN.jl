module BayesKNN

using Random
using Statistics
using Distributions
using NearestNeighbors
using AbstractMCMC


export BayesKNNFit,
       fit_bayesknn,
       predict_proba,
       predict_class,
       posterior_k_pmf,
       posterior_beta_summary,
       anomaly_score

# ============================================================
# Input validation
# ============================================================

function _validate_feature_matrix(X::AbstractMatrix, name::AbstractString)
    if !(eltype(X) <: Real)
        throw(ArgumentError("$name must contain real-valued predictors."))
    end
    if isempty(X) || size(X, 1) == 0 || size(X, 2) == 0
        throw(ArgumentError("$name must have at least one row and one column."))
    end
    if any(x -> !isfinite(x), X)
        throw(ArgumentError("$name must contain only finite values."))
    end
    return nothing
end

function _validate_training_inputs(Xtrain::AbstractMatrix, ytrain::AbstractVector)
    _validate_feature_matrix(Xtrain, "Xtrain")

    n_train = size(Xtrain, 1)
    if n_train < 2
        throw(ArgumentError("Xtrain must contain at least two observations."))
    end
    if length(ytrain) != n_train
        throw(DimensionMismatch("length(ytrain) must equal size(Xtrain, 1)."))
    end
    if any(ismissing, ytrain)
        throw(ArgumentError("ytrain must not contain missing labels."))
    end

    return nothing
end

function _validate_test_matrix(Xtest::AbstractMatrix, fit)
    _validate_feature_matrix(Xtest, "Xtest")

    if size(Xtest, 2) != size(fit.Xtrain, 2)
        throw(DimensionMismatch("size(Xtest, 2) must equal size(fit.Xtrain, 2)."))
    end

    return nothing
end

# ============================================================
# Probability utilities
# ============================================================

function _softmax_probabilities(logits::AbstractVector{<:Real})
    offset = maximum(logits)
    probs = Vector{Float64}(undef, length(logits))
    total = 0.0

    @inbounds for i in eachindex(logits)
        p = exp(logits[i] - offset)
        probs[i] = p
        total += p
    end

    @inbounds for i in eachindex(probs)
        probs[i] /= total
    end

    return probs
end


# ============================================================
# Types
# ============================================================

"""
    BayesKNNFit

Container for a fitted multiclass probabilistic nearest-neighbour model.

Fields:
- `chain`: named tuple `(beta::Vector{Float64}, k_idx::Vector{Int})` of
  posterior draws.
- `Xtrain`: training predictor matrix, observations in rows.
- `ytrain`: encoded training labels as integers `1, 2, ..., M`.
- `classes`: sorted original class labels; column `m` of any probability
  matrix corresponds to `classes[m]`.
- `k_values`: candidate neighbourhood sizes used during fitting.
- `train_order`: `n_train × (n_train - 1)` matrix of neighbour indices for
  each training point, sorted by increasing distance.
- `cumcounts_train`: `n_train × length(k_values) × M` array of cumulative
  class counts among neighbours for each training point.
- `tree`: KDTree built from the training data, reused for test-set queries.
- `diagnostics`: named tuple of convergence diagnostics. Always contains
  `ess_beta` and `ess_k` (effective sample size for `beta` and `k`). When
  `nchains > 1`, also contains `rhat_beta` and `rhat_k` (split-R-hat);
  values near 1.0 indicate good mixing, values above 1.1 suggest the chains
  have not converged. `fake_fit` and other test helpers may store `nothing`.
"""
struct BayesKNNFit{TX<:AbstractMatrix,
              TY<:AbstractVector{Int},
              TC<:AbstractVector,
              TK<:AbstractVector{Int},
              TO<:AbstractMatrix{Int},
              TCU<:Array{Int, 3},
              TT,
              TCH,
              TD}
    chain::TCH
    Xtrain::TX
    ytrain::TY
    classes::TC
    k_values::TK
    train_order::TO
    cumcounts_train::TCU
    tree::TT
    diagnostics::TD
end

# ============================================================
# Label handling
# ============================================================

"""
    _encode_labels(y)

Encode labels as integers `1, 2, ..., M`.

Returns a named tuple with:
- `y_encoded`: encoded integer labels
- `classes`: sorted unique original labels
"""
function _encode_labels(y::AbstractVector)
    classes = sort(collect(unique(y)))
    class_to_index = Dict(c => i for (i, c) in enumerate(classes))
    y_encoded = Int[class_to_index[yi] for yi in y]
    return (; y_encoded, classes)
end

"""
    _decode_labels(y_encoded, classes)

Map encoded labels `1, 2, ..., M` back to the original label values.
"""
function _decode_labels(y_encoded::AbstractVector{<:Integer}, classes::AbstractVector)
    return [classes[Int(i)] for i in y_encoded]
end

"""
    _nclasses(y)

Return the number of distinct encoded classes in `y`.
Assumes labels are encoded as `1, 2, ..., M`.
"""
function _nclasses(y::AbstractVector{<:Integer})
    M = maximum(y)
    if sort(unique(y)) != collect(1:M)
        throw(ArgumentError("Encoded labels must be exactly 1:M."))
    end
    return M
end

# ============================================================
# k validation
# ============================================================

"""
    _validate_k_values(k_values, n_train)

Validate candidate neighbourhood sizes.
"""
function _validate_k_values(k_values::AbstractVector{<:Integer}, n_train::Integer)
    kv = unique(sort(Int.(k_values)))

    if isempty(kv)
        throw(ArgumentError("k_values must not be empty."))
    end
    if any(k -> k < 1 || k >= n_train, kv)
        throw(ArgumentError("All k_values must satisfy 1 ≤ k < n_train."))
    end

    return kv
end

# ============================================================
# Nearest-neighbour ordering
# ============================================================

"""
    _neighbour_order_train(Xtrain)

For each training observation, return the indices of all other training points
sorted by increasing distance.

Returns an `n_train × (n_train - 1)` integer matrix `order`, where
`order[i, j]` is the index of the `j`th nearest neighbour of training point `i`
among the other training points.
"""
function _neighbour_order_train(Xtrain::AbstractMatrix)
    n_train, _ = size(Xtrain)

    data = permutedims(Matrix(Xtrain))  # p × n_train
    tree = KDTree(data)

    return _neighbour_order_train(tree, data, n_train)
end

function _neighbour_order_train(tree, data::AbstractMatrix, n_train::Integer)
    idxs, _ = knn(tree, data, n_train, true)

    order = Matrix{Int}(undef, n_train, n_train - 1)

    for i in 1:n_train
        nbrs = idxs[i]
        nbrs_no_self = Vector{Int}(undef, n_train - 1)

        t = 1
        @inbounds for j in nbrs
            if j != i
                nbrs_no_self[t] = j
                t += 1
            end
        end

        order[i, :] = nbrs_no_self
    end

    return order
end

"""
    _neighbour_order_test(Xtest::AbstractMatrix, Xtrain::AbstractMatrix)

For each test observation, return the indices of training points sorted by
increasing distance.

Arguments:
- `Xtest::AbstractMatrix`: real-valued test predictors, with observations in
  rows and predictors in columns.
- `Xtrain::AbstractMatrix`: real-valued training predictors, with observations
  in rows and predictors in columns. `size(Xtest, 2)` should equal
  `size(Xtrain, 2)`.

Returns an `n_test × n_train` integer matrix `order`, where `order[i, j]` is the
index of the `j`th nearest training neighbour of test point `i`.
"""
function _neighbour_order_test(Xtest::AbstractMatrix, Xtrain::AbstractMatrix)
    n_train = size(Xtrain, 1)

    data_train = permutedims(Matrix(Xtrain))  # p × n_train
    tree = KDTree(data_train)

    return _neighbour_order_test(Xtest, tree, n_train)
end

function _neighbour_order_test(Xtest::AbstractMatrix, tree, n_train::Integer)
    n_test = size(Xtest, 1)
    data_test = permutedims(Matrix(Xtest))    # p × n_test

    idxs, _ = knn(tree, data_test, n_train, true)

    order = Matrix{Int}(undef, n_test, n_train)

    for i in 1:n_test
        order[i, :] = idxs[i]
    end

    return order
end

# ============================================================
# Cumulative neighbour class counts
# ============================================================

"""
    _cumulative_class_counts(order, y, M, k_values)

Given:
- `order`: neighbour order matrix
- `y`: encoded labels in `1:M`
- `M`: number of classes
- `k_values`: candidate neighbourhood sizes

return a 3D integer array `cumcounts` with dimensions:

    size(cumcounts) == (n_rows, length(k_values), M)

where `cumcounts[i, j, m]` is the number of class-`m` labels among the first
`k_values[j]` neighbours for row `i`.
"""
function _cumulative_class_counts(
    order::AbstractMatrix{Int},
    y::AbstractVector{<:Integer},
    M::Integer,
    k_values::AbstractVector{<:Integer}
)
    n_rows, n_neighbours = size(order)
    nk = length(k_values)
    maxk = maximum(k_values)

    if maxk > n_neighbours
        throw(ArgumentError("All k_values must be no larger than the number of neighbours."))
    end

    cumcounts = Array{Int}(undef, n_rows, nk, M)

    for i in 1:n_rows
        counts = zeros(Int, M)
        k_idx = 1

        @inbounds for depth in 1:maxk
            cls = y[order[i, depth]]
            counts[cls] += 1

            if depth == k_values[k_idx]
                for m in 1:M
                    cumcounts[i, k_idx, m] = counts[m]
                end
                k_idx += 1
                k_idx > nk && break
            end
        end
    end

    return cumcounts
end

# ============================================================
# MCMC model and sampler
# ============================================================

struct BayesKNNModel{TB} <: AbstractMCMC.AbstractModel
    y::Vector{Int}
    cumcounts::Array{Int,3}
    k_values::Vector{Int}
    beta_prior::TB
end

struct BayesKNNSampler <: AbstractMCMC.AbstractSampler
    beta_step::Float64
end

struct BayesKNNTransition
    beta::Float64
    k_idx::Int
end

const _DEFAULT_BETA_PRIOR = truncated(Normal(0.0, 5.0), 0.0, Inf)

function _validate_beta_prior(beta_prior)
    try
        lower = minimum(beta_prior)
        upper = maximum(beta_prior)

        if lower < 0 || upper <= 0
            throw(ArgumentError("beta_prior must have support only on positive beta values."))
        end
    catch err
        err isa ArgumentError && rethrow()
        throw(ArgumentError("beta_prior must provide finite support bounds via minimum and maximum."))
    end

    try
        test_beta = max(float(minimum(beta_prior)), eps(Float64))
        logp = logpdf(beta_prior, test_beta)
        if isnan(logp)
            throw(ArgumentError("beta_prior must return a valid log density for positive beta values."))
        end
    catch err
        err isa ArgumentError && rethrow()
        throw(ArgumentError("beta_prior must support logpdf(beta_prior, beta) for positive beta values."))
    end

    return nothing
end

# Log-joint: log p(beta, k_idx, y | cumcounts, k_values)
function _log_joint(
    y::Vector{Int},
    cumcounts::Array{Int,3},
    k_values::Vector{Int},
    beta_prior,
    beta::Float64,
    k_idx::Int
)
    M = size(cumcounts, 3)
    k = k_values[k_idx]
    lp = logpdf(beta_prior, beta)
    logits = Vector{Float64}(undef, M)

    @inbounds for i in eachindex(y)
        for m in 1:M
            logits[m] = beta * (cumcounts[i, k_idx, m] / k)
        end
        offset = maximum(logits)
        sum_exp = 0.0
        for m in 1:M
            sum_exp += exp(logits[m] - offset)
        end
        lp += (logits[y[i]] - offset) - log(sum_exp)
    end

    return lp
end

# Initial step: draw from prior
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::BayesKNNModel,
    ::BayesKNNSampler;
    kwargs...
)
    nk = length(model.k_values)
    k_idx = rand(rng, 1:nk)
    beta = rand(rng, model.beta_prior)
    lp = _log_joint(model.y, model.cumcounts, model.k_values, model.beta_prior, beta, k_idx)
    state = (beta = beta, k_idx = k_idx, lp = lp)
    return BayesKNNTransition(beta, k_idx), state
end

# Subsequent step: Gibbs scan over k_idx then MH for beta
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::BayesKNNModel,
    sampler::BayesKNNSampler,
    state;
    kwargs...
)
    y, cumcounts, k_values = model.y, model.cumcounts, model.k_values
    beta_prior = model.beta_prior
    beta, k_idx = state.beta, state.k_idx
    nk = length(k_values)

    # Exact Gibbs step for k_idx: enumerate full conditional and sample directly
    log_cond = Vector{Float64}(undef, nk)
    for j in 1:nk
        log_cond[j] = _log_joint(y, cumcounts, k_values, beta_prior, beta, j)
    end
    k_idx = rand(rng, Categorical(_softmax_probabilities(log_cond)))
    lp = log_cond[k_idx]

    # MH step for beta using a random walk on the log scale
    log_beta = log(beta)
    log_beta_prop = log_beta + sampler.beta_step * randn(rng)
    beta_prop = exp(log_beta_prop)
    lp_prop = _log_joint(y, cumcounts, k_values, beta_prior, beta_prop, k_idx)

    # Accept/reject; log_beta_prop - log_beta is the log-Jacobian correction
    if log(rand(rng)) < lp_prop - lp + log_beta_prop - log_beta
        beta = beta_prop
        lp = lp_prop
    end

    state = (beta = beta, k_idx = k_idx, lp = lp)
    return BayesKNNTransition(beta, k_idx), state
end

# ============================================================
# R-hat convergence diagnostics
# ============================================================

# Gelman-Rubin R-hat for m chains of equal-or-truncated length
function _rhat(chains::AbstractVector{<:AbstractVector{<:Real}})
    m = length(chains)
    m < 2 && return NaN
    n = minimum(length.(chains))
    n < 2 && return NaN
    chain_means = [mean(c[1:n]) for c in chains]
    chain_vars  = [var(c[1:n]; corrected = true) for c in chains]
    grand_mean  = mean(chain_means)
    W = mean(chain_vars)
    B = n / (m - 1) * sum((μ - grand_mean)^2 for μ in chain_means)
    W == 0 && return NaN
    var_hat = (n - 1) / n * W + B / n
    return sqrt(var_hat / W)
end

# Split R-hat: split each chain in half, then apply R-hat to the 2m half-chains
function _split_rhat(chains::AbstractVector{<:AbstractVector{<:Real}})
    split = Vector{Vector{Float64}}()
    for chain in chains
        half = length(chain) ÷ 2
        push!(split, Float64.(chain[1:half]))
        push!(split, Float64.(chain[(half + 1):end]))
    end
    return _rhat(split)
end

# ESS via the initial positive sequence estimator (Geyer 1992).
# Sums per-chain ESS; chains with zero variance contribute nothing.
function _ess(chains::AbstractVector{<:AbstractVector{<:Real}})
    ess = 0.0
    for chain in chains
        n = length(chain)
        n < 4 && continue
        μ  = mean(chain)
        σ² = var(chain; corrected = false)
        σ² == 0 && continue
        rho_sum = 0.0
        for lag in 1:(n - 1)
            rho = 0.0
            @inbounds for t in 1:(n - lag)
                rho += (chain[t] - μ) * (chain[t + lag] - μ)
            end
            rho /= (n * σ²)
            rho <= 0 && break
            rho_sum += rho
        end
        ess += n / (1 + 2 * rho_sum)
    end
    return ess
end

# ============================================================
# Fitting
# ============================================================

"""
    fit_bayesknn(
        Xtrain::AbstractMatrix,
        ytrain::AbstractVector;
        k_values = collect(1:min(size(Xtrain, 1) - 1, 50)),
        beta_prior = truncated(Normal(0.0, 5.0), 0.0, Inf),
        beta_step::Float64 = 0.5,
        nsamples::Int = 5_000,
        discard_initial::Int = 1_000,
        nchains::Int = 1,
        rng = Random.default_rng(),
    )

Fit the multiclass probabilistic nearest-neighbour model.

Arguments:
- `Xtrain::AbstractMatrix`: real-valued training predictors, with observations
  in rows and predictors in columns. Values must be finite. Predictors are
  assumed to already be standardized or otherwise on an appropriate common
  scale.
- `ytrain::AbstractVector`: training labels, with length equal to
  `size(Xtrain, 1)`. Labels must not be missing and must be sortable because
  class labels are stored in sorted order.

Keyword arguments:
- `k_values::AbstractVector{<:Integer}`: candidate neighbourhood sizes.
  Defaults to `collect(1:min(size(Xtrain, 1) - 1, 50))`. Values must satisfy
  `1 <= k < size(Xtrain, 1)`.
- `beta_prior`: prior distribution for the nonnegative neighbour-strength
  parameter `beta`. Defaults to `truncated(Normal(0.0, 5.0), 0.0, Inf)`.
  The prior must support `rand`, `logpdf`, `minimum`, and `maximum`, and its
  support must not include negative values.
- `beta_step::Float64`: standard deviation of the log-scale random-walk
  proposal for `beta`. Defaults to `0.5`. Increase if the acceptance rate for
  `beta` is too high; decrease if it is too low.
- `nsamples::Int`: number of posterior samples returned per chain. Defaults to
  `5_000`. The chain runs for `discard_initial + nsamples` steps in total.
- `discard_initial::Int`: number of initial samples discarded as burn-in.
  Defaults to `1_000`. These steps are run but not stored.
- `nchains::Int`: number of MCMC chains. Defaults to `1`. When `nchains > 1`,
  chains are sampled in parallel using available threads; if Julia was started
  with one thread, a warning is issued and chains run serially.
- `rng`: random number generator. Defaults to `Random.default_rng()`.

Returns a `BayesKNNFit` containing the posterior draws, encoded training labels,
original class labels, candidate `k` values, neighbour information, and
training KDTree.
"""
function fit_bayesknn(
    Xtrain::AbstractMatrix,
    ytrain::AbstractVector;
    k_values = collect(1:min(size(Xtrain, 1) - 1, 50)),
    beta_prior = _DEFAULT_BETA_PRIOR,
    beta_step::Float64 = 0.5,
    nsamples::Int = 5_000,
    discard_initial::Int = 1_000,
    nchains::Int = 1,
    rng = Random.default_rng()
)
    _validate_training_inputs(Xtrain, ytrain)
    _validate_beta_prior(beta_prior)

    n_train = size(Xtrain, 1)
    encoded = _encode_labels(ytrain)
    y_encoded = encoded.y_encoded
    classes = encoded.classes
    M = length(classes)

    k_values_valid = _validate_k_values(k_values, n_train)
    Xtrain_mat = Matrix(Xtrain)

    data_train = permutedims(Xtrain_mat)
    tree = KDTree(data_train)
    train_order = _neighbour_order_train(tree, data_train, n_train)
    cumcounts_train = _cumulative_class_counts(train_order, y_encoded, M, k_values_valid)

    bayesknn_model = BayesKNNModel(y_encoded, cumcounts_train, k_values_valid, beta_prior)
    bayesknn_sampler = BayesKNNSampler(beta_step)

    if nchains > 1 && Threads.nthreads() < 2
        @warn "nchains=$nchains requested but Julia was started with one thread; " *
              "chains will run serially. Restart Julia with --threads=auto or " *
              "--threads=$nchains to sample in parallel."
    end

    transitions =
        nchains == 1 ?
        sample(rng, bayesknn_model, bayesknn_sampler, nsamples; progress = false, discard_initial) :
        sample(rng, bayesknn_model, bayesknn_sampler, MCMCThreads(), nsamples, nchains; progress = false, discard_initial)

    diagnostics, chain = if nchains == 1
        beta_vec  = [t.beta  for t in transitions]
        k_idx_vec = [t.k_idx for t in transitions]
        diag = (
            ess_beta = _ess([beta_vec]),
            ess_k    = _ess([Float64.(k_idx_vec)]),
        )
        diag, (; beta = beta_vec, k_idx = k_idx_vec)
    else
        beta_chains    = [[t.beta          for t in c] for c in transitions]
        k_idx_chains_f = [[Float64(t.k_idx) for t in c] for c in transitions]
        diag = (
            ess_beta  = _ess(beta_chains),
            ess_k     = _ess(k_idx_chains_f),
            rhat_beta = _split_rhat(beta_chains),
            rhat_k    = _split_rhat(k_idx_chains_f),
        )
        chain = (
            beta  = vcat(beta_chains...),
            k_idx = Int.(vcat(k_idx_chains_f...)),
        )
        diag, chain
    end

    return BayesKNNFit(
        chain,
        Xtrain_mat,
        y_encoded,
        classes,
        k_values_valid,
        train_order,
        cumcounts_train,
        tree,
        diagnostics,
    )
end

# ============================================================
# Posterior draw extraction
# ============================================================

"""
    _extract_bayesknn_draws(chain)

Return posterior draws of `beta` and `k_idx` from a fitted model's chain.

Returns a named tuple with:
- `beta`
- `k_idx`
"""
_extract_bayesknn_draws(chain) = chain

function Base.show(io::IO, fit::BayesKNNFit)
    draws = _extract_bayesknn_draws(fit.chain)
    print(
        io,
        "BayesKNNFit(",
        "n_train=", size(fit.Xtrain, 1),
        ", n_features=", size(fit.Xtrain, 2),
        ", n_classes=", length(fit.classes),
        ", n_k=", length(fit.k_values),
        ", n_draws=", length(draws.beta),
        ")",
    )
    if !isnothing(fit.diagnostics)
        d = fit.diagnostics
        print(io, "\n  ESS: beta=", round(d.ess_beta; digits = 0),
              ", k=", round(d.ess_k; digits = 0))
        if haskey(d, :rhat_beta)
            print(io, "\n  R-hat: beta=", round(d.rhat_beta; digits = 4),
                  ", k=", round(d.rhat_k; digits = 4))
        end
    end
end

# ============================================================
# Test-set preparation
# ============================================================

"""
    _prepare_test_cumcounts(fit::BayesKNNFit, Xtest::AbstractMatrix)

Compute neighbour ordering and cumulative class counts for test points relative
to the training set stored in `fit`.

Arguments:
- `fit::BayesKNNFit`: fitted model returned by [`fit_bayesknn`](@ref).
- `Xtest::AbstractMatrix`: real-valued test predictors, with observations in
  rows and predictors in columns. Values must be finite, and
  `size(Xtest, 2)` must equal `size(fit.Xtrain, 2)`.

Returns a named tuple with:
- `test_order`
- `cumcounts_test`
"""
function _prepare_test_cumcounts(fit::BayesKNNFit, Xtest::AbstractMatrix)
    _validate_test_matrix(Xtest, fit)

    Xtest_mat = Matrix(Xtest)
    test_order = _neighbour_order_test(Xtest_mat, fit.tree, size(fit.Xtrain, 1))
    M = length(fit.classes)
    cumcounts_test = _cumulative_class_counts(test_order, fit.ytrain, M, fit.k_values)

    return (
        test_order = test_order,
        cumcounts_test = cumcounts_test
    )
end

# ============================================================
# Prediction
# ============================================================

"""
    predict_proba(fit::BayesKNNFit, Xtest::AbstractMatrix)

Posterior predictive class probabilities for each test point.

Arguments:
- `fit::BayesKNNFit`: fitted model returned by [`fit_bayesknn`](@ref).
- `Xtest::AbstractMatrix`: real-valued test predictors, with observations in
  rows and predictors in columns. Values must be finite, and
  `size(Xtest, 2)` must equal `size(fit.Xtrain, 2)`.

Returns a named tuple with:
- `p_mean`: `n_test × M` matrix of posterior mean probabilities
- `p_lo`: `n_test × M` matrix of 2.5% quantiles
- `p_hi`: `n_test × M` matrix of 97.5% quantiles
- `p_draws`: `n_draws × n_test × M` array of posterior predictive probabilities
- `classes`: original class labels corresponding to columns of the probability matrices
"""
function predict_proba(fit::BayesKNNFit, Xtest::AbstractMatrix)
    prep = _prepare_test_cumcounts(fit, Xtest)
    cumcounts_test = prep.cumcounts_test

    draws = _extract_bayesknn_draws(fit.chain)
    beta_draws = draws.beta
    k_idx_draws = draws.k_idx

    ns = length(beta_draws)
    n_test = size(Xtest, 1)
    M = length(fit.classes)

    p_draws = Array{Float64}(undef, ns, n_test, M)

    for s in 1:ns
        beta = beta_draws[s]
        k_idx = k_idx_draws[s]
        k = fit.k_values[k_idx]
        logits = Vector{Float64}(undef, M)

        for i in 1:n_test
            @inbounds for m in 1:M
                logits[m] = beta * (cumcounts_test[i, k_idx, m] / k)
            end

            probs = _softmax_probabilities(logits)
            @inbounds for m in 1:M
                p_draws[s, i, m] = probs[m]
            end
        end
    end

    p_mean = Matrix{Float64}(undef, n_test, M)
    p_lo   = Matrix{Float64}(undef, n_test, M)
    p_hi   = Matrix{Float64}(undef, n_test, M)

    for i in 1:n_test
        for m in 1:M
            v = @view p_draws[:, i, m]
            p_mean[i, m] = mean(v)
            p_lo[i, m] = quantile(v, 0.025)
            p_hi[i, m] = quantile(v, 0.975)
        end
    end

    return (
        p_mean = p_mean,
        p_lo = p_lo,
        p_hi = p_hi,
        p_draws = p_draws,
        classes = fit.classes
    )
end

"""
    predict_class(fit::BayesKNNFit, Xtest::AbstractMatrix)

Posterior class predictions for each test point.

Arguments:
- `fit::BayesKNNFit`: fitted model returned by [`fit_bayesknn`](@ref).
- `Xtest::AbstractMatrix`: real-valued test predictors, with observations in
  rows and predictors in columns. Values must be finite, and
  `size(Xtest, 2)` must equal `size(fit.Xtrain, 2)`.

Returns a named tuple with:
- `yhat_encoded`: predicted encoded labels in `1:M`
- `yhat`: predicted original labels
- `p_mean`: posterior mean class probabilities
- `p_lo`: lower credible limits for class probabilities
- `p_hi`: upper credible limits for class probabilities
- `classes`: original class labels corresponding to columns of the probability matrices
"""
function predict_class(fit::BayesKNNFit, Xtest::AbstractMatrix)
    pred = predict_proba(fit, Xtest)
    n_test, M = size(pred.p_mean)

    yhat_encoded = Vector{Int}(undef, n_test)
    for i in 1:n_test
        yhat_encoded[i] = argmax(@view pred.p_mean[i, :])
    end

    yhat = _decode_labels(yhat_encoded, fit.classes)

    return (
        yhat_encoded = yhat_encoded,
        yhat = yhat,
        p_mean = pred.p_mean,
        p_lo = pred.p_lo,
        p_hi = pred.p_hi,
        classes = pred.classes
    )
end

# ============================================================
# Posterior summaries
# ============================================================

"""
    posterior_k_pmf(fit)

Posterior PMF over candidate neighbourhood sizes.

Returns a named tuple with:
- `k`: candidate neighbourhood sizes
- `posterior_prob`: posterior probabilities corresponding to `k`
"""
function posterior_k_pmf(fit::BayesKNNFit)
    draws = _extract_bayesknn_draws(fit.chain)
    k_idx = draws.k_idx
    nk = length(fit.k_values)

    probs = zeros(Float64, nk)
    for j in 1:nk
        probs[j] = mean(k_idx .== j)
    end

    return (k = fit.k_values, posterior_prob = probs)
end

"""
    posterior_beta_summary(fit)

Posterior summary for the neighbour-strength parameter `beta`.
"""
function posterior_beta_summary(fit::BayesKNNFit)
    draws = _extract_bayesknn_draws(fit.chain)
    beta = draws.beta

    return (
        mean = mean(beta),
        median = median(beta),
        q025 = quantile(beta, 0.025),
        q975 = quantile(beta, 0.975)
    )
end

# ============================================================
# Distance-based anomaly scoring
# ============================================================

"""
    anomaly_score(
        fit::BayesKNNFit,
        Xtest::AbstractMatrix;
        threshold = nothing,
    )

Distance-based anomaly scores for test points using the posterior over `k`.

For each posterior draw of `k`, the anomaly score is the distance from the test
point to its `k`th nearest training neighbour.

Arguments:
- `fit::BayesKNNFit`: fitted model returned by [`fit_bayesknn`](@ref).
- `Xtest::AbstractMatrix`: real-valued test predictors, with observations in
  rows and predictors in columns. Values must be finite, and
  `size(Xtest, 2)` must equal `size(fit.Xtrain, 2)`.
- `threshold`: optional distance threshold; if provided, the function also
  returns the posterior probability that the `k`th neighbour distance exceeds
  this threshold

Returns a named tuple with:
- `mean_distance`
- `median_distance`
- `q025_distance`
- `q975_distance`
- `p_gt_threshold`
- `distance_draws`
"""
function anomaly_score(
    fit::BayesKNNFit,
    Xtest::AbstractMatrix;
    threshold = nothing
)
    _validate_test_matrix(Xtest, fit)

    Xtest_mat = Matrix(Xtest)
    maxk = maximum(fit.k_values)

    data_test = permutedims(Xtest_mat)     # p × n_test

    _, dists = knn(fit.tree, data_test, maxk, true)

    n_test = size(Xtest_mat, 1)
    kth_dist = Matrix{Float64}(undef, n_test, maxk)

    for i in 1:n_test
        kth_dist[i, :] = dists[i]
    end

    draws = _extract_bayesknn_draws(fit.chain)
    k_draws = fit.k_values[draws.k_idx]

    ns = length(k_draws)
    distance_draws = Matrix{Float64}(undef, ns, n_test)

    for s in 1:ns
        k = k_draws[s]
        @inbounds for i in 1:n_test
            distance_draws[s, i] = kth_dist[i, k]
        end
    end

    mean_distance = Vector{Float64}(undef, n_test)
    median_distance = Vector{Float64}(undef, n_test)
    q025_distance = Vector{Float64}(undef, n_test)
    q975_distance = Vector{Float64}(undef, n_test)

    for i in 1:n_test
        v = @view distance_draws[:, i]
        mean_distance[i] = mean(v)
        median_distance[i] = median(v)
        q025_distance[i] = quantile(v, 0.025)
        q975_distance[i] = quantile(v, 0.975)
    end

    p_gt_threshold =
        isnothing(threshold) ? nothing : vec(mean(distance_draws .> threshold; dims = 1))

    return (
        mean_distance = mean_distance,
        median_distance = median_distance,
        q025_distance = q025_distance,
        q975_distance = q975_distance,
        p_gt_threshold = p_gt_threshold,
        distance_draws = distance_draws
    )
end

end
