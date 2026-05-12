using BayesKNN
using Distributions
using Random
using Test

function fake_fit(Xtrain, ytrain; beta = [0.0, 2.0], k_idx = [1, 1], k_values = [1])
    encoded = BayesKNN._encode_labels(ytrain)
    y_encoded = encoded.y_encoded
    classes = encoded.classes
    train_order = BayesKNN._neighbour_order_train(Xtrain)
    cumcounts_train = BayesKNN._cumulative_class_counts(train_order, y_encoded, length(classes), k_values)
    chain = (beta = beta, k_idx = k_idx)
    tree = BayesKNN.KDTree(permutedims(Matrix(Xtrain)))

    return BayesKNNFit(
        chain,
        Matrix(Xtrain),
        y_encoded,
        classes,
        k_values,
        train_order,
        cumcounts_train,
        tree,
        nothing,
    )
end

@testset "Label handling" begin
    encoded = BayesKNN._encode_labels(["b", "a", "b"])

    @test encoded.classes == ["a", "b"]
    @test encoded.y_encoded == [2, 1, 2]
    @test BayesKNN._decode_labels(encoded.y_encoded, encoded.classes) == ["b", "a", "b"]
    @test BayesKNN._nclasses([1, 2, 1]) == 2
    @test_throws ArgumentError BayesKNN._nclasses([1, 3])
end

@testset "Neighbour ordering and counts" begin
    X = reshape([0.0, 1.0, 3.0], 3, 1)
    order = BayesKNN._neighbour_order_train(X)

    @test order == [2 3; 1 3; 2 1]

    cumcounts = BayesKNN._cumulative_class_counts(order, [1, 2, 2], 2, [1, 2])

    @test size(cumcounts) == (3, 2, 2)
    @test cumcounts[1, 1, :] == [0, 1]
    @test cumcounts[1, 2, :] == [0, 2]
    @test cumcounts[2, 1, :] == [1, 0]
    @test cumcounts[2, 2, :] == [1, 1]

    sparse_cumcounts = BayesKNN._cumulative_class_counts(order, [1, 2, 2], 2, [2])
    @test size(sparse_cumcounts) == (3, 1, 2)
    @test sparse_cumcounts[1, 1, :] == [0, 2]
end

@testset "Softmax probabilities" begin
    p = BayesKNN._softmax_probabilities([1.0, 2.0, 3.0])

    @test all(p .> 0)
    @test sum(p) ≈ 1.0
    @test argmax(p) == 3
    @test BayesKNN._softmax_probabilities([1000.0, 1000.0]) ≈ [0.5, 0.5]
end

@testset "Validation" begin
    @test_throws ArgumentError fit_bayesknn([1.0;;], [1])
    @test_throws ArgumentError fit_bayesknn([1.0; Inf;;], [1, 2])
    @test_throws DimensionMismatch fit_bayesknn(reshape([1.0, 2.0], 2, 1), [1])
    @test_throws ArgumentError fit_bayesknn(reshape([1.0, 2.0], 2, 1), [1, missing])
    @test_throws ArgumentError fit_bayesknn(reshape([1.0, 2.0], 2, 1), [1, 2]; k_values = Int[])
    @test_throws ArgumentError fit_bayesknn(reshape([1.0, 2.0], 2, 1), [1, 2]; k_values = [2])
    @test_throws ArgumentError fit_bayesknn(reshape([1.0, 2.0], 2, 1), [1, 2]; beta_prior = Normal())

    fit = fake_fit(reshape([0.0, 4.0], 2, 1), ["a", "b"])

    @test_throws DimensionMismatch predict_proba(fit, reshape([1.0, 2.0], 1, 2))
    @test_throws ArgumentError predict_proba(fit, [NaN;;])
end

@testset "Posterior predictions" begin
    fit = fake_fit(reshape([0.0, 4.0], 2, 1), ["a", "b"])
    pred = predict_proba(fit, reshape([1.0, 3.0], 2, 1))

    @test size(pred.p_mean) == (2, 2)
    @test size(pred.p_draws) == (2, 2, 2)
    @test pred.classes == ["a", "b"]
    @test all(isapprox.(sum(pred.p_mean; dims = 2), 1.0; atol = 1e-12))

    cls = predict_class(fit, reshape([1.0, 3.0], 2, 1))

    @test cls.yhat == ["a", "b"]
    @test cls.yhat_encoded == [1, 2]
end

@testset "Non-string labels" begin
    int_fit = fake_fit(reshape([0.0, 4.0], 2, 1), [10, 20])
    int_pred = predict_class(int_fit, reshape([1.0, 3.0], 2, 1))

    @test int_fit.classes == [10, 20]
    @test int_pred.yhat == [10, 20]
    @test eltype(int_pred.yhat) <: Integer

    symbol_fit = fake_fit(reshape([0.0, 4.0], 2, 1), [:control, :treated])
    symbol_pred = predict_class(symbol_fit, reshape([1.0, 3.0], 2, 1))

    @test symbol_fit.classes == [:control, :treated]
    @test symbol_pred.yhat == [:control, :treated]
    @test eltype(symbol_pred.yhat) <: Symbol
end

@testset "Posterior summaries and anomaly scores" begin
    fit = fake_fit(
        reshape([0.0, 4.0], 2, 1),
        ["a", "b"];
        beta = [0.0, 1.0, 2.0],
        k_idx = [1, 1, 1],
    )

    k_pmf = posterior_k_pmf(fit)
    beta_summary = posterior_beta_summary(fit)
    anomaly = anomaly_score(fit, reshape([3.0], 1, 1); threshold = 0.5)

    @test k_pmf.k == [1]
    @test k_pmf.posterior_prob == [1.0]
    @test beta_summary.mean == 1.0
    @test anomaly.mean_distance == [1.0]
    @test anomaly.median_distance == [1.0]
    @test anomaly.p_gt_threshold == [1.0]
    @test sprint(show, fit) == "BayesKNNFit(n_train=2, n_features=1, n_classes=2, n_k=1, n_draws=3)"

    # k_pmf with multiple k values and known draw distribution
    fit2 = fake_fit(
        reshape([0.0, 2.0, 4.0], 3, 1),
        ["a", "b", "a"];
        beta = [0.0, 0.0, 1.0],
        k_idx = [1, 2, 1],
        k_values = [1, 2],
    )
    k_pmf2 = posterior_k_pmf(fit2)
    @test k_pmf2.k == [1, 2]
    @test k_pmf2.posterior_prob ≈ [2/3, 1/3]
    @test all(0 .<= k_pmf2.posterior_prob .<= 1)
    @test sum(k_pmf2.posterior_prob) ≈ 1.0

    # anomaly score without threshold returns nothing for p_gt_threshold
    fit3 = fake_fit(reshape([0.0, 4.0], 2, 1), ["a", "b"]; beta = [1.0], k_idx = [1])
    score = anomaly_score(fit3, reshape([2.0], 1, 1))
    @test isnothing(score.p_gt_threshold)
    @test score.mean_distance ≈ [2.0]
end

@testset "Beta = 0 gives uniform probabilities" begin
    fit = fake_fit(reshape([0.0, 4.0], 2, 1), ["a", "b"]; beta = [0.0], k_idx = [1])
    pred = predict_proba(fit, reshape([0.0, 2.0, 4.0], 3, 1))
    @test pred.p_mean ≈ fill(0.5, 3, 2)
    @test all(isapprox.(sum(pred.p_mean; dims = 2), 1.0; atol = 1e-12))
end

@testset "Three classes" begin
    X = reshape([0.0, 5.0, 10.0], 3, 1)
    y = ["a", "b", "c"]
    fit = fake_fit(X, y; beta = [10.0], k_idx = [1], k_values = [1])

    @test length(fit.classes) == 3

    pred = predict_proba(fit, reshape([-1.0, 5.0, 11.0], 3, 1))
    @test size(pred.p_mean) == (3, 3)
    @test all(isapprox.(sum(pred.p_mean; dims = 2), 1.0; atol = 1e-12))

    cls = predict_class(fit, reshape([-1.0, 5.0, 11.0], 3, 1))
    @test cls.yhat == ["a", "b", "c"]
end

@testset "Imbalanced classes" begin
    X = reshape([0.0, 0.1, 0.2, 0.3, 4.0], 5, 1)
    y = ["a", "a", "a", "a", "b"]
    fit = fake_fit(X, y; beta = [5.0], k_idx = [1], k_values = [1])
    pred = predict_proba(fit, reshape([0.15, 3.9], 2, 1))

    @test all(isapprox.(sum(pred.p_mean; dims = 2), 1.0; atol = 1e-12))
    @test pred.p_mean[1, 1] > 0.9    # near "a" cluster → high probability for "a"
    @test pred.p_mean[2, 2] > 0.9    # near "b" → high probability for "b"
end

@testset "Credible interval ordering" begin
    fit = fake_fit(
        reshape([0.0, 4.0], 2, 1),
        ["a", "b"];
        beta = [0.5, 1.0, 2.0, 5.0],
        k_idx = [1, 1, 1, 1],
    )
    pred = predict_proba(fit, reshape([1.0, 3.0], 2, 1))

    @test all(pred.p_lo .<= pred.p_mean)
    @test all(pred.p_mean .<= pred.p_hi)
    @test all(pred.p_lo .>= 0.0)
    @test all(pred.p_hi .<= 1.0)
end

@testset "Tiny seeded fit" begin
    rng = MersenneTwister(123)
    X = reshape([0.0, 0.2, 3.8, 4.0], 4, 1)
    y = ["a", "a", "b", "b"]

    fit = fit_bayesknn(X, y; k_values = [1, 2], nsamples = 20, discard_initial = 0, rng = rng)
    pred = predict_proba(fit, reshape([0.1, 3.9], 2, 1))

    @test fit.classes == ["a", "b"]
    @test fit.k_values == [1, 2]
    @test size(pred.p_mean) == (2, 2)
    @test all(isapprox.(sum(pred.p_mean; dims = 2), 1.0; atol = 1e-12))

    # Symmetry: well-separated classes should give mirrored probabilities
    @test pred.p_mean[1, 1] ≈ pred.p_mean[2, 2]
    @test pred.p_mean[1, 2] ≈ pred.p_mean[2, 1]
    # Directionality: test point near "a" should favour class "a"
    @test pred.p_mean[1, 1] > 0.5
    @test pred.p_mean[2, 2] > 0.5
    # k_idx draws must be valid indices into k_values
    draws = BayesKNN._extract_bayesknn_draws(fit.chain)
    @test all(1 .<= draws.k_idx .<= length(fit.k_values))
    @test all(draws.beta .> 0)

    # single chain has ESS but no R-hat
    @test !isnothing(fit.diagnostics)
    @test haskey(fit.diagnostics, :ess_beta)
    @test haskey(fit.diagnostics, :ess_k)
    @test !haskey(fit.diagnostics, :rhat_beta)
    @test fit.diagnostics.ess_beta > 0
end

@testset "Custom beta prior" begin
    rng = MersenneTwister(456)
    X = reshape([0.0, 0.2, 3.8, 4.0], 4, 1)
    y = ["a", "a", "b", "b"]
    prior = Gamma(2.0, 1.0)

    fit = fit_bayesknn(
        X,
        y;
        k_values = [1, 2],
        beta_prior = prior,
        nsamples = 10,
        discard_initial = 0,
        rng = rng,
    )

    draws = BayesKNN._extract_bayesknn_draws(fit.chain)

    @test length(draws.beta) == 10
    @test all(draws.beta .> 0)
    @test all(isfinite.(logpdf.(Ref(prior), draws.beta)))
end

@testset "Multiple chains" begin
    rng = MersenneTwister(321)
    X = reshape([0.0, 0.2, 3.8, 4.0], 4, 1)
    y = [:a, :a, :b, :b]

    fit = fit_bayesknn(
        X,
        y;
        k_values = [1, 2],
        nsamples = 10,
        nchains = 2,
        rng = rng,
    )

    draws = BayesKNN._extract_bayesknn_draws(fit.chain)
    pred = predict_proba(fit, reshape([0.1, 3.9], 2, 1))
    k_pmf = posterior_k_pmf(fit)

    @test length(draws.beta) == 20
    @test length(draws.k_idx) == 20
    @test size(pred.p_draws) == (20, 2, 2)
    @test k_pmf.k == [1, 2]
    @test sum(k_pmf.posterior_prob) ≈ 1.0

    # diagnostics present for multi-chain fit: ESS + R-hat
    @test !isnothing(fit.diagnostics)
    @test haskey(fit.diagnostics, :ess_beta)
    @test haskey(fit.diagnostics, :ess_k)
    @test haskey(fit.diagnostics, :rhat_beta)
    @test haskey(fit.diagnostics, :rhat_k)
    @test fit.diagnostics.ess_beta > 0
    @test fit.diagnostics.rhat_beta isa Float64
    @test fit.diagnostics.rhat_k isa Float64
end
