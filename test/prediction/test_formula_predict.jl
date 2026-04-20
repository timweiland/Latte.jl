using Test
using Latte
using GaussianMarkovRandomFields
using Distributions
using SparseArrays
using Random
using LinearAlgebra
using DataFrames
using StatsModels

@testset "predict(result, new_df)" begin

    @testset "Error for non-formula models" begin
        Random.seed!(42)
        n_base = 5
        n_obs = 10
        A = randn(n_obs, n_base) / sqrt(n_base)

        spec = @hyperparams begin
            (σ ~ InverseGamma(2, 1), transform = log, space = natural)
            (τ ~ Gamma(2, 1), transform = log, space = natural)
        end
        ltom = LinearlyTransformedObservationModel(ExponentialFamily(Normal), A)
        model = LatentGaussianModel(spec, IIDModel(n_base), ltom)
        y = randn(n_obs)
        result = inla(model, y; progress = false, diff_strategy = FiniteDiffStrategy())

        @test_throws ArgumentError predict(result, DataFrame(x = [1, 2, 3]))
    end

    @testset "IID formula model — subset of levels" begin
        Random.seed!(42)
        n = 20
        n_groups = 4
        groups = repeat(1:n_groups, inner = n ÷ n_groups)

        intercept = 1.0
        group_effects = [1.0, -0.5, 0.3, 0.8]
        y = [intercept + group_effects[g] + 0.3 * randn() for g in groups]

        df = DataFrame(y = y, group = groups)

        iid = IID()
        f = @formula(y ~ 1 + iid(group))
        hp = @hyperparams begin
            (σ ~ InverseGamma(2, 1), transform = log, space = natural)
            (τ_iid ~ Gamma(2, 1), transform = log, space = natural)
        end

        result = inla(f, hp, df; family = Normal, progress = false, diff_strategy = FiniteDiffStrategy())

        # Predict at a SUBSET of groups — exercises predict_cols with missing levels
        pred_df = DataFrame(group = [1, 3, 4])
        pred_marginals = predict(result, pred_df)

        @test length(pred_marginals) == 3
        for m in pred_marginals
            @test isfinite(mean(m))
            @test std(m) > 0
        end

        # Compare with manual linear_combinations
        # Augmented field: [η₁...η_n; x_iid₁...x_iid₄; x_intercept]
        n_total = length(result.latent_marginals)
        n_obs_aug = result.augmentation_info.n_linear_predictors

        pred_groups = [1, 3, 4]
        A_manual = spzeros(3, n_total)
        for (i, g) in enumerate(pred_groups)
            A_manual[i, n_obs_aug + g] = 1.0   # IID group effect
            A_manual[i, n_total] = 1.0           # intercept (last component)
        end
        lc_marginals = linear_combinations(result, A_manual)

        for i in 1:3
            @test mean(pred_marginals[i]) ≈ mean(lc_marginals[i]) atol = 1.0e-10
            @test std(pred_marginals[i]) ≈ std(lc_marginals[i]) atol = 1.0e-10
        end
    end

    @testset "IID predict at subset — sensible means" begin
        Random.seed!(123)
        n = 30
        n_groups = 3
        groups = repeat(1:n_groups, inner = n ÷ n_groups)

        intercept = 2.0
        group_effects = [0.5, -1.0, 0.3]
        y = [intercept + group_effects[g] + 0.2 * randn() for g in groups]

        df = DataFrame(y = y, group = groups)

        iid = IID()
        f = @formula(y ~ 1 + iid(group))
        hp = @hyperparams begin
            (σ ~ InverseGamma(2, 1), transform = log, space = natural)
            (τ_iid ~ Gamma(2, 1), transform = log, space = natural)
        end

        result = inla(f, hp, df; family = Normal, progress = false, diff_strategy = FiniteDiffStrategy())

        # Predict for only groups 1 and 2 — subset of training levels
        pred_df = DataFrame(group = [1, 2])
        pred_marginals = predict(result, pred_df)

        @test length(pred_marginals) == 2
        for m in pred_marginals
            @test isfinite(mean(m))
            @test std(m) > 0
        end

        # Predictions should include the intercept — means should be roughly
        # intercept + group_effect, not just the group effect alone
        @test mean(pred_marginals[1]) > 1.0   # intercept + 0.5 ≈ 2.5
        @test mean(pred_marginals[2]) < 2.0   # intercept + (-1.0) ≈ 1.0
    end

    @testset "Matern formula model — finite predictions" begin
        Random.seed!(456)
        n_obs = 40

        x_train = 10.0 * rand(n_obs)
        y_train = 10.0 * rand(n_obs)
        y_vals = sin.(x_train) .+ 0.5 .* randn(n_obs)

        df = DataFrame(x = x_train, y_coord = y_train, y = y_vals)

        matern = Matern(smoothness = 1)
        f = @formula(y ~ 1 + matern(x, y_coord))
        hp = @hyperparams begin
            (σ ~ InverseGamma(2, 1), transform = log, space = natural)
            (τ_matern ~ PCPrior.Precision(1.0, α = 0.01), transform = log)
            (range_matern ~ Exponential(5.0), transform = log, space = natural)
        end

        result = inla(f, hp, df; family = Normal, progress = false, diff_strategy = FiniteDiffStrategy())

        # Predict at a small grid (new locations not in training data)
        pred_df = DataFrame(
            x = [2.0, 5.0, 8.0, 2.0, 5.0, 8.0],
            y_coord = [2.0, 2.0, 2.0, 8.0, 8.0, 8.0]
        )
        pred_marginals = predict(result, pred_df)

        @test length(pred_marginals) == 6
        for m in pred_marginals
            @test isfinite(mean(m))
            @test std(m) > 0
            @test !isnan(mean(m))
        end
    end
end
