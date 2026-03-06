using Test
using IntegratedNestedLaplace
using GaussianMarkovRandomFields
using Distributions
using SparseArrays
using Statistics
using Random

@testset "rand(::INLAResult)" begin

    # Shared model constructors
    function make_normal_iid_model(n)
        spec = @hyperparams begin
            (σ ~ InverseGamma(2, 1), transform = log, space = natural)
        end
        function latent_func(; σ, kwargs...)
            Q = spdiagm(0 => fill(1 / σ^2, n))
            return GMRF(zeros(n), Q)
        end
        obs_model = ExponentialFamily(Normal)
        return INLAModel(spec, FunctionLatentModel(latent_func, n), obs_model)
    end

    function make_poisson_iid_model(n)
        spec = @hyperparams begin
            (τ ~ Gamma(2, 1), transform = log, space = natural)
        end
        function latent_func(; τ, kwargs...)
            Q = spdiagm(0 => fill(τ, n))
            return GMRF(zeros(n), Q)
        end
        obs_model = ExponentialFamily(Poisson)
        return INLAModel(spec, FunctionLatentModel(latent_func, n), obs_model)
    end

    @testset "Deterministic with seed" begin
        n = 10
        model = make_normal_iid_model(n)
        Random.seed!(42)
        y = randn(n)
        result = inla(model, y; progress = false)

        samples1 = rand(MersenneTwister(123), result, 5)
        samples2 = rand(MersenneTwister(123), result, 5)

        for i in 1:5
            @test samples1[i].x == samples2[i].x
            @test samples1[i].θ == samples2[i].θ
        end
    end

    @testset "Output structure" begin
        n = 10
        model = make_normal_iid_model(n)
        Random.seed!(42)
        y = randn(n)
        result = inla(model, y; progress = false)

        # Multiple samples without include_y
        samples = rand(MersenneTwister(1), result, 3)
        @test length(samples) == 3
        for s in samples
            @test haskey(s, :θ)
            @test haskey(s, :x)
            @test !haskey(s, :y)
            @test length(s.x) == n
        end

        # Multiple samples with include_y
        samples_y = rand(MersenneTwister(1), result, 3; include_y = true)
        @test length(samples_y) == 3
        for s in samples_y
            @test haskey(s, :θ)
            @test haskey(s, :x)
            @test haskey(s, :y)
            @test length(s.x) == n
            @test length(s.y) == n
        end

        # Single sample (no n argument)
        s = rand(MersenneTwister(1), result)
        @test haskey(s, :θ)
        @test haskey(s, :x)
        @test length(s.x) == n

        # Single sample with include_y
        s_y = rand(MersenneTwister(1), result; include_y = true)
        @test haskey(s_y, :y)

        # Default RNG convenience methods
        s_default = rand(result)
        @test haskey(s_default, :x)
        samples_default = rand(result, 3)
        @test length(samples_default) == 3
    end

    @testset "Hyperparameter values come from integration points" begin
        n = 10
        model = make_normal_iid_model(n)
        Random.seed!(42)
        y = randn(n)
        result = inla(model, y; progress = false)

        samples = rand(MersenneTwister(1), result, 50)

        # All sampled θ should be from the integration points
        integration_points = result.exploration.grid_points[result.exploration.integration_indices]
        valid_θ_vecs = Set([p.θ.θ for p in integration_points])

        for s in samples
            @test s.θ.θ in valid_θ_vecs
        end
    end

    @testset "Sample statistics match posterior (Normal IID)" begin
        n = 20
        model = make_normal_iid_model(n)
        Random.seed!(42)
        y = randn(n)
        result = inla(model, y; progress = false)

        n_samples = 2000
        samples = rand(MersenneTwister(99), result, n_samples)

        # Check that sample means approximate posterior means
        x_matrix = hcat([s.x for s in samples]...)  # n × n_samples
        sample_means = mean(x_matrix, dims = 2)[:]
        sample_stds = std(x_matrix, dims = 2)[:]

        posterior_means = [mean(result.latent_marginals[i]) for i in 1:n]
        posterior_stds = [std(result.latent_marginals[i]) for i in 1:n]

        # Sample means should be close to posterior means (with tolerance for Monte Carlo error)
        for i in 1:n
            mc_se = posterior_stds[i] / sqrt(n_samples)
            @test abs(sample_means[i] - posterior_means[i]) < 5 * mc_se
        end

        # Sample stds should be in the right ballpark
        for i in 1:n
            @test 0.5 < sample_stds[i] / posterior_stds[i] < 2.0
        end
    end

    @testset "Poisson model: y samples are non-negative integers" begin
        n = 10
        model = make_poisson_iid_model(n)
        Random.seed!(42)
        y = rand(Poisson(3.0), n)
        result = inla(model, y; progress = false)

        samples = rand(MersenneTwister(1), result, 20; include_y = true)

        for s in samples
            @test all(s.y .>= 0)
            @test all(isinteger.(s.y))
        end
    end

    @testset "Prediction model: samples cover full latent field" begin
        n = 10
        model = make_normal_iid_model(n)

        y = Vector{Union{Missing, Float64}}(randn(n))
        y[3] = missing
        y[7] = missing

        result = inla(model, y; progress = false)

        samples = rand(MersenneTwister(1), result, 10)

        for s in samples
            @test length(s.x) == n
            @test all(isfinite.(s.x))
        end
    end

    @testset "Edge case: n = 1" begin
        n = 10
        model = make_normal_iid_model(n)
        Random.seed!(42)
        y = randn(n)
        result = inla(model, y; progress = false)

        samples = rand(MersenneTwister(1), result, 1)
        @test length(samples) == 1
        @test length(samples[1].x) == n
    end
end
