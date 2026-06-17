using Test
using Latte
using GaussianMarkovRandomFields
using Distributions
using SparseArrays
using Statistics
using DataFrames
using Random

# `derived` is the nonlinear, sampling-based analogue of `linear_combinations`: it pushes
# posterior draws through an arbitrary g(latents) and returns the empirical marginals. The
# key correctness anchor is that for a LINEAR g it must agree with the analytic
# `linear_combinations`.

@testset "derived" begin
    function make_normal_iid_model(n)
        spec = @hyperparams begin
            (σ ~ InverseGamma(2, 1), transform = log, space = natural)
        end
        function latent_func(; σ, kwargs...)
            Q = spdiagm(0 => fill(1 / σ^2, n))
            return (zeros(n), Q)
        end
        return LatentGaussianModel(spec, FunctionLatentModel(latent_func, n), ExponentialFamily(Normal))
    end

    @testset "output structure (scalar / vector)" begin
        n = 8
        model = make_normal_iid_model(n)
        Random.seed!(1)
        y = randn(n)
        result = inla(model, y; progress = false)

        d = derived(result, z -> sum(z.latent); n_samples = 400, rng = MersenneTwister(7))
        @test d isa SampleMarginal
        @test isfinite(mean(d))

        ds = derived(result, z -> z.latent[1:3]; n_samples = 400, rng = MersenneTwister(7))
        @test ds isa Vector{<:SampleMarginal}
        @test length(ds) == 3
    end

    @testset "linear g agrees with linear_combinations" begin
        n = 8
        model = make_normal_iid_model(n)
        Random.seed!(2)
        y = randn(n)
        result = inla(model, y; progress = false)

        A = randn(3, n)
        lc = linear_combinations(result, A)                 # analytic Gaussian functionals
        d = derived(result, z -> A * z.latent; n_samples = 6000, rng = MersenneTwister(11))
        for k in 1:3
            ## Deterministic (seeded) Monte Carlo vs the exact linear result.
            @test mean(d[k]) ≈ mean(lc[k]) atol = 0.08
            @test std(d[k]) ≈ std(lc[k]) rtol = 0.12
        end
    end

    @testset "reproducible + summary_df" begin
        n = 6
        model = make_normal_iid_model(n)
        Random.seed!(3)
        y = randn(n)
        result = inla(model, y; progress = false)

        d1 = derived(result, z -> sum(z.latent); n_samples = 300, rng = MersenneTwister(99))
        d2 = derived(result, z -> sum(z.latent); n_samples = 300, rng = MersenneTwister(99))
        @test d1.samples == d2.samples

        ds = derived(result, z -> z.latent[1:2]; n_samples = 300, rng = MersenneTwister(99))
        df = summary_df(ds)
        @test nrow(df) == 2
        @test all(isfinite, df.mode)
        @test all(isfinite, df.std)

        @test_throws ArgumentError derived(result, z -> sum(z.latent); n_samples = 0)
    end

    @testset "named latent groups via @latte" begin
        @latte function reg(y, xobs)
            σ ~ Gamma(2.0, 1.0)
            @random β ~ MvNormal(zeros(2), 3.0)
            for i in eachindex(y)
                y[i] ~ Normal(β[1] + β[2] * xobs[i], σ)
            end
        end

        Random.seed!(4)
        xobs = randn(25)
        yobs = 1.0 .+ 2.0 .* xobs .+ 0.3 .* randn(25)
        lgm = reg(yobs, xobs)
        result = inla(lgm, yobs; progress = false)

        ## g reads the latent field by group name.
        d = derived(result, z -> z.β[1] + z.β[2]; n_samples = 1500, rng = MersenneTwister(5))
        @test d isa SampleMarginal
        @test isfinite(mean(d))

        ## Per-component derived must reproduce the latent marginals of :β.
        comps = derived(result, z -> z.β; n_samples = 4000, rng = MersenneTwister(6))
        βm = latent_marginals(result, :β)
        for k in 1:2
            @test mean(comps[k]) ≈ mean(βm[k]) atol = 0.1
        end
    end
end
