using Test
using Latte
using GaussianMarkovRandomFields: IIDModel
using Distributions
using Statistics
using Random

# A Gaussian observation site whose mean is nonlinear in the latent field x —
# `y[i] ~ Normal(f(x), σ)` — should be recognized and dispatched to GMRFs'
# NonlinearLeastSquaresModel (Gauss-Newton), not the generic AutoDiff path.
@testset "NLS recognition: Normal with nonlinear-in-x mean" begin
    @latte function nlmodel(y, n)
        τ ~ truncated(Normal(1.0, 0.5); lower = 0.1)
        x ~ IIDModel(n)(τ = τ)
        for i in eachindex(y)
            y[i] ~ Normal(exp(x[i]), 0.1)   # nonlinear (exp) in x
        end
    end
    n = 5
    Random.seed!(1)
    y = exp.(0.2 .* randn(n)) .+ 0.1 .* randn(n)

    @testset "obs model recognized as NonlinearLeastSquares (default)" begin
        lgm = nlmodel(y, n)
        @test occursin("NonlinearLeastSquares", string(typeof(lgm.observation_model)))
    end

    @testset "inla runs and gives finite marginals" begin
        res = inla(nlmodel(y, n), y; latent_marginalization_method = GaussianMarginal(), progress = false)
        lm = latent_marginals(res)
        @test all(m -> isfinite(mean(m)) && isfinite(std(m)), lm)
    end

    # Safety: a Gaussian site whose σ depends on the latent looks homoskedastic
    # with constant σ at the zero seed, but NLS would freeze σ and fit the wrong
    # model. It must punt to the AD fallback, not fire NLS.
    @testset "latent-dependent σ punts to AD (not NLS)" begin
        @latte function nlmodel_hetσ(y, n)
            τ ~ truncated(Normal(1.0, 0.5); lower = 0.1)
            x ~ IIDModel(n)(τ = τ)
            for i in eachindex(y)
                y[i] ~ Normal(exp(x[i]), exp(x[i]))   # σ depends on the latent
            end
        end
        lgm = nlmodel_hetσ(y, n)
        @test !occursin("NonlinearLeastSquares", string(typeof(lgm.observation_model)))
    end

    # A mildly-curved Normal mean is too gently curved for the tiny-step affine
    # probe to flag, so it would be silently linearized. The curvature-direct
    # check must route it to NLS, not the linear fast path.
    @testset "mildly-curved Normal mean recognized as NLS (not linearized)" begin
        @latte function nlmodel_mild(y, n)
            τ ~ truncated(Normal(1.0, 0.5); lower = 0.1)
            x ~ IIDModel(n)(τ = τ)
            for i in eachindex(y)
                y[i] ~ Normal(x[i] + 0.01 * x[i]^2, 0.1)
            end
        end
        lgm = nlmodel_mild(y, n)
        @test occursin("NonlinearLeastSquares", string(typeof(lgm.observation_model)))
    end

    # Opt-out: `nls = false` forces the exact full-Hessian AD path instead of the
    # Gauss–Newton NLS approximation — for both genuinely-nonlinear and
    # mildly-curved Normal means. The curved case must reach AD, never the wrong
    # affine linearization.
    @testset "nls = false opts out to the exact AD path" begin
        @latte function nlmodel_optout(y, n)
            τ ~ truncated(Normal(1.0, 0.5); lower = 0.1)
            x ~ IIDModel(n)(τ = τ)
            for i in eachindex(y)
                y[i] ~ Normal(exp(x[i]), 0.1)
            end
        end
        @test occursin("NonlinearLeastSquares", string(typeof(nlmodel_optout(y, n).observation_model)))
        optout = nlmodel_optout(y, n; nls = false)
        @test !occursin("NonlinearLeastSquares", string(typeof(optout.observation_model)))
        @test occursin("AutoDiff", string(typeof(optout.observation_model)))

        @latte function nlmodel_optout_mild(y, n)
            τ ~ truncated(Normal(1.0, 0.5); lower = 0.1)
            x ~ IIDModel(n)(τ = τ)
            for i in eachindex(y)
                y[i] ~ Normal(x[i] + 0.01 * x[i]^2, 0.1)
            end
        end
        mild_optout = nlmodel_optout_mild(y, n; nls = false)
        @test !occursin("NonlinearLeastSquares", string(typeof(mild_optout.observation_model)))
        @test occursin("AutoDiff", string(typeof(mild_optout.observation_model)))
    end

    # Heteroskedastic but constant σ (per-site fixed noise) is still NLS — the
    # per-site σ vector is frozen into the model.
    @testset "heteroskedastic constant σ is recognized as NLS" begin
        @latte function nls_het(y, n)
            τ ~ truncated(Normal(1.0, 0.5); lower = 0.1)
            x ~ IIDModel(n)(τ = τ)
            for i in eachindex(y)
                y[i] ~ Normal(exp(x[i]), 0.05 + 0.02 * i)
            end
        end
        # The per-site σ vector freeze happens at model construction, so the
        # dispatch assert is the whole claim; GN inference itself is covered by
        # the canonical smoke test above.
        lgm = nls_het(y, n)
        @test occursin("NonlinearLeastSquares", string(typeof(lgm.observation_model)))
    end

    # σ driven by a noise-scale hyperparameter named `σ` flows as a hyperparameter
    # (inferred), rather than being frozen — the Gauss–Newton x-Hessian with exact
    # θ-gradients through σ.
    @testset "hp-dependent σ (named σ) flows as a hyperparameter" begin
        @latte function nls_hpσ(y, n)
            τ ~ truncated(Normal(1.0, 0.5); lower = 0.1)
            σ ~ truncated(Normal(0.3, 0.2); lower = 0.01)
            x ~ IIDModel(n)(τ = τ)
            for i in eachindex(y)
                y[i] ~ Normal(exp(x[i]), σ)
            end
        end
        lgm = nls_hpσ(y, n)
        @test occursin("NonlinearLeastSquares", string(typeof(lgm.observation_model)))
        # σ is NOT pre-bound (no _FixedKwargs wrapper): it flows as a hyperparameter.
        @test !occursin("FixedKwargs", string(typeof(lgm.observation_model)))
        # inla running to completion proves σ is routed: NLS requires σ as a kwarg
        # at materialization, so a missing/misrouted σ would error here.
        res = inla(nls_hpσ(y, n), y; latent_marginalization_method = GaussianMarginal(), progress = false)
        @test all(m -> isfinite(mean(m)) && isfinite(std(m)), latent_marginals(res))
    end

    # A σ that is a *transform* of a hyperparameter can't be routed 1:1, so it
    # must punt to the exact AD path rather than silently routing the wrong σ.
    @testset "σ as a transform of a hyperparameter punts to AD" begin
        @latte function nls_σtransform(y, n)
            τ ~ truncated(Normal(1.0, 0.5); lower = 0.1)
            λ ~ truncated(Normal(1.0, 0.5); lower = 0.1)
            x ~ IIDModel(n)(τ = τ)
            for i in eachindex(y)
                y[i] ~ Normal(exp(x[i]), 1.0 / sqrt(λ))
            end
        end
        lgm = nls_σtransform(y, n)
        @test !occursin("NonlinearLeastSquares", string(typeof(lgm.observation_model)))
    end

    # The latent field can enter the forward map nonlinearly *and* the map can
    # depend on a hyperparameter (a parameterized residual). NLS carries the hp
    # into the residual via its `hyperparams`, so the outer θ-gradient stays
    # exact. The dependence is multiplicative (`exp(α·x)`), invisible at x = 0 —
    # the recognizer must probe at a nonzero latent to catch it.
    @testset "hp-dependent nonlinear mean routes the hp into the NLS residual" begin
        @latte function nls_hpmean(y, n)
            α ~ truncated(Normal(1.0, 0.5); lower = 0.1)
            τ ~ truncated(Normal(1.0, 0.5); lower = 0.1)
            x ~ IIDModel(n)(τ = τ)
            for i in eachindex(y)
                y[i] ~ Normal(exp(α * x[i]), 0.1)
            end
        end
        # Dispatch is asserted here; that the hp is routed *correctly* into the
        # residual is verified numerically (Gauss–Newton vs exact-AD agreement
        # on α) in test_nlsq_composite.jl.
        lgm = nls_hpmean(y, n)
        @test occursin("NonlinearLeastSquares", string(typeof(lgm.observation_model)))
    end
end
