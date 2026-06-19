using Test
using Latte
using DynamicPPL: @model
using Distributions, Random, LinearAlgebra
import GaussianMarkovRandomFields as G

# Issue #23: broadcast (dotted-tilde) priors `u .~ Dist.(…)` must build without crashing and produce
# the same latent prior as the explicit loop form `for i; u[i] ~ Dist(…); end`. A non-Gaussian
# broadcast prior should additionally auto-structure into a factor-graph `StructuredLatentPrior`.

# `(μ, Q)` of a built Gaussian latent prior at a hyperparameter value — the prior the engine sees.
_mu_Q(model, hp::NamedTuple) =
    (Distributions.mean(model; hp...), Matrix(Latte.precision_matrix(model; hp...)))

@testset "IID broadcast prior matches the loop form's latent prior" begin
    # `u .~ Normal.(0.0, τ)` (all-scalar broadcast) used to crash at construction with
    # `UndefVarError: τ` (recognition mis-fire) and `BoundsError` (the broadcast collapsed to one
    # distribution). It must now lower to the IID Gaussian prior `N(0, τ²·I)`, identical to the loop.
    n = 12

    @latte function iid_dotted(y, n)
        log_τ ~ Normal(0.0, 1.0)
        τ = exp(log_τ)
        u = Vector{Real}(undef, n)
        u .~ Normal.(0.0, τ)
        for i in 1:n
            y[i] ~ Poisson(exp(u[i]); check_args = false)
        end
    end

    # Plain-DPPL loop reference: the same model written element-wise.
    @model function iid_loop(y, n)
        log_τ ~ Normal(0.0, 1.0)
        τ = exp(log_τ)
        u = Vector{Real}(undef, n)
        for i in 1:n
            u[i] ~ Normal(0.0, τ)
        end
        for i in 1:n
            y[i] ~ Poisson(exp(u[i]); check_args = false)
        end
    end

    Random.seed!(20260619)
    y = [rand(Poisson(exp(0.5 * randn()))) for _ in 1:n]

    lgm = iid_dotted(y, n)
    @test length(lgm.latent_prior) == n   # not collapsed to length 1

    dotted = Latte.build_latent_model(Latte._LATTE_DPPL_CONSTRUCTORS[iid_dotted](y, n), (:u,), (:log_τ,))[1]
    loop = Latte.build_latent_model(iid_loop(y, n), (:u,), (:log_τ,))[1]
    for v in (-0.3, 0.4)
        μd, Qd = _mu_Q(dotted, (log_τ = v,))
        μl, Ql = _mu_Q(loop, (log_τ = v,))
        @test μd ≈ μl atol = 1.0e-10
        @test Qd ≈ Ql atol = 1.0e-10
        # The IID prior is `N(0, τ²·I)`: zero mean, precision `(1/τ²)·I`.
        @test μd ≈ zeros(n) atol = 1.0e-10
        @test Qd ≈ exp(-2v) * I(n) atol = 1.0e-9
    end
end

@testset "Broadcast prior runs end-to-end through inla" begin
    n = 25

    @latte function iid_dotted(y, n)
        log_τ ~ Normal(0.0, 1.0)
        τ = exp(log_τ)
        u = Vector{Real}(undef, n)
        u .~ Normal.(0.0, τ)
        for i in 1:n
            y[i] ~ Poisson(exp(u[i]); check_args = false)
        end
    end

    Random.seed!(20260619)
    u_true = 0.6 .* randn(n)
    y = [rand(Poisson(exp(u))) for u in u_true]

    r = inla(iid_dotted(y, n), y; progress = false)
    @test converged(r)
    @test length(latent_marginals(r)) == n
    # Loose sanity: the latent posterior tracks the truth within 3 posterior SE.
    means = mean.(latent_marginals(r))
    stds = std.(latent_marginals(r))
    @test count(abs.(means .- u_true) .< 3 .* stds) >= n - 1
end

@testset "Data-mean broadcast prior matches the loop form's latent prior" begin
    # `u .~ Normal.(μ, σ)` with an array mean (data) must lower to `N(μ, σ²·I)` — the broadcast does
    # not collapse here, but the same lowering must keep it correct.
    n = 12

    @latte function mean_dotted(y, μ, n)
        log_σ ~ Normal(0.0, 1.0)
        σ = exp(log_σ)
        u = Vector{Real}(undef, n)
        u .~ Normal.(μ, σ)
        for i in 1:n
            y[i] ~ Poisson(exp(u[i]); check_args = false)
        end
    end

    @model function mean_loop(y, μ, n)
        log_σ ~ Normal(0.0, 1.0)
        σ = exp(log_σ)
        u = Vector{Real}(undef, n)
        for i in 1:n
            u[i] ~ Normal(μ[i], σ)
        end
        for i in 1:n
            y[i] ~ Poisson(exp(u[i]); check_args = false)
        end
    end

    Random.seed!(20260619)
    μ = collect(range(-1.0, 1.0; length = n))
    y = [rand(Poisson(exp(m + 0.3 * randn()))) for m in μ]

    dotted = Latte.build_latent_model(Latte._LATTE_DPPL_CONSTRUCTORS[mean_dotted](y, μ, n), (:u,), (:log_σ,))[1]
    loop = Latte.build_latent_model(mean_loop(y, μ, n), (:u,), (:log_σ,))[1]
    for v in (-0.2, 0.5)
        μd, Qd = _mu_Q(dotted, (log_σ = v,))
        μl, Ql = _mu_Q(loop, (log_σ = v,))
        @test μd ≈ μl atol = 1.0e-10
        @test Qd ≈ Ql atol = 1.0e-10
        @test μd ≈ μ atol = 1.0e-10           # prior mean is the data mean
        @test Qd ≈ exp(-2v) * I(n) atol = 1.0e-9
    end
end

@testset "Non-Gaussian broadcast prior auto-structures" begin
    # A Logistic broadcast prior is non-Gaussian (value-dependent Hessian) → the structured path must
    # engage, reproducing the monolithic prior (so `isa StructuredLatentPrior` is the assertion that
    # the extracted factor graph matched the monolith; the local quadratic is checked too).
    n = 12

    @latte function logistic_dotted(y, n)
        log_s ~ Normal(0.0, 0.5)
        s = exp(log_s)
        u = Vector{Real}(undef, n)
        u .~ Logistic.(0.0, s)
        for i in 1:n
            y[i] ~ Poisson(exp(u[i]); check_args = false)
        end
    end

    Random.seed!(20260619)
    y = [rand(Poisson(exp(0.4 * randn()))) for _ in 1:n]

    lgm = logistic_dotted(y, n)
    @test lgm.latent_prior isa G.StructuredLatentPrior

    dppl = Latte._LATTE_DPPL_CONSTRUCTORS[logistic_dotted](y, n)
    mono, path = Latte.build_latent_model(dppl, (:u,), (:log_s,))
    @test path === :sparse_nongaussian
    @test !(mono isa G.StructuredLatentPrior)

    hp = (log_s = -0.2,)
    for x in (0.3 .* sin.(1:n), -0.4 .* cos.(1:n))
        lqs = G.local_quadratic(lgm.latent_prior, x; hp...)
        lqm = G.local_quadratic(mono, x; hp...)
        @test maximum(abs.(Matrix(lqs.Q) .- Matrix(lqm.Q))) < 1.0e-9
        @test maximum(abs.(lqs.h .- lqm.h)) < 1.0e-8
        @test abs(lqs.logp_ref - lqm.logp_ref) < 1.0e-8
    end
end

@testset "Factor extraction recovers a broadcast prior site" begin
    # The loop-preserving walker must pick up `u .~ Logistic.(0.0, s)` as exactly one prior factor
    # template, synthesizing a loop over the broadcast axis.
    n = 8
    body = quote
        log_s ~ Normal(0.0, 0.5)
        s = exp(log_s)
        u = Vector{Real}(undef, n)
        u .~ Logistic.(0.0, s)
        for i in 1:n
            y[i] ~ Poisson(exp(u[i]); check_args = false)
        end
    end
    @test length(Latte._walk_factor_templates(body, (:u,))) == 1
end

@testset "Self-referential broadcast is rejected with a clear error" begin
    # `x[2:n] .~ Normal.(x[1:n-1], σ)` reads not-yet-sampled latents — ill-posed as a broadcast. It
    # must fail fast with an actionable message rather than as a downstream `BoundsError`.
    err = try
        @eval @latte function selfref(y, n)
            log_σ ~ Normal(0.0, 1.0)
            σ = exp(log_σ)
            x = Vector{Real}(undef, n)
            x[1] ~ Normal(0.0, 1.0)
            x[2:n] .~ Normal.(x[1:(n - 1)], σ)
            for i in 1:n
                y[i] ~ Normal(x[i], 0.1)
            end
        end
        nothing
    catch e
        e
    end
    @test err !== nothing
    @test occursin("loop", lowercase(sprint(showerror, err)))
end
