using Test
using Latte
using Distributions
using GaussianMarkovRandomFields
using LinearAlgebra
using LogDensityProblems
using SparseArrays
using Random
using Statistics

isdefined(@__MODULE__, :make_poisson_iid_model) ||
    include(joinpath(@__DIR__, "..", "..", "shared_test_models.jl"))

# Three-hyperparameter Poisson-IID LGM whose latent constructor counts its own
# invocations. Both gradient paths call it the same number of times per
# objective evaluation, so the ratio of counts between two gradient calls is
# exactly the ratio of objective evaluations.
function make_counting_poisson_model(n, counter)
    spec = @hyperparams begin
        (τ1 ~ Gamma(2, 1), transform = log, space = natural)
        (τ2 ~ Gamma(2, 1), transform = log, space = natural)
        (τ3 ~ Gamma(2, 1), transform = log, space = natural)
    end
    block = [min(div(3 * (i - 1), n) + 1, 3) for i in 1:n]
    function counting_latent(; τ1, τ2, τ3, kwargs...)
        counter[] += 1
        τs = (τ1, τ2, τ3)
        return (zeros(n), spdiagm(0 => [τs[block[i]] for i in 1:n]))
    end
    return LatentGaussianModel(
        spec, FunctionLatentModel(counting_latent, n), ExponentialFamily(Poisson)
    )
end

@testset "hmc_laplace gradient differentiation strategy" begin
    n = 30
    counter = Ref(0)
    model = make_counting_poisson_model(n, counter)
    Random.seed!(2026)
    y = rand(Poisson(3.0), n)
    y_obs, prep_model, _ = Latte._prepare_for_prediction(model, y)
    spec = prep_model.hyperparameter_spec
    θ = [0.3, -0.2, 0.1]

    target_ad = Latte._hmc_target(prep_model, y_obs, θ, spec, ADStrategy())
    target_fd = Latte._hmc_target(prep_model, y_obs, θ, spec, FiniteDiffStrategy())

    @testset "AD gradient matches the finite-difference gradient" begin
        v_ad, g_ad = LogDensityProblems.logdensity_and_gradient(target_ad, θ)
        v_fd, g_fd = LogDensityProblems.logdensity_and_gradient(target_fd, θ)

        # Both paths evaluate the same objective, so the values agree exactly.
        @test v_ad ≈ v_fd rtol = 1.0e-12
        @test v_ad ≈ LogDensityProblems.logdensity(target_ad, θ) rtol = 1.0e-12

        # FD is the reference here, and its own accuracy is the binding
        # constraint on the comparison.
        @test g_ad ≈ g_fd rtol = 1.0e-4 atol = 1.0e-6
        @test length(g_ad) == 3
        @test all(isfinite, g_ad)
    end

    @testset "AD costs O(1) objective evaluations per gradient" begin
        counter[] = 0
        LogDensityProblems.logdensity_and_gradient(target_ad, θ)
        ad_calls = counter[]

        counter[] = 0
        LogDensityProblems.logdensity_and_gradient(target_fd, θ)
        fd_calls = counter[]

        # A single value-only evaluation sets the per-objective unit.
        counter[] = 0
        LogDensityProblems.logdensity(target_ad, θ)
        unit = counter[]
        @test unit >= 1

        # FD needs one objective evaluation per hyperparameter (2d for its
        # central stencil); AD needs a fixed number regardless of d.
        d = length(θ)
        @test fd_calls >= d * unit
        @test ad_calls <= 2 * unit
        @test ad_calls * d <= fd_calls
    end

    @testset "hmc_laplace defaults to the AD gradient path" begin
        # Same model, same seed, so the chain — and therefore the number of
        # objective evaluations — is deterministic. A default-kwarg run that
        # costs exactly what the explicit-AD run costs is on the AD path; the
        # FD run costs strictly more.
        counter[] = 0
        hmc_laplace(model, y; n_samples = 20, n_warmup = 10, rng = MersenneTwister(3))
        default_calls = counter[]

        counter[] = 0
        hmc_laplace(
            model, y; n_samples = 20, n_warmup = 10,
            diff_strategy = ADStrategy(), rng = MersenneTwister(3),
        )
        ad_calls = counter[]

        counter[] = 0
        hmc_laplace(
            model, y; n_samples = 20, n_warmup = 10,
            diff_strategy = FiniteDiffStrategy(), rng = MersenneTwister(3),
        )
        fd_calls = counter[]

        @test default_calls == ad_calls
        @test ad_calls < fd_calls
    end

    @testset "AD and FD chains agree statistically" begin
        m = make_poisson_iid_model(25)
        Random.seed!(11)
        yy = rand(Poisson(4.0), 25)

        r_ad = hmc_laplace(
            m, yy; n_samples = 300, n_warmup = 150,
            diff_strategy = ADStrategy(), rng = MersenneTwister(5),
        )
        r_fd = hmc_laplace(
            m, yy; n_samples = 300, n_warmup = 150,
            diff_strategy = FiniteDiffStrategy(), rng = MersenneTwister(5),
        )

        θ_ad = vec(samples(r_ad))
        θ_fd = vec(samples(r_fd))
        # Same target, so the chains are two MCMC estimates of one posterior.
        # Judge their difference against Monte Carlo error, estimated by batch
        # means — the draws are autocorrelated, so the independent-draws
        # formula understates it by roughly a factor of two here.
        function batch_mcse(x)
            b = floor(Int, sqrt(length(x)))
            m = [mean(x[((i - 1) * b + 1):(i * b)]) for i in 1:div(length(x), b)]
            return std(m) / sqrt(length(m))
        end
        se = sqrt(batch_mcse(θ_ad)^2 + batch_mcse(θ_fd)^2)
        @test abs(mean(θ_ad) - mean(θ_fd)) < 4 * se
        @test isapprox(std(θ_ad), std(θ_fd); rtol = 0.35)

        # Latent marginals from the two paths should be indistinguishable.
        m_ad = [mean(d) for d in latent_marginals(r_ad)]
        m_fd = [mean(d) for d in latent_marginals(r_fd)]
        @test maximum(abs.(m_ad .- m_fd)) < 0.1
    end
end
