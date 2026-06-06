using Test
using Latte
using Latte: resolve_targets, DataDependentQuantity, DerivedTargetDescriptor,
    AbstractTargetDescriptor, TargetDescriptor, _sbc_loglik, _sbc_complete,
    _prior_simulate
using DynamicPPL: @model
using Distributions
using GaussianMarkovRandomFields: IIDModel, ExponentialFamily
using StableRNGs: StableRNG
using Statistics

# Small Gaussian–Gaussian IID LGM with a FIXED observation σ — the
# conditional observation density is a plain Normal, so the joint
# log-likelihood is hand-verifiable.
function _gauss_iid_lgm(n; σ = 0.5)
    hp = @hyperparams begin
        (τ ~ PCPrior.Precision(1.0, α = 0.01), transform = log, space = natural)
        σ = σ
    end
    return LatentGaussianModel(hp, IIDModel(n), ExponentialFamily(Normal))
end

@testset "DataDependentQuantity SBC target" begin

    @testset "resolve_targets yields one derived descriptor" begin
        lgm = _gauss_iid_lgm(4)
        descs = resolve_targets(DataDependentQuantity(), lgm)
        @test length(descs) == 1
        @test descs[1] isa DerivedTargetDescriptor
        @test descs[1] isa AbstractTargetDescriptor
        @test descs[1].label == :loglik

        descs_c = resolve_targets(DataDependentQuantity(quantity = :complete), lgm)
        @test descs_c[1].label == :log_complete
    end

    @testset "hyperparameter descriptors are still TargetDescriptors" begin
        lgm = _gauss_iid_lgm(4)
        d = resolve_targets(Hyperparameters(), lgm)[1]
        @test d isa TargetDescriptor
        @test d isa AbstractTargetDescriptor
        @test d.label == :τ
    end

    @testset "loglik eval matches the hand-computed observation density" begin
        n = 5
        σ = 0.5
        lgm = _gauss_iid_lgm(n; σ = σ)
        θ_nt = (τ = 2.0, σ = σ)
        x = collect(range(-1.0, 1.0; length = n))
        y = x .+ 0.1 .* collect(1:n)
        got = _sbc_loglik(lgm, θ_nt, x, y)
        want = sum(logpdf(Normal(x[i], σ), y[i]) for i in 1:n)
        @test got ≈ want
    end

    @testset "complete = loglik + latent log-prior" begin
        n = 4
        σ = 0.5
        lgm = _gauss_iid_lgm(n; σ = σ)
        θ_nt = (τ = 3.0, σ = σ)
        x = [0.2, -0.5, 0.1, 0.3]
        y = [0.1, -0.4, 0.0, 0.6]
        ll = _sbc_loglik(lgm, θ_nt, x, y)
        lc = _sbc_complete(lgm, θ_nt, x, y)
        @test lc > ll || lc < ll          # differs by the latent log-prior term
        @test isfinite(lc)
        # latent prior is the IID GMRF at τ: N(0, 1/τ I)
        @test isapprox(lc - ll, sum(logpdf(Normal(0.0, 1 / sqrt(3.0)), xi) for xi in x); rtol = 1.0e-6)
    end

    @testset "LGM prior_simulate records the latent truth" begin
        n = 6
        lgm = _gauss_iid_lgm(n)
        rep = _prior_simulate(lgm, identity, zeros(n), StableRNG(1); replicate_id = 1)
        @test rep.latent_truth !== nothing
        @test length(rep.latent_truth) == n
    end

    @testset "end-to-end sbc_run with DataDependentQuantity (LGM path, :tmb)" begin
        n = 5
        build = y -> _gauss_iid_lgm(n)
        r = sbc_run(
            build, Vector{Missing}(missing, n);
            n_attempted = 40, n_posterior = 150,
            engine = :tmb, targets = DataDependentQuantity(),
            base_seed = UInt64(0x0005bcdd), progress = false,
        )
        @test r isa SBCResult
        @test length(r.targets) == 1
        @test r.targets[1].label == :loglik
        @test all(0 .<= r.ranks .<= r.n_posterior)
        @test r.status != :invalid
        q = sbc_quantile_position(r, 1)
        @test all(0.0 .< q .< 1.0)
    end

    @testset "DPPL path errors clearly (latent truth not recorded)" begin
        @model function m_dppl(y, n)
            τ ~ PCPrior.Precision(1.0, α = 0.01)
            x ~ IIDModel(n)(τ = τ)
            for i in eachindex(y)
                y[i] ~ Normal(x[i], 0.5)
            end
        end
        n = 4
        build = y -> m_dppl(y, n)
        @test_throws Exception sbc_run(
            build, Vector{Missing}(missing, n);
            n_attempted = 3, n_posterior = 20, engine = :inla,
            random = (:x,), targets = DataDependentQuantity(), progress = false,
        )
    end
end
