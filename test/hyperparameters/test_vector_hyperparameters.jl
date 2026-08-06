using Test
using Latte
using Distributions
using Bijectors
using LinearAlgebra

# Vector-valued hyperparameters (issue #41): a free hyperparameter may carry a
# multivariate prior (e.g. MvNormal with non-diagonal covariance). The flat θ
# vector concatenates per-name blocks in declaration order.

@testset "Vector hyperparameters" begin

    Σ_κ = [1.0 0.5; 0.5 2.0]
    μ_κ = [0.0, 1.0]

    # Mixed spec used throughout: scalar σ (natural-space Exponential, log
    # transform), vector κ (working-space MvNormal, identity), vector τv
    # (natural-space positive pair, elementwise log). Total working dim = 5.
    hp_σ = Hyperparameter(Exponential(1.0), transform = elementwise(log), prior_space = :natural)
    hp_κ = Hyperparameter(MvNormal(μ_κ, Σ_κ))
    hp_τv = Hyperparameter(
        product_distribution([Exponential(1.0), Exponential(2.0)]),
        transform = elementwise(log), prior_space = :natural,
    )
    spec = HyperparameterSpec(free = (σ = hp_σ, κ = hp_κ, τv = hp_τv), fixed = (μ = 0.0,))

    w = [0.3, -0.4, 1.2, 0.1, -0.2]   # [σ; κ; τv] in working space
    n = [exp(0.3), -0.4, 1.2, exp(0.1), exp(-0.2)]   # natural space

    @testset "Spec construction and @hyperparams" begin
        mspec = @hyperparams begin
            (σ ~ Exponential(1.0), transform = log, space = natural)
            κ ~ MvNormal([0.0, 1.0], [1.0 0.5; 0.5 2.0])
        end
        @test mspec isa HyperparameterSpec
        @test mspec.free.κ.prior isa MvNormal
        @test length(mspec.free.κ.prior) == 2
    end

    @testset "Transform validation for vector entries" begin
        # Dimension-changing bijectors can't share one flat layout between
        # working and natural space — reject with an actionable error.
        dir = Dirichlet(3, 1.0)
        @test_throws ArgumentError Hyperparameter(
            dir; transform = bijector(dir), prior_space = :natural,
        )
        # Non-elementwise transforms on a vector prior are rejected too.
        @test_throws ArgumentError Hyperparameter(
            MvNormal(zeros(2), I); transform = Bijectors.Logit(0.0, 1.0),
        )
        # identity and elementwise are accepted.
        @test Hyperparameter(MvNormal(zeros(2), I)) isa Hyperparameter
        @test Hyperparameter(
            product_distribution([Exponential(1.0), Exponential(1.0)]),
            transform = elementwise(log), prior_space = :natural,
        ) isa Hyperparameter
    end

    @testset "Wrapper construction, length, getproperty" begin
        θ_w = WorkingHyperparameters(copy(w), spec)
        @test length(θ_w) == 5
        @test θ_w.σ ≈ 0.3
        @test θ_w.κ ≈ [-0.4, 1.2]
        @test θ_w.τv ≈ [0.1, -0.2]
        @test θ_w.μ == 0.0   # fixed passthrough

        θ_n = NaturalHyperparameters(copy(n), spec)
        @test length(θ_n) == 5
        @test θ_n.σ ≈ exp(0.3)
        @test θ_n.κ ≈ [-0.4, 1.2]
        @test θ_n.τv ≈ exp.([0.1, -0.2])

        # Flat length must match the total dimension, not the number of names.
        @test_throws ErrorException WorkingHyperparameters([1.0, 2.0, 3.0], spec)
        @test_throws ErrorException NaturalHyperparameters([1.0, 2.0, 3.0], spec)
    end

    @testset "Space conversion round-trips" begin
        θ_w = WorkingHyperparameters(copy(w), spec)
        θ_n = convert(NaturalHyperparameters, θ_w)
        @test θ_n.θ ≈ n atol = 1.0e-12

        θ_w2 = convert(WorkingHyperparameters, θ_n)
        @test θ_w2.θ ≈ w atol = 1.0e-12

        nt = convert(NamedTuple, θ_n)
        @test nt.σ ≈ exp(0.3)
        @test nt.κ isa AbstractVector
        @test nt.κ ≈ [-0.4, 1.2]
        @test nt.τv ≈ exp.([0.1, -0.2])
        @test nt.μ == 0.0

        nt_w = convert(NamedTuple, θ_w)
        @test nt_w.κ ≈ [-0.4, 1.2]
        @test nt_w.τv ≈ [0.1, -0.2]
    end

    @testset "logdetjac" begin
        θ_w = WorkingHyperparameters(copy(w), spec)
        θ_n = NaturalHyperparameters(copy(n), spec)
        # working → natural: exp for σ, identity for κ (0), exp per τv entry.
        @test logdetjac(θ_w) ≈ w[1] + 0.0 + w[4] + w[5]
        # natural → working: 1/x per log-transformed coordinate.
        @test logdetjac(θ_n) ≈ -log(n[1]) - log(n[4]) - log(n[5])
    end

    @testset "logpdf_prior matches hand-computed block sum" begin
        θ_w = WorkingHyperparameters(copy(w), spec)
        θ_n = convert(NaturalHyperparameters, θ_w)

        # Natural space: plain sum of natural-space prior densities.
        expected_natural =
            logpdf(Exponential(1.0), n[1]) +
            logpdf(MvNormal(μ_κ, Σ_κ), n[2:3]) +
            logpdf(product_distribution([Exponential(1.0), Exponential(2.0)]), n[4:5])
        @test logpdf_prior(θ_n) ≈ expected_natural

        # Working space: natural density plus the working→natural Jacobian.
        @test logpdf_prior(θ_w) ≈ expected_natural + logdetjac(θ_w)

        # Legacy NamedTuple form flattens vector entries.
        @test logpdf_prior((σ = n[1], κ = n[2:3], τv = n[4:5]), spec) ≈ expected_natural
    end

    @testset "Mode-finding initialisation" begin
        θ0 = initial_hyperparameter_guess(spec)
        @test θ0 isa WorkingHyperparameters
        @test length(θ0) == 5
        @test θ0.κ ≈ μ_κ   # MvNormal working-space prior mode

        starts = Latte.resolve_mode_starts(RandomStarts(3), spec)
        @test length(starts) == 3
        @test all(s -> length(s) == 5, starts)

        # NamedTuple mode_init with a vector-valued entry (natural space).
        starts_nt = Latte.resolve_mode_starts(
            (σ = 2.0, κ = [0.5, -0.5], τv = [1.0, 1.0]), spec,
        )
        @test length(starts_nt) == 1
        @test starts_nt[1].σ ≈ log(2.0)
        @test starts_nt[1].κ ≈ [0.5, -0.5]
        @test starts_nt[1].τv ≈ [0.0, 0.0] atol = 1.0e-12
    end

    @testset "Scalar-only specs unchanged" begin
        sspec = HyperparameterSpec(free = (σ = hp_σ,), fixed = NamedTuple())
        θ_w = WorkingHyperparameters([0.5], sspec)
        @test θ_w.σ ≈ 0.5
        @test convert(NaturalHyperparameters, θ_w).σ ≈ exp(0.5)
        @test logpdf_prior(θ_w) ≈ logpdf(transformed(Exponential(1.0), elementwise(log)), 0.5)
    end
end
