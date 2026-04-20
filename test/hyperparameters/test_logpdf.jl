using Test
using Latte
using Distributions
using Bijectors

@testset "logpdf_prior" begin

    @testset "WorkingHyperparameters dispatch" begin
        # Prior specified in natural space, stored in working space
        hp_σ_nat = Hyperparameter(Exponential(1.0), transform = elementwise(log), prior_space = :natural)
        spec_nat = HyperparameterSpec(free = (σ = hp_σ_nat,), fixed = NamedTuple())

        θ_w = WorkingHyperparameters([log(2.0)], spec_nat)
        log_p = logpdf_prior(θ_w)

        # Expected: prior is TransformedDistribution in working space
        # log p(η) where η = log(σ), prior is Exponential(1) transformed by log
        # TransformedDistribution automatically handles Jacobian
        @test isfinite(log_p)

        # Prior specified in working space
        hp_σ_work = Hyperparameter(Normal(0, 1), transform = elementwise(log), prior_space = :working)
        spec_work = HyperparameterSpec(free = (σ = hp_σ_work,), fixed = NamedTuple())

        θ_w2 = WorkingHyperparameters([log(2.0)], spec_work)
        log_p2 = logpdf_prior(θ_w2)

        # Expected: log p(log(2)) where p is Normal(0, 1)
        expected = logpdf(Normal(0, 1), log(2.0))
        @test log_p2 ≈ expected atol = 1.0e-10
    end

    @testset "NaturalHyperparameters dispatch" begin
        # Prior specified in natural space
        hp_σ_nat = Hyperparameter(Exponential(1.0), transform = elementwise(log), prior_space = :natural)
        spec_nat = HyperparameterSpec(free = (σ = hp_σ_nat,), fixed = NamedTuple())

        θ_n = NaturalHyperparameters([2.0], spec_nat)
        log_p = logpdf_prior(θ_n)

        # Expected: converts to working, evaluates, adds Jacobian
        # log p(log(2)) + log|dη/dθ| where η = log(θ)
        # = log p(log(2)) - log(2)  (since dη/dθ = 1/θ)
        θ_w = convert(WorkingHyperparameters, θ_n)
        expected = logpdf_prior(θ_w) + logdetjac(θ_n)
        @test log_p ≈ expected atol = 1.0e-10

        # Prior specified in working space
        hp_σ_work = Hyperparameter(Normal(0, 1), transform = elementwise(log), prior_space = :working)
        spec_work = HyperparameterSpec(free = (σ = hp_σ_work,), fixed = NamedTuple())

        θ_n2 = NaturalHyperparameters([2.0], spec_work)
        log_p2 = logpdf_prior(θ_n2)

        # Expected: same logic, converts to working and adds Jacobian
        θ_w2 = convert(WorkingHyperparameters, θ_n2)
        expected2 = logpdf_prior(θ_w2) + logdetjac(θ_n2)
        @test log_p2 ≈ expected2 atol = 1.0e-10
    end

    @testset "NamedTuple dispatch (legacy interface)" begin
        # Prior specified in natural space
        hp_σ_nat = Hyperparameter(Exponential(1.0), transform = elementwise(log), prior_space = :natural)
        spec_nat = HyperparameterSpec(free = (σ = hp_σ_nat,), fixed = NamedTuple())

        θ_natural_nt = (σ = 2.0,)
        log_p_nat = logpdf_prior(θ_natural_nt, spec_nat)

        # Should match NaturalHyperparameters dispatch
        θ_n = NaturalHyperparameters([2.0], spec_nat)
        expected_nat = logpdf_prior(θ_n)
        @test log_p_nat ≈ expected_nat atol = 1.0e-10

        # Prior specified in working space
        hp_σ_work = Hyperparameter(Normal(0, 1), transform = elementwise(log), prior_space = :working)
        spec_work = HyperparameterSpec(free = (σ = hp_σ_work,), fixed = NamedTuple())

        θ_natural_nt2 = (σ = 2.0,)
        log_p_work = logpdf_prior(θ_natural_nt2, spec_work)

        # Should match NaturalHyperparameters dispatch
        θ_n2 = NaturalHyperparameters([2.0], spec_work)
        expected_work = logpdf_prior(θ_n2)
        @test log_p_work ≈ expected_work atol = 1.0e-10
    end

    @testset "Multiple parameters" begin
        hp_σ = Hyperparameter(Exponential(1.0), transform = elementwise(log), prior_space = :natural)
        hp_ρ = Hyperparameter(Beta(2, 2), transform = Bijectors.Logit(0.0, 1.0), prior_space = :natural)
        spec = HyperparameterSpec(free = (σ = hp_σ, ρ = hp_ρ), fixed = NamedTuple())

        θ_w = WorkingHyperparameters([log(2.0), Bijectors.Logit(0.0, 1.0)(0.7)], spec)
        log_p_w = logpdf_prior(θ_w)

        # Should be sum of individual log priors in working space
        @test isfinite(log_p_w)

        θ_n = NaturalHyperparameters([2.0, 0.7], spec)
        log_p_n = logpdf_prior(θ_n)

        # Should equal working space prior plus Jacobian
        expected_n = logpdf_prior(convert(WorkingHyperparameters, θ_n)) + logdetjac(θ_n)
        @test log_p_n ≈ expected_n atol = 1.0e-10
    end

    @testset "Identity transform" begin
        # Identity transform with working space
        hp_μ_work = Hyperparameter(Normal(0, 10), transform = identity, prior_space = :working)
        spec_work = HyperparameterSpec(free = (μ = hp_μ_work,), fixed = NamedTuple())

        θ_w = WorkingHyperparameters([5.0], spec_work)
        log_p_w = logpdf_prior(θ_w)
        expected_w = logpdf(Normal(0, 10), 5.0)
        @test log_p_w ≈ expected_w atol = 1.0e-10

        θ_n = NaturalHyperparameters([5.0], spec_work)
        log_p_n = logpdf_prior(θ_n)
        # With identity, Jacobian is 0, so should match
        @test log_p_n ≈ expected_w atol = 1.0e-10

        # Identity transform with natural space
        hp_μ_nat = Hyperparameter(Normal(0, 10), transform = identity, prior_space = :natural)
        spec_nat = HyperparameterSpec(free = (μ = hp_μ_nat,), fixed = NamedTuple())

        θ_w2 = WorkingHyperparameters([5.0], spec_nat)
        log_p_w2 = logpdf_prior(θ_w2)
        @test log_p_w2 ≈ expected_w atol = 1.0e-10

        θ_n2 = NaturalHyperparameters([5.0], spec_nat)
        log_p_n2 = logpdf_prior(θ_n2)
        @test log_p_n2 ≈ expected_w atol = 1.0e-10
    end

    @testset "Type stability" begin
        hp_σ = Hyperparameter(Exponential(1.0), transform = elementwise(log), prior_space = :natural)
        spec = HyperparameterSpec(free = (σ = hp_σ,), fixed = NamedTuple())

        θ_w = WorkingHyperparameters([log(2.0)], spec)
        θ_n = NaturalHyperparameters([2.0], spec)

        @test @inferred(logpdf_prior(θ_w)) isa Float64
        @test @inferred(logpdf_prior(θ_n)) isa Float64
        @test @inferred(logpdf_prior((σ = 2.0,), spec)) isa Float64
    end

end
