using Test
using IntegratedNestedLaplace
using Distributions
using Bijectors

@testset "@hyperparams Macro" begin

    @testset "Basic syntax with builtin transforms" begin
        # PC priors with log and logit transforms
        spec = @hyperparams begin
            (σ ~ Exponential(1.0), transform = log, space = natural)
            (ρ ~ Beta(2, 2), transform = logit, space = natural)
        end

        @test spec isa HyperparameterSpec
        @test length(keys(spec.free)) == 2
        @test length(keys(spec.fixed)) == 0
        @test keys(spec.free) == (:σ, :ρ)

        # Test that transforms are applied correctly
        hp_σ = spec.free.σ
        hp_ρ = spec.free.ρ
        @test IntegratedNestedLaplace.prior_space(hp_σ) == :natural
        @test IntegratedNestedLaplace.prior_space(hp_ρ) == :natural
        # When space=natural, priors are transformed to working space
        @test hp_σ.prior isa Bijectors.TransformedDistribution
        @test hp_ρ.prior isa Bijectors.TransformedDistribution
    end

    @testset "Call aliases (Log, Logit, Identity)" begin
        spec = @hyperparams begin
            (σ ~ Exponential(1.0), transform = Log(), space = natural)
            (ρ ~ Beta(2, 2), transform = Logit(), space = natural)
            (μ ~ Normal(0, 10), transform = Identity())
        end

        @test length(keys(spec.free)) == 3
        @test keys(spec.free) == (:σ, :ρ, :μ)

        # When space=natural, priors are transformed to working space
        @test spec.free.σ.prior isa Bijectors.TransformedDistribution
        @test spec.free.ρ.prior isa Bijectors.TransformedDistribution
        @test spec.free.μ.prior isa Normal  # Identity doesn't transform (working space by default)
    end

    @testset "Custom bijector expression" begin
        spec = @hyperparams begin
            (τ ~ Gamma(2, 1), transform = elementwise(x -> log(x + 1)), space = natural)
        end

        @test length(keys(spec.free)) == 1
        # When space=natural, priors are transformed to working space
        @test spec.free.τ.prior isa Bijectors.TransformedDistribution
    end

    @testset "Identity transform (default)" begin
        spec = @hyperparams begin
            μ ~ Normal(0, 10)
            σ ~ Gamma(2, 1)
        end

        @test length(keys(spec.free)) == 2
        @test keys(spec.free) == (:μ, :σ)

        # Default is identity transform, natural space (natural = working when transform is identity)
        @test IntegratedNestedLaplace.prior_space(spec.free.μ) == :natural
        @test IntegratedNestedLaplace.prior_space(spec.free.σ) == :natural
        @test spec.free.μ.prior isa Normal
        @test spec.free.σ.prior isa Gamma
        @test spec.free.μ.transform === identity
        @test spec.free.σ.transform === identity
    end

    @testset "Mixed free and fixed parameters" begin
        spec = @hyperparams begin
            (σ ~ Exponential(1.0), transform = log, space = natural)
            (ρ ~ Beta(2, 2), transform = logit, space = natural)
            μ = 0.0
            n = 100
        end

        @test length(keys(spec.free)) == 2
        @test length(keys(spec.fixed)) == 2
        @test keys(spec.free) == (:σ, :ρ)
        @test keys(spec.fixed) == (:μ, :n)
        @test spec.fixed.μ == 0.0
        @test spec.fixed.n == 100
    end

    @testset "prior_space keyword (synonym for space)" begin
        spec = @hyperparams begin
            (σ ~ Exponential(1.0), transform = log, prior_space = natural)
        end

        @test length(keys(spec.free)) == 1
        @test IntegratedNestedLaplace.prior_space(spec.free.σ) == :natural
        # When prior_space=natural, priors are transformed to working space
        @test spec.free.σ.prior isa Bijectors.TransformedDistribution
    end

    @testset "Equivalence with manual construction" begin
        # Using macro
        spec_macro = @hyperparams begin
            (σ ~ Exponential(1.0), transform = log, space = natural)
            (ρ ~ Beta(2, 2), transform = logit, space = natural)
            μ = 0.0
        end

        # Manual construction
        spec_manual = HyperparameterSpec(
            free = (
                σ = Hyperparameter(Exponential(1.0), transform = elementwise(log), prior_space = :natural),
                ρ = Hyperparameter(Beta(2, 2), transform = Bijectors.Logit(0.0, 1.0), prior_space = :natural),
            ),
            fixed = (μ = 0.0,)
        )

        # Test that they have the same structure
        @test keys(spec_macro.free) == keys(spec_manual.free)
        @test keys(spec_macro.fixed) == keys(spec_manual.fixed)
        @test spec_macro.fixed.μ == spec_manual.fixed.μ

        # Test that they behave the same
        θ_w = WorkingHyperparameters([log(2.0), Bijectors.Logit(0.0, 1.0)(0.7)], spec_macro)
        θ_w_manual = WorkingHyperparameters([log(2.0), Bijectors.Logit(0.0, 1.0)(0.7)], spec_manual)

        θ_natural_macro = convert(NamedTuple, convert(NaturalHyperparameters, θ_w))
        θ_natural_manual = convert(NamedTuple, convert(NaturalHyperparameters, θ_w_manual))

        @test θ_natural_macro.σ ≈ θ_natural_manual.σ atol = 1.0e-10
        @test θ_natural_macro.ρ ≈ θ_natural_manual.ρ atol = 1.0e-10
        @test θ_natural_macro.μ == θ_natural_manual.μ

        log_p_macro = logpdf_prior(θ_w)
        log_p_manual = logpdf_prior(θ_w_manual)
        @test log_p_macro ≈ log_p_manual atol = 1.0e-10
    end

    @testset "Error handling: Forgot parentheses (options on separate lines)" begin
        # When user writes options on separate lines instead of in parentheses
        # We detect that option names like "transform" or "space" are suspicious as fixed parameters
        try
            @macroexpand @hyperparams begin
                σ ~ Exponential(1.0)
                transform = log
            end
            @test false  # Should not reach here
        catch e
            @test occursin("forget parentheses", e.msg)
            @test occursin("transform", e.msg)
            @test occursin("(param ~ prior, transform = value)", e.msg)
        end

        # Test with "space" option name too
        try
            @macroexpand @hyperparams begin
                σ ~ Exponential(1.0)
                space = natural
            end
            @test false
        catch e
            @test occursin("forget parentheses", e.msg)
            @test occursin("space", e.msg)
        end
    end

    @testset "Error handling: Invalid option" begin
        # Verify the error message lists valid options
        try
            @macroexpand @hyperparams begin
                (σ ~ Exponential(1.0), invalid_option = log)
            end
            @test false  # Should not reach here
        catch e
            @test occursin("unsupported option", e.msg)
            @test occursin("transform", e.msg)
            @test occursin("space", e.msg)
            @test occursin("prior_space", e.msg)
        end
    end

    @testset "Error handling: Duplicate parameters" begin
        # Verify error message mentions duplication
        try
            @macroexpand @hyperparams begin
                (σ ~ Exponential(1.0), transform = log, space = natural)
                σ = 1.0
            end
            @test false
        catch e
            @test occursin("more than once", e.msg)
        end
    end

    @testset "Error handling: No free parameters" begin
        # Verify error message mentions need for free parameters
        try
            @macroexpand @hyperparams begin
                μ = 0.0
            end
            @test false
        catch e
            @test occursin("at least one free parameter", e.msg)
        end
    end

    @testset "Error handling: Invalid prior space" begin
        # Verify error mentions valid spaces
        try
            @macroexpand @hyperparams begin
                (σ ~ Exponential(1.0), transform = log, space = invalid)
            end
            @test false
        catch e
            @test occursin("natural", e.msg)
            @test occursin("working", e.msg)
        end
    end

    @testset "Error handling: Conflicting space and prior_space" begin
        # Verify error mentions conflict
        try
            @macroexpand @hyperparams begin
                (σ ~ Exponential(1.0), transform = log, space = natural, prior_space = working)
            end
            @test false
        catch e
            @test occursin("conflicting", e.msg)
        end
    end

    @testset "Error handling: Duplicate option specification" begin
        # Verify error mentions duplication
        try
            @macroexpand @hyperparams begin
                (σ ~ Exponential(1.0), transform = log, transform = exp)
            end
            @test false
        catch e
            @test occursin("more than once", e.msg)
        end
    end

    @testset "Functional validation: Transformations work correctly" begin
        spec = @hyperparams begin
            (σ ~ Exponential(1.0), transform = log, space = natural)
            (ρ ~ Beta(2, 2), transform = logit, space = natural)
        end

        # Test transformations
        θ_w = WorkingHyperparameters([log(2.0), Bijectors.Logit(0.0, 1.0)(0.7)], spec)
        θ_natural = convert(NamedTuple, convert(NaturalHyperparameters, θ_w))

        @test θ_natural.σ ≈ 2.0 atol = 1.0e-10
        @test θ_natural.ρ ≈ 0.7 atol = 1.0e-10

        # Test round-trip
        θ_n = NaturalHyperparameters([2.0, 0.7], spec)
        θ_w_back = convert(WorkingHyperparameters, θ_n)
        @test θ_w_back[1] ≈ log(2.0) atol = 1.0e-10
        @test θ_w_back[2] ≈ Bijectors.Logit(0.0, 1.0)(0.7) atol = 1.0e-10
    end

    @testset "Functional validation: Prior evaluation" begin
        spec = @hyperparams begin
            (σ ~ Exponential(1.0), transform = log, space = natural)
        end

        # logpdf_prior evaluates in natural space (no Jacobian needed when space=natural)
        θ_natural = (σ = 2.0,)
        log_p = logpdf_prior(θ_natural, spec)

        # Should equal: log p(σ) in natural space (no Jacobian)
        log_p_expected = logpdf(Exponential(1.0), 2.0)

        @test log_p ≈ log_p_expected atol = 1.0e-10
    end

    @testset "Type stability" begin
        spec = @hyperparams begin
            (σ ~ Exponential(1.0), transform = log, space = natural)
            (ρ ~ Beta(2, 2), transform = logit, space = natural)
            μ = 0.0
        end

        θ_w = WorkingHyperparameters([log(2.0), Bijectors.Logit(0.0, 1.0)(0.7)], spec)
        θ_n = NaturalHyperparameters([2.0, 0.7], spec)

        # Test type stability of key functions with the macro-generated spec
        @test @inferred(convert(NaturalHyperparameters, θ_w)) isa NaturalHyperparameters
        @test @inferred(logpdf_prior(θ_w)) isa Float64
        @test @inferred(logpdf_prior((σ = 2.0, ρ = 0.7), spec)) isa Float64

        result_nt = convert(NamedTuple, θ_w)
        @test result_nt isa NamedTuple
        @test result_nt.σ ≈ log(2.0)
        @test result_nt.ρ ≈ Bijectors.Logit(0.0, 1.0)(0.7)
        @test result_nt.μ == 0.0  # Fixed parameter
    end

    @testset "Documentation examples" begin
        # Example 1: PC priors
        spec1 = @hyperparams begin
            (σ ~ Exponential(1.0), transform = log, space = natural)
            (ρ ~ Beta(2, 2), transform = logit, space = natural)
            μ = 0.0
        end
        @test keys(spec1.free) == (:σ, :ρ)
        @test spec1.fixed.μ == 0.0

        # Example 2: Call aliases
        spec2 = @hyperparams begin
            (σ ~ Exponential(1.0), transform = Log(), space = natural)
            (ρ ~ Beta(2, 2), transform = Logit(), space = natural)
        end
        @test keys(spec2.free) == (:σ, :ρ)

        # Example 3: Custom bijector
        spec3 = @hyperparams begin
            (τ ~ Gamma(2, 1), transform = elementwise(x -> log(x + 1)), space = natural)
        end
        @test keys(spec3.free) == (:τ,)

        # Example 4: Identity transform (default)
        spec4 = @hyperparams begin
            μ ~ Normal(0, 10)
            σ ~ Gamma(2, 1)
        end
        @test keys(spec4.free) == (:μ, :σ)
        @test spec4.free.μ.transform === identity

        # Example 5: Mixed free and fixed
        spec5 = @hyperparams begin
            (σ ~ Exponential(1.0), transform = log, space = natural)
            (ρ ~ Beta(2, 2), transform = logit, space = natural)
            μ = 0.0
            n = 100
        end
        @test keys(spec5.free) == (:σ, :ρ)
        @test keys(spec5.fixed) == (:μ, :n)

        # Example 6: prior_space keyword
        spec6 = @hyperparams begin
            (σ ~ Exponential(1.0), transform = log, prior_space = natural)
        end
        @test IntegratedNestedLaplace.prior_space(spec6.free.σ) == :natural
    end

end
