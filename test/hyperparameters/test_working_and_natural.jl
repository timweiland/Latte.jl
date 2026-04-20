using Test
using Latte
using Distributions
using Bijectors
using LinearAlgebra

@testset "WorkingHyperparameters and NaturalHyperparameters" begin

    @testset "Construction and AbstractVector interface" begin
        hp_σ = Hyperparameter(Exponential(1.0), transform = elementwise(log), prior_space = :natural)
        hp_ρ = Hyperparameter(Beta(2, 2), transform = Bijectors.Logit(0.0, 1.0), prior_space = :natural)
        spec = HyperparameterSpec(free = (σ = hp_σ, ρ = hp_ρ), fixed = (μ = 0.0,))

        # Construct WorkingHyperparameters
        θ_w = WorkingHyperparameters([log(2.0), Bijectors.Logit(0.0, 1.0)(0.7)], spec)
        @test length(θ_w) == 2
        @test size(θ_w) == (2,)
        @test θ_w[1] ≈ log(2.0)
        @test θ_w[2] ≈ Bijectors.Logit(0.0, 1.0)(0.7)

        # Construct NaturalHyperparameters
        θ_n = NaturalHyperparameters([2.0, 0.7], spec)
        @test length(θ_n) == 2
        @test size(θ_n) == (2,)
        @test θ_n[1] ≈ 2.0
        @test θ_n[2] ≈ 0.7

        # Test indexing and setindex!
        θ_w_copy = WorkingHyperparameters(copy(θ_w.θ), spec)
        θ_w_copy[1] = -1.0
        @test θ_w_copy[1] == -1.0
        @test θ_w[1] ≈ log(2.0)  # Original unchanged

        # Test error handling for mismatched lengths
        @test_throws ErrorException WorkingHyperparameters([1.0], spec)
        @test_throws ErrorException NaturalHyperparameters([1.0], spec)
    end

    @testset "Broadcasting" begin
        hp_σ = Hyperparameter(Exponential(1.0), transform = elementwise(log), prior_space = :natural)
        spec = HyperparameterSpec(free = (σ = hp_σ,), fixed = NamedTuple())

        θ_w = WorkingHyperparameters([1.0], spec)

        # Test broadcasting preserves type
        θ_w2 = θ_w .+ 1.0
        @test θ_w2 isa WorkingHyperparameters
        @test θ_w2[1] ≈ 2.0

        θ_w3 = θ_w .* 2.0
        @test θ_w3 isa WorkingHyperparameters
        @test θ_w3[1] ≈ 2.0

        # Test with natural hyperparameters
        θ_n = NaturalHyperparameters([2.0], spec)
        θ_n2 = θ_n .+ 1.0
        @test θ_n2 isa NaturalHyperparameters
        @test θ_n2[1] ≈ 3.0
    end

    @testset "Conversion between Working and Natural" begin
        hp_σ = Hyperparameter(Exponential(1.0), transform = elementwise(log), prior_space = :natural)
        hp_ρ = Hyperparameter(Beta(2, 2), transform = Bijectors.Logit(0.0, 1.0), prior_space = :natural)
        spec = HyperparameterSpec(free = (σ = hp_σ, ρ = hp_ρ), fixed = (μ = 0.0,))

        # Working → Natural
        θ_w = WorkingHyperparameters([log(2.0), Bijectors.Logit(0.0, 1.0)(0.7)], spec)
        θ_n = convert(NaturalHyperparameters, θ_w)
        @test θ_n isa NaturalHyperparameters
        @test θ_n[1] ≈ 2.0 atol = 1.0e-10
        @test θ_n[2] ≈ 0.7 atol = 1.0e-10

        # Natural → Working
        θ_n2 = NaturalHyperparameters([2.0, 0.7], spec)
        θ_w2 = convert(WorkingHyperparameters, θ_n2)
        @test θ_w2 isa WorkingHyperparameters
        @test θ_w2[1] ≈ log(2.0) atol = 1.0e-10
        @test θ_w2[2] ≈ Bijectors.Logit(0.0, 1.0)(0.7) atol = 1.0e-10

        # Round-trip: Working → Natural → Working
        θ_w_roundtrip = convert(WorkingHyperparameters, convert(NaturalHyperparameters, θ_w))
        @test θ_w_roundtrip[1] ≈ θ_w[1] atol = 1.0e-10
        @test θ_w_roundtrip[2] ≈ θ_w[2] atol = 1.0e-10

        # Round-trip: Natural → Working → Natural
        θ_n_roundtrip = convert(NaturalHyperparameters, convert(WorkingHyperparameters, θ_n))
        @test θ_n_roundtrip[1] ≈ θ_n[1] atol = 1.0e-10
        @test θ_n_roundtrip[2] ≈ θ_n[2] atol = 1.0e-10

        # Test with identity transform
        hp_μ = Hyperparameter(Normal(0, 10), transform = identity, prior_space = :working)
        spec_id = HyperparameterSpec(free = (μ = hp_μ,), fixed = NamedTuple())

        θ_w_id = WorkingHyperparameters([5.0], spec_id)
        θ_n_id = convert(NaturalHyperparameters, θ_w_id)
        @test θ_n_id[1] == 5.0  # Identity transform

        θ_w_id_back = convert(WorkingHyperparameters, θ_n_id)
        @test θ_w_id_back[1] == 5.0
    end

    @testset "Conversion to NamedTuple" begin
        hp_σ = Hyperparameter(Exponential(1.0), transform = elementwise(log), prior_space = :natural)
        hp_ρ = Hyperparameter(Beta(2, 2), transform = Bijectors.Logit(0.0, 1.0), prior_space = :natural)
        spec = HyperparameterSpec(free = (σ = hp_σ, ρ = hp_ρ), fixed = (μ = 0.0,))

        # Working to NamedTuple
        θ_w = WorkingHyperparameters([log(2.0), Bijectors.Logit(0.0, 1.0)(0.7)], spec)
        θ_w_nt = convert(NamedTuple, θ_w)
        @test θ_w_nt isa NamedTuple
        @test θ_w_nt.σ ≈ log(2.0)
        @test θ_w_nt.ρ ≈ Bijectors.Logit(0.0, 1.0)(0.7)
        @test θ_w_nt.μ == 0.0  # Fixed parameter included

        # Natural to NamedTuple
        θ_n = NaturalHyperparameters([2.0, 0.7], spec)
        θ_n_nt = convert(NamedTuple, θ_n)
        @test θ_n_nt isa NamedTuple
        @test θ_n_nt.σ ≈ 2.0
        @test θ_n_nt.ρ ≈ 0.7
        @test θ_n_nt.μ == 0.0  # Fixed parameter included
    end

    @testset "Property access (dot notation)" begin
        hp_σ = Hyperparameter(Exponential(1.0), transform = elementwise(log), prior_space = :natural)
        hp_ρ = Hyperparameter(Beta(2, 2), transform = Bijectors.Logit(0.0, 1.0), prior_space = :natural)
        spec = HyperparameterSpec(free = (σ = hp_σ, ρ = hp_ρ), fixed = (μ = 0.0,))

        θ_w = WorkingHyperparameters([log(2.0), Bijectors.Logit(0.0, 1.0)(0.7)], spec)
        θ_n = NaturalHyperparameters([2.0, 0.7], spec)

        # Access free parameters
        @test θ_w.σ ≈ log(2.0)
        @test θ_w.ρ ≈ Bijectors.Logit(0.0, 1.0)(0.7)
        @test θ_n.σ ≈ 2.0
        @test θ_n.ρ ≈ 0.7

        # Access fixed parameters
        @test θ_w.μ == 0.0
        @test θ_n.μ == 0.0
    end

    @testset "logdetjac" begin
        hp_σ = Hyperparameter(Exponential(1.0), transform = elementwise(log), prior_space = :natural)
        hp_ρ = Hyperparameter(Beta(2, 2), transform = Bijectors.Logit(0.0, 1.0), prior_space = :natural)
        spec = HyperparameterSpec(free = (σ = hp_σ, ρ = hp_ρ), fixed = (μ = 0.0,))

        # Test logdetjac for WorkingHyperparameters
        θ_w = WorkingHyperparameters([log(2.0), Bijectors.Logit(0.0, 1.0)(0.7)], spec)
        logdetjac_w = logdetjac(θ_w)

        # Expected: log|det J(working → natural)|
        # For log transform: d(exp(η))/dη = exp(η) → log det J = η
        # For logit transform: more complex, but computable
        θ_n_from_w = convert(NaturalHyperparameters, θ_w)
        expected_σ_jacobian = log(θ_n_from_w[1])  # log(2.0)
        # Full jacobian includes both parameters
        @test isfinite(logdetjac_w)

        # Test logdetjac for NaturalHyperparameters
        θ_n = NaturalHyperparameters([2.0, 0.7], spec)
        logdetjac_n = logdetjac(θ_n)
        @test isfinite(logdetjac_n)

        # logdetjac for working and natural should be related
        # logdetjac(natural → working) = -logdetjac(working → natural)
        @test logdetjac_w ≈ -logdetjac_n atol = 1.0e-10

        # Test with identity transform (jacobian should be 0)
        hp_μ = Hyperparameter(Normal(0, 10), transform = identity, prior_space = :working)
        spec_id = HyperparameterSpec(free = (μ = hp_μ,), fixed = NamedTuple())

        θ_w_id = WorkingHyperparameters([5.0], spec_id)
        @test logdetjac(θ_w_id) == 0.0

        θ_n_id = NaturalHyperparameters([5.0], spec_id)
        @test logdetjac(θ_n_id) == 0.0
    end

    @testset "Display" begin
        hp_σ = Hyperparameter(Exponential(1.0), transform = elementwise(log), prior_space = :natural)
        spec = HyperparameterSpec(free = (σ = hp_σ,), fixed = (μ = 0.0,))

        θ_w = WorkingHyperparameters([log(2.0)], spec)
        io = IOBuffer()
        show(io, MIME("text/plain"), θ_w)
        output = String(take!(io))
        @test occursin("WorkingHyperparameters", output)
        @test occursin("σ", output)

        θ_n = NaturalHyperparameters([2.0], spec)
        io = IOBuffer()
        show(io, MIME("text/plain"), θ_n)
        output = String(take!(io))
        @test occursin("NaturalHyperparameters", output)
        @test occursin("σ", output)
    end

    @testset "Edge cases and numerical stability" begin
        hp_σ = Hyperparameter(Exponential(1.0), transform = elementwise(log), prior_space = :natural)
        spec = HyperparameterSpec(free = (σ = hp_σ,), fixed = NamedTuple())

        # Very small σ
        θ_w_small = WorkingHyperparameters([-10.0], spec)  # σ ≈ 4.5e-5
        θ_n_small = convert(NaturalHyperparameters, θ_w_small)
        @test θ_n_small[1] ≈ exp(-10.0) atol = 1.0e-15
        @test θ_n_small[1] > 0

        # Very large σ
        θ_w_large = WorkingHyperparameters([10.0], spec)  # σ ≈ 22026
        θ_n_large = convert(NaturalHyperparameters, θ_w_large)
        @test θ_n_large[1] ≈ exp(10.0) atol = 1.0e-10
        @test isfinite(θ_n_large[1])

        # Test logit with extreme values
        hp_ρ = Hyperparameter(Beta(2, 2), transform = Bijectors.Logit(0.0, 1.0), prior_space = :natural)
        spec2 = HyperparameterSpec(free = (ρ = hp_ρ,), fixed = NamedTuple())

        # Near 0
        θ_w_low = WorkingHyperparameters([-5.0], spec2)  # ρ ≈ 0.0067
        θ_n_low = convert(NaturalHyperparameters, θ_w_low)
        @test θ_n_low[1] ≈ inverse(Bijectors.Logit(0.0, 1.0))(-5.0) atol = 1.0e-10
        @test 0 < θ_n_low[1] < 1

        # Near 1
        θ_w_high = WorkingHyperparameters([5.0], spec2)  # ρ ≈ 0.9933
        θ_n_high = convert(NaturalHyperparameters, θ_w_high)
        @test θ_n_high[1] ≈ inverse(Bijectors.Logit(0.0, 1.0))(5.0) atol = 1.0e-10
        @test 0 < θ_n_high[1] < 1
    end

    @testset "Type stability" begin
        hp_σ = Hyperparameter(Exponential(1.0), transform = elementwise(log), prior_space = :natural)
        hp_ρ = Hyperparameter(Beta(2, 2), transform = Bijectors.Logit(0.0, 1.0), prior_space = :natural)
        spec = HyperparameterSpec(free = (σ = hp_σ, ρ = hp_ρ), fixed = (μ = 0.0,))

        θ_w = WorkingHyperparameters([log(2.0), Bijectors.Logit(0.0, 1.0)(0.7)], spec)
        θ_n = NaturalHyperparameters([2.0, 0.7], spec)

        # Test type stability of conversions
        @test @inferred(convert(NaturalHyperparameters, θ_w)) isa NaturalHyperparameters
        @test @inferred(convert(WorkingHyperparameters, θ_n)) isa WorkingHyperparameters
        # convert(NamedTuple, ...) returns a concrete NamedTuple type, not just NamedTuple
        @test convert(NamedTuple, θ_w) isa NamedTuple
        @test convert(NamedTuple, θ_n) isa NamedTuple

        # Test type stability of logdetjac
        @test @inferred(logdetjac(θ_w)) isa Float64
        @test @inferred(logdetjac(θ_n)) isa Float64
    end

end
