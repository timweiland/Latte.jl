using Test
using Latte
using Latte: compute_point_summary, accumulate!, finalize!,
    DICPointSummary, WAICPointSummary, CPOPointSummary,
    _waic_pointwise_integrals, _cpo_pointwise_integrals
using GaussianMarkovRandomFields
using Distributions
using SparseArrays
using LinearAlgebra

@testset "compute_point_summary" begin

    @testset "DICPointSummary" begin
        acc = DICAccumulator()
        summary = compute_point_summary(acc; total_loglikelihood = -10.5, ga = nothing, obs_lik = nothing)
        @test summary isa DICPointSummary
        @test summary.deviance == 21.0  # -2 * -10.5

        # Two-step path matches one-step
        acc_twostep = DICAccumulator()
        accumulate!(acc_twostep, summary; is_mode = true)
        @test acc_twostep.deviances == [21.0]
        @test acc_twostep.mode_deviance == 21.0

        acc_onestep = DICAccumulator()
        accumulate!(acc_onestep; total_loglikelihood = -10.5, is_mode = true)
        @test acc_onestep.deviances == acc_twostep.deviances
        @test acc_onestep.mode_deviance == acc_twostep.mode_deviance
    end

    @testset "WAICPointSummary" begin
        # Create a simple GA and observation likelihood
        n = 5
        μ = randn(n)
        v = rand(n) .+ 0.1
        Q = Diagonal(1.0 ./ v)
        ga = GMRF(μ, Q)
        σ_obs = 1.5
        y = randn(n)
        obs_lik = ExponentialFamily(Normal)(y; σ = σ_obs)

        acc = WAICAccumulator()
        summary = compute_point_summary(acc; ga = ga, obs_lik = obs_lik)
        @test summary isa WAICPointSummary
        @test length(summary.integrated_ll) == n
        @test length(summary.expected_log_ll) == n

        # Two-step path: compute_point_summary → accumulate!
        acc_twostep = WAICAccumulator()
        accumulate!(acc_twostep, summary)

        # One-step path: original accumulate!
        acc_onestep = WAICAccumulator()
        accumulate!(acc_onestep; ga = ga, obs_lik = obs_lik)

        # Must be identical
        @test acc_twostep.integrated_lls == acc_onestep.integrated_lls
        @test acc_twostep.expected_log_lls == acc_onestep.expected_log_lls
    end

    @testset "CPOPointSummary" begin
        n = 5
        μ = randn(n)
        v = rand(n) .+ 0.1
        Q = Diagonal(1.0 ./ v)
        ga = GMRF(μ, Q)
        σ_obs = 1.5
        y = randn(n)
        obs_lik = ExponentialFamily(Normal)(y; σ = σ_obs)

        acc = CPOAccumulator()
        summary = compute_point_summary(acc; ga = ga, obs_lik = obs_lik)
        @test summary isa CPOPointSummary
        @test length(summary.log_inv_lik_exp) == n
        @test length(summary.pit_exp) == n
        @test length(summary.pareto_k) == n

        # Two-step vs one-step
        acc_twostep = CPOAccumulator()
        accumulate!(acc_twostep, summary)

        acc_onestep = CPOAccumulator()
        accumulate!(acc_onestep; ga = ga, obs_lik = obs_lik)

        @test acc_twostep.log_inv_lik_expectations == acc_onestep.log_inv_lik_expectations
        @test acc_twostep.pit_expectations == acc_onestep.pit_expectations
        # NaN-aware equality (analytic paths emit NaN k̂ values).
        @test isequal(acc_twostep.pareto_k, acc_onestep.pareto_k)
    end

    @testset "CPOPointSummary without PIT" begin
        n = 5
        μ = randn(n)
        v = rand(n) .+ 0.1
        Q = Diagonal(1.0 ./ v)
        ga = GMRF(μ, Q)
        σ_obs = 1.5
        y = randn(n)
        obs_lik = ExponentialFamily(Normal)(y; σ = σ_obs)

        acc = CPOAccumulator(compute_pit = false)
        summary = compute_point_summary(acc; ga = ga, obs_lik = obs_lik)
        @test summary isa CPOPointSummary
        # pit_exp should be empty when compute_pit=false
        @test isempty(summary.pit_exp)
    end

    @testset "MarginalLogLikelihoodAccumulator" begin
        acc = MarginalLogLikelihoodAccumulator()
        summary = compute_point_summary(acc; ga = nothing, obs_lik = nothing)
        @test summary === nothing
    end

    @testset "Tuple map preserves types" begin
        # Verify that map over an accumulator Tuple returns typed summaries
        n = 5
        μ = randn(n)
        v = rand(n) .+ 0.1
        Q = Diagonal(1.0 ./ v)
        ga = GMRF(μ, Q)
        σ_obs = 1.5
        y = randn(n)
        obs_lik = ExponentialFamily(Normal)(y; σ = σ_obs)

        accumulators = (DICAccumulator(), MarginalLogLikelihoodAccumulator(), WAICAccumulator(), CPOAccumulator())

        kwargs = (total_loglikelihood = -10.5, ga = ga, obs_lik = obs_lik)
        summaries = map(accumulators) do acc
            compute_point_summary(acc; kwargs...)
        end

        @test summaries isa Tuple
        @test summaries[1] isa DICPointSummary
        @test summaries[2] === nothing
        @test summaries[3] isa WAICPointSummary
        @test summaries[4] isa CPOPointSummary

        # Zip and accumulate
        for (acc, summary) in zip(accumulators, summaries)
            if summary !== nothing
                accumulate!(acc, summary; is_mode = false)
            end
        end

        @test length(accumulators[1].deviances) == 1
        @test length(accumulators[3].integrated_lls) == 1
        @test length(accumulators[4].log_inv_lik_expectations) == 1
    end
end
