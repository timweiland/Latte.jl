using Test
using Latte
using GaussianMarkovRandomFields
using Distributions
using LinearAlgebra
using SparseArrays
using Random
using LDLFactorizations
using JLD2
using Statistics

@testset "End-to-End Test: AR-1 Poisson Model (Fast)" begin

    # Load pre-computed MCMC reference data
    reference_file = joinpath(@__DIR__, "reference_data.jld2")

    # The MCMC reference is regenerated on demand (`make generate-reference`)
    # and not committed, so it is absent in CI and fresh clones — skip there.
    if !isfile(reference_file)
        @warn "Reference data not found: $reference_file"
        @warn "Run `make generate-reference` to enable the end-to-end tests"
        @test_skip "Reference data not available"
        return
    end

    @load reference_file y_gt τ_gmrf_log_samples η_samples x_samples model_params

    # Extract model parameters
    k = model_params.k
    σ_gmrf_true = model_params.σ_gmrf_true
    ρ_true = model_params.ρ_true
    τ_gmrf_log_true = model_params.τ_gmrf_log_true
    η_true = model_params.η_true
    desired_std_dev = model_params.desired_std_dev

    # Set same seed for reproducibility
    Random.seed!(model_params.seed)

    # AR-1 precision matrix function
    function ar_precision(ρ, k)
        return spdiagm(-1 => -ρ * ones(k - 1), 0 => ones(k) .+ ρ^2, 1 => -ρ * ones(k - 1))
    end

    # Model setup (same as reference generation, using new API)
    spec = @hyperparams begin
        (τ_gmrf ~ Normal(0, 1), transform = log, space = working)
        (η ~ Normal(atanh(0.95), desired_std_dev), transform = identity, space = working)
    end

    function latent_gmrf(; τ_gmrf, η, kwargs...)
        ρ = tanh(η)
        Q = ar_precision(ρ, k) .* τ_gmrf
        μ₀ = log(1000.0)
        μ = μ₀ .* [ρ^i for i in 1:k]
        return (μ, Q)
    end
    obs_model = ExponentialFamily(Poisson)
    model = LatentGaussianModel(spec, FunctionLatentModel(latent_gmrf, k), obs_model)

    # Run INLA inference (fast!)
    inla_start_time = time()
    inla_result = inla(
        model,
        y_gt,
        progress = false,
        latent_marginalization_method = LaplaceMarginal(),
        hyperparameter_marginalization_method = AutoHyperparameterMarginal()
    )
    inla_time = time() - inla_start_time

    @testset "Reference Data Validation" begin
        @test length(y_gt) == k
        @test length(τ_gmrf_log_samples) > 1000  # Should have many samples
        @test length(η_samples) > 1000
        @test size(x_samples, 2) == k
        @test size(x_samples, 1) == length(τ_gmrf_log_samples)
    end

    @testset "INLA Result Structure" begin
        @test isa(inla_result, INLAResult)
        @test length(inla_result.hyperparameter_marginals) == 2
        @test isa(inla_result.latent_marginals, Vector{WeightedMixture})
        @test length(inla_result.latent_marginals) == k
        @test length(inla_result.hyperparameter_mode) == 2
        @test inla_result.convergence.mode_converged == true
    end

    @testset "Statistical Comparison" begin
        # MCMC samples are in working space (log scale for τ_gmrf, direct η)
        # Transform to natural space for comparison with INLA marginals (which are now in natural space)
        τ_gmrf_samples_natural = exp.(τ_gmrf_log_samples)  # Transform from log(τ) to τ
        η_samples_natural = η_samples  # η has identity transform

        τ_gmrf_marginal = inla_result.hyperparameter_marginals.τ_gmrf
        η_marginal = inla_result.hyperparameter_marginals.η

        # Compare hyperparameter posterior means in natural space
        @test mean(τ_gmrf_marginal) ≈ mean(τ_gmrf_samples_natural) rtol = 0.1
        @test mean(η_marginal) ≈ mean(η_samples) rtol = 0.1

        # Compare hyperparameter credible interval bounds in natural space
        inla_τ_ci = quantile(τ_gmrf_marginal, [0.025, 0.975])
        inla_η_ci = quantile(η_marginal, [0.025, 0.975])
        mcmc_τ_ci = quantile(τ_gmrf_samples_natural, [0.025, 0.975])
        mcmc_η_ci = quantile(η_samples_natural, [0.025, 0.975])

        @test inla_τ_ci[1] ≈ mcmc_τ_ci[1] rtol = 0.2  # Lower bound
        @test inla_τ_ci[2] ≈ mcmc_τ_ci[2] rtol = 0.2  # Upper bound
        @test inla_η_ci[1] ≈ mcmc_η_ci[1] rtol = 0.2
        @test inla_η_ci[2] ≈ mcmc_η_ci[2] rtol = 0.2

        # Compare latent field marginals (test a subset for speed)
        test_indices = [1, 10, 50, 100, 150, 200]  # Sample across the field
        for i in test_indices
            inla_latent_marginal = inla_result.latent_marginals[i]
            mcmc_latent_samples = x_samples[:, i]

            # Compare means
            @test mean(inla_latent_marginal) ≈ mean(mcmc_latent_samples) rtol = 0.15 atol = 0.1

            # Compare standard deviations
            @test std(inla_latent_marginal) ≈ std(mcmc_latent_samples) rtol = 0.2 atol = 0.1

            # Compare quantiles
            inla_latent_ci = quantile(inla_latent_marginal, [0.025, 0.975])
            mcmc_latent_ci = quantile(mcmc_latent_samples, [0.025, 0.975])
            @test inla_latent_ci[1] ≈ mcmc_latent_ci[1] rtol = 0.25 atol = 0.1
            @test inla_latent_ci[2] ≈ mcmc_latent_ci[2] rtol = 0.25 atol = 0.1
        end
    end

    # Shared MCMC log-likelihood matrix for WAIC, DIC, and CPO comparisons
    # Poisson log-link: log p(y_i | x_i) = y_i * x_i - exp(x_i) - log(y_i!)
    n_samples = size(x_samples, 1)
    n_obs = length(y_gt)

    ll_matrix = Matrix{Float64}(undef, n_samples, n_obs)
    for s in 1:n_samples, i in 1:n_obs
        ll_matrix[s, i] = logpdf(Poisson(exp(x_samples[s, i])), y_gt[i])
    end

    @testset "WAIC and DIC vs MCMC" begin
        # MCMC lppd: Σ_i log(mean_s(exp(ll_si)))
        mcmc_lppd = sum(
            let col = @view(ll_matrix[:, i])
                    m = maximum(col)
                    m + log(mean(exp.(col .- m)))
            end for i in 1:n_obs
        )

        # MCMC p_WAIC1: 2 * (lppd - Σ_i mean_s(ll_si))
        mcmc_mean_ll = sum(mean(@view(ll_matrix[:, i])) for i in 1:n_obs)
        mcmc_p_waic1 = 2.0 * (mcmc_lppd - mcmc_mean_ll)
        mcmc_waic = -2 * (mcmc_lppd - mcmc_p_waic1)

        # MCMC DIC: D_bar + p_D  where D_bar = mean(-2*ll), D_mode ≈ min deviance
        mcmc_deviances = [-2 * sum(ll_matrix[s, :]) for s in 1:n_samples]
        mcmc_D_bar = mean(mcmc_deviances)
        mcmc_D_mode = minimum(mcmc_deviances)  # Approximate mode deviance
        mcmc_p_D = mcmc_D_bar - mcmc_D_mode
        mcmc_dic = mcmc_D_bar + mcmc_p_D

        # Extract INLA accumulators
        dic_acc = inla_result.accumulators[1]
        waic_acc = inla_result.accumulators[3]

        # WAIC comparison (rtol=0.15 accounts for GA and grid approximation)
        @test waic_acc.lppd ≈ mcmc_lppd rtol = 0.15
        @test waic_acc.p_WAIC ≈ mcmc_p_waic1 rtol = 0.3
        @test waic_acc.WAIC ≈ mcmc_waic rtol = 0.15

        # DIC comparison (looser tolerance since D_mode from MCMC is approximate)
        @test dic_acc.D_bar ≈ mcmc_D_bar rtol = 0.15
        @test dic_acc.DIC ≈ mcmc_dic rtol = 0.3
    end

    @testset "CPO and PIT vs MCMC" begin
        # MCMC CPO_i = 1 / mean_s(1/p(y_i|x_s_i)) = 1 / mean_s(exp(-ll_s_i))
        mcmc_log_cpo = Vector{Float64}(undef, n_obs)
        for i in 1:n_obs
            neg_ll = -@view(ll_matrix[:, i])
            # Use logsumexp for numerical stability: log(mean(exp(neg_ll)))
            m = maximum(neg_ll)
            log_inv_cpo = m + log(mean(exp.(neg_ll .- m)))
            mcmc_log_cpo[i] = -log_inv_cpo
        end
        mcmc_lpml = sum(mcmc_log_cpo)

        cpo_acc = inla_result.accumulators[4]

        # LPML: INLA is systematically conservative (more negative) due to GA +
        # harmonic mean amplifying variance at AR-1 boundaries. Check direction
        # and bound the gap.
        @test cpo_acc.LPML <= mcmc_lpml
        @test cpo_acc.LPML ≈ mcmc_lpml rtol = 0.15

        # Pointwise log-CPO: mid-field only (indices 41+) where GA is accurate.
        # Boundary indices (1-40) have known excess variance from the GA.
        # atol=0.5 covers typical sites; rtol gives the low-CPO outliers (e.g.
        # the worst-predicted mid-field point) proportional slack, where a fixed
        # absolute tolerance is too tight for an intrinsic GA-vs-exact gap.
        mid_field_indices = [50, 80, 100, 120, 140, 160, 180, 200]
        for i in mid_field_indices
            @test cpo_acc.log_CPO[i] ≈ mcmc_log_cpo[i] atol = 0.5 rtol = 0.12
        end

        # Failure detection: per-observation reliability scores are computed.
        @test length(cpo_acc.failure) == k
        # PSIS-LOO is *supposed* to flag observations whose leave-one-out
        # estimate is unreliable (Pareto k̂ > 0.7) — that is the diagnostic
        # working, not a numerical failure (R-INLA reports the same via
        # cpo$failure). For this AR-1 Poisson the GA overdisperses at the
        # series boundary, so flags should be there, not in the well-
        # approximated mid-field. Verify the diagnostic behaves correctly:
        #   - no genuinely-broken CPO (every CPO finite and > 0);
        #   - flags are boundary-concentrated (more in 1-40 than in 41+);
        #   - the unreliable fraction stays modest.
        @test all(i -> isfinite(cpo_acc.CPO[i]) && cpo_acc.CPO[i] > 0, 1:k)
        flagged = findall(>(0), cpo_acc.failure)
        @test !isempty(flagged)
        @test count(<=(40), flagged) > count(>(40), flagged)
        @test cpo_acc.n_failures < 0.25 * k

        # PIT structural properties
        @test all(0 .<= cpo_acc.PIT .<= 1)
        @test 0.3 < mean(cpo_acc.PIT) < 0.7

        # Mid-field PIT should be closer to uniform than boundary PIT
        # (boundary GA overdispersion compresses PIT toward 0.5)
        boundary_pit_std = std(cpo_acc.PIT[1:40])
        midfield_pit_std = std(cpo_acc.PIT[41:end])
        @test midfield_pit_std > boundary_pit_std
    end

    @testset "Model Properties" begin
        # Verify nonlinear model handling
        @test isa(inla_result.model.observation_model, ExponentialFamily{Poisson})

        # Latent field should give reasonable Poisson rates
        latent_means = [mean(m) for m in inla_result.latent_marginals[1:10]]
        poisson_rates = exp.(latent_means)
        @test all(0 < r < 10000 for r in poisson_rates)
    end

    @testset "Performance" begin
        @test inla_time < 60.0  # Should be very fast without MCMC
        @test inla_time < model_params.mcmc_time  # Should be faster than reference MCMC
        @test model_params.mcmc_time / inla_time > 5.0  # INLA should be much faster
    end

    @testset "Error Handling" begin
        @test_throws ArgumentError inla(model, Float64[])
        @test_throws ArgumentError inla(model, y_gt, latent_indices = Int[])
        @test_throws ArgumentError inla(model, y_gt, latent_indices = [k + 1])
    end

    @testset "Reference Data Quality" begin
        # Verify reference data makes sense
        @test all(y_gt .>= 0)  # Poisson observations should be non-negative
        @test all(isfinite.(τ_gmrf_log_samples))  # MCMC samples should be finite
        @test all(isfinite.(η_samples))
        @test all(isfinite.(x_samples))

        # Check MCMC sample quality
        @test length(unique(τ_gmrf_log_samples)) > 100  # Should have good mixing
        @test length(unique(η_samples)) > 100

        # Parameters should be in reasonable ranges
        @test all(-10 .< τ_gmrf_log_samples .< 10)  # Log precision should be reasonable
        @test all(-5 .< η_samples .< 5)  # atanh(ρ) should be reasonable
    end

    @testset "SimplifiedLaplace variant" begin
        # Run same model with SimplifiedLaplace
        inla_result_sl = inla(
            model,
            y_gt,
            progress = false,
            latent_marginalization_method = SimplifiedLaplace(),
            hyperparameter_marginalization_method = AutoHyperparameterMarginal(),
        )

        @test isa(inla_result_sl, INLAResult)
        @test length(inla_result_sl.latent_marginals) == k
        @test inla_result_sl.convergence.mode_converged == true

        # Compare against MCMC reference (same tolerances as LaplaceMarginal)
        test_indices = [10, 50, 100, 150, 200]
        for i in test_indices
            mcmc_samples = x_samples[:, i]
            sl_m = inla_result_sl.latent_marginals[i]

            @test mean(sl_m) ≈ mean(mcmc_samples) rtol = 0.15 atol = 0.1
            @test std(sl_m) ≈ std(mcmc_samples) rtol = 0.2 atol = 0.1
        end

        # Cross-check: SimplifiedLaplace vs LaplaceMarginal should agree closely
        for i in test_indices
            sl_m = inla_result_sl.latent_marginals[i]
            la_m = inla_result.latent_marginals[i]

            @test mean(sl_m) ≈ mean(la_m) rtol = 0.05 atol = 0.05
            @test std(sl_m) ≈ std(la_m) rtol = 0.1 atol = 0.05
        end
    end
end
