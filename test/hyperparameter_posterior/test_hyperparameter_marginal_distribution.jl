using Test
using IntegratedNestedLaplace
using GaussianMarkovRandomFields
using Distributions
using LinearAlgebra
using StatsFuns
using SparseArrays
using HCubature
using Random
using Bijectors

@testset "HyperparameterMarginalDistribution Tests" begin

    # Setup test data following the ar1_gmrf_2d example
    function setup_test_data()
        # AR-1 precision matrix function
        function ar_precision(ρ, k)
            return spdiagm(-1 => -ρ * ones(k - 1), 0 => ones(k), 1 => -ρ * ones(k - 1))
        end

        # Model parameters (smaller for faster testing)
        k = 100
        σ_gmrf_true = 2.0
        ρ_true = 0.3

        # Hyperparameter prior using new API
        spec = @hyperparams begin
            (σ_gmrf ~ Gamma(2, 3), transform = log, space = natural)
            (ρ ~ Uniform(0, 0.5), transform = logit, space = natural)
            σ = 1.0e-6  # Fixed parameter
        end

        # Function to create latent GMRF
        function latent_gmrf(; σ_gmrf, ρ, kwargs...)
            Q = ar_precision(ρ, k) ./ σ_gmrf^2
            μ = zeros(k)
            return GMRF(μ, Q)
        end

        # Observation model
        obs_model = ExponentialFamily(Normal)

        # Create INLA model
        inla_model = INLAModel(spec, latent_gmrf, obs_model)

        # Generate synthetic data
        Random.seed!(123)
        x_gt = rand(latent_gmrf(; σ_gmrf = σ_gmrf_true, ρ = ρ_true))
        y_gt = rand(conditional_distribution(obs_model, x_gt; σ = spec.fixed.σ))

        return inla_model, y_gt, k
    end

    @testset "Constructor and Basic Properties" begin
        @test HyperparameterMarginalDistribution <: ContinuousUnivariateDistribution
        @test Distributions.partype(HyperparameterMarginalDistribution{Float64}) == Float64
    end

    @testset "Complete Workflow Integration Test" begin
        inla_model, y_gt, k = setup_test_data()

        # Step 1: Find mode
        θ_star, mode_points, mode_logdensities = find_hyperparameter_mode(inla_model, y_gt)

        # Step 2: Explore posterior
        exploration = explore_hyperparameter_posterior(
            inla_model, y_gt, θ_star, GaussianMarginal(), 1:k
        )

        # Step 3: Build interpolant
        posterior_approx = build_posterior_interpolant(exploration)

        # Get spec and free names for use in all tests
        spec = inla_model.hyperparameter_spec
        free_names = collect(keys(spec.free))

        # Create marginal distributions for both dimensions (natural space)
        marginal_1 = HyperparameterMarginalDistribution(posterior_approx, 1; rtol = 1.0e-3, atol = 1.0e-6)
        marginal_2 = HyperparameterMarginalDistribution(posterior_approx, 2; rtol = 1.0e-3, atol = 1.0e-6)

        @testset "Basic Properties" begin
            # Bounds are already in natural space
            natural_bounds_1 = (exploration.integration_bounds[1, 1], exploration.integration_bounds[1, 2])
            natural_bounds_2 = (exploration.integration_bounds[2, 1], exploration.integration_bounds[2, 2])

            @test minimum(marginal_1) ≈ natural_bounds_1[1]
            @test maximum(marginal_1) ≈ natural_bounds_1[2]
            @test minimum(marginal_2) ≈ natural_bounds_2[1]
            @test maximum(marginal_2) ≈ natural_bounds_2[2]

            # Test insupport - θ_star is already in natural space NamedTuple
            @test insupport(marginal_1, θ_star[free_names[1]])
            @test insupport(marginal_2, θ_star[free_names[2]])
            @test !insupport(marginal_1, minimum(marginal_1) - 1)
            @test !insupport(marginal_2, maximum(marginal_2) + 1)
        end

        @testset "PDF/LogPDF Consistency" begin
            # Test several points in natural space (θ_star is already in natural space)
            test_points_1 = [minimum(marginal_1) + 0.1, θ_star[free_names[1]], maximum(marginal_1) - 0.1]
            test_points_2 = [minimum(marginal_2) + 0.01, θ_star[free_names[2]], maximum(marginal_2) - 0.01]

            for x in test_points_1
                @test pdf(marginal_1, x) ≈ exp(logpdf(marginal_1, x)) rtol = 1.0e-10
            end

            for x in test_points_2
                @test pdf(marginal_2, x) ≈ exp(logpdf(marginal_2, x)) rtol = 1.0e-10
            end
        end

        @testset "Moments vs Numerical Integration" begin
            # Test moments against numerical integration using logpdf
            # This is the "double integration" we wanted to avoid in the implementation
            # but it serves as a good test for correctness

            # Get bounds for integration
            a1, b1 = minimum(marginal_1), maximum(marginal_1)

            # Integrate E[θ₁] using logpdf
            mean_integrand(x_vec) = x_vec[1] * exp(logpdf(marginal_1, x_vec[1]))
            true_mean_1, _ = hcubature(mean_integrand, [a1], [b1], rtol = 1.0e-3, atol = 1.0e-6)

            # Integrate E[θ₁²] using logpdf
            second_moment_integrand(x_vec) = x_vec[1]^2 * exp(logpdf(marginal_1, x_vec[1]))
            true_second_moment_1, _ = hcubature(second_moment_integrand, [a1], [b1], rtol = 1.0e-3, atol = 1.0e-6)

            true_var_1 = true_second_moment_1 - true_mean_1^2

            # Compare with distribution methods
            @test mean(marginal_1) ≈ true_mean_1 rtol = 1.0e-2
            @test var(marginal_1) ≈ true_var_1 rtol = 1.0e-2

            # Repeat for dimension 2
            a2, b2 = minimum(marginal_2), maximum(marginal_2)

            mean_integrand_2(x_vec) = x_vec[1] * exp(logpdf(marginal_2, x_vec[1]))
            true_mean_2, _ = hcubature(mean_integrand_2, [a2], [b2], rtol = 1.0e-3, atol = 1.0e-6)

            second_moment_integrand_2(x_vec) = x_vec[1]^2 * exp(logpdf(marginal_2, x_vec[1]))
            true_second_moment_2, _ = hcubature(second_moment_integrand_2, [a2], [b2], rtol = 1.0e-3, atol = 1.0e-6)

            true_var_2 = true_second_moment_2 - true_mean_2^2

            @test mean(marginal_2) ≈ true_mean_2 rtol = 1.0e-2
            @test var(marginal_2) ≈ true_var_2 rtol = 1.0e-2
        end

        @testset "CDF Properties" begin
            # Test boundary conditions
            @test cdf(marginal_1, minimum(marginal_1)) ≈ 0.0 atol = 1.0e-3
            @test cdf(marginal_1, maximum(marginal_1)) ≈ 1.0 atol = 1.0e-3
            @test cdf(marginal_2, minimum(marginal_2)) ≈ 0.0 atol = 1.0e-3
            @test cdf(marginal_2, maximum(marginal_2)) ≈ 1.0 atol = 1.0e-3

            # Test monotonicity
            test_points_1 = range(minimum(marginal_1), maximum(marginal_1), length = 10)
            cdf_values_1 = [cdf(marginal_1, x) for x in test_points_1]

            for i in 2:length(cdf_values_1)
                @test cdf_values_1[i] >= cdf_values_1[i - 1] - 1.0e-6  # Allow small numerical errors
            end

            # Test that CDF integrates the PDF
            mid_point = (minimum(marginal_1) + maximum(marginal_1)) / 2

            # Integrate PDF from minimum to mid_point
            pdf_integrand(x_vec) = exp(logpdf(marginal_1, x_vec[1]))
            integrated_pdf, _ = hcubature(
                pdf_integrand, [minimum(marginal_1)], [mid_point],
                rtol = 1.0e-3, atol = 1.0e-6
            )

            @test cdf(marginal_1, mid_point) ≈ integrated_pdf rtol = 1.0e-2
        end

        @testset "Quantile-CDF Inverse Relationship" begin
            # Test several quantile levels
            test_levels = [0.1, 0.25, 0.5, 0.75, 0.9]

            for q in test_levels
                x1 = quantile(marginal_1, q)
                cdf_x1 = cdf(marginal_1, x1)

                # CDF(quantile(q)) should equal q
                @test cdf_x1 ≈ q rtol = 1.0e-2

                # Test the inverse relationship
                if minimum(marginal_1) < x1 < maximum(marginal_1)
                    q_roundtrip = quantile(marginal_1, cdf_x1)
                    @test q_roundtrip ≈ x1 rtol = 1.0e-2
                end
            end
        end

        @testset "PDF Normalization" begin
            # Test that PDF integrates to 1
            pdf_integrand(x_vec) = pdf(marginal_1, x_vec[1])
            total_mass_1, _ = hcubature(
                pdf_integrand, [minimum(marginal_1)], [maximum(marginal_1)],
                rtol = 1.0e-3, atol = 1.0e-6
            )

            @test total_mass_1 ≈ 1.0 rtol = 1.0e-3
        end

        @testset "Sampling Statistics" begin
            Random.seed!(456)
            n_samples = 1000  # More samples for better statistics
            samples_1 = [rand(marginal_1) for _ in 1:n_samples]

            # Sample mean should approximate true mean (reasonable tolerance for sampling variability)
            sample_mean_1 = sum(samples_1) / n_samples
            true_mean_1 = mean(marginal_1)
            @test sample_mean_1 ≈ true_mean_1 atol = 1.0e-2

            # All samples should be in support
            @test all(minimum(marginal_1) <= s <= maximum(marginal_1) for s in samples_1)

            # Sample variance should be reasonable (moderate tolerance due to sampling variability)
            sample_var_1 = sum((s - sample_mean_1)^2 for s in samples_1) / (n_samples - 1)
            true_var_1 = var(marginal_1)

            @test sample_var_1 ≈ true_var_1 atol = 1.0e-1
        end

        @testset "Caching Behavior" begin

            # Create a fresh marginal distribution to test caching
            fresh_marginal = HyperparameterMarginalDistribution(posterior_approx, 1; rtol = 1.0e-3, atol = 1.0e-6)

            # Test that moments are cached
            @test isnothing(fresh_marginal._moments)

            # First call should compute and cache
            μ1 = mean(fresh_marginal)
            @test !isnothing(fresh_marginal._moments)
            @test fresh_marginal._moments[1] == μ1

            # Second call should use cache
            μ2 = mean(fresh_marginal)
            @test μ2 == μ1

            # Variance should also be cached
            σ²1 = var(fresh_marginal)
            @test fresh_marginal._moments[2] == σ²1
        end

        @testset "Error Handling" begin
            # Test quantile bounds checking
            @test_throws ArgumentError quantile(marginal_1, -0.1)
            @test_throws ArgumentError quantile(marginal_1, 1.1)

            # Test constructor validation
            @test_throws ArgumentError HyperparameterMarginalDistribution(
                posterior_approx, 0  # Invalid dimension
            )
            @test_throws ArgumentError HyperparameterMarginalDistribution(
                posterior_approx, 3  # Too high dimension for 2D problem
            )
        end
    end
end
