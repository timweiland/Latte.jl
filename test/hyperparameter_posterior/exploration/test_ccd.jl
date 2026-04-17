using Test
using IntegratedNestedLaplace
using IntegratedNestedLaplace: generate_ccd_points, generate_factorial_points,
    ccd_integration_weights
using GaussianMarkovRandomFields
using Distributions
using SparseArrays
using LinearAlgebra
using Statistics
using Random

@testset "CCD Exploration" begin

    @testset "generate_ccd_points" begin
        f0 = 1.1  # default

        @testset "d=1: center + 2 axial" begin
            points = generate_ccd_points(1)
            @test length(points) == 3
            # Center
            @test [0.0] in points
            # Axial at ±f0*√1 = ±1.1
            radius = f0 * sqrt(1.0)
            @test [radius] in points
            @test [-radius] in points
        end

        @testset "d=2: center + 4 axial + 4 factorial = 9" begin
            points = generate_ccd_points(2)
            @test length(points) == 9
            radius = f0 * sqrt(2.0)
            # Center
            @test [0.0, 0.0] in points
            # All non-center points at distance f0*√d
            for p in points
                r = norm(p)
                if r > 0
                    @test r ≈ radius
                end
            end
        end

        @testset "d=3: 1 + 6 + 8 = 15" begin
            points = generate_ccd_points(3)
            @test length(points) == 15
            radius = f0 * sqrt(3.0)
            @test zeros(3) in points
            # All non-center points at radius f0*√d
            for p in points
                r = norm(p)
                if r > 0
                    @test r ≈ radius
                end
            end
        end

        @testset "d=5: fractional factorial" begin
            points = generate_ccd_points(5)
            # Should be 1 + 10 + 16 = 27
            @test length(points) == 27
            radius = f0 * sqrt(5.0)
            @test zeros(5) in points
            # Axial: 10 points (single nonzero entry)
            axial_pts = filter(p -> count(!iszero, p) == 1, points)
            @test length(axial_pts) == 10
            for p in axial_pts
                @test norm(p) ≈ radius
            end
        end

        @testset "Custom f0" begin
            points = generate_ccd_points(3; f0 = 1.5)
            radius = 1.5 * sqrt(3.0)
            for p in points
                r = norm(p)
                if r > 0
                    @test r ≈ radius
                end
            end
        end

        @testset "Symmetry: for each z, -z is in design" begin
            for d in [1, 2, 3, 4, 5]
                points = generate_ccd_points(d)
                for p in points
                    @test (-p) in points
                end
            end
        end
    end

    @testset "Mathematical properties (Rue et al. 2009, Section 6.5)" begin
        # Verify point counts match Rue et al. 2009 Table:
        # m=3: 1+6+8=15, m=4: 1+8+16=25, m=5: 1+10+16=27
        @testset "Point counts match Rue et al. 2009" begin
            expected_factorial = Dict(3 => 8, 4 => 16, 5 => 16, 6 => 32)
            for (d, n_fac) in expected_factorial
                pts = generate_ccd_points(d)
                expected_total = 1 + 2d + n_fac
                @test length(pts) == expected_total
            end
        end

        @testset "Spherical design: all non-center points at radius f0√d" begin
            for d in 2:5
                f0 = 1.1
                points = generate_ccd_points(d; f0 = f0)
                radius = f0 * sqrt(Float64(d))
                for p in points
                    r = norm(p)
                    if r > 0  # skip center
                        @test r ≈ radius
                    end
                end
            end
        end

        @testset "Second-moment balance: Σ zᵢzⱼ = 0 for i≠j over design" begin
            # For a balanced CCD, the cross-moments should vanish
            for d in 2:5
                points = generate_ccd_points(d)
                for i in 1:d, j in (i + 1):d
                    cross_moment = sum(p[i] * p[j] for p in points)
                    @test abs(cross_moment) < 1.0e-10
                end
            end
        end

        @testset "Second-moment isotropy: Σ zᵢ² equal across dimensions" begin
            # For a balanced CCD, the sum of squared coordinates should
            # be the same for each dimension
            for d in 2:5
                points = generate_ccd_points(d)
                col_sums = [sum(p[i]^2 for p in points) for i in 1:d]
                @test all(s ≈ col_sums[1] for s in col_sums)
            end
        end

        @testset "Analytical weights sum to 1" begin
            for d in 2:6
                n_p = length(generate_ccd_points(d))
                w_sphere, w_center = ccd_integration_weights(n_p, d, 1.1)
                total = w_center + (n_p - 1) * w_sphere
                @test total ≈ 1.0
                @test w_center > 0
                @test w_sphere > 0
            end
        end

        @testset "Analytical weights exact for Gaussian: E[zᵀz] = d" begin
            # The Rue et al. weights are derived so that the CCD quadrature is exact for
            # ∫zᵀz · N(z;0,I)dz = d — the Gaussian density is part of the integrand.
            for d in 2:5
                f0 = 1.1
                points = generate_ccd_points(d; f0 = f0)
                n_p = length(points)
                w_sphere, w_center = ccd_integration_weights(n_p, d, f0)

                # Assign quadrature weights Δ_k
                Δ = [all(iszero, p) ? w_center : w_sphere for p in points]

                # Compute E[zᵀz] = Σ Δ_k · π(z_k) · zᵀz / Σ Δ_k · π(z_k)
                # where π(z) = N(z;0,I) ∝ exp(-½||z||²)
                density = [exp(-0.5 * dot(p, p)) for p in points]
                numerator = sum(Δ[k] * density[k] * dot(points[k], points[k]) for k in 1:n_p)
                denominator = sum(Δ[k] * density[k] for k in 1:n_p)
                expected_ztz = numerator / denominator
                @test expected_ztz ≈ Float64(d) atol = 1.0e-10
            end
        end

        @testset "CCD integration with Rue et al. weights on Gaussian" begin
            # Full test: analytical weights × Gaussian density → correct moments
            for d in 2:4
                f0 = 1.1
                points = generate_ccd_points(d; f0 = f0)
                n_p = length(points)
                w_sphere, w_center = ccd_integration_weights(n_p, d, f0)

                # Weights = Δ_k * π(z_k), normalized
                log_densities = [-0.5 * dot(p, p) for p in points]
                Δ = [all(iszero, p) ? w_center : w_sphere for p in points]
                weighted = Δ .* exp.(log_densities)
                weights = weighted ./ sum(weighted)

                # E[z] = 0 by symmetry
                for i in 1:d
                    mean_i = sum(w * p[i] for (w, p) in zip(weights, points))
                    @test abs(mean_i) < 1.0e-10
                end

                # E[zᵢ²] should be close to 1 and isotropic
                vars = [sum(w * p[i]^2 for (w, p) in zip(weights, points)) for i in 1:d]
                @test all(v ≈ vars[1] for v in vars)
                # With proper weights, variance should be close to 1
                for v in vars
                    @test v > 0.5
                    @test v < 2.0
                end
            end
        end
    end

    @testset "generate_factorial_points" begin
        @testset "d <= 4: full factorial" begin
            for d in 1:4
                pts = generate_factorial_points(d)
                @test length(pts) == 2^d
                # All entries are ±1
                for p in pts
                    @test all(abs.(p) .≈ 1.0)
                end
                # All unique
                @test length(unique(pts)) == length(pts)
            end
        end

        @testset "d=5: fractional factorial" begin
            pts = generate_factorial_points(5)
            @test length(pts) == 16  # 2^(5-1)
            for p in pts
                @test all(abs.(p) .≈ 1.0)
                @test length(p) == 5
            end
            @test length(unique(pts)) == length(pts)
        end
    end

    # Helper: create a 3-hyperparameter model for CCD testing
    # Uses three InverseGamma priors with log transforms for numerical stability
    function make_3hp_model(k = 20)
        spec = @hyperparams begin
            (σ_latent ~ InverseGamma(2, 1), transform = log, space = natural)
            (σ ~ InverseGamma(2, 1), transform = log, space = natural)
            (τ ~ InverseGamma(2, 1), transform = log, space = natural)
        end
        function latent_func(; σ_latent, τ, kwargs...)
            # Precision = (1/σ_latent² + 1/τ²) on diagonal
            prec = 1.0 / σ_latent^2 + 1.0 / τ^2
            Q = spdiagm(0 => fill(prec, k))
            return (zeros(k), Q)
        end
        obs_model = ExponentialFamily(Normal)  # Uses σ hyperparameter
        return INLAModel(spec, FunctionLatentModel(latent_func, k), obs_model), k
    end

    @testset "CCD exploration on 3-HP model" begin
        Random.seed!(42)
        model, k = make_3hp_model(20)

        # Generate data
        σ_latent_true = 1.5
        σ_true = 0.5
        τ_true = 2.0
        x_true = rand(model.latent_prior(; σ_latent = σ_latent_true, σ = σ_true, τ = τ_true))
        y = x_true .+ σ_true .* randn(k)

        # Find mode
        θ_star, _, _ = find_hyperparameter_mode(model, y)

        # Run CCD exploration
        exploration, accs = explore_hyperparameter_posterior(
            CCDExplorationStrategy(),
            model, y, θ_star,
            GaussianMarginal(), collect(1:k);
            accumulators = (DICAccumulator(), WAICAccumulator())
        )

        @test exploration isa HyperparameterExploration
        @test length(exploration.grid_points) > 0
        @test !isempty(exploration.integration_indices)

        # Should have ~15 points for d=3 (some may be filtered if -Inf)
        n_points = length(exploration.grid_points)
        @test n_points <= 15
        @test n_points >= 5  # At minimum center + some axial/factorial

        # All integration points should have marginal results
        for idx in exploration.integration_indices
            @test exploration.grid_points[idx].marginal_result !== nothing
        end

        # Weights should be valid
        integration_points = exploration.grid_points[exploration.integration_indices]
        log_weights = [p.log_density for p in integration_points]
        weights = exp.(log_weights)
        weights ./= sum(weights)
        @test all(isfinite, weights)
        @test all(w -> w >= 0, weights)
        @test sum(weights) ≈ 1.0

        # Latent marginals via create_weighted_mixtures should work
        mixture_result = create_weighted_mixtures(exploration)
        @test length(mixture_result.marginals) == k
        for m in mixture_result.marginals
            @test isfinite(mean(m))
            @test std(m) > 0
        end

        # Accumulators should have been finalized
        dic_acc = accs[1]
        waic_acc = accs[2]
        @test isfinite(dic_acc.DIC)
        @test isfinite(waic_acc.WAIC)
    end

    @testset "inla() with exploration_strategy" begin
        Random.seed!(42)

        # 1-HP model (d=1): auto should use grid
        function make_1hp_model(n)
            spec = @hyperparams begin
                (σ ~ InverseGamma(2, 1), transform = log, space = natural)
            end
            function latent_func(; σ, kwargs...)
                Q = spdiagm(0 => fill(1 / σ^2, n))
                return (zeros(n), Q)
            end
            return INLAModel(spec, FunctionLatentModel(latent_func, n), ExponentialFamily(Normal))
        end

        n = 10
        model_1d = make_1hp_model(n)
        y_1d = randn(n)

        # Default (:auto) on 1D → should use grid (produces many points)
        result_auto = inla(model_1d, y_1d; progress = false)
        @test length(result_auto.exploration.grid_points) > 3  # Grid produces more than CCD's 3

        # Explicit :ccd on 1D → should use CCD (only 3 points)
        result_ccd = inla(model_1d, y_1d; progress = false, exploration_strategy = CCDExplorationStrategy())
        @test length(result_ccd.exploration.grid_points) == 3
        @test all(isfinite(mean(m)) for m in result_ccd.latent_marginals)

        # Explicit grid on 1D → should use grid
        result_grid = inla(model_1d, y_1d; progress = false, exploration_strategy = GridExplorationStrategy())
        @test length(result_grid.exploration.grid_points) > 3
    end

    @testset "inla() CCD end-to-end on 3-HP model" begin
        Random.seed!(42)
        model, k = make_3hp_model(20)

        σ_latent_true = 1.5
        σ_true = 0.5
        τ_true = 2.0
        x_true = rand(model.latent_prior(; σ_latent = σ_latent_true, σ = σ_true, τ = τ_true))
        y = x_true .+ σ_true .* randn(k)

        result = inla(model, y; progress = false, exploration_strategy = CCDExplorationStrategy())

        # Result should be complete
        @test result isa INLAResult
        @test length(result.latent_marginals) == k
        @test all(isfinite(mean(m)) for m in result.latent_marginals)

        # Hyperparameter marginals should exist
        @test length(result.hyperparameter_marginals) == 3

        # Accumulators should be populated
        @test !isempty(result.accumulators)

        # Posterior sampling should work
        samples = rand(MersenneTwister(1), result, 10)
        @test length(samples) == 10
        @test length(samples[1].x) == k
    end
end
