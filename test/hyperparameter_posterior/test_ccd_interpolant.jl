using Test
using IntegratedNestedLaplace
using Distributions
using LinearAlgebra
using SparseArrays

@testset "CCDInterpolant" begin

    # Helper: create a ReparameterizationTransform from known Hessian
    function make_transform(θ_star_vec, H; spec = nothing)
        if spec === nothing
            spec = @hyperparams begin
                (θ1 ~ Normal(0, 10), transform = identity, space = working)
                (θ2 ~ Normal(0, 10), transform = identity, space = working)
            end
        end
        θ_star = WorkingHyperparameters(θ_star_vec, spec)
        eigen_result = eigen(H)
        V = eigen_result.vectors
        Λ_inv_sqrt = Diagonal(1.0 ./ sqrt.(eigen_result.values))
        return ReparameterizationTransform(θ_star, V, Λ_inv_sqrt, H)
    end

    @testset "Evaluate: Known Standard Gaussian" begin
        # CCD interpolant with sigma_corr = 1 should match standard Gaussian in z-space:
        # log p(z) = c - 0.5 * ||z||^2
        H = [1.0 0.0; 0.0 1.0]  # Identity Hessian
        transform = make_transform([0.0, 0.0], H)

        interp = CCDInterpolant(
            0.0,               # mode_log_density
            [1.0, 1.0],        # sigma_corr_plus
            [1.0, 1.0],        # sigma_corr_minus
            transform,
            inv(H)
        )

        # At z = [0, 0]: log p = 0.0
        @test interp(zeros(2)) ≈ 0.0

        # At z = [1, 0]: log p = -0.5
        @test interp([1.0, 0.0]) ≈ -0.5

        # At z = [0, 1]: log p = -0.5
        @test interp([0.0, 1.0]) ≈ -0.5

        # At z = [1, 1]: log p = -1.0
        @test interp([1.0, 1.0]) ≈ -1.0

        # At z = [2, 0]: log p = -2.0
        @test interp([2.0, 0.0]) ≈ -2.0
    end

    @testset "Evaluate: Asymmetric Skewness Corrections" begin
        # Different sigma_corr for positive and negative z directions
        H = [1.0 0.0; 0.0 1.0]
        transform = make_transform([0.0, 0.0], H)

        interp = CCDInterpolant(
            0.0,
            [2.0, 1.0],        # sigma_corr_plus: wider in dim 1 positive
            [0.5, 1.0],        # sigma_corr_minus: narrower in dim 1 negative
            transform,
            inv(H)
        )

        # At z = [0, 0]: log p = 0.0
        @test interp(zeros(2)) ≈ 0.0

        # Positive z1: uses sigma_corr_plus[1] = 2.0
        # log p = -0.5 * (1.0)^2 / (2.0)^2 = -0.5 * 0.25 = -0.125
        @test interp([1.0, 0.0]) ≈ -0.125

        # Negative z1: uses sigma_corr_minus[1] = 0.5
        # log p = -0.5 * (-1.0)^2 / (0.5)^2 = -0.5 * 4.0 = -2.0
        @test interp([-1.0, 0.0]) ≈ -2.0

        # Symmetric in dim 2 (sigma = 1.0 both ways)
        @test interp([0.0, 1.0]) ≈ interp([0.0, -1.0])
    end

    @testset "Evaluate: Non-zero Mode Log-density" begin
        H = [1.0 0.0; 0.0 1.0]
        transform = make_transform([0.0, 0.0], H)

        mode_logp = -5.0
        interp = CCDInterpolant(mode_logp, [1.0, 1.0], [1.0, 1.0], transform, inv(H))

        # At z=0, should return mode_log_density
        @test interp(zeros(2)) ≈ mode_logp

        # At z=[1,0], should return mode_logp - 0.5
        @test interp([1.0, 0.0]) ≈ mode_logp - 0.5
    end

    @testset "Profile Marginal: Diagonal Hessian" begin
        # For a diagonal Hessian (no correlation), profiling along theta_k
        # should give a pure quadratic profile (since conditional mode of other
        # dims is always theta*)
        H = [4.0 0.0; 0.0 1.0]  # Different curvatures
        transform = make_transform([1.0, 2.0], H)

        interp = CCDInterpolant(0.0, [1.0, 1.0], [1.0, 1.0], transform, inv(H))

        bounds = [-2.0 4.0; -1.0 5.0]  # [n_dim x 2] integration bounds

        # Profile along dim 1
        θ_grid, log_profile = profile_marginal(interp, 1, 50, bounds)

        # Grid should span the bounds for dim 1
        @test θ_grid[1] ≈ bounds[1, 1]
        @test θ_grid[end] ≈ bounds[1, 2]
        @test length(θ_grid) == 50

        # Profile should be maximal at the mode theta*[1] = 1.0
        mode_idx = argmax(log_profile)
        @test θ_grid[mode_idx] ≈ 1.0 atol = 0.2

        # Profile should be symmetric around the mode (sigma_corr = 1)
        # Find log-density at equal distances from mode
        half_n = div(50, 2)
        # Due to discrete grid, just check that profile decreases away from mode
        @test log_profile[mode_idx] > log_profile[1]
        @test log_profile[mode_idx] > log_profile[end]
    end

    @testset "Profile Marginal: Correlated Hessian" begin
        # Correlated Hessian: eigenvectors are rotated, so z-space doesn't
        # align with theta-space. Profiling should still work correctly.
        H = [2.0 0.5; 0.5 1.0]  # Positive definite, correlated
        @test isposdef(H)  # Verify positive definiteness

        θ_star = [0.0, 0.0]
        transform = make_transform(θ_star, H)
        Σ = inv(H)

        interp = CCDInterpolant(0.0, [1.0, 1.0], [1.0, 1.0], transform, Σ)
        bounds = [-3.0 3.0; -3.0 3.0]

        # Profile along dim 1
        θ_grid_1, log_profile_1 = profile_marginal(interp, 1, 100, bounds)

        # Profile along dim 2
        θ_grid_2, log_profile_2 = profile_marginal(interp, 2, 100, bounds)

        # Both profiles should peak at the mode (0, 0)
        @test θ_grid_1[argmax(log_profile_1)] ≈ 0.0 atol = 0.1
        @test θ_grid_2[argmax(log_profile_2)] ≈ 0.0 atol = 0.1

        # For a correlated Gaussian, the profile variance along theta_k equals
        # Sigma_kk (the marginal variance). The profile is:
        # log p_profile(theta_k) = c - 0.5 * theta_k^2 / Sigma_kk
        # So the curvature at the mode is -1/Sigma_kk

        # For dim 1: Sigma_11 = inv(H)[1,1]
        # For dim 2: Sigma_22 = inv(H)[2,2]
        # Check curvature by looking at drop at theta_k = 1.0
        idx_at_one_dim1 = argmin(abs.(θ_grid_1 .- 1.0))
        expected_drop_dim1 = 0.5 / Σ[1, 1]
        actual_drop_dim1 = log_profile_1[argmax(log_profile_1)] - log_profile_1[idx_at_one_dim1]
        @test actual_drop_dim1 ≈ expected_drop_dim1 atol = 0.15

        idx_at_one_dim2 = argmin(abs.(θ_grid_2 .- 1.0))
        expected_drop_dim2 = 0.5 / Σ[2, 2]
        actual_drop_dim2 = log_profile_2[argmax(log_profile_2)] - log_profile_2[idx_at_one_dim2]
        @test actual_drop_dim2 ≈ expected_drop_dim2 atol = 0.15
    end

    @testset "Profile Marginal: 1D" begin
        # CCD interpolant should also work for 1D (trivial case, no profiling needed)
        H = reshape([4.0], 1, 1)
        spec_1d = @hyperparams begin
            (θ1 ~ Normal(0, 10), transform = identity, space = working)
        end
        θ_star = WorkingHyperparameters([0.0], spec_1d)
        transform = ReparameterizationTransform(θ_star, ones(1, 1), Diagonal([0.5]), H)

        interp = CCDInterpolant(0.0, [1.0], [1.0], transform, inv(H))
        bounds = reshape([-3.0 3.0], 1, 2)

        θ_grid, log_profile = profile_marginal(interp, 1, 50, bounds)
        @test length(θ_grid) == 50
        @test θ_grid[argmax(log_profile)] ≈ 0.0 atol = 0.2
    end

end
