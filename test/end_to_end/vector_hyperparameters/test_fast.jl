using Test
using Latte
using GaussianMarkovRandomFields
using Distributions
using LinearAlgebra
using SparseArrays
using Random
using Statistics

# End-to-end inference with a vector-valued hyperparameter (issue #41):
# κ = [log τ₁, log τ₂] carries a joint MvNormal prior; the two blocks of an
# IID latent field use precisions exp(κ[1]) and exp(κ[2]). Observations are
# Normal with fixed σ. The same model factored into two scalar
# hyperparameters with independent Normal priors must give the same
# posterior when the MvNormal covariance is diagonal.

const N_HALF = 3
const N_LAT = 2 * N_HALF

function two_block_latent_vec(; κ, kwargs...)
    Q = spdiagm(0 => [fill(exp(κ[1]), N_HALF); fill(exp(κ[2]), N_HALF)])
    return (zeros(N_LAT), Q)
end

function two_block_latent_scl(; κ1, κ2, kwargs...)
    Q = spdiagm(0 => [fill(exp(κ1), N_HALF); fill(exp(κ2), N_HALF)])
    return (zeros(N_LAT), Q)
end

@testset "Vector hyperparameters end-to-end" begin
    rng = MersenneTwister(41)
    y = [0.8, -0.5, 1.1, -1.4, 2.0, -0.3]

    spec_diag = @hyperparams begin
        κ ~ MvNormal(zeros(2), Diagonal([0.7, 1.3]))
        σ = 0.5
    end
    spec_scl = @hyperparams begin
        κ1 ~ Normal(0.0, sqrt(0.7))
        κ2 ~ Normal(0.0, sqrt(1.3))
        σ = 0.5
    end
    obs = ExponentialFamily(Normal)

    model_vec = LatentGaussianModel(spec_diag, FunctionLatentModel(two_block_latent_vec, N_LAT), obs)
    model_scl = LatentGaussianModel(spec_scl, FunctionLatentModel(two_block_latent_scl, N_LAT), obs)

    @testset "INLA protocol shapes and naming" begin
        r = inla(model_vec, y)

        hms = hyperparameter_marginals(r)
        @test length(hms) == 2

        groups = hyperparameter_groups(r)
        @test collect(keys(groups)) == [:κ]
        @test groups[:κ] == 1:2

        # By-name accessor returns all components of the vector block.
        @test length(hyperparameter_marginals(r, :κ)) == 2

        # Internal storage keyed by expanded per-coordinate names.
        @test collect(keys(r.hyperparameter_marginals)) ==
            [Symbol("κ[1]"), Symbol("κ[2]")]

        # Mode exposes the vector block by name in natural space.
        θ_mode = hyperparameter_mode(r)
        @test length(θ_mode.κ) == 2

        # Joint posterior draws carry all free coordinates.
        s = rand(rng, r, 8)
        @test size(s.θ) == (8, 2)
        @test size(s.x) == (8, N_LAT)

        @test sprint(show, r) isa String
    end

    @testset "Diagonal MvNormal ≡ independent scalar factorization" begin
        r_vec = inla(model_vec, y)
        r_scl = inla(model_scl, y)

        m_vec = hyperparameter_marginals(r_vec)
        m_scl = hyperparameter_marginals(r_scl)
        for i in 1:2
            @test mean(m_vec[i]) ≈ mean(m_scl[i]) rtol = 1.0e-5
            @test std(m_vec[i]) ≈ std(m_scl[i]) rtol = 1.0e-5
        end

        lm_vec = latent_marginals(r_vec)
        lm_scl = latent_marginals(r_scl)
        @test mean.(lm_vec) ≈ mean.(lm_scl) rtol = 1.0e-6 atol = 1.0e-8
        @test std.(lm_vec) ≈ std.(lm_scl) rtol = 1.0e-6
    end

    @testset "Non-diagonal covariance runs and shifts the posterior" begin
        Σ = [0.7 0.6; 0.6 1.3]   # strong prior correlation between the blocks
        spec_corr = @hyperparams begin
            κ ~ MvNormal(zeros(2), Σ)
            σ = 0.5
        end
        model_corr = LatentGaussianModel(
            spec_corr, FunctionLatentModel(two_block_latent_vec, N_LAT), obs,
        )
        r_corr = inla(model_corr, y)
        r_diag = inla(model_vec, y)

        m_corr = hyperparameter_marginals(r_corr)
        @test all(isfinite, mean.(m_corr))
        @test all(isfinite, std.(m_corr))
        # The correlated prior must actually inform the posterior.
        m_diag = hyperparameter_marginals(r_diag)
        @test !isapprox(mean(m_corr[1]), mean(m_diag[1]); atol = 1.0e-8) ||
            !isapprox(mean(m_corr[2]), mean(m_diag[2]); atol = 1.0e-8)
    end

    @testset "TMB with a vector hyperparameter" begin
        r = tmb(model_vec, y)
        @test length(hyperparameter_marginals(r)) == 2
        @test hyperparameter_groups(r)[:κ] == 1:2
        @test length(r.θ_map) == 2
        @test length(hyperparameter_mode(r).κ) == 2
        @test sprint(show, r) isa String
    end

    @testset "HMC-Laplace with a vector hyperparameter" begin
        r = hmc_laplace(model_vec, y; n_samples = 100, n_warmup = 50, rng = MersenneTwister(7))
        @test length(hyperparameter_marginals(r)) == 2
        @test hyperparameter_groups(r)[:κ] == 1:2
        ch = Latte.chain(r)
        @test Symbol("κ[1]") in names(ch) && Symbol("κ[2]") in names(ch)
        @test sprint(show, r) isa String
    end
end
