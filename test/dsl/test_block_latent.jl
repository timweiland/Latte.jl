using Test
using Latte
using Distributions, Random, LinearAlgebra
using Statistics: mean
import GaussianMarkovRandomFields as G

# A matrix-slice (multivariate-block) latent `x[:, t] ~ MvNormal(d)` over `n` columns has `n * d`
# scalar entries, not `n`. `variable_length` used to count the `~` sites (`n`), which sized the
# probe seed vector wrong and crashed model construction with a BoundsError. It must count scalars.

@testset "Block (matrix-slice MvNormal) latent dimension" begin
    n, d = 6, 2

    @latte function mvblock_dim(y, n, d)
        log_τ ~ Normal(0.0, 1.0)
        τ = exp(log_τ)
        x = Matrix{Real}(undef, d, n)
        x[:, 1] ~ MvNormal(zeros(d), 1.0 * I(d))
        for t in 2:n
            x[:, t] ~ MvNormal(x[:, t - 1], (1.0 / τ) * I(d))
        end
        for t in 1:n
            y[t] ~ Normal(sum(x[:, t]), 0.1)
        end
    end

    Random.seed!(20260618)
    y = randn(n)
    dppl = Latte._LATTE_DPPL_CONSTRUCTORS[mvblock_dim](y, n, d)

    # The fix: total scalar dimension is n*d, not n.
    @test Latte.variable_length(dppl, :x, (log_τ = 1.0,)) == n * d

    # And the model builds.
    @test mvblock_dim(y, n, d) isa Latte.LatentGaussianModel
end

# A multivariate-block latent written with a slice LHS (`x[:, t] ~ MvNormal(x[:, t-1], Σ)`) is a
# correlated/vector state-space prior. It must auto-take the structured factor-graph path: the
# monolithic sparse-AD prior `convert`-ambiguates between `ForwardDiff.Dual` and
# `SparseConnectivityTracer.Dual` when it builds the `MvNormal` under the nested θ-gradient, so the
# factor-graph prior (per-factor closures, plain ForwardDiff) is the *only* working path. The macro
# must extract it automatically, and `inla` must run and agree with the element-wise scalar form.
@testset "Block latent auto-structures and runs inla" begin
    n, d = 6, 2

    # Vector state-space: a block MvNormal AR(1) with isotropic precision τ·I.
    @latte function mvblock(y, n, d)
        log_τ ~ Normal(0.0, 1.0)
        τ = exp(log_τ)
        x = Matrix{Real}(undef, d, n)
        x[:, 1] ~ MvNormal(zeros(d), 1.0 * I(d))
        for t in 2:n
            x[:, t] ~ MvNormal(x[:, t - 1], (1.0 / τ) * I(d))
        end
        for t in 1:n
            y[t] ~ Normal(sum(x[:, t]), 0.1)
        end
    end

    # The element-wise scalar equivalent (the documented workaround): `MvNormal(μ, (1/τ)·I)` factors
    # into independent `Normal(μ_a, 1/√τ)`. Same joint posterior, but it takes the already-working
    # scalar path — so its `inla` result is the reference.
    @latte function mvscalar(y, n, d)
        log_τ ~ Normal(0.0, 1.0)
        τ = exp(log_τ)
        x = Matrix{Real}(undef, d, n)
        for a in 1:d
            x[a, 1] ~ Normal(0.0, 1.0)
        end
        for t in 2:n
            for a in 1:d
                x[a, t] ~ Normal(x[a, t - 1], sqrt(1.0 / τ))
            end
        end
        for t in 1:n
            y[t] ~ Normal(sum(x[:, t]), 0.1)
        end
    end

    Random.seed!(20260618)
    y = randn(n)

    lgm = mvblock(y, n, d)
    # The block prior must auto-engage the factor-graph path (its guard verified it reproduces the
    # monolithic prior, so `isa StructuredLatentPrior` is itself the correctness assertion).
    @test lgm.latent_prior isa G.StructuredLatentPrior

    # inla must run end to end (it threw a `convert` MethodError on the monolithic sparse-AD path).
    r = inla(lgm, y; progress = false)
    @test all(isfinite, mean.(latent_marginals(r, :x)))

    # And it must agree with the element-wise scalar form.
    r_scalar = inla(mvscalar(y, n, d), y; progress = false)
    m_block = vec(collect(mean.(latent_marginals(r, :x))))
    m_scalar = vec(collect(mean.(latent_marginals(r_scalar, :x))))
    @test m_block ≈ m_scalar atol = 1.0e-4
end
