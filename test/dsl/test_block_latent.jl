using Test
using Latte
using Distributions, Random, LinearAlgebra

# A matrix-slice (multivariate-block) latent `x[:, t] ~ MvNormal(d)` over `n` columns has `n * d`
# scalar entries, not `n`. `variable_length` used to count the `~` sites (`n`), which sized the
# probe seed vector wrong and crashed model construction with a BoundsError. It must count scalars.
# (Running inla on such a model still hits a separate nested-AD-through-MvNormal ambiguity in the
# sparse-AD path — tracked separately; this test covers only the dimension/build fix.)

@testset "Block (matrix-slice MvNormal) latent dimension" begin
    n, d = 6, 2

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

    Random.seed!(20260618)
    y = randn(n)
    dppl = Latte._LATTE_DPPL_CONSTRUCTORS[mvblock](y, n, d)

    # The fix: total scalar dimension is n*d, not n.
    @test Latte.variable_length(dppl, :x, (log_τ = 1.0,)) == n * d

    # And the model now builds (was a BoundsError in the prior probe).
    @test mvblock(y, n, d) isa Latte.LatentGaussianModel
end
