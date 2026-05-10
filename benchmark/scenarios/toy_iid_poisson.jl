# Toy IID Poisson scenario — smallest possible LGM, used as a smoke
# test for the benchmark harness. Not a serious benchmark.
#
# Model:
#   τ ~ PCPrior.Precision(1.0, α = 0.01)
#   x ~ IIDModel(n)(τ = τ)
#   y_i ~ Poisson(exp(x_i))

using DynamicPPL: @model
using Distributions
using GaussianMarkovRandomFields: IIDModel
using StableRNGs

const SCENARIO_ID = "toy_iid_poisson"

# Symbols in the @model that are random effects / latent fields. Everything
# else is treated as a hyperparameter by `latte_from_dppl`.
const RANDOM_SYMS = (:x,)

# Hyperparameter symbols, in the same order Latte's flat θ vector
# presents them. The NUTS reference engine summarises these and the
# other engines compare their hyperparameter marginals against this
# ordering. Keep in sync with what `latte_from_dppl` produces.
const HP_SYMS = (:τ,)

"""
    scenario() -> Scenario

Smoke-test scenario. Tiny n, fast on every engine, used to validate the
harness end-to-end.
"""
function scenario()
    return Scenario(
        id = SCENARIO_ID,
        title = "Toy IID Poisson",
        description = "Smallest possible LGM. Hyperparameter τ, IID Gaussian latent, Poisson likelihood. Used as a smoke test for the benchmark runner.",
        target = "Posterior marginal of τ; posterior means of x[i].",
        engines = [:latte_inla, :latte_tmb, :latte_hmc_laplace, :nuts_reference],
        quick_n = 10,
        full_n = 50,
        timeout_seconds = 120.0,
        repetitions = 5,
        # τ is a precision parameter; at small n the posterior has a long
        # right tail. Bumped chains and tighter `target_accept` keep ESS
        # respectable in q975 and avoid divergences in the tail.
        nuts_full_samples = 5_000,
        nuts_full_warmup = 2_000,
        nuts_full_chains = 4,
        nuts_target_accept = 0.95,
        notes = ["Smoke test only; not a serious benchmark."],
    )
end

# ─── Model + data ─────────────────────────────────────────────────────

@model function toy_model(y, n)
    τ ~ PCPrior.Precision(1.0, α = 0.01)
    x ~ IIDModel(n)(τ = τ)
    for i in eachindex(y)
        y[i] ~ Poisson(exp(x[i]); check_args = false)
    end
end

"""
    generate_data(n; seed) -> NamedTuple

Deterministic data generation: same seed ⇒ same dataset. Returned y
includes a small dispersion to keep the posterior well-defined.
"""
function generate_data(n::Int; seed::UInt64)
    rng = StableRNG(seed)
    true_x = randn(rng, n) .* 0.4 .+ 0.7
    y = rand.(rng, Poisson.(exp.(true_x)))
    return (; n = n, y = y, true_x = true_x)
end

"""
    build_model(data) -> DynamicPPL.Model
"""
build_model(data) = toy_model(data.y, data.n)
