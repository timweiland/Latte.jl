# Strict, MC-free correctness gate for the analytic-conjugate backbone.
#   julia --project=benchmark benchmark/validate_ci.jl
# Loads NO committed reference and runs NO MCMC for the reference: it is the exact
# Gaussian–Gaussian quadrature posterior, computed in-process. INLA is run at its
# DEFAULT exploration grid — the τ-marginal gap that surfaces is reported
# honestly, not hidden by switching to a finer grid.
using Pkg
Pkg.activate(@__DIR__)
using Latte, Distributions, StableRNGs
using Distributions: cdf
using GaussianMarkovRandomFields: IIDModel, ExponentialFamily
using Random: MersenneTwister
using Test

include(joinpath(@__DIR__, "utils", "reporting.jl"))    # defines AccuracyMetrics
include(joinpath(@__DIR__, "utils", "reference_store.jl"))
include(joinpath(@__DIR__, "utils", "accuracy.jl"))
include(joinpath(@__DIR__, "scenarios", "analytic_iid_normal.jl"))

# Wrap the analytic NamedTuple into a ReferenceSummary (n_chains=0 ⇒ no MC floor ⇒
# raw KS, exactly right for an exact oracle).
function _analytic_ref(scenario_id, data)
    s = analytic_iid_normal_reference(data)
    np = length(s.parameter_names)
    return ReferenceSummary(
        scenario_id = scenario_id, data_id = string(hash(data.y), base = 16),
        parameter_names = s.parameter_names,
        posterior_cdf_grids = s.cdf_grids, posterior_cdf_values = s.cdf_values,
        posterior_q025 = s.q025, posterior_q25 = s.q25, posterior_median = s.median,
        posterior_q75 = s.q75, posterior_q975 = s.q975, posterior_q99 = s.q99,
        ess = zeros(np), rhat = ones(np), n_chains = 0, n_samples_per_chain = 0,
        n_warmup = 0, seed = UInt64(0), timestamp_iso = "analytic",
        notes = ["Exact Gaussian–Gaussian quadrature reference; in-process, no MCMC."],
    )
end

# Hand-built LGM ⇒ empty latent_layout ⇒ compare the full latent_marginals directly.
_metrics(res, ref) = accuracy_against_reference(
    Latte.hyperparameter_marginals(res), Latte.latent_marginals(res), ref
)

_median_relerr(res, ref) =
    abs(quantile(Latte.hyperparameter_marginals(res)[1], 0.5) - ref.posterior_median[1]) /
    max(ref.posterior_median[1], eps())

const SEED = UInt64(0x0badcafe)
const N = 200
data = generate_data(N; seed = SEED)
ref = _analytic_ref(SCENARIO_ID, data)
lgm = build_lgm(data)

res_inla = inla(lgm, data.y; progress = false)                 # DEFAULT grid (gap reported, not hidden)
res_tmb = tmb(lgm, data.y)
res_hmc = hmc_laplace(
    lgm, data.y; rng = MersenneTwister(SEED), n_samples = 2000, n_warmup = 1000, progress = false
)

m_inla = _metrics(res_inla, ref)
m_tmb = _metrics(res_tmb, ref)
m_hmc = _metrics(res_hmc, ref)

@testset "analytic-conjugate backbone (exact, MC-free)" begin
    # ── STRICT GATE: deterministic engines on cells that are exact here ──
    @testset "TMB strict KS ≤ 0.02 (both blocks)" begin
        @test m_tmb.posterior_ks_max ≤ 0.02       # τ block — TMB Gaussianizes to exact at n=200
        @test m_tmb.latent_ks_max ≤ 0.02          # latent block
        @test m_tmb.worst_ks ≤ 0.02
    end
    @testset "INLA latent strict KS ≤ 0.02 (inner Laplace exact here)" begin
        @test m_inla.latent_ks_max ≤ 0.02
    end
    @test _median_relerr(res_tmb, ref) ≤ 0.05
    @test _median_relerr(res_inla, ref) ≤ 0.08    # INLA's default τ grid is coarser; honest bound

    # ── DOCUMENTED, NON-MASKING bound: INLA's τ marginal floors ~0.025–0.03 at
    # the default 5-point exploration grid (a real approximation, surfaced in the
    # report; a finer grid closes it). Gate only at the honest yellow band.
    @testset "INLA τ documented default-grid gap (≤ 0.05)" begin
        @test m_inla.posterior_ks_max ≤ 0.05
        if m_inla.posterior_ks_max > 0.02
            @info "INLA τ KS exceeds strict 0.02 at the default grid (expected; coarse hp grid under-resolves the skewed τ tail)" ks = m_inla.posterior_ks_max
        end
    end

    # ── REPORT-ONLY: hmc_laplace is seeded MCMC over the Laplace marginal ──
    @testset "HMC-Laplace report-only (loose, seeded)" begin
        @test m_hmc.worst_ks ≤ 0.05               # generous; chain noise at 2000 draws
        @info "hmc_laplace KS (report-only)" τ = m_hmc.posterior_ks_max x = m_hmc.latent_ks_max
    end
end

println(
    "validate_ci OK | TMB worst=", round(m_tmb.worst_ks, digits = 4),
    " | INLA τ=", round(m_inla.posterior_ks_max, digits = 4),
    " x=", round(m_inla.latent_ks_max, digits = 4),
    " | HMC worst=", round(m_hmc.worst_ks, digits = 4)
)
