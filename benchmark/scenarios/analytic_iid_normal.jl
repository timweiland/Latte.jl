# Analytic-conjugate backbone: a Gaussian–Gaussian IID LGM with KNOWN observation
# variance. The conditional latent posterior is exactly Gaussian (so INLA's inner
# Laplace is exact here) and the marginal likelihood is closed form, so the τ
# posterior and the marginal latent posteriors are available EXACTLY by 1-D
# quadrature — no MCMC, no Laplace. This is the strict, MC-free numeric backbone
# of the validation report.
#
# Driven directly by benchmark/validate_ci.jl (build_lgm / generate_data /
# analytic_iid_normal_reference), NOT through runbench.jl's generic engine
# harness: that path goes via latte_from_dppl, whose Normal fast-path forces σ to
# be a free hyperparameter, whereas this backbone needs σ FIXED so τ is the sole
# hyperparameter. A DPPL-expressible variant (once the adapter can fix σ) is
# tracked as future work.

using Latte
using Distributions: Normal, logpdf, cdf
using GaussianMarkovRandomFields: IIDModel, ExponentialFamily
using StableRNGs: StableRNG

const SCENARIO_ID = "analytic_iid_normal"
const RANDOM_SYMS = ()           # hand-built LGM ⇒ empty latent_layout; compare latent_marginals directly
const HP_SYMS = (:τ,)
const SIGMA_OBS = 0.5            # KNOWN observation SD (σ² = 0.25)
const PC_U = 1.0                 # PC-prior calibration, shared by model + reference
const PC_ALPHA = 0.01
const TAU_TRUE = 2.0

"""
    generate_data(n; seed) -> (; n, y, true_x)

Deterministic data: x ~ N(0, 1/τ_true) (IID latent), y_i ~ N(x_i, σ²).
"""
function generate_data(n::Int; seed)
    rng = StableRNG(UInt64(seed))
    x_true = randn(rng, n) ./ sqrt(TAU_TRUE)
    y = x_true .+ SIGMA_OBS .* randn(rng, n)
    return (; n = n, y = y, true_x = x_true)
end

"""
    build_lgm(data) -> LatentGaussianModel

Hand-built single-τ LGM with a FIXED σ (the DPPL Normal fast-path would require σ
as a free hyperparameter; `@hyperparams` supports a fixed `σ = value`).
"""
function build_lgm(data)
    hp = @hyperparams begin
        (τ ~ PCPrior.Precision(PC_U, α = PC_ALPHA), transform = log, space = natural)
        σ = SIGMA_OBS
    end
    return LatentGaussianModel(hp, IIDModel(data.n), ExponentialFamily(Normal))
end

# ── Exact reference (1-D quadrature; no MCMC) ─────────────────────────────────
_trap(v, Δ) = Δ * (sum(v) - 0.5 * (first(v) + last(v)))
function _cumtrap(w, Δ)
    c = similar(w); c[1] = 0.0
    @inbounds for k in 2:length(w)
        c[k] = c[k - 1] + 0.5 * (w[k - 1] + w[k]) * Δ
    end
    return c
end
function _invcdf(grid, cdfv, p)
    n = length(cdfv); k = searchsortedfirst(cdfv, p)
    k <= 1 && return grid[1]
    k > n && return grid[end]
    t = (p - cdfv[k - 1]) / (cdfv[k] - cdfv[k - 1])
    return grid[k - 1] + t * (grid[k] - grid[k - 1])
end

"""
    analytic_iid_normal_reference(data; n_grid_τ=8001, n_grid_x=601,
                                  log_τ_min=-12.0, log_τ_max=12.0, x_pad_sds=8.0)

EXACT marginal posteriors for the Gaussian–Gaussian IID model with KNOWN σ.
τ posterior via the closed-form marginal likelihood p(y|τ) = ∏ N(y_i; 0, σ²+1/τ)
times the PC prior, normalized by 1-D trapezoid in η = log τ. Latent marginals are
the exact quadrature Gaussian MIXTURE p(x_i|y) = ∫ p(x_i|y_i,τ) p(τ|y) dτ.
Returns the NamedTuple shape ReferenceSummary / accuracy.jl consume.
"""
function analytic_iid_normal_reference(
        data; n_grid_τ::Int = 8001, n_grid_x::Int = 601,
        log_τ_min::Float64 = -12.0, log_τ_max::Float64 = 12.0,
        x_pad_sds::Float64 = 8.0,
    )
    y = data.y; n = length(y); σ2 = SIGMA_OBS^2
    prior = PCPrior.Precision(PC_U, α = PC_ALPHA)

    η = collect(range(log_τ_min, log_τ_max, length = n_grid_τ))
    τg = exp.(η); Δη = η[2] - η[1]

    # log p(y|τ) + log π(τ) + η  (η-Jacobian: dτ = τ dη)
    logpost = Vector{Float64}(undef, n_grid_τ)
    @inbounds for j in 1:n_grid_τ
        v = σ2 + 1 / τg[j]                                   # Var(y_i|τ) = σ² + 1/τ
        ll = -0.5 * sum(log(2π * v) + (yi * yi) / v for yi in y)
        logpost[j] = ll + logpdf(prior, τg[j]) + η[j]
    end
    logpost .-= maximum(logpost)
    w = exp.(logpost); Z = _trap(w, Δη); w ./= Z

    cdf_τ = _cumtrap(w, Δη)
    @assert cdf_τ[end] > 1 - 1.0e-8 "τ grid truncated (cdf_τ[end]=$(cdf_τ[end])); widen log_τ range"
    τ_q = (
        q025 = _invcdf(τg, cdf_τ, 0.025), q25 = _invcdf(τg, cdf_τ, 0.25),
        med = _invcdf(τg, cdf_τ, 0.5), q75 = _invcdf(τg, cdf_τ, 0.75),
        q975 = _invcdf(τg, cdf_τ, 0.975), q99 = _invcdf(τg, cdf_τ, 0.99),
    )

    cdf_grids = Vector{Vector{Float64}}(undef, n + 1)
    cdf_values = Vector{Vector{Float64}}(undef, n + 1)
    cdf_grids[1] = collect(τg); cdf_values[1] = cdf_τ

    xq = ntuple(_ -> Vector{Float64}(undef, n), 6)   # q025,q25,med,q75,q975,q99
    keep = w .> 1.0e-12 * maximum(w)
    s = [sqrt(1 / (τ + 1 / σ2)) for τ in τg]             # conditional sd (i-independent)
    @inbounds for i in 1:n
        m = [(1 / (τ + 1 / σ2)) * (y[i] / σ2) for τ in τg]  # conditional mean
        x_lo = minimum(@view(m[keep]) .- x_pad_sds .* @view(s[keep]))
        x_hi = maximum(@view(m[keep]) .+ x_pad_sds .* @view(s[keep]))
        xgrid = collect(range(x_lo, x_hi, length = n_grid_x))
        Fx = zeros(Float64, n_grid_x)
        for j in 1:n_grid_τ
            w[j] < 1.0e-14 && continue
            wt = ((j == 1 || j == n_grid_τ) ? 0.5Δη : Δη) * w[j]
            dist = Normal(m[j], s[j])
            for k in 1:n_grid_x
                Fx[k] += wt * cdf(dist, xgrid[k])
            end
        end
        Fx ./= Fx[end]
        cdf_grids[i + 1] = xgrid; cdf_values[i + 1] = Fx
        xq[1][i] = _invcdf(xgrid, Fx, 0.025); xq[2][i] = _invcdf(xgrid, Fx, 0.25)
        xq[3][i] = _invcdf(xgrid, Fx, 0.5); xq[4][i] = _invcdf(xgrid, Fx, 0.75)
        xq[5][i] = _invcdf(xgrid, Fx, 0.975); xq[6][i] = _invcdf(xgrid, Fx, 0.99)
    end

    parameter_names = String["τ"; ["x[$(i)]" for i in 1:n]]
    return (
        parameter_names = parameter_names,
        cdf_grids = cdf_grids, cdf_values = cdf_values,
        q025 = vcat(τ_q.q025, xq[1]), q25 = vcat(τ_q.q25, xq[2]),
        median = vcat(τ_q.med, xq[3]), q75 = vcat(τ_q.q75, xq[4]),
        q975 = vcat(τ_q.q975, xq[5]), q99 = vcat(τ_q.q99, xq[6]),
    )
end
