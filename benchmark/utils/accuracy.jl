# Headline metric is per-parameter KS distance against the reference
# CDF; signed gap at argmax + reference-CI mass errors are the
# secondary diagnostics. Aggregation: max KS / max |CI error| within
# each block (hp, latent), cross-block worst drives the verdict band.

using Statistics: mean, std, quantile
using Distributions: cdf
using Latte: Latte

# Accuracy bands (descriptive only; KS number is the headline):
#   green ≤ 0.02 / yellow ≤ 0.05 / orange ≤ 0.15 / red > 0.15
const _BAND_GREEN = 0.02
const _BAND_YELLOW = 0.05
const _BAND_ORANGE = 0.15

# CI levels we report mass errors at.
const _CI_LEVELS = (
    (:ci50, 0.5),
    (:ci90, 0.9),
    (:ci95, 0.95),
)

"""
    accuracy_against_reference(hp_marginals, latent_marginals, ref;
                               yellow=0.02, red=0.10) -> AccuracyMetrics

Compare engine posterior marginals to a reference and produce the
KS-based `AccuracyMetrics` payload. `hp_marginals` and
`latent_marginals` are `Vector{<:Distribution}`-like — anything
supporting `cdf(d, x)` works.

Returns `nothing` if the reference's parameter list doesn't cover
the engine's marginals.
"""
function accuracy_against_reference(
        hp_marginals, latent_marginals, ref::ReferenceSummary;
        green::Float64 = _BAND_GREEN,
        yellow::Float64 = _BAND_YELLOW,
        orange::Float64 = _BAND_ORANGE,
    )
    n_hp = length(hp_marginals)
    n_latent = length(latent_marginals)
    n_total = n_hp + n_latent
    length(ref.parameter_names) >= n_total || return nothing

    hp_block = _block_metrics(hp_marginals, ref, 1:n_hp)
    latent_block = n_latent > 0 ?
        _block_metrics(latent_marginals, ref, (n_hp + 1):(n_hp + n_latent)) :
        _empty_block()

    ks_candidates = filter(!isnothing, (hp_block.ks_max, latent_block.ks_max))
    worst_ks = isempty(ks_candidates) ? nothing : maximum(ks_candidates)
    accuracy_band = if worst_ks === nothing
        :unknown
    elseif worst_ks > orange
        :red
    elseif worst_ks > yellow
        :orange
    elseif worst_ks > green
        :yellow
    else
        :green
    end

    # NUTS-reference noise floor: KS values at or below `1.36/√ESS`
    # are indistinguishable from the chain's own MC error. `nothing`
    # for quadrature oracles (n_chains == 0).
    posterior_floor, latent_floor = _ks_mc_floors(ref, n_hp, n_latent)

    return AccuracyMetrics(
        posterior_ks_max = hp_block.ks_max,
        posterior_ks_signed_at_argmax = hp_block.ks_signed,
        latent_ks_max = latent_block.ks_max,
        latent_ks_signed_at_argmax = latent_block.ks_signed,
        posterior_ks_mc_floor = posterior_floor,
        latent_ks_mc_floor = latent_floor,
        posterior_ci50_mass_error = hp_block.ci50,
        posterior_ci90_mass_error = hp_block.ci90,
        posterior_ci95_mass_error = hp_block.ci95,
        latent_ci50_mass_error = latent_block.ci50,
        latent_ci90_mass_error = latent_block.ci90,
        latent_ci95_mass_error = latent_block.ci95,
        worst_ks = worst_ks,
        accuracy_band = accuracy_band,
    )
end

# Per-block 1.36/√(min ESS) noise floor for NUTS references; nothing
# when the reference is a quadrature oracle (n_chains == 0).
function _ks_mc_floors(ref::ReferenceSummary, n_hp::Int, n_latent::Int)
    ref.n_chains == 0 && return (nothing, nothing)
    hp_idx = 1:n_hp
    latent_idx = (n_hp + 1):(n_hp + n_latent)
    posterior_floor = isempty(hp_idx) ? nothing :
        1.36 / sqrt(minimum(@view ref.ess[hp_idx]))
    latent_floor = isempty(latent_idx) ? nothing :
        1.36 / sqrt(minimum(@view ref.ess[latent_idx]))
    return posterior_floor, latent_floor
end

# ─── Per-block KS + CI mass error ──────────────────────────────────────

function _block_metrics(marginals, ref::ReferenceSummary, indices)
    isempty(marginals) && return _empty_block()

    ks_per_param = Float64[]
    ks_signed_per_param = Float64[]
    ci_errors = Dict{Symbol, Vector{Float64}}(
        sym => Float64[] for (sym, _) in _CI_LEVELS
    )

    for (k, j) in enumerate(indices)
        ref_grid = ref.posterior_cdf_grids[j]
        ref_cdf = ref.posterior_cdf_values[j]
        # Skip parameters with no CDF rather than silently using a
        # weaker metric — caller should populate CDFs for every param.
        (isempty(ref_grid) || isempty(ref_cdf)) && continue

        eng = marginals[k]
        ks_max, ks_signed = _ks_distance(eng, ref_grid, ref_cdf)
        push!(ks_per_param, ks_max)
        push!(ks_signed_per_param, ks_signed)

        for (sym, α) in _CI_LEVELS
            err = _ci_mass_error(eng, ref, j, α)
            push!(ci_errors[sym], err)
        end
    end

    isempty(ks_per_param) && return _empty_block()

    i_worst = argmax(ks_per_param)
    return (
        ks_max = ks_per_param[i_worst],
        ks_signed = ks_signed_per_param[i_worst],
        ci50 = maximum(abs, ci_errors[:ci50]),
        ci90 = maximum(abs, ci_errors[:ci90]),
        ci95 = maximum(abs, ci_errors[:ci95]),
    )
end

function _empty_block()
    return (
        ks_max = nothing, ks_signed = nothing,
        ci50 = nothing, ci90 = nothing, ci95 = nothing,
    )
end

# Kolmogorov distance between an engine marginal and a reference CDF
# given as a sorted (grid, F_grid) pair. Returns the unsigned max and
# the signed gap at the argmax (positive = engine CDF above reference).
function _ks_distance(engine, ref_grid::Vector{Float64}, ref_cdf::Vector{Float64})
    best_abs = 0.0
    best_signed = 0.0
    @inbounds for k in eachindex(ref_grid)
        x = ref_grid[k]
        F_eng = cdf(engine, x)
        gap = F_eng - ref_cdf[k]
        if abs(gap) > best_abs
            best_abs = abs(gap)
            best_signed = gap
        end
    end
    return best_abs, best_signed
end

# CI mass error: P_engine(θ ∈ [Q_ref((1-α)/2), Q_ref((1+α)/2)]) − α.
# Positive = engine puts more mass than reference inside the central
# α-CI; negative = less. We aggregate at the block level by max-abs.
function _ci_mass_error(engine, ref::ReferenceSummary, j::Int, α::Float64)
    lo, hi = _ref_central_quantiles(ref, j, α)
    mass = cdf(engine, hi) - cdf(engine, lo)
    return mass - α
end

# Pre-stored quantile pairs for standard CI levels; otherwise fall
# back to interpolating the CDF grid.
function _ref_central_quantiles(ref::ReferenceSummary, j::Int, α::Float64)
    if α ≈ 0.5
        return ref.posterior_q25[j], ref.posterior_q75[j]
    elseif α ≈ 0.95
        return ref.posterior_q025[j], ref.posterior_q975[j]
    elseif α ≈ 0.9
        return _cdf_quantile(ref, j, 0.05), _cdf_quantile(ref, j, 0.95)
    else
        return _cdf_quantile(ref, j, (1 - α) / 2), _cdf_quantile(ref, j, (1 + α) / 2)
    end
end

# Linear interpolation along the reference's stored CDF grid to find
# the x at which F(x) ≈ p.
function _cdf_quantile(ref::ReferenceSummary, j::Int, p::Float64)
    grid = ref.posterior_cdf_grids[j]
    cdf_vals = ref.posterior_cdf_values[j]
    n = length(grid)
    n == 0 && return NaN
    k = searchsortedfirst(cdf_vals, p)
    k <= 1 && return grid[1]
    k > n && return grid[end]
    t = (p - cdf_vals[k - 1]) / (cdf_vals[k] - cdf_vals[k - 1])
    return grid[k - 1] + t * (grid[k] - grid[k - 1])
end

# ─── Latent slicing helper (unchanged) ─────────────────────────────────

"""
    user_named_latents(result, random_syms) -> Vector

Slice an inference result's full augmented latent vector down to just
the user-named random effects in declaration order, dropping any
augmentation latents (η, etc.) introduced by the LGM construction.

Output is shape-compatible with the reference's latent block, which
only ever sees user-level DPPL variables.
"""
function user_named_latents(result, random_syms)
    groups = Latte.latent_groups(result)
    out = Any[]
    for sym in random_syms
        idx = get(groups, sym, nothing)
        idx === nothing && continue
        for i in idx
            push!(out, result.latent_marginals[i])
        end
    end
    return out
end
