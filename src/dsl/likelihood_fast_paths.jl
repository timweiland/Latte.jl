# Likelihood fast-paths: detect common `y[i] ~ Distribution(link(linear(x)))`
# patterns in a DPPL model and substitute GMRFs.jl's hand-coded
# `ExponentialFamily` observation model for the default
# `AutoDiffObservationModel`. Dramatically faster; avoids a nested-AD bug
# in the default path.
#
# Fast-path is a pure optimization: anything we can't recognise falls
# through to the existing AD wrapping unchanged.

using SparseArrays
using ADTypes: AutoSparse, AutoForwardDiff
using DifferentiationInterface
using SparseConnectivityTracer: TracerLocalSparsityDetector
using SparseMatrixColorings: GreedyColoringAlgorithm
using Distributions: Poisson, Bernoulli, Normal, mean
using GaussianMarkovRandomFields:
    ExponentialFamily, LinearlyTransformedObservationModel,
    LogLink, LogitLink, IdentityLink

# ─── Custom DPPL accumulator: records observation distributions ───────────
# `PriorDistributionAccumulator` skips observed (y) sites. We mirror its
# shape but hook `accumulate_observe!!` instead, collecting per-site
# distributions in the order the model emits them.
mutable struct _ObsDistributionAccumulator <: DynamicPPL.AbstractAccumulator
    dists::Vector{Any}
end
_ObsDistributionAccumulator() = _ObsDistributionAccumulator(Any[])

DynamicPPL.accumulator_name(::Type{_ObsDistributionAccumulator}) = :ObsDistributionAccumulator
Base.copy(a::_ObsDistributionAccumulator) = _ObsDistributionAccumulator(copy(a.dists))
DynamicPPL.reset(a::_ObsDistributionAccumulator) = (empty!(a.dists); a)
DynamicPPL.split(::_ObsDistributionAccumulator) = _ObsDistributionAccumulator()
function DynamicPPL.combine(a::_ObsDistributionAccumulator, b::_ObsDistributionAccumulator)
    return _ObsDistributionAccumulator(vcat(a.dists, b.dists))
end
function DynamicPPL.accumulate_observe!!(a::_ObsDistributionAccumulator, right, left, vn, template)
    push!(a.dists, right)
    return a
end
function DynamicPPL.accumulate_assume!!(a::_ObsDistributionAccumulator, val, tv, lj, vn, d, t)
    return a
end

"""
    _probe_obs_distributions(dppl_model, hp_nt, latent_nt)

Run `dppl_model` once with hyperparameters fixed to `hp_nt` and latent
variables initialised from `latent_nt`, recording the distribution at
each observation `~` site. Returns a Vector of distributions in emission
order.
"""
function _probe_obs_distributions(dppl_model, hp_nt::NamedTuple, latent_nt::NamedTuple)
    cond = DynamicPPL.fix(dppl_model, hp_nt)
    vi = DynamicPPL.OnlyAccsVarInfo((_ObsDistributionAccumulator(),))
    vi = last(DynamicPPL.init!!(cond, vi, DynamicPPL.InitFromParams(latent_nt, nothing), DynamicPPL.UnlinkAll()))
    return DynamicPPL.getacc(vi, Val(:ObsDistributionAccumulator)).dists
end

# ─── Family dispatch: distribution type → (family, link, natural-param) ──
# Extension point: one line per new family. `nothing` signals "not
# supported, punt to AD fallback".
_ef_family_info(::Type{<:Poisson}) = (Poisson, LogLink(), d -> log(mean(d)))
_ef_family_info(::Type{<:Bernoulli}) = (Bernoulli, LogitLink(), d -> (p = mean(d); log(p / (1 - p))))
_ef_family_info(::Type{<:Normal}) = (Normal, IdentityLink(), d -> mean(d))
_ef_family_info(_) = nothing

# ─── Main detection + assembly ────────────────────────────────────────────
"""
    try_exponential_family_fast_path(dppl_model, random_syms, dims, hp_names)
        -> ObservationModel or nothing

Detect whether the DPPL likelihood is a homogeneous single-family
distribution with a canonical link and a linear predictor in `x`. If so,
return a `LinearlyTransformedObservationModel` wrapping an
`ExponentialFamily` (optionally further wrapped in
`OffsetObservationModel` when the linear predictor has a non-zero
constant term). Return `nothing` for anything non-conformant; caller
falls through to the AD-based wrapping.

Current support:
- Family: `Poisson` + `LogLink`, `Bernoulli` + `LogitLink`, `Normal` +
  `IdentityLink`.
- Predictor: affine in the concatenated latent vector `x = [β; u; ...]`.
  Non-zero constant term (e.g. Poisson log-exposure, Bernoulli logit
  shift, Normal mean offset) is captured by wrapping the base obs model
  in `OffsetObservationModel` — works uniformly for augmented and
  non-augmented LGMs.
- Likelihood: homogeneous (all y sites use the same distribution family).
"""
function try_exponential_family_fast_path(
        dppl_model, random_syms::Tuple, dims::Dict{Symbol, Int}, hp_names::Tuple
    )
    probe_hp = NamedTuple{hp_names}(Tuple(1.0 for _ in hp_names))
    probe_x_nt = NamedTuple{random_syms}(Tuple(zeros(dims[s]) for s in random_syms))

    # 1) probe per-site y distributions
    y_dists = _probe_obs_distributions(dppl_model, probe_hp, probe_x_nt)
    isempty(y_dists) && return nothing

    # 2) homogeneous single-family check + supported family lookup
    T = typeof(first(y_dists))
    all(d -> typeof(d) === T, y_dists) || return nothing
    fam_info = _ef_family_info(T)
    fam_info === nothing && return nothing
    family, link, natural_param = fam_info

    # 3) linearity probe via sparse-AD Jacobian of the natural predictor
    n_latent = sum(dims[s] for s in random_syms)

    # Offsets for splitting a flat x vector into per-random-symbol components
    offsets = _component_offsets(random_syms, dims)

    function η_of_x(x_vec)
        x_nt = NamedTuple{random_syms}(Tuple(Vector(x_vec[offsets[s]]) for s in random_syms))
        dists = _probe_obs_distributions(dppl_model, probe_hp, x_nt)
        return [natural_param(d) for d in dists]
    end

    backend = AutoSparse(
        AutoForwardDiff();
        sparsity_detector = TracerLocalSparsityDetector(),
        coloring_algorithm = GreedyColoringAlgorithm(),
    )
    prep = prepare_jacobian(η_of_x, backend, zeros(n_latent))
    A = jacobian(η_of_x, prep, backend, zeros(n_latent))
    A_check = jacobian(η_of_x, prep, backend, ones(n_latent))
    isapprox(A, A_check; atol = 1.0e-10) || return nothing           # nonlinear → punt

    # 4) assemble. Non-zero `b = η(0)` → wrap in OffsetObservationModel
    # (obs-layer offset, works for any LGM shape).
    b = η_of_x(zeros(n_latent))
    A_sp = SparseMatrixCSC(A)
    dropzeros!(A_sp)
    base = ExponentialFamily(family, link)
    obs = all(iszero, b) ? base : OffsetObservationModel(base, Vector{Float64}(b))
    return LinearlyTransformedObservationModel(obs, A_sp)
end

# ─── Small helper shared with obs_model.jl ────────────────────────────────
function _component_offsets(random_syms::Tuple, dims::Dict{Symbol, Int})
    offsets = Dict{Symbol, UnitRange{Int}}()
    off = 0
    for s in random_syms
        offsets[s] = (off + 1):(off + dims[s])
        off += dims[s]
    end
    return offsets
end
