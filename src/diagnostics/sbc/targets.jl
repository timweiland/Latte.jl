using Distributions: VariateForm, Univariate, Multivariate, logpdf

"""
    resolve_targets(target::SBCTarget, lgm) -> Vector{TargetDescriptor}

Expand a high-level target specification into concrete per-scalar
descriptors, walking the LGM's hyperparameter spec to pin down column
positions in the posterior θ matrix.

Latte's DPPL adapter currently restricts hyperparameters to scalar
univariate priors (multivariate priors are classified as random
effects — see `latte_from_dppl`). So in practice every sym here
resolves to one descriptor. The vector-valued fan-out (one descriptor
per component, labelled `Symbol("β[i]")`) is forward-compatible
infrastructure for the planned `LatentFunctional` target type that
will let users rank scalar functionals of the latent field.
"""
function resolve_targets(::Hyperparameters, lgm)
    hp_names = collect(keys(lgm.hyperparameter_spec.free))
    return _build_descriptors(hp_names, lgm)
end

function resolve_targets(spec::NamedScalars, lgm)
    hp_names = collect(keys(lgm.hyperparameter_spec.free))
    for sym in spec.symbols
        sym in hp_names || throw(
            ArgumentError(
                "NamedScalars: sym `$(sym)` is not a free hyperparameter " *
                    "in this LGM (available: $(hp_names))."
            )
        )
    end
    return _build_descriptors(spec.symbols, lgm)
end

# Walk every free hyperparameter, accumulating the running column
# offset into the posterior θ matrix, and emit descriptors only for
# the requested subset. This keeps column indices correct even when
# `NamedScalars` picks out a subset in arbitrary order.
function _build_descriptors(requested::AbstractVector{Symbol}, lgm)
    hp_spec = lgm.hyperparameter_spec.free
    requested_set = Set(requested)
    descriptors = TargetDescriptor[]

    col_offset = 0
    for sym in keys(hp_spec)
        dim = _hp_dim(hp_spec[sym])
        if sym in requested_set
            _append_descriptors!(descriptors, sym, col_offset, dim)
        end
        col_offset += dim
    end
    return descriptors
end

# `_hp_dim` (scalar components contributed to θ by a Hyperparameter) comes
# from the shared layout helpers in model/hyperparameter_layout.jl.

function _append_descriptors!(descriptors, sym::Symbol, col_offset::Int, dim::Int)
    if dim == 1
        push!(descriptors, _scalar_descriptor(sym, col_offset + 1))
    else
        for i in 1:dim
            label = Symbol(sym, "[", i, "]")
            push!(
                descriptors, _vector_descriptor(
                    label, sym, i, col_offset + i,
                )
            )
        end
    end
    return descriptors
end

function _scalar_descriptor(sym::Symbol, col::Int)
    extract_truth = truth_nt -> _as_scalar(truth_nt[sym])
    extract_posterior = θ_mat -> Vector{Float64}(view(θ_mat, :, col))
    return TargetDescriptor(sym, sym, nothing, extract_truth, extract_posterior)
end

function _vector_descriptor(label::Symbol, sym::Symbol, idx::Int, col::Int)
    extract_truth = truth_nt -> Float64(truth_nt[sym][idx])
    extract_posterior = θ_mat -> Vector{Float64}(view(θ_mat, :, col))
    return TargetDescriptor(label, sym, idx, extract_truth, extract_posterior)
end

# DPPL hands back `[x]` for a length-1 Multivariate (e.g.
# `β ~ MvNormal(zeros(1), I)`). Collapse uniformly to a scalar.
_as_scalar(x::Real) = Float64(x)
_as_scalar(x::AbstractVector{<:Real}) =
    length(x) == 1 ? Float64(x[1]) :
    error(
        "SBC target extraction: sym expects a scalar but produced a length-$(length(x)) vector. " *
        "Did you pass a multivariate sym via `NamedScalars`? That is handled via vector-valued descriptors."
    )

# ─── Data-dependent (derived) quantities ──────────────────────────────

"""Observation log-likelihood `log p(y | x, θ)`: the exact density the
prior-predictive `y` was drawn from. `θ_nt` carries free hyperparameters
in natural space plus any fixed ones."""
function _sbc_loglik(lgm, θ_nt::NamedTuple, x::AbstractVector, y)
    η = _x_for_obs_model(lgm, x)
    dist = GaussianMarkovRandomFields.conditional_distribution(lgm.observation_model, η; θ_nt...)
    return logpdf(dist, y)
end

"""Complete-data log-likelihood `log p(y, x | θ) = log p(y | x, θ) +
log p(x | θ)`, adding the latent GMRF log-prior at the natural-space θ."""
function _sbc_complete(lgm, θ_nt::NamedTuple, x::AbstractVector, y)
    return _sbc_loglik(lgm, θ_nt, x, y) + logpdf(latent_gmrf(lgm, θ_nt), x)
end

"""
    resolve_targets(t::DataDependentQuantity, lgm) -> [DerivedTargetDescriptor]

Resolve a derived-quantity target. The descriptor evaluates `t.f` at the
true `(θ, x)` and at every posterior draw, ranking the truth among them.
Both the true and posterior hyperparameters are reconstructed to the full
natural-space NamedTuple (free draws + fixed values) so `t.f` sees a
consistent representation.
"""
function resolve_targets(t::DataDependentQuantity, lgm)
    spec = lgm.hyperparameter_spec

    extract_truth = ctx -> begin
        ctx.latent_truth === nothing && throw(
            ArgumentError(
                "DataDependentQuantity requires the prior-drawn latent truth, which is recorded " *
                    "only on the LGM SBC path (build_model returning a LatentGaussianModel). " *
                    "DPPL-path latent assembly is future work."
            )
        )
        # truth_nt is already the full natural-space NamedTuple (free + fixed).
        return Float64(t.f(ctx.lgm, ctx.truth_nt, ctx.latent_truth, ctx.y))
    end

    extract_posterior = ctx -> begin
        ctx.x_mat === nothing && throw(
            ArgumentError("DataDependentQuantity requires posterior latent draws (samples.x).")
        )
        npost = size(ctx.θ_mat, 1)
        out = Vector{Float64}(undef, npost)
        for l in 1:npost
            θ_nt = convert(NamedTuple, NaturalHyperparameters(collect(view(ctx.θ_mat, l, :)), spec))
            out[l] = Float64(t.f(ctx.lgm, θ_nt, view(ctx.x_mat, l, :), ctx.y))
        end
        return out
    end

    return DerivedTargetDescriptor[DerivedTargetDescriptor(t.label, extract_truth, extract_posterior)]
end
