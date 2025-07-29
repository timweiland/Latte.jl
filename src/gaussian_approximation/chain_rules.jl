using ChainRulesCore
using Zygote
using LinearAlgebra
using GaussianMarkovRandomFields
using SparseArrays

"""
    _add_namedtuples(nt1::Union{NamedTuple, Nothing}, nt2::Union{NamedTuple, Nothing}) -> Union{NamedTuple, Nothing}

Add two NamedTuples (or nothing) with smart handling of all cases.

**Top-level nothing handling:**
- If both arguments are `nothing`, return `nothing`
- If one argument is `nothing`, return the other argument
- If both are NamedTuples, proceed to key-wise addition

**Key-wise addition (when both are NamedTuples):**
- If one value is `nothing`, use the other value
- If both values are non-`nothing`, add them together
- If both values are `nothing`, the result is `nothing`

# Arguments
- `nt1::Union{NamedTuple, Nothing}`: First NamedTuple or nothing
- `nt2::Union{NamedTuple, Nothing}`: Second NamedTuple or nothing (must have same keys as `nt1` if both are NamedTuples)

# Returns
- `Union{NamedTuple, Nothing}`: Result with smart combination of inputs

# Examples
```julia
# Top-level nothing handling
add_namedtuples(nothing, nothing) == nothing
add_namedtuples(nothing, (a=1,)) == (a=1,)
add_namedtuples((a=1,), nothing) == (a=1,)

# Key-wise addition
nt1 = (a = 1.0, b = nothing, c = [1, 2])
nt2 = (a = 2.0, b = 3.0, c = [3, 4])
result = add_namedtuples(nt1, nt2)
# result = (a = 3.0, b = 3.0, c = [4, 6])
```
"""
function _add_namedtuples(nt1::Union{NamedTuple, Nothing}, nt2::Union{NamedTuple, Nothing})
    # Handle top-level nothing cases first
    if nt1 === nothing && nt2 === nothing
        return nothing
    elseif nt1 === nothing
        return nt2
    elseif nt2 === nothing
        return nt1
    end

    # Create result by combining values for each key
    result_pairs = map(keys(nt1)) do key
        val1 = getproperty(nt1, key)
        val2 = getproperty(nt2, key)

        # Handle different combinations of nothing/non-nothing
        if val1 === nothing && val2 === nothing
            combined_val = nothing
        elseif val1 === nothing
            combined_val = val2
        elseif val2 === nothing
            combined_val = val1
        else
            # Both are non-nothing, add them
            combined_val = val1 + val2
        end

        key => combined_val
    end

    return NamedTuple(result_pairs)
end

"""
    negate_namedtuple(nt::Union{NamedTuple, Nothing}) -> Union{NamedTuple, Nothing}

Negate a NamedTuple (or nothing) with smart handling of `nothing`.

**Top-level nothing handling:**
- If argument is `nothing`, return `nothing`
- If argument is a NamedTuple, proceed to key-wise negation

**Key-wise negation:**
- If a value is `nothing`, keep it as `nothing`
- If a value is non-`nothing`, negate it with `-`

# Arguments
- `nt::Union{NamedTuple, Nothing}`: NamedTuple to negate or nothing

# Returns
- `Union{NamedTuple, Nothing}`: Negated result

# Examples
```julia
negate_namedtuple(nothing) == nothing
negate_namedtuple((a = 1.0, b = nothing, c = [1, 2])) == (a = -1.0, b = nothing, c = [-1, -2])
```
"""
function _negate_namedtuple(nt::Union{NamedTuple, Nothing})
    # Handle top-level nothing case
    if nt === nothing
        return nothing
    end

    # Negate each value in the NamedTuple
    result_pairs = map(keys(nt)) do key
        val = getproperty(nt, key)

        # Handle nothing vs non-nothing values
        if val === nothing
            negated_val = nothing
        else
            negated_val = -val
        end

        key => negated_val
    end

    return NamedTuple(result_pairs)
end

function ChainRulesCore.rrule(
        ::typeof(gaussian_approximation),
        prior_gmrf::GMRF, obs_lik::ObservationLikelihood
    )
    # === Forward pass ===
    d_star = gaussian_approximation(prior_gmrf, obs_lik)
    x_star = mean(d_star)
    Q_star_cho = d_star.solver.precision_chol

    # === Pullback ===
    function pullback(ȳ)
        # IFT pullback:
        μ̄ = ȳ.mean
        Q̄ = ȳ.precision
        λ = Q_star_cho \ μ̄       # λ = Hᵗ⁻¹ ȳ
        _, vjp_pullback = Zygote.pullback(∇ₓ_neg_log_posterior, prior_gmrf, obs_lik, x_star)

        prior_gmrf_adj, obs_lik_adj, _ = vjp_pullback(λ)

        prior_gmrf_adj = (
            mean = prior_gmrf_adj.mean,
            precision = prior_gmrf_adj.precision + Q̄,
            solver = prior_gmrf_adj.solver,
        )

        obs_lik_adj_Q̄, _ = Zygote.pullback(loghessian, obs_lik, x_star)[2](Q̄)
        obs_lik_adj = _add_namedtuples(obs_lik_adj, _negate_namedtuple(obs_lik_adj_Q̄))

        return NoTangent(), prior_gmrf_adj, obs_lik_adj
    end

    return d_star, pullback
end
