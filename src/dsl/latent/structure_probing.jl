# DPPL model structure probing: extract priors without sampling, classify
# variables as atomic Gaussians vs non-Gaussian vs loop-built.

using DynamicPPL: extract_priors, init!!, OnlyAccsVarInfo,
    PriorDistributionAccumulator, UnlinkAll, getacc, PRIOR_ACCNAME, InitFromParams
using DynamicPPL.AbstractPPL: getsym
import DynamicPPL
using GaussianMarkovRandomFields: AbstractGMRF
using Distributions: UnivariateDistribution, MvNormal, Normal, cov, mean

"""
    extract_priors_no_sample(cond_model, values::NamedTuple)

Extract the prior distribution objects for each `~` statement in `cond_model`,
evaluated at `values` for all the sampled quantities. Uses DPPL's accumulator
infrastructure to avoid the default sampling-based path (which triggers
`randn(::Type{SCT.Dual})` when called through the sparse-AD tracer).
"""
function extract_priors_no_sample(cond_model, values::NamedTuple)
    vi = OnlyAccsVarInfo((PriorDistributionAccumulator(),))
    strat = InitFromParams(values, nothing)
    vi = last(init!!(cond_model, vi, strat, UnlinkAll()))
    return getacc(vi, Val(PRIOR_ACCNAME)).values
end

"""
    find_dist(priors, sym::Symbol)

Return the distribution(s) associated with variable `sym` in a priors dict. If
`sym` appears once (as with an MvNormal), returns the distribution directly;
otherwise returns a vector of distributions (loop-built).
"""
function find_dist(priors, sym::Symbol)
    matches = [d for (vn, d) in pairs(priors) if getsym(vn) === sym]
    return length(matches) == 1 ? matches[1] : matches
end

"""
    variable_length(dppl_model, sym::Symbol, hp_values::NamedTuple)

Dimension of the random variable `sym` with hyperparameters fixed to
`hp_values`. Works for both loop-built (counts matches) and atomic cases.
"""
function variable_length(dppl_model, sym::Symbol, hp_values::NamedTuple)
    cond = DynamicPPL.fix(dppl_model, hp_values)
    priors = extract_priors(cond)
    matches = [d for (vn, d) in pairs(priors) if getsym(vn) === sym]
    if length(matches) == 1
        d = matches[1]
        return d isa UnivariateDistribution ? 1 : length(d)
    else
        return length(matches)
    end
end

"""
    classify_sym(dppl_model, sym::Symbol, hp_values::NamedTuple)

Classify variable `sym` as:
- `:atomic_gaussian` — single `~` statement producing a Normal / MvNormal / GMRF
- `:non_gaussian` — single `~` statement with a non-Gaussian distribution
- `:loop_built` — multiple `~` statements (e.g. a for-loop RW prior)
"""
function classify_sym(dppl_model, sym::Symbol, hp_values::NamedTuple)
    cond = DynamicPPL.fix(dppl_model, hp_values)
    priors = extract_priors(cond)
    matches = [d for (vn, d) in pairs(priors) if getsym(vn) === sym]
    if length(matches) == 1
        d = matches[1]
        if d isa AbstractGMRF || d isa MvNormal || d isa Normal
            return :atomic_gaussian
        else
            return :non_gaussian
        end
    else
        return :loop_built
    end
end

"""
    priors_differ(d1, d2)

Whether two prior distributions differ in their mean or covariance structure.
Used to detect causal edges in the DAG analysis.
"""
function priors_differ(d1, d2)
    d1 === d2 && return false
    try
        return !(mean(d1) ≈ mean(d2)) || !(cov(d1) ≈ cov(d2))
    catch
        return d1 != d2
    end
end
