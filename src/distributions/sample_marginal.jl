using Distributions
using KernelDensity
using Random
using StatsBase: ecdf
import Statistics

export SampleMarginal

"""
    SampleMarginal <: ContinuousUnivariateDistribution

A 1D marginal distribution defined empirically by a vector of samples.

Moments, quantiles, and CDF are computed from the samples directly
(`Statistics.mean/var/quantile`, `StatsBase.ecdf`). The PDF is a kernel
density estimate (built once and cached). `rand` bootstraps from the
stored samples.

Intended for posterior marginals from sample-based inference (e.g.
HMC-on-Laplace), giving the same `Distribution` interface that INLA's
spline marginals and TMB's Gaussian marginals provide.
"""
mutable struct SampleMarginal{T <: AbstractFloat} <: ContinuousUnivariateDistribution
    samples::Vector{T}
    _ecdf::Any
    _kde::Any
end

function SampleMarginal(samples::AbstractVector{<:Real})
    s = collect(Float64, samples)
    isempty(s) && throw(ArgumentError("SampleMarginal needs at least one sample"))
    return SampleMarginal{Float64}(s, nothing, nothing)
end

function _get_ecdf(d::SampleMarginal)
    d._ecdf === nothing && (d._ecdf = ecdf(d.samples))
    return d._ecdf
end

function _get_kde(d::SampleMarginal)
    d._kde === nothing && (d._kde = kde(d.samples))
    return d._kde
end

Distributions.mean(d::SampleMarginal) = Statistics.mean(d.samples)
Distributions.var(d::SampleMarginal) = Statistics.var(d.samples)
Distributions.std(d::SampleMarginal) = Statistics.std(d.samples)
Distributions.median(d::SampleMarginal) = Statistics.median(d.samples)
Distributions.quantile(d::SampleMarginal, q::Real) = Statistics.quantile(d.samples, q)
Distributions.minimum(d::SampleMarginal) = minimum(d.samples)
Distributions.maximum(d::SampleMarginal) = maximum(d.samples)
Distributions.insupport(d::SampleMarginal, x::Real) =
    isfinite(x) && minimum(d) <= x <= maximum(d)

Distributions.cdf(d::SampleMarginal, x::Real) = _get_ecdf(d)(x)

function Distributions.pdf(d::SampleMarginal, x::Real)
    k = _get_kde(d)
    (x < first(k.x) || x > last(k.x)) && return 0.0
    return max(KernelDensity.pdf(k, x), 0.0)
end
Distributions.logpdf(d::SampleMarginal, x::Real) = log(pdf(d, x))

# Bootstrap a draw from the stored samples. Matches what MCMC users expect
# from "rand on a posterior-samples object" and avoids imposing a KDE
# smoothing assumption on draws.
Base.rand(rng::AbstractRNG, d::SampleMarginal) = d.samples[rand(rng, 1:length(d.samples))]
Base.rand(d::SampleMarginal) = rand(Random.default_rng(), d)
