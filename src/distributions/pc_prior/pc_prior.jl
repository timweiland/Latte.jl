"""
    PCPrior

Namespace for penalized-complexity priors. Build a prior by calling the
constructor for the parameterization you need — `PCPrior.precision(...)`,
`PCPrior.sigma(...)`, `PCPrior.range(...)`, `PCPrior.ar1_correlation(...)`,
`PCPrior.bym_proportion(...)` — each returning a `ContinuousUnivariateDistribution`.
"""
module PCPrior

    using Distributions
    using Bijectors

    using Random

    import Distributions: ContinuousUnivariateDistribution, RealInterval, logpdf, support, mode

    include("common.jl")
    include("precision.jl")
    include("sigma.jl")
    include("range.jl")
    include("ar1_correlation.jl")
    include("bym_proportion.jl")

end # module PCPrior

export PCPrior
