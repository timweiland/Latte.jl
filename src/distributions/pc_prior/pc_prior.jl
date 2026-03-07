module PCPrior

    using Distributions
    using Random

    import Distributions: ContinuousUnivariateDistribution, RealInterval, logpdf, support, mode

    include("common.jl")
    include("precision.jl")
    include("sigma.jl")
    include("ar1_correlation.jl")
    include("bym_proportion.jl")

end # module PCPrior

export PCPrior
