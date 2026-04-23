module PCPrior

    using Distributions
    using Bijectors

    using Random

    import Distributions: ContinuousUnivariateDistribution, RealInterval, logpdf, support, mode

    include("common.jl")
    include("precision.jl")
    include("sigma.jl")
    include("ar1_correlation.jl")
    include("bym_proportion.jl")

    # Support on (0, ∞) ⇒ log transform to working space.
    Bijectors.bijector(::Precision) = Bijectors.elementwise(log)

end # module PCPrior

export PCPrior
