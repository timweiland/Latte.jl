# Include modular components in dependency order
include("adaptive_hessian.jl")  # Must come before transformation.jl
include("transformation.jl")
include("types.jl")
include("utils.jl")        # Must come before algorithms
include("algorithms.jl")   # Uses functions from utils.jl
include("ccd.jl")          # CCD exploration strategy
