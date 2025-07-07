# Include modular components in dependency order
include("transformation.jl")
include("types.jl")
include("utils.jl")        # Must come before algorithms
include("algorithms.jl")   # Uses functions from utils.jl
