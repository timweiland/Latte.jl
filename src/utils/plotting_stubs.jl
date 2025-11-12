# Plotting functionality stubs
# These are implemented in the Makie extension when Makie is loaded

export joyplot, joyplot!

"""
    joyplot(distributions::Vector; kwargs...)
    joyplot!(ax, distributions::Vector; kwargs...)

Create a joy plot (ridgeline plot) for multiple INLA marginal distributions.

**Note**: This function requires Makie to be loaded. Please run `using CairoMakie`
(or `using GLMakie`/`using WGLMakie`) before using this function.

# Example
```julia
using IntegratedNestedLaplace
using CairoMakie  # Load Makie to enable plotting

result = inla(...)
dists = [result.latent_marginals[i] for i in 10:10:100]
joyplot(dists)
```

See the documentation in the Makie extension for full details on available options.
"""
function joyplot(args...; kwargs...)
    error("joyplot requires Makie to be loaded.")
end

"""
    joyplot!(ax, distributions::Vector; kwargs...)

In-place version of `joyplot` that plots into an existing axis.

**Note**: This function requires Makie to be loaded. Please run `using CairoMakie`
(or `using GLMakie`/`using WGLMakie`) before using this function.
"""
function joyplot!(args...; kwargs...)
    error("joyplot! requires Makie to be loaded.")
end
