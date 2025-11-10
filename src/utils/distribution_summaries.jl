"""
Utilities for summarizing distributions into DataFrames.
"""

using DataFrames
using Distributions

export summary_df

"""
    summary_df(marginals::AbstractVector{<:Distribution})

Create a summary DataFrame of marginal distributions with key statistics.

# Arguments
- `marginals::AbstractVector`: Vector of distributions (e.g., latent marginals)

# Returns
DataFrame with one row per distribution containing:
- `index`: Index in the vector
- `mode`: Mode of the marginal
- `median`: Median (50th percentile)
- `q2_5`: 2.5th percentile
- `q97_5`: 97.5th percentile
- `mean`: Mean
- `std`: Standard deviation

# Example
```julia
result = inla(model, y)
df = summary_df(result.latent_marginals)
```
"""
function summary_df(marginals::AbstractVector{<:Distribution})
    # Compute statistics for each marginal
    modes = Float64[]
    medians = Float64[]
    q025s = Float64[]
    q975s = Float64[]
    means = Float64[]
    stds = Float64[]

    for marginal in marginals
        push!(modes, mode(marginal))
        push!(medians, median(marginal))
        push!(q025s, quantile(marginal, 0.025))
        push!(q975s, quantile(marginal, 0.975))
        push!(means, mean(marginal))
        push!(stds, std(marginal))
    end

    return DataFrame(
        mode = modes,
        median = medians,
        q2_5 = q025s,
        q97_5 = q975s,
        mean = means,
        std = stds
    )
end
