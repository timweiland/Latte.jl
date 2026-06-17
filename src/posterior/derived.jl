export derived

"""
    derived(result, g; n_samples = 1000, rng = Random.default_rng())

Posterior of a *derived quantity*: a (possibly nonlinear) function `g` of the latent field.
Where [`linear_combinations`](@ref) handles linear functionals `z = A x` analytically, `derived`
handles arbitrary `g` by Monte Carlo — it draws `n_samples` from the posterior, evaluates `g`
on each draw, and returns the empirical marginal(s) of the output.

`g` receives a `NamedTuple` mapping each latent group symbol (see [`latent_groups`](@ref)) to
its values for one posterior draw; for a model with no named groups it receives `(; latent)`,
the full latent vector. `g` returns either

- a scalar, in which case `derived` returns a single [`SampleMarginal`](@ref), or
- a vector, in which case it returns a `Vector{SampleMarginal}`, one per output component.

A `SampleMarginal` supports `mean`, `std`, `median`, `quantile`, `mode`, and feeds
[`summary_df`](@ref), so derived quantities summarise and plot like any other marginal. Draws
carry both the hyperparameter and latent-field uncertainty integrated over the posterior.

# Example

Spawning-stock biomass and average fishing mortality from an age-structured assessment, where
`logN`/`logF` are flattened age×year fields (`reshape` to `(nA, nY)`):

```julia
ssb = derived(result; n_samples = 2000) do z
    logN = reshape(z.logN, nA, nY)
    [sum(exp.(logN[:, y])) for y in 1:nY]      # total numbers per year
end
mean.(ssb)               # posterior mean SSB by year
quantile.(ssb, 0.975)    # upper 95% credible bound by year
```
"""
# Two argument orders: function-first enables `derived(result) do z ... end` (the `do` block
# binds the closure as the first positional argument); result-first reads naturally inline.
derived(g::Function, result; kwargs...) = _derived(g, result; kwargs...)
derived(result, g::Function; kwargs...) = _derived(g, result; kwargs...)

function _derived(g, result; n_samples::Int = 1000, rng::AbstractRNG = Random.default_rng())
    n_samples > 0 || throw(ArgumentError("n_samples must be positive, got $n_samples"))
    samples = rand(rng, result, n_samples)
    X = samples.x                                   # n_samples × n_latent
    groups = latent_groups(result)
    syms = Tuple(keys(groups))
    latents(i) = isempty(syms) ?
        (; latent = X[i, :]) :
        NamedTuple{syms}(Tuple(X[i, groups[s]] for s in syms))

    first_out = g(latents(1))
    if first_out isa Number
        vals = Vector{Float64}(undef, n_samples)
        vals[1] = first_out
        for i in 2:n_samples
            vals[i] = g(latents(i))
        end
        return SampleMarginal(vals)
    end

    k = length(first_out)
    out = Matrix{Float64}(undef, n_samples, k)
    out[1, :] .= first_out
    for i in 2:n_samples
        gi = g(latents(i))
        length(gi) == k || throw(
            DimensionMismatch(
                "derived: g returned $(length(gi)) values on draw $i but $k on the first draw; " *
                    "g must return the same length on every draw."
            )
        )
        out[i, :] .= gi
    end
    return [SampleMarginal(out[:, j]) for j in 1:k]
end
