# Hand-built LGM constructors shared by the engine and diagnostics suites. Defining the
# closure once gives every file the same `latent_func` type, so the inference pipeline
# compiles one specialization instead of one per file (and the method-overwrite warnings
# from per-file redefinition disappear). Include via:
#     isdefined(@__MODULE__, :make_poisson_iid_model) || include(<relative path>)

using Latte
using Distributions
using GaussianMarkovRandomFields
using SparseArrays

function make_poisson_iid_model(n)
    spec = @hyperparams begin
        (τ ~ Gamma(2, 1), transform = log, space = natural)
    end
    function poisson_iid_latent(; τ, kwargs...)
        Q = spdiagm(0 => fill(τ, n))
        return (zeros(n), Q)
    end
    obs_model = ExponentialFamily(Poisson)
    return LatentGaussianModel(spec, FunctionLatentModel(poisson_iid_latent, n), obs_model)
end

function make_normal_iid_model(n)
    spec = @hyperparams begin
        (σ ~ InverseGamma(2, 1), transform = log, space = natural)
    end
    function normal_iid_latent(; σ, kwargs...)
        Q = spdiagm(0 => fill(1 / σ^2, n))
        return (zeros(n), Q)
    end
    obs_model = ExponentialFamily(Normal)
    return LatentGaussianModel(spec, FunctionLatentModel(normal_iid_latent, n), obs_model)
end
