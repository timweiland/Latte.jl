# Threaded vs Sequential speedup for full inla().
#
# Uses the AR-1 Poisson reference problem. Shows what users can expect on
# their machine when they pass ThreadedExecutor through.
#
# Usage:
#   JULIA_NUM_THREADS=4 julia --project=examples examples/threaded_speedup_benchmark.jl
#
# Note: observed speedup is bounded by CHOLMOD's global factorization lock.
# For very symbolic-heavy workloads (e.g. SPDE Matern on dense meshes), the
# `CliqueTreesBackend` upstream is fully thread-safe and gives near-linear
# scaling — configure via GMRFs.jl if you need to push past the ~2× wall.

using TestEnv;
TestEnv.activate()
using IntegratedNestedLaplace
using GaussianMarkovRandomFields
using Distributions
using SparseArrays
using JLD2
using Random
using Printf

reference_file = joinpath(@__DIR__, "..", "test", "end_to_end", "ar1_poisson", "reference_data.jld2")
if !isfile(reference_file)
    error("Reference data not found at $reference_file. Run `make generate-reference` first.")
end
@load reference_file y_gt model_params
k = length(y_gt)

spec = @hyperparams begin
    (τ_gmrf ~ Normal(0, 1), transform = log, space = working)
    (η ~ Normal(atanh(0.95), model_params.desired_std_dev), transform = identity, space = working)
end

function ar_precision(ρ, k)
    return spdiagm(-1 => -ρ * ones(k - 1), 0 => ones(k) .+ ρ^2, 1 => -ρ * ones(k - 1))
end

function latent_gmrf(; τ_gmrf, η, kwargs...)
    ρ = tanh(η)
    Q = ar_precision(ρ, k) .* τ_gmrf
    μ₀ = log(1000.0)
    μ = μ₀ .* [ρ^i for i in 1:k]
    return (μ, Q)
end

model = INLAModel(spec, FunctionLatentModel(latent_gmrf, k), ExponentialFamily(Poisson))
y = PoissonObservations(y_gt)

function bench(exec; n = 3)
    # warm JIT
    inla(
        model, y; progress = false,
        latent_marginalization_method = SimplifiedLaplace(),
        hyperparameter_marginalization_method = AutoHyperparameterMarginal(),
        executor = exec,
    )
    t = Inf
    for _ in 1:n
        GC.gc()
        s = @timed inla(
            model, y; progress = false,
            latent_marginalization_method = SimplifiedLaplace(),
            hyperparameter_marginalization_method = AutoHyperparameterMarginal(),
            executor = exec,
        )
        t = min(t, s.time)
    end
    return t
end

nthreads = Threads.nthreads()
println("="^70)
println("Threaded vs Sequential speedup (AR-1 Poisson, k=$k, SimplifiedLaplace)")
println("Julia threads available: $nthreads")
println("="^70)
println()

@printf "%-22s │ %12s │ %10s\n"  "executor"  "wall-clock"  "speedup"
@printf "%-22s─┼─%12s─┼─%10s\n"   "─"^22 "─"^12 "─"^10

t_seq = bench(SequentialExecutor())
@printf "%-22s │ %10.3f s │     1.00×\n"  "SequentialExecutor"  t_seq

for nw in (2, 4, 8)
    nw > nthreads && continue
    t = bench(ThreadedExecutor(nworkers = nw))
    @printf "%-22s │ %10.3f s │     %4.2f×\n"  "ThreadedExecutor($nw)"  t  t_seq / t
end

println()
println("Notes:")
println("  - CHOLMOD (default backend) serializes numeric factorizations via a global lock.")
println("    Speedup plateau of ~2× is expected. Switch to CliqueTreesBackend for linear scaling.")
println("  - Sequential and Threaded give bit-for-bit identical results (tested in test/parallel/).")
