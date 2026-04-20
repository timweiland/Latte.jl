# Workspace speedup benchmark
#
# Measures how much wall-clock time the new persistent-workspace path saves
# in hyperparameter inner loops, as a function of problem size.
#
# Measures per-call hyperparameter_logpdf cost with fresh vs. reused workspace.
# This is the direct measurement of the symbolic-factorization amortization —
# the core optimization Phase 1 enables.
#
# Usage:
#   julia --project=examples examples/workspace_speedup_benchmark.jl
#
# Notes:
# - Uses @elapsed with best-of-3 to smooth out one-off jitter.
# - First call per configuration is discarded as warm-up (JIT compilation).
# - The "cold" path here constructs a fresh GMRFWorkspace per logpdf call,
#   which reproduces the work the pre-Phase-1 code was doing implicitly via
#   the cold-path latent_gmrf(model, θ). It over-counts slightly (the old
#   path didn't go through make_workspace) but is a fair upper bound.

using Latte
using GaussianMarkovRandomFields
using Distributions
using SparseArrays
using LinearAlgebra
using Random
using Printf

# ---------------------------------------------------------------------------
# AR-1 Poisson model (scales trivially with k)
# ---------------------------------------------------------------------------

function build_ar1_model(k::Int)
    spec = @hyperparams begin
        (τ_gmrf ~ LogNormal(1, 1), transform = log, space = natural)
        (η ~ Uniform(0, 1), transform = logit, space = natural)
    end
    function latent_gmrf(; τ_gmrf, η, kwargs...)
        ρ = 2 * η - 1
        Q = spdiagm(-1 => fill(-ρ, k - 1), 0 => fill(1 + ρ^2, k), 1 => fill(-ρ, k - 1))
        Q[1, 1] = 1
        Q[k, k] = 1
        Q .*= τ_gmrf
        return (zeros(k), Q)
    end
    model = LatentGaussianModel(spec, FunctionLatentModel(latent_gmrf, k), ExponentialFamily(Poisson))

    Random.seed!(42)
    σ_true = 0.5
    ρ_true = 0.4
    x_gt = rand(model.latent_prior(; τ_gmrf = 1 / σ_true^2, η = (ρ_true + 1) / 2))
    y_int = Int[rand(Poisson(exp(1.0 + xi))) for xi in x_gt]
    y = PoissonObservations(y_int)

    return model, y, "AR-1 (tridiagonal, k=$k)"
end

function build_laplacian2d_model(n::Int)
    # 2D Laplacian on an n×n grid: 5-point stencil, penta-diagonal precision.
    # k = n²; symbolic factorization has non-trivial cost and fill-in.
    k = n * n
    spec = @hyperparams begin
        (τ_gmrf ~ LogNormal(1, 1), transform = log, space = natural)
    end
    # Pre-build the Laplacian once (pattern is τ-invariant)
    function laplacian_indices(n)
        I = Int[]; J = Int[]; V = Float64[]
        for i in 1:n, j in 1:n
            v = (i - 1) * n + j
            push!(I, v); push!(J, v); push!(V, 4.0)
            for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1))
                ii, jj = i + di, j + dj
                if 1 <= ii <= n && 1 <= jj <= n
                    push!(I, v); push!(J, (ii - 1) * n + jj); push!(V, -1.0)
                end
            end
        end
        return sparse(I, J, V, n * n, n * n)
    end
    L = laplacian_indices(n)
    # A small ridge keeps L SPD
    ridge = 1.0e-2 * I
    function latent_gmrf(; τ_gmrf, kwargs...)
        Q = τ_gmrf .* L + ridge
        return (zeros(k), Q)
    end
    model = LatentGaussianModel(spec, FunctionLatentModel(latent_gmrf, k), ExponentialFamily(Poisson))

    Random.seed!(42)
    x_gt = rand(model.latent_prior(; τ_gmrf = 1.0))
    y_int = Int[rand(Poisson(exp(1.0 + 0.3 * xi))) for xi in x_gt]
    y = PoissonObservations(y_int)

    return model, y, "2D Laplacian ($(n)×$(n), k=$k)"
end

# ---------------------------------------------------------------------------
# Per-call logpdf benchmark
# ---------------------------------------------------------------------------

function benchmark_logpdf(model, y, θ_vec::Vector{<:Vector}, spec; n_calls::Int, best_of::Int = 3)
    # Warm JIT
    θ_w = WorkingHyperparameters(θ_vec[1], spec)
    θ_nt = convert(NamedTuple, convert(NaturalHyperparameters, θ_w))
    ws0 = make_workspace(model.latent_prior; θ_nt...)
    hyperparameter_logpdf(model, θ_w, y; ws = ws0)

    # Warm: one workspace, reused across all calls
    t_warm = Inf
    for _ in 1:best_of
        θ_first = WorkingHyperparameters(θ_vec[1], spec)
        θ_first_nt = convert(NamedTuple, convert(NaturalHyperparameters, θ_first))
        ws = make_workspace(model.latent_prior; θ_first_nt...)
        t = @elapsed for θ_v in θ_vec[1:n_calls]
            θ_w = WorkingHyperparameters(θ_v, spec)
            hyperparameter_logpdf(model, θ_w, y; ws = ws)
        end
        t_warm = min(t_warm, t)
    end

    # Cold: fresh workspace per call (discards symbolic each time)
    t_cold = Inf
    for _ in 1:best_of
        t = @elapsed for θ_v in θ_vec[1:n_calls]
            θ_w = WorkingHyperparameters(θ_v, spec)
            θ_nt = convert(NamedTuple, convert(NaturalHyperparameters, θ_w))
            ws_fresh = make_workspace(model.latent_prior; θ_nt...)
            hyperparameter_logpdf(model, θ_w, y; ws = ws_fresh)
        end
        t_cold = min(t_cold, t)
    end

    return (warm = t_warm, cold = t_cold, speedup = t_cold / t_warm)
end
# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

# θ perturbations around the prior mean — used for the per-call measurement
function sample_θ_vec(spec; n::Int = 20)
    Random.seed!(1)
    θ_vec = Vector{Float64}[]
    for _ in 1:n
        push!(θ_vec, randn(length(keys(spec.free))) .* 0.2)
    end
    return θ_vec
end

println("="^78)
println("Workspace speedup benchmark")
println("="^78)
println()
println("Measures the symbolic-factorization amortization in a tight loop of")
println("hyperparameter_logpdf calls: same model, varying θ, N calls.")
println()

n_calls = 20

function run_case(builder, label_header::String, sizes)
    println(label_header)
    @printf "  %-34s │ %14s │ %14s │ %10s\n"  "config"  "per-call cold"  "per-call warm"  "speedup"
    @printf "  %-34s─┼─%14s─┼─%14s─┼─%10s\n"  "─"^34  "─"^14  "─"^14  "─"^10
    for s in sizes
        model, y, label = builder(s)
        spec = model.hyperparameter_spec
        θ_vec = sample_θ_vec(spec; n = n_calls)
        t = benchmark_logpdf(model, y, θ_vec, spec; n_calls = n_calls)
        @printf "  %-34s │ %12.3f ms │ %12.3f ms │ %8.2f×\n"  label  t.cold * 1.0e3 / n_calls  t.warm * 1.0e3 / n_calls  t.speedup
    end
    return println()
end

run_case(
    build_ar1_model, "AR-1 (tridiagonal — symbolic is ~free):",
    (100, 1_000, 5_000, 10_000)
)

run_case(
    build_laplacian2d_model, "2D Laplacian (penta-diagonal, non-trivial symbolic + fill-in):",
    (10, 25, 50, 100)
)

println("Legend:")
println("  per-call cold : fresh workspace each call — reproduces pre-Phase-1 work")
println("                  (symbolic + numeric Cholesky every iteration)")
println("  per-call warm : workspace reused across calls (Phase 1) —")
println("                  symbolic done once, numeric-only refactorization each call")
println("  speedup       : cold / warm — dominated by symbolic-Cholesky savings.")
println("                  AR-1's tridiagonal has trivial symbolic, so speedup is modest.")
println("                  Realistic spatial models (2D Laplacian, SPDE Matern) pay")
println("                  much more symbolic cost, so the workspace wins are bigger.")
