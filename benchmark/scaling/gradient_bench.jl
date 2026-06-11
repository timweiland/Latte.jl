# Is the mode-finding gap a ForwardDiff overhead, or the irreducible GA cost?
# R-INLA gets its hyperparameter gradient from central finite differences
# (~2k+1 objective evals for k hyperparameters). Latte uses ForwardDiff. AD
# *should* be cheaper per gradient than FD here (k=2 ⇒ one Dual{2} pass vs ~4
# primal evals) — unless the Dual path through the GA carries overhead.
#
# This measures, on the actual mode-finding objective at a real mesh level:
#   (a) per-gradient cost: 1 primal eval vs ForwardDiff vs FiniteDiff (central),
#       both warm (x0 = mode, the in-optimisation regime) and cold,
#   (b) end-to-end find_hyperparameter_mode under ADStrategy vs FiniteDiffStrategy.
#
#   julia --project=benchmark benchmark/scaling/gradient_bench.jl [level=3]

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Latte, GaussianMarkovRandomFields, Distributions, DynamicPPL
using LinearAlgebra, SparseArrays, CSV, DataFrames, Printf, Statistics
using Ferrite, FerriteGmsh, Gmsh, LibGEOS
using ForwardDiff, FiniteDiff

const WORKDIR = joinpath(@__DIR__, "_workdir")
lev = isempty(ARGS) ? 3 : parse(Int, ARGS[1])
d = joinpath(WORKDIR, "mesh_$lev")

function load_disc(dir)
    nodes = CSV.read(joinpath(dir, "nodes.csv"), DataFrame)
    tris = CSV.read(joinpath(dir, "triangles.csv"), DataFrame)
    fn = [Ferrite.Node((nodes.x[i], nodes.y[i])) for i in 1:nrow(nodes)]
    el = [Ferrite.Triangle((tris.v1[i], tris.v2[i], tris.v3[i])) for i in 1:nrow(tris)]
    return FEMDiscretization(Ferrite.Grid(el, fn), Ferrite.Lagrange{Ferrite.RefTriangle, 1}(), Ferrite.QuadratureRule{Ferrite.RefTriangle}(2))
end

@latte function scaling_poisson(y, base_matern, A_obs)
    τ_matern ~ PCPrior.Precision(1.0; α = 0.5)
    range_matern ~ PCPrior.Range(0.3; p = 0.5)
    β ~ MvNormal(zeros(1), 100.0 * I(1))
    field ~ base_matern(τ = τ_matern, range = range_matern)
    η = β[1] .+ A_obs * field
    for i in eachindex(y)
        y[i] ~ Poisson(exp(η[i]); check_args = false)
    end
end

disc = load_disc(d)
base_matern = MaternModel(disc; smoothness = 0)
obs = CSV.read(joinpath(d, "obs_coords.csv"), DataFrame)
coords = Matrix(hcat(obs.s1, obs.s2))
A_obs = evaluation_matrix(base_matern, coords)
y = CSV.read(joinpath(d, "y.csv"), DataFrame).y
lgm = scaling_poisson(y, base_matern, A_obs)
yp, model, _ = Latte._prepare_for_prediction(lgm, y)
spec = model.hyperparameter_spec

println("=== gradient bench (level $lev, n=", length(base_matern), ", m=", length(y), ") ===")
flush(stdout)

# Find the mode so we benchmark at a representative interior θ*, and grab the
# GA mode there as the warm-start seed.
θ_star, _, _, _ = find_hyperparameter_mode(model, yp; progress_callback = nothing)
θvec = collect(θ_star.θ)
println("k = ", length(θvec), " hyperparameters;  θ* = ", round.(θvec; digits = 4))

θ0_nt = convert(NamedTuple, convert(NaturalHyperparameters, θ_star))
pool = Latte.make_workspace_pool(model.latent_prior; size = 1, θ0_nt...)

bel(f, n = 5) = (f(); minimum(@elapsed(f()) for _ in 1:n))

Latte.with_workspace(pool) do ws
    buf = Ref(Float64[])
    # warm-seed objective: x0 = mode (the regime the optimiser spends time in)
    f_warm = let
        mode = buf
        θv -> -hyperparameter_logpdf(
            model, WorkingHyperparameters(θv, spec), yp;
            ws = ws, x0 = isempty(mode[]) ? nothing : mode[]
        )
    end
    # populate the warm seed: one primal eval at θ* with mode_out
    hyperparameter_logpdf(
        model, WorkingHyperparameters(θvec, spec), yp;
        ws = ws, mode_out = buf
    )
    x_star = copy(buf[])
    f_cold = θv -> -hyperparameter_logpdf(model, WorkingHyperparameters(θv, spec), yp; ws = ws)

    # --- agreement check: AD vs FD gradients must match ---
    g_ad = ForwardDiff.gradient(f_warm, θvec)
    g_fd = FiniteDiff.finite_difference_gradient(f_warm, θvec)   # central by default
    @printf(
        "\ngradient agreement (warm): AD=%s  FD=%s  max|Δ|=%.2e\n",
        string(round.(g_ad; digits = 5)), string(round.(g_fd; digits = 5)),
        maximum(abs.(g_ad .- g_fd))
    )

    for (label, f) in (("WARM (x0 = mode)", f_warm), ("COLD (x0 = nothing)", f_cold))
        t_primal = bel(() -> f(θvec))
        t_ad = bel(() -> ForwardDiff.gradient(f, θvec), 3)
        t_fd = bel(() -> FiniteDiff.finite_difference_gradient(f, θvec), 3)
        @printf("\n[%s]\n", label)
        @printf("  primal eval        : %8.2f ms\n", t_primal * 1.0e3)
        @printf("  ForwardDiff grad   : %8.2f ms   (%.2fx primal)\n", t_ad * 1.0e3, t_ad / t_primal)
        @printf("  FiniteDiff  grad   : %8.2f ms   (%.2fx primal)\n", t_fd * 1.0e3, t_fd / t_primal)
        @printf(
            "  AD / FD            : %.2fx  %s\n", t_ad / t_fd,
            t_ad < t_fd ? "(AD faster)" : "(FD faster)"
        )
    end
    flush(stdout)
end

# --- end-to-end: full mode-finding under each strategy ---
println("\n=== end-to-end find_hyperparameter_mode ===")
function run_mode(strategy)
    res = find_hyperparameter_mode(model, yp; diff_strategy = strategy, progress_callback = nothing)
    θ, pts, _, _ = res
    return (θ = collect(θ.θ), nevals = length(pts))
end
# warmup both paths
run_mode(ADStrategy()); run_mode(FiniteDiffStrategy())
for (label, strat) in (
        ("ADStrategy   (ForwardDiff)", ADStrategy()),
        ("FiniteDiffStrategy        ", FiniteDiffStrategy()),
    )
    t = @elapsed r = run_mode(strat)
    @printf(
        "  %s : %7.3f s   θ*=%s   (%d primal evals collected)\n",
        label, t, string(round.(r.θ; digits = 4)), r.nevals
    )
end
