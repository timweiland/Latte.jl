# Decompose a single WARM primal hyperparameter_logpdf eval (the mode-finding /
# exploration unit of work) into its components, and count sparse Cholesky
# factorizations per eval. Mode-finding = ~10 BFGS iters x (value + gradient),
# each ~3.6x a primal eval, so every redundant factorization here multiplies.
#
#   julia --project=benchmark benchmark/scaling/ga_eval_profile.jl [level=4]

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Latte, GaussianMarkovRandomFields, Distributions, DynamicPPL
using LinearAlgebra, SparseArrays, CSV, DataFrames, Printf, Statistics, Profile
using Ferrite, FerriteGmsh, Gmsh, LibGEOS

const WORKDIR = joinpath(@__DIR__, "_workdir")
lev = isempty(ARGS) ? 4 : parse(Int, ARGS[1])
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

println("=== GA-eval profile (level $lev, n=", length(base_matern), ", m=", length(y), ") ===")
flush(stdout)

θ_star, _, _, _ = find_hyperparameter_mode(model, yp; progress_callback = nothing)
θ_nt = convert(NamedTuple, convert(NaturalHyperparameters, θ_star))
pool = Latte.make_workspace_pool(model.latent_prior; size = 1, θ_nt...)

bel(f, n = 8) = (f(); minimum(@elapsed(f()) for _ in 1:n))

Latte.with_workspace(pool) do ws
    # warm seed: the GA mode at θ*
    buf = Ref(Float64[])
    hyperparameter_logpdf(model, WorkingHyperparameters(collect(θ_star.θ), spec), yp; ws = ws, mode_out = buf)
    x_star = copy(buf[])

    # full warm eval baseline
    t_full = bel(() -> hyperparameter_logpdf(model, WorkingHyperparameters(collect(θ_star.θ), spec), yp; ws = ws, x0 = x_star))

    # --- component decomposition (mirrors hyperparameter_logpdf body) ---
    t_obs = bel(() -> model.observation_model(yp; θ_nt...))
    t_prior = bel(() -> Latte.latent_gmrf(model, ws, θ_nt))
    latent_prior = Latte.latent_gmrf(model, ws, θ_nt)
    obs_lik = model.observation_model(yp; θ_nt...)
    t_ga = bel(() -> gaussian_approximation(latent_prior, obs_lik; x0 = x_star))
    x_G = gaussian_approximation(latent_prior, obs_lik; x0 = x_star)
    t_lp_prior = bel(() -> logpdf(latent_prior, x_star))   # logdet(Q_prior)
    t_loglik = bel(() -> loglik(x_star, obs_lik))
    t_lp_ga = bel(() -> logpdf(x_G, x_star))               # logdet(Q_post)

    @printf("\nfull warm eval        : %7.2f ms  (100%%)\n", t_full * 1.0e3)
    comps = [
        ("observation_model(y)", t_obs),
        ("latent_gmrf (prior)", t_prior),
        ("gaussian_approximation", t_ga),
        ("logpdf(prior, x*)", t_lp_prior),
        ("loglik(x*, obs)", t_loglik),
        ("logpdf(GA, x*)", t_lp_ga),
    ]
    for (lab, t) in comps
        @printf("  %-22s: %7.2f ms  (%4.1f%%)\n", lab, t * 1.0e3, 100 * t / t_full)
    end
    @printf("  %-22s: %7.2f ms\n", "[sum of components]", sum(last, comps) * 1.0e3)
    flush(stdout)

    # --- profile: count Cholesky / factorization frames over many warm evals ---
    Profile.clear()
    Profile.init(n = 10^8, delay = 0.0005)
    Profile.@profile for _ in 1:200
        hyperparameter_logpdf(model, WorkingHyperparameters(collect(θ_star.θ), spec), yp; ws = ws, x0 = x_star)
    end
    open("/tmp/ga_eval_prof.txt", "w") do io
        Profile.print(IOContext(io, :displaysize => (240, 320)); format = :flat, sortedby = :count, mincount = 20)
    end
    println("\nflat profile (200 warm evals) -> /tmp/ga_eval_prof.txt")
end
