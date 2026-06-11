# Settle whether the check_sparse / Sparse-conversion overhead in GMRFs
# refactorize! is real on the ACTUAL posterior precision Q_post = Q_prior + A'WA
# (the matrix the GA factorizes), not a prior-only proxy. Builds the real GA at a
# mesh level, extracts Q_post, and times:
#   A) cholesky!(F, Symmetric(Q))    — current path (Sparse(A) + check_sparse + factorize)
#   B) cholesky!(F, S; check=false)  — S built once (factorize only)
#   + direct check_sparse(Sparse(Q)) cost and nnz/row.
#
#   julia --project=benchmark benchmark/scaling/chol_post_bench.jl [level=4]

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Latte, GaussianMarkovRandomFields, Distributions, DynamicPPL
using LinearAlgebra, SparseArrays, CSV, DataFrames, Printf
using Ferrite, FerriteGmsh, Gmsh, LibGEOS
using SparseArrays.CHOLMOD: Sparse, check_sparse

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

θ_star, _, _, _ = find_hyperparameter_mode(model, yp; progress_callback = nothing)
θ_nt = convert(NamedTuple, convert(NaturalHyperparameters, θ_star))
pool = Latte.make_workspace_pool(model.latent_prior; size = 1, θ_nt...)

bel(f, n = 30) = (f(); minimum(@elapsed(f()) for _ in 1:n))

Latte.with_workspace(pool) do ws
    buf = Ref(Float64[])
    hyperparameter_logpdf(model, WorkingHyperparameters(collect(θ_star.θ), spec), yp; ws = ws, mode_out = buf)
    x_star = copy(buf[])
    obs_lik = model.observation_model(yp; θ_nt...)
    latent_prior = Latte.latent_gmrf(model, ws, θ_nt)
    x_G = gaussian_approximation(latent_prior, obs_lik; x0 = x_star)

    # Extract the actual posterior precision the GA factorizes.
    Qsp = sparse(precision_map(x_G))
    Q = issymmetric(Qsp) ? Symmetric(Qsp) : Symmetric(Qsp + Qsp' - Diagonal(Qsp))
    n = size(Q, 1)
    println("=== posterior precision Q_post (level $lev, n=$n) ===")
    @printf(
        "nnz(Q)/row = %.1f   (prior precision nnz/row = %.1f)\n",
        nnz(sparse(Q)) / n, nnz(sparse(precision_matrix(base_matern; τ = 1.0, range = 0.3))) / length(base_matern)
    )

    F = cholesky(Q)
    S = Sparse(Q)
    fA() = cholesky!(F, Q)
    fB() = cholesky!(F, S; check = false)
    fchk() = check_sparse(Sparse(Q))
    fconv() = Sparse(Q)

    fA(); ldA = logdet(F); fB(); ldB = logdet(F)
    tA = bel(fA); tB = bel(fB); tchk = bel(fchk); tconv = bel(fconv)
    @printf("\nA: chol!(Symmetric(Q))      : %7.2f ms\n", tA * 1.0e3)
    @printf(
        "B: chol!(Sparse,check=false): %7.2f ms   (%.2fx faster, |Δlogdet|=%.1e)\n",
        tB * 1.0e3, tA / tB, abs(ldA - ldB)
    )
    @printf("   Sparse(Q) conversion     : %7.2f ms\n", tconv * 1.0e3)
    @printf("   check_sparse(Sparse(Q))  : %7.2f ms\n", tchk * 1.0e3)
    @printf("   => removable overhead    : %7.2f ms  (%.0f%% of A)\n", (tA - tB) * 1.0e3, 100 * (tA - tB) / tA)
end
