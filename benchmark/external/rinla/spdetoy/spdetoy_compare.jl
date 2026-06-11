# SPDEtoy: Latte INLA vs R-INLA on a Matérn-SPDE model, on the SAME mesh.
#
# This is the SPDE cross-validation. To isolate the inference comparison from
# meshing differences, both engines solve the SAME discretized problem: R-INLA
# builds the mesh (inla.mesh.2d) and dumps nodes.csv + triangles.csv; this
# script rebuilds the identical Ferrite grid + FEM discretization and fits the
# matching Matérn-SPDE Gaussian model. We compare per-node field marginals, the
# intercept, and the three hyperparameter posteriors (obs SD, field range, field
# Stdev), plus warm/cold timing.
#
# Dependency note (vs the other scenarios): the shared mesh is produced BY the R
# run, so R must run first. main() ensures the reference CSVs exist before Latte
# builds its discretization.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using CSV
using DataFrames
using Distributions
using DynamicPPL                                   # @latte's expansion references the module
using GaussianMarkovRandomFields
using GaussianMarkovRandomFields: FEMDiscretization, evaluation_matrix, node_selection_matrix
using Ferrite, FerriteGmsh, Gmsh, LibGEOS          # activate the FEM extension
using JSON3
using Latte
using LinearAlgebra
using Printf
using Profile
using SparseArrays
using Statistics
import Random

const SPDETOY_CSV = joinpath(@__DIR__, "spdetoy_data.csv")
const WORKDIR = joinpath(@__DIR__, "_workdir")

# The Matérn range prior is `PCPrior.Range` (prototyped in Latte for this work;
# matches R-INLA's inla.spde2.pcmatern prior.range — in 2D the SPDE PC range
# prior is exactly 1/range ~ Exponential(-ρ0·log p)).

# ── KS helper (verbatim from tokyo_compare.jl), plus a cdf-function variant ────
function _ks_cdf(cdf_fn, ref_grid::Vector{Float64}, ref_density::Vector{Float64})
    n = length(ref_grid)
    cdf_vals = Vector{Float64}(undef, n)
    cdf_vals[1] = 0.0
    for k in 2:n
        cdf_vals[k] = cdf_vals[k - 1] +
            0.5 * (ref_density[k - 1] + ref_density[k]) * (ref_grid[k] - ref_grid[k - 1])
    end
    Z = cdf_vals[end]
    Z > 0 && (cdf_vals ./= Z)
    best_abs = 0.0
    best_signed = 0.0
    for k in 1:n
        gap = cdf_fn(ref_grid[k]) - cdf_vals[k]
        if abs(gap) > best_abs
            best_abs = abs(gap)
            best_signed = gap
        end
    end
    return best_abs, best_signed
end
_ks_density(engine, g, d) = _ks_cdf(x -> cdf(engine, x), g, d)

# ── data + shared mesh ─────────────────────────────────────────────────────────
function load_spdetoy()
    df = CSV.read(SPDETOY_CSV, DataFrame)
    coords = Matrix{Float64}(hcat(df.s1, df.s2))
    return (n = nrow(df), coords = coords, y = Vector{Float64}(df.y))
end

# Rebuild R-INLA's exact mesh as a Ferrite grid → FEM discretization.
function load_discretization(workdir)
    nodes = CSV.read(joinpath(workdir, "nodes.csv"), DataFrame)
    tris = CSV.read(joinpath(workdir, "triangles.csv"), DataFrame)
    ferrite_nodes = [Ferrite.Node((nodes.x[i], nodes.y[i])) for i in 1:nrow(nodes)]
    elements = [Ferrite.Triangle((tris.v1[i], tris.v2[i], tris.v3[i])) for i in 1:nrow(tris)]
    grid = Ferrite.Grid(elements, ferrite_nodes)
    ip = Ferrite.Lagrange{Ferrite.RefTriangle, 1}()
    qr = Ferrite.QuadratureRule{Ferrite.RefTriangle}(2)
    disc = FEMDiscretization(grid, ip, qr)
    return disc, nrow(nodes)
end

# node id (R order) → Latte field DOF. Lagrange-P1 ⇒ one DOF per node.
function node_to_dof(disc, n_nodes)
    S = node_selection_matrix(disc, collect(1:n_nodes))
    I_, J_, _ = findnz(S)
    perm = Vector{Int}(undef, n_nodes)
    for t in eachindex(I_)
        perm[I_[t]] = J_[t]
    end
    return perm
end

# ── model ──────────────────────────────────────────────────────────────────────
@latte function spdetoy_model(y, base_matern, A_obs, p)
    σ ~ PCPrior.Sigma(p.sigma_obs_U; α = p.sigma_obs_alpha)
    τ_matern ~ PCPrior.Precision(p.sigma_field_U; α = p.sigma_field_p)
    range_matern ~ PCPrior.Range(p.range_U; p = p.range_p)
    β ~ MvNormal(zeros(1), (1 / p.prec_intercept) * I(1))
    field ~ base_matern(τ = τ_matern, range = range_matern)
    η = β[1] .+ A_obs * field
    for i in eachindex(y)
        y[i] ~ Normal(η[i], σ)
    end
end

# read a 2-column (x, density) R marginal CSV
_read_marg(path) = (df = CSV.read(path, DataFrame); (Vector{Float64}(df.x), Vector{Float64}(df.density)))

function main(args::Vector{String} = ARGS)
    mkpath(WORKDIR)
    pj = joinpath(WORKDIR, "params.json")
    isfile(pj) || error("missing $pj — write the matched-prior params first")
    p_raw = JSON3.read(read(pj, String))
    p = (
        sigma_obs_U = Float64(p_raw.sigma_obs_U), sigma_obs_alpha = Float64(p_raw.sigma_obs_alpha),
        sigma_field_U = Float64(p_raw.sigma_field_U), sigma_field_p = Float64(p_raw.sigma_field_p),
        range_U = Float64(p_raw.range_U), range_p = Float64(p_raw.range_p),
        prec_intercept = Float64(p_raw.prec_intercept),
    )

    # R-INLA reference + shared mesh must exist first (dependency inversion).
    if !isfile(joinpath(WORKDIR, "rinla_field_summary.csv")) || "--refresh-rinla" in args
        @info "running R-INLA (produces the shared mesh + reference marginals)"
        rscript = joinpath(@__DIR__, "spdetoy_compare.R")
        run(`Rscript $rscript $(WORKDIR) $(WORKDIR)`)
    end
    rinla_meta = JSON3.read(read(joinpath(WORKDIR, "rinla_meta.json"), String))

    data = load_spdetoy()
    disc, n_nodes = load_discretization(WORKDIR)
    @info "mesh rebuilt" n_nodes n_obs = data.n

    A_obs = evaluation_matrix(disc, data.coords)
    rowsums = vec(sum(A_obs; dims = 2))
    @assert all(>(0.99), rowsums) && all(<(1.01), rowsums) "some obs points fell outside the mesh (A rows ≉ 1): extrema=$(extrema(rowsums))"

    # GMRFs smoothness_to_ν(s, D=2) = s + 1, so smoothness=0 ⇒ ν=1 ⇒ alpha=2,
    # matching R-INLA's inla.spde2.pcmatern(alpha=2).
    base_matern = MaternModel(disc; smoothness = 0)
    perm = node_to_dof(disc, n_nodes)

    # ── Latte fit: cold once, warm median-of-5 ──
    # Gaussian observations ⇒ the latent posterior is exactly Gaussian, so
    # GaussianMarginal() is exact and avoids the (unnecessary, slow) spline
    # augmentation the SimplifiedLaplace default would add.
    augmented = "--augmented" in args
    # R-INLA (as called) computes neither DIC/WAIC/CPO; Latte's default does. For a
    # fair comparison (and to test the lever), --lean-accum keeps only the marginal
    # likelihood INLA needs.
    accum = "--lean-accum" in args ? (MarginalLogLikelihoodStrategy(),) :
        (DICStrategy(), MarginalLogLikelihoodStrategy(), WAICStrategy(), CPOStrategy())
    @info "running Latte INLA (GaussianMarginal — exact for Gaussian obs)" augment = augmented accumulators = length(accum)
    lgm = spdetoy_model(data.y, base_matern, A_obs, p; augment = augmented)

    # --inspect: what LatentGaussianModel did @latte construct? Recognized
    # sparse latent + conjugate obs (fast path) vs an AD-closure fallback?
    if "--inspect" in args
        println("latent_prior      :: ", typeof(lgm.latent_prior))
        println("observation_model :: ", typeof(lgm.observation_model))
        println("augmentation_info  : ", lgm.augmentation_info === nothing ? "none" : typeof(lgm.augmentation_info))
        println("latent_layout      : ", lgm.latent_layout)
        aug_dim = isempty(lgm.latent_layout) ? -1 : maximum(last, values(lgm.latent_layout))
        println("augmented latent dim: ", aug_dim)
        return nothing
    end

    # --profile: bisect where the warm time goes. If computing marginals for 20
    # nodes is ≪ all 1680, the per-node latent-marginal step dominates.
    if "--profile" in args
        gm = GaussianMarginal()
        inla(lgm, data.y; progress = false, latent_marginalization_method = gm)  # warmup
        t_full = @elapsed inla(lgm, data.y; progress = false, latent_marginalization_method = gm)
        @info "PROFILE t_full (warm)" t_full = round(t_full, digits = 2)
        try
            t_mode = @elapsed Latte.find_hyperparameter_mode(lgm, data.y)
            @info "PROFILE t_mode" t_mode = round(t_mode, digits = 2) explore_plus_marg = round(t_full - t_mode, digits = 2)
        catch e
            @warn "mode timing failed" exception = e
        end
        Profile.clear()
        Profile.init(n = 10^8, delay = 0.0005)
        @profile inla(lgm, data.y; progress = false, latent_marginalization_method = gm)
        open("/tmp/spde_prof_flat.txt", "w") do io
            Profile.print(IOContext(io, :displaysize => (240, 260)); format = :flat, sortedby = :count, mincount = 40)
        end
        @info "PROFILE flat → /tmp/spde_prof_flat.txt"
        return nothing
    end
    t_latte_cold = @elapsed result = inla(lgm, data.y; progress = false, latent_marginalization_method = GaussianMarginal(), accumulators = accum)
    n_warm = "--quick" in args ? 1 : 5
    warm_times = Float64[]
    for _ in 1:n_warm
        push!(warm_times, @elapsed inla(lgm, data.y; progress = false, latent_marginalization_method = GaussianMarginal(), accumulators = accum))
    end
    t_latte_warm = median(warm_times)
    @info "Latte done" cold = round(t_latte_cold, digits = 2) warm = round(t_latte_warm, digits = 3)

    # localise any disagreement: where do Latte's hyperparameters land vs R-INLA?
    let σm = result.hyperparameter_marginals[:σ], rm = result.hyperparameter_marginals[:range_matern],
            τm = result.hyperparameter_marginals[:τ_matern], βm = result.latent_marginals.β[1]
        @info "Latte posteriors vs R-INLA" σ_obs = round(mean(σm), digits = 3) range = round(mean(rm), digits = 3) σ_field_approx = round(1 / sqrt(mean(τm)), digits = 3) β = round(mean(βm), digits = 3) R_range = Float64(rinla_meta.range_mean) R_stdev = Float64(rinla_meta.stdev_mean) R_β = Float64(rinla_meta.intercept_mean)
    end

    # ── field nodes: sanity (means) then per-node KS ──
    latte_field = result.latent_marginals.field          # DOF order
    field_summary = CSV.read(joinpath(WORKDIR, "rinla_field_summary.csv"), DataFrame)
    latte_node_mean = [mean(latte_field[perm[k]]) for k in 1:n_nodes]
    latte_node_sd = [std(latte_field[perm[k]]) for k in 1:n_nodes]
    mean_maxdiff = maximum(abs.(latte_node_mean .- field_summary.mean))
    sd_maxdiff = maximum(abs.(latte_node_sd .- field_summary.sd))
    @info "field summary agreement" mean_maxdiff = round(mean_maxdiff, digits = 4) sd_maxdiff = round(sd_maxdiff, digits = 4)
    if mean_maxdiff > 0.5
        @warn "field means disagree strongly — node→DOF bridge or mesh likely wrong; KS below is suspect"
    end

    field_marg = CSV.read(joinpath(WORKDIR, "rinla_field_marginals.csv"), DataFrame)
    ks_field = Vector{Float64}(undef, n_nodes)
    for k in 1:n_nodes
        sub = field_marg[field_marg.node .== k, :]
        ks_field[k], _ = _ks_density(latte_field[perm[k]], Vector{Float64}(sub.x), Vector{Float64}(sub.density))
    end
    worst_field = argmax(ks_field)

    # ── intercept ──
    g, d = _read_marg(joinpath(WORKDIR, "rinla_intercept_marginal.csv"))
    ks_intercept, _ = _ks_density(result.latent_marginals.β[1], g, d)

    # ── hyperparameters on the interpretable scale ──
    g, d = _read_marg(joinpath(WORKDIR, "rinla_sigma_obs_marginal.csv"))
    ks_sigma_obs, _ = _ks_density(result.hyperparameter_marginals[:σ], g, d)

    g, d = _read_marg(joinpath(WORKDIR, "rinla_range_marginal.csv"))
    ks_range, _ = _ks_density(result.hyperparameter_marginals[:range_matern], g, d)

    # field Stdev: compare on the σ_field = 1/√τ scale via the τ marginal's cdf
    τ_marg = result.hyperparameter_marginals[:τ_matern]
    g, d = _read_marg(joinpath(WORKDIR, "rinla_stdev_marginal.csv"))
    ks_stdev, _ = _ks_cdf(s -> 1 - cdf(τ_marg, 1 / s^2), g, d)

    t_rinla = Float64(rinla_meta.elapsed_seconds)

    println()
    println("SPDEtoy — Latte vs R-INLA (Matérn SPDE, shared mesh), n=$(data.n), nodes=$(n_nodes)")
    println("="^70)
    @printf "%-26s max %.4f (node %d)   median %.4f   count > 0.05: %d / %d\n" "field node KS" ks_field[worst_field] worst_field median(ks_field) count(>(0.05), ks_field) n_nodes
    @printf "%-26s mean max-diff %.4f   sd max-diff %.4f\n" "field summary" mean_maxdiff sd_maxdiff
    @printf "%-26s %.4f\n" "intercept β KS" ks_intercept
    @printf "%-26s %.4f\n" "obs SD KS" ks_sigma_obs
    @printf "%-26s %.4f\n" "field range KS" ks_range
    @printf "%-26s %.4f\n" "field Stdev KS" ks_stdev
    println()
    @printf "Timing: Latte cold = %.2f s,  warm (median of 5) = %.3f s\n" t_latte_cold t_latte_warm
    @printf "        R-INLA = %.2f s   (INLA %s)\n" t_rinla String(rinla_meta.inla_version)
    @printf "        warm speedup: %.1fx\n" t_rinla / t_latte_warm
    println()

    out = (
        scenario = "spdetoy", n = data.n, n_nodes = n_nodes,
        ks_field = ks_field, worst_field = worst_field,
        ks_field_max = ks_field[worst_field], ks_field_median = median(ks_field),
        ks_intercept = ks_intercept, ks_sigma_obs = ks_sigma_obs,
        ks_range = ks_range, ks_stdev = ks_stdev,
        field_mean_maxdiff = mean_maxdiff, field_sd_maxdiff = sd_maxdiff,
        t_latte_cold = t_latte_cold, t_latte_warm = t_latte_warm, t_rinla = t_rinla,
        inla_version = String(rinla_meta.inla_version),
    )
    open(joinpath(WORKDIR, "result.json"), "w") do io
        JSON3.write(io, out)
    end
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
