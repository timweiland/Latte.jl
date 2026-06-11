# Paraná precipitation: Latte INLA vs R-INLA on the SPDE book §2.8 model, same mesh.
#
# Gamma likelihood (log link), η = intercept + rw1(seaDist) + Matérn-SPDE field, on
# the shared non-convex mesh R-INLA built. R produces the mesh + seaDist + its rw1
# grouping (parana_data.csv, rw1_groups.csv); this script rebuilds the identical
# discretisation + a 3-component latent and cross-validates field/rw1/intercept
# marginals, the hyperparameters, and timing.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using CSV
using DataFrames
using Distributions
using DynamicPPL
using GaussianMarkovRandomFields
using GaussianMarkovRandomFields: FEMDiscretization, evaluation_matrix, node_selection_matrix, RWModel
using Ferrite, FerriteGmsh, Gmsh, LibGEOS
using JSON3
using Latte
using LinearAlgebra
using Printf
using Profile
using SparseArrays
using Statistics

const WORKDIR = joinpath(@__DIR__, "_workdir")

# ── KS helpers (verbatim from spdetoy) ──
function _ks_cdf(cdf_fn, g::Vector{Float64}, d::Vector{Float64})
    n = length(g); c = Vector{Float64}(undef, n); c[1] = 0.0
    for k in 2:n
        c[k] = c[k - 1] + 0.5 * (d[k - 1] + d[k]) * (g[k] - g[k - 1])
    end
    Z = c[end]; Z > 0 && (c ./= Z)
    best = 0.0
    for k in 1:n
        gap = abs(cdf_fn(g[k]) - c[k]); gap > best && (best = gap)
    end
    return best
end
_ks_density(engine, g, d) = _ks_cdf(x -> cdf(engine, x), g, d)
_read_marg(path) = (df = CSV.read(path, DataFrame); (Vector{Float64}(df.x), Vector{Float64}(df.density)))

function load_discretization(workdir)
    nodes = CSV.read(joinpath(workdir, "nodes.csv"), DataFrame)
    tris = CSV.read(joinpath(workdir, "triangles.csv"), DataFrame)
    fn = [Ferrite.Node((nodes.x[i], nodes.y[i])) for i in 1:nrow(nodes)]
    els = [Ferrite.Triangle((tris.v1[i], tris.v2[i], tris.v3[i])) for i in 1:nrow(tris)]
    grid = Ferrite.Grid(els, fn)
    disc = FEMDiscretization(grid, Ferrite.Lagrange{Ferrite.RefTriangle, 1}(), Ferrite.QuadratureRule{Ferrite.RefTriangle}(2))
    return disc, nrow(nodes)
end

function node_to_dof(disc, n_nodes)
    S = node_selection_matrix(disc, collect(1:n_nodes))
    I_, J_, _ = findnz(S); perm = Vector{Int}(undef, n_nodes)
    for t in eachindex(I_)
        perm[I_[t]] = J_[t]
    end
    return perm
end

# Gamma SPDE + rw1 model: η = β + A_rw1·u + A_spde·field, y ~ Gamma(phi, exp(η)/phi)
@latte function parana_model(y, base_matern, A_spde, rw1, A_rw1, p)
    phi ~ Gamma(2.0, 5.0)   # interior mode (shape>1) — avoids log(0) init for the dispersion
    τ_field ~ PCPrior.Precision(p.sigma_field_U; α = p.sigma_field_p)
    range_field ~ PCPrior.Range(p.range_U; p = p.range_p)
    τ_rw1 ~ PCPrior.Precision(p.rw1_sigma_U; α = p.rw1_sigma_alpha)
    β ~ MvNormal(zeros(1), (1 / p.prec_intercept) * I(1))
    field ~ base_matern(τ = τ_field, range = range_field)
    u ~ rw1(τ = τ_rw1)
    η = β[1] .+ A_rw1 * u .+ A_spde * field
    for i in eachindex(y)
        y[i] ~ Gamma(phi, exp(η[i]) / phi)
    end
end

function main(args::Vector{String} = ARGS)
    pj = joinpath(WORKDIR, "params.json")
    p_raw = JSON3.read(read(pj, String))
    p = (
        sigma_field_U = Float64(p_raw.sigma_field_U), sigma_field_p = Float64(p_raw.sigma_field_p),
        range_U = Float64(p_raw.range_U), range_p = Float64(p_raw.range_p),
        rw1_sigma_U = Float64(p_raw.rw1_sigma_U), rw1_sigma_alpha = Float64(p_raw.rw1_sigma_alpha),
        prec_intercept = Float64(p_raw.prec_intercept),
    )
    if !isfile(joinpath(WORKDIR, "rinla_field_summary.csv")) || "--refresh-rinla" in args
        run(`Rscript $(joinpath(@__DIR__, "paranaprec_compare.R")) $(WORKDIR) $(WORKDIR)`)
    end
    rinla_meta = JSON3.read(read(joinpath(WORKDIR, "rinla_meta.json"), String))

    data = CSV.read(joinpath(WORKDIR, "parana_data.csv"), DataFrame)
    coords = Matrix{Float64}(hcat(data.s1, data.s2))
    y = Vector{Float64}(data.y)
    grp = Vector{Int}(data.grp)
    n_groups = nrow(CSV.read(joinpath(WORKDIR, "rw1_groups.csv"), DataFrame))
    n = length(y)

    disc, n_nodes = load_discretization(WORKDIR)
    @info "mesh rebuilt" n_nodes n_obs = n n_groups
    A_spde = evaluation_matrix(disc, coords)
    @assert all(0.99 .< vec(sum(A_spde; dims = 2)) .< 1.01) "obs points off mesh"
    A_rw1 = sparse(1:n, grp, ones(n), n, n_groups)
    base_matern = MaternModel(disc; smoothness = 0)
    rw1 = RWModel{1}(n_groups; scale_model = true)   # match R-INLA's scale.model=TRUE (PC prior on marginal SD)
    perm = node_to_dof(disc, n_nodes)

    augmented = "--augmented" in args   # legacy: augment=true + SimplifiedLaplace
    mode_str = augmented ? "augmented (legacy)" : "compact (resolved default)"
    @info "running Latte INLA (Gamma + rw1 + SPDE)" mode = mode_str
    lgm = parana_model(y, base_matern, A_spde, rw1, A_rw1, p; augment = augmented)
    !augmented && @info "resolved defaults" augmented = (lgm.augmentation_info !== nothing) method = typeof(Latte.default_marginalization(lgm))
    accum = (MarginalLogLikelihoodStrategy(),)

    if "--profile" in args
        inla(lgm, y; progress = false, accumulators = accum)  # warmup
        t_full = @elapsed inla(lgm, y; progress = false, accumulators = accum)
        @info "PROFILE t_full (warm)" t_full = round(t_full, digits = 2)
        try
            t_mode = @elapsed Latte.find_hyperparameter_mode(lgm, y)
            @info "PROFILE t_mode" t_mode = round(t_mode, digits = 2) explore_plus_marg = round(t_full - t_mode, digits = 2)
        catch e
            @warn "mode timing failed" exception = e
        end
        Profile.clear(); Profile.init(n = 10^8, delay = 0.0005)
        @profile inla(lgm, y; progress = false, accumulators = accum)
        open("/tmp/parana_prof_flat.txt", "w") do io
            Profile.print(IOContext(io, :displaysize => (240, 260)); format = :flat, sortedby = :count, mincount = 30)
        end
        @info "PROFILE flat → /tmp/parana_prof_flat.txt"
        return nothing
    end
    vbc_short = nothing
    for a in args
        startswith(a, "--vbc-short=") && (vbc_short = parse(Int, last(split(a, "="))))
    end
    marg = if augmented
        SimplifiedLaplace()
    elseif "--gaussian-marg" in args
        GaussianMarginal()
    elseif vbc_short !== nothing
        VBCMarginal(AutoVBCIndexSet(short_dim = vbc_short))   # explicit VBC short dim override
    else
        nothing   # inla resolves via default_marginalization (→ VBC for this compact LTM)
    end
    t_cold = @elapsed result = inla(lgm, y; progress = false, accumulators = accum, latent_marginalization_method = marg)
    n_warm = "--quick" in args ? 1 : 3
    warm = median([(@elapsed inla(lgm, y; progress = false, accumulators = accum, latent_marginalization_method = marg)) for _ in 1:n_warm])

    # diagnostics vs R-INLA
    let rm = result.hyperparameter_marginals[:range_field], τm = result.hyperparameter_marginals[:τ_field],
            βm = result.latent_marginals.β[1]
        @info "Latte posteriors vs R-INLA" range = round(mean(rm), digits = 3) σ_field = round(1 / sqrt(mean(τm)), digits = 3) β = round(mean(βm), digits = 3) R_range = Float64(rinla_meta.range_mean) R_stdev = Float64(rinla_meta.stdev_mean) R_β = Float64(rinla_meta.intercept_mean)
    end

    # ── field nodes ──
    latte_field = result.latent_marginals.field
    fsum = CSV.read(joinpath(WORKDIR, "rinla_field_summary.csv"), DataFrame)
    mean_maxdiff = maximum(abs.([mean(latte_field[perm[k]]) for k in 1:n_nodes] .- fsum.mean))
    fmarg = CSV.read(joinpath(WORKDIR, "rinla_field_marginals.csv"), DataFrame)
    ks_field = [(_ks_density(latte_field[perm[k]], Vector{Float64}((s = fmarg[fmarg.node .== k, :]).x), Vector{Float64}(s.density))) for k in 1:n_nodes]

    # ── rw1 nodes (direct order, no DOF reshuffle) ──
    latte_u = result.latent_marginals.u
    rwmarg = CSV.read(joinpath(WORKDIR, "rinla_rw1_marginals.csv"), DataFrame)
    ks_rw1 = [(_ks_density(latte_u[k], Vector{Float64}((s = rwmarg[rwmarg.node .== k, :]).x), Vector{Float64}(s.density))) for k in 1:n_groups]

    # ── intercept + hyperparameters ──
    g, d = _read_marg(joinpath(WORKDIR, "rinla_intercept_marginal.csv")); ks_intercept = _ks_density(result.latent_marginals.β[1], g, d)
    g, d = _read_marg(joinpath(WORKDIR, "rinla_range_marginal.csv")); ks_range = _ks_density(result.hyperparameter_marginals[:range_field], g, d)
    τm = result.hyperparameter_marginals[:τ_field]
    g, d = _read_marg(joinpath(WORKDIR, "rinla_stdev_marginal.csv")); ks_stdev = _ks_cdf(s -> 1 - cdf(τm, 1 / s^2), g, d)
    τr = result.hyperparameter_marginals[:τ_rw1]
    g, d = _read_marg(joinpath(WORKDIR, "rinla_rw1_sd_marginal.csv")); ks_rw1_sd = _ks_cdf(s -> 1 - cdf(τr, 1 / s^2), g, d)

    t_rinla = Float64(rinla_meta.elapsed_seconds)
    println()
    println("Paraná precipitation — Latte vs R-INLA (Gamma + rw1 + SPDE, shared mesh)")
    println("="^70)
    @printf "%-22s max %.4f  median %.4f  (>0.05: %d/%d)\n" "field node KS" maximum(ks_field) median(ks_field) count(>(0.05), ks_field) n_nodes
    @printf "%-22s mean max-diff %.4f\n" "field summary" mean_maxdiff
    @printf "%-22s max %.4f  median %.4f\n" "rw1(seaDist) KS" maximum(ks_rw1) median(ks_rw1)
    @printf "%-22s %.4f\n" "intercept β KS" ks_intercept
    @printf "%-22s %.4f\n" "field range KS" ks_range
    @printf "%-22s %.4f\n" "field Stdev KS" ks_stdev
    @printf "%-22s %.4f\n" "rw1 Stdev KS" ks_rw1_sd
    println()
    @printf "Timing: Latte cold = %.2f s,  warm = %.3f s\n" t_cold warm
    @printf "        R-INLA = %.2f s   (INLA %s)\n        warm speedup: %.1fx\n" t_rinla String(rinla_meta.inla_version) t_rinla / warm

    out = (
        scenario = "paranaprec", n = n, n_nodes = n_nodes, n_groups = n_groups,
        ks_field = ks_field, ks_field_max = maximum(ks_field), ks_field_median = median(ks_field),
        ks_rw1 = ks_rw1, ks_intercept = ks_intercept, ks_range = ks_range, ks_stdev = ks_stdev, ks_rw1_sd = ks_rw1_sd,
        field_mean_maxdiff = mean_maxdiff,
        t_latte_cold = t_cold, t_latte_warm = warm, t_rinla = t_rinla, inla_version = String(rinla_meta.inla_version),
    )
    open(joinpath(WORKDIR, "result.json"), "w") do io
        JSON3.write(io, out)
    end
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
