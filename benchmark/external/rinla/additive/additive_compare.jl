# Empirical comparison of η marginals on an additive Poisson model:
# Latte (augmented mode, η is a primary latent) vs R-INLA (compact mode,
# η is a derived lincomb), both judged against a long NUTS run.
#
# Model:
#   β ~ Normal(0, 1)                   (fixed effect)
#   τ ~ PCPrior.Precision(1.0, α=0.01)
#   x_i ~ N(0, 1/τ)  iid               (random effect, n=50)
#   y_i ~ Poisson(exp(β + x_i))
#
# Each η_i = β + x_i is a sum of two posterior-correlated latent
# components; getting its marginal accurately requires joint
# information that pure x-marginals don't carry.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using CSV
using DataFrames
using Distributions
using DynamicPPL          # full module: the @latte macro's expansion references it
using JSON3
using Latte
using GaussianMarkovRandomFields
using LinearAlgebra
using MCMCChains
using Printf
using Random
using Serialization
using StableRNGs
using Statistics
using Turing

const WORKDIR = joinpath(@__DIR__, "_workdir")
const N_OBS = 50
const SEED = UInt64(0x0badcafe)

@latte function additive_iid_poisson(y, n)
    # MvNormal-of-length-1 instead of plain Normal because the DPPL
    # adapter expects vector latents — needs to register `β` in the
    # latent layout as a block of size 1.
    β ~ MvNormal(zeros(1), ones(1))
    τ ~ PCPrior.Precision(1.0, α = 0.01)
    x ~ IIDModel(n)(τ = τ)
    for i in eachindex(y)
        y[i] ~ Poisson(exp(β[1] + x[i]); check_args = false)
    end
end

# Deterministic data: same y for every run of this script. Mirrors the
# benchmark scenario style (StableRNG-seeded).
function generate_data(n::Int; seed::UInt64 = SEED)
    rng = StableRNG(seed)
    true_β = 0.5
    true_x = randn(rng, n) .* 0.4
    y = rand.(rng, Poisson.(exp.(true_β .+ true_x)))
    return (; n = n, y = y, true_β = true_β, true_x = true_x)
end

# Build empirical CDFs from a chain for each η_i = β + x[i].
function nuts_eta_cdfs(chain::Chains, n::Int)
    # `β` is a length-1 MvNormal, exposed in the chain as `β[1]`.
    β_samples = vec(chain[Symbol("β[1]")])
    cdfs = Vector{Tuple{Vector{Float64}, Vector{Float64}}}(undef, n)
    for i in 1:n
        x_samples = vec(chain[Symbol("x[$(i)]")])
        η = sort!(β_samples .+ x_samples)
        m = length(η)
        cdfs[i] = (collect(Float64, η), collect(range(1, m) ./ m))
    end
    return cdfs
end

# Evaluate engine.cdf on the union of evaluation points; return
# (max_abs_gap, signed_at_argmax).
function ks_distance(engine, ref_grid::Vector{Float64}, ref_cdf::Vector{Float64})
    best_abs = 0.0
    best_signed = 0.0
    for k in eachindex(ref_grid)
        gap = cdf(engine, ref_grid[k]) - ref_cdf[k]
        if abs(gap) > best_abs
            best_abs = abs(gap)
            best_signed = gap
        end
    end
    return best_abs, best_signed
end

# KS via interpolated CDF on a (grid, density) marginal, used for
# R-INLA's marginals.linear.predictor.
function ks_distance_density(
        rinla_grid::Vector{Float64}, rinla_density::Vector{Float64},
        ref_grid::Vector{Float64}, ref_cdf::Vector{Float64},
    )
    n = length(rinla_grid)
    rinla_cdf = Vector{Float64}(undef, n)
    rinla_cdf[1] = 0.0
    for k in 2:n
        rinla_cdf[k] = rinla_cdf[k - 1] +
            0.5 * (rinla_density[k - 1] + rinla_density[k]) * (rinla_grid[k] - rinla_grid[k - 1])
    end
    Z = rinla_cdf[end]
    Z > 0 && (rinla_cdf ./= Z)

    eval_pts = sort!(unique(vcat(rinla_grid, ref_grid)))
    best_abs = 0.0
    best_signed = 0.0
    for x in eval_pts
        f_rinla = _interp_cdf(rinla_grid, rinla_cdf, x)
        f_ref = _interp_cdf(ref_grid, ref_cdf, x)
        gap = f_rinla - f_ref
        if abs(gap) > best_abs
            best_abs = abs(gap)
            best_signed = gap
        end
    end
    return best_abs, best_signed
end

function _interp_cdf(grid::Vector{Float64}, cdf::Vector{Float64}, x::Real)
    n = length(grid)
    x <= grid[1] && return 0.0
    x >= grid[end] && return 1.0
    k = searchsortedfirst(grid, x)
    t = (x - grid[k - 1]) / (grid[k] - grid[k - 1])
    return cdf[k - 1] + t * (cdf[k] - cdf[k - 1])
end

function main(args::Vector{String} = ARGS)
    mkpath(WORKDIR)
    @info "additive_iid_poisson η-marginal comparison" n = N_OBS seed = SEED

    data = generate_data(N_OBS)

    # ── 1. NUTS reference ─────────────────────────────────────────────
    # Cached on disk: NUTS is the slow step (~5 minutes). The cache key
    # is the data hash + a fixed config string; bump CACHE_VERSION below
    # if the NUTS settings change.
    cache_path = joinpath(
        WORKDIR, "nuts_chain_$(string(hash(data.y), base = 16))_v1.bin",
    )
    nuts_chain = if isfile(cache_path) && !("--refresh-nuts" in args)
        @info "loading NUTS chain from cache" cache_path
        deserialize(cache_path)
    else
        @info "running NUTS reference (4 chains × 5000 × 2000 warmup, target 0.95)"
        rng = MersenneTwister(SEED)
        backend = Threads.nthreads() > 1 ? Turing.MCMCThreads() : Turing.MCMCSerial()
        # @latte's call form returns an LGM; the DPPL model for Turing is
        # recovered via `Latte.dppl_model(...)`.
        turing_model = Latte.dppl_model(additive_iid_poisson)(data.y, data.n)
        t_nuts = @elapsed chain = sample(
            rng, turing_model,
            Turing.NUTS(2000, 0.95), backend, 5000, 4;
            progress = false, verbose = false,
        )
        @info "NUTS done" elapsed = round(t_nuts, digits = 1)
        serialize(cache_path, chain)
        chain
    end
    nuts_cdfs = nuts_eta_cdfs(nuts_chain, data.n)

    # ── 2. Latte INLA ─────────────────────────────────────────────────
    # Default: the @latte macro builds a compact LGM (the Poisson likelihood
    # with the `β[1] + x[i]` predictor is a LinearlyTransformedObservationModel)
    # and inla resolves the VBC mean correction. `--augmented` opts into the
    # legacy augmented + simplified.laplace mode, where each η_i is a primary
    # latent.
    #
    # Full Laplace is dropped from the standard comparison — it's
    # validated separately and adds ~5+ minutes per run. Pass
    # `--with-full-laplace` to include it.
    augmented = "--augmented" in args
    marg = augmented ? SimplifiedLaplace() : nothing   # nothing ⇒ resolve (→ VBC, compact LTM)
    lgm = additive_iid_poisson(data.y, data.n; augment = augmented)
    resolved = string(typeof(augmented ? SimplifiedLaplace() : Latte.default_marginalization(lgm)).name.name)

    @info "running Latte INLA" mode = (augmented ? "augmented + simplified.laplace (legacy)" : "compact + VBC (default)") resolved
    t_latte = @elapsed latte_result = inla(
        lgm, data.y;
        latent_marginalization_method = marg,
        progress = false,
    )
    @info "Latte done" elapsed = round(t_latte, digits = 2) resolved
    latte_lp = eta_marginals(latte_result, data.n)

    if "--with-full-laplace" in args
        @info "running Latte INLA (LaplaceMarginal)"
        t_latte_full = @elapsed latte_full_result = inla(
            lgm, data.y;
            latent_marginalization_method = LaplaceMarginal(),
            progress = false,
        )
        @info "Latte (Full Laplace) done" elapsed = round(t_latte_full, digits = 2)
    else
        latte_full_result = nothing
    end

    # ── 3. R-INLA (cached unless --refresh-rinla) ────────────────────
    rinla_marker = joinpath(WORKDIR, "rinla_eta_marginals.csv")
    if !isfile(rinla_marker) || "--refresh-rinla" in args
        @info "running R-INLA (compact, simplified.laplace)"
        CSV.write(joinpath(WORKDIR, "data.csv"), DataFrame(y = data.y))
        open(joinpath(WORKDIR, "params.json"), "w") do io
            JSON3.write(
                io, Dict(
                    "pc_U" => 1.0, "pc_alpha" => 0.01,
                    "strategy" => "simplified.laplace",
                )
            )
        end
        rscript = joinpath(@__DIR__, "additive_compare.R")
        t_rinla = @elapsed run(`Rscript $rscript $(WORKDIR) $(WORKDIR)`)
        @info "R-INLA done" elapsed = round(t_rinla, digits = 2)
    else
        @info "loading R-INLA outputs from cache" workdir = WORKDIR
    end
    rinla_meta = JSON3.read(read(joinpath(WORKDIR, "rinla_meta.json"), String))

    rinla_marg_df = CSV.read(joinpath(WORKDIR, "rinla_eta_marginals.csv"), DataFrame)

    # ── 4. KS distances vs NUTS for each η_i ─────────────────────────
    latte_ks = Float64[]
    latte_signed = Float64[]
    rinla_ks = Float64[]
    rinla_signed = Float64[]
    for i in 1:data.n
        ref_grid, ref_cdf = nuts_cdfs[i]

        # Latte: η_i lives at the i-th linear-predictor index.
        ks_l, sgn_l = ks_distance(latte_lp[i], ref_grid, ref_cdf)
        push!(latte_ks, ks_l)
        push!(latte_signed, sgn_l)

        # R-INLA: marginal(η_i) from the dumped (x, density) pairs.
        sub = filter(row -> row.i == i, rinla_marg_df)
        sort!(sub, :x)
        ks_r, sgn_r = ks_distance_density(
            Vector{Float64}(sub.x), Vector{Float64}(sub.density),
            ref_grid, ref_cdf,
        )
        push!(rinla_ks, ks_r)
        push!(rinla_signed, sgn_r)
    end

    function _aggregate(name, ks_vec)
        worst_idx = argmax(ks_vec)
        med = sort(ks_vec)[div(length(ks_vec), 2) + 1]
        return @sprintf(
            "%-30s max %.4f (i*=%d)   median %.4f   count > 0.05: %d / %d",
            name, ks_vec[worst_idx], worst_idx, med, count(>(0.05), ks_vec), length(ks_vec),
        )
    end

    # Sanity: β posterior across all three sources. If priors agree
    # (we set prec.intercept=1.0 in the R script to mirror the
    # `MvNormal(0, 1)` Julia prior) the means/SDs should be close.
    nuts_β = vec(nuts_chain[Symbol("β[1]")])
    nuts_β_mean = mean(nuts_β)
    nuts_β_sd = std(nuts_β)
    nuts_β_q025 = quantile(nuts_β, 0.025)
    nuts_β_q975 = quantile(nuts_β, 0.975)

    # Latte β marginals across both default (compact + VBC) and Full Laplace
    # strategies — useful for telling whether any β bias is specific to the
    # marginalization method vs a deeper issue with mode finding /
    # hyperparameter integration. `β` is a length-1 latent block; its index is
    # read from the model's named layout (works in both compact and augmented
    # modes).
    β_latent_idx = first(Latte.latent_groups(latte_result)[:β])
    latte_β = latte_result.latent_marginals[β_latent_idx]

    function _stats(d)
        return (mean(d), std(d), quantile(d, 0.025), quantile(d, 0.975))
    end

    nuts_β_q025 = quantile(nuts_β, 0.025)
    nuts_β_q975 = quantile(nuts_β, 0.975)
    latte_simp_stats = _stats(latte_β)
    latte_full_stats = latte_full_result === nothing ? nothing : _stats(latte_full_result.latent_marginals[β_latent_idx])

    rinla_β_summary = CSV.read(joinpath(WORKDIR, "rinla_beta_summary.csv"), DataFrame)
    rinla_β_mean = rinla_β_summary.mean[1]
    rinla_β_sd = rinla_β_summary.sd[1]
    rinla_β_q025 = rinla_β_summary.q025[1]
    rinla_β_q975 = rinla_β_summary.q975[1]

    println()
    println("β posterior (intercept) — sanity check on priors and approximation methods")
    println("="^80)
    @printf "%-30s mean %+.4f  sd %.4f  q025 %+.4f  q975 %+.4f\n" "NUTS" nuts_β_mean nuts_β_sd nuts_β_q025 nuts_β_q975
    @printf "%-30s mean %+.4f  sd %.4f  q025 %+.4f  q975 %+.4f\n" "Latte ($(resolved))" latte_simp_stats[1] latte_simp_stats[2] latte_simp_stats[3] latte_simp_stats[4]
    if latte_full_stats !== nothing
        @printf "%-30s mean %+.4f  sd %.4f  q025 %+.4f  q975 %+.4f\n" "Latte (Full Laplace)" latte_full_stats[1] latte_full_stats[2] latte_full_stats[3] latte_full_stats[4]
    end
    @printf "%-30s mean %+.4f  sd %.4f  q025 %+.4f  q975 %+.4f\n" "R-INLA" rinla_β_mean rinla_β_sd rinla_β_q025 rinla_β_q975

    println()
    println("η_i = β + x_i marginal accuracy vs NUTS reference (n=$(data.n))")
    println("="^80)
    println(_aggregate("Latte ($(augmented ? "augmented + SLA" : "compact + " * resolved))", latte_ks))
    println(_aggregate("R-INLA (compact mode)", rinla_ks))
    println()

    return (
        latte_ks = latte_ks, rinla_ks = rinla_ks,
        latte_signed = latte_signed, rinla_signed = rinla_signed,
    )
end

# Latte's η_i = β + x_i marginals, in both modes:
#   - augmented: η is a primary latent; the predictor block of
#     `latent_marginals` carries the simplified-Laplace skew correction.
#   - compact (default): η is the linear functional β·1 + x[i] of the latent
#     field. `linear_combinations` builds its marginal as a weighted mixture of
#     Gaussian conditionals over the hyperparameter integration points, with the
#     VBC-corrected mean threaded through (so the mean matches the corrected
#     latent marginals).
function eta_marginals(result, n_obs::Int)
    info = result.augmentation_info
    if info !== nothing
        pred_idx = info.linear_predictor_indices
        return [result.latent_marginals[i] for i in pred_idx[1:n_obs]]
    end
    # Compact mode: η_i = 1·β + x[i]. Coefficient 1 on the single β component
    # for every row; the identity on the n-dim x block selects x[i] per row.
    return linear_combinations(result; β = ones(n_obs), x = Matrix(1.0I, n_obs, n_obs))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
