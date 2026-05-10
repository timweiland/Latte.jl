# Quick diagnostic on β bias:
#  1. Compare Latte's β under GaussianMarginal vs SimplifiedLaplace.
#     If only SLA is biased ⇒ surgical fix is the suspect.
#     If both are biased ⇒ issue is upstream of marginalization.
#  2. Show the joint Laplace mode for β at the τ-MAP, vs NUTS.
#     If the mode itself is biased ⇒ mode finding / augmentation issue.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using Distributions
using DynamicPPL: @model
using GaussianMarkovRandomFields
using Latte
using Printf
using Random
using Serialization
using StableRNGs
using Statistics

const SEED = UInt64(0x0badcafe)
const N_OBS = 50
const WORKDIR = joinpath(@__DIR__, "_workdir")

@model function additive_iid_poisson(y, n)
    β ~ MvNormal(zeros(1), ones(1))
    τ ~ PCPrior.Precision(1.0, α = 0.01)
    x ~ IIDModel(n)(τ = τ)
    for i in eachindex(y)
        y[i] ~ Poisson(exp(β[1] + x[i]); check_args = false)
    end
end

function generate_data(n::Int; seed::UInt64 = SEED)
    rng = StableRNG(seed)
    true_β = 0.5
    true_x = randn(rng, n) .* 0.4
    y = rand.(rng, Poisson.(exp.(true_β .+ true_x)))
    return (; n = n, y = y, true_β = true_β, true_x = true_x)
end

function β_summary(result)
    info = result.augmentation_info
    β_idx = first(info.base_latent_indices)
    m = result.latent_marginals[β_idx]
    return (
        idx = β_idx, mean = mean(m), sd = std(m),
        q025 = quantile(m, 0.025), q975 = quantile(m, 0.975),
    )
end

function nuts_β_stats()
    cache_path = joinpath(WORKDIR, "nuts_chain_$(string(hash(generate_data(N_OBS).y), base = 16))_v1.bin")
    isfile(cache_path) || error("NUTS cache missing")
    chain = deserialize(cache_path)
    β = vec(chain[Symbol("β[1]")])
    return (
        mean = mean(β), sd = std(β),
        q025 = quantile(β, 0.025), q975 = quantile(β, 0.975),
    )
end

function main()
    data = generate_data(N_OBS)
    @info "diagnose β bias" n = data.n true_β = 0.5
    nuts = nuts_β_stats()

    dppl = additive_iid_poisson(data.y, data.n)
    lgm = latte_from_dppl(dppl; random = (:β, :x))

    println()
    @printf "%-30s mean %+.4f  sd %.4f  q025 %+.4f  q975 %+.4f\n" "NUTS (gold)" nuts.mean nuts.sd nuts.q025 nuts.q975

    @info "fitting Gaussian"
    t1 = @elapsed r1 = inla(
        lgm, data.y;
        latent_marginalization_method = GaussianMarginal(), progress = false
    )
    s1 = β_summary(r1)
    @printf "%-30s mean %+.4f  sd %.4f  q025 %+.4f  q975 %+.4f   %.1fs   Δμ=%+.4f Δσ=%+.4f\n" "Latte (Gaussian)" s1.mean s1.sd s1.q025 s1.q975 t1 (s1.mean - nuts.mean) (s1.sd - nuts.sd)

    @info "fitting Simplified"
    t2 = @elapsed r2 = inla(
        lgm, data.y;
        latent_marginalization_method = SimplifiedLaplace(), progress = false
    )
    s2 = β_summary(r2)
    @printf "%-30s mean %+.4f  sd %.4f  q025 %+.4f  q975 %+.4f   %.1fs   Δμ=%+.4f Δσ=%+.4f\n" "Latte (Simplified)" s2.mean s2.sd s2.q025 s2.q975 t2 (s2.mean - nuts.mean) (s2.sd - nuts.sd)

    # Joint mode at τ-MAP and τ posterior summary.
    θ_star = r1.hyperparameter_mode
    info = r1.augmentation_info
    θ_natural = convert(NamedTuple, convert(Latte.NaturalHyperparameters, θ_star))
    prior_gmrf = lgm.latent_prior(; θ_natural...)
    obs_lik = lgm.observation_model(GaussianMarkovRandomFields.PoissonObservations(data.y); θ_natural...)
    ga = gaussian_approximation(prior_gmrf, obs_lik)
    β_idx = first(info.base_latent_indices)
    β_mode = mean(ga)[β_idx]
    β_mode_sd = std(ga)[β_idx]
    @printf "\n%-30s mode %+.4f  laplace_sd %.4f   Δmode_vs_NUTSμ %+.4f\n" "Latte joint mode β @ τ-MAP" β_mode β_mode_sd (β_mode - nuts.mean)
    @printf "%-30s τ-MAP (natural) = %+.4f\n" "" first(θ_natural)

    # Latte's τ marginal vs NUTS's τ samples.
    cache_path = joinpath(WORKDIR, "nuts_chain_$(string(hash(data.y), base = 16))_v1.bin")
    chain = deserialize(cache_path)
    nuts_τ = vec(chain[:τ])
    @printf "%-30s mean %+.4f  sd %.4f  q025 %+.4f  q975 %+.4f\n" "NUTS τ" mean(nuts_τ) std(nuts_τ) quantile(nuts_τ, 0.025) quantile(nuts_τ, 0.975)
    latte_τ = r1.hyperparameter_marginals[1]
    @printf "%-30s mean %+.4f  sd %.4f  q025 %+.4f  q975 %+.4f\n" "Latte τ" mean(latte_τ) std(latte_τ) quantile(latte_τ, 0.025) quantile(latte_τ, 0.975)

    println()
    println("Interpretation:")
    println("  - If both Gaussian and Simplified show similar β bias  ⇒ upstream issue")
    println("    (mode, augmentation, or τ-mixing); SLA surgical fix is innocent.")
    println("  - If only Simplified is biased                        ⇒ SLA surgical fix")
    println("    is overcorrecting; revisit `_correct_augmentation_shadow!`.")
    println("  - If joint mode β already differs from NUTS mean      ⇒ mode finding")
    return println("    or augmented Q construction.")
end

main()
