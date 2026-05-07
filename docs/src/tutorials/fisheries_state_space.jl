using Latte
using DynamicPPL: @model
using Distributions
using GaussianMarkovRandomFields
using GaussianMarkovRandomFields: IIDModel
using LinearAlgebra
using Random

function schaefer_step(log_B0, r, K, F, dt; n_sub::Int = 8)
    h = dt / n_sub
    log_B = log_B0
    @inbounds for _ in 1:n_sub
        f(lB) = r * (1 - exp(lB) / K) - F   # d(log B)/dt
        k1 = f(log_B)
        k2 = f(log_B + 0.5h * k1)
        k3 = f(log_B + 0.5h * k2)
        k4 = f(log_B + h * k3)
        log_B = log_B + (h / 6) * (k1 + 2k2 + 2k3 + k4)
    end
    return log_B
end

function simulate_biomass(log_K, ε, catches, r, K, σ_proc)
    T = length(catches) + 1
    log_B = Vector{Float64}(undef, T)
    log_B[1] = log_K
    for t in 1:(T - 1)
        F_t = min(catches[t] / exp(log_B[t]), r - 1.0e-3)
        log_B[t + 1] = schaefer_step(log_B[t], r, K, F_t, 1.0) +
            σ_proc * ε[t]
    end
    return log_B
end

Random.seed!(20260502)

const T = 25
true_r = 0.3
true_K = 1000.0
true_σ_proc = 0.1
true_q_com = 0.001;
true_σ_com = 0.2
true_q_sur = 0.5;
true_σ_sur = 0.1
true_q_rec = 0.0005;
true_σ_rec = 0.3

# Catch trajectory: development → peak → decline (typical fishery history)
true_catches = [50.0 + 250.0 * exp(-0.5 * ((t - 12) / 5.0)^2) for t in 1:(T - 1)]

true_ε = randn(T - 1)
true_log_B = simulate_biomass(
    log(true_K), true_ε, true_catches, true_r, true_K, true_σ_proc,
)

# Three observation series, each with their own (q, σ) calibration
true_log_I_com = log(true_q_com) .+ true_log_B .+ true_σ_com .* randn(T)
true_log_I_sur = log(true_q_sur) .+ true_log_B .+ true_σ_sur .* randn(T)
true_log_I_rec = log(true_q_rec) .+ true_log_B .+ true_σ_rec .* randn(T)

using AlgebraOfGraphics, CairoMakie

fig = Figure(size = (1100, 360))
ax_b = Axis(
    fig[1, 1], title = "Biomass trajectory + catches",
    xlabel = "year", ylabel = "B (tonnes)",
)
lines!(ax_b, 1:T, exp.(true_log_B); color = :black, linewidth = 2, label = "B")
barplot!(
    ax_b, 1:(T - 1), true_catches; color = (:firebrick, 0.4),
    label = "catch",
)
axislegend(ax_b; position = :rt, framevisible = false)
ax_i = Axis(
    fig[1, 2], title = "Three CPUE indices (log-scale)",
    xlabel = "year", ylabel = "log I",
)
scatter!(ax_i, 1:T, true_log_I_com; color = :steelblue, label = "commercial")
scatter!(ax_i, 1:T, true_log_I_sur; color = :forestgreen, label = "survey")
scatter!(ax_i, 1:T, true_log_I_rec; color = :goldenrod, label = "recreational")
axislegend(ax_i; position = :rb, framevisible = false)
fig

@model function multifleet_schaefer(
        log_I_com, log_I_sur, log_I_rec, catches, T,
    )
    log_r ~ Normal(log(0.3), 0.7)
    log_K ~ Normal(log(1000.0), 1.0)
    log_σ_proc ~ Normal(log(0.1), 0.5)
    # Loose priors for the commercial and recreational fleets...
    log_q_com ~ Normal(log(0.001), 1.0)
    log_σ_com ~ Normal(log(0.2), 0.5)
    # ...tight prior on the calibrated research survey:
    log_q_sur ~ Normal(log(0.5), 0.05)
    log_σ_sur ~ Normal(log(0.1), 0.4)
    log_q_rec ~ Normal(log(0.0005), 1.5)
    log_σ_rec ~ Normal(log(0.3), 0.5)

    r = exp(log_r)
    K = exp(log_K)
    σ_proc = exp(log_σ_proc)

    # Latent: T-1 process-noise increments, IID standard normal under
    # the prior. σ_proc is folded into the dynamics, not the prior — the
    # "non-centered" parameterisation that keeps the latent prior
    # independent of hyperparameters and the LGM Gaussian.
    ε ~ IIDModel(T - 1)(τ = 1.0)

    # Forward-simulate biomass via RK4. Buffer eltype must promote
    # across both ε (latent) and the closure-captured Dual hyperparameters.
    Tp = promote_type(eltype(ε), typeof(log_K), typeof(σ_proc))
    log_B = Vector{Tp}(undef, T)
    log_B[1] = log_K
    for t in 1:(T - 1)
        F_t = min(catches[t] / exp(log_B[t]), r - 1.0e-3)
        log_B[t + 1] = schaefer_step(log_B[t], r, K, F_t, 1.0) +
            σ_proc * ε[t]
    end

    # Three independent observation likelihoods, each LogNormal(q·B, σ).
    σ_com = exp(log_σ_com)
    σ_sur = exp(log_σ_sur)
    σ_rec = exp(log_σ_rec)
    for t in 1:T
        log_I_com[t] ~ Normal(log_q_com + log_B[t], σ_com)
        log_I_sur[t] ~ Normal(log_q_sur + log_B[t], σ_sur)
        log_I_rec[t] ~ Normal(log_q_rec + log_B[t], σ_rec)
    end
end

dppl = multifleet_schaefer(
    true_log_I_com, true_log_I_sur, true_log_I_rec, true_catches, T,
)
lgm = latte_from_dppl(
    dppl; random = :ε, likelihood_hessian_pattern = :dense,
)

y_joint = vcat(true_log_I_com, true_log_I_sur, true_log_I_rec)
result = tmb(lgm, y_joint; diff_strategy = Latte.FiniteDiffStrategy())

using DataFrames
truth_natural = (
    r = true_r, K = true_K, σ_proc = true_σ_proc,
    q_com = true_q_com, σ_com = true_σ_com,
    q_sur = true_q_sur, σ_sur = true_σ_sur,
    q_rec = true_q_rec, σ_rec = true_σ_rec,
)
hp_keys = collect(keys(lgm.hyperparameter_spec.free))
summary_df_tmb = DataFrame(
    parameter = [Symbol(string(k)[5:end]) for k in hp_keys],
    truth = [getproperty(truth_natural, Symbol(string(k)[5:end])) for k in hp_keys],
    map = [exp(result.θ_map[i]) for i in eachindex(hp_keys)],
    se_working = [sqrt(max(result.θ_cov[i, i], 0.0)) for i in eachindex(hp_keys)],
)

ε_map = mean.(result.latent_marginals)
r_map = exp(result.θ_map[1]);
K_map = exp(result.θ_map[2])
σ_proc_map = exp(result.θ_map[3])
log_B_map = simulate_biomass(
    log(K_map), ε_map, true_catches, r_map, K_map, σ_proc_map,
)

biomass_df = DataFrame(
    year = repeat(1:T, 2),
    B = vcat(exp.(true_log_B), exp.(log_B_map)),
    series = repeat(["truth", "MAP"]; inner = T),
)

fig2 = Figure(size = (820, 380))
ax = Axis(
    fig2[1, 1], title = "Reconstructed biomass at the MAP",
    xlabel = "year", ylabel = "B (tonnes)",
)
lines!(ax, 1:T, exp.(true_log_B); color = :black, linewidth = 2.5, label = "truth")
lines!(ax, 1:T, exp.(log_B_map); color = :crimson, linewidth = 2, linestyle = :dash, label = "MAP")
axislegend(ax; position = :rt, framevisible = false)
fig2

result.latent_marginals[1:3]

result.log_marginal_likelihood

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
