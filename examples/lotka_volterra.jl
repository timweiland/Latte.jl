# Phase 2 prototype: Latte fit of a Lotka–Volterra ODE.
#
# Model:
#   du[1]/dt = α·u[1] − β·u[1]·u[2]
#   du[2]/dt = −γ·u[2] + δ·u[1]·u[2]
# Latent (random-effect block): log-parameters log_p = [log α, log β, log γ, log δ].
# Hyperparameter: σ (Gaussian obs noise).
# Observations: (prey, predator) at a handful of time points.

using Latte
using DynamicPPL: @model
using Distributions
using LinearAlgebra
using Random
using OrdinaryDiffEqTsit5
using Printf

# ── ODE ─────────────────────────────────────────────────────────────────
function lv!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[1] * u[2]
    du[2] = -γ * u[2] + δ * u[1] * u[2]
    return
end

# Prior mean on log-parameters, centred on the "true" values.
μ0 = log.([1.5, 1.0, 3.0, 1.0])

# ── DPPL model ──────────────────────────────────────────────────────────
@model function lv_fit(Y, t_obs, u0, μ0)
    σ ~ Gamma(2.0, 0.5)
    log_p ~ MvNormal(μ0, 0.5^2 * I(4))

    prob = ODEProblem(lv!, u0, (t_obs[1], t_obs[end]), exp.(log_p))
    sol = solve(prob, Tsit5(); saveat = t_obs, abstol = 1.0e-8, reltol = 1.0e-8)

    # Y is a 2×n matrix: first row prey, second row predator
    for i in eachindex(t_obs)
        Y[1, i] ~ Normal(sol.u[i][1], σ)
        Y[2, i] ~ Normal(sol.u[i][2], σ)
    end
end

# ── Simulate data ───────────────────────────────────────────────────────
Random.seed!(20260423)
p_true = [1.5, 1.0, 3.0, 1.0]
u0 = [1.0, 1.0]
t_obs = collect(range(0.0, 10.0; length = 21))
prob_truth = ODEProblem(lv!, u0, (0.0, 10.0), p_true)
sol_truth = solve(prob_truth, Tsit5(); saveat = t_obs, abstol = 1.0e-10, reltol = 1.0e-10)
σ_true = 0.25
Y = hcat(sol_truth.u...) .+ σ_true .* randn(2, length(t_obs))

println("True parameters: α,β,γ,δ = ", p_true)
println("True σ:          ", σ_true)
println("Observations:    2 × $(length(t_obs)) = $(2 * length(t_obs)) scalar measurements")

# ── Build LGM and fit via TMB ───────────────────────────────────────────
@info "Building LGM via latte_from_dppl (defaults — adapter falls back through the ODE solver gracefully)"
lgm = latte_from_dppl(
    lv_fit(Y, t_obs, u0, μ0);
    random = (:log_p,),
)

@info "Observation model type" typeof(lgm.observation_model).name.name

@info "Running TMB (FiniteDiffStrategy — avoids nested-AD issue through AD obs model)"
t0 = time()
# Y is 2×n; Latte's outer `y` channel expects a vector, but the DPPL model
# closure already holds Y, so this is metadata-only.
result = tmb(lgm, vec(Y); diff_strategy = FiniteDiffStrategy())
@printf "TMB done in %.2f s\n\n" (time() - t0)

# ── Results ─────────────────────────────────────────────────────────────
log_p_post = mean.(latent_marginals(result))
base_idx = lgm.augmentation_info === nothing ?
    eachindex(log_p_post) : lgm.augmentation_info.base_latent_indices
p_hat = exp.(log_p_post[base_idx])
σ_hat = convert(NamedTuple, hyperparameter_mode(result)).σ

println("Posterior point estimates (TMB mode):")
for (name, est, truth) in zip(
        ("α", "β", "γ", "δ"), p_hat, p_true,
    )
    @printf "  %s = %.3f  (true %.3f)\n" name est truth
end
@printf "  σ = %.3f  (true %.3f)\n" σ_hat σ_true
