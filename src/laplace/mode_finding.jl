"""
Mode finding functions for hyperparameter posterior optimization.

This module contains the core functions for finding the mode of the hyperparameter
posterior π(θ | y) and computing the associated reparameterization.
"""

using LinearAlgebra
using Optim
using Optim.LineSearches: LineSearches
using FiniteDiff
using ForwardDiff: ForwardDiff
using GaussianMarkovRandomFields: GMRF
using Distributions

export hyperparameter_logpdf, find_hyperparameter_mode, initial_hyperparameter_guess

"""
    _is_numerical_failure(e) -> Bool

Classifier for exception types that should be silently mapped to `-Inf`
(numerical failures at extreme hyperparameter values) versus exceptions
that indicate a real bug — typically AD-incompatibility, missing
methods, or malformed inputs — and should be re-thrown so the user sees
them.

Numerical: `DomainError` (e.g. `log(-x)`), Cholesky / linear-solve
failures (`PosDefException`, `SingularException`,
`LinearAlgebra.ZeroPivotException`).

NOT numerical (re-thrown): `MethodError` (typically a `Float64(::Dual)`
conversion in a non-AD-friendly path), `ArgumentError`, `BoundsError`,
etc. Without this distinction the catch sites below would silently turn
a broken AD pass into a constant `-Inf`, making the gradient zero and
fooling BFGS into early convergence.
"""
@inline _is_numerical_failure(e) =
    e isa DomainError ||
    e isa PosDefException ||
    e isa LinearAlgebra.SingularException ||
    e isa LinearAlgebra.ZeroPivotException

"""
    initial_hyperparameter_guess(spec::HyperparameterSpec)

Compute an initial guess for hyperparameter optimization in working space.

Returns `WorkingHyperparameters` with initial guesses from the prior modes/means.

# Details
Since priors are stored in working space, we directly use the mode (or mean for distributions
with boundary modes like Exponential) from the working-space prior.
"""
function initial_hyperparameter_guess(spec::HyperparameterSpec)
    θ_init = _flatten_hp_blocks([_initial_guess_for_hyperparameter(hp) for hp in values(spec.free)])
    return WorkingHyperparameters(θ_init, spec)
end

"""
    _robust_initial_value(dist::Distribution)

Compute a robust initial value for optimization from a distribution.
Uses mode by default, but can be specialized for distributions with boundary modes.
"""
function _robust_initial_value(dist::Distribution)
    try
        return mode(dist)
    catch e
        # A prior with no `Distributions.mode` falls through to the generic
        # `mode`, which tries to iterate the distribution and throws a cryptic
        # MethodError. Turn that into an actionable message.
        e isa MethodError || rethrow(e)
        throw(
            ArgumentError(
                "Hyperparameter mode-finding needs an initial guess from the prior " *
                    "$(typeof(dist)), but `Distributions.mode` is not available for it. " *
                    "Define `Distributions.mode` (or `Distributions.median`) for this prior, " *
                    "or pass an explicit `mode_init = (; hpname = value, …)` to inla / tmb / hmc_laplace.",
            ),
        )
    end
end

# Specializations for distributions with boundary modes
_robust_initial_value(dist::Exponential) = mean(dist)  # mode=0, use mean instead

# AR1Correlation's PC prior shrinks toward the base model (ρ=0), so its mode is
# on the boundary (like Exponential). Seed at the median of the |ρ| distribution
# instead: the distance d(ρ)=√(-log(1-ρ²)) is Exponential(rate λ), whose median
# is log(2)/λ.
function _robust_initial_value(dist::PCPrior.AR1Correlation)
    d_med = log(2) / dist.λ
    return sqrt(1 - exp(-d_med^2))
end

# For a TransformedDistribution we want the *working-space* mode, not the
# natural-space mode mapped through the bijector. They differ by a Jacobian
# term: π_w(u) = π_n(t⁻¹(u)) * |dt⁻¹/du|. Skipping the Jacobian (the old
# behavior) drops u into a heavy-tailed flank of the working density and can
# leave BFGS far enough from the data-driven mode to take a wild first step.
#
# Brent's method on log π_w in a bracket centered at the natural-mode-mapped
# point handles all priors uniformly without per-prior special cases.
function _robust_initial_value(dist::Bijectors.TransformedDistribution)
    base_value = _robust_initial_value(dist.dist)
    u_natural = dist.transform(base_value)
    # Vector-valued (multivariate) priors skip the 1-D Brent polish; the
    # mapped mode is a serviceable multi-dimensional starting point.
    u_natural isa AbstractVector && return u_natural
    return _working_space_mode_1d(dist, u_natural)
end

# Product-form multivariate priors delegate to the per-component rule, so
# boundary-mode components (e.g. Exponential) keep their robust seeds.
_robust_initial_value(dist::Distributions.Product) =
    [_robust_initial_value(c) for c in dist.v]

function _working_space_mode_1d(dist::Bijectors.TransformedDistribution, u0::Real)
    neg_log_pw = u -> begin
        v = try
            logpdf(dist, u)
        catch e
            _is_numerical_failure(e) || rethrow(e)
            -Inf
        end
        return isfinite(v) ? -v : Inf
    end
    # If the seed point is itself outside the working support (e.g. a
    # transformed Uniform whose mode lies on the boundary), give up cleanly.
    isfinite(neg_log_pw(u0)) || return u0
    # Shrink the bracket until both ends sit on finite density. Handles
    # half-open working-space supports where Brent would otherwise see Inf.
    lo, hi = _finite_bracket(neg_log_pw, u0)
    lo == hi && return u0
    res = Optim.optimize(neg_log_pw, lo, hi, Optim.Brent())
    return Optim.converged(res) && isfinite(Optim.minimum(res)) ?
        Optim.minimizer(res) : u0
end

function _finite_bracket(neg_log_pw, u0::Real; max_radius::Real = 10.0)
    r = max_radius
    while r > 1.0e-3
        lo_ok = isfinite(neg_log_pw(u0 - r))
        hi_ok = isfinite(neg_log_pw(u0 + r))
        if lo_ok && hi_ok
            return u0 - r, u0 + r
        end
        # Asymmetric fallback: keep the side that's finite, snap the other to u0.
        if lo_ok && !hi_ok
            return u0 - r, u0
        elseif hi_ok && !lo_ok
            return u0, u0 + r
        end
        r /= 2
    end
    return u0, u0
end

function _initial_guess_for_hyperparameter(hp::Hyperparameter{T, S}) where {T, S}
    # Prior is always stored in working space now
    # Just extract the mode/mean from the working-space prior
    return _robust_initial_value(hp.prior)
end

"""
    hyperparameter_logpdf(model::LatentGaussianModel, θ, y, ga=nothing)

Evaluate log π(θ | y) ∝ log π(θ) + log π(x*(θ), θ, y) - log π̃_G(x*(θ) | θ, y)

This is the INLA approximation to the hyperparameter posterior.

# Arguments
- `model::LatentGaussianModel`: The INLA model specification
- `θ`: Hyperparameters (WorkingHyperparameters or NaturalHyperparameters)
- `y`: Observed data
- `ga`: Optional pre-computed Gaussian approximation (GMRF object). If `nothing`, will be computed.

# Details
- Main implementation is for `WorkingHyperparameters` (working space)
- `NaturalHyperparameters` converts to working space and adds Jacobian correction
"""
# The primal (Float64) inner-Newton mode carried by a GA solution, for warm
# starts. Under a ForwardDiff-Dual θ the IFT path converges the primal Newton
# in Float64 before attaching tangents, so the Dual value parts ARE that
# converged mode — extracting them lets a fused value-and-gradient evaluation
# keep the warm start fresh without a separate primal solve. Other element
# types have no extractable primal mode.
_primal_mode(x::AbstractVector{<:AbstractFloat}) = collect(x)
function _primal_mode(x::AbstractVector{<:ForwardDiff.Dual})
    v = ForwardDiff.value.(x)
    return eltype(v) <: AbstractFloat ? v : nothing
end
_primal_mode(x) = nothing

function hyperparameter_logpdf(
        model::LatentGaussianModel, θ::WorkingHyperparameters, y, ga = nothing;
        ws, x0 = nothing, mode_out = nothing,
        mean_change_tol::Real = 1.0e-8,
        newton_dec_tol::Real = 1.0e-10,
    )
    # Inner-solve tolerances default far below the outer optimizer's g_abstol
    # (1e-6): the objective this function returns is consumed by FD gradients
    # and FD Hessian stencils, whose accuracy is bounded by the inner solve's
    # noise floor. GMRFs' own defaults (1e-4/1e-5) INVERT that ordering and
    # make the outer tolerances unreachable. Newton converges quadratically,
    # so the tighter tolerance typically costs one extra inner iteration.
    # Compute INLA approximation: log π(x*, θ, y) - log π̃_G(x* | θ, y)

    # Evaluate prior in working space
    log_prior_θ = logpdf_prior(θ)

    if log_prior_θ === -Inf
        return -Inf
    end

    # Convert to natural space for model evaluation
    θ_nt = convert(NamedTuple, convert(NaturalHyperparameters, θ))

    obs_lik = model.observation_model(y; θ_nt...)

    # Non-Gaussian latent prior (e.g. AutoDiffLatentPrior): there is no fixed GMRF to
    # materialise via `latent_gmrf`. Run the iterated-Laplace GA directly on the prior
    # and route the marginal through GMRFs' `marginal_loglikelihood`, which uses the
    # EXACT prior log-density at the mode (not a once-linearised Gaussian surrogate).
    # The Gaussian branch below is untouched (incl. its factor-reuse term ordering).
    if model.latent_prior isa NonGaussianLatentPrior
        prior = model.latent_prior
        # The workspace is reused across the whole θ-grid AND through the Dual-θ AD-gradient
        # evals: GMRFs' IFT path forwards `ws` to its primal Newton (#174), so one symbolic
        # factorisation backs both. The workspace carries the prior∪obs Hessian pattern (see the
        # make_workspace call site), so a nonlinear obs (SAM's Baranov logN×logF coupling) stays
        # inside the workspace pattern.
        x_G = ga === nothing ?
            gaussian_approximation(prior, obs_lik; θ = θ_nt, ws = ws, x0 = x0) : ga
        x_star = mean(x_G)
        if mode_out !== nothing
            m = _primal_mode(x_star)
            m !== nothing && (mode_out[] = m)
        end
        ml = marginal_loglikelihood(prior, obs_lik, x_G; θ_nt...)
        return isfinite(ml) ? log_prior_θ + ml : -Inf
    end

    latent_prior = latent_gmrf(model, ws, θ_nt)

    # The prior-logdet fast path applies only to provably unconstrained
    # priors: a plain GMRF (e.g. the Dual-θ path of the DPPL adapter), or a
    # workspace variant whose `constraints` field is empty. Anything else —
    # ConstrainedGMRF carries its constraints in the type, not a field — takes
    # the general `logpdf` fallback below, which is always correct.
    unconstrained = latent_prior isa GMRF ||
        (hasproperty(latent_prior, :constraints) && latent_prior.constraints === nothing)

    # Prior logdet BEFORE the GA: the prior density is assembled manually
    # further down (matvec quadform), so the workspace is never reloaded at
    # Q_prior after the GA and the evaluation ends with the Q_post factor
    # alive for reuse by `logpdf(x_G, ...)` and the next warm-started solve.
    ldc_prior = unconstrained ? logdetcov(latent_prior) : nothing

    if ga === nothing
        x_G = gaussian_approximation(
            latent_prior, obs_lik; x0 = x0,
            mean_change_tol = mean_change_tol, newton_dec_tol = newton_dec_tol
        )
    else
        x_G = ga
    end

    x_star = mean(x_G)
    # Expose the GA mode for callers that warm-start the next solve from it
    # (mode-finding). On Dual evaluations the value parts carry the mode.
    if mode_out !== nothing
        m = _primal_mode(x_star)
        m !== nothing && (mode_out[] = m)
    end

    # Evaluate the Gaussian (posterior) term FIRST. `latent_prior` and `x_G` share
    # one GMRFWorkspace, which holds a single numeric factorization. The Gaussian
    # approximation leaves that workspace factorized at Q_post and (via the GMRFs
    # forward-mode GA) tags x_G as its owner, so `logpdf(x_G, x_star)` reuses the
    # factor instead of refactorizing. Evaluating `logpdf(latent_prior, ...)` first
    # would reload Q_prior into the workspace and clobber that factor, forcing
    # `logpdf(x_G, ...)` to redo the Q_post factorization — one extra factorization
    # per hyperparameter gradient. (The marginal likelihood is a sum, so the term
    # order is otherwise immaterial.)
    gaussian_logpdf = logpdf(x_G, x_star)
    if !isfinite(gaussian_logpdf)
        return -Inf
    end

    if ldc_prior === nothing
        log_prior_x = logpdf(latent_prior, x_star)   # constrained fallback: old path
    else
        r_prior = x_star .- mean(latent_prior)
        log_prior_x = -0.5 * dot(r_prior, latent_prior.precision * r_prior) -
            0.5 * ldc_prior - 0.5 * length(r_prior) * log(2π)
    end
    log_likelihood = loglik(x_star, obs_lik)

    joint_logpdf = log_prior_θ + log_prior_x + log_likelihood
    if !isfinite(joint_logpdf)
        return -Inf
    end

    return joint_logpdf - gaussian_logpdf
end

function hyperparameter_logpdf(model::LatentGaussianModel, θ::NaturalHyperparameters, y, ga = nothing; ws)
    # Convert to working space and evaluate
    θ_working = convert(WorkingHyperparameters, θ)
    log_p_working = hyperparameter_logpdf(model, θ_working, y, ga; ws = ws)

    # Add Jacobian correction to get natural-space density
    return log_p_working + logdetjac(θ)
end

"""
    _StallDetector(window, f_tol)

Track objective values across outer iterations and flag a stall: `window`
consecutive iterations without an improvement larger than `f_tol`. A window
of `0` disables detection. Objective values are for minimization (smaller is
better).
"""
mutable struct _StallDetector
    best_f::Float64
    since_best::Int
    window::Int
    f_tol::Float64
    triggered::Bool
end
_StallDetector(window::Int, f_tol::Real) =
    _StallDetector(Inf, 0, window, Float64(f_tol), false)

function _stall_check!(det::_StallDetector, fval::Real)
    det.window <= 0 && return false
    if fval < det.best_f - det.f_tol
        det.best_f = fval
        det.since_best = 0
    else
        det.best_f = min(det.best_f, Float64(fval))
        det.since_best += 1
        det.since_best >= det.window && (det.triggered = true)
    end
    return det.triggered
end

"""
    _sr1_update!(B, s, y; r=1e-8) -> Bool

Symmetric-rank-1 update of the model Hessian `B` from step `s` and gradient
difference `y`. Skips the update (returning `false`) when the standard
denominator safeguard `|(y - Bs)ᵀs| ≥ r‖s‖‖y - Bs‖` fails, which also covers
the case where `B` already satisfies the secant equation.
"""
function _sr1_update!(B::AbstractMatrix, s::AbstractVector, y::AbstractVector; r::Real = 1.0e-8)
    v = y .- B * s
    denom = dot(v, s)
    # `<=` so that v = 0 (secant equation already satisfied) skips instead of
    # dividing 0/0.
    abs(denom) <= r * norm(s) * norm(v) && return false
    B .+= v .* v' ./ denom
    return true
end

"""
    _forward_diff_hessian(f, backend, x, g_center)

Model Hessian of `f` at `x` from forward differences of AD gradients, reusing
the already-computed gradient `g_center` at `x` — `length(x)` gradient
evaluations instead of the `2·length(x)` central-difference stencil of
[`ad_negative_hessian`](@ref). The O(h) truncation bias is acceptable for a
trust-region model Hessian (SR1 secant updates correct it between refreshes);
the CCD's reparameterization Hessian keeps the central-difference version.
"""
function _forward_diff_hessian(f, backend, x::AbstractVector{T}, g_center::AbstractVector) where {T}
    d = length(x)
    h = cbrt(eps(T))
    H = Matrix{T}(undef, d, d)
    xp = similar(x)
    for i in 1:d
        copyto!(xp, x)
        xp[i] += h
        gp = DifferentiationInterface.gradient(f, backend, xp)
        @. H[:, i] = (gp - g_center) / h
    end
    return (H .+ H') ./ 2
end

# Model-Hessian policy for second-order mode finding: a fresh
# forward-difference AD-gradient Hessian every `refresh_every` accepted
# iterates; between refreshes, an SR1 secant update from every gradient
# evaluation — including rejected trust-region proposals, whose curvature
# information is exactly where the model mispredicted. Each refresh costs
# dim(θ) gradient evaluations; the secant updates are free.
mutable struct _SR1HessianState
    B::Matrix{Float64}
    x_prev::Vector{Float64}
    g_prev::Vector{Float64}
    have_prev::Bool
    refreshed::Bool
    refresh_every::Int
    updates_since_refresh::Int
end
_SR1HessianState(d::Int, refresh_every::Int) = _SR1HessianState(
    Matrix{Float64}(I, d, d), zeros(d), zeros(d), false, false, refresh_every, 0,
)

# One secant pair per gradient evaluation. Steps below the gradient-noise
# scale are skipped without advancing the anchor (the pair to the next
# far-enough point stays valid — SR1 pairs need not be consecutive), so a
# noise-dominated y never enters the model.
function _observe_pair!(hs::_SR1HessianState, x::AbstractVector, g::AbstractVector)
    if !hs.have_prev
        copyto!(hs.x_prev, x)
        copyto!(hs.g_prev, g)
        hs.have_prev = true
        return nothing
    end
    s = x .- hs.x_prev
    norm(s) < 1.0e-6 && return nothing
    _sr1_update!(hs.B, s, g .- hs.g_prev)
    copyto!(hs.x_prev, x)
    copyto!(hs.g_prev, g)
    return nothing
end

"""
    _resolve_mode_method(method, diff_strategy, d) -> Optim method

Resolve the default optimization method for mode finding. An explicitly
passed `method` is used as-is. The default (`nothing`) is trust-region
Newton for AD gradients, `dim(θ) ≤ 6`, and models that run the inner-Newton
warm start (`warm`) — measured dominant or tied on every benchmark family
at d ∈ {1, 2, 4}, and mechanically equivalent at 5–6, where a model-Hessian
refresh still costs ≈ d/5 gradient evaluations per accepted iterate. BFGS +
backtracking otherwise: finite-difference gradients are too noisy to
difference into a model Hessian; at higher dim the refresh cost dominates
and is unmeasured; and the trust region's fused value-and-gradient
evaluations assume the warm start, which augmented models disable — without
it the gradient certificate degrades.
"""
function _resolve_mode_method(method, diff_strategy, d::Int, warm::Bool)
    method === nothing || return method
    diff_strategy isa ADStrategy && warm && d <= 6 && return NewtonTrustRegion()
    return BFGS(linesearch = LineSearches.BackTracking(order = 3, maxstep = 5.0))
end

# A stall with a near-tolerance gradient is a benign stop at the objective
# noise floor: the mode is found, only the g_abstol certificate is out of
# reach. Observed benign stalls sit at |∇| ≈ 1e-5–5e-3; genuinely stuck
# optimizations carry |∇| well above 1e-1.
_benign_stall(g_norm) = isfinite(g_norm) && g_norm < 1.0e-2

"""
    find_hyperparameter_mode(model::LatentGaussianModel, y; method=nothing, collect_points=true, progress_callback=nothing)

Find the mode θ* of the hyperparameter posterior π(θ | y).

# Arguments
- `model`: INLA model specification
- `y`: Observed data
- `method`: Optimization method (from Optim.jl). The default (`nothing`)
  resolves per differentiation strategy, dimension, and warm-start
  availability: `NewtonTrustRegion()` for AD gradients, `dim(θ) ≤ 6`, and
  non-augmented models; `BFGS` + backtracking otherwise
  (`_resolve_mode_method`). First-order methods use the AD gradient.
  Second-order methods (`NewtonTrustRegion()`, `Newton()`) additionally get a
  model Hessian built from forward differences of AD gradients, refreshed
  every `hessian_refresh` accepted iterates with SR1 secant updates from
  every gradient evaluation in between, and require
  `diff_strategy = ADStrategy()`.
- `collect_points`: Whether to collect intermediate points during optimization
- `progress_callback`: Optional function for progress updates with signature `f(; kwargs...)`
- `hessian_refresh`: For second-order methods, how many accepted iterates to
  carry SR1 curvature updates before recomputing the forward-difference AD
  Hessian; `0` computes it only once at the start. The default (`nothing`)
  resolves to 1 for `dim(θ) ≤ 2` — there a refresh costs at most two gradient
  evaluations, and low-dimensional secant updates go stale near the mode —
  and 5 for higher dimensions, where the refresh is the dominant cost.
- `stall_iterations`, `stall_f_tol`: Stop an optimization early when
  `stall_iterations` consecutive outer iterations improve the objective by less
  than `stall_f_tol` (roughly the inner-solve noise floor). The best point
  found is still returned and the stop is reported in `mode_info.stalled`.
  The default (`nothing`) resolves to 6 for second-order methods (accepted
  trust-region steps decrease f monotonically, so a sub-noise run is terminal)
  and 50 for line-search methods (which may pass through f-increases).
  `stall_iterations = 0` disables the check.

# Returns
- `θ_star`: The posterior mode in working space (WorkingHyperparameters)
- `mode_points`: WorkingHyperparameters evaluated during optimization (if collect_points=true)
- `mode_logdensities`: Log-densities at mode_points (if collect_points=true)

# Details
Optimization is performed in working (unconstrained) space. The mode is returned in working space.
"""
function find_hyperparameter_mode(
        model::LatentGaussianModel, y;
        method = nothing,
        iterations::Int = 1000,
        collect_points = true, progress_callback = nothing,
        diff_strategy::DifferentiationStrategy = ADStrategy(),
        mode_init = PriorModeStart(),
        latent_init = ZeroLatentStart(),
        executor::ParallelExecutor = SequentialExecutor(),
        warm_start::Union{Nothing, Bool} = nothing,
        hessian_refresh::Union{Nothing, Int} = nothing,
        stall_iterations::Union{Nothing, Int} = nothing,
        stall_f_tol::Real = 1.0e-8,
    )
    # Normalize y (Vector{Int} → PoissonObservations, etc.) so direct
    # callers behave the same as inla() / tmb() which pre-wrap via
    # `_prepare_for_prediction`. Without this the objective's try/catch
    # would silently swallow the obs-model MethodError and return -Inf,
    # making the gradient zero and BFGS terminate at the starting point.
    y, model, _ = _prepare_for_prediction(model, y)
    spec = model.hyperparameter_spec

    # Warm-start each θ-step's GA Newton solve from the previous step's mode.
    # Default: on for compact models (well-conditioned ⇒ the GA mode is start-
    # invariant, so this only saves Newton iterations); off for augmented models,
    # whose ~1e13 conditioning makes the loosely-converged mode start-dependent
    # and shifts θ* in the flat (weakly-identified) directions.
    do_warm = warm_start === nothing ? (model.augmentation_info === nothing) : warm_start

    starts = resolve_mode_starts(mode_init, spec)
    n_starts = length(starts)

    method = _resolve_mode_method(method, diff_strategy, length(first(starts)), do_warm)

    # Trust-region accepted steps decrease f monotonically and, once a
    # rejection has shrunk the radius, the quadratic model is first-order
    # exact in the remaining region — so a short run of sub-noise iterations
    # is a terminal signature (gradient at the noise floor), not a plateau.
    # Line-search quasi-Newton can wander through f-increases before
    # recovering, so it gets a wide window.
    stall_window = stall_iterations !== nothing ? stall_iterations :
        (method isa Optim.SecondOrderOptimizer ? 6 : 50)

    # A full model-Hessian refresh costs dim(θ) gradient evaluations. At
    # dim(θ) ≤ 2 that is at most two evaluations — cheaper than one wasted
    # rejection from a stale model — and the low-dimensional secant updates
    # are the most noise-fragile: near the mode the pair guard leaves B
    # frozen at its last refresh exactly when the endgame needs accurate
    # curvature (measured: refresh-1 certifies where refresh-5 freezes at
    # the noise floor at d = 1). At higher dim the refresh is the dominant
    # cost and a 5-iterate cadence measures best.
    refresh = hessian_refresh !== nothing ? hessian_refresh :
        (length(first(starts)) <= 2 ? 1 : 5)

    # Resolve the FIRST inner-Newton latent start (latent_init). Subsequent θ-steps warm-start
    # from the previous mode; this only seeds each start's first GA solve. Resolved once at the
    # first θ start (the prior mode is a basin-correct seed for all starts).
    _latent_x0 = resolve_latent_start(
        latent_init, model, convert(NamedTuple, convert(NaturalHyperparameters, first(starts))),
    )

    if progress_callback === nothing
        progress_callback = (; kwargs...) -> nothing
    end

    best_θ = first(starts)
    best_logp = -Inf
    best_idx = 0
    best_points = WorkingHyperparameters[]
    best_logps = Float64[]
    best_converged = false
    final_logdensities = Vector{Float64}(undef, n_starts)
    any_converged = false

    # Each start is an independent optimisation from its own working-space
    # initial point; the only cross-start coupling is selecting the best mode
    # afterwards. Run them through the executor with a pooled workspace per
    # task — sequential reuses one workspace, threaded checks out one per
    # worker via `with_workspace` so concurrent factor-updates never race. The
    # precision pattern is θ-invariant, so a pool built at the first start
    # covers every start (each refactorises numerically at its own θ).
    report_progress = executor isa SequentialExecutor

    function _one_start((i, θ_init), ws)
        points = WorkingHyperparameters[]
        logps = Float64[]
        # Shared warm-start state: the last primal GA mode, reused as the Newton
        # seed by both the value and gradient evaluations. Only primal (Float64)
        # value evals update it; gradient (Dual) evals read it.
        last_mode = Ref{Union{Nothing, Vector{Float64}}}(
            _latent_x0 === nothing ? nothing : copy(_latent_x0),
        )
        mode_buf = Ref(Float64[])

        objective = let _spec = spec, _model = model, _y = y, _ws = ws,
                _points = points, _logps = logps, _collect = collect_points,
                _warm = do_warm, _last = last_mode, _buf = mode_buf
            function (θ_vec)
                θ = WorkingHyperparameters(θ_vec, _spec)
                logpdf_val = 0.0
                try
                    logpdf_val = hyperparameter_logpdf(
                        _model, θ, _y; ws = _ws,
                        x0 = _warm ? _last[] : nothing,
                        mode_out = _warm ? _buf : nothing,
                    )
                catch e
                    _is_numerical_failure(e) || rethrow(e)
                    return Inf
                end
                isfinite(logpdf_val) || return Inf
                _warm && !isempty(_buf[]) && (_last[] = copy(_buf[]))
                if _collect
                    push!(_points, WorkingHyperparameters(copy(θ_vec), _spec))
                    push!(_logps, logpdf_val)
                end
                return -logpdf_val
            end
        end

        record! = function (θ_vec, logp)
            collect_points || return nothing
            push!(points, WorkingHyperparameters(copy(θ_vec), spec))
            push!(logps, logp)
            return nothing
        end

        stall = _StallDetector(stall_window, stall_f_tol)
        optim_callback = function (state)
            report_progress && progress_callback(
                start_index = i, n_starts = n_starts,
                iteration = state.iteration,
                objective = state.value,
                gradient_norm = state.g_norm,
            )
            return _stall_check!(stall, state.value)
        end

        options = Optim.Options(
            f_reltol = 0.0,
            f_abstol = 0.0,
            g_abstol = 1.0e-6,
            x_reltol = 0.0,
            x_abstol = 0.0,
            iterations = iterations,
            show_trace = false,
            allow_f_increases = true,
            callback = optim_callback,
        )

        result, neg_hess = _run_optimization(
            diff_strategy, objective, model, y, spec, θ_init, ws, method, options,
            last_mode, do_warm; hessian_refresh = refresh, record! = record!,
        )
        return (
            idx = i,
            θ_final = WorkingHyperparameters(Optim.minimizer(result), spec),
            final_logp = -Optim.minimum(result),
            converged = Optim.converged(result),
            stalled = stall.triggered,
            g_norm = Optim.g_residual(result),
            negative_hessian = neg_hess,
            points = points,
            logps = logps,
        )
    end

    θ0_nt = convert(NamedTuple, convert(NaturalHyperparameters, first(starts)))
    pool = make_workspace_pool(model.latent_prior; size = _pool_size(executor), θ0_nt...)
    on_complete = report_progress ? nothing :
        function (done)
            progress_callback(
                start_index = done, n_starts = n_starts,
                iteration = 0, objective = NaN, gradient_norm = NaN,
            )
            return nothing
    end
    results = pmap_executor(
        _one_start, collect(enumerate(starts)), executor, pool; on_complete = on_complete,
    )

    best_stalled = false
    best_g_norm = NaN
    best_neg_hess = nothing
    for r in results
        final_logdensities[r.idx] = r.final_logp
        any_converged |= r.converged
        if r.final_logp > best_logp
            best_logp = r.final_logp
            best_θ = r.θ_final
            best_idx = r.idx
            best_points = r.points
            best_logps = r.logps
            best_converged = r.converged
            best_stalled = r.stalled
            best_g_norm = r.g_norm
            best_neg_hess = r.negative_hessian
        end
    end

    if !any_converged
        if _benign_stall(best_g_norm)
            # Near-tolerance gradient: the mode is found, only the g_abstol
            # certificate is out of reach at the objective noise floor —
            # regardless of whether the stop came from the stall detector, a
            # collapsed trust radius, or a failed line search.
            @info "Hyperparameter mode optimization stopped near the mode without " *
                "certifying the gradient tolerance (best log-density " *
                "$(round(best_logp; digits = 3)), gradient norm " *
                "$(round(best_g_norm; sigdigits = 3)))."
        elseif best_stalled
            @warn "Hyperparameter mode optimization stalled: no objective improvement " *
                "beyond $stall_f_tol over $stall_window consecutive iterations " *
                "(best log-density $(round(best_logp; digits = 3)), " *
                "gradient norm $(round(best_g_norm; sigdigits = 3))). " *
                "Returning the best point found; the gradient tolerance was not certified."
        else
            @warn "Hyperparameter mode optimization did not converge for any start " *
                "(n_starts = $n_starts)"
        end
    end

    # Runner-up gap (best - second best). nothing for single-start runs.
    runner_up_gap = if n_starts >= 2
        sorted = sort(final_logdensities; rev = true)
        sorted[1] - sorted[2]
    else
        nothing
    end

    mode_info = (
        n_starts = n_starts,
        best_start_index = best_idx,
        final_logdensities = final_logdensities,
        converged = best_converged,
        stalled = best_stalled,
        gradient_norm = best_g_norm,
        # Second-order methods: the model Hessian (negative log-posterior
        # curvature) at the selected mode, reusable as the exploration
        # reparameterization. `nothing` for first-order methods.
        negative_hessian = best_neg_hess,
        runner_up_gap = runner_up_gap,
    )

    if collect_points
        return best_θ, best_points, best_logps, mode_info
    else
        return best_θ, nothing, nothing, mode_info
    end
end

function _run_optimization(::FiniteDiffStrategy, objective, model, y, spec, θ_init, ws, method, options, last_mode, do_warm; hessian_refresh::Int = 5, record! = (θ, lp) -> nothing)
    method isa Optim.SecondOrderOptimizer && throw(
        ArgumentError(
            "Second-order mode-finding methods ($(typeof(method).name.name)) build their " *
                "Hessian from AD gradients; use diff_strategy = ADStrategy() (the default).",
        ),
    )
    # The passed `objective` already warm-starts (and updates last_mode) on every
    # primal evaluation, which is all finite differencing does.
    return Optim.optimize(objective, θ_init.θ, method, options), nothing
end

function _run_optimization(strategy::ADStrategy, objective, model, y, spec, θ_init, ws, method, options, last_mode, do_warm; hessian_refresh::Int = 5, record! = (θ, lp) -> nothing)
    # Clean objective for AD (no side effects, safe for Dual numbers).
    # ws is captured by the closure; AD flows through the numeric values only.
    # The Dual gradient solve reads the last primal mode as its Newton seed (a
    # plain Float64 vector, promoted by the solve) but never updates it.
    function objective_clean(θ_vec)
        θ = WorkingHyperparameters(θ_vec, spec)
        logpdf_val = try
            hyperparameter_logpdf(model, θ, y; ws = ws, x0 = do_warm ? last_mode[] : nothing)
        catch e
            _is_numerical_failure(e) || rethrow(e)
            oftype(θ_vec[1], -Inf)
        end
        return isfinite(logpdf_val) ? -logpdf_val : oftype(logpdf_val, Inf)
    end

    if method isa Optim.SecondOrderOptimizer
        # Dual-safe objective that also refreshes the warm-start mode: the IFT
        # path converges the primal Newton before attaching tangents, and
        # `_primal_mode` extracts that mode from the Dual GA solution. The
        # second-order path evaluates value-and-gradient fused (one Dual pass,
        # no separate primal solve), so without this the warm start would stay
        # frozen at the initial mode.
        mode_buf = Ref(Float64[])
        function objective_ad(θ_vec)
            θ = WorkingHyperparameters(θ_vec, spec)
            logpdf_val = try
                hyperparameter_logpdf(
                    model, θ, y; ws = ws,
                    x0 = do_warm ? last_mode[] : nothing,
                    mode_out = do_warm ? mode_buf : nothing,
                )
            catch e
                _is_numerical_failure(e) || rethrow(e)
                oftype(θ_vec[1], -Inf)
            end
            isfinite(logpdf_val) || return oftype(logpdf_val, Inf)
            do_warm && !isempty(mode_buf[]) && (last_mode[] = copy(mode_buf[]))
            return -logpdf_val
        end
        td, hs = _second_order_objective(
            strategy, objective, objective_ad, θ_init.θ, hessian_refresh, record!,
        )
        result = Optim.optimize(td, θ_init.θ, method, options)
        # Hand the model Hessian at the final accepted iterate to the caller —
        # exploration reuses it as the CCD/grid reparameterization curvature,
        # skipping its own stencil. Guarded on `refreshed`: before the first
        # full refresh B is only the identity placeholder.
        return result, hs.refreshed ? Matrix(Symmetric(hs.B)) : nothing
    end

    # Explicit gradient via DifferentiationInterface
    function gradient!(G, θ_vec)
        return copyto!(G, DifferentiationInterface.gradient(objective_clean, strategy.backend, θ_vec))
    end

    return Optim.optimize(objective, gradient!, θ_init.θ, method, options), nothing
end

# TwiceDifferentiable plumbing for trust-region / Newton mode finding. Value
# and gradient are computed in one Dual pass of `objective_ad` (its primal
# part is exact and it keeps the warm start fresh); the side-effecting primal
# `objective` only serves value-only requests; the Hessian is the
# SR1-with-periodic-refresh model above.
function _second_order_objective(
        strategy::ADStrategy, objective, objective_ad,
        θ0::AbstractVector{Float64}, hessian_refresh::Int, record!,
    )
    d = length(θ0)
    hs = _SR1HessianState(d, hessian_refresh)
    # Memo of the most recent gradient evaluation: the Hessian refresh needs
    # the gradient at its evaluation point, which the optimizer has always
    # just computed. The fallback recomputation should never fire in practice.
    memo_x = fill(NaN, d)
    memo_g = fill(NaN, d)

    function gradient!(G, θ_vec)
        DifferentiationInterface.gradient!(objective_ad, G, strategy.backend, θ_vec)
        copyto!(memo_x, θ_vec)
        copyto!(memo_g, G)
        _observe_pair!(hs, θ_vec, G)
        return G
    end

    function fg!(G, θ_vec)
        fval, _ = DifferentiationInterface.value_and_gradient!(
            objective_ad, G, strategy.backend, θ_vec,
        )
        copyto!(memo_x, θ_vec)
        copyto!(memo_g, G)
        if isfinite(fval)
            _observe_pair!(hs, θ_vec, G)
            record!(θ_vec, -fval)
        end
        return fval
    end

    function h!(H, θ_vec)
        refresh_due = hs.refresh_every > 0 &&
            hs.updates_since_refresh >= hs.refresh_every
        if !hs.refreshed || refresh_due
            g = θ_vec == memo_x ? memo_g : gradient!(similar(memo_g), θ_vec)
            hs.B = _forward_diff_hessian(objective_ad, strategy.backend, θ_vec, g)
            copyto!(hs.x_prev, θ_vec)
            copyto!(hs.g_prev, g)
            hs.have_prev = true
            hs.refreshed = true
            hs.updates_since_refresh = 0
        else
            hs.updates_since_refresh += 1
        end
        copyto!(H, hs.B)
        return H
    end

    return Optim.TwiceDifferentiable(objective, gradient!, fg!, h!, collect(θ0)), hs
end
