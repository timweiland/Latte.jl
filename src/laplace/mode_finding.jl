"""
Mode finding functions for hyperparameter posterior optimization.

This module contains the core functions for finding the mode of the hyperparameter
posterior π(θ | y) and computing the associated reparameterization.
"""

using LinearAlgebra
using Optim
using Optim.LineSearches: LineSearches
using FiniteDiff
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
    θ_init = [_initial_guess_for_hyperparameter(hp) for hp in values(spec.free)]
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
    return _working_space_mode_1d(dist, u_natural)
end

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
function hyperparameter_logpdf(
        model::LatentGaussianModel, θ::WorkingHyperparameters, y, ga = nothing;
        ws, x0 = nothing, mode_out = nothing,
    )
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
        mode_out !== nothing && eltype(x_star) <: AbstractFloat && (mode_out[] = collect(x_star))
        ml = marginal_loglikelihood(prior, obs_lik, x_G; θ_nt...)
        return isfinite(ml) ? log_prior_θ + ml : -Inf
    end

    latent_prior = latent_gmrf(model, ws, θ_nt)

    # Use provided Gaussian approximation or compute it
    if ga === nothing
        # Find Gaussian approximation (warm-start from x0 if provided)
        x_G = gaussian_approximation(latent_prior, obs_lik; x0 = x0)
    else
        x_G = ga
    end

    x_star = mean(x_G)
    # Expose the GA mode for callers that warm-start the next solve from it
    # (mode-finding). Only meaningful on primal (Float64) evaluations.
    mode_out !== nothing && eltype(x_star) <: AbstractFloat && (mode_out[] = collect(x_star))

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

    log_prior_x = logpdf(latent_prior, x_star)
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
    find_hyperparameter_mode(model::LatentGaussianModel, y; method=BFGS(), collect_points=true, progress_callback=nothing)

Find the mode θ* of the hyperparameter posterior π(θ | y).

# Arguments
- `model`: INLA model specification
- `y`: Observed data
- `method`: Optimization method (from Optim.jl)
- `collect_points`: Whether to collect intermediate points during optimization
- `progress_callback`: Optional function for progress updates with signature `f(; kwargs...)`

# Returns
- `θ_star`: The posterior mode in working space (WorkingHyperparameters)
- `mode_points`: WorkingHyperparameters evaluated during optimization (if collect_points=true)
- `mode_logdensities`: Log-densities at mode_points (if collect_points=true)

# Details
Optimization is performed in working (unconstrained) space. The mode is returned in working space.
"""
function find_hyperparameter_mode(
        model::LatentGaussianModel, y;
        method = BFGS(linesearch = LineSearches.BackTracking(order = 3, maxstep = 5.0)),
        iterations::Int = 1000,
        collect_points = true, progress_callback = nothing,
        diff_strategy::DifferentiationStrategy = ADStrategy(),
        mode_init = PriorModeStart(),
        latent_init = ZeroLatentStart(),
        executor::ParallelExecutor = SequentialExecutor(),
        warm_start::Union{Nothing, Bool} = nothing,
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

        optim_callback = function (state)
            report_progress && progress_callback(
                start_index = i, n_starts = n_starts,
                iteration = state.iteration,
                objective = state.value,
                gradient_norm = state.g_norm,
            )
            return false
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

        result = _run_optimization(
            diff_strategy, objective, model, y, spec, θ_init, ws, method, options,
            last_mode, do_warm,
        )
        return (
            idx = i,
            θ_final = WorkingHyperparameters(Optim.minimizer(result), spec),
            final_logp = -Optim.minimum(result),
            converged = Optim.converged(result),
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
        end
    end

    if !any_converged
        @warn "Hyperparameter mode optimization did not converge for any start " *
            "(n_starts = $n_starts)"
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
        runner_up_gap = runner_up_gap,
    )

    if collect_points
        return best_θ, best_points, best_logps, mode_info
    else
        return best_θ, nothing, nothing, mode_info
    end
end

function _run_optimization(::FiniteDiffStrategy, objective, model, y, spec, θ_init, ws, method, options, last_mode, do_warm)
    # The passed `objective` already warm-starts (and updates last_mode) on every
    # primal evaluation, which is all finite differencing does.
    return Optim.optimize(objective, θ_init.θ, method, options)
end

function _run_optimization(strategy::ADStrategy, objective, model, y, spec, θ_init, ws, method, options, last_mode, do_warm)
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

    # Explicit gradient via DifferentiationInterface
    function gradient!(G, θ_vec)
        return copyto!(G, DifferentiationInterface.gradient(objective_clean, strategy.backend, θ_vec))
    end

    return Optim.optimize(objective, gradient!, θ_init.θ, method, options)
end
