# PrecompileTools workload for Latte.
#
# Goal: shrink the cold-start cost of `inla(lgm, y)` so that
# statisticians don't bounce on first-call latency. Runs a small
# end-to-end `inla()` cycle on each fast-path likelihood family
# (Poisson / Bernoulli / Binomial / Normal) with a 1-hyperparameter
# IID-Gaussian latent prior. This forces type inference + native code
# specialisation across the hot path: mode finding, GMRF gaussian
# approximation, grid exploration, and SimplifiedLaplace's
# SkewNormal correction loop.
#
# What this CANNOT precompile:
#   * `using Latte` itself (package load is dominated by deps).
#   * `latte_from_dppl(dppl_model)` for arbitrary user models — DPPL
#     specialises on the `typeof(@model_function)`, which only exists
#     at user-call time. Users who want zero TTFX should put their own
#     `Latte.warmup(model, y)` call inside their app's
#     `@compile_workload` or PackageCompiler sysimage.
#
# Coverage rationale: each LGM here matches the *augmented*
# `LinearlyTransformedObservationModel` shape that `latte_from_dppl`'s
# fast-path produces for `y[i] ~ Family(link(linear(x)))` patterns —
# `LatentGaussianModel(spec, IIDModel, LTM(EF, A))` unwraps into
# `(AugmentedLatentModel, base_obs_model)`, which is what the inla()
# pipeline actually dispatches on. A final default-kwargs call covers
# the AdaptiveMarginal + AutoExploration + DIC/MLL/WAIC/CPO chain hit
# by users who type `inla(lgm, y)` with no kwargs.
using PrecompileTools: PrecompileTools, @setup_workload, @compile_workload

@setup_workload begin
    using Distributions: LogNormal, Beta, Poisson, Bernoulli, Binomial, Normal
    using GaussianMarkovRandomFields: IIDModel, ExponentialFamily, LinearlyTransformedObservationModel
    using SparseArrays: sparse, SparseMatrixCSC

    _ttfx_n = 4
    _ttfx_y_pois = [3, 0, 1, 2]
    _ttfx_y_bern = [true, false, true, false]
    _ttfx_y_binom = [3, 1, 4, 2]
    _ttfx_y_norm = [0.1, -0.2, 0.0, 0.3]

    # Single-hp spec used by IID Poisson/Bernoulli/Binomial. Normal needs
    # a σ on top, defined just below.
    _ttfx_spec = @hyperparams begin
        (τ ~ LogNormal(0.0, 1.0), transform = log, space = natural)
    end
    _ttfx_norm_spec = @hyperparams begin
        (τ ~ LogNormal(0.0, 1.0), transform = log, space = natural)
        (σ ~ LogNormal(0.0, 1.0), transform = log, space = natural)
    end

    # Sparse identity design matrix (n_obs × n_base, both = _ttfx_n). The
    # DPPL fast-path adapter produces LTM(EF, A)-shaped observation models
    # for `y[i] ~ Family(link(linear(x)))` patterns — which the LGM ctor
    # then unwraps into `(AugmentedLatentModel, base_obs_model)`. Building
    # the workload LGMs via the LTM constructor matches that production
    # path exactly, so precompile coverage transfers to any DPPL fast-path
    # user.
    _ttfx_A::SparseMatrixCSC{Float64, Int64} = sparse(1.0I, _ttfx_n, _ttfx_n)

    _ttfx_pois_lgm = LatentGaussianModel(
        _ttfx_spec, IIDModel(_ttfx_n),
        LinearlyTransformedObservationModel(ExponentialFamily(Poisson), _ttfx_A),
    )
    _ttfx_bern_lgm = LatentGaussianModel(
        _ttfx_spec, IIDModel(_ttfx_n),
        LinearlyTransformedObservationModel(ExponentialFamily(Bernoulli), _ttfx_A),
    )
    _ttfx_binom_lgm = LatentGaussianModel(
        _ttfx_spec, IIDModel(_ttfx_n),
        LinearlyTransformedObservationModel(
            BinomialTrialsObservationModel(ExponentialFamily(Binomial), fill(5, _ttfx_n)),
            _ttfx_A,
        ),
    )
    _ttfx_norm_lgm = LatentGaussianModel(
        _ttfx_norm_spec, IIDModel(_ttfx_n),
        LinearlyTransformedObservationModel(ExponentialFamily(Normal), _ttfx_A),
    )

    # Tight, deterministic exploration grid. We don't care about
    # accuracy here — only that the methods specialise.
    _ttfx_grid_strategy = GridExplorationStrategy(
        integration_step_z = 1.0,
        max_log_drop = 1.5,
        interpolation_subdivisions = 1,
    )

    # The default `inla(lgm, y)` call path goes through
    # `AutoExplorationStrategy()`, which dispatches to
    # `GridExplorationStrategy` for D ≤ 2. We hit that wrapper at least
    # once below so the auto-dispatch path is also baked in.
    _ttfx_auto_strategy = AutoExplorationStrategy(
        low_dim = _ttfx_grid_strategy,
        high_dim = _ttfx_grid_strategy,
    )

    @compile_workload begin
        # SimplifiedLaplace is the most common latent strategy and exercises
        # the SkewNormal/γ_3 path. AdaptiveMarginal in Gaussian-likelihood
        # cases short-circuits to GaussianMarginal, so we cover both
        # explicitly via Poisson (SLA) and Normal (Gaussian).
        inla(
            _ttfx_pois_lgm, _ttfx_y_pois;
            latent_marginalization_method = SimplifiedLaplace(),
            exploration_strategy = _ttfx_grid_strategy,
            progress = false, accumulators = (),
        )
        inla(
            _ttfx_bern_lgm, _ttfx_y_bern;
            latent_marginalization_method = SimplifiedLaplace(),
            exploration_strategy = _ttfx_grid_strategy,
            progress = false, accumulators = (),
        )
        inla(
            _ttfx_binom_lgm, _ttfx_y_binom;
            latent_marginalization_method = SimplifiedLaplace(),
            exploration_strategy = _ttfx_grid_strategy,
            progress = false, accumulators = (),
        )
        inla(
            _ttfx_norm_lgm, _ttfx_y_norm;
            latent_marginalization_method = GaussianMarginal(),
            exploration_strategy = _ttfx_grid_strategy,
            progress = false, accumulators = (),
        )
        # One Poisson run with the **default method stack** —
        # AdaptiveMarginal + AutoHyperparameterMarginal + the full
        # DIC/MLL/WAIC/CPO accumulator chain — wrapped in
        # AutoExplorationStrategy so the auto-dispatch shape that
        # `inla(lgm, y)` (no kwargs) hits is also baked in. We pin
        # the cheap-grid AutoExplorationStrategy variant (not the
        # default-budget one) to keep precompile time bounded; the
        # method specialisations are the same.
        inla(
            _ttfx_pois_lgm, _ttfx_y_pois;
            exploration_strategy = _ttfx_auto_strategy,
            progress = false,
        )
    end
end
