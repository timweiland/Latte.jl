# R-INLA `int.strategy = "grid"` design for D=1 and D=2 — hardcoded
# Gauss-Hermite-style point set with precomputed quadrature weights.
#
# Ported verbatim from R-INLA's `GMRFLib_design_grid` in
# `gmrflib/design.c`. R-INLA's grid is *not* a walked Cartesian grid —
# for D ∈ {1, 2} it uses a fixed point set whose weights are precomputed
# to match a higher-order quadrature than trapezoidal would give.
# Hence `dz` and `diff.logdens` are ignored for D ≤ 2 in R-INLA, and
# this strategy similarly takes no tuning knobs.
#
# For D ≥ 3 this strategy delegates to `CCDExplorationStrategy()` —
# matching R-INLA's `int.strategy = "auto"` for high D.

# ── Hardcoded designs ────────────────────────────────────────────────

const _INLA_GRID_X1 = (
    -3.5, -2.5, -1.75, -1.0, -0.5, 0.0, 0.5, 1.0, 1.75, 2.5, 3.5,
)

const _INLA_GRID_W1 = (
    3.187537795, 1.811358205, 1.937929918, 1.431919577, 1.288639321,
    1.0,
    1.288639321, 1.431919577, 1.937929918, 1.811358205, 3.187537795,
)

# 35 points laid out in row-major order. Each row is one (z₁, z₂) pair.
# Matches the `x2[]` array in R-INLA's `gmrflib/design.c`.
const _INLA_GRID_X2 = (
    (-2.25, -1.25), (-2.25, -0.5), (-2.25, 0.0), (-2.25, 0.5), (-2.25, 1.25),
    (-1.25, -2.25), (-1.25, -1.25), (-1.25, -0.5), (-1.25, 0.0), (-1.25, 0.5), (-1.25, 1.25), (-1.25, 2.25),
    (-0.5, -2.25), (-0.5, -1.25), (-0.5, -0.5), (-0.5, 0.0), (-0.5, 0.5), (-0.5, 1.25), (-0.5, 2.25),
    (0.0, -2.25), (0.0, -1.25), (0.0, -0.5), (0.0, 0.0), (0.0, 0.5), (0.0, 1.25), (0.0, 2.25),
    (0.5, -2.25), (0.5, -1.25), (0.5, -0.5), (0.5, 0.0), (0.5, 0.5), (0.5, 1.25), (0.5, 2.25),
    (1.25, -2.25), (1.25, -1.25), (1.25, -0.5), (1.25, 0.0), (1.25, 0.5), (1.25, 1.25), (1.25, 2.25),
    (2.25, -1.25), (2.25, -0.5), (2.25, 0.0), (2.25, 0.5), (2.25, 1.25),
)

const _INLA_GRID_W2 = (
    2.277250821, 1.248862019, 1.93160554, 1.248862019, 2.277250821,
    2.277250821, 1.389904145, 0.762234217, 1.17894196, 0.762234217, 1.389904145, 2.277250821,
    1.248862019, 0.762234217, 0.4180151587, 0.646540918, 0.4180151587, 0.762234217, 1.248862019,
    1.93160554, 1.17894196, 0.646540918, 1.0, 0.646540918, 1.17894196, 1.93160554,
    1.248862019, 0.762234217, 0.4180151587, 0.646540918, 0.4180151587, 0.762234217, 1.248862019,
    2.277250821, 1.389904145, 0.762234217, 1.17894196, 0.762234217, 1.389904145, 2.277250821,
    2.277250821, 1.248862019, 1.93160554, 1.248862019, 2.277250821,
)

# Returns (z_points, weights) for D ∈ {1, 2}. Caller handles D ≥ 3.
function _inla_grid_design(d::Int)
    if d == 1
        z_points = [Float64[z] for z in _INLA_GRID_X1]
        weights = collect(_INLA_GRID_W1)
        return z_points, weights
    elseif d == 2
        z_points = [collect(Float64, p) for p in _INLA_GRID_X2]
        weights = collect(_INLA_GRID_W2)
        return z_points, weights
    else
        error("INLA grid design only defined for d ∈ {1, 2}, got d=$d")
    end
end

# ── Strategy entry point ─────────────────────────────────────────────

function explore_hyperparameter_posterior(
        strategy::INLAGridStrategy,
        model::LatentGaussianModel, y, θ_star::WorkingHyperparameters,
        marginalization_method, marginalization_indices;
        progress_callback = nothing,
        accumulators::Tuple = (),
        executor::ParallelExecutor = SequentialExecutor(),
        diff_strategy::DifferentiationStrategy = ADStrategy()
    )
    d = length(θ_star)
    if d >= 3
        return explore_hyperparameter_posterior(
            CCDExplorationStrategy(),
            model, y, θ_star, marginalization_method, marginalization_indices;
            progress_callback, accumulators, executor, diff_strategy,
        )
    end

    if progress_callback === nothing
        progress_callback = (; kwargs...) -> nothing
    end

    θ_star_nt = convert(NamedTuple, convert(NaturalHyperparameters, θ_star))
    pool = make_workspace_pool(model.latent_prior; size = _pool_size(executor), θ_star_nt...)

    progress_callback(status = "Computing reparameterization", dimensions = d)
    transform = compute_reparameterization(
        model, y, θ_star;
        pool = pool, executor = executor, diff_strategy = diff_strategy,
    )

    z_points, weights = _inla_grid_design(d)
    n_design = length(z_points)
    progress_callback(status = "INLA grid design", n_points = n_design, dimensions = d)

    # Build work items with pre-computed θ values; flag the centre point so
    # accumulators can treat it as the mode (matches CCD's `is_mode = true`).
    work_items = [
        (z = z, θ = transform(z), is_center = all(iszero, z), weight = w)
            for (z, w) in zip(z_points, weights)
    ]

    # Warm-start seed: the latent mode x*(θ*) (see CCD). Seeding every design
    # point's GA from it halves the per-point Newton iterations, KS-unchanged.
    x0_seed = with_workspace(pool) do ws
        try
            latent_mode(model, y, θ_star_nt, ws)
        catch e
            _is_numerical_failure(e) || rethrow(e)
            nothing
        end
    end

    n_inla_grid_points = length(work_items)
    eval_results = pmap_executor(
        work_items, executor, pool;
        on_complete = function (done)
            return progress_callback(
                status = "Evaluating INLA grid points",
                points_evaluated = done,
                total_points = n_inla_grid_points,
                progress = done / n_inla_grid_points,
            )
        end,
    ) do item, ws
        result = evaluate_at_grid_point(
            model, y, item.θ;
            ws = ws, x0 = x0_seed,
            compute_marginals = true,
            marginalization_method = marginalization_method,
            marginalization_indices = marginalization_indices,
        )
        summaries = if result.log_density > -Inf
            map(accumulators) do acc
                compute_point_summary(acc; result...)
            end
        else
            map(_ -> nothing, accumulators)
        end
        return (; result..., θ = item.θ, z = item.z, is_center = item.is_center, weight = item.weight, summaries)
    end

    progress_callback(status = "INLA grid evaluation complete", n_evaluated = length(eval_results))

    # Build grid points: bake `log(weight) + raw_log_density` into the
    # log_density field (same trick CCD uses with `Δ_k`) so downstream
    # integration / accumulator code reads the correct quadrature
    # weights via `get_integration_weights`.
    GP = GridPoint{typeof(θ_star)}
    grid_points = Vector{GP}()
    sizehint!(grid_points, n_design)
    weighted_log_densities = Float64[]

    for r in eval_results
        if r.log_density > -Inf
            weighted = log(r.weight) + r.log_density
            push!(grid_points, GridPoint(r.θ, weighted, r.marginal_result))
            push!(weighted_log_densities, weighted)

            if !isempty(accumulators)
                for (acc, summary) in zip(accumulators, r.summaries)
                    if summary !== nothing
                        accumulate!(acc, summary; is_mode = r.is_center)
                    end
                end
            end
        end
    end

    progress_callback(status = "Computing normalization", n_valid = length(grid_points))

    # Normalisation: Σ_k w_k · π̃(θ_k|y) · |J|. In log-space, this is
    # logsumexp of the already-weighted log_densities + logdet_jacobian.
    # No explicit cell-volume term — the weights are absolute quadrature
    # weights, not multiplicative cell sizes.
    log_normalization_constant = logsumexp(weighted_log_densities) + logdet_jacobian(transform)

    normalized_points = [
        GridPoint(p.θ, p.log_density - log_normalization_constant, p.marginal_result)
            for p in grid_points
    ]

    integration_indices = collect(1:length(normalized_points))
    accumulator_reorder = collect(1:length(normalized_points))

    exploration = GridExploration(
        normalized_points,
        integration_indices,
        transform,
        log_normalization_constant;
        accumulator_reorder = accumulator_reorder,
    )

    if !isempty(accumulators)
        progress_callback(status = "Finalizing accumulators", n_accumulators = length(accumulators))
        for acc in accumulators
            finalize!(acc, exploration)
        end
    end

    progress_callback(status = "INLA grid exploration complete", final_points = length(normalized_points))

    return exploration, accumulators
end
