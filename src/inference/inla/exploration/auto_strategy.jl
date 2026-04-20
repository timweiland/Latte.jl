# AutoExplorationStrategy dispatch: picks grid for small θ-spaces, CCD for
# large ones. Must load after grid.jl and ccd.jl (uses both strategies).
function explore_hyperparameter_posterior(
        strategy::AutoExplorationStrategy,
        model::LatentGaussianModel, y, θ_star::WorkingHyperparameters,
        marginalization_method, marginalization_indices;
        kwargs...
    )
    actual_strategy = length(θ_star) > 2 ? strategy.ccd : strategy.grid
    return explore_hyperparameter_posterior(
        actual_strategy, model, y, θ_star, marginalization_method, marginalization_indices;
        kwargs...
    )
end
