# Include modular components in dependency order
include("adaptive_hessian.jl")  # Must come before transformation.jl
include("transformation.jl")
include("types.jl")
include("utils.jl")        # Must come before grid/ccd
include("grid.jl")         # Grid-based exploration
include("ccd.jl")          # CCD exploration

# AutoExplorationStrategy dispatch (needs both grid.jl and ccd.jl loaded)
function explore_hyperparameter_posterior(
        strategy::AutoExplorationStrategy,
        model::INLAModel, y, θ_star::WorkingHyperparameters,
        marginalization_method, marginalization_indices;
        kwargs...
    )
    actual_strategy = length(θ_star) > 2 ? strategy.ccd : strategy.grid
    return explore_hyperparameter_posterior(
        actual_strategy, model, y, θ_star, marginalization_method, marginalization_indices;
        kwargs...
    )
end
