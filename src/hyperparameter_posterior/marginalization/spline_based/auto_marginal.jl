"""
AutoHyperparameterMarginal: dispatches based on dimension.

- D=1: GridSumMarginal (rectangle rule on 1D exploration grid)
- D≥2: CCDInterpolantMarginal (CCD interpolant + profiling)
"""
function _marginalize_impl(
        method::AutoHyperparameterMarginal,
        exploration::AbstractHyperparameterExploration,
        model::INLAModel,
        y,
        progress_callback
    )
    n_dim = length(exploration.transform.θ_star)
    if n_dim == 1
        return _marginalize_impl(GridSumMarginal(), exploration, model, y, progress_callback)
    else
        return _marginalize_impl(
            CCDInterpolantMarginal(method.n_grid), exploration, model, y, progress_callback
        )
    end
end
