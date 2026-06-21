export GridSumMarginal, CCDInterpolantMarginal, AutoHyperparameterMarginal

"""
    GridSumMarginal <: HyperparameterMarginalizationMethod

Rectangle rule on grid points -> 1D cubic spline marginals (the grid-sum approach).

Extracts (θ, log_density) pairs from the exploration grid for each dimension,
normalizes, and fits cubic splines via `SplineMarginalDistribution`.

Best for D=1 where the grid is inherently 1D. For D>=2 with a rotated grid
(standard reparameterization), this gives z-marginals not θ-marginals — use
`CCDInterpolantMarginal` instead.
"""
struct GridSumMarginal <: HyperparameterMarginalizationMethod end

"""
    CCDInterpolantMarginal <: HyperparameterMarginalizationMethod

CCD interpolant (skewness-corrected Gaussian in z-space) + profiling -> 1D
cubic spline marginals.

Builds a lightweight parametric interpolant from the exploration data plus
1+2d extra `hyperparameter_logpdf` evaluations for skewness corrections.
Then profiles along each θ dimension using inverse-Hessian conditional modes
(no further logpdf evaluations needed). Each profile is fit with a cubic spline.

Works for any D, and is the default approach for D>=2.

# Fields
- `n_grid::Int`: Number of profile grid points per dimension (default 200)
"""
struct CCDInterpolantMarginal <: HyperparameterMarginalizationMethod
    n_grid::Int
end
CCDInterpolantMarginal(; n_grid::Int = 200) = CCDInterpolantMarginal(n_grid)

"""
    AutoHyperparameterMarginal <: HyperparameterMarginalizationMethod

Auto-selects the marginalization strategy based on dimensionality:
- D=1: `GridSumMarginal` (rectangle rule, exact for 1D grids)
- D>=2: `CCDInterpolantMarginal` (CCD interpolant + profiling)

This is the default in `inla()`.

# Fields
- `n_grid::Int`: Profile grid resolution passed to CCDInterpolantMarginal (default 200)
"""
struct AutoHyperparameterMarginal <: HyperparameterMarginalizationMethod
    n_grid::Int
end
AutoHyperparameterMarginal(; n_grid::Int = 200) = AutoHyperparameterMarginal(n_grid)
