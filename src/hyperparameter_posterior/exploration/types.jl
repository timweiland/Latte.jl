using Printf

export GridPoint, AbstractHyperparameterExploration, GridExploration, CCDExploration
export ExplorationStrategy, GridExplorationStrategy, CCDExplorationStrategy, AutoExplorationStrategy

# Backward-compat alias: `isa HyperparameterExploration` still works
export HyperparameterExploration

"""
    ExplorationStrategy

Abstract type for hyperparameter exploration strategies.

Concrete subtypes:
- `GridExplorationStrategy`: Cartesian grid exploration
- `CCDExplorationStrategy`: Central Composite Design exploration
- `AutoExplorationStrategy`: Automatic selection (grid for D ≤ 2, CCD for D ≥ 3)
"""
abstract type ExplorationStrategy end

"""
    GridExplorationStrategy(; integration_step_z=0.75, max_log_drop=6.0, interpolation_subdivisions=1)

Cartesian grid exploration strategy. Explores the hyperparameter posterior on a
regular grid in standardized z-space.

# Fields
- `integration_step_z::Float64`: Step size in z-space (1.0 = one standard deviation)
- `max_log_drop::Float64`: Stop exploring when log-density drops this much from mode
- `interpolation_subdivisions::Int`: Fine-grid steps per coarse integration step
"""
struct GridExplorationStrategy <: ExplorationStrategy
    integration_step_z::Float64
    max_log_drop::Float64
    interpolation_subdivisions::Int
end

function GridExplorationStrategy(;
        integration_step_z::Float64 = 0.75,
        max_log_drop::Float64 = 6.0,
        interpolation_subdivisions::Int = 1
    )
    return GridExplorationStrategy(integration_step_z, max_log_drop, interpolation_subdivisions)
end

"""
    CCDExplorationStrategy(; f0=1.1)

Central Composite Design exploration strategy (Rue et al. 2009, Section 6.5).
Uses O(2d² + 1) design points instead of a full Cartesian grid.

# Fields
- `f0::Float64`: Scaling factor. All non-center design points lie on a sphere of radius `f0 * √d`.
"""
struct CCDExplorationStrategy <: ExplorationStrategy
    f0::Float64
end

CCDExplorationStrategy(; f0::Float64 = 1.1) = CCDExplorationStrategy(f0)

"""
    AutoExplorationStrategy(; grid=GridExplorationStrategy(), ccd=CCDExplorationStrategy())

Automatic exploration strategy selection: grid for D ≤ 2, CCD for D ≥ 3.

# Fields
- `grid::GridExplorationStrategy`: Strategy used for D ≤ 2
- `ccd::CCDExplorationStrategy`: Strategy used for D ≥ 3
"""
struct AutoExplorationStrategy <: ExplorationStrategy
    grid::GridExplorationStrategy
    ccd::CCDExplorationStrategy
end

function AutoExplorationStrategy(;
        grid::GridExplorationStrategy = GridExplorationStrategy(),
        ccd::CCDExplorationStrategy = CCDExplorationStrategy()
    )
    return AutoExplorationStrategy(grid, ccd)
end

"""
Represents a single point on the exploration grid with its computed results.

# Type Parameters
- `W <: WorkingHyperparameters`: Type of working hyperparameters

# Fields
- `θ::W`: Hyperparameters in working space
- `log_density::Float64`: Log density at this point
- `marginal_result::Union{Nothing, MarginalResult}`: Optional marginal results
"""
struct GridPoint{W <: WorkingHyperparameters}
    θ::W
    log_density::Float64
    marginal_result::Union{Nothing, MarginalResult}
end

"""
    AbstractHyperparameterExploration{GP}

Abstract base type for hyperparameter exploration results.

Concrete subtypes:
- `GridExploration`: Cartesian grid exploration
- `CCDExploration`: Central Composite Design exploration (carries CCD-specific data)

All subtypes share the same common fields:
- `grid_points`, `integration_indices`, `transform`,
  `log_normalization_constant`, `integration_bounds`, `accumulator_reorder`
"""
abstract type AbstractHyperparameterExploration{GP <: GridPoint} end

const HyperparameterExploration = AbstractHyperparameterExploration

"""
    GridExploration{GP}

Grid-based hyperparameter exploration (Cartesian product of 1D grids).
"""
struct GridExploration{GP <: GridPoint} <: AbstractHyperparameterExploration{GP}
    grid_points::Vector{GP}
    integration_indices::Vector{Int}
    transform::ReparameterizationTransform
    log_normalization_constant::Float64
    integration_bounds::Matrix{Float64}
    accumulator_reorder::Vector{Int}

    function GridExploration(
            grid_points::Vector{GP}, integration_indices, transform, log_normalization_constant;
            accumulator_reorder::Vector{Int} = collect(1:length(integration_indices))
        ) where {GP <: GridPoint}
        bounds = _compute_integration_bounds(grid_points, integration_indices)
        return new{GP}(grid_points, integration_indices, transform, log_normalization_constant, bounds, accumulator_reorder)
    end
end

"""
    CCDExploration{GP}

Central Composite Design exploration. Carries CCD-specific raw log-density data
at the mode and axial points, enabling the CCD interpolant to be built without
any additional `hyperparameter_logpdf` evaluations.

# Extra fields (beyond common fields)
- `mode_raw_logp`: Raw (unweighted) log-density at the mode
- `axial_raw_logp_plus`: Raw log-density at +f₀√d along each axis
- `axial_raw_logp_minus`: Raw log-density at -f₀√d along each axis
- `f0`: CCD scaling factor
"""
struct CCDExploration{GP <: GridPoint} <: AbstractHyperparameterExploration{GP}
    grid_points::Vector{GP}
    integration_indices::Vector{Int}
    transform::ReparameterizationTransform
    log_normalization_constant::Float64
    integration_bounds::Matrix{Float64}
    accumulator_reorder::Vector{Int}
    mode_raw_logp::Float64
    axial_raw_logp_plus::Vector{Float64}
    axial_raw_logp_minus::Vector{Float64}
    f0::Float64

    function CCDExploration(
            grid_points::Vector{GP}, integration_indices, transform, log_normalization_constant,
            mode_raw_logp, axial_raw_logp_plus, axial_raw_logp_minus, f0;
            accumulator_reorder::Vector{Int} = collect(1:length(integration_indices))
        ) where {GP <: GridPoint}
        bounds = _compute_integration_bounds(grid_points, integration_indices)
        return new{GP}(
            grid_points, integration_indices, transform, log_normalization_constant,
            bounds, accumulator_reorder, mode_raw_logp, axial_raw_logp_plus, axial_raw_logp_minus, f0
        )
    end
end

"""Compute integration bounds from grid points (shared by both exploration types)."""
function _compute_integration_bounds(grid_points, integration_indices)
    integration_points = grid_points[integration_indices]

    if isempty(integration_points)
        error("No integration points found in exploration")
    end

    θ_values = [point.θ for point in integration_points]
    n_dims = length(θ_values[1])

    bounds = Matrix{Float64}(undef, n_dims, 2)
    for dim in 1:n_dims
        dim_values = [θ[dim] for θ in θ_values]
        bounds[dim, 1] = minimum(dim_values)
        bounds[dim, 2] = maximum(dim_values)
    end
    return bounds
end

# Custom show methods

function Base.show(io::IO, gp::GridPoint)
    print(io, "GridPoint(θ=")
    θ_vec = gp.θ.θ
    if length(θ_vec) <= 3
        print(io, "[", join([@sprintf("%.4f", x) for x in θ_vec], ", "), "]")
    else
        print(io, "[", @sprintf("%.4f", θ_vec[1]), ", ", @sprintf("%.4f", θ_vec[2]), ", ..., ", @sprintf("%.4f", θ_vec[end]), "]")
    end
    print(io, ", log_density=", @sprintf("%.4f", gp.log_density))
    if gp.marginal_result !== nothing
        print(io, ", marginals=", length(gp.marginal_result.marginals), " variables")
    else
        print(io, ", marginals=none")
    end
    return print(io, ")")
end

function Base.show(io::IO, exploration::AbstractHyperparameterExploration)
    typename = nameof(typeof(exploration))
    println(io, typename, ":")
    println(io, "  Grid points: ", length(exploration.grid_points))
    println(io, "  Integration points: ", length(exploration.integration_indices))

    if !isempty(exploration.grid_points)
        n_dim = length(exploration.grid_points[1].θ)
        println(io, "  Parameter dimensions: ", n_dim)

        if !isempty(exploration.integration_indices)
            integration_points = exploration.grid_points[exploration.integration_indices]
            for dim in 1:n_dim
                dim_values = [point.θ[dim] for point in integration_points]
                min_val = minimum(dim_values)
                max_val = maximum(dim_values)
                println(io, "    Dimension ", dim, ": [", @sprintf("%.4f", min_val), ", ", @sprintf("%.4f", max_val), "]")
            end
        end
    end

    log_densities = [point.log_density for point in exploration.grid_points]
    if !isempty(log_densities)
        println(io, "  Log density range: [", @sprintf("%.4f", minimum(log_densities)), ", ", @sprintf("%.4f", maximum(log_densities)), "]")
    end

    return print(io, "  Log normalization constant: ", @sprintf("%.4f", exploration.log_normalization_constant))
end
