using LinearAlgebra

export NewtonStats, NewtonResult, NewtonOptions

"""
    NewtonStats

Statistics collected during Newton-Raphson optimization.

# Fields
- `iteration::Int`: Current iteration number
- `newton_decrement::Float64`: Newton decrement (λ²/2)
- `step_size::Float64`: Step size taken
- `gradient_norm::Float64`: L2 norm of the gradient
- `converged::Bool`: Whether convergence criteria are met
"""
struct NewtonStats
    iteration::Int
    newton_decrement::Float64
    step_size::Float64
    gradient_norm::Float64
    converged::Bool
end

"""
    NewtonResult

Result of Newton-Raphson optimization.

# Fields
- `μ::Vector{Float64}`: Final mean estimate
- `precision::AbstractMatrix`: Final precision matrix
- `precision_chol`: Cholesky factorization of precision matrix
- `stats::Vector{NewtonStats}`: Statistics from each iteration
- `converged::Bool`: Whether optimization converged
- `iterations::Int`: Number of iterations performed
"""
struct NewtonResult{T<:AbstractMatrix, C}
    μ::Vector{Float64}
    precision::T
    precision_chol::C
    stats::Vector{NewtonStats}
    converged::Bool
    iterations::Int
end

"""
    NewtonOptions

Options for Newton-Raphson optimization.

# Fields
- `max_iterations::Int = 100`: Maximum number of iterations
- `tol_gradient::Float64 = 1e-6`: Convergence tolerance for gradient norm
- `tol_decrement::Float64 = 1e-8`: Convergence tolerance for Newton decrement
- `min_step_size::Float64 = 1e-12`: Minimum allowed step size
- `verbose::Bool = false`: Whether to print progress information
"""
Base.@kwdef struct NewtonOptions
    max_iterations::Int = 100
    tol_gradient::Float64 = 1e-6
    tol_decrement::Float64 = 1e-8
    min_step_size::Float64 = 1e-12
    verbose::Bool = false
end