using DifferentiationInterface
using ADTypes

export DifferentiationStrategy, FiniteDiffStrategy, ADStrategy

"""
    DifferentiationStrategy

Abstract type for differentiation backends used in hyperparameter optimization.

Controls how gradients (for mode finding) and Hessians (for reparameterization) are computed.

Concrete subtypes:
- `FiniteDiffStrategy`: Pure finite differences (legacy behavior)
- `ADStrategy`: Automatic differentiation via DifferentiationInterface.jl
"""
abstract type DifferentiationStrategy end

"""
    FiniteDiffStrategy()

Use finite differences for all derivatives. This is the legacy behavior:
- Mode finding: Optim's internal finite-difference gradient
- Hessian: Adaptive step size selection with 3pt/5pt stencil comparison
"""
struct FiniteDiffStrategy <: DifferentiationStrategy end

"""
    ADStrategy(backend=AutoForwardDiff())

Use automatic differentiation for gradients, with finite-diff-of-AD-gradient for Hessians.
The backend is swappable via DifferentiationInterface.jl.

- Mode finding: Explicit AD gradient provided to Optim
- Hessian: Central finite differences of exact AD gradient (no adaptive step size needed)

# Examples
```julia
ADStrategy()                    # Default: ForwardDiff
ADStrategy(AutoForwardDiff())   # Explicit ForwardDiff
```
"""
struct ADStrategy{B <: ADTypes.AbstractADType} <: DifferentiationStrategy
    backend::B
end

ADStrategy() = ADStrategy(AutoForwardDiff())

"""
    ad_negative_hessian(f, x, backend; executor=SequentialExecutor())

Compute the negative Hessian of `f` at `x` using central finite differences of the AD
gradient. Each gradient evaluation is independent and can be parallelized via `executor`.

Since the AD gradient is exact, no adaptive step size search is needed — a standard
`cbrt(eps)` step size works well.

# Arguments
- `f`: Scalar-valued function `f(x) → Real`
- `x`: Evaluation point (AbstractVector)
- `backend`: AD backend (e.g., `AutoForwardDiff()`)
- `executor`: Parallel executor for gradient evaluations (default: `SequentialExecutor()`)

# Returns
- `Symmetric` negative Hessian matrix
"""
function ad_negative_hessian(
        f, x::AbstractVector{T}, backend;
        executor::ParallelExecutor = SequentialExecutor()
    ) where {T}
    d = length(x)
    h = cbrt(eps(T))

    # Build perturbation points: x ± h*eᵢ for central differences
    eval_points = Vector{Vector{T}}(undef, 2d)
    for i in 1:d
        xp = copy(x); xp[i] += h
        xm = copy(x); xm[i] -= h
        eval_points[2i - 1] = xp
        eval_points[2i] = xm
    end

    # Evaluate AD gradients at all perturbation points (PARALLEL)
    grads = pmap_executor(eval_points, executor) do xi
        DifferentiationInterface.gradient(f, backend, xi)
    end

    # Assemble Hessian via central differences of gradient
    H = Matrix{T}(undef, d, d)
    for i in 1:d
        @. H[:, i] = (grads[2i - 1] - grads[2i]) / (2h)
    end

    return Matrix(Symmetric(-(H + H') / 2))
end

"""
    _compute_negative_hessian(strategy, f, x; executor)

Dispatch Hessian computation based on differentiation strategy.
"""
function _compute_negative_hessian(::FiniteDiffStrategy, f, x; executor = SequentialExecutor())
    return adaptive_negative_hessian(f, x; executor = executor)
end

function _compute_negative_hessian(strategy::ADStrategy, f, x; executor = SequentialExecutor())
    return ad_negative_hessian(f, x, strategy.backend; executor = executor)
end
