using DifferentiationInterface
using GaussianMarkovRandomFields
using SparseArrays
using Mooncake
using SelectedInversion
using ChainRulesCore
using LinearAlgebra

function ar_precision(ρ, k)
    return spdiagm(-1 => -ρ * ones(k - 1), 0 => ones(k) .+ ρ^2, 1 => -ρ * ones(k - 1))
end

function ar_deriv(ρ, k)
    return spdiagm(-1 => -ones(k - 1), 0 => ones(k) .* (2 * ρ), 1 => -ones(k - 1))
end

k = 50000
function f(θ)
    ρ = θ
    Q = ar_precision(ρ, k)
    x = GMRF(zeros(k), Q)
    return my_logpdf(x)
end

z = rand(GMRF(zeros(k), ar_precision(0.94, k)))

function my_logpdf(x::GMRF)
    μ = mean(x)
    Q = precision_matrix(x)
    r = z - μ
    return -0.5 * (logdet(x.solver.precision_chol) + dot(r, Q * r))
end

function ChainRulesCore.rrule(::typeof(my_logpdf), x::GMRF)
    μ = mean(x)
    Q = precision_matrix(x)
    r = z - μ
    chol = x.solver.precision_chol
    val = -0.5 * (logdet(chol) + dot(r, Q * r))

    function my_logpdf_pullback(ȳ)
        # Forward: logpdf = -0.5 * (logdet(Q) + rᵀ Q r)
        # So ∂/∂μ = Qr
        # And ∂/∂Q = -0.5 * (I + rrᵀ)

        @time Qinv = selinv(chol; depermute = true).Z
        Qr = Q * r

        # Derivatives
        μ̄ = ȳ * Qr                    # ∂logpdf/∂μ = Qr → chain rule

        rows, cols, vals = findnz(Qinv)  # nonzero structure of Qinv
        rr_vals = r[rows] .* r[cols]
        Q̄ = sparse(rows, cols, (-0.5 * ȳ) .* (vals .+ rr_vals), size(Q)...)

        # Tangent for x
        x̄ = Tangent{typeof(x)}(;
            solver = Tangent{typeof(x.solver)}(; precision_chol = NoTangent()),  # fixed
            mean = -μ̄,                     # ∂L/∂μ = Qr → ∂L/∂mean = -Qr
            precision = Q̄          # ∂L/∂Q
        )

        return NoTangent(), x̄
    end

    return val, my_logpdf_pullback
end

function ChainRulesCore.rrule(::Type{GMRF}, μ::AbstractVector, Q::SparseMatrixCSC)
    x = GMRF(μ, Q)  # constructs the GMRF, including solver

    function GMRF_pullback(Δx̄)
        # Here, Δx̄ is a Tangent{GMRF} — unpack what you want
        μ̄ = Δx̄.mean
        Q̄ = Δx̄.precision

        return NoTangent(), μ̄, Q̄
    end

    return x, GMRF_pullback
end

cache = Mooncake.prepare_pullback_cache(my_logdet, A)
