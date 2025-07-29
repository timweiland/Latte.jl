using Test
using IntegratedNestedLaplace
using ForwardDiff
using LinearAlgebra
using Distributions

@testset "Custom Observation Likelihoods and AD Fallbacks" begin

    @testset "Custom Likelihood with only loglik" begin
        # Define a custom observation likelihood - negative binomial
        struct NegativeBinomialLikelihood <: ObservationLikelihood
            r::Float64  # Number of failures parameter
            y::Vector{Int}  # Observed data
        end

        function IntegratedNestedLaplace.loglik(lik::NegativeBinomialLikelihood, x)
            μ = exp.(x)  # Mean of negative binomial
            r = lik.r
            y = lik.y
            # Negative binomial parameterization: p = r/(r+μ)
            p = r ./ (r .+ μ)
            return sum(logpdf.(NegativeBinomial.(r, p), y))
        end

        # Test the likelihood
        y = [3, 1, 8]
        lik = NegativeBinomialLikelihood(5.0, y)
        x = [1.0, 0.5, 2.0]

        # Test that loglik works
        ll = loglik(lik, x)
        @test ll isa Float64
        @test isfinite(ll)

        # Test that AD fallbacks work for gradient
        grad = loggrad(lik, x)
        grad_fd = ForwardDiff.gradient(xi -> loglik(lik, xi), x)
        @test grad ≈ grad_fd rtol = 1.0e-10

        # Test that AD fallbacks work for hessian
        hess = loghessian(lik, x)
        hess_fd = ForwardDiff.hessian(xi -> loglik(lik, xi), x)
        @test hess ≈ hess_fd rtol = 1.0e-8
    end

    @testset "Custom Likelihood with optimized gradient" begin
        # Custom likelihood that provides its own gradient
        struct CustomNormalLikelihood <: ObservationLikelihood
            σ²::Float64
            y::Vector{Float64}
        end

        function IntegratedNestedLaplace.loglik(lik::CustomNormalLikelihood, x)
            return -0.5 * sum((lik.y .- x) .^ 2) / lik.σ² - 0.5 * length(lik.y) * log(2π * lik.σ²)
        end

        function IntegratedNestedLaplace.loggrad(lik::CustomNormalLikelihood, x)
            return (lik.y .- x) ./ lik.σ²
        end

        y = [0.6, 1.1, -0.4]
        lik = CustomNormalLikelihood(0.25, y)
        x = [0.5, 1.0, -0.5]

        # Test that custom gradient is used
        grad_custom = loggrad(lik, x)
        grad_expected = (y .- x) ./ lik.σ²
        @test grad_custom ≈ grad_expected

        # Test that it matches ForwardDiff
        grad_fd = ForwardDiff.gradient(xi -> loglik(lik, xi), x)
        @test grad_custom ≈ grad_fd rtol = 1.0e-10

        # Test that hessian still uses AD fallback
        hess = loghessian(lik, x)
        hess_fd = ForwardDiff.hessian(xi -> loglik(lik, xi), x)
        @test hess ≈ hess_fd rtol = 1.0e-10
    end

    @testset "Custom Likelihood with both gradient and hessian" begin
        # Simple quadratic likelihood for testing
        struct QuadraticLikelihood <: ObservationLikelihood
            y::Vector{Float64}
        end

        function IntegratedNestedLaplace.loglik(lik::QuadraticLikelihood, x)
            return -0.5 * sum((x .- lik.y) .^ 2)
        end

        function IntegratedNestedLaplace.loggrad(lik::QuadraticLikelihood, x)
            return -(x .- lik.y)
        end

        function IntegratedNestedLaplace.loghessian(lik::QuadraticLikelihood, x)
            return -I(length(x))
        end

        y = [1.1, 1.9, 3.2]
        lik = QuadraticLikelihood(y)
        x = [1.0, 2.0, 3.0]

        # Test custom implementations
        @test loggrad(lik, x) ≈ -(x .- y)
        @test loghessian(lik, x) ≈ -I(3)

        # Verify against ForwardDiff
        grad_fd = ForwardDiff.gradient(xi -> loglik(lik, xi), x)
        @test loggrad(lik, x) ≈ grad_fd

        hess_fd = ForwardDiff.hessian(xi -> loglik(lik, xi), x)
        @test Matrix(loghessian(lik, x)) ≈ hess_fd
    end

    @testset "Sparse Hessian Detection" begin
        # Likelihood that should produce sparse hessian
        struct SparseLikelihood <: ObservationLikelihood
            y::Vector{Float64}
        end

        function IntegratedNestedLaplace.loglik(lik::SparseLikelihood, x)
            # Only neighboring elements interact
            ll = 0.0
            for i in eachindex(lik.y)
                ll += -0.5 * (x[i] - lik.y[i])^2
                if i > 1
                    ll += -0.1 * x[i - 1] * x[i]  # Sparse interaction
                end
            end
            return ll
        end

        y = ones(5) .+ 0.1
        lik = SparseLikelihood(y)
        x = ones(5)

        # Test that hessian computation works
        hess = loghessian(lik, x)
        @test hess isa AbstractMatrix

        # Should be close to ForwardDiff result
        hess_fd = ForwardDiff.hessian(xi -> loglik(lik, xi), x)
        @test hess ≈ hess_fd rtol = 1.0e-8
    end

    @testset "Error handling" begin
        # Test that missing loglik implementation throws appropriate error
        struct IncompleteLikelihood <: ObservationLikelihood
            y::Vector{Float64}
        end

        y = [1.0]
        lik = IncompleteLikelihood(y)
        x = [1.0]

        @test_throws MethodError loglik(lik, x)
    end
end
