using Test
using IntegratedNestedLaplace
using ForwardDiff
using LinearAlgebra
using Distributions

@testset "Custom Observation Models and AD Fallbacks" begin
    
    @testset "Custom Model with only logpdf" begin
        # Define a custom observation model - negative binomial
        struct NegativeBinomialModel <: ObservationModel
            r::Float64  # Number of failures parameter
        end
        
        function IntegratedNestedLaplace.loglik(model::NegativeBinomialModel, x, θ, y)
            μ = exp.(x)  # Mean of negative binomial
            r = model.r
            # Negative binomial parameterization: p = r/(r+μ)
            p = r ./ (r .+ μ)
            return sum(logpdf.(NegativeBinomial.(r, p), y))
        end
        
        # Test the model
        model = NegativeBinomialModel(5.0)
        x = [1.0, 0.5, 2.0]
        θ = Float64[]
        y = [3, 1, 8]
        
        # Test that loglik works
        ll = loglik(model, x, θ, y)
        @test ll isa Float64
        @test isfinite(ll)
        
        # Test that AD fallbacks work for gradient
        grad = loggrad(model, x, θ, y)
        grad_fd = ForwardDiff.gradient(xi -> loglik(model, xi, θ, y), x)
        @test grad ≈ grad_fd rtol=1e-10
        
        # Test that AD fallbacks work for hessian
        hess = loghessian(model, x, θ, y)
        hess_fd = ForwardDiff.hessian(xi -> loglik(model, xi, θ, y), x)
        @test hess ≈ hess_fd rtol=1e-8
    end
    
    @testset "Custom Model with optimized gradient" begin
        # Custom model that provides its own gradient
        struct CustomNormalModel <: ObservationModel
            σ²::Float64
        end
        
        function IntegratedNestedLaplace.loglik(model::CustomNormalModel, x, θ, y)
            return -0.5 * sum((y .- x).^2) / model.σ² - 0.5 * length(y) * log(2π * model.σ²)
        end
        
        function IntegratedNestedLaplace.loggrad(model::CustomNormalModel, x, θ, y)
            return (y .- x) ./ model.σ²
        end
        
        model = CustomNormalModel(0.25)
        x = [0.5, 1.0, -0.5]
        θ = Float64[]
        y = [0.6, 1.1, -0.4]
        
        # Test that custom gradient is used
        grad_custom = loggrad(model, x, θ, y)
        grad_expected = (y .- x) ./ model.σ²
        @test grad_custom ≈ grad_expected
        
        # Test that it matches ForwardDiff
        grad_fd = ForwardDiff.gradient(xi -> loglik(model, xi, θ, y), x)
        @test grad_custom ≈ grad_fd rtol=1e-10
        
        # Test that hessian still uses AD fallback
        hess = loghessian(model, x, θ, y)
        hess_fd = ForwardDiff.hessian(xi -> loglik(model, xi, θ, y), x)
        @test hess ≈ hess_fd rtol=1e-10
    end
    
    @testset "Custom Model with both gradient and hessian" begin
        # Simple quadratic model for testing
        struct QuadraticModel <: ObservationModel end
        
        function IntegratedNestedLaplace.loglik(model::QuadraticModel, x, θ, y)
            return -0.5 * sum((x .- y).^2)
        end
        
        function IntegratedNestedLaplace.loggrad(model::QuadraticModel, x, θ, y)
            return -(x .- y)
        end
        
        function IntegratedNestedLaplace.loghessian(model::QuadraticModel, x, θ, y)
            return -I(length(x))
        end
        
        model = QuadraticModel()
        x = [1.0, 2.0, 3.0]
        θ = Float64[]
        y = [1.1, 1.9, 3.2]
        
        # Test custom implementations
        @test loggrad(model, x, θ, y) ≈ -(x .- y)
        @test loghessian(model, x, θ, y) ≈ -I(3)
        
        # Verify against ForwardDiff
        grad_fd = ForwardDiff.gradient(xi -> loglik(model, xi, θ, y), x)
        @test loggrad(model, x, θ, y) ≈ grad_fd
        
        hess_fd = ForwardDiff.hessian(xi -> loglik(model, xi, θ, y), x)
        @test Matrix(loghessian(model, x, θ, y)) ≈ hess_fd
    end
    
    @testset "Sparse Hessian Detection" begin
        # Model that should produce sparse hessian
        struct SparseModel <: ObservationModel end
        
        function IntegratedNestedLaplace.loglik(model::SparseModel, x, θ, y)
            # Only neighboring elements interact
            ll = 0.0
            for i in eachindex(y)
                ll += -0.5 * (x[i] - y[i])^2
                if i > 1
                    ll += -0.1 * x[i-1] * x[i]  # Sparse interaction
                end
            end
            return ll
        end
        
        model = SparseModel()
        x = ones(5)
        θ = Float64[]
        y = ones(5) .+ 0.1
        
        # Test that hessian computation works
        hess = loghessian(model, x, θ, y)
        @test hess isa AbstractMatrix
        
        # Should be close to ForwardDiff result
        hess_fd = ForwardDiff.hessian(xi -> loglik(model, xi, θ, y), x)
        @test hess ≈ hess_fd rtol=1e-8
    end
    
    @testset "Error handling" begin
        # Test that missing loglik implementation throws appropriate error
        struct IncompleteModel <: ObservationModel end
        
        model = IncompleteModel()
        x = [1.0]
        θ = Float64[]
        y = [1.0]
        
        @test_throws ErrorException loglik(model, x, θ, y)
    end
end
