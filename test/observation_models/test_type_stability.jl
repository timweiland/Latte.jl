using Test
using IntegratedNestedLaplace

@testset "Type Stability" begin
    
    @testset "ExponentialFamily Type Stability" begin
        # Test that ExponentialFamily models are type stable
        
        # Poisson model
        poisson_model = ExponentialFamily(Poisson)
        x = [1.0, 2.0]
        θ = Float64[]
        y = [1, 3]
        
        @inferred loglik(poisson_model, x, θ, y)
        @inferred loggrad(poisson_model, x, θ, y)
        @inferred loghessian(poisson_model, x, θ, y)
        
        # Bernoulli model
        bernoulli_model = ExponentialFamily(Bernoulli)
        y_bool = [0, 1]
        
        @inferred loglik(bernoulli_model, x, θ, y_bool)
        @inferred loggrad(bernoulli_model, x, θ, y_bool)
        @inferred loghessian(bernoulli_model, x, θ, y_bool)
        
        # Normal model
        normal_model = ExponentialFamily(Normal)
        θ_normal = [1.0]  # sigma
        y_float = [1.1, 2.2]
        
        @inferred loglik(normal_model, x, θ_normal, y_float)
        @inferred loggrad(normal_model, x, θ_normal, y_float)
        @inferred loghessian(normal_model, x, θ_normal, y_float)
    end
    
    @testset "Link Function Type Stability" begin
        # Test link functions
        @inferred apply_link(IdentityLink(), 1.0)
        @inferred apply_invlink(IdentityLink(), 1.0)
        @inferred IntegratedNestedLaplace.derivative_invlink(IdentityLink(), 1.0)
        @inferred IntegratedNestedLaplace.second_derivative_invlink(IdentityLink(), 1.0)
        
        @inferred apply_link(LogLink(), 1.0)
        @inferred apply_invlink(LogLink(), 1.0)
        @inferred IntegratedNestedLaplace.derivative_invlink(LogLink(), 1.0)
        @inferred IntegratedNestedLaplace.second_derivative_invlink(LogLink(), 1.0)
        
        @inferred apply_link(LogitLink(), 0.5)
        @inferred apply_invlink(LogitLink(), 0.0)
        @inferred IntegratedNestedLaplace.derivative_invlink(LogitLink(), 0.0)
        @inferred IntegratedNestedLaplace.second_derivative_invlink(LogitLink(), 0.0)
        
        # Test broadcasting
        x = [1.0, 2.0, 3.0]
        @inferred (() -> apply_link.(Ref(IdentityLink()), x))()
        @inferred (() -> apply_invlink.(Ref(LogLink()), x))()
        @inferred (() -> IntegratedNestedLaplace.derivative_invlink.(Ref(LogitLink()), x))()
    end
    
    @testset "Custom Model Type Stability" begin
        # Simple custom model for testing
        struct SimpleCustomModel <: ObservationModel end
        
        function IntegratedNestedLaplace.loglik(::SimpleCustomModel, x, θ, y)
            return -0.5 * sum((x .- y).^2)
        end
        
        model = SimpleCustomModel()
        x = [1.0, 2.0]
        θ = Float64[]
        y = [1.1, 2.1]
        
        @inferred loglik(model, x, θ, y)
        # Note: AD fallbacks may not be type stable due to ForwardDiff internals,
        # but we can still test that they return correct types
        grad_result = loggrad(model, x, θ, y)
        hess_result = loghessian(model, x, θ, y)
        
        @test grad_result isa Vector{Float64}
        @test hess_result isa AbstractMatrix{Float64}
    end
    
    @testset "Parametric Type Consistency" begin
        # Test that parametric types work as expected
        poisson_log = ExponentialFamily{Poisson, LogLink}(Poisson, LogLink())
        poisson_identity = ExponentialFamily{Poisson, IdentityLink}(Poisson, IdentityLink())
        
        @test typeof(poisson_log) != typeof(poisson_identity)
        @test poisson_log.family == poisson_identity.family
        @test typeof(poisson_log.link) != typeof(poisson_identity.link)
        
        # Both should be type stable but with different signatures
        x = [1.0, 2.0]
        θ = Float64[]
        y = [1, 2]
        
        @inferred loglik(poisson_log, x, θ, y)
        @inferred loglik(poisson_identity, x, θ, y)
    end
end