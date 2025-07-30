using Test
using IntegratedNestedLaplace
using Distributions

@testset "Type Stability" begin

    @testset "ExponentialFamily Type Stability" begin
        # Test that ObservationLikelihood models are type stable using new API

        # Poisson model
        poisson_model = ExponentialFamily(Poisson)
        x = [1.0, 2.0]
        y = [1, 3]
        poisson_lik = poisson_model(y)

        @inferred loglik(poisson_lik, x)
        @inferred loggrad(poisson_lik, x)
        @inferred loghessian(poisson_lik, x)

        # Bernoulli model
        bernoulli_model = ExponentialFamily(Bernoulli)
        y_bool = [0, 1]
        bernoulli_lik = bernoulli_model(y_bool)

        @inferred loglik(bernoulli_lik, x)
        @inferred loggrad(bernoulli_lik, x)
        @inferred loghessian(bernoulli_lik, x)

        # Normal model
        normal_model = ExponentialFamily(Normal)
        y_float = [1.1, 2.2]
        normal_lik = normal_model(y_float; σ = 1.0)

        @inferred loglik(normal_lik, x)
        @inferred loggrad(normal_lik, x)
        @inferred loghessian(normal_lik, x)
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

    @testset "Custom ObservationLikelihood Type Stability" begin
        # Simple custom likelihood for testing
        struct SimpleCustomLikelihood <: ObservationLikelihood
            y::Vector{Float64}
        end

        function IntegratedNestedLaplace.loglik(lik::SimpleCustomLikelihood, x)
            return -0.5 * sum((x .- lik.y) .^ 2)
        end

        y = [1.1, 2.1]
        lik = SimpleCustomLikelihood(y)
        x = [1.0, 2.0]

        @inferred loglik(lik, x)
        # Note: AD fallbacks may not be type stable due to ForwardDiff internals,
        # but we can still test that they return correct types
        grad_result = loggrad(lik, x)
        hess_result = loghessian(lik, x)

        @test grad_result isa Vector{Float64}
        @test hess_result isa AbstractMatrix{Float64}
    end

    @testset "Parametric Type Consistency" begin
        # Test that parametric types work as expected
        poisson_log = ExponentialFamily(Poisson, LogLink())
        poisson_identity = ExponentialFamily(Poisson, IdentityLink())

        @test typeof(poisson_log) != typeof(poisson_identity)
        @test poisson_log.family == poisson_identity.family
        @test typeof(poisson_log.link) != typeof(poisson_identity.link)

        # Both should be type stable with new API
        x = [1.0, 2.0]
        y = [1, 2]

        poisson_log_lik = poisson_log(y)
        poisson_identity_lik = poisson_identity(y)

        @inferred loglik(poisson_log_lik, x)
        @inferred loglik(poisson_identity_lik, x)
    end
end
