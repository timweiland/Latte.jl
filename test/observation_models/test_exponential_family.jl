using Test
using IntegratedNestedLaplace
using Distributions
using ForwardDiff
using LinearAlgebra
using StatsFuns

# Helper function to test gradient and hessian against ForwardDiff using new API
function test_against_autodiff(obs_lik, η)
    # Test gradient
    grad = loggrad(obs_lik, η)
    grad_fd = ForwardDiff.gradient(x -> loglik(obs_lik, x), η)
    @test grad ≈ grad_fd rtol = 1.0e-10

    # Test hessian
    hess = loghessian(obs_lik, η)
    hess_fd = ForwardDiff.hessian(x -> loglik(obs_lik, x), η)
    return @test Matrix(hess) ≈ hess_fd rtol = 1.0e-8
end

@testset "ExponentialFamily Models" begin

    @testset "Poisson Family" begin
        # Test with different link functions
        links = [LogLink(), IdentityLink()]

        for link in links
            model = ExponentialFamily{Poisson, typeof(link)}(Poisson, link)

            # Generate appropriate test data for this link
            if link isa LogLink
                η = randn(5)  # Any real values work for log link
            else  # IdentityLink
                η = abs.(randn(5)) .+ 0.1  # Must be positive for Poisson rates
            end

            y = rand(0:10, 5)  # Random count data

            # Create materialized likelihood using new API
            obs_lik = model(y)

            test_against_autodiff(obs_lik, η)
        end
    end

    @testset "Bernoulli Family" begin
        # Test with different link functions
        links = [LogitLink(), LogLink()]

        for link in links
            model = ExponentialFamily{Bernoulli, typeof(link)}(Bernoulli, link)

            # Generate appropriate test data for this link
            if link isa LogitLink
                η = randn(5)  # Any real values work for logit link
            else  # LogLink
                η = log.(rand(5) .* 0.8 .+ 0.1)  # log of probabilities in (0,1)
            end

            y = rand([0, 1], 5)  # Random binary data

            # Create materialized likelihood using new API
            obs_lik = model(y)

            test_against_autodiff(obs_lik, η)
        end
    end

    @testset "Binomial Family" begin
        # Test with different link functions
        links = [LogitLink(), IdentityLink()]
        n = 10  # Fixed number of trials

        for link in links
            model = ExponentialFamily{Binomial, typeof(link)}(Binomial, link)

            # Generate appropriate test data for this link
            if link isa LogitLink
                η = randn(5)  # Any real values work for logit link
            else  # IdentityLink
                η = rand(5) .* 0.8 .+ 0.1  # Probabilities in (0,1)
            end

            y = rand(0:n, 5)  # Random binomial data

            # Create materialized likelihood using new API
            obs_lik = model(y; n = n)

            test_against_autodiff(obs_lik, η)
        end
    end

    @testset "Normal Family" begin
        # Test with different link functions
        links = [IdentityLink(), LogLink()]
        σ = 0.5  # Fixed standard deviation

        for link in links
            model = ExponentialFamily{Normal, typeof(link)}(Normal, link)

            # Generate appropriate test data for this link
            if link isa IdentityLink
                η = randn(5)  # Any real values work for identity link
            else  # LogLink
                η = log.(abs.(randn(5)) .+ 0.1)  # log of positive values
            end

            y = randn(5)  # Random normal data

            # Create materialized likelihood using new API
            obs_lik = model(y; σ = σ)

            test_against_autodiff(obs_lik, η)
        end
    end
end
