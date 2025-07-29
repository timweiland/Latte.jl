using Test
using IntegratedNestedLaplace
using Distributions

@testset "Materialized Likelihood Function" begin

    @testset "Factory Pattern and Materialization" begin
        # Test that obs_model(y; params...) creates correct likelihood objects

        # Test Poisson
        poisson_model = ExponentialFamily(Poisson)
        y_poisson = [1, 3]
        poisson_lik = poisson_model(y_poisson)
        @test poisson_lik isa PoissonLikelihood{LogLink}
        @test poisson_lik.y == y_poisson

        # Test Normal
        normal_model = ExponentialFamily(Normal)
        y_normal = [0.1, 1.2]
        normal_lik = normal_model(y_normal; σ = 0.5)
        @test normal_lik isa NormalLikelihood{IdentityLink}
        @test normal_lik.y == y_normal
        @test normal_lik.σ == 0.5

        # Test Bernoulli
        bernoulli_model = ExponentialFamily(Bernoulli)
        y_bernoulli = [0, 1]
        bernoulli_lik = bernoulli_model(y_bernoulli)
        @test bernoulli_lik isa BernoulliLikelihood{LogitLink}
        @test bernoulli_lik.y == y_bernoulli

        # Test Binomial
        binomial_model = ExponentialFamily(Binomial)
        y_binomial = [3, 8]
        binomial_lik = binomial_model(y_binomial; n = 10)
        @test binomial_lik isa BinomialLikelihood{LogitLink}
        @test binomial_lik.y == y_binomial
        @test binomial_lik.n == 10
    end

    @testset "Likelihood Evaluation with Materialized Objects" begin
        # Test that materialized likelihoods work correctly

        test_cases = [
            (ExponentialFamily(Poisson), [1, 3], NamedTuple(), [1.0, 2.0]),
            (ExponentialFamily(Normal), [0.1, 1.2], (σ = 0.5,), [0.0, 1.0]),
            (ExponentialFamily(Bernoulli), [0, 1], NamedTuple(), [0.0, 1.0]),
            (ExponentialFamily(Binomial), [3, 8], (n = 10,), [0.0, 0.5]),
        ]

        for (model, y, θ_named, x) in test_cases
            # Create materialized likelihood
            obs_lik = model(y; θ_named...)

            # Test that loglik works
            ll = loglik(obs_lik, x)
            @test ll isa Float64
            @test isfinite(ll)

            # Test that gradients work
            grad = loggrad(obs_lik, x)
            @test grad isa Vector{Float64}
            @test length(grad) == length(x)

            # Test that hessians work
            hess = loghessian(obs_lik, x)
            @test hess isa AbstractMatrix{Float64}
            @test size(hess) == (length(x), length(x))
        end
    end
end
