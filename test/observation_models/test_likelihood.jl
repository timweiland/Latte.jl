using Test
using IntegratedNestedLaplace
using Distributions

@testset "Likelihood Function" begin
    
    @testset "Returns Distribution Objects" begin
        # Test Poisson
        poisson_model = ExponentialFamily(Poisson)
        η = [1.0, 2.0]
        θ = Float64[]
        
        dist = likelihood(poisson_model, η, θ)
        @test dist isa Distribution
        @test all(d -> d isa Poisson, dist.v)  # Product distribution components
        
        # Test Normal
        normal_model = ExponentialFamily(Normal)
        η_norm = [0.0, 1.0]
        θ_norm = [0.5]
        
        dist_norm = likelihood(normal_model, η_norm, θ_norm)
        @test dist_norm isa Distribution
        
        # Test Bernoulli
        bernoulli_model = ExponentialFamily(Bernoulli)
        η_bern = [0.0, 1.0]
        θ_bern = Float64[]
        
        dist_bern = likelihood(bernoulli_model, η_bern, θ_bern)
        @test dist_bern isa Distribution
        @test all(d -> d isa Bernoulli, dist_bern.v)
    end
    
    @testset "Consistency with loglik" begin
        # Test that loglik(model, x, θ, y) ≈ logpdf(likelihood(model, x, θ), y)
        
        models_and_data = [
            (ExponentialFamily(Poisson), [1.0, 2.0], Float64[], [1, 3]),
            (ExponentialFamily(Normal), [0.0, 1.0], [0.5], [0.1, 1.2]),
            (ExponentialFamily(Bernoulli), [0.0, 1.0], Float64[], [0, 1]),
            (ExponentialFamily(Binomial), [0.0, 0.5], [10.0], [3, 8]),
        ]
        
        for (model, η, θ, y) in models_and_data
            dist = likelihood(model, η, θ)
            ll_via_likelihood = logpdf(dist, y)
            ll_direct = loglik(model, η, θ, y)
            
            @test ll_via_likelihood ≈ ll_direct rtol=1e-12
        end
    end
end