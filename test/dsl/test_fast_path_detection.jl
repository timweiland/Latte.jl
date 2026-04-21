using Test
using Latte
using DynamicPPL: @model
using Distributions
using GaussianMarkovRandomFields
using LinearAlgebra
using Random

# Does `latte_from_dppl` correctly detect the fast path for supported
# families, and fall through to the AD wrapping otherwise?
#
# Each model needs ≥ 1 hyperparameter (Latte requires it). We include a
# τ_u on a random-effect covariance to give the adapter something to pin.
@testset "Fast-path detection" begin

    @testset "Poisson + LogLink fires fast path" begin
        @model function m_poisson(y, X, group)
            τ_u ~ Gamma(2, 1)
            β ~ MvNormal(zeros(size(X, 2)), 100.0 * I(size(X, 2)))
            u ~ MvNormal(zeros(maximum(group)), (1 / τ_u) * I(maximum(group)))
            for i in eachindex(y)
                y[i] ~ Poisson(exp(X[i, :] ⋅ β + u[group[i]]); check_args = false)
            end
        end
        Random.seed!(1)
        n, p, G = 30, 2, 3
        X = [ones(n) randn(n)]
        group = rand(1:G, n)
        β_true = [0.3, 0.5]
        u_true = randn(G) ./ 2
        y_obs = [rand(Poisson(exp(X[i, :] ⋅ β_true + u_true[group[i]]))) for i in 1:n]

        lgm = latte_from_dppl(m_poisson(y_obs, X, group); random = (:β, :u))
        @test lgm.observation_model isa ExponentialFamily{Poisson, LogLink}
        @test lgm.augmentation_info !== nothing
    end

    @testset "Bernoulli + LogitLink fires fast path" begin
        @model function m_bernoulli(y, X, group)
            τ_u ~ Gamma(2, 1)
            β ~ MvNormal(zeros(size(X, 2)), 100.0 * I(size(X, 2)))
            u ~ MvNormal(zeros(maximum(group)), (1 / τ_u) * I(maximum(group)))
            for i in eachindex(y)
                p_i = 1 / (1 + exp(-(X[i, :] ⋅ β + u[group[i]])))
                y[i] ~ Bernoulli(p_i; check_args = false)
            end
        end
        Random.seed!(2)
        n, p, G = 30, 2, 3
        X = [ones(n) randn(n)]
        group = rand(1:G, n)
        β_true = [0.1, 0.4]
        u_true = randn(G) ./ 2
        y_obs = [
            rand(Bernoulli(1 / (1 + exp(-(X[i, :] ⋅ β_true + u_true[group[i]])))))
                for i in 1:n
        ]

        lgm = latte_from_dppl(m_bernoulli(y_obs, X, group); random = (:β, :u))
        @test lgm.observation_model isa ExponentialFamily{Bernoulli, LogitLink}
    end

    @testset "force_ad_obs_model=true takes the AD path" begin
        @model function m_poisson(y, X, group)
            τ_u ~ Gamma(2, 1)
            β ~ MvNormal(zeros(size(X, 2)), 100.0 * I(size(X, 2)))
            u ~ MvNormal(zeros(maximum(group)), (1 / τ_u) * I(maximum(group)))
            for i in eachindex(y)
                y[i] ~ Poisson(exp(X[i, :] ⋅ β + u[group[i]]); check_args = false)
            end
        end
        Random.seed!(3)
        n, G = 20, 3
        X = [ones(n) randn(n)]
        group = rand(1:G, n)
        y_obs = [rand(Poisson(exp(X[i, :] ⋅ [0.3, 0.5]))) for i in 1:n]

        lgm = latte_from_dppl(
            m_poisson(y_obs, X, group);
            random = (:β, :u), force_ad_obs_model = true,
        )
        @test !(lgm.observation_model isa ExponentialFamily)
    end

    @testset "Mixed-family likelihood falls through to AD path" begin
        # Half the sites Poisson, half Normal — fast path demands a
        # homogeneous family and must punt.
        @model function m_mixed(y, z, X, group)
            τ_u ~ Gamma(2, 1)
            β ~ MvNormal(zeros(size(X, 2)), 100.0 * I(size(X, 2)))
            u ~ MvNormal(zeros(maximum(group)), (1 / τ_u) * I(maximum(group)))
            for i in eachindex(y)
                y[i] ~ Poisson(exp(X[i, :] ⋅ β + u[group[i]]); check_args = false)
            end
            for i in eachindex(z)
                z[i] ~ Normal(X[i, :] ⋅ β + u[group[i]], 1.0)
            end
        end
        Random.seed!(4)
        n, G = 10, 3
        X = [ones(n) randn(n)]
        group = rand(1:G, n)
        y_obs = rand(Poisson(2.0), n)
        z_obs = randn(n)

        lgm = latte_from_dppl(m_mixed(y_obs, z_obs, X, group); random = (:β, :u))
        @test !(lgm.observation_model isa ExponentialFamily)
    end
end
