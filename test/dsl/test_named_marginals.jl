using Test
using Latte
using Distributions
using LinearAlgebra
using Random
using DynamicPPL: @model
using OrderedCollections: OrderedDict

isdefined(@__MODULE__, :shared_hier_poisson) || include("shared_models.jl")

# DPPL-built LGMs populate a latent_layout; result.latent_marginals and
# result.base_latent_marginals should surface that layout, so users can
# index or property-access by DPPL symbol name.
@testset "Named marginals access via symbols" begin

    # Named access is a pure property of the result layout, so one fit serves
    # every testset below.
    result, n, p, G = let
        Random.seed!(42)
        n, p, G = 25, 2, 3
        X = [ones(n) randn(n)]
        group = rand(1:G, n)
        y = [rand(Poisson(exp(X[i, :] ⋅ [0.3, 0.5]))) for i in 1:n]
        lgm = latte_from_dppl(
            shared_hier_poisson(y, X, group, G); random = (:β, :u), augment = true,
        )
        inla(lgm, y; progress = false), n, p, G
    end

    @testset "Property access on augmented latent_marginals" begin
        # .β and .u return sub-vectors of the right length
        β_marg = result.latent_marginals.β
        u_marg = result.latent_marginals.u
        @test length(β_marg) == p
        @test length(u_marg) == G

        # Contents match the layout-indexed slices
        layout = latent_groups(result)
        @test β_marg == result.latent_marginals[layout[:β]]
        @test u_marg == result.latent_marginals[layout[:u]]
    end

    @testset "Symbol indexing and property access are equivalent" begin
        @test result.latent_marginals[:β] == result.latent_marginals.β
        @test result.latent_marginals[:u] == result.latent_marginals.u
    end

    @testset "Property access on base_latent_marginals (augmented LGM)" begin
        base = result.base_latent_marginals
        @test length(base) == p + G
        @test length(base.β) == p
        @test length(base.u) == G
        # The base view starts at index 1 locally — the layout should already
        # be shifted so the user doesn't have to subtract n.
        @test base.β == base[1:p]
        @test base.u == base[(p + 1):(p + G)]
    end

    @testset "Unknown symbol throws informative error" begin
        @test_throws KeyError result.latent_marginals.nope
        @test_throws KeyError result.latent_marginals[:nope]
    end

    @testset "Iteration and full-vector operations still work" begin
        # AbstractVector interface
        @test length(result.latent_marginals) ==
            length(latent_marginals(result))

        # Iteration (mean of each marginal)
        means = [mean(d) for d in result.latent_marginals]
        @test length(means) == length(result.latent_marginals)
    end

    @testset "Hand-built LGMs have no named access (layout empty)" begin
        # Backdoor: construct an INLAResult from a hand-built LGM and check
        # that property access degrades gracefully.
        using GaussianMarkovRandomFields: IIDModel, ExponentialFamily
        spec = @hyperparams begin
            (τ ~ Gamma(2, 1), transform = log, space = natural)
        end
        latent = IIDModel(5)
        obs_model = ExponentialFamily(Poisson)
        lgm = LatentGaussianModel(spec, latent, obs_model)
        Random.seed!(7)
        y = rand(Poisson(1.0), 5)
        r_plain = inla(lgm, y; progress = false)
        # No layout — raw Vector{Distribution} comes back, .β errors
        # with whatever type of error a plain Vector raises (ErrorException
        # on Julia 1.10).
        @test_throws Exception r_plain.latent_marginals.β
        # Raw indexing still works.
        @test length(r_plain.latent_marginals) == 5
    end
end
