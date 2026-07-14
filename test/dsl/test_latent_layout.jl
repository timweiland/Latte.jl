using Test
using Latte
using Distributions
using LinearAlgebra
using Random
using DynamicPPL: @model
using GaussianMarkovRandomFields: IIDModel
using OrderedCollections: OrderedDict

isdefined(@__MODULE__, :shared_hier_poisson) || include("shared_models.jl")

# The DPPL adapter should thread each random-effect symbol's position in the
# augmented latent vector onto the LGM so downstream machinery (prediction,
# marginal lookup) can work by name.
@testset "Latent layout from latte_from_dppl" begin

    @testset "Augmented fast-path: sym → augmented-latent range" begin
        Random.seed!(1)
        n, p, G = 30, 2, 3
        X = [ones(n) randn(n)]
        group = rand(1:G, n)
        y = [rand(Poisson(exp(X[i, :] ⋅ [0.3, 0.5]))) for i in 1:n]

        lgm = latte_from_dppl(
            shared_hier_poisson(y, X, group, G); random = (:β, :u), augment = true,
        )

        layout = latent_groups(lgm)
        @test layout isa OrderedDict{Symbol, UnitRange{Int}}
        @test haskey(layout, :β)
        @test haskey(layout, :u)

        # Augmented layout is [η (n); β (p); u (G)] — so β sits at n+1 : n+p
        # and u at n+p+1 : n+p+G.
        @test layout[:β] == (n + 1):(n + p)
        @test layout[:u] == (n + p + 1):(n + p + G)
    end

    @testset "Non-augmented fast-path: sym → base-latent range" begin
        Random.seed!(2)
        H = 8
        y = rand(Poisson(1.0), H)
        lgm = latte_from_dppl(shared_iid_sumtozero_poisson(y, H); random = (:u,), augment = false)

        layout = latent_groups(lgm)
        @test layout[:u] == 1:H
    end
end

# One shared fit serves both the result-layout check and the named
# linear_combinations checks below.
@testset "latent_groups / linear_combinations on a result" begin
    Random.seed!(3)
    n, p, G = 25, 2, 3
    X = [ones(n) randn(n)]
    group = rand(1:G, n)
    y = [rand(Poisson(exp(X[i, :] ⋅ [0.3, 0.5]))) for i in 1:n]

    lgm = latte_from_dppl(shared_hier_poisson(y, X, group, G); random = (:β, :u))
    result = inla(lgm, y; progress = false)

    @testset "latent_groups(result) matches latent_groups(model)" begin
        @test latent_groups(result) == latent_groups(lgm)
    end

    @testset "linear_combinations named form equals hand-built design matrix" begin
        # Query β at a new "configuration": 2×p matrix.
        β_design = [1.0 0.5; 1.0 -0.5]
        # Pick group 1 and group 3 respectively.
        u_design = zeros(2, G)
        u_design[1, 1] = 1.0
        u_design[2, 3] = 1.0

        # Named form
        named_marginals = linear_combinations(result; β = β_design, u = u_design)

        # Hand-built equivalent
        n_lat = length(latent_marginals(result))
        manual = zeros(2, n_lat)
        manual[:, latent_groups(result)[:β]] = β_design
        manual[:, latent_groups(result)[:u]] = u_design
        manual_marginals = linear_combinations(result, manual)

        for i in 1:2
            @test mean(named_marginals[i]) ≈ mean(manual_marginals[i])
            @test std(named_marginals[i]) ≈ std(manual_marginals[i])
        end
    end

end

@testset "Scalar coefficient is broadcast to a column of ones" begin
    # Needs a 1-dimensional β: the scalar kwarg form `β = 1.0` must equal an explicit
    # ones-column design.
    @model function m_scalar_β(y, group, G)
        τ_u ~ Gamma(2.0, 1.0)
        β ~ MvNormal(zeros(1), 100.0 * I(1))
        u ~ MvNormal(zeros(G), (1 / τ_u) * I(G))
        for i in eachindex(y)
            y[i] ~ Poisson(exp(β[1] + u[group[i]]); check_args = false)
        end
    end
    Random.seed!(5)
    n, G = 20, 3
    group = rand(1:G, n)
    y = rand(Poisson(2.0), n)

    lgm = latte_from_dppl(m_scalar_β(y, group, G); random = (:β, :u))
    result = inla(lgm, y; progress = false)

    u_design = Matrix{Float64}(I, G, G)
    scalar_form = linear_combinations(result; β = 1.0, u = u_design)
    full_form = linear_combinations(result; β = ones(G, 1), u = u_design)

    for i in 1:G
        @test mean(scalar_form[i]) ≈ mean(full_form[i])
        @test std(scalar_form[i]) ≈ std(full_form[i])
    end
end
