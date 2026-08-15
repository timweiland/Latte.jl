using Test
using Latte
using GaussianMarkovRandomFields
using LDLFactorizations
using Distributions
using LinearAlgebra
using SparseArrays
using Optim
using Optim.LineSearches: BackTracking
using FiniteDiff
using Random
using DynamicPPL: @model

# Regression: a prior without `Distributions.mode` must yield an actionable
# error from the mode-finder, not a cryptic `MethodError: iterate`.
struct _NoModePrior <: ContinuousUnivariateDistribution end
Distributions.logpdf(::_NoModePrior, x::Real) = -abs(x)

@testset "mode-finder: actionable error for a prior lacking `mode`" begin
    err = try
        Latte._robust_initial_value(_NoModePrior())
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("mode_init", err.msg)
    @test occursin("_NoModePrior", err.msg)
end

@testset "Mode Finding" begin

    @testset "Basic hyperparameter_logpdf" begin
        # Create a simple test model for consistent testing
        function create_simple_model(n = 10)
            # Single hyperparameter controlling latent field precision
            spec = @hyperparams begin
                (τ ~ Gamma(2, 1), transform = log, space = natural)
            end

            function simple_latent(; τ, kwargs...)
                Q = spdiagm(0 => fill(τ, n))  # White noise precision
                return (zeros(n), Q)
            end
            obs_model = ExponentialFamily(Bernoulli)  # No hyperparameters
            return LatentGaussianModel(spec, FunctionLatentModel(simple_latent, n), obs_model)
        end

        model = create_simple_model(5)
        y_test = [true, false, true, false, true]
        spec = model.hyperparameter_spec
        ws = make_workspace(model.latent_prior; τ = 1.0)

        # Test basic hyperparameter_logpdf evaluation
        θ_test_vec = [log(1.5)]  # Working space
        θ_test = WorkingHyperparameters(θ_test_vec, spec)
        logpdf_val = hyperparameter_logpdf(model, θ_test, y_test; ws = ws)
        @test isfinite(logpdf_val)

        # Test that function works at various points
        θ_low_vec = [log(0.5)]
        θ_low = WorkingHyperparameters(θ_low_vec, spec)
        logpdf_low = hyperparameter_logpdf(model, θ_low, y_test; ws = ws)
        @test isfinite(logpdf_low)
    end

    @testset "Optimality Conditions" begin
        # Create test model
        spec = @hyperparams begin
            (σ ~ InverseGamma(3, 2), transform = log, space = natural)
        end

        function precision_latent(; σ, kwargs...)
            n = 8
            Q = spdiagm(0 => fill(1 / σ^2, n))
            return (zeros(n), Q)
        end
        obs_model = ExponentialFamily(Normal)
        model = LatentGaussianModel(spec, FunctionLatentModel(precision_latent, 8), obs_model)

        # Generate test data (seeded — Optim's f_reltol=1e-3 stop can leave
        # `|grad|` above the test threshold on flat realisations)
        Random.seed!(42)
        y_test = randn(8)

        # Find mode
        θ_star, mode_points, mode_logdensities = find_hyperparameter_mode(model, y_test)

        @test θ_star isa WorkingHyperparameters
        @test length(θ_star) == 1  # WorkingHyperparameters with 1 free parameter

        # Convert to natural space to check the value
        θ_star_natural = convert(NaturalHyperparameters, θ_star)
        θ_star_nt = convert(NamedTuple, θ_star_natural)
        @test isfinite(θ_star_nt.σ)  # σ should be finite in natural space
        @test θ_star_nt.σ > 0  # σ must be positive in natural space

        # Test optimality condition: gradient should be ≈ 0 at mode
        # Use the working space vector for gradient computation
        θ_star_vec = θ_star.θ
        ws = make_workspace(model.latent_prior; σ = 1.0)
        function objective(θ_vec)
            θ_w = WorkingHyperparameters(θ_vec, spec)
            return hyperparameter_logpdf(model, θ_w, y_test; ws = ws)
        end

        grad_at_mode = FiniteDiff.finite_difference_gradient(objective, θ_star_vec)
        @test abs(grad_at_mode[1]) < 1.0e-3  # Gradient should be near zero

        # Test second-order condition: Hessian should be negative definite
        hess_at_mode = FiniteDiff.finite_difference_hessian(objective, θ_star_vec)
        @test hess_at_mode[1, 1] < 0  # Negative definite for 1D case

        # Test mode collection during optimization
        if mode_points !== nothing
            @test length(mode_points) > 0
            @test length(mode_points) == length(mode_logdensities)
            @test all(isfinite, mode_logdensities)
        end
    end

    @testset "Local optimality" begin
        spec = @hyperparams begin
            (τ ~ Gamma(2, 2), transform = log, space = natural)
        end

        function test_latent(; τ, kwargs...)
            n = 6
            Q = spdiagm(0 => fill(τ, n))
            return (zeros(n), Q)
        end
        obs_model = ExponentialFamily(Bernoulli)
        model = LatentGaussianModel(spec, FunctionLatentModel(test_latent, 6), obs_model)

        y_test = [true, true, false, true, false, false]

        # Find mode with default starting point
        θ_star1, _, _ = find_hyperparameter_mode(model, y_test)

        # Test that the mode is actually better than nearby points
        δ = 0.1
        θ_nearby = θ_star1 .+ δ  # Broadcasting preserves WorkingHyperparameters type
        ws = make_workspace(model.latent_prior; τ = 1.0)
        logpdf_mode = hyperparameter_logpdf(model, θ_star1, y_test; ws = ws)
        logpdf_nearby = hyperparameter_logpdf(model, θ_nearby, y_test; ws = ws)

        @test logpdf_mode >= logpdf_nearby  # Mode should be at least as good
    end

    @testset "Robustness and Edge Cases" begin
        # Test behavior with very peaked/flat posteriors
        spec = @hyperparams begin
            (λ ~ Exponential(1), transform = log, space = natural)
        end

        function exponential_latent(; λ, kwargs...)
            n = 3
            Q = spdiagm(0 => fill(λ + 1.0e-6, n))  # Add small regularization
            return (zeros(n), Q)
        end
        obs_model = ExponentialFamily(Bernoulli)
        model = LatentGaussianModel(spec, FunctionLatentModel(exponential_latent, 3), obs_model)

        # Test with moderate data (not too extreme to avoid numerical issues)
        y_moderate = [true, false, true]

        θ_star, _, _ = find_hyperparameter_mode(model, y_moderate)
        @test isfinite(θ_star[1])
        # θ_star is in working space (log scale)
        # Just verify we get a reasonable finite value
    end

    @testset "Initial hyperparameter guess" begin
        # Test the initial guess from HyperparameterSpec
        spec = @hyperparams begin
            (σ ~ InverseGamma(2, 1), transform = log, space = natural)
            (ρ ~ Beta(2, 2), transform = logit, space = natural)
            (λ ~ Exponential(1), transform = log, space = natural)
        end

        θ_init = initial_hyperparameter_guess(spec)

        @test length(θ_init) == 3
        @test all(isfinite, θ_init)
        # Initial guesses are working-space modes computed via Brent search
        # (see _working_space_mode_1d). Tolerance reflects Brent's accuracy.
        # Beta(2,2) under logit and Exp(1) under log both have working-space
        # modes at u = 0 by analytical argument.
        @test θ_init[1] < 0
        @test θ_init[2] ≈ 0 atol = 1.0e-6
        @test θ_init[3] ≈ 0 atol = 1.0e-6
    end

    @testset "Constrained prior through the DPPL adapter under AD" begin
        # Regression: a Besag (sum-to-zero) prior materializes as a plain
        # ConstrainedGMRF on the Dual-θ DAG path — no `constraints` FIELD, the
        # constraint lives in the type. The prior-logdet fast path must not
        # claim it; misclassifying it as unconstrained either throws (missing
        # Dual logdetcov method) or silently drops the constraint correction.
        using DynamicPPL: @model
        n = 6
        W = spdiagm(-1 => ones(n - 1), 1 => ones(n - 1))
        @model function besag_chain(y, W, n)
            τ ~ Gamma(2.0, 1.0)
            u ~ BesagModel(W)(τ = τ)
            fixed ~ MvNormal(zeros(1), 100.0 * I(1))
            for i in 1:n
                y[i] ~ Poisson(exp(fixed[1] + u[i]))
            end
        end
        Random.seed!(5)
        y = rand(1:5, n)
        lgm = latte_from_dppl(besag_chain(y, W, n); random = (:fixed, :u))

        θ_ad, _, _, info_ad = find_hyperparameter_mode(lgm, y)
        @test all(isfinite, collect(θ_ad))
        @test isfinite(maximum(info_ad.final_logdensities))

        # The AD path must agree with finite differences (which never enter
        # the Dual branch) — a silently dropped constraint correction would
        # shift the mode.
        θ_fd, _, _, _ = find_hyperparameter_mode(
            lgm, y; diff_strategy = FiniteDiffStrategy(),
        )
        @test collect(θ_ad) ≈ collect(θ_fd) atol = 1.0e-3
    end

    @testset "Second-order mode finding (trust region)" begin
        # Two hyperparameters so the SR1 curvature update is actually exercised.
        function create_two_hp_model()
            spec = @hyperparams begin
                (σ_latent ~ InverseGamma(2, 1), transform = log, space = natural)
                (σ ~ InverseGamma(2, 1), transform = log, space = natural)
            end
            latent(; σ_latent, kwargs...) = (zeros(4), spdiagm(0 => fill(1 / σ_latent^2, 4)))
            return LatentGaussianModel(
                spec, FunctionLatentModel(latent, 4), ExponentialFamily(Normal),
            )
        end

        @testset "NewtonTrustRegion reaches the BFGS optimum" begin
            model = create_two_hp_model()
            y = [0.5, -0.3, 0.8, -0.2]

            θ_bfgs, _, _, info_bfgs = find_hyperparameter_mode(
                model, y; method = BFGS(linesearch = BackTracking(order = 3, maxstep = 5.0)),
            )
            θ_ntr, pts, lps, info_ntr = find_hyperparameter_mode(
                model, y; method = NewtonTrustRegion(),
            )

            @test info_ntr.converged
            @test collect(θ_ntr) ≈ collect(θ_bfgs) atol = 1.0e-4
            @test maximum(info_ntr.final_logdensities) ≈
                maximum(info_bfgs.final_logdensities) atol = 1.0e-6
            # Primal evaluations are still collected through the fused fg path
            @test length(pts) == length(lps) > 0
        end

        @testset "SR1-only Hessian model (hessian_refresh = 0)" begin
            model = create_two_hp_model()
            y = [0.5, -0.3, 0.8, -0.2]
            θ_bfgs, _, _, _ = find_hyperparameter_mode(
                model, y; method = BFGS(linesearch = BackTracking(order = 3, maxstep = 5.0)),
            )
            θ_sr1, _, _, info = find_hyperparameter_mode(
                model, y; method = NewtonTrustRegion(), hessian_refresh = 0,
            )
            @test info.converged
            @test collect(θ_sr1) ≈ collect(θ_bfgs) atol = 1.0e-4
        end

        @testset "mode_info carries the model Hessian for exploration handoff" begin
            model = create_two_hp_model()
            y = [0.5, -0.3, 0.8, -0.2]

            _, _, _, info_ntr = find_hyperparameter_mode(
                model, y; method = NewtonTrustRegion(),
            )
            H = info_ntr.negative_hessian
            @test H isa Matrix{Float64}
            @test size(H) == (2, 2)
            @test issymmetric(H)
            @test all(eigvals(H) .> 0)

            # First-order methods have no model Hessian to hand off
            _, _, _, info_bfgs = find_hyperparameter_mode(
                model, y; method = BFGS(linesearch = BackTracking(order = 3, maxstep = 5.0)),
            )
            @test info_bfgs.negative_hessian === nothing
        end

        @testset "FiniteDiffStrategy rejects second-order methods" begin
            model = create_two_hp_model()
            y = [0.5, -0.3, 0.8, -0.2]
            @test_throws ArgumentError find_hyperparameter_mode(
                model, y; method = NewtonTrustRegion(),
                diff_strategy = FiniteDiffStrategy(),
            )
        end

        @testset "default method resolution" begin
            @test Latte._resolve_mode_method(nothing, ADStrategy(), 1) isa NewtonTrustRegion
            @test Latte._resolve_mode_method(nothing, ADStrategy(), 6) isa NewtonTrustRegion
            @test Latte._resolve_mode_method(nothing, ADStrategy(), 7) isa BFGS
            @test Latte._resolve_mode_method(nothing, FiniteDiffStrategy(), 1) isa BFGS
            explicit = BFGS()
            @test Latte._resolve_mode_method(explicit, ADStrategy(), 1) === explicit

            # Integration: the resolved default at small d is the trust
            # region — observable through the exploration Hessian handoff —
            # and the FD default silently falls back to BFGS instead of
            # throwing.
            model = create_two_hp_model()
            y = [0.5, -0.3, 0.8, -0.2]
            _, _, _, info_default = find_hyperparameter_mode(model, y)
            @test info_default.negative_hessian !== nothing
            _, _, _, info_fd = find_hyperparameter_mode(
                model, y; diff_strategy = FiniteDiffStrategy(),
            )
            @test info_fd.negative_hessian === nothing
        end

        @testset "benign stall classification" begin
            @test Latte._benign_stall(1.0e-5)
            @test Latte._benign_stall(5.0e-3)
            @test !Latte._benign_stall(0.1)
            @test !Latte._benign_stall(NaN)
            @test !Latte._benign_stall(Inf)
        end

        @testset "forward-difference model Hessian is exact on quadratics" begin
            Random.seed!(11)
            n = 3
            M = randn(n, n)
            A = Symmetric(M + M' + 2n * I)
            b = randn(n)
            f(x) = 0.5 * dot(x, A * x) + dot(b, x)
            x0 = randn(n)
            H = Latte._forward_diff_hessian(
                f, Latte.ADStrategy().backend, x0, A * x0 + b,
            )
            @test H ≈ Matrix(A) rtol = 1.0e-6
        end

        @testset "SR1 update reconstructs a quadratic's Hessian" begin
            Random.seed!(7)
            n = 4
            M = randn(n, n)
            A = Symmetric(M + M' + 2n * I)
            B = Matrix(1.0 * I, n, n)
            # On a quadratic (y = A s exactly), SR1 has the hereditary property:
            # n well-defined updates along independent steps recover A exactly.
            for _ in 1:n
                s = randn(n)
                @test Latte._sr1_update!(B, s, A * s)
            end
            @test B ≈ Matrix(A) rtol = 1.0e-8
            # Degenerate step (y = B s already satisfied) is skipped, B unchanged
            s = randn(n)
            @test !Latte._sr1_update!(B, s, B * s)
        end
    end

    @testset "Stall detection" begin
        @testset "detector unit behavior" begin
            det = Latte._StallDetector(3, 1.0e-8)
            @test !Latte._stall_check!(det, 100.0)      # first value: improvement
            @test !Latte._stall_check!(det, 99.0)       # improvement resets
            @test !Latte._stall_check!(det, 99.0)       # 1
            @test !Latte._stall_check!(det, 99.0 - 1.0e-10)  # 2: below tol
            @test Latte._stall_check!(det, 99.0)        # 3 in a row: stalled
            @test det.triggered

            # Real progress never triggers
            det2 = Latte._StallDetector(3, 1.0e-8)
            for i in 1:20
                @test !Latte._stall_check!(det2, 100.0 - i)
            end

            # window = 0 disables detection
            det3 = Latte._StallDetector(0, 1.0e-8)
            for _ in 1:50
                @test !Latte._stall_check!(det3, 1.0)
            end
        end

        @testset "stalled optimization stops early and is reported" begin
            spec = @hyperparams begin
                (τ ~ Gamma(2, 1), transform = log, space = natural)
            end
            latent(; τ, kwargs...) = (zeros(5), spdiagm(0 => fill(τ, 5)))
            model = LatentGaussianModel(
                spec, FunctionLatentModel(latent, 5), ExponentialFamily(Bernoulli),
            )
            y = [true, false, true, false, true]

            # stall_f_tol = Inf treats every iteration as non-improving, so the
            # stall window is the effective iteration budget. Window 1 stops at
            # the first callback — the model itself converges within a few
            # iterations, so a larger window would never fire.
            # Start far from the mode so the stop carries a large gradient and
            # deterministically takes the :warn (non-benign) branch.
            θ, _, _, info = @test_logs (:warn, r"stalled") match_mode = :any find_hyperparameter_mode(
                model, y; stall_iterations = 1, stall_f_tol = Inf,
                mode_init = [(; τ = 1.0e6)],
            )
            @test info.stalled
            @test !info.converged
            @test all(isfinite, collect(θ))

            # A healthy run does not stall
            _, _, _, info_ok = find_hyperparameter_mode(model, y)
            @test !info_ok.stalled
            @test info_ok.converged
        end
    end

    @testset "Parallel multistart matches serial" begin
        # Multi-start mode-finding parallelized over an executor must give the
        # *same* result as the sequential path: identical starts (same seeded
        # RNG) → identical per-start optima and the same selected best mode,
        # regardless of execution order or threading.
        n = 6
        spec = @hyperparams begin
            (τ ~ Gamma(2, 1), transform = log, space = natural)
        end
        latent(; τ, kwargs...) = (zeros(n), spdiagm(0 => fill(τ, n)))
        lgm = LatentGaussianModel(
            spec, FunctionLatentModel(latent, n), ExponentialFamily(Poisson),
        )
        Random.seed!(99)
        y = rand(0:4, n)

        mk_starts() = RandomStarts(4; rng = MersenneTwister(2024))
        θ_seq, _, _, info_seq = find_hyperparameter_mode(
            lgm, y; mode_init = mk_starts(), executor = SequentialExecutor(),
            diff_strategy = FiniteDiffStrategy(),
        )
        θ_par, _, _, info_par = find_hyperparameter_mode(
            lgm, y; mode_init = mk_starts(), executor = ThreadedExecutor(nworkers = 2),
            diff_strategy = FiniteDiffStrategy(),
        )

        @test collect(θ_seq) ≈ collect(θ_par) rtol = 1.0e-10
        @test info_seq.final_logdensities ≈ info_par.final_logdensities rtol = 1.0e-10
        @test info_seq.best_start_index == info_par.best_start_index
        @test info_seq.runner_up_gap ≈ info_par.runner_up_gap rtol = 1.0e-10
    end

end
