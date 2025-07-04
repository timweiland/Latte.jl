using GaussianMarkovRandomFields

export to_gmrf

"""
    to_gmrf(result::NewtonResult) -> GMRF

Convert a NewtonResult to a GMRF distribution.
"""
function to_gmrf(result::NewtonResult)
    return GMRF(result.μ, result.precision, CholeskySolverBlueprint())
end

"""
    summary(result::NewtonResult)

Print a summary of the Newton optimization result.
"""
function Base.summary(result::NewtonResult)
    println("Newton-Raphson Optimization Summary")
    println("="^40)
    println("Converged: ", result.converged)
    println("Iterations: ", result.iterations)

    if !isempty(result.stats)
        final_stats = result.stats[end]
        println("Final gradient norm: ", final_stats.gradient_norm)
        println("Final Newton decrement: ", final_stats.newton_decrement)
        println("Final step size: ", final_stats.step_size)
    end

    println("Mode dimension: ", length(result.μ))
    println("Precision matrix size: ", size(result.precision))

    return if result.converged
        println("✓ Optimization successful")
    else
        println("⚠ Optimization did not converge")
    end
end

"""
    plot_convergence(result::NewtonResult)

Plot convergence statistics from Newton optimization.
Requires Plots.jl to be loaded.
"""
function plot_convergence(result::NewtonResult)
    if !isdefined(Main, :Plots)
        error("Plots.jl must be loaded to use plot_convergence")
    end

    iterations = [s.iteration for s in result.stats]
    newton_decrements = [s.newton_decrement for s in result.stats]
    gradient_norms = [s.gradient_norm for s in result.stats]
    step_sizes = [s.step_size for s in result.stats]

    p1 = Main.Plots.plot(
        iterations, log10.(newton_decrements .+ 1.0e-16),
        xlabel = "Iteration", ylabel = "log₁₀(Newton Decrement)",
        title = "Newton Decrement", marker = :circle
    )

    p2 = Main.Plots.plot(
        iterations, log10.(gradient_norms .+ 1.0e-16),
        xlabel = "Iteration", ylabel = "log₁₀(Gradient Norm)",
        title = "Gradient Norm", marker = :circle
    )

    p3 = Main.Plots.plot(
        iterations, log10.(step_sizes .+ 1.0e-16),
        xlabel = "Iteration", ylabel = "log₁₀(Step Size)",
        title = "Step Size", marker = :circle
    )

    return Main.Plots.plot(p1, p2, p3, layout = (1, 3), size = (900, 300))
end
