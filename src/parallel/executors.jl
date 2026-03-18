using LinearAlgebra: BLAS

export ParallelExecutor, SequentialExecutor, ThreadedExecutor, pmap_executor

"""
    ParallelExecutor

Abstract type for parallel execution backends. Controls how independent grid point
evaluations are distributed across compute resources.

Concrete subtypes:
- `SequentialExecutor`: Default sequential execution (equivalent to `map`)
- `ThreadedExecutor`: Task-based multi-threaded execution

# Extending
To add a new backend, define a subtype and a method for `pmap_executor`:
```julia
struct MyExecutor <: ParallelExecutor end
IntegratedNestedLaplace.pmap_executor(f, xs, ::MyExecutor) = my_parallel_map(f, xs)
```
"""
abstract type ParallelExecutor end

"""
    SequentialExecutor()

Default executor. Evaluates grid points sequentially using `map`.
"""
struct SequentialExecutor <: ParallelExecutor end

"""
    ThreadedExecutor(; nworkers=Threads.nthreads())

Evaluate grid points across Julia threads using task-based parallelism.

Automatically sets BLAS threads to 1 during execution to avoid oversubscription
(sparse Cholesky does not benefit from BLAS threading), and restores the original
value after.

`nworkers` controls the maximum number of concurrent tasks — useful for
memory-bandwidth-bound problems where fewer workers can be faster.
"""
struct ThreadedExecutor <: ParallelExecutor
    nworkers::Int
end

ThreadedExecutor(; nworkers::Int = Threads.nthreads()) = ThreadedExecutor(nworkers)

"""
    pmap_executor(f, xs, executor::ParallelExecutor) -> Vector

Apply `f` to each element of `xs` using the given executor. Returns results in the
same order as `xs`.
"""
function pmap_executor(f, xs, ::SequentialExecutor)
    return map(f, xs)
end

function pmap_executor(f, xs, executor::ThreadedExecutor)
    n = length(xs)
    n == 0 && return []

    old_blas = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        nw = executor.nworkers
        results = Vector{Any}(undef, n)
        for batch_start in 1:nw:n
            batch_end = min(batch_start + nw - 1, n)
            tasks = map(batch_start:batch_end) do i
                Threads.@spawn f(xs[i])
            end
            for (j, task) in enumerate(tasks)
                results[batch_start + j - 1] = fetch(task)
            end
        end
        return results
    finally
        BLAS.set_num_threads(old_blas)
    end
end
