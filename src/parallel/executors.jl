using LinearAlgebra: BLAS
using GaussianMarkovRandomFields: AbstractLatentWorkspacePool, with_workspace

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
Latte.pmap_executor(f, xs, ::MyExecutor) = my_parallel_map(f, xs)
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
    pmap_executor(f, xs, executor::ParallelExecutor; on_complete=nothing) -> Vector

Apply `f` to each element of `xs` using the given executor. Returns
results in the same order as `xs`.

The optional `on_complete::Function` callback fires per item *from the
main thread*, with the 1-based completion index — safe to call into
`ProgressMeter` from there. For the threaded backend it fires as each
`fetch` returns within a batch, giving real-time progress during long
parallel evaluations.
"""
function pmap_executor(f, xs, ::SequentialExecutor; on_complete = nothing)
    on_complete === nothing && return map(f, xs)
    return [(r = f(x); on_complete(i); r) for (i, x) in enumerate(xs)]
end

function pmap_executor(f, xs, executor::ThreadedExecutor; on_complete = nothing)
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
                idx = batch_start + j - 1
                results[idx] = fetch(task)
                on_complete === nothing || on_complete(idx)
            end
        end
        return results
    finally
        BLAS.set_num_threads(old_blas)
    end
end

# ---------------------------------------------------------------------------
# Pool-aware overloads (Phase 2).
#
# Signature: `f(item, ws) -> result`. Each evaluation receives a workspace
# checked out from the pool via `with_workspace`. Sequential path uses one
# workspace for the entire loop; threaded path checks out a workspace per
# task so concurrent Newton/factor-update work never races on a shared
# GMRFWorkspace.
# ---------------------------------------------------------------------------

"""
    pmap_executor(f, xs, executor::ParallelExecutor, pool::AbstractLatentWorkspacePool) -> Vector

Pool-aware variant of [`pmap_executor`](@ref). `f` has signature
`f(item, ws)` and receives a workspace checked out from `pool` for each
evaluation. Returns results in `xs` order.
"""
function pmap_executor(
        f, xs, ::SequentialExecutor, pool::AbstractLatentWorkspacePool;
        on_complete = nothing,
    )
    return with_workspace(pool) do ws
        on_complete === nothing && return map(x -> f(x, ws), xs)
        return [(r = f(x, ws); on_complete(i); r) for (i, x) in enumerate(xs)]
    end
end

function pmap_executor(
        f, xs, executor::ThreadedExecutor, pool::AbstractLatentWorkspacePool;
        on_complete = nothing,
    )
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
                Threads.@spawn with_workspace(pool) do ws
                    f(xs[i], ws)
                end
            end
            for (j, task) in enumerate(tasks)
                idx = batch_start + j - 1
                results[idx] = fetch(task)
                on_complete === nothing || on_complete(idx)
            end
        end
        return results
    finally
        BLAS.set_num_threads(old_blas)
    end
end

"""
    _pool_size(executor::ParallelExecutor) -> Int

How many workspaces a pool needs to cover the active concurrency of a
given executor. Sequential → 1; Threaded → `nworkers`. Custom executors
default to 1; override by dispatching on the executor subtype.
"""
_pool_size(::SequentialExecutor) = 1
_pool_size(e::ThreadedExecutor) = e.nworkers
_pool_size(::ParallelExecutor) = 1
