using Test
using Latte
using Latte: pmap_executor
using LinearAlgebra

@testset "ParallelExecutor" begin

    @testset "SequentialExecutor" begin
        ex = SequentialExecutor()
        @test ex isa ParallelExecutor

        # Behaves like map
        result = pmap_executor(x -> x^2, [1, 2, 3, 4, 5], ex)
        @test result == [1, 4, 9, 16, 25]

        # Works with empty collection
        @test pmap_executor(x -> x^2, Int[], ex) == Int[]

        # Works with complex return types
        result = pmap_executor(x -> (val = x, sq = x^2), [1, 2, 3], ex)
        @test result[1] == (val = 1, sq = 1)
        @test result[3] == (val = 3, sq = 9)
    end

    @testset "ThreadedExecutor construction" begin
        # Default: uses all available threads
        ex = ThreadedExecutor()
        @test ex isa ParallelExecutor
        @test ex.nworkers == Threads.nthreads()

        # Custom nworkers
        ex4 = ThreadedExecutor(nworkers = 4)
        @test ex4.nworkers == 4

        ex1 = ThreadedExecutor(nworkers = 1)
        @test ex1.nworkers == 1
    end

    @testset "ThreadedExecutor produces correct results" begin
        ex = ThreadedExecutor(nworkers = 2)

        # Same results as map
        result = pmap_executor(x -> x^2, [1, 2, 3, 4, 5], ex)
        @test result == [1, 4, 9, 16, 25]

        # Works with complex return types
        result = pmap_executor(x -> (val = x, sq = x^2), [1, 2, 3], ex)
        @test result[1] == (val = 1, sq = 1)
        @test result[3] == (val = 3, sq = 9)

        # Works with empty collection
        @test pmap_executor(x -> x^2, Int[], ex) == []

        # Works with single element
        @test pmap_executor(x -> x * 10, [7], ex) == [70]
    end

    @testset "ThreadedExecutor runs concurrently" begin
        if Threads.nthreads() > 1
            ex = ThreadedExecutor(nworkers = Threads.nthreads())

            # Collect thread IDs to verify multiple threads are used
            thread_ids = pmap_executor(1:20, ex) do _
                sleep(0.01)  # Small sleep to encourage scheduling on different threads
                Threads.threadid()
            end

            # Should use more than 1 thread
            @test length(unique(thread_ids)) > 1
        else
            @info "Skipping concurrency test: only 1 Julia thread available"
        end
    end

    @testset "ThreadedExecutor manages BLAS threads" begin
        ex = ThreadedExecutor(nworkers = 2)

        # Record BLAS threads before
        blas_before = BLAS.get_num_threads()

        # During execution, BLAS threads should be set to 1
        blas_during = Ref(0)
        pmap_executor([1], ex) do _
            blas_during[] = BLAS.get_num_threads()
            return nothing
        end

        # After execution, BLAS threads should be restored
        blas_after = BLAS.get_num_threads()

        @test blas_during[] == 1
        @test blas_after == blas_before
    end

    @testset "ThreadedExecutor restores BLAS threads on error" begin
        ex = ThreadedExecutor(nworkers = 2)
        blas_before = BLAS.get_num_threads()

        try
            pmap_executor([1, 2, 3], ex) do x
                if x == 2
                    error("deliberate test error")
                end
                return x
            end
        catch e
            # Expected
        end

        # BLAS threads should still be restored
        @test BLAS.get_num_threads() == blas_before
    end

    @testset "Sequential and Threaded produce identical results" begin
        seq = SequentialExecutor()
        thr = ThreadedExecutor(nworkers = 2)

        # Test with a deterministic function (no randomness)
        f(x) = (sum = sum(1:x), prod = prod(1:min(x, 10)))
        inputs = collect(1:20)

        result_seq = pmap_executor(f, inputs, seq)
        result_thr = pmap_executor(f, inputs, thr)

        @test result_seq == result_thr
    end
end
