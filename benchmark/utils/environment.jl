# Capture hardware + software metadata for a benchmark run.
#
# Goal: every Result records enough about its environment that someone
# else, on a different machine, can tell whether differences in numbers
# are method differences or environment differences.

using LinearAlgebra: BLAS
using Pkg

"""
    capture_environment() -> Dict{Symbol, Any}

Snapshot the host + Julia + package versions for a benchmark run.

Returns a `Dict` (not a struct) because what we want to capture grows
over time; a dict round-trips through JSON cleanly without a schema
migration.
"""
function capture_environment()
    return Dict{Symbol, Any}(
        :os => string(Sys.KERNEL),
        :arch => string(Sys.ARCH),
        :cpu_brand => _cpu_brand(),
        :cpu_threads => Sys.CPU_THREADS,
        :total_memory_gb => round(Sys.total_memory() / 2^30, digits = 1),
        :julia_version => string(VERSION),
        :julia_threads => Threads.nthreads(),
        :blas_threads => BLAS.get_num_threads(),
        :blas_vendor => string(BLAS.vendor()),
        :package_versions => _package_versions(),
        :hostname => gethostname(),
    )
end

# Best-effort: read from sysctl on Darwin, /proc/cpuinfo on Linux.
function _cpu_brand()
    try
        if Sys.isapple()
            return strip(read(`sysctl -n machdep.cpu.brand_string`, String))
        elseif Sys.islinux()
            for line in eachline("/proc/cpuinfo")
                m = match(r"^model name\s*:\s*(.+)$", line)
                m === nothing || return strip(m.captures[1])
            end
        end
    catch
        # Fall through to unknown
    end
    return "unknown"
end

# Versions of every dep currently active in the project. Filtered to the
# packages that actually drive benchmark results — Pkg's full manifest
# is verbose and most of it is transitive build deps no one cares about.
const _BENCH_RELEVANT_PKGS = Set(
    [
        "Latte",
        "GaussianMarkovRandomFields",
        "DynamicPPL",
        "Distributions",
        "AdvancedHMC",
        "Turing",
        "ForwardDiff",
        "Optim",
    ]
)

function _package_versions()
    versions = OrderedDict{String, String}()
    deps = Pkg.dependencies()
    for (uuid, info) in deps
        info.name in _BENCH_RELEVANT_PKGS || continue
        versions[info.name] = info.version === nothing ? "dev" : string(info.version)
    end
    return versions
end

"""
    git_sha() -> Union{Nothing, String}

Best-effort git SHA of the working tree. Returns `nothing` if not in a
git repo or `git` isn't available.
"""
function git_sha()
    try
        sha = strip(read(pipeline(`git rev-parse HEAD`, stderr = devnull), String))
        return isempty(sha) ? nothing : sha
    catch
        return nothing
    end
end
