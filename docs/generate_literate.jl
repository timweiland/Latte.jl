using Literate

# Build modes (set via env var):
#   LATTE_REBUILD_TUTORIALS=1  → force full rebuild (use in CI / after API changes)
#   LATTE_SKIP_TUTORIALS=1     → skip every tutorial; reuse existing .md outputs
# Default behaviour: incremental — rebuild a tutorial only when its .jl source
# is newer than the corresponding .md output.
const REBUILD_ALL = get(ENV, "LATTE_REBUILD_TUTORIALS", "0") in ("1", "true")
const SKIP_ALL = get(ENV, "LATTE_SKIP_TUTORIALS", "0") in ("1", "true")

const TUTORIALS_IN = joinpath(@__DIR__, "src", "literate-tutorials")
const TUTORIALS_OUT = joinpath(@__DIR__, "src", "tutorials")
mkpath(TUTORIALS_OUT)

const TUTORIALS_DATA = joinpath(@__DIR__, "src", "literate-tutorials", "data")
mkpath(TUTORIALS_DATA)

# A tutorial is "up to date" if its .md output exists and is at least as new
# as the .jl source. mtime comparison only — does not catch upstream package
# code changes, so use LATTE_REBUILD_TUTORIALS=1 after touching package internals.
function _md_up_to_date(program, out_dir)
    name = basename(program)
    out_md = joinpath(out_dir, replace(name, ".jl" => ".md"))
    return isfile(out_md) && mtime(out_md) >= mtime(program)
end

counts = Dict(:built => 0, :skipped => 0, :failed => 0)

for (IN, OUT) in [(TUTORIALS_IN, TUTORIALS_OUT)]
    ## First pass: process .jl files and copy known assets
    for program in readdir(IN; join = true)
        name = basename(program)
        if endswith(program, ".jl")
            if SKIP_ALL
                @info "[literate] skip $name (LATTE_SKIP_TUTORIALS=1)"
                counts[:skipped] += 1
                continue
            end
            if !REBUILD_ALL && _md_up_to_date(program, OUT)
                @info "[literate] cached $name"
                counts[:skipped] += 1
                continue
            end
            @info "[literate] build $name"
            # `execute = true, documenter = false` runs each cell during
            # Literate's pass and bakes the outputs straight into plain
            # markdown code blocks. The alternative — Literate emitting
            # @example blocks for Documenter to execute — re-runs every
            # tutorial cell on every docs build, so the LATTE_SKIP_TUTORIALS
            # short-circuit doesn't actually skip anything.
            #
            # Each tutorial is isolated: one failing to build (e.g. a missing
            # optional extension in the build env) should not abort the whole
            # docs build. Failures are reported loudly and the existing .md, if
            # any, is left in place.
            try
                Literate.script(program, OUT)
                Literate.markdown(program, OUT; execute = true, documenter = false)
                Literate.notebook(program, OUT)
                counts[:built] += 1
            catch e
                @error "[literate] FAILED to build $name — skipping (keeping any existing .md)" exception = (e, catch_backtrace())
                counts[:failed] += 1
            end
        elseif any(endswith.(name, [".png", ".jpg", ".gif"]))
            cp(program, joinpath(OUT, name); force = true)
        elseif !isdir(program)
            @warn "ignoring $program"
        end
    end
    ## Second pass: copy any assets generated during Literate execution
    for asset in readdir(IN; join = true)
        name = basename(asset)
        if any(endswith.(name, [".png", ".jpg", ".gif"]))
            cp(asset, joinpath(OUT, name); force = true)
        end
    end
end

@info "[literate] tutorials built: $(counts[:built]), cached/skipped: $(counts[:skipped]), failed: $(counts[:failed])" *
    (
    counts[:skipped] > 0 && !REBUILD_ALL && !SKIP_ALL ?
        " — pass LATTE_REBUILD_TUTORIALS=1 to force rebuild" : ""
)
counts[:failed] > 0 && @warn "[literate] $(counts[:failed]) tutorial(s) FAILED to build — their pages are stale; see the errors above."
