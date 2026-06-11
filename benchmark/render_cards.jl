# Aggregate per-scenario result.json (timing + per-component KS vectors) and
# overlay.json (representative latent marginal + hyperparameter marginal curves)
# into docs/src/data/benchmark_cards.json, consumed by the BenchCard component
# (one tabbed card per benchmark: Marginals / KS spread / Speed).
#
#   julia --project=benchmark benchmark/render_cards.jl

using Pkg
Pkg.activate(@__DIR__)

using JSON3
using Statistics

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
const RINLA_DIR = joinpath(REPO_ROOT, "benchmark", "external", "rinla")
const DATA_OUT = joinpath(REPO_ROOT, "docs", "src", "data", "benchmark_cards.json")

# Human title/scenario + the primary latent-marginal KS vector (field in result.json).
const CARDS = [
    (id = "seeds", title = "Crowder seeds", scenario = "Binomial GLMM · IID plate", ksfield = :ks_b, kslabel = "plate random effects"),
    (id = "scotland", title = "Scottish lip cancer", scenario = "Poisson + Besag · log-offset", ksfield = :ks_u, kslabel = "district spatial effects"),
    (id = "nhtemp", title = "New Haven temperature", scenario = "Normal + RW2 · 1912–1971", ksfield = :ks_x, kslabel = "RW2 trend nodes"),
    (id = "tokyo", title = "Tokyo rainfall", scenario = "Binomial + RW2 · 366 days", ksfield = :ks_x, kslabel = "RW2 day nodes"),
    (id = "epil", title = "Epil (BUGS)", scenario = "Poisson + IID · 59 subjects", ksfield = :ks_subj, kslabel = "subject random effects"),
    (id = "spdetoy", title = "SPDEtoy", scenario = "Gaussian + Matérn SPDE", ksfield = :ks_field, kslabel = "SPDE field nodes"),
    (id = "paranaprec", title = "Paraná precipitation", scenario = "Gamma + RW1 + Matérn SPDE", ksfield = :ks_field, kslabel = "SPDE field nodes"),
]

# Histogram of a KS vector into nb bins over [0, hi].
function _hist(v::Vector{Float64}; nb = 24)
    hi = max(0.13, maximum(v) * 1.02)
    edges = collect(range(0.0, hi; length = nb + 1))
    counts = zeros(Int, nb)
    for x in v
        b = clamp(searchsortedlast(edges, x), 1, nb)
        counts[b] += 1
    end
    return edges, counts
end

function _card(c)
    rfile = joinpath(RINLA_DIR, c.id, "_workdir", "result.json")
    ofile = joinpath(RINLA_DIR, c.id, "_workdir", "overlay.json")
    (isfile(rfile) && isfile(ofile)) || return nothing
    r = JSON3.read(read(rfile, String))
    o = JSON3.read(read(ofile, String))

    ksv = sort(collect(Float64, getproperty(r, c.ksfield)))
    edges, counts = _hist(ksv)

    marginals = Any[
        Dict(
            "kind" => "latent", "label" => String(o.label), "ks" => Float64(o.ks),
            "grid" => o.grid, "latte" => o.latte, "rinla" => o.rinla,
        ),
    ]
    if haskey(o, :hypers)
        for h in o.hypers
            push!(
                marginals, Dict(
                    "kind" => "hyper", "label" => String(h.name), "ks" => Float64(h.ks),
                    "grid" => h.grid, "latte" => h.latte, "rinla" => h.rinla,
                )
            )
        end
    end

    return Dict(
        "id" => c.id, "title" => c.title, "scenario" => c.scenario,
        "timing" => Dict(
            "warm" => round(Float64(r.t_latte_warm), digits = 4),
            "cold" => round(Float64(r.t_latte_cold), digits = 2),
            "rinla" => round(Float64(r.t_rinla), digits = 3),
            "speedup" => round(Float64(r.t_rinla) / Float64(r.t_latte_warm), digits = 1),
        ),
        "ks_block" => Dict(
            "label" => c.kslabel, "n" => length(ksv),
            "max" => round(maximum(ksv), digits = 3),
            "median" => round(ksv[div(length(ksv), 2) + 1], digits = 3),
            "edges" => round.(edges, digits = 4), "counts" => counts,
        ),
        "marginals" => marginals,
    )
end

cards = filter(!isnothing, [_card(c) for c in CARDS])
mkpath(dirname(DATA_OUT))
open(DATA_OUT, "w") do io
    JSON3.pretty(io, cards)
end
println("Wrote $(length(cards)) cards → $(relpath(DATA_OUT, REPO_ROOT))")
for c in cards
    println("  - ", rpad(c["id"], 12), " marginals=", length(c["marginals"]), "  ks_n=", c["ks_block"]["n"], "  speedup=", c["timing"]["speedup"], "x")
end
