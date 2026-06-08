# Experiment (not committed): quantify the biggest perf lever found by profiling
# — re-assembling the κ-independent FEM matrices C, G on every θ evaluation.
#
# A cached model assembles C, G ONCE and reuses GMRFs' own inner
# `_matern_precision_only(C, K, α, …)` per θ, so the precision is bit-identical
# (accuracy preserved). We verify Q matches, then time precision_matrix
# cached-vs-uncached to see how much the redundant assembly costs.

include(joinpath(@__DIR__, "spdetoy_compare.jl"))   # definitions only (main is guarded)

using SparseArrays

const GMRF = GaussianMarkovRandomFields
const FEMExt = Base.get_extension(GMRF, :GaussianMarkovRandomFieldsFEM)

disc, n_nodes = load_discretization(WORKDIR)
base = MaternModel(disc; smoothness = 0)            # ν = 1, alpha = 2

# Assemble C, G ONCE (κ-independent).
ν = GMRF.smoothness_to_ν(0, 2)                       # = 1
α_val = Integer(ν + 2 ÷ 2)                           # = 2
cellvalues = Ferrite.CellValues(disc.quadrature_rule, disc.interpolation, disc.geom_interpolation)
C_cache, G_cache = GMRF.assemble_C_G_matrices(cellvalues, disc.dof_handler, disc.interpolation, Matrix{Float64}(I, 2, 2))

# Per-θ precision from cached C, G — mirrors matern_precision_only + MaternModel's τ·Q.
function cached_precision(τ, range)
    κ = GMRF.range_to_κ(range, ν)
    K = κ^2 * C_cache + G_cache
    σ²_natural = 1 / (4π * κ^2)                       # ν=1: gamma(1)/(gamma(2)·4π·κ²)
    Qu = FEMExt._matern_precision_only(C_cache, K, α_val, disc.constraint_handler, disc.constraint_noise, σ²_natural)
    return τ * Qu
end

# ── verify bit-identical precision ──
Q_orig = GMRF.precision_matrix(base; τ = 2.0, range = 0.4)
Q_cached = cached_precision(2.0, 0.4)
maxdiff = maximum(abs, Q_orig - Q_cached)
relerr = maxdiff / maximum(abs, Q_orig)
@info "precision match" maxdiff relerr identical = (maxdiff == 0.0)

# ── time precision_matrix: original (re-assembles C,G) vs cached ──
GMRF.precision_matrix(base; τ = 2.0, range = 0.4); cached_precision(2.0, 0.4)   # warmup
N = 200
t_orig = @elapsed for _ in 1:N
    GMRF.precision_matrix(base; τ = 2.0, range = 0.4)
end
t_cached = @elapsed for _ in 1:N
    cached_precision(2.0, 0.4)
end
@info "precision_matrix timing" calls = N t_orig = round(t_orig, digits = 3) t_cached = round(t_cached, digits = 3) per_call_orig_ms = round(1000t_orig / N, digits = 2) per_call_cached_ms = round(1000t_cached / N, digits = 2) speedup = round(t_orig / t_cached, digits = 2)
