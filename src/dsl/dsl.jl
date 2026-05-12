# DSL — DPPL `@model` → `LatentGaussianModel` pipeline.
#
# Subdirs:
#   latent/  — prior extraction (structure probing, hp spec, DAG, sparsity)
#   obs/     — observation models (fast paths, AD fallback, composite groups)
#   macro/   — `@latte` macro: markers, AST walker, macro body, prelude lift
#
# `adapter.jl` is the top-level `latte_from_dppl(...)` entry that glues
# the three together.

include("latent/structure_probing.jl")
include("latent/pattern_augment.jl")
include("latent/dag_extraction.jl")
include("latent/hp_spec.jl")
include("latent/latent_prior.jl")

include("obs/obs_model.jl")
include("obs/fixed_kwargs_obs_model.jl")
include("obs/fast_paths.jl")
include("obs/obs_groups.jl")

include("adapter.jl")

include("macro/markers_and_meta.jl")
include("macro/ast_walker.jl")
include("macro/latte.jl")
include("macro/prelude_lift.jl")
