Implement the tutorial described in tutorial-ideas/02-nonlinear-regression-bayesian-gam.md on a feature branch feature/tutorial-nonlinear-regression-bayesian-gam.

## Key context from a previous tutorial implementation:
                                  
- Tutorials live in docs/src/literate-tutorials/<name>.jl as Literate.jl source files (use # # for section headers, # for
narrative text).
- Register the tutorial in docs/make.jl under the "Tutorials" section. Don't use @ref links in the tutorial — they don't
resolve in Literate.jl-generated markdown.
- Follow the patterns in the existing tutorials (especially temporal_trend_earthquakes.jl and disease_mapping_spatial.jl)
for imports, formula interface, plotting, and result exploration.
- Formula interface: create functor instances before the formula (e.g., rw1 = RandomWalk(); @formula(y ~ 1 + rw1(time))).
Hyperparameter names are derived from model names (e.g., τ_rw1, τ_besag, τ_iid).
- Use plot! (not distplot!) for hyperparameter marginal plots — Makie dispatches automatically.
- rand(result, n; include_y=true) returns a Vector{NamedTuple} with keys (:θ, :x, :y). The y vector covers the full
augmented latent field — slice to [1:n_obs] for actual observation samples.
- Suppress large outputs: use first(df, 5) for DataFrames, size(matrix) for large matrices.
- Verify the tutorial runs end-to-end with julia --project=docs -e 'include("docs/src/literate-tutorials/<name>.jl")'. Then
verify docs build with make docs.

**Critical instruction**: If at any point you discover a missing feature in the package or a critical bug, do NOT work around it. Stop and report back to me with a detailed bug report (see bug_reports/ for the format we use).

