# Fit the SPDEtoy dataset with R-INLA's Matérn-SPDE model and dump everything
# the Julia side needs to cross-validate against it:
#   - the SHARED mesh (nodes.csv, triangles.csv) — source of truth, Julia
#     rebuilds its FEM discretization from these so both engines solve the
#     SAME discretized problem (isolates inference from meshing).
#   - per-node field marginals + summary, intercept marginal, and the three
#     hyperparameter marginals (obs SD, field range, field Stdev) on the
#     interpretable scale.
#   - rinla_meta.json with cpu.used[["Total"]] for the timing comparison.
#
# Inputs:  params.json (matched PC-prior parameters), data(SPDEtoy)
# Outputs: spdetoy_data.csv + the CSV/JSON files above, under <output_dir>.

suppressPackageStartupMessages({
    library(INLA)
    library(jsonlite)
})

argv <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(argv) >= 1) argv[1] else "."
output_dir <- if (length(argv) >= 2) argv[2] else input_dir

params <- fromJSON(file.path(input_dir, "params.json"))

## ── Data: freeze SPDEtoy to CSV so the Julia side never needs R for it ──
data(SPDEtoy)
df <- data.frame(s1 = SPDEtoy$s1, s2 = SPDEtoy$s2, y = SPDEtoy$y)
write.csv(df, file.path(output_dir, "spdetoy_data.csv"), row.names = FALSE)
coords <- as.matrix(df[, c("s1", "s2")])
n <- nrow(df)
cat(sprintf("SPDEtoy: n = %d, y range [%.2f, %.2f]\n", n, min(df$y), max(df$y)))

## ── Mesh (pinned settings; this is the shared mesh) ──
mesh <- inla.mesh.2d(
    loc = coords,
    max.edge = c(params$mesh_max_edge_inner, params$mesh_max_edge_outer),
    cutoff = params$mesh_cutoff,
    offset = c(params$mesh_offset_inner, params$mesh_offset_outer)
)
cat(sprintf("mesh: %d nodes, %d triangles\n", mesh$n, nrow(mesh$graph$tv)))

## ── SPDE (PC-matern, alpha = 2 → ν = 1 in 2D) ──
spde <- inla.spde2.pcmatern(
    mesh, alpha = 2,
    prior.range = c(params$range_U, params$range_p),
    prior.sigma = c(params$sigma_field_U, params$sigma_field_p)
)

A <- inla.spde.make.A(mesh, loc = coords)
idx <- inla.spde.make.index("spatial", spde$n.spde)
stk <- inla.stack(
    data = list(y = df$y),
    A = list(A, 1),
    effects = list(idx, list(Intercept = rep(1, n))),
    tag = "est"
)

t_rinla <- system.time({
    res <- inla(
        y ~ 0 + Intercept + f(spatial, model = spde),
        data = inla.stack.data(stk),
        family = "gaussian",
        control.predictor = list(A = inla.stack.A(stk)),
        control.family = list(hyper = list(prec = list(
            prior = "pc.prec", param = c(params$sigma_obs_U, params$sigma_obs_alpha)
        ))),
        control.fixed = list(prec.intercept = params$prec_intercept),
        control.inla = list(int.strategy = "grid")
    )
})[["elapsed"]]

## ── Shared mesh (source of truth) ──
write.csv(
    data.frame(node = seq_len(mesh$n), x = mesh$loc[, 1], y = mesh$loc[, 2]),
    file.path(output_dir, "nodes.csv"), row.names = FALSE
)
tv <- mesh$graph$tv
write.csv(
    data.frame(tri = seq_len(nrow(tv)), v1 = tv[, 1], v2 = tv[, 2], v3 = tv[, 3]),
    file.path(output_dir, "triangles.csv"), row.names = FALSE
)

## ── Field node marginals (long form) + summary ──
fm <- res$marginals.random$spatial
field_long <- do.call(rbind, lapply(seq_along(fm), function(i) {
    m <- fm[[i]]
    data.frame(node = i, x = m[, "x"], density = m[, "y"])
}))
write.csv(field_long, file.path(output_dir, "rinla_field_marginals.csv"), row.names = FALSE)

fs <- res$summary.random$spatial
write.csv(
    data.frame(node = seq_len(nrow(fs)), mean = fs$mean, sd = fs$sd),
    file.path(output_dir, "rinla_field_summary.csv"), row.names = FALSE
)

## ── Intercept marginal ──
im <- res$marginals.fixed$Intercept
write.csv(data.frame(x = im[, "x"], density = im[, "y"]),
    file.path(output_dir, "rinla_intercept_marginal.csv"), row.names = FALSE)

## ── Hyperparameter marginals on the interpretable scale ──
# obs precision → SD
pm <- res$marginals.hyperpar[["Precision for the Gaussian observations"]]
sm <- inla.tmarginal(function(x) 1 / sqrt(x), pm)
write.csv(data.frame(x = sm[, "x"], density = sm[, "y"]),
    file.path(output_dir, "rinla_sigma_obs_marginal.csv"), row.names = FALSE)

rgm <- res$marginals.hyperpar[["Range for spatial"]]
write.csv(data.frame(x = rgm[, "x"], density = rgm[, "y"]),
    file.path(output_dir, "rinla_range_marginal.csv"), row.names = FALSE)

sdm <- res$marginals.hyperpar[["Stdev for spatial"]]
write.csv(data.frame(x = sdm[, "x"], density = sdm[, "y"]),
    file.path(output_dir, "rinla_stdev_marginal.csv"), row.names = FALSE)

## ── Meta ──
meta <- list(
    n = n, n_nodes = mesh$n, n_triangles = nrow(tv),
    inla_version = as.character(packageVersion("INLA")),
    elapsed_seconds = as.numeric(res$cpu.used[["Total"]]),
    elapsed_wall = as.numeric(t_rinla),
    intercept_mean = res$summary.fixed["Intercept", "mean"],
    intercept_sd = res$summary.fixed["Intercept", "sd"],
    range_mean = res$summary.hyperpar["Range for spatial", "mean"],
    stdev_mean = res$summary.hyperpar["Stdev for spatial", "mean"],
    status = "ok"
)
writeLines(toJSON(meta, auto_unbox = TRUE, pretty = TRUE),
    file.path(output_dir, "rinla_meta.json"))

cat("done.\n")
