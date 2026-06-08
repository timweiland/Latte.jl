# Paraná precipitation — R-INLA reference for the SPDE book §2.8 model.
#
# Faithful structure: per-station mean daily rainfall (2011), Gamma likelihood
# (log link), linear predictor = intercept + rw1(seaDist) + Matérn-SPDE field,
# on a NON-CONVEX-hull mesh of the 616 stations. seaDist = great-circle distance
# (km) to the Paraná coast (PRborder[1034:1078,]).
#
# Dumps the shared mesh (nodes/triangles), the shared seaDist + its rw1 grouping,
# and reference marginals (field nodes, rw1, intercept, hyperparameters) for the
# Julia side to cross-validate against.

suppressPackageStartupMessages({
    library(INLA)
    library(jsonlite)
})

argv <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(argv) >= 1) argv[1] else "."
output_dir <- if (length(argv) >= 2) argv[2] else input_dir
p <- fromJSON(file.path(input_dir, "params.json"))

## ── Data ──
data(PRprec); data(PRborder)
coords <- as.matrix(PRprec[, 1:2])
n <- nrow(coords)
ybar <- rowMeans(PRprec[, -(1:3)], na.rm = TRUE)          # mean daily rainfall, 2011

## ── seaDist: great-circle km to the coast (PRborder[1034:1078,]) ──
hav <- function(lo1, la1, lo2, la2) {
    R <- 6371; dlo <- (lo2 - lo1) * pi / 180; dla <- (la2 - la1) * pi / 180
    a <- sin(dla / 2)^2 + cos(la1 * pi / 180) * cos(la2 * pi / 180) * sin(dlo / 2)^2
    2 * R * asin(pmin(1, sqrt(a)))
}
coast <- PRborder[1034:1078, ]
seaDist <- sapply(seq_len(n), function(i) min(hav(coords[i, 1], coords[i, 2], coast[, 1], coast[, 2])))

## ── Mesh: non-convex hull of the stations ──
bnd <- inla.nonconvex.hull(coords, convex = p$mesh_convex, concave = p$mesh_concave, resolution = c(100, 100))
mesh <- inla.mesh.2d(
    loc = coords, boundary = bnd,
    max.edge = c(p$mesh_max_edge_inner, p$mesh_max_edge_outer),
    cutoff = p$mesh_cutoff
)
cat(sprintf("n=%d  mesh: %d nodes, %d triangles\n", n, mesh$n, nrow(mesh$graph$tv)))

## ── SPDE (pcmatern, alpha=2) ──
spde <- inla.spde2.pcmatern(
    mesh, alpha = 2,
    prior.range = c(p$range_U, p$range_p),
    prior.sigma = c(p$sigma_field_U, p$sigma_field_p)
)
A.spde <- inla.spde.make.A(mesh, loc = coords)
idx.spde <- inla.spde.make.index("spatial", spde$n.spde)

## ── rw1 on grouped seaDist ──
seaDist.grp <- inla.group(seaDist, n = p$n_groups)
rw1.hyper <- list(prec = list(prior = "pc.prec", param = c(p$rw1_sigma_U, p$rw1_sigma_alpha)))

stk <- inla.stack(
    data = list(y = ybar),
    A = list(A.spde, 1, 1),
    effects = list(idx.spde, list(Intercept = rep(1, n)), list(seaDist = seaDist.grp)),
    tag = "est"
)

t_rinla <- system.time({
    res <- inla(
        y ~ 0 + Intercept + f(seaDist, model = "rw1", hyper = rw1.hyper, scale.model = TRUE) + f(spatial, model = spde),
        data = inla.stack.data(stk),
        family = "gamma",
        # match Latte's phi ~ Gamma(shape=2, scale=5) == Gamma(shape=2, rate=0.2)
        control.family = list(hyper = list(prec = list(prior = "loggamma", param = c(2, 0.2)))),
        control.predictor = list(A = inla.stack.A(stk)),
        control.fixed = list(prec.intercept = p$prec_intercept),
        control.inla = list(int.strategy = "ccd")
    )
})[["elapsed"]]

## ── Shared mesh ──
write.csv(data.frame(node = seq_len(mesh$n), x = mesh$loc[, 1], y = mesh$loc[, 2]),
    file.path(output_dir, "nodes.csv"), row.names = FALSE)
tv <- mesh$graph$tv
write.csv(data.frame(tri = seq_len(nrow(tv)), v1 = tv[, 1], v2 = tv[, 2], v3 = tv[, 3]),
    file.path(output_dir, "triangles.csv"), row.names = FALSE)

## ── Shared data: response, coords, seaDist + group mapping ──
ugrp <- sort(unique(seaDist.grp))                          # rw1 node locations (group centres)
grp_idx <- match(seaDist.grp, ugrp)                        # station -> rw1 node (1-based)
write.csv(data.frame(s1 = coords[, 1], s2 = coords[, 2], y = ybar,
        seaDist = seaDist, grp = grp_idx),
    file.path(output_dir, "parana_data.csv"), row.names = FALSE)
write.csv(data.frame(group = seq_along(ugrp), seaDist = ugrp),
    file.path(output_dir, "rw1_groups.csv"), row.names = FALSE)

## ── Reference marginals ──
fm <- res$marginals.random$spatial
write.csv(do.call(rbind, lapply(seq_along(fm), function(i) data.frame(node = i, x = fm[[i]][, "x"], density = fm[[i]][, "y"]))),
    file.path(output_dir, "rinla_field_marginals.csv"), row.names = FALSE)
fs <- res$summary.random$spatial
write.csv(data.frame(node = seq_len(nrow(fs)), mean = fs$mean, sd = fs$sd),
    file.path(output_dir, "rinla_field_summary.csv"), row.names = FALSE)

rwm <- res$marginals.random$seaDist
write.csv(do.call(rbind, lapply(seq_along(rwm), function(i) data.frame(node = i, x = rwm[[i]][, "x"], density = rwm[[i]][, "y"]))),
    file.path(output_dir, "rinla_rw1_marginals.csv"), row.names = FALSE)
rws <- res$summary.random$seaDist
write.csv(data.frame(node = seq_len(nrow(rws)), mean = rws$mean, sd = rws$sd),
    file.path(output_dir, "rinla_rw1_summary.csv"), row.names = FALSE)

im <- res$marginals.fixed$Intercept
write.csv(data.frame(x = im[, "x"], density = im[, "y"]),
    file.path(output_dir, "rinla_intercept_marginal.csv"), row.names = FALSE)

# hyperparameters on interpretable scales
rgm <- res$marginals.hyperpar[["Range for spatial"]]
write.csv(data.frame(x = rgm[, "x"], density = rgm[, "y"]), file.path(output_dir, "rinla_range_marginal.csv"), row.names = FALSE)
sdm <- res$marginals.hyperpar[["Stdev for spatial"]]
write.csv(data.frame(x = sdm[, "x"], density = sdm[, "y"]), file.path(output_dir, "rinla_stdev_marginal.csv"), row.names = FALSE)
# rw1 precision -> sd
rwp <- res$marginals.hyperpar[["Precision for seaDist"]]
rwsd <- inla.tmarginal(function(x) 1 / sqrt(x), rwp)
write.csv(data.frame(x = rwsd[, "x"], density = rwsd[, "y"]), file.path(output_dir, "rinla_rw1_sd_marginal.csv"), row.names = FALSE)

meta <- list(
    n = n, n_nodes = mesh$n, n_triangles = nrow(tv), n_groups = length(ugrp),
    inla_version = as.character(packageVersion("INLA")),
    elapsed_seconds = as.numeric(res$cpu.used[["Total"]]),
    intercept_mean = res$summary.fixed["Intercept", "mean"],
    range_mean = res$summary.hyperpar["Range for spatial", "mean"],
    stdev_mean = res$summary.hyperpar["Stdev for spatial", "mean"],
    status = "ok"
)
writeLines(toJSON(meta, auto_unbox = TRUE, pretty = TRUE), file.path(output_dir, "rinla_meta.json"))
cat("done. range_mean =", round(meta$range_mean, 3), " stdev_mean =", round(meta$stdev_mean, 3),
    " intercept_mean =", round(meta$intercept_mean, 3), " elapsed =", round(t_rinla, 2), "s\n")
