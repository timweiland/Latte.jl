# R-INLA side of the scaling benchmark. For each mesh level: regenerate the
# IDENTICAL inla.mesh.2d (same args as gen_meshes.R → same nodes Latte rebuilt),
# read the shared y.csv (written by scaling_latte.jl), fit the Poisson SPDE with
# priors matched to the Latte model, and record cpu.used. Also dumps the field
# posterior mean/sd per node for the cross-engine accuracy-at-scale check.
#
#   Rscript scaling_rinla.R <workdir>   (run AFTER scaling_latte.jl writes y.csv)

suppressPackageStartupMessages({
    library(INLA)
    library(jsonlite)
})

argv <- commandArgs(trailingOnly = TRUE)
out <- if (length(argv) >= 1) argv[1] else "."
levels <- fromJSON(file.path(out, "levels.json"))

results <- list()
for (i in seq_len(nrow(levels))) {
    level <- levels$level[i]
    me <- levels$max_edge[i]
    d <- file.path(out, sprintf("mesh_%d", level))
    yf <- file.path(d, "y.csv")
    if (!file.exists(yf)) {
        cat(sprintf("level %d: no y.csv yet, skipping\n", level))
        next
    }
    obs <- read.csv(file.path(d, "obs_coords.csv"))
    coords <- as.matrix(obs[, c("s1", "s2")])
    y <- read.csv(yf)$y

    # Reconstruct the EXACT shared mesh from the dumped nodes/triangles (the
    # source of truth Latte rebuilt from). Regenerating via inla.mesh.2d is not
    # bit-reproducible after the CSV round-trip, so build the mesh explicitly.
    nodes <- read.csv(file.path(d, "nodes.csv"))
    tris <- read.csv(file.path(d, "triangles.csv"))
    mesh <- inla.mesh.create(loc = as.matrix(nodes[, c("x", "y")]), tv = as.matrix(tris[, c("v1", "v2", "v3")]))
    stopifnot(mesh$n == nrow(nodes))   # identical to the mesh Latte solved on

    # Priors matched to the Latte model: PC range c(0.3, 0.5), PC sigma c(1.0, 0.5).
    spde <- inla.spde2.pcmatern(mesh, alpha = 2, prior.range = c(0.3, 0.5), prior.sigma = c(1.0, 0.5))
    A <- inla.spde.make.A(mesh, loc = coords)
    idx <- inla.spde.make.index("spatial", spde$n.spde)
    stk <- inla.stack(
        data = list(y = y), A = list(A, 1),
        effects = list(idx, list(Intercept = rep(1, length(y)))), tag = "est"
    )

    el <- system.time({
        res <- inla(
            y ~ 0 + Intercept + f(spatial, model = spde),
            data = inla.stack.data(stk), family = "poisson",
            control.predictor = list(A = inla.stack.A(stk)),
            control.fixed = list(prec.intercept = 0.01),
            control.inla = list(int.strategy = "grid")
        )
    })[["elapsed"]]

    fs <- res$summary.random$spatial
    write.csv(
        data.frame(node = seq_len(nrow(fs)), mean = fs$mean, sd = fs$sd),
        file.path(d, "rinla_field_summary.csv"), row.names = FALSE
    )
    cpu <- as.numeric(res$cpu.used[["Total"]])
    results[[length(results) + 1]] <- list(
        level = level, n_nodes = mesh$n, m_obs = length(y),
        cpu_total = cpu, elapsed_wall = as.numeric(el)
    )
    cat(sprintf("level %d: n=%d, cpu=%.2fs, wall=%.2fs\n", level, mesh$n, cpu, el))
}

writeLines(
    toJSON(list(inla_version = as.character(packageVersion("INLA")), levels = results),
        auto_unbox = TRUE, pretty = TRUE),
    file.path(out, "rinla_scaling.json")
)
cat("done.\n")
