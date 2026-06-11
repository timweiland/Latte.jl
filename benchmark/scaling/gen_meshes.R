# Generate a refinable sequence of shared SPDE meshes for the scaling benchmark.
# Each level: an observation grid + an inla.mesh.2d at decreasing max.edge, so the
# latent (mesh) dimension n grows. Dumps nodes/triangles/obs_coords per level —
# the SAME mesh every engine then solves on (Latte rebuilds via Ferrite, R-INLA
# native, Turing via the precision). Mesh generation only; no fitting here.
#
#   Rscript gen_meshes.R <output_dir>

suppressPackageStartupMessages({
    library(INLA)
    library(jsonlite)
})

argv <- commandArgs(trailingOnly = TRUE)
out <- if (length(argv) >= 1) argv[1] else "."
dir.create(out, showWarnings = FALSE, recursive = TRUE)

# Per level: observation-grid side, and the inner mesh max.edge. Both refine
# together so observations track the field as n grows.
gms <- c(20, 35, 55, 90, 140)
edges <- c(0.100, 0.055, 0.032, 0.018, 0.011)

levels <- list()
for (i in seq_along(gms)) {
    gm <- gms[i]
    me <- edges[i]
    g <- seq(0.03, 0.97, length.out = gm)
    coords <- as.matrix(expand.grid(s1 = g, s2 = g))
    mesh <- inla.mesh.2d(
        loc = coords,
        max.edge = c(me, me * 3),
        cutoff = me * 0.5,
        offset = c(me * 2, 0.2)
    )
    d <- file.path(out, sprintf("mesh_%d", i))
    dir.create(d, showWarnings = FALSE, recursive = TRUE)
    write.csv(
        data.frame(node = seq_len(mesh$n), x = mesh$loc[, 1], y = mesh$loc[, 2]),
        file.path(d, "nodes.csv"), row.names = FALSE
    )
    tv <- mesh$graph$tv
    write.csv(
        data.frame(tri = seq_len(nrow(tv)), v1 = tv[, 1], v2 = tv[, 2], v3 = tv[, 3]),
        file.path(d, "triangles.csv"), row.names = FALSE
    )
    write.csv(
        data.frame(s1 = coords[, 1], s2 = coords[, 2]),
        file.path(d, "obs_coords.csv"), row.names = FALSE
    )
    cat(sprintf("level %d: %d obs, max.edge=%.3f -> %d mesh nodes, %d triangles\n",
        i, nrow(coords), me, mesh$n, nrow(tv)))
    levels[[i]] <- list(level = i, m_obs = nrow(coords), max_edge = me, n_nodes = mesh$n)
}

writeLines(toJSON(levels, auto_unbox = TRUE, pretty = TRUE), file.path(out, "levels.json"))
cat("done.\n")
