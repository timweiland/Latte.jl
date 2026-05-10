# Fit Scottish lip cancer with R-INLA and dump fixed-effect / u / τ
# marginals for cross-validation against Latte's INLA.
#
# Inputs:
#   scotland_data.csv   — Counts, E, X, Region (from R-INLA::data(Scotland))
#   scotland_edges.csv  — i, j (undirected adjacency edges, 1-indexed)
#   params.json         — { "pc_U", "pc_alpha", "strategy" }
#
# Outputs:
#   rinla_fixed_marginals.csv  — long form (name, x, density)
#   rinla_u_marginals.csv      — long form (i, x, density)
#   rinla_tau_marginal.csv     — (x, density)
#   rinla_summary.csv          — name, mean, sd, q025, q5, q975
#   rinla_meta.json

suppressPackageStartupMessages({
    library(INLA)
    library(jsonlite)
})

argv <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(argv) >= 1) argv[1] else "."
output_dir <- if (length(argv) >= 2) argv[2] else input_dir

scot <- read.csv(file.path(input_dir, "scotland_data.csv"))
edges <- read.csv(file.path(input_dir, "scotland_edges.csv"))
params <- fromJSON(file.path(input_dir, "params.json"))
strategy <- params$strategy
n <- nrow(scot)

cat(sprintf("R-INLA scotland: n = %d, edges = %d, strategy = %s\n",
    n, nrow(edges), strategy))

# Build INLA graph from the symmetric edge list.
graph_path <- file.path(output_dir, "scotland.graph")
{
    nbrs <- vector("list", n)
    for (i in seq_len(n)) nbrs[[i]] <- integer(0)
    for (k in seq_len(nrow(edges))) {
        a <- edges$i[k]; b <- edges$j[k]
        nbrs[[a]] <- c(nbrs[[a]], b)
        nbrs[[b]] <- c(nbrs[[b]], a)
    }
    con <- file(graph_path, "w")
    writeLines(as.character(n), con)
    for (i in seq_len(n)) {
        writeLines(paste(c(i, length(nbrs[[i]]), nbrs[[i]]), collapse = " "), con)
    }
    close(con)
}

# Match the Latte parameterisation: log-rate = α + β·(X/10) + u, with
# log(E) as offset. We pass `X / 10` so the fixed-effect prior has the
# same scale as Latte's `fixed[2]`.
scot$x_scaled <- scot$X / 10

formula <- Counts ~ 1 + x_scaled + f(Region,
    model = "besag",
    graph = graph_path,
    scale.model = FALSE,
    hyper = list(prec = list(prior = "pc.prec",
                             param = c(params$pc_U, params$pc_alpha)))
)

result <- inla(formula,
    data = scot,
    family = "poisson",
    E = scot$E,
    control.fixed = list(prec = 1.0e-2, prec.intercept = 1.0e-2),
    control.inla = list(strategy = strategy, int.strategy = "grid"),
    control.compute = list(return.marginals.predictor = TRUE),
)

# Fixed-effect marginals
fixed_names <- names(result$marginals.fixed)
fixed_long <- do.call(rbind, lapply(fixed_names, function(nm) {
    m <- result$marginals.fixed[[nm]]
    data.frame(name = nm, x = m[, "x"], density = m[, "y"])
}))
write.csv(fixed_long, file.path(output_dir, "rinla_fixed_marginals.csv"), row.names = FALSE)

# Spatial random effect u
u_long <- do.call(rbind, lapply(seq_len(n), function(i) {
    m <- result$marginals.random$Region[[i]]
    data.frame(i = i, x = m[, "x"], density = m[, "y"])
}))
write.csv(u_long, file.path(output_dir, "rinla_u_marginals.csv"), row.names = FALSE)

# τ marginal
tau_marg <- result$marginals.hyperpar$`Precision for Region`
tau_df <- data.frame(x = tau_marg[, "x"], density = tau_marg[, "y"])
write.csv(tau_df, file.path(output_dir, "rinla_tau_marginal.csv"), row.names = FALSE)

# Summaries
summary_rows <- list()
for (nm in fixed_names) {
    s <- result$summary.fixed[nm, ]
    summary_rows[[length(summary_rows) + 1]] <- data.frame(
        name = nm, mean = s$mean, sd = s$sd,
        q025 = s$`0.025quant`, q5 = s$`0.5quant`, q975 = s$`0.975quant`
    )
}
for (i in seq_len(n)) {
    s <- result$summary.random$Region[i, ]
    summary_rows[[length(summary_rows) + 1]] <- data.frame(
        name = sprintf("u[%d]", i), mean = s$mean, sd = s$sd,
        q025 = s$`0.025quant`, q5 = s$`0.5quant`, q975 = s$`0.975quant`
    )
}
write.csv(do.call(rbind, summary_rows),
    file.path(output_dir, "rinla_summary.csv"), row.names = FALSE)

meta <- list(
    strategy = strategy, n = n,
    inla_version = as.character(packageVersion("INLA")),
    elapsed_seconds = result$cpu.used[["Total"]],
    status = "ok"
)
writeLines(toJSON(meta, auto_unbox = TRUE, pretty = TRUE),
    file.path(output_dir, "rinla_meta.json"))

cat("done.\n")
