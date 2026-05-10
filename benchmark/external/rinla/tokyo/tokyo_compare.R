# Fit Tokyo rainfall with R-INLA and dump per-day x_t marginals + τ
# marginal for cross-validation against Latte's INLA.
#
# Inputs:
#   tokyo_data.csv    — y, n, time
#   params.json       — { "pc_U", "pc_alpha", "strategy" }
#
# Outputs:
#   rinla_x_marginals.csv  — long form (i, x, density)
#   rinla_tau_marginal.csv — (x, density)
#   rinla_x_summary.csv    — per-day mean / sd / quantiles
#   rinla_meta.json

suppressPackageStartupMessages({
    library(INLA)
    library(jsonlite)
})

argv <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(argv) >= 1) argv[1] else "."
output_dir <- if (length(argv) >= 2) argv[2] else input_dir

tokyo <- read.csv(file.path(input_dir, "tokyo_data.csv"))
params <- fromJSON(file.path(input_dir, "params.json"))
strategy <- params$strategy
n <- nrow(tokyo)

cat(sprintf("R-INLA Tokyo: n = %d, strategy = %s\n", n, strategy))

formula <- y ~ -1 + f(time,
    model = "rw2",
    scale.model = FALSE,
    hyper = list(prec = list(prior = "pc.prec",
                             param = c(params$pc_U, params$pc_alpha)))
)

result <- inla(formula,
    data = tokyo,
    family = "binomial",
    Ntrials = tokyo$n,
    control.inla = list(strategy = strategy, int.strategy = "grid"),
    control.compute = list(return.marginals.predictor = TRUE),
)

# x[t] marginals (R-INLA's latent random effect for `time` term).
x_long <- do.call(rbind, lapply(seq_len(n), function(i) {
    m <- result$marginals.random$time[[i]]
    data.frame(i = i, x = m[, "x"], density = m[, "y"])
}))
write.csv(x_long, file.path(output_dir, "rinla_x_marginals.csv"), row.names = FALSE)

# τ marginal (precision of the RW2 increments).
tau_marg <- result$marginals.hyperpar$`Precision for time`
tau_df <- data.frame(x = tau_marg[, "x"], density = tau_marg[, "y"])
write.csv(tau_df, file.path(output_dir, "rinla_tau_marginal.csv"), row.names = FALSE)

# Per-x summary.
x_summary <- do.call(rbind, lapply(seq_len(n), function(i) {
    s <- result$summary.random$time[i, ]
    data.frame(
        name = sprintf("x[%d]", i),
        mean = s$mean, sd = s$sd,
        q025 = s$`0.025quant`, q5 = s$`0.5quant`, q975 = s$`0.975quant`
    )
}))
write.csv(x_summary, file.path(output_dir, "rinla_x_summary.csv"), row.names = FALSE)

meta <- list(
    strategy = strategy, n = n,
    inla_version = as.character(packageVersion("INLA")),
    elapsed_seconds = result$cpu.used[["Total"]],
    status = "ok"
)
writeLines(toJSON(meta, auto_unbox = TRUE, pretty = TRUE),
    file.path(output_dir, "rinla_meta.json"))

cat("done.\n")
