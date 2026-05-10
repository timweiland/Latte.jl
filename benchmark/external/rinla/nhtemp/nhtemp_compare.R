# Fit New Haven temperature with R-INLA and dump fixed-effect / x / τ_x / τ_obs
# marginals for cross-validation against Latte's INLA.
#
# Inputs:
#   nhtemp_data.csv   — year, temp
#   params.json       — { "pc_U_x", "pc_alpha_x", "pc_U_obs", "pc_alpha_obs", "strategy" }
#
# Outputs:
#   rinla_fixed_marginals.csv   — long form (name, x, density)
#   rinla_x_marginals.csv       — long form (i, x, density)
#   rinla_tau_x_marginal.csv    — (x, density) — RW2 precision
#   rinla_tau_obs_marginal.csv  — (x, density) — observation precision
#   rinla_summary.csv           — name, mean, sd, q025, q5, q975
#   rinla_meta.json

suppressPackageStartupMessages({
    library(INLA)
    library(jsonlite)
})

argv <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(argv) >= 1) argv[1] else "."
output_dir <- if (length(argv) >= 2) argv[2] else input_dir

nh <- read.csv(file.path(input_dir, "nhtemp_data.csv"))
params <- fromJSON(file.path(input_dir, "params.json"))
strategy <- params$strategy
n <- nrow(nh)

cat(sprintf("R-INLA nhtemp: n = %d, strategy = %s\n", n, strategy))

# Match Latte: intercept α + RW2(year) + Gaussian noise. Year index passed
# 1..n so the RW2 sits on integer time steps (Latte's RW2Model is on
# integer indices; the actual calendar year doesn't enter the prior).
nh$idx <- seq_len(n)

formula <- temp ~ 1 + f(idx,
    model = "rw2",
    scale.model = FALSE,
    hyper = list(prec = list(prior = "pc.prec",
                             param = c(params$pc_U_x, params$pc_alpha_x)))
)

result <- inla(formula,
    data = nh,
    family = "gaussian",
    control.family = list(
        hyper = list(prec = list(prior = "pc.prec",
                                 param = c(params$pc_U_obs, params$pc_alpha_obs)))
    ),
    control.fixed = list(prec = 1.0e-4, prec.intercept = 1.0e-4),
    control.inla = list(strategy = strategy, int.strategy = "grid"),
    control.compute = list(return.marginals.predictor = TRUE),
)

fixed_names <- names(result$marginals.fixed)
fixed_long <- do.call(rbind, lapply(fixed_names, function(nm) {
    m <- result$marginals.fixed[[nm]]
    data.frame(name = nm, x = m[, "x"], density = m[, "y"])
}))
write.csv(fixed_long, file.path(output_dir, "rinla_fixed_marginals.csv"), row.names = FALSE)

x_long <- do.call(rbind, lapply(seq_len(n), function(i) {
    m <- result$marginals.random$idx[[i]]
    data.frame(i = i, x = m[, "x"], density = m[, "y"])
}))
write.csv(x_long, file.path(output_dir, "rinla_x_marginals.csv"), row.names = FALSE)

tau_x <- result$marginals.hyperpar$`Precision for idx`
write.csv(data.frame(x = tau_x[, "x"], density = tau_x[, "y"]),
    file.path(output_dir, "rinla_tau_x_marginal.csv"), row.names = FALSE)

tau_obs <- result$marginals.hyperpar$`Precision for the Gaussian observations`
write.csv(data.frame(x = tau_obs[, "x"], density = tau_obs[, "y"]),
    file.path(output_dir, "rinla_tau_obs_marginal.csv"), row.names = FALSE)

summary_rows <- list()
for (nm in fixed_names) {
    s <- result$summary.fixed[nm, ]
    summary_rows[[length(summary_rows) + 1]] <- data.frame(
        name = nm, mean = s$mean, sd = s$sd,
        q025 = s$`0.025quant`, q5 = s$`0.5quant`, q975 = s$`0.975quant`
    )
}
for (i in seq_len(n)) {
    s <- result$summary.random$idx[i, ]
    summary_rows[[length(summary_rows) + 1]] <- data.frame(
        name = sprintf("x[%d]", i), mean = s$mean, sd = s$sd,
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
