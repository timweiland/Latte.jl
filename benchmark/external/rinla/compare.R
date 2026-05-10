# Fit toy_iid_poisson with R-INLA and dump posterior marginals.
#
# Inputs (read via getOption / args):
#   data.csv      — single column `y` (the observation vector)
#   params.json   — `{ "pc_U": 1.0, "pc_alpha": 0.01, "strategy": "simplified.laplace" }`
#
# Outputs:
#   rinla_x_marginals.csv  — long form (i, x, density) for each x[i]
#   rinla_tau_marginal.csv — (x, density) for τ
#   rinla_summary.csv      — per-parameter (name, mean, sd, q025, q5, q975)
#   rinla_meta.json        — strategy, n, INLA version, status

suppressPackageStartupMessages({
    library(INLA)
    library(jsonlite)
})

argv <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(argv) >= 1) argv[1] else "."
output_dir <- if (length(argv) >= 2) argv[2] else input_dir

data <- read.csv(file.path(input_dir, "data.csv"))
params <- fromJSON(file.path(input_dir, "params.json"))
y <- data$y
n <- length(y)
strategy <- params$strategy

cat(sprintf("R-INLA fit: n = %d, strategy = %s, pc_U = %g, pc_alpha = %g\n",
    n, strategy, params$pc_U, params$pc_alpha))

# Model: y_i ~ Poisson(exp(x_i)),  x_i ~ N(0, 1/τ) iid,
# τ ~ PCPrior.Precision(U = pc_U, α = pc_alpha).
#
# Use `f(idx, model = "iid")` with a `pc.prec` hyperparameter prior.
# `-1` removes the implicit intercept so the latent is just x.
df <- data.frame(y = y, idx = 1:n)
formula <- y ~ -1 + f(idx,
    model = "iid",
    hyper = list(prec = list(prior = "pc.prec",
                             param = c(params$pc_U, params$pc_alpha)))
)

result <- inla(formula,
    data = df,
    family = "poisson",
    control.inla = list(strategy = strategy),
    control.compute = list(return.marginals.predictor = TRUE),
)

# ── x[i] marginals ────────────────────────────────────────────────────
x_long <- do.call(rbind, lapply(seq_len(n), function(i) {
    m <- result$marginals.random$idx[[i]]
    data.frame(i = i, x = m[, "x"], density = m[, "y"])
}))
write.csv(x_long, file.path(output_dir, "rinla_x_marginals.csv"), row.names = FALSE)

# ── τ marginal ────────────────────────────────────────────────────────
tau_marg <- result$marginals.hyperpar$`Precision for idx`
tau_df <- data.frame(x = tau_marg[, "x"], density = tau_marg[, "y"])
write.csv(tau_df, file.path(output_dir, "rinla_tau_marginal.csv"), row.names = FALSE)

# ── per-parameter summary ─────────────────────────────────────────────
summary_rows <- list()
for (i in seq_len(n)) {
    s <- result$summary.random$idx[i, ]
    summary_rows[[length(summary_rows) + 1L]] <- data.frame(
        name = sprintf("x[%d]", i),
        mean = s$mean, sd = s$sd,
        q025 = s$`0.025quant`, q5 = s$`0.5quant`, q975 = s$`0.975quant`
    )
}
s_tau <- result$summary.hyperpar["Precision for idx", ]
summary_rows[[length(summary_rows) + 1L]] <- data.frame(
    name = "tau", mean = s_tau$mean, sd = s_tau$sd,
    q025 = s_tau$`0.025quant`, q5 = s_tau$`0.5quant`, q975 = s_tau$`0.975quant`
)
write.csv(do.call(rbind, summary_rows),
    file.path(output_dir, "rinla_summary.csv"), row.names = FALSE)

# ── metadata ──────────────────────────────────────────────────────────
meta <- list(
    strategy = strategy,
    n = n,
    inla_version = as.character(packageVersion("INLA")),
    elapsed_seconds = result$cpu.used[["Total"]],
    status = "ok"
)
writeLines(toJSON(meta, auto_unbox = TRUE, pretty = TRUE),
    file.path(output_dir, "rinla_meta.json"))

cat("done.\n")
