# Fit additive_iid_poisson with R-INLA and dump linear-predictor
# marginals.
#
# Model: y_i ~ Poisson(exp(β + x_i)) where β ~ N(0,1) (fixed) and
# x_i ~ N(0, 1/τ) iid with τ ~ PCPrior.Precision(1, α=0.01).
#
# Outputs a long-format CSV of (i, x, density) for each linear
# predictor η_i = β + x_i.

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

cat(sprintf("R-INLA additive: n = %d, strategy = %s\n", n, strategy))

df <- data.frame(y = y, idx = 1:n)
formula <- y ~ 1 + f(idx,
    model = "iid",
    hyper = list(prec = list(prior = "pc.prec",
                             param = c(params$pc_U, params$pc_alpha)))
)

result <- inla(formula,
    data = df,
    family = "poisson",
    control.fixed = list(prec.intercept = 1.0),
    control.inla = list(strategy = strategy),
    control.predictor = list(compute = TRUE),
    control.compute = list(return.marginals.predictor = TRUE),
)

# η_i = (intercept) + f(idx_i): marginals.linear.predictor.
eta_long <- do.call(rbind, lapply(seq_len(n), function(i) {
    m <- result$marginals.linear.predictor[[i]]
    data.frame(i = i, x = m[, "x"], density = m[, "y"])
}))
write.csv(eta_long, file.path(output_dir, "rinla_eta_marginals.csv"), row.names = FALSE)

# Per-η_i summary for sanity checks.
eta_summary <- do.call(rbind, lapply(seq_len(n), function(i) {
    s <- result$summary.linear.predictor[i, ]
    data.frame(
        name = sprintf("eta[%d]", i),
        mean = s$mean, sd = s$sd,
        q025 = s$`0.025quant`, q5 = s$`0.5quant`, q975 = s$`0.975quant`
    )
}))
write.csv(eta_summary, file.path(output_dir, "rinla_eta_summary.csv"), row.names = FALSE)

# Intercept β: marginal density + summary, so the Julia driver can
# verify the priors agree across implementations.
beta_marg <- result$marginals.fixed$`(Intercept)`
beta_df <- data.frame(x = beta_marg[, "x"], density = beta_marg[, "y"])
write.csv(beta_df, file.path(output_dir, "rinla_beta_marginal.csv"), row.names = FALSE)

beta_s <- result$summary.fixed["(Intercept)", ]
beta_summary <- data.frame(
    name = "(Intercept)",
    mean = beta_s$mean, sd = beta_s$sd,
    q025 = beta_s$`0.025quant`, q5 = beta_s$`0.5quant`, q975 = beta_s$`0.975quant`
)
write.csv(beta_summary, file.path(output_dir, "rinla_beta_summary.csv"), row.names = FALSE)

meta <- list(
    strategy = strategy, n = n,
    inla_version = as.character(packageVersion("INLA")),
    elapsed_seconds = result$cpu.used[["Total"]],
    status = "ok"
)
writeLines(toJSON(meta, auto_unbox = TRUE, pretty = TRUE),
    file.path(output_dir, "rinla_meta.json"))

cat("done.\n")
