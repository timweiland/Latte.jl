# Tokyo Model B: explicit intercept + slope + rw2 with BOTH null-space
# constraints (sum-to-zero AND linear). Mirrors Latte's stock RW2Model
# (which imposes both constraints on the random effect) plus diffuse
# fixed effects for intercept and slope.
#
# This is a different model spec from `tokyo_compare.R` — Model A there
# matches R-INLA's default rw2 (one constraint). Model B here adds the
# linear constraint and explicit fixed effects so the linear predictor's
# overall offset and trend are absorbed by α + β·t rather than baked
# into the random effect.

suppressPackageStartupMessages({
    library(INLA)
    library(jsonlite)
})

argv <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(argv) >= 1) argv[1] else "."
output_dir <- if (length(argv) >= 2) argv[2] else input_dir

tokyo <- read.csv(file.path(input_dir, "tokyo_data.csv"))
# R-INLA can't use the same column as both fixed-effect covariate and f() index.
tokyo$time_idx <- tokyo$time
params <- fromJSON(file.path(input_dir, "params.json"))
strategy <- params$strategy
n <- nrow(tokyo)

cat(sprintf("R-INLA Tokyo Model B: n = %d, strategy = %s\n", n, strategy))

# `1 + time` adds intercept (default vague Normal prior) and slope as fixed
# effects. `extraconstr = list(A=seq_len(n), e=0)` adds the linear-trend-zero
# constraint on top of the default sum-to-zero — matches Latte's RW2Model.
linear_A <- matrix(seq_len(n), nrow = 1)
formula <- y ~ 1 + time + f(time_idx,
    model = "rw2",
    scale.model = FALSE,
    constr = TRUE,
    extraconstr = list(A = linear_A, e = 0),
    hyper = list(prec = list(prior = "pc.prec",
                             param = c(params$pc_U, params$pc_alpha)))
)

result <- inla(formula,
    data = tokyo,
    family = "binomial",
    Ntrials = tokyo$n,
    control.inla = list(strategy = strategy, int.strategy = "grid"),
    control.compute = list(return.marginals.predictor = TRUE),
    control.predictor = list(compute = TRUE),
)

# Linear predictor marginals η_t = α + β·t + x_t — that's what we compare
# against Latte Model B's `linear_predictor_marginals` (or moments-built η_t).
eta_long <- do.call(rbind, lapply(seq_len(n), function(i) {
    m <- result$marginals.linear.predictor[[i]]
    data.frame(i = i, x = m[, "x"], density = m[, "y"])
}))
write.csv(eta_long, file.path(output_dir, "rinla_eta_marginals.csv"), row.names = FALSE)

eta_summary <- do.call(rbind, lapply(seq_len(n), function(i) {
    s <- result$summary.linear.predictor[i, ]
    data.frame(
        name = sprintf("eta[%d]", i),
        mean = s$mean, sd = s$sd,
        q025 = s$`0.025quant`, q5 = s$`0.5quant`, q975 = s$`0.975quant`
    )
}))
write.csv(eta_summary, file.path(output_dir, "rinla_eta_summary.csv"), row.names = FALSE)

# Fixed effects (intercept and slope) for sanity.
fixed_summary <- result$summary.fixed
write.csv(fixed_summary, file.path(output_dir, "rinla_fixed_summary.csv"))

tau_marg <- result$marginals.hyperpar$`Precision for time_idx`
tau_df <- data.frame(x = tau_marg[, "x"], density = tau_marg[, "y"])
write.csv(tau_df, file.path(output_dir, "rinla_tau_marginal_b.csv"), row.names = FALSE)

meta <- list(
    strategy = strategy, n = n, model = "B (intercept+slope+two_constraints)",
    inla_version = as.character(packageVersion("INLA")),
    elapsed_seconds = result$cpu.used[["Total"]],
    status = "ok"
)
writeLines(toJSON(meta, auto_unbox = TRUE, pretty = TRUE),
    file.path(output_dir, "rinla_meta_b.json"))

cat("done.\n")
