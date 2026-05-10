# Fit BUGS Epil with R-INLA and dump fixed-effect / b_subject / τ_subj / τ_obs
# marginals for cross-validation against Latte's INLA.
#
# Inputs:
#   epil_data.csv    — y, Trt, Base, Age, V4, rand, Ind (R-INLA::data(Epil))
#   params.json      — { "pc_U", "pc_alpha", "strategy" }
#
# Outputs:
#   rinla_fixed_marginals.csv   — long form (name, x, density)
#   rinla_subj_marginals.csv    — long form (i, x, density), per-subject RE
#   rinla_tau_subj_marginal.csv — (x, density)
#   rinla_tau_obs_marginal.csv  — (x, density)
#   rinla_summary.csv           — name, mean, sd, q025, q5, q975
#   rinla_meta.json

suppressPackageStartupMessages({
    library(INLA)
    library(jsonlite)
})

argv <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(argv) >= 1) argv[1] else "."
output_dir <- if (length(argv) >= 2) argv[2] else input_dir

epil <- read.csv(file.path(input_dir, "epil_data.csv"))
params <- fromJSON(file.path(input_dir, "params.json"))
strategy <- params$strategy
n <- nrow(epil)
n_subj <- max(epil$Ind)

cat(sprintf("R-INLA epil: n = %d, n_subj = %d, strategy = %s\n", n, n_subj, strategy))

# Match Latte: log(Base/4), Trt, Trt × log(Base/4), log(Age), V4 + subject + obs RE.
epil$log_base4 <- log(epil$Base / 4)
epil$trt_logbase4 <- epil$Trt * epil$log_base4
epil$log_age <- log(epil$Age)
epil$obs_id <- seq_len(n)

formula <- y ~ 1 + log_base4 + Trt + trt_logbase4 + log_age + V4 +
    f(Ind, model = "iid",
        hyper = list(prec = list(prior = "pc.prec",
                                 param = c(params$pc_U, params$pc_alpha)))) +
    f(obs_id, model = "iid",
        hyper = list(prec = list(prior = "pc.prec",
                                 param = c(params$pc_U, params$pc_alpha))))

result <- inla(formula,
    data = epil,
    family = "poisson",
    control.fixed = list(prec = 1.0e-2, prec.intercept = 1.0e-2),
    control.inla = list(strategy = strategy, int.strategy = "grid"),
    control.compute = list(return.marginals.predictor = TRUE),
)

fixed_names <- names(result$marginals.fixed)
fixed_long <- do.call(rbind, lapply(fixed_names, function(nm) {
    m <- result$marginals.fixed[[nm]]
    data.frame(name = nm, x = m[, "x"], density = m[, "y"])
}))
write.csv(fixed_long, file.path(output_dir, "rinla_fixed_marginals.csv"), row.names = FALSE)

subj_long <- do.call(rbind, lapply(seq_len(n_subj), function(i) {
    m <- result$marginals.random$Ind[[i]]
    data.frame(i = i, x = m[, "x"], density = m[, "y"])
}))
write.csv(subj_long, file.path(output_dir, "rinla_subj_marginals.csv"), row.names = FALSE)

tau_subj <- result$marginals.hyperpar$`Precision for Ind`
write.csv(data.frame(x = tau_subj[, "x"], density = tau_subj[, "y"]),
    file.path(output_dir, "rinla_tau_subj_marginal.csv"), row.names = FALSE)

tau_obs <- result$marginals.hyperpar$`Precision for obs_id`
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
write.csv(do.call(rbind, summary_rows),
    file.path(output_dir, "rinla_summary.csv"), row.names = FALSE)

meta <- list(
    strategy = strategy, n = n, n_subj = n_subj,
    inla_version = as.character(packageVersion("INLA")),
    elapsed_seconds = result$cpu.used[["Total"]],
    status = "ok"
)
writeLines(toJSON(meta, auto_unbox = TRUE, pretty = TRUE),
    file.path(output_dir, "rinla_meta.json"))

cat("done.\n")
