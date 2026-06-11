# Full R-INLA scaling curve: fit time vs n across all mesh levels, parallel
# (num.threads "10:1") and forced serial ("1:1"), int.strategy="grid", priors
# matched to the Latte model. Reconstructs the EXACT shared mesh from the dumped
# nodes/triangles. Persists per level.
#   Rscript benchmark/scaling/rinla_curve.R
suppressPackageStartupMessages({ library(INLA); library(jsonlite) })
out <- "benchmark/scaling/_workdir"
levels <- fromJSON(file.path(out, "levels.json"))

fit <- function(d, nthreads) {
  obs <- read.csv(file.path(d, "obs_coords.csv")); coords <- as.matrix(obs[, c("s1","s2")])
  y <- read.csv(file.path(d, "y.csv"))$y
  nodes <- read.csv(file.path(d,"nodes.csv")); tris <- read.csv(file.path(d,"triangles.csv"))
  mesh <- inla.mesh.create(loc=as.matrix(nodes[,c("x","y")]), tv=as.matrix(tris[,c("v1","v2","v3")]))
  spde <- inla.spde2.pcmatern(mesh, alpha=2, prior.range=c(0.3,0.5), prior.sigma=c(1.0,0.5))
  A <- inla.spde.make.A(mesh, loc=coords); idx <- inla.spde.make.index("spatial", spde$n.spde)
  stk <- inla.stack(data=list(y=y), A=list(A,1),
    effects=list(idx, list(Intercept=rep(1,length(y)))), tag="est")
  inla.setOption(num.threads = nthreads)
  el <- system.time({
    res <- inla(y ~ 0 + Intercept + f(spatial, model=spde), data=inla.stack.data(stk),
      family="poisson", control.predictor=list(A=inla.stack.A(stk)),
      control.fixed=list(prec.intercept=0.01), control.inla=list(int.strategy="grid"))
  })[["elapsed"]]
  list(wall=as.numeric(el), cpu=as.numeric(res$cpu.used[["Total"]]), n=mesh$n)
}

results <- list()
for (i in seq_len(nrow(levels))) {
  lev <- levels$level[i]; d <- file.path(out, sprintf("mesh_%d", lev))
  if (!file.exists(file.path(d, "y.csv"))) next
  par <- fit(d, "10:1"); ser <- fit(d, "1:1")
  results[[length(results)+1]] <- list(level=lev, n=par$n,
    parallel_s=par$wall, serial_s=ser$wall)
  cat(sprintf("L%d  n=%6d  serial=%7.2fs  parallel=%7.2fs  self-speedup=%.2fx\n",
      lev, par$n, ser$wall, par$wall, ser$wall/par$wall))
  writeLines(toJSON(list(inla_version=as.character(packageVersion("INLA")), levels=results),
    auto_unbox=TRUE, pretty=TRUE), file.path(out, "rinla_curve.json"))
}
cat("done -> _workdir/rinla_curve.json\n")
