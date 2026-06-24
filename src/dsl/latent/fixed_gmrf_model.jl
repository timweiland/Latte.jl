# A fixed (hyperparameter-independent) GMRF latent prior.
#
# Some latent fields have a Gaussian prior that is specified up front and does
# not depend on any of the model's hyperparameters. `@latte` recognizes such a
# `@random x ~ g`, where `g` is a runtime GMRF value, as a hyperparameter-free
# fixed prior; `_coerce_latent` then wraps `g` in this `LatentModel` so the rest
# of the pipeline — which speaks the `LatentModel` contract — sees a constant
# prior that carries no hyperparameters.
#
# This adapter lives in Latte, not GMRFs: it is glue created by recognition, not
# a modeling primitive a GMRFs user would reach for, and users never name it.
# Latte owns the type, so defining methods on GMRFs' `LatentModel` generics for
# it is not piracy.

struct FixedGMRFModel{G <: GaussianMarkovRandomFields.AbstractGMRF} <: LatentModel
    gmrf::G
end

# Unwrap a constraint layer to reach the unconstrained base prior. The
# `LatentModel` contract reports mean / precision / constraint separately, so
# mean and precision must come from the base prior (a `ConstrainedGMRF` reports
# the *constrained* mean from `mean`), with the constraint reported on its own.
_base_gmrf(g::GaussianMarkovRandomFields.AbstractGMRF) = g
_base_gmrf(g::GaussianMarkovRandomFields.ConstrainedGMRF) = g.base_gmrf

_fixed_gmrf_constraints(::GaussianMarkovRandomFields.AbstractGMRF) = nothing
_fixed_gmrf_constraints(g::GaussianMarkovRandomFields.ConstrainedGMRF) =
    (g.constraint_matrix, g.constraint_vector)

Base.length(m::FixedGMRFModel) = length(m.gmrf)

GaussianMarkovRandomFields.hyperparameters(::FixedGMRFModel) = NamedTuple()
GaussianMarkovRandomFields.model_name(::FixedGMRFModel) = :fixed_gmrf

# Base mean, base precision, and any linear-equality constraint are read off the
# wrapped GMRF and are hyperparameter-independent. None of these densify (only
# `cov`/`var` do), so a fixed GMRF prior never trips the dense-covariance guard.
GaussianMarkovRandomFields.mean(m::FixedGMRFModel; kwargs...) = GaussianMarkovRandomFields.mean(_base_gmrf(m.gmrf))
GaussianMarkovRandomFields.precision_matrix(m::FixedGMRFModel; kwargs...) = GaussianMarkovRandomFields.precision_matrix(_base_gmrf(m.gmrf))
GaussianMarkovRandomFields.constraints(m::FixedGMRFModel; kwargs...) = _fixed_gmrf_constraints(m.gmrf)

# The default `(::LatentModel)(; θ...)` rebuilds a GMRF from mean/precision/
# constraints and needs an `alg` field this model does not have. Return the
# already-materialized wrapped GMRF instead — this also preserves its concrete
# type and any metadata it carries.
(m::FixedGMRFModel)(; kwargs...) = m.gmrf
