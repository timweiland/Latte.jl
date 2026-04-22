module Latte

# Import observation model functionality from GaussianMarkovRandomFields.jl v0.4+
using GaussianMarkovRandomFields: ObservationModel, ObservationLikelihood,
    ExponentialFamily, BinomialObservations, PoissonObservations,
    LinearlyTransformedObservationModel,
    CompositeObservationModel, loglik, loggrad, loghessian, hyperparameters,
    latent_dimension, gaussian_approximation, successes, trials,
    IdentityLink, LogLink, LogitLink, LinkFunction,
    conditional_distribution, apply_link, apply_invlink,
    pointwise_loglik, NormalLikelihood,
    PoissonLikelihood, BernoulliLikelihood, BinomialLikelihood

# Re-export observation model types and functions for user convenience
export ObservationModel, ObservationLikelihood, ExponentialFamily,
    BinomialObservations, PoissonObservations, LinearlyTransformedObservationModel,
    CompositeObservationModel, gaussian_approximation,
    loglik, loggrad, loghessian, hyperparameters, latent_dimension,
    successes, trials, IdentityLink, LogLink, LogitLink, LinkFunction,
    conditional_distribution, apply_link, apply_invlink,
    pointwise_loglik

# Hyperparameter marginalization exports (were previously in an orchestrator file)
export HyperparameterMarginalizationMethod, marginalize_hyperparameters

# ─── Infrastructure (no inter-deps) ──────────────────────────────────────────
include("parallel/parallel.jl")
include("differentiation/differentiation.jl")

include("utils/selinv.jl")
include("utils/owens_t.jl")
include("utils/kld.jl")
include("utils/distribution_summaries.jl")
include("utils/plotting_stubs.jl")

include("distributions/distributions.jl")

# ─── Model layer (types used throughout) ─────────────────────────────────────
include("model/hyperparameter.jl")
include("model/hyperparameter_spec.jl")
include("model/working_and_natural.jl")
include("model/logpdf.jl")
include("model/hyperparams_macro.jl")

include("model/loghessian_derivatives.jl")
include("model/link_to_bijector.jl")
include("model/augmentation_info.jl")
include("model/augmented_latent_model.jl")
include("model/offset_observation_model.jl")
include("model/latent_gaussian_model.jl")

# ─── InferenceResult protocol (abstract supertype + Tier 1 methods) ──────────
# Declared early so both INLAResult and TMBResult can subtype it.
include("posterior/result_protocol.jl")

# ─── Laplace approximation (shared inner machinery) ──────────────────────────
include("laplace/types.jl")
include("laplace/mode_finding.jl")
include("laplace/gaussian_marginal.jl")
include("laplace/spline_augmented_gaussian.jl")
include("laplace/laplace_cache.jl")
include("laplace/laplace_marginal.jl")
include("laplace/simplified_laplace.jl")
include("laplace/marginalize.jl")

# ─── INLA inference (grid / CCD over θ) ──────────────────────────────────────
include("inference/inla/exploration/adaptive_hessian.jl")
include("inference/inla/exploration/transformation.jl")
include("inference/inla/exploration/types.jl")
include("inference/inla/exploration/utils.jl")
include("inference/inla/exploration/grid.jl")
include("inference/inla/exploration/ccd.jl")
include("inference/inla/exploration/auto_strategy.jl")

include("inference/inla/interpolation.jl")
include("inference/inla/spline_marginal_builders.jl")

include("inference/inla/hp_marginals/types.jl")
include("inference/inla/hp_marginals/spline_based_types.jl")
include("inference/inla/hp_marginals/gridsum_marginal.jl")
include("inference/inla/hp_marginals/ccd_interpolant_marginal.jl")
include("inference/inla/hp_marginals/auto_marginal.jl")

include("inference/inla/latent_marginals/adaptive_marginal.jl")

include("inference/inla/types.jl")        # INLAResult
include("inference/inla/validation.jl")
include("inference/inla/progress.jl")

# `posterior/prediction.jl` defines `_prepare_for_prediction` (used inside
# `inla()`) plus `predicted_marginals(::INLAResult)` (dispatches on INLAResult).
# It must come after INLAResult is defined but before `inla()` uses it.
include("posterior/prediction.jl")

# INLAResult's implementation of the shared InferenceResult protocol. Needs
# INLAResult, NaturalHyperparameters, and `_prepare_for_prediction` available.
include("inference/inla/result_protocol.jl")

include("inference/inla/inference.jl")   # `inla(...)`

# ─── TMB inference ───────────────────────────────────────────────────────────
include("inference/tmb/types.jl")
include("inference/tmb/inference.jl")

# ─── HMC-Laplace inference (tmbstan-style NUTS on the Laplace marginal) ──────
# Depends on TMB's warm-start + covariance, so must come after TMB.
include("inference/hmc_laplace/types.jl")
include("inference/hmc_laplace/inference.jl")

# ─── Diagnostics (PSIS-k̂ for Laplace-based inference results) ───────────────
include("diagnostics/gpd_fit.jl")
include("diagnostics/psis.jl")
include("diagnostics/laplace_diagnostic.jl")

# ─── DSL: DynamicPPL → LatentGaussianModel adapter ───────────────────────────
# Structure probing, DAG extraction, pattern augmentation, hp spec / obs model
# extraction, and the `latte_from_dppl` entry point.
include("dsl/structure_probing.jl")
include("dsl/pattern_augment.jl")
include("dsl/dag_extraction.jl")
include("dsl/hp_spec.jl")
include("dsl/latent_prior.jl")
include("dsl/obs_model.jl")
include("dsl/likelihood_fast_paths.jl")
include("dsl/adapter.jl")

# ─── Posterior post-processing (method-agnostic in spirit) ───────────────────
include("posterior/accumulators/interface.jl")
include("posterior/accumulators/dic.jl")
include("posterior/accumulators/marginal_likelihood.jl")
include("posterior/accumulators/waic.jl")
include("posterior/accumulators/cpo.jl")

include("posterior/observation_marginals.jl")
include("posterior/sampling.jl")
include("posterior/linear_combinations.jl")
include("posterior/predict.jl")
include("posterior/model_averaging.jl")

end
