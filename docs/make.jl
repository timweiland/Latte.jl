using Latte
using Documenter
using DocumenterVitepress

include("generate_literate.jl")

DocMeta.setdocmeta!(Latte, :DocTestSetup, :(using Latte); recursive = true)

makedocs(;
    authors = "Tim Weiland <hello@timwei.land> and contributors",
    sitename = "Latte.jl",
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/timweiland/Latte.jl",
        devbranch = "main",
        devurl = "dev"
    ),
    pages = [
        "Home" => "index.md",
        "Tutorials" => [
            "Overview" => "tutorials/index.md",
            "Foundations" => [
                "Getting started" => "tutorials/getting_started.md",
                "Getting familiar with INLA" => "tutorials/inla_in_depth.md",
            ],
            "Spatial & spatio-temporal" => [
                "Spatial disease mapping" => "tutorials/disease_mapping_spatial.md",
                "Spatial SPDE: earthquake intensity" => "tutorials/spatial_spde.md",
                "Spatial barrier models: coastlines" => "tutorials/barrier_coastline.md",
                "Spatial survival: leukemia" => "tutorials/spatial_survival_leukemia.md",
                "Spatio-temporal modeling" => "tutorials/spatio_temporal_separable.md",
            ],
            "Temporal & regression" => [
                "Temporal trends" => "tutorials/temporal_trend_earthquakes.md",
                "Nonlinear regression" => "tutorials/nonlinear_regression_gam.md",
            ],
            "Custom & nonlinear models" => [
                "Custom likelihoods: Tweedie" => "tutorials/tweedie_insurance.md",
                "Age-structured fisheries assessment (SAM)" => "tutorials/age_structured_sam.md",
            ],
            "Model checking & comparison" => [
                "Bayesian model averaging" => "tutorials/bayesian_model_averaging.md",
                "Posterior predictive checks" => "tutorials/posterior_predictive_checks.md",
                "Simulation-based calibration" => "tutorials/sbc_calibration.md",
            ],
            "Inference engines & interop" => [
                "Handoff to Turing" => "tutorials/turing_handoff.md",
                "Sampling hyperparameters (HMC-Laplace)" => "tutorials/hmc_laplace_when.md",
            ],
        ],
        "Coming from R-INLA" => "coming_from_rinla.md",
        "Benchmarks" => "benchmarks/index.md",
        "Validation" => "validation/index.md",
        "Inference engines" => [
            "INLA" => "engines/inla.md",
            "TMB" => "engines/tmb.md",
            "HMC-Laplace" => "engines/hmc_laplace.md",
        ],
        "Reference" => [
            "Overview" => "reference/index.md",
            "Defining models: @latte" => "reference/latte.md",
            "Working with results" => "reference/results.md",
            "Lower-level construction" => "reference/lower_level.md",
            "Observation models" => "reference/observation_models.md",
            "Gaussian approximation" => "reference/gaussian_approximation.md",
            "Marginalization" => "reference/marginalization.md",
            "Hyperparameter posterior" => "reference/hyperparameter_posterior.md",
        ],
    ]
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/timweiland/Latte.jl",
    devbranch = "main"
)
