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
            "Getting started" => "tutorials/getting_started.md",
            "Spatial disease mapping" => "tutorials/disease_mapping_spatial.md",
            "Bayesian model averaging" => "tutorials/bayesian_model_averaging.md",
            "Temporal trends" => "tutorials/temporal_trend_earthquakes.md",
            "Nonlinear regression" => "tutorials/nonlinear_regression_gam.md",
            "Spatial SPDE: earthquake intensity" => "tutorials/spatial_spde.md",
            "Spatio-temporal modeling" => "tutorials/spatio_temporal_separable.md",
        ],
        "Main Interface" => "main_interface.md",
        "Reference" => [
            "Observation Models" => "reference/observation_models.md",
            "Hyperparameters" => "reference/hyperparameters.md",
            "Gaussian Approximation" => "reference/gaussian_approximation.md",
            "Marginalization" => "reference/marginalization.md",
            "INLA Model" => "reference/inla_model.md",
            "Hyperparameter Posterior" => "reference/hyperparameter_posterior.md",
        ],
    ],
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/timweiland/Latte.jl",
    devbranch = "main",
)
