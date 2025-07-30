using IntegratedNestedLaplace
using Documenter
using DocumenterVitepress

DocMeta.setdocmeta!(IntegratedNestedLaplace, :DocTestSetup, :(using IntegratedNestedLaplace); recursive = true)

makedocs(;
    authors = "Tim Weiland <hello@timwei.land> and contributors",
    sitename = "IntegratedNestedLaplace.jl",
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/timweiland/IntegratedNestedLaplace.jl",
        devbranch = "main",
        devurl = "dev"
    ),
    pages = [
        "Home" => "index.md",
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
    repo = "github.com/timweiland/IntegratedNestedLaplace.jl",
    devbranch = "main",
)
