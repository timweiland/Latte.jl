using IntegratedNestedLaplace
using Documenter

DocMeta.setdocmeta!(IntegratedNestedLaplace, :DocTestSetup, :(using IntegratedNestedLaplace); recursive=true)

makedocs(;
    authors="Tim Weiland <tim@weiland-lahnstein.de> and contributors",
    sitename="IntegratedNestedLaplace.jl",
    format=Documenter.HTML(;
        canonical="https://timweiland.github.io/IntegratedNestedLaplace.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Reference" => [
            "Observation Models" => "reference/observation_models.md",
            "Hyperparameters" => "reference/hyperparameters.md",
            "Gaussian Approximation" => "reference/gaussian_approximation.md",
            "INLA Model" => "reference/inla_model.md",
            "Hyperparameter Posterior" => "reference/hyperparameter_posterior.md",
        ],
    ],
)

deploydocs(;
    repo="github.com/timweiland/IntegratedNestedLaplace.jl",
    devbranch="main",
)
