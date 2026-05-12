using BayesKNN
using Documenter

DocMeta.setdocmeta!(BayesKNN, :DocTestSetup, :(using BayesKNN); recursive=true)

makedocs(;
    modules=[BayesKNN],
    checkdocs=:exports,
    authors="Stanley E. Lazic",
    sitename="BayesKNN.jl",
    format=Documenter.HTML(;
        canonical="https://stanlazic.github.io/BayesKNN.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/stanlazic/BayesKNN.jl",
    devbranch="master",
)
