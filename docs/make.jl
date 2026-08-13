using InterfacialWaves
using Documenter
using DocumenterCitations

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style=:numeric)

DocMeta.setdocmeta!(
    InterfacialWaves,
    :DocTestSetup,
    :(using InterfacialWaves),
    recursive=true,
)

makedocs(
    modules=[InterfacialWaves],
    checkdocs=:none,
    authors="Nikhil Yewale and Ratul Dasgupta",
    sitename="InterfacialWaves.jl",
    format=Documenter.HTML(
        canonical="https://yewalenikhil65.github.io/InterfacialWaves.jl",
        edit_link="main",
        assets=String[],
        mathengine=MathJax3(),
        prettyurls=get(ENV, "CI", "false") == "true",
        example_size_threshold=nothing,
        size_threshold=nothing,
        size_threshold_warn=nothing,
    ),
    pages=[
        "Home" => "index.md",
        "Travelling Waves" => [
            "Overview" => "travelling/index.md",
            "Pure Gravity Waves" => "travelling/gravity.md",
            "Gravity–Capillary Waves" => "travelling/gravity_capillary.md",
            "Viscous Gravity–Capillary Waves" => "travelling/viscous.md",
            "Interfacial Waves" => "travelling/interfacial.md",
        ],
        "Standing Waves" => "standing/index.md",
        "Stability" => "stability/index.md",
        "Progress" => "progress.md",
        "API Reference" => "api.md",
        "References" => "references.md",
    ],
    plugins=[bib],
    remotes=nothing,
)

deploydocs(
    repo="github.com/yewalenikhil65/InterfacialWaves.jl.git",
    devbranch="main",
    push_preview=false,
)
