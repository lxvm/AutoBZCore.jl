push!(LOAD_PATH, "../src/")
using Documenter, AutoBZCore

Documenter.HTML(
    mathengine = MathJax3(Dict(
        :loader => Dict("load" => ["[tex]/physics"]),
        :tex => Dict(
            "inlineMath" => [["\$","\$"], ["\\(","\\)"]],
            "tags" => "ams",
            "packages" => ["base", "ams", "autoload", "physics"],
        ),
    )),
)

makedocs(
    sitename="AutoBZCore.jl",
    modules=[AutoBZCore],
    pages = [
        "Home" => "index.md",
        "Examples" => "examples.md",
        "Problems" => "problems.md",
        "Integrands" => "integrands.md",
        "Algorithms" => "algorithms.md",
        "Reference" => "reference.md",
        "Extensions" => "extensions.md",
    ],
)

deploydocs(
    repo = "github.com/lxvm/AutoBZCore.jl.git",
)
