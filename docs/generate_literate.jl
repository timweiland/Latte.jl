using Literate

# Tutorials
TUTORIALS_IN = joinpath(@__DIR__, "src", "literate-tutorials")
TUTORIALS_OUT = joinpath(@__DIR__, "src", "tutorials")
mkpath(TUTORIALS_OUT)

TUTORIALS_DATA = joinpath(@__DIR__, "src", "literate-tutorials", "data")
mkpath(TUTORIALS_DATA)

for (IN, OUT) in [(TUTORIALS_IN, TUTORIALS_OUT)]
    ## First pass: process .jl files and copy known assets
    for program in readdir(IN; join = true)
        name = basename(program)
        if endswith(program, ".jl")
            println(name)
            Literate.script(program, OUT)
            Literate.markdown(program, OUT)
            Literate.notebook(program, OUT)
        elseif any(endswith.(name, [".png", ".jpg", ".gif"]))
            cp(program, joinpath(OUT, name); force = true)
        elseif !isdir(program)
            @warn "ignoring $program"
        end
    end
    ## Second pass: copy any assets generated during Literate execution
    for asset in readdir(IN; join = true)
        name = basename(asset)
        if any(endswith.(name, [".png", ".jpg", ".gif"]))
            cp(asset, joinpath(OUT, name); force = true)
        end
    end
end
