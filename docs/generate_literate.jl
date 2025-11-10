using Literate

# Tutorials
TUTORIALS_IN = joinpath(@__DIR__, "src", "literate-tutorials")
TUTORIALS_OUT = joinpath(@__DIR__, "src", "tutorials")
mkpath(TUTORIALS_OUT)

TUTORIALS_DATA = joinpath(@__DIR__, "src", "literate-tutorials", "data")
mkpath(TUTORIALS_DATA)

for (IN, OUT) in [(TUTORIALS_IN, TUTORIALS_OUT)], program in readdir(IN; join = true)
    name = basename(program)
    if endswith(program, ".jl")
        println(name)
        Literate.script(program, OUT)
        Literate.markdown(program, OUT)
        Literate.notebook(program, OUT)
    elseif any(endswith.(program, [".png", ".jpg", ".gif"]))
        cp(program, joinpath(OUT, name); force = true)
    else
        @warn "ignoring $program"
    end
end
