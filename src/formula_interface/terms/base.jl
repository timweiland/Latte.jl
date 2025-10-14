using StatsModels

export RandomEffectTerm

"""
    RandomEffectTerm <: AbstractTerm

Abstract base type for all random effect terms in the formula interface.

All concrete subtypes must implement:
- `StatsModels.termvars(term)`: Variables used by the term
- `StatsModels.modelcols(term, data)`: Design matrix columns  
- `gmrf_block(term, data, θ_named)`: Precision matrix block
- `hyperparameters(term)`: Required hyperparameter names
- `Base.show(io, term)`: String representation
"""
abstract type RandomEffectTerm <: StatsModels.AbstractTerm end
