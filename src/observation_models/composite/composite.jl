# Composite likelihood implementation
# This module provides support for combining multiple observation models

include("composite_observations.jl")
include("composite_observation_model.jl")
include("composite_evaluation.jl")

export CompositeObservations, CompositeObservationModel, CompositeLikelihood
