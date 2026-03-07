"""
    lambda_from_tail(U, α)

Calibrate a PC prior via tail probability: P(distance > U) = α ⇒ λ = -log(α) / U.

# Arguments
- `U > 0`: soft upper bound for the distance measure
- `α ∈ (0,1)`: prior tail mass beyond U
"""
lambda_from_tail(U, α) = -log(α) / U
