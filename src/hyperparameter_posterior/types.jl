"""
Type definitions for hyperparameter posterior exploration and approximation.
"""

using Distributions
# Method extension for computing mode of Product distributions
"""
    mode(d::Product)

Compute the mode of a Product distribution by computing the mode of each marginal distribution.

The mode of a product of independent distributions is the vector of modes of the marginal distributions.
"""
function Distributions.mode(d::Product)
    return [mode(marginal) for marginal in d.v]
end
