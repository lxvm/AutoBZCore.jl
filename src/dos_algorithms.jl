# BCD
# - number of k points
# - H_R or DFT data

# LTM
# - number of k points
# - any function or H_R

"""
    GGR(; npt=50)

Generalized Gilat-Raubenheimer method as in ["Generalized Gilat–Raubenheimer method for
density-of-states calculation in photonic
crystals"](https://doi.org/10.1088/2040-8986/aaae52).
This method requires the Hamiltonian and its derivatives, and performs a linear
extrapolation at each k-point in an equispace grid. The algorithm is expected to show
second-order convergence and suffer reduced error at band crossings compared to
interpolatory methods.

## Arguments
- `npt`: the number of k-points per dimension
"""
struct GGR <: DOSAlgorithm
    npt::Int
end
GGR(; npt=50) = GGR(npt)


"""
    ImplicitIntegrationJL(; kws...)

This algorithm is implemented in an extension. Try it with `using ImplicitIntegration`.
"""
struct ImplicitIntegrationJL{K} <: DOSAlgorithm
    kws::K
end
ImplicitIntegrationJL(; kws...) = ImplicitIntegrationJL(kws)

"""
    BCD(npt, α, ΔE)

## Arguments
- `npt`: the number of k-points per dimension
- `α`: a scaling parameter for the deformation
- `ΔE`: a parameter for the cut-off function in the deformation
"""
struct BCD <: DOSAlgorithm
    npt::Int
    α::Float64
    ΔE::Float64
    η::Float64
end
BCD(; npt=50, α=0.1/(2π), ΔE=0.9, η=0) = BCD(npt, α, ΔE, η)