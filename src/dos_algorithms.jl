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
GGR(; npt = 50) = GGR(npt)


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

Brillouin Contour Deformation method was developed in ["Efficient extraction of resonant states 
in systems with defects"](https://doi.org/10.1016/j.jcp.2023.111928).
This method requires the Hamiltonian and its first and second order derivatives. 
It performs a deformation of the Brillouin zone into the complex planes based on
the first derivative of the Hamiltonian at singularity points. The deformation is controlled by 
two parameters α and ΔE which scale the deformation around singularity respectively in amplitude and width.
This method is expected to show exponential convergence and is not highly sensitive 
to the choice of parameters. Therefore it can be used with default parameters in most cases.

## Arguments
- `npt`: the number of k-points per dimension
- `α`: a scaling parameter for the amplitude of the deformation
- `ΔE`: a parameter which impacts the width of the deformation at singularities
"""
struct BCD <: DOSAlgorithm
    npt::Int
    α::Float64
    ΔE::Float64
    η::Float64
end
BCD(; npt = 50, α = 0.1 / (2π), ΔE = 0.9, η = 0) = BCD(npt, α, ΔE, η)

"""
    LT(npt)

Linear Tetrahedron method ["High-precision sampling for Brillouin-zone integration in metals"](https://doi.org/10.1103/PhysRevB.40.3616).
This method requires Hamiltonian's eigenvalues. It performs a linear interpolation of the eigenvalues on a tetrahedric decomposition
of the Brillouin zone. Therefore only the eigenvalues are needed at each k-point to perform interpolation. This method is expected to show
quadratic convergence.

## Arguments
- `npt`: the number of k-points per dimension
"""
struct LT <: DOSAlgorithm
    npt::Int
end
LT(; npt = 50) = LT(npt)
