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
