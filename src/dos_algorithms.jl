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

# Extension algorithms

"""
    RationalRichardson(; alg=IAI(), m=2, h=1.0, power=1, contract=0.125, breaktol=2, kwargs...)

Compute a density of states extrapolation by smearing the Dirac distribution
with a rational kernel of order `m`, as presented in ["Computing Spectral
Measures of Self-Adjoint operators"](https://doi.org/10.1137/20M1330944).
This algorithm requires `using Richardson`.

This algorithm assumes that the operator `H` is a function of a continuous
crystal momentum, `H(k)`, and will use the integration algorithm `alg` to
estimate that integral. Additionally, the parameter `p` passed to `DOSProblem`
should be the domain of integration (e.g. the Brillouin zone). Finally, the
other `kwargs` passed to `RationalRichardson` will be used as the integration
solver keywords, such as `abstol`, `reltol`, and `maxiters`, which the caller
should set.

The parameters `h>0` and `power` control the initial evaluation point and the
exponent of the expansion in `h`, respectively. See
[Richardson.jl](https://github.com/JuliaMath/Richardson.jl) for more details on
these keywords as well as `contract` and `breaktol`.
"""
struct RationalRichardson{A<:IntegralAlgorithm,T<:Number,P<:Real,K,C<:Number,B<:Real} <: DOSAlgorithm
    m::Int
    h::T
    power::P
    contract::C
    breaktol::B
    alg::A
    kwargs::K
end
function RationalRichardson(; alg=IAI(), h=1.0, power=1, contract=oftype(float(real(h)), 0.125), breaktol=2, m=2, kws...)
    return RationalRichardson(m, h, power, contract, breaktol, alg, NamedTuple(kws))
end
