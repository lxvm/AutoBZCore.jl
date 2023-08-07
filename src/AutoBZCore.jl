"""
The package providing the functionality and abstractions for `AutoBZ.jl`. It
provides both [`SymmetricBZ`](@ref), which is a type that stores information
about the Brillouin zone (BZ), and [`IntegralSolver`](@ref), a type that
provides a functor interface to parametrize `IntegralProblem`s as defined by
[Integrals.jl](https://docs.sciml.ai/Integrals/stable/). The package also
provides the optimized [`FourierIntegrand`](@ref). The package integrates with
[FourierSeriesEvaluators.jl](https://github.com/lxvm/FourierSeriesEvaluators.jl),
[IteratedIntegration.jl](https://github.com/lxvm/IteratedIntegration.jl), and
[AutoSymPTR.jl](https://github.com/lxvm/AutoSymPTR.jl) to provide a generic
interface to efficient algorithms for BZ integration.

For example, computing the local Green's function can be done as follows:

    using LinearAlgebra
    using FourierSeriesEvaluators
    using AutoBZCore

    gloc_integrand(k, h; η, ω) = inv(complex(ω,η)*I-h(k))   # define integrand evaluator
    h = FourierSeries([0.5, 0.0, 0.5]; period=1, offset=-2) # construct cos(2πk) 1D integer lattice Hamiltonian
    integrand = ParameterIntegrand(gloc_integrand, h, η=0.1)         # construct integrand with Fourier series h and parameter η=0.1
    prob = IntegralProblem(integrand, 0, 1)                 # setup the integral problem
    alg = QuadGKJL()                                        # choose integration algorithm (also AutoPTR() and PTR())
    gloc = IntegralSolver(prob, alg; abstol=1e-3)           # construct a solver for gloc to within specified tolerance
    gloc(ω=0.0)                                             # evaluate gloc at frequency ω=0.0


    gloc_integrand(h_k::FourierValue; η, ω) = inv(complex(ω,η)*I-h_k.s)     # define integrand evaluator
    h = FourierSeries([0.0; 0.5; 0.0;; 0.5; 0.0; 0.5;; 0.0; 0.5; 0.0]; period=1, offset=-2) # construct cos(2πk) 1D integer lattice Hamiltonian
    bz = FullBZ(2pi*I(2))                                   # construct BZ from lattice vectors A=2pi*I
    integrand = FourierIntegrand(gloc_integrand, h, η=0.1)   # construct integrand with Fourier series h and parameter η=0.1
    prob = IntegralProblem(integrand, bz)                   # setup the integral problem
    alg = IAI()                                             # choose integration algorithm (also AutoPTR() and PTR())
    gloc = IntegralSolver(prob, alg; abstol=1e-3)           # construct a solver for gloc to within specified tolerance
    gloc(ω=0.0)                                             # evaluate gloc at frequency ω=0.0

!!! note "Assumptions"
    `AutoBZCore` assumes that all calculations occur in the
    reciprocal lattice basis, since that is the basis in which Wannier
    interpolants are most efficiently described. See [`SymmetricBZ`](@ref) for
    details. We also assume that the integrands are cheap to evaluate, which is why we
    provide adaptive methods in the first place, so that return types can be determined at
    runtime (and mechanisms are in place for compile time as well)
"""
module AutoBZCore

using LinearAlgebra: I, norm, det, checksquare

using StaticArrays: SVector, SMatrix, pushfirst
using FunctionWrappers: FunctionWrapper
using Reexport
@reexport using AutoSymPTR
@reexport using FourierSeriesEvaluators
@reexport using IteratedIntegration
@reexport using QuadGK
@reexport using HCubature

using IteratedIntegration: limit_iterate, interior_point
using HCubature: hcubature

export PuncturedInterval, HyperCube
include("domains.jl")

export InplaceIntegrand
include("inplace.jl")

export BatchIntegrand
include("batch.jl")

export IntegralAlgorithm, QuadGKJL, HCubatureJL, QuadratureFunction
export AuxQuadGKJL, ContQuadGKJL, MeroQuadGKJL
export MonkhorstPack, AutoSymPTRJL
export NestedQuad, AbsoluteEstimate
include("algorithms.jl")

export SymmetricBZ, FullBZ, nsyms
export AbstractSymRep, SymRep, UnknownRep, TrivialRep
export AutoBZAlgorithm, IAI, PTR, AutoPTR, TAI, PTR_IAI, AutoPTR_IAI
include("brillouin.jl")

export MixedParameters, paramzip, paramproduct
export IntegralSolver, batchsolve
export ParameterIntegrand
include("interfaces.jl")

export FourierIntegrand, FourierValue
include("rules.jl")


end
