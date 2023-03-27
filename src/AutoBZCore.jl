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

    gloc_integrand(h_k; η, ω) = inv(complex(ω,η)*I-h_k)     # define integrand evaluator
    h = FourierSeries([0.5, 0.0, 0.5]; offset=-2)           # construct cos(k) 1D integer lattice Hamiltonian
    bz = FullBZ(2pi*I(1))                                   # construct BZ from lattice vectors A=2pi*I
    integrand = FourierIntegrand(gloc_integrand, h, η=0.1)    # construct integrand with Fourier series h and parameter η=0.1
    alg = IAI()                                             # choose integration algorithm (also AutoPTR() and PTR())
    gloc = IntegralSolver(integrand, bz, alg; abstol=1e-3)  # construct a solver for gloc to within specified tolerance
    gloc(ω=0.0)                                               # evaluate gloc at frequency ω=0.0

!!! note "Assumptions"
    `AutoBZCore` assumes that all calculations occur in the
    reciprocal lattice basis, since that is the basis in which Wannier
    interpolants are most efficiently described. See [`SymmetricBZ`](@ref) for
    details.
"""
module AutoBZCore

using LinearAlgebra: I, norm, det, checksquare

using StaticArrays: SMatrix
using Reexport
@reexport using Integrals
@reexport using FourierSeriesEvaluators
@reexport using AutoSymPTR
@reexport using IteratedIntegration

import Integrals: IntegralProblem, __solvebp_call, SciMLBase.NullParameters, ReCallVJP, ZygoteVJP
import AutoSymPTR: autosymptr, symptr, symptr_rule!, symptr_rule, ptr, ptr_rule!, ptrindex, alloc_autobuffer
import IteratedIntegration: iterated_integrand, iterated_pre_eval, alloc_segbufs


export SymmetricBZ, FullBZ, nsyms
export AbstractSymRep, SymRep, UnknownRep, TrivialRep, FaithfulRep, LatticeRep
include("brillouin_zone.jl")

export AbstractAutoBZAlgorithm, IAI, PTR, AutoPTR, PTR_IAI, AutoPTR_IAI, TAI
include("algorithms.jl")

export MixedParameters, IntegralSolver, batchsolve
include("solver.jl")

export Integrand
include("integrand.jl")

"""
    NTHREADS_KSUM = fill(Threads.nthreads())

This represents the number of threads to parallelize over for k-sums in PTR. To
change the number of threads to 1 to completely bypass multi-threading, write
`AutoBZCore.NTHREADS_KSUM[] = 1`.
"""
const NTHREADS_KSUM = fill(Threads.nthreads())

export FourierIntegrand
include("fourier_integration.jl")


end