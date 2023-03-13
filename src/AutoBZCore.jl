"""
The package providing the functionality and abstractions for `AutoBZ.jl`. It
provides both [`SymmetricBZ`](@ref), which is a type that stores information
about the Brillouin zone (BZ), and [`FourierIntegrator`](@ref), a type that
provides a functor interface to compute user-defined Brillouin zone integrals
depending on Wannier-interpolated quantities. The package integrates with
[`FourierSeriesEvaluators`](@ref), [`IteratedIntegration`](@ref), and
[`AutoSymPTR`](@ref) to provide a generic interface to efficient algorithms for
BZ integration.

For example, computing the local Green's function can be done as follows:

    using LinearAlgebra
    using FourierSeriesEvaluators
    using AutoBZCore

    gloc_integrand(h_k, η, ω) = inv(complex(ω,η)*I-h_k)     # define integrand evaluator
    h = FourierSeries([0.5, 0.0, 0.5]; offset=-2)           # construct cos(k) 1D integer lattice Hamiltonian
    bz = FullBZ(2pi*I(1))                                   # construct BZ from lattice vectors A=2pi*I
    integrand = FourierIntegrand(gloc_integrand, h, 0.1)    # construct integrand with Fourier series h and parameter η=0.1
    alg = IAI()                                             # choose integration algorithm (also AutoPTR() and PTR())
    gloc = IntegralSolver(integrand, bz, alg; abstol=1e-3)  # construct a solver for gloc to within specified tolerance
    gloc(0.0)                                               # evaluate gloc at frequency ω=0.0

!!! note "Assumptions"
    `AutoBZCore` assumes that all calculations occur in the reciprocal lattice
    basis, since that is the basis in which Wannier interpolants are most
    efficiently described. This means that user-provided integration limits and
    symmetries should be in the reciprocal lattice basis and additionally they
    should be in fractional coordinates (e.g. the BZ in these coordinates has
    vertices (0,0,0) and (1,1,1)). All symmetry transformations must be dealt
    with by the user, who can specialize the [`symmetrize`](@ref) routine to
    automate that step.
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

export FourierIntegrand
include("fourier_integration.jl")

export AbstractAutoBZAlgorithm, IAI, PTR, AutoPTR, PTR_IAI, AutoPTR_IAI, TAI
include("algorithms.jl")

export Integrand
include("integrand.jl")

export IntegralSolver, batchsolve
include("solver.jl")

end