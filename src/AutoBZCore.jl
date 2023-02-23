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

    gloc_integrand(H, η, ω) = inv(complex(ω,η)*I-H) # define integrand evaluator
    h = FourierSeries([0.5, 0.0, 0.5]; offset=-2) # define 1D integer lattice Hamiltonian
    fbz = FullBZ(I(1)) # construct BZ from lattice with default limits of integration
    ps = (1.0, 0.0) # representative values for (η,ω), the last arguments of the integrand evaluator
    gloc = FourierIntegrator(gloc_integrand, fbz, h; ps=ps) # initialize default integration routine
    gloc(ps...) # evaluate gloc at parameter points

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

using Printf: @sprintf
using LinearAlgebra: I, norm, det, checksquare

using StaticArrays: SMatrix
using IteratedIntegration
using AutoSymPTR
using Reexport
@reexport using Integrals
@reexport using FourierSeriesEvaluators

import Integrals: IntegralProblem, __solvebp_call, SciMLBase.NullParameters
import AutoSymPTR: autosymptr, symptr, symptr_rule!, symptr_rule, ptr, ptr_rule!, ptrindex
import IteratedIntegration: iterated_integration, iterated_integrand, iterated_pre_eval


export SymmetricBZ, FullBZ, nsyms, symmetrize
export AbstractSymRep, SymRep, UnknownRep, TrivialRep, FaithfulRep
include("brillouin_zone.jl")

export FourierIntegrand
include("fourier_integration.jl")

export IAI, PTR, AutoPTR, PTR_IAI, AutoPTR_IAI, TAI
include("algorithms.jl")

export IntegralSolver, parallel_solve
include("jobs.jl")

end