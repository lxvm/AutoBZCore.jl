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

using LinearAlgebra: I, norm, det, checksquare

using StaticArrays: SMatrix
using QuadGK: quadgk

using FourierSeriesEvaluators
using IteratedIntegration
using AutoSymPTR

import AutoSymPTR: symptr, autosymptr, ptr, ptr!, ptr_, evalptr, ptr_integrand,
    symptr_kwargs, autosymptr_kwargs
import IteratedIntegration: iterated_integration, iterated_integration_kwargs,
    iterated_integrand, iterated_pre_eval


export AbstractIntegrand, QuadIntegrand, Integrator, quad_args
include("definitions.jl")

export SymmetricBZ, FullBZ, nsyms, symmetrize
include("brillouin_zone.jl")

export FourierIntegrand, FourierIntegrator
include("fourier_integration.jl")


end