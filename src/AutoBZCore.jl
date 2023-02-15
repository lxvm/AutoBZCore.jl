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
"""
module AutoBZCore

using LinearAlgebra

using StaticArrays

using FourierSeriesEvaluators
using IteratedIntegration
using AutoSymPTR

import AutoSymPTR: symptr, autosymptr, ptr, ptr!, ptr_, evalptr, ptr_integrand,
    symptr_kwargs, autosymptr_kwargs
import IteratedIntegration: iterated_integration, iterated_integration_kwargs,
    iterated_integral_type, iterated_inference, alloc_segbufs,
    iterated_integrand, iterated_pre_eval,
    quad_limits, quad_integrand, quad_kwargs

# component 1: Brillouin zone type with IAI & PTR bindings

export SymmetricBZ, FullBZ, nsyms, symmetrize
include("sym_bz.jl")

include("iai_bz.jl")

include("ptr_bz.jl")


# component 2: Fourier integrands with optimized IAI & PTR methods

export fourier_ptr!
include("fourier_ptr.jl")

export AbstractFourierIntegrand, FourierIntegrand, IteratedFourierIntegrand
include("fourier_integrands.jl")

export FourierIntegrator
include("fourier_integrator.jl")

end