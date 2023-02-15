# TODO: release as a separate package
"""
    AutoBZCore

The package providing the functionality and abstractions for `AutoBZ.jl`
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