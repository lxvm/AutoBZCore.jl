# TODO: release as a separate package
"""
    AutoBZCore

The package providing the functionality and abstractions for `AutoBZ.jl`
"""
module AutoBZCore

using LinearAlgebra

using StaticArrays

using AbstractFourierSeriesEvaluators
using IteratedIntegration

import AutoSymPTR: symptr, autosymptr, ptr, ptr!, evalptr, ptr_integrand
import IteratedIntegration: iterated_integration, iterated_integration_kwargs,
    iterated_integral_type, iterated_inference, alloc_segbufs

# component 1: Brillouin zone type with IAI & PTR bindings

export SymmetricBZ, FullBZ, basis, nsyms, symmetries, symmetrize, limits
include("sym_bz.jl")

include("iai_bz.jl")

include("ptr_bz.jl")


# component 2: Fourier integrands with optimized IAI & PTR methods

export fourier_ptr!
include("fourier_ptr.jl")

export AbstractFourierIntegrand, finner, ftotal, series, params
export FourierIntegrand, IteratedFourierIntegrand
include("fourier_integrands.jl")

export FourierIntegrator
include("fourier_integrator.jl")

end