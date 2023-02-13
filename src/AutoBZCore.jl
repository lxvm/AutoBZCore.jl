# TODO: release as a separate package
"""
    AutoBZCore

The package providing the functionality and abstractions for `AutoBZ.jl`
"""
module AutoBZCore

using StaticArrays


using AbstractFourierSeriesEvaluators
using IteratedIntegration

import AutoSymPTR: symptr, autosymptr, ptr, ptr!, evalptr

# component 1: Brillouin zone abstractions

export AbstractBZ, FullBZ, SymmetricBZ
export basis, nsyms, symmetries, symmetrize, limits, boundingbox, vol, coefficient_type
include("AbstractBZ.jl")


# component 2: symmetrized integration methods

include("iai_bz.jl")
include("ptr_bz.jl")


# component 3: Fourier integrands with optimized IAI & PTR methods

export fourier_ptr!
include("fourier_rule.jl")

export AbstractFourierIntegrand, finner, ftotal, series, params
export FourierIntegrand, IteratedFourierIntegrand
include("fourier_integrands.jl")

export FourierIntegrator
include("FourierIntegrator.jl")

end