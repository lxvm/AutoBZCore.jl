"""
A package providing a common interface to integration algorithms intended for applications
including Brillouin-zone integration and Wannier interpolation. Its design is influenced by
high-level libraries like Integrals.jl, and it makes use of Julia's multiple dispatch to
provide the same interface for integrands with optimized inplace, batched, and Fourier
series evaluation.

### Quickstart

As a first example, we integrate sine over [0,1] as a function of its period.
```
julia> using AutoBZCore

julia> f = IntegralSolver((x,p) -> sin(p*x), 0, 1, QuadGKJL());

julia> f(0.3) # solves the integral of sin(p*x) over [0,1] with p=0.3
0.14887836958131329
```
Notice that we construct an [`IntegralSolver`](@ref) object that we can evaluate at
different parameters with a function-like interface. For more examples, see the
documentation.

### Features

Special integrand interfaces
- [`ParameterIntegrand`](@ref): allows user integrand to use keyword arguments
- [`InplaceIntegrand`](@ref): allows an integrand to write its result inplace to an array
- [`BatchIntegrand`](@ref): allows user-side parallelization on e.g. shared memory,
  distributed memory, or the gpu
- [`FourierIntegrand`](@ref): efficient evaluation of Fourier series for cubatures with
  hierachical grids

Quadrature algorithms:
- Trapezoidal rule and FastGaussQuadrature.jl: [`QuadratureFunction`](@ref)
- h-adaptive quadrature (Gauss-Kronrod): [`QuadGKJL`](@ref)
- h-adaptive cubature (Genz-Malik): [`HCubatureJL`](@ref)
- p-adaptive, symmetrized Monkhorst-Pack: [`AutoSymPTRJL`](@ref)

Meta-Algorithms:
- Iterated integration: [`NestedQuad`](@ref)
- Integrand evaluation counter: [`EvalCounter`](@ref)

# Extended help

If you experience issues with AutoBZCore.jl, please report a bug on the [GitHub
page](https://github.com/lxvm/AutoBZCore.jl) to contact the developers.
"""
module AutoBZCore

using LinearAlgebra: I, norm, det, checksquare, isdiag, Diagonal, tr, diag, eigen, Hermitian

using StaticArrays: SVector, SMatrix, pushfirst, sacollect
using FunctionWrappers: FunctionWrapper
using ChunkSplitters: chunks, getchunk
using Reexport
@reexport using AutoSymPTR
@reexport using FourierSeriesEvaluators
@reexport using IteratedIntegration
@reexport using QuadGK
@reexport using HCubature

using FourierSeriesEvaluators: workspace_allocate, workspace_contract!, workspace_evaluate!, workspace_evaluate, period
using IteratedIntegration: limit_iterate, interior_point
using HCubature: hcubature

export PuncturedInterval, HyperCube
include("domains.jl")

export InplaceIntegrand
include("inplace.jl")

export BatchIntegrand, NestedBatchIntegrand
include("batch.jl")

# export IntegralProblem, solve, init, solve! # we don't export the SciML interface
export IntegralSolver, batchsolve
include("interfaces.jl")

export QuadGKJL, HCubatureJL, QuadratureFunction
export AuxQuadGKJL, ContQuadGKJL, MeroQuadGKJL
export MonkhorstPack, AutoSymPTRJL
export NestedQuad, AbsoluteEstimate, EvalCounter
include("algorithms.jl")

export SymmetricBZ, nsyms
export load_bz, FBZ, IBZ, InversionSymIBZ, CubicSymIBZ
export AbstractSymRep, SymRep, UnknownRep, TrivialRep
export IAI, PTR, AutoPTR, TAI, PTR_IAI, AutoPTR_IAI
include("brillouin.jl")

export ParameterIntegrand, paramzip, paramproduct
include("parameters.jl")

export FourierIntegrand, FourierValue
include("fourier.jl")

export DOSProblem
include("dos_interfaces.jl")

export GGR, RationalRichardson
include("dos_algorithms.jl")
include("dos_ggr.jl")

end
