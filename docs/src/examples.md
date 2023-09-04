# Examples

The following are several examples of how to use the algorithms and integrands
provided by AutoBZCore.jl.

# `IntegralSolver`

```@docs
AutoBZCore.IntegralSolver
```

# Density of states

The [repo's demo](https://github.com/lxvm/AutoBZCore.jl/tree/main/aps_example)
on density of states provides a complete example of how to compute and
interpolate an integral as a function of its parameters.

# Green's function

For example, computing the local Green's function can be done as follows:

```julia
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
```

When performing integrals over multiple dimensions, it is much more efficient to
evaluate a Fourier series one dimension at a time. This capability is provided
by [`FourierIntegrand`](@ref) and is illustrated below
```julia
gloc_integrand(h_k::FourierValue; η, ω) = inv(complex(ω,η)*I-h_k.s)     # define integrand evaluator
h = FourierSeries([0.0; 0.5; 0.0;; 0.5; 0.0; 0.5;; 0.0; 0.5; 0.0]; period=1, offset=-2) # construct cos(2πk) 1D integer lattice Hamiltonian
bz = load_bz(FBZ(2), 2pi*I(2))                          # construct BZ from lattice vectors A=2pi*I
integrand = FourierIntegrand(gloc_integrand, h, η=0.1)  # construct integrand with Fourier series h and parameter η=0.1
prob = IntegralProblem(integrand, bz)                   # setup the integral problem
alg = IAI()                                             # choose integration algorithm (also AutoPTR() and PTR())
gloc = IntegralSolver(prob, alg; abstol=1e-3)           # construct a solver for gloc to within specified tolerance
gloc(ω=0.0)                                             # evaluate gloc at frequency ω=0.0
```

## SciML interface

AutoBZCore.jl replicates the Integrals.jl interface, but does not export it.

```julia
using AutoBZCore: IntegralProblem, init, solve!

prob = IntegralProblem((x,p) -> sin(p*x), 0, 1, 0.3)
cache = init(prob, QuadGKJL())
solve!(cache)   # 0.14887836958131329

# solve again at a new parameter
cache.p = 0.4
solve!(cache)   # 0.1973475149927873

```

## Batched evaluation

The routine [`batchsolve`](@ref) allows multi-threaded evaluation of an
[`IntegralSolver`](@ref) at many parameter points.