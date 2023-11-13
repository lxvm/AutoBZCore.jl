# Problem definitions

The design of AutoBZCore.jl is heavily influenced by the
[SciML](https://sciml.ai/) package
[Integrals.jl](https://github.com/SciML/Integrals.jl)
and may eventually become implemented in it.

## SciML interface

AutoBZCore.jl replicates the Integrals.jl interface, but does not export it in
order to avoid name conflicts with other SciML packages.

### Quickstart

```julia
using AutoBZCore: IntegralProblem, init, solve!

prob = IntegralProblem((x,p) -> sin(p*x), 0, 1, 0.3)
cache = init(prob, QuadGKJL())
solve!(cache)   # 0.14887836958131329

# solve again at a new parameter
cache.p = 0.4
solve!(cache)   # 0.1973475149927873
```

### Reference

```@docs
AutoBZCore.IntegralProblem
AutoBZCore.solve
AutoBZCore.init
AutoBZCore.solve!
AutoBZCore.NullParameters
```

## Functor interface

As shown in the quickstart of the [`AutoBZCore`](@ref) page, AutoBZCore.jl also defines
a functor interface to solving integrals

```@docs
AutoBZCore.IntegralSolver
```

The functor interface is also extended by [`ParameterIntegrand`](@ref) and
[`FourierIntegrand`](@ref) to
allow a more flexible interface for passing (partial) positional and keyword
arguments to user-defined integrands.

```@docs
AutoBZCore.MixedParameters
AutoBZCore.paramzip
AutoBZCore.paramproduct
```

## Batched evaluation

The routine [`batchsolve`](@ref) allows multi-threaded evaluation of an
[`IntegralSolver`](@ref) at many parameter points.

```@docs
AutoBZCore.batchsolve
```