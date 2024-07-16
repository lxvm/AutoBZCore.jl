# Problem definitions

The design of AutoBZCore.jl is heavily influenced by
[SciML](https://sciml.ai/) packages and uses the 
[CommonSolve.jl](https://github.com/SciML/CommonSolve.jl)
interface. Eventually, this package may contribute to
[Integrals.jl](https://github.com/SciML/Integrals.jl).

## Problem interface

AutoBZCore.jl replicates the Integrals.jl interface, using an
[`IntegralProblem`](@ref) type to setup an integral from an
integrand, a domain, and parameters.

```@example prob
using AutoBZCore

f = (x,p) -> sin(p*x)
dom = (0, 1)
p = 0.3
prob = IntegralProblem(f, dom, p)
```

```@docs
AutoBZCore.IntegralProblem
AutoBZCore.NullParameters
```

## `solve`

To solve an integral problem, pick an algorithm and call [`solve`](@ref)

```@example prob
alg = QuadGKJL()
solve(prob, alg)
```

```@docs
AutoBZCore.solve
```

## `init` and `solve!`

To solve many problems with the same integrand but different domains or
parameters, use [`init`](@ref) to allocate a solver and
[`solve!`](@ref) to get the solution

```@example prob
solver = init(prob, alg)
solve!(solver).value
```

To solve again, update the parameters of the solver in place and `solve!` again
```@example prob
# solve again at a new parameter
solver.p = 0.4
solve!(solver).value
```


```@docs
AutoBZCore.init
AutoBZCore.solve!
```

## Additional problems

```@docs
AutoBZCore.AutoBZProblem
AutoBZCore.DOSProblem
```
