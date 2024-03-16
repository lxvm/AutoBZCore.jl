# Density of States

Computing the density of states (DOS) of a self-adjoint, or Hermitian, operator is a
related, but distinct problem to the integrals also presented in this package.
In fact, many DOS algorithms will compute integrals to approximate the DOS of an
operator that depends on a parameter such as crystal momentum. To solve these
types of problems, this package defines the following problem type:

```@docs
AutoBZCore.DOSProblem
```

## Algorithms

Currently the available algorithms are experimental and we would like to include
the following reference algorithms that are more common in the literature in a future release:
- (Linear) Tetrahedron Method
- Adaptive Gaussian broadening

```@docs
AutoBZCore.DOSAlgorithm
AutoBZCore.GGR
```

## Caching interface

Using the [`AutoBZCore.init`](@ref) and [`AutoBZCore.solve!`](@ref) functions, it is possible to
construct a cache to solve a [`DOSProblem`](@ref) for several energies or
several Hamiltonians. As an example of solving for several energies,
```julia
using AutoBZCore, StaticArrays
h = FourierSeries(SMatrix{1,1,Float64,1}.([0.5, 0.0, 0.5]), period=1.0, offset=-2)
E = 0.3
bz = load_bz(FBZ(), [2pi;;])
prob = DOSProblem(h, E, bz)
alg = GGR(; npt=100)
cache = AutoBZCore.init(prob, alg)
Es = range(-2, 2, length=1000)
data = map(Es) do E
    cache.domain = E
    AutoBZCore.solve!(cache).u
end
```

As an example of interchanging Hamiltonians, which must remain the same type,
```julia
using AutoBZCore, StaticArrays

h = FourierSeries(SMatrix{1,1,Float64,1}.([0.5, 0.0, 0.5]), period=1.0, offset=-2)
bz = load_bz(FBZ(), [2pi;;])
prob = DOSProblem(h, 0.0, bz)
alg = GGR()

cache = AutoBZCore.init(prob, alg)
sol1 = AutoBZCore.solve!(cache)

h.c .*= 2
cache.isfresh = true

sol2 = AutoBZCore.solve!(cache)

sol1.u*2 ≈ sol2.u
```