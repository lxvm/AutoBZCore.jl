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
- Adaptive Gaussian briadening

```@docs
AutoBZCore.DOSAlgorithm
AutoBZCore.GGR
AutoBZCore.RationalRichardson
```