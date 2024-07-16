# Algorithms

## `IntegralProblem` algorithms

```@docs
AutoBZCore.IntegralAlgorithm
```

### Quadrature

```@docs
AutoBZCore.QuadratureFunction
AutoBZCore.QuadGKJL
AutoBZCore.AuxQuadGKJL
AutoBZCore.ContQuadGKJL
AutoBZCore.MeroQuadGKJL
```

### Cubature

```@docs
AutoBZCore.HCubatureJL
AutoBZCore.MonkhorstPack
AutoBZCore.AutoSymPTRJL
```

### Meta-algorithms

```@docs
AutoBZCore.NestedQuad
```

## `AutoBZProblem` algorithms

In order to make algorithms domain-agnostic, the BZ loaded from
[`load_bz`](@ref) can be called with the algorithms below, which are aliases
for algorithms above

```@docs
AutoBZCore.AutoBZAlgorithm
AutoBZCore.IAI
AutoBZCore.TAI
AutoBZCore.PTR
AutoBZCore.AutoPTR
```

## `DOSProblem` algorithms

Currently the available algorithms are an initial release and we would like to include
the following reference algorithms that are also common in the literature in a future release:
- (Linear) Tetrahedron Method
- Adaptive Gaussian broadening

```@docs
AutoBZCore.DOSAlgorithm
AutoBZCore.GGR
```
