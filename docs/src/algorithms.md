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
AutoBZCore.EvalCounter
AutoBZCore.EvalLogger
```

## `AutoBZProblem` algorithms

Although different algorithms may use different representations of the BZ, the BZ loaded from
[`load_bz`](@ref) can be called with any of the algorithms below, which are aliases
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
- Adaptive Gaussian broadening

```@docs
AutoBZCore.DOSAlgorithm
AutoBZCore.GGR
AutoBZCore.BCD
AutoBZCore.ImplicitIntegrationJL
AutoBZCore.LT
```
