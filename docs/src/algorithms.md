# Integral algorithms

```@docs
AutoBZCore.IntegralAlgorithm
```

## Quadrature

```@docs
AutoBZCore.QuadratureFunction
AutoBZCore.QuadGKJL
AutoBZCore.AuxQuadGKJL
AutoBZCore.ContQuadGKJL
AutoBZCore.MeroQuadGKJL
```

## Cubature

```@docs
AutoBZCore.HCubatureJL
AutoBZCore.MonkhorstPack
AutoBZCore.AutoSymPTRJL
```

## Meta-algorithms

```@docs
AutoBZCore.NestedQuad
AutoBZCore.EvalCounter
AutoBZCore.AbsoluteEstimate
```

# BZ-specific integral algorithms

In order to make algorithms domain-agnostic, the BZ loaded from
[`load_bz`](@ref) can be called with the algorithms below, which are wrappers
for algorithms above with the additional capability of mapping integrals over
the IBZ to the FBZ.

```@docs
AutoBZCore.AutoBZAlgorithm
AutoBZCore.IAI
AutoBZCore.TAI
AutoBZCore.PTR
AutoBZCore.AutoPTR
AutoBZCore.PTR_IAI
AutoBZCore.AutoPTR_IAI
```