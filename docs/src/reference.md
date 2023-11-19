# Function reference

The following symbols are exported by AutoBZCore.jl

## Domains

```@docs
AutoBZCore.PuncturedInterval
AutoBZCore.HyperCube
AutoBZCore.load_bz
AutoBZCore.SymmetricBZ
```

## Brillouin-zone kinds

```@docs
AutoBZCore.AbstractBZ
AutoBZCore.FBZ
AutoBZCore.IBZ
AutoBZCore.InversionSymIBZ
AutoBZCore.CubicSymIBZ
```

## Symmetry representations

```@docs
AutoBZCore.AbstractSymRep
AutoBZCore.SymRep
AutoBZCore.TrivialRep
AutoBZCore.UnknownRep
AutoBZCore.symmetrize_
```

## Internal

The following docstrings belong to internal functions that may change between
versions of AutoBZCore.

```@docs
AutoBZCore.trapz
AutoBZCore.cube_automorphisms
AutoBZCore.batchparam
AutoBZCore.symmetrize
AutoBZCore.integralerror
AutoBZCore.FourierValue
```