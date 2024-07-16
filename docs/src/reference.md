# Function reference

The following symbols are exported by AutoBZCore.jl

## Brillouin-zone kinds

```@docs
AutoBZCore.load_bz
AutoBZCore.load_bz(::IBZ, A, B, species, positions)
AutoBZCore.AbstractBZ
AutoBZCore.FBZ
AutoBZCore.IBZ
AutoBZCore.InversionSymIBZ
AutoBZCore.CubicSymIBZ
```

## Symmetry representations

```@docs
AutoBZCore.AbstractSymRep
AutoBZCore.TrivialRep
AutoBZCore.UnknownRep
AutoBZCore.symmetrize
```

## Internal

The following docstrings belong to internal types and functions that may change between
versions of AutoBZCore.

```@docs
AutoBZCore.PuncturedInterval
AutoBZCore.HyperCube
AutoBZCore.SymmetricBZ
AutoBZCore.trapz
AutoBZCore.cube_automorphisms
```