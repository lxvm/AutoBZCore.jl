# Brillouin zones

The first [Brillouin zone](https://en.wikipedia.org/wiki/Brillouin_zone) (FBZ)
or full BZ, is the domain of integration for many quantum-mechanical observables
in crystalline systems with translation symmetry. In systems with additional
point-group symmetries, the symmetries can be used to down-fold the FBZ onto the
irreducible Brillouin zone (IBZ) that can be used to reduce the computational
cost by a factor of at most ``2^d d!`` in ``d`` spatial dimensions. While the
FBZ is isomorphic to a hypercube, the IBZ is a polytope. The following type
stores the data used by various algorithms to do IBZ integration.

```@docs
SymmetricBZ
FullBZ
```


## Integration limits

The [`IAI`](@ref) algorithm can do IBZ integration with knowledge of the convex
hull of the IBZ. The `calc_ibz` routine in
[SymmetryReduceBZ.jl](https://github.com/jerjorg/SymmetryReduceBZ.jl) does
exactly this step. Then polyhedral manipulation algorithms can be used to do
this step (search for `PolyhedralLimits` in IteratedIntegration.jl).


## Lattice symmetries

The [`PTR`](@ref) and [`AutoPTR`](@ref) algorithms can do IBZ integration (also
known as Monkhorst-Pack integration) simply with the crystallographic point
group. This can be provided as a vector of the point group operators in the
`syms` argument of [`SymmetricBZ`](@ref).


## Symmetry representations

The result of the integral on the IBZ still needs to be mapped to the FBZ. The
[`SymRep`](@ref) trait is provided so that the caller can specify the symmetry
representation of their integrand to automate this mapping.

```@docs
SymRep
AbstractSymRep
FaithfulRep
```

The following representations are provided out-of-the box:

```@docs
TrivialRep
LatticeRep
UnknownRep
```


### User-defined `SymRep`s

These are the methods to be aware of extending to implement the action of a
symmetry group for mapping an integral from the IBZ to the FBZ.

```@docs
AutoBZCore.symmetrize
AutoBZCore.symmetrize_
```

If you are an expert on the symmetry representations of Wannier orbitals or
Bloch bands and can offer assistance in automating the IBZ to FBZ mapping,
please contact the developers by opening a Github issue or pr.
