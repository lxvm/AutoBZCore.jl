module AtomsBaseExt

using StaticArrays: SMatrix

using AtomsBase
using AutoBZCore: AbstractBZ, FBZ, IBZ, canonical_reciprocal_basis
import AutoBZCore: load_bz

"""
    load_bz(::AbstractBZ, ::AbstractSystem; kws...)

Automatically load a BZ using data from AtomsBase.jl-compatible `AbstractSystem`.
"""
function load_bz(bz::AbstractBZ, system::AbstractSystem)
    @assert all(==(Periodic()), boundary_conditions(system))
    bz_ = convert(AbstractBZ{n_dimensions(system)}, bz)
    bb = bounding_box(system)
    A = reinterpret(reshape, eltype(eltype(bb)), bb)
    return load_bz(bz_, A)
end
load_bz(system::AbstractSystem) = load_bz(FBZ(), system)
function load_bz(bz::IBZ, system::AbstractSystem; kws...)
    @assert all(==(Periodic()), boundary_conditions(system))
    d = n_dimensions(system)
    bz_ = convert(AbstractBZ{d}, bz)
    bb = bounding_box(system)
    A = SMatrix{d,d}(reinterpret(reshape, eltype(eltype(bb)), bb))
    B = canonical_reciprocal_basis(A)
    species = atomic_symbol(system)
    pos = position(system)
    atom_pos = reinterpret(reshape, eltype(eltype(pos)), pos)
    return load_bz(bz_, A, B, species, atom_pos; kws..., coordinates="Cartesian")
end

end
