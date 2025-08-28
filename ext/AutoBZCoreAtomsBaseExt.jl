module AutoBZCoreAtomsBaseExt

using StaticArrays: SMatrix

using AtomsBase
using AutoBZCore: AbstractBZ, FBZ, IBZ, canonical_reciprocal_basis
import AutoBZCore: load_bz

"""
    load_bz(::AbstractBZ, ::AbstractSystem; kws...)

Automatically load a BZ using data from AtomsBase.jl-compatible `AbstractSystem`.
"""
function load_bz(bz::AbstractBZ, system::AbstractSystem)
    @assert all(periodicity(system))
    bz_ = convert(AbstractBZ{n_dimensions(system)}, bz)
    A = stack(cell_vectors(system))
    return load_bz(bz_, A)
end
load_bz(system::AbstractSystem) = load_bz(FBZ(), system)
function load_bz(bz::IBZ, system::AbstractSystem; kws...)
    @assert all(periodicity(system))
    d = n_dimensions(system)
    bz_ = convert(AbstractBZ{d}, bz)
    A = SMatrix{d,d}(stack(cell_vectors(system)))
    B = canonical_reciprocal_basis(A)
    species = atomic_symbol(system, :)
    pos = position(system, :)
    atom_pos = stack(pos)
    return load_bz(bz_, A, B, species, atom_pos; kws..., coordinates="Cartesian")
end

end
