module WannierIOExt

using WannierIO
using AutoBZCore: AbstractBZ, FBZ, IBZ
import AutoBZCore: load_bz

"""
    load_bz(::AbstractBZ, filename; kws...)

Automatically load a BZ using data from a "seedname.wout" file.
"""
function load_bz(bz::AbstractBZ, filename::String; atol=1e-5)
    out = WannierIO.read_wout(filename)
    bz_ = convert(AbstractBZ{3}, bz)
    return load_bz(bz_, out.lattice, out.recip_lattice; atol=atol)
end
load_bz(filename::String; kws...) = load_bz(FBZ(), filename; kws...)
function load_bz(bz::IBZ, filename::String; kws...)
    out = WannierIO.read_wout(filename)
    bz_ = convert(AbstractBZ{3}, bz)
    atom_pos = reinterpret(reshape, eltype(eltype(out.atom_positions)), out.atom_positions)
    return load_bz(bz_, out.lattice, out.recip_lattice, out.atom_labels, atom_pos; kws..., coordinates="lattice")
end

end
