module WannierIOExt

using WannierIO
using AutoBZCore
import AutoBZCore: AbstractBZ, load_bz

"""
    load_autobz(::AbstractBZ, filename; kws...)

Automatically load a BZ using data from a "seedname.wout" file.
"""
function load_bz(bz::AbstractBZ, filename::String; atol=1e-5)
    out = WannierIO.read_wout(filename)
    return load_bz(bz, out.lattice, out.recip_lattice; atol=atol)
end
load_bz(filename::String; kws...) = load_bz(FBZ(), filename; kws...)
function load_bz(bz::IBZ, filename::String; kws...)
    out = WannierIO.read_wout(filename)
    atom_pos = reinterpret(reshape, eltype(eltype(out.atom_positions)), out.atom_positions)
    return load_bz(convert(AbstractBZ{3}, bz), out.lattice, out.atom_labels, atom_pos; kws..., coordinates="lattice")
end

end
