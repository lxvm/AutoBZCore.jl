module SymmetryReduceBZExt

using LinearAlgebra
using Polyhedra: Polyhedron, polyhedron, doubledescription, hrepiscomputed, hrep
using StaticArrays

using SymmetryReduceBZ
using AutoBZCore: canonical_reciprocal_basis, SymmetricBZ, IBZ, DefaultPolyhedron,
    CubicLimits, AbstractIteratedLimits, load_limits
import AutoBZCore: IteratedIntegration.fixandeliminate, IteratedIntegration.segments


include("ibzlims.jl")

function get_segs(vert::AbstractMatrix)
    rtol = atol = sqrt(eps(eltype(vert)))
    uniquepts=Vector{eltype(vert)}(undef, size(vert, 1))
    numpts = 0
    for i in axes(vert,1)
        v = vert[i,end]
        test = isapprox(v, atol=atol, rtol=rtol)
        if !any(test, @view(uniquepts[begin:begin+numpts-1,end]))
            numpts += 1
            uniquepts[numpts] = v
        end
    end
    @assert numpts >= 2 uniquepts
    resize!(uniquepts,numpts)
    sort!(uniquepts)
    return uniquepts
end

struct Polyhedron3{T<:Real} <: AbstractIteratedLimits{3,T}
    face_coord::Vector{Matrix{T}}
    segs::Vector{T}
end
function segments(ph::Polyhedron3, dim)
    @assert dim == 3
    return ph.segs
end

struct Polygon2{T<:Real} <: AbstractIteratedLimits{2,T}
    vert::Matrix{T}
    segs::Vector{T}
end
function segments(pg::Polygon2, dim)
    @assert dim == 2
    return pg.segs
end

function fixandeliminate(ph::Polyhedron3, z, ::Val{3})
    pg_vert = pg_vert_from_zslice(z, ph.face_coord)
    segs = get_segs(pg_vert)
    return Polygon2(pg_vert, segs)
end
function fixandeliminate(pg::Polygon2, y, ::Val{2})
    return CubicLimits(xlim_from_yslice(y, pg.vert)...)
end

function (::IBZ{n,Polyhedron})(real_latvecs, atom_types, atom_pos, coordinates; ibzformat="half-space", makeprim=false, convention="ordinary") where {n}
    hull_cart = calc_ibz(real_latvecs, atom_types, atom_pos, coordinates, ibzformat, makeprim, convention)
    hull = real_latvecs' * polyhedron(doubledescription(hull_cart)) # rotate Cartesian basis to lattice basis in reciprocal coordinates
    hrepiscomputed(hull) || hrep(hull) # precompute hrep if it isn't already
    return load_limits(hull)
end

function (::IBZ{3,DefaultPolyhedron})(real_latvecs, atom_types, atom_pos, coordinates; ibzformat="convex hull", makeprim=false, convention="ordinary", digits=12)
    hull = calc_ibz(real_latvecs, atom_types, atom_pos, coordinates, ibzformat, makeprim, convention)
    tri_idx = hull.simplices
    ph_vert = hull.points * real_latvecs
    # tidy_vertices!(ph_vert, digits) # this should take care of rounding errors
    face_idx = faces_from_triangles(tri_idx, ph_vert)
    # face_idx = SymmetryReduceBZ.Utilities.get_uniquefacets(hull)
    face_coord = face_coord_from_idx(face_idx, ph_vert)
    segs = get_segs(ph_vert)
    return Polyhedron3(face_coord, segs)
end

fixsign(x) = iszero(x) ? abs(x) : x
function tidy_vertices!(points, digits)
    for (i, p) in enumerate(points)
        points[i] = fixsign(round(p; digits=digits))
    end
    return points
end

"""
    load_ibz(::IBZ, A, B, species, positions; coordinates="lattice", rtol=nothing, atol=1e-9, digits=12)

Use `SymmetryReduceBZ` to automatically load the IBZ. Since this method lives in
an extension module, make sure you write `using SymmetryReduceBZ` before `using
AutoBZ`.
"""
function load_ibz(bz::IBZ{N}, A::SMatrix{N,N}, B::SMatrix{N,N}, species::AbstractVector, positions::AbstractMatrix;
    coordinates="lattice", rtol=nothing, atol=1e-9, digits=12) where {N}
    # we need to convert arguments to unit-free since SymmetryReduceBZ doesn't support them
    # and our limits objects must be unitless
    real_latvecs = A / oneunit(eltype(A))
    atom_species = unique(species)
    atom_types = map(e -> findfirst(==(e), atom_species) - 1, species)
    atom_pos = positions / oneunit(eltype(positions))
    # get symmetries
    sg = SymmetryReduceBZ.Symmetry.calc_spacegroup(real_latvecs, atom_types, atom_pos, coordinates)
    pg_ = SymmetryReduceBZ.Utilities.remove_duplicates(sg[2], rtol=something(rtol, sqrt(eps(float(maximum(real_latvecs))))), atol=atol)
    pg = Ref(real_latvecs') .* pg_ .* Ref(inv(real_latvecs')) # rotate operator from Cartesian basis to lattice basis in reciprocal coordinates
    syms = convert(Vector{SMatrix{3,3,Float64,9}}, pg) # deal with type instability in SymmetryReduceBZ
    map!(s -> fixsign.(round.(s, digits=digits)), syms, syms)   # clean up matrix elements
    # get convex hull
    hull = bz(real_latvecs, atom_types, atom_pos, coordinates)
    # now limits and symmetries should be in reciprocal coordinates in the lattice basis
    return SymmetricBZ(A, B, hull, syms)
end

end
