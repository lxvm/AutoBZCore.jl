module AutoBZCoreSymmetryReduceBZExt

using LinearAlgebra
using Polyhedra: Polyhedron, polyhedron, doubledescription, hrepiscomputed, hrep
using StaticArrays

using SymmetryReduceBZ
using AutoBZCore: canonical_reciprocal_basis, SymmetricBZ, IBZ, DefaultPolyhedron,
    CubicLimits, AbstractIteratedLimits, load_limits
import IteratedIntegration: fixandeliminate, segments, eliminate

include("ibzlims.jl")

function get_segs(vert::AbstractMatrix, dim=size(vert, 2))
    rtol = atol = sqrt(eps(eltype(vert)))
    uniquepts=Vector{eltype(vert)}(undef, size(vert, 1))
    numpts = 0
    for i in axes(vert,1)
        v = vert[i,dim]
        test = isapprox(v, atol=atol, rtol=rtol)
        if !any(test, @view(uniquepts[begin:begin+numpts-1]))
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
    segs3::Vector{T}
    segs2::Vector{T}
    segs1::Vector{T}
end
function segments(ph::Polyhedron3, dim)
    if dim == 3
        return ph.segs3
    elseif dim == 2
        return ph.segs2
    elseif dim == 1
        return ph.segs1
    else
        error("dim must be 1, 2, or 3")
    end
end

struct Polygon2{T<:Real} <: AbstractIteratedLimits{2,T}
    vert::Matrix{T}
    segs2::Vector{T}
    segs1::Vector{T}
end
function segments(pg::Polygon2, dim)
    if dim == 2
        return pg.segs2
    elseif dim == 1
        return pg.segs1
    else
        error("dim must be 1 or 2")
    end
end

function fixandeliminate(ph::Polyhedron3, z, ::Val{3})
    pg_vert = pg_vert_from_zslice(z, ph.face_coord)
    segs2 = get_segs(pg_vert, 2)
    segs1 = get_segs(pg_vert, 1)
    return Polygon2(pg_vert, segs2, segs1)
end
function fixandeliminate(pg::Polygon2, y, ::Val{2})
    return CubicLimits(xlim_from_yslice(y, pg.vert)...)
end

function eliminate(ph::Polyhedron3{T}, dims::Int) where {T<:Real}
    idx = setdiff(1:3, dims)
    @assert length(idx) == 2
    vert = tidy_vertices!(reduce(vcat, (view(v, :, idx) for v in ph.face_coord)), floor(Int, abs(log10(eps(T))))-1) |> eachrow |> unique |> stack |> permutedims
    if dims == 1
        Polygon2(vert, ph.segs3, ph.segs2)
    elseif dims == 2
        Polygon2(vert, ph.segs3, ph.segs1)
    elseif dims == 3
        Polygon2(vert, ph.segs2, ph.segs1)
    else
        error("dims must be 1, 2, or 3")
    end
end
function eliminate(ph::Polyhedron3, dims)
    1 in dims && 2 in dims && 3 in dims && return nothing
    length(dims) == 1 && return eliminate(ph, only(dims))
    length(dims) == 2 && return CubicLimits(extrema(segments(ph, only(setdiff(1:3, dims))))...)
    error("dims must be 1, 2, or 3")
end
function eliminate(pg::Polygon2, dims::Int)
    CubicLimits(segments(pg, dims)...)
end
function eliminate(pg::Polygon2, dims)
    1 in dims && 2 in dims && return nothing
    length(dims) == 1 && return eliminate(pg, only(dims))
    error("dims must be 1 or 2")
end


function (::IBZ{n,Polyhedron})(real_latvecs, atom_types, atom_pos, coordinates; makeprim=false, convention="ordinary") where {n}
    ibz_cart = calc_ibz(real_latvecs, atom_types, atom_pos, coordinates, makeprim, convention)
    ibz_lat = real_latvecs' * ibz_cart # rotate Cartesian basis to lattice basis in reciprocal coordinates
    hrepiscomputed(ibz_lat) || hrep(ibz_lat) # precompute hrep if it isn't already
    return load_limits(ibz_lat)
end

function (::IBZ{3,DefaultPolyhedron})(real_latvecs, atom_types, atom_pos, coordinates; makeprim=false, convention="ordinary")
    ibz_cart = calc_ibz(real_latvecs, atom_types, atom_pos, coordinates, makeprim, convention)
    # tri_idx = hull.simplices
    # ph_vert = hull.points * real_latvecs
    # face_idx = faces_from_triangles(tri_idx, ph_vert)
    # face_coord = face_coord_from_idx(face_idx, ph_vert)
    ibz_lat = real_latvecs' * ibz_cart
    ph_vert = permutedims(reduce(hcat, SymmetryReduceBZ.Utilities.vertices(ibz_lat)))
    face_coord = map(x -> permutedims(reduce(hcat, x)), SymmetryReduceBZ.Utilities.get_uniquefacets(ibz_lat))
    segs3 = get_segs(ph_vert, 3)
    segs2 = get_segs(ph_vert, 2)
    segs1 = get_segs(ph_vert, 1)
    return Polyhedron3(face_coord, segs3, segs2, segs1)
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
