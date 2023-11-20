"""
    PuncturedInterval(s)

Represent an interval `(a, b)` with interior points deleted by `s = (a, c1, ..., cN, b)`, so
that the integration algorithm can avoid the points `c1, ..., cN` for e.g. discontinuities.
`s` must be a tuple or vector.
"""
struct PuncturedInterval{T,S}
    s::S
    PuncturedInterval(s::S) where {N,S<:NTuple{N}} = new{eltype(s),S}(s)
    PuncturedInterval(s::S) where {T,S<:AbstractVector{T}} = new{T,S}(s)
end
Base.eltype(::Type{PuncturedInterval{T,S}}) where {T,S} = T
Base.ndims(::PuncturedInterval) = 1
segments(p::PuncturedInterval) = p.s
endpoints(p::PuncturedInterval) = (p.s[begin], p.s[end])

"""
    HyperCube(a, b)

Represents a hypercube spanned by the vertices `a, b`, which must be iterables of the same length.
"""
struct HyperCube{d,T}
    a::SVector{d,T}
    b::SVector{d,T}
end
function HyperCube(a::NTuple{d}, b::NTuple{d}) where {d}
    F = promote_type(eltype(a), eltype(b))
    return HyperCube{d,F}(SVector{d,F}(a), SVector{d,F}(b))
end
HyperCube(a, b) = HyperCube(promote(a...), promote(b...))
Base.eltype(::Type{HyperCube{d,T}}) where {d,T} = T
Base.ndims(::HyperCube{d}) where {d} = d
endpoints(c::HyperCube) = (c.a, c.b)
