endpoints(dom) = (first(dom), last(dom))
breakpoints(dom) = dom[begin+1:end-1] # or Iterators.drop(Iterators.take(dom, length(dom)-1), 1)
_segments(dom) = dom
function get_prototype(dom)
    a, b, = dom
    return (a+b)/2
end

function get_prototype(B::Basis)
    return B * zero(SVector{ndims(B),typeof(float(one(eltype(B))))})
end

get_basis(B::Basis) = B
get_basis(B::AbstractMatrix) = Basis(B)

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
PuncturedInterval(s::PuncturedInterval) = s
Base.eltype(::Type{PuncturedInterval{T,S}}) where {T,S} = T
Base.ndims(::PuncturedInterval) = 1
_segments(p::PuncturedInterval) = p.s
endpoints(p::PuncturedInterval) = endpoints(p.s)
breakpoints(p::PuncturedInterval) = breakpoints(p.s)
function get_prototype(p::PuncturedInterval)
    a, b, = _segments(p)
    return (a + b)/2
end
load_limits(p::PuncturedInterval) = CubicLimits(endpoints(p)...)


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
function get_prototype(p::HyperCube)
    a, b = endpoints(p)
    return (a + b)/2
end

get_prototype(l::AbstractIteratedLimits) = interior_point(l)

load_limits(c::HyperCube) = CubicLimits(endpoints(c)...)