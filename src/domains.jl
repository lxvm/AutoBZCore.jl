abstract type AbstractDomain end

"""
    pointtype(domain::AbstractDomain)

Return the type of points in the domain (e.g. Float or SVector)
"""
function pointtype end
pointtype(l::AbstractIteratedLimits) = typeof(interior_point(l))

"""
    asdomain(alg::IntegralAlgorithm, domain::AbstractDomain)

Transform a domain representation into one suitable for the given algorithm.
"""
function asdomain end

struct DomainWrapper{T} <: AbstractDomain
    domain::T
end

struct PuncturedInterval{T,S}
    s::S
    PuncturedInterval(s::S) where {N,T,S<:NTuple{N,T}} = new{T,S}(s)
    PuncturedInterval(s::S) where {T,S<:AbstractVector{T}} = new{T,S}(s)
end
Base.eltype(::Type{PuncturedInterval{T,S}}) where {T,S} = T
segments(p::PuncturedInterval) = p.s
endpoints(p::PuncturedInterval) = (p.s[begin], p.s[end])
pointtype(p::PuncturedInterval) = float(eltype(p))

struct HyperCube{d,T}
    a::SVector{d,T}
    b::SVector{d,T}
end
function HyperCube(a::NTuple{d,T}, b::NTuple{d,S}) where {d,T,S}
    F = promote_type(T,S)
    return HyperCube{d,F}(SVector{d,F}(a), SVector{d,F}(b))
end
HyperCube(a, b) = HyperCube(promote(a...), promote(b...))
Base.eltype(::Type{HyperCube{d,T}}) where {d,T} = T

endpoints(c::HyperCube) = (c.a, c.b)
pointtype(::HyperCube{d,T}) where {d,T} = SVector{d,float(T)}
