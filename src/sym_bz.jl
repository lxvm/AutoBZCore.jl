# utilities
function basis_to_limits(B::SMatrix{d,d}) where d
    half_b = SVector{d}(ntuple(n -> norm(B[:,n])/2, Val{d}()))
    CubicLimits(-half_b, half_b)
end
function check_bases_canonical(A::AbstractMatrix, B::AbstractMatrix, atol)
    norm(A'B - 2pi*I) < atol || throw("Real and reciprocal Bravais lattice bases non-orthogonal to tolerance $atol")
end
canonical_reciprocal_basis(A::AbstractMatrix) = A' \ (2pi*one(A))

# main data type
"""
    SymmetricBZ(A, B, lims::AbstractLimits, syms=nothing; atol=sqrt(eps()))

Data type representing a Brillouin zone reduced by a set of symmetries, `syms`
with iterated integration limits `lims`, both of which are assumed to be in the
lattice basis (since the Fourier series is). `A` and `B` should be
identically-sized square matrices containing the real and reciprocal basis
vectors in their columns.
"""
struct SymmetricBZ{S,L,d,T,d2}
    A::SMatrix{d,d,T,d2}
    B::SMatrix{d,d,T,d2}
    lims::L
    syms::S
    SymmetricBZ(A::M, B::M, lims::L, syms::S) where {d,T,d2,M<:SMatrix{d,d,T,d2},L<:AbstractLimits{d},S} =
        new{S,L,d,T,d2}(A, B, lims, syms)
end

# eventually limits could be computed from B and symmetries
function SymmetricBZ(A::AbstractMatrix{T}, B::AbstractMatrix{S}, lims, syms=nothing; atol=nothing) where {T,S}
    F = float(promote_type(T, S))
    (d = LinearAlgebra.checksquare(A)) == LinearAlgebra.checksquare(B) ||
        throw(DimensionMismatch("Bravais lattices $A and $B must have the same shape"))
    check_bases_canonical(A, B, something(atol, sqrt(eps(F))))
    M = SMatrix{d,d,F,d^2}
    SymmetricBZ(convert(M, A), convert(M, B), lims, syms)
end

nsyms(bz::SymmetricBZ) = length(bz.syms)
symmetries(bz::SymmetricBZ) = bz.syms
limits(bz::SymmetricBZ) = bz.lims
Base.ndims(::SymmetricBZ{S,L,d}) where {S,L,d} = d
Base.eltype(::Type{<:SymmetricBZ{S,L,d,T}}) where {S,L,d,T} = T

"""
    symmetrize(f, ::SymmetricBZ, xs...)
    symmetrize(f, ::SymmetricBZ, x::Number)

Tranform `x` by the symmetries of the parametrization used to reduce the
domain, thus mapping the value of `x` on the parametrization to the full domain.
"""
symmetrize(f, bz::SymmetricBZ, xs...) = map(x -> symmetrize(f, bz, x), xs)
symmetrize(_, bz::SymmetricBZ, x::Number) = nsyms(bz)*x
function symmetrize(_, ::SymmetricBZ, x)
    @warn "Symmetric BZ detected and returning integral computed from limits. Define a method for symmetrize() for your integrand type that maps to the full BZ value"
    x
end



"""
    FullBZ(A, B, lims; atol=sqrt(eps()))

A type alias for `SymmetricBZ{Nothing}` when there are no symmetries applied to BZ
"""
const FullBZ = SymmetricBZ{Nothing}
FullBZ(A, B, lims; kwargs...) = SymmetricBZ(A, B, lims; kwargs...)

nsyms(::FullBZ) = 1
symmetries(::FullBZ) = nothing
limits(bz::FullBZ) = bz.lims
Base.convert(::Type{<:FullBZ}, fbz::FullBZ) = fbz
symmetrize(_, ::FullBZ, x) = x
symmetrize(_, ::FullBZ, x::Number) = x