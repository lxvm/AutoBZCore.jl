# utilities
function lattice_bz_limits(B::AbstractMatrix)
    d = checksquare(B)
    CubicLimits(zeros(d), ones(d))
end
function check_bases_canonical(A::AbstractMatrix, B::AbstractMatrix, atol)
    norm(A'B - 2pi*I) < atol || throw("Real and reciprocal Bravais lattice bases non-orthogonal to tolerance $atol")
end
canonical_reciprocal_basis(A::AbstractMatrix) = A' \ (2pi*one(A))

# main data type
"""
    SymmetricBZ(A, B, lims::AbstractLimits, syms; atol=sqrt(eps()))

Data type representing a Brillouin zone reduced by a set of symmetries, `syms`
with iterated integration limits `lims`, both of which are assumed to be in the
lattice basis (since the Fourier series is). `A` and `B` should be
identically-sized square matrices containing the real and reciprocal basis
vectors in their columns. `lims` should be limits compatible with the
`IteratedIntegration` package that represent the BZ in fractional lattice
coordinates (e.g. the full BZ with vertices (0,0,0) & (1,1,1)). `syms` should be
a collection of symmetries compatible with `AutoSymPTR` and the symmetry
operators should be in the lattice basis (if necessary, rotate them from the
Cartesian basis).
"""
struct SymmetricBZ{S,L,d,T,d2}
    A::SMatrix{d,d,T,d2}
    B::SMatrix{d,d,T,d2}
    lims::L
    syms::S
    SymmetricBZ(A::M, B::M, lims::L, syms::S) where {d,T,d2,M<:SMatrix{d,d,T,d2},L,S} =
        new{S,L,d,T,d2}(A, B, lims, syms)
end

# eventually limits could be computed from B and symmetries
function SymmetricBZ(A::AbstractMatrix{T}, B::AbstractMatrix{S}, lims, syms; atol=nothing) where {T,S}
    F = float(promote_type(T, S))
    (d = checksquare(A)) == checksquare(B) ||
        throw(DimensionMismatch("Bravais lattices $A and $B must have the same shape"))
    check_bases_canonical(A, B, something(atol, sqrt(eps(F))))
    M = SMatrix{d,d,F,d^2}
    SymmetricBZ(convert(M, A), convert(M, B), lims, syms)
end

nsyms(bz::SymmetricBZ) = length(bz.syms)
Base.ndims(::SymmetricBZ{S,L,d}) where {S,L,d} = d
Base.eltype(::Type{<:SymmetricBZ{S,L,d,T}}) where {S,L,d,T} = T

# Define traits for symmetrization based on symmetry representations

abstract type AbstractSymRep end
abstract type FaithfulRep <: AbstractSymRep end

struct UnknownRep <: AbstractSymRep end
struct TrivialRep <: AbstractSymRep end
struct LatticeRep <: FaithfulRep end

SymRep(::Any) = UnknownRep()
const TrivialRepType = Union{Number,AbstractArray{<:Any,0}}

"""
    symmetrize(f, ::SymmetricBZ, xs...)
    symmetrize(f, ::SymmetricBZ, x::Number)

Transform `x` by the symmetries of the parametrization used to reduce the
domain, thus mapping the value of `x` on the parametrization to the full domain.
"""
symmetrize(f, bz::SymmetricBZ, xs...) = map(x -> symmetrize(f, bz, x), xs)
symmetrize(f, bz::SymmetricBZ, x) = symmetrize_(SymRep(f), bz, x)
symmetrize(f, bz::SymmetricBZ, x::TrivialRepType) =
    symmetrize_(TrivialRep(), bz, x)
symmetrize_(::TrivialRep, bz::SymmetricBZ, x) = nsyms(bz)*x
function symmetrize_(::LatticeRep, bz::SymmetricBZ, x::AbstractVector)
    r = zero(x)
    for S in bz.syms
        r += S * x
    end
    r
end
function symmetrize_(::LatticeRep, bz::SymmetricBZ, x::AbstractMatrix)
    r = zero(x)
    for S in bz.syms
        r += S * x * S'
    end
    r
end
function symmetrize_(::UnknownRep, ::SymmetricBZ, x)
    @warn "Symmetric BZ detected but the integrand's symmetry representation is unknown. Define a trait for your integrand by extending SymRep"
    x
end


"""
    FullBZ(A, B=canonical_reciprocal_basis(A); atol=sqrt(eps()))

Constructs a [`SymmetricBZ`](@ref) with trivial symmetries.
"""
FullBZ(A, B=canonical_reciprocal_basis(A), lims=lattice_bz_limits(B); kwargs...) =
    SymmetricBZ(A, B, lims, nothing; kwargs...)

const FullBZType = SymmetricBZ{Nothing}
nsyms(::FullBZType) = 1
symmetrize(_, ::FullBZType, x) = x
symmetrize(_, ::FullBZType, x::TrivialRepType) = x
