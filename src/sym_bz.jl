# utilities
function basis_to_limits(B::SMatrix{d,d}) where d
    half_b = SVector{d}(ntuple(n -> norm(B[:,n])/2, Val{d}()))
    CubicLimits(-half_b, half_b)
end
function check_bases_canonical(A::AbstractMatrix, B::AbstractMatrix, atol)
    norm(A'B - 2pi*I) < atol || throw("Real and reciprocal Bravais lattice bases non-orthogonal to tolerance $atol")
end
canonical_reciprocal_basis(A::T) where {T<:AbstractMatrix} = A' \ (2pi*one(T))

# main data type
"""
    SymmetricBZ(A, B, lims, syms=nothing)

Data type representing a Brillouin zone reduced by a set of symmetries, with
integration limits `lims`
"""
struct SymmetricBZ{basis,S,L,d,T,d2}
    A::SMatrix{d,d,T,d2}
    B::SMatrix{d,d,T,d2}
    lims::L
    syms::S
    SymmetricBZ{basis}(A::M, B::M, lims::L, syms::S) where {basis,d,T,d2,M<:SMatrix{d,d,T,d2},L<:AbstractLimits{d},S} =
        new{basis,S,L,d,T,d2}(A, B, lims, syms)
end

function SymmetricBZ(A::AbstractMatrix{T}, B::AbstractMatrix{S}, lims, syms=nothing; atol=sqrt(eps()), basis=:lattice) where {T,S}
    F = float(promote_type(T, S))
    (d = LinearAlgebra.checksquare(A)) == LinearAlgebra.checksquare(B) ||
        throw(DimensionMismatch("Bravais lattices $A and $B must have the same shape"))
    check_bases_canonical(A, B, something(atol, sqrt(eps(F))))
    M = SMatrix{d,d,F,d^2}
    SymmetricBZ{basis}(convert(M, A), convert(M, B), lims, syms)
end

nsyms(l::SymmetricBZ) = length(l.syms)
symmetries(l::SymmetricBZ) = l.syms
limits(bz::SymmetricBZ) = bz.lims
Base.ndims(::SymmetricBZ{d}) where d = d
Base.eltype(::Type{<:SymmetricBZ{d,T}}) where {d,T} = T
basis(::SymmetricBZ{d,T,b}) where {d,T,b} = b

"""
    symmetrize(f, ::SymmetricBZ, xs...)
    symmetrize(f, ::SymmetricBZ, x::Number)

Tranform `x` by the symmetries of the parametrization used to reduce the
domain, thus mapping the value of `x` on the parametrization to the full domain.

"""
symmetrize(f, l::SymmetricBZ, xs...) = map(x -> symmetrize(f, l, x), xs)
symmetrize(_, l::SymmetricBZ, x::Number) = nsyms(l)*x
function symmetrize(_, ::SymmetricBZ, x)
    @warn "Specialized BZ detected and returning integral computed from limits. Define a method for symmetrize() for your integrand type that maps to the full BZ value"
    x
end



# Define FullBZ as a type alias
const FullBZ{basis} = SymmetricBZ{basis,Nothing}

nsyms(::FullBZ) = 1
symmetries(::FullBZ) = tuple(I)
limits(bz::FullBZ) = bz.lims
Base.convert(::Type{<:FullBZ}, fbz::FullBZ) = fbz
symmetrize(_, ::FullBZ, x) = x
symmetrize(_, ::FullBZ, x::Number) = x

Base.convert(::Type{<:FullBZ{basis}}, bz::SymmetricBZ) where basis = FullBZ{basis}(bz.A, bz.B, basis_to_limits(bz.B))
