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
    SymmetricBZ(A, B, lims::AbstractLimits, syms=nothing; atol=sqrt(eps()))

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
function SymmetricBZ(A::AbstractMatrix{T}, B::AbstractMatrix{S}, lims, syms=nothing; atol=nothing) where {T,S}
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

"""
    symmetrize(f, ::SymmetricBZ, xs...)
    symmetrize(f, ::SymmetricBZ, x::Number)

Tranform `x` by the symmetries of the parametrization used to reduce the
domain, thus mapping the value of `x` on the parametrization to the full domain.
"""
symmetrize(f, bz::SymmetricBZ, xs...) = map(x -> symmetrize(f, bz, x), xs)
symmetrize(_, bz::SymmetricBZ, x::Number) = nsyms(bz)*x
function symmetrize(f, ::SymmetricBZ, x)
    @warn "Symmetric BZ detected and returning integral computed from limits. Define a method for symmetrize() for your integrand type that maps to the full BZ value"
    @show f
    x
end


"""
    FullBZ(A, B=canonical_reciprocal_basis(A), lims=lattice_bz_limits(B); atol=sqrt(eps()))

A type alias for `SymmetricBZ{Nothing}` when there are no symmetries applied to BZ
"""
const FullBZ = SymmetricBZ{Nothing}
FullBZ(A, B=canonical_reciprocal_basis(A), lims=lattice_bz_limits(B); kwargs...) =
    SymmetricBZ(A, B, lims; kwargs...)

nsyms(::FullBZ) = 1
symmetrize(_, ::FullBZ, x) = x
symmetrize(_, ::FullBZ, x::Number) = x

# TODO: In Julia 1.9 put these definitions in extensions modules

"""
    iterated_integration(f, bz::SymmetricBZ; kwargs...)
"""
function iterated_integration(f::F, bz::BZ; kwargs...) where {F,BZ<:SymmetricBZ}
    kw = iterated_integration_kwargs(f, bz; kwargs...)
    j = det(bz.B)
    atol = kw.atol/nsyms(bz)/j # reduce absolute tolerance by symmetry factor
    int, err = iterated_integration(f, bz.lims; kw..., atol=atol)
    symmetrize(f, bz, j*int, j*err)
end
iterated_integration_kwargs(f, bz::SymmetricBZ; kwargs...) =
    iterated_integration_kwargs(f, bz.lims; kwargs...)

quad_args(::typeof(iterated_integration), f, l) = (f, l)
quad_kwargs(::typeof(iterated_integration), f, l; kwargs...) =
    iterated_integration_kwargs(f, l; kwargs...)

"""
    symptr(f, bz::SymmetricBZ; kwargs...)
"""
function symptr(f, bz::SymmetricBZ; kwargs...)
    int, rules = symptr(f, bz.B, bz.syms; kwargs...)
    symmetrize(f, bz, int), rules
end
symptr_kwargs(f, bz::SymmetricBZ; kwargs...) =
    symptr_kwargs(f, bz.B, bz.syms; kwargs...)

quad_args(::typeof(symptr), f, bz) = (f, bz)
quad_kwargs(::typeof(symptr), f, bz; kwargs...) =
    symptr_kwargs(f, bz; kwargs...)

"""
    autosymptr(f, bz::SymmetricBZ; kwargs...)
"""
function autosymptr(f, bz::SymmetricBZ; kwargs...)
    int, err, numevals, rules = autosymptr(f, bz.B, bz.syms; kwargs...)
    symmetrize(f, bz, int, err)..., numevals, rules
end
autosymptr_kwargs(f, bz::SymmetricBZ; kwargs...) =
    autosymptr_kwargs(f, bz.B, bz.syms; kwargs...)

quad_args(::typeof(autosymptr), f, bz) = (f, bz)
quad_kwargs(::typeof(autosymptr), f, bz; kwargs...) =
    autosymptr_kwargs(f, bz; kwargs...)
