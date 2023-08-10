# utilities
function lattice_bz_limits(B::AbstractMatrix)
    T = SVector{checksquare(B),typeof(one(eltype(B)))}
    CubicLimits(zero(T), ones(T))   # unitless canonical bz
end
function check_bases_canonical(A::AbstractMatrix, B::AbstractMatrix, atol)
    norm(A'B - 2pi*I) < atol || throw("Real and reciprocal Bravais lattice bases non-orthogonal to tolerance $atol")
end
canonical_reciprocal_basis(A::AbstractMatrix) = A' \ (2pi*one(A))

# main data type
"""
    SymmetricBZ(A, B, lims::AbstractIteratedLimits, syms; atol=sqrt(eps()))

Data type representing a Brillouin zone reduced by a set of symmetries, `syms`
with iterated integration limits `lims`, both of which are assumed to be in the
lattice basis (since the Fourier series is). `A` and `B` should be
identically-sized square matrices containing the real and reciprocal basis
vectors in their columns.

!!! note "Convention"
    This type assumes all integration limit data is in the reciprocal lattice
    basis with fractional coordinates, where the FBZ is just the hypercube
    spanned by the vertices (0,…,0) & (1,…,1). If necessary, use `A` or `B` to
    rotate these quantities into the convention.

`lims` should be limits compatible with
[IteratedIntegration.jl](https://github.com/lxvm/IteratedIntegration.jl).
`syms` should be an iterable collection of point group symmetries compatible
with [AutoSymPTR.jl](https://github.com/lxvm/AutoSymPTR.jl).
"""
struct SymmetricBZ{S,L,d,TA,TB,d2}
    A::SMatrix{d,d,TA,d2}
    B::SMatrix{d,d,TB,d2}
    lims::L
    syms::S
    function SymmetricBZ(A::MA, B::MB, lims::L, syms::S) where {d,TA,TB,d2,MA<:SMatrix{d,d,TA,d2},MB<:SMatrix{d,d,TB,d2},L,S}
        return new{S,L,d,TA,TB,d2}(A, B, lims, syms)
    end
end

nsyms(bz::SymmetricBZ) = length(bz.syms)

const FullBZ = SymmetricBZ{Nothing}
nsyms(::FullBZ) = 1

# Define traits for symmetrization based on symmetry representations

"""
    AbstractSymRep

Abstract supertype of symmetry representation traits.
"""
abstract type AbstractSymRep end

"""
    UnknownRep()

Fallback symmetry representation for array types without a user-defined `SymRep`.
"""
struct UnknownRep <: AbstractSymRep end

"""
    TrivialRep()

Symmetry representation of objects with trivial transformation under the group.
"""
struct TrivialRep <: AbstractSymRep end


"""
    SymRep(f)

`SymRep` specifies the symmetry representation of the integral of the function
`f`. When you define a new integrand, you can choose to implement this trait to
specify how the integral is transformed under the symmetries of the lattice in
order to map the integral of `f` on the IBZ to the result for the FBZ.

New types for `SymRep` should also extend a corresponding method for
[`AutoBZCore.symmetrize_`](@ref).
"""
SymRep(::Any) = UnknownRep()
const TrivialRepType = Union{Number,AbstractArray{<:Any,0}}

"""
    symmetrize(f, ::SymmetricBZ, xs...)
    symmetrize(f, ::SymmetricBZ, x::Union{Number,AbstractArray{<:Any,0}})

Transform `x` by the symmetries of the parametrization used to reduce the
domain, thus mapping the value of `x` on the parametrization to the full domain.
"""
symmetrize(f, bz, xs...) = map(x -> symmetrize(f, bz, x), xs)
symmetrize(f, bz, x) = symmetrize_(SymRep(f), bz, x)
symmetrize(f, bz, x::TrivialRepType) =
    symmetrize_(TrivialRep(), bz, x)

"""
    symmetrize_(rep::AbstractSymRep, bz::SymmetricBZ, x)

Transform `x` under representation `rep` using the symmetries in `bz` to obtain
the result of an integral on the FBZ from `x`, which was calculated on the IBZ.
"""
symmetrize_(::TrivialRep, bz::SymmetricBZ, x) = nsyms(bz)*x
function symmetrize_(::UnknownRep, ::SymmetricBZ, x)
    @warn "Symmetric BZ detected but the integrand's symmetry representation is unknown. Define a trait for your integrand by extending SymRep"
    x
end

symmetrize(_, ::FullBZ, x) = x
symmetrize(_, ::FullBZ, x::TrivialRepType) = x

symmetrize(f, bz, x::AuxValue) = AuxValue(symmetrize(f, bz, x.val, x.aux)...)
symmetrize(_, ::FullBZ, x::AuxValue) = x

# Here we provide utilities to build BZs

"""
    AbstractBZ{d}

Abstract supertype for all Brillouin zone data types parametrized by dimension.
"""
abstract type AbstractBZ{d} end

"""
    load_bz(::AbstractBZ, T::Type)
    load_bz(::AbstractBZ, A::AbstractMatrix, [B::AbstractMatrix])

Interface to loading Brillouin zones
"""
function load_bz end

function load_bz(bz::AbstractBZ{N}, A::AbstractMatrix{T}, B::AbstractMatrix{S}=canonical_reciprocal_basis(A); atol=nothing) where {N,T,S}
    (d = checksquare(A)) == checksquare(B) ||
        throw(DimensionMismatch("Bravais lattices $A and $B must have the same shape"))
    bz_ = if N isa Integer
        @assert d == N
        bz
    else
        convert(AbstractBZ{d}, bz)
    end
    check_bases_canonical(A, B, something(atol, sqrt(eps(oneunit(T)*oneunit(S)))))
    MA = SMatrix{d,d,T,d^2}; MB = SMatrix{d,d,S,d^2}
    return load_bz(bz_, convert(MA, A), convert(MB, B))
end

function load_bz(bz::AbstractBZ{d}, ::Type{T}=Float64) where {d,T}
    d isa Integer || throw(ArgumentError("BZ dimension must be integer"))
    A = oneunit(SMatrix{d,d,T,d^2})
    return load_bz(bz, A)
end

"""
    FBZ{N} <: AbstractBZ

Singleton type representing first/full Brillouin zones of `N` dimensions.
By default, `N` is nothing and the dimension is obtained from input files.
"""
struct FBZ{N} <: AbstractBZ{N} end
FBZ(n=nothing) = FBZ{n}()
Base.convert(::Type{AbstractBZ{d}}, ::FBZ) where {d} = FBZ{d}()

function load_bz(::FBZ{N}, A::SMatrix{N,N}, B::SMatrix{N,N}) where {N}
    lims = lattice_bz_limits(B)
    return SymmetricBZ(A, B, lims, nothing)
end

"""
    IBZ <: AbstractBZ

Singleton type representing irreducible Brillouin zones. Load
[SymmetryReduceBZ.jl](https://github.com/jerjorg/SymmetryReduceBZ.jl) to use this.
"""
struct IBZ{d,P} <: AbstractBZ{d} end

struct DefaultPolyhedron end

IBZ(n=nothing,) = IBZ{n,DefaultPolyhedron}()
Base.convert(::Type{AbstractBZ{d}}, ::IBZ{N,P}) where {d,N,P} = IBZ{d,P}()

checkorthog(A::AbstractMatrix) = isdiag(transpose(A)*A)

sign_flip_tuples(n::Val{d}) where {d} = Iterators.product(ntuple(_ -> (1,-1), n)...)
sign_flip_matrices(n::Val{d}) where {d} = (Diagonal(SVector{d,Int}(A)) for A in sign_flip_tuples(n))
n_sign_flips(d::Integer) = 2^d

"""
    InversionSymIBZ{N} <: AbstractBZ

Singleton type representing Brillouin zones with full inversion symmetry

!!! warning "Assumptions"
    Only expect this to work for systems with orthogonal lattice vectors
"""
struct InversionSymIBZ{N} <: AbstractBZ{N} end
InversionSymIBZ(n=nothing) = InversionSymIBZ{n}()
Base.convert(::Type{AbstractBZ{d}}, ::InversionSymIBZ) where {d} = InversionSymIBZ{d}()

function load_bz(::InversionSymIBZ{N}, A::SMatrix{N,N}, B::SMatrix{N,N,TB}) where {N,TB}
    checkorthog(A) || @warn "Non-orthogonal lattice vectors detected with InversionSymIBZ. Unexpected behavior may occur"
    T = typeof(one(TB)); V = SVector{N,T}
    lims = CubicLimits(zero(V), fill(1//2, V))
    syms = map(S -> one(T)*S, sign_flip_matrices(Val(N)))
    return SymmetricBZ(A, B, lims, syms)
end

function permutation_matrices(t::Val{n}) where {n}
    permutations = permutation_tuples(ntuple(identity, t))
    (sacollect(SMatrix{n,n,Int,n^2}, ifelse(j == p[i], 1, 0) for i in 1:n, j in 1:n) for p in permutations)
end
permutation_tuples(C::NTuple{N,T}) where {N,T} = @inbounds((C[i], p...)::NTuple{N,T} for i in eachindex(C) for p in permutation_tuples(C[[j for j in eachindex(C) if j != i]]))
permutation_tuples(C::NTuple{1}) = C
n_permutations(n::Integer) = factorial(n)

"""
    cube_automorphisms(::Val{d}) where d

return a generator of the symmetries of the cube in `d` dimensions including the
identity.
"""
cube_automorphisms(n::Val{d}) where {d} = (S*P for S in sign_flip_matrices(n), P in permutation_matrices(n))
n_cube_automorphisms(d) = n_sign_flips(d) * n_permutations(d)

"""
    CubicSymIBZ{N} <: AbstractBZ

Singleton type representing Brillouin zones with full cubic symmetry

!!! warning "Assumptions"
    Only expect this to work for systems with orthogonal lattice vectors
"""
struct CubicSymIBZ{N} <: AbstractBZ{N} end
CubicSymIBZ(n=nothing) = CubicSymIBZ{n}()
Base.convert(::Type{AbstractBZ{d}}, ::CubicSymIBZ) where {d} = CubicSymIBZ{d}()

function load_bz(::CubicSymIBZ{N}, A::SMatrix{N,N}, B::SMatrix{N,N,TB}) where {N,TB}
    checkorthog(A) || @warn "Non-orthogonal lattice vectors detected with CubicSymIBZ. Unexpected behavior may occur"
    T = typeof(one(TB))
    lims = TetrahedralLimits(fill(1//2, SVector{N,T}))
    syms = map(S -> one(T)*S, cube_automorphisms(Val{N}()))
    return SymmetricBZ(A, B, lims, syms)
end

# Now we provide the BZ integration algorithms effectively as aliases to the libraries

"""
    AutoBZAlgorithm

Abstract supertype for Brillouin zone integration algorithms.
All integration problems on the BZ get rescaled to fractional coordinates so that the
Brillouin zone becomes `[0,1]^d`, and integrands should have this periodicity. If the
integrand depends on the Brillouin zone basis, then it may have to be transformed to the
Cartesian coordinates as a post-processing step.
"""
abstract type AutoBZAlgorithm <: IntegralAlgorithm end

function init_cacheval(f, bz::SymmetricBZ, p, bzalg::AutoBZAlgorithm)
    _, dom, alg = bz_to_standard(bz, bzalg)
    return init_cacheval(f, dom, p, alg)
end

function do_solve(f, bz::SymmetricBZ, p, bzalg::AutoBZAlgorithm, cacheval; _kws...)
    bz_, dom, alg = bz_to_standard(bz, bzalg)

    j = abs(det(bz_.B))  # rescale tolerance to (I)BZ coordinate and get the right number of digits
    kws = NamedTuple(_kws)
    kws_ = haskey(kws, :abstol) ? merge(kws, (abstol=kws.abstol / (j * nsyms(bz_)),)) : kws

    @show sol = do_solve(f, dom, p, alg, cacheval; kws_...)
    val = j*symmetrize(f, bz_, sol.u)
    err = sol.resid === nothing ? nothing : j*symmetrize(f, bz_, sol.resid)
    return IntegralSolution(val, err, sol.retcode)
end

# AutoBZAlgorithms must implement:
# - bz_to_standard: (transformed) bz, unitless domain, standard algorithm

"""
    IAI(alg::IntegralAlgorithm=AuxQuadGKJL())
    IAI(algs::IntegralAlgorithm...)

Iterated-adaptive integration using `nested_quad` from
[IteratedIntegration.jl](https://github.com/lxvm/IteratedIntegration.jl).
**This algorithm is the most efficient for localized integrands**.
"""
struct IAI{T} <: AutoBZAlgorithm
    algs::T
    IAI(alg::IntegralAlgorithm=AuxQuadGKJL()) = new{typeof(alg)}(alg)
    IAI(algs::Tuple{Vararg{IntegralAlgorithm}}) = new{typeof(algs)}(algs)
end
IAI(algs::IntegralAlgorithm...) = IAI(algs)

function bz_to_standard(bz::SymmetricBZ, alg::IAI)
    return bz, bz.lims, NestedQuad(alg.algs)
end

"""
    PTR(; npt=50, parallel=nothing)

Periodic trapezoidal rule with a fixed number of k-points per dimension, `npt`,
using the routine `ptr` from [AutoSymPTR.jl](https://github.com/lxvm/AutoSymPTR.jl).
**The caller should check that the integral is converged w.r.t. `npt`**.
"""
struct PTR <: AutoBZAlgorithm
    npt::Int
    nthreads::Int
end
PTR(; npt=50, nthreads=Threads.nthreads()) = PTR(npt, nthreads)

function bz_to_standard(bz::SymmetricBZ, alg::PTR)
    return bz, Basis(one(bz.B)), MonkhorstPack(npt=alg.npt, syms=bz.syms, nthreads=alg.nthreads)
end


"""
    AutoPTR(; norm=norm, a=1.0, nmin=50, nmax=1000, n₀=6, Δn=log(10), keepmost=2, parallel=nothing)

Periodic trapezoidal rule with automatic convergence to tolerances passed to the
solver with respect to `norm` using the routine `autosymptr` from
[AutoSymPTR.jl](https://github.com/lxvm/AutoSymPTR.jl).
**This algorithm is the most efficient for smooth integrands**.
"""
struct AutoPTR{F} <: AutoBZAlgorithm
    norm::F
    a::Float64
    nmin::Int
    nmax::Int
    n₀::Float64
    Δn::Float64
    keepmost::Int
    nthreads::Int
end
function AutoPTR(; norm=norm, a=1.0, nmin=50, nmax=1000, n₀=6.0, Δn=log(10), keepmost=2, nthreads=Threads.nthreads())
    return AutoPTR(norm, a, nmin, nmax, n₀, Δn, keepmost, nthreads)
end
function bz_to_standard(bz::SymmetricBZ, alg::AutoPTR)
    return bz, Basis(one(bz.B)), AutoSymPTRJL(norm=alg.norm, a=alg.a, nmin=alg.nmin, nmax=alg.nmax, n₀=alg.n₀, Δn=alg.Δn, keepmost=alg.keepmost, syms=bz.syms, nthreads=alg.nthreads)
end

"""
    TAI(; norm=norm, initdivs=1)

Tree-adaptive integration using `hcubature` from
[HCubature.jl](https://github.com/JuliaMath/HCubature.jl). This routine is
limited to integration over hypercube domains and may not use all symmetries.
"""
struct TAI{N} <: AutoBZAlgorithm
    norm::N
    initdiv::Int
end
TAI(; norm=norm, initdiv=1) = TAI(norm, initdiv)

function bz_to_standard(bz::SymmetricBZ, alg::TAI)
    bz_ = bz.lims isa CubicLimits ? bz : SymmetricBZ(bz.A, bz.B, lattice_bz_limits(bz.B), nothing)
    l = bz_.lims
    return bz_, HyperCube(l.a, l.b), HCubatureJL(norm=alg.norm, initdiv = alg.initdiv)
end


"""
    PTR_IAI(; ptr=PTR(), iai=IAI())

Multi-algorithm that returns an `IAI` calculation with an `abstol` determined
from the given `reltol` and a `PTR` estimate, `I`, as `reltol*norm(I)`.
This addresses the issue that `IAI` does not currently use a globally-adaptive
algorithm and may not have the expected scaling with localization length unless
an `abstol` is used since computational effort may be wasted via a `reltol` with
the naive `nested_quadgk`.
"""
PTR_IAI(; ptr=PTR(), iai=IAI(), kws...) = AbsoluteEstimate(ptr, iai; kws...)


"""
    AutoPTR_IAI(; reltol=1.0, ptr=AutoPTR(), iai=IAI())

Multi-algorithm that returns an `IAI` calculation with an `abstol` determined
from an `AutoPTR` estimate, `I`, computed to `reltol` precision, and the `rtol`
given to the solver as `abstol=rtol*norm(I)`.
This addresses the issue that `IAI` does not currently use a globally-adaptive
algorithm and may not have the expected scaling with localization length unless
an `abstol` is used since computational effort may be wasted via a `reltol` with
the naive `nested_quadgk`.
"""
AutoPTR_IAI(; reltol=1.0, ptr=AutoPTR(), iai=IAI(), kws...) = AbsoluteEstimate(ptr, iai; reltol=reltol, kws...)
