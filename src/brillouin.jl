# utilities
function lattice_bz_limits(B::AbstractMatrix)
    T = SVector{checksquare(B),typeof(one(eltype(B)))}
    CubicLimits(zero(T), ones(T))   # unitless canonical bz
end
function check_bases_canonical(A::AbstractMatrix, B::AbstractMatrix, atol)
    norm(A'B - 2pi*I) < atol || throw("Real and reciprocal Bravais lattice bases non-orthogonal to tolerance $atol")
end
canonical_reciprocal_basis(A::AbstractMatrix) = A' \ (pi*(one(A)+one(A)))
canonical_ptr_basis(B) = Basis(one(B))

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

Base.summary(bz::SymmetricBZ) = string(checksquare(bz.A), "-dimensional Brillouin zone with ", bz isa FullBZ ? "trivial" : nsyms(bz), " symmetries")
Base.show(io::IO, bz::SymmetricBZ) = print(io, summary(bz))
Base.ndims(::SymmetricBZ{S,L,d}) where {S,L,d} = d
Base.eltype(::Type{<:SymmetricBZ{S,L,d,TA,TB}}) where {S,L,d,TA,TB} = TB
get_prototype(bz::SymmetricBZ) = interior_point(bz.lims)

# Define traits for symmetrization based on symmetry representations

"""
    AbstractSymRep

Abstract supertype of symmetry representation traits.
"""
abstract type AbstractSymRep end

"""
    UnknownRep()

Fallback symmetry representation for array types without a user-defined `SymRep`.
Will perform FBZ integration regardless of available BZ symmetries.
"""
struct UnknownRep <: AbstractSymRep end

"""
    TrivialRep()

Symmetry representation of objects with trivial transformation under the group.
"""
struct TrivialRep <: AbstractSymRep end

"""
    symmetrize(rep::AbstractSymRep, ::SymmetricBZ, x)

Transform `x` by the representation of the symmetries of the point group used to reduce the
domain, thus mapping the value of `x` on to the full Brillouin zone.
"""
symmetrize(rep, bz::SymmetricBZ, x) = symmetrize_(rep, bz, x)
symmetrize(_, ::FullBZ, x) = x

symmetrize_(rep, bz, x) = symmetrize__(rep, bz, x)
symmetrize_(rep, bz, x::AuxValue) = AuxValue(symmetrize__(rep, bz, x.val), symmetrize__(rep, bz, x.aux))

symmetrize__(::TrivialRep, bz, x) = nsyms(bz)*x
symmetrize__(::UnknownRep, bz, x) = error("unknown representation cannot be symmetrized")

struct SymmetricRule{R,U,B}
    rule::R
    rep::U
    bz::B
end

Base.getindex(r::SymmetricRule, i) = getindex(r.rule, i)
Base.eltype(::Type{SymmetricRule{R,U,B}}) where {R,U,B} = eltype(R)
Base.length(r::SymmetricRule) = length(r.rule)
Base.iterate(r::SymmetricRule, args...) = iterate(r.rule, args...)
function (r::SymmetricRule)(f::F, args...) where {F}
    out = r.rule(f, args...)
    val = symmetrize(r.rep, r.bz, out)
    return val
end

struct SymmetricRuleDef{R,U,B}
    rule::R
    rep::U
    bz::B
end

AutoSymPTR.nsyms(r::SymmetricRuleDef) = AutoSymPTR.nsyms(r.r)
function (r::SymmetricRuleDef)(::Type{T}, v::Val{d}) where {T,d}
    return SymmetricRule(r.rule(T, v), r.rep, r.bz)
end
function AutoSymPTR.nextrule(r::SymmetricRule, ruledef::SymmetricRuleDef)
    return SymmetricRule(AutoSymPTR.nextrule(r.rule, ruledef.rule), ruledef.rep, ruledef.bz)
end

# Here we provide utilities to build BZs

"""
    AbstractBZ{d}

Abstract supertype for all Brillouin zone data types parametrized by dimension.
"""
abstract type AbstractBZ{d} end

"""
    load_bz(bz::AbstractBZ, [T::Type=Float64])
    load_bz(bz::AbstractBZ, A::AbstractMatrix, [B::AbstractMatrix])

Interface to loading Brillouin zones.

## Arguments
- `bz::AbstractBZ`: a kind of Brillouin zone to construct, e.g. [`FBZ`](@ref) or
  [`IBZ`](@ref)
- `T::Type`: a numeric type to set the precision of the domain (default: `Float64`)
- `A::AbstractMatrix`: a ``d \\times d`` matrix whose columns are the real-space lattice
  vectors of a ``d``-dimensional crystal
- `B::AbstractMatrix`: a ``d \\times d`` matrix whose columns are the reciprocal-space
  lattice vectors of a ``d``-dimensional Brillouin zone (default: `A' \\ 2πI`)

!!! note "Assumptions"
    `AutoBZCore` assumes that all calculations occur in the reciprocal
    lattice basis, since that is the basis in which Wannier interpolants are most
    efficiently described. See [`SymmetricBZ`](@ref) for details. We also assume that the
    integrands are cheap to evaluate, which is why we provide adaptive methods in the first
    place, so that return types can be determined at runtime (and mechanisms are in place
    for compile time as well)
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

"""
    load_bz(::IBZ, A, B, species::AbstractVector, positions::AbstractMatrix; kws...)

`species` must have distinct labels for each atom type (e.g. can be any string or integer)
and `positions` must be a matrix whose columns give the coordinates of the atom of the
corresponding species.
"""
function load_bz(bz::IBZ, A, B, species, positions; kws...)
    ext = Base.get_extension(@__MODULE__(), :SymmetryReduceBZExt)
    if ext !== nothing
        return ext.load_ibz(bz, A, B, species, positions; kws...)
    else
        error("SymmetryReduceBZ extension not loaded")
    end
end
function load_bz(bz::IBZ{N}, A::SMatrix{N,N}, B::SMatrix{N,N}) where {N}
    return load_bz(bz, A, B, nothing, nothing)
end

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
    t = one(TB); V = SVector{N,typeof(t)}
    lims = CubicLimits(zero(V), fill(1//2, V))
    syms = map(S -> t*S, sign_flip_matrices(Val(N)))
    return SymmetricBZ(A, B, lims, syms)
end

function permutation_matrices(t::Val{n}) where {n}
    permutations = permutation_tuples(ntuple(identity, t))
    (sacollect(SMatrix{n,n,Int,n^2}, ifelse(j == p[i], 1, 0) for i in 1:n, j in 1:n) for p in permutations)
end
permutation_tuples(C::NTuple{N}) where {N} = @inbounds((C[i], p...)::typeof(C) for i in eachindex(C) for p in permutation_tuples(C[[j for j in eachindex(C) if j != i]]))
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
    t = one(TB)
    lims = TetrahedralLimits(fill(1//2, SVector{N,typeof(t)}))
    syms = map(S -> t*S, cube_automorphisms(Val{N}()))
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
These algorithms also use the symmetries of the Brillouin zone and the integrand.
"""
abstract type AutoBZAlgorithm <: IntegralAlgorithm end

"""
    AutoBZProblem([rep], f, bz, [p]; kwargs...)

Construct a BZ integration problem.

## Arguments
- `rep::AbstractSymRep`: The symmetry representation of `f` (default: `UnknownRep()`)
- `f::AbstractIntegralFunction`: The integrand
- `bz::SymmetricBZ`: The Brillouin zone to integrate over
- `p`: parameters for the integrand (default: `NullParameters()`)

## Keywords
Additional keywords are passed directly to the solver
"""
struct AutoBZProblem{R<:AbstractSymRep,F<:AbstractIntegralFunction,BZ<:SymmetricBZ,P,K<:NamedTuple}
    rep::R
    f::F
    bz::BZ
    p::P
    kwargs::K
end

const WARN_UNKNOWN_SYMMETRY = """
A symmetric BZ was used with an integrand whose symmetry representation is unknown.
For correctness, the calculation will proceed on the full BZ, i.e. without symmetry.
To integrate with symmetry, define an AbstractSymRep for your integrand.
"""

function AutoBZProblem(rep::AbstractSymRep, f::AbstractIntegralFunction, bz::SymmetricBZ, p=NullParameters(); kws...)
    proto = get_prototype(f, get_prototype(bz), p)
    if rep isa UnknownRep && !(bz isa FullBZ)
        @warn WARN_UNKNOWN_SYMMETRY
        fbz = SymmetricBZ(bz.A, bz.B, lattice_bz_limits(bz.B), nothing)
        return AutoBZProblem(rep, f, fbz, p, NamedTuple(kws))
    else
        return AutoBZProblem(rep, f, bz, p, NamedTuple(kws))
    end
end
function AutoBZProblem(f::AbstractIntegralFunction, bz::SymmetricBZ, p=NullParameters(); kws...)
    return AutoBZProblem(UnknownRep(), f, bz, p; kws...)
end
function AutoBZProblem(f, bz::SymmetricBZ, p=NullParameters(); kws...)
    return AutoBZProblem(IntegralFunction(f), bz, p; kws...)
end

mutable struct AutoBZCache{R,F,BZ,P,A,C,K}
    rep::R
    f::F
    bz::BZ
    p::P
    alg::A
    cacheval::C
    kwargs::K
end

function init(prob::AutoBZProblem, alg::AutoBZAlgorithm; kwargs...)
    rep = prob.rep; f = prob.f; bz = prob.bz; p = prob.p
    kws = (; prob.kwargs..., kwargs...)
    checkkwargs(kws)
    cacheval = init_cacheval(rep, f, bz, p, alg; kws...)
    return AutoBZCache(rep, f, bz, p, alg, cacheval, kws)
end

"""
    solve(::AutoBZProblem, ::AutoBZAlgorithm; kws...)::IntegralSolution
"""
solve(prob::AutoBZProblem, alg::AutoBZAlgorithm; kwargs...)

"""
    solve!(::IntegralCache)::IntegralSolution

Compute the solution to an [`IntegralProblem`](@ref) constructed from [`init`](@ref).
"""
function solve!(c::AutoBZCache)
    return do_solve_autobz(c.rep, c.f, c.bz, c.p, c.alg, c.cacheval; c.kwargs...)
end

function init_cacheval(rep, f, bz::SymmetricBZ, p, bzalg::AutoBZAlgorithm; kws...)
    prob, alg = bz_to_standard(f, bz, p, bzalg; kws...)
    return init(prob, alg)
end

function do_solve_autobz(rep, f, bz, p, bzalg::AutoBZAlgorithm, cacheval; _kws...)
    j = abs(det(bz.B))  # rescale tolerance to (I)BZ coordinate and get the right number of digits
    kws = NamedTuple(_kws)
    cacheval.f = f
    cacheval.p = p
    cacheval.kwargs = haskey(kws, :abstol) ? merge(kws, (abstol=kws.abstol / (j * nsyms(bz)),)) : kws

    sol = solve!(cacheval)
    value = j*symmetrize(rep, bz, sol.value)
    stats = (; sol.stats...)
    # err = sol.resid === nothing ? nothing : j*symmetrize(f, bz_, sol.resid)
    return IntegralSolution(value, sol.retcode, stats)
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
struct IAI{T,S} <: AutoBZAlgorithm
    algs::T
    specialize::S
    IAI(alg::IntegralAlgorithm=AuxQuadGKJL(), specialize::AbstractSpecialization=FunctionWrapperSpecialize()) = new{typeof(alg),typeof(specialize)}(alg, specialize)
    IAI(algs::Tuple{Vararg{IntegralAlgorithm}}, specialize::Tuple{Vararg{AbstractSpecialization}}=ntuple(_->FunctionWrapperSpecialize(),length(algs))) = new{typeof(algs),typeof(specialize)}(algs, specialize)
end
IAI(algs::IntegralAlgorithm...) = IAI(algs)

function bz_to_standard(f, bz, p, bzalg::IAI; kws...)
    return IntegralProblem(f, bz.lims, p; kws...), NestedQuad(bzalg.algs, bzalg.specialize)
end

"""
    PTR(; npt=50, nthreads=1)

Periodic trapezoidal rule with a fixed number of k-points per dimension, `npt`,
using the routine `ptr` from [AutoSymPTR.jl](https://github.com/lxvm/AutoSymPTR.jl).
**The caller should check that the integral is converged w.r.t. `npt`**.
"""
struct PTR <: AutoBZAlgorithm
    npt::Int
    nthreads::Int
end
PTR(; npt=50, nthreads=1) = PTR(npt, nthreads)

function bz_to_standard(f, bz, p, alg::PTR; kws...)
    return IntegralProblem(f, canonical_ptr_basis(bz.B), p; kws...), MonkhorstPack(npt=alg.npt, syms=bz.syms, nthreads=alg.nthreads)
end


"""
    AutoPTR(; norm=norm, a=1.0, nmin=50, nmax=1000, n₀=6, Δn=log(10), keepmost=2, nthreads=1)

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
function AutoPTR(; norm=norm, a=1.0, nmin=50, nmax=1000, n₀=6.0, Δn=log(10), keepmost=2, nthreads=1)
    return AutoPTR(norm, a, nmin, nmax, n₀, Δn, keepmost, nthreads)
end


struct RepBZ{R,B}
    rep::R
    bz::B
end
Base.ndims(dom::RepBZ) = ndims(dom.bz)
Base.eltype(::Type{RepBZ{R,B}}) where {R,B} = eltype(B)
get_prototype(dom::RepBZ) = get_prototype(dom.bz)


function init_cacheval(rep, f, bz::SymmetricBZ, p, bzalg::AutoPTR; kws...)
    prob = IntegralProblem(f, RepBZ(rep, bz), p; kws...)
    alg = AutoSymPTRJL(norm=bzalg.norm, a=bzalg.a, nmin=bzalg.nmin, nmax=bzalg.nmax, n₀=bzalg.n₀, Δn=bzalg.Δn, keepmost=bzalg.keepmost, syms=bz.syms, nthreads=bzalg.nthreads)
    return init(prob, alg)
end
get_basis(dom::RepBZ) = canonical_ptr_basis(dom.bz.B)
function init_rule(dom::RepBZ, alg::AutoSymPTRJL)
    B = get_basis(dom)
    rule = init_rule(B, alg)
    return SymmetricRuleDef(rule, dom.rep, dom.bz)
end
# The spectral convergence of the PTR for integrands with non-trivial symmetry action
# requires symmetrizing inside the quadrature
function do_solve_autobz(rep, f, bz, p, bzalg::AutoPTR, cacheval; _kws...)
    j = abs(det(bz.B))  # rescale tolerance to (I)BZ coordinate and get the right number of digits
    kws = NamedTuple(_kws)
    cacheval.f = f
    cacheval.p = p
    cacheval.kwargs = haskey(kws, :abstol) ? merge(kws, (abstol=kws.abstol / j,)) : kws

    sol = solve!(cacheval)
    value = j*sol.value
    stats = (; sol.stats..., error=sol.stats.error*j)
    return IntegralSolution(value, sol.retcode, stats)
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

function bz_to_standard(f, bz, p, alg::TAI; kws...)
    @assert bz.lims isa CubicLimits "TAI can only integrate rectangular regions"
    return IntegralProblem(f, HyperCube(bz.lims.a, bz.lims.b), p; kws...), HCubatureJL(norm=alg.norm, initdiv = alg.initdiv)
end

struct AutoBZEvalCounter{T<:AutoBZAlgorithm} <: AutoBZAlgorithm
    alg::T
end
function bz_to_standard(f, bz, p, bzalg::AutoBZEvalCounter; kws...)
    prob, alg = bz_to_standard(f, bz, p, bzalg.alg; kws...)
    return prob, EvalCounter(alg)
end
EvalCounter(alg::AutoBZAlgorithm) = AutoBZEvalCounter(alg)
#=
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

function count_bz_to_standard(bz, alg)
    _bz, dom, _alg = bz_to_standard(bz, alg)
    return _bz, dom, EvalCounter(_alg)
end

function do_solve(f, bz::SymmetricBZ, p, alg::EvalCounter{<:AutoBZAlgorithm}, cacheval; kws...)
    return do_solve_autobz(count_bz_to_standard, f, bz, p, alg.alg, cacheval; kws...)
end
=#
