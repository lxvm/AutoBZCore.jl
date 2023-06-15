# we recreate a lot of the SciML Integrals.jl functionality, but only for our algorithms
# the features we omit are: inplace integrands, infinite limit transformations, nout and
# batch keywords. Otherwise, there is a correspondence between.
# solve -> do_solve
# init -> make_cache

"""
    IntegralAlgorithm

Abstract supertype for integration algorithms.
"""
abstract type IntegralAlgorithm end


struct IntegralProblem{F,D,P}
    f::F
    dom::D
    p::P
    IntegralProblem{F,D,P}(f::F, dom::D, p::P) where {F,D,P} = new{F,D,P}(f, dom, p)
end
IntegralProblem(f::F, dom::D, p::P=()) where {F,D,P} = IntegralProblem{F,D,P}(f, dom, p)
function IntegralProblem(f::F, a::T, b::T, p::P=()) where {F,T,P}
    dom = T <: Real ? PuncturedInterval((a, b)) : HyperCube(a, b)
    return IntegralProblem{F,typeof(dom),P}(f, dom, p)
end

struct IntegralCache{F,D,P,A,C,K}
    f::F
    dom::D
    p::P
    alg::A
    cacheval::C
    kwargs::K
end

function make_cache(f, dom, p, alg; kwargs...)
    cacheval = init_cacheval(f, dom, p, alg)
    return IntegralCache(f, dom, p, alg, cacheval, NamedTuple(kwargs))
end

function checkkwargs(kwargs)
    for key in keys(kwargs)
        key in (:abstol, :reltol, :maxiters) || throw(ArgumentError("keyword $key unrecognized"))
    end
    return nothing
end

function make_cache(prob::IntegralProblem, alg::IntegralAlgorithm; kwargs...)
    checkkwargs(NamedTuple(kwargs))
    f = prob.f; dom = prob.dom; p = prob.p
    return make_cache(f, dom, p, alg; kwargs...)
end

function solve(prob::IntegralProblem, alg::IntegralAlgorithm; kwargs...)
    cache = make_cache(prob, alg; kwargs...)
    return do_solve(cache)
end

function do_solve(c::IntegralCache)
    return do_solve(c.f, c.dom, c.p, c.alg, c.cacheval; c.kwargs...)
end

# Methods an algorithm must define
# - init_cacheval
# - do_solve

# this method can be extended to different integrands so that it doesn't have to be
# evaluated to get the type, which is also useful in case parameters are incomplete
integrand_return_type(f, x, p) = typeof(f(x, p))

struct IntegralSolution{T,E}
    u::T
    resid::E
    retcode::Bool
end


# Here we replicate the algorithms provided by SciML

"""
    QuadGKJL(; order = 7, norm = norm, parallel = Sequential())

Generalization of the QuadGKJL provided by Integrals.jl that allows for `AuxValue`d
integrands for auxiliary integration and multi-threaded evaluation with a `parallel`
argument. See `auxquadgk` from IteratedIntegration.jl for more detail.
"""
struct QuadGKJL{F,P} <: IntegralAlgorithm
    order::Int
    norm::F
    parallel::P
end
function QuadGKJL(; order = 7, norm = norm, parallel = Sequential())
    return QuadGKJL(order, norm, parallel)
end

rule_type(::GaussKronrod{T}) where {T} = T
init_rule(dom, alg::QuadGKJL) = GaussKronrod(eltype(dom), alg.order)
function init_segbuf(f, dom, p, rule)
    TX = float(eltype(dom))
    TF = integrand_return_type(f, zero(rule_type(rule)), p)
    TI = typeof(zero(TF)* float(real(one(TX))))
    TE = typeof(norm(zero(TI)))
    return TF, IteratedIntegration.alloc_segbuf(TX, TI, TE)
end
init_parallel(p::Sequential, _, _, _) = p
function init_parallel(p::Parallel, T, S, order)
    p isa Parallel{T,S} && return p
    return Parallel(Vector{T}(undef, 2*order+1), Vector{S}(undef, 1), Vector{S}(undef, 2))
end
function init_cacheval(f, dom, p, alg::QuadGKJL)
    rule = init_rule(dom, alg)
    TF, segbuf = init_segbuf(f, dom, p, rule)
    parallel = init_parallel(alg.parallel, TF, eltype(segbuf), alg.order)
    return (rule=rule, segbuf=segbuf, parallel=parallel)
end

function do_solve(f, dom, p, alg::QuadGKJL, cacheval;
                    reltol = nothing, abstol = nothing, maxiters = typemax(Int))
    g = x -> f(x, p)

    segs = segments(dom)
    val, err = auxquadgk(g, segs..., parallel = cacheval.parallel, rule = cacheval.rule, maxevals = maxiters,
                      rtol = reltol, atol = abstol, order = alg.order, norm = alg.norm, segbuf=cacheval.segbuf)
    return IntegralSolution(val, err, true)
end

"""
    HCubatureJL(; norm=norm, initdiv=1)

A copy of `HCubatureJL` from Integrals.jl.
"""
struct HCubatureJL{N} <: IntegralAlgorithm
    norm::N
    initdiv::Int
end
HCubatureJL(; norm=norm, initdiv=1) = HCubatureJL(norm, initdiv)

init_cacheval(f, dom, p, ::HCubatureJL) = Some(nothing)

function do_solve(f, dom, p, alg::HCubatureJL, cacheval;
                    reltol = nothing, abstol = nothing, maxiters = typemax(Int))

    abstol_ = (abstol===nothing) ? zero(eltype(dom)) : abstol
    reltol_ = (reltol===nothing) ? (iszero(abstol_) ? sqrt(eps(typeof(abstol_))) : zero(abstol_)) : reltol

    g = x -> f(x, p)
    a, b = endpoints(dom)
    routine = a isa Number ? hquadrature : hcubature
    val, err = routine(g, a, b; norm = alg.norm, initdiv = alg.initdiv, atol=abstol_, rtol=reltol_, maxevals=maxiters)
    return IntegralSolution(val, err, true)
end

# Now we provide the BZ integration algorithms

"""
    AutoBZAlgorithm

Abstract supertype for Brillouin zone integration algorithms.
"""
abstract type AutoBZAlgorithm <: IntegralAlgorithm end

"""
    IAI(; order=7, norm=norm, initdivs=nothing, segbufs=nothing, parallels=nothing)

Iterated-adaptive integration using `nested_quad` from
[IteratedIntegration.jl](https://github.com/lxvm/IteratedIntegration.jl).
**This algorithm is the most efficient for localized integrands**.
"""
struct IAI{F,I,P} <: AutoBZAlgorithm
    order::Int
    norm::F
    initdivs::I
    parallels::P
end
function IAI(; order=7, norm=norm, initdivs=nothing, parallels=nothing)
    return IAI(order, norm, initdivs, parallels)
end

function init_rule(bz::SymmetricBZ, alg::IAI)
    return NestedGaussKronrod(eltype(bz.lims), alg.order, Val(ndims(bz.lims)))
end

rule_type(r::NestedGaussKronrod) = IteratedIntegration.rule_type(r)
function init_segbufs(f, dom, p, rule)
    TX = float(eltype(dom))
    TF = integrand_return_type(f, zero(rule_type(rule)), p)
    TI = typeof(zero(TF) * float(real(one(TX))))
    TE = typeof(norm(zero(TI)))
    return TF, alloc_segbufs(ndims(dom), TX, TI, TE)
end
init_parallels(::Nothing, args...) = nothing
init_parallels(::Tuple{}, args...) = ()
function init_parallels(p::Tuple, args...)
    return (init_parallel(first(p), args...), init_parallels(Base.tail(p), args...)...)
end
function init_cacheval(f, bz::SymmetricBZ, p, alg::IAI)
    rule = init_rule(bz, alg)
    TF, segbufs = init_segbufs(f, bz.lims, p, rule)
    parallels = init_parallels(alg.parallels, TF, eltype(eltype(segbufs)), alg.order)
    return (rule=rule, segbufs=segbufs, parallels=parallels)
end

"""
    PTR(; npt=50, parallel=nothing)

Periodic trapezoidal rule with a fixed number of k-points per dimension, `npt`,
using the routine `ptr` from [AutoSymPTR.jl](https://github.com/lxvm/AutoSymPTR.jl).
**The caller should check that the integral is converged w.r.t. `npt`**.
"""
struct PTR{P} <: AutoBZAlgorithm
    npt::Int
    parallel::P
end
PTR(; npt=50, parallel=nothing) = PTR(npt, parallel)
function init_rule(bz::FullBZType, alg::PTR)
    dom = Basis(bz.B)
    return AutoSymPTR.PTR(eltype(dom), Val(ndims(dom)), alg.npt)
end
function init_rule(bz::SymmetricBZ, alg::PTR)
    dom = Basis(bz.B)
    return AutoSymPTR.MonkhorstPack(eltype(dom), Val(ndims(dom)), alg.npt, bz.syms)
end

rule_type(::AutoSymPTR.PTR{N,T}) where {N,T} = SVector{N,T}
function init_buffer(f, p, rule, parallel)
    parallel === nothing && return nothing
    T = integrand_return_type(f, zero(rule_type(rule)), p)
    return T[]
end
function init_cacheval(f, bz::SymmetricBZ , p, alg::PTR)
    rule = init_rule(bz, alg)
    buf = init_buffer(f, p, rule, alg.parallel)
    return (rule=rule, buffer=buf)
end

"""
    AutoPTR(; norm=norm, a=1.0, nmin=50, nmax=1000, n₀=6, Δn=log(10), keepmost=2, parallel=nothing)

Periodic trapezoidal rule with automatic convergence to tolerances passed to the
solver with respect to `norm` using the routine `autosymptr` from
[AutoSymPTR.jl](https://github.com/lxvm/AutoSymPTR.jl).
**This algorithm is the most efficient for smooth integrands**.
"""
struct AutoPTR{F,P} <: AutoBZAlgorithm
    norm::F
    a::Float64
    nmin::Int
    nmax::Int
    n₀::Float64
    Δn::Float64
    keepmost::Int
    parallel::P
end
function AutoPTR(; norm=norm, a=1.0, nmin=50, nmax=1000, n₀=6.0, Δn=log(10), keepmost=2, parallel=nothing)
    return AutoPTR(norm, a, nmin, nmax, n₀, Δn, keepmost, parallel)
end
function init_rule(bz::SymmetricBZ, alg::AutoPTR)
    return AutoSymPTR.MonkhorstPackRule(bz.syms, alg.a, alg.nmin, alg.nmax, alg.n₀, alg.Δn)
end
rule_type(::AutoSymPTR.MonkhorstPack{N,T}) where {N,T} = SVector{N,T}
function init_cacheval(f, bz::SymmetricBZ, p, alg::AutoPTR)
    rule = init_rule(bz, alg)
    cache = AutoSymPTR.alloc_cache(eltype(bz), Val(ndims(bz)), rule)
    buffer = init_buffer(f, p, cache[1], alg.parallel)
    return (rule=rule, cache=cache, buffer=buffer)
end
function reduce_ptr_cache!(cache::Vector, nrule::Integer)
    @assert nrule > 1
    # keep at most the `nrule` most refined rules
    (nelem = min(length(cache), nrule)) == length(cache) && return cache
    reverse!(cache)
    reverse!(cache, firstindex(cache), firstindex(cache)+nelem-1)
    resize!(cache, nelem)
    return cache
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
struct PTR_IAI{P,I} <: AutoBZAlgorithm
    ptr::P
    iai::I
end
PTR_IAI(; ptr=PTR(), iai=IAI()) = PTR_IAI(ptr, iai)


function init_cacheval(f, bz::SymmetricBZ, p, alg::PTR_IAI)
    ptr_cacheval = init_cacheval(f, bz, p, alg.ptr)
    iai_cacheval = init_cacheval(f, bz, p, alg.iai)
    return (ptr=ptr_cacheval, iai=iai_cacheval)
end

"""
    AutoPTR_IAI(; reltol=1.0, ptr=AutoPTR(), iai=IAI())

Multi-algorithm that returns an `IAI` calculation with an `abstol` determined
from an `AutoPTR` estimate, `I`, computed to `reltol` precision, and the `rtol`
given to the solver as `rtol*norm(I)`.
This addresses the issue that `IAI` does not currently use a globally-adaptive
algorithm and may not have the expected scaling with localization length unless
an `abstol` is used since computational effort may be wasted via a `reltol` with
the naive `nested_quadgk`.
"""
struct AutoPTR_IAI{P,I} <: AutoBZAlgorithm
    reltol::Float64
    ptr::P
    iai::I
end
AutoPTR_IAI(; reltol=1.0, ptr=AutoPTR(), iai=IAI()) = AutoPTR_IAI(reltol, ptr, iai)

function init_cacheval(f, bz::SymmetricBZ, p, alg::AutoPTR_IAI)
    ptr_cacheval = init_cacheval(f, bz, p, alg.ptr)
    iai_cacheval = init_cacheval(f, bz, p, alg.iai)
    return (ptr=ptr_cacheval, iai=iai_cacheval)
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

init_cacheval(f, bz::SymmetricBZ, p, alg::TAI) = Some(nothing)


function do_solve(f, bz::SymmetricBZ, p, alg::AutoBZAlgorithm, cacheval;
                    reltol = nothing, abstol = nothing, maxiters = typemax(Int))

    abstol_ = (abstol===nothing) ? zero(eltype(bz)) : abstol
    reltol_ = (reltol===nothing) ? (iszero(abstol_) ? sqrt(eps(typeof(abstol_))) : zero(abstol_)) : reltol

    g = x -> f(x, p)

    if alg isa IAI
        j = abs(det(bz.B))  # include jacobian determinant for map from fractional reciprocal lattice coordinates to Cartesian reciprocal lattice
        atol = abstol_/nsyms(bz)/j # reduce absolute tolerance by symmetry factor
        val, err = nested_quad(g, bz.lims; atol=atol, rtol=reltol_, maxevals = maxiters, rule = cacheval.rule, parallels=cacheval.parallels,
                                        norm = alg.norm, order = alg.order, initdivs = alg.initdivs, segbufs = cacheval.segbufs)
        val_ = symmetrize(f, bz, j*val)
        err_ = symmetrize(f, bz, j*err)
        IntegralSolution(val_, err_, true)
    elseif alg isa PTR
        val = cacheval.rule(g, Basis(bz.B), cacheval.buffer)
        val_ = symmetrize(f, bz, val)
        IntegralSolution(val_, nothing, true)
    elseif alg isa AutoPTR
        val, err = autosymptr(g, Basis(bz.B); syms = bz.syms, rule = cacheval.rule, cache = cacheval.cache,
                        abstol = abstol_, reltol = reltol_, maxevals = maxiters, norm=alg.norm, buffer=cacheval.buffer)
        reduce_ptr_cache!(cacheval.cache, alg.keepmost)
        val_ = symmetrize(f, bz, val)
        err_ = symmetrize(f, bz, err)
        IntegralSolution(val_, err_, true)
    elseif alg isa PTR_IAI
        sol = do_solve(f, bz, p, alg.ptr, cacheval.ptr; reltol = reltol_, abstol = abstol, maxiters = maxiters)
        atol = max(abstol_, reltol_*alg.iai.norm(sol.u))
        do_solve(f, bz, p, alg.ptr, cacheval.ptr; reltol = zero(atol), abstol = atol, maxiters = maxiters)
    elseif alg isa AutoPTR_IAI
        sol = do_solve(f, bz, p, alg.ptr, cacheval.ptr; reltol = reltol_, abstol = abstol, maxiters = maxiters)
        atol = max(abstol_, reltol_*alg.iai.norm(sol.u))
        do_solve(f, bz, p, alg.ptr, cacheval.ptr; reltol = zero(atol), abstol = atol, maxiters = maxiters)
    elseif alg isa TAI
        # Fallback to FBZ if the domain is not a cube
        l, nsym = bz.lims isa CubicLimits ? (bz.lims, nsyms(bz)) : (lattice_bz_limits(bz.B), 1)
        j = abs(det(bz.B))
        atol = abstol_/nsym/j # reduce absolute tolerance by symmetry factor
        val, err = hcubature(g, l.a, l.b; norm=alg.norm, initdiv=alg.initdiv, atol=atol, rtol=reltol_, maxevals=maxiters)
        if bz.lims isa CubicLimits
            IntegralSolution(symmetrize(f, bz, j*val), symmetrize(f, bz, j*err), true)
        else
            IntegralSolution(j*val, j*err, true)
        end
    end
end
