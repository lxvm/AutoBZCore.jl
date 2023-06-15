# Here we provide optimizations of multidimensional Fourier series evaluation for the
# various algorithms. It could be a package extension, but we keep it in the main library
# because it provides the infrastructure of the main application of library

# In multiple dimensions, these specialized rules can provide a benefit over batch
# integrands since the multidimensional structure of the quadrature rule can be lost when
# batching many points together and passing them to an integrand to solve simultaneously. In
# some cases we can also cache the rule with series evaluations and apply it to different
# integrands, which again could only be achieved with a batched and vector-valued integrand.
# The ethos of this package is to let the user provide the kernel for the integrand and to
# have the library take care of the details of fast evaluation and such. Automating batched
# and vector-valued integrands is another worthwhile approach, but it is not well
# established in existing Julia libraries or Integrals.jl, so in the meantime I strive to
# provide these efficient rules for Wannier interpolation. In the long term, the batched and
# vector-valued approach will allow distributed computing and other benefits that are beyond
# the scope of what this package aims to provide.

# I would like to just make the user pass in some `FourierRule` to indicate that their
# integrand accepts `FourierValue`s, however in previous versions we used the
# `FourierIntegrand` concept, which we now convert to a Fourier 'rule' if there is a benefit
# to doing so. (i.e. multidimensional evaluation of Fourier series on hierarchical grids.)
# Also, rules don't exist in the SciML interface we are copying, so we have to dispatch on
# the integrand

struct FourierIntegrand{F,P,S<:AbstractFourierSeries}
    f::Integrand{F,P}
    s::S
end

function FourierIntegrand(f::F, s::AbstractFourierSeries, args...; kwargs...) where {F}
    p = MixedParameters(args...; kwargs...)
    return FourierIntegrand(Integrand{F}(f, p), s)
end

function Base.getproperty(f::FourierIntegrand, name::Symbol)
    name === :f && return getfield(f, :f).f
    name === :p && return getfield(f, :f).p
    return getfield(f, name)
end

function (s::IntegralSolver{<:FourierIntegrand})(args...; kwargs...)
    p = MixedParameters(args...; kwargs...)
    sol = do_solve(s, p)
    return sol.u
end

function remake_cache(f::FourierIntegrand, dom, p, alg, cacheval, kwargs)
    new = FourierIntegrand(f.f, f.s)
    return remake_integrand_cache(new, dom, merge(f.p, p), alg, cacheval, kwargs)
end

# FourierIntegrands should expect a FourierValue as input

struct FourierValue{X,S}
    x::X
    s::S
end

Base.zero(::Type{FourierValue{X,S}}) where {X,S} = FourierValue(zero(X), zero(S))
Base.:*(B::AbstractMatrix, f::FourierValue) = FourierValue(B*f.x, f.s)

(f::FourierIntegrand)(x::FourierValue, p=()) = getfield(f, :f)(x, p)
# fallback evaluator of FourierIntegrand for algorithms without specialized rules
(f::FourierIntegrand)(x, p=()) = f(FourierValue(x, f.s(x)), p)

# HCubature rule (no specialization for FourierIntegrand)

# QuadGK rule (no specialization for FourierIntegrand)

#=
# TODO: we can pre-evaluate the twiddle factors for the GK points as an optimization, but
# until we do so, there is no benefit to a dedicated FourierGaussKronrod rule since we could
# obtain the same improvement with a batched integrand, which isn't implemented either
# Here is a template to start from in case further optimization is desired

struct FourierGaussKronrod{S<:AbstractFourierSeries,T}
    s::S
    r::GaussKronrod{T}
end

function (r::FourierGaussKronrod)(f::F, s, nrm=norm, buffer=nothing) where {F}
    return r.r(s, nrm, buffer) do x
        return f(FourierValue(x, r.s(x)))
    end
end

function init_fourier_rule(s::AbstractFourierSeries, dom, alg::QuadGKJL)
    gk = GaussKronrod(eltype(dom), alg.order)
    return FourierGaussKronrod(s, gk)
end
function init_cacheval(f::FourierIntegrand, dom, p, alg::QuadGKJL)
    rule = init_fourier_rule(f.s, dom, alg)
    TF, segbuf = init_segbuf(f, dom, p, rule)
    parallel = init_parallel(alg.parallel, TF, eltype(segbuf), alg.order)
    return (rule=rule, segbuf=segbuf, parallel=parallel)
end
=#

# IAI rules

struct NestedFourierGaussKronrod{dim,d,X,T,S}
    r::NestedGaussKronrod{dim,d,X,T}
    s::S
    function NestedFourierGaussKronrod(r::NestedGaussKronrod{dim,d,X,T}, s::S) where {dim,d,X,T,S<:AbstractFourierSeries}
        @assert dim == ndims(s)
        return new{dim,d,X,T,S}(r, s)
    end
end

# base case
function IteratedIntegration.countevals(g::NestedFourierGaussKronrod)
    return IteratedIntegration.countevals(g.r)
end
function IteratedIntegration.rule_type(g::NestedFourierGaussKronrod)
    X = IteratedIntegration.rule_type(g.r)
    S = FourierSeriesEvaluators.fourier_type(g.s, X)
    return FourierValue{X,S}
end
function (g::NestedFourierGaussKronrod{1})(f::F, s, nrm=norm, buffer=nothing) where {F}
    return g.r(x -> f(FourierValue(x, g.s(x[1]))), s, nrm, buffer)
end

# multidimensional
IteratedIntegration.nextdim(::NestedFourierGaussKronrod{dim}) where dim = dim
function IteratedIntegration.nextrule(r::NestedFourierGaussKronrod, x, v::Val{dim}) where {dim}
    nr = nextrule(r.r, x, v)
    ns = contract(r.s, x, v)
    return NestedFourierGaussKronrod(nr, ns)
end

function (g::NestedFourierGaussKronrod)(f::F, s, nrm=norm, buffer=nothing) where {F}
    return g.r(f, s, nrm, buffer)
end

function rule_type(r::NestedFourierGaussKronrod)
    X = rule_type(r.r)
    S = fourier_type(r.s, eltype(X))
    return FourierValue{X,S}
end
function init_fourier_rule(s::AbstractFourierSeries, bz::SymmetricBZ, alg::IAI)
    gk = NestedGaussKronrod(eltype(bz.lims), alg.order, Val(ndims(bz.lims)))
    return NestedFourierGaussKronrod(gk, s)
end
function init_cacheval(f::FourierIntegrand, bz::SymmetricBZ, p, alg::IAI)
    rule = init_fourier_rule(f.s, bz, alg)
    TF, segbufs = init_segbufs(f, bz.lims, p, rule)
    parallels = init_parallels(alg.parallels, TF, eltype(eltype(segbufs)), alg.order)
    return (rule=rule, segbufs=segbufs, parallels=parallels)
end

# PTR rules

# no symmetries
struct FourierPTR{N,T,S,X} <: AbstractArray{Tuple{AutoSymPTR.One,FourierValue{SVector{N,T},S}},N}
    s::Array{S,N}
    p::AutoSymPTR.PTR{N,T,X}
end

function fourier_ptr!(vals::AbstractArray{T,1}, s::AbstractFourierSeries{1}, x::AbstractVector) where {T}
    # Threads.@threads # we know that FourierSeriesEvaluators.evaluate is thread safe
    for (i, y) in zip(eachindex(vals), x)
        @inbounds vals[i] = s(y)
    end
    return vals
end
function fourier_ptr!(vals::AbstractArray{T,d}, s::AbstractFourierSeries{d}, x::AbstractVector) where {T,d}
    # probably unsafe to parallelize this loop since s could be inplace
    for (y, v) in zip(x, eachslice(vals, dims=d))
        fourier_ptr!(v, contract(s, y, Val(d)), x)
    end
    return vals
end

function FourierPTR(s::AbstractFourierSeries, ::Type{T}, v::Val{d}, npt) where {T,d}
    p = AutoSymPTR.PTR(T, v, npt)
    vals = similar(p, fourier_type(s, eltype(p.x)))
    fourier_ptr!(vals, s, p.x)
    return FourierPTR(vals, p)
end

rule_type(::FourierPTR{N,T,S}) where {N,T,S} = FourierValue{SVector{N,T},S}
function init_fourier_rule(s::AbstractFourierSeries, bz::FullBZType , alg::PTR)
    dom = Basis(bz.B)
    return FourierPTR(s, eltype(dom), Val(ndims(dom)), alg.npt)
end
function init_cacheval(f::FourierIntegrand, bz::SymmetricBZ , p, alg::PTR)
    rule = init_fourier_rule(f.s, bz, alg)
    buf = init_buffer(f, p, rule, alg.parallel)
    return (rule=rule, buffer=buf)
end

# Array interface
Base.size(r::FourierPTR) = size(r.s)
function Base.getindex(r::FourierPTR{N}, i::Vararg{Int,N}) where {N}
    w, x = r.p[i...]
    return w, FourierValue(x, r.s[i...])
end

# iteration
function Base.iterate(p::FourierPTR)
    next1 = iterate(p.s)
    next1 === nothing && return nothing
    next2 = iterate(p.p)
    next2 === nothing && return nothing
    s, state1 = next1
    (w, x), state2 = next2
    return (w, FourierValue(x, s)), (state1, state2)
end
Base.isdone(::FourierPTR, state) = any(isnothing, state)
function Base.iterate(p::FourierPTR, state)
    next1 = iterate(p.s, state[1])
    next1 === nothing && return nothing
    next2 = iterate(p.p, state[2])
    next2 === nothing && return nothing
    s, state1 = next1
    (w, x), state2 = next2
    return (w, FourierValue(x, s)), (state1, state2)
end

AutoSymPTR.countevals(r::FourierPTR) = AutoSymPTR.countevals(r.p)

function (rule::FourierPTR)(f, B::Basis, buffer=nothing)
    return AutoSymPTR.quadsum(rule, f, B, buffer) * (abs(det(B.B)) / AutoSymPTR.countevals(rule))
end

# SymPTR rules
struct FourierMonkhorstPack{d,T,S}
    npt::Int64
    nsyms::Int64
    wxs::Vector{Tuple{Int64,FourierValue{SVector{d,T},S}}}
end

function fourier_symptr!!(vals::AbstractVector{V}, n, s::AbstractFourierSeries{1}, x::AbstractVector{T}, flag::AbstractVector{Bool}, wsym, nsym, z::NTuple{d}) where {V,T,d}
    # Threads.@threads # we know that FourierSeriesEvaluators.evaluate is thread safe
    # but to multithread we must compute an index array from flag, which could be an
    # extension of wsym
    @inbounds for (y, f) in zip(x, flag)
        n[] >= nsym && break
        f || continue
        n[] += 1
        vals[n[]] = (wsym[n[]], FourierValue(SVector{d+1,T}((y, z...)), s(y)))
    end
    return vals
end
function fourier_symptr!!(vals::AbstractVector{T}, n, s::AbstractFourierSeries{d}, x::AbstractVector, flag::AbstractArray{Bool,d}, wsym, nsym, z) where {T,d}
    # probably unsafe to parallelize this loop since s could be inplace
    for (y, f) in zip(x, eachslice(flag, dims=d))
        n[] >= nsym && break
        # any(f) || continue # this lookahead may be expensive, but it prevents unnecessary contractions
        fourier_symptr!!(vals, n, contract(s, y, Val(d)), x, f, wsym, nsym, (y, z...))
    end
    return vals
end

function FourierMonkhorstPack(s::AbstractFourierSeries, ::Type{T}, v::Val{d}, npt, syms) where {d,T}
    u = AutoSymPTR.ptrpoints(npt)
    flag, wsym, nsym = AutoSymPTR.symptr_rule(npt, Val(d), syms)
    wxs = Vector{Tuple{Int64,FourierValue{SVector{d,T},fourier_type(s,T)}}}(undef, nsym)
    fourier_symptr!!(wxs, fill(0), s, u, flag, wsym, nsym, ())
    return FourierMonkhorstPack(npt, length(syms), wxs)
end

function init_fourier_rule(s::AbstractFourierSeries, bz::SymmetricBZ, alg::PTR)
    dom = Basis(bz.B)
    return FourierMonkhorstPack(s, eltype(dom), Val(ndims(dom)), alg.npt, bz.syms)
end

# indexing
Base.getindex(rule::FourierMonkhorstPack, i::Int) = rule.wxs[i]

# iteration
Base.eltype(::Type{FourierMonkhorstPack{d,T,S}}) where {d,T,S} = Tuple{Int64,FourierValue{SVector{d,T},S}}
Base.length(r::FourierMonkhorstPack) = length(r.wxs)
Base.iterate(rule::FourierMonkhorstPack, args...) = iterate(rule.wxs, args...)

rule_type(::FourierMonkhorstPack{d,T,S}) where {d,T,S} = FourierValue{SVector{d,T},S}
AutoSymPTR.countevals(rule::FourierMonkhorstPack) = length(rule)

function (rule::FourierMonkhorstPack{d})(f, B::Basis, buffer=nothing) where d
    return AutoSymPTR.quadsum(rule, f, B, buffer) * ((abs(det(B.B)) / (rule.npt^d * rule.nsyms)))
end

# rule definition

struct FourierMonkhorstPackRule{S,M}
    s::S
    m::M
end

function FourierMonkhorstPackRule(s, syms, a, nmin, nmax, n₀, Δn)
    mp = AutoSymPTR.MonkhorstPackRule(syms, a, nmin, nmax, n₀, Δn)
    return FourierMonkhorstPackRule(s, mp)
end
AutoSymPTR.nsyms(r::FourierMonkhorstPackRule) = AutoSymPTR.nsyms(r.m)

function init_fourier_rule(s::AbstractFourierSeries, bz::SymmetricBZ, alg::AutoPTR)
    return FourierMonkhorstPackRule(s, bz.syms, alg.a, alg.nmin, alg.nmax, alg.n₀, alg.Δn)
end
function init_cacheval(f::FourierIntegrand, bz::SymmetricBZ, p, alg::AutoPTR)
    rule = init_fourier_rule(f.s, bz, alg)
    cache = AutoSymPTR.alloc_cache(eltype(bz), Val(ndims(bz)), rule)
    buffer = init_buffer(f, p, cache[1], alg.parallel)
    return (rule=rule, cache=cache, buffer=buffer)
end

function (r::FourierMonkhorstPackRule)(::Type{T}, v::Val{d}) where {T,d}
    if r.m.syms isa Nothing
        FourierPTR(r.s, T, v, r.m.n₀)
    else
        FourierMonkhorstPack(r.s, T, v, r.m.n₀, r.m.syms)
    end
end

function AutoSymPTR.nextrule(p::FourierPTR{d,T}, r::FourierMonkhorstPackRule) where {d,T}
    return FourierPTR(r.s, T, Val(d), length(p.p.x)+r.m.Δn)
end

function AutoSymPTR.nextrule(p::FourierMonkhorstPack{d,T}, r::FourierMonkhorstPackRule) where {d,T}
    return FourierMonkhorstPack(r.s, T, Val(d), p.npt+r.m.Δn, r.m.syms)
end

# TAI rule (no optimization for FourierIntegrand)
