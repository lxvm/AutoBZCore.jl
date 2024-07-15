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

# We use the pattern of allowing the user to pass a container with the integrand, Fourier
# series and workspace, and use dispatch to enable the optimizations

# the nested batched integrand is optional, but when included it allows for thread-safe
# parallelization

abstract type AbstractFourierIntegralFunction <: AbstractIntegralFunction end

"""
    FourierIntegralFunction(f, s, [prototype=nothing]; alias=false)

## Arguments
- `f`: The integrand, accepting inputs `f(x, s(x), p)`
- `s::AbstractFourierSeries`: The Fourier series to evaluate
- `prototype`:
- `alias::Bool`: whether to `deepcopy` the series (false) or use the series as-is (true)
"""
struct FourierIntegralFunction{F,S,P} <: AbstractFourierIntegralFunction
    f::F
    s::S
    prototype::P
    alias::Bool
end
FourierIntegralFunction(f, s, p=nothing; alias=false) = FourierIntegralFunction(f, s, p, alias)

function get_prototype(f::FourierIntegralFunction, x, ws, p)
    f.prototype === nothing ? f.f(x, ws(x), p) : f.prototype
end
get_prototype(f::FourierIntegralFunction, x, p) = get_prototype(f, x, f.s, p)

function get_fourierworkspace(f::AbstractFourierIntegralFunction)
    f.s isa FourierWorkspace ? f.s : FourierSeriesEvaluators.workspace_allocate(f.alias ? f.s : deepcopy(f.s), FourierSeriesEvaluators.period(f.s))
end

# TODO implement FourierInplaceIntegrand FourierInplaceBatchIntegrand

"""
    CommonSolveFourierIntegralFunction(prob, alg, update!, postsolve, s, [prototype, specialize]; alias=false, kws...)

Constructor for an integrand that solves a problem defined with the CommonSolve.jl
interface, `prob`, which is instantiated using `init(prob, alg; kws...)`. Helper functions
include: `update!(cache, x, s(x), p)` is called before
`solve!(cache)`, followed by `postsolve(sol, x, s(x), p)`, which should return the value of the solution.
The `prototype` argument can help control how much to `specialize` on the type of the
problem, which defaults to `FullSpecialize()` so that run times are improved. However
`FunctionWrapperSpecialize()` may help reduce compile times.
"""
struct CommonSolveFourierIntegralFunction{P,A,S,K,U,PS,T,M<:AbstractSpecialization} <: AbstractFourierIntegralFunction
    prob::P
    alg::A
    s::S
    kwargs::K
    update!::U
    postsolve::PS
    prototype::T
    specialize::M
    alias::Bool
end
function CommonSolveFourierIntegralFunction(prob, alg, update!, postsolve, s, prototype=nothing, specialize=FullSpecialize(); alias=false, kws...)
    return CommonSolveFourierIntegralFunction(prob, alg, s, NamedTuple(kws), update!, postsolve, prototype, specialize, alias)
end

function do_solve!(cache, f::CommonSolveFourierIntegralFunction, x, s, p)
    f.update!(cache, x, s, p)
    sol = solve!(cache)
    return f.postsolve(sol, x, s, p)
end
function get_prototype(f::CommonSolveFourierIntegralFunction, x, ws, p)
    if isnothing(f.prototype)
        cache = init(f.prob, f.alg; f.kwargs...)
        do_solve!(cache, f, x, ws(x), p)
    else
        f.prototype
    end
end
get_prototype(f::CommonSolveFourierIntegralFunction, x, p) = get_prototype(f, x, f.s, p)

function init_specialized_fourierintegrand(cache, f, dom, p; x=get_prototype(dom), ws=f.s, s = ws(x), prototype=f.prototype)
    proto = prototype === nothing ? do_solve!(cache, f, x, s, p) : prototype
    func = (x, s, p) -> do_solve!(cache, f, x, s, p)
    integrand = if f.specialize isa FullSpecialize
        func
    elseif f.specialize isa FunctionWrapperSpecialize
        FunctionWrapper{typeof(prototype), typeof((x, s, p))}(func)
    else
        throw(ArgumentError("$(f.specialize) is not implemented"))
    end
    return integrand, proto
end
function _init_commonsolvefourierfunction(f, dom, p; kws...)
    cache = init(f.prob, f.alg; f.kwargs...)
    integrand, prototype = init_specialized_fourierintegrand(cache, f, dom, p; kws...)
    return cache, integrand, prototype
end

# TODO implement CommonSolveFourierInplaceIntegrand CommonSolveFourierInplaceBatchIntegrand

# similar to workspace_allocate, but more type-stable because of loop unrolling and vector types
function workspace_allocate_vec(s::AbstractFourierSeries{N}, x::NTuple{N,Any}, len::NTuple{N,Integer}=ntuple(one,Val(N))) where {N}
    # Only the top-level workspace has an AbstractFourierSeries in the series field
    # In the lower level workspaces the series field has a cache that can be contract!-ed
    # into a series
    dim = Val(N)
    if N == 1
        c = FourierSeriesEvaluators.allocate(s, x[N], dim)
        ws = Vector{typeof(c)}(undef, len[N])
        ws[1] = c
        for n in 2:len[N]
            ws[n] = FourierSeriesEvaluators.allocate(s, x[N], dim)
        end
    else
        c = FourierSeriesEvaluators.allocate(s, x[N], dim)
        t = FourierSeriesEvaluators.contract!(c, s, x[N], dim)
        c_ = FourierWorkspace(c, FourierSeriesEvaluators.workspace_allocate(t, x[1:N-1], len[1:N-1]).cache)
        ws = Vector{typeof(c_)}(undef, len[N])
        ws[1] = c_
        for n in 2:len[N]
            _c = FourierSeriesEvaluators.allocate(s, x[N], dim)
            _t = FourierSeriesEvaluators.contract!(_c, s, x[N], dim)
            ws[n] = FourierWorkspace(_c, FourierSeriesEvaluators.workspace_allocate(_t, x[1:N-1], len[1:N-1]).cache)
        end
    end
    return FourierWorkspace(s, ws)
end

struct FourierValue{X,S}
    x::X
    s::S
end
@inline AutoSymPTR.mymul(w, x::FourierValue) = FourierValue(AutoSymPTR.mymul(w, x.x), x.s)
@inline AutoSymPTR.mymul(::AutoSymPTR.One, x::FourierValue) = x

function init_cacheval(f::FourierIntegralFunction, dom, p, alg::QuadGKJL; kws...)
    segs = PuncturedInterval(dom)
    ws = get_fourierworkspace(f)
    prototype = get_prototype(f, get_prototype(segs), ws, p)
    return init_segbuf(prototype, segs, alg), ws
end
function init_cacheval(f::CommonSolveFourierIntegralFunction, dom, p, alg::QuadGKJL; kws...)
    segs = PuncturedInterval(dom)
    x = get_prototype(segs)
    ws = get_fourierworkspace(f)
    cache, integrand, prototype = _init_commonsolvefourierfunction(f, dom, p; x, ws)
    return init_segbuf(prototype, segs, alg), ws, cache, integrand
end
function call_quadgk(f::FourierIntegralFunction, p, u, usegs, cacheval; kws...)
    segbuf, ws = cacheval
    quadgk(x -> (ux =
     u*x; f.f(ux, ws(ux), p)), usegs...; kws..., segbuf)
end
function call_quadgk(f::CommonSolveFourierIntegralFunction, p, u, usegs, cacheval; kws...)
    segbuf, ws, _, integrand = cacheval
    quadgk(x -> (ux = u*x; integrand(ux, ws(ux), p)), usegs...; kws..., segbuf)
end

function init_cacheval(f::FourierIntegralFunction, dom, p, ::HCubatureJL; kws...)
    # TODO utilize hcubature_buffer
    ws = get_fourierworkspace(f)
    return ws
end
function hcubature_integrand(f::FourierIntegralFunction, p, a, b, ws)
    x -> f.f(x, ws(x), p)
end

function init_autosymptr_cache(f::FourierIntegralFunction, dom, p, bufsize; kws...)
    ws = get_fourierworkspace(f)
    return (; buffer=nothing, ws)
end
function init_autosymptr_cache(f::CommonSolveFourierIntegralFunction, dom, p, bufsize; kws...)
    ws = get_fourierworkspace(f)
    cache, integrand, = _init_commonsolvefourierfunction(f, dom, p; ws)
    return (; buffer=nothing, ws, cache, integrand)
end
function autosymptr_integrand(f::FourierIntegralFunction, p, segs, cacheval)
    ws = cacheval.ws
    x -> x isa FourierValue ? f.f(x.x, x.s, p) : f.f(x, ws(x), p)
end
function autosymptr_integrand(f::CommonSolveFourierIntegralFunction, p, segs, cacheval)
    integrand = cacheval.integrand
    ws = cacheval.ws
    return x -> x isa FourierValue ? integrand(x.x, x.s, p) : integrand(x, ws(x), p)
end


function init_cacheval(f::FourierIntegralFunction, dom, p, alg::AuxQuadGKJL; kws...)
    segs = PuncturedInterval(dom)
    ws = get_fourierworkspace(f)
    prototype = get_prototype(f, get_prototype(segs), ws, p)
    return init_segbuf(prototype, segs, alg), ws
end
function init_cacheval(f::CommonSolveFourierIntegralFunction, dom, p, alg::AuxQuadGKJL; kws...)
    segs = PuncturedInterval(dom)
    ws = get_fourierworkspace(f)
    x = get_prototype(segs)
    cache, integrand, prototype = _init_commonsolvefourierfunction(f, dom, p; x, ws)
    return init_segbuf(prototype, segs, alg), ws, cache, integrand
end
function call_auxquadgk(f::FourierIntegralFunction, p, u, usegs, cacheval; kws...)
    segbuf, ws = cacheval
    auxquadgk(x -> (ux=u*x; f.f(ux, ws(ux), p)), usegs...; kws..., segbuf)
end
function call_auxquadgk(f::CommonSolveFourierIntegralFunction, p, u, usegs, cacheval; kws...)
    # cache = cacheval[2] could call do_solve!(cache, f, x, p) to fully specialize
    segbuf, ws, _, integrand = cacheval
    auxquadgk(x -> (ux=u*x; integrand(ux, ws(ux), p)), usegs...; kws..., segbuf)
end


function init_cacheval(f::FourierIntegralFunction, dom, p, alg::ContQuadGKJL; kws...)
    segs = PuncturedInterval(dom)
    ws = get_fourierworkspace(f)
    prototype = get_prototype(f, get_prototype(segs), ws, p)
    segbufs = init_csegbuf(prototype, dom, alg)
    return (; segbufs..., ws)
end
function init_cacheval(f::CommonSolveFourierIntegralFunction, dom, p, alg::ContQuadGKJL; kws...)
    segs = PuncturedInterval(dom)
    ws = get_fourierworkspace(f)
    cache, integrand, prototype = _init_commonsolvefourierfunction(f, dom, p; ws, x=get_prototype(segs))
    segbufs = init_csegbuf(prototype, dom, alg)
    return (; segbufs..., ws, cache, integrand)
end
function call_contquadgk(f::FourierIntegralFunction, p, segs, cacheval; kws...)
    ws = cacheval.ws
    contquadgk(x -> f.f(x, ws(x), p), segs; kws...)
end
function call_contquadgk(f::CommonSolveFourierIntegralFunction, p, segs, cacheval; kws...)
    integrand = cacheval.integrand
    ws = cacheval.ws
    contquadgk(x -> integrand(x, ws(x), p), segs...; kws...)
end

function init_cacheval(f::FourierIntegralFunction, dom, p, alg::MeroQuadGKJL; kws...)
    segs = PuncturedInterval(dom)
    ws = get_fourierworkspace(f)
    prototype = get_prototype(f, get_prototype(segs), ws, p)
    segbuf = init_msegbuf(prototype, dom, alg)
    return (; segbuf, ws)
end
function init_cacheval(f::CommonSolveFourierIntegralFunction, dom, p, alg::MeroQuadGKJL; kws...)
    segs = PuncturedInterval(dom)
    ws = get_fourierworkspace(f)
    cache, integrand, prototype = _init_commonsolvefourierfunction(f, dom, p; ws, x=get_prototype(segs))
    segbuf = init_msegbuf(prototype, dom, alg)
    return (; segbuf, ws, cache, integrand)
end
function call_meroquadgk(f::FourierIntegralFunction, p, segs, cacheval; kws...)
    ws = cacheval.ws
    meroquadgk(x -> f.f(x, ws(x), p), segs; kws...)
end
function call_meroquadgk(f::CommonSolveFourierIntegralFunction, p, segs, cacheval; kws...)
    integrand = cacheval.integrand
    ws = cacheval.ws
    meroquadgk(x -> integrand(x, ws(x), p), segs...; kws...)
end

function _fourier_update!(cache, x, p)
    _update!(cache, x, p)
    s = workspace_contract!(p.ws, x)
    cache.cacheval.p = (; cache.cacheval.p..., ws=s)
    return
end
function inner_integralfunction(f::FourierIntegralFunction, x0, p)
    ws = get_fourierworkspace(f)
    proto = get_prototype(f, x0, ws, p)
    func = IntegralFunction(proto) do x, (; p, ws, lims_state)
        f.f(limit_iterate(lims_state..., x), workspace_evaluate!(ws, x), p)
    end
    return func, ws
end
function outer_integralfunction(f::FourierIntegralFunction, x0, p)
    ws = get_fourierworkspace(f)
    proto = get_prototype(f, x0, ws, p)
    s = workspace_contract!(ws, x0[end])
    func = FourierIntegralFunction(f.f, s, proto; alias=true)
    return func, ws, _fourier_update!, _postsolve
end
# TODO it would be desirable to allow the inner integralfunction to be of the
# same type as f, which requires moving workspace out of the parameters into
# some kind of mutable storage
function inner_integralfunction(f::CommonSolveFourierIntegralFunction, x0, p)
    ws = get_fourierworkspace(f)
    proto = get_prototype(f, x0, ws, p)
    cache = init(f.prob, f.alg; f.kwargs...)
    func = IntegralFunction(proto) do x, (; p, ws, lims_state)
        y = limit_iterate(lims_state..., x)
        s = workspace_evaluate!(ws, x)
        do_solve!(cache, f, y, s, p)
    end
    return func, ws
end
function outer_integralfunction(f::CommonSolveFourierIntegralFunction, x0, p)
    ws = get_fourierworkspace(f)
    proto = get_prototype(f, x0, ws, p)
    s = workspace_contract!(ws, x0[end])
    func = CommonSolveFourierIntegralFunction(f.prob, f.alg, f.update!, f.postsolve, s, proto, f.specialize; alias=true, f.kwargs...)
    return func, ws, _fourier_update!, _postsolve
end

# PTR rules

# no symmetries
struct FourierPTR{N,T,S,X} <: AbstractArray{Tuple{AutoSymPTR.One,FourierValue{SVector{N,T},S}},N}
    s::Array{S,N}
    p::AutoSymPTR.PTR{N,T,X}
end

function fourier_ptr!(vals::AbstractArray{T,1}, w::FourierWorkspace, x::AbstractVector) where {T}
    t = period(w.series, 1)
    if length(w.cache) === 1
        for (i, y) in zip(eachindex(vals), x)
            @inbounds vals[i] = workspace_evaluate!(w, t*y)
        end
    else
        # we batch for memory locality in vals array on each thread
        Threads.@threads for (vrange, ichunk) in chunks(axes(vals, 1), length(w.cache), :batch)
            for i in vrange
                @inbounds vals[i] = workspace_evaluate!(w, t*x[i], ichunk)
            end
        end
    end
    return vals
end
function fourier_ptr!(vals::AbstractArray{T,d}, w::FourierWorkspace, x::AbstractVector) where {T,d}
    t = period(w.series, d)
    if length(w.cache) === 1
        for (y, v) in zip(x, eachslice(vals, dims=d))
            fourier_ptr!(v, workspace_contract!(w, t*y), x)
        end
    else
        # we batch for memory locality in vals array on each thread
        Threads.@threads for (vrange, ichunk) in chunks(axes(vals, d), length(w.cache), :batch)
            for i in vrange
                ws = workspace_contract!(w, t*x[i], ichunk)
                fourier_ptr!(view(vals, ntuple(_->(:),Val(d-1))..., i), ws, x)
            end
        end
    end
    return vals
end

function FourierPTR(w::FourierWorkspace, ::Type{T}, ndim, npt) where {T}
    FourierSeriesEvaluators.isinplace(w.series) && throw(ArgumentError("inplace series not supported for PTR - please file a bug report"))
    # unitless quadrature weight/node, but unitful value to Fourier series
    p = AutoSymPTR.PTR(typeof(float(real(one(T)))), ndim, npt)
    s = workspace_evaluate(w, ntuple(_->zero(T), ndim))
    vals = similar(p, typeof(s))
    fourier_ptr!(vals, w, p.x)
    return FourierPTR(vals, p)
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

function (rule::FourierPTR)(f::F, B::Basis, buffer=nothing) where {F}
    arule = AutoSymPTR.AffineQuad(rule, B)
    return AutoSymPTR.quadsum(arule, f, arule.vol / length(rule), buffer)
end

# SymPTR rules
struct FourierMonkhorstPack{d,W,T,S}
    npt::Int64
    nsyms::Int64
    wxs::Vector{Tuple{W,FourierValue{SVector{d,T},S}}}
end

function _fourier_symptr!(vals::AbstractVector, w::FourierWorkspace, x::AbstractVector, npt, wsym, ::Tuple{}, idx, coord, offset)
    t = period(w.series, 1)
    o = offset-1
    # we can't parallelize the inner loop without knowing the offsets of each contiguous
    # chunk, which would require a ragged array to store. We would be better off with
    # changing the symptr algorithm to compute a convex ibz
    # but for 3D grids this inner loop should be a large enough base case to make
    # parallelizing worth it, although the workloads will vary piecewise linearly as a
    # function of the slice, so we should distribute points using :scatter
    n = 0
    for i in 1:npt
        @inbounds wi = wsym[i, idx...]
        iszero(wi) && continue
        @inbounds xi = x[i]
        vals[o+(n+=1)] = (wi, FourierValue(SVector(xi, coord...), workspace_evaluate!(w, t*xi)))
    end
    return vals
end
function _fourier_symptr!(vals::AbstractVector, w::FourierWorkspace, x::AbstractVector, npt, wsym, flags, idx, coord, offset)
    d = ndims(w.series)
    t = period(w.series, d)
    flag, f = flags[begin:end-1], flags[end]
    if (len = length(w.cache)) === 1 # || len <= w.basecasesize[d]
        for i in 1:npt
            @inbounds(fi = f[i, idx...]) == 0 && continue
            @inbounds xi = x[i]
            ws = workspace_contract!(w, t*xi)
            _fourier_symptr!(vals, ws, x, npt, wsym, flag, (i, idx...), (xi, coord...), fi)
        end
    else
        # since we don't know the distribution of ibz nodes, other than that it will be
        # piecewise linear, our best chance for a speedup from parallelizing is to scatter
        Threads.@threads for (vrange, ichunk) in chunks(1:npt, len, :scatter)
            for i in vrange
                @inbounds(fi = f[i, idx...]) == 0 && continue
                @inbounds xi = x[i]
                ws = workspace_contract!(w, t*xi, ichunk)
                _fourier_symptr!(vals, ws, x, npt, wsym, flag, (i, idx...), (xi, coord...), fi)
            end
        end
    end
    return vals
end

function fourier_symptr!(wxs, w, u, npt, wsym, flags)
    flag, f = flags[begin:end-1], flags[end]
    return _fourier_symptr!(wxs, w, u, npt, wsym, flag, (), (), f[])
end

function FourierMonkhorstPack(w::FourierWorkspace, ::Type{T}, ndim::Val{d}, npt, syms) where {d,T}
    # unitless quadrature weight/node, but unitful value to Fourier series
    FourierSeriesEvaluators.isinplace(w.series) && throw(ArgumentError("inplace series not supported for PTR - please file a bug report"))
    u = AutoSymPTR.ptrpoints(typeof(float(real(one(T)))), npt)
    s = w(map(*, period(w.series), ntuple(_->zero(T), ndim)))
    # the bottleneck is likely to be symptr_rule, which is not a fast or parallel algorithm
    wsym, flags, nsym = AutoSymPTR.symptr_rule(npt, ndim, syms)
    wxs = Vector{Tuple{eltype(wsym),FourierValue{SVector{d,eltype(u)},typeof(s)}}}(undef, nsym)
    # fourier_symptr! may be worth parallelizing for expensive Fourier series, but may not
    # be the bottleneck
    fourier_symptr!(wxs, w, u, npt, wsym, flags)
    return FourierMonkhorstPack(npt, length(syms), wxs)
end

# indexing
Base.getindex(rule::FourierMonkhorstPack, i::Int) = rule.wxs[i]

# iteration
Base.eltype(::Type{FourierMonkhorstPack{d,W,T,S}}) where {d,W,T,S} = Tuple{W,FourierValue{SVector{d,T},S}}
Base.length(r::FourierMonkhorstPack) = length(r.wxs)
Base.iterate(rule::FourierMonkhorstPack, args...) = iterate(rule.wxs, args...)

function (rule::FourierMonkhorstPack{d})(f::F, B::Basis, buffer=nothing) where {d,F}
    arule = AutoSymPTR.AffineQuad(rule, B)
    return AutoSymPTR.quadsum(arule, f, arule.vol / (rule.npt^d * rule.nsyms), buffer)
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

function AutoSymPTR.nextrule(p::FourierMonkhorstPack{d,W,T}, r::FourierMonkhorstPackRule) where {d,W,T}
    return FourierMonkhorstPack(r.s, T, Val(d), p.npt+r.m.Δn, r.m.syms)
end

# dispatch on PTR algorithms

# function init_buffer(f::FourierIntegrand, len)
#     return f.nest isa NestedBatchIntegrand ? Vector{eltype(f.nest.y)}(undef, len) : nothing
# end

function init_fourier_rule(w::FourierWorkspace, dom, alg::MonkhorstPack)
    @assert ndims(w.series) == ndims(dom)
    if alg.syms === nothing
        return FourierPTR(w, eltype(dom), Val(ndims(dom)), alg.npt)
    else
        return FourierMonkhorstPack(w, eltype(dom), Val(ndims(dom)), alg.npt, alg.syms)
    end
end
function init_cacheval(f::AbstractFourierIntegralFunction, dom , p, alg::MonkhorstPack; kws...)
    cache = init_autosymptr_cache(f, dom, p, alg.nthreads; kws...)
    ws = cache.ws
    rule = init_fourier_rule(ws, dom, alg)
    return (; rule, buffer=nothing, ws, cache...)
end

function init_fourier_rule(w::FourierWorkspace, dom, alg::AutoSymPTRJL)
    @assert ndims(w.series) == ndims(dom)
    return FourierMonkhorstPackRule(w, alg.syms, alg.a, alg.nmin, alg.nmax, alg.n₀, alg.Δn)
end
function init_fourier_rule(w::FourierWorkspace, dom::RepBZ, alg::AutoSymPTRJL)
    B = get_basis(dom)
    rule = init_fourier_rule(w, B, alg)
    return SymmetricRuleDef(rule, dom.rep, dom.bz)
end
function init_cacheval(f::AbstractFourierIntegralFunction, dom, p, alg::AutoSymPTRJL; kws...)
    cache = init_autosymptr_cache(f, dom, p, alg.nthreads; kws...)
    ws = cache.ws
    rule = init_fourier_rule(ws, dom, alg)
    rule_cache = AutoSymPTR.alloc_cache(eltype(dom), Val(ndims(dom)), rule)
    return (; rule, rule_cache, cache...)
end
