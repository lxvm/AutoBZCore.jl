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
    CommonSolveFourierIntegralFunction(solve!, prob, alg, s, [prototype, specialize, executor]; alias=false, kws...)

Constructor for an integrand that solves a problem defined with the CommonSolve.jl
interface, `prob`, which is instantiated using `init(prob, alg; kws...)`. The function `sol = solve!(solver, x, s(x), p)` should return the value of the solution and `s` must be a Fourier series.
The `prototype` argument can help control how much to `specialize` on the solution type of the
problem. By default, `specialize=DefaultSpecialize()` uses Julia's default heuristics, which can give up on inference in complicated codes.
Additionally, `FullSpecialize()` can obtain the fastest run times with the longest compile times, `NoSpecialize()` strikes a good balance of run time, compile time and inference, and `FunctionWrapperSpecialize()` may have the fastest compile time and very good run times (comparable to `FullSpecialize()`) but with possible issues regarding world age.
The `executor` keyword specifies how to schedule and run the integrand evaluation, defaulting to `SerialExecutor()` with an additional option for `ThreadedExecutor(::Integer)`.
"""
struct CommonSolveFourierIntegralFunction{F,P,A,S,K,T,M<:AbstractSpecialization,E<:AbstractExecutor} <: AbstractFourierIntegralFunction
    solve!::F
    prob::P
    alg::A
    s::S
    kwargs::K
    prototype::T
    specialize::M
    executor::E
    alias::Bool
end
function CommonSolveFourierIntegralFunction(solve!, prob, alg, s, prototype=nothing, specialize=FullSpecialize(), executor=SerialExecutor(); alias=false, kws...)
    return CommonSolveFourierIntegralFunction(solve!, prob, alg, s, NamedTuple(kws), prototype, specialize, executor, alias)
end

function do_solve!(solver, f::CommonSolveFourierIntegralFunction, x, s, p)
    return f.solve!(solver, x, s, p)
end
function get_prototype(f::CommonSolveFourierIntegralFunction, x, ws, p)
    if isnothing(f.prototype)
        solver = init(f.prob, f.alg; f.kwargs...)
        do_solve!(solver, f, x, ws(x), p)
    else
        f.prototype
    end
end
get_prototype(f::CommonSolveFourierIntegralFunction, x, p) = get_prototype(f, x, f.s, p)

function init_specialized_fourierintegrand(solver, f, dom, p; x=get_prototype(dom), ws=f.s, s = ws(x), prototype=f.prototype)
    proto = prototype === nothing ? do_solve!(solver, f, x, s, p) : prototype
    func = (x, s, p) -> do_solve!(solver, f, x, s, p)
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
    solver = init(f.prob, f.alg; f.kwargs...)
    integrand, prototype = init_specialized_fourierintegrand(solver, f, dom, p; kws...)
    return solver, integrand, prototype
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
init_cacheval(f::CommonSolveFourierIntegralFunction, dom, p, alg::HCubatureJL; kws...) = init_cacheval_cs(f.executor, f, dom, p, alg; kws...)
function init_cacheval_cs(::SerialExecutor, f::CommonSolveFourierIntegralFunction, dom, p, alg::HCubatureJL; kws...)
    ws = get_fourierworkspace(f)
    solver, integrand,  = init_commonsolvefunction(f, dom, p)
    return (; ws, solver, integrand)
end
function init_cacheval_cs(exec::ThreadedExecutor, f::CommonSolveFourierIntegralFunction, dom, p, alg::HCubatureJL; kws...)
    throw(ArgumentError("HCubatureJL does not support threaded execution because it does not support batched integrands"))
end
function hcubature_integrand(f::FourierIntegralFunction, p, a, b, ws)
    x -> f.f(x, ws(x), p)
end
function hcubature_integrand(f::CommonSolveFourierIntegralFunction, p, a, b, cacheval)
    integrand = cacheval.integrand
    ws = cacheval.ws
    x -> integrand(x, ws(x), p)
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


function init_cacheval(f::AbstractFourierIntegralFunction, dom, p, alg::AuxQuadGKJL; kws...)
    g = fourier_to_standard(f, get_prototype(dom), p)
    return g, init_cacheval(g, dom, p, alg; kws...)
end
function call_auxquadgk(f::AbstractFourierIntegralFunction, p, u, usegs, (g, cacheval); kws...)
    return call_auxquadgk(g, p, u, usegs, cacheval; kws...)
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


function insert_counter(f::CommonSolveFourierIntegralFunction, x, p, channel)
    input = (; x, p, s=f.s(x))
    prob = ComposedCommonSolveProblem(input, CounterProblem(; channel), f.prob) do (; x, s, p), countersolver, probsolver
        step!(countersolver)
        return f.solve!(probsolver, x, s, p)
    end
    alg = ComposedCommonSolveAlgorithm(SingleCount(), f.alg)
    return CommonSolveFourierIntegralFunction(prob, alg, f.s, f.prototype, f.specialize, f.executor) do solver, x, s, p
        # solver.input = (; solver.input..., x, s, p)
        # return solve!(solver)
        ## using out-of-place semantics can be faster
        return solver.solve!((; x, s, p), solver.solvers...)
    end
end

struct FourierEliminationProblem{F<:AbstractFourierSeries,X,D,K}
    f::F
    x::X
    dim::D
    kwargs::K
end
FourierEliminationProblem(f::AbstractFourierSeries, x, dim; kws...) = FourierEliminationProblem(f, x, dim, kws)
struct FourierEliminationAlgorithm end
mutable struct FourierEliminationSolver{F,X,D,A,C,K}
    f::F
    x::X
    dim::D
    alg::A
    cacheval::C
    kwargs::K
end
function init(prob::FourierEliminationProblem, alg::FourierEliminationAlgorithm; kws...)
    kwargs = (; prob.kwargs..., kws...)
    cacheval = FourierSeriesEvaluators.allocate(prob.f, prob.x, prob.dim)
    return FourierEliminationSolver(prob.f, prob.x, prob.dim, alg, cacheval, kwargs)
end
function solve!(solver::FourierEliminationSolver)
    return FourierSeriesEvaluators.contract!(solver.cacheval, solver.f, solver.x, solver.dim)
end

function nested_innerintegralfunction(f::FourierIntegralFunction, x0, p)
    proto = get_prototype(f, x0, p)
    func = IntegralFunction(proto) do x, (; p, state, fouriercache, fourierseries)
        f.f(SVector(promote(x, state...)), FourierSeriesEvaluators.evaluate!(fouriercache, fourierseries, x), p)
    end
    return func
end

struct FourierEvaluationProblem{F<:AbstractFourierSeries,X,K}
    f::F
    x::X
    alias::Bool
    kwargs::K
end
FourierEvaluationProblem(f::AbstractFourierSeries, x, alias=false; kws...) = FourierEvaluationProblem(f, x, alias, kws)
struct FourierEvaluationAlgorithm end
mutable struct FourierEvaluationSolver{F,X,A,C,K}
    f::F
    x::X
    alg::A
    cacheval::C
    kwargs::K
end
function init(prob::FourierEvaluationProblem, alg::FourierEvaluationAlgorithm; kws...)
    kwargs = (; prob.kwargs..., kws...)
    cacheval = FourierSeriesEvaluators.allocate(prob.f, prob.x, Val(1))#prob.dim)
    return FourierEvaluationSolver(prob.f, prob.x, alg, cacheval, kwargs)
end
function solve!(solver::FourierEvaluationSolver)
    return FourierSeriesEvaluators.evaluate!(solver.cacheval, solver.f, solver.x)
end

function fourier_to_standard(func::FourierIntegralFunction, x0, p)
    (; f, s, prototype, alias) = func
    prob = FourierEvaluationProblem(s, x0, alias)
    alg = FourierEvaluationAlgorithm()
    _solve! = (solver, x, p) -> begin
        solver.x = x
        sol = solve!(solver)
        sol = FourierSeriesEvaluators.evaluate!(solver.cacheval, solver.f, x)
        return f(x, sol, p)
    end
    CommonSolveIntegralFunction(_solve!, prob, alg, prototype)
end
function fourier_to_standard(func::CommonSolveFourierIntegralFunction, x0, p)
    (; prob, alg, s, kwargs, prototype, specialize, executor, alias) = func
    fourierprob = FourierEvaluationProblem(s, x0, alias)
    fourieralg = FourierEvaluationAlgorithm()
    fullprob = ComposedCommonSolveProblem((; x=x0, p=p), fourierprob, prob) do ((; x, p), fouriersolver, probsolver)
        fouriersolver.x = x
        fx = solve!(fouriersolver)
        return func.solve!(probsolver, x, fx, p)
    end
    fullalg = ComposedCommonSolveAlgorithm(fourieralg, alg)
    _solve! = (solver, x, p) -> begin
        solver.input = (; x, p)
        return solve!(solver)
    end
    CommonSolveIntegralFunction(_solve!, fullprob, fullalg, prototype, specialize, executor; kwargs...)
end

struct ComposedCommonSolveProblem{P,S,I,K}
    problems::P
    solve!::S
    input::I
    kwargs::K
    ComposedCommonSolveProblem(solve!, input, probs...; kws...) = new{typeof(probs),typeof(solve!),typeof(input),typeof(kws)}(probs, solve!, input, kws)
end

struct ComposedCommonSolveAlgorithm{A}
    algorithms::A
    ComposedCommonSolveAlgorithm(algs...) = new{typeof(algs)}(algs)
end

mutable struct ComposedCommonSolveSolver{S,SS,I,K}
    solvers::S
    solve!::SS
    input::I
    kwargs::K
end
function init(prob::ComposedCommonSolveProblem, alg::ComposedCommonSolveAlgorithm; kws...)
    kwargs = (; prob.kwargs..., kws...)
    solvers = map(init, prob.problems, alg.algorithms)
    return ComposedCommonSolveSolver(solvers, prob.solve!, prob.input, kwargs)
end
function solve!(solver::ComposedCommonSolveSolver)
    return solver.solve!(solver.input, solver.solvers...; solver.kwargs...)
end

function init_cacheval(f::AbstractFourierIntegralFunction, dom, p, alg::NestedQuad; kws...)
    x0, segs, lims, states = unroll_limits(dom)
    series = unroll_series(f.s)
    algs = alg.algs isa IntegralAlgorithm ? ntuple(i -> alg.algs, Val(ndims(dom))) : alg.algs
    spec = alg.specialize isa AbstractSpecialization ? ntuple(i -> alg.specialize, Val(ndims(dom))) : alg.specialize
    exec = alg.executor isa AbstractExecutor ? ntuple(i -> alg.executor, Val(ndims(dom))) : alg.executor

    _f = nested_innerfourierintegralfunction(f, x0, series[1], x0[1], p)

    intprob = IntegralProblem(_f, segs[1], (; p, state=states[1], series=series[1]); _rescale_abstol(1/real(prod(x0[begin+1:end])); kws...)...)
    segprob = SegmentProblem(lims[1], 1)
    innerinput = (; lims=lims[1], dim=Val(1), state=states[1], p, series=series[1], kws=intprob.kwargs)
    innerprob = ComposedCommonSolveProblem(innerinput, segprob, intprob) do (; lims, dim, state, p, series, kws), segsolver, intsolver
        segsolver.lims = lims
        segsolver.dim = _val_unwrap(dim)
        _segs = solve!(segsolver)
        intsolver.dom = _segs
        intsolver.p = (; intsolver.p..., p, state, series)
        intsolver.kwargs = (; intsolver.kwargs..., kws...)
        return solve!(intsolver)
    end
    inneralg = ComposedCommonSolveAlgorithm(SegmentAlgorithm(), algs[1])
    prob, alg = nested_fourierprob(innerprob, inneralg, _f.prototype, p, x0, segs[2:end], lims[2:end], states[2:end], algs[2:end], spec[2:end], exec[2:end], series[2:end]; kws...)
    return init(prob, alg)
end
function nested_fourierprob(innerprob, inneralg, prototype, p, x0, segs, lims, states, algs, spec, exec, series; kws...)
    length(states) == 0 && return innerprob, inneralg
    elimprob = EliminationProblem(lims[1], x0[ndims(lims[1])], Val(ndims(lims[1])))
    eliminput = (; x=x0[ndims(lims[1])], lims=lims[1], state=states[1], dim=Val(ndims(lims[1])), series=series[1], p, kws=innerprob.input.kws)
    fourierprob = FourierEliminationProblem(series[1], x0[ndims(lims[1])], Val(ndims(lims[1])))
    _prob = ComposedCommonSolveProblem(eliminput, elimprob, fourierprob, innerprob) do (; x, lims, state, dim, p, series, kws), elimsolver, fouriersolver, innersolver
        elimsolver.x = x
        elimsolver.lims = lims
        elimsolver.dim = dim
        _state = (x, state...)
        _lims = solve!(elimsolver)
        fouriersolver.f = series
        fouriersolver.x = x
        fouriersolver.dim = dim
        _series = solve!(fouriersolver)
        innersolver.input = (; innersolver.input..., lims=_lims, state=_state, dim=Val(_val_unwrap(dim)-1), series=_series, p, kws)
        return solve!(innersolver)
    end
    _alg = ComposedCommonSolveAlgorithm(EliminationAlgorithm(), FourierEliminationAlgorithm(), inneralg)
    _f = CommonSolveIntegralFunction(_prob, _alg, prototype*real(prod(x0[begin:ndims(lims[1])-1])), spec[1], exec[1]) do solver, x, p
        # solver.input = (; solver.input..., p..., x)
        # sol = solve!(solver)
        ## out-of-place semantics may be faster
        sol = solver.solve!((; solver.input..., p..., x), solver.solvers...)
        return sol.value
    end
    __f = nested_integralfunction(_f, x0[ndims(lims[1])], p)
    _kws = _rescale_abstol(1/real(prod(x0[begin+ndims(lims[1]):end])); kws...)
    innerinput = (; lims=lims[1], dim=Val(ndims(lims[1])), state=states[1], series=series[1], p, kws=_kws)
    intprob = IntegralProblem(__f, segs[1], (; innerinput..., kws=eliminput.kws); _kws...)

    segprob = SegmentProblem(lims[1], ndims(lims[1]))
    _innerprob = ComposedCommonSolveProblem(innerinput, segprob, intprob) do (; lims, dim, state, series, p, kws), segsolver, intsolver
        segsolver.lims = lims
        segsolver.dim = _val_unwrap(dim)
        _segs = solve!(segsolver)
        intsolver.dom = _segs
        len = abs(_segs[end]-_segs[begin])
        # TODO figure out type instability/GC in line below
        intsolver.p = (; intsolver.p..., lims, state, series, dim, p, kws=_rescale_abstol(1/len; kws...))
        intsolver.kwargs = (; intsolver.kwargs..., kws...)
        return solve!(intsolver)
    end
    _inneralg = ComposedCommonSolveAlgorithm(SegmentAlgorithm(), algs[1])

    nested_fourierprob(_innerprob, _inneralg, prototype, p, x0, segs[2:end], lims[2:end], states[2:end], algs[2:end], spec[2:end], exec[2:end], series[2:end]; kws...)
end

function unroll_series(s::AbstractFourierSeries)
    if ndims(s) == 1
        return (s,)
    else
        (unroll_series(FourierSeriesEvaluators.contract(s, FourierSeriesEvaluators.period(s, ndims(s)), Val(ndims(s))))..., s)
    end
end

function nested_innerfourierintegralfunction(f::FourierIntegralFunction, x0, series, x1, p)
    proto = get_prototype(f, x0, p)

    prob = FourierEvaluationProblem(series, x1)
    alg = FourierEvaluationAlgorithm()

    _f = f.f
    func = CommonSolveIntegralFunction(prob, alg, proto) do solver, x, p
        # solver.x = x
        # solver.f = p.series
        # sol = solve!(solver)
        ## using out-of-place semantics can be faster
        sol = FourierSeriesEvaluators.evaluate!(solver.cacheval, p.series, x)
        return _f(SVector(promote(x, state...)), sol, p)
    end
    return func
end
function nested_innerfourierintegralfunction(f::CommonSolveFourierIntegralFunction, x0, series, x1, p)
    return nested_innerfourierintegralfunction_cs(f.executor, f, x0, series, x1, p)
end
function nested_innerfourierintegralfunction_cs(::SerialExecutor, f, x0, series, x1, p)
    proto = get_prototype(f, x0, p)

    prob = FourierEvaluationProblem(series, x1)
    alg = FourierEvaluationAlgorithm()
    input = (; x=x0, x1, series, p)
    cprob = ComposedCommonSolveProblem(input, prob, f.prob) do (; x, x1, p, series), fouriersolver, fsolver
        # fouriersolver.f = series
        # fouriersolver.x = x1
        # s = solve!(fouriersolver)
        ## for performance, eliding the setfield! call is helpful for small fourier series
        s = FourierSeriesEvaluators.evaluate!(fouriersolver.cacheval, series, x1)
        return f.solve!(fsolver, x, s, p)
    end

    calg = ComposedCommonSolveAlgorithm(alg, f.alg)
    return CommonSolveIntegralFunction(cprob, calg, proto, f.specialize, f.executor; f.kwargs...) do solver, x, (; p, state, series)
        # solver.input = (; solver.input..., x1=x, x=SVector(promote(x, state...)), p, series)
        # return solve!(solver)
        ## out-of-place semantics may be faster
        return solver.solve!((; solver.input..., x1=x, x=SVector(promote(x, state...)), p, series), solver.solvers...)
    end
end
function nested_innerfourierintegralfunction_cs(exec::ThreadedExecutor, f, x0, series, x1, p)
    _f = nested_innerfourierintegralfunction_cs(SerialExecutor(), f, x0, series, x1, p)
    return nested_integralfunction_cs(exec, _f, x1, p)
end