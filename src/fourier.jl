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

# TODO think about generalizing FourierIntegralFunction to a EliminationIntegralFunction
# TODO implement CommonSolveFourierInplaceIntegrand CommonSolveFourierInplaceBatchIntegrand

"""
    FourierIntegralFunction(f, s, [prototype=nothing, executor=SerialExecutor()]; alias=false)

## Arguments
- `f`: The integrand, accepting inputs `f(x, s(x), p)`
- `s::AbstractFourierSeries`: The Fourier series to evaluate
- `prototype`:
- `alias::Bool`: whether to `deepcopy` the series (false) or use the series as-is (true)
"""
struct FourierIntegralFunction{F,S,P,E<:AbstractExecutor} <: AbstractFourierIntegralFunction
    f::F
    s::S
    prototype::P
    executor::E
    alias::Bool
end
FourierIntegralFunction(f, s, p=nothing, exec=SerialExecutor(); alias=false) = FourierIntegralFunction(f, s, p, exec, alias)

function _get_prototype(f::FourierIntegralFunction, x, ws, p)
    f.prototype === nothing ? f.f(x, ws(x), p) : f.prototype
end
get_prototype(f::FourierIntegralFunction, x, p) = _get_prototype(f, x, f.s, p)

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
function CommonSolveFourierIntegralFunction(solve!, prob, alg, s, prototype=nothing, specialize=DefaultSpecialize(), executor=SerialExecutor(); alias=false, kws...)
    return CommonSolveFourierIntegralFunction(solve!, prob, alg, s, NamedTuple(kws), prototype, specialize, executor, alias)
end

function do_solve!(solver, f::CommonSolveFourierIntegralFunction, x, s, p)
    sol = f.solve!(solver, x, s, p)
    if sol isa CommonSolutionStats
        return sol.value
    else
        return sol
    end
end
function _get_prototype(f::CommonSolveFourierIntegralFunction, x, ws, p, _solver=nothing)
    if isnothing(f.prototype)
        solver = isnothing(_solver) ? init(f.prob, f.alg; f.kwargs...) : _solver
        do_solve!(solver, f, x, ws(x), p)
    else
        f.prototype
    end
end
get_prototype(f::CommonSolveFourierIntegralFunction, x, p, _solver=nothing) = _get_prototype(f, x, f.s, p, _solver)

function fourier_to_standard(func::FourierIntegralFunction, x0, p)
    (; f, s, prototype, alias, executor) = func
    prob = FourierEvaluationProblem(s, x0, alias)
    alg = FourierEvaluationAlgorithm()
    _solve! = (solver, x, p) -> begin
        solver.x = x
        sol = solve!(solver)
        return f(x, sol, p)
    end
    CommonSolveIntegralFunction(_solve!, prob, alg, prototype, DefaultSpecialize(), executor)
end
function fourier_to_standard(func::CommonSolveFourierIntegralFunction, x0, p)
    (; prob, alg, s, kwargs, prototype, specialize, executor, alias) = func
    fourierprob = FourierEvaluationProblem(s, x0, alias)
    fourieralg = FourierEvaluationAlgorithm()
    fullprob = ComposedCommonSolveProblem((; x=x0, p=p), fourierprob, prob) do (; x, p), fouriersolver, probsolver
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

struct FourierValue{X,S}
    x::X
    s::S
end
@inline AutoSymPTR.mymul(w, x::FourierValue) = FourierValue(AutoSymPTR.mymul(w, x.x), x.s)
@inline AutoSymPTR.mymul(::AutoSymPTR.One, x::FourierValue) = x

# standard integrators with a full series evaluation at each point

function init_integrand_cacheval(f::AbstractFourierIntegralFunction, dom, p)
    g = fourier_to_standard(f, get_prototype(dom), p)
    prototype, cacheval = init_integrand_cacheval(g, dom, p)
    return prototype, (g, cacheval)
end

function quadgk_integrand(f::AbstractFourierIntegralFunction, p, u, alg_cache, (g, cacheval))
    return quadgk_integrand(g, p, u, alg_cache, cacheval)
end

function autosymptr_integrand(f::AbstractFourierIntegralFunction, p, segs, alg_cache, (g, cacheval))
    return autosymptr_integrand(g, p, segs, alg_cache, cacheval)
end

# PTR rules with special series evaluation on rectangular grids

# no symmetries
struct FourierPTR{N,T,S,X} <: AbstractArray{Tuple{AutoSymPTR.One,FourierValue{SVector{N,T},S}},N}
    s::Array{S,N}
    p::AutoSymPTR.PTR{N,T,X}
end


struct ProductArray{T,N,X<:NTuple{N,<:AbstractVector{T}}} <: AbstractArray{T,N}
    xs::X
end
Base.size(p::ProductArray) = map(length, p.xs)
Base.getindex(p::ProductArray{T,N}, idx::Vararg{Int,N}) where {T,N} = SVector{N,T}(map(getindex, p.xs, idx))

Base.IteratorSize(::Type{<:ProductArray{T,N}}) where {T,N} = Base.HasShape{N}()
Base.size(p::ProductArray, dim) = length(p.xs[dim])
Base.axes(p::ProductArray) = map(eachindex, p.xs)

function FourierPTR(f::AbstractFourierSeries, ::Type{T}, ndim, npt, exec) where {T}
    FourierSeriesEvaluators.isinplace(f) && throw(ArgumentError("inplace series not supported for PTR - please file a bug report"))
    # unitless quadrature weight/node, but unitful value to Fourier series
    p = AutoSymPTR.PTR(typeof(float(real(one(T)))), ndim, npt)
    prob = FourierEvaluationProblem(f, BatchArray(ProductArray(ntuple(_->p.x * oneunit(T),ndim))))
    vals = solve(prob, FourierEvaluationAlgorithm(exec))
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
    s = w(map(*, period(w.series), ntuple(_->zero(eltype(u)), ndim)))
    # the bottleneck is likely to be symptr_rule, which is not a fast or parallel algorithm
    wsym, flags, nsym = AutoSymPTR.symptr_rule(npt, ndim, syms)
    wxs = Vector{Tuple{eltype(wsym),FourierValue{SVector{d,eltype(u)},typeof(s)}}}(undef, nsym)
    # fourier_symptr! may be worth parallelizing for expensive Fourier series, but may not
    # be the bottleneck
    fourier_symptr!(wxs, w, u, npt, wsym, flags)
    return FourierMonkhorstPack(npt, length(syms), wxs)
end

function FourierMonkhorstPack(f::AbstractFourierSeries, ::Type{T}, ndim::Val{d}, npt, syms, exec) where {d,T}
    # unitless quadrature weight/node, but unitful value to Fourier series
    # TODO recast the _fourier_symptr function as a FourierEvaluationProblem with
    # specialized batched input type
    w = workspace_allocate(f, period(f), exec isa SerialExecutor ? ntuple(_->1,ndim) : ntuple(n->n==d ? exec.ntasks : 1, ndim))
    return FourierMonkhorstPack(w, T, ndim, npt, syms)
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

struct FourierMonkhorstPackRule{S,M,E}
    s::S
    m::M
    executor::E
end

function FourierMonkhorstPackRule(s, syms, a, nmin, nmax, n₀, Δn, exec)
    mp = AutoSymPTR.MonkhorstPackRule(syms, a, nmin, nmax, n₀, Δn)
    return FourierMonkhorstPackRule(s, mp, exec)
end
AutoSymPTR.nsyms(r::FourierMonkhorstPackRule) = AutoSymPTR.nsyms(r.m)

function (r::FourierMonkhorstPackRule)(::Type{T}, v::Val{d}) where {T,d}
    if r.m.syms isa Nothing
        FourierPTR(r.s, T, v, r.m.n₀, r.executor)
    else
        FourierMonkhorstPack(r.s, T, v, r.m.n₀, r.m.syms, r.executor)
    end
end

function AutoSymPTR.nextrule(p::FourierPTR{d,T}, r::FourierMonkhorstPackRule) where {d,T}
    return FourierPTR(r.s, T, Val(d), length(p.p.x)+r.m.Δn, r.executor)
end

function AutoSymPTR.nextrule(p::FourierMonkhorstPack{d,W,T}, r::FourierMonkhorstPackRule) where {d,W,T}
    return FourierMonkhorstPack(r.s, T, Val(d), p.npt+r.m.Δn, r.m.syms, r.executor)
end

# dispatch on PTR algorithms

function init_fourier_rule(f::AbstractFourierSeries, dom, alg::MonkhorstPack, exec)
    @assert ndims(f) == ndims(dom)
    if alg.syms === nothing
        return FourierPTR(f, eltype(dom), Val(ndims(dom)), alg.npt, exec)
    else
        return FourierMonkhorstPack(f, eltype(dom), Val(ndims(dom)), alg.npt, alg.syms, exec)
    end
end
function fourier_to_partial(func::FourierIntegralFunction)
    IntegralFunction(func.prototype, func.executor) do _x, p
        _x isa FourierValue || throw(ArgumentError("expected a FourierValue"))
        return func.f(_x.x, _x.s, p)
    end
end
function fourier_to_partial(func::CommonSolveFourierIntegralFunction)
    CommonSolveIntegralFunction(func.prob, func.alg, func.prototype, func.specialize, func.executor; func.kwargs...) do solver, _x, p
        _x isa FourierValue || throw(ArgumentError("expected a FourierValue"))
        return func.solve!(solver, _x.x, _x.s, p)
    end
end

struct FourierDomain{F,D,E}
    f::F
    dom::D
    executor::E
end
Base.ndims(dom::FourierDomain) = ndims(dom.dom)
Base.eltype(::Type{FourierDomain{F,D,E}}) where {F,D,E} = eltype(D) # ? FourierValue{eltype(D),?eltype(F)}
function init_rule(dom::FourierDomain, alg::MonkhorstPack)
    return init_fourier_rule(dom.f, dom.dom, alg, dom.executor)
end
function get_prototype(dom::FourierDomain)
    x = get_prototype(dom.dom)
    return FourierValue(x, dom.f(x))
end
get_basis(dom::FourierDomain) = get_basis(dom.dom)

function init_cacheval(f::AbstractFourierIntegralFunction, dom , p, alg::MonkhorstPack; kws...)
    g = fourier_to_partial(f)
    (; rule, algorithm_cacheval, integrand_cacheval) = init_cacheval(g, FourierDomain(f.s, dom, f.executor), p, alg; kws...)
    return (; rule, algorithm_cacheval, integrand_cacheval=(g, integrand_cacheval))
end

function init_fourier_rule(f::AbstractFourierSeries, dom, alg::AutoSymPTRJL, exec)
    @assert ndims(f) == ndims(dom)
    return FourierMonkhorstPackRule(f, alg.syms, alg.a, alg.nmin, alg.nmax, alg.n₀, alg.Δn, exec)
end
function init_fourier_rule(f::AbstractFourierSeries, dom::RepBZ, alg::AutoSymPTRJL, exec)
    B = get_basis(dom)
    rule = init_fourier_rule(f, B, alg, exec)
    return SymmetricRuleDef(rule, dom.rep, dom.bz)
end
function init_rule(dom::FourierDomain, alg::AutoSymPTRJL)
    # TODO smarter parallelization of init_fourier_rule
    return init_fourier_rule(dom.f, dom.dom, alg, dom.executor)
end
function init_cacheval(f::AbstractFourierIntegralFunction, dom, p, alg::AutoSymPTRJL; kws...)
    g = fourier_to_partial(f)
    (; rule_cache, rule, algorithm_cacheval, integrand_cacheval) = init_cacheval(g, FourierDomain(f.s, dom, f.executor), p, alg; kws...)
    return (; rule_cache, rule, algorithm_cacheval, integrand_cacheval=(g, integrand_cacheval))
end


function insert_stats(alg::SolverStats, f::FourierIntegralFunction, x, p, channel)
    proto = get_prototype(f, x, p)
    prob = StatsProblem((x, f.s(x), p, proto), channel, false)
    CommonSolveFourierIntegralFunction(prob, alg.stats, f.s, proto, DefaultSpecialize(), f.executor) do solver, x, s, p
        value = f.f(x, s, p)
        step_stats!(solver, x, s, p, value)
        return value
    end
end
function insert_stats(alg::SolverStats, f::CommonSolveFourierIntegralFunction, x, p, channel)
    input = (; x, p, s=f.s(x))
    fsolve! = f.solve!
    prob = ComposedCommonSolveProblem(input, f.prob, StatsProblem((x, input.s, p, f.prototype), channel, false)) do (; x, s, p), probsolver, statssolver
        out = fsolve!(probsolver, x, s, p)
        step_stats!(statssolver, x, s, p, out)
        return out
    end
    alg = ComposedCommonSolveAlgorithm(f.alg, alg.stats)
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

struct FourierEvaluationProblem{F<:AbstractFourierSeries,X,K}
    f::F
    x::X
    alias::Bool
    kwargs::K
end
FourierEvaluationProblem(f::AbstractFourierSeries, x, alias=false; kws...) = FourierEvaluationProblem(f, x, alias, kws)
struct FourierEvaluationAlgorithm{E}
    exec::E
end
FourierEvaluationAlgorithm(; exec=SerialExecutor()) = FourierEvaluationAlgorithm(exec)
mutable struct FourierEvaluationSolver{F,X,A,C,K}
    f::F
    x::X
    alg::A
    cacheval::C
    kwargs::K
end
function init(prob::FourierEvaluationProblem, alg::FourierEvaluationAlgorithm; kws...)
    kwargs = (; prob.kwargs..., kws...)
    FourierSeriesEvaluators.isinplace(prob.f) && throw(ArgumentError("Cannot evaluate an inplace Fourier series. Please file a bug report"))
    cacheval = init_fourierevalcache(prob.f |> (prob.alias ? identity : deepcopy), prob.x isa BatchArray ? prob.x : Tuple(prob.x), alg.exec)
    return FourierEvaluationSolver(prob.f, prob.x, alg, cacheval, kwargs)
end
function init_fourierevalcache(f::AbstractFourierSeries, x::Tuple, exec::AbstractExecutor)
    nd = ndims(f)
    vd = Val(nd)
    xd = x[nd]
    if nd == 1
        return (FourierSeriesEvaluators.allocate(f, xd, vd),)
    else
        solver = init(FourierEliminationProblem(f, xd, vd), FourierEliminationAlgorithm())
        return (init_fourierevalcache(solve!(solver), x[1:nd-1], exec)..., solver)
    end
end
_firstpoint(x) = first(x)
function init_fourierevalcache(f::AbstractFourierSeries, x::BatchArray, exec::SerialExecutor)
    # serial execution only needs the memory for a single evaluator
    x1 = _firstpoint(x.data)
    solver = init_fourierevalcache(f, Tuple(x1), exec)
    sol = solve_fourierevalcache!(solver, f, Tuple(x1), exec)
    out = similar(x.data, typeof(sol))
    # the output array above would not work for inplace series
    return (solver, out)
end
function init_fourierevalcache(f::AbstractFourierSeries, x::BatchArray, exec::ThreadedExecutor)
    # threaded execution needs multiple solvers
    # heuristically, we assume that the workload can be distributed uniformly over the
    # outermost variable (otherwise an adaptive scheme is possible like for IAI)
    x1 = _firstpoint(x.data)
    channel = fillchannel(exec) do
        init_fourierevalcache(f, Tuple(x1), exec)
    end
    solver = first(channel)
    sol = solve_fourierevalcache!(solver, f, Tuple(x1), exec)
    out = similar(x.data, typeof(sol))
    # the output array above would not work for inplace series
    return (channel, out)
end

function solve!(solver::FourierEvaluationSolver)
    return solve_fourierevalcache!(solver.cacheval, solver.f, solver.x isa BatchArray ? solver.x : Tuple(solver.x), solver.alg.exec)
end
function solve_fourierevalcache!(cacheval, f::AbstractFourierSeries, x::Tuple, exec::AbstractExecutor)
    nd = ndims(f)
    vd = Val(nd)
    xd = x[nd]
    cache = cacheval[nd]
    if nd == 1
        return FourierSeriesEvaluators.evaluate!(cache, f, xd)
    else
        cache.f = f
        cache.x = xd
        return solve_fourierevalcache!(cacheval[1:nd-1], solve!(cache), x[1:nd-1], exec)
    end
end
function solve_fourierevalcache!((cacheval, out), f::AbstractFourierSeries, x::BatchArray, exec::AbstractExecutor)
    batchsolve_fourierevalcache!(out, cacheval, f, x.data, exec)
end
function batchsolve_fourierevalcache!(out, cacheval, f, x::ProductArray, exec::SerialExecutor)
    nd = ndims(x)
    vd = Val(nd)
    if nd == 1
        for (i,xi) in zip(eachindex(out), x.xs[1])
            out[i] = solve_fourierevalcache!(cacheval, f, Tuple(xi), exec)
        end
    else
        solver = cacheval[nd]
        for (i,ix) in zip(axes(out)[nd], axes(x)[nd])
            solver.f = f
            solver.x = x.xs[nd][ix]
            g = solve!(solver)
            batchsolve_fourierevalcache!(view(out, ntuple(_->(:), Val(nd-1))..., i), cacheval[1:nd-1], g, ProductArray(x.xs[1:nd-1]), exec)
        end
    end
    return out
end
function batchsolve_fourierevalcache!(out, channel, f::AbstractFourierSeries, x::ProductArray, exec::ThreadedExecutor)
    # partition ProductArray into exec.ntasks chunks and then launch work
    d, r = divrem(size(x)[end], exec.ntasks)
    ix = firstindex(axes(x)[end])
    io = firstindex(axes(out)[end])
    for n in 1:exec.ntasks
        chunk = ((n-1)*d+(n > r ? r : n-1)):(n*d-1+(n > r ? r : n))
        cacheval = take!(channel)
        Threads.@spawn begin
            nd = ndims(x)
            batchsolve_fourierevalcache!(view(out, ntuple(_->(:),Val(nd-1))..., chunk .+ io), cacheval, f, ProductArray((ntuple(n->x.xs[n],Val(nd-1))..., x.xs[nd][chunk .+ ix])), SerialExecutor())
            put!(channel, cacheval)
        end
    end
    return out
end

# Nested quadrature with special series evaluation on a hierarchical grid
function init_cacheval(f::AbstractFourierIntegralFunction, dom, p, alg::NestedQuad; kws...)
    x0, segs, lims, states = unroll_limits(dom)
    series = unroll_series(f.s)
    algs = alg.algs isa IntegralAlgorithm ? ntuple(i -> alg.algs, Val(ndims(dom))) : alg.algs
    spec = alg.specialize isa AbstractSpecialization ? ntuple(i -> alg.specialize, Val(ndims(dom)-1)) : alg.specialize
    exec = alg.executor isa AbstractExecutor ? ntuple(i -> alg.executor, Val(ndims(dom)-1)) : alg.executor
    tolalg = alg.tolerance_alg
    _f, fprototype = nested_innerfourierintegralfunction(f, x0, series[1], x0[1], p)


    _kws, kws_ = if tolalg isa StaticTolAlg && tolalg.method == project
        elimdom = eliminate(lims[end], 1)
        invoutermeasure = inv(elimdom === nothing ? 1 : measure(quadgk, elimdom))
        _rescale_abstol(invoutermeasure; kws...), (; kws...)
    elseif tolalg isa v04TolAlg
        _meas = 1
        _lims = lims[end]
        while ndims(_lims) > 1
            s = segments(_lims, ndims(_lims))
            _meas *= abs(s[end]-s[begin])
            _lims = fixandeliminate(_lims, (s[begin]+s[end])/2, Val(ndims(_lims)))
        end
        init_kws = _rescale_abstol(1/_meas; kws...)
        _tmp_kws = _rescale_abstol(1/real(prod(x0[begin+1:end])); kws...)
        _tmp_kws, _tmp_kws
    elseif tolalg isa v03TolAlg
        _rescale_abstol(1/real(prod(x0[begin+1:end])); kws...), _rescale_abstol(1/real(prod(x0[begin+2:end])); kws...)
    else
        _tmp_kws = _rescale_abstol(1/real(prod(x0[begin+1:end])); kws...)
        _tmp_kws, _tmp_kws
    end

    intprob = IntegralProblem(_f, segs[1], (; p, state=states[1], series=series[1]); _kws...)
    segprob = SegmentProblem(lims[1], 1)
    innerinput = (; lims=lims[1], dim=Val(1), state=states[1], p, series=series[1], kws=kws_)
    innerprob = ComposedCommonSolveProblem(innerinput, segprob, intprob) do (; lims, dim, state, p, series, kws), segsolver, intsolver
        segsolver.lims = lims
        segsolver.dim = _val_unwrap(dim)
        _segs = solve!(segsolver)
        intsolver.dom = _segs
        len = abs(_segs[end]-_segs[begin])
        intsolver.p = (; intsolver.p..., p, state, series)
        __kws = if tolalg isa StaticTolAlg
            if tolalg.method == bbox
                kws
            elseif tolalg.method == project
                _rescale_abstol(invoutermeasure; kws...)
            else
                error("$(tolalg.method) not implemented")
            end
        elseif tolalg isa AdaptiveTolAlg
            kws
        elseif tolalg isa v04TolAlg
            length(states) == 1 ? kws : init_kws
        elseif tolalg isa v03TolAlg
            _rescale_abstol(length(states) == 1 ? one(1/len) : 1/len; kws...)
        else
            error("tolalg not recognized")
        end
        intsolver.kwargs = (; intsolver.kwargs..., __kws...)
        return solve!(intsolver)
    end
    inneralg = ComposedCommonSolveAlgorithm(SegmentAlgorithm(), algs[1])
    prob, alg = nested_fourierprob(innerprob, inneralg, fprototype, p, x0, segs[2:end], lims[2:end], states[2:end], algs[2:end], spec, exec, series[2:end], tolalg; kws...)
    return init(prob, alg)
end
function nested_fourierprob(innerprob, inneralg, prototype, p, x0, segs, lims, states, algs, spec, exec, series, tolalg; kws...)
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
        return CommonSolutionStats(sol.value, sol.stats)
    end
    __f = nested_integralfunction(_f, x0[ndims(lims[1])], p)
    
    _kws, kws_ = if tolalg isa StaticTolAlg
        if tolalg.method == bbox
            _a, _b = segments(lims[end], ndims(lims[1]))
            invouterlen =(1/abs(_b-_a))
            _tmpk = _rescale_abstol(1/real(prod(x0[begin+ndims(lims[1]):end])); kws...)
            _tmpk, _tmpk
        elseif tolalg.method == project
            elimdom = eliminate(lims[end], 1:ndims(lims[1]))
            invouterlen = (elimdom === nothing ? 1 : inv(measure(quadgk, elimdom)))
            _rescale_abstol(invouterlen; kws...), (; kws...)
        else
            error("$(tolalg.method) not implemented")
        end
    elseif tolalg isa v04TolAlg
        _meas = 1
        _lims = lims[end]
        while ndims(_lims) > ndims(lims[1])
            s = segments(_lims, ndims(_lims))
            _meas *= abs(s[end]-s[begin])
            _lims = fixandeliminate(_lims, (s[begin]+s[end])/2, Val(ndims(_lims)))
        end
        init_kws = _rescale_abstol(1/_meas; kws...)
        ekws = eliminput.kws
        init_kws, init_kws
    elseif tolalg isa v03TolAlg
        _rescale_abstol(1/real(prod(x0[begin+ndims(lims[1]):end])); kws...), _rescale_abstol(1/real(prod(x0[begin+ndims(lims[1])+1:end])); kws...)
    else
        tmpkw = _rescale_abstol(1/real(prod(x0[begin+ndims(lims[1]):end])); kws...)
        tmpkw, tmpkw
    end

    innerinput = (; lims=lims[1], dim=Val(ndims(lims[1])), state=states[1], series=series[1], p, kws=kws_)
    intprob = IntegralProblem(__f, segs[1], (; innerinput..., kws=eliminput.kws); _kws...)

    segprob = SegmentProblem(lims[1], ndims(lims[1]))
    _innerprob = ComposedCommonSolveProblem(innerinput, segprob, intprob) do (; lims, dim, state, series, p, kws), segsolver, intsolver
        segsolver.lims = lims
        segsolver.dim = _val_unwrap(dim)
        _segs = solve!(segsolver)
        intsolver.dom = _segs
        len = abs(_segs[end]-_segs[begin])
        __kws, kws__ = if tolalg isa StaticTolAlg
            if tolalg.method == bbox
                kws, _rescale_abstol(invouterlen; kws...)
            elseif tolalg.method == project
                _rescale_abstol(invouterlen; kws...), (; kws...)
            else
                error("not implemented")
            end
        elseif tolalg isa AdaptiveTolAlg
            kws, _rescale_abstol(1/len; kws...)
        elseif tolalg isa v04TolAlg
            (length(states) == 1 ? kws : init_kws), _rescale_abstol(1/len; ekws...) # latter is ignored
        elseif tolalg isa v03TolAlg
            kw = _rescale_abstol(length(states) == 1 ? one(1/len) : 1/len; kws...)
            kw, kw
        else
            error("tolalg not recognized")
        end
        intsolver.p = (; intsolver.p..., lims, state, series, dim, p, kws=kws__)
        intsolver.kwargs = (; intsolver.kwargs..., __kws...)
        return solve!(intsolver)
    end
    _inneralg = ComposedCommonSolveAlgorithm(SegmentAlgorithm(), algs[1])

    nested_fourierprob(_innerprob, _inneralg, prototype, p, x0, segs[2:end], lims[2:end], states[2:end], algs[2:end], spec[2:end], exec[2:end], series[2:end], tolalg; kws...)
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
    alg = FourierEvaluationAlgorithm(SerialExecutor())

    _f = f.f
    exec = f.executor
    func = CommonSolveIntegralFunction(prob, alg, proto, DefaultSpecialize(), exec) do solver, x, (; series, p, state)
        # solver.x = x
        # solver.f = p.series
        # sol = solve!(solver)
        ## using out-of-place semantics can be faster
        sol = solve_fourierevalcache!(solver.cacheval, series, Tuple(x), SerialExecutor())
        return _f(SVector(promote(x, state...)), sol, p)
    end
    return func, proto
end
function nested_innerfourierintegralfunction(f::CommonSolveFourierIntegralFunction, x0, series, x1, p)
    proto = get_prototype(f, x0, p)
    return nested_innerfourierintegralfunction_cs(f.executor, f, x0, series, x1, p, proto), proto
end
function nested_innerfourierintegralfunction_cs(exec::SerialExecutor, f, x0, series, x1, p, proto)

    prob = FourierEvaluationProblem(series, x1)
    alg = FourierEvaluationAlgorithm(exec)
    input = (; x=x0, x1, series, p)
    fsolve! = f.solve!
    cprob = ComposedCommonSolveProblem(input, prob, f.prob) do (; x, x1, p, series), fouriersolver, fsolver
        # fouriersolver.f = series
        # fouriersolver.x = x1
        # s = solve!(fouriersolver)
        ## for performance, eliding the setfield! call is helpful for small fourier series
        s = solve_fourierevalcache!(fouriersolver.cacheval, series, Tuple(x1), SerialExecutor())
        return fsolve!(fsolver, x, s, p)
    end

    calg = ComposedCommonSolveAlgorithm(alg, f.alg)
    return CommonSolveIntegralFunction(cprob, calg, proto, f.specialize, f.executor; f.kwargs...) do solver, x, (; p, state, series)
        # solver.input = (; solver.input..., x1=x, x=SVector(promote(x, state...)), p, series)
        # return solve!(solver)
        ## out-of-place semantics may be faster
        return solver.solve!((; solver.input..., x1=x, x=SVector(promote(x, state...)), p, series), solver.solvers...)
    end
end
function nested_innerfourierintegralfunction_cs(exec::ThreadedExecutor, f, x0, series, x1, p, proto)
    _f = nested_innerfourierintegralfunction_cs(SerialExecutor(), f, x0, series, x1, p, proto)
    return nested_integralfunction_cs(exec, _f, x1, p)
end
