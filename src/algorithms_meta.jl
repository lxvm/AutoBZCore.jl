"""
    NestedQuad(alg::IntegralAlgorithm)
    NestedQuad(algs::IntegralAlgorithm...)

Nested integration by repeating one quadrature algorithm or composing a list of algorithms.
The domain of integration must be an `AbstractIteratedLimits` from the
IteratedIntegration.jl package. Analogous to `nested_quad` from IteratedIntegration.jl.
The integrand should expect `SVector` inputs. Do not use this for very high-dimensional
integrals, since the compilation time scales very poorly with respect to dimensionality.
"""
struct NestedQuad{T,S,E} <: IntegralAlgorithm
    algs::T
    specialize::S
    executor::E
    NestedQuad(alg::IntegralAlgorithm, specialize::AbstractSpecialization=NoSpecialize(), executor::AbstractExecutor=SerialExecutor()) = new{typeof(alg),typeof(specialize),typeof(executor)}(alg, specialize, executor)
    NestedQuad(algs::Tuple{Vararg{IntegralAlgorithm}}, specialize::Tuple{Vararg{AbstractSpecialization}}=ntuple(_->NoSpecialize(), length(algs)), executor::Tuple{Vararg{AbstractExecutor}}=ntuple(_->SerialExecutor(), length(algs))) = new{typeof(algs),typeof(specialize),typeof(executor)}(algs, specialize, executor)
end
NestedQuad(algs::IntegralAlgorithm...) = NestedQuad(algs)
# TODO add a parallelization option for use when it is safe to do so

struct NestedIntegralProblem{F,P,L,S,K}
    f::F
    p::P
    lims::L
    state::S
    kwargs::K
end
mutable struct NestedIntegralSolver{F,P,L,S,A,C,K}
    f::F
    p::P
    lims::L
    state::S
    alg::A
    cacheval::C
    kwargs::K
end
struct IteratedIntegrationJL end
function init(prob::NestedIntegralProblem, alg::IteratedIntegrationJL; kws...)
    kwargs = (; prob.kwargs..., kws...)
    cacheval = if prob.f isa CommonSolveIntegralFunction
        init(prob.f.prob, prob.f.alg; prob.f.kwargs...)
    else
        nothing
    end
    return NestedIntegralSolver(prob.f, prob.p, prob.lims, prob.state, alg, cacheval, kwargs)
end
function solve!(solver::NestedIntegralSolver)
    # solver.cacheval.p = solver.p
    next = limit_iterate(solver.lims, solver.state...)
    if next isa SVector
        if solver.f isa IntegralFunction
            return solver.f.f(next, solver.p)
        end
    else
        segs, lims, state = next
        solver.cacheval.dom = segs
        solver.cacheval.p = (; solver.cacheval.p..., p=solver.p, lims_state=(lims, state))
        return solve!(solver.cacheval)
    end

end

function _update!(cache, x, (; p, lims_state))
    segs, lims, state = limit_iterate(lims_state..., x)
    len = segs[end] - segs[begin]
    kws = cache.kwargs
    cache.p = p
    cache.cacheval.dom = segs
    cache.cacheval.kwargs = haskey(kws, :abstol) ? merge(kws, (abstol = kws.abstol / len,)) : kws
    cache.cacheval.p = (; cache.cacheval.p..., lims_state = (lims, state))
    return
end
_postsolve(sol, x, p) = sol.value
#=
function init_cacheval(f, nextdom, p, alg::NestedQuad; kws...)
    x0, (segs, lims, state) = if nextdom isa AbstractIteratedLimits
        interior_point(nextdom), limit_iterate(nextdom)
    else
        nextdom
    end
    algs = alg.algs isa IntegralAlgorithm ? ntuple(i -> alg.algs, Val(ndims(lims))) : alg.algs
    spec = alg.specialize isa AbstractSpecialization ? ntuple(i -> alg.specialize, Val(ndims(lims))) : alg.specialize
    exec = alg.executor isa AbstractExecutor ? ntuple(i -> alg.executor isa SharedThreadedExecutor ? SharedThreadedExecutor(alg.executor.ntasks) : alg.executor, Val(ndims(lims))) : alg.executor
    if ndims(lims) == 1
        func, ws = inner_integralfunction(f, x0, p)
    else
        integrand, ws, update!, postsolve = outer_integralfunction(f, x0, p)
        proto = get_prototype(integrand, x0, p)
        a, b, = segs
        x = (a + b) / 2
        next = (x0[begin:end-1], limit_iterate(lims, state, x))
        kws = NamedTuple(kws)
        len = segs[end] - segs[begin]
        kwargs = haskey(kws, :abstol) ? merge(kws, (abstol = kws.abstol / len,)) : kws
        subprob = IntegralProblem(integrand, next, p; kwargs...)
        func = CommonSolveIntegralFunction(subprob, NestedQuad(algs[1:ndims(lims)-1], spec[1:ndims(lims)-1], exec[1:ndims(lims)-1]), update!, postsolve, proto*x^(ndims(lims)-1), spec[ndims(lims)], exec[ndims(lims)])
    end
    prob = IntegralProblem(func, segs, (; p, lims_state = (lims, state), ws); kws...)
    return init(prob, algs[ndims(lims)])
    # the order of updates is somewhat tricky. I think some could be simplified if instead
    # we use an IntegralProblem modified to contain lims_state, instead of passing the
    # parameter as well
end
=#
unroll_limits(dom) = unroll_limits(limit_iterate(dom)...)
function unroll_limits(segs, lims, state)
    a, b, = segs
    x = (a + b)/2
    next = limit_iterate(lims, state, x)
    next isa SVector && return next, (segs,), (lims,), (state,)
    x0, _segs, _lims, _state = unroll_limits(next...)
    return x0, (_segs..., segs), (_lims..., lims), (_state..., state)
end
function __update!(solver, x, p)
    solver.p = p
    segs, lims, state = limit_iterate(lims_state..., x)
    len = segs[end] - segs[begin]
    kws = cache.kwargs
    cache.p = p
    cache.cacheval.dom = segs
    cache.cacheval.kwargs = haskey(kws, :abstol) ? merge(kws, (abstol=kws.abstol/len,)) : kws
    cache.cacheval.p = (; cache.cacheval.p..., lims_state=(lims, state))
    return
end

struct SegmentProblem{T,A,L,S,D,P,U,K}
    prob::T
    alg::A
    lims::L
    state::S
    dim::D
    p::P
    update!::U
    kwargs::K
end

struct SegmentAlgorithm end

mutable struct SegmentSolver{C,L,S,D,P,U}
    cacheval::C
    lims::L
    state::S
    dim::D
    p::P
    update!::U
end

function init(prob::SegmentProblem, ::SegmentAlgorithm; kws...)
    return SegmentSolver(init(prob.prob, prob.alg; prob.kwargs..., kws...), prob.lims, prob.state, prob.dim, prob.p, prob.update!)
end

function solve!(solver::SegmentSolver)
    solver.update!(solver.cacheval, solver.lims, IteratedIntegration.segments(solver.lims, solver.dim), solver.state, solver.p)
    return solve!(solver.cacheval)
end

struct EliminationProblem{T,A,P,X,D,S,L,U,K}
    prob::T
    alg::A
    p::P
    x::X
    dim::D
    state::S
    lims::L
    update!::U
    kwargs::K
end

struct EliminationAlgorithm end

mutable struct EliminationSolver{C,P,X,D,S,L,U}
    cacheval::C
    p::P
    x::X
    dim::D
    state::S
    lims::L
    update!::U
end

function init(prob::EliminationProblem, ::EliminationAlgorithm; kws...)
    return EliminationSolver(init(prob.prob, prob.alg; prob.kwargs..., kws...), prob.p, prob.x, prob.dim, prob.state, prob.lims, prob.update!)
end

function solve!(solver::EliminationSolver)
    lims = IteratedIntegration.fixandeliminate(solver.lims, solver.x, solver.dim)
    solver.update!(solver.cacheval, lims, (solver.x, solver.state...), solver.dim, solver.p)
    return solve!(solver.cacheval)
end

_val_unwrap(::Val{X}) where {X} = X
function nested_prob(segprob, prototype, p, x0, segs, lims, states, algs, spec, exec; kws...)
    length(states) == 0 && return segprob
    segup! = (solver, lims, state, dim, (; p, kws, len)) -> begin
        solver.lims = lims
        solver.state = state
        solver.dim = _val_unwrap(dim)-1
        solver.p = (; solver.p..., p, kws=_rescale_abstol(1/len; kws...))
        return
    end
    _kws = _rescale_abstol(1/real(prod(x0[begin+ndims(lims[1]):end])); kws...)
    elimprob = EliminationProblem(segprob, SegmentAlgorithm(), (; p, kws=_kws, len=abs(segs[1][end]-segs[1][begin])), x0[ndims(lims[1])], Val{ndims(lims[1])}(), states[1], lims[1], segup!, (;))
    elimup! = (solver, x, (; p, lims, state, len, kws)) -> begin
        solver.x = x
        solver.lims = lims
        solver.state = state
        solver.p = (; solver.p..., p, kws, len)
        return
    end
    elimpost = (sol, x, p) -> sol.value
    _f = CommonSolveIntegralFunction(elimprob, EliminationAlgorithm(), elimup!, elimpost, prototype*real(prod(x0[begin:ndims(lims[1])-1])), spec[1], exec[1])
    __f = nested_integralfunction(_f, x0[ndims(lims[1])], p)
    intprob = IntegralProblem(__f, segs[1], (; p, lims=lims[1], state=states[1], len=abs(segs[1][end]-segs[1][begin]), kws=_kws); _kws...)
    intup! = (solver, lims, segs, state, (; p, kws)) -> begin
        solver.dom = segs
        solver.kwargs = (; solver.kwargs..., kws...)
        solver.p = (; solver.p..., p, lims, state, len=abs(segs[end]-segs[begin]), kws)
        return
    end
    _segprob = SegmentProblem(intprob, algs[1], lims[1], states[1], ndims(lims[1]), (; p, kws=(; intprob.kwargs...)), intup!, kws)
    return nested_prob(_segprob, prototype, p, x0, segs[2:end], lims[2:end], states[2:end], algs[2:end], spec[2:end], exec[2:end]; kws...)
end
function nested_integralfunction(f::CommonSolveIntegralFunction, x0, p)
    return nested_integralfunction_cs(f.executor, f, x0, p)
end
function nested_integralfunction_cs(::SerialExecutor, f, x0, p)
    return f
end
function nested_integralfunction_cs(exec::ThreadedExecutor, f, x0, p)
    channel, integrand, prototype = init_commonsolvefunction_(exec, f, x0, p; kws...)
    proto = [prototype]
    func = InplaceBatchIntegralFunction(proto; max_batch=exec.ntasks) do y, x, (; p, state)
        do_threaded_solve!(integrand, channel, f, y, x, p)
    end
    return func
end
function nested_innerintegralfunction(f::IntegralFunction, x0, p)
    proto = get_prototype(f, x0, p)
    func = IntegralFunction(proto) do x, (; p, state)
        f.f(SVector(promote(x, state...)), p)
    end
    return func
end
function nested_innerintegralfunction(f::CommonSolveIntegralFunction, x0, p)
    return nested_innerintegralfunction_cs(f.executor, f, x0, p)
end
function nested_innerintegralfunction_cs(::SerialExecutor, f, x0, p)
    up! = (solver, x, (; p, state)) -> f.update!(solver, SVector(promote(x, state...)), p)
    post = (sol, x, (; p, state)) -> f.postsolve(sol, SVector(promote(x, state...)), p)
    return CommonSolveIntegralFunction(f.prob, f.alg, up!, post, f.prototype, f.specialize, f.executor; f.kwargs...)
end
function nested_innerintegralfunction_cs(exec::ThreadedExecutor, f, x0, p)
    _f = nested_innerintegralfunction_cs(SerialExecutor(), f, x0, p)
    return nested_integralfunction_cs(exec, _f, x0, p)
end
_rescale_abstol(s; kws...) = haskey(kws, :abstol) ? (; kws..., abstol=kws[:abstol]*s) : (; kws...)
function init_cacheval(f, dom, p, alg::NestedQuad; kws...)
    x0, segs, lims, states = unroll_limits(dom)
    algs = alg.algs isa IntegralAlgorithm ? ntuple(i -> alg.algs, Val(ndims(dom))) : alg.algs
    spec = alg.specialize isa AbstractSpecialization ? ntuple(i -> alg.specialize, Val(ndims(dom))) : alg.specialize
    exec = alg.executor isa AbstractExecutor ? ntuple(i -> alg.executor isa SharedThreadedExecutor ? SharedThreadedExecutor(alg.executor.ntasks) : alg.executor, Val(ndims(dom))) : alg.executor

    _f = nested_innerintegralfunction(f, x0, p)
    innerprob = IntegralProblem(_f, segs[1], (; p, state=states[1]); _rescale_abstol(1/real(prod(x0[begin+1:end])); kws...)...)
    innerup! = (solver, lims, segs, state, (; p, kws)) -> begin
        solver.dom = segs
        solver.p = (; p, state)
        solver.kwargs = (; solver.kwargs..., kws...)
        return
    end
    segprob = SegmentProblem(innerprob, algs[1], lims[1], states[1], 1, (; p, kws=(; innerprob.kwargs...)), innerup!, (;))

    prob = nested_prob(segprob, _f.prototype, p, x0, segs[2:end], lims[2:end], states[2:end], algs[2:end], spec[2:end], exec[2:end]; kws...)
    return init(prob, SegmentAlgorithm(); kws...)
end

function do_integral(f, dom, p, alg::NestedQuad, cacheval; kws...)
    cacheval.p = (; cacheval.p..., p, kws=(; kws...))
    cacheval.lims = dom
    return solve!(cacheval)
end
function inner_integralfunction(f::IntegralFunction, x0, p)
    proto = get_prototype(f, x0, p)
    func = IntegralFunction(proto) do x, (; p, state)
        f.f(SVector(promote(x, state...)), p)
    end
    ws = nothing
    return func, ws
end
function outer_integralfunction(f::IntegralFunction, x0, p)
    proto = get_prototype(f, x0, p)
    func = IntegralFunction(f.f, proto)
    ws = nothing
    return func, ws, _update!, _postsolve
end
function inner_integralfunction(f::CommonSolveIntegralFunction, x0, p)
    proto = get_prototype(f, x0, p)
    up = (cache, x, (; p, lims_state)) -> f.update!(cache, limit_iterate(lims_state..., x), p)
    func = CommonSolveIntegralFunction(f.prob, f.alg, f.kwargs, up, f.postsolve, proto, f.specialize, f.executor)
    ws = nothing
    return func, ws
end
function outer_integralfunction(f::CommonSolveIntegralFunction, x0, p)
    proto = get_prototype(f, x0, p)
    func = CommonSolveIntegralFunction(f.prob, f.alg, f.kwargs, f.update!, f.postsolve, proto, f.specialize, f.executor)
    ws = nothing
    return func, ws, _update!, _postsolve
end
#=
"""
    AbsoluteEstimate(est_alg, abs_alg; kws...)

Most algorithms are efficient when using absolute error tolerances, but how do you know the
size of the integral? One option is to estimate it using second algorithm.

A multi-algorithm to estimate an integral using an `est_alg` to generate a rough estimate of
the integral that is combined with a user's relative tolerance to re-calculate the integral
to higher accuracy using the `abs_alg`. The keywords passed to the algorithm may include
`reltol`, `abstol` and `maxiters` and are given to the `est_alg` solver. They should limit
the amount of work of `est_alg` so as to only generate an order-of-magnitude estimate of the
integral. The tolerances passed to `abs_alg` are `abstol=max(abstol,reltol*norm(I))` and
`reltol=0`.
"""
struct AbsoluteEstimate{E<:IntegralAlgorithm,A<:IntegralAlgorithm,F,K<:NamedTuple} <: IntegralAlgorithm
    est_alg::E
    abs_alg::A
    norm::F
    kws::K
end
function AbsoluteEstimate(est_alg, abs_alg; norm=norm, kwargs...)
    kws = NamedTuple(kwargs)
    checkkwargs(kws)
    return AbsoluteEstimate(est_alg, abs_alg, norm, kws)
end

function init_cacheval(f, dom, p, alg::AbsoluteEstimate)
    return (est=init_cacheval(f, dom, p, alg.est_alg),
            abs=init_cacheval(f, dom, p, alg.abs_alg))
end

function do_solve(f, dom, p, alg::AbsoluteEstimate, cacheval;
                    abstol=nothing, reltol=nothing, maxiters=typemax(Int))
    sol = do_solve(f, dom, p, alg.est_alg, cacheval.est; alg.kws...)
    val = alg.norm(sol.u) # has same units as sol
    rtol = reltol === nothing ? sqrt(eps(one(val))) : reltol # use the precision of the solution to set the default relative tolerance
    atol = max(abstol === nothing ? zero(val) : abstol, rtol*val)
    return do_solve(f, dom, p, alg.abs_alg, cacheval.abs;
                    abstol=atol, reltol=zero(rtol), maxiters=maxiters)
end
=#


"""
    EvalCounter(::IntegralAlgorithm)

An algorithm which counts the evaluations used by another algorithm.
The count is stored in the `sol.stats.numevals` field.
"""
struct EvalCounter{T <: IntegralAlgorithm} <: IntegralAlgorithm
    alg::T
end

struct CounterFunction{F}
    counter::Base.RefValue{Int64}
    f::F
end
(f::CounterFunction)(args...; kws...) = (f.counter[] += 1; f.f(args...; kws...))
struct BatchCounterFunction{F}
    counter::Base.RefValue{Int64}
    f::F
end
(f::BatchCounterFunction)(y, x, p) = (f.counter[] += size(x)[end]; f.f(y, x, p))


struct CounterProblem{F,A<:Tuple,K<:NamedTuple}
    f::F
    args::A
    kwargs::K
    CounterProblem(f, args...; kws...) = new{typeof(f),typeof(args),typeof(NamedTuple(kws))}(f, args, NamedTuple(kws))
end

mutable struct CounterSolver{F,A,K,G}
    f::F
    args::A
    kwargs::K
    alg::G
    counter::Int
end
abstract type CounterAlgorithm end
# this could
struct SingleCount <: CounterAlgorithm end
struct RemoteCount <: CounterAlgorithm
    ch::Channel{CounterSolver}
end
function init(prob::CounterProblem, alg::CounterAlgorithm; kws...)
    solver = CounterSolver(prob.f, prob.args, (; prob.kwargs..., kws...), alg, 0)
    if alg isa RemoteCount
        put!(alg.ch, solver)
    end
    return solver
end
@inline function solve!(solver::CounterSolver)
    solver.counter += 1
    return solver.f(solver.args...; solver.kwargs...)
end
function insert_counter(f::IntegralFunction, x, p, channel)
    prob = CounterProblem(f.f, x, p)
    alg = RemoteCount(channel)
    up = (solver, x, p) -> solver.args = (x, p)
    post = (sol, x, p) -> sol
    CommonSolveIntegralFunction(prob, alg, up, post, f.prototype)
    # IntegralFunction(CounterFunction(numevals, f.f), f.prototype)
end
# insert_counter(f::IntegralFunction, numevals) = IntegralFunction(CounterFunction(numevals, f.f), f.prototype)
insert_counter(f::InplaceIntegralFunction, numevals) = InplaceIntegralFunction(CounterFunction(numevals, f.f!), f.prototype)
insert_counter(f::InplaceBatchIntegralFunction, numevals) = InplaceBatchIntegralFunction(BatchCounterFunction(numevals, f.f!), f.prototype; max_batch=f.max_batch)
function insert_counter(f::CommonSolveIntegralFunction, numevals)
    f.executor isa SerialExecutor || throw(ArgumentError("Can only count serial integrands"))
    CommonSolveIntegralFunction(f.prob, f.alg, CounterFunction(numevals, f.update!), f.postsolve, f.prototype, f.specialize; f.kwargs...)
end
function init_cacheval(f, dom, p, alg::EvalCounter; kws...)
    channel = Channel{CounterSolver}(Inf)
    g = insert_counter(f, get_prototype(dom), p, channel)
    return g, channel, init_cacheval(g, dom, p, alg.alg; kws...)
end
function do_integral(f, dom, p, alg::EvalCounter, (g, channel, cacheval); kws...)
    for c in channel.data
        c.counter = 0
    end
    sol = do_integral(g, dom, p, alg.alg, cacheval; kws...)
    numevals = 0
    for c in channel.data
        numevals += c.counter
    end
    return IntegralSolution(sol.value, sol.retcode, (; sol.stats..., numevals))
end
# elseif InplaceIntegrand
#     ni::Int = 0
#     gi = (y, x, p) -> (ni += 1; f.f!(y, x, p))
#     soli = do_solve(InplaceIntegrand(gi, f.I), dom, p, alg.alg, cacheval; kws...)
#     return IntegralSolution(soli.u, soli.resid, soli.retcode, ni)
# elseif f isa BatchIntegrand
#     nb::Int = 0
#     gb = (y, x, p) -> (nb += length(x); f.f!(y, x, p))
#     solb = do_solve(BatchIntegrand(gb, f.y, f.x, max_batch=f.max_batch), dom, p, alg.alg, cacheval; kws...)
#     return IntegralSolution(solb.u, solb.resid, solb.retcode, nb)
# else
#     n::Int = 0
#     g = (x, p) -> (n += 1; f(x, p)) # we need let to prevent Core.Box around the captured variable
#     sol = do_solve(g, dom, p, alg.alg, cacheval; kws...)
#     return IntegralSolution(sol.u, sol.resid, sol.retcode, n)
