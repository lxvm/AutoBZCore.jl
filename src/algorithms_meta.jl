"""
    AbstolAlgorithm

Supertype for choice of absolute tolerances of inner integrals in [`NestedQuad`](@ref).
"""
abstract type AbstolAlgorithm end

"""
    StaticTolAlgMethod

Enum for choosing the [`StaticTolAlg`](@ref). Two options:
- `bbox`: this can set tighter tolerances than the default because it calculates the volume of a bounding box of the domain.
- `project`: this is the default choice and computes the exact volume of the integration domain, slower than `bbox`.
"""
@enum StaticTolAlgMethod begin
    bbox
    project
end

"""
    StaticTolAlg(; method::StaticTolAlgMethod=project)

Pick the `abstol` for an inner integral by dividing the problem `abstol` by the volume of the outer variables of integration. 
The `method` keyword determines how accurately this volume is calculated. See [`StaticTolAlgMethod`](@ref).
These algorithms do not adapt to changing the domain of integration, so it is recommended to create a new solver whenever the domain needs to be changed.
"""
struct StaticTolAlg <: AbstolAlgorithm
    method::StaticTolAlgMethod
end
StaticTolAlg(; method=project) = StaticTolAlg(method)

"""
    AdaptiveTolAlg()

Choose the `abstol` for the next inner integral by dividing the tolerance of the previous integral by the length of its integral segment.
Fixes the bugs in [`v03TolAlg`](@ref) and [`v04TolAlg`](@ref).
"""
struct AdaptiveTolAlg <: AbstolAlgorithm end

"""
    v04TolAlg()

Uses the same choice of tolerances as AutoBZCore v0.4.
It is not recommended to use it as it has a bug, so it is mainly for reference.
"""
struct v04TolAlg <: AbstolAlgorithm end

"""
    v03TolAlg()

Uses the same choice of tolerances as AutoBZCore v0.3.
It is not recommended to use it as it has a bug, so it is mainly for reference.
"""
struct v03TolAlg <: AbstolAlgorithm end

"""
    NestedQuad(alg::IntegralAlgorithm=QuadGKJL(), specialize=NoSpecialize(), executor=SerialExecutor(); tolalg=StaticTolAlg(project))
    NestedQuad(algs::Tuple{Vararg{IntegralAlgorithm,N}}, specializations=ntuple(_->NoSpecialize(), N-1), executors=ntuple(_->SerialExecutor(),N-1); tolalg=StaticTolAlg(project)) where {N}
    NestedQuad(algs::IntegralAlgorithm...; tolalg=StaticTolAlg(project))

Nested integration done variable by variable using one quadrature algorithm for all variables or a tuple of algorithms for each variable.
In `algs`, the first algorithm corresponds to the "hot" innermost integration variable, and the last corresponds to the outer integration variable.
The domain of integration must be an `AbstractIteratedLimits` from the IteratedIntegration.jl package.
The integrand should expect `SVector` inputs. Do not use this for very high-dimensional
integrals, since the compilation time scales very poorly with respect to dimensionality.

The `specialize` arguments allow control over how to trade compilation time for run time.
If specifying a tuple of these, it should be one element shorter than `algs` since specialization of the inner integration variable is controlled by the integral function.

The `executor` arguments allow control over running the integrand in serial or parallel.
When using a `ThreadedExecutor`, the `ntasks` workers are shared in the same pool in the corresponding dimension.
This means that the `ntasks` of the first/inner variable should be greater than or equal to the second, and so on, to have enough workers for each task.
If specifying a tuple of these, it should be one element shorter than `algs` since specialization of the inner integration variable is controlled by the integral function.

The `tolalg` keyword controls how `abstol` is set for inner integrals based on the value requested for the integral problem.
The default choice of [`StaticTolAlg`](@ref) chooses a simple and robust scaling of tolerances.
"""
struct NestedQuad{T,S,E,A} <: IntegralAlgorithm
    algs::T
    specialize::S
    executor::E
    tolerance_alg::A
    NestedQuad(alg::IntegralAlgorithm=QuadGKJL(), specialize::AbstractSpecialization=NoSpecialize(), executor::AbstractExecutor=SerialExecutor(); tolalg::AbstolAlgorithm=StaticTolAlg()) = new{typeof(alg),typeof(specialize),typeof(executor),typeof(tolalg)}(alg, specialize, executor, tolalg)
    NestedQuad(algs::Tuple{IntegralAlgorithm,Vararg{IntegralAlgorithm,N}}, specialize::Tuple{Vararg{AbstractSpecialization,N}}=ntuple(_->NoSpecialize(), N), executor::Tuple{Vararg{AbstractExecutor}}=ntuple(_->SerialExecutor(), N); tolalg::AbstolAlgorithm=StaticTolAlg()) where {N} = new{typeof(algs),typeof(specialize),typeof(executor),typeof(tolalg)}(algs, specialize, executor, tolalg)
end
NestedQuad(algs::IntegralAlgorithm...) = NestedQuad(algs)

unroll_limits(dom::AbstractIteratedLimits) = unroll_limits(segments(dom, ndims(dom)), dom, ())
unroll_limits(dom) = unroll_limits(load_limits(dom))
function unroll_limits(segs, lims, state)
    a, b, = segs
    x = (a + b)/2
    # next = limit_iterate(lims, state, x)
    if ndims(lims) == 1
        return SVector(promote(x, state...)), (segs,), (lims,), (state,)
    else
        lx = fixandeliminate(lims, x, Val(ndims(lims)))
        x0, _segs, _lims, _state = unroll_limits(segments(lx, ndims(lx)), lx, (x, state...))
        return x0, (_segs..., segs), (_lims..., lims), (_state..., state)
    end
end

struct SegmentProblem{L,D,K}
    lims::L
    dim::D
    kwargs::K
end
SegmentProblem(lims, dim; kws...) = SegmentProblem(lims, dim, kws)

struct SegmentAlgorithm end

mutable struct SegmentSolver{C,L,D,K}
    cacheval::C
    lims::L
    dim::D
    kwargs::K
end

function init(prob::SegmentProblem, ::SegmentAlgorithm; kws...)
    kwargs = (; prob.kwargs..., kws...)
    return SegmentSolver(nothing, prob.lims, prob.dim, kwargs)
end

function solve!(solver::SegmentSolver)
    return segments(solver.lims, solver.dim)
end

struct EliminationProblem{L,X,D,K}
    lims::L
    x::X
    dim::D
    kwargs::K
end
EliminationProblem(lims, x, dim; kws...) = EliminationProblem(lims, x, dim, kws)

struct EliminationAlgorithm end

mutable struct EliminationSolver{L,X,D,A,C,K}
    lims::L
    x::X
    dim::D
    alg::A
    cacheval::C
    kwargs::K
end

function init(prob::EliminationProblem, alg::EliminationAlgorithm; kws...)
    cacheval = nothing
    kwargs = (; prob.kwargs..., kws...)
    return EliminationSolver(prob.lims, prob.x, prob.dim, alg, cacheval, kwargs)
end

function solve!(solver::EliminationSolver)
    return fixandeliminate(solver.lims, solver.x, solver.dim)
end

_val_unwrap(::Val{X}) where {X} = X
function nested_prob(innerprob, inneralg, prototype, p, x0, segs, lims, states, algs, spec, exec, tolalg; kws...)
    length(states) == 0 && return innerprob, inneralg
    elimprob = EliminationProblem(lims[1], x0[ndims(lims[1])], Val(ndims(lims[1])))
    eliminput = (; x=x0[ndims(lims[1])], lims=lims[1], state=states[1], dim=Val(ndims(lims[1])), p, kws=innerprob.input.kws)
    _prob = ComposedCommonSolveProblem(eliminput, elimprob, innerprob) do (; x, lims, state, dim, p, kws), elimsolver, innersolver
        elimsolver.x = x
        elimsolver.lims = lims
        elimsolver.dim = dim
        _state = (x, state...)
        lims_ = solve!(elimsolver)
        innersolver.input = (; innersolver.input..., lims=lims_, state=_state, dim=Val(_val_unwrap(dim)-1), p, kws)
        return solve!(innersolver)
    end
    _alg = ComposedCommonSolveAlgorithm(EliminationAlgorithm(), inneralg)
    _f = CommonSolveIntegralFunction(_prob, _alg, prototype*real(prod(x0[begin:ndims(lims[1])-1])), spec[1], exec[1]) do solver, x, p
        # solver.input = (; solver.input..., p..., x)
        # sol = solve!(solver)
        ## out-of-place semantics may be faster
        sol = solver.solve!((; solver.input..., p..., x), solver.solvers...)
        return CommonSolutionStats(sol.value, sol.stats)
    end
    __f = nested_integralfunction(_f, x0[ndims(lims[1])], p)

    _kws, kws_, tolalg_params = if tolalg isa StaticTolAlg
        if tolalg.method == bbox
            _a, _b = segments(lims[end], ndims(lims[1]))
            invouterlen =(1/abs(_b-_a))
            _tmpk = _rescale_abstol(1/real(prod(x0[begin+ndims(lims[1]):end])); kws...)
            _tmpk, _tmpk, (; invouterlen)
        elseif tolalg.method == project
            elimdom = eliminate(lims[end], 1:ndims(lims[1]))
            invouterlen = (elimdom === nothing ? 1 : inv(measure(quadgk, elimdom)))
            _rescale_abstol(invouterlen; kws...), (; kws...), (; invouterlen)
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
        init_kws, init_kws, (; init_kws, ekws)
    elseif tolalg isa v03TolAlg
        _rescale_abstol(1/real(prod(x0[begin+ndims(lims[1]):end])); kws...), _rescale_abstol(1/real(prod(x0[begin+ndims(lims[1])+1:end])); kws...), nothing
    else
        tmpkw = _rescale_abstol(1/real(prod(x0[begin+ndims(lims[1]):end])); kws...)
        tmpkw, tmpkw, nothing
    end

    innerinput = (; lims=lims[1], dim=Val(ndims(lims[1])), state=states[1], p, kws=kws_)
    intprob = IntegralProblem(__f, segs[1], (; innerinput..., kws=eliminput.kws); _kws...)

    segprob = SegmentProblem(lims[1], ndims(lims[1]))
    _innerprob = ComposedCommonSolveProblem(innerinput, segprob, intprob) do (; lims, dim, state, p, kws), segsolver, intsolver
        segsolver.lims = lims
        segsolver.dim = _val_unwrap(dim)
        _segs = solve!(segsolver)
        intsolver.dom = _segs
        len = abs(_segs[end]-_segs[begin])
        __kws, kws__ = if tolalg isa StaticTolAlg
            if tolalg.method == bbox
                kws, _rescale_abstol(tolalg_params.invouterlen; kws...)
            elseif tolalg.method == project
                _rescale_abstol(tolalg_params.invouterlen; kws...), (; kws...)
            else
                error("not implemented")
            end
        elseif tolalg isa AdaptiveTolAlg
            kws, _rescale_abstol(1/len; kws...)
        elseif tolalg isa v04TolAlg
            (length(states) == 1 ? kws : tolalg_params.init_kws), _rescale_abstol(1/len; tolalg_params.ekws...) # latter is ignored
        elseif tolalg isa v03TolAlg
            kw = _rescale_abstol(length(states) == 1 ? one(1/len) : 1/len; kws...)
            kw, kw
        else
            error("tolalg not recognized")
        end
        intsolver.p = (; intsolver.p..., lims, state, dim, p, kws=kws__)
        intsolver.kwargs = (; intsolver.kwargs..., __kws...)
        return solve!(intsolver)
    end
    _inneralg = ComposedCommonSolveAlgorithm(SegmentAlgorithm(), algs[1])

    nested_prob(_innerprob, _inneralg, prototype, p, x0, segs[2:end], lims[2:end], states[2:end], algs[2:end], spec[2:end], exec[2:end], tolalg; kws...)
end
function nested_integralfunction(f::CommonSolveIntegralFunction, x0, p)
    return nested_integralfunction_cs(f.executor, f, x0, p)
end
function nested_integralfunction_cs(::SerialExecutor, f, x0, p)
    return f
end
function nested_integralfunction_cs(exec::ThreadedExecutor, f, x0, p)
    channel, integrand, prototype = init_commonsolvefunction_(exec, f, x0, p)
    proto = [prototype]
    minchunksize = exec.min_chunksize
    ntasks = exec.ntasks
    func = InplaceBatchIntegralFunction(proto; max_batch=exec.max_batch) do y, x, p
        do_threaded_solve!(integrand, channel, f, y, x, p, ntasks, minchunksize)
    end
    return func
end
function nested_innerintegralfunction(f::IntegralFunction, x0, p)
    proto = get_prototype(f, x0, p)
    func = IntegralFunction(proto, f.executor) do x, (; p, state)
        f.f(SVector(promote(x, state...)), p)
    end
    return func, proto
end
function nested_innerintegralfunction(f::CommonSolveIntegralFunction, x0, p)
    proto = get_prototype(f, x0, p)
    return nested_innerintegralfunction_cs(f.executor, f, x0, p, proto), proto
end
function nested_innerintegralfunction_cs(::SerialExecutor, f, x0, p, prototype)
    fsolve! = f.solve!
    _solve! = (solver, x, (; p, state)) -> fsolve!(solver, SVector(promote(x, state...)), p)
    return CommonSolveIntegralFunction(_solve!, f.prob, f.alg, prototype, f.specialize, f.executor; f.kwargs...)
end
function nested_innerintegralfunction_cs(exec::ThreadedExecutor, f, x0, p, prototype)
    _f = nested_innerintegralfunction_cs(SerialExecutor(), f, x0, p, prototype)
    return nested_integralfunction_cs(exec, _f, x0, p)
end
_rescale_abstol(s; kws...) = haskey(kws, :abstol) ? (; kws..., abstol=kws[:abstol]*s) : (; kws...)
function init_cacheval(f, dom, p, alg::NestedQuad; kws...)
    x0, segs, lims, states = unroll_limits(dom)
    algs = alg.algs isa IntegralAlgorithm ? ntuple(i -> alg.algs, Val(ndims(dom))) : alg.algs
    spec = alg.specialize isa AbstractSpecialization ? ntuple(i -> alg.specialize, Val(ndims(dom)-1)) : alg.specialize
    exec = alg.executor isa AbstractExecutor ? ntuple(i -> alg.executor, Val(ndims(dom)-1)) : alg.executor
    tolalg = alg.tolerance_alg

    _f, fprototype = nested_innerintegralfunction(f, x0, p)

    # WARNING: Boxing of variables defined conditionally may lead to incorrect results, so always pack into tolalg_params
    # TODO: rewrite this big if with a function barrier and dispatch on tolalg
    _kws, kws_, tolalg_params = if tolalg isa StaticTolAlg && tolalg.method == project
        elimdom = eliminate(lims[end], 1)
        invoutermeasure = inv(elimdom === nothing ? 1 : measure(quadgk, elimdom))
        _rescale_abstol(invoutermeasure; kws...), (; kws...), (; invoutermeasure)
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
        _tmp_kws, _tmp_kws, (; init_kws)
    elseif tolalg isa v03TolAlg
        _rescale_abstol(1/real(prod(x0[begin+1:end])); kws...), _rescale_abstol(1/real(prod(x0[begin+2:end])); kws...), nothing
    else
        _tmp_kws = _rescale_abstol(1/real(prod(x0[begin+1:end])); kws...)
        _tmp_kws, _tmp_kws, nothing
    end

    intprob = IntegralProblem(_f, segs[1], (; p, state=states[1]); _kws...)
    segprob = SegmentProblem(lims[1], 1)
    innerinput = (; lims=lims[1], dim=Val(1), state=states[1], p, kws=kws_)
    innerprob = ComposedCommonSolveProblem(innerinput, segprob, intprob) do (; lims, dim, state, p, kws), segsolver, intsolver
        segsolver.lims = lims
        segsolver.dim = _val_unwrap(dim)
        _segs = solve!(segsolver)
        intsolver.dom = _segs
        len = abs(_segs[end]-_segs[begin])
        intsolver.p = (; p, state)
        __kws = if tolalg isa StaticTolAlg
            if tolalg.method == bbox
                kws
            elseif tolalg.method == project
                _rescale_abstol(tolalg_params.invoutermeasure; kws...)
            else
                error("$(tolalg.method) not implemented")
            end
        elseif tolalg isa AdaptiveTolAlg
            kws
        elseif tolalg isa v04TolAlg
            length(states) == 1 ? kws : tolalg_params.init_kws
        elseif tolalg isa v03TolAlg
            _rescale_abstol(length(states) == 1 ? one(1/len) : 1/len; kws...)
        else
            error("tolalg not recognized")
        end
        intsolver.kwargs = (; intsolver.kwargs..., __kws...)
        return solve!(intsolver)
    end

    inneralg = ComposedCommonSolveAlgorithm(SegmentAlgorithm(), algs[1])
    prob, alg = nested_prob(innerprob, inneralg, fprototype, p, x0, segs[2:end], lims[2:end], states[2:end], algs[2:end], spec, exec, tolalg; kws...)
    return init(prob, alg)
end

function do_integral(f, dom, p, alg::NestedQuad, cacheval; kws...)
    lims = dom isa AbstractIteratedLimits ? dom : load_limits(dom)
    cacheval.input = (; cacheval.input..., p, lims, kws=(; kws...))
    return solve!(cacheval)
end

abstract type StatsAlgorithm end

struct StatsProblem{A,C}
    args::A
    channel::C
    batch::Bool
end

mutable struct StatsSolver{A,C}
    batch::Bool
    alg::A
    cacheval::C
end

function init(prob::StatsProblem, alg::StatsAlgorithm)
    cacheval = init_stats_cacheval(alg, prob.batch, prob.args...)
    put!(prob.channel, cacheval)
    return StatsSolver(prob.batch, alg, cacheval)
end

function step_stats!(solver::StatsSolver, args...)
    return stats_step!(solver.cacheval, solver.alg, solver.batch, args...)
end
struct SolverStats{A <: IntegralAlgorithm, S <: StatsAlgorithm} <: IntegralAlgorithm
    alg::A
    stats::S
end

function init_cacheval(f, dom, p, alg::SolverStats; kws...)
    channel = Channel(Inf)
    g = insert_stats(alg, f, get_prototype(dom), p, channel)
    return g, channel, init_cacheval(g, dom, p, alg.alg; kws...)
end
function do_integral(f, dom, p, alg::SolverStats, (g, channel, cacheval); kws...)
    stats_reset(alg.stats, channel)
    sol = do_integral(g, dom, p, alg.alg, cacheval; kws...)
    stats = stats_summary(alg.stats, channel, sol)
    return IntegralSolution(sol.value, sol.retcode, stats)
end
function insert_stats(alg::SolverStats, f::IntegralFunction, x, p, channel)
    proto = get_prototype(f, x, p)
    prob = StatsProblem((x, p, proto), channel, false)
    CommonSolveIntegralFunction(prob, alg.stats, proto, DefaultSpecialize(), f.executor) do solver, x, p
        value = f.f(x, p)
        step_stats!(solver, x, p, value)
        return value
    end
end
function insert_stats(alg::SolverStats, f::InplaceIntegralFunction, x, p, channel)
    prob = StatsProblem((x, p, f.prototype), channel, false)
    solver = init(prob, alg.stats)
    _f = (y, x, p) -> begin
        out = f.f!(y, x, p)
        step_stats!(solver, x, p, y)
        return out
    end
    # TODO return a commonsolve function because this integrand is not thread safe
    return InplaceIntegralFunction(_f, f.prototype)
end
function insert_stats(alg::SolverStats, f::InplaceBatchIntegralFunction, x, p, channel)
    prob = StatsProblem((x, p, f.prototype), channel, true)
    solver = init(prob, alg.stats)
    _f = (y, x, p) -> begin
        out = f.f!(y, x, p)
        step_stats!(solver, x, p, y)
        return out
    end
    # TODO return a commonsolve function because this integrand is not thread safe
    return InplaceBatchIntegralFunction(_f, f.prototype; max_batch= f.max_batch)
end
function insert_stats(alg::SolverStats, f::CommonSolveIntegralFunction, x, p, channel)
    input = (; x, p)
    prob = ComposedCommonSolveProblem(input, f.prob, StatsProblem((x, p, f.prototype), channel, false)) do (; x, p), probsolver, statssolver
        out = f.solve!(probsolver, x, p)
        step_stats!(statssolver, x, p, out)
        return out
    end
    alg = ComposedCommonSolveAlgorithm(f.alg, alg.stats)
    return CommonSolveIntegralFunction(prob, alg, f.prototype, f.specialize, f.executor) do solver, x, p
        # solver.input = (; solver.input..., x, p)
        # return solve!(solver)
        ## using out-of-place semantics can be faster
        return solver.solve!((; x, p), solver.solvers...)
    end
end

"""
    EvalCounter(alg::IntegralAlgorithm)

This algorithm wrapper will count the number of function evaluations used by integration algorithm `alg` during a solve. This information is found in the `sol.stats.numevals` field of an `IntegralSolution`.
"""
struct EvalCounter <: StatsAlgorithm end
EvalCounter(alg) = SolverStats(alg, EvalCounter())

function init_stats_cacheval(::EvalCounter, batch, args...)
    Ref(0)
end
function stats_step!(counter, ::EvalCounter, batch, args...)
    x = args[1]
    sol = args[end]
    if sol isa CommonSolutionStats
        if haskey(sol.stats, :numevals)
            counter[] += sol.stats.numevals
        else
            counter[] += 1
        end
    elseif batch
        counter[] += size(x)[end]
    else
        counter[] += 1
    end
    return
end
function stats_reset(::EvalCounter, channel)
    for c in channel.data
        c[] = 0
    end
end
function stats_summary(::EvalCounter, channel, sol)
    numevals = 0
    for c in channel.data
        numevals += c[]
    end
    return (; sol.stats..., numevals)
end

"""
    EvalLogger(alg::IntegralAlgorithm)

This algorithm wrapper will record the quadrature points used by integration algorithm `alg` during a solve. This information is found in the `sol.stats.evallog` field of an `IntegralSolution`.
"""
struct EvalLogger <: StatsAlgorithm end
EvalLogger(alg) = SolverStats(alg, EvalLogger())

function init_stats_cacheval(::EvalLogger, args...)
    x, = args
    Vector{typeof(x)}(undef, 0)
end
function stats_step!(log, ::EvalLogger, batch, args...)
    x, = args
    push!(log, x)
end
function stats_reset(::EvalLogger, channel)
    for c in channel.data
        empty!(c)
    end
end
function stats_summary(::EvalLogger, channel, sol)
    evallog = Iterators.flatten((log for log in channel.data))
    return (; sol.stats..., evallog)
end
