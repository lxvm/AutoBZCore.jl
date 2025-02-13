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

unroll_limits(dom) = unroll_limits(limit_iterate(dom)...)
function unroll_limits(segs, lims, state)
    a, b, = segs
    x = (a + b)/2
    next = limit_iterate(lims, state, x)
    next isa SVector && return next, (segs,), (lims,), (state,)
    x0, _segs, _lims, _state = unroll_limits(next...)
    return x0, (_segs..., segs), (_lims..., lims), (_state..., state)
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
    return IteratedIntegration.segments(solver.lims, solver.dim)
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
    return IteratedIntegration.fixandeliminate(solver.lims, solver.x, solver.dim)
end

_val_unwrap(::Val{X}) where {X} = X
function nested_prob(innerprob, inneralg, prototype, p, x0, segs, lims, states, algs, spec, exec; kws...)
    length(states) == 0 && return innerprob, inneralg
    elimprob = EliminationProblem(lims[1], x0[ndims(lims[1])], Val(ndims(lims[1])))
    eliminput = (; x=x0[ndims(lims[1])], lims=lims[1], state=states[1], dim=Val(ndims(lims[1])), p, kws=innerprob.input.kws)
    _prob = ComposedCommonSolveProblem(eliminput, elimprob, innerprob) do (; x, lims, state, dim, p, kws), elimsolver, innersolver
        elimsolver.x = x
        elimsolver.lims = lims
        elimsolver.dim = dim
        _state = (x, state...)
        _lims = solve!(elimsolver)
        innersolver.input = (; innersolver.input..., lims=_lims, state=_state, dim=Val(_val_unwrap(dim)-1), p, kws)
        return solve!(innersolver)
    end
    _alg = ComposedCommonSolveAlgorithm(EliminationAlgorithm(), inneralg)
    _f = CommonSolveIntegralFunction(_prob, _alg, prototype*real(prod(x0[begin:ndims(lims[1])-1])), spec[1], exec[1]) do solver, x, p
        # solver.input = (; solver.input..., p..., x)
        # sol = solve!(solver)
        ## out-of-place semantics may be faster
        sol = solver.solve!((; solver.input..., p..., x), solver.solvers...)
        return sol.value
    end
    __f = nested_integralfunction(_f, x0[ndims(lims[1])], p)
    _kws = _rescale_abstol(1/real(prod(x0[begin+ndims(lims[1]):end])); kws...)
    innerinput = (; lims=lims[1], dim=Val(ndims(lims[1])), state=states[1], p, kws=_kws)
    intprob = IntegralProblem(__f, segs[1], (; innerinput..., kws=eliminput.kws); _kws...)

    segprob = SegmentProblem(lims[1], ndims(lims[1]))
    _innerprob = ComposedCommonSolveProblem(innerinput, segprob, intprob) do (; lims, dim, state, p, kws), segsolver, intsolver
        segsolver.lims = lims
        segsolver.dim = _val_unwrap(dim)
        segs = solve!(segsolver)
        intsolver.dom = segs
        len = abs(segs[end]-segs[begin])
        intsolver.p = (; intsolver.p..., lims, state, dim, p, kws=_rescale_abstol(1/len; kws...))
        intsolver.kwargs = (; intsolver.kwargs..., kws...)
        return solve!(intsolver)
    end
    _inneralg = ComposedCommonSolveAlgorithm(SegmentAlgorithm(), algs[1])

    nested_prob(_innerprob, _inneralg, prototype, p, x0, segs[2:end], lims[2:end], states[2:end], algs[2:end], spec[2:end], exec[2:end]; kws...)
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
    func = InplaceBatchIntegralFunction(proto; max_batch=exec.ntasks) do y, x, p
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
    _solve! = (solver, x, (; p, state)) -> f.solve!(solver, SVector(promote(x, state...)), p)
    return CommonSolveIntegralFunction(_solve!, f.prob, f.alg, f.prototype, f.specialize, f.executor; f.kwargs...)
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
    exec = alg.executor isa AbstractExecutor ? ntuple(i -> alg.executor, Val(ndims(dom))) : alg.executor

    _f = nested_innerintegralfunction(f, x0, p)

    intprob = IntegralProblem(_f, segs[1], (; p, state=states[1]); _rescale_abstol(1/real(prod(x0[begin+1:end])); kws...)...)
    segprob = SegmentProblem(lims[1], 1)
    innerinput = (; lims=lims[1], dim=Val(1), state=states[1], p, kws=intprob.kwargs)
    innerprob = ComposedCommonSolveProblem(innerinput, segprob, intprob) do (; lims, dim, state, p, kws), segsolver, intsolver
        segsolver.lims = lims
        segsolver.dim = _val_unwrap(dim)
        segs = solve!(segsolver)
        intsolver.dom = segs
        intsolver.p = (; p, state)
        intsolver.kwargs = (; intsolver.kwargs..., kws...)
        return solve!(intsolver)
    end

    inneralg = ComposedCommonSolveAlgorithm(SegmentAlgorithm(), algs[1])
    prob, alg = nested_prob(innerprob, inneralg, _f.prototype, p, x0, segs[2:end], lims[2:end], states[2:end], algs[2:end], spec[2:end], exec[2:end]; kws...)
    return init(prob, alg)
end

function do_integral(f, dom, p, alg::NestedQuad, cacheval; kws...)
    cacheval.input = (; cacheval.input..., p, lims=dom, kws=(; kws...))
    return solve!(cacheval)
end


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


struct CounterProblem{K}
    kwargs::K
end
CounterProblem(; kws...) = CounterProblem(kws)

mutable struct CounterSolver{K,G}
    kwargs::K
    alg::G
    counter::Int
end
abstract type CounterAlgorithm end
# this could
struct SingleCount <: CounterAlgorithm end
function init(prob::CounterProblem, alg::CounterAlgorithm; kws...)
    kwargs = (; prob.kwargs..., kws...)
    solver = CounterSolver(kwargs, alg, 0)
    if haskey(kwargs, :channel)
        put!(kwargs.channel, solver)
    end
    return solver
end
function step!(solver::CounterSolver)
    solver.counter += 1
    return solver.counter
end
function insert_counter(f::IntegralFunction, x, p, channel)
    prob = CounterProblem(; channel)
    alg = SingleCount()
    proto = get_prototype(f, x, p)
    CommonSolveIntegralFunction(prob, alg, proto) do solver, x, p
        step!(solver)
        return f.f(x, p)
    end
end
function insert_counter(f::InplaceIntegralFunction, x, p, channel)
    prob = CounterProblem(; channel)
    alg = SingleCount()
    solver = init(prob, alg)
    _f = (y, x, p) -> begin
        step!(solver)
        return f.f!(y, x, p)
    end
    # TODO return a commonsolve function because this integrand is not thread safe
    return InplaceIntegralFunction(_f, f.prototype)
end
function insert_counter(f::InplaceBatchIntegralFunction, x, p, channel)
    prob = CounterProblem(; channel)
    alg = SingleCount()
    solver = init(prob, alg)
    _f = (y, x, p) -> begin
        for _ in 1:size(x)[end]
            step!(solver)
        end
        return f.f!(y, x, p)
    end
    # TODO return a commonsolve function because this integrand is not thread safe
    return InplaceBatchIntegralFunction(_f, f.prototype; max_batch= f.max_batch)
end
function insert_counter(f::CommonSolveIntegralFunction, x, p, channel)
    input = (; x, p)
    prob = ComposedCommonSolveProblem(input, CounterProblem(; channel), f.prob) do (; x, p), countersolver, probsolver
        step!(countersolver)
        return f.solve!(probsolver, x, p)
    end
    alg = ComposedCommonSolveAlgorithm(SingleCount(), f.alg)
    return CommonSolveIntegralFunction(prob, alg, f.prototype, f.specialize, f.executor) do solver, x, p
        # solver.input = (; solver.input..., x, p)
        # return solve!(solver)
        ## using out-of-place semantics can be faster
        return solver.solve!((; x, p), solver.solvers...)
    end
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
