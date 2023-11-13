# we recreate a lot of the SciML Integrals.jl functionality, but only for our algorithms
# the features we omit are: inplace integrands, infinite limit transformations, nout and
# batch keywords. Otherwise, there is a correspondence between.
# solve -> solve!
# init -> init

"""
    IntegralAlgorithm

Abstract supertype for integration algorithms.
"""
abstract type IntegralAlgorithm end

# Methods an algorithm must define
# - init_cacheval
# - solve!

struct NullParameters end

struct IntegralProblem{F,D,P}
    f::F
    dom::D
    p::P
    function IntegralProblem{F,D,P}(f::F, dom::D, p::P) where {F,D,P}
        return new{F,D,P}(f, dom, p)
    end
end
function IntegralProblem(f::F, dom::D, p::P=NullParameters()) where {F,D,P}
    return IntegralProblem{F,D,P}(f, dom, p)
end
function IntegralProblem(f::F, a::T, b::T, p::P=NullParameters()) where {F,T,P}
    dom = T <: Number ? PuncturedInterval((a, b)) : HyperCube(a, b)
    return IntegralProblem{F,typeof(dom),P}(f, dom, p)
end

mutable struct IntegralCache{F,D,P,A,C,K}
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

"""
    init(::IntegralProblem, ::IntegralAlgorithm; kws...)::IntegralCache
"""
function init(prob::IntegralProblem, alg::IntegralAlgorithm; kwargs...)
    checkkwargs(NamedTuple(kwargs))
    f = prob.f; dom = prob.dom; p = prob.p
    return make_cache(f, dom, p, alg; kwargs...)
end

"""
    solve(::IntegralProblem, ::IntegralAlgorithm; kws...)::IntegralSolution
"""
function solve(prob::IntegralProblem, alg::IntegralAlgorithm; kwargs...)
    cache = init(prob, alg; kwargs...)
    return solve!(cache)
end

"""
    solve!(::IntegralCache)::IntegralSolution
"""
function solve!(c::IntegralCache)
    return do_solve(c.f, c.dom, c.p, c.alg, c.cacheval; c.kwargs...)
end

struct IntegralSolution{T,E}
    u::T
    resid::E
    retcode::Bool
    numevals::Int
end
# a value of numevals < 0 means it was undefined


"""
    IntegralSolver(cache::IntegralCache)

This struct is a functor that solves an integral problem as a function of the problem
parameters for a given algorithms and tolerances.
"""
struct IntegralSolver{F,D,A,T,K} <: Function
    f::F
    dom::D
    alg::A
    cacheval::T
    kwargs::K
end

# some algorithms can already allocate a rule based on `dom`, but we leave extending this
# method as an optimization to the caller
init_solver_cacheval(f, dom, alg) = nothing

"""
    IntegralSolver(f, dom, alg; abstol, reltol, maxiters)
"""
function IntegralSolver(f, dom, alg::IntegralAlgorithm; kwargs...)
    checkkwargs(NamedTuple(kwargs))
    cacheval = init_solver_cacheval(f, dom, alg)
    return IntegralSolver(f, dom, alg, cacheval, NamedTuple(kwargs))
end

"""
    IntegralSolver(f, lb, ub, alg::AbstractIntegralAlgorithm; abstol, reltol, maxiters)

Returns a functor, `fun`, that accepts `MixedParameters` for input via the
following interface `fun(args...; kwargs...) -> solve(IntegralProblem(f, lb, ub,
MixedParameters(args..., kwargs...)), alg)`.
"""
function IntegralSolver(f, a::T, b::T, alg::IntegralAlgorithm; kwargs...) where {T}
    dom = T <: Number ? PuncturedInterval((a, b)) : HyperCube(a, b)
    return IntegralSolver(f, dom, alg; kwargs...)
end

function IntegralSolver(prob::IntegralProblem, alg::IntegralAlgorithm; kwargs...)
    return IntegralSolver(prob.f, prob.dom, alg; kwargs...)
end


# layer to intercept the problem parameters & algorithm and transform them
remake_cache(args...) = IntegralCache(args...)
remake_cache(c, p) = remake_cache(c.f, c.dom, p, c.alg, c.cacheval, c.kwargs)

function solve_p(s::IntegralSolver, p)
    c = if s.cacheval===nothing
        prob = IntegralProblem(s.f, s.dom, p)
        init(prob, s.alg; s.kwargs...)
    else
        remake_cache(s, p)
    end
    return solve!(c)
end

function (s::IntegralSolver)(p)
    sol = solve_p(s, p)
    return sol.u
end

# parallelization

"""
    batchparam(ps, nthreads)

If the cost of a calculation smoothly varies with the parameters `ps`, then
batch `ps` into `nthreads` groups where the `i`th element of group `j` is
`ps[j+(i-1)*nthreads]` along the longest axis of `ps`. We assume that multidimensional
arrays of parameters have smoothest cost along their longest axis
"""
function batchparam(xs::AbstractArray{T,N}, nthreads) where {T,N}
    (s = size(xs)) === () && return (((CartesianIndex(()), only(xs)),),)
    @assert nthreads >= 1
    len, dim = findmax(s)
    batches = [Tuple{CartesianIndex{N},T}[] for _ in 1:min(nthreads, len)]
    for i in CartesianIndices(xs)
        push!(batches[mod(i[dim]-1, nthreads)+1], (i, xs[i]))
    end
    return batches
end

function batchsolve!(out, f, ps, nthreads, callback)
    n = Threads.Atomic{Int}(0)
    Threads.@threads for batch in batchparam(ps, nthreads)
        f_ = Threads.threadid() == 1 ? f : deepcopy(f) # avoid data races for in place integrators
        for (i, p) in batch
            t = time()
            sol = solve_p(f_, p)
            callback(f, i, Threads.atomic_add!(n, 1) + 1, p, sol, time() - t)
            out[i] = sol.u
        end
    end
    return out
end

solver_type(::F, ::P) where {F,P} = Base.promote_op((f, p) -> solve_p(f, p).u, F, P)

"""
    batchsolve(f::IntegralSolver, ps::AbstractArray, [T]; nthreads=Threads.nthreads())

Evaluate the [`IntegralSolver`](@ref) `f` at each of the parameters `ps` in
parallel. Returns an array similar to `ps` containing the evaluated integrals `I`. This is
a form of multithreaded broadcasting. Providing the return type `f(eltype(ps))::T` is
optional, but will help in case inference of that type fails.
"""
function batchsolve(f::IntegralSolver, ps::AbstractArray, T=solver_type(f, ps[begin]); nthreads=Threads.nthreads(), callback=(x...)->nothing)
    solver = if f.cacheval === nothing
        prob = IntegralProblem(f.f, f.dom, ps[begin])
        cache = init(prob, f.alg; f.kwargs...)
        IntegralSolver(f.f, f.dom, f.alg, cache.cacheval, f.kwargs)
    else
        f
    end
    return batchsolve!(similar(ps, T), solver, ps, nthreads, callback)
end
