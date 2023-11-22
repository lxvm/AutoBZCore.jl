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

"""
    NullParameters()

A singleton type representing absent parameters
"""
struct NullParameters end

"""
    IntegralProblem(f, domain, [p=NullParameters])
    IntegralProblem(f, a::T, b::T, [p=NullParameters]) where {T}

Collects the data need to define an integral of a function `f(x, p)` over a `domain`
containing the points, `x`, and set with parameters `p` (default: [`NullParameters`](@ref)).
If the domain is an interval or hypercube, it can also be specified by its endpoints `a, b`,
and it gets converted to a [`PuncturedInterval`](@ref) or [`HyperCube`](@ref).
"""
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

function make_cache(f, dom, p, alg::IntegralAlgorithm; kwargs...)
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

Construct a cache for an [`IntegralProblem`](@ref), [`IntegralAlgorithm`](@ref), and the
keyword arguments to the solver (i.e. `abstol`, `reltol`, or `maxiters`) that can be reused
for solving the problem for multiple different parameters of the same type.
"""
function init(prob::IntegralProblem, alg::IntegralAlgorithm; kwargs...)
    checkkwargs(NamedTuple(kwargs))
    f = prob.f; dom = prob.dom; p = prob.p
    return make_cache(f, dom, p, alg; kwargs...)
end

"""
    solve(::IntegralProblem, ::IntegralAlgorithm; kws...)::IntegralSolution

Compute the solution to the given [`IntegralProblem`](@ref) using the given
[`IntegralAlgorithm`](@ref) for the given keyword arguments to the solver (i.e. `abstol`,
`reltol`, or `maxiters`).

## Keywords
- `abstol`: an absolute error tolerance to get the solution to a specified number of
  absolute digits, e.g. 1e-3 requests accuracy to 3 decimal places.  Note that this number
  must have the same units as the integral. (default: nothing)
- `reltol`: a relative error tolerance equivalent to specifying a number of significant
  digits of accuracy, e.g. 1e-4 requests accuracy to roughly 4 significant digits. (default:
  nothing)
- `maxiters`: a soft upper limit on the number of integrand evaluations (default:
  `typemax(Int)`)

Solvers typically converge only to the weakest error condition. For example, a relative
tolerance can be used in combination with a smaller-than necessary absolute tolerance so
that the solution is resolved up to the requested significant digits, unless the integral is
smaller than the absolute tolerance.
"""
function solve(prob::IntegralProblem, alg::IntegralAlgorithm; kwargs...)
    cache = init(prob, alg; kwargs...)
    return solve!(cache)
end

"""
    solve!(::IntegralCache)::IntegralSolution

Compute the solution to an [`IntegralProblem`](@ref) constructed from [`init`](@ref).
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
    IntegralSolver(f, dom, alg; [abstol, reltol, maxiters])
    IntegralSolver(f, lb, ub, alg::AbstractIntegralAlgorithm; [abstol, reltol, maxiters])

Returns a functor, `fun`, that accepts input parameters `p` and solves the corresponding
integral `fun(p) -> solve(IntegralProblem(f, lb, ub, p), alg).u`. See [`solve`](@ref) for
details on the keywords.

If `f` is a [`ParameterIntegrand`](@ref) or [`FourierIntegrand`](@ref), then the functor
interface is modified to accept parameters as function arguments, and the following is done:
`fun(args...; kwargs...) ->  solve(IntegralProblem(f, lb, ub, merge(f.p,
MixedParameters(args...; kwargs...))), alg).u` where `f.p` are the preset parameters of `f`.
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

function IntegralSolver(f, dom, alg::IntegralAlgorithm; kwargs...)
    checkkwargs(NamedTuple(kwargs))
    cacheval = init_solver_cacheval(f, dom, alg)
    return IntegralSolver(f, dom, alg, cacheval, NamedTuple(kwargs))
end

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
