abstract type AbstractIntegralFunction end
# should have at least two fields:
# - f
# - integrand_prototype

"""
    IntegralFunction(f, [prototype=nothing])

Constructor for an out-of-place integrand of the form `f(x, p)`.
Optionally, a `prototype` can be provided for the output of the function.
"""
struct IntegralFunction{F,P} <: AbstractIntegralFunction
    f::F
    prototype::P
end
IntegralFunction(f) = IntegralFunction(f, nothing)

function get_prototype(f::IntegralFunction, x, p)
    f.prototype === nothing ? f.f(x, p) : f.prototype
end

"""
    InplaceIntegralFunction(f!, prototype::AbstractArray)

Constructor for an inplace integrand of the form `f!(y, x, p)`.
A `prototype` array is required to store the same type and size as the result, `y`.
"""
struct InplaceIntegralFunction{F,P<:AbstractArray} <: AbstractIntegralFunction
    # in-place function f!(y, x, p) that takes one x value and outputs an array of results in-place
    f!::F
    prototype::P
end

function get_prototype(f::InplaceIntegralFunction, x, p)
    # iip is required to have a prototype array
    f.prototype
end

"""
    InplaceBatchIntegralFunction(f!, prototype; max_batch::Integer=typemax(Int))

Constructor for an inplace, batched integrand of the form `f!(y, x, p)` that accepts an
array `x` containing a batch of evaluation points stored along the last axis of the array.
A `prototype` array is required to store the same type and size as the result, `y`, however
the last axis, which is reserved for batching, which should contain at least one element.
The `max_batch` keyword sets a soft limit on the number of points batched simultaneously.
"""
struct InplaceBatchIntegralFunction{F,P<:AbstractArray} <: AbstractIntegralFunction
    f!::F
    prototype::P
    max_batch::Int
end

function InplaceBatchIntegralFunction(f!, p::AbstractArray; max_batch::Integer=typemax(Int))
    return InplaceBatchIntegralFunction(f!, p, max_batch)
end

function get_prototype(f::InplaceBatchIntegralFunction, x, p)
    # iip is required to have a prototype array
    f.prototype
end

abstract type AbstractSpecialization end
struct NoSpecialize <: AbstractSpecialization end
struct FunctionWrapperSpecialize <: AbstractSpecialization end
struct FullSpecialize <: AbstractSpecialization end

"""
    CommonSolveIntegralFunction(prob, alg, update!, postsolve, [prototype, specialize]; kws...)

Constructor for an integrand that solves a problem defined with the CommonSolve.jl
interface, `prob`, which is instantiated using `init(prob, alg; kws...)`. Helper functions
include: `update!(cache, x, p)` is called before
`solve!(cache)`, followed by `postsolve(sol, x, p)`, which should return the value of the solution.
The `prototype` argument can help control how much to `specialize` on the type of the
problem, which defaults to `FullSpecialize()` so that run times are improved. However
`FunctionWrapperSpecialize()` may help reduce compile times.
"""
struct CommonSolveIntegralFunction{P,A,K,U,S,T,M<:AbstractSpecialization} <: AbstractIntegralFunction
    prob::P
    alg::A
    kwargs::K
    update!::U
    postsolve::S
    prototype::T
    specialize::M
end
function CommonSolveIntegralFunction(prob, alg, update!, postsolve, prototype=nothing, specialize=FullSpecialize(); kws...)
    return CommonSolveIntegralFunction(prob, alg, NamedTuple(kws), update!, postsolve, prototype, specialize)
end

function do_solve!(cache, f::CommonSolveIntegralFunction, x, p)
    f.update!(cache, x, p)
    sol = solve!(cache)
    return f.postsolve(sol, x, p)
end
function get_prototype(f::CommonSolveIntegralFunction, x, p)
    if isnothing(f.prototype)
        cache = init(f.prob, f.alg; f.kwargs...)
        do_solve!(cache, f, x, p)
    else
        f.prototype
    end
end
function init_specialized_integrand(cache, f, dom, p; x=get_prototype(dom), prototype=f.prototype)
    proto = prototype === nothing ? do_solve!(cache, f, x, p) : prototype
    func = (x, p) -> do_solve!(cache, f, x, p)
    integrand = if f.specialize isa FullSpecialize
        func
    elseif f.specialize isa FunctionWrapperSpecialize
        FunctionWrapper{typeof(prototype), typeof((x, p))}(func)
    else
        throw(ArgumentError("$(f.specialize) is not implemented"))
    end
    return integrand, proto
end
function _init_commonsolvefunction(f, dom, p; kws...)
    cache = init(f.prob, f.alg; f.kwargs...)
    integrand, prototype = init_specialized_integrand(cache, f, dom, p; kws...)
    return cache, integrand, prototype
end

# TODO add InplaceCommonSolveIntegralFunction and InplaceBatchCommonSolveIntegralFunction
# TODO add ThreadedCommonSolveIntegralFunction and DistributedCommonSolveIntegralFunction

"""
    IntegralAlgorithm

Abstract supertype for integration algorithms.
"""
abstract type IntegralAlgorithm end

"""
    NullParameters()

A singleton type representing absent parameters
"""
struct NullParameters end

"""
    IntegralProblem(f, domain, [p=NullParameters]; kwargs...)

## Arguments
- `f::AbstractIntegralFunction`: The function to integrate
- `domain`: The domain to integrate over, e.g. `(lb, ub)`
- `p`: Parameters to pass to the integrand

## Keywords
Additional keywords are passed directly to the solver
"""
struct IntegralProblem{F<:AbstractIntegralFunction,D,P,K<:NamedTuple}
    f::F
    dom::D
    p::P
    kwargs::K
end
function IntegralProblem(f::AbstractIntegralFunction, dom, p=NullParameters(); kws...)
    return IntegralProblem(f, dom, p, NamedTuple(kws))
end
function IntegralProblem(f, dom, p=NullParameters(); kws...)
    return IntegralProblem(IntegralFunction(f), dom, p; kws...)
end

mutable struct IntegralSolver{F,D,P,A,C,K}
    f::F
    dom::D
    p::P
    alg::A
    cacheval::C
    kwargs::K
end

function checkkwargs(kwargs)
    for key in keys(kwargs)
        key in (:abstol, :reltol, :maxiters) || throw(ArgumentError("keyword $key unrecognized"))
    end
    return nothing
end

"""
    init(::IntegralProblem, ::IntegralAlgorithm; kws...)::IntegralSolver

Construct a cache for an [`IntegralProblem`](@ref), [`IntegralAlgorithm`](@ref), and the
keyword arguments to the solver (i.e. `abstol`, `reltol`, or `maxiters`) that can be reused
for solving the problem for multiple different parameters of the same type.
"""
function init(prob::IntegralProblem, alg::IntegralAlgorithm; kwargs...)
    f = prob.f; dom = prob.dom; p = prob.p
    kws = (; prob.kwargs..., kwargs...)
    checkkwargs(kws)
    cacheval = init_cacheval(f, dom, p, alg; kws...)
    return IntegralSolver(f, dom, p, alg, cacheval, kws)
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
solve(prob::IntegralProblem, alg::IntegralAlgorithm; kwargs...)

"""
    solve!(::IntegralSolver)::IntegralSolution

Compute the solution to an [`IntegralProblem`](@ref) constructed from [`init`](@ref).
"""
function solve!(c::IntegralSolver)
    return do_integral(c.f, c.dom, c.p, c.alg, c.cacheval; c.kwargs...)
end

@enum ReturnCode begin
    Success
    Failure
    MaxIters
end

struct IntegralSolution{T,S}
    value::T
    retcode::ReturnCode
    stats::S
end
