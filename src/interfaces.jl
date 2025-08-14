abstract type AbstractIntegralFunction end
# should have at least two fields:
# - f
# - integrand_prototype


"""
    AbstractSpecialization

Supertype for compiler specializations of commonsolve functions to control code generation and inference.
"""
abstract type AbstractSpecialization end

"""
    AbstractExecutor

Supertype of policies for how to schedule the execution of commonsolve functions.
"""
abstract type AbstractExecutor end

"""
    IntegralFunction(f, [prototype=nothing, executor=SerialExecutor()])

Constructor for an out-of-place integrand of the form `f(x, p)`.
Optionally, a `prototype` can be provided for the output of the function.
"""
struct IntegralFunction{F,P,E<:AbstractExecutor} <: AbstractIntegralFunction
    f::F
    prototype::P
    executor::E
end
IntegralFunction(f, proto=nothing) = IntegralFunction(f, proto, SerialExecutor())

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
the last axis, which is reserved for batching, should contain at least one element.
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

"""
    DefaultSpecialize()

Specialize a commonsolve function using default heuristics of Julia's compiler.
"""
struct DefaultSpecialize <: AbstractSpecialization end

"""
    NoSpecialize()

Type-stable specialization of a commonsolve function without code generation or inference based on the solver type using the `@nospecializeinfer` macro.
Asserts that the returned value is of the same type as the prototype.
Strikes a good balance of compile time and run time.
"""
struct NoSpecialize <: AbstractSpecialization end

"""
    FullSpecialize()

Type-stable specialization of a commonsolve function with most code generation and full inference.
Asserts that the returned value is of the same type as the prototype.
This may lead to excessive compile times but will usually give the fastest runtime.
"""
struct FullSpecialize <: AbstractSpecialization end

"""
    FunctionWrapperSpecialize()

Type-stable specialization of a commonsolve function behind a C-function points.
Requires `using FunctionWrappers` as this is implemented in a package extension.
Asserts that the returned value is of the same type as the prototype.
This gives both very fast runtimes, compile times, and zero allocations, but may be brittle w.r.t. world age and is not as flexible with types of integration limits.
"""
struct FunctionWrapperSpecialize <: AbstractSpecialization end

"""
    SerialExecutor()

Policy that a commonsolve function be executed on a single thread.
"""
struct SerialExecutor <: AbstractExecutor end


"""
    ThreadedExecutor(ntasks::Int, max_batch::Int)
    ThreadedExecutor(; ntasks::Int, max_batch::Int)

Policy that a commonsolve function be executed on multiple threads, scheduling up to `ntasks` tasks at a time which may exceed the number of threads.
The pool of commonsolve workers is of size `ntasks`.
`max_batch` sets a soft limit on the number of quadrature points assigned to any task so that the program does not run out of memory.
If `max_batch` is too small, the overhead of scheduling tasks could negate the speedup of parallelization.
"""
Base.@kwdef struct ThreadedExecutor <: AbstractExecutor
    ntasks::Int
    max_batch::Int
end


"""
    CommonSolveIntegralFunction(solve!, prob, alg, [prototype, specialize, executor]; kws...)

Constructor for an integrand that solves a problem defined with the CommonSolve.jl
interface, `prob`, which is instantiated using `init(prob, alg; kws...)`.
The `solution = solve!(solver, x, p)` function supplied by the caller must do the work of the problem, although it need not be a method of `CommonSolve.solve!` as the out-of-place semantics of passing the arguments `x, p` can provide a speedup.
The `prototype` argument can help control how much to `specialize` on the solution type of the
problem. By default, `specialize=DefaultSpecialize()` uses Julia's default heuristics, which can give up on inference in complicated codes.
Additionally, `FullSpecialize()` can obtain the fastest run times with the longest compile times, `NoSpecialize()` strikes a good balance of run time, compile time and inference, and `FunctionWrapperSpecialize()` may have the fastest compile time and very good run times (comparable to `FullSpecialize()`) but with possible issues regarding world age.
The `executor` keyword specifies how to schedule and run the integrand evaluation, defaulting to `SerialExecutor()` with an additional option for `ThreadedExecutor(::Integer)`.
"""
struct CommonSolveIntegralFunction{F,P,A,K,T,M<:AbstractSpecialization,E<:AbstractExecutor} <: AbstractIntegralFunction
    solve!::F
    prob::P
    alg::A
    kwargs::K
    prototype::T
    specialize::M
    executor::E
end
function CommonSolveIntegralFunction(solve!, prob, alg, prototype=nothing, specialize=DefaultSpecialize(), executor=SerialExecutor(); kws...)
    return CommonSolveIntegralFunction(solve!, prob, alg, NamedTuple(kws), prototype, specialize, executor)
end

"""
    CommonSolutionStats(value, stats)

When a `CommonSolveIntegralFunction` or `CommonSolveFourierIntegralFunction` returns its solution in this struct in the `value` field, additional information about the solve may also be passed in the `stats` field.
Currently, the only use is for `EvalCounter` to count integrand evaluations for an IntegralProblem solved within another integral problem.
"""
struct CommonSolutionStats{V,S}
    value::V
    stats::S
end

function do_solve!(solver, f::CommonSolveIntegralFunction, x, p)
    sol = f.solve!(solver, x, p)
    if sol isa CommonSolutionStats
        return sol.value
    else
        return sol
    end
end
Base.@nospecializeinfer function do_solve_nsp!(@nospecialize(solver), f::CommonSolveIntegralFunction, x, p)
    return do_solve!(solver, f, x, p)
end
function get_prototype(f::CommonSolveIntegralFunction, x, p, _solver=nothing)
    sol = if isnothing(f.prototype)
        solver = isnothing(_solver) ? init(f.prob, f.alg; f.kwargs...) : _solver
        do_solve!(solver, f, x, p)
    else
        f.prototype
    end
    if sol isa CommonSolutionStats
        return sol.value
    else
        return sol
    end
end
function init_specialized_integrand(::DefaultSpecialize, solver, f, x, p, prototype)
    do_solve!
end
function init_specialized_integrand(::NoSpecialize, solver, f, x, p, prototype)
    (solver, f, x, p) -> do_solve_nsp!(solver, f, x, p)::typeof(prototype)
end
function init_specialized_integrand(::FullSpecialize, solver, f, x, p, prototype)
    (solver, f, x, p) -> do_solve!(solver, f, x, p)::typeof(prototype)
end

init_commonsolvefunction(f, dom, p) = init_commonsolvefunction_(f.executor, f, get_prototype(dom), p)
function init_commonsolvefunction_(::SerialExecutor, f, x, p)
    solver = init(f.prob, f.alg; f.kwargs...)
    prototype = get_prototype(f, x, p, solver)
    integrand = init_specialized_integrand(f.specialize, solver, f, x, p, prototype)
    return solver, integrand, prototype
end
function init_commonsolvefunction_(exec::ThreadedExecutor, f, x, p)
    channel = fillchannel(exec) do
        init(f.prob, f.alg; f.kwargs...)
    end
    solver = fetch(channel)
    prototype = get_prototype(f, x, p, solver)
    integrand = init_specialized_integrand(f.specialize, solver, f, x, p, prototype)
    return channel, integrand, prototype
end


function fillchannel(f, exec::ThreadedExecutor)
    return fillchannel(f, exec.ntasks)
end
function fillchannel(f, n::Integer)
    item = f()
    ch = Channel{typeof(item)}(n)
    put!(ch, item)
    for _ in 2:n
        put!(ch, f())
    end
    return ch
end

function do_threaded_solve!(integrand, channel, f, y, x, p)
    @sync for (iy, xi) in zip(eachindex(y), x)
        solver = take!(channel)
        Threads.@spawn begin
            # TODO mini-batch the x evaluations
            y[iy] = integrand(solver, f, xi, p)
            put!(channel, solver)
        end
    end
end

# TODO add InplaceCommonSolveIntegralFunction and InplaceBatchCommonSolveIntegralFunction

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
