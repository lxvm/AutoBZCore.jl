"""
    MixedParameters(args::Tuple, kwargs::NamedTuple)

A struct to store full and partial sets of parameters used to evaluate
integrands.
"""
struct MixedParameters{A<:Tuple,K<:NamedTuple}
    args::A
    kwargs::K
end
MixedParameters(args...; kwargs...) = MixedParameters(args, NamedTuple(kwargs))

# parameter fusion and iteration utilities

Base.getindex(p::MixedParameters, i::Int) = getindex(getfield(p, :args), i)
Base.getproperty(p::MixedParameters, name::Symbol) = getproperty(getfield(p, :kwargs), name)

Base.merge(p::MixedParameters, q) =
    MixedParameters((getfield(p, :args)..., q), getfield(p, :kwargs))
Base.merge(q, p::MixedParameters) =
    MixedParameters((q, getfield(p, :args)...), getfield(p, :kwargs))
Base.merge(p::MixedParameters, q::NamedTuple) =
    MixedParameters(getfield(p, :args), merge(getfield(p, :kwargs), q))
Base.merge(q::NamedTuple, p::MixedParameters) =
    MixedParameters(getfield(p, :args), merge(q, getfield(p, :kwargs)))
Base.merge(p::MixedParameters, q::Tuple) =
    MixedParameters((getfield(p, :args)..., q...), getfield(p, :kwargs))
Base.merge(q::Tuple, p::MixedParameters) =
    MixedParameters((q..., getfield(p, :args)...), getfield(p, :kwargs))
Base.merge(p::MixedParameters, q::MixedParameters) =
    MixedParameters((getfield(p, :args)..., getfield(q, :args)...), merge(getfield(p, :kwargs), getfield(q, :kwargs)))

function paramzip_(::Tuple{}, ::NamedTuple{(), Tuple{}})
    MixedParameters{Tuple{},NamedTuple{(), Tuple{}}}[]
end
function paramzip_(args::Tuple, ::NamedTuple{(), Tuple{}})
    [MixedParameters(arg, NamedTuple()) for arg in zip(args...)]
end
function paramzip_(::Tuple{}, kwargs::NamedTuple)
    [MixedParameters((), NamedTuple{keys(kwargs)}(val)) for val in zip(values(kwargs)...)]
end
function paramzip_(args::Tuple, kwargs::NamedTuple)
    [MixedParameters(arg, NamedTuple{keys(kwargs)}(val)) for (arg, val) in zip(zip(args...), zip(values(kwargs)...))]
end
paramzip(args...; kwargs...) = paramzip_(args, NamedTuple(kwargs))

function paramproduct_(args::Tuple, kwargs::NamedTuple)
    [MixedParameters(item[1:length(args)], NamedTuple{keys(kwargs)}(item[length(args)+1:end])) for item in Iterators.product(args..., values(kwargs)...)]
end
paramproduct(args...; kwargs...) = paramproduct_(args, NamedTuple(kwargs))

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
    dom = T <: Real ? PuncturedInterval((a, b)) : HyperCube(a, b)
    return IntegralSolver(f, dom, alg; kwargs...)
end

function IntegralSolver(prob::IntegralProblem, alg::IntegralAlgorithm; kwargs...)
    return IntegralSolver(prob.f, prob.dom, alg; kwargs...)
end


# layer to intercept the problem parameters & algorithm and transform them
remake_cache(args...) = IntegralCache(args...)
remake_cache(c, p) = remake_cache(c.f, c.dom, p, c.alg, c.cacheval, c.kwargs)

function do_solve(s::IntegralSolver, p)
    c = if s.cacheval===nothing
        prob = IntegralProblem(s.f, s.dom, p)
        make_cache(prob, s.alg; s.kwargs...)
    else
        remake_cache(s, p)
    end
    return do_solve(c)
end

function (s::IntegralSolver)(p)
    sol = do_solve(s, p)
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
            sol = do_solve(f_, p)
            callback(f, i, Threads.atomic_add!(n, 1) + 1, p, sol, time() - t)
            out[i] = sol.u
        end
    end
    return out
end

solver_type(::F, ::P) where {F,P} = Base.promote_op((f, p) -> do_solve(f, p).u, F, P)

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
        cache = make_cache(prob, f.alg; f.kwargs...)
        IntegralSolver(f.f, f.dom, f.alg, cache.cacheval, f.kwargs)
    else
        f
    end
    return batchsolve!(similar(ps, T), solver, ps, nthreads, callback)
end

"""
    ParameterIntegrand(f, args...; kwargs...)

Represent an integrand with a partial collection of parameters `p`. When the
`ParameterIntegrand` is invoked with one argument, e.g. `int(x)`, it evaluates `f(x,
p...; kwargs...)`. However when invoked with two arguments, as in an `IntegralProblem`,
e.g. `int(x, p2)`, it evaluates the union of parameters `f(x, p..., p2...; kwargs...)`.
This allows for convenient parametrization of the integrand.
"""
struct ParameterIntegrand{F,P}
    f::F
    p::P
    function ParameterIntegrand{F}(f::F, p::P) where {F,P<:MixedParameters}
        return new{F,P}(f, p)
    end
end

function ParameterIntegrand(f, args...; kwargs...)
    p = MixedParameters(args...; kwargs...)
    return ParameterIntegrand{typeof(f)}(f, p)
end

# provide Integrals.jl interface
function (f::ParameterIntegrand)(x, q=())
    p = merge(f.p, q)
    return f.f(x, getfield(p, :args)...; getfield(p, :kwargs)...)
end

# move all parameters from f.p to p for convenience
remake_integrand_cache(args...) = IntegralCache(args...)
function remake_cache(f::ParameterIntegrand, dom, p, alg, cacheval, kwargs)
    new = ParameterIntegrand(f.f)
    return remake_integrand_cache(new, dom, merge(f.p, p), alg, cacheval, kwargs)
end

function (s::IntegralSolver{<:ParameterIntegrand})(args...; kwargs...)
    p = MixedParameters(args...; kwargs...)
    sol = do_solve(s, p)
    return sol.u
end
