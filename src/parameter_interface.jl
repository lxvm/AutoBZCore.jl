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

evaluate_integrand(f, x, p::MixedParameters) = f(x, p.args...; p.kwargs...)

Base.getindex(p::MixedParameters, i::Int) = getindex(p.args, i)

Base.merge(p::MixedParameters, ::NullParameters) = p
Base.merge(p::MixedParameters, q) = MixedParameters((p.args..., q), p.kwargs)
Base.merge(p::MixedParameters, q::Tuple) = MixedParameters((p.args..., q...), p.kwargs)
Base.merge(p::MixedParameters, q::MixedParameters) =
    MixedParameters((p.args..., q.args...), merge(p.kwargs, q.kwargs))


function paramzip(::Tuple{}, ::NamedTuple{(), Tuple{}})
    MixedParameters{Tuple{},NamedTuple{(), Tuple{}}}[]
end
function paramzip(args::Tuple, ::NamedTuple{(), Tuple{}})
    [MixedParameters(arg, NamedTuple()) for arg in zip(args...)]
end
function paramzip(::Tuple{}, kwargs::NamedTuple)
    [MixedParameters((), NamedTuple{keys(kwargs)}(val)) for val in zip(values(kwargs)...)]
end
function paramzip(args::Tuple, kwargs::NamedTuple)
    [MixedParameters(arg, NamedTuple{keys(kwargs)}(val)) for (arg, val) in zip(zip(args...), zip(values(kwargs)...))]
end


# provide our own parameter interface for our integrands
abstract type AbstractAutoBZIntegrand end
const AutoBZIntegralSolver = IntegralSolver{<:Any,<:AbstractAutoBZIntegrand}


(s::AutoBZIntegralSolver)(args...; kwargs...) =
    do_solve(s, MixedParameters(args...; kwargs...)).u

"""
    batchsolve(T::Type, f::IntegralSolver, args...; nthreads=Threads.nthreads(), kwargs...)

Evaluate the [`IntegralSolver`](@ref) `f` at each of the parameters `ps` in
parallel. Returns a vector containing the evaluated integrals `I`. This is
a form of multithreaded broadcasting.
"""
function batchsolve(T::Type, f::AutoBZIntegralSolver, args...; itr=paramzip, nthreads=Threads.nthreads(), callback=(x...)->nothing, kwargs...)
    ps = itr(args, NamedTuple(kwargs))
    batchsolve(f, ps, T; nthreads=nthreads, callback=callback)
end
