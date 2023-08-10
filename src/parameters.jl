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
    sol = solve_p(s, p)
    return sol.u
end
