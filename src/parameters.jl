"""
    MixedParameters(args::Tuple, kwargs::NamedTuple)
    MixedParameters(args...; kwargs...)

A struct to store the arguments and keyword arguments to a function. Indicies access
`args`, i.e. `MixedParameters(args...; kwargs...)[i] == args[i]` and properties access
`kwargs`, i.e. `MixedParameters(args...; kwargs...).name == kwargs.name`.

Used internally to store partially complete collections of function arguments or parameters.
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
"""
    paramzip(args...; kwargs...)

Behaves similarly to `zip(zip(args...), zip(kwargs...))` with [`MixedParameters`](@ref)
return values so that `paramzip(args...; kwargs...)[i][j] == args[j][i]` and
`paramzip(args...; kwargs...)[i].name == kwargs.name[i]`.
"""
paramzip(args...; kwargs...) = paramzip_(args, NamedTuple(kwargs))

function paramproduct_(args::Tuple, kwargs::NamedTuple)
    [MixedParameters(item[1:length(args)], NamedTuple{keys(kwargs)}(item[length(args)+1:end])) for item in Iterators.product(args..., values(kwargs)...)]
end

"""
    paramproduct(args...; kwargs...)

Behaves similarly to `product(args..., kwargs...)` with [`MixedParameters`](@ref) return
values so that `paramzip(args...; kwargs...)[i1, ...,ij, il, ...in] ==
MixedParameters(args[begin][i1], ..., args[end][ij]; kwargs[begin][il], ..., kwargs[end][in])`
"""
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
function (f::ParameterIntegrand)(x, q)
    p = merge(f.p, q)
    return f.f(x, getfield(p, :args)...; getfield(p, :kwargs)...)
end
(f::ParameterIntegrand)(x, ::NullParameters) = f(x, ())

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

function do_solve(f::ParameterIntegrand, dom, p, alg::EvalCounter, cacheval; kws...)
    n::Int = 0
    function g(x, args...; kwargs...)
        n += 1
        return f.f(x, args...; kwargs...)
    end
    sol = do_solve(ParameterIntegrand{typeof(g)}(g, f.p), dom, p, alg.alg, cacheval; kws...)
    return IntegralSolution(sol.u, sol.resid, true, n)
end

# ambiguity
function do_solve(f::ParameterIntegrand, bz::SymmetricBZ, p, alg::EvalCounter{<:AutoBZAlgorithm}, cacheval; kws...)
    return do_solve(f, bz, p, AutoBZEvalCounter(bz_to_standard(bz, alg.alg)...), cacheval; kws...)
end
