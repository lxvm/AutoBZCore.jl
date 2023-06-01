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

# we only merge non-MixedParameters in the right argument
Base.merge(p::MixedParameters, ::NullParameters) = p
Base.merge(p::MixedParameters, q) =
    MixedParameters((getfield(p, :args)..., q), getfield(p, :kwargs))
Base.merge(p::MixedParameters, q::NamedTuple) =
    MixedParameters(getfield(p, :args), merge(getfield(p, :kwargs), q))
Base.merge(p::MixedParameters, q::Tuple) =
    MixedParameters((getfield(p, :args)..., q...), getfield(p, :kwargs))
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
    IntegralSolver(f, lb, ub, alg; abstol=0, reltol=sqrt(eps()), maxiters=typemax(Int))
    IntegralSolver(f, bz::SymmetricBZ, alg::AbstractAutoBZAlgorithm; kwargs...)

Constructs a functor that solves an integral of `f` over the given domain (e.g.
`lb` to `ub` or a `bz`) using the given `alg` to within the given tolerances.
Calling this functor, `fun` with parameters `p` using the syntax `fun(p)`
returns the estimated integral `I`. Under the hood, this uses the [Integrals.jl
interface](https://docs.sciml.ai/Integrals/stable/) for defining an
`IntegralProblem`, so `f` must be a 2-argument function `f(x,p)`, or if
in-place, a 3-argument function `f(y,x,p)`.

Also, the types [`Integrand`](@ref) and [`FourierIntegrand`](@ref) allow for
providing a partial set of parameters so that the `IntegralSolver` can interface
easily with other algorithms, such as root-finding and interpolation.
"""
struct IntegralSolver{iip,F,B,A,S,D,AT,RT,K}
    f::F
    lb::B
    ub::B
    alg::A
    sensealg::S
    do_inf_transformation::D
    abstol::AT
    reltol::RT
    maxiters::Int
    kwargs::K
    function IntegralSolver{iip}(f, lb, ub, alg;
                                sensealg = ReCallVJP(ZygoteVJP()),
                                do_inf_transformation = nothing,
                                abstol=0.0, reltol=iszero(abstol) ? sqrt(eps()) : zero(abstol),
                                maxiters=typemax(Int), kwargs...) where iip
        @assert typeof(lb)==typeof(ub) "Type of lower and upper bound must match"
        new{iip, typeof(f), typeof(lb), typeof(alg), typeof(sensealg), typeof(do_inf_transformation),
            typeof(abstol), typeof(reltol), typeof(kwargs)}(f, lb, ub, alg, sensealg, do_inf_transformation, abstol, reltol, maxiters, kwargs)
    end
end

function IntegralSolver(f, lb, ub, alg; kwargs...)
    iip = isinplace(f, 3)
    IntegralSolver{iip}(f, lb, ub, alg; kwargs...)
end

function construct_problem(s::IntegralSolver{iip}, p) where iip
    IntegralProblem{iip}(s.f, s.lb, s.ub, p; s.kwargs...)
end

function do_solve(s::S, p::P) where {S<:IntegralSolver,P}
    prob = construct_problem(s, p)
    return solve(prob, s.alg,
                    abstol = s.abstol, reltol = s.reltol, maxiters = s.maxiters,
                    do_inf_transformation=s.do_inf_transformation, sensealg=s.sensealg)
end

# provide plain SciML interface
function (s::IntegralSolver)(p=NullParameters())
    sol = do_solve(s, p)
    return sol.u
end

# imitate general interface
IntegralSolver(f, bz::SymmetricBZ, alg::AbstractAutoBZAlgorithm; kwargs...) =
    IntegralSolver{isinplace(f, 3)}(f, bz, bz, alg; do_inf_transformation=Val(false), kwargs...)

# parallelization

"""
    batchparam(ps, nthreads)

If the cost of a calculation smoothly varies with the parameters `ps`, then
batch `ps` into `nthreads` groups where the `i`th element of group `j` is
`ps[j+(i-1)*nthreads]`
"""
function batchparam(xs, nthreads)
    batches = [Tuple{Int,eltype(xs)}[] for _ in 1:min(nthreads, length(xs))]
    for (i, x) in pairs(IndexStyle(xs), xs)
        push!(batches[mod(i-1, nthreads)+1], (i, x))
    end
    batches
end

function batcheval(i, p, f, callback)
    t = time()
    sol = do_solve(f, p)
    t = time() - t
    callback(f, i, p, sol, t)
    sol.u
end


function batchsolve!(out, f, ps, nthreads, callback)
    Threads.@threads for batch in batchparam(ps, nthreads)
        f_ = Threads.threadid() == 1 ? f : deepcopy(f) # avoid data races for in place integrators
        for (i, p) in batch
            out[i] = batcheval(i, p, f_, callback)
        end
    end
    out
end

solver_type(F, P) = Base.promote_op((f,p) -> do_solve(f,p).u, F, P)

"""
    batchsolve(f::IntegralSolver, ps::AbstractArray, [T]; nthreads=Threads.nthreads())

Evaluate the [`IntegralSolver`](@ref) `f` at each of the parameters `ps` in
parallel. Returns an array similar to `ps` containing the evaluated integrals `I`. This is
a form of multithreaded broadcasting. Providing the return type `f(eltype(ps))::T` is
optional, but will help in case inference of that type fails.
"""
function batchsolve(f::IntegralSolver, ps::AbstractArray, T=solver_type(typeof(f), eltype(ps)); nthreads=Threads.nthreads(), callback=(x...)->nothing)
    batchsolve!(similar(ps, T), f, ps, nthreads, callback)
end


# provide our own parameter interface for our integrands
abstract type AbstractAutoBZIntegrand{F} end
# Create a type for the convenience of hacking Integrals.jl
const AutoBZIntegralSolver{iip} = IntegralSolver{iip,<:AbstractAutoBZIntegrand}

function (s::AutoBZIntegralSolver)(args...; kwargs...)
    p = MixedParameters(args...; kwargs...)
    sol = do_solve(s, p)
    return sol.u
end

"""
    Integrand(f, args...; kwargs...)

Represent an integrand with a partial collection of parameters `p`. When the
`Integrand` is invoked with one argument, e.g. `int(x)`, it evaluates `f(x,
p...; kwargs...)`. However when invoked with two arguments, as in an `IntegralProblem`,
e.g. `int(x, p2)`, it evaluates the union of parameters `f(x, p..., p2...; kwargs...)`.
This allows for convenient parametrization of the integrand.
"""
struct Integrand{F,P<:MixedParameters} <: AbstractAutoBZIntegrand{F}
    f::F
    p::P
end

function Integrand(f, args...; kwargs...)
    p = MixedParameters(args...; kwargs...)
    return Integrand(f, p)
end

# provide Integrals.jl interface while still using functor interface
(f::Integrand)(x, p) = Integrand(f.f, merge(f.p, p))(x)

function evaluate_integrand(f, x, p::MixedParameters)
    return f(x, getfield(p, :args)...; getfield(p, :kwargs)...)
end
evaluate_integrand(f::Integrand, x) = evaluate_integrand(f.f, x, f.p)

(f::Integrand)(x) = evaluate_integrand(f, x)

construct_integrand(f::Integrand, iip, p) = construct_autobz_integrand(f.f, merge(f.p, p))
construct_autobz_integrand(f, p) = Integrand(f, p)
