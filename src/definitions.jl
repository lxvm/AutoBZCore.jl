# definitions of abstractions of integration routines

"""
    AbstractIntegrand{F} <: Function

Supertype for classes of integrands with a field `F` to specialize on the
integrand evaluator. `T<:AbstractIntegrand` must to implement a constructor
`T{F}`.
"""
abstract type AbstractIntegrand{F} <: Function end

"""
    QuadIntegrand(f, params, kwargs)
    QuadIntegrand{F}(params...; kwargs...) where F

Constructs an integrand `g` such that `g(x) = f(x, params...; kwargs...)`
"""
struct QuadIntegrand{F,P,K} <: AbstractIntegrand{F}
    f::F
    p::P
    k::K
end
QuadIntegrand{F}(params...; kwargs...) where {F<:Function} =
    QuadIntegrand(F.instance, params, kwargs)

(f::QuadIntegrand)(x) = f.f(x, f.p...; f.k...)

# Main type

"""
    Integrator{T}(routine, f, l, args...; callargs=nothing, inargs=(;), kwargs...) where {T<:AbstractIntegrand}

Composite type that stores limits of integration, `l`, and parts of an
[`AbstractIntegrand`](@ref) of type `T`, including a user-provided integrand
evaluator `f` and additional parameters for the integrand `p`. Uses the
integration `routine` specified by the caller.

This type provides a functor interface to various integration algorithms so that
an `f::Integrator{T,F}` called with this syntax `f(ps...; kwargs...)` can
construct the integrand with this syntax `T{F}(f.p...; ps...)`, where the
constructor `T{F}` is usually a type alias. Then the integration routine is
called `routine(quad_args(routine, integrand, f.l)...; f.kwargs..., kwargs...)`
and its output can be stored in the integrator (see details below).

Integration algorithms that implement the `Integrator` interface must extend the
following methods for a given `routine`:

`quad_args(routine, ::AbstractIntegrand, l)`: return a tuple of the arguments
that should be splatted into the integration routine

Creating a type alias for `Integrator{MyIntegrand}` is recommended, either to
use the default methods or to provide an alternative constructor.

The keyword `inargs` stores arguments that should be passed to the integrand
constructor, instead of `kwargs` passed to the routine.
"""
struct Integrator{T,F,L,P,A,R,K} <: Function
    f::F # integrand
    l::L # limits
    p::P # integrand parameters
    a::A # integrand keywords
    r::R # routine
    k::K # routine keywords
    Integrator{T}(f::F, l::L, p::P, a::A, r::R, k::K) where {T<:AbstractIntegrand,F,L,P,A,R,K} =
        new{T,F,L,P,A,R,K}(f, l, p, a, r, k)
end

function Integrator{T}(routine, f::F, l, params...; callargs=nothing, inargs=(;), kwargs...) where {T<:AbstractIntegrand,F}
    if !isnothing(callargs)
        # attempt to construct kwargs from a test integrand
        test = T{F}(params..., callargs...; inargs...)
        kwargs = (; kwargs..., quad_kwargs(routine, test, l; kwargs...)...)
    end
    Integrator{T}(f, l, params, inargs, routine, kwargs)
end


function (f::Integrator{T,F})(args...; inargs=(;), kwargs...) where {T,F}
    integrand = T{F}(f.p..., args...; f.a..., inargs...) # allows for dispatch based on type aliases
    f.r(quad_args(f.r, integrand, f.l)...; f.k..., kwargs...)
end

Base.nameof(::U) where {T,F,U<:Integrator{T,F}} = Symbol(T, nameof(F)) # todo: display aliases

Base.show(io::IO, ::MIME"text/plain", f::Integrator) = show(io, f)
Base.show(io::IO, f::Integrator) =
    print(io, nameof(f), " using routine ", nameof(f.r))

"""
    quad_args(routine, f, l)

Return the tuple of arguments needed by the quadrature `routine` depending on
the limits `l` and integrand `f`.
"""
function quad_args end

"""
    quad_kwargs(routine, f, l; kwargs)

Return the named tuple of keyword arguments needed
"""
function quad_kwargs end


"""
    QuadIntegrator

Alias for `Integrator{QuadIntegrand}`
"""
const QuadIntegrator = Integrator{QuadIntegrand}

"""
    QuadGKIntegrator(args...; kwargs...)

Alias for [`QuadIntegrator`](@ref) with the routine set to `quadgk`.
"""
QuadGKIntegrator(args...; kwargs...) =
    QuadIntegrator(quadgk, args...; kwargs...)

quad_args(::typeof(quadgk), f, segs::NTuple{N,T}) where {N,T} = (f, segs...)
function quad_kwargs(::typeof(quadgk), f, segs::NTuple{N,T};
    atol=zero(T), rtol=iszero(atol) ? sqrt(eps(T)) : zero(atol),
    order=7, maxevals=10^7, norm=norm, segbuf=nothing) where {N,T}
    F = Base.promote_op(f, T)
    segbuf_ = segbuf === nothing ? alloc_segbuf(T, F, Base.promote_op(norm, F)) : segbuf
    (rtol=rtol, atol=atol, order=order, maxevals=maxevals, norm=norm, segbuf=segbuf_)
end
