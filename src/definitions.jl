# definitions of abstractions of integration routines

"""
    AbstractIntegrand{F} <: Function

Supertype for classes of integrands with a field `F` to specialize on the
integrand evaluator. `T<:AbstractIntegrand` must to implement a constructor
`T{F}`.
"""
abstract type AbstractIntegrand{F} <: Function end

struct QuadIntegrand{F,P,K} <: AbstractIntegrand{F}
    f::F
    p::P
    k::K
end
QuadIntegrand{F}(p...; kwargs...) where {F<:Function} =
    QuadIntegrand(F.instance, p, kwargs)
(f::QuadIntegrand)(x) = f.f(x, f.p...; f.k...)


# Main type

"""
    Integrator{T}(routine, l, f, p...; inargs=(;), kwargs...) where {T<:AbstractIntegrand}

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
struct Integrator{T,F,L,P,R,A,K} <: Function
    routine::R
    l::L
    f::F
    p::P
    inargs::A
    kwargs::K
    Integrator{T}(r::R, l::L, f::F, p::P, a::A, k::K) where {T<:AbstractIntegrand,F,L,P,R,A,K} =
        new{T,F,L,P,R,A,K}(r, l, f, p, a, k)
end

function Integrator{T}(routine, l, f::F, p...; ps=nothing, inargs=(;), kwargs...) where {T<:AbstractIntegrand,F}
    if !isnothing(ps)
        # attempt to construct kwargs from a test integrand
        test = T{F}(p..., ps...; inargs...)
        kwargs = (; kwargs..., quad_kwargs(routine, l, test; kwargs...)...)
    end
    Integrator{T}(routine, l, f, p, inargs, kwargs)
end


function (f::Integrator{T,F})(ps...; inargs=(;), kwargs...) where {T,F}
    integrand = T{F}(f.p..., ps...; f.inargs..., inargs...) # allows for dispatch based on type aliases
    f.routine(quad_args(f.routine, f.l, integrand)...; f.kwargs..., kwargs...)
end

Base.nameof(::U) where {T,F,U<:Integrator{T,F}} = T{F} # todo: display aliases

Base.show(io::IO, ::MIME"text/plain", f::Integrator) = show(io, f)
Base.show(io::IO, f::Integrator) =
    print(io, nameof(f), " using routine ", nameof(f.routine))

"""
    quad_args(routine, l, f)

Return the tuple of arguments needed by the quadrature `routine` depending on
the limits `l` and integrand `f`.
"""
function quad_args end

"""
    quad_kwargs(routine, l, f; kwargs)

Return 
"""
function quad_kwargs end

