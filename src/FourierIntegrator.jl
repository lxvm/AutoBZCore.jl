"""
    FourierIntegrator(f, bz, s, p...; ps=0.0, routine=iterated_integration, args=(), kwargs=(;))
    FourierIntegrator{F}(args...; kwargs...) where {F<:Function}

Composite type that stores parts of a [`AutoBZ.FourierIntegrand`](@ref),
including the user-defined integrand `f`, the [`AutoBZ.AbstractBZ`] `bz`,
a [`AutoBZ.AbstractFourierSeries`] `s`, and additional parameters for the
integrand `ps`. Uses the integration routine specified by the caller.
Please provide a sample input `ps` (possibly a tuple of inputs) to the
constructor so that it can infer the type of outcome.

DEV notes: When called as a functor, this integrator concatenates `ps` followed
by the input `xs...` to the functor call to construct the integrand like so
`FourierIntegrand{typeof(f)}(s, ps..., xs...)`, which means that the arguments
passed to `f` which will be passed to the integrator must go in the last
positions. This mostly applies to aliases that may have specialized behavior
while also having an interface compatible with other routines (e.g. interpolation).
"""
struct FourierIntegrator{F,S<:AbstractFourierSeries,P<:Tuple,BZ<:AbstractBZ,R,K<:NamedTuple} <: AbstractIntegrator{F}
    f::F
    bz::BZ
    s::S
    p::P
    routine::R
    kwargs::K
end

function FourierIntegrator(f::F, bz, s, p...; ps=0.0, routine=iterated_integration, kwargs...) where F
    check_period_match(s, bz)
    test = FourierIntegrand{F}(s, p..., ps...)
    FourierIntegrator(f, bz, s, p, routine, quad_kwargs(routine, test, bz; kwargs...))
end

FourierIntegrator{F}(args...; kwargs...) where {F<:Function} =
    FourierIntegrator(F.instance, args...; kwargs...)
FourierIntegrator{F}(args...; kwargs...) where {F<:Tuple{Vararg{Function}}} =
    FourierIntegrator(tuple(map(f -> f.instance, F.parameters)...), args...; kwargs...)

# methods

limits(f::FourierIntegrator) = (f.bz,)

quad_integrand(f::FourierIntegrator{F}, ps...) where {F<:Function} =
    FourierIntegrand{F}(f.s, f.p..., ps...)
quad_integrand(f::FourierIntegrator{F}, ps...) where {F<:Tuple} =
    IteratedFourierIntegrand{F}(f.s, f.p..., ps...)

quad_kwargs(::typeof(iterated_integration), f, bz::AbstractBZ; kwargs...) =
    iterated_integration_kwargs(f, limits(bz); kwargs...)
quad_kwargs(::typeof(autosymptr), f, bz; kwargs...) =
    autosymptr_kwargs(f, bz; kwargs...)
quad_kwargs(::typeof(symptr), f, bz; kwargs...) =
    symptr_kwargs(f, bz; kwargs...)