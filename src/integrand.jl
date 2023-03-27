# Create a type for the convenience of hacking Integrals.jl
"""
    Integrand(f, args...; kwargs...)

Represent an integrand with a partial collection of parameters `p`. When the
`Integrand` is invoked with one argument, e.g. `int(x)`, it evaluates `f(x,
p...; kwargs...)`. However when invoked with two arguments, as in an `IntegralProblem`,
e.g. `int(x, p2)`, it evaluates the union of parameters `f(x, p..., p2...; kwargs...)`.
This allows for convenient parametrization of the integrand.
"""
struct Integrand{F,P<:MixedParameters} <: AbstractAutoBZIntegrand
    f::F
    p::P
end
Integrand(f, args...; kwargs...) =
    Integrand(f, MixedParameters(args...; kwargs...))

# provide Integrals.jl interface while still using functor interface
(f::Integrand)(x, p) = Integrand(f.f, merge(f.p, p))(x)

evaluate_integrand(f::Integrand, x) = evaluate_integrand(f.f, x, f.p)

(f::Integrand)(x) = evaluate_integrand(f, x)

