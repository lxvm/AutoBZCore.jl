# Create a type for the convenience of hacking Integrals.jl
"""
    Integrand(f, p...)

Represent an integrand with a partial collection of parameters `p`. When the
`Integrand` is invoked with one argument, e.g. `int(x)`, it evaluates `f(x,
p...)`. However when invoked with two arguments, as in an `IntegralProblem`,
e.g. `int(x, p2)`, it evaluates the union of parameters `f(x, p..., p2...)`.
This allows for convenient parametrization of the integrand.
"""
struct Integrand{F,P<:Tuple}
    f::F
    p::P
end
Integrand(f, p...) = Integrand(f, p)

# provide Integrals.jl interface while still using functor interface
(f::Integrand)(x, p) = Integrand(f.f, (f.p..., p))(x)
(f::Integrand)(x, p::Tuple) = Integrand(f.f, (f.p..., p...))(x)
(f::Integrand)(x, ::NullParameters) = f(x)

(f::Integrand)(x) = f.f(x, f.p...)

