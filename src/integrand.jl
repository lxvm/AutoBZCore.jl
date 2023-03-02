# Create a type for the convenience of hacking Integrals.jl
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

