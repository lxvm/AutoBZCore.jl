# BCD
# - number of k points
# - H_R or DFT data

# LTM
# - number of k points
# - any function or H_R

"""
    RationalRichardson(; alg=IAI(), m=2, h=1.0, power=1, contract=0.125, breaktol=2, kwargs...)

Compute a density of states extrapolation by smearing the Dirac distribution
with a rational kernel of order `m`, as presented in ["Computing Spectral
Measures of Self-Adjoint operators"](https://doi.org/10.1137/20M1330944).

This algorithm assumes that the operator `H` is a function of a continuous
crystal momentum, `H(k)`, and will use the integration algorithm `alg` to
estimate that integral. Additionally, the parameter `p` passed to `DOSProblem`
should be the domain of integration (e.g. the Brillouin zone). Finally, the
other `kwargs` passed to `RationalRichardson` will be used as the integration
solver keywords, such as `abstol`, `reltol`, and `maxiters`, which the caller
should set.

The parameters `h>0` and `power` control the initial evaluation point and the
exponent of the expansion in `h`, respectively. See
[Richardson.jl](https://github.com/JuliaMath/Richardson.jl) for more details on
these keywords as well as `contract` and `breaktol`.
"""
struct RationalRichardson{A<:IntegralAlgorithm,T<:Number,P<:Real,K,C<:Number,B<:Real} <: DOSAlgorithm
    m::Int
    h::T
    power::P
    contract::C
    breaktol::B
    alg::A
    kwargs::K
end
function RationalRichardson(; alg=IAI(), h=1.0, power=1, contract=oftype(float(real(h)), 0.125), breaktol=2, m=2, kws...)
    return RationalRichardson(m, h, power, contract, breaktol, alg, NamedTuple(kws))
end

function init_cacheval(h, dom, p, alg::RationalRichardson)
    f = extrap_integrand(h, alg.m)
    prob = IntegralProblem(f, p, (dom, alg.h))
    return init(prob, alg.alg; alg.kwargs...)
end

function dos_solve(h, domain, p, alg::RationalRichardson, cacheval;
    maxiters=typemax(Int), abstol::Real=0,
    reltol::Real = abstol > zero(abstol) ? zero(one(float(real(alg.h)))) : sqrt(eps(typeof(one(float(real(alg.h)))))))
    domain isa Number || throw(ArgumentError("RationalRichardson only supports estimating the DOS at an eigenvalue"))
    E = domain

    f_ = η -> begin
        cacheval.p = (E, η)
        return solve!(cacheval).u
    end
    f0, err = extrapolate(f_, alg.h, power=alg.power, atol=abstol, rtol=reltol,
                            maxeval=maxiters, contract=alg.contract, breaktol=alg.breaktol)
    return DOSSolution(f0, err, true, -1)
end

# assume H is a function of k
function extrap_integrand(h, m)
    return (k, (E, η)) -> _rational_dos_integrand(h(k), η, E, m)
end
function extrap_integrand(h::Union{FourierSeries,FourierWorkspace}, m)
    f = (h_k::FourierValue, m, E, η) -> _rational_dos_integrand(h_k.s, η, E, m)
    return FourierIntegrand(f, h, m)
end
function _rational_dos_integrand(h_k, η, E, m)
    a, α = cachedkernel(typeof(float(real(one(η)))), m)
    g = α[1]*inv((E+a[1]*η)*I - h_k)
    for i in 2:m
        g += α[i]*inv((E+a[i]*η)*I - h_k)
    end
    return -imag(tr(g))/pi
end

# compute the coefficients of a m-th order rational kernel (e.g. generalization of a
# Lorentzian m=1)
function equi_vandermonde(::Type{T}, m::Int) where {T}
    a = Complex{T}[complex(((i+i)/(T(m)+one(T)))-one(T),one(T)) for i in 1:m]
    V = Complex{T}[a[i]^n for n in 0:m-1, i in 1:m]
    t = zeros(Complex{T},m)
    t[1] = one(Complex{T})
    α = V \ t
    return a, α
end

const kernelcache = Dict{Type,Dict}()

@generated function _cachedkernel(::Type{T}, m::Int) where {T}
    cache = haskey(kernelcache, T) ? kernelcache[T] : (kernelcache[T] = Dict{Int,NTuple{2,Vector{Complex{T}}}}())
    return :(haskey($cache, m) ? $cache[m] : ($cache[m] = equi_vandermonde($T, m)))
end

# cache for kernel to not have to recompute it
function cachedkernel(::Type{T}, m::Integer) where {T}
    return _cachedkernel(typeof(float(real(one(T)))), Int(m))
end