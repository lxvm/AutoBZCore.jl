module AutoBZCoreRichardsonExt

using LinearAlgebra
using Richardson
using AutoBZCore

function AutoBZCore.init_cacheval(h, dom, p, alg::RationalRichardson)
    f = extrap_integrand(h, alg.m)
    prob = AutoBZCore.IntegralProblem(f, p, (dom, alg.h))
    return AutoBZCore.init(prob, alg.alg; alg.kwargs...)
end

function AutoBZCore.dos_solve(h, domain, p, alg::RationalRichardson, cacheval;
    maxiters=typemax(Int), abstol::Real=0,
    reltol::Real = abstol > zero(abstol) ? zero(one(float(real(alg.h)))) : sqrt(eps(typeof(one(float(real(alg.h)))))))
    domain isa Number || throw(ArgumentError("RationalRichardson only supports estimating the DOS at an eigenvalue"))
    E = domain

    f_ = η -> begin

        if cacheval.alg isa AutoPTR
            ptr = cacheval.alg
            cacheval.alg = AutoPTR(; norm=ptr.norm, a=oftype(ptr.a, η), nmin=ptr.nmin, nmax=ptr.nmax,
                                    n₀=ptr.n₀, Δn=ptr.Δn, keepmost=2, nthreads=ptr.nthreads)
        end

        cacheval.p = (E, η)
        return AutoBZCore.solve!(cacheval).u
    end
    f0, err = extrapolate(f_, alg.h, power=alg.power, atol=abstol, rtol=reltol,
                            maxeval=maxiters, contract=alg.contract, breaktol=alg.breaktol)
    return AutoBZCore.DOSSolution(f0, err, true, -1)
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

end
