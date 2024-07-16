# We could move these into an extension, although QuadratureFunction also uses AutoSymPTR.jl
# for evaluation


"""
    MonkhorstPack(; npt=50, syms=nothing, nthreads=1)

Periodic trapezoidal rule with a fixed number of k-points per dimension, `npt`,
using the `PTR` rule from [AutoSymPTR.jl](https://github.com/lxvm/AutoSymPTR.jl).
`nthreads` sets the numbers of threads used to parallelize the quadrature only when the
integrand is a , in which case the user must parallelize the
integrand evaluations. For no threading set `nthreads=1`.
**The caller should check that the integral is converged w.r.t. `npt`**.
"""
struct MonkhorstPack{S} <: IntegralAlgorithm
    npt::Int
    syms::S
    nthreads::Int
end
MonkhorstPack(; npt=50, syms=nothing, nthreads=1) = MonkhorstPack(npt, syms, nthreads)
function init_rule(dom, alg::MonkhorstPack)
    # rule = AutoSymPTR.MonkhorstPackRule(alg.syms, alg.a, alg.nmin, alg.nmax, alg.n₀, alg.Δn)
    # return rule(eltype(dom), Val(ndims(dom)))
    if alg.syms === nothing
        return AutoSymPTR.PTR(eltype(dom), Val(ndims(dom)), alg.npt)
    else
        return AutoSymPTR.MonkhorstPack(eltype(dom), Val(ndims(dom)), alg.npt, alg.syms)
    end
end

function init_cacheval(f, dom, p, alg::MonkhorstPack; kws...)
    b = get_basis(dom)
    rule = init_rule(b, alg)
    cache = init_autosymptr_cache(f, b, p, alg.nthreads; kws...)
    return (; rule, cache...)
end
function do_integral(f, dom, p, alg::MonkhorstPack, cacheval;
                    reltol = nothing, abstol = nothing, maxiters = typemax(Int))
    b = get_basis(dom)
    g = autosymptr_integrand(f, p, b, cacheval)
    value = cacheval.rule(g, b, cacheval.buffer)
    retcode = Success
    stats = (; numevals=length(cacheval.rule))
    return IntegralSolution(value, retcode, stats)
end

"""
    AutoSymPTRJL(; norm=norm, a=1.0, nmin=50, nmax=1000, n₀=6, Δn=log(10), keepmost=2, nthreads=1)

Periodic trapezoidal rule with automatic convergence to tolerances passed to the
solver with respect to `norm` using the routine `autosymptr` from
[AutoSymPTR.jl](https://github.com/lxvm/AutoSymPTR.jl).
`nthreads` sets the numbers of threads used to parallelize the quadrature only when the
integrand is a  in which case the user must parallelize the
integrand evaluations. For no threading set `nthreads=1`.
**This algorithm is the most efficient for smooth integrands**.
"""
struct AutoSymPTRJL{F,S} <: IntegralAlgorithm
    norm::F
    a::Float64
    nmin::Int
    nmax::Int
    n₀::Float64
    Δn::Float64
    keepmost::Int
    syms::S
    nthreads::Int
end
function AutoSymPTRJL(; norm=norm, a=1.0, nmin=50, nmax=1000, n₀=6.0, Δn=log(10), keepmost=2, syms=nothing, nthreads=1)
    return AutoSymPTRJL(norm, a, nmin, nmax, n₀, Δn, keepmost, syms, nthreads)
end
function init_rule(dom, alg::AutoSymPTRJL)
    return AutoSymPTR.MonkhorstPackRule(alg.syms, alg.a, alg.nmin, alg.nmax, alg.n₀, alg.Δn)
end


function init_cacheval(f, dom, p, alg::AutoSymPTRJL; kws...)
    b = get_basis(dom)
    rule = init_rule(dom, alg)
    rule_cache = AutoSymPTR.alloc_cache(eltype(dom), Val(ndims(dom)), rule)
    cache = init_autosymptr_cache(f, b, p, alg.nthreads; kws...)
    return (; rule, rule_cache, cache...)
end

function do_integral(f, dom, p, alg::AutoSymPTRJL, cacheval;
                    reltol = nothing, abstol = nothing, maxiters = typemax(Int))

    g = autosymptr_integrand(f, p, dom, cacheval)
    bas = get_basis(dom)
    value, error = autosymptr(g, bas; syms = alg.syms, rule = cacheval.rule, cache = cacheval.rule_cache, keepmost = alg.keepmost,
        abstol = abstol, reltol = reltol, maxevals = maxiters, norm=alg.norm, buffer=cacheval.buffer)
    retcode = error < max(something(abstol, zero(error)), alg.norm(value)*something(reltol, isnothing(abstol) ? sqrt(eps(eltype(bas))) : abstol)) ? Success : Failure
    stats = (; error)
    return IntegralSolution(value, retcode, stats)
end
