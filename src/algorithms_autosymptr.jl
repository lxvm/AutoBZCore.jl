# We could move these into an extension, although QuadratureFunction also uses AutoSymPTR.jl
# for evaluation

# high-level wrapper of AutoSymPTR.AffineQuad for use by multiple algorithms
struct AffineQuad{R} <: IntegralAlgorithm
    rule::R
end

function init_rule(dom, alg::AffineQuad)
    return alg.rule
end

function init_cacheval(f::AbstractIntegralFunction, dom, p, alg::AffineQuad; kws...)
    rule = init_rule(dom, alg)
    prototype, integrand_cacheval = init_integrand_cacheval(f, dom, p)
    algorithm_cacheval = if prototype isa BatchArray
        (data = prototype.data) isa AbstractVector || throw(ArgumentError("AutoSymPTR.jl does not support batched functions with multidimensional outputs"))
        bufsize = 0 # a buffer of size zero will be filled with the default number of threads
        x0 = get_prototype(dom) # the number of threads should be chosen to prevent false sharing
        (; buffer=similar(data, bufsize), y=similar(data, bufsize), x=Vector{typeof(x0)}(undef, bufsize))
    elseif prototype isa InplaceArray
        x = get_prototype(dom)
        ytmp = similar(prototype.data)
        I = ytmp * prod(x)
        Itmp = similar(I)
        (; I, Itmp, ytmp, buffer=nothing)
    else
        (; buffer=nothing)
    end
    return (; rule, algorithm_cacheval, integrand_cacheval)
end

# TODO define do_integral for AffineQuad

autosymptr_integrand(f::IntegralFunction, p, segs, alg_cache, cacheval) = autosymptr_integrand_if(f.executor, f, p, segs, alg_cache, cacheval)
function autosymptr_integrand_if(::SerialExecutor, f::IntegralFunction, p, segs, alg_cache, cacheval)
    x -> f.f(x, p)
end
function autosymptr_integrand_if(exec::ThreadedExecutor, f::IntegralFunction, p, segs, alg_cache, cacheval)
    func, proto, cache = cacheval
    return autosymptr_integrand(func, p, segs, alg_cache, cache)
end
function autosymptr_integrand(f::InplaceIntegralFunction, p, segs, alg_cache, cacheval)
    AutoSymPTR.InplaceIntegrand((y,x) -> f.f!(y,x,p), alg_cache.I, alg_cache.Itmp, cacheval, alg_cache.ytmp)
end
function autosymptr_integrand(f::InplaceBatchIntegralFunction, p, segs, alg_cache, cacheval)
    AutoSymPTR.BatchIntegrand((y,x) -> f.f!(y,x,p), cacheval, alg_cache.x, max_batch=f.max_batch)
end
autosymptr_integrand(f::CommonSolveIntegralFunction, p, segs, alg_cache, cacheval) = autosymptr_integrand_cs(f.executor, f, p, segs, alg_cache, cacheval)
function autosymptr_integrand_cs(::SerialExecutor, f, p, segs, alg_cache, cacheval)
    solver, integrand = cacheval
    x -> integrand(solver, f, x, p)
end
function autosymptr_integrand_cs(exec::ThreadedExecutor, f, p, segs, alg_cache, cacheval)
    func, channel, integrand, proto, cache = cacheval
    return autosymptr_integrand(func, p, segs, alg_cache, cache)
end

"""
    MonkhorstPack(; npt=50, syms=nothing)

Periodic trapezoidal rule with a fixed number of k-points per dimension, `npt`,
using the `PTR` rule from [AutoSymPTR.jl](https://github.com/lxvm/AutoSymPTR.jl).
**The caller should check that the integral is converged w.r.t. `npt`**.
"""
struct MonkhorstPack{S} <: IntegralAlgorithm
    npt::Int
    syms::S
end
MonkhorstPack(; npt=50, syms=nothing) = MonkhorstPack(npt, syms)
function init_rule(dom, alg::MonkhorstPack)
    # rule = AutoSymPTR.MonkhorstPackRule(alg.syms, alg.a, alg.nmin, alg.nmax, alg.n₀, alg.Δn)
    # return rule(eltype(dom), Val(ndims(dom)))
    if alg.syms === nothing
        return AutoSymPTR.PTR(eltype(dom), Val(ndims(dom)), alg.npt)
    else
        return AutoSymPTR.MonkhorstPack(eltype(dom), Val(ndims(dom)), alg.npt, alg.syms)
    end
end

function init_cacheval(f::AbstractIntegralFunction, dom, p, alg::MonkhorstPack; kws...)
    b = get_basis(dom)
    rule = init_rule(dom, alg)
    return init_cacheval(f, dom, p, AffineQuad(rule); kws...)
end

function do_integral(f, dom, p, alg::MonkhorstPack, cacheval;
                    reltol = nothing, abstol = nothing, maxiters = typemax(Int))
    b = get_basis(dom)
    g = autosymptr_integrand(f, p, b, cacheval.algorithm_cacheval, cacheval.integrand_cacheval)
    value = cacheval.rule(g, b, cacheval.algorithm_cacheval.buffer)
    retcode = Success
    stats = (; numevals=length(cacheval.rule))
    return IntegralSolution(value, retcode, stats)
end

"""
    AutoSymPTRJL(; norm=norm, a=1.0, nmin=50, nmax=1000, n₀=6, Δn=log(10), keepmost=2)

Periodic trapezoidal rule with automatic convergence to tolerances passed to the
solver with respect to `norm` using the routine `autosymptr` from
[AutoSymPTR.jl](https://github.com/lxvm/AutoSymPTR.jl).
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
end
function AutoSymPTRJL(; norm=norm, a=1.0, nmin=50, nmax=1000, n₀=6.0, Δn=log(10), keepmost=2, syms=nothing)
    return AutoSymPTRJL(norm, a, nmin, nmax, n₀, Δn, keepmost, syms)
end
function init_rule(dom, alg::AutoSymPTRJL)
    return AutoSymPTR.MonkhorstPackRule(alg.syms, alg.a, alg.nmin, alg.nmax, alg.n₀, alg.Δn)
end


function init_cacheval(f::AbstractIntegralFunction, dom, p, alg::AutoSymPTRJL; kws...)
    b = get_basis(dom)
    rule = init_rule(dom, alg)
    cache = init_cacheval(f, dom, p, AffineQuad(rule); kws...)
    rule_cache = AutoSymPTR.alloc_cache(eltype(dom), Val(ndims(dom)), rule)
    return (; rule_cache, cache...)
end

function do_integral(f, dom, p, alg::AutoSymPTRJL, cacheval;
                    reltol = nothing, abstol = nothing, maxiters = typemax(Int))

    g = autosymptr_integrand(f, p, dom, cacheval.algorithm_cacheval, cacheval.integrand_cacheval)
    bas = get_basis(dom)
    value, error = autosymptr(g, bas; syms = alg.syms, rule = cacheval.rule, cache = cacheval.rule_cache, keepmost = alg.keepmost,
        abstol = abstol, reltol = reltol, maxevals = maxiters, norm=alg.norm, buffer=cacheval.algorithm_cacheval.buffer)
    retcode = error < max(something(abstol, zero(error)), alg.norm(value)*something(reltol, isnothing(abstol) ? sqrt(eps(eltype(bas))) : abstol)) ? Success : Failure
    stats = (; error)
    return IntegralSolution(value, retcode, stats)
end
