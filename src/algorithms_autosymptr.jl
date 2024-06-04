# We could move these into an extension, although QuadratureFunction also uses AutoSymPTR.jl
# for evaluation


"""
    MonkhorstPack(; npt=50, syms=nothing, nthreads=1)

Periodic trapezoidal rule with a fixed number of k-points per dimension, `npt`,
using the `PTR` rule from [AutoSymPTR.jl](https://github.com/lxvm/AutoSymPTR.jl).
`nthreads` sets the numbers of threads used to parallelize the quadrature only when the
integrand is a [`BatchIntegrand`](@ref), in which case the user must parallelize the
integrand evaluations. For no threading set `nthreads=1`.
**The caller should check that the integral is converged w.r.t. `npt`**.
"""
struct MonkhorstPack{S} <: IntegralAlgorithm
    npt::Int
    syms::S
    nthreads::Int
end
MonkhorstPack(; npt=50, syms=nothing, nthreads=1) = MonkhorstPack(npt, syms, nthreads)
function init_rule(dom::Basis, alg::MonkhorstPack)
    # rule = AutoSymPTR.MonkhorstPackRule(alg.syms, alg.a, alg.nmin, alg.nmax, alg.n₀, alg.Δn)
    # return rule(eltype(dom), Val(ndims(dom)))
    if alg.syms === nothing
        return AutoSymPTR.PTR(eltype(dom), Val(ndims(dom)), alg.npt)
    else
        return AutoSymPTR.MonkhorstPack(eltype(dom), Val(ndims(dom)), alg.npt, alg.syms)
    end
end

rule_type(::AutoSymPTR.PTR{N,T}) where {N,T} = SVector{N,T}
rule_type(::AutoSymPTR.MonkhorstPack{N,T}) where {N,T} = SVector{N,T}

function init_cacheval(f, dom::Basis, p, alg::MonkhorstPack)
    f isa NestedBatchIntegrand && throw(ArgumentError("MonkhorstPack doesn't support nested batching"))
    rule = init_rule(dom, alg)
    buf = init_buffer(f, alg.nthreads)
    return (rule=rule, buffer=buf)
end

function do_solve(f, dom, p, alg::MonkhorstPack, cacheval;
                    reltol = nothing, abstol = nothing, maxiters = typemax(Int))
    g = if f isa BatchIntegrand
        xx = eltype(f.x) === Nothing ? typeof(dom*zero(rule_type(cacheval.rule)))[] : f.x
        AutoSymPTR.BatchIntegrand((y,x) -> f.f!(y,x,p), f.y, xx, max_batch=f.max_batch)
    elseif f isa InplaceIntegrand
        AutoSymPTR.InplaceIntegrand((y,x) -> f.f!(y,x,p), f.I)
    else
        x -> f(x, p)
    end
    I = cacheval.rule(g, dom, cacheval.buffer)
    return IntegralSolution(I, nothing, true, -1)
end

"""
    AutoSymPTRJL(; norm=norm, a=1.0, nmin=50, nmax=1000, n₀=6, Δn=log(10), keepmost=2, nthreads=1)

Periodic trapezoidal rule with automatic convergence to tolerances passed to the
solver with respect to `norm` using the routine `autosymptr` from
[AutoSymPTR.jl](https://github.com/lxvm/AutoSymPTR.jl).
`nthreads` sets the numbers of threads used to parallelize the quadrature only when the
integrand is a [`BatchIntegrand`](@ref), in which case the user must parallelize the
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
function init_rule(dom::Basis, alg::AutoSymPTRJL)
    return AutoSymPTR.MonkhorstPackRule(alg.syms, alg.a, alg.nmin, alg.nmax, alg.n₀, alg.Δn)
end
function init_cacheval(f, dom::Basis, p, alg::AutoSymPTRJL)
    f isa NestedBatchIntegrand && throw(ArgumentError("AutoSymPTRJL doesn't support nested batching"))
    rule = init_rule(dom, alg)
    cache = AutoSymPTR.alloc_cache(eltype(dom), Val(ndims(dom)), rule)
    buffer = init_buffer(f, alg.nthreads)
    return (rule=rule, cache=cache, buffer=buffer)
end

function do_solve(f, dom, p, alg::AutoSymPTRJL, cacheval;
                    reltol = nothing, abstol = nothing, maxiters = typemax(Int))

    g = if f isa BatchIntegrand
        xx = eltype(f.x) === Nothing ? typeof(dom*zero(rule_type(cacheval.cache[1])))[] : f.x
        AutoSymPTR.BatchIntegrand((y,x) -> f.f!(y,x,p), f.y, xx, max_batch=f.max_batch)
    elseif f isa InplaceIntegrand
        AutoSymPTR.InplaceIntegrand((y,x) -> f.f!(y,x,p), f.I)
    else
        x -> f(x, p)
    end
    val, err = autosymptr(g, dom; syms = alg.syms, rule = cacheval.rule, cache = cacheval.cache, keepmost = alg.keepmost,
        abstol = abstol, reltol = reltol, maxevals = maxiters, norm=alg.norm, buffer=cacheval.buffer)
    return IntegralSolution(val, err, true, -1)
end
