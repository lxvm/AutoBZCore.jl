# Methods an algorithm must define
# - init_cacheval
# - do_integral

"""
    QuadGKJL(; order = 7, norm = norm)

Duplicate of the QuadGKJL provided by Integrals.jl.
"""
struct QuadGKJL{F} <: IntegralAlgorithm
    order::Int
    norm::F
end
function QuadGKJL(; order = 7, norm = norm)
    return QuadGKJL(order, norm)
end

function init_midpoint_scale(a::T, b::T) where {T}
    # we try to reproduce the initial midpoint used by QuadGK, and scale just needs right units
    s = float(oneunit(T))
    if one(T) isa Real
        x = if (infa = isinf(a)) & (infb = isinf(b))
            float(zero(T))
        elseif infa
            float(b - oneunit(b))
        elseif infb
            float(a + oneunit(a))
        else
            (a+b)/2
        end
        return x, s
    else
        return (a+b)/2, s
    end
end
init_midpoint_scale(dom::PuncturedInterval) = init_midpoint_scale(endpoints(dom)...)
function init_segbuf(prototype, segs, alg)
    x, s = init_midpoint_scale(segs)
    u = x/oneunit(x)
    TX = typeof(u)
    fx_s = prototype * s/oneunit(s)
    TI = typeof(fx_s)
    TE = typeof(alg.norm(fx_s))
    return IteratedIntegration.alloc_segbuf(TX, TI, TE)
end

function init_cacheval(f::IntegralFunction, dom, p, alg::QuadGKJL; kws...)
    segs = PuncturedInterval(dom)
    prototype = get_prototype(f, get_prototype(segs), p)
    return init_segbuf(prototype, segs, alg)
end
function init_cacheval(f::InplaceIntegralFunction, dom, p, alg::QuadGKJL; kws...)
    segs = PuncturedInterval(dom)
    prototype = get_prototype(f, get_prototype(segs), p)
    return init_segbuf(prototype, segs, alg), similar(prototype)
end
function init_cacheval(f::InplaceBatchIntegralFunction, dom, p, alg::QuadGKJL; kws...)
    segs = PuncturedInterval(dom)
    pt = get_prototype(segs)
    prototype = get_prototype(f, pt, p)
    prototype isa AbstractVector || throw(ArgumentError("QuadGKJL only supports batch integrands with vector outputs"))
    pts = zeros(typeof(pt), 2*alg.order+1)
    upts = pts / pt
    return init_segbuf(first(prototype), segs, alg), similar(prototype), pts, upts
end
function init_cacheval(f::CommonSolveIntegralFunction, dom, p, alg::QuadGKJL; kws...)
    segs = PuncturedInterval(dom)
    x = get_prototype(segs)
    cache, integrand, prototype = _init_commonsolvefunction(f, dom, p; x)
    return init_segbuf(prototype, segs, alg), cache, integrand
end
function do_integral(f, dom, p, alg::QuadGKJL, cacheval;
                    reltol = nothing, abstol = nothing, maxiters = typemax(Int))
    # we need to strip units from the limits since infinity transformations change the units
    # of the limits, which can break the segbuf
    u = oneunit(eltype(dom))
    usegs = map(x -> x/u, dom)
    atol = isnothing(abstol) ? abstol : abstol/u
    val, err = call_quadgk(f, p, u, usegs, cacheval; maxevals = maxiters, rtol = reltol, atol, order = alg.order, norm = alg.norm)
    value = u*val
    retcode = err < max(something(atol, zero(err)), alg.norm(val)*something(reltol, isnothing(atol) ? sqrt(eps(one(eltype(usegs)))) : 0)) ? Success : Failure
    stats = (; error=u*err)
    return IntegralSolution(value, retcode, stats)
end
function call_quadgk(f::IntegralFunction, p, u, usegs, cacheval; kws...)
    quadgk(x -> f.f(u*x, p), usegs...; kws..., segbuf=cacheval)
end
function call_quadgk(f::InplaceIntegralFunction, p, u, usegs, cacheval; kws...)
    # TODO allocate everything in the QuadGK.InplaceIntegrand in the cacheval
    quadgk!((y, x) -> f.f!(y, u*x, p), cacheval[2], usegs...; kws..., segbuf=cacheval[1])
end
function call_quadgk(f::InplaceBatchIntegralFunction, p, u, usegs, cacheval; kws...)
    pts = cacheval[3]
    g = BatchIntegrand((y, x) -> f.f!(y, resize!(pts, length(x)) .= u .* x, p), cacheval[2], cacheval[4]; max_batch=f.max_batch)
    quadgk(g, usegs...; kws..., segbuf=cacheval[1])
end
function call_quadgk(f::CommonSolveIntegralFunction, p, u, usegs, cacheval; kws...)
    # cache = cacheval[2] could call do_solve!(cache, f, x, p) to fully specialize
    integrand = cacheval[3]
    quadgk(x -> integrand(u * x, p), usegs...; kws..., segbuf=cacheval[1])
end


"""
    HCubatureJL(; norm=norm, initdiv=1)

Multi-dimensional h-adaptive cubature from HCubature.jl.
"""
struct HCubatureJL{N} <: IntegralAlgorithm
    norm::N
    initdiv::Int
end
HCubatureJL(; norm=norm, initdiv=1) = HCubatureJL(norm, initdiv)

function init_cacheval(f::IntegralFunction, dom, p, ::HCubatureJL; kws...)
    # TODO utilize hcubature_buffer
    return
end
function init_cacheval(f::CommonSolveIntegralFunction, dom, p, ::HCubatureJL; kws...)
    cache, integrand, = _init_commonsolvefunction(f, dom, p)
    return cache, integrand
end

function do_integral(f, dom, p, alg::HCubatureJL, cacheval; reltol = 0, abstol = 0, maxiters = typemax(Int))
    a, b = endpoints(dom)
    g = hcubature_integrand(f, p, a, b, cacheval)
    routine = a isa Number ? hquadrature : hcubature
    value, error = routine(g, a, b; norm = alg.norm, initdiv = alg.initdiv, atol=abstol, rtol=reltol, maxevals=maxiters)
    retcode = error < max(something(abstol, zero(error)), alg.norm(value)*something(reltol, isnothing(abstol) ? sqrt(eps(eltype(a))) : abstol)) ? Success : Failure
    stats = (; error)
    return IntegralSolution(value, retcode, stats)
end
function hcubature_integrand(f::IntegralFunction, p, a, b, cacheval)
    x -> f.f(x, p)
end
function hcubature_integrand(f::CommonSolveIntegralFunction, p, a, b, cacheval)
    integrand = cacheval[2]
    return x -> integrand(x, p)
end

"""
    trapz(n::Integer)

Return the weights and nodes on the standard interval [-1,1] of the [trapezoidal
rule](https://en.wikipedia.org/wiki/Trapezoidal_rule).
"""
function trapz(n::Integer)
    @assert n > 1
    r = range(-1, 1, length=n)
    x = collect(r)
    halfh = step(r)/2
    h = step(r)
    w = [ (i == 1) || (i == n) ? halfh : h for i in 1:n ]
    return (x, w)
end

"""
    QuadratureFunction(; fun=trapz, npt=50, nthreads=1)

Quadrature rule for the standard interval [-1,1] computed from a function `x, w = fun(npt)`.
The nodes and weights should be set so the integral of `f` on [-1,1] is `sum(w .* f.(x))`.
The default quadrature rule is [`trapz`](@ref), although other packages provide rules, e.g.

    using FastGaussQuadrature
    alg = QuadratureFunction(fun=gausslegendre, npt=100)

`nthreads` sets the numbers of threads used to parallelize the quadrature only when the
integrand is a , in which case the user must parallelize the
integrand evaluations. For no threading set `nthreads=1`.
"""
struct QuadratureFunction{F} <: IntegralAlgorithm
    fun::F
    npt::Int
    nthreads::Int
end
QuadratureFunction(; fun=trapz, npt=50, nthreads=1) = QuadratureFunction(fun, npt, nthreads)

function init_rule(dom, alg::QuadratureFunction)
    x, w = alg.fun(alg.npt)
    return [(w,x) for (w,x) in zip(w,x)]
end
function init_autosymptr_cache(f::IntegralFunction, dom, p, bufsize; kws...)
    return (; buffer=nothing)
end
function init_autosymptr_cache(f::InplaceIntegralFunction, dom, p, bufsize; kws...)
    x = get_prototype(dom)
    proto = get_prototype(f, x, p)
    y = similar(proto)
    ytmp = similar(proto)
    I = y * prod(x)
    Itmp = similar(I)
    return (; buffer=nothing, I, Itmp, y, ytmp)
end
function init_autosymptr_cache(f::InplaceBatchIntegralFunction, dom, p, bufsize; kws...)
    x0 = get_prototype(dom)
    proto=get_prototype(f, x0, p)
    return (; buffer=similar(proto, bufsize), y=similar(proto, bufsize), x=Vector{typeof(x0)}(undef, bufsize))
end
function init_autosymptr_cache(f::CommonSolveIntegralFunction, dom, p, bufsize; kws...)
    cache, integrand, = _init_commonsolvefunction(f, dom, p)
    return (; buffer=nothing, cache, integrand)
end
function init_cacheval(f, dom, p, alg::QuadratureFunction; kws...)
    rule = init_rule(dom, alg)
    cache = init_autosymptr_cache(f, dom, p, alg.npt; kws...)
    return (; rule, cache...)
end

function do_integral(f, dom, p, alg::QuadratureFunction, cacheval;
                    reltol = nothing, abstol = nothing, maxiters = typemax(Int))
    rule = cacheval.rule; buffer=cacheval.buffer
    segs = segments(dom)
    g = autosymptr_integrand(f, p, segs, cacheval)
    A = sum(1:length(segs)-1) do i
        a, b = segs[i], segs[i+1]
        s = (b-a)/2
        arule = AutoSymPTR.AffineQuad(rule, s, a, 1, s)
        return AutoSymPTR.quadsum(arule, g, s, buffer)
    end

    return IntegralSolution(A, Success, (; numevals = length(cacheval.rule)*(length(segs)-1)))
end
function autosymptr_integrand(f::IntegralFunction, p, segs, cacheval)
    x -> f.f(x, p)
end
function autosymptr_integrand(f::InplaceIntegralFunction, p, segs, cacheval)
    AutoSymPTR.InplaceIntegrand((y,x) -> f.f!(y,x,p), cacheval.I, cacheval.Itmp, cacheval.y, cacheval.ytmp)
end
function autosymptr_integrand(f::InplaceBatchIntegralFunction, p, segs, cacheval)
    AutoSymPTR.BatchIntegrand((y,x) -> f.f!(y,x,p), cacheval.y, cacheval.x, max_batch=f.max_batch)
end
function autosymptr_integrand(f::CommonSolveIntegralFunction, p, segs, cacheval)
    integrand = cacheval.integrand
    return x -> integrand(x, p)
end
