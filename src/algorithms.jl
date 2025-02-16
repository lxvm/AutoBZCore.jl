# Methods an algorithm must define
# - init_cacheval
# - do_integral

init_integrand_cacheval(f::IntegralFunction, dom, p) = init_integrand_cacheval_if(f.executor, f, dom, p)
init_integrand_cacheval(f::CommonSolveIntegralFunction, dom, p) = init_integrand_cacheval_cs(f.executor, f, dom, p)

function init_integrand_cacheval_if(::SerialExecutor, f::IntegralFunction, dom, p)
    prototype = get_prototype(f, get_prototype(dom), p)
    cacheval = nothing
    return prototype, cacheval
end
function init_integrand_cacheval_if(exec::ThreadedExecutor, f::IntegralFunction, dom, p)
    prototype = get_prototype(f, get_prototype(dom), p)
    proto = [prototype]
    func = InplaceBatchIntegralFunction(proto; max_batch=exec.ntasks) do y, x, p
        @sync for (iy, xi) in zip(eachindex(y), x)
            Threads.@spawn begin
                y[iy] = f.f(xi, p)
            end
        end
    end
    _prototype, _cacheval = init_integrand_cacheval(func, dom, p)
    cacheval = (func, proto, _cacheval)
    return _prototype, cacheval
end
struct InplaceArray{T<:AbstractArray}
    data::T
end
function init_integrand_cacheval(f::InplaceIntegralFunction, dom, p)
    prototype = get_prototype(f, get_prototype(dom), p)
    cacheval = similar(prototype)
    return InplaceArray(prototype), cacheval
end
struct BatchArray{T<:AbstractArray}
    data::T
end
function init_integrand_cacheval(f::InplaceBatchIntegralFunction, dom, p)
    prototype = get_prototype(f, get_prototype(dom), p)
    cacheval = similar(prototype)
    return BatchArray(prototype), cacheval
end
function init_integrand_cacheval_cs(::SerialExecutor, f::CommonSolveIntegralFunction, dom, p)
    solver, integrand, prototype = init_commonsolvefunction(f, dom, p)
    cacheval = (solver, integrand)
    return prototype, cacheval
end
function init_integrand_integrand_cacheval_cs(exec::ThreadedExecutor, f::CommonSolveIntegralFunction, dom, p)
    channel, integrand, prototype = init_commonsolvefunction(f, dom, p)
    proto = [prototype]
    func = InplaceBatchIntegralFunction(proto; max_batch=exec.ntasks) do y, x, p
        do_threaded_solve!(integrand, channel, f, y, x, p)
    end
    _prototype, _cacheval = init_integrand_cacheval(func, dom, p)
    cacheval = (func, channel, integrand, proto, _cacheval)
    return _prototype, cacheval
end


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
function init_cacheval(f::AbstractIntegralFunction, dom, p, alg::QuadGKJL; kws...)
    segs = PuncturedInterval(dom)
    prototype, integrand_cacheval = init_integrand_cacheval(f, dom, p)
    proto = if prototype isa BatchArray
        (data = prototype.data) isa AbstractVector || throw(ArgumentError("QuadGK does not support batched functions with multidimensional outputs"))
        first(data)
    elseif prototype isa InplaceArray
        prototype.data
    else
        prototype
    end
    algorithm_cacheval = if prototype isa BatchArray
        pt = get_prototype(segs)
        pts = zeros(typeof(pt), 2*alg.order+1)
        upts = pts / pt
        (pts, upts)
    elseif prototype isa InplaceArray
        fg = similar(prototype.data)
        I = fg * real(oneunit(eltype(segs)))
        (fg, similar(fg), similar(fg), similar(fg), similar(I), I)
    else
        nothing
    end
    return init_segbuf(proto, segs, alg), algorithm_cacheval, integrand_cacheval
end
function do_integral(f, dom, p, alg::QuadGKJL, (segbuf, alg_cache, cacheval);
                    reltol = nothing, abstol = nothing, maxiters = typemax(Int))
    # we need to strip units from the limits since infinity transformations change the units
    # of the limits, which can break the segbuf
    u = oneunit(eltype(dom))
    usegs = map(x -> x/u, dom)
    atol = isnothing(abstol) ? abstol : abstol/u
    g = quadgk_integrand(f, p, u, alg_cache, cacheval)
    val, err = quadgk(g, usegs...; segbuf, maxevals = maxiters, rtol = reltol, atol, order = alg.order, norm = alg.norm)
    value = u*val
    retcode = err < max(something(atol, zero(err)), alg.norm(val)*something(reltol, isnothing(atol) ? sqrt(eps(one(eltype(usegs)))) : 0)) ? Success : Failure
    stats = (; error=u*err)
    return IntegralSolution(value, retcode, stats)
end
quadgk_integrand(f::IntegralFunction, p, u, alg_cache, cacheval) = quadgk_integrand_if(f.executor, f, p, u, alg_cache, cacheval)
function quadgk_integrand_if(::SerialExecutor, f::IntegralFunction, p, u, alg_cache, cacheval)
    x -> f.f(u*x, p)
end
function quadgk_integrand_if(exec::ThreadedExecutor, f::IntegralFunction, p, u, alg_cache, cacheval)
    func, proto, cache = cacheval
    quadgk_integrand(func, p, u, alg_cache, cache)
end
function quadgk_integrand(f::InplaceIntegralFunction, p, u, alg_cache, cacheval)
    # TODO allocate everything in the QuadGK.InplaceIntegrand in the cacheval
    _f = (y, x) -> f.f!(y, u*x, p)
    fg, fk, Ig, Ik, Idiff, I = alg_cache
    fx = cacheval
    InplaceIntegrand(_f, fg, fk, Ig, Ik, fx, Idiff, I)
end
function quadgk_integrand(f::InplaceBatchIntegralFunction, p, u, alg_cache, cacheval)
    pts, upts = alg_cache
    BatchIntegrand((y, x) -> f.f!(y, resize!(pts, length(x)) .= u .* x, p), cacheval, upts; max_batch=f.max_batch)
end
quadgk_integrand(f::CommonSolveIntegralFunction, p, u, alg_cache, cacheval) = quadgk_integrand_cs(f.executor, f, p, u, alg_cache, cacheval)
function quadgk_integrand_cs(::SerialExecutor, f::CommonSolveIntegralFunction, p, u, alg_cache, cacheval)
    solver, integrand = cacheval
    x -> integrand(solver, f, u * x, p)
end
function quadgk_integrand_cs(exec::ThreadedExecutor, f::CommonSolveIntegralFunction, p, u, alg_cache, cacheval)
    func, channel, integrand, proto, cache = cacheval
    quadgk_integrand(func, p, u, alg_cache, cache)
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

function init_cacheval(f::AbstractIntegralFunction, dom, p, alg::HCubatureJL; kws...)
    f isa IntegralFunction && f.executor isa ThreadedExecutor && throw(ArgumentError("HCubatureJL does not support threaded execution because it does not support batched integrands"))
    f isa CommonSolveIntegralFunction && f.executor isa ThreadedExecutor && throw(ArgumentError("HCubatureJL does not support threaded execution because it does not support batched integrands"))
    f isa InplaceIntegralFunction && throw(ArgumentError("HCubatureJL does not support inplace integrands"))
    f isa InplaceBatchIntegralFunction && throw(ArgumentError("HCubatureJL does not support inplace, batched integrands"))
    prototype, integrand_cacheval = init_integrand_cacheval(f, dom, p)
    return integrand_cacheval
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
    @assert f.executor isa SerialExecutor
    x -> f.f(x, p)
end
function hcubature_integrand(f::CommonSolveIntegralFunction, p, a, b, cacheval)
    @assert f.executor isa SerialExecutor
    solver, integrand = cacheval
    return x -> integrand(solver, f, x, p)
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
    QuadratureFunction(; fun=trapz, npt=50)

Quadrature rule for the standard interval [-1,1] computed from a function `x, w = fun(npt)`.
The nodes and weights should be set so the integral of `f` on [-1,1] is `sum(w .* f.(x))`.
The default quadrature rule is [`trapz`](@ref), although other packages provide rules, e.g.

    using FastGaussQuadrature
    alg = QuadratureFunction(fun=gausslegendre, npt=100)
"""
struct QuadratureFunction{F} <: IntegralAlgorithm
    fun::F
    npt::Int
end
QuadratureFunction(; fun=trapz, npt=50) = QuadratureFunction(fun, npt)

function init_rule(dom, alg::QuadratureFunction)
    x, w = alg.fun(alg.npt)
    return [(w,x) for (w,x) in zip(w,x)]
end

function init_cacheval(f::AbstractIntegralFunction, dom, p, alg::QuadratureFunction; kws...)
    rule = init_rule(dom, alg)
    return init_cacheval(f, dom, p, AffineQuad(rule); kws...)
end

function do_integral(f, dom, p, alg::QuadratureFunction, cacheval;
                    reltol = nothing, abstol = nothing, maxiters = typemax(Int))
    rule = cacheval.rule; buffer=cacheval.algorithm_cacheval.buffer
    segs = segments(dom)
    g = autosymptr_integrand(f, p, segs, cacheval.algorithm_cacheval, cacheval.integrand_cacheval)
    A = sum(1:length(segs)-1) do i
        a, b = segs[i], segs[i+1]
        s = (b-a)/2
        arule = AutoSymPTR.AffineQuad(rule, s, a, 1, s)
        return AutoSymPTR.quadsum(arule, g, s, buffer)
    end

    return IntegralSolution(A, Success, (; numevals = length(cacheval.rule)*(length(segs)-1)))
end
