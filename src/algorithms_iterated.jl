"""
    AuxQuadGKJL(; order = 7, norm = norm)

Generalization of the QuadGKJL provided by Integrals.jl that allows for `AuxValue`d
integrands for auxiliary integration and multi-threaded evaluation with the `batch` argument
to `IntegralProblem`
"""
struct AuxQuadGKJL{F} <: IntegralAlgorithm
    order::Int
    norm::F
end
function AuxQuadGKJL(; order = 7, norm = norm)
    return AuxQuadGKJL(order, norm)
end

function init_cacheval(f::AbstractIntegralFunction, dom, p, alg::AuxQuadGKJL; kws...)
    return init_cacheval(f, dom, p, QuadGKJL(; order=alg.order, norm=alg.norm); kws...)
end
function do_integral(f, dom, p, alg::AuxQuadGKJL, (segbuf, alg_cache, cacheval);
                    reltol = nothing, abstol = nothing, maxiters = typemax(Int))
    # we need to strip units from the limits since infinity transformations change the units
    # of the limits, which can break the segbuf
    u = oneunit(eltype(dom))
    usegs = map(x -> x/u, dom)
    atol = isnothing(abstol) ? abstol : abstol/u
    val, err = call_auxquadgk(f, p, u, usegs, segbuf, alg_cache, cacheval; maxevals = maxiters, rtol = reltol, atol, order = alg.order, norm = alg.norm)
    value = u*val
    retcode = err < max(something(atol, zero(err)), alg.norm(val)*something(reltol, isnothing(atol) ? sqrt(eps(one(eltype(usegs)))) : 0)) ? Success : Failure
    stats = (; error=u*err)
    return IntegralSolution(value, retcode, stats)
end
call_auxquadgk(f::IntegralFunction, p, u, usegs, segbuf, alg_cache, cacheval; kws...) = call_auxquadgk_if(f.executor, f, p, u, usegs, segbuf, alg_cache, cacheval; kws...)
function call_auxquadgk_if(::SerialExecutor, f::IntegralFunction, p, u, usegs, segbuf, alg_cache, cacheval; kws...)
    quadgk(x -> f.f(u*x, p), usegs...; kws..., segbuf)
end
function call_auxquadgk_if(exec::ThreadedExecutor, f::IntegralFunction, p, u, usegs, segbuf, alg_cache, cacheval; kws...)
    func, proto, cache = cacheval
    call_quadgk(func, p, u, usegs, segbuf, alg_cache, cache; kws...)
end
function call_auxquadgk(f::InplaceIntegralFunction, p, u, usegs, segbuf, alg_cache, cacheval; kws...)
    # TODO allocate everything in the AuxQuadGK.InplaceIntegrand in the cacheval
    auxquadgk!((y, x) -> f.f!(y, u*x, p), cacheval, usegs...; kws..., segbuf)
end
function call_auxquadgk(f::InplaceBatchIntegralFunction, p, u, usegs, segbuf, alg_cache, cacheval; kws...)
    pts, upts = alg_cache
    g = IteratedIntegration.AuxQuadGK.BatchIntegrand((y, x) -> f.f!(y, resize!(pts, length(x)) .= u .* x, p), cacheval, upts; max_batch=f.max_batch)
    auxquadgk(g, usegs...; kws..., segbuf)
end
call_auxquadgk(f::CommonSolveIntegralFunction, p, u, usegs, segbuf, alg_cache, cacheval; kws...) = call_auxquadgk_cs(f.executor, f, p, u, usegs, segbuf, alg_cache, cacheval; kws...)
function call_auxquadgk_cs(::SerialExecutor, f::CommonSolveIntegralFunction, p, u, usegs, segbuf, alg_cache, cacheval; kws...)
    solver, integrand = cacheval
    auxquadgk(x -> integrand(solver, f, u * x, p), usegs...; kws..., segbuf)
end
function call_auxquadgk_cs(exec::ThreadedExecutor, f::CommonSolveIntegralFunction, p, u, usegs, segbuf, alg_cache, cacheval; kws...)
    func, channel, integrand, proto, cache = cacheval
    call_auxquadgk(func, p, u, usegs, segbuf, alg_cache, cache; kws...)
end
