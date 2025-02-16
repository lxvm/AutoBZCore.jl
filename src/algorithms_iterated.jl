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
    g = auxquadgk_integrand(f, p, u, alg_cache, cacheval)
    val, err = auxquadgk(g, usegs; segbuf, maxevals = maxiters, rtol = reltol, atol, order = alg.order, norm = alg.norm)
    value = u*val
    retcode = err < max(something(atol, zero(err)), alg.norm(val)*something(reltol, isnothing(atol) ? sqrt(eps(one(eltype(usegs)))) : 0)) ? Success : Failure
    stats = (; error=u*err)
    return IntegralSolution(value, retcode, stats)
end
auxquadgk_integrand(f::AbstractIntegralFunction, p, u, alg_cache, cacheval) = quadgk_integrand(f, p, u, alg_cache, cacheval)
function auxquadgk_integrand(f::InplaceBatchIntegralFunction, p, u, alg_cache, cacheval)
    pts, upts = alg_cache
    IteratedIntegration.AuxQuadGK.BatchIntegrand((y, x) -> f.f!(y, resize!(pts, length(x)) .= u .* x, p), cacheval, upts; max_batch=f.max_batch)
end
