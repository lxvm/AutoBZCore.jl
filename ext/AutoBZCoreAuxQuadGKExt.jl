module AutoBZCoreAuxQuadGKExt


using AutoBZCore
using AutoBZCore: InplaceArray, BatchArray, init_integrand_cacheval, quadgk_integrand, symmetrize__, AbstractIntegralFunction, AbstractFourierIntegralFunction, Success, Failure, IntegralSolution
import AutoBZCore: init_cacheval, do_integral, auxquadgk_integrand, symmetrize_, _segments, PuncturedInterval
using AuxQuadGK
import QuadGK

function init_cacheval(f::AbstractIntegralFunction, dom, p, alg::AuxQuadGKJL; kws...)
    return init_cacheval(f, dom, p, QuadGKJL(; order=alg.order, norm=alg.norm); kws...)
end
function do_integral(f, dom, p, alg::AuxQuadGKJL, (segbuf, alg_cache, cacheval);
                    reltol = nothing, abstol = nothing, maxiters = typemax(Int))
    # we need to strip units from the limits since infinity transformations change the units
    # of the limits, which can break the segbuf
    segs = PuncturedInterval(dom)
    u = oneunit(real(eltype(segs)))
    usegs = map(x -> x/u, _segments(segs))
    atol = isnothing(abstol) ? abstol : abstol/u
    # @show abstol reltol
    g = auxquadgk_integrand(f, p, u, alg_cache, cacheval)
    val, err = auxquadgk(g, usegs; segbuf, maxevals = maxiters, rtol = reltol, atol, order = alg.order, norm = alg.norm)
    value = u*val
    retcode = err < max(something(atol, zero(err)), alg.norm(val)*something(reltol, isnothing(atol) ? sqrt(eps(float(one(u)))) : 0)) ? Success : Failure
    stats = (; error=u*err)
    return IntegralSolution(value, retcode, stats)
end
function auxquadgk_integrand(f::AbstractIntegralFunction, p, u, alg_cache, cacheval)
    g = quadgk_integrand(f, p, u, alg_cache, cacheval)
    if g isa QuadGK.BatchIntegrand
        return AuxQuadGK.BatchIntegrand(g.f!, g.y, g.x, g.max_batch)
    else
        return g
    end
end
function auxquadgk_integrand(f::InplaceBatchIntegralFunction, p, u, alg_cache, cacheval)
    pts, upts = alg_cache
    AuxQuadGK.BatchIntegrand((y, x) -> f.f!(y, resize!(pts, length(x)) .= u .* x, p), cacheval, upts; max_batch=f.max_batch)
end


function auxquadgk_integrand(f::AbstractFourierIntegralFunction, p, u, alg_cache, (g, cacheval))
    return auxquadgk_integrand(g, p, u, alg_cache, cacheval)
end

symmetrize_(rep, bz, x::AuxValue) = AuxValue(symmetrize__(rep, bz, x.val), symmetrize__(rep, bz, x.aux))

end