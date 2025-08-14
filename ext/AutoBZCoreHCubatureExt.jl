module AutoBZCoreHCubatureExt

using HCubature: hcubature, hquadrature
using AutoBZCore
using AutoBZCore: InplaceArray, BatchArray, init_integrand_cacheval, Success, Failure, IntegralSolution, AbstractIntegralFunction, AbstractFourierIntegralFunction
import AutoBZCore: init_cacheval, do_integral, hcubature_integrand, endpoints

function init_cacheval(f::AbstractIntegralFunction, dom, p, alg::HCubatureJL; kws...)
    prototype, integrand_cacheval = init_integrand_cacheval(f, dom, p)
    prototype isa InplaceArray && throw(ArgumentError("HCubatureJL does not support inplace integrands"))
    prototype isa BatchArray && throw(ArgumentError("HCubatureJL does not support batched integrands"))
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

function hcubature_integrand(f::AbstractFourierIntegralFunction, p, a, b, (g, cacheval))
    return hcubature_integrand(g, p, a, b, cacheval)
end

end