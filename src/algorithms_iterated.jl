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

function init_cacheval(f::IntegralFunction, dom, p, alg::AuxQuadGKJL; kws...)
    segs = PuncturedInterval(dom)
    prototype = get_prototype(f, get_prototype(segs), p)
    return init_segbuf(prototype, segs, alg)
end
function init_cacheval(f::InplaceIntegralFunction, dom, p, alg::AuxQuadGKJL; kws...)
    segs = PuncturedInterval(dom)
    prototype = get_prototype(f, get_prototype(segs), p)
    return init_segbuf(prototype, segs, alg), similar(prototype)
end
function init_cacheval(f::InplaceBatchIntegralFunction, dom, p, alg::AuxQuadGKJL; kws...)
    segs = PuncturedInterval(dom)
    pt = get_prototype(segs)
    prototype = get_prototype(f, pt, p)
    prototype isa AbstractVector || throw(ArgumentError("QuadGKJL only supports batch integrands with vector outputs"))
    pts = zeros(typeof(pt), 2*alg.order+1)
    upts = pts / pt
    return init_segbuf(first(prototype), segs, alg), similar(prototype), pts, upts
end
init_cacheval(f::CommonSolveIntegralFunction, dom, p, alg::AuxQuadGKJL; kws...) = init_cacheval_cs(f.executor, f, dom, p, alg; kws...)
function init_cacheval_cs(::SerialExecutor, f::CommonSolveIntegralFunction, dom, p, alg::AuxQuadGKJL; kws...)
    segs = PuncturedInterval(dom)
    solver, integrand, prototype = init_commonsolvefunction(f, dom, p)
    return init_segbuf(prototype, segs, alg), solver, integrand
end
function init_cacheval_cs(exec::ThreadedExecutor, f::CommonSolveIntegralFunction, dom, p, alg::AuxQuadGKJL; kws...)
    channel, integrand, prototype = init_commonsolvefunction(f, dom, p)
    proto = [prototype]
    func = InplaceBatchIntegralFunction(proto; max_batch=exec.ntasks) do y, x, p
        do_threaded_solve!(integrand, channel, f, y, x, p)
    end
    channel, integrand, proto, init_cacheval(func, dom, p, alg; kws...)
end
function do_integral(f, dom, p, alg::AuxQuadGKJL, cacheval;
                    reltol = nothing, abstol = nothing, maxiters = typemax(Int))
    # we need to strip units from the limits since infinity transformations change the units
    # of the limits, which can break the segbuf
    u = oneunit(eltype(dom))
    usegs = map(x -> x/u, dom)
    atol = isnothing(abstol) ? abstol : abstol/u
    val, err = call_auxquadgk(f, p, u, usegs, cacheval; maxevals = maxiters, rtol = reltol, atol, order = alg.order, norm = alg.norm)
    value = u*val
    retcode = err < max(something(atol, zero(err)), alg.norm(val)*something(reltol, isnothing(atol) ? sqrt(eps(one(eltype(usegs)))) : 0)) ? Success : Failure
    stats = (; error=u*err)
    return IntegralSolution(value, retcode, stats)
end
function call_auxquadgk(f::IntegralFunction, p, u, usegs, cacheval; kws...)
    auxquadgk(x -> f.f(u*x, p), usegs...; kws..., segbuf=cacheval)
end
function call_auxquadgk(f::InplaceIntegralFunction, p, u, usegs, cacheval; kws...)
    # TODO allocate everything in the AuxQuadGK.InplaceIntegrand in the cacheval
    auxquadgk!((y, x) -> f.f!(y, u*x, p), cacheval[2], usegs...; kws..., segbuf=cacheval[1])
end
function call_auxquadgk(f::InplaceBatchIntegralFunction, p, u, usegs, cacheval; kws...)
    pts = cacheval[3]
    g = IteratedIntegration.AuxQuadGK.BatchIntegrand((y, x) -> f.f!(y, resize!(pts, length(x)) .= u .* x, p), cacheval[2], cacheval[4]; max_batch=f.max_batch)
    auxquadgk(g, usegs...; kws..., segbuf=cacheval[1])
end
call_auxquadgk(f::CommonSolveIntegralFunction, p, u, usegs, cacheval; kws...) = call_auxquadgk_cs(f.executor, f, p, u, usegs, cacheval; kws...)
function call_auxquadgk_cs(::SerialExecutor, f::CommonSolveIntegralFunction, p, u, usegs, cacheval; kws...)
    segbuf, solver, integrand = cacheval
    auxquadgk(x -> integrand(solver, f, u * x, p), usegs...; kws..., segbuf)
end
function call_auxquadgk_cs(exec::ThreadedExecutor, f::CommonSolveIntegralFunction, p, u, usegs, cacheval; kws...)
    channel, integrand, proto, cache = cacheval
    func = InplaceBatchIntegralFunction(proto; max_batch=exec.ntasks) do y, x, p
        do_threaded_solve!(integrand, channel, f, y, x, p)
    end
    call_auxquadgk(func, p, u, usegs, cache; kws...)
end