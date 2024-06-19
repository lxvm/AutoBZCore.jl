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
function init_cacheval(f::CommonSolveIntegralFunction, dom, p, alg::AuxQuadGKJL; kws...)
    segs = PuncturedInterval(dom)
    x = get_prototype(segs)
    cache, integrand, prototype = _init_commonsolvefunction(f, dom, p; x)
    return init_segbuf(prototype, segs, alg), cache, integrand
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
function call_auxquadgk(f::CommonSolveIntegralFunction, p, u, usegs, cacheval; kws...)
    # cache = cacheval[2] could call do_solve!(cache, f, x, p) to fully specialize
    integrand = cacheval[3]
    auxquadgk(x -> integrand(u*x, p), usegs...; kws..., segbuf=cacheval[1])
end

"""
    ContQuadGKJL(; order = 7, norm = norm, rho = 1.0, rootmeth = IteratedIntegration.ContQuadGK.NewtonDeflation())

A 1d contour deformation quadrature scheme for scalar, complex-valued integrands. It
defaults to regular `quadgk` behavior on the real axis, but if it finds a root of 1/f
nearby, in the sense of Bernstein ellipse for the standard segment `[-1,1]` with semiaxes
`cosh(rho)` and `sinh(rho)`, on either the upper/lower half planes, then it dents the
contour away from the presumable pole.
"""
struct ContQuadGKJL{F,M} <: IntegralAlgorithm
    order::Int
    norm::F
    rho::Float64
    rootmeth::M
end
function ContQuadGKJL(; order = 7, norm = norm, rho = 1.0, rootmeth = IteratedIntegration.ContQuadGK.NewtonDeflation())
    return ContQuadGKJL(order, norm, rho, rootmeth)
end

function init_csegbuf(prototype, dom, alg::ContQuadGKJL)
    segs = PuncturedInterval(dom)
    a, b = endpoints(segs)
    x, s = (a+b)/2, (b-a)/2
    TX = typeof(x)
    convert(ComplexF64, prototype)
    fx_s = one(ComplexF64) * s # currently the integrand is forcibly written to a ComplexF64 buffer
    TI = typeof(fx_s)
    TE = typeof(alg.norm(fx_s))
    r_segbuf = IteratedIntegration.ContQuadGK.PoleSegment{TX,TI,TE}[]
    fc_s = prototype * complex(s) # the regular evalrule is used on complex segments
    TCX = typeof(complex(x))
    TCI = typeof(fc_s)
    TCE = typeof(alg.norm(fc_s))
    c_segbuf = IteratedIntegration.ContQuadGK.Segment{TCX,TCI,TCE}[]
    return (r=r_segbuf, c=c_segbuf)
end
function init_cacheval(f::IntegralFunction, dom, p, alg::ContQuadGKJL; kws...)
    segs = PuncturedInterval(dom)
    prototype = get_prototype(f, get_prototype(segs), p)
    init_csegbuf(prototype, dom, alg)
end
function init_cacheval(f::CommonSolveIntegralFunction, dom, p, alg::ContQuadGKJL; kws...)
    segs = PuncturedInterval(dom)
    cache, integrand, prototype = _init_commonsolvefunction(f, dom, p; x=get_prototype(segs))
    segbufs = init_csegbuf(prototype, dom, alg)
    return (; segbufs..., cache, integrand)
end

function do_integral(f, dom, p, alg::ContQuadGKJL, cacheval;
                    reltol = nothing, abstol = nothing, maxiters = typemax(Int))

    value, err = call_contquadgk(f, p, dom, cacheval;  maxevals = maxiters, rho = alg.rho, rootmeth = alg.rootmeth,
                    rtol = reltol, atol = abstol, order = alg.order, norm = alg.norm, r_segbuf=cacheval.r, c_segbuf=cacheval.c)
    retcode = err < max(something(abstol, zero(err)), alg.norm(value)*something(reltol, isnothing(abstol) ? sqrt(eps(one(eltype(dom)))) : 0)) ? Success : Failure
    stats = (; error=err)
    return IntegralSolution(value, retcode, stats)
end
function call_contquadgk(f::IntegralFunction, p, segs, cacheval; kws...)
    contquadgk(x -> f.f(x, p), segs; kws...)
end
function call_contquadgk(f::CommonSolveIntegralFunction, p, segs, cacheval; kws...)
    integrand = cacheval.integrand
    contquadgk(x -> integrand(x, p), segs...; kws...)
end

"""
    MeroQuadGKJL(; order = 7, norm = norm, rho = 1.0, rootmeth = IteratedIntegration.MeroQuadGK.NewtonDeflation())

A 1d pole subtraction quadrature scheme for scalar, complex-valued integrands that are
meromorphic. It defaults to regular `quadgk` behavior on the real axis, but if it finds
nearby roots of 1/f, in the sense of Bernstein ellipse for the standard segment `[-1,1]`
with semiaxes `cosh(rho)` and `sinh(rho)`, it attempts pole subtraction on that segment.
"""
struct MeroQuadGKJL{F,M} <: IntegralAlgorithm
    order::Int
    norm::F
    rho::Float64
    rootmeth::M
end
function MeroQuadGKJL(; order = 7, norm = norm, rho = 1.0, rootmeth = IteratedIntegration.MeroQuadGK.NewtonDeflation())
    return MeroQuadGKJL(order, norm, rho, rootmeth)
end

function init_msegbuf(prototype, dom, alg::MeroQuadGKJL; kws...)
    segs = PuncturedInterval(dom)
    a, b = endpoints(segs)
    x, s = (a + b)/2, (b-a)/2
    convert(ComplexF64, prototype)
    fx_s = one(ComplexF64) * s # ignore the actual integrand since it is written to CF64 array
    err = alg.norm(fx_s)
    return IteratedIntegration.alloc_segbuf(typeof(x), typeof(fx_s), typeof(err))
end
function init_cacheval(f::IntegralFunction, dom, p, alg::MeroQuadGKJL; kws...)
    segs = PuncturedInterval(dom)
    prototype = get_prototype(f, get_prototype(segs), p)
    segbuf = init_msegbuf(prototype, dom, alg)
    return (; segbuf)
end
function init_cacheval(f::CommonSolveIntegralFunction, dom, p, alg::MeroQuadGKJL; kws...)
    segs = PuncturedInterval(dom)
    cache, integrand, prototype = _init_commonsolvefunction(f, dom, p; x=get_prototype(segs))
    segbuf = init_msegbuf(prototype, dom, alg)
    return (; segbuf, cache, integrand)
end

function do_integral(f, dom, p, alg::MeroQuadGKJL, cacheval;
                    reltol = nothing, abstol = nothing, maxiters = typemax(Int))
    value, err = call_meroquadgk(f, p, dom, cacheval;  maxevals = maxiters, rho = alg.rho, rootmeth = alg.rootmeth,
            rtol = reltol, atol = abstol, order = alg.order, norm = alg.norm, segbuf=cacheval.segbuf)
    retcode = err < max(something(abstol, zero(err)), alg.norm(value)*something(reltol, isnothing(abstol) ? sqrt(eps(one(eltype(dom)))) : 0)) ? Success : Failure
    stats = (; error=err)
    return IntegralSolution(value, retcode, stats)
end
function call_meroquadgk(f::IntegralFunction, p, segs, cacheval; kws...)
    meroquadgk(x -> f.f(x, p), segs; kws...)
end
function call_meroquadgk(f::CommonSolveIntegralFunction, p, segs, cacheval; kws...)
    integrand = cacheval.integrand
    meroquadgk(x -> integrand(x, p), segs...; kws...)
end
