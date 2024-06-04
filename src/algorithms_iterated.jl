# TODO move these to an extension if SymmetricBZ can be simplified

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

function init_cacheval(f, dom, p, alg::AuxQuadGKJL)
    f isa NestedBatchIntegrand && throw(ArgumentError("AuxQuadGKJL doesn't support nested batching"))
    return init_segbuf(f, dom, p, alg.norm)
end

function do_solve(f, dom, p, alg::AuxQuadGKJL, cacheval;
                    reltol = nothing, abstol = nothing, maxiters = typemax(Int))

    segs = segments(dom)
    u = oneunit(eltype(dom))
    usegs = map(x -> x/u, segs)
    if f isa InplaceIntegrand
        g! = (y, x) -> f.f!(y, u*x, p)
        result = f.I / u
        val, err = auxquadgk!(g!, result, usegs, maxevals = maxiters,
                        rtol = reltol, atol = isnothing(abstol) ? abstol : abstol/u, order = alg.order, norm = alg.norm, segbuf=cacheval)
        return IntegralSolution(f.I .= u .* val, u*err, true, -1)
    elseif f isa BatchIntegrand
        xx = eltype(f.x) === Nothing ? typeof((segs[1]+segs[end])/2)[] : f.x
        g_ = (y, x) -> (resize!(xx, length(x)); f.f!(y, xx .= u .* x, p))
        g = IteratedIntegration.AuxQuadGK.BatchIntegrand(g_, f.y, xx/u, max_batch=f.max_batch)
        val, err = auxquadgk(g, usegs, maxevals = maxiters,
                        rtol = reltol, atol = isnothing(abstol) ? abstol : abstol/u, order = alg.order, norm = alg.norm, segbuf=cacheval)
        return IntegralSolution(u*val, u*err, true, -1)
    else
        g = x -> f(u*x, p)
        val, err = auxquadgk(g, usegs, maxevals = maxiters,
                        rtol = reltol, atol = isnothing(abstol) ? abstol : abstol/u, order = alg.order, norm = alg.norm, segbuf=cacheval)
        return IntegralSolution(u*val, u*err, true, -1)
    end
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

function init_cacheval(f, dom, p, alg::ContQuadGKJL)
    f isa NestedBatchIntegrand && throw(ArgumentError("ContQuadGK doesn't support nested batching"))
    f isa BatchIntegrand && throw(ArgumentError("ContQuadGK doesn't support batching"))
    f isa InplaceIntegrand && throw(ArgumentError("ContQuadGK doesn't support inplace integrands"))

    a, b = endpoints(dom)
    x, s = (a+b)/2, (b-a)/2
    TX = typeof(x)
    fx_s = one(ComplexF64) * s # currently the integrand is forcibly written to a ComplexF64 buffer
    TI = typeof(fx_s)
    TE = typeof(alg.norm(fx_s))
    r_segbuf = IteratedIntegration.ContQuadGK.PoleSegment{TX,TI,TE}[]
    fc_s = f(complex(x), p) * complex(s) # the regular evalrule is used on complex segments
    TCX = typeof(complex(x))
    TCI = typeof(fc_s)
    TCE = typeof(alg.norm(fc_s))
    c_segbuf = IteratedIntegration.ContQuadGK.Segment{TCX,TCI,TCE}[]
    return (r=r_segbuf, c=c_segbuf)
end

function do_solve(f, dom, p, alg::ContQuadGKJL, cacheval;
                    reltol = nothing, abstol = nothing, maxiters = typemax(Int))

    segs = segments(dom)
    g = x -> f(x, p)
    val, err = contquadgk(g, segs, maxevals = maxiters, rho = alg.rho, rootmeth = alg.rootmeth,
                    rtol = reltol, atol = abstol, order = alg.order, norm = alg.norm, r_segbuf=cacheval.r, c_segbuf=cacheval.c)
    return IntegralSolution(val, err, true, -1)
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

function init_cacheval(f, dom, p, alg::MeroQuadGKJL)
    f isa NestedBatchIntegrand && throw(ArgumentError("MeroQuadGK doesn't support nested batching"))
    f isa BatchIntegrand && throw(ArgumentError("MeroQuadGK doesn't support batching"))
    f isa InplaceIntegrand && throw(ArgumentError("MeroQuadGK doesn't support inplace integrands"))
    a, b = endpoints(dom)
    x, s = (a + b)/2, (b-a)/2
    fx_s = one(ComplexF64) * s # ignore the actual integrand since it is written to CF64 array
    err = alg.norm(fx_s)
    return IteratedIntegration.alloc_segbuf(typeof(x), typeof(fx_s), typeof(err))
end

function do_solve(f, dom, p, alg::MeroQuadGKJL, cacheval;
                    reltol = nothing, abstol = nothing, maxiters = typemax(Int))

    segs = segments(dom)
    g = x -> f(x, p)
    val, err = meroquadgk(g, segs, maxevals = maxiters, rho = alg.rho, rootmeth = alg.rootmeth,
                    rtol = reltol, atol = abstol, order = alg.order, norm = alg.norm, segbuf=cacheval)
    return IntegralSolution(val, err, true, -1)
end
