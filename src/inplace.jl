"""
    InplaceIntegrand(f!, result::AbstractArray)

Constructor for a `InplaceIntegrand` accepting an integrand of the form `f!(y,x,p)`. The
caller also provides an output array needed to store the result of the quadrature.
Intermediate `y` arrays are allocated during the calculation, and the final result is
may or may not be written to `result`, so use the IntegralSolution immediately after the
calculation to read the result, and don't expect it to persist if the same integrand is used
for another calculation.
"""
struct InplaceIntegrand{F,T<:AbstractArray}
    # in-place function f!(y, x, p) that takes one x value and outputs an array of results in-place
    f!::F
    I::T
end

function init_segbuf(f::InplaceIntegrand, dom, p, norm)
    x, s = init_midpoint_scale(dom)
    u = x/oneunit(x)
    TX = typeof(u)
    TI = typeof(f.I *s/oneunit(s))
    fill!(f.I, zero(eltype(f.I)))
    TE = typeof(norm(f.I)*s/oneunit(s))
    return IteratedIntegration.alloc_segbuf(TX, TI, TE)
end
function do_solve_quadgk(f::InplaceIntegrand, segs, p, cacheval, order, norm, reltol, abstol, maxiters)
    # we need to strip units from the limits since infinity transformations change the units
    # of the limits, which can break the segbuf
    u = oneunit(eltype(segs))
    usegs = map(x -> x/u, segs)
    g! = (y, x) -> f.f!(y, u*x, p)
    result = f.I / u
    val, err = quadgk!(g!, result, usegs..., maxevals = maxiters,
                    rtol = reltol, atol = isnothing(abstol) ? abstol : abstol/u, order = order, norm = norm, segbuf=cacheval)
    return IntegralSolution(f.I .= u .* val, u*err, true, -1)
end

function assemble_hintegrand(f::InplaceIntegrand, dom, p)
    fx = f.I / oneunit(eltype(dom))
    g = x -> (f.f!(fx, x, p); fx*one(eltype(dom)))
    return g
end

function assemble_pintegrand(f::InplaceIntegrand, p, dom, rule)
    return AutoSymPTR.InplaceIntegrand((y,x) -> f.f!(y,x,p), f.I)
end

function do_solve_auxquadgk(f::InplaceIntegrand, segs, p,  cacheval, order, norm, reltol, abstol, maxiters)
    u = oneunit(eltype(segs))
    usegs = map(x -> x/u, segs)
    g! = (y, x) -> f.f!(y, u*x, p)
    result = f.I / u
    val, err = auxquadgk!(g!, result, usegs, maxevals = maxiters,
                    rtol = reltol, atol = isnothing(abstol) ? abstol : abstol/u, order = order, norm = norm, segbuf=cacheval)
    return IntegralSolution(f.I .= u .* val, u*err, true, -1)
end

function assemble_cont_integrand(::InplaceIntegrand, p)
    throw(ArgumentError("ContQuadGK.jl doesn't support inplace evaluation. Consider opening an issue upstream."))
end

function assemble_mero_integrand(::InplaceIntegrand, p)
    throw(ArgumentError("MeroQuadGK.jl doesn't support inplace evaluation. Consider opening an issue upstream."))
end

function do_solve_evalcounter(f::InplaceIntegrand, dom, p, alg, cacheval; kws...)
    n::Int = 0
    g = (y, x, p) -> (n += 1; f.f!(y, x, p))
    sol = do_solve(InplaceIntegrand(g, f.I), dom, p, alg, cacheval; kws...)
    return IntegralSolution(sol.u, sol.resid, sol.retcode, n)
end

function init_nested_cacheval(f::InplaceIntegrand, p, segs, lims, state, alg::IntegralAlgorithm)
    dom = PuncturedInterval(segs)
    a, b = segs[1], segs[2]
    mid = (a+b)/2 # sample point that should be safe to evaluate
    next = limit_iterate(lims, state, mid) # see what the next limit gives
    fxi = f.I*mid/oneunit(prod(next))
    cacheval = init_cacheval(InplaceIntegrand(nothing, fxi), dom, p, alg)
    return (nothing, cacheval, fxi)
end

function assemble_nested_integrand(f::InplaceIntegrand, fxx, dom, p, lims, state, ::Tuple{}, cacheval; kws...)
    xx = float(oneunit(eltype(dom)))
    FX = typeof(fxx/xx)
    TX = typeof(xx)
    TP = typeof(p)
    return InplaceIntegrand(FunctionWrapper{Nothing,Tuple{FX,TX,TP}}() do y, x, p
        f.f!(y, limit_iterate(lims, state, x), p)
        return nothing
    end, f.I)
end
