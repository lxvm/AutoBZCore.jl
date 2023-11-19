"""
    NestedIntegrand(f, [g=nothin])

Use this wrapper to signal that the integrand `f(x,p)` returns an
[`IntegralSolution`](@ref), meaning that the integrand is itself an integral. Optionally, a
function `g(x, p, f(x,p).u)` can be provided that will transform the value of the inner
integral can be provided. The purpose of this interface is to allow for counting integrand
evaluations correctly when nesting integrals.
"""
struct NestedIntegrand{F,G}
    f::F
    g::G
end

NestedIntegrand(f) = NestedIntegrand(f, nothing)

assemble_nested_integrand(f::NestedIntegrand) = assemble_nested_integrand(f.f, f.g)
assemble_nested_integrand(f, g) = (x, p) -> g(x, p, f(x, p).u)
assemble_nested_integrand(f, ::Nothing) = (x, p) -> f(x, p).u

function init_segbuf(f::NestedIntegrand, dom, p, norm)
    h = assemble_nested_integrand(f)
    return init_segbuf(h, dom, p, norm)
end

function do_solve_quadgk(f::NestedIntegrand, segs, p, cacheval, order, norm, reltol, abstol, maxiters)
    g = assemble_nested_integrand(f)
    return do_solve_quadgk(g, segs, p, cacheval, order, norm, reltol, abstol, maxiters)
end

function assemble_hintegrand(f::NestedIntegrand, dom, p)
    g = assemble_nested_integrand(f)
    return assemble_hintegrand(g, dom, p)
end

function assemble_pintegrand(f::NestedIntegrand, p, dom, rule)
    g = assemble_nested_integrand(f)
    return assemble_pintegrand(g, p, dom, rule)
end

function do_solve_auxquadgk(f::NestedIntegrand, segs, p, cacheval, order, norm, reltol, abstol, maxiters)
    g = assemble_nested_integrand(f)
    return do_solve_auxquadgk(g, segs, p, cacheval, order, norm, reltol, abstol, maxiters)
end

function assemble_cont_integrand(f::NestedIntegrand, p)
    g = assemble_nested_integrand(f)
    return assemble_cont_integrand(g, p)
end

function assemble_mero_integrand(f::NestedIntegrand, p)
    g = assemble_nested_integrand(f)
    return assemble_mero_integrand(g, p)
end

function do_solve_evalcounter(f::NestedIntegrand, dom, p, alg, cacheval; kws...)
    return do_solve_evalcounter_nested(f.f, f.g, dom, p, alg, cacheval; kws...)
end

function do_solve_evalcounter_nested(f, g, dom, p, alg, cacheval; kws...)
    n::Int = 0
    h = (x, p) -> begin
        sol = f(x, p)
        n += sol.numevals > 0 ? sol.numevals : 0
        return isnothing(g) ? sol.u : g(x, p, sol.u)
    end
    sol = do_solve(h, dom, p, alg, cacheval; kws...)
    return IntegralSolution(sol.u, sol.resid, sol.retcode, iszero(n) ? -1 : n)
end

function init_cacheval(f, dom, p, alg::NestedQuad)
    dom isa AbstractIteratedLimits || throw(ArgumentError("NestedQuad only supports AbstractIteratedLimits for domain. Please open an issue."))
    algs = alg.algs isa IntegralAlgorithm ? ntuple(i -> alg.algs, Val(ndims(dom))) : alg.algs
    return init_nested_cacheval(f, p, limit_iterate(dom)..., algs...)
end

# we construct a cacheval with 3 entries:
# 1. storage for the cache of the inner integrals
# 2. The cache used for the current integral
# 3. a value of the same type as the current integral
function init_nested_cacheval(f::F, p, segs, lims, state, alg::IntegralAlgorithm) where {F}
    dom = PuncturedInterval(segs)
    a, b = segs[1], segs[2]
    mid = (a+b)/2 # sample point that should be safe to evaluate
    next = limit_iterate(lims, state, mid) # see what the next limit gives
    fx = f(next,p)
    cacheval = init_cacheval((x, p) -> fx, dom, p, alg)
    return (nothing, cacheval, fx*mid)
end
function init_nested_cacheval(f::F, p, segs, lims, state, alg_::IntegralAlgorithm, algs_::IntegralAlgorithm...) where {F}
    dim = ndims(lims)
    algs = (alg_, algs_...)
    alg = algs[dim]
    dom = PuncturedInterval(segs)
    a, b = segs[1], segs[2]
    mid = (a+b)/2 # sample point that should be safe to evaluate
    next = limit_iterate(lims, state, mid) # see what the next limit gives
    nest = init_nested_cacheval(f, p, next..., algs[1:dim-1]...)
    h = nest[3]
    hx = h*mid
    # units may change for outer integral
    cacheval = init_cacheval((x, p) -> h, dom, p, alg)
    return (nest, cacheval, hx)
end

function do_solve(f::F, lims, p, alg::NestedQuad, cacheval; kws...) where {F}
    lims isa AbstractIteratedLimits || throw(ArgumentError("NestedQuad only supports AbstractIteratedLimits for domain. Please open an issue."))
    segs, lims_, state = limit_iterate(lims)
    dom = PuncturedInterval(segs)
    dim = ndims(lims) # constant propagation :)
    algs = alg.algs isa IntegralAlgorithm ? ntuple(i -> alg.algs, Val(dim)) : alg.algs
    g = assemble_nested_integrand(f, cacheval[3], dom, p, lims_, state, algs[1:dim-1], cacheval[1]; kws...)
    return do_solve(g, dom, p, algs[dim], cacheval[2]; kws...)
end

struct StatefulLimits{d,T,U,S,L<:AbstractIteratedLimits{d,T}} <: AbstractIteratedLimits{d,T}
    segs::U
    state::S
    lims::L
end
function IteratedIntegration.limit_iterate(lims::StatefulLimits)
    return lims.segs, lims.lims, lims.state
end

function assemble_nested_integrand(f::F, fxx, dom, p, lims, state, ::Tuple{}, cacheval; kws...) where {F}
    return (x, p) -> f(limit_iterate(lims, state, x), p)
end
# TODO: specialize with a function wrapper only when the integrand is passed as a wrapper
function assemble_nested_integrand(f::F, fxx, dom, p, lims, state, algs, cacheval; kws_...) where {F}
    kws = NamedTuple(kws_)
    xx = float(oneunit(eltype(dom)))
    TX = typeof(xx)
    TP = typeof(p)
    err = integralerror(last(algs), fxx)
    f_ = FunctionWrapper{IntegralSolution{typeof(fxx),typeof(err)},Tuple{TX,TP}}() do x, p
        # here we rescale the next absolute tolerance to have the right units
        segs, lims_, state_ = limit_iterate(lims, state, x)
        len = segs[end] - segs[1]
        kwargs = haskey(kws, :abstol) ? merge(kws, (abstol=kws.abstol/len,)) : kws
        return do_solve(f, StatefulLimits(segs, state_, lims_), p, NestedQuad(algs), cacheval; kwargs...)
    end
    return NestedIntegrand(f_)
end


function init_nested_cacheval(f::NestedIntegrand, p, segs, lims, state, alg::IntegralAlgorithm)
    g = assemble_nested_integrand(f)
    return init_nested_cacheval(g, p, segs, lims, state, alg)
end
function init_nested_cacheval(f::NestedIntegrand, p, segs, lims, state, alg_::IntegralAlgorithm, algs_::IntegralAlgorithm...)
    g = assemble_nested_integrand(f)
    return init_nested_cacheval(g, p, segs, lims, state, alg_, algs_...)
end
function assemble_nested_integrand(f::NestedIntegrand, fxx, dom, p, lims, state, algs::Tuple{}, cacheval; kws...)
    g = assemble_nested_integrand(f)
    return assemble_nested_integrand(g, fxx, dom, p, lims, state, algs, cacheval; kws...)
end
function assemble_nested_integrand(f::NestedIntegrand, fxx, dom, p, lims, state, algs, cacheval; kws...)
    g = assemble_nested_integrand(f)
    return assemble_nested_integrand(g, fxx, dom, p, lims, state, algs, cacheval; kws...)
end
