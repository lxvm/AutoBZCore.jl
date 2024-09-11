"""
    NestedQuad(alg::IntegralAlgorithm)
    NestedQuad(algs::IntegralAlgorithm...)

Nested integration by repeating one quadrature algorithm or composing a list of algorithms.
The domain of integration must be an `AbstractIteratedLimits` from the
IteratedIntegration.jl package. Analogous to `nested_quad` from IteratedIntegration.jl.
The integrand should expect `SVector` inputs. Do not use this for very high-dimensional
integrals, since the compilation time scales very poorly with respect to dimensionality.
In order to improve the compilation time, FunctionWrappers.jl is used to enforce type
stability of the integrand, so you should always pick the widest integration limit type so
that inference works properly. For example, if [`ContQuadGKJL`](@ref) is used as an
algorithm in the nested scheme, then the limits of integration should be made complex.
"""
struct NestedQuad{T,S} <: IntegralAlgorithm
    algs::T
    specialize::S
    NestedQuad(alg::IntegralAlgorithm, specialize::AbstractSpecialization=FunctionWrapperSpecialize()) = new{typeof(alg),typeof(specialize)}(alg, specialize)
    NestedQuad(algs::Tuple{Vararg{IntegralAlgorithm}}, specialize::Tuple{Vararg{AbstractSpecialization}}=ntuple(_->FunctionWrapperSpecialize(), length(algs))) = new{typeof(algs),typeof(specialize)}(algs, specialize)
end
NestedQuad(algs::IntegralAlgorithm...) = NestedQuad(algs)
# TODO add a parallelization option for use when it is safe to do so

function _update!(cache, x, (; p, lims_state))
    segs, lims, state = limit_iterate(lims_state..., x)
    len = segs[end] - segs[begin]
    kws = cache.kwargs
    cache.p = p
    cache.cacheval.dom = segs
    cache.cacheval.kwargs = haskey(kws, :abstol) ? merge(kws, (abstol=kws.abstol/len,)) : kws
    cache.cacheval.p = (; cache.cacheval.p..., lims_state=(lims, state))
    return
end
_postsolve(sol, x, p) = sol.value
function init_cacheval(f, nextdom, p, alg::NestedQuad; kws...)
    x0, (segs, lims, state) = if nextdom isa AbstractIteratedLimits
        interior_point(nextdom), limit_iterate(nextdom)
    else
        nextdom
    end
    algs = alg.algs isa IntegralAlgorithm ? ntuple(i -> alg.algs, Val(ndims(lims))) : alg.algs
    spec = alg.specialize isa AbstractSpecialization ? ntuple(i -> alg.specialize, Val(ndims(lims))) : alg.specialize
    if ndims(lims) == 1
        func, ws = inner_integralfunction(f, x0, p)
    else
        integrand, ws, update!, postsolve = outer_integralfunction(f, x0, p)
        proto = get_prototype(integrand, x0, p)
        a, b, = segs
        x = (a+b)/2
        next = (x0[begin:end-1], limit_iterate(lims, state, x))
        kws = NamedTuple(kws)
        len = segs[end] - segs[begin]
        kwargs = haskey(kws, :abstol) ? merge(kws, (abstol=kws.abstol/len,)) : kws
        subprob = IntegralProblem(integrand, next, p; kwargs...)
        func = CommonSolveIntegralFunction(subprob, NestedQuad(algs[1:ndims(lims)-1], spec[1:ndims(lims)-1]), update!, postsolve, proto*x^(ndims(lims)-1), spec[ndims(lims)])
    end
    prob = IntegralProblem(func, segs, (; p, lims_state=(lims, state), ws); kws...)
    return init(prob, algs[ndims(lims)])
    # the order of updates is somewhat tricky. I think some could be simplified if instead
    # we use an IntegralProblem modified to contain lims_state, instead of passing the
    # parameter as well
end


function do_integral(f, dom, p, alg::NestedQuad, cacheval; kws...)
    cacheval.p = (; cacheval.p..., p)
    cacheval.kwargs = (; cacheval.kwargs..., kws...)
    return solve!(cacheval)
end
function inner_integralfunction(f::IntegralFunction, x0, p)
    proto = get_prototype(f, x0, p)
    func = IntegralFunction(proto) do x, (; p, lims_state)
        f.f(limit_iterate(lims_state..., x), p)
    end
    ws = nothing
    return func, ws
end
function outer_integralfunction(f::IntegralFunction, x0, p)
    proto = get_prototype(f, x0, p)
    func = IntegralFunction(f.f, proto)
    ws = nothing
    return func, ws, _update!, _postsolve
end
#=
"""
    AbsoluteEstimate(est_alg, abs_alg; kws...)

Most algorithms are efficient when using absolute error tolerances, but how do you know the
size of the integral? One option is to estimate it using second algorithm.

A multi-algorithm to estimate an integral using an `est_alg` to generate a rough estimate of
the integral that is combined with a user's relative tolerance to re-calculate the integral
to higher accuracy using the `abs_alg`. The keywords passed to the algorithm may include
`reltol`, `abstol` and `maxiters` and are given to the `est_alg` solver. They should limit
the amount of work of `est_alg` so as to only generate an order-of-magnitude estimate of the
integral. The tolerances passed to `abs_alg` are `abstol=max(abstol,reltol*norm(I))` and
`reltol=0`.
"""
struct AbsoluteEstimate{E<:IntegralAlgorithm,A<:IntegralAlgorithm,F,K<:NamedTuple} <: IntegralAlgorithm
    est_alg::E
    abs_alg::A
    norm::F
    kws::K
end
function AbsoluteEstimate(est_alg, abs_alg; norm=norm, kwargs...)
    kws = NamedTuple(kwargs)
    checkkwargs(kws)
    return AbsoluteEstimate(est_alg, abs_alg, norm, kws)
end

function init_cacheval(f, dom, p, alg::AbsoluteEstimate)
    return (est=init_cacheval(f, dom, p, alg.est_alg),
            abs=init_cacheval(f, dom, p, alg.abs_alg))
end

function do_solve(f, dom, p, alg::AbsoluteEstimate, cacheval;
                    abstol=nothing, reltol=nothing, maxiters=typemax(Int))
    sol = do_solve(f, dom, p, alg.est_alg, cacheval.est; alg.kws...)
    val = alg.norm(sol.u) # has same units as sol
    rtol = reltol === nothing ? sqrt(eps(one(val))) : reltol # use the precision of the solution to set the default relative tolerance
    atol = max(abstol === nothing ? zero(val) : abstol, rtol*val)
    return do_solve(f, dom, p, alg.abs_alg, cacheval.abs;
                    abstol=atol, reltol=zero(rtol), maxiters=maxiters)
end


"""
    EvalCounter(::IntegralAlgorithm)

An algorithm which counts the evaluations used by another algorithm.
The count is stored in the `sol.numevals` field.
"""
struct EvalCounter{T<:IntegralAlgorithm} <: IntegralAlgorithm
    alg::T
end

function init_cacheval(f, dom, p, alg::EvalCounter)
    return init_cacheval(f, dom, p, alg.alg)
end

function do_solve(f, dom, p, alg::EvalCounter, cacheval; kws...)
    if f isa InplaceIntegrand
        ni::Int = 0
        gi = (y, x, p) -> (ni += 1; f.f!(y, x, p))
        soli = do_solve(InplaceIntegrand(gi, f.I), dom, p, alg.alg, cacheval; kws...)
        return IntegralSolution(soli.u, soli.resid, soli.retcode, ni)
    elseif f isa BatchIntegrand
        nb::Int = 0
        gb = (y, x, p) -> (nb += length(x); f.f!(y, x, p))
        solb = do_solve(BatchIntegrand(gb, f.y, f.x, max_batch=f.max_batch), dom, p, alg.alg, cacheval; kws...)
        return IntegralSolution(solb.u, solb.resid, solb.retcode, nb)
    elseif f isa NestedBatchIntegrand
        # TODO allocate a bunch of accumulators associated with the leaves of the nested
        # integrand or rewrap the algorithms in NestedQuad
        error("NestedBatchIntegrand not yet supported with EvalCounter")
    else
        n::Int = 0
        g = (x, p) -> (n += 1; f(x, p)) # we need let to prevent Core.Box around the captured variable
        sol = do_solve(g, dom, p, alg.alg, cacheval; kws...)
        return IntegralSolution(sol.u, sol.resid, sol.retcode, n)
    end
end
=#
