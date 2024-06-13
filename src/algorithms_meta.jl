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

#=
# this function helps create a tree of the cachevals used by each quadrature
function nested_cacheval(f::F, p::P, algs, segs, lims, state, x, xs...) where {F,P}
    dom = PuncturedInterval(segs)
    a, b = segs[1], segs[2]
    dim = ndims(lims)
    alg = algs[dim]
    mid = (a+b)/2 # sample point that should be safe to evaluate
    next = limit_iterate(lims, state, mid) # see what the next limit gives
    if xs isa Tuple{} # the next integral takes us to the inner integral
        # base case test integrand of the inner integral
        # we need to pass dummy integrands to all the outer integrals so that they can build
        # their caches with the right types
        if f isa BatchIntegrand || f isa NestedBatchIntegrand
            # Batch integrate the inner integral only
            cacheval = init_cacheval(BatchIntegrand(nothing, f.y, f.x, max_batch=f.max_batch), dom, p, alg)
            return (nothing, cacheval, oneunit(eltype(f.y))*mid)
        elseif f isa InplaceIntegrand
            # Inplace integrate through the whole nest structure
            fxi = f.I*mid/oneunit(prod(next))
            cacheval = init_cacheval(InplaceIntegrand(nothing, fxi), dom, p, alg)
            return (nothing, cacheval, fxi)
        else
            fx = f(next,p)
            cacheval = init_cacheval((x, p) -> fx, dom, p, alg)
            return (nothing, cacheval, fx*mid)
        end
    elseif f isa NestedBatchIntegrand
        algs_ = algs[1:dim-1]
        # numbered names to avoid type instabilities (we are not using dispatch, but the
        # compiler's optimization for the recursive function's argument types)
        nest0 = nested_cacheval(f.f[1], p, algs_, next..., x, xs[1:dim-2]...)
        cacheval = init_cacheval(BatchIntegrand(nothing, f.y, f.x, max_batch=f.max_batch), dom, p, alg)
        return (ntuple(n -> n == 1 ? nest0 : deepcopy(nest0), Val(length(f.f))), cacheval, nest0[3]*mid)
    else
        algs_ = algs[1:dim-1]
        nest1 = nested_cacheval(f, p, algs_, next..., x, xs[1:dim-2]...)
        h = nest1[3]
        hx = h*mid
        # units may change for outer integral
        if f isa InplaceIntegrand
            cacheval = init_cacheval(InplaceIntegrand(nothing, hx), dom, p, alg)
            return (nest1, cacheval, hx)
        else
            cacheval = init_cacheval((x, p) -> h, dom, p, alg)
            return (nest1, cacheval, hx)
        end
    end
end
function init_cacheval(f, dom::AbstractIteratedLimits, p, alg::NestedQuad)
    algs = alg.algs isa IntegralAlgorithm ? ntuple(i -> alg.algs, Val(ndims(dom))) : alg.algs
    return nested_cacheval(f, p, algs, limit_iterate(dom)..., interior_point(dom)...)
end
=#

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
function init_cacheval(f::IntegralFunction, nextdom, p, alg::NestedQuad; kws...)
    x0, (segs, lims, state) = if nextdom isa AbstractIteratedLimits
        interior_point(nextdom), limit_iterate(nextdom)
    else
        nothing, nextdom
    end
    algs = alg.algs isa IntegralAlgorithm ? ntuple(i -> alg.algs, Val(ndims(lims))) : alg.algs
    spec = alg.specialize isa AbstractSpecialization ? ntuple(i -> alg.specialize, Val(ndims(lims))) : alg.specialize
    proto = get_prototype(f, x0, p)
    func = if ndims(lims) == 1
        inner_integralfunction(f, proto)
    else
        len = segs[end] - segs[begin]
        a, b, = segs
        x = (a+b)/2
        next = limit_iterate(lims, state, x)
        kws = NamedTuple(kws)
        kwargs = haskey(kws, :abstol) ? merge(kws, (abstol=kws.abstol/len,)) : kws
        integrand = outer_integralfunction(f, proto)
        subprob = IntegralProblem(integrand, next, p; kwargs...)
        CommonSolveIntegralFunction(subprob, NestedQuad(algs[1:ndims(lims)-1], spec[1:ndims(lims)-1]), _update!, _postsolve, proto*x^(ndims(lims)-1), spec[ndims(lims)])
    end
    prob = IntegralProblem(func, segs, (; p, lims_state=(lims, state)); kws...)
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
function inner_integralfunction(f::IntegralFunction, proto)
    IntegralFunction(proto) do x, (; p, lims_state)
        f.f(limit_iterate(lims_state..., x), p)
    end
end
function outer_integralfunction(f::IntegralFunction, proto)
    return IntegralFunction(f.f, proto)
end
#=
function init_nest(f::F, fxx, dom, p,lims, state, algs, cacheval; kws_...) where {F}
    kws = NamedTuple(kws_)
    xx = float(oneunit(eltype(dom)))
    FX = typeof(fxx/xx)
    TX = typeof(xx)
    TP = Tuple{typeof(p),typeof(lims),typeof(state)}
    if algs isa Tuple{} # inner integral
        if f isa BatchIntegrand
            return f
        elseif f isa NestedBatchIntegrand
            nchunk = length(f.f)
            return BatchIntegrand(FunctionWrapper{Nothing,Tuple{typeof(f.y),typeof(f.x),TP}}() do y, x, (p, lims, state)
                Threads.@threads for ichunk in 1:min(nchunk, length(x))
                    for (i, j) in zip(getchunk(x, ichunk, nchunk, :scatter), getchunk(y, ichunk, nchunk, :scatter))
                        xi = x[i]
                        y[j] = f.f[ichunk](limit_iterate(lims, state, xi), p)
                    end
                end
                return nothing
            end, f.y, f.x, max_batch=f.max_batch)
        elseif f isa InplaceIntegrand
            return InplaceIntegrand(FunctionWrapper{Nothing,Tuple{FX,TX,TP}}() do y, x, (p, lims, state)
                f.f!(y, limit_iterate(lims, state, x), p)
                return nothing
            end, f.I)
        else
            return FunctionWrapper{FX,Tuple{TX,TP}}() do x, (p, lims, state)
                return f(limit_iterate(lims, state, x), p)
            end
        end
    else
        if f isa InplaceIntegrand
            return InplaceIntegrand(FunctionWrapper{Nothing,Tuple{FX,TX,TP}}() do y, x, (p, lims, state)
                segs, lims_, state_ = limit_iterate(lims, state, x)
                len = segs[end] - segs[1]
                kwargs = haskey(kws, :abstol) ? merge(kws, (abstol=kws.abstol/len,)) : kws
                do_solve(InplaceIntegrand(f.f!, y), lims_, NestState(p, segs, state_), NestedQuad(algs), cacheval; kwargs...)
                return nothing
            end, f.I)
        elseif f isa NestedBatchIntegrand
            nchunks = length(f.f)
            return BatchIntegrand(FunctionWrapper{Nothing,Tuple{typeof(f.y),typeof(f.x),TP}}() do y, x, (p, lims, state)
                Threads.@threads for ichunk in 1:min(nchunks, length(x))
                    for (i, j) in zip(getchunk(x, ichunk, nchunks, :scatter), getchunk(y, ichunk, nchunks, :scatter))
                        xi = x[i]
                        segs, lims_, state_ = limit_iterate(lims, state, xi)
                        len = segs[end] - segs[1]
                        kwargs = haskey(kws, :abstol) ? merge(kws, (abstol=kws.abstol/len,)) : kws
                        y[j] = do_solve(f.f[ichunk], lims_, NestState(p, segs, state_), NestedQuad(algs), cacheval[ichunk]; kwargs...).u
                    end
                end
                return nothing
            end, f.y, f.x, max_batch=f.max_batch)
        else
            return FunctionWrapper{FX,Tuple{TX,TP}}() do x, (p, lims, state)
                segs, lims_, state_ = limit_iterate(lims, state, x)
                len = segs[end] - segs[1]
                kwargs = haskey(kws, :abstol) ? merge(kws, (abstol=kws.abstol/len,)) : kws
                sol = do_solve(f, lims_, NestState(p, segs, state_), NestedQuad(algs), cacheval; kwargs...)
                return sol.u
            end
        end
    end
end

struct NestState{P,G,S}
    p::P
    segs::G
    state::S
end

function do_solve(f::F, lims::AbstractIteratedLimits, p_, alg::NestedQuad, cacheval; kws...) where {F}
    g, p, segs, state = if p_ isa NestState
        gg = if f isa NestedBatchIntegrand
            fx = eltype(f.x) === Nothing ? float(eltype(p_.segs))[] : f.x
            NestedBatchIntegrand(f.f, f.y, fx, max_batch=f.max_batch)
        else
            f
        end
        gg, p_.p, p_.segs, p_.state
    else
        seg, lim, sta = limit_iterate(lims)
        gg = if f isa BatchIntegrand
            fx = eltype(f.x) === Nothing ? typeof(interior_point(lims))[] : f.x
            BatchIntegrand(f.y, similar(f.x, eltype(eltype(fx))), max_batch=f.max_batch) do y, xs, (p, lims, state)
                resize!(fx, length(xs))
                f.f!(y, map!(x -> limit_iterate(lims, state, x), fx, xs), p)
            end
        elseif f isa NestedBatchIntegrand
            # this should be done recursively at the outermost level, but it is lazy.
            fx = eltype(f.x) === Nothing ? float(eltype(seg))[] : f.x
            NestedBatchIntegrand(f.f, f.y, fx, max_batch=f.max_batch)
        else
            f
        end
        gg, p_, seg, sta
    end
    dom = PuncturedInterval(segs)
    dim = ndims(lims) # constant propagation :)
    algs = alg.algs isa IntegralAlgorithm ? ntuple(i -> alg.algs, Val(dim)) : alg.algs
    nest = init_nest(g, cacheval[3], dom, p, lims, state, algs[1:dim-1], cacheval[1]; kws...)
    return do_solve(nest, dom, (p, lims, state), algs[dim], cacheval[2]; kws...)
end
=#
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
#=
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
=#

"""
    EvalCounter(::IntegralAlgorithm)

An algorithm which counts the evaluations used by another algorithm.
The count is stored in the `sol.numevals` field.
"""
struct EvalCounter{T<:IntegralAlgorithm} <: IntegralAlgorithm
    alg::T
end
#=
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
