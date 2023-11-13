# Here we provide optimizations of multidimensional Fourier series evaluation for the
# various algorithms. It could be a package extension, but we keep it in the main library
# because it provides the infrastructure of the main application of library

# In multiple dimensions, these specialized rules can provide a benefit over batch
# integrands since the multidimensional structure of the quadrature rule can be lost when
# batching many points together and passing them to an integrand to solve simultaneously. In
# some cases we can also cache the rule with series evaluations and apply it to different
# integrands, which again could only be achieved with a batched and vector-valued integrand.
# The ethos of this package is to let the user provide the kernel for the integrand and to
# have the library take care of the details of fast evaluation and such. Automating batched
# and vector-valued integrands is another worthwhile approach, but it is not well
# established in existing Julia libraries or Integrals.jl, so in the meantime I strive to
# provide these efficient rules for Wannier interpolation. In the long term, the batched and
# vector-valued approach will allow distributed computing and other benefits that are beyond
# the scope of what this package aims to provide.

# We use the pattern of allowing the user to pass a container with the integrand, Fourier
# series and workspace, and use dispatch to enable the optimizations

# the nested batched integrand is optional, but when included it allows for thread-safe parallelization
struct FourierIntegrand{F,P,W,N}
    f::ParameterIntegrand{F,P}
    w::W
    nest::N
    function FourierIntegrand(f::ParameterIntegrand{F,P}, w::FourierWorkspace) where {F,P}
        return new{F,P,typeof(w),Nothing}(f, w, nothing)
    end
    function FourierIntegrand(f::ParameterIntegrand{F,P}, w::FourierWorkspace, nest::NestedBatchIntegrand{<:ParameterIntegrand{F}}) where {F,P}
        return new{F,P,typeof(w),typeof(nest)}(f, w, nest)
    end
    function FourierIntegrand(f::ParameterIntegrand{F,P}, w::FourierWorkspace, nest::ParameterIntegrand{F}) where {F,P}
        return new{F,P,typeof(w),typeof(nest)}(f, w, nest)
    end
end

"""
    FourierIntegrand(f, w::FourierWorkspace, args...; kws...)

Constructs an integrand of the form `f(FourierValue(x,w(x)), args...; kws...)` where the
Fourier series in `w` is evaluated efficiently, i.e. one dimension at a time, with
compatible algorithms. `f` should accept parameters as arguments and keywords, similar to a
[`ParameterIntegrand`](@ref) although the first argument to `f` will always be a
[`FourierValue`](@ref).
"""
function FourierIntegrand(f, w::FourierWorkspace, args...; kws...)
    return FourierIntegrand(ParameterIntegrand(f, args...; kws...), w)
end

"""
    FourierIntegrand(f, s::AbstractFourierSeries, args...; kws...)

Outer constructor for `FourierIntegrand` that wraps the Fourier series `s` into a
single-threaded `FourierWorkspace`.
"""
function FourierIntegrand(f, s::AbstractFourierSeries, args...; kws...)
    return FourierIntegrand(f, workspace_allocate(s, period(s)), args...; kws...)
end

function (s::IntegralSolver{<:FourierIntegrand})(args...; kwargs...)
    p = MixedParameters(args...; kwargs...)
    sol = solve_p(s, p)
    return sol.u
end

function remake_cache(f::FourierIntegrand, dom, p, alg, cacheval, kwargs)
    # TODO decide what to do with the nest, since it may have allocated
    fp = ParameterIntegrand(f.f.f)
    new = f.nest === nothing ? FourierIntegrand(fp, f.w) : FourierIntegrand(fp, f.w, f.nest)
    return remake_integrand_cache(new, dom, merge(f.f.p, p), alg, cacheval, kwargs)
end

# FourierIntegrands should expect a FourierValue as input

"""
    FourierValue(x, s)

A container used by [`FourierIntegrand`](@ref) to pass a point, `x`, and the value of a
Fourier series evaluated at the point, `s`, to integrands. The properties `x` and `s` of a
`FourierValue` store the point and evaluated series, respectively.
"""
struct FourierValue{X,S}
    x::X
    s::S
end
Base.convert(::Type{T}, f::FourierValue) where {T<:FourierValue} = T(f.x,f.s)

Base.zero(::Type{FourierValue{X,S}}) where {X,S} = FourierValue(zero(X), zero(S))
Base.:*(B::AbstractMatrix, f::FourierValue) = FourierValue(B*f.x, f.s)

(f::FourierIntegrand)(x::FourierValue, p) = f.f(x, p)
# fallback evaluator of FourierIntegrand for algorithms without specialized rules
(f::FourierIntegrand)(x, p) = f(FourierValue(x, f.w(x)), p)

# PTR rules

# no symmetries
struct FourierPTR{N,T,S,X} <: AbstractArray{Tuple{AutoSymPTR.One,FourierValue{SVector{N,T},S}},N}
    s::Array{S,N}
    p::AutoSymPTR.PTR{N,T,X}
end

function fourier_ptr!(vals::AbstractArray{T,1}, w::FourierWorkspace, x::AbstractVector) where {T}
    t = period(w.series, 1)
    if length(w.cache) === 1
        for (i, y) in zip(eachindex(vals), x)
            @inbounds vals[i] = workspace_evaluate!(w, t*y)
        end
    else
        # we batch for memory locality in vals array on each thread
        Threads.@threads for (vrange, ichunk) in chunks(axes(vals, 1), length(w.cache), :batch)
            for i in vrange
                @inbounds vals[i] = workspace_evaluate!(w, t*x[i], ichunk)
            end
        end
    end
    return vals
end
function fourier_ptr!(vals::AbstractArray{T,d}, w::FourierWorkspace, x::AbstractVector) where {T,d}
    t = period(w.series, d)
    if length(w.cache) === 1
        for (y, v) in zip(x, eachslice(vals, dims=d))
            fourier_ptr!(v, workspace_contract!(w, t*y), x)
        end
    else
        # we batch for memory locality in vals array on each thread
        Threads.@threads for (vrange, ichunk) in chunks(axes(vals, d), length(w.cache), :batch)
            for i in vrange
                ws = workspace_contract!(w, t*x[i], ichunk)
                fourier_ptr!(view(vals, ntuple(_->(:),Val(d-1))..., i), ws, x)
            end
        end
    end
    return vals
end

function FourierPTR(w::FourierWorkspace, ::Type{T}, ndim, npt) where {T}
    FourierSeriesEvaluators.isinplace(w.series) && throw(ArgumentError("inplace series not supported for PTR - please file a bug report"))
    # unitless quadrature weight/node, but unitful value to Fourier series
    p = AutoSymPTR.PTR(typeof(float(real(one(T)))), ndim, npt)
    s = workspace_evaluate(w, ntuple(_->zero(T), ndim))
    vals = similar(p, typeof(s))
    fourier_ptr!(vals, w, p.x)
    return FourierPTR(vals, p)
end

# Array interface
Base.size(r::FourierPTR) = size(r.s)
function Base.getindex(r::FourierPTR{N}, i::Vararg{Int,N}) where {N}
    w, x = r.p[i...]
    return w, FourierValue(x, r.s[i...])
end

# iteration
function Base.iterate(p::FourierPTR)
    next1 = iterate(p.s)
    next1 === nothing && return nothing
    next2 = iterate(p.p)
    next2 === nothing && return nothing
    s, state1 = next1
    (w, x), state2 = next2
    return (w, FourierValue(x, s)), (state1, state2)
end
Base.isdone(::FourierPTR, state) = any(isnothing, state)
function Base.iterate(p::FourierPTR, state)
    next1 = iterate(p.s, state[1])
    next1 === nothing && return nothing
    next2 = iterate(p.p, state[2])
    next2 === nothing && return nothing
    s, state1 = next1
    (w, x), state2 = next2
    return (w, FourierValue(x, s)), (state1, state2)
end

function (rule::FourierPTR)(f, B::Basis, buffer=nothing)
    arule = AutoSymPTR.AffineQuad(rule, B)
    return AutoSymPTR.quadsum(arule, f, arule.vol / length(rule), buffer)
end

# SymPTR rules
struct FourierMonkhorstPack{d,W,T,S}
    npt::Int64
    nsyms::Int64
    wxs::Vector{Tuple{W,FourierValue{SVector{d,T},S}}}
end

function _fourier_symptr!(vals::AbstractVector, w::FourierWorkspace, x::AbstractVector, npt, wsym, ::Tuple{}, idx, coord, offset)
    t = period(w.series, 1)
    o = offset-1
    # we can't parallelize the inner loop without knowing the offsets of each contiguous
    # chunk, which would require a ragged array to store. We would be better off with
    # changing the symptr algorithm to compute a convex ibz
    # but for 3D grids this inner loop should be a large enough base case to make
    # parallelizing worth it, although the workloads will vary piecewise linearly as a
    # function of the slice, so we should distribute points using :scatter
    n = 0
    for i in 1:npt
        @inbounds wi = wsym[i, idx...]
        iszero(wi) && continue
        @inbounds xi = x[i]
        vals[o+(n+=1)] = (wi, FourierValue((xi, coord...), workspace_evaluate!(w, t*xi)))
    end
    return vals
end
function _fourier_symptr!(vals::AbstractVector, w::FourierWorkspace, x::AbstractVector, npt, wsym, flags, idx, coord, offset)
    d = ndims(w.series)
    t = period(w.series, d)
    flag, f = flags[begin:end-1], flags[end]
    if (len = length(w.cache)) === 1 # || len <= w.basecasesize[d]
        for i in 1:npt
            @inbounds(fi = f[i, idx...]) == 0 && continue
            @inbounds xi = x[i]
            ws = workspace_contract!(w, t*xi)
            _fourier_symptr!(vals, ws, x, npt, wsym, flag, (i, idx...), (xi, coord...), fi)
        end
    else
        # since we don't know the distribution of ibz nodes, other than that it will be
        # piecewise linear, our best chance for a speedup from parallelizing is to scatter
        Threads.@threads for (vrange, ichunk) in chunks(1:npt, len, :scatter)
            for i in vrange
                @inbounds(fi = f[i, idx...]) == 0 && continue
                @inbounds xi = x[i]
                ws = workspace_contract!(w, t*xi, ichunk)
                _fourier_symptr!(vals, ws, x, npt, wsym, flag, (i, idx...), (xi, coord...), fi)
            end
        end
    end
    return vals
end

function fourier_symptr!(wxs, w, u, npt, wsym, flags)
    flag, f = flags[begin:end-1], flags[end]
    return _fourier_symptr!(wxs, w, u, npt, wsym, flag, (), (), f[])
end

function FourierMonkhorstPack(w::FourierWorkspace, ::Type{T}, ndim::Val{d}, npt, syms) where {d,T}
    # unitless quadrature weight/node, but unitful value to Fourier series
    FourierSeriesEvaluators.isinplace(w.series) && throw(ArgumentError("inplace series not supported for PTR - please file a bug report"))
    u = AutoSymPTR.ptrpoints(typeof(float(real(one(T)))), npt)
    s = w(map(*, period(w.series), ntuple(_->zero(T), ndim)))
    # the bottleneck is likely to be symptr_rule, which is not a fast or parallel algorithm
    wsym, flags, nsym = AutoSymPTR.symptr_rule(npt, ndim, syms)
    wxs = Vector{Tuple{eltype(wsym),FourierValue{SVector{d,eltype(u)},typeof(s)}}}(undef, nsym)
    # fourier_symptr! may be worth parallelizing for expensive Fourier series, but may not
    # be the bottleneck
    fourier_symptr!(wxs, w, u, npt, wsym, flags)
    return FourierMonkhorstPack(npt, length(syms), wxs)
end

# indexing
Base.getindex(rule::FourierMonkhorstPack, i::Int) = rule.wxs[i]

# iteration
Base.eltype(::Type{FourierMonkhorstPack{d,W,T,S}}) where {d,W,T,S} = Tuple{W,FourierValue{SVector{d,T},S}}
Base.length(r::FourierMonkhorstPack) = length(r.wxs)
Base.iterate(rule::FourierMonkhorstPack, args...) = iterate(rule.wxs, args...)

rule_type(::FourierMonkhorstPack{d,W,T,S}) where {d,W,T,S} = FourierValue{SVector{d,T},S}

function (rule::FourierMonkhorstPack{d})(f, B::Basis, buffer=nothing) where d
    arule = AutoSymPTR.AffineQuad(rule, B)
    return AutoSymPTR.quadsum(arule, f, arule.vol / (rule.npt^d * rule.nsyms), buffer)
end

# rule definition

struct FourierMonkhorstPackRule{S,M}
    s::S
    m::M
end

function FourierMonkhorstPackRule(s, syms, a, nmin, nmax, n₀, Δn)
    mp = AutoSymPTR.MonkhorstPackRule(syms, a, nmin, nmax, n₀, Δn)
    return FourierMonkhorstPackRule(s, mp)
end
AutoSymPTR.nsyms(r::FourierMonkhorstPackRule) = AutoSymPTR.nsyms(r.m)

function (r::FourierMonkhorstPackRule)(::Type{T}, v::Val{d}) where {T,d}
    if r.m.syms isa Nothing
        FourierPTR(r.s, T, v, r.m.n₀)
    else
        FourierMonkhorstPack(r.s, T, v, r.m.n₀, r.m.syms)
    end
end

function AutoSymPTR.nextrule(p::FourierPTR{d,T}, r::FourierMonkhorstPackRule) where {d,T}
    return FourierPTR(r.s, T, Val(d), length(p.p.x)+r.m.Δn)
end

function AutoSymPTR.nextrule(p::FourierMonkhorstPack{d,W,T}, r::FourierMonkhorstPackRule) where {d,W,T}
    return FourierMonkhorstPack(r.s, T, Val(d), p.npt+r.m.Δn, r.m.syms)
end

# dispatch on PTR algorithms

function init_buffer(f::FourierIntegrand, len)
    return f.nest isa NestedBatchIntegrand ? Vector{eltype(f.nest.y)}(undef, len) : nothing
end

rule_type(::FourierPTR{N,T,S}) where {N,T,S} = FourierValue{SVector{N,T},S}
function init_fourier_rule(w::FourierWorkspace, dom::Basis, alg::MonkhorstPack)
    @assert ndims(w.series) == ndims(dom)
    if alg.syms === nothing
        return FourierPTR(w, eltype(dom), Val(ndims(dom)), alg.npt)
    else
        return FourierMonkhorstPack(w, eltype(dom), Val(ndims(dom)), alg.npt, alg.syms)
    end
end
function init_cacheval(f::FourierIntegrand, dom::Basis , p, alg::MonkhorstPack)
    rule = init_fourier_rule(f.w, dom, alg)
    buf = init_buffer(f, alg.nthreads)
    return (rule=rule, buffer=buf)
end

function init_fourier_rule(w::FourierWorkspace, dom::Basis, alg::AutoSymPTRJL)
    @assert ndims(w.series) == ndims(dom)
    return FourierMonkhorstPackRule(w, alg.syms, alg.a, alg.nmin, alg.nmax, alg.n₀, alg.Δn)
end
function init_cacheval(f::FourierIntegrand, dom::Basis, p, alg::AutoSymPTRJL)
    rule = init_fourier_rule(f.w, dom, alg)
    cache = AutoSymPTR.alloc_cache(eltype(dom), Val(ndims(dom)), rule)
    buffer = init_buffer(f, alg.nthreads)
    return (rule=rule, cache=cache, buffer=buffer)
end

function init_fourier_rule(s::AbstractFourierSeries, bz::SymmetricBZ, alg::PTR)
    dom = Basis(bz.B)
    return FourierMonkhorstPack(s, eltype(dom), Val(ndims(dom)), alg.npt, bz.syms)
end

function nested_to_batched(f::FourierIntegrand, dom::Basis, rule)
    ys = f.nest.y/prod(ntuple(n -> oneunit(eltype(dom)), Val(ndims(dom))))
    xs = rule_type(rule)[]
    return BatchIntegrand(ys, xs, max_batch=f.nest.max_batch) do y, x, p
        # would be better to fully unwrap the nested structure, but this is one level
        nchunk = length(f.nest.f)
        Threads.@threads for ichunk in 1:min(nchunk, length(x))
            for (i, j) in zip(getchunk(x, ichunk, nchunk, :batch), getchunk(y, ichunk, nchunk, :batch))
                y[j] = FourierIntegrand(f.nest.f[ichunk], FourierWorkspace(nothing,nothing))(x[i], p)
            end
        end
        return nothing
    end
end

function do_solve(f::FourierIntegrand, dom, p, alg::MonkhorstPack, cacheval; kws...)
    g = f.nest isa NestedBatchIntegrand ? nested_to_batched(f, dom, cacheval.rule) : f.f
    return do_solve(g, dom, p, alg, cacheval; kws...)
end

function do_solve(f::FourierIntegrand, dom, p, alg::AutoSymPTRJL, cacheval; kws...)
    g = f.nest isa NestedBatchIntegrand ? nested_to_batched(f, dom, cacheval.cache[1]) : f.f
    return do_solve(g, dom, p, alg, cacheval; kws...)
end


# dispatch on IAI algorithms

function nested_cacheval(f::FourierIntegrand, p::P, algs, segs, lims, state, x, xs...) where {P}
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
        if f.nest isa NestedBatchIntegrand
            # Batch integrate the inner integral only
            cacheval = init_cacheval(BatchIntegrand(nothing, f.nest.y, f.nest.x, max_batch=f.nest.max_batch), dom, p, alg)
            return (nothing, cacheval, zero(eltype(f.nest.y))*mid)
        else
            fx = f(next,p)
            cacheval = init_cacheval((x, p) -> fx, dom, p, alg)
            return (nothing, cacheval, fx*mid)
        end
    elseif f.nest isa NestedBatchIntegrand
        algs_ = algs[1:dim-1]
        # numbered names to avoid type instabilities (we are not using dispatch, but the
        # compiler's optimization for the recursive function's argument types)
        nest0 = nested_cacheval(FourierIntegrand(f.f, f.w, f.nest.f[1]), p, algs_, next..., x, xs[1:dim-2]...)
        cacheval = init_cacheval(BatchIntegrand(nothing, f.nest.y, f.nest.x, max_batch=f.nest.max_batch), dom, p, alg)
        return (ntuple(n -> n == 1 ? nest0 : deepcopy(nest0), Val(length(f.nest.f))), cacheval, nest0[3]*mid)
    else
        algs_ = algs[1:dim-1]
        nest1 = nested_cacheval(f, p, algs_, next..., x, xs[1:dim-2]...)
        h = nest1[3]
        hx = h*mid
        # units may change for outer integral
        cacheval = init_cacheval((x, p) -> h, dom, p, alg)
        return (nest1, cacheval, hx)
    end
end

function init_nest(f::FourierIntegrand, fxx, dom, p,lims, state, algs, cacheval; kws_...)
    kws = NamedTuple(kws_)
    xx = float(oneunit(eltype(dom)))
    FX = typeof(fxx/xx)
    TX = typeof(xx)
    TP = Tuple{typeof(p),typeof(lims),typeof(state)}
    if algs isa Tuple{} # inner integral
        if f.nest isa NestedBatchIntegrand
            nchunk = min(length(f.nest.f), length(f.w.cache))
            return BatchIntegrand(f.nest.y, f.nest.x, max_batch=f.nest.max_batch) do y, x, (p, lims, state)
                Threads.@threads for ichunk in 1:min(nchunk, length(x))
                    for (i, j) in zip(getchunk(x, ichunk, nchunk, :scatter), getchunk(y, ichunk, nchunk, :scatter))
                        xi = x[i]
                        v = FourierValue(limit_iterate(lims, state, xi), workspace_evaluate!(f.w, xi, ichunk))
                        y[j] = f.nest.f[ichunk](v, p)
                    end
                end
                return nothing
            end
        else
            return (x, (p, lims, state)) -> begin
            # return FunctionWrapper{FX,Tuple{TX,TP}}() do x, (p, lims, state)
                v = FourierValue(limit_iterate(lims, state, x), workspace_evaluate!(f.w, x))
                return f.f(v, p)
            end
        end
    else
        if f.nest isa NestedBatchIntegrand
            nchunks = min(length(f.nest.f), length(f.w.cache))
            return BatchIntegrand(FunctionWrapper{Nothing,Tuple{typeof(f.nest.y),typeof(f.nest.x),TP}}() do y, x, (p, lims, state)
                Threads.@threads for ichunk in 1:min(nchunks, length(x))
                    for (i, j) in zip(getchunk(x, ichunk, nchunks, :scatter), getchunk(y, ichunk, nchunks, :scatter))
                        xi = x[i]
                        segs, lims_, state_ = limit_iterate(lims, state, xi)
                        len = segs[end] - segs[1]
                        kwargs = haskey(kws, :abstol) ? merge(kws, (abstol=kws.abstol/len,)) : kws
                        fx = FourierIntegrand(f.f, workspace_contract!(f.w, xi, ichunk), f.nest.f[ichunk])
                        y[j] = do_solve(fx, lims_, NestState(p, segs, state_), NestedQuad(algs), cacheval[ichunk]; kwargs...).u
                    end
                end
                return nothing
            end, f.nest.y, f.nest.x, max_batch=f.nest.max_batch)
        else
            g = f.nest === nothing ? f.f : f.nest
            return FunctionWrapper{FX,Tuple{TX,TP}}() do x, (p, lims, state)
                segs, lims_, state_ = limit_iterate(lims, state, x)
                fx = FourierIntegrand(g, workspace_contract!(f.w, x))
                len = segs[end] - segs[1]
                kwargs = haskey(kws, :abstol) ? merge(kws, (abstol=kws.abstol/len,)) : kws
                sol = do_solve(fx, lims_, NestState(p, segs, state_), NestedQuad(algs), cacheval; kwargs...)
                return sol.u
            end
        end
    end
end

function init_cacheval(f::FourierIntegrand, dom::AbstractIteratedLimits, p, alg::NestedQuad)
    algs = alg.algs isa IntegralAlgorithm ? ntuple(i -> alg.algs, Val(ndims(dom))) : alg.algs
    return nested_cacheval(f, p, algs, limit_iterate(dom)..., interior_point(dom)...)
end

function do_solve(f::FourierIntegrand, lims::AbstractIteratedLimits, p_, alg::NestedQuad, cacheval; kws...)
    p, segs, state = if p_ isa NestState
        p_.p, p_.segs, p_.state
    else
        seg, lim, sta = limit_iterate(lims)
        p_, seg, sta
    end
    dom = PuncturedInterval(segs)
    g = if f.nest isa NestedBatchIntegrand && eltype(f.nest.x) === Nothing
        FourierIntegrand(f.f, f.w, NestedBatchIntegrand(f.nest.f, f.nest.y, float(eltype(dom))[], max_batch=f.nest.max_batch))
    else
        f
    end
    (dim = ndims(lims)) == ndims(f.w.series) || throw(ArgumentError("variables in Fourier series don't match domain"))
    algs = alg.algs isa IntegralAlgorithm ? ntuple(i -> alg.algs, Val(dim)) : alg.algs
    nest = init_nest(g, cacheval[3], dom, p, lims, state, algs[1:dim-1], cacheval[1]; kws...)
    return do_solve(nest, dom, (p, lims, state), algs[dim], cacheval[2]; kws...)
end

function do_solve(f::FourierIntegrand, dom, p, alg::EvalCounter, cacheval; kws...)
    if f.nest isa NestedBatchIntegrand
        error("NestedBatchIntegrand not yet supported with EvalCounter")
    else
        n::Int = 0
        function g(args...; kwargs...)
            n += 1
            return f.f.f(args...; kwargs...)
        end
        h = FourierIntegrand(ParameterIntegrand{typeof(g)}(g, f.f.p), f.w)
        sol = do_solve(h, dom, p, alg.alg, cacheval; kws...)
        return IntegralSolution(sol.u, sol.resid, true, n)
    end
end

# method needed to resolve ambiguities
function do_solve(f::FourierIntegrand, bz::SymmetricBZ, p, alg::EvalCounter{<:AutoBZAlgorithm}, cacheval; kws...)
    return do_solve(f, bz, p, AutoBZEvalCounter(bz_to_standard(bz, alg.alg)...), cacheval; kws...)
end
