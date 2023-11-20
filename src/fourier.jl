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
struct FourierIntegrand{F,P,W,N,I}
    f::I
    w::W
    nest::N
end

function FourierIntegrand(f::ParameterIntegrand{F,P}, w::FourierWorkspace) where {F,P}
    return FourierIntegrand{F,P,typeof(w),Nothing,typeof(f)}(f, w, nothing)
end
function FourierIntegrand(f::ParameterIntegrand{F,P}, w::FourierWorkspace, nest::NestedBatchIntegrand{<:ParameterIntegrand{F}}) where {F,P}
    return FourierIntegrand{F,P,typeof(w),typeof(nest),typeof(f)}(f, w, nest)
end
function FourierIntegrand(f::ParameterIntegrand{F,P}, w::FourierWorkspace, nest::ParameterIntegrand{F}) where {F,P}
    return FourierIntegrand{F,P,typeof(w),typeof(nest),typeof(f)}(f, w, nest)
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
    return FourierIntegrand(f, workspace_allocate_vec(s, period(s)), args...; kws...)
end

# similar to workspace_allocate, but more type-stable because of loop unrolling and vector types
function workspace_allocate_vec(s::AbstractFourierSeries{N}, x::NTuple{N,Any}, len::NTuple{N,Integer}=ntuple(one,Val(N))) where {N}
    # Only the top-level workspace has an AbstractFourierSeries in the series field
    # In the lower level workspaces the series field has a cache that can be contract!-ed
    # into a series
    dim = Val(N)
    if N == 1
        c = FourierSeriesEvaluators.allocate(s, x[N], dim)
        ws = Vector{typeof(c)}(undef, len[N])
        ws[1] = c
        for n in 2:len[N]
            ws[n] = FourierSeriesEvaluators.allocate(s, x[N], dim)
        end
    else
        c = FourierSeriesEvaluators.allocate(s, x[N], dim)
        t = FourierSeriesEvaluators.contract!(c, s, x[N], dim)
        c_ = FourierWorkspace(c, workspace_allocate_vec(t, x[1:N-1], len[1:N-1]).cache)
        ws = Vector{typeof(c_)}(undef, len[N])
        ws[1] = c_
        for n in 2:len[N]
            _c = FourierSeriesEvaluators.allocate(s, x[N], dim)
            _t = FourierSeriesEvaluators.contract!(_c, s, x[N], dim)
            ws[n] = FourierWorkspace(_c, workspace_allocate_vec(_t, x[1:N-1], len[1:N-1]).cache)
        end
    end
    return FourierWorkspace(s, ws)
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

function init_fourier_rule(w::FourierWorkspace, dom::Basis, alg::MonkhorstPack)
    @assert ndims(w.series) == ndims(dom)
    if alg.syms === nothing
        return FourierPTR(w, eltype(dom), Val(ndims(dom)), alg.npt)
    else
        return FourierMonkhorstPack(w, eltype(dom), Val(ndims(dom)), alg.npt, alg.syms)
    end
end

function init_cacheval(f::FourierIntegrand, dom, p, alg::MonkhorstPack)
    dom isa Basis || throw(ArgumentError("MonkhorstPack only supports Basis for domain. Please open an issue."))
    rule = init_fourier_rule(f.w, dom, alg)
    buf = init_buffer(f, alg.nthreads)
    return (rule=rule, buffer=buf)
end

function init_fourier_rule(w::FourierWorkspace, dom::Basis, alg::AutoSymPTRJL)
    @assert ndims(w.series) == ndims(dom)
    return FourierMonkhorstPackRule(w, alg.syms, alg.a, alg.nmin, alg.nmax, alg.n₀, alg.Δn)
end
function init_cacheval(f::FourierIntegrand, dom, p, alg::AutoSymPTRJL)
    dom isa Basis || throw(ArgumentError("MonkhorstPack only supports Basis for domain. Please open an issue."))
    rule = init_fourier_rule(f.w, dom, alg)
    cache = AutoSymPTR.alloc_cache(eltype(dom), Val(ndims(dom)), rule)
    buffer = init_buffer(f, alg.nthreads)
    return (rule=rule, cache=cache, buffer=buffer)
end

function init_fourier_rule(s::AbstractFourierSeries, bz::SymmetricBZ, alg::PTR)
    dom = Basis(bz.B)
    return FourierMonkhorstPack(s, eltype(dom), Val(ndims(dom)), alg.npt, bz.syms)
end

function assemble_pintegrand(f::FourierIntegrand, p, dom, rule)
    g = f.nest isa NestedBatchIntegrand ? f.nest : f.f
    return assemble_pintegrand(g, p, dom, rule)
end

function do_solve_evalcounter(f::FourierIntegrand, dom, p, alg, cacheval; kws...)
    if f.nest isa NestedBatchIntegrand
        w = wrap_with_counter(f.f)
        u = NestedBatchIntegrand(w, f.y, f.x, max_batch=f.max_batch)
        FourierIntegrand(WrapperCounter(f.f), f.w, u)
        sol = do_solve(g, dom, p, alg, cacheval; kws...)
        n = sum(s -> s.numevals, FlatView(u), init=0)
        return IntegralSolution(sol.u, sol.resid, sol.retcode, iszero(n) ? -1 : n)
    else
        n::Int = 0
        function g(args...; kwargs...)
            n += 1
            return f.f.f(args...; kwargs...)
        end
        h = FourierIntegrand(ParameterIntegrand{typeof(g)}(g, f.f.p), f.w)
        sol = do_solve(h, dom, p, alg, cacheval; kws...)
        return IntegralSolution(sol.u, sol.resid, true, n)
    end
end

function init_nested_cacheval(f::FourierIntegrand, p, segs, lims, state, alg::IntegralAlgorithm)
    g = f.nest isa NestedBatchIntegrand ? f.nest : (x, p) -> f(FourierValue(x, f.w(x)), p)
    return init_nested_cacheval(g, p, segs, lims, state, alg)
end
function init_nested_cacheval(f::FourierIntegrand, p, segs, lims, state, alg_::IntegralAlgorithm, algs_::IntegralAlgorithm...)
    if f.nest isa NestedBatchIntegrand
        dim = ndims(lims)
        algs = (alg_, algs_...)
        alg = algs[dim]
        dom = PuncturedInterval(segs)
        a, b = segs[1], segs[2]
        mid = (a+b)/2 # sample point that should be safe to evaluate
        next = limit_iterate(lims, state, mid) # see what the next limit gives
        nest = init_nested_cacheval(FourierIntegrand(f.f, f.w, f.nest.f[1]), p, next..., algs[1:dim-1]...)
        cacheval = init_cacheval(BatchIntegrand(nothing, f.nest.y, f.nest.x, max_batch=f.nest.max_batch), dom, p, alg)
        fx = eltype(f.nest.x) === Nothing ? typeof(mid)[] : f.nest.x
        return ((fx, map(n -> deepcopy(nest), f.nest.f)), cacheval, nest[3]*mid)
     else
        g = (x, p) -> f(FourierValue(x, f.w(x)), p)
        return init_nested_cacheval(g, p, segs, lims, state, alg_, algs_...)
    end
end

function assemble_nested_integrand(f::FourierIntegrand, fxx, dom, p, lims, state, alg::Tuple{}, cacheval; kws...)
    if f.nest isa NestedBatchIntegrand
        nchunk = min(length(f.nest.f), length(f.w.cache))
        return BatchIntegrand(f.nest.y, cacheval[1], max_batch=f.nest.max_batch) do y, x, p
            Threads.@threads for ichunk in 1:min(nchunk, length(x))
                for (i, j) in zip(getchunk(x, ichunk, nchunk, :scatter), getchunk(y, ichunk, nchunk, :scatter))
                    xi = x[i]
                    v = FourierValue(limit_iterate(lims, state, xi), workspace_evaluate!(f.w, xi, ichunk))
                    y[j] = f.nest.f[ichunk](v, p)
                end
            end
            return nothing
        end
        #=
        return nested_to_batched(f.nest) do i, w, x, p
            n = 0
            while (len = length(FlatView(w.nest.f[n+=1]))) < i
                i -= len
            end
            v = FourierValue(limit_iterate(lims, state, x), workspace_evaluate!(f.w[i], x, i))
            f.nest.f[i](v, p)
        end
        =#
    else
        return (x, p) -> begin
            v = FourierValue(limit_iterate(lims, state, x), workspace_evaluate!(f.w, x))
            return f.f(v, p)
        end
    end
end
function assemble_nested_integrand(f::FourierIntegrand, fxx, dom, p, lims, state, algs, cacheval; kws_...)
    kws = NamedTuple(kws_)
    xx = float(oneunit(eltype(dom)))
    TX = typeof(xx)
    TP = typeof(p)
    err = integralerror(last(algs), fxx)
    if f.nest isa NestedBatchIntegrand
        # TODO: make this function return a NestedIntegrand
        nchunks = min(length(f.nest.f), length(f.w.cache))
        return BatchIntegrand(FunctionWrapper{Nothing,Tuple{typeof(f.nest.y),typeof(cacheval[1]),TP}}() do y, x, p
            Threads.@threads for ichunk in 1:min(nchunks, length(x))
                for (i, j) in zip(getchunk(x, ichunk, nchunks, :scatter), getchunk(y, ichunk, nchunks, :scatter))
                    xi = x[i]
                    segs, lims_, state_ = limit_iterate(lims, state, xi)
                    len = segs[end] - segs[1]
                    kwargs = haskey(kws, :abstol) ? merge(kws, (abstol=kws.abstol/len,)) : kws
                    fx = FourierIntegrand(f.f, workspace_contract!(f.w, xi, ichunk), f.nest.f[ichunk])
                    y[j] = do_solve(fx, StatefulLimits(segs, state_, lims_), p, NestedQuad(algs), cacheval[2][ichunk]; kwargs...).u
                end
            end
            return nothing
        end, f.nest.y, cacheval[1], max_batch=f.nest.max_batch)
    else
        g = f.nest === nothing ? f.f : f.nest
        f_ = FunctionWrapper{IntegralSolution{typeof(fxx),typeof(err)},Tuple{TX,TP}}() do x, p
            segs, lims_, state_ = limit_iterate(lims, state, x)
            fx = FourierIntegrand(g, workspace_contract!(f.w, x))
            len = segs[end] - segs[1]
            kwargs = haskey(kws, :abstol) ? merge(kws, (abstol=kws.abstol/len,)) : kws
            return do_solve(fx, StatefulLimits(segs, state_, lims_), p, NestedQuad(algs), cacheval; kwargs...)
        end
        return NestedIntegrand(f_)
    end
end
