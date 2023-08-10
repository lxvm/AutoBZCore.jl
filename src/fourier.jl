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

# I would like to just make the user pass in some `FourierRule` to indicate that their
# integrand accepts `FourierValue`s, however in previous versions we used the
# `FourierIntegrand` concept, which we now convert to a Fourier 'rule' if there is a benefit
# to doing so. (i.e. multidimensional evaluation of Fourier series on hierarchical grids.)
# Also, rules don't exist in the SciML interface we are copying, so we have to dispatch on
# the integrand

# needs to contain enough information to decide when/how to do threading/batching and how
# much workspace to pre-allocate, and it should also allow thread-unsafe integrands?
function workspace_allocate(s, x)
    dim = Val((d = ndims(s)))
    d === 1 && return ()
    cache = allocate(s, x[d], dim)
    d === 2 && return (cache,)
    return (workspace_allocate(contract!(cache, s, x, x[d], dim), x[1:d-1])..., cache)
end

struct FourierWorkspace{S,C}
    series::S
    cache::C
end

# Only the top-level workspace has an AbstractFourierSeries in the series field
# In the lower level workspaces the series field has a cache that can be contract!-ed into a series
function workspace_allocate(s::AbstractFourierSeries{1}, x=0.0, len::NTuple{1}=(1,))
    return FourierWorkspace(s, ntuple(_->nothing, Val(len[1])))
end
function workspace_allocate(s::AbstractFourierSeries{N}, x=0.0, len::NTuple{N}=ntuple(_->1,Val(N))) where{N}
    dim = Val(ndims(s))
    ws = ntuple(Val(len[N])) do n
        cache = allocate(s, x, dim)
        t = contract!(cache, s, x, dim)
        return FourierWorkspace(cache, workspace_allocate(t, x, Base.front(len)).cache)
    end
    return FourierWorkspace(s, ws)
end

function workspace_contract!(ws, x, dim, i=1)
    s = contract!(ws.cache[i].series, ws.series, x, dim)
    return FourierWorkspace(s, ws.cache[i].cache)
end

# evaluate a Fourier series using the workspace storage
workspace_evaluate(ws, x::NTuple{1}) = evaluate(ws.series, x)
function workspace_evaluate(ws, x::NTuple{N}) where {N}
    workspace_evaluate(workspace_contract!(ws, x[N], Val(N)), x[1:N-1])
end
workspace_evaluate(ws, x) = workspace_evaluate(ws, promote(x...))

struct FourierIntegrand{F,P,S,C}
    f::ParameterIntegrand{F,P}
    w::FourierWorkspace{S,C}
end

function FourierIntegrand(f, w::FourierWorkspace, args...; kws...)
    return FourierIntegrand(ParameterIntegrand(f, args...; kws...), w)
end
function FourierIntegrand(f, s::AbstractFourierSeries, args...; kws...)
    return FourierIntegrand(f, workspace_allocate(s), args...; kws...)
end

function Base.getproperty(f::FourierIntegrand, name::Symbol)
    name === :f && return getfield(f, :f).f
    name === :p && return getfield(f, :f).p
    return getfield(f, name)
end

function (s::IntegralSolver{<:FourierIntegrand})(args...; kwargs...)
    p = MixedParameters(args...; kwargs...)
    sol = do_solve(s, p)
    return sol.u
end

function remake_cache(f::FourierIntegrand, dom, p, alg, cacheval, kwargs)
    new = FourierIntegrand(f.f, f.w)
    return remake_integrand_cache(new, dom, merge(f.p, p), alg, cacheval, kwargs)
end

# FourierIntegrands should expect a FourierValue as input

struct FourierValue{X,S}
    x::X
    s::S
end
Base.convert(::Type{T}, f::FourierValue) where {T<:FourierValue} = T(f.x,f.s)

Base.zero(::Type{FourierValue{X,S}}) where {X,S} = FourierValue(zero(X), zero(S))
Base.:*(B::AbstractMatrix, f::FourierValue) = FourierValue(B*f.x, f.s)

(f::FourierIntegrand)(x::FourierValue, p=()) = getfield(f, :f)(x, p)
# fallback evaluator of FourierIntegrand for algorithms without specialized rules
(f::FourierIntegrand)(x, p=()) = f(FourierValue(x, workspace_evaluate(f.w, x)), p)

# PTR rules

# no symmetries
struct FourierPTR{N,T,S,X} <: AbstractArray{Tuple{AutoSymPTR.One,FourierValue{SVector{N,T},S}},N}
    s::Array{S,N}
    p::AutoSymPTR.PTR{N,T,X}
end

function fourier_ptr!(vals::AbstractArray{T,1}, w::FourierWorkspace, x::AbstractVector) where {T}
    p = period(w.series, 1)
    if length(w.cache) === 1
        for (i, y) in zip(eachindex(vals), x)
            @inbounds vals[i] = w.series(p*y)
        end
    else
        # we batch for memory locality in vals array on each thread
        Threads.@threads for (vrange, ichunk) in chunks(axes(vals, 1), length(w.cache), :batch)
            for i in vrange
                @inbounds vals[i] = w.series(p*x[i])
            end
        end
    end
    return vals
end
function fourier_ptr!(vals::AbstractArray{T,d}, w::FourierWorkspace, x::AbstractVector) where {T,d}
    p = period(w.series, d)
    if length(w.cache) === 1
        for (y, v) in zip(x, eachslice(vals, dims=d))
            fourier_ptr!(v, workspace_contract!(w, p*y, Val(d)), x)
        end
    else
        # we batch for memory locality in vals array on each thread
        Threads.@threads for (vrange, ichunk) in chunks(axes(vals, d), length(w.cache), :batch)
            for i in vrange
                ws = workspace_contract!(w, p*x[i], Val(d), ichunk)
                fourier_ptr!(view(vals, ntuple(_->(:),Val(d-1))..., i), ws, x)
            end
        end
    end
    return vals
end

function FourierPTR(w::FourierWorkspace, ::Type{T}, ndim, npt) where {T}
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

#=
stateful_workspace_evaluate(ws::FourierWorkspace, x::NTuple{1}) = workspace_evaluate(ws, x), (ws,)
function stateful_workspace_evaluate(ws::FourierWorkspace, x::NTuple{N}) where {N}
    w = workspace_contract!(ws, x[N], Val(N))
    val, state = stateful_workspace_evaluate(w, x[1:N-1])
    return val, (state..., ws)
end

function lazyeval(s, state, prev::NTuple{1}, u::NTuple{1})
    if prev[1] != u[1]
        return stateful_workspace_evaluate(state[1], map(*, u, period(state[1].series)))
    else
        return s, state
    end
end
function lazyeval(s, state, prev::NTuple{N}, u::NTuple{N}) where {N}
    if prev[N] != u[N]
        stateful_workspace_evaluate(state[N], map(*, u, period(state[N].series)))
    else
        val, new_state = lazyeval(s, state[1:N-1], prev[1:N-1], u[1:N-1])
        return val, (new_state..., state[N])
    end
end
# TODO think of ways to parallelize this by precomputing output indices
function fourier_symptr!(vals, w::FourierWorkspace, x::AbstractVector, wsym::AbstractArray, nsym)
    n = 0
    prev = ntuple(j -> zero(eltype(x)), Val(ndims(wsym)))
    s, state = stateful_workspace_evaluate(w, prev)
    for i in CartesianIndices(wsym)
        n < nsym || break
        iszero(wsym[i]) && continue
        u = ntuple(j -> x[i[j]], Val(ndims(wsym)))
        s, state = lazyeval(s, state, prev, u)
        vals[n += 1] = (wsym[i], FourierValue(u, s))
        prev = u
    end
end
=#


function _fourier_symptr!(vals::AbstractVector, w::FourierWorkspace, x::AbstractVector, wsym::AbstractVector, ::Tuple{}, offset, a, b, z, npt)
    p = period(w.series, 1)
    # offset -= 1
    o = offset-a
    if (len = length(w.cache)) === 1 # || len <= w.basecasesize[1]
        for i in a:b # 1:npt # a:b
            wi = wsym[i]
            iszero(wi) && (offset -= 1; continue)
            y = x[i]
            # @show i+o, y, wi
            vals[i+o] = (wi, FourierValue(SVector(y, z...), w.series(p*y)))
        end
    else
        # since the ibz is convex we scatter points over threads to distribute the workload
        Threads.@threads for (vrange, ichunk) in chunks(1:npt, len, :scatter)
            for i in vrange
                wi = wsym[i]
                iszero(wi) && (offset -= 1; continue)
                y = x[i]
                vals[i+offset] = (wi, FourierValue(SVector(y, z...),w.series(p*y)))
            end
        end
    end
    return vals
end
function _fourier_symptr!(vals::AbstractVector, w::FourierWorkspace, x::AbstractVector, wsym::AbstractArray, flags, offset, a, b, z, npt)
    d = ndims(wsym)
    p = period(w.series, d)
    flag, f = flags[begin:end-1], flags[end]
    if (len = length(w.cache)) === 1 # || len <= w.basecasesize[d]
        for i in a:b # 1:npt # a:b
            fi = f[i]
            fi[1] == 0 && continue
            y = x[i]
            ws = workspace_contract!(w, p*y, Val(d))
            _fourier_symptr!(vals, ws, x, selectdim(wsym, d, i), flag, fi..., (y, z...), npt)
        end
    else
        # since the ibz is convex we scatter points over threads to distribute the workload
        Threads.@threads for (vrange, ichunk) in chunks(1:npt, len, :scatter)
            for i in vrange
                fi = f[i]
                fi[1] == 0 && continue
                y = x[i]
                ws = workspace_contract!(w, p*y, Val(d), ichunk)
                _fourier_symptr!(vals, ws, x, selectdim(wsym, d, i), flag, fi..., (y, z...), npt)
            end
        end
    end
    return vals
end

function fourier_symptr!(wxs, w, u, wsym, flags, npt)
    flag, f = flags[begin:end-1], flags[end]
    return _fourier_symptr!(wxs, w, u, wsym, flag, f[]..., (), npt)
end

function FourierMonkhorstPack(w::FourierWorkspace, ::Type{T}, ndim::Val{d}, npt, syms) where {d,T}
    # unitless quadrature weight/node, but unitful value to Fourier series
    u = AutoSymPTR.ptrpoints(typeof(float(real(one(T)))), npt)
    s = workspace_evaluate(w, ntuple(_->zero(T), ndim))
    wsym, flags, nsym = AutoSymPTR.symptr_rule(npt, ndim, syms)
    wxs = Vector{Tuple{eltype(wsym),FourierValue{SVector{d,eltype(u)},typeof(s)}}}(undef, nsym)
    fourier_symptr!(wxs, w, u, wsym, flags, npt)
    return FourierMonkhorstPack(npt, length(syms), wxs)
end

# indexing
Base.getindex(rule::FourierMonkhorstPack, i::Int) = rule.wxs[i]

# iteration
Base.eltype(::Type{FourierMonkhorstPack{d,T,S}}) where {d,T,S} = Tuple{Int64,FourierValue{SVector{d,T},S}}
Base.length(r::FourierMonkhorstPack) = length(r.wxs)
Base.iterate(rule::FourierMonkhorstPack, args...) = iterate(rule.wxs, args...)

rule_type(::FourierMonkhorstPack{d,T,S}) where {d,T,S} = FourierValue{SVector{d,T},S}

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

function AutoSymPTR.nextrule(p::FourierMonkhorstPack{d,T}, r::FourierMonkhorstPackRule) where {d,T}
    return FourierMonkhorstPack(r.s, T, Val(d), p.npt+r.m.Δn, r.m.syms)
end

# dispatch on algorithms

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


# IAI rules
