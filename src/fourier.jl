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


struct FourierIntegrand{F,P,S,C}
    f::ParameterIntegrand{F,P}
    w::FourierWorkspace{S,C}
end

function FourierIntegrand(f, w::FourierWorkspace, args...; kws...)
    return FourierIntegrand(ParameterIntegrand(f, args...; kws...), w)
end
function FourierIntegrand(f, s::AbstractFourierSeries, args...; kws...)
    return FourierIntegrand(f, workspace_allocate(s, period(s)), args...; kws...)
end

function (s::IntegralSolver{<:FourierIntegrand})(args...; kwargs...)
    p = MixedParameters(args...; kwargs...)
    sol = solve_p(s, p)
    return sol.u
end

function remake_cache(f::FourierIntegrand, dom, p, alg, cacheval, kwargs)
    new = FourierIntegrand(f.f.f, f.w)
    return remake_integrand_cache(new, dom, merge(f.f.p, p), alg, cacheval, kwargs)
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
(f::FourierIntegrand)(x, p=()) = f(FourierValue(x, f.w(x)), p)

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

function _fourier_symptr!(vals::AbstractVector, w::FourierWorkspace, x::AbstractVector, wsym::AbstractVector, ::Tuple{}, offset, a, b, z)
    t = period(w.series, 1)
    o = offset-a
    if (len = length(w.cache)) === 1 # || len <= w.basecasesize[1]
        for i in a:b
            wi = wsym[i]
            iszero(wi) && continue
            y = x[i]
            vals[i+o] = (wi, FourierValue((y, z...), workspace_evaluate!(w, t*y)))
        end
    else
        # since the ibz is convex we scatter points over threads to distribute the workload
        Threads.@threads for (vrange, ichunk) in chunks(a:b, len, :scatter)
            for i in vrange
                wi = wsym[i]
                iszero(wi) && continue
                y = x[i]
                vals[i+o] = (wi, FourierValue((y, z...), workspace_evaluate!(w, t*y, ichunk)))
            end
        end
    end
    return vals
end
function _fourier_symptr!(vals::AbstractVector, w::FourierWorkspace, x::AbstractVector, wsym::AbstractArray, flags, offset, a, b, z)
    d = ndims(wsym)
    t = period(w.series, d)
    flag, f = flags[begin:end-1], flags[end]
    if (len = length(w.cache)) === 1 # || len <= w.basecasesize[d]
        for i in a:b
            fi = f[i]
            fi[1] == 0 && continue
            y = x[i]
            ws = workspace_contract!(w, t*y)
            _fourier_symptr!(vals, ws, x, selectdim(wsym, d, i), flag, fi..., (y, z...))
        end
    else
        # since the ibz is convex we scatter points over threads to distribute the workload
        Threads.@threads for (vrange, ichunk) in chunks(a:b, len, :scatter)
            for i in vrange
                fi = f[i]
                fi[1] == 0 && continue
                y = x[i]
                ws = workspace_contract!(w, t*y, ichunk)
                _fourier_symptr!(vals, ws, x, selectdim(wsym, d, i), flag, fi..., (y, z...))
            end
        end
    end
    return vals
end

function fourier_symptr!(wxs, w, u, wsym, flags)
    flag, f = flags[begin:end-1], flags[end]
    return _fourier_symptr!(wxs, w, u, wsym, flag, f[]..., ())
end

function FourierMonkhorstPack(w::FourierWorkspace, ::Type{T}, ndim::Val{d}, npt, syms) where {d,T}
    # unitless quadrature weight/node, but unitful value to Fourier series
    u = AutoSymPTR.ptrpoints(typeof(float(real(one(T)))), npt)
    s = w(map(*, period(w.series), ntuple(_->zero(T), ndim)))
    wsym, flags, nsym = AutoSymPTR.symptr_rule(npt, ndim, syms)
    wxs = Vector{Tuple{eltype(wsym),FourierValue{SVector{d,eltype(u)},typeof(s)}}}(undef, nsym)
    fourier_symptr!(wxs, w, u, wsym, flags)
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

function init_buffer(f::FourierIntegrand, len)
    return nothing
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


# IAI rules

function init_nest(f::FourierIntegrand, fxx, dom, p,lims, state, algs, cacheval; kws_...)
    kws = NamedTuple(kws_)
    FX = typeof(fxx/oneunit(eltype(dom)))
    TX = eltype(dom)
    TP = Tuple{typeof(p),typeof(lims),typeof(state)}
    if algs isa Tuple{} # inner integral
        return FunctionWrapper{FX,Tuple{TX,TP}}() do x, (p, lims, state)
            v = FourierValue(limit_iterate(lims, state, x), workspace_evaluate!(f.w, x))
            return f.f(v, p)
        end
    else
        return FunctionWrapper{FX,Tuple{TX,TP}}() do x, (p, lims, state)
            segs, lims_, state_ = limit_iterate(lims, state, x)
            fx = FourierIntegrand(f.f, workspace_contract!(f.w, x))
            len = segs[end] - segs[1]
            kwargs = haskey(kws, :abstol) ? merge(kws, (abstol=kws.abstol/len,)) : kws
            sol = do_solve(fx, lims_, NestState(p, segs, state_), NestedQuad(algs), cacheval; kwargs...)
            return sol.u
        end
    end
end

function do_solve(f::FourierIntegrand, lims::AbstractIteratedLimits, p_, alg::NestedQuad, cacheval; kws...)
    g, p, segs, state = if p_ isa NestState
        f, p_.p, p_.segs, p_.state
    else
        seg, lim, sta = limit_iterate(lims)
        f, p_, seg, sta
    end
    dom = PuncturedInterval(segs)
    (dim = ndims(lims)) == ndims(f.w.series) || throw(ArgumentError("variables in Fourier series don't match domain"))
    algs = alg.algs isa IntegralAlgorithm ? ntuple(i -> alg.algs, Val(dim)) : alg.algs
    nest = init_nest(g, cacheval[dim][2], dom, p, lims, state, algs[1:dim-1], cacheval[1:dim-1]; kws...)
    return do_solve(nest, dom, (p, lims, state), algs[dim], cacheval[dim][1]; kws...)
end
