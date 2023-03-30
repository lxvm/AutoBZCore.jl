"""
    FourierIntegrand(f, s::AbstractFourierSeries, args...; kwargs...)

A type generically representing an integrand `f` whose entire dependence on the
variables of integration is in a Fourier series `s`, and which may also accept
some input parameters `p`. The caller must know that their function, `f`, will
be evaluated at many points, `x`, in the following way: `f(s(x), p...)`.
Therefore the caller is expected to know the type of `s(x)`
and the layout of the parameters in the tuple `p` (hint: it should correspond to
the arguments of the function). This type is optimized for the IAI and PTR routines.
"""
struct FourierIntegrand{F,S<:AbstractFourierSeries,P<:MixedParameters} <: AbstractAutoBZIntegrand
    f::F
    s::S
    p::P
end
FourierIntegrand(f, s, args...; kwargs...) =
    FourierIntegrand(f, s, MixedParameters(args...; kwargs...))

# provide Integrals.jl interface while still using functor interface
(f::FourierIntegrand)(x, p) = FourierIntegrand(f.f, f.s, merge(f.p, p))(x)

# intercept integrand construction when solving integral problem
# because the IAI routines dispatch on the integrand type
construct_integrand(f::FourierIntegrand, iip, p) =
    FourierIntegrand(f.f, f.s, merge(f.p, p))

evaluate_integrand(f::FourierIntegrand, s_x) = evaluate_integrand(f.f, s_x, f.p)

# IAI customizations that copy behavior of AbstractIteratedIntegrand
iterated_integrand(_::FourierIntegrand, y, ::Val{d}) where d = y
iterated_integrand(f::FourierIntegrand, x, ::Val{1}) =
    evaluate_integrand(f, f.s(x))

iterated_pre_eval(f::FourierIntegrand, x, ::Val{d}) where d =
    FourierIntegrand(f.f, contract(f.s, x, Val(d)), f.p)

(f::FourierIntegrand)(::Tuple{}) = f
function (f::FourierIntegrand)(x::NTuple{N}) where N
    if (d = ndims(f.s)) == N == 1
        iterated_integrand(f, x[1], Val(1))
    else
        iterated_pre_eval(f, x[N], Val(d))(x[1:N-1])
    end
end
(f::FourierIntegrand)(x) = f(promote(x...))

# PTR customizations

# no symmetries
struct FourierPTRRule{N,X,S<:AbstractFourierSeries{N}}
    x::Vector{X}
    s::S
    n::Array{Int,0}
end
Base.size(r::FourierPTRRule{N}) where N = ntuple(_->r.n[], Val(N))
Base.length(r::FourierPTRRule) = length(r.x)
function Base.copy!(r::T, s::T) where {T<:FourierPTRRule}
    copy!(r.x, s.x)
    r
end
Base.getindex(p::FourierPTRRule{N}, i::Int) where {N} = p.x[i]
Base.getindex(p::FourierPTRRule{N}, i::CartesianIndex{N}) where {N} =
    p.x[ptrindex(p.n[], i)]

Base.isdone(p::FourierPTRRule, state) = !(1 <= state <= length(p))
function Base.iterate(p::FourierPTRRule, state=1)
    Base.isdone(p, state) && return nothing
    (p[state], state+1)
end

struct FourierPTR{T<:AbstractFourierSeries}
    s::T
end
function (f::FourierPTR)(::Type{T}, ::Val{N}) where {T,N}
    S = Base.promote_op(f.s, NTuple{N,T})
    x = Vector{S}(undef, 0)
    FourierPTRRule(x, f.s, Array{Int,0}(undef))
end

@generated function ptr_rule!(rule::FourierPTRRule, npt, ::Val{N}) where {N}
    f_N = Symbol(:f_, N)
    quote
        $f_N = rule.s
        rule.n[] = npt
        resize!(rule.x, npt^N)
        box = period($f_N)
        n = 0
        Base.Cartesian.@nloops $N i _ -> Base.OneTo(npt) (d -> d==1 ? nothing : f_{d-1} = contract(f_d, box[d]*(i_d-1)/npt, Val(d))) begin
            n += 1
            rule.x[n] = f_1(box[1]*(i_1-1)/npt)
        end
        rule
    end
end

function ptr(f::FourierIntegrand, B::AbstractMatrix; npt=npt_update(f,0), rule=nothing, min_per_thread=1, nthreads=NTHREADS_KSUM[])
    N = checksquare(B); T = float(eltype(B))
    rule_x = (rule===nothing) ? ptr_rule!(FourierPTR(f.s)(T, Val(N)), npt, Val(N)) : rule
    n = length(rule_x); dvol = abs(det(B))/npt^N
    nthreads == 1 && return sum(s_x -> evaluate_integrand(f, s_x), rule_x)*dvol

    acc = evaluate_integrand(f, rule_x[n]) # unroll first term in sum to get right types
    n == 1 && return acc*dvol
    runthreads = min(nthreads, div(n-1, min_per_thread)) # choose the actual number of threads
    d, r = divrem(n-1, runthreads)
    partial_sums = fill!(Vector{typeof(acc)}(undef, runthreads), zero(acc)) # allocations :(
    Threads.@threads for i in Base.OneTo(runthreads)
        # batch nodes into `runthreads` continguous groups of size d or d+1 (remainder)
        jmax = (i <= r ? d+1 : d)
        offset = min(i-1, r)*(d+1) + max(i-1-r, 0)*d
        @inbounds for j in 1:jmax
            partial_sums[i] += evaluate_integrand(f, rule_x[offset + j])
        end
    end
    for part in partial_sums
        acc += part
    end
    acc*dvol
end


# general symmetries
struct FourierSymPTRRule{X,S<:AbstractFourierSeries}
    w::Vector{Int}
    x::Vector{X}
    s::S
end
Base.length(r::FourierSymPTRRule) = length(r.x)
function Base.copy!(r::T, s::T) where {T<:FourierSymPTRRule}
    copy!(r.w, s.w)
    copy!(r.x, s.x)
    r
end

struct FourierSymPTR{T<:AbstractFourierSeries}
    s::T
end
function (f::FourierSymPTR)(::Type{T}, ::Val{N}) where {T,N}
    S = Base.promote_op(f.s, NTuple{N,T})
    w = Vector{Int}(undef, 0); x = Vector{S}(undef, 0)
    FourierSymPTRRule(w, x, f.s)
end

@generated function symptr_rule!(rule::FourierSymPTRRule, npt, ::Val{N}, syms) where {N}
    f_N = Symbol(:f_, N)
    quote
        $f_N = rule.s
        flag, wsym, nsym = symptr_rule(npt, Val(N), syms)
        n = 0
        box = period($f_N)
        resize!(rule.w, nsym)
        resize!(rule.x, nsym)
        Base.Cartesian.@nloops $N i flag (d -> d==1 ? nothing : f_{d-1} = contract(f_d, box[d]*(i_d-1)/npt, Val(d))) begin
            (Base.Cartesian.@nref $N flag i) || continue
            n += 1
            rule.x[n] = f_1(box[1]*(i_1-1)/npt)
            rule.w[n] = wsym[n]
            n >= nsym && break
        end
        rule
    end
end

# enables kpt parallelization by default for all BZ integrals
# with symmetries
function symptr(f::FourierIntegrand, B::AbstractMatrix, syms; npt=npt_update(f, 0), rule=nothing, min_per_thread=1, nthreads=NTHREADS_KSUM[])
    N = checksquare(B); T = float(eltype(B))
    rule_ = (rule===nothing) ? symptr_rule!(FourierSymPTR(f.s)(T, Val(N)), npt, Val(N), syms) : rule
    n = length(rule_); dvol = abs(det(B))/length(syms)/npt^N
    nthreads == 1 && return sum(((w, s_x),) -> w*evaluate_integrand(f, s_x), zip(rule_.w, rule_.x))*dvol
    
    acc = rule_.w[n]*evaluate_integrand(f, rule_.x[n]) # unroll first term in sum to get right types
    n == 1 && return acc*dvol
    runthreads = min(nthreads, div(n-1, min_per_thread)) # choose the actual number of threads
    d, r = divrem(n-1, runthreads)
    partial_sums = fill!(Vector{typeof(acc)}(undef, runthreads), zero(acc)) # allocations :(
    Threads.@threads for i in Base.OneTo(runthreads)
        # batch nodes into `runthreads` continguous groups of size d or d+1 (remainder)
        jmax = (i <= r ? d+1 : d)
        offset = min(i-1, r)*(d+1) + max(i-1-r, 0)*d
        @inbounds for j in 1:jmax
            partial_sums[i] += rule_.w[offset + j]*evaluate_integrand(f, rule_.x[offset + j])
        end
    end
    for part in partial_sums
        acc += part
    end
    acc*dvol
end

# Defining defaults without symmetry

symptr_rule!(rule::FourierPTRRule, npt, ::Val{d}, ::Nothing) where d =
    ptr_rule!(rule, npt, Val(d))

symptr(f::FourierIntegrand, B::AbstractMatrix, ::Nothing; kwargs...) =
    ptr(f, B; kwargs...)

autosymptr(f::FourierIntegrand, B::AbstractMatrix, ::Nothing; kwargs...) =
    autosymptr(f, B, nothing, FourierPTR(f.s); kwargs...)
autosymptr(f::FourierIntegrand, B::AbstractMatrix, syms; kwargs...) =
    autosymptr(f, B, syms, FourierSymPTR(f.s); kwargs...)

# helper routine to allocate rules
"""
    alloc_rule(f::AbstractFourierSeries, ::Type, syms)

Compute the values of `f` on the PTR grid as well as the quadrature weights for
the given `syms` to use across multiple compatible calls of the [`PTR`](@ref) algorithm.
"""
alloc_rule(f::AbstractFourierSeries{N}, ::Type{T}, ::Nothing, npt::Int) where {N,T} =
    ptr_rule!(FourierPTR(f)(T, Val(N)), npt, Val(N))
alloc_rule(f::AbstractFourierSeries{N}, ::Type{T}, syms, npt::Int) where {N,T}=
    symptr_rule!(FourierSymPTR(f)(T, Val(N)), npt, Val(N), syms)

"""
    alloc_autobuffer(f::AbstractFourierSeries, ::Type, syms)

Initialize an empty buffer of PTR rules with pre-evaluated Fourier series
evaluated on a domain of type `T` with symmetries `syms` to use across multiple
compatible calls of the [`AutoPTR`](@ref) algorithm.
"""
alloc_autobuffer(f::AbstractFourierSeries{N}, ::Type{T}, ::Nothing) where {N,T} =
    alloc_autobuffer(T, Val(N), FourierPTR(f))
alloc_autobuffer(f::AbstractFourierSeries{N}, ::Type{T}, syms) where {N,T} =
    alloc_autobuffer(T, Val(N), FourierSymPTR(f))

"""
    alloc_segbufs(coefficient_type, range_type, norm_type, ndim)

Allocate a segment buffer for multiple compatible calls to [`IAI`](@ref).
Typically `coefficient_type` would be `eltype(bz)`, `range_type` would be the
return type of the integrand, and `norm_type` would be `Base.promote_op(norm,
range_type)`. `ndim` should be the number of dimensions of integration.
"""
function alloc_segbufs end