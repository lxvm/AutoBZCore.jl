"""
    FourierIntegrand(f, s::AbstractFourierSeries)

A type generically representing an integrand `f` whose entire dependence on the
variables of integration is in a Fourier series `s`, and which may also accept
some input parameters `ps`. The caller must know that their function, `f`, will
be evaluated at many points, `x`, in the following way: `f(s(x), ps...)`.
Therefore the caller is expected to know the type of `s(x)` (hint: `eltype(s)`)
and the layout of the parameters in the tuple `ps`. Additionally, `f` is assumed
to be type-stable, and is compatible with the equispace integration routines.
"""
struct FourierIntegrand{F,S<:AbstractFourierSeries,P<:Tuple}
    f::F
    s::S
    p::P
end
FourierIntegrand(f, s, p...) = FourierIntegrand(f, s, p)
(f::FourierIntegrand)(x, p) = f.f(f.s(x), f.p..., p...)
construct_integrand(f::FourierIntegrand, iip, p) =
    FourierIntegrand(f.f, f.s, (f.p..., p...))

# IAI customizations that copy behavior of AbstractIteratedIntegrand

iterated_integrand(_::FourierIntegrand, y, ::Val{d}) where d = y
iterated_integrand(f::FourierIntegrand, x, ::Val{1}) =
    f.f(f.s(x), f.p...)

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

function ptr(f::FourierIntegrand, B::AbstractMatrix; npt=npt_update(f,0), rule=nothing, min_per_thread=1, nthreads=Threads.nthreads())
    N = checksquare(B); T = float(eltype(B))
    rule_ = (rule===nothing) ? ptr_rule!(FourierPTR(f.s)(T, Val(N)), npt, Val(N)) : rule
    n = length(rule_)

    acc = f.f(rule_.x[n], f.p...) # unroll first term in sum to get right types
    n == 1 && return acc*det(B)/npt^N
    runthreads = min(nthreads, div(n-1, min_per_thread)) # choose the actual number of threads
    d, r = divrem(n-1, runthreads)
    partial_sums = fill!(Vector{typeof(acc)}(undef, runthreads), zero(acc)) # allocations :(
    Threads.@threads for i in Base.OneTo(runthreads)
        # batch nodes into `runthreads` continguous groups of size d or d+1 (remainder)
        jmax = (i <= r ? d+1 : d)
        offset = min(i-1, r)*(d+1) + max(i-1-r, 0)*d
        @inbounds for j in 1:jmax
            partial_sums[i] += f.f(rule_.x[offset + j], f.p...)
        end
    end
    for part in partial_sums
        acc += part
    end
    acc*det(B)/npt^N
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
function symptr(f::FourierIntegrand, B::AbstractMatrix, syms; npt=npt_update(f, 0), rule=nothing, min_per_thread=1, nthreads=Threads.nthreads())
    N = checksquare(B); T = float(eltype(B))
    rule_ = (rule===nothing) ? symptr_rule!(FourierSymPTR(f.s)(T, Val(N)), npt, Val(N), syms) : rule
    n = length(rule_)

    acc = rule_.w[n]*f.f(rule_.x[n], f.p...) # unroll first term in sum to get right types
    n == 1 && return acc*det(B)/length(syms)/npt^N
    runthreads = min(nthreads, div(n-1, min_per_thread)) # choose the actual number of threads
    d, r = divrem(n-1, runthreads)
    partial_sums = fill!(Vector{typeof(acc)}(undef, runthreads), zero(acc)) # allocations :(
    Threads.@threads for i in Base.OneTo(runthreads)
        # batch nodes into `runthreads` continguous groups of size d or d+1 (remainder)
        jmax = (i <= r ? d+1 : d)
        offset = min(i-1, r)*(d+1) + max(i-1-r, 0)*d
        @inbounds for j in 1:jmax
            partial_sums[i] += rule_.w[offset + j]*f.f(rule_.x[offset + j], f.p...)
        end
    end
    for part in partial_sums
        acc += part
    end
    acc*det(B)/length(syms)/npt^N
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
