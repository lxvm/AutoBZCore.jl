# TODO: In Julia 1.9 this could become an extension
"""
    FourierIntegrand(f, s::AbstractFourierSeries, ps...)

A type generically representing an integrand `f` whose entire dependence on the
variables of integration is in a Fourier series `s`, and which may also accept
some input parameters `ps`. The caller must know that their function, `f`, will
be evaluated at many points, `x`, in the following way: `f(s(x), ps...)`.
Therefore the caller is expected to know the type of `s(x)` (hint: `eltype(s)`)
and the layout of the parameters in the tuple `ps`. Additionally, `f` is assumed
to be type-stable, and is compatible with the equispace integration routines.
"""
struct FourierIntegrand{F,S,P,K} <: AbstractIntegrand{F}
    f::F
    s::S
    p::P
    kwargs::K
    FourierIntegrand(f::F, s::S, p::P, k::K) where {F,S<:AbstractFourierSeries,P<:Tuple,K} =
        new{F,S,P,K}(f,s,p,k)
end
FourierIntegrand(f, s::AbstractFourierSeries, p...; kwargs...) =
    FourierIntegrand(f, s, p, kwargs)
# hack to allow user-defined integrands without a type alias
FourierIntegrand{F}(s::AbstractFourierSeries, p...; kwargs...) where {F<:Function} =
    FourierIntegrand(F.instance, s, p, kwargs)


# IAI customizations that copy behavior of AbstractIteratedIntegrand

iterated_integrand(_::FourierIntegrand, y, ::Val{d}) where d = y
iterated_integrand(f::FourierIntegrand, x, ::Val{1}) =
    f.f(f.s(x), f.p...; f.kwargs...)

iterated_pre_eval(f::FourierIntegrand, x, ::Val{d}) where d =
    FourierIntegrand(f.f, contract(f.s, x, Val(d)), f.p, f.kwargs)

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

# general symmetries
function ptr(npt, f::FourierIntegrand, ::Val{d}, ::Type{T}, syms) where {d,T}
    x = Vector{Base.promote_op(f.s,NTuple{d,T})}(undef, 0)
    w = Vector{Int}(undef, 0)
    rule = (; x=x, w=w)
    ptr!(rule, npt, f, Val(d), T, syms)
end

# no symmetries
function ptr(npt, f::FourierIntegrand, ::Val{d}, ::Type{T}, syms::Nothing) where {d,T}
    x = Vector{Base.promote_op(f.s,NTuple{d,T})}(undef, 0)
    rule = (; x=x)
    ptr!(rule, npt, f, Val(d), T, syms)
end

ptr!(rule, npt, f::FourierIntegrand, ::Val{d}, ::Type{T}, syms) where {d,T} =
fourier_ptr!(rule, npt, f.s, Val(d), T, syms)

"""
    fourier_ptr!(rule, npt, f::AbstractFourierSeries, ::Val{d}, ::Type{T}, syms) where {d,T}

Modifies and returns the NamedTuple `rule` with fields `x,w` to contain the
Fourier series evaluated at the symmetrized PTR grid points
"""
# no symmetries
@generated function fourier_ptr!(rule, npt, f::AbstractFourierSeries{N}, ::Val{N}, ::Type{T}, ::Nothing) where {N,T}
    f_N = Symbol(:f_, N)
    quote
        $f_N = f
        resize!(rule.x, npt^N)
        box = period(f)
        n = 0
        Base.Cartesian.@nloops $N i _ -> Base.OneTo(npt) (d -> d==1 ? nothing : f_{d-1} = contract(f_d, box[d]*(i_d-1)/npt, Val(d))) begin
            n += 1
            rule.x[n] = f_1(box[1]*(i_1-1)/npt)
        end
        rule
    end
end

# general symmetries
@generated function fourier_ptr!(rule, npt, f::AbstractFourierSeries{N}, ::Val{N}, ::Type{T}, syms) where {N,T}
    f_N = Symbol(:f_, N)
    quote
        $f_N = f
        flag, wsym, nsym = ptr_(Val(N), npt, syms)
        n = 0
        box = period(f)
        resize!(rule.x, nsym)
        resize!(rule.w, nsym)
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

ptr_integrand(f::FourierIntegrand, s_x) = f.f(s_x, f.p...)

function evalptr(rule, npt, f::FourierIntegrand, B::SMatrix{d,d}, syms) where d
    int = mapreduce((w, x) -> w*ptr_integrand(f, x), +, rule.w, rule.x)
    int * det(B)/npt^d/length(syms)
end

function evalptr(rule, npt, f::FourierIntegrand, B::SMatrix{d,d}, ::Nothing) where d
    int = sum(x -> ptr_integrand(f, x), rule.x)
    int * det(B)/npt^d
end

"""
    FourierIntegrator(routine, f, bz, s, p...; kwargs...)

An [`Integrator`](@ref) that is specialized for [`FourierIntegrand`](@ref).

    FourierIntegrator{F}(routine, bz, s, p...; kwargs...) where {F<:Function}

Defining a type alias for `FourierIntegrand{F}` allows for omitting `f`.
"""
const FourierIntegrator{F} = Integrator{FourierIntegrand,F}

# hack to allow integrators via type alias
FourierIntegrator{F}(routine, bz, s, p...; kwargs...) where {F<:Function} =
    FourierIntegrator(routine, F.instance, bz, s, p...; kwargs...)
