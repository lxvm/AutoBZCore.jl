"""
    AbstractFourierIntegrand{S<:AbstractFourierSeries} <: AbstractIteratedIntegrand

Supertype representing Fourier integrands
"""
abstract type AbstractFourierIntegrand{d,S<:AbstractFourierSeries} <: AbstractIteratedIntegrand{d} end

#= interface
Should have fields f, s, p representing the integrand, the series, and params
=#

# abstract methods to optimize IAI

@generated iterated_pre_eval(f::T, x, ::Val{dim}) where {dim,d,T<:AbstractFourierIntegrand{d}} =
    :($(nameof(T)){$d}(f.f, contract(f.s, x, Val(dim)), f.p))

iterated_value(f::AbstractFourierIntegrand) = f.s

# abstract methods to optimize PTR

ptr!(rule, npt, f::AbstractFourierIntegrand, ::Val{d}, ::Type{T}, syms) where {d,T} =
    fourier_ptr!(rule, npt, f.s, Val(d), T, syms)

# general symmetries
function ptr(npt, f::AbstractFourierIntegrand, ::Val{d}, ::Type{T}, syms) where {d,T}
    x = Vector{Base.promote_op(f.s,NTuple{d,T})}(undef, 0)
    w = Vector{Int}(undef, 0)
    rule = (; x=x, w=w)
    ptr!(rule, npt, f, Val(d), T, syms)
end

# no symmetries
function ptr(npt, f::AbstractFourierIntegrand, ::Val{d}, ::Type{T}, syms::Nothing) where {d,T}
    x = Vector{Base.promote_op(f.s,NTuple{d,T})}(undef, 0)
    rule = (; x=x)
    ptr!(rule, npt, f, Val(d), T, syms)
end

# implementations

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
struct FourierIntegrand{F,d,S,P} <: AbstractFourierIntegrand{d,S}
    f::F
    s::S
    p::P
    FourierIntegrand{d}(f::F, s::S, p::P) where {d,F<:Function,S<:AbstractFourierSeries,P<:Tuple} =
        new{F,d,S,P}(f,s,p)
end
FourierIntegrand{F}(s::AbstractFourierSeries{N}, p...) where {F<:Function,N} =
    FourierIntegrand{N}(F.instance, s, p) # allows dispatch by aliases
FourierIntegrand{d}(f, s, p...) where d = FourierIntegrand{d}(f, s, p)

iterated_integrand(_::FourierIntegrand, y, ::Val{d}) where d = y
iterated_integrand(f::FourierIntegrand, x, ::Val{1}) =
    f.f(f.s(x), f.p...)

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
    IteratedFourierIntegrand(fs::Tuple, s::AbstractFourierSeries, ps...)

Integrand type similar to `FourierIntegrand`, but allowing nested integrand
functions `fs` with `fs[1]` the innermost function. Only the innermost integrand
is allowed to depend on parameters, but this could be implemented to allow the
inner function to also be multivariate Fourier series.

!!! note "Incompatibility with symmetries"
    In practice, it is only safe to use the output of this integrand when
    integrated over a domain with symmetries when the functions`fs` preserve the
    periodicity of the functions being integrated, such as linear functions.
    When used as an equispace integrand, this type does each integral one
    variable at a time applying PTR to each dimension. Note that 
"""
struct IteratedFourierIntegrand{F,d,S,P} <: AbstractFourierIntegrand{d,S}
    f::F
    s::S
    p::P
    IteratedFourierIntegrand{d}(f::F, s::S, p::P) where {d,F<:Tuple,S<:AbstractFourierSeries,P<:Tuple} =
        new{F,d,S,P}(f,s,p)
end
IteratedFourierIntegrand{F}(s, ps...) where {F<:Tuple{Vararg{Function}}} =
    IteratedFourierIntegrand{length(F.parameters)}(tuple(map(f -> f.instance, F.parameters)...), s, ps) # allows dispatch by aliases
IteratedFourierIntegrand{d}(f, s, ps...) where d =
    IteratedFourierIntegrand{d}(f, s, ps)

# IAI customizations

iterated_integrand(f::IteratedFourierIntegrand, x, ::Val{1}) = f.f[1](f.s(x), f.p...)
iterated_integrand(f::IteratedFourierIntegrand, y, ::Val{d}) where d = f.f[d](y)
iterated_integrand(_::IteratedFourierIntegrand, y, ::Val{0}) = y

# PTR customizations

evalptr(_, _, ::IteratedFourierIntegrand, _, _) =
    throw(ArgumentError("iterated integrands and PTR are only allowed without symmetries"))

function evalptr(rule, npt, f::IteratedFourierIntegrand, B, ::Nothing)
    @warn "Do not trust an iterated integrand with equispace integration unless for linear integrands"
    iterated_fourier_evalptr(rule.x, npt, f, B)
end

@generated function iterated_fourier_evalptr(x::Vector{T}, npt, f::IteratedFourierIntegrand, B::SMatrix{N,N}) where {T,N}
    I_N = Symbol(:I_, N)
    quote
        # infer return types of individual integrals
        T_1 = Base.promote_op(*, Base.promote_op(ptr_integrand, F, T), W)
        Base.Cartesian.@nexprs $(N-1) d -> T_{d+1} = Base.promote_op(iterated_integrand, F, T_d, Type{Val{d+1}})
        # compute quadrature
        n = 0
        $I_N = zero($(Symbol(:T_, N)))
        Base.Cartesian.@nloops $N i rule (d -> d==1 ? nothing : I_{d-1} = zero(T_{d-1})) (d -> d==1 ? nothing : I_d += iterated_integrand(f, I_{d-1}, Val{d})) begin
            n += 1
            I_1 += iterated_integrand(f, x[n], Val{1}) # implicitly assuming outer functions are linear
        end
        iterated_integrand(f, $I_N, Val{0}) * det(B)/npt^d
    end
end