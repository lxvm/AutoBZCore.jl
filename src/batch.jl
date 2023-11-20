"""
    BatchIntegrand(f!, y::AbstractArray, x::AbstractVector, max_batch=typemax(Int))

Constructor for a `BatchIntegrand` accepting an integrand of the form `f!(y,x,p) = y .= f!.(x, Ref(p))`
that can evaluate the integrand at multiple quadrature nodes using, for example, threads,
the GPU, or distributed-memory. The `max_batch` keyword is a soft limit on the number of
nodes passed to the integrand. The buffers `y,x` must both be `resize!`-able since the
number of evaluation points may vary between calls to `f!`.
"""
struct BatchIntegrand{F,Y,X}
    # in-place function f!(y, x, p) that takes an array of x values and outputs an array of results in-place
    f!::F
    y::Y
    x::X
    max_batch::Int # maximum number of x to supply in parallel
    function BatchIntegrand(f!, y::AbstractArray, x::AbstractVector, max_batch::Integer=typemax(Int))
        max_batch > 0 || throw(ArgumentError("maximum batch size must be positive"))
        return new{typeof(f!),typeof(y),typeof(x)}(f!, y, x, max_batch)
    end
end


"""
    BatchIntegrand(f!, y, x; max_batch=typemax(Int))

Constructor for a `BatchIntegrand` with pre-allocated buffers.
"""
BatchIntegrand(f!, y, x; max_batch::Integer=typemax(Int)) =
    BatchIntegrand(f!, y, x, max_batch)

"""
    BatchIntegrand(f!, y::Type, x::Type=Nothing; max_batch=typemax(Int))

Constructor for a `BatchIntegrand` whose range type is known. The domain type is optional.
Array buffers for those types are allocated internally.
"""
BatchIntegrand(f!, Y::Type, X::Type=Nothing; max_batch::Integer=typemax(Int)) =
    BatchIntegrand(f!, Y[], X[], max_batch)


function init_segbuf(f::BatchIntegrand, dom, p, norm)
    x, s = init_midpoint_scale(dom)
    u = x/oneunit(x)
    TX = typeof(u)
    fx_s = zero(eltype(f.y)) * s/oneunit(s)    # TODO BatchIntegrand(InplaceIntegrand) should depend on size of result
    TI = typeof(fx_s)
    TE = typeof(norm(fx_s))
    return IteratedIntegration.alloc_segbuf(TX, TI, TE)
end
function do_solve_quadgk(::BatchIntegrand, segs, p, order, norm, cacheval, reltol, abstol, maxiters)
    throw(ArgumentError("QuadGKJL has not implemented BatchIntegrand. Please open an issue."))
end

function assemble_hintegrand(::BatchIntegrand, dom, p)
    throw(ArgumentError("HCubature.jl does not support batching. Consider opening an issue upstream."))
end

init_buffer(f::BatchIntegrand, len) = Vector{eltype(f.y)}(undef, len)
function assemble_pintegrand(f::BatchIntegrand, p, dom, rule)
    xx = if eltype(f.x) === Nothing
        x = last(first(rule))
        dom isa Basis ? typeof(dom*x)[] : typeof(oneunit(eltype(dom))*x)[]
    else
        f.x
    end
    return AutoSymPTR.BatchIntegrand((y,x) -> f.f!(y,x,p), f.y, xx, max_batch=f.max_batch)
end

function do_solve_auxquadgk(f::BatchIntegrand, segs, p, cacheval, order, norm, reltol, abstol, maxiters)
    u = oneunit(eltype(segs))
    usegs = map(x -> x/u, segs)
    xx = eltype(f.x) === Nothing ? typeof((segs[1]+segs[end])/2)[] : f.x
    g_ = (y, x) -> (resize!(xx, length(x)); f.f!(y, xx .= u .* x, p))
    g = IteratedIntegration.AuxQuadGK.BatchIntegrand(g_, f.y, xx/u, max_batch=f.max_batch)
    val, err = auxquadgk(g, usegs, maxevals = maxiters,
                    rtol = reltol, atol = isnothing(abstol) ? abstol : abstol/u, order = order, norm = norm, segbuf=cacheval)
    return IntegralSolution(u*val, u*err, true, -1)
end

function assemble_cont_integrand(::BatchIntegrand, p)
    throw(ArgumentError("ContQuadGK.jl doesn't support batching. Consider opening an issue upstream."))
end

function assemble_mero_integrand(::BatchIntegrand, p)
    throw(ArgumentError("MeroQuadGK.jl doesn't support batching. Consider opening an issue upstream."))
end

function do_solve_evalcounter(f::BatchIntegrand, dom, p, alg, cacheval; kws...)
    n::Int = 0
    g = (y, x, p) -> (n += length(x); f.f!(y, x, p); return nothing)
    sol = do_solve(BatchIntegrand(g, f.y, f.x, max_batch=f.max_batch), dom, p, alg, cacheval; kws...)
    return IntegralSolution(sol.u, sol.resid, sol.retcode, n)
end

function init_nested_cacheval(f::BatchIntegrand, p, segs, lims, state, alg::IntegralAlgorithm)
    dom = PuncturedInterval(segs)
    a, b = segs[1], segs[2]
    mid = (a+b)/2 # sample point that should be safe to evaluate
    cacheval = init_cacheval(BatchIntegrand(nothing, f.y, f.x, max_batch=f.max_batch), dom, p, alg)
    next = limit_iterate(lims, state, mid)
    fx = eltype(f.x) === Nothing ? typeof(next)[] : f.x
    # pre-allocate buffers in output
    return ((fx, typeof(mid)[]), cacheval, zero(eltype(f.y))*mid)
end
# we let the outer integrals be unbatched
function assemble_nested_integrand(f::BatchIntegrand, fxx, dom, p, lims, state, ::Tuple{}, (fx, x_); kws...)
    BatchIntegrand(f.y, x_, max_batch=f.max_batch) do y, xs, p
        resize!(fx, length(xs))
        f.f!(y, map!(x -> limit_iterate(lims, state, x), fx, xs), p)
    end
end

function assemble_nested_integrand(::BatchIntegrand, ::Nothing)
    throw(ArgumentError("NestedIntegrand(BatchIntegrand) is not support. Please open an issue"))
end
function assemble_nested_integrand(::BatchIntegrand, g)
    throw(ArgumentError("NestedIntegrand(BatchIntegrand) is not support. Please open an issue"))
end

"""
    NestedBatchIntegrand(f::Tuple, y::AbstractVector, x::AbstractVector, max_batch::Integer)

An integrand type intended for multi-threaded evaluation of [`NestedQuad`](@ref). The caller
provides a tuple `f` of worker functions that can evaluate the same integrand on different
threads, so as to avoid race conditions. These workers can also be `NestedBatchIntegrand`s
depending on if the user wants to parallelize the integration at multiple levels of nesting.
The other arguments are the same as for [`BatchIntegrand`](@ref).
"""
struct NestedBatchIntegrand{F,T,Y<:AbstractVector,X<:AbstractVector}
    f::T
    y::Y
    x::X
    max_batch::Int
end
function NestedBatchIntegrand(f::NTuple, y::Y, x::X, max_batch::Integer) where {Y,X}
    if eltype(f) <: NestedBatchIntegrand
        return NestedBatchIntegrand{_nesttype(eltype(f)),typeof(f),Y,X}(f, y, x, max_batch)
    else
        return NestedBatchIntegrand{eltype(f),typeof(f),Y,X}(f, y, x, max_batch)
    end
end
function NestedBatchIntegrand(f::AbstractArray{F}, y::Y, x::X, max_batch::Integer) where {F,Y,X}
    return NestedBatchIntegrand{F,typeof(f),Y,X}(f, y, x, max_batch)
end
function NestedBatchIntegrand(f::AbstractArray{T}, y::Y, x::X, max_batch::Integer) where {F,T<:NestedBatchIntegrand{F},Y,X}
    return NestedBatchIntegrand{F,typeof(f),Y,X}(f, y, x, max_batch)
end

_nesttype(::Type{<:NestedBatchIntegrand{F}}) where {F} = F
function NestedBatchIntegrand(f, y, x; max_batch::Integer=typemax(Int))
    return NestedBatchIntegrand(f, y, x, max_batch)
end

function NestedBatchIntegrand(f, ::Type{Y}, ::Type{X}=Nothing; kws...) where {Y,X}
    return NestedBatchIntegrand(f, Y[], X[]; kws...)
end

# this is the implementation of the multi-threading
function nested_to_batched(callback::C, f::NestedBatchIntegrand) where {C}
    workers = FlatView(f)
    return BatchIntegrand(f.y, f.x, max_batch=f.max_batch) do y, x, p
        nchunk = length(workers)
        Threads.@threads for ichunk in 1:min(nchunk, length(x))
            for (i, j) in zip(getchunk(x, ichunk, nchunk, :batch), getchunk(y, ichunk, nchunk, :batch))
                y[j] = callback(ichunk, workers, x[i], p)
            end
        end
        return nothing
    end
end
nested_to_batched(f::NestedBatchIntegrand) = nested_to_batched((i, w, x, p) -> w[i](x, p), f)

struct FlatView{T}
    nest::T
end
Base.eltype(::Type{FlatView{T}}) where {T} = _nesttype(T)
function Base.length(f::FlatView)
    if eltype(f.nest.f) <: NestedBatchIntegrand
        return sum(length∘FlatView, f.nest.f, init=0)
    else
        return length(f.nest.f)
    end
end
# I think the complexity is O(depth)
function Base.getindex(f::FlatView, i::Int)::_nesttype(typeof(f.nest))
    if eltype(f.nest.f) <: NestedBatchIntegrand
        n = 0
        while (len = length(FlatView(f.nest.f[n+=1]))) < i
            i -= len
        end
        Base.getindex(FlatView(f.nest.f[n]), i)
    else
        return Base.getindex(f.nest.f, i)
    end
end
function Base.iterate(f::FlatView)
    (len = length(f)) == 0 && return nothing
    return f[1], (2, len)
end
function Base.iterate(f::FlatView, state)
    n, len = state
    n > len && return nothing
    item = f[n]
    n += 1
    return item, (n, len)
end

function init_segbuf(f::NestedBatchIntegrand, dom, p, norm)
    x, s = init_midpoint_scale(dom)
    u = x/oneunit(x)
    TX = typeof(u)
    fx_s = zero(eltype(f.y)) * s/oneunit(s)    # TODO BatchIntegrand(InplaceIntegrand) should depend on size of result
    TI = typeof(fx_s)
    TE = typeof(norm(fx_s))
    return IteratedIntegration.alloc_segbuf(TX, TI, TE)
end

function do_solve_quadgk(f::NestedBatchIntegrand, segs, p, order, norm, cacheval, reltol, abstol, maxiters)
    w = nested_to_batched(f)
    return do_solve_quadgk(w, segs, p, order, norm, cacheval, reltol, abstol, maxiters)
end

function assemble_hintegrand(f::NestedBatchIntegrand, dom, p)
    w = nested_to_batched(f)
    return assemble_hintegrand(w, dom, p)
end

init_buffer(f::NestedBatchIntegrand, len) = Vector{eltype(f.y)}(undef, len)

function assemble_pintegrand(f::NestedBatchIntegrand, p, dom, rule)
    w = nested_to_batched(f)
    ys = w.y/prod(ntuple(n -> oneunit(eltype(dom)), Val(ndims(dom))))
    xs = if eltype(w.x) === Nothing
        x = last(first(rule))
        dom isa Basis ? typeof(dom*x)[] : typeof(oneunit(eltype(dom))*x)[]
    else
        w.x
    end
    return AutoSymPTR.BatchIntegrand((y, x) -> w.f!(y, x, p), ys, xs, max_batch=f.max_batch)
end

function do_solve_auxquadgk(f::NestedBatchIntegrand, segs, p, cacheval, order, norm, reltol, abstol, maxiters)
    w = nested_to_batched(f)
    return do_solve_auxquadgk(w,  segs, p, cacheval, order, norm, reltol, abstol, maxiters)
end

function assemble_cont_integrand(f::NestedBatchIntegrand, p)
    w = nested_to_batched(f)
    return assemble_cont_integrand(w, p)
end

function assemble_mero_integrand(f::NestedBatchIntegrand, p)
    w = nested_to_batched(f)
    return assemble_mero_integrand(w, p)
end

mutable struct WrapperCounter{F}
    numevals::Int
    const f::F
end
WrapperCounter(f) = WrapperCounter(0, f)
function (f::WrapperCounter)(args...; kws...)
    if f.f isa NestedIntegrand
        sol = f.f.f(args...; kws...)
        sol.numevals
        f.numevals += sol.numevals > 0 ? sol.numevals : 0
        return isnothing(f.f.g) ? sol.u : f.f.g(args..., sol.u; kws...)
    else
        f.numevals += 1
        return f.f(args...; kws...)
    end
end

# we allocate when creating the wrapper, and since this is typically once per integral it
# should be fine
function do_solve_evalcounter(f::NestedBatchIntegrand, dom, p, alg, cacheval; kws...)
    w = wrap_with_counter(f.f)
    g = NestedBatchIntegrand(w, f.y, f.x, max_batch=f.max_batch)
    sol = do_solve(g, dom, p, alg, cacheval; kws...)
    n = sum(s -> s.numevals, FlatView(g), init=0)
    return IntegralSolution(sol.u, sol.resid, sol.retcode, iszero(n) ? -1 : n)
end

function wrap_with_counter(nest)
    if eltype(nest) <: NestedBatchIntegrand
        map(f -> NestedBatchIntegrand(wrap_with_counter(f.f), f.y, f.x, max_batch=f.max_batch), nest)
    else
        map(WrapperCounter, nest)
    end
end

function init_nested_cacheval(f::NestedBatchIntegrand, p, segs, lims, state, alg::IntegralAlgorithm)
    dom = PuncturedInterval(segs)
    a, b = segs[1], segs[2]
    mid = (a+b)/2 # sample point that should be safe to evaluate
    cacheval = init_cacheval(BatchIntegrand(nothing, f.y, f.x, max_batch=f.max_batch), dom, p, alg)
    fx = eltype(f.x) === Nothing ? typeof(mid)[] : f.x
    return ((fx, nothing), cacheval, zero(eltype(f.y))*mid)
end

function init_nested_cacheval(f::NestedBatchIntegrand, p, segs, lims, state, alg_::IntegralAlgorithm, algs_::IntegralAlgorithm...)
    dim = ndims(lims)
    algs = (alg_, algs_...)
    alg = algs[dim]
    dom = PuncturedInterval(segs)
    a, b = segs[1], segs[2]
    mid = (a+b)/2 # sample point that should be safe to evaluate
    next = limit_iterate(lims, state, mid) # see what the next limit gives
    nest = init_nested_cacheval(f.f[1], p, next..., algs[1:dim-1]...)
    cacheval = init_cacheval(BatchIntegrand(nothing, f.y, f.x, max_batch=f.max_batch), dom, p, alg)
    fx = eltype(f.x) === Nothing ? typeof(mid)[] : f.x
    return ((fx, map(n -> deepcopy(nest), f.f)), cacheval, nest[3]*mid)
end

function assemble_nested_integrand(f::NestedBatchIntegrand, fxx, dom, p, lims, state, ::Tuple{}, (fx, cacheval); kws...)
    return nested_to_batched((i, w, x, p) -> w[i](limit_iterate(lims, state, x), p), f)
 end

# this wrapper avoids unnecessary allocations in the common case of nested quad
# if we just did nested_to_batched then the inner integrals wouldn't be counted as nested integrands
struct WrappedIntegrandBuffer{B,P,S,L,A,C,K,F,X,E}
    fbuf::B
    p::P
    state::S
    lims::L
    algs::A
    cbuf::C
    kws::K
    fxx::F
    xx::X
    err::E
end
Base.length(w::WrappedIntegrandBuffer) = length(w.fbuf)
function Base.eltype(::Type{WrappedIntegrandBuffer{B,P,S,L,A,C,K,F,X,E}}) where {B,P,S,L,A,C,K,F,X,E}
    return NestedIntegrand{FunctionWrapper{IntegralSolution{F,E},Tuple{X,P}},Nothing}
end
function Base.getindex(w::WrappedIntegrandBuffer, n::Int)
    f, lims, algs, state, kws, c, fxx, err, xx, p = w.fbuf[n], w.lims, w.algs, w.state, w.kws, w.cbuf[n], w.fxx, w.err, w.xx, w.p
    w_ = NestedIntegralWorker(f, lims, algs, state, kws, c)
    f_ = FunctionWrapper{IntegralSolution{typeof(fxx),typeof(err)},Tuple{typeof(xx),typeof(p)}}(w_)
    return NestedIntegrand(f_)
end
function Base.iterate(w::WrappedIntegrandBuffer)
    next = iterate(w.fbuf)
    isnothing(next) && return nothing
    return w[1], (2, next[2])
end
function Base.iterate(w::WrappedIntegrandBuffer, (n, state))
    next = iterate(w.fbuf, state)
    isnothing(next) && return nothing
    item = w[n]
    n += 1
    return item, (n, next[2])
end


# This function returns a  NestedBatchIntegrand(NestedIntegrand)
# TODO: only use a functionwrapper when provided with the integrand
function assemble_nested_integrand(f::NestedBatchIntegrand, fxx, dom, p, lims, state, algs, (fx, cacheval); kws...)
    xx = float(oneunit(eltype(dom)))
    TX = typeof(xx)
    TP = typeof(p)
    err = integralerror(last(algs), fxx)
    w = WrappedIntegrandBuffer(f.f, p, state, lims, algs, cacheval, NamedTuple(kws), fxx, xx, err)
    T = NestedIntegrand{FunctionWrapper{IntegralSolution{typeof(fxx),typeof(err)},Tuple{TX,TP}},Nothing}
    return NestedBatchIntegrand{T,typeof(w),typeof(f.y),typeof(f.x)}(w, f.y, f.x, f.max_batch)
end
