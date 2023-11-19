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
    g = (y, x, p) -> (n += length(x); f.f!(y, x, p))
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
    function NestedBatchIntegrand(f::NTuple, y::Y, x::X, max_batch::Integer) where {Y,X}
        if eltype(f) <: NestedBatchIntegrand
            return new{_nesttype(eltype(f)),typeof(f),Y,X}(f, y, x, max_batch)
        else
            return new{eltype(f),typeof(f),Y,X}(f, y, x, max_batch)
        end
    end
    function NestedBatchIntegrand(f::AbstractArray{F}, y::Y, x::X, max_batch::Integer) where {F,Y,X}
        return new{F,typeof(f),Y,X}(f, y, x, max_batch)
    end
    function NestedBatchIntegrand(f::AbstractArray{T}, y::Y, x::X, max_batch::Integer) where {F,T<:NestedBatchIntegrand{F},Y,X}
        return new{F,typeof(f),Y,X}(f, y, x, max_batch)
    end
end

_nesttype(::Type{<:NestedBatchIntegrand{F}}) where {F} = F
function NestedBatchIntegrand(f, y, x; max_batch::Integer=typemax(Int))
    return NestedBatchIntegrand(f, y, x, max_batch)
end

function NestedBatchIntegrand(f, ::Type{Y}, ::Type{X}=Nothing; kws...) where {Y,X}
    return NestedBatchIntegrand(f, Y[], X[]; kws...)
end

function init_segbuf(::NestedBatchIntegrand, dom, p, norm)
    throw(ArgumentError("Cannot allocate a segbuf for NestedBatchIntegrand. Please open an issue."))
end
function do_solve_quadgk(::NestedBatchIntegrand, segs, p, cacheval, reltol, abstol, maxiters)
    throw(ArgumentError("QuadGKJL has not implemented NestedBatchIntegrand. Please open an issue."))
end

function assemble_hintegrand(::NestedBatchIntegrand, dom, p)
    throw(ArgumentError("HCubature.jl does not support batching. Consider opening an issue upstream."))
end

init_buffer(f::NestedBatchIntegrand, len) = Vector{eltype(f.y)}(undef, len)

# TODO: unwrap the NestedBatchIntegrand recursively
function assemble_pintegrand(f::NestedBatchIntegrand, p, dom, rule)
    ys = f.y/prod(ntuple(n -> oneunit(eltype(dom)), Val(ndims(dom))))
    xs = if eltype(f.x) === Nothing
        x = last(first(rule))
        dom isa Basis ? typeof(dom*x)[] : typeof(oneunit(eltype(dom))*x)[]
    else
        f.x
    end
    return AutoSymPTR.BatchIntegrand(ys, xs, max_batch=f.max_batch) do y, x
        # would be better to fully unwrap the nested structure, but this is one level
        nchunk = length(f.f)
        Threads.@threads for ichunk in 1:min(nchunk, length(x))
            for (i, j) in zip(getchunk(x, ichunk, nchunk, :batch), getchunk(y, ichunk, nchunk, :batch))
                y[j] = f.f[ichunk](x[i], p)
            end
        end
        return nothing
    end
end

function do_solve_auxquadgk(::NestedBatchIntegrand, segs, p, cacheval, order, norm, reltol, abstol, maxiters)
    throw(ArgumentError("AuxQuadGKJL has not implemented NestedBatchIntegrand. Please open an issue."))
end

function assemble_cont_integrand(::NestedBatchIntegrand, p)
    throw(ArgumentError("ContQuadGK.jl doesn't support batching. Consider opening an issue upstream."))
end

function assemble_mero_integrand(::NestedBatchIntegrand, p)
    throw(ArgumentError("MeroQuadGK.jl doesn't support batching. Consider opening an issue upstream."))
end

function do_solve_evalcounter(f::NestedBatchIntegrand, dom, p, alg, cacheval; kws...)
    throw(ArgumentError("EvalCounter has not implemented NestedBatchIntegrand. Please open an issue."))
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
    TP = typeof(p)
    nchunk = length(f.f)
    return BatchIntegrand(FunctionWrapper{Nothing,Tuple{typeof(f.y),typeof(fx),TP}}() do y, x, p
        Threads.@threads for ichunk in 1:min(nchunk, length(x))
            for (i, j) in zip(getchunk(x, ichunk, nchunk, :scatter), getchunk(y, ichunk, nchunk, :scatter))
                xi = x[i]
                y[j] = f.f[ichunk](limit_iterate(lims, state, xi), p)
            end
        end
        return nothing
    end, f.y, fx, max_batch=f.max_batch)
end

# TODO: make this function return a NestedIntegrand
function assemble_nested_integrand(f::NestedBatchIntegrand, fxx, dom, p, lims, state, algs, (fx, cacheval); kws_...)
    kws = NamedTuple(kws_)
    TP = typeof(p)
    nchunks = length(f.f)
    return BatchIntegrand(FunctionWrapper{Nothing,Tuple{typeof(f.y),typeof(fx),TP}}() do y, x, p
        Threads.@threads for ichunk in 1:min(nchunks, length(x))
            for (i, j) in zip(getchunk(x, ichunk, nchunks, :scatter), getchunk(y, ichunk, nchunks, :scatter))
                xi = x[i]
                segs, lims_, state_ = limit_iterate(lims, state, xi)
                len = segs[end] - segs[1]
                kwargs = haskey(kws, :abstol) ? merge(kws, (abstol=kws.abstol/len,)) : kws
                y[j] = do_solve(f.f[ichunk], StatefulLimits(segs, state_, lims_), p, NestedQuad(algs), cacheval[ichunk]; kwargs...).u
            end
        end
        return nothing
    end, f.y, fx, max_batch=f.max_batch)
end
