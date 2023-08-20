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

struct NestedBatchIntegrand{F,N,T,Y<:AbstractVector,X<:AbstractVector}
    f::NTuple{N,T}
    y::Y
    x::X
    max_batch::Int
    function NestedBatchIntegrand(f::NTuple{N,F}, y::Y, x::X, max_batch::Integer) where {N,F,Y,X}
        return new{F,N,F,Y,X}(f, y, x, max_batch)
    end
    function NestedBatchIntegrand(f::NTuple{N,T}, y::Y, x::X, max_batch::Integer) where {N,F,T<:NestedBatchIntegrand{F},Y,X}
        return new{F,N,T,Y,X}(f, y, x, max_batch)
    end
end

function NestedBatchIntegrand(f, y, x; max_batch::Integer=typemax(Int))
    return NestedBatchIntegrand(f, y, x, max_batch)
end

function NestedBatchIntegrand(f, ::Type{Y}, ::Type{X}=Nothing; kws...) where {Y,X}
    return NestedBatchIntegrand(f, Y[], X[]; kws...)
end
