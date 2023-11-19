"""
    IntegrandWrapper{Ret,Args<:Tuple}()
Use this wrapper to signal that the integrand is type-stable
Uses FunctionWrappers.jl to improve compiler performance
"""
struct IntegrandWrapper{Ret,Args<:Tuple,F}
    f::F
end

IntegrandWrapper{Ret,Args}(f::IntegrandWrapper{Ret,Args}) where {Ret,Args} = f

"""
    ThreadedBatchIntegrand(workers::Channel, [integrand_prototype])

This integrand gets multi-threaded dynamically from the available worker pool
"""
struct ThreadedBatchIntegrand{W<:Channel,P}
    workers::W
    integrand_protoype::P
end
