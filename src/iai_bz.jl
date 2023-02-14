function iterated_integration(f, bz::Symmetric; atol=nothing, kwargs...)
    atol = something(atol, zero(eltype(bz)))/nsyms(bz) # rescaling by symmetries
    int, err = iterated_integration(f, limits(bz); atol=atol, kwargs...)
    symmetrize(f, bz, int, err)
end
function iterated_integration_kwargs(f, bz::Symmetric; atol=nothing, kwargs...)
    atol = something(atol, zero(eltype(bz)))/nsyms(bz) # rescaling by symmetries
    iterated_integration_kwargs(f, limits(bz); atol=atol, kwargs...)
end
iterated_integral_type(f, bz::SymmetricBZ) = iterated_integral_type(f, limits(bz))
iterated_inference(f, bz::SymmetricBZ) = iterated_inference(f, limits(bz))
alloc_segbufs(f, bz::SymmetricBZ) = alloc_segbufs(f, limits(bz))
