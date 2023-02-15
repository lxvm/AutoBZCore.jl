function iterated_integration(f, bz::SymmetricBZ; kwargs...)
    kw = iterated_integration_kwargs(f, bz; kwargs...)
    int, err = iterated_integration(f, bz.lims; kw...)
    symmetrize(f, bz, int, err)
end
function iterated_integration_kwargs(f, bz::SymmetricBZ; kwargs...)
    kw = iterated_integration_kwargs(f, bz.lims; kwargs...)
    (; kw..., atol=kw.atol/nsyms(bz)) # rescaling by symmetries
end
iterated_integral_type(f, bz::SymmetricBZ) = iterated_integral_type(f, bz.lims)
iterated_inference(f, bz::SymmetricBZ) = iterated_inference(f, bz.lims)
alloc_segbufs(f, bz::SymmetricBZ) = alloc_segbufs(f, bz.lims)
