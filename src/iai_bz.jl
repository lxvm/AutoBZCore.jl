"""
    iterated_integration(f, bz::SymmetricBZ; kwargs...)
"""
function iterated_integration(f, bz::SymmetricBZ; kwargs...)
    kw = iterated_integration_kwargs(f, bz; kwargs...)
    atol = kw.atol/nsyms(bz) # reduce absolute tolerance by symmetry factor
    int, err = iterated_integration(f, bz.lims; kw..., atol=atol)
    symmetrize(f, bz, int, err)
end
iterated_integration_kwargs(f, bz::SymmetricBZ; kwargs...) =
    iterated_integration_kwargs(f, bz.lims; kwargs...)
