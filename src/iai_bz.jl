"""
    iterated_integration(f, bz::SymmetricBZ; kwargs...)
"""
function iterated_integration(f, bz::SymmetricBZ; kwargs...)
    kw = iterated_integration_kwargs(f, bz; kwargs...)
    j = det(bz.B)
    atol = kw.atol/nsyms(bz)/j # reduce absolute tolerance by symmetry factor
    int, err = iterated_integration(f, bz.lims; kw..., atol=atol)
    symmetrize(f, bz, j*int, j*err)
end
iterated_integration_kwargs(f, bz::SymmetricBZ; kwargs...) =
    iterated_integration_kwargs(f, bz.lims; kwargs...)
