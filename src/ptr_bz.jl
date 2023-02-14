symptr(f, bz::SymmetricBZ; kwargs...) =
    symptr(f, bz.B, symmmetries(bz); kwargs...)

autosymptr(f, bz::SymmetricBZ; kwargs...) =
    autosymptr(f, bz.B, symmetries(bz); kwargs...)