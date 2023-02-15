"""
    symptr(f, bz::SymmetricBZ; kwargs...)
"""
symptr(f, bz::SymmetricBZ; kwargs...) =
    symptr(f, bz.B, bz.syms; kwargs...)
symptr_kwargs(f, bz::SymmetricBZ; kwargs...) =
    symptr_kwargs(f, bz.B, bz.syms; kwargs...)

"""
    autosymptr(f, bz::SymmetricBZ; kwargs...)
"""
autosymptr(f, bz::SymmetricBZ; kwargs...) =
    autosymptr(f, bz.B, bz.syms; kwargs...)
autosymptr_kwargs(f, bz::SymmetricBZ; kwargs...) =
    autosymptr_kwargs(f, bz.B, bz.syms; kwargs...)
