"""
    symptr(f, bz::SymmetricBZ; kwargs...)
"""
function symptr(f, bz::SymmetricBZ; kwargs...)
    int, rules = symptr(f, bz.B, bz.syms; kwargs...)
    symmetrize(f, bz, int), rules
end
symptr_kwargs(f, bz::SymmetricBZ; kwargs...) =
    symptr_kwargs(f, bz.B, bz.syms; kwargs...)

"""
    autosymptr(f, bz::SymmetricBZ; kwargs...)
"""
function autosymptr(f, bz::SymmetricBZ; kwargs...)
    int, err, numevals, rules = autosymptr(f, bz.B, bz.syms; kwargs...)
    symmetrize(f, bz, int, err)..., numevals, rules
end
autosymptr_kwargs(f, bz::SymmetricBZ; kwargs...) =
    autosymptr_kwargs(f, bz.B, bz.syms; kwargs...)
