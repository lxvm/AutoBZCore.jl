module UnitfulExt

    using Unitful: Quantity, unit, ustrip
    import AutoBZCore: canonical_reciprocal_basis, canonical_ptr_basis

    function canonical_reciprocal_basis(A::AbstractMatrix{<:Quantity})
        return canonical_reciprocal_basis(ustrip.(A)) / unit(eltype(A))
    end
    function canonical_ptr_basis(B::AbstractMatrix{<:Quantity})
        return canonical_ptr_basis(ustrip.(B))
    end
end
