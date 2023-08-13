module UnitfulExt

    using Unitful: Quantity, unit, ustrip
    import AutoBZCore: canonical_reciprocal_basis

    function canonical_reciprocal_basis(A::AbstractMatrix{<:Quantity})
        return canonical_reciprocal_basis(ustrip(A)) / unit(eltype(A))
    end
end
