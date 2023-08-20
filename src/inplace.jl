"""
    InplaceIntegrand(f!, result::AbstractArray)

Constructor for a `InplaceIntegrand` accepting an integrand of the form `f!(y,x,p)`. The
caller also provides an output array needed to store the result of the quadrature.
Intermediate `y` arrays are allocated during the calculation, and the final result is
may or may not be written to `result`, so use the IntegralSolution immediately after the
calculation to read the result, and don't expect it to persist if the same integrand is used
for another calculation.
"""
struct InplaceIntegrand{F,T<:AbstractArray}
    # in-place function f!(y, x, p) that takes one x value and outputs an array of results in-place
    f!::F
    I::T
end
