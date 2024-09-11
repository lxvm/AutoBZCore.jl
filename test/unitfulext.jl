using Test
using Unitful
using AutoBZCore
using AutoBZCore: canonical_reciprocal_basis, canonical_ptr_basis
using LinearAlgebra: I
using StaticArrays

for A in [rand(3, 3) * u"m", rand(SMatrix{3,3,Float64,9})*u"m"]
    B = canonical_reciprocal_basis(A)
    @test B'A ≈ 2pi*I
    pB = canonical_ptr_basis(B)
    @test pB isa AutoBZCore.Basis
    @test pB.B ≈ I
end