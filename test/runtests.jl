using Test
using LinearAlgebra

using StaticArrays

# using FourierSeriesEvaluators
# using IteratedIntegration
# using AutoSymPTR

using AutoBZCore
using AutoBZCore.IteratedIntegration


@testset "AutoBZCore" begin
    
    @testset "SymmetricBZ" begin
        dims = 3
        A = I(dims)
        B = AutoBZCore.canonical_reciprocal_basis(A)
        lims = CubicLimits(zeros(3), ones(3))
        fbz = SymmetricBZ(A, B, lims)
    end

    @testset "FourierIntegrand" begin
        
    end

    @testset "IteratedFourierIntegrand" begin
        
    end

    @testset "FourierIntegrator" begin
        
    end

end