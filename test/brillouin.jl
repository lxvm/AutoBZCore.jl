using Test
using LinearAlgebra
using AutoBZCore
using AutoBZCore: PuncturedInterval, HyperCube, segments, endpoints

@testset "domains" begin
    @testset "SymmetricBZ" begin
        dims = 3
        A = I(dims)
        B = AutoBZCore.canonical_reciprocal_basis(A)
        fbz = load_bz(FBZ(), A)
        @test fbz.A ≈ A
        @test fbz.B ≈ B
        @test nsyms(fbz) == 1
        @test fbz.lims == AutoBZCore.CubicLimits(zeros(3), ones(3))

        ibz = load_bz(InversionSymIBZ(), A)
        @test ibz.A ≈ A
        @test ibz.B ≈ B
        @test nsyms(ibz) == 2^dims
        @test all(isdiag, ibz.syms)
        @test ibz.lims == AutoBZCore.CubicLimits(zeros(3), 0.5*ones(3))

        cbz = load_bz(CubicSymIBZ(), A)
        @test cbz.A ≈ A
        @test cbz.B ≈ B
        @test nsyms(cbz) == factorial(dims)*2^dims
        @test cbz.lims == AutoBZCore.TetrahedralLimits(ntuple(n -> 0.5, dims))
    end
end

@testset "algorithms" begin
    dims = 3
    A = I(dims)
    vol = (2π)^dims
    for bz in (load_bz(FBZ(), A), load_bz(InversionSymIBZ(), A))
        ip = AutoBZProblem((x,p) -> 1.0, bz)  # unit measure
        for alg in (IAI(), TAI(), PTR(), AutoPTR())
            solver = init(ip, alg)
            @test @inferred(solve!(solver)).value ≈ vol
        end
    end
end
