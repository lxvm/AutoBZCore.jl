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

        ibz = IBZ()
        @test_throws ArgumentError load_bz(ibz)
        ibz3 = IBZ(3)
        @test_throws "SymmetryReduceBZ" load_bz(ibz3)
    end
    @testset "n symmetries" for d in 1:3
        @test AutoBZCore.n_permutations(d) == factorial(d)
        @test AutoBZCore.n_sign_flips(d) == 2^d
        @test AutoBZCore.n_cube_automorphisms(d) == 2^d * factorial(d)
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
        @test solve(ip, EvalCounter(PTR(; npt=10))).stats.numevals == 10^dims
    end
end
