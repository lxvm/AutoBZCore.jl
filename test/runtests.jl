using Test
using LinearAlgebra

using StaticArrays
using OffsetArrays

using AutoBZCore


function integer_lattice(n)
    C = OffsetArray(zeros(ntuple(_ -> 3, n)), repeat([-1:1], n)...)
    for i in 1:n, j in (-1, 1)
        C[CartesianIndex(ntuple(k -> k == i ? j : 0, n))] = 1/2n
    end
    C
end

@testset "AutoBZCore" begin

    @testset "SymmetricBZ" begin
        dims = 3
        A = I(dims)
        B = AutoBZCore.canonical_reciprocal_basis(A)
        lims = AutoBZCore.CubicLimits(zeros(3), ones(3))
        fbz = FullBZ(A, B, lims)
        @test fbz == SymmetricBZ(A, B, lims, nothing)
        @test fbz.A ≈ A
        @test fbz.B ≈ B
        @test nsyms(fbz) == 1
        @test fbz.syms === nothing
        @test ndims(fbz) == dims
        @test eltype(fbz) == float(eltype(B))
        fbz = FullBZ(A)
        @test fbz.lims isa AutoBZCore.CubicLimits

        nsym = 8
        syms = rand(SMatrix{dims,dims}, nsym)
        bz = SymmetricBZ(A, B, lims, syms)
        @test nsyms(bz) == nsym
        @test bz.syms == syms
        @test ndims(bz) == dims
        @test eltype(bz) == float(eltype(B))
    end

    @testset "FourierIntegrand" begin
        dos_integrand(H::AbstractMatrix, M) = imag(tr(inv(M-H)))/(-pi)
        s = InplaceFourierSeries(rand(SMatrix{3,3,ComplexF64}, 3,3,3))
        p = -I
        f = AutoBZCore.construct_integrand(FourierIntegrand(dos_integrand, s), false, tuple(p))
        @test f == FourierIntegrand(dos_integrand, s, p)
        @test AutoBZCore.iterated_integrand(f, (1.0, 1.0, 1.0), Val(1)) == f((1.0, 1.0, 1.0))
        @test AutoBZCore.iterated_integrand(f, 809, Val(0)) == AutoBZCore.iterated_integrand(f, 809, Val(2)) == 809
    end

    dims = 3
    A = I(dims)
    B = AutoBZCore.canonical_reciprocal_basis(A)
    fbz = FullBZ(A, B)
    bz = SymmetricBZ(A, B, fbz.lims, (I,))

    dos_integrand(H, M) = imag(tr(inv(M-H)))/(-pi)
    s = InplaceFourierSeries(integer_lattice(dims), period=1)
    p = complex(1.0,1.0)*I
    f = FourierIntegrand(dos_integrand, s)

    ip_fbz = IntegralProblem(f, fbz, (p,))
    ip_bz = IntegralProblem(f, bz, (p,))
    
    @testset "IntegralProblem interface" begin
        g = FourierIntegrand(dos_integrand, s, p)
        ip_fbz_g = IntegralProblem(g, fbz)
        ip_bz_g = IntegralProblem(g, bz)

        for (ip1, ip2) in ((ip_fbz, ip_fbz_g), (ip_bz, ip_bz_g))
            intf = AutoBZCore.construct_integrand(ip1.f, isinplace(ip1), ip1.p)
            intg = AutoBZCore.construct_integrand(ip2.f, isinplace(ip2), ip2.p)
            @test intf == intg
        end
    end

    @testset "Algorithms" begin
        @test solve(ip_fbz, IAI()) ≈ solve(ip_bz, IAI())
        @test solve(ip_fbz, TAI()) ≈ solve(ip_bz, TAI())
        @test solve(ip_fbz, PTR()) ≈ solve(ip_bz, PTR())
        @test solve(ip_fbz, AutoPTR()) ≈ solve(ip_bz, AutoPTR())
        @test solve(ip_fbz, PTR_IAI()) ≈ solve(ip_bz, PTR_IAI())
        @test solve(ip_fbz, AutoPTR_IAI()) ≈ solve(ip_bz, AutoPTR_IAI())
        # @test solve(ip_fbz, VEGAS()) ≈ solve(ip_bz, VEGAS()) # skip for now or
        # set larger tolerance
    end

end