using Test
using LinearAlgebra

using StaticArrays
using OffsetArrays

# using FourierSeriesEvaluators
# using IteratedIntegration
# using AutoSymPTR

using AutoBZCore
using AutoBZCore.IteratedIntegration
using AutoBZCore.AutoSymPTR
using AutoBZCore.FourierSeriesEvaluators


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
        lims = CubicLimits(zeros(3), ones(3))
        @test SymmetricBZ(A, B, lims) isa FullBZ
        fbz = FullBZ(A, B, lims)
        @test fbz isa FullBZ
        @test fbz.A ≈ A
        @test fbz.B ≈ B
        @test nsyms(fbz) == 1
        @test fbz.syms === nothing
        @test ndims(fbz) == dims
        @test eltype(fbz) == float(eltype(B))
        fbz = FullBZ(A)
        @test fbz.lims isa CubicLimits

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
        f = FourierIntegrand{3}(dos_integrand, s, p)
        @test FourierIntegrand{typeof(dos_integrand)}(s, p) == f
        @test iterated_integrand(f, (1.0, 1.0, 1.0), Val(1)) == f((1.0, 1.0, 1.0))
        @test iterated_integrand(f, 809, Val(0)) == iterated_integrand(f, 809, Val(2)) == 809
    end

    @testset "IteratedFourierIntegrand" begin
        safedos_integrand(H::AbstractMatrix, M) = tr(inv(M-H))/(-pi)
        fs = (safedos_integrand, imag, identity)
        s = InplaceFourierSeries(rand(SMatrix{3,3,ComplexF64}, 3,3,3))
        p = -I
        f = IteratedFourierIntegrand{3}(fs, s, p)
        @test IteratedFourierIntegrand{typeof(fs)}(s, p) == f
        @test iterated_integrand(f, (1.0, 1.0, 1.0), Val(1)) == f((1.0, 1.0, 1.0))
        @test iterated_integrand(f, complex(2.0, 1.1), Val(2)) == 1.1
        @test iterated_integrand(f, 809, Val(0)) == iterated_integrand(f, 809, Val(3)) == 809
    end

    @testset "IAI" begin
        dims = 3
        A = I(dims)
        B = AutoBZCore.canonical_reciprocal_basis(A)
        fbz = FullBZ(A, B)

        dos_integrand(H, M) = imag(tr(inv(M-H)))/(-pi)
        s = InplaceFourierSeries(integer_lattice(dims))
        p = complex(1.0,1.0)*I
        f = FourierIntegrand{3}(dos_integrand, s, p)

        iterated_integration(f, fbz)
    end

    @testset "PTR" begin
        dims = 3
        A = I(dims)
        B = AutoBZCore.canonical_reciprocal_basis(A)
        fbz = FullBZ(A, B)
        bz = SymmetricBZ(A, B, fbz.lims, (I,))

        dos_integrand(H, M) = imag(tr(inv(M-H)))/(-pi)
        s = InplaceFourierSeries(integer_lattice(dims))
        p = complex(1.0,1.0)*I
        f = FourierIntegrand{3}(dos_integrand, s, p)

        @test symptr(f, fbz)[1] ≈ symptr(f, bz)[1]
        @test autosymptr(f, fbz)[1] ≈ autosymptr(f, bz)[1]

    end

    @testset "FourierIntegrator" begin
        dims = 3
        A = I(dims)
        B = AutoBZCore.canonical_reciprocal_basis(A)
        fbz = FullBZ(A, B)
        bz = SymmetricBZ(A, B, fbz.lims, (I,))

        dos_integrand(H, M) = imag(tr(inv(M-H)))/(-pi)
        s = InplaceFourierSeries(integer_lattice(dims))
        p = complex(1.0,1.0)*I
        
        fiai = FourierIntegrator(dos_integrand, fbz, s; ps=(p,), routine=iterated_integration)
        fptr = FourierIntegrator(dos_integrand, fbz, s; ps=(p,), routine=symptr)
        fsym = FourierIntegrator(dos_integrand, fbz, s; ps=(p,), routine=autosymptr)

        bziai = FourierIntegrator(dos_integrand, bz, s; ps=(p,), routine=iterated_integration)
        bzptr = FourierIntegrator(dos_integrand, bz, s; ps=(p,), routine=symptr)
        bzsym = FourierIntegrator(dos_integrand, bz, s; ps=(p,), routine=autosymptr)

        @test fiai(p)[1] ≈ bziai(p)[1]
        @test fptr(p)[1] ≈ bzptr(p)[1]
        @test fsym(p)[1] ≈ bzsym(p)[1]
    end

end