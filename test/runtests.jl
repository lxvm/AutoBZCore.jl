using Test
using LinearAlgebra

using StaticArrays
using OffsetArrays

using HDF5
using FourierSeriesEvaluators
using AutoBZCore


function integer_lattice(n)
    C = OffsetArray(zeros(ntuple(_ -> 3, n)), repeat([-1:1], n)...)
    for i in 1:n, j in (-1, 1)
        C[CartesianIndex(ntuple(k -> k == i ? j : 0, n))] = 1/2n
    end
    C
end

@testset "AutoBZCore" begin

    @testset "domains" begin
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
    end

    @testset "algorithms" begin
        dims = 3
        A = I(dims)
        B = AutoBZCore.canonical_reciprocal_basis(A)
        bz = FullBZ(A, B)
        vol = (2π)^dims
        ip = IntegralProblem((x,p) -> 1.0, bz)  # unit measure
        T = SciMLBase.IntegralSolution{Float64,0}
        for alg in (IAI(), TAI(), PTR(), AutoPTR(), PTR_IAI(), AutoPTR_IAI(), AuxIAI())
            # we need to specify do_inf_transformation=Val(true/false) for any type stability
            @test @inferred(T, solve(ip, alg; do_inf_transformation=Val(false))).u ≈ vol
        end
        @test @inferred(T, solve(IntegralProblem((x, p) -> exp(-x^2), -Inf, Inf), AuxQuadGK())).u ≈ sqrt(pi)
    end

    @testset "interfaces" begin
        @testset "MixedParameters" begin
            args = (1, 2)
            kwargs = (a = "a", b = "b")
            p = MixedParameters(args...)
            q = MixedParameters(; kwargs...)
            # we only merge non-MixedParameters in the right argument
            for pq = (merge(p, q), merge(p, kwargs), merge(q, args))
                @test args[1] == pq[1]
                @test args[2] == pq[2]
                @test kwargs.a == pq.a
                @test kwargs.b == pq.b
            end
            @test 3 == merge(p, 3)[3] == merge(q, 3)[1] # generic types are appended
            @test "c" == merge(p, (a="c",)).a == merge(q, (a="c",)).a # keywords overwritten
        end
        @testset "IntegralSolver" begin
            f = (x,p) -> p
            p = 0.81
            # SciML interface: (a, b, alg)
            a = 0; b = 1
            solver = IntegralSolver(f, a, b, QuadGKJL())
            @test solver(p) == solve(IntegralProblem(f, a, b, p), QuadGKJL()).u
            # AutoBZ interface: (bz, alg) and do_inf_transformation=Val(false)
            dims = 3
            A = I(dims)
            B = AutoBZCore.canonical_reciprocal_basis(A)
            bz = FullBZ(A, B)
            solver = IntegralSolver(f, bz, IAI())
            @test solver(p) == solve(IntegralProblem(f, bz, p), IAI(), do_inf_transformation=Val(false)).u
        end
        @testset "Integrands" begin
            # AutoBZ interface user function: f(x, args...; kwargs...) where args & kwargs
            # stored in MixedParameters
            f(x, a; b) = a*x+b
            # SciML interface for Integrand: f(x, p) (# and parameters can be preloaded and
            # p is merged with MixedParameters)
            @test f(6.7, 1.3, b=4.2) == Integrand(f, 1.3, b=4.2)(6.7) == Integrand(f)(6.7, MixedParameters(1.3, b=4.2))
            # IntegralSolver will accept args & kwargs for an Integrand
            u = IntegralSolver(Integrand(f, 1.3, b=4.2), 0, 1, QuadGKJL())()
            v = IntegralSolver(Integrand(f), 0, 1, QuadGKJL())(1.3, b=4.2)
            @test u == v
        end
        @testset "batchsolve" begin
            # SciML interface: iterable of parameters
            solver = IntegralSolver((x, p) -> p, 0, 1, QuadGKJL())
            params = range(1, 2, length=3)
            @test [solver(p) for p in params] == batchsolve(solver, params)
            # AutoBZ interface: array of MixedParameters
            f(x, a; b) = a*x+b
            solver = IntegralSolver(Integrand(f), 0, 1, QuadGKJL(), do_inf_transformation=Val(false))
            as = rand(3); bs = rand(3)
            @test [solver(a, b=b) for (a,b) in Iterators.zip(as, bs)] == batchsolve(solver, paramzip(as, b=bs))
            @test [solver(a, b=b) for (a,b) in Iterators.product(as, bs)] == batchsolve(solver, paramproduct(as, b=bs))
        end
    end

end
#=
@testset "FourierExt" begin
    @testset "FourierIntegrand" begin


    for (integrand, p, T, args, kwargs...) in (
        (dos_integrand, p_dos, Float64, ([i*I for i in 1:3],), ),
        (gloc_integrand, p_gloc, ComplexF64, (), :η => ones(3), :ω => 1:3)
    )

        f = Integrand(integrand)

        ip_fbz = IntegralProblem(f, fbz, p)
        ip_bz = IntegralProblem(f, bz, p)
    end
    dims = 3
    A = I(dims)
    B = AutoBZCore.canonical_reciprocal_basis(A)
    fbz = FullBZ(A, B)
    bz = SymmetricBZ(A, B, fbz.lims, (I,))

    s = InplaceFourierSeries(integer_lattice(dims), period=1)

    dos_integrand(k, H, M) = imag(tr(inv(M-H(k))))/(-pi)      # test integrand with positional arguments
    p_dos = MixedParameters(s, complex(1.0,1.0)*I)

    gloc_integrand(k, h; η, ω) = inv(complex(ω,η)*I-h(k)) # test integrand with keyword arguments
    p_gloc = MixedParameters(s; η=1.0, ω=0.0)

        dos_integrand(H::AbstractMatrix, M) = imag(tr(inv(M-H)))/(-pi)
        s = InplaceFourierSeries(rand(SMatrix{3,3,ComplexF64}, 3,3,3))
        p = -I
        f = AutoBZCore.construct_integrand(Integrand(dos_integrand, s), false, tuple(p))
        @test f == Integrand(dos_integrand, s, p)
        @test AutoBZCore.iterated_integrand(f, (1.0, 1.0, 1.0), Val(1)) == f((1.0, 1.0, 1.0))
        @test AutoBZCore.iterated_integrand(f, 809, Val(0)) == AutoBZCore.iterated_integrand(f, 809, Val(2)) == 809
    end
end

@testset "HDF5Ext" begin

    @testset "begin" begin

    end
    @testset "SArray" begin

    end
    @testset "AuxValue" begin

    end
end
=#
