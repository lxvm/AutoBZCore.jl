using Test, LinearAlgebra
using AutoBZCore
using AutoBZCore: IntegralProblem, solve, MixedParameters
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
        ip = IntegralProblem((x,p) -> 1.0, bz)  # unit measure
        for alg in (IAI(), TAI(), PTR(), AutoPTR())
            @test @inferred(solve(ip, alg)).u ≈ vol
        end
        @test @inferred(solve(IntegralProblem((x, p) -> exp(-x^2), -Inf, Inf), QuadGKJL())).u ≈ sqrt(pi)
    end
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
        prob = IntegralProblem(f, a, b, 33) # ordinary integrands get parameters replaced
        solver = IntegralSolver(prob, QuadGKJL())
        @test solver(p) == solve(IntegralProblem(f, a, b, p), QuadGKJL()).u
        # AutoBZ interface: (bz, alg)
        dims = 3
        A = I(dims)
        B = AutoBZCore.canonical_reciprocal_basis(A)
        bz = load_bz(FBZ(), A, B)
        prob = IntegralProblem(f, bz, p)
        solver = IntegralSolver(IntegralProblem(f, bz), IAI())
        @test solver(p) == solve(prob, IAI()).u # use the plain interface
        # g = (x,p) -> sum(x)*p[1]+p.a
        # solver2 = IntegralSolver(ParameterIntegrand(g), bz, IAI()) # use the MixedParameters interface
        # prob2 = IntegralProblem(g, bz, MixedParameters(12.0, a=1.0))
        # @test solver2(12.0, a=1.0) == solve(prob2, IAI()).u
    end
    @testset "ParameterIntegrand" begin
        # AutoBZ interface user function: f(x, args...; kwargs...) where args & kwargs
        # stored in MixedParameters
        f(x, a; b) = a*x+b
        # SciML interface for ParameterIntegrand: f(x, p) (# and parameters can be preloaded and
        # p is merged with MixedParameters)
        @test f(6.7, 1.3, b=4.2) == ParameterIntegrand(f, 1.3, b=4.2)(6.7, AutoBZCore.NullParameters()) == ParameterIntegrand(f)(6.7, MixedParameters(1.3, b=4.2))
        # A ParameterIntegrand merges its parameters with the problem's
        prob = IntegralProblem(ParameterIntegrand(f, 1.3, b=4.2), 0, 1)
        u = IntegralSolver(prob, QuadGKJL())()
        v = IntegralSolver(ParameterIntegrand(f), 0, 1, QuadGKJL())(1.3, b=4.2)
        w = IntegralSolver(ParameterIntegrand(f, b=4.2), 0, 1, QuadGKJL())(1.3)
        @test u == v == w
        @test solve(prob, EvalCounter(QuadGKJL(order=7))).numevals == 15
    end
    @testset "batchsolve" begin
        # SciML interface: iterable of parameters
        prob = IntegralProblem((x, p) -> p, 0, 1)
        solver = IntegralSolver(prob, QuadGKJL())
        params = range(1, 2, length=3)
        @test [solver(p) for p in params] == batchsolve(solver, params)
        # AutoBZ interface: array of MixedParameters
        f(x, a; b) = a*x+b
        solver = IntegralSolver(ParameterIntegrand(f), 0, 1, QuadGKJL())
        as = rand(3); bs = rand(3)
        @test [solver(a, b=b) for (a,b) in Iterators.zip(as, bs)] == batchsolve(solver, paramzip(as, b=bs))
        @test [solver(a, b=b) for (a,b) in Iterators.product(as, bs)] == batchsolve(solver, paramproduct(as, b=bs))
    end
end
