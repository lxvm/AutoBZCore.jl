using Test
using LinearAlgebra
using AutoBZCore
using AutoBZCore: PuncturedInterval, HyperCube, segments, endpoints
using AutoBZCore: CubicLimits


struct TestProblem{A,B}
    a::A
    b::B
end
struct TestAlgorithm end
mutable struct TestSolver{A,B,K}
    a::A
    b::B
    k::K
end
function AutoBZCore.init(prob::TestProblem, alg::TestAlgorithm; kws...)
    return TestSolver(prob.a, prob.b, NamedTuple(kws))
end
function AutoBZCore.solve!(solver::TestSolver)
    (; a, b, k) = solver
    return a+b
end
function testup!(solver, x, p)
    solver.a = x
    solver.b = p
    return
end
function testpost(sol, x, p)
    return sol
end

@testset "domains" begin
    # PuncturedInterval
    a = (0.0, 1.0, 2.0)
    b = collect(a)
    sa = PuncturedInterval(a)
    sb = PuncturedInterval(b)
    @test all(segments(sa) .== segments(sb))
    @test (0.0, 2.0) == endpoints(sa) == endpoints(sb)
    @test Float64 == eltype(sa) == eltype(sb)

    # HyperCube
    for d = 1:3
        c = HyperCube(zeros(d), ones(d))
        @test eltype(c) == Float64
        a, b = endpoints(c)
        @test all(a .== zeros(d))
        @test all(b .== ones(d))
    end
end

@testset "quadrature" begin
    a = 0.0
    b = 2pi
    abstol=1e-5
    p=3.0
    # QuadratureFunction QuadGKJL AuxQuadGKJL ContQuadGKJL MeroQuadGKJL
    for (f, ref) in (
        ((x,p) -> p*sin(x), 0.0),
        ((x,p) -> p*one(x), p*(b-a)),
        ((x,p) -> inv(p-cos(x)), (b-a)/sqrt(p^2-1)),
    )
        prob = IntegralProblem(f, (a, b), p; abstol)
        for alg in (QuadratureFunction(), QuadGKJL(), AuxQuadGKJL(), ContQuadGKJL(), MeroQuadGKJL())
            sol = solve(prob, alg)
            @test ref ≈ sol.value atol=abstol
        end
    end
    @test @inferred(solve(IntegralProblem((x, p) -> exp(-x^2), (-Inf, Inf)), QuadGKJL())).value ≈ sqrt(pi)
end

@testset "commonproblem" begin
    a = 1.0
    b = 2pi
    abstol=1e-5
    p0=3.0
    # QuadratureFunction QuadGKJL AuxQuadGKJL ContQuadGKJL MeroQuadGKJL
    update! = (cache, x, p) -> cache.p = (x, p)
    postsolve = (sol, x, p) -> sol.value
    f = (x, (y, p)) -> p*(y + x)
    subprob = IntegralProblem(f, (a, b), ((a+b)/2, p0); abstol)
    integrand = CommonSolveIntegralFunction(subprob, QuadGKJL(), update!, postsolve)
    prob = IntegralProblem(integrand, (a, b), p0; abstol)
    for alg in (QuadratureFunction(), QuadGKJL(), HCubatureJL(), AuxQuadGKJL(), ContQuadGKJL(), MeroQuadGKJL())
        cache = init(prob, alg)
        for p in [3.0, 4.0]
            ref = p*(b-a)*(b^2-a^2)
            cache.p = p
            sol = solve!(cache)
            @test ref ≈ sol.value atol=abstol
        end
    end
    f = (x, (y, p)) -> p*(sin(only(y))^2 + x)
    subprob = IntegralProblem(f, (a, b), ([b/2], p0); abstol)
    integrand = CommonSolveIntegralFunction(subprob, QuadGKJL(), update!, postsolve)
    prob = IntegralProblem(integrand, AutoBZCore.Basis(b*I(1)), p0; abstol)
    for alg in (MonkhorstPack(), AutoSymPTRJL(),)
        cache = init(prob, alg)
        for p in [3.0, 4.0]
            ref = p*((b-a)+(b^2-a^2))*b/2
            cache.p = p
            sol = solve!(cache)
            @test ref ≈ sol.value atol=abstol
        end
    end
end

@testset "cubature" begin
    # HCubatureJL MonkhorstPack AutoSymPTRJL NestedQuad
    a = 0.0
    b = 2pi
    abstol=1e-5
    p = 3.0
    for dim in 1:3, (f, ref) in (
        ((x,p) -> p*sum(sin, x), 0.0),
        ((x,p) -> p*one(eltype(x)), p*(b-a)^dim),
        ((x,p) -> prod(y -> inv(p-cos(y)), x), ((b-a)/sqrt(p^2-1))^dim),
    )
        prob = IntegralProblem(f, (fill(a, dim), fill(b, dim)), p; abstol)
        for alg in (HCubatureJL(),)
            @test ref ≈ solve(prob, alg).value atol=abstol
        end
        prob = IntegralProblem(f, AutoBZCore.Basis(b*I(dim)), p; abstol)
        for alg in (MonkhorstPack(), AutoSymPTRJL(),)
            @test ref ≈ solve(prob, alg).value atol=abstol
        end
    end
end

@testset "inplace" begin
    # QuadratureFunction QuadGKJL AuxQuadGKJL HCubatureJL MonkhorstPack AutoSymPTRJL
    a = 0.0
    b = 2pi
    abstol=1e-5
    p = 3.0
    for (f, ref) in (
        ((y,x,p) -> y .= p*sin(only(x)), [0.0]),
        ((y,x,p) -> y .= p*one(only(x)), [p*(b-a)]),
        ((y,x,p) -> y .= inv(p-cos(only(x))), [(b-a)/sqrt(p^2-1)]),
    )
        integrand = InplaceIntegralFunction(f, [0.0])
        inplaceprob = IntegralProblem(integrand, (a, b), p; abstol)
        for alg in (QuadGKJL(), QuadratureFunction(), QuadGKJL(), AuxQuadGKJL())
            @test ref ≈ solve(inplaceprob, alg).value atol=abstol
        end
        inplaceprob = IntegralProblem(integrand, AutoBZCore.Basis([b;;]), p; abstol)
        for alg in (MonkhorstPack(), AutoSymPTRJL())
            @test ref ≈ solve(inplaceprob, alg).value atol=abstol
        end
    end
end


@testset "batch" begin
    # QuadratureFunction AuxQuadGKJL MonkhorstPack AutoSymPTRJL
    a = 0.0
    b = 2pi
    abstol=1e-5
    p = 3.0
    for (f, ref) in (
        ((y,x,p) -> y .= p .* sin.(only.(x)), 0.0),
        ((y,x,p) -> y .= p .* one.(only.(x)), p*(b-a)),
        ((y,x,p) -> y .= inv.(p .- cos.(only.(x))), (b-a)/sqrt(p^2-1)),
    )
        integrand = InplaceBatchIntegralFunction(f, zeros(1))
        batchprob = IntegralProblem(integrand, (a, b), p; abstol)
        for alg in (QuadGKJL(), QuadratureFunction(), AuxQuadGKJL())
            @test ref ≈ solve(batchprob, alg).value atol=abstol
        end
        batchprob = IntegralProblem(integrand, AutoBZCore.Basis([b;;]), p; abstol)
        for alg in (MonkhorstPack(), AutoSymPTRJL())
            @test ref ≈ solve(batchprob, alg).value atol=abstol
        end
    end
end
@testset "multi-algorithms" begin
    # NestedQuad
    f(x, p) = 1.0 + p*sum(abs2 ∘ cos, x)
    abstol=1e-3
    p0 = 0.0
    for dim in 1:3, alg in (QuadratureFunction(), QuadGKJL(), AuxQuadGKJL())
        dom = CubicLimits(zeros(dim), 2pi*ones(dim))
        prob = IntegralProblem(f, dom, p0; abstol)
        ndalg = NestedQuad(alg)
        cache = init(prob, ndalg)
        for p in [5.0, 7.0]
            cache.p = p
            ref = (2pi)^dim + dim*p*pi*(2pi)^(dim-1)
            @test ref ≈ solve!(cache).value atol=abstol
            # TODO implement CommonSolveInplaceIntegralFunction
            inplaceprob = IntegralProblem(InplaceIntegralFunction((y,x,p) -> y .= f(x,p), [0.0]), dom, p)
            @test_broken [ref] ≈ solve(inplaceprob, ndalg, abstol=abstol).value atol=abstol
            # TODO implement CommonSolveInplaceBatchIntegralFunction
            batchprob = IntegralProblem(InplaceBatchIntegralFunction((y,x,p) -> y .= f.(x,Ref(p)), zeros(Float64, 1)), dom, p)
            @test_broken ref ≈ solve(batchprob, ndalg, abstol=abstol).value atol=abstol
        end
    end
    #=
    # AbsoluteEstimate
    est_alg = QuadratureFunction()
    abs_alg = QuadGKJL()
    alg = AbsoluteEstimate(est_alg, abs_alg)
    ref_alg = MeroQuadGKJL()
    f2(x, p) = inv(complex(p...) - cos(x))
    prob = IntegralProblem(f2, 0.0, 2pi, (0.5, 1e-3))
    abstol = 1e-5; reltol=1e-5
    @test solve(prob, alg, reltol=reltol).value ≈ solve(prob, ref_alg, abstol=abstol).value atol=abstol
    =#

    # EvalCounter
    for prob in (
        IntegralProblem((x, p) -> 1.0, (0, 1)),
        IntegralProblem(InplaceIntegralFunction((y, x, p) -> y .= 1.0, fill(0.0)), (0, 1)),
        IntegralProblem(InplaceBatchIntegralFunction((y, x, p) -> y .= 1.0, [0.0]), (0, 1)),
        IntegralProblem(CommonSolveIntegralFunction(TestProblem(0.0, 0.0), TestAlgorithm(), testup!, testpost, 0.0), (0, 1), 3.0),
    )
        # constant integrand should always use the same number of evaluations as the
        # base quadrature rule
        for (alg, numevals) in (
            (QuadratureFunction(npt=10), 10),
            (QuadGKJL(order=7), 15),
            (QuadGKJL(order=9), 19),
        )
            @test solve(prob, EvalCounter(alg)).stats.numevals == numevals
        end
    end
    @test solve(IntegralProblem((x, p) -> 1.0, CubicLimits((0,0), (1,1))), EvalCounter(NestedQuad(QuadGKJL(order=7)))).stats.numevals == 15^2
end
