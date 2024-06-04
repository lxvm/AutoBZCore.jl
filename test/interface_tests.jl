using Test
using LinearAlgebra
using AutoBZCore
using AutoBZCore: PuncturedInterval, HyperCube, segments, endpoints
using AutoBZCore: CubicLimits

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
        for alg in (QuadratureFunction(), QuadGKJL(),)#(QuadratureFunction(), QuadGKJL(), AuxQuadGKJL(), ContQuadGKJL(), MeroQuadGKJL())
            sol = solve(prob, alg)
            @test ref ≈ sol.value atol=abstol
        end
    end
end

@testset "commonproblem" begin
    a = 1.0
    b = 2pi
    abstol=1e-5
    p=3.0
    # QuadratureFunction QuadGKJL AuxQuadGKJL ContQuadGKJL MeroQuadGKJL
    update! = (cache, x, p) -> cache.p = x
    postsolve = (sol, x, p) -> sol.value
    f = (x, p) -> p + x
    ref = (b-a)*(b^2-a^2)
    subprob = IntegralProblem(f, (a, b), (a+b)/2; abstol)
    integrand = CommonSolveIntegralFunction(subprob, QuadGKJL(), update!, postsolve)
    prob = IntegralProblem(integrand, (a, b), p; abstol)
    for alg in (QuadratureFunction(), QuadGKJL(), HCubatureJL()) #(QuadratureFunction(), QuadGKJL(), AuxQuadGKJL(), ContQuadGKJL(), MeroQuadGKJL())
        sol = solve(prob, alg)
        @test ref ≈ sol.value atol=abstol
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
        #=
        prob = IntegralProblem(f, Basis(b*I(dim)), p)
        for alg in (MonkhorstPack(), AutoSymPTRJL(),)
            @test ref ≈ solve(prob, alg, abstol=abstol).value atol=abstol
        end
        =#
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
        for alg in (QuadGKJL(),)#(QuadratureFunction(), QuadGKJL(), AuxQuadGKJL(), HCubatureJL(),)
            @test ref ≈ solve(inplaceprob, alg).value atol=abstol
        end
        #=
        inplaceprob = IntegralProblem(integrand, Basis([b;;]), p)
        for alg in (MonkhorstPack(), AutoSymPTRJL())
            @test ref ≈ solve(inplaceprob, alg, abstol=abstol).value atol=abstol
        end
        =#
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
        batchprob = IntegralProblem(integrand, (a, b), p)
        for alg in (QuadGKJL(),)#(QuadratureFunction(), AuxQuadGKJL())
            @test ref ≈ solve(batchprob, alg, abstol=abstol).value atol=abstol
        end
        #=
        batchprob = IntegralProblem(integrand, Basis([b;;]), p)
        for alg in (MonkhorstPack(), AutoSymPTRJL())
            @test ref ≈ solve(batchprob, alg, abstol=abstol).value atol=abstol
        end
        =#
    end
end

@testset "multi-algorithms" begin
    # NestedQuad
    f(x, p) = 1.0 + p*sum(cos, x)
    p = 7.0
    abstol=1e-3
    for dim in 1:3, alg in (QuadratureFunction(), QuadGKJL())# AuxQuadGKJL())
        ref = (2pi)^dim
        dom = CubicLimits(zeros(dim), 2pi*ones(dim))
        prob = IntegralProblem(f, dom, p; abstol)
        ndalg = NestedQuad(alg)
        @test ref ≈ solve(prob, ndalg).value atol=abstol
        # inplaceprob = IntegralProblem(InplaceIntegrand((y,x,p) -> y .= f(x,p), [0.0]), dom, p)
        # @test [ref] ≈ solve(inplaceprob, ndalg, abstol=abstol).value atol=abstol
        # batchprob = IntegralProblem(BatchIntegrand((y,x,p) -> y .= f.(x,Ref(p)), Float64), dom, p)
        # @test ref ≈ solve(batchprob, ndalg, abstol=abstol).value atol=abstol
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

    # EvalCounter
    for prob in (
        IntegralProblem((x, p) -> 1.0, 0, 1),
        IntegralProblem(InplaceIntegrand((y, x, p) -> y .= 1.0, fill(0.0)), 0, 1),
        IntegralProblem(BatchIntegrand((y, x, p) -> y .= 1.0, Float64), 0, 1)
    )
        # constant integrand should always use the same number of evaluations as the
        # base quadrature rule
        for (alg, numevals) in (
            (QuadratureFunction(npt=10), 10),
            (QuadGKJL(order=7), 15),
            (QuadGKJL(order=9), 19),
        )
            prob.f isa BatchIntegrand && alg isa QuadGKJL && continue
            @test solve(prob, EvalCounter(alg)).numevals == numevals
        end
    end
    =#
end
