using Test
using LinearAlgebra
using StaticArrays
using FourierSeriesEvaluators
using AutoBZCore
using AutoBZCore: CubicLimits
using AutoBZCore: PuncturedInterval, HyperCube, segments, endpoints

@testset "FourierIntegralFunction" begin
    @testset "quadrature" begin
        a = 0
        b = 1
        p = 0.0
        t = 1.0
        s = FourierSeries([1, 0, 1]/2; period=t, offset=-2)
        int(x, s, p) = x * s + p
        ref = (b-a)*p + t*(b*sin(b/t*2pi) + t*cos(b/t*2pi) - (a*sin(a/t*2pi) + t*cos(a/t*2pi)))
        abstol = 1e-5
        prob = IntegralProblem(FourierIntegralFunction(int, s), (a, b), p; abstol)
        for alg in (QuadGKJL(), QuadratureFunction(), AuxQuadGKJL(), ContQuadGKJL(), MeroQuadGKJL())
            @test solve(prob, alg).value ≈ ref atol=abstol
        end
    end
    @testset "commonproblem" begin
        a = 0
        b = 1
        p = 0.0
        t = 1.0
        update! = (cache, x, s, p) -> cache.p = (x, s, p)
        postsolve = (sol, x, s, p) -> sol.value
        s = FourierSeries([1, 0, 1]/2; period=t, offset=-2)
        f = (x, (y, s, p)) -> x * s + p + y
        subprob = IntegralProblem(f, (a, b), ((a+b)/2, s((a+b)/2), 1.0))
        abstol = 1e-5
        prob = IntegralProblem(CommonSolveFourierIntegralFunction(subprob, QuadGKJL(), update!, postsolve, s), (a, b), p; abstol)
        for alg in (QuadGKJL(), HCubatureJL(), QuadratureFunction(), AuxQuadGKJL(), ContQuadGKJL(), MeroQuadGKJL())
            cache = init(prob, alg)
            for p in [3.0, 4.0]
                ref = (b-a)*(t*p + (b-a)^2/2) + (b-a)^2/2*t*(b*sin(b/t*2pi) + t*cos(b/t*2pi) - (a*sin(a/t*2pi) + t*cos(a/t*2pi)))
                cache.p = p
                sol = solve!(cache)
                @test ref ≈ sol.value atol=abstol
            end
        end
        f = (x, (y, s, p)) -> x * s + p
        subprob = IntegralProblem(f, (a, b), ([(a+b)/2], s((a+b)/2), 1.0))
        abstol = 1e-5
        prob = IntegralProblem(CommonSolveFourierIntegralFunction(subprob, QuadGKJL(), update!, postsolve, s), AutoBZCore.Basis(t*I(1)), p; abstol)
        for alg in (MonkhorstPack(), AutoSymPTRJL(),)
            cache = init(prob, alg)
            for p in [3.0, 4.0]
                ref = (b-a)*(t*p) + (b-a)^2/2*t*(b*sin(b/t*2pi) + t*cos(b/t*2pi) - (a*sin(a/t*2pi) + t*cos(a/t*2pi)))
                cache.p = p
                sol = solve!(cache)
                @test ref ≈ sol.value atol=abstol
            end
        end
    end
    @testset "cubature" for dim in 2:3
        a = zeros(dim)
        b = ones(dim)
        p = 0.0
        t = 1.0
        s = FourierSeries([prod(x) for x in Iterators.product([(0.1, 0.5, 0.3) for i in 1:dim]...)]; period=t, offset=-2)
        f = (x, s, p) -> prod(x) * s + p
        abstol = 1e-4
        prob = IntegralProblem(FourierIntegralFunction(f, s), (a, b), p; abstol)
        refprob = IntegralProblem(IntegralFunction(let f=f; (x, p) -> f(x, s(x), p); end), (a, b), p; abstol)
        for alg in (HCubatureJL(),)
            @test solve(prob, alg).value ≈ solve(refprob, alg).value atol=abstol
        end
        p=1.3
        f = (x, s, p) -> inv(s + im*p)
        prob = IntegralProblem(FourierIntegralFunction(f, s), AutoBZCore.Basis(t*I(dim)), p; abstol)
        refprob = IntegralProblem(IntegralFunction(let f=f; (x, p) -> f(x, s(x), p); end), AutoBZCore.Basis(t*I(dim)), p; abstol)
        for alg in (MonkhorstPack(), AutoSymPTRJL(),)
            @test solve(prob, alg).value ≈ solve(refprob, alg).value atol=abstol
        end
    end
    @testset "meta-algorithms" for dim in 2:3
        # NestedQuad
        a = zeros(dim)
        b = ones(dim)
        p0 = 0.0
        t = 1.0
        s = FourierSeries([prod(x) for x in Iterators.product([(0.1, 0.5, 0.3) for i in 1:dim]...)]; period=t, offset=-2)
        int(x, s, p) = prod(x) * s + p
        abstol = 1e-4
        prob = IntegralProblem(FourierIntegralFunction(int, s), CubicLimits(a, b), p0; abstol)
        refprob = IntegralProblem(FourierIntegralFunction(int, s), (a, b), p0; abstol)
        for alg in (QuadGKJL(), AuxQuadGKJL())
            cache = init(prob, NestedQuad(alg))
            refcache = init(refprob, HCubatureJL())
            for p in [5.0, 6.0]
                cache.p = p
                refcache.p = p
                @test solve!(cache).value ≈ solve!(refcache).value atol=abstol
            end
        end
        # TODO implement CommonSolveFourierInplaceIntegralFunction
        # TODO implement CommonSolveFourierInplaceBatchIntegralFunction
    end

    # EvalCounter
    @testset "evalcounter" for prob in (
        IntegralProblem(FourierIntegralFunction((x, s, p) -> x * s + p, FourierSeries([1, 0, 1]/2; period=1.0, offset=-2)), (0.0, 1.0), 0.0; abstol=1e-3),
        IntegralProblem(CommonSolveFourierIntegralFunction(IntegralProblem((x, (y, s, p)) -> x * s + p + y, (0.0, 1.0), (0.5, 0.0, 1.0)), QuadGKJL(), (cache, x, s, p) -> (cache.p = (x, s, p)), (sol, x, s, p) -> sol.value, FourierSeries([1, 0, 1]/2; period=1.0, offset=-2)), (0.0, 1.0), 1.0),
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
end

#=
@testset "FourierIntegrand" begin
    for dims in 1:3
        s = FourierSeries(integer_lattice(dims), period=1)
        # AutoBZ interface user function: f(x, args...; kwargs...) where args & kwargs
        # stored in MixedParameters
        # a FourierIntegrand should expect a FourierValue in the first argument
        # a FourierIntegrand is just a wrapper around an integrand
        f(x::FourierValue, a; b) = a*x.s*x.x .+ b
        # IntegralSolver will accept args & kwargs for a FourierIntegrand
        prob = IntegralProblem(FourierIntegrand(f, s, 1.3, b=4.2), zeros(dims), ones(dims))
        u = IntegralSolver(prob, HCubatureJL())()
        v = IntegralSolver(FourierIntegrand(f, s), zeros(dims), ones(dims), HCubatureJL())(1.3, b=4.2)
        w = IntegralSolver(FourierIntegrand(f, s, b=4.2), zeros(dims), ones(dims), HCubatureJL())(1.3)
        @test u == v == w

        # tests for the nested integrand
        nouter = 3
        ws = FourierSeriesEvaluators.workspace_allocate(s, FourierSeriesEvaluators.period(s), ntuple(n -> n == dims ? nouter : 1,dims))
        p = ParameterIntegrand(f, 1.3, b=4.2)
        nest = NestedBatchIntegrand(ntuple(n -> deepcopy(p), nouter), SVector{dims,ComplexF64})
        for (alg, dom) in (
            (HCubatureJL(), HyperCube(zeros(dims), ones(dims))),
            (NestedQuad(AuxQuadGKJL()), CubicLimits(zeros(dims), ones(dims))),
            (MonkhorstPack(), Basis(one(SMatrix{dims,dims}))),
        )
            prob1 = IntegralProblem(FourierIntegrand(p, s), dom)
            prob2 = IntegralProblem(FourierIntegrand(p, ws, nest), dom)
            @test solve(prob1, alg).u ≈ solve(prob2, alg).u
        end
    end
end
@testset "algorithms" begin
    f(x::FourierValue, a; b) = a*x.s+b
    for dims in 1:3
        vol = (2pi)^dims
        A = I(dims)
        s = FourierSeries(integer_lattice(dims), period=1)
        for bz in (load_bz(FBZ(), A), load_bz(InversionSymIBZ(), A))
            integrand = FourierIntegrand(f, s, 1.3, b=1.0)
            prob = IntegralProblem(integrand, bz)
            for alg in (IAI(), PTR(), AutoPTR(), TAI()), counter in (false, true)
                new_alg = counter ? EvalCounter(alg) : alg
                solver = IntegralSolver(prob, new_alg, reltol=0, abstol=1e-6)
                @test solver() ≈ vol atol=1e-6
            end
        end
    end
end
=#
