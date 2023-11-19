using Test, LinearAlgebra, OffsetArrays, StaticArrays, AutoBZCore
using AutoBZCore: IntegralProblem, solve, MixedParameters

function integer_lattice(n, t=1/n)
    C = OffsetArray(zeros(ntuple(_ -> 3, n)), repeat([-1:1], n)...)
    for i in 1:n, j in (-1, 1)
        C[CartesianIndex(ntuple(k -> k == i ? j : 0, n))] = t/2
    end
    C
end

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
            @test solve(prob1, alg) == solve(prob2, alg)
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

    # check NestedIntegrand gets integrand evaluations correct (i.e. NestedQuad with EvalCounter)
    prob = IntegralProblem(FourierIntegrand(f, FourierSeries(integer_lattice(2), period=3), b=1.0), CubicLimits((0.0, 0.0), (1.0, 1.0)), 2.0)
    sol1 = solve(prob, EvalCounter(NestedQuad(QuadGKJL(order=7))))
    sol2 = solve(prob, NestedQuad(EvalCounter(QuadGKJL(order=7))))
    nest = NestedIntegrand(FourierIntegrand(FourierSeries(integer_lattice(1, 1/2), period=3)) do x, p
        f_ = FourierIntegrand((y,q; kws...) -> f(y,q; kws...)+x.s*p, FourierSeries(integer_lattice(1, 1/2), period=3), b=1.0)
        prb = IntegralProblem(f_, 0.0, 1.0, p)
        return solve(prb, EvalCounter(QuadGKJL(order=7)))
    end)
    sol3 = solve(IntegralProblem(nest, 0.0, 1.0, 2.0), EvalCounter(QuadGKJL(order=7)))
    @test sol1.numevals == sol2.numevals == sol3.numevals == (2*7+1)^2
end
