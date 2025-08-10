using Test
using AutoBZCore
using FourierSeriesEvaluators
using LinearAlgebra: I

ntasks = 6
max_batch = 1000
#=
@testset "threaded executor" begin
    let
        f = (x, p) -> sin(x[1]*p)
        proto = 0.0
        ifs = IntegralFunction(f, proto, SerialExecutor())
        dom = (0.0, 1.0)
        p = 2.0
        probs = IntegralProblem(ifs, dom, p)
        alg = QuadGKJL()
        sols = solve(probs, alg)
        ift = IntegralFunction(f, proto, ThreadedExecutor(ntasks, max_batch))
        probt = IntegralProblem(ift, dom, p)
        solt = solve(probt, alg)
        @test sols.value ≈ solt.value

        alg = QuadratureFunction()
        sols = solve(probs, alg)
        solt = solve(probt, alg)
        @test sols.value ≈ solt.value

        @test solve(IntegralProblem(ift, AutoBZCore.PuncturedInterval(dom), p), NestedQuad(QuadGKJL())).value ≈ solve(IntegralProblem(ifs, AutoBZCore.PuncturedInterval(dom), p), NestedQuad(QuadGKJL())).value

        @test_throws ["ArgumentError", "HCubatureJL does not support"] solve(probt, HCubatureJL())
    end
    let
        iprob = IntegralProblem((x, p) -> x*p, (0, 1), 3.0)
        proto = 0.0
        _solve! = function (solver, y, p)
            solver.p = p + sum(y)
            return solve!(solver).value
        end
        ifs = CommonSolveIntegralFunction(_solve!, iprob, QuadGKJL(), proto, DefaultSpecialize(), SerialExecutor())
        dom = (0.0, 1.0)
        p = 2.0
        probs = IntegralProblem(ifs, dom, p)
        alg = QuadGKJL()
        sols = solve(probs, alg)
        ift = CommonSolveIntegralFunction(_solve!, iprob, QuadGKJL(), proto, DefaultSpecialize(), ThreadedExecutor(ntasks, max_batch))
        probt = IntegralProblem(ift, dom, p)
        solt = solve(probt, alg)
        @test sols.value ≈ solt.value

        alg = QuadratureFunction()
        sols = solve(probs, alg)
        solt = solve(probt, alg)
        @test sols.value ≈ solt.value

        @test solve(IntegralProblem(ift, AutoBZCore.PuncturedInterval(dom), p), NestedQuad(QuadGKJL())).value ≈ solve(IntegralProblem(ifs, AutoBZCore.PuncturedInterval(dom), p), NestedQuad(QuadGKJL())).value

        @test_throws ["ArgumentError", "HCubatureJL does not support"] solve(probt, HCubatureJL())

        dom = AutoBZCore.HyperCube((0,0), (1,1))
        solss = solve(IntegralProblem(ifs, dom, p), NestedQuad(QuadGKJL(), QuadGKJL()))
        solst = solve(IntegralProblem(ifs, dom, p), NestedQuad((QuadGKJL(), QuadGKJL()), (NoSpecialize(),), (ThreadedExecutor(ntasks, max_batch),)))
        soltt = solve(IntegralProblem(ift, dom, p), NestedQuad((QuadGKJL(), QuadGKJL()), (NoSpecialize(),), (ThreadedExecutor(ntasks, max_batch),)))
        solts = solve(IntegralProblem(ift, dom, p), NestedQuad(QuadGKJL()))
        @test solss.value ≈ solts.value ≈ soltt.value ≈ solst.value

    end
end
=#
@testset "fourier threaded executor" begin
    let
        f = (x, s, p) -> sin(x[1]*p) + s
        proto = 0.0
        s = FourierSeries([1, 0, 1]/2; period=1.0, offset=-2)
        ifs = FourierIntegralFunction(f, s, proto, SerialExecutor())
        dom = (0.0, 1.0)
        p = 2.0
        probs = IntegralProblem(ifs, dom, p)
        alg = QuadGKJL()
        sols = solve(probs, alg)
        ift = FourierIntegralFunction(f, s, proto, ThreadedExecutor(ntasks, max_batch))
        probt = IntegralProblem(ift, dom, p)
        solt = solve(probt, alg)
        @test sols.value ≈ solt.value

        alg = QuadratureFunction()
        sols = solve(probs, alg)
        solt = solve(probt, alg)
        @test sols.value ≈ solt.value

        @test solve(IntegralProblem(ift, AutoBZCore.PuncturedInterval(dom), p), NestedQuad(QuadGKJL())).value ≈ solve(IntegralProblem(ifs, AutoBZCore.PuncturedInterval(dom), p), NestedQuad(QuadGKJL())).value

        @test_throws ["ArgumentError", "HCubatureJL does not support"] solve(probt, HCubatureJL())
    end
    let
        iprob = IntegralProblem((x, p) -> x*p, (0, 1), 3.0)
        proto = 0.0im
        _solve! = function (solver, y, s, p)
            solver.p = p + sum(y)
            return solve!(solver).value + s
        end
        s = FourierSeries([1, 0, 1]/2; period=1.0, offset=-2)
        ifs = CommonSolveFourierIntegralFunction(_solve!, iprob, QuadGKJL(), s, proto, DefaultSpecialize(), SerialExecutor())
        dom = (0.0, 1.0)
        p = 2.0
        probs = IntegralProblem(ifs, dom, p)
        alg = QuadGKJL()
        sols = solve(probs, alg)
        ift = CommonSolveFourierIntegralFunction(_solve!, iprob, QuadGKJL(), s, proto, DefaultSpecialize(), ThreadedExecutor(ntasks, max_batch))
        probt = IntegralProblem(ift, dom, p)
        solt = solve(probt, alg)
        @test sols.value ≈ solt.value

        alg = QuadratureFunction()
        sols = solve(probs, alg)
        solt = solve(probt, alg)
        @test sols.value ≈ solt.value

        @test solve(IntegralProblem(ift, AutoBZCore.PuncturedInterval(dom), p), NestedQuad(QuadGKJL())).value ≈ solve(IntegralProblem(ifs, AutoBZCore.PuncturedInterval(dom), p), NestedQuad(QuadGKJL())).value

        @test_throws ["ArgumentError", "HCubatureJL does not support"] solve(probt, HCubatureJL())

        dom = AutoBZCore.HyperCube((0,0), (1,1))

        s = FourierSeries([0 1 0; 1 0 1; 0 1 0]/2; period=1.0, offset=-2)
        ifs = CommonSolveFourierIntegralFunction(_solve!, iprob, QuadGKJL(), s, proto, DefaultSpecialize(), SerialExecutor())
        ift = CommonSolveFourierIntegralFunction(_solve!, iprob, QuadGKJL(), s, proto, DefaultSpecialize(), ThreadedExecutor(ntasks, max_batch))
        solss = solve(IntegralProblem(ifs, dom, p), NestedQuad(QuadGKJL(), QuadGKJL()))
        solst = solve(IntegralProblem(ifs, dom, p), NestedQuad((QuadGKJL(), QuadGKJL()), (NoSpecialize(),), (ThreadedExecutor(ntasks, max_batch),)))
        soltt = solve(IntegralProblem(ift, dom, p), NestedQuad((QuadGKJL(), QuadGKJL()), (NoSpecialize(),), (ThreadedExecutor(ntasks, max_batch),)))
        solts = solve(IntegralProblem(ift, dom, p), NestedQuad(QuadGKJL()))
        @test solss.value ≈ solts.value ≈ soltt.value ≈ solst.value
    end
end
@testset "SymmetricRule" begin
    dims = 3
    A = I(dims)
    vol = (2π)^dims
    bz = load_bz(InversionSymIBZ(), A)
    f = IntegralFunction((x, p) -> 1.0, 1.0, ThreadedExecutor(ntasks, max_batch))
    ip = AutoBZProblem(TrivialRep(), f, bz)  # unit measure
    refsol = solve(ip, IAI(AuxQuadGKJL(), AuxQuadGKJL(), AuxQuadGKJL()))
    sol = solve(ip, AutoPTR())
    @test sol.value ≈ refsol.value ≈ vol
end