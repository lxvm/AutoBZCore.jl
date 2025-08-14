using Test
using Unitful
using AutoBZCore
using AutoBZCore: canonical_reciprocal_basis, canonical_ptr_basis
using LinearAlgebra: I, Diagonal
using StaticArrays
using FourierSeriesEvaluators: FourierSeries

for A in [rand(3, 3) * u"m", rand(SMatrix{3,3,Float64,9})*u"m"]
    B = canonical_reciprocal_basis(A)
    @test B'A ≈ 2pi*I
    pB = canonical_ptr_basis(B)
    @test pB isa AutoBZCore.Basis
    @test pB.B ≈ I
end


quad = QuadGKJL(order=1)
let dims=1
    for f in [IntegralFunction((x, p) -> 1.0u"s", 1.0u"s"), FourierIntegralFunction((x, s, p) -> 1.0u"s", FourierSeries(integer_lattice(dims); period=2pi*u"m"), 1.0u"s")]
        abstol=1e-5u"s * m^1"
        dom = (0.0u"m", 2.0u"m")
        prob = IntegralProblem(f, dom)
        alg = quad
        sol = solve(prob, alg; abstol)
        @test sol.value ≈ 2.0u"m" * 1.0u"s"
    end
end
let dims=2
    for f in [IntegralFunction((x, p) -> 1.0u"s", 1.0u"s"), FourierIntegralFunction((x, s, p) -> 1.0u"s", FourierSeries(integer_lattice(dims); period=2pi*u"m"), 1.0u"s")]
        abstol=1e-5u"s * m^2"
        dom = AutoBZCore.get_basis(Diagonal([2, 4])*u"m")
        prob = IntegralProblem(f, dom)
        alg = MonkhorstPack(; npt=10)
        sol = solve(prob, alg; abstol)
        @test sol.value ≈ 2.0u"m" * 4.0u"m" * 1.0u"s"
    end
end

let
    maxdims = 3
    bounds = 2.0u"m" .* (1:maxdims)
    for dims in 1:maxdims
        for f in [IntegralFunction((x, p) -> 1.0u"s", 1.0u"s"), FourierIntegralFunction((x, s, p) -> 1.0u"s", FourierSeries(integer_lattice(dims); period=2pi*u"m"), 1.0u"s")]
            lim = bounds[1:dims]
            abstol=1e-5u"s" * oneunit(prod(lim))
            dom = AutoBZCore.HyperCube(fill(0.0u"m", dims), lim)
            prob = IntegralProblem(f, dom)
            let tolalg = AutoBZCore.StaticTolAlg(AutoBZCore.project)
                alg = NestedQuad(quad; tolalg)
                sol = solve(prob, alg; abstol)
                @test sol.value ≈ 1.0u"s" * prod(lim)
            end
            let tolalg = AutoBZCore.StaticTolAlg(AutoBZCore.bbox)
                alg = NestedQuad(quad; tolalg)
                sol = solve(prob, alg; abstol)
                @test sol.value ≈ 1.0u"s" * prod(lim)
            end
            let tolalg = AutoBZCore.AdaptiveTolAlg()
                alg = NestedQuad(quad; tolalg)
                sol = solve(prob, alg; abstol)
                @test sol.value ≈ 1.0u"s" * prod(lim)
            end
            let tolalg = AutoBZCore.v03TolAlg()
                alg = NestedQuad(quad; tolalg)
                sol = solve(prob, alg; abstol)
                @test sol.value ≈ 1.0u"s" * prod(lim)
            end
            let tolalg = AutoBZCore.v04TolAlg()
                alg = NestedQuad(quad; tolalg)
                if dims == 1
                    sol = solve(prob, alg; abstol)
                    @test sol.value ≈ 1.0u"s" * prod(lim)
                else
                    @test_throws "DimensionError" solve(prob, alg; abstol)
                end
            end
        end
    end
end