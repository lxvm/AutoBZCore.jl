using Test
using LinearAlgebra
using AutoBZCore
using AutoBZCore: PuncturedInterval, HyperCube, segments, endpoints
using AuxQuadGK: AuxValue

@testset "AuxValue" begin
    dims = 3
    A = I(dims)
    vol = (2π)^dims
    bz = load_bz(CubicSymIBZ(), A)
    f = IntegralFunction((x,p) -> AuxValue(1.0, 1.0))
    ip = AutoBZProblem(TrivialRep(), f, bz)  # unit measure
    alg = IAI(ntuple(_->AuxQuadGKJL(), dims)...)
    solver = init(ip, alg)
    sol = solve!(solver)
    @test sol.value.val ≈ sol.value.aux ≈ vol
end
