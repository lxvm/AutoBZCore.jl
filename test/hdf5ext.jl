using Test
using LinearAlgebra
using StaticArrays
using HDF5
using AutoBZCore

fn = tempname()
h5open(fn, "w") do io
    g1 = create_group(io, "Number")
    @testset "Number" begin
        prob   = IntegralProblem((x, p) -> p, 0, 1)
        solver = IntegralSolver(prob, QuadGKJL())
        params = 1:1.0:10
        values = batchsolve(g1, solver, params, verb=false)
        @test g1["I"][:] ≈ values
    end
    g2 = create_group(io, "SArray")
    @testset "SArray" begin
        f(x,p) = ((s,c) = sincos(p*x) ; SHermitianCompact{2,Float64,3}([s, c, -s]))
        prob   = IntegralProblem(f, 0.0, 1pi)
        solver = IntegralSolver(prob, QuadGKJL())
        params = [0.8, 0.9, 1.0]
        values = batchsolve(g2, solver, params, verb=false)
        # Arrays of SArray are flattened to multidimensional arrays
        @test g2["I"][:,:,:] ≈ reshape(collect(Iterators.flatten(values)), 2, 2, :)
    end
    g3 = create_group(io, "AuxValue")
    @testset "AuxValue" begin
        f(x,p) = ((re,im) = reim(inv(complex(cos(x), p))) ; AuxValue(re, im))
        prob   = IntegralProblem(f, 0.0, 2pi)
        solver = IntegralSolver(prob, QuadGKJL(), abstol=1e-3)
        params = [2.0, 1.0, 0.5]
        values = batchsolve(g3, solver, params, verb=false)
        @test g3["I/val"][:] ≈ getproperty.(values, :val)
        @test g3["I/aux"][:] ≈ getproperty.(values, :aux)
    end
    g4 = create_group(io, "0d")
    g5 = create_group(io, "3d")
    @testset "parameter dimensions" begin
        prob   = IntegralProblem((x, p) -> p[1] + p[2] + p[3], 0, 1)
        solver = IntegralSolver(prob, QuadGKJL())
        # 0-d parameters
        params = paramzip(0, 1, 2)
        values = batchsolve(g4, solver, params, verb=false)
        @test g4["I"][] ≈ values[] ≈ 3
        @test g4["args/1"][] ≈ 0
        @test g4["args/2"][] ≈ 1
        @test g4["args/3"][] ≈ 2
        # 3-d parameters
        params = paramproduct(1:2, 1:2, 1:2)
        values = batchsolve(g5, solver, params, verb=false)
        @test g5["I"][1,1,1] ≈ values[1] ≈ 3
        @test g5["I"][2,2,2] ≈ values[8] ≈ 6
    end
end
rm(fn)
