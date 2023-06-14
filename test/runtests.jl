using Test
using LinearAlgebra

using StaticArrays
using OffsetArrays

using HDF5
using AutoBZCore


function integer_lattice(n)
    C = OffsetArray(zeros(ntuple(_ -> 3, n)), repeat([-1:1], n)...)
    for i in 1:n, j in (-1, 1)
        C[CartesianIndex(ntuple(k -> k == i ? j : 0, n))] = 1/2n
    end
    C
end
function make_inversion_ibz(A, B, dims)
    lims = CubicLimits(zeros(dims),ones(dims)/2)
    syms = [Diagonal(SVector{dims}(v)) for v in Iterators.product([(1,-1) for i in 1:dims]...)]
    return SymmetricBZ(A,B,lims,syms)
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
        vol = (2π)^dims
        for bz in (FullBZ(A, B), make_inversion_ibz(A,B,dims))
            ip = IntegralProblem((x,p) -> 1.0, bz)  # unit measure
            T = AutoBZCore.IntegralSolution{Float64}
            for alg in (IAI(), TAI(), PTR(), AutoPTR(), PTR_IAI(), AutoPTR_IAI())
                @test @inferred(T, solve(ip, alg)).u ≈ vol
            end
            @test @inferred(T, solve(IntegralProblem((x, p) -> exp(-x^2), -Inf, Inf), QuadGKJL())).u ≈ sqrt(pi)
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
            bz = FullBZ(A, B)
            prob = IntegralProblem(f, bz, p)
            solver = IntegralSolver(IntegralProblem(f, bz), IAI())
            @test solver(p) == solve(prob, IAI()).u # use the plain interface
            # g = (x,p) -> sum(x)*p[1]+p.a
            # solver2 = IntegralSolver(Integrand(g), bz, IAI()) # use the MixedParameters interface
            # prob2 = IntegralProblem(g, bz, MixedParameters(12.0, a=1.0))
            # @test solver2(12.0, a=1.0) == solve(prob2, IAI()).u
        end
        @testset "Integrands" begin
            # AutoBZ interface user function: f(x, args...; kwargs...) where args & kwargs
            # stored in MixedParameters
            f(x, a; b) = a*x+b
            # SciML interface for Integrand: f(x, p) (# and parameters can be preloaded and
            # p is merged with MixedParameters)
            @test f(6.7, 1.3, b=4.2) == Integrand(f, 1.3, b=4.2)(6.7) == Integrand(f)(6.7, MixedParameters(1.3, b=4.2))
            # An Integrand merges its parameters with the problem's
            prob = IntegralProblem(Integrand(f, 1.3, b=4.2), 0, 1)
            u = IntegralSolver(prob, QuadGKJL())()
            v = IntegralSolver(Integrand(f), 0, 1, QuadGKJL())(1.3, b=4.2)
            w = IntegralSolver(Integrand(f, b=4.2), 0, 1, QuadGKJL())(1.3)
            @test u == v == w
        end
        @testset "batchsolve" begin
            # SciML interface: iterable of parameters
            prob = IntegralProblem((x, p) -> p, 0, 1)
            solver = IntegralSolver(prob, QuadGKJL())
            params = range(1, 2, length=3)
            @test [solver(p) for p in params] == batchsolve(solver, params)
            # AutoBZ interface: array of MixedParameters
            f(x, a; b) = a*x+b
            solver = IntegralSolver(Integrand(f), 0, 1, QuadGKJL())
            as = rand(3); bs = rand(3)
            @test [solver(a, b=b) for (a,b) in Iterators.zip(as, bs)] == batchsolve(solver, paramzip(as, b=bs))
            @test [solver(a, b=b) for (a,b) in Iterators.product(as, bs)] == batchsolve(solver, paramproduct(as, b=bs))
        end
    end

end


@testset "FourierExt" begin
    @testset "FourierIntegrand" begin
        for dims in 1:3
            s = InplaceFourierSeries(integer_lattice(dims), period=1)
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
        end
    end
    @testset "algorithms" begin
        f(x::FourierValue, a; b) = a*x.s+b
        for dims in 1:3
            vol = (2pi)^dims
            A = I(dims)
            B = AutoBZCore.canonical_reciprocal_basis(A)
            s = InplaceFourierSeries(integer_lattice(dims), period=1)
            for bz in (FullBZ(A, B), make_inversion_ibz(A,B,dims))
                integrand = FourierIntegrand(f, s, 1.3, b=1.0)
                prob = IntegralProblem(integrand, bz)
                for alg in (IAI(), PTR(), AutoPTR(), TAI())
                    solver = IntegralSolver(prob, alg, reltol=0, abstol=1e-6)
                    @test solver() ≈ vol*one(eltype(s)) atol=1e-6
                end
            end
        end
    end

end

@testset "HDF5Ext" begin
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
    end
    rm(fn)
end
