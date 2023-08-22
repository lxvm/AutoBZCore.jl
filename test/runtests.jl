using Test
using LinearAlgebra

using StaticArrays
using OffsetArrays

using HDF5
using SymmetryReduceBZ

using AutoBZCore
using AutoBZCore: IntegralProblem, solve, MixedParameters
using AutoBZCore: PuncturedInterval, HyperCube, segments, endpoints


function integer_lattice(n)
    C = OffsetArray(zeros(ntuple(_ -> 3, n)), repeat([-1:1], n)...)
    for i in 1:n, j in (-1, 1)
        C[CartesianIndex(ntuple(k -> k == i ? j : 0, n))] = 1/2n
    end
    C
end

# tests for SciML interface

@testset "AutoBZCore - SciML" begin
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
        # QuadratureFunction QuadGKJL AuxQuadGKJL ContQuadGKJL MeroQuadGKJL
        a = 0.0
        b = 2pi
        abstol=1e-5
        p = 3.0
        for (f, ref) in (
            ((x,p) -> p*sin(x), 0.0),
            ((x,p) -> p*one(x), p*(b-a)),
            ((x,p) -> inv(p-cos(x)), (b-a)/sqrt(p^2-1)),
        )
            prob = IntegralProblem(f, a, b, p)
            for alg in (QuadratureFunction(), QuadGKJL(), AuxQuadGKJL(), ContQuadGKJL(), MeroQuadGKJL())
                @test ref ≈ solve(prob, alg, abstol=abstol).u atol=abstol
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
            prob = IntegralProblem(f, fill(a, dim), fill(b, dim), p)
            for alg in (HCubatureJL(),)
                @test ref ≈ solve(prob, alg, abstol=abstol).u atol=abstol
            end
            prob = IntegralProblem(f, Basis(b*I(dim)), p)
            for alg in (MonkhorstPack(), AutoSymPTRJL(),)
                @test ref ≈ solve(prob, alg, abstol=abstol).u atol=abstol
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
            integrand = InplaceIntegrand(f, [0.0])
            inplaceprob = IntegralProblem(integrand, a, b, p)
            for alg in (QuadratureFunction(), QuadGKJL(), AuxQuadGKJL(), HCubatureJL(),)
                @test ref ≈ solve(inplaceprob, alg, abstol=abstol).u atol=abstol
            end
            inplaceprob = IntegralProblem(integrand, Basis([b;;]), p)
            for alg in (MonkhorstPack(), AutoSymPTRJL())
                @test ref ≈ solve(inplaceprob, alg, abstol=abstol).u atol=abstol
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
            integrand = BatchIntegrand(f, Float64)
            batchprob = IntegralProblem(integrand, a, b, p)
            for alg in (QuadratureFunction(), AuxQuadGKJL())
                @test ref ≈ solve(batchprob, alg, abstol=abstol).u atol=abstol
            end
            batchprob = IntegralProblem(integrand, Basis([b;;]), p)
            for alg in (MonkhorstPack(), AutoSymPTRJL())
                @test ref ≈ solve(batchprob, alg, abstol=abstol).u atol=abstol
            end
        end
    end

    @testset "multi-algorithms" begin
        # NestedQuad
        f(x, p) = 1.0 + p*sum(cos, x)
        p = 7.0
        abstol=1e-3
        for dim in 1:3, alg in (QuadratureFunction(), AuxQuadGKJL())
            ref = (2pi)^dim
            dom = CubicLimits(zeros(dim), 2pi*ones(dim))
            prob = IntegralProblem(f, dom, p)
            ndalg = NestedQuad(alg)
            @test ref ≈ solve(prob, ndalg, abstol=abstol).u atol=abstol
            inplaceprob = IntegralProblem(InplaceIntegrand((y,x,p) -> y .= f(x,p), [0.0]), dom, p)
            @test [ref] ≈ solve(inplaceprob, ndalg, abstol=abstol).u atol=abstol
            batchprob = IntegralProblem(BatchIntegrand((y,x,p) -> y .= f.(x,Ref(p)), Float64), dom, p)
            @test ref ≈ solve(batchprob, ndalg, abstol=abstol).u atol=abstol
            nestedbatchprob = IntegralProblem(NestedBatchIntegrand(ntuple(n -> f, 3), Float64), dom, p)
            @test ref ≈ solve(nestedbatchprob, ndalg, abstol=abstol).u atol=abstol
        end

        # AbsoluteEstimate
        est_alg = QuadratureFunction()
        abs_alg = QuadGKJL()
        alg = AbsoluteEstimate(est_alg, abs_alg)
        ref_alg = MeroQuadGKJL()
        f2(x, p) = inv(complex(p...) - cos(x))
        prob = IntegralProblem(f2, 0.0, 2pi, (0.5, 1e-3))
        abstol = 1e-5; reltol=1e-5
        @test solve(prob, alg, reltol=reltol).u ≈ solve(prob, ref_alg, abstol=abstol).u atol=abstol

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
    end
end

# tests for AutoBZ features
@testset "AutoBZCore - AutoBZ" begin

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

end

@testset "FourierExt" begin
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
end

@testset "SymmetryReduceBZExt" begin
    include("test_ibz.jl")
end

#=
using Unitful
using UnitfulAtomic
@testset "AtomsBaseExt" begin
    using AtomsBase
    # do the example of getting the volume of the bz of silicon
    bounding_box = 10.26 / 2 * [[0, 0, 1], [1, 0, 1], [1, 1, 0]]u"bohr"
    silicon = periodic_system([:Si =>  ones(3)/8,
                            :Si => -ones(3)/8],
                            bounding_box, fractional=true)
    A = reinterpret(reshape,eltype(eltype(bounding_box)),AtomsBase.bounding_box(silicon))
    using AutoBZCore
    recip_vol = det(AutoBZCore.canonical_reciprocal_basis(A))
    fbz = load_bz(FBZ(), silicon)
    fprob = AutoBZCore.IntegralProblem((x,p) -> 1.0, fbz)
    using SymmetryReduceBZ
    ibz = load_bz(IBZ(), silicon)
    iprob = AutoBZCore.IntegralProblem((x,p) -> 1.0, ibz)
    for alg in (IAI(), PTR(), TAI())
        @test recip_vol ≈ AutoBZCore.solve(fprob, alg).u
        @test recip_vol ≈ AutoBZCore.solve(iprob, alg).u
    end
end

@testset "WannierIOExt" begin
    using WannierIO
    # use artefacts to provide an input wout file
    using AutoBZCore
    fbz = load_bz(FBZ(), "svo.wout")
    using SymmetryReduceBZ
    ibz = load_bz(IBZ(), "svo.wout")
end
=#
