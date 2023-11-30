using Test, AutoBZCore, LinearAlgebra, Richardson, StaticArrays

# cos(x) band with dos 1/sqrt(1-ω^2)/pi
for h in (x -> sum(y -> cospi(2y), x), FourierSeries(SMatrix{1,1,Float64,1}.([0.5, 0.0, 0.5]), period=1.0, offset=-2))
    E = 0.3
    bz = load_bz(FBZ(), [2pi;;])

    prob = DOSProblem(h, E, bz)

    atol = 1e-6

    for alg in (RationalRichardson(; abstol=atol/10), GGR(npt=1000))
        alg isa GGR && !(h isa FourierSeries) && continue
        sol = AutoBZCore.solve(prob, alg; abstol=atol)
        @test only(sol.u)/det(bz.B) ≈ 1/sqrt(1-E^2)/pi atol=1e-3
    end
end

# test caching behavior
let h = FourierSeries(SMatrix{1,1,Float64,1}.([0.5, 0.0, 0.5]), period=1.0, offset=-2)
    bz = load_bz(FBZ(), [2pi;;])
    prob = DOSProblem(h, 0.0, bz)
    alg = GGR()

    cache = AutoBZCore.init(prob, alg)
    sol1 = AutoBZCore.solve!(cache)

    h.c .*= 2
    cache.isfresh = true

    sol2 = AutoBZCore.solve!(cache)

    @test sol1.u*2 ≈ sol2.u

    cache.H = FourierSeries(2*h.c, period=h.t, offset=h.o)
    sol3 = AutoBZCore.solve!(cache)
    @test sol2.u*2 ≈ sol3.u
end
