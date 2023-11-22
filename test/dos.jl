using Test, AutoBZCore, LinearAlgebra

# cos(x) band with dos 1/sqrt(1-ω^2)/pi
for h in (x -> sum(y -> cospi(2y), x), FourierSeries([0.5, 0.0, 0.5], period=1.0, offset=-2))
    E = 0.3
    bz = load_bz(FBZ(), [2pi;;])

    prob = DOSProblem(h, E, bz)

    atol = 1e-6

    alg = RationalRichardson(; abstol=atol/10)

    sol = AutoBZCore.solve(prob, alg; abstol=atol)
    @test sol.u/det(bz.B) ≈ 1/sqrt(1-E^2)/pi
end