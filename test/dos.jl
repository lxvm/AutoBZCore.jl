using Test, AutoBZCore, LinearAlgebra, StaticArrays, OffsetArrays, Elliptic
using GeneralizedGaussianQuadrature: generalizedquadrature
using FourierSeriesEvaluators, QuadGK

# test set of known DOS examples

# TODO : check that the exact formulas correctly integrate to unity

tb_graphene = let t=1.0
    ax = CartesianIndices((-2:2, -2:2))
    hm = OffsetMatrix([zero(MMatrix{2,2,typeof(t),4}) for i in ax], ax)
    hm[1,1][1,2]   = hm[1,-2][1,2] = hm[-2,1][1,2] = t
    hm[-1,-1][2,1] = hm[-1,2][2,1] = hm[2,-1][2,1] = t
    FourierSeries(SMatrix{2,2,typeof(t),4}.(hm), period=1.0)
end

# https://arxiv.org/abs/1311.2514v1
function dos_graphene_exact(E::Real, t=oneunit(E))
    E = abs(E)
    x = abs(E/t)
    if x <= 1
        f = (1+x)^2 - (x^2-1)^2/4
        2E/((pi*t)^2*sqrt(f))*Elliptic.K(4x/f)
    elseif 1 < x < 3
        f = (1+x)^2 - (x^2-1)^2/4
        2E/((pi*t)^2*sqrt(4x))*Elliptic.K(f/4x)
    else
        zero(inv(oneunit(t)))
    end
end

# The following three examples are taken from sec 5.3 of
# https://link.springer.com/book/10.1007/3-540-28841-4

function tb_integer(n, t=1.0)
    ax = CartesianIndices(ntuple(_ -> -1:1, n))
    C = OffsetArray([zero(MMatrix{1,1,typeof(t),1}) for i in ax], ax)
    for i in 1:n, j in (-1, 1)
        C[CartesianIndex(ntuple(k -> k ≈ i ? j : 0, n))][1,1] = t
    end
    return FourierSeries(SMatrix{1,1,typeof(t),1}.(C), period=1.0)
end

tb_integer_1d = tb_integer(1)

function dos_integer_1d_exact(E::Real, t=oneunit(E))
    x = abs(E/2t)
    if x <= 1
        1/sqrt(1 - x^2)/(pi*2t)
    else
        zero(inv(oneunit(t)))
    end
end

tb_integer_2d = tb_integer(2)

function dos_integer_2d_exact(E::Real, t=oneunit(E))
    x = abs(E/4t)
    if x <= 1
        # note Elliptic and SpecialFunctions accept the elliptic modulus m = k^2
        1/(pi^2*2t)*Elliptic.K(1 - x^2)
    else
        zero(inv(oneunit(t)))
    end
end

tb_integer_3d = tb_integer(3)

# https://doi.org/10.1143/JPSJ.30.957 (also includes FCC and BCC lattices)
function dos_integer_3d_exact(E::Real, t=oneunit(E))
    x = abs(E/6t)
    # note Elliptic and SpecialFunctions accept the elliptic modulus m = k^2
    f = u -> Elliptic.K(1 - ((3x-cos(u))/2)^2)
    if 3x < 1
        n, w = generalizedquadrature(30) # quadrature designed for logarithmic singularity
        u′ = acos(3x)   # breakpoint for logarithmic singularity in the interval (0, pi)
        I1 = sum(w .* f.(u′ .+ n .* -u′)) * u′ # (0, u′)
        I2 = sum(w .* f.(u′ .+ n .* (pi - u′))) * (pi - u′) # (u′, pi)
        oftype(zero(inv(oneunit(t))), 1/(pi^3*2t) * (I1+I2))
        # since we may loose precision when the breakpoint is near the boundary, consider
        # oftype(zero(inv(oneunit(t))), 1/(pi^3*2t) * ((isfinite(I1) ? I1 : zero(I1)) + (isfinite(I2) ? I2 : zero(I2))))
    elseif x < 1
        1/(pi^3*2t)*quadgk(f, 0, acos(3x-2))[1]
    else
        zero(inv(oneunit(t)))
    end
end

for (model, solution, bandwidth, bzkind) in (
    (tb_graphene,   dos_graphene_exact,   4, FBZ()),
    (tb_integer_1d, dos_integer_1d_exact, 2, FBZ()),
    (tb_integer_2d, dos_integer_2d_exact, 4, FBZ()),
    (tb_integer_3d, dos_integer_3d_exact, 6, FBZ()),
    (tb_integer_1d, dos_integer_1d_exact, 2, InversionSymIBZ()),
    (tb_integer_2d, dos_integer_2d_exact, 4, InversionSymIBZ()),
    (tb_integer_3d, dos_integer_3d_exact, 6, InversionSymIBZ()),
    (tb_integer_1d, dos_integer_1d_exact, 2, CubicSymIBZ()),
    (tb_integer_2d, dos_integer_2d_exact, 4, CubicSymIBZ()),
    (tb_integer_3d, dos_integer_3d_exact, 6, CubicSymIBZ()),
)
    B = bandwidth
    bz = load_bz(bzkind, I(ndims(model)))
    prob = DOSProblem(model, float(zero(B)), bz)
    E = Float64[-B - 1, -0.8B, -0.6B, -0.2B, 0.1B, 0.3B, 0.5B, 0.7B, 0.9B, B + 2]
    for alg in (GGR(; npt=200),)
        cache = AutoBZCore.init(prob, alg)
        for e in E
            cache.domain = e
            @test AutoBZCore.solve!(cache).value ≈ solution(e) atol=1e-2
        end
    end
end

# test caching behavior
let h = FourierSeries(SMatrix{1,1,Float64,1}.([0.5, 0.0, 0.5]), period=1.0, offset=-2), E = 0.1
    bz = load_bz(FBZ(), [2pi;;])
    prob = DOSProblem(h, E, bz)
    alg = GGR()

    cache = AutoBZCore.init(prob, alg)
    sol1 = AutoBZCore.solve!(cache)

    h.c .*= 2
    cache.isfresh = true
    cache.domain = 2E

    sol2 = AutoBZCore.solve!(cache)

    @test sol1.value ≈ 2sol2.value

    cache.H = FourierSeries(2*h.c, period=h.t, offset=h.o)
    cache.domain = 4E
    sol3 = AutoBZCore.solve!(cache)
    @test sol2.value ≈ 2sol3.value
end
