# import Pkg
# Pkg.activate(".")           # reproducible environment included
# Pkg.instantiate()           # install dependencies

using WannierIO

hrdat = read_w90_hrdat("svo_hr.dat")

Rmin, Rmax = extrema(hrdat.Rvectors)
Rsize = Tuple(Rmax .- Rmin .+ 1)
n, m = size(first(hrdat.H))

using StaticArrays, OffsetArrays

H_R = OffsetArray(
    Array{SMatrix{n,m,eltype(eltype(hrdat.H)),n*m}}(undef, Rsize...),
    map(:, Rmin, Rmax)...
)
for (i, h, n) in zip(hrdat.Rvectors, hrdat.H, hrdat.Rdegens)
    H_R[CartesianIndex(Tuple(i))] = h / n
end

using FourierSeriesEvaluators, LinearAlgebra

h = FourierSeries(H_R, period=1.0)

η = 1e-2                    # 10 meV (scattering amplitude)
ω_min = 10
ω_max = 15
p0 = (; η, ω=(ω_min + ω_max)/2) # initial parameters
# BUG cannot redefine this function without breaking functionwrappers
# https://github.com/JuliaLang/julia/issues/52635#issuecomment-2150808569
greens_function(k, h_k, (; η, ω)) = tr(inv((ω+im*η)*I - h_k))
prototype = let k = FourierSeriesEvaluators.period(h)
    greens_function(k, h(k), p0)
end

using AutoBZCore
bz = load_bz(CubicSymIBZ(), "svo.wout")
# bz = load_bz(IBZ(), "svo.wout") # works with SymmetryReduceBZ.jl installed

integrand = FourierIntegralFunction(greens_function, h, prototype)
prob_dos = AutoBZProblem(integrand, bz, p0; abstol=1e-3)

using HChebInterp

cheb_order = 15
#=
batch_iai = let prob = prob_dos, alg = IAI(QuadGKJL()), nthreads=min(cheb_order+1, Threads.nthreads())
    cache = init(prob, alg)
    ω -> begin
        cache.p = (; cache.p..., ω)
        solve!(cache).value
    end
    #=
    caches = [init(prob, alg) for _ in 1:nthreads]
    BatchFunction() do ωs
        out = Vector{typeof(prototype)}(undef, length(ωs))
        Threads.@threads for i in 1:nthreads
            cache = caches[i]
            for j in i:nthreads:length(ωs)
                ω = ωs[j]
                cache.p = (; cache.p..., ω)
                out[j] = solve!(cache).value
            end
        end
        return out
    end
    =#
end
=#
batch_iai = let cache = init(prob_dos, IAI((QuadGKJL(), QuadGKJL(), QuadGKJL()), (AutoBZCore.FullSpecialize(), AutoBZCore.FunctionWrapperSpecialize(), AutoBZCore.FullSpecialize())))
    ω -> begin
        cache.p = (; cache.p..., ω)
        solve!(cache).value
    end
end
@time greens_iai = hchebinterp(batch_iai, ω_min, ω_max; atol=1e-2)
#=
batch_ptr = let prob = prob_dos, nthreads=cheb_order+1
    ω -> batchsolve(prob, PTR(; nthreads), ω; nthreads=1)
end
greens_ptr = hchebinterp(dos_solver_ptr, 10, 15; atol=1e-2)
=#
using CairoMakie

set_theme!(fontsize=24, linewidth=4)

fig1 = Figure()
ax1 = Axis(fig1[1,1], limits=((10,15), (0,6)), xlabel="ω (eV)", ylabel="SVO DOS (eV⁻¹)")
p1 = lines!(ax1, 10:η/100:15, ω -> -imag(greens_iai(ω))/pi/det(bz.B); label="IAI, η=$η")
axislegend(ax1)
save("iai_svo_dos.pdf", fig1)
#=
fig2 = Figure()
ax2 = Axis(fig2[1,1], limits=((10,15), (0,det(bz.B)*6)), xlabel="ω (eV)", ylabel="SVO DOS (eV⁻¹ Å⁻³)")
p2 = lines!(ax2, 10:η/100:15, ω -> -imag(greens_ptr(ω))/pi; label="PTR, η=$η")
axislegend(ax2)
save("ptr_svo_dos.pdf", fig2)
=#
