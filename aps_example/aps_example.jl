import Pkg
Pkg.activate(".")           # reproducible environment included
Pkg.instantiate()           # install dependencies

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

h = HermitianFourierSeries(FourierSeries(H_R, period=1.0))

η = 1e-2                    # 10 meV (scattering amplitude)
ω_min = 10
ω_max = 15
p0 = (; η, ω=(ω_min + ω_max)/2) # initial parameters
greens_function(k, h_k, (; η, ω)) = tr(inv((ω+im*η)*I - h_k))
prototype = let k = FourierSeriesEvaluators.period(h)
    greens_function(k, h(k), p0)
end

using AutoBZCore
bz = load_bz(CubicSymIBZ(), "svo.wout")
# bz = load_bz(IBZ(), "svo.wout") # works with SymmetryReduceBZ.jl installed

integrand = FourierIntegralFunction(greens_function, h, prototype)
prob_dos = AutoBZProblem(TrivialRep(), integrand, bz, p0; abstol=1e-3)

using HChebInterp

cheb_order = 15

function dos_solver(prob, alg)
    solver = init(prob, alg)
    ω -> begin
        solver.p = (; solver.p..., ω)
        solve!(solver).value
    end
end
function threaded_dos_solver(prob, alg; nthreads=min(cheb_order, Threads.nthreads()))
    solvers = [init(prob, alg) for _ in 1:nthreads]
    BatchFunction() do ωs
        out = Vector{typeof(prototype)}(undef, length(ωs))
        Threads.@threads for i in 1:nthreads
            solver = solvers[i]
            for j in i:nthreads:length(ωs)
                ω = ωs[j]
                solver.p = (; solver.p..., ω)
                out[j] = solve!(solver).value
            end
        end
        return out
    end
end

dos_solver_iai = dos_solver(prob_dos, IAI(QuadGKJL()))
@time greens_iai = hchebinterp(dos_solver_iai, ω_min, ω_max; atol=1e-2, order=cheb_order)

dos_solver_ptr = dos_solver(prob_dos, PTR(; npt=100))
@time greens_ptr = hchebinterp(dos_solver_ptr, ω_min, ω_max; atol=1e-2, order=cheb_order)

using CairoMakie

set_theme!(fontsize=24, linewidth=4)

fig1 = Figure()
ax1 = Axis(fig1[1,1], limits=((10,15), (0,6)), xlabel="ω (eV)", ylabel="SVO DOS (eV⁻¹)")
p1 = lines!(ax1, 10:η/100:15, ω -> -imag(greens_iai(ω))/pi/det(bz.B); label="IAI, η=$η")
axislegend(ax1)
save("iai_svo_dos.pdf", fig1)

fig2 = Figure()
ax2 = Axis(fig2[1,1], limits=((10,15), (0,6)), xlabel="ω (eV)", ylabel="SVO DOS (eV⁻¹)")
p2 = lines!(ax2, 10:η/100:15, ω -> -imag(greens_ptr(ω))/pi/det(bz.B); label="PTR, η=$η")
axislegend(ax2)
save("ptr_svo_dos.pdf", fig2)
