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
    Rmin[1]:Rmax[1], Rmin[2]:Rmax[2], Rmin[3]:Rmax[3]
)
for (i, h) in zip(hrdat.Rvectors, hrdat.H)
    H_R[CartesianIndex(Tuple(i))] = h
end

using AutoBZCore, WannierIO, HChebInterp, LinearAlgebra

bz = load_bz(CubicSymIBZ(), "svo.wout")
# bz = load_bz(IBZ(), "svo.wout") # works with SymmetryReduceBZ.jl installed
h = FourierSeries(H_R, period=1.0)

η = 1e-2                    # 10 meV (scattering amplitude)
dos_integrand(h_k::FourierValue, η, ω) = -imag(tr(inv((ω+im*η)*I - h_k.s)))/pi
integrand = FourierIntegrand(dos_integrand, h, η)

alg = IAI()
dos_solver_iai = IntegralSolver(integrand, bz, alg; abstol=1e-3)
dos_iai = hchebinterp(dos_solver_iai, 10, 15; atol=1e-2)

alg = PTR(npt=100)
dos_solver_ptr = IntegralSolver(integrand, bz, alg; abstol=1e-3)
dos_ptr = hchebinterp(dos_solver_ptr, 10, 15; atol=1e-2)

using CairoMakie

set_theme!(fontsize=24, linewidth=4)

fig1 = Figure()
ax1 = Axis(fig1[1,1], limits=((10,15), (0,det(bz.B)*6)), xlabel="ω (eV)", ylabel="SVO DOS (eV⁻¹ Å⁻³)")
p1 = lines!(ax1, 10:η/100:15, dos_iai; label="IAI(), η=$η")
axislegend(ax1)
save("iai_svo_dos.pdf", fig1)

fig2 = Figure()
ax2 = Axis(fig2[1,1], limits=((10,15), (0,det(bz.B)*6)), xlabel="ω (eV)", ylabel="SVO DOS (eV⁻¹ Å⁻³)")
p2 = lines!(ax2, 10:η/100:15, dos_ptr; label="PTR(), η=$η")
axislegend(ax2)
save("ptr_svo_dos.pdf", fig2)
