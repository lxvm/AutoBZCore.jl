import Pkg
Pkg.activate(".")           # reproducible environment included
Pkg.instantiate()           # install dependencies
include("autobz_extras.jl") # extra functions from AutoBZ.jl

h, bz = load_wannier90_data("svo")

bz = cubic_sym_ibz(bz)      # construct ibz for cubic system

using AutoBZCore, HChebInterp, Plots, LinearAlgebra

η = 1e-2                    # 10 meV (scattering amplitude)
dos_integrand(h_k, η, ω) = -imag(tr(inv(complex(ω, η)*I - h_k)))/pi
integrand = FourierIntegrand(dos_integrand, h, η)

alg = IAI(; segbufs=AutoBZCore.alloc_segbufs(Float64,Float64,Float64,3))
dos_solver = IntegralSolver(integrand, bz, alg; abstol=1e-3)
dos = hchebinterp(dos_solver, 10, 15; atol=1e-2)

plot(; xguide="ω (eV)", yguide="SVO DOS (eV⁻¹ Å⁻³)", framestyle=:box)
plot!(10:η/100:15, dos; label="IAI(), η=$η", legend=:topleft)
savefig("iai_svo_dos.pdf")

npt = 100
alg = PTR(; npt=npt, rule=AutoBZCore.alloc_rule(h, eltype(bz), bz.syms, npt))
dos_solver_ptr = IntegralSolver(integrand, bz, alg; abstol=1e-3)
dos_ptr = hchebinterp(dos_solver_ptr, 10, 15; atol=1e-2)

plot(; xguide="ω (eV)", yguide="SVO DOS (eV⁻¹ Å⁻³)", framestyle=:box)
plot!(10:η/100:15, dos_ptr; label="PTR(; npt=$npt), η=$η", legend=:topleft)
savefig("ptr_svo_dos.pdf")