"""
    AbstractAutoBZAlgorithm

Abstract supertype for Brillouin zone integration algorithms.
"""
abstract type AbstractAutoBZAlgorithm <: SciMLBase.AbstractIntegralAlgorithm end

"""
    IAI(; order=7, norm=norm, initdivs=nothing, segbufs=nothing)

Iterated-adaptive integration using `nested_quadgk` from
[IteratedIntegration.jl](https://github.com/lxvm/IteratedIntegration.jl).
**This algorithm is the most efficient for localized integrands**.
See [`alloc_segbufs`](@ref) for how to pre-allocate segment buffers for
`nested_quadgk`.
"""
struct IAI{F,I,S} <: AbstractAutoBZAlgorithm
    order::Int
    norm::F
    initdivs::I
    segbufs::S
end
IAI(; order=7, norm=norm, initdivs=nothing, segbufs=nothing) = IAI(order, norm, initdivs, segbufs)


"""
    PTR(; npt=50, rule=nothing)

Periodic trapezoidal rule with a fixed number of k-points per dimension, `npt`,
using the routine `ptr` from [AutoSymPTR.jl](https://github.com/lxvm/AutoSymPTR.jl).
**The caller should check that the integral is converged w.r.t. `npt`**.
See [`alloc_rule`](@ref) for how to pre-evaluate a PTR rule for use across calls
with compatible integrands.
"""
struct PTR{R} <: AbstractAutoBZAlgorithm
    npt::Int
    rule::R
end
PTR(; npt=50, rule=nothing) = PTR(npt, rule)

"""
    AutoPTR(; norm=norm, buffer=nothing)

Periodic trapezoidal rule with automatic convergence to tolerances passed to the
solver with respect to `norm` using the routine `autosymptr` from
[AutoSymPTR.jl](https://github.com/lxvm/AutoSymPTR.jl).
**This algorithm is the most efficient for smooth integrands**.
See [`alloc_autobuffer`](@ref) for how to pre-evaluate a buffer for `autosymptr`
for use across calls with compatible integrands.
"""
struct AutoPTR{F,B} <: AbstractAutoBZAlgorithm
    norm::F
    buffer::B
end
AutoPTR(; norm=norm, buffer=nothing) = AutoPTR(norm, buffer)

"""
    PTR_IAI(; ptr=PTR(), iai=IAI())

Multi-algorithm that returns an `IAI` calculation with an `abstol` determined
from the given `reltol` and a `PTR` estimate, `I`, as `reltol*norm(I)`.
This addresses the issue that `IAI` does not currently use a globally-adaptive
algorithm and may not have the expected scaling with localization length unless
an `abstol` is used since computational effort may be wasted via a `reltol` with
the naive `nested_quadgk`.
"""
struct PTR_IAI{P,I} <: AbstractAutoBZAlgorithm
    ptr::P
    iai::I
end
PTR_IAI(; ptr=PTR(), iai=IAI()) = PTR_IAI(ptr, iai)

"""
    AutoPTR_IAI(; reltol=1.0, ptr=AutoPTR(), iai=IAI())


Multi-algorithm that returns an `IAI` calculation with an `abstol` determined
from an `AutoPTR` estimate, `I`, computed to `reltol` precision, and the `rtol`
given to the solver as `rtol*norm(I)`.
This addresses the issue that `IAI` does not currently use a globally-adaptive
algorithm and may not have the expected scaling with localization length unless
an `abstol` is used since computational effort may be wasted via a `reltol` with
the naive `nested_quadgk`.
"""
struct AutoPTR_IAI{P,I} <: AbstractAutoBZAlgorithm
    reltol::Float64
    ptr::P
    iai::I
end
AutoPTR_IAI(; reltol=1.0, ptr=AutoPTR(), iai=IAI()) = AutoPTR_IAI(reltol, ptr, iai)

"""
    TAI(; rule=HCubatureJL())

Tree-adaptive integration using `hcubature` from
[HCubature.jl](https://github.com/JuliaMath/HCubature.jl). This routine is
limited to integration over hypercube domains and may not use all symmetries.
"""
struct TAI{T<:HCubatureJL} <: AbstractAutoBZAlgorithm
    rule::T
end
TAI(; rule=HCubatureJL()) = TAI(rule)

# Imitate original interface
IntegralProblem(f, bz::SymmetricBZ, args...; kwargs...) =
    IntegralProblem{isinplace(f, 3)}(f, bz, bz, args...; kwargs...)

# layer to intercept integrand construction
function construct_integrand(f, iip, p)
    if iip
        (y, x) -> (f(y, x, p); y)
    else
        x -> f(x, p)
    end
end

function __solvebp_call(prob::IntegralProblem, alg::AbstractAutoBZAlgorithm,
                                sensealg, bz::SymmetricBZ, ::SymmetricBZ, p;
                                reltol = nothing, abstol = nothing, maxiters = typemax(Int))

    abstol_ = (abstol===nothing) ? zero(eltype(bz)) : abstol
    reltol_ = (abstol===nothing) ? (iszero(abstol_) ? sqrt(eps(typeof(abstol_))) : zero(abstol_)) : reltol
    f = construct_integrand(prob.f, isinplace(prob), prob.p)

    if alg isa IAI
        j = abs(det(bz.B))  # include jacobian determinant for map from fractional reciprocal lattice coordinates to Cartesian reciprocal lattice
        atol = abstol_/nsyms(bz)/j # reduce absolute tolerance by symmetry factor
        val, err = nested_quadgk(f, bz.lims; atol=atol, rtol=reltol_, maxevals = maxiters,
                                        norm = alg.norm, order = alg.order, initdivs = alg.initdivs, segbufs = alg.segbufs)
        val, err = symmetrize(f, bz, j*val, j*err)
        SciMLBase.build_solution(prob, alg, val, err, retcode = ReturnCode.Success)
    elseif alg isa PTR
        val = symptr(f, bz.B, bz.syms; npt = alg.npt, rule = alg.rule)
        val = symmetrize(f, bz, val)
        err = nothing
        SciMLBase.build_solution(prob, alg, val, err, retcode = ReturnCode.Success)
    elseif alg isa AutoPTR
        val, err = autosymptr(f, bz.B, bz.syms;
                        atol = abstol_, rtol = reltol_, maxevals = maxiters, norm=alg.norm, buffer=alg.buffer)
        val, err = symmetrize(f, bz, val, err)
        SciMLBase.build_solution(prob, alg, val, err, retcode = ReturnCode.Success)
    elseif alg isa PTR_IAI
        sol = __solvebp_call(prob, alg.ptr, sensealg, bz, bz, p;
                                reltol = reltol_, abstol = abstol_, maxiters = maxiters)
        atol = max(abstol_, reltol_*alg.iai.norm(sol))
        __solvebp_call(prob, alg.iai, sensealg, bz, bz, p;
                                reltol = zero(atol), abstol = atol, maxiters = maxiters)
    elseif alg isa AutoPTR_IAI
        sol = __solvebp_call(prob, alg.ptr, sensealg, bz, bz, p;
                                reltol = alg.reltol, abstol = abstol_, maxiters = maxiters)
        atol = max(abstol_, reltol_*alg.iai.norm(sol))
        __solvebp_call(prob, alg.iai, sensealg, bz, bz, p;
                                reltol = zero(atol), abstol = atol, maxiters = maxiters)
    elseif alg isa TAI
        l, nsym = bz.lims isa CubicLimits ? (bz.lims, nsyms(bz)) : (lattice_bz_limits(bz.B), 1)
        a = l.a
        b = l.b
        j = abs(det(bz.B))
        atol = abstol_/nsym/j # reduce absolute tolerance by symmetry factor
        sol = __solvebp_call(prob, alg.rule, sensealg, a, b, p;
                                abstol=atol, reltol=reltol_, maxiters=maxiters)
        val, err = bz.lims isa CubicLimits ? symmetrize(f, bz, j*sol.u, j*sol.resid) : (j*sol.u, j*sol.resid)
        SciMLBase.build_solution(sol.prob, sol.alg, val, err, retcode = sol.retcode, chi = sol.chi)
    end
end

"""
    AuxIAI(; order=7, norm=norm, initdivs=nothing, segbufs=nothing, parallel=nothing)

Iterated-adaptive integration using `nested_quadgk` from
[IteratedIntegration.jl](https://github.com/lxvm/IteratedIntegration.jl).
**This algorithm is the most efficient for localized integrands**.
See [`alloc_segbufs`](@ref) for how to pre-allocate segment buffers for
`nested_quadgk`.
"""
function AuxIAI end
