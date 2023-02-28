abstract type AbstractAutoBZAlgorithm <: SciMLBase.AbstractIntegralAlgorithm end

struct IAI{F,I,S} <: AbstractAutoBZAlgorithm
    order::Int
    norm::F
    initdivs::I
    segbufs::S
end
IAI(; order=7, norm=norm, initdivs=nothing, segbufs=nothing) = IAI(order, norm, initdivs, segbufs)

struct PTR{R} <: AbstractAutoBZAlgorithm
    npt::Int
    rule::R
end
PTR(; npt=50, rule=nothing) = PTR(npt, rule)

struct AutoPTR{F,B} <: AbstractAutoBZAlgorithm
    norm::F
    buffer::B
end
AutoPTR(; norm=norm, buffer=nothing) = AutoPTR(norm, buffer)

struct PTR_IAI{P,I} <: AbstractAutoBZAlgorithm
    ptr::P
    iai::I
end
PTR_IAI(; ptr=PTR(), iai=IAI()) = PTR_IAI(ptr, iai)

struct AutoPTR_IAI{P,I} <: AbstractAutoBZAlgorithm
    reltol::Float64
    ptr::P
    iai::I
end
AutoPTR_IAI(; reltol=1.0, ptr=AutoPTR(), iai=IAI()) = AutoPTR_IAI(reltol, ptr, iai)

TAI(; kwargs...) = HCubatureJL(; kwargs...)

# Imitate original interface
IntegralProblem(f, bz::SymmetricBZ, args...; kwargs...) =
    IntegralProblem(f, (bz,), (), args...; kwargs...)

# layer to intercept integrand construction
function construct_integrand(f, iip, p)
    if iip
        (y, x) -> (f(y, x, p); y)
    else
        x -> f(x, p)
    end
end

function __solvebp_call(prob::IntegralProblem, alg::AbstractAutoBZAlgorithm,
                                sensealg, (bz,)::Tuple{SymmetricBZ}, ::Tuple{}, p;
                                reltol = nothing, abstol = nothing, maxiters = typemax(Int))

    abstol_ = (abstol===nothing) ? zero(eltype(bz)) : abstol
    reltol_ = (abstol===nothing) ? (iszero(abstol_) ? sqrt(eps(typeof(abstol_))) : zero(abstol_)) : reltol
    f = construct_integrand(prob.f, isinplace(prob), prob.p)

    if alg isa IAI
        j = abs(det(bz.B))  # include jacobian determinant for map from fractional reciprocal lattice coordinates to Cartesian reciprocal lattice
        atol = abstol_/nsyms(bz)/j # reduce absolute tolerance by symmetry factor
        val, err = iterated_integration(f, bz.lims; atol=atol, rtol=reltol_, maxevals = maxiters,
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
        sol = __solvebp_call(prob, alg.ptr, sensealg, (bz,), (), p;
                                reltol = reltol_, abstol = abstol_, maxiters = maxiters)
        atol = max(abstol_, reltol_*alg.iai.norm(sol))
        __solvebp_call(prob, alg.iai, sensealg, (bz,), (), p;
                                reltol = zero(atol), abstol = atol, maxiters = maxiters)
    elseif alg isa AutoPTR_IAI
        sol = __solvebp_call(prob, alg.ptr, sensealg, (bz,), (), p;
                                reltol = alg.reltol, abstol = abstol_, maxiters = maxiters)
        atol = max(abstol_, reltol_*alg.iai.norm(sol))
        __solvebp_call(prob, alg.iai, sensealg, (bz,), (), p;
                                reltol = zero(atol), abstol = atol, maxiters = maxiters)
    end
end

for alg in (:HCubatureJL, :VEGAS)

    @eval function __solvebp_call(prob::IntegralProblem, alg::$alg,
        sensealg, (bz,)::Tuple{SymmetricBZ}, ::Tuple{}, p; kwargs...)
        (; a, b) = lattice_bz_limits(bz.B)
        j = abs(det(bz.B))
        sol = __solvebp_call(prob, alg, sensealg, a, b, p; kwargs...)
        SciMLBase.build_solution(sol.prob, sol.alg, sol.u*j, sol.resid*j, retcode = sol.retcode, chi = sol.chi)
    end

end
