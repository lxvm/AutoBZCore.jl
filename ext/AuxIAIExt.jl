module AuxIAIExt

using AutoBZCore
import AutoBZCore: AuxIAI, symmetrize

using IteratedIntegration: nested_auxquadgk
import Integrals: __solvebp_call


struct AuxIAIJL{F,I,S,P} <: AbstractAutoBZAlgorithm
    order::Int
    norm::F
    initdivs::I
    segbufs::S
    parallels::P
end

function AuxIAI(; order=7, norm=norm, initdivs=nothing, segbufs=nothing, parallels=nothing)
    AuxIAIJL(order, norm, initdivs, segbufs, parallels)
end

function __solvebp_call(prob::IntegralProblem, alg::AuxIAIJL,
                                sensealg, bz::SymmetricBZ, ::SymmetricBZ, p;
                                reltol = nothing, abstol = nothing, maxiters = typemax(Int))

    abstol_ = (abstol===nothing) ? zero(eltype(bz)) : abstol
    reltol_ = (abstol===nothing) ? (iszero(abstol_) ? sqrt(eps(typeof(abstol_))) : zero(abstol_)) : reltol
    f = construct_integrand(prob.f, isinplace(prob), prob.p)

    j = abs(det(bz.B))  # include jacobian determinant for map from fractional reciprocal lattice coordinates to Cartesian reciprocal lattice
    atol = abstol_/nsyms(bz)/j # reduce absolute tolerance by symmetry factor
    val, err = nested_auxquadgk(f, bz.lims; atol=atol, rtol=reltol_, maxevals = maxiters, parallels=alg.parallels,
                                    norm = alg.norm, order = alg.order, initdivs = alg.initdivs, segbufs = alg.segbufs)
    val, err = symmetrize(f, bz, j*val, j*err)
    SciMLBase.build_solution(prob, alg, val, err, retcode = ReturnCode.Success)
end

symmetrize(f, bz, x::Integrands) = Integrands(symmetrize(f, bz, x.vals...)...)
symmetrize(f, bz, x::Errors) = Errors(symmetrize(f, bz, x.vals...)...)
symmetrize(_, ::FullBZType, x::Integrands) = x
symmetrize(_, ::FullBZType, x::Errors) = x


end