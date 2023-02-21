abstract type AbstractAutoBZAlgorithm <: SciMLBase.AbstractIntegralAlgorithm end

struct IAI <: AbstractAutoBZAlgorithm
end

struct PTR <: AbstractAutoBZAlgorithm
end

struct AutoPTR <: AbstractAutoBZAlgorithm
end

struct PTR_IAI <: AbstractAutoBZAlgorithm
end

struct AutoPTR_IAI <: AbstractAutoBZAlgorithm
end

function Integrals.__solvebp_call(prob::IntegralProblem, alg::AbstractAutoBZAlgorithm,
    sensealg,
    lb, ub, p;
    reltol = 1e-8, abstol = 1e-8,
    maxiters = alg isa IAI ? 10^7 : typemax(Int))

    if alg isa IAI
        out = iterated_integration
    elseif alg isa PTR
        out = symptr
    elseif alg isa AutoPTR
        out = autosymptr
    elseif alg isa PTR_IAI
        out_ = symptr
        out = iterated_integration
    elseif alg isa AutoPTR_IAI
        out_ = autosymptr
        out = iterated_integration
    end

    SciMLBase.build_solution(prob, alg, val, err, retcode = ReturnCode.Success)
end


# New definitions of QuadGKJL to use InplaceIntegrand, segs, and segbuf features

"""
    QuadGKBZ(; order = 7, norm=norm)
One-dimensional Gauss-Kronrod integration from QuadGK.jl.
This method also takes the optional arguments `order` and `norm`.
Which are the order of the integration rule
and the norm for calculating the error, respectively.
Compare to `QuadGKJL` from Integrals.jl.
"""
struct QuadGKBZ{F,S} <: SciMLBase.AbstractIntegralAlgorithm where {F}
    order::Int
    norm::F
    segbuf::S
end
QuadGKBZ(; order = 7, norm = norm, segbuf = nothing) = QuadGKBZ(order, norm, segbuf)


function __solvebp_call(prob::IntegralProblem, alg::QuadGKBZ, sensealg, segs, qq, p;
                        reltol = nothing, abstol = nothing, maxiters = 10^7)
    if segs isa AbstractArray
        error("QuadGKBZ only accepts one-dimensional quadrature problems.")
    end
    @assert prob.batch == 0
    @assert prob.nout == 1
    p = p
    if isinplace(prob)
        f! = (y, x) -> prob.f(y, x, p)
        val, err = quadgk!(f!, segs...;
        rtol = reltol, atol = abstol, order = alg.order, norm = alg.norm, maxevals = maxiters, segbuf = alg.segbuf)
    else
    f = x -> prob.f(x, p)
    val, err = quadgk(f, segs...;
    rtol = reltol, atol = abstol, order = alg.order, norm = alg.norm, maxevals = maxiters, segbuf = alg.segbuf)
    end
    SciMLBase.build_solution(prob, QuadGKBZ(), val, err, retcode = ReturnCode.Success)
end

# TODO: when HCubature
