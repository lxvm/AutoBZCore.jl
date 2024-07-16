"""
    DOSAlgorithm

Abstract supertype for algorithms for computing density of states
"""
abstract type DOSAlgorithm end

"""
    DOSProblem(H, domain, [p=NullParameters()])

Define a problem for the density of states of a Hermitian or self-adjoint
operator depending on a parameter, H(p), on a given `domain` in its spectrum.
The mathematical definition we use is
```math
D(E) = \\sum_{k \\in p} \\sum_{\\lambda \\in \\text{spectrum}(H(k))} \\delta(E - \\lambda)
```
where ``E \\in \\text{domain}`` and ``\\delta`` is the Dirac Delta distribution.

## Arguments
- `H`: a linear operator depending on a parameter, H(p), that is finite
  dimensional (e.g: tight binding model) or infinite dimensional (e.g. DFT data)
- `domain`: a set in the spectrum for which an approximation of the
  density-of-states is desired. Can be a single point, in which case the
  solution will return the estimated density of states at that eigenvalue, or an
  interval, in which case the solution will return a function approximation to
  the density of states on that interval in the spectrum that should be
  understood as a distribution or measure.
- `p`: optional parameters on which `H` depends for which the density of states
  should sum over. Can be discrete (e.g. for `H` a Hamiltonian with spin degrees
  of freedom) or continuous (e.g. for `H` a Hamiltonian parameterized by crystal
  momentum).
"""
struct DOSProblem{H,D,P,K<:NamedTuple}
    H::H
    domain::D
    p::P
    kwargs::K
end
function DOSProblem(H, domain, p=NullParameters(); kws...)
    return DOSProblem(H, domain, p, NamedTuple(kws))
end

struct DOSSolution{V,S}
    value::V
    retcode::ReturnCode
    stats::S
end

# store the data in a mutable cache so that the user can update the cache and
# compute the result again without setting up new problems.
mutable struct DOSCache{H,D,P,A,C,K}
    H::H
    domain::D
    p::P
    alg::A
    cacheval::C
    isfresh::Bool # true if H has been replaced/modified, otherwise false
    kwargs::K
end

function Base.setproperty!(cache::DOSCache, name::Symbol, item)
    if name === :H
        setfield!(cache, :isfresh, true)
    end
    return setfield!(cache, name, item)
end

# by default, algorithms won't have anything in the cache
init_cacheval(h, dom, p, ::DOSAlgorithm) = nothing

# check same keywords as for integral problems: abstol, reltol, maxiters
checkkwargs_dos(kws) = checkkwargs(kws)

"""
    init(::DOSProblem, ::DOSAlgorithm; kwargs...)::DOSCache

Create a cache of the data used by an algorithm to solve the given problem.
"""
function init(prob::DOSProblem, alg::DOSAlgorithm; kwargs...)
    h = prob.H; dom = prob.domain; p = prob.p
    kws = (; prob.kwargs..., kwargs...)
    checkkwargs_dos(kws)
    cacheval = init_cacheval(h, dom, p, alg)
    return DOSCache(h, dom, p, alg, cacheval, false, kws)
end

"""
    solve!(::DOSCache)::DOSSolution

Compute the solution of a problem from the initialized cache
"""
function solve!(c::DOSCache)

    if c.isfresh
        c.cacheval = init_cacheval(c.H, c.domain, c.p, c.alg)
        c.isfresh = false
    end

    return dos_solve(c.H, c.domain, c.p, c.alg, c.cacheval; c.kwargs...)
end

function dos_solve end

"""
    solve(::DOSProblem, ::DOSAlgorithm; kws...)::DOSSolution
"""
solve(prob::DOSProblem, alg::DOSAlgorithm; kwargs...)
