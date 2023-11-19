# Methods an algorithm must define a dispatch for
# - init_cacheval
# - do_solve
# - error_type
# no other arguments should be dispatched - instead define a function specific to the
# algorithm to use for dispatch on the integrand type

# typically each new integrand type will have to implement some algorithm-specific functions
# to be compatible. Here we just define the default behavior for out-of-place integrands

# Here we replicate the algorithms provided by SciML

"""
    QuadGKJL(; order = 7, norm = norm)

Duplicate of the QuadGKJL provided by Integrals.jl.
"""
struct QuadGKJL{F} <: IntegralAlgorithm
    order::Int
    norm::F
end
function QuadGKJL(; order = 7, norm = norm)
    return QuadGKJL(order, norm)
end

function init_cacheval(f, dom, p, alg::QuadGKJL)
    return init_segbuf(f, dom, p, alg.norm)
end
function init_segbuf(f, dom, p, norm)
    x, s = init_midpoint_scale(dom)
    u = x/oneunit(x)
    TX = typeof(u)
    fx_s = f(x, p) * s/oneunit(s)
    TI = typeof(fx_s)
    TE = typeof(norm(fx_s))
    return IteratedIntegration.alloc_segbuf(TX, TI, TE)
end
init_midpoint_scale(dom::PuncturedInterval) = init_midpoint_scale(endpoints(dom)...)
function init_midpoint_scale(a::T, b::T) where {T}
    # we try to reproduce the initial midpoint used by QuadGK, and scale just needs right units
    s = float(oneunit(T))
    if one(T) isa Real
        x = if (infa = isinf(a)) & (infb = isinf(b))
            float(zero(T))
        elseif infa
            float(b - oneunit(b))
        elseif infb
            float(a + oneunit(a))
        else
            (a+b)/2
        end
        return x, s
    else
        return (a+b)/2, s
    end
end

function do_solve(f::F, dom, p, alg::QuadGKJL, cacheval;
                    reltol = nothing, abstol = nothing, maxiters = typemax(Int)) where {F}
    segs = segments(dom)
    return do_solve_quadgk(f, segs, p, cacheval, alg.order, alg.norm, reltol, abstol, maxiters)
end
function do_solve_quadgk(f, segs, p, cacheval, order, norm, reltol, abstol, maxiters)
    # we need to strip units from the limits since infinity transformations change the units
    # of the limits, which can break the segbuf
    u = oneunit(eltype(segs))
    usegs = map(x -> x/u, segs)
    g = x -> f(u*x, p)
    val, err = quadgk(g, usegs..., maxevals = maxiters,
                    rtol = reltol, atol = isnothing(abstol) ? abstol : abstol/u, order = order, norm = norm, segbuf=cacheval)
    return IntegralSolution(u*val, u*err, true, -1)
end

"""
    integralerror(alg::IntegralAlgorithm, I)

Return the norm/error of the value of the integral `I` as computed by the algorithm `alg`.
"""
integralerror(alg::QuadGKJL, I) = alg.norm(I)

"""
    HCubatureJL(; norm=norm, initdiv=1)

A copy of `HCubatureJL` from Integrals.jl.
"""
struct HCubatureJL{N} <: IntegralAlgorithm
    norm::N
    initdiv::Int
end
HCubatureJL(; norm=norm, initdiv=1) = HCubatureJL(norm, initdiv)

function init_cacheval(f, dom, p, ::HCubatureJL)
    return Some(nothing)
end

function do_solve(f, dom, p, alg::HCubatureJL, cacheval;
                    reltol = 0, abstol = 0, maxiters = typemax(Int))
    a, b = endpoints(dom)
    g = assemble_hintegrand(f, dom, p)
    routine = a isa Number ? hquadrature : hcubature
    val, err = routine(g, a, b; norm = alg.norm, initdiv = alg.initdiv, atol=abstol, rtol=reltol, maxevals=maxiters)
    return IntegralSolution(val, err, true, -1)
end
assemble_hintegrand(f, dom, p) = x -> f(x, p)

integralerror(alg::HCubatureJL, I) = alg.norm(I)

"""
    trapz(n::Integer)

Return the weights and nodes on the standard interval [-1,1] of the [trapezoidal
rule](https://en.wikipedia.org/wiki/Trapezoidal_rule).
"""
function trapz(n::Integer)
    @assert n > 1
    r = range(-1, 1, length=n)
    x = collect(r)
    halfh = step(r)/2
    h = step(r)
    w = [ (i == 1) || (i == n) ? halfh : h for i in 1:n ]
    return (x, w)
end

"""
    QuadratureFunction(; fun=trapz, npt=50, nthreads=Threads.nthreads())

Quadrature rule for the standard interval [-1,1] computed from a function `x, w = fun(npt)`.
The nodes and weights should be set so the integral of `f` on [-1,1] is `sum(w .* f.(x))`.
The default quadrature rule is [`trapz`](@ref), although other packages provide rules, e.g.

    using FastGaussQuadrature
    alg = QuadratureFunction(fun=gausslegendre, npt=100)

`nthreads` sets the numbers of threads used to parallelize the quadrature only when the
integrand is a [`BatchIntegrand`](@ref), in which case the user must parallelize the
integrand evaluations. For no threading set `nthreads=1`.
"""
struct QuadratureFunction{F} <: IntegralAlgorithm
    fun::F
    npt::Int
    nthreads::Int
end
QuadratureFunction(; fun=trapz, npt=50, nthreads=Threads.nthreads()) = QuadratureFunction(fun, npt, nthreads)
function init_cacheval(f, dom, p, alg::QuadratureFunction)
    buf = init_buffer(f, alg.nthreads)
    x, w = alg.fun(alg.npt)
    return (rule=[(w,x) for (w,x) in zip(w,x)], buffer=buf)
end
init_buffer(f, len) = nothing


function do_solve(f, dom, p, ::QuadratureFunction, cacheval;
                    reltol = nothing, abstol = nothing, maxiters = typemax(Int))
    rule = cacheval.rule; buffer=cacheval.buffer
    dom isa PuncturedInterval || throw(ArgumentError("QuadratureFunction only supports 1d quadrature problems"))
    segs = segments(dom)
    g = assemble_pintegrand(f, p, dom, rule)
    A = sum(1:length(segs)-1) do i
        a, b = segs[i], segs[i+1]
        s = (b-a)/2
        arule = AutoSymPTR.AffineQuad(rule, s, a, 1, s)
        return AutoSymPTR.quadsum(arule, g, s, buffer)
    end

    return IntegralSolution(A, nothing, true, -1)
end
assemble_pintegrand(f, p, dom, rule) = x -> f(x, p)

integralerror(::QuadratureFunction, I) = nothing

# Here we put the quadrature algorithms from IteratedIntegration

"""
    AuxQuadGKJL(; order = 7, norm = norm)

Generalization of the QuadGKJL provided by Integrals.jl that allows for `AuxValue`d
integrands for auxiliary integration and multi-threaded evaluation with the `batch` argument
to `IntegralProblem`
"""
struct AuxQuadGKJL{F} <: IntegralAlgorithm
    order::Int
    norm::F
end
function AuxQuadGKJL(; order = 7, norm = norm)
    return AuxQuadGKJL(order, norm)
end

function init_cacheval(f, dom, p, alg::AuxQuadGKJL)
    return init_segbuf(f, dom, p, alg.norm)
end

function do_solve(f, dom, p, alg::AuxQuadGKJL, cacheval;
                    reltol = nothing, abstol = nothing, maxiters = typemax(Int))

    segs = segments(dom)
    return do_solve_auxquadgk(f, segs, p, cacheval, alg.order, alg.norm, reltol, abstol, maxiters)
end

function do_solve_auxquadgk(f, segs, p, cacheval, order, norm, reltol, abstol, maxiters)
    u = oneunit(eltype(segs))
    usegs = map(x -> x/u, segs)
    g = x -> f(u*x, p)
    val, err = auxquadgk(g, usegs, maxevals = maxiters,
                    rtol = reltol, atol = isnothing(abstol) ? abstol : abstol/u, order = order, norm = norm, segbuf=cacheval)
    return IntegralSolution(u*val, u*err, true, -1)
end

integralerror(alg::AuxQuadGKJL, I) = alg.norm(I)


"""
    ContQuadGKJL(; order = 7, norm = norm, rho = 1.0, rootmeth = IteratedIntegration.ContQuadGK.NewtonDeflation())

A 1d contour deformation quadrature scheme for scalar, complex-valued integrands. It
defaults to regular `quadgk` behavior on the real axis, but if it finds a root of 1/f
nearby, in the sense of Bernstein ellipse for the standard segment `[-1,1]` with semiaxes
`cosh(rho)` and `sinh(rho)`, on either the upper/lower half planes, then it dents the
contour away from the presumable pole.
"""
struct ContQuadGKJL{F,M} <: IntegralAlgorithm
    order::Int
    norm::F
    rho::Float64
    rootmeth::M
end
function ContQuadGKJL(; order = 7, norm = norm, rho = 1.0, rootmeth = IteratedIntegration.ContQuadGK.NewtonDeflation())
    return ContQuadGKJL(order, norm, rho, rootmeth)
end

function init_cacheval(f, dom, p, alg ::ContQuadGKJL)
    a, b = endpoints(dom)
    x, s = (a+b)/2, (b-a)/2
    TX = typeof(x)
    fx_s = one(ComplexF64) * s # currently the integrand is forcibly written to a ComplexF64 buffer
    TI = typeof(fx_s)
    TE = typeof(alg.norm(fx_s))
    r_segbuf = IteratedIntegration.ContQuadGK.PoleSegment{TX,TI,TE}[]
    fc_s = f(complex(x), p) * complex(s) # the regular evalrule is used on complex segments
    TCX = typeof(complex(x))
    TCI = typeof(fc_s)
    TCE = typeof(alg.norm(fc_s))
    c_segbuf = IteratedIntegration.ContQuadGK.Segment{TCX,TCI,TCE}[]
    return (r=r_segbuf, c=c_segbuf)
end

function do_solve(f, dom, p, alg::ContQuadGKJL, cacheval;
                    reltol = nothing, abstol = nothing, maxiters = typemax(Int))

    dom isa PuncturedInterval || throw(ArgumentError("ContQuadGKJL only supports 1d quadrature problems"))
    segs = segments(dom)
    g = assemble_cont_integrand(f, p)
    val, err = contquadgk(g, segs, maxevals = maxiters, rho = alg.rho, rootmeth = alg.rootmeth,
                    rtol = reltol, atol = abstol, order = alg.order, norm = alg.norm, r_segbuf=cacheval.r, c_segbuf=cacheval.c)
    return IntegralSolution(val, err, true, -1)
end
assemble_cont_integrand(f, p) = x -> f(x, p)

integralerror(alg::ContQuadGKJL, I) = alg.norm(I)

"""
    MeroQuadGKJL(; order = 7, norm = norm, rho = 1.0, rootmeth = IteratedIntegration.MeroQuadGK.NewtonDeflation())

A 1d pole subtraction quadrature scheme for scalar, complex-valued integrands that are
meromorphic. It defaults to regular `quadgk` behavior on the real axis, but if it finds
nearby roots of 1/f, in the sense of Bernstein ellipse for the standard segment `[-1,1]`
with semiaxes `cosh(rho)` and `sinh(rho)`, it attempts pole subtraction on that segment.
"""
struct MeroQuadGKJL{F,M} <: IntegralAlgorithm
    order::Int
    norm::F
    rho::Float64
    rootmeth::M
end
function MeroQuadGKJL(; order = 7, norm = norm, rho = 1.0, rootmeth = IteratedIntegration.MeroQuadGK.NewtonDeflation())
    return MeroQuadGKJL(order, norm, rho, rootmeth)
end

function init_cacheval(f, dom, p, alg::MeroQuadGKJL)
    a, b = endpoints(dom)
    x, s = (a + b)/2, (b-a)/2
    fx_s = one(ComplexF64) * s # ignore the actual integrand since it is written to CF64 array
    err = alg.norm(fx_s)
    return IteratedIntegration.alloc_segbuf(typeof(x), typeof(fx_s), typeof(err))
end

function do_solve(f, dom, p, alg::MeroQuadGKJL, cacheval;
                    reltol = nothing, abstol = nothing, maxiters = typemax(Int))

    dom isa PuncturedInterval || throw(ArgumentError("MeroQuadGKJL only supports 1d quadrature problems"))
    segs = segments(dom)
    g = assemble_mero_integrand(f, p)
    val, err = meroquadgk(g, segs, maxevals = maxiters, rho = alg.rho, rootmeth = alg.rootmeth,
                    rtol = reltol, atol = abstol, order = alg.order, norm = alg.norm, segbuf=cacheval)
    return IntegralSolution(val, err, true, -1)
end
assemble_mero_integrand(f, p) = x -> f(x, p)

integralerror(alg::MeroQuadGKJL, I) = alg.norm(I)


# Algorithms from AutoSymPTR.jl

"""
    MonkhorstPack(; npt=50, syms=nothing, nthreads=Threads.nthreads())

Periodic trapezoidal rule with a fixed number of k-points per dimension, `npt`,
using the `PTR` rule from [AutoSymPTR.jl](https://github.com/lxvm/AutoSymPTR.jl).
`nthreads` sets the numbers of threads used to parallelize the quadrature only when the
integrand is a [`BatchIntegrand`](@ref), in which case the user must parallelize the
integrand evaluations. For no threading set `nthreads=1`.
**The caller should check that the integral is converged w.r.t. `npt`**.
"""
struct MonkhorstPack{S} <: IntegralAlgorithm
    npt::Int
    syms::S
    nthreads::Int
end
MonkhorstPack(; npt=50, syms=nothing, nthreads=Threads.nthreads()) = MonkhorstPack(npt, syms, nthreads)
function init_rule(dom::Basis, alg::MonkhorstPack)
    # rule = AutoSymPTR.MonkhorstPackRule(alg.syms, alg.a, alg.nmin, alg.nmax, alg.n₀, alg.Δn)
    # return rule(eltype(dom), Val(ndims(dom)))
    if alg.syms === nothing
        return AutoSymPTR.PTR(eltype(dom), Val(ndims(dom)), alg.npt)
    else
        return AutoSymPTR.MonkhorstPack(eltype(dom), Val(ndims(dom)), alg.npt, alg.syms)
    end
end

function init_cacheval(f, dom, p, alg::MonkhorstPack)
    dom isa Basis || throw(ArgumentError("MonkhorstPack only supports Basis for domain. Please open an issue."))
    rule = init_rule(dom, alg)
    buf = init_buffer(f, alg.nthreads)
    return (rule=rule, buffer=buf)
end

function do_solve(f, dom, p, alg::MonkhorstPack, cacheval;
                    reltol = nothing, abstol = nothing, maxiters = typemax(Int))
    dom isa Basis || throw(ArgumentError("MonkhorstPack only supports Basis for domain. Please open an issue."))
    g = assemble_pintegrand(f, p, dom, cacheval.rule)
    I = cacheval.rule(g, dom, cacheval.buffer)
    return IntegralSolution(I, nothing, true, -1)
end

integralerror(::MonkhorstPack, I) = nothing

"""
    AutoSymPTRJL(; norm=norm, a=1.0, nmin=50, nmax=1000, n₀=6, Δn=log(10), keepmost=2, nthreads=Threads.nthreads())

Periodic trapezoidal rule with automatic convergence to tolerances passed to the
solver with respect to `norm` using the routine `autosymptr` from
[AutoSymPTR.jl](https://github.com/lxvm/AutoSymPTR.jl).
`nthreads` sets the numbers of threads used to parallelize the quadrature only when the
integrand is a [`BatchIntegrand`](@ref), in which case the user must parallelize the
integrand evaluations. For no threading set `nthreads=1`.
**This algorithm is the most efficient for smooth integrands**.
"""
struct AutoSymPTRJL{F,S} <: IntegralAlgorithm
    norm::F
    a::Float64
    nmin::Int
    nmax::Int
    n₀::Float64
    Δn::Float64
    keepmost::Int
    syms::S
    nthreads::Int
end
function AutoSymPTRJL(; norm=norm, a=1.0, nmin=50, nmax=1000, n₀=6.0, Δn=log(10), keepmost=2, syms=nothing, nthreads=Threads.nthreads())
    return AutoSymPTRJL(norm, a, nmin, nmax, n₀, Δn, keepmost, syms, nthreads)
end
function init_rule(dom::Basis, alg::AutoSymPTRJL)
    return AutoSymPTR.MonkhorstPackRule(alg.syms, alg.a, alg.nmin, alg.nmax, alg.n₀, alg.Δn)
end
function init_cacheval(f, dom, p, alg::AutoSymPTRJL)
    dom isa Basis || throw(ArgumentError("AutoSymPTRJL only supports Basis for domain. Please open an issue."))
    rule = init_rule(dom, alg)
    cache = AutoSymPTR.alloc_cache(eltype(dom), Val(ndims(dom)), rule)
    buffer = init_buffer(f, alg.nthreads)
    return (rule=rule, cache=cache, buffer=buffer)
end

function do_solve(f, dom, p, alg::AutoSymPTRJL, cacheval;
                    reltol = nothing, abstol = nothing, maxiters = typemax(Int))

    dom isa Basis || throw(ArgumentError("AutoSymPTRJL only supports Basis for domain. Please open an issue."))
    g = assemble_pintegrand(f, p, dom, first(cacheval.cache))
    val, err = autosymptr(g, dom; syms = alg.syms, rule = cacheval.rule, cache = cacheval.cache, keepmost = alg.keepmost,
        abstol = abstol, reltol = reltol, maxevals = maxiters, norm=alg.norm, buffer=cacheval.buffer)
    return IntegralSolution(val, err, true, -1)
end

integralerror(alg::AutoSymPTRJL, I) = alg.norm(I)

# Meta-algorithms

"""
    NestedQuad(alg::IntegralAlgorithm)
    NestedQuad(algs::IntegralAlgorithm...)

Nested integration by repeating one quadrature algorithm or composing a list of algorithms.
The domain of integration must be an `AbstractIteratedLimits` from the
IteratedIntegration.jl package. Analogous to `nested_quad` from IteratedIntegration.jl.
The integrand should expect `SVector` inputs. Do not use this for very high-dimensional
integrals, since the compilation time scales very poorly with respect to dimensionality.
In order to improve the compilation time, FunctionWrappers.jl is used to enforce type
stability of the integrand, so you should always pick the widest integration limit type so
that inference works properly. For example, if [`ContQuadGKJL`](@ref) is used as an
algorithm in the nested scheme, then the limits of integration should be made complex.
"""
struct NestedQuad{T} <: IntegralAlgorithm
    algs::T
    NestedQuad(alg::IntegralAlgorithm) = new{typeof(alg)}(alg)
    NestedQuad(algs::Tuple{Vararg{IntegralAlgorithm}}) = new{typeof(algs)}(algs)
end
NestedQuad(algs::IntegralAlgorithm...) = NestedQuad(algs)

integralerror(alg::NestedQuad, I) = integralerror(alg.algs isa IntegralAlgorithm ? alg.algs : last(alg.algs), I)

"""
    AbsoluteEstimate(est_alg, abs_alg; kws...)

Most algorithms are efficient when using absolute error tolerances, but how do you know the
size of the integral? One option is to estimate it using second algorithm.

A multi-algorithm to estimate an integral using an `est_alg` to generate a rough estimate of
the integral that is combined with a user's relative tolerance to re-calculate the integral
to higher accuracy using the `abs_alg`. The keywords passed to the algorithm may include
`reltol`, `abstol` and `maxiters` and are given to the `est_alg` solver. They should limit
the amount of work of `est_alg` so as to only generate an order-of-magnitude estimate of the
integral. The tolerances passed to `abs_alg` are `abstol=max(abstol,reltol*norm(I))` and
`reltol=0`.
"""
struct AbsoluteEstimate{E<:IntegralAlgorithm,A<:IntegralAlgorithm,F,K<:NamedTuple} <: IntegralAlgorithm
    est_alg::E
    abs_alg::A
    norm::F
    kws::K
end
function AbsoluteEstimate(est_alg, abs_alg; norm=norm, kwargs...)
    kws = NamedTuple(kwargs)
    checkkwargs(kws)
    return AbsoluteEstimate(est_alg, abs_alg, norm, kws)
end

function init_cacheval(f, dom, p, alg::AbsoluteEstimate)
    return (est=init_cacheval(f, dom, p, alg.est_alg),
            abs=init_cacheval(f, dom, p, alg.abs_alg))
end

function do_solve(f, dom, p, alg::AbsoluteEstimate, cacheval;
                    abstol=nothing, reltol=nothing, maxiters=typemax(Int))
    sol = do_solve(f, dom, p, alg.est_alg, cacheval.est; alg.kws...)
    val = alg.norm(sol.u) # has same units as sol
    rtol = reltol === nothing ? sqrt(eps(one(val))) : reltol # use the precision of the solution to set the default relative tolerance
    atol = max(abstol === nothing ? zero(val) : abstol, rtol*val)
    return do_solve(f, dom, p, alg.abs_alg, cacheval.abs;
                    abstol=atol, reltol=zero(rtol), maxiters=maxiters)
end

integralerror(alg::AbsoluteEstimate, I) = integralerror(alg, I)


"""
    EvalCounter(::IntegralAlgorithm)

An algorithm which counts the evaluations used by another algorithm.
The count is stored in the `sol.numevals` field.
"""
struct EvalCounter{T<:IntegralAlgorithm} <: IntegralAlgorithm
    alg::T
end

function init_cacheval(f, dom, p, alg::EvalCounter)
    return init_cacheval(f, dom, p, alg.alg)
end

function do_solve(f, dom, p, alg::EvalCounter, cacheval; kws...)
    return do_solve_evalcounter(f, dom, p, alg.alg, cacheval; kws...)
end

function do_solve_evalcounter(f, dom, p, alg, cacheval; kws...)
    n::Int = 0
    g = (x, p) -> (n += 1; f(x, p))
    sol = do_solve(g, dom, p, alg, cacheval; kws...)
    return IntegralSolution(sol.u, sol.resid, sol.retcode, n)
end

integralerror(alg::EvalCounter, I) = integralerror(alg.alg, I)
