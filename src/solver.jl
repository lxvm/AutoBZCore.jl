"""
    IntegralSolver(f, lb, ub, alg; abstol=0, reltol=sqrt(eps()), maxiters=typemax(Int))
    IntegralSolver(f, bz::SymmetricBZ, alg::AbstractAutoBZAlgorithm; kwargs...)

Constructs a functor that solves an integral of `f` over the given domain (e.g.
`lb` to `ub` or a `bz`) using the given `alg` to within the given tolerances.
Calling this functor, `fun` with parameters `p` using the syntax `fun(p)`
returns the estimated integral `I`. Under the hood, this uses the [Integrals.jl
interface](https://docs.sciml.ai/Integrals/stable/) for defining an
`IntegralProblem`, so `f` must be a 2-argument function `f(x,p)`, or if
in-place, a 3-argument function `f(y,x,p)`.

Also, the types [`Integrand`](@ref) and [`FourierIntegrand`](@ref) allow for
providing a partial set of parameters so that the `IntegralSolver` can interface
easily with other algorithms, such as root-finding and interpolation.
"""
struct IntegralSolver{iip,F,B,A,S,D,AT,RT,K}
    f::F
    lb::B
    ub::B
    alg::A
    sensealg::S
    do_inf_transformation::D
    abstol::AT
    reltol::RT
    maxiters::Int
    kwargs::K
    function IntegralSolver{iip}(f, lb, ub, alg;
                                sensealg = ReCallVJP(ZygoteVJP()),
                                do_inf_transformation = nothing,
                                abstol=0.0, reltol=iszero(abstol) ? sqrt(eps()) : zero(abstol),
                                maxiters=typemax(Int), kwargs...) where iip
        @assert typeof(lb)==typeof(ub) "Type of lower and upper bound must match"
        new{iip, typeof(f), typeof(lb), typeof(alg), typeof(sensealg), typeof(do_inf_transformation),
            typeof(abstol), typeof(reltol), typeof(kwargs)}(f, lb, ub, alg, sensealg, do_inf_transformation, abstol, reltol, maxiters, kwargs)
    end
end

IntegralSolver(f, lb, ub, alg; kwargs...) =
    IntegralSolver{isinplace(f, 3)}(f, lb, ub, alg; kwargs...)

construct_problem(s::IntegralSolver{iip}, p) where iip =
    IntegralProblem{iip}(s.f, s.lb, s.ub, p; s.kwargs...)

do_solve(s::IntegralSolver, p) = solve(construct_problem(s, p), s.alg,
    abstol = s.abstol, reltol = s.reltol, maxiters = s.maxiters,
    do_inf_transformation=s.do_inf_transformation, sensealg=s.sensealg)

# provide plain SciML interface
(s::IntegralSolver)(p=NullParameters()) =
    do_solve(s, p).u

# imitate general interface
IntegralSolver(f, bz::SymmetricBZ, alg::AbstractAutoBZAlgorithm; kwargs...) =
    IntegralSolver{isinplace(f, 3)}(f, bz, bz, alg; do_inf_transformation=Val(false), kwargs...)

# parallelization

"""
    batchparam(ps, nthreads)

If the cost of a calculation smoothly varies with the parameters `ps`, then
batch `ps` into `nthreads` groups where the `i`th element of group `j` is
`ps[j+(i-1)*nthreads]`
"""
function batchparam(xs, nthreads)
    batches = [Tuple{Int,eltype(xs)}[] for _ in 1:min(nthreads, length(xs))]
    for (i, x) in enumerate(xs)
        push!(batches[mod(i-1, nthreads)+1], (i, x))
    end
    batches
end

function batcheval(i, p, f, callback)
    t = time()
    sol = do_solve(f, p)
    t = time() - t
    callback(f, i, p, sol, t)
    sol.u
end


function batchsolve!(out::Vector, f::IntegralSolver, ps, nthreads, callback)
    Threads.@threads for batch in batchparam(ps, nthreads)
        f_ = Threads.threadid() == 1 ? f : deepcopy(f) # avoid data races for in place integrators
        for (i, p) in batch
            out[i] = batcheval(i, p, f_, callback)
        end
    end
    out
end

"""
    batchsolve(f, ps, [T=Base.promote_op(f, eltype(ps))]; nthreads=Threads.nthreads())

Evaluate the [`IntegralSolver`](@ref) `f` at each of the parameters `ps` in
parallel. Returns a vector containing the evaluated integrals `I`. This is
a form of multithreaded broadcasting.
"""
batchsolve(f::IntegralSolver, ps, T=Base.promote_op(f, eltype(ps)); nthreads=Threads.nthreads(), callback=(x...)->nothing) =
    batchsolve!(Vector{T}(undef, length(ps)), f, ps, nthreads, callback)
