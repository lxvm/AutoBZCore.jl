struct IntegralSolver{iip,F,L,U,A,K}
    f::F
    lb::L
    ub::U
    alg::A
    abstol::Float64
    reltol::Float64
    maxiters::Int
    kwargs::K
    function IntegralSolver{iip}(f, lb, ub, alg;
                                abstol=0.0, reltol=iszero(abstol) ? sqrt(eps()) : zero(abstol),
                                maxiters=typemax(Int), kwargs...) where iip
        new{iip, typeof(f), typeof(lb), typeof(ub), typeof(alg),
            typeof(kwargs)}(f, lb, ub, alg, abstol, reltol, maxiters, kwargs)
    end
end

IntegralSolver(f, lb, ub, alg; kwargs...) =
    IntegralSolver{isinplace(f, 3)}(f, lb, ub, alg; kwargs...)

construct_problem(s::IntegralSolver{iip}, p) where iip =
    IntegralProblem{iip}(s.f, s.lb, s.ub, p; s.kwargs...)

do_solve(s::IntegralSolver, p) = solve(construct_problem(s, p), s.alg,
    abstol = s.abstol, reltol = s.reltol, maxiters = s.maxiters)

(s::IntegralSolver)(p) = do_solve(s, p).u

# imitate general interface
IntegralSolver(f, bz::SymmetricBZ, alg; kwargs...) =
    IntegralSolver(f, (bz,), (), alg; kwargs...)

# parallelization

"""
    batchparam(xs, nthreads)

If the cost of a calculation smoothly varies with the parameters `xs`, then
batch `xs` into `nthreads` groups where the `i`th element of group `j` is
`xs[j+(i-1)*nthreads]`
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
        f_ = deepcopy(f) # to avoid data races for in place integrators
        for (i, p) in batch
            out[i] = batcheval(i, p, f_, callback)
        end
    end
    out
end

"""
    batchsolve(f, ps; nthreads=Threads.nthreads())

Evaluate the [`IntegralSolver`](@ref) `f` at each of the parameters `ps` in
parallel. Returns a vector containing the evaluated integrals `I`.
"""
function batchsolve(f, ps; T=Base.promote_op(f, eltype(ps)), nthreads=Threads.nthreads(), callback=(w,v,x,y,z)->nothing)
    batchsolve!(Vector{T}(undef, length(ps)), f, ps, nthreads, callback)
end