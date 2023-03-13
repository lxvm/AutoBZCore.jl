struct IntegralSolver{iip,F,B,A,S,D,K}
    f::F
    lb::B
    ub::B
    alg::A
    sensealg::S
    do_inf_transformation::D
    abstol::Float64
    reltol::Float64
    maxiters::Int
    kwargs::K
    function IntegralSolver{iip}(f, lb, ub, alg, ;
                                sensealg = ReCallVJP(ZygoteVJP()),
                                do_inf_transformation = nothing,
                                abstol=0.0, reltol=iszero(abstol) ? sqrt(eps()) : zero(abstol),
                                maxiters=typemax(Int), kwargs...) where iip
        @assert typeof(lb)==typeof(ub) "Type of lower and upper bound must match"
        new{iip, typeof(f), typeof(lb), typeof(alg), typeof(sensealg), typeof(do_inf_transformation),
            typeof(kwargs)}(f, lb, ub, alg, sensealg, do_inf_transformation, abstol, reltol, maxiters, kwargs)
    end
end

IntegralSolver(f, args...; kwargs...) =
    IntegralSolver{isinplace(f, 3)}(f, args...; kwargs...)

construct_problem(s::IntegralSolver{iip}, p) where iip =
    IntegralProblem{iip}(s.f, s.lb, s.ub, p; s.kwargs...)

do_solve(s::IntegralSolver, p) = solve(construct_problem(s, p), s.alg,
    abstol = s.abstol, reltol = s.reltol, maxiters = s.maxiters,
    do_inf_transformation=s.do_inf_transformation, sensealg=s.sensealg)

(s::IntegralSolver)(p=NullParameters()) = do_solve(s, p).u

# imitate general interface
IntegralSolver(f, bz::SymmetricBZ, args...; kwargs...) =
    IntegralSolver{isinplace(f, 3)}(f, bz, bz, args...; do_inf_transformation=Val(false), kwargs...)

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
batchsolve(f, ps, T=Base.promote_op(f, eltype(ps)); nthreads=Threads.nthreads(), callback=(x...)->nothing) =
    batchsolve!(Vector{T}(undef, length(ps)), f, ps, nthreads, callback)