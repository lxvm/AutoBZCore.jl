# parallelization

"""
    batch_smooth_param(xs, nthreads)

If the cost of a calculation smoothly varies with the parameters `xs`, then
batch `xs` into `nthreads` groups where the `i`th element of group `j` is
`xs[j+(i-1)*nthreads]`
"""
function batch_smooth_param(xs, nthreads)
    batches = [Tuple{Int,eltype(xs)}[] for _ in 1:min(nthreads, length(xs))]
    for (i, x) in enumerate(xs)
        push!(batches[mod(i-1, nthreads)+1], (i, x))
    end
    batches
end

"""
    parallel_integration(f, ps; nthreads=Threads.nthreads())

Evaluate the `AbstractIntegrator` `f` at each of the parameters `ps` in
parallel. Returns a named tuple `(I, E, t, p)` containing the integrals `I`, the
extra data from the integration routine `E`, timings `t`, and the original
parameters `p`. The parameter layout in `ps` should such that `f(ps[i]...)` runs
"""
function parallel_integration(f, ps; nthreads=Threads.nthreads())
    T = Base.promote_op(first∘f, eltype(ps))
    ints = Vector{T}(undef, length(ps))
    # extra = Vector{???}(undef, length(ps))
    ts = Vector{Float64}(undef, length(ps))
    @info "Beginning parameter sweep using $(f.routine)"
    @info "using $nthreads threads for parameter parallelization"
    batches = batch_smooth_param(ps, nthreads)
    t = time()
    Threads.@threads for batch in batches
        f_ = deepcopy(f) # to avoid data races for in place integrators
        for (i, p) in batch
            @info @sprintf "starting parameter %i" i
            t_ = time()
            ints[i], = f_(p...)
            # TODO: ints[i], extra[i] = quad_return(routine(f), f_(p...))
            ts[i] = time() - t_
            @info @sprintf "finished parameter %i in %e (s) wall clock time" i ts[i]
        end
    end
    @info @sprintf "Finished parameter sweep in %e (s) CPU time and %e (s) wall clock time" sum(ts) (time()-t)
    (I=ints, t=ts, p=ps)
end
