# Performance Tips

## `executor` interface

The executor interface for integral functions and `NestedQuad` specifies whether the integrand can be evaluated in serial or in parallel and currently allows for either serial or multi-threaded execution.

```@docs
AutoBZCore.AbstractExecutor
AutoBZCore.SerialExecutor
AutoBZCore.ThreadedExecutor
```

At the time of writing, multi-threading gives a noticeable speedup to the `NestedQuad` implementation (see example below for guidance), however, the performance of the `AutoSymPTRJL` implementation is suboptimal due to an [upstream issue](https://github.com/lxvm/AutoSymPTR.jl/issues/6).
Parallelization performance improves as the cost per (batch) integrand evaluation approaches the overhead of spawning a task.

## `specialization` interface

The `specialize` interface of CommonSolve integral functions and `NestedQuad` allows control of compilation specialization and inference.

```@docs
AutoBZCore.AbstractSpecialization
AutoBZCore.DefaultSpecialize
AutoBZCore.NoSpecialize
AutoBZCore.FullSpecialize
AutoBZCore.FunctionWrapperSpecialize
```

At the time of writing, benchmarks on Julia 1.10 show that `NoSpecialize` performs reasonably well in both compilation time and run time.
Previous versions of AutoBZCore used an equivalent of `FunctionWrapperSpecialize`, which has similar performance characteristics but may break codes.
Developments in Julia's compiler may change the performance of the various options, however it is recommended to avoid using `DefaultSpecialize` and `FullSpecialize` in `NestedQuad` due to excessive compilation times.

## `NestedQuad`

If an integrand is thread-safe, then it can be run in parallel on multiple threads to speed up the evaluation of a quadrature rule.
Quadrature rules computed by `NestedQuad` have more structure than a list of points and weights because they are inherently hierarchical as each multi-dimensional integral is evaluated dimension by dimension.
This means that the dimension in the outermost integral can be parallelized at the same time as the inner dimensions, and for this reason the `NestedQuad` implementation allows the caller to control which dimensions are parallelized.
In each dimension, the quadrature rule is usually built adaptively, meaning that the usually a small number (order 10) of quadrature points can be parallelized over, so this is why parallelizing over multiple integrals can lead to greater parallel improvements.

In order to parallelize `NestedQuad`, one specifies the integration algorithm used in each dimension as a tuple of algorithms, then the `specialization` of the integral problem in each dimension, and finally the `executor` for each dimension.
Two points to remember are the order of dimensions, with the first dimension/algorithm being in the hot "inner" loop and the last one being the "outer" loop is most practical to parallelize over, and that the `specialization` and `executor` of the integrand are used for the inner integral so that the tuple of specializations and executors is one element shorts than the algorithms / dimensions.

Here is an example showing how to parallelize a 3d integral with `NestedQuad` over the outermost dimension
```@example nestedouter
using AutoBZCore
ntasks = 4 # the maximum number of tasks to schedule simultaneously on all available threads
max_batch = 1_000 # the maximum number of quadrature nodes to assign per task
f = IntegralFunction((x, p) -> prod(x) + p)
prob = IntegralProblem(f, AutoBZCore.CubicLimits((0,0,0), (1,1,1)), 3.0)
algs = (QuadratureFunction(; npt=50), QuadGKJL(), QuadGKJL()) # use trapezoidal rule on inner integral and quadgk on outer integrals
spec = (DefaultSpecialize(), NoSpecialize()) # compiler specialization more on the middle integral than the outer integral
exec = (SerialExecutor(), ThreadedExecutor(ntasks, max_batch)) # run the middle integral in serial and the outer with multi-threading
alg = NestedQuad(algs, spec, exec)
solve(prob, alg)
```
Here is an example showing how to parallelize a 3d integral with `NestedQuad` over all dimensions
```@example nestedall
using AutoBZCore
ntasks = 4 # the maximum number of tasks to schedule simultaneously on all available threads
max_batch = 1_000 # the maximum number of quadrature nodes to assign per task
prototype = 0.0 # sample object with same return type as integrand
f = IntegralFunction((x, p) -> prod(x) + p, prototype, ThreadedExecutor(ntasks, max_batch)) # multi-thread the inner integral
prob = IntegralProblem(f, AutoBZCore.CubicLimits((0,0,0), (1,1,1)), 3.0)
algs = (QuadratureFunction(; npt=50), QuadGKJL(), QuadGKJL()) # use trapezoidal rule on inner integral and quadgk on outer integrals
spec = (DefaultSpecialize(), NoSpecialize()) # compiler specialization more on the middle integral than the outer integral
exec = (ThreadedExecutor(ntasks, max_batch), ThreadedExecutor(ntasks, max_batch)) # run the middle and outer integral with multi-threading
alg = NestedQuad(algs, spec, exec)
solve(prob, alg)
```
Note that `ThreadedExecutor` uses at most `ntasks` workers in the specified dimension and that this worker pool is shared amongst all integral solvers in that dimension.
Therefore, the `ntasks` of the inner integral should always be greater than or equal to the `ntasks` of the outer integral as this will guarantee a solver for the inner integral is available for each outer integral call.
When a dimension is serial, there is a unique worker spawned for each outer integral.