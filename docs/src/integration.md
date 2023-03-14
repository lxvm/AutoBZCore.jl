# Integration

The integration tools provided by AutoBZCore.jl are very flexible and allow
for many kinds of user-defined integrals with specialized behaviors. Examples
implementations can be found in AutoBZ.jl.

## Parametrizing integrals

[Integrals.jl](https://docs.sciml.ai/Integrals/stable/) provides a generic and
unified interface for computing integrals that works well for all kinds of
problems. This package builds on that interface by offering
[`IntegralSolver`](@ref)s to easily parametrize an integral for a given
integrand, limits of integration, algorithm, and error tolerances. Until the
SciML community arrives at a
[consensus](https://github.com/SciML/DifferentialEquations.jl/issues/881) of
what a parametrization interface looks like, this gets the job done.

```@docs
IntegralSolver
```

## Integrands

User-defined integrands should be able to take any number of parameters and
should be entirely general. We only require that the integrand functions take
only positional arguments.

The integrand types below solve the problem of providing the integrand evaluator
parameters a partial set of parameters so that the [`IntegralSolver`](@ref) can
parametrize the remaining ones. Namely, the full set of parameters is the union
of the ones included in the integrand follow by those passed to the solver,
therefore **the caller is responsible for ensuring that the ordered union of
parameters they pass to the integrand and solver matches the positional
arguments of their integrand evaluator**. Note that this behavior slightly
differs from using the SciML routine `remake` for an `IntegralProblem`.

```@docs
Integrand
FourierIntegrand
```

## Algorithms

There are many Brillouin zone integration algorithms, but the ones provided by
this package work for black-box integrands and converge automatically to a
requested tolerance, except where noted. Additionally, the algorithms are
fully compatible with integration over the IBZ, except where noted. More details
on the algorithms can be found from their respective packages.

```@docs
AbstractAutoBZAlgorithm
IAI
PTR
AutoPTR
PTR_IAI
AutoPTR_IAI
TAI
```

### Allocations for algorithms

In practice, allocating memory for an algorithm can help improve its
performance. This is particularly noticeable for PTR, where one of the main
optimizations is to store the Fourier series evaluated on the PTR grid for reuse
across multiple compatible calls. The routines below show how to pre-evaluate or
pre-allocate the memory necessary for different algorithms.

```@docs
AutoBZCore.alloc_rule
AutoBZCore.alloc_autobuffer
AutoBZCore.alloc_segbufs
```

### Parallelization

This package parallelizes ``k``-sums in the [`PTR`](@ref) and [`AutoPTR`](@ref)
routines by default, which gives good multi-threaded performance for those
algorithms when the number of ``k``-points is large. On the other hand, if the
number of parameter points for which you would like to evaluate an integral is
large, then parameter-parallelization is an effective strategy and done by the
routines below.

```@docs
batchsolve
AutoBZCore.batchparam
```