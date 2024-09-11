# Examples

The following are several examples of how to use the algorithms and integrands
provided by AutoBZCore.jl.
For background on the essential interface see the [Problem definitions](@ref) page

## Green's function integration

A common integral appearing in [Dynamical mean-field
theory](https://en.wikipedia.org/wiki/Dynamical_mean-field_theory) is that of
the local Green's function:
```math
G(\omega) = \int d^d \vec{k}\ \operatorname{Tr} \left[ \left( \omega - H \left( \vec{k} \right) - \Sigma(\omega) \right)^{-1} \right].
```

For simplicity, we take ``\Sigma(\omega) = -i\eta``. We can define the integrand
as a function of ``\vec{k}`` and ``H`` and parameters ``\eta, \omega``.
```@example gloc
using LinearAlgebra
gloc_integrand(k, (; h, η, ω)) = tr(inv(complex(ω,η)*I-h(k)))
```
Here we use named tuple destructuring syntax to unpack a named tuple of
parameters in the function definition.

Commonly, ``H(\vec{k})`` is evaluated using Wannier interpolation, i.e. as a
Fourier series. For a simple tight-binding model, the integer lattice, the
Hamiltonian is given by
```math
H(k) = \cos(2\pi k) = \frac{1}{2} \left( e^{2\pi ik} + e^{-2\pi ik} \right)
```
We can use the built-in function `cos` to evaluate this, however, for more
complex Fourier series it becomes easier to use the representation in terms of
Fourier coefficients. Using the package
[FourierSeriesEvaluators.jl](https://github.com/lxvm/FourierSeriesEvaluators.jl),
we can define ``H(k) = \cos(2\pi k)`` by the following:
```@example gloc
using FourierSeriesEvaluators
h = FourierSeries([0.5, 0.0, 0.5]; period=1, offset=-2)
```
The coefficient values of ``1/2`` can be determined from Euler's formula, as
used in the expansion of ``\cos`` above, and the value of `offset` is chosen to
offset the coefficient array indices, `1:3` since Julia has 1-based indexing, to
correspond to values of ``n`` in the phase factors ``e^{2\pi i n k}`` used in
the Fourier series above, i.e. `-1:1`. Now we proceed to the define the
[`IntegralProblem`](@ref) and solve it with a generic adaptive
integration scheme, [`QuadGKJL`](@ref)
```@example gloc
using AutoBZCore
dom = (0, 1)
p = (; h, η=0.1, ω=0.0)
prob = IntegralProblem(gloc_integrand, dom, p)
alg = QuadGKJL()
solve(prob, alg; abstol=1e-3).value
```

## BZ integration

To perform integration over a Brillouin zone, we can load one using the
[`load_bz`](@ref) function and then construct an [`AutoBZProblem`](@ref) to
solve. Since the Brillouin zone may be reduced using point group symmetries, a
common optimization, it is also required to specify the symmetry representation
of the integrand. Continuing the previous example, the trace of the Green's
function has no band/orbital degrees of freedom and transforms trivially under
the point group, so it is a [`TrivialRep`](@ref). The previous calculation can
be replicated as:

```@example gloc
using AutoBZCore
bz = load_bz(FBZ(), 2pi*I(1))
p = (; h, η=0.1, ω=0.0)
prob = AutoBZProblem(TrivialRep(), IntegralFunction(gloc_integrand), bz, p)
alg = TAI()
solve(prob, alg; abstol=1e-3).value
```

Now we proceed to multi-dimensional integrals. In this case, Wannier
interpolation is much more efficient when Fourier series are evaluated one
variable at a time. To understand, this suppose we have a series defined by ``M
\times M`` coefficients (i.e. a 2d series) that we want to evaluate on an ``N
\times N`` grid. Naively evaluating the series at each grid
point will require ``\mathcal{O}(M^{2} N^{2})`` operations, however, we can
reduce the complexity by pre-evaluating certain coefficients as follows
```math
f(x, y) = \sum_{m,n=1}^{M} f_{nm} e^{i(nx + my)} = \sum_{n=1}^{M} e^{inx} \left( \sum_{m=1}^{M} f_{nm} e^{imy} \right) = \sum_{n=1}^{M} e^{inx} \tilde{f}_{n}(y)
```
This means we can evaluate the series on the grid in ``\mathcal{O}(M N^2 + M^2
N)`` operations. When ``N \gg M``, this is ``\mathcal{O}(M N^{2})`` operations,
which is comparable to the computational complexity of a [multi-dimensional
FFT](https://en.wikipedia.org/wiki/Fast_Fourier_transform#Multidimensional_FFTs).
Since the constants of a FFT may not be trivial, this scheme is competitive.

Let's use this with a Fourier series corresponding to
``H(\vec{k}) = \cos(2\pi k_{x}) + \cos(2\pi k_{y})``
and define a new method of `gloc_integrand` that accepts the (efficiently)
evaluated Fourier series in the second argument
```@example gloc
h = FourierSeries([0.0; 0.5; 0.0;; 0.5; 0.0; 0.5;; 0.0; 0.5; 0.0]; period=1, offset=-2)
gloc_integrand(k, h_k, (; η, ω)) = tr(inv(complex(ω,η)*I-h_k))
```
Similar to before, we construct an [`AutoBZCore.IntegralProblem`](@ref) and this time we
take the integration domain to correspond to the full Brillouin zone of a square
lattice with lattice vectors `2pi*I(2)`.
```@example gloc
integrand = FourierIntegralFunction(gloc_integrand, h)
bz = load_bz(FBZ(2), 2pi*I(2))
p = (; η=0.1, ω=0.0)
prob = AutoBZProblem(TrivialRep(), integrand, bz, p)
alg = IAI()
solve(prob, alg).value
```
This package provides several [`AutoBZProblem` algorithms](@ref) that we
can use to solve the multidimensional integral.

The [repo's demo](https://github.com/lxvm/AutoBZCore.jl/tree/main/aps_example)
on density of states provides a complete example of how to compute and
interpolate an integral as a function of its parameters using the [`init` and
`solve!`](@ref) interface


## Density of States

Computing the density of states (DOS) of a self-adjoint, or Hermitian, operator is a
related, but distinct problem to the integrals also presented in this package.
In fact, many DOS algorithms will compute integrals to approximate the DOS of an
operator by introducing an artificial broadening.
To handle the ``T=0^{+}`` limit of the broadening, we implement the well-known
[Gilat-Raubenheimer method](https://arxiv.org/abs/1711.07993) as an algorithm
for the [`AutoBZCore.DOSProblem`](@ref)

Using the [`AutoBZCore.init`](@ref) and [`AutoBZCore.solve!`](@ref) functions, it is possible to
construct a cache to solve a [`DOSProblem`](@ref) for several energies or
several Hamiltonians. As an example of solving for several energies,
```@example dos
using AutoBZCore, FourierSeriesEvaluators, StaticArrays
h = FourierSeries(SMatrix{1,1,Float64,1}.([0.5, 0.0, 0.5]), period=1.0, offset=-2)
E = 0.3
bz = load_bz(FBZ(), [2pi;;])
prob = DOSProblem(h, E, bz)
alg = GGR(; npt=100)
cache = init(prob, alg)
Es = range(-1, 1, length=10) * 1.1
data = map(Es) do E
    cache.domain = E
    solve!(cache).value
end
```

As an example of interchanging Hamiltonians, which must remain the same type, we
can double the energies, which will halve the DOS
```@example dos
cache.domain = E
sol1 = AutoBZCore.solve!(cache)

h.c .*= 2
cache.isfresh = true
cache.domain = 2E

sol2 = AutoBZCore.solve!(cache)

sol1.value ≈ 2sol2.value
```