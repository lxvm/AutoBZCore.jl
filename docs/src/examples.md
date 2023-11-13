# Examples

The following are several examples of how to use the algorithms and integrands
provided by AutoBZCore.jl.
For background on defining integrals see the [Problem definitions](@ref) page

## Green's function

A common integral appearing in [Dynamical mean-field
theory](https://en.wikipedia.org/wiki/Dynamical_mean-field_theory) is that of
the local Green's function:
```math
G(\omega) = \int d^d \vec{k}\ \operatorname{Tr} \left[ \left( \omega - H \left( \vec{k} \right) - \Sigma(\omega) \right)^{-1} \right].
```

For simplicity, we take ``\Sigma(\omega) = -i\eta``. We can define the integrand
as a function of ``\vec{k}`` and ``H`` and (required) parameters ``\eta, \omega``.
```julia
using LinearAlgebra
gloc_integrand(k, h; η, ω) = inv(complex(ω,η)*I-h(k))
```

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
```julia
using FourierSeriesEvaluators
h = FourierSeries([0.5, 0.0, 0.5]; period=1, offset=-2)
```
The coefficient values of ``1/2`` can be determined from Euler's formula, as
used in the expansion of ``cos`` above, and the value of `offset` is chosen to
offset the coefficient array indices, `1:3` since Julia has 1-based indexing, to
correspond to values of ``n`` in the phase factors ``e^{2\pi i n k}`` used in
the Fourier series above, i.e. `-1:1`. Now we proceed to the define the integral problem
```julia
using AutoBZCore
using AutoBZCore: IntegralProblem
integrand = ParameterIntegrand(gloc_integrand, h, η=0.1)
prob = IntegralProblem(integrand, 0, 1)
```
Here we wrapped our function with two of its arguments, `h, η` as a
[`ParameterIntegrand`](@ref) that allows us to provide partial arguments so that
we can solve the integral as a function of the remaining parameters, in this
case `ω`. We also created an [`AutoBZCore.IntegralProblem`](@ref) to integrate our function
over its period ``[0,1]``. To solve this problem, we pick any of the package's
[Integral algorithms](@ref) and the
tolerance to which we would like the solution. Then we make an
[`IntegralSolver`](@ref) to evaluate ``G(\omega)`` as a function.
```julia
alg = QuadGKJL()
gloc = IntegralSolver(prob, alg; abstol=1e-3)
gloc(ω=0.0) # -2.7755575615628914e-17 - 0.9950375451895513im
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

This capability is provided by [`FourierIntegrand`](@ref).
Let's use this with a Fourier series corresponding to
``H(\vec{k}) = \cos(2\pi k_{x}) + \cos(2\pi k_{y})``
```julia
h = FourierSeries([0.0; 0.5; 0.0;; 0.5; 0.0; 0.5;; 0.0; 0.5; 0.0]; period=1, offset=-2)
integrand = FourierIntegrand(gloc_integrand, h, η=0.1)
```
However, since [`FourierIntegrand`](@ref) evaluates ``H(k)`` for us and gives it
as a `FourierValue` together with ``k``, we need to define another method for
our integrand to comply with the interface
```julia
gloc_integrand(h_k::FourierValue; η, ω) = inv(complex(ω,η)*I-h_k.s)
```
Similar to before, we construct an [`AutoBZCore.IntegralProblem`](@ref) and this time we
take the integration domain to correspond to the full Brillouin zone of a square
lattice with lattice vectors `2pi*I(2)`. (See the
[Reference](@ref) for more details on constructing BZs.)
```julia
bz = load_bz(FBZ(2), 2pi*I(2))
prob = IntegralProblem(integrand, bz)
```
This package provides several [BZ-specific integral algorithms](@ref) that we
can use to solve the multidimensional integral.
```julia
alg = IAI()
gloc = IntegralSolver(prob, alg; abstol=1e-3)
gloc(ω=0.0) # 1.5265566588595902e-16 - 1.3941704019631334im
```

## Density of states

The [repo's demo](https://github.com/lxvm/AutoBZCore.jl/tree/main/aps_example)
on density of states provides a complete example of how to compute and
interpolate an integral as a function of its parameters.
