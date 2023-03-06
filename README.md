# AutoBZCore.jl

[Documentation](https://lxvm.github.io/AutoBZCore.jl/dev/)

This package provides interfaces to integration algorithms for periodic
functions that typically occur in computational solid state physics in the form
of Brillouin-zone integrals.

The main types it exports are `FourierIntegrand`, which is a constructor for
user-defined functions of tight-binding Hamiltonians and such (represented using
[`AbstractFourierSeries`](https://github.com/lxvm/FourierSeriesEvaluators.jl)),
`SymmetricBZ`, which represents a Brillouin zone and its symmetries, and
`IntegralSolver`, which parametrizes the interfaces defined in
[Integrals.jl](https://github.com/SciML/Integrals.jl) to solve integrals using a
[functor
interface](https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects-1).
The package also exports integration algorithms including: `IAI`, `PTR`,
`AutoPTR`, and `TAI`, and it  evaluates the `FourierIntegrand` efficiently using
the structure of the algorithms.

Thus, it provides the core, user-extensible functionality of
[AutoBZ.jl](https://lxvm.github.io/AutoBZ.jl/dev/).


## Usage

For an example of defining a BZ integral for density of states, see the
[aps_example](https://github.com/lxvm/AutoBZCore.jl/tree/main/aps_example/). To
do additional integrals that are not over a BZ, directly use the
`IntegralSolver` with a provided function or with an `Integrand`, which is a
type similar to `FourierIntegrand` that is exported for convenience.
Also note that `IntegralSolver`s can be composed in order to do multiple nested
integrals.


## Research

The paper by [Kaye et
al., (2023)](http://arxiv.org/abs/2211.12959) is the motivation for this package.
If you find this package useful for your research, please consider citing the
paper.

```
@misc{Kaye:AutoBZ:22,
	title = {Automatic, high-order, and adaptive algorithms for {Brillouin} zone integration},
	copyright = {All rights reserved},
	url = {http://arxiv.org/abs/2211.12959},
	doi = {10.48550/arXiv.2211.12959},
	urldate = {2022-11-26},
	publisher = {arXiv},
	author = {Kaye, Jason and Beck, Sophie and Barnett, Alex and Van Muñoz, Lorenzo and Parcollet, Olivier},
	month = nov,
	year = {2022},
	note = {arXiv:2211.12959 [cond-mat]},
	keywords = {Condensed Matter - Strongly Correlated Electrons, Mathematics - Numerical Analysis, Condensed Matter - Materials Science},
}
```


## Author and Copyright

AutoBZCore.jl was written by [Lorenzo Van Muñoz](https://web.mit.edu/lxvm/www/),
and is free/open-source software under the MIT license.


## Related packages
- [Integrals.jl](https://github.com/SciML/Integrals.jl)
- [FourierSeriesEvaluators.jl](https://github.com/lxvm/FourierSeriesEvaluators.jl)
- [SymmetryReduceBZ.jl](https://github.com/jerjorg/SymmetryReduceBZ.jl)
- [Brillouin.jl](https://github.com/thchr/Brillouin.jl)