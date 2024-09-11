# AutoBZCore.jl

| Documentation | Build Status | Coverage | Version |
| :-: | :-: | :-: | :-: |
| [![][docs-stable-img]][docs-stable-url] | [![][action-img]][action-url] | [![][codecov-img]][codecov-url] | [![ver-img]][ver-url] |
| [![][docs-dev-img]][docs-dev-url] | [![][pkgeval-img]][pkgeval-url] | [![][aqua-img]][aqua-url] | [![deps-img]][deps-url] |

This package provides a common interface to integration algorithms that are
efficient and high-order accurate for computational tasks including
Brillouin-zone integration and Wannier interpolation. For further information on
integrand interfaces, including optimizations for Wannier interpolation, please see [the
documentation](https://lxvm.github.io/AutoBZCore.jl/stable/).

## Research and citation

If you use AutoBZCore.jl in your software or published research works, please
cite one, or, all of the following. Citations help to encourage the development
and maintenance of open-source scientific software.
- This repository: https://github.com/lxvm/AutoBZCore.jl
- Our paper on BZ integration: [Automatic, high-order, and adaptive algorithms
  for Brillouin zone integration. Jason Kaye, Sophie Beck, Alex Barnett, Lorenzo
  Van Muñoz, Olivier Parcollet. SciPost Phys. 15, 062
  (2023)](https://scipost.org/SciPostPhys.15.2.062)


## Author and Copyright

AutoBZCore.jl was written by [Lorenzo Van Muñoz](https://web.mit.edu/lxvm/www/),
and is free/open-source software under the MIT license.


## Related packages
- [AutoBZ.jl](https://github.com/lxvm/AutoBZ.jl)
- [`wannier-berri`](https://github.com/wannier-berri/wannier-berri)
- [FourierSeriesEvaluators.jl](https://github.com/lxvm/FourierSeriesEvaluators.jl)
- [SymmetryReduceBZ.jl](https://github.com/jerjorg/SymmetryReduceBZ.jl)
- [AtomsBase.jl](https://github.com/qiaojunfeng/WannierIO.jl)
- [HDF5.jl](https://github.com/JuliaIO/HDF5.jl)
- [WannierIO.jl](https://github.com/qiaojunfeng/WannierIO.jl)
- [Integrals.jl](https://github.com/SciML/Integrals.jl)
- [Brillouin.jl](https://github.com/thchr/Brillouin.jl)
- [TightBinding.jl](https://github.com/cometscome/TightBinding.jl)

<!-- badges -->

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://lxvm.github.io/AutoBZCore.jl/stable/

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://lxvm.github.io/AutoBZCore.jl/dev/

[action-img]: https://github.com/lxvm/AutoBZCore.jl/actions/workflows/CI.yml/badge.svg?branch=main
[action-url]: https://github.com/lxvm/AutoBZCore.jl/actions/?query=workflow:CI

[pkgeval-img]: https://juliahub.com/docs/General/AutoBZCore/stable/pkgeval.svg
[pkgeval-url]: https://juliahub.com/ui/Packages/General/AutoBZCore

[codecov-img]: https://codecov.io/github/lxvm/AutoBZCore.jl/branch/main/graph/badge.svg
[codecov-url]: https://app.codecov.io/github/lxvm/AutoBZCore.jl

[aqua-img]: https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg
[aqua-url]: https://github.com/JuliaTesting/Aqua.jl

[ver-img]: https://juliahub.com/docs/AutoBZCore/version.svg
[ver-url]: https://juliahub.com/ui/Packages/AutoBZCore/UDEDl

[deps-img]: https://juliahub.com/docs/General/AutoBZCore/stable/deps.svg
[deps-url]: https://juliahub.com/ui/Packages/General/AutoBZCore?t=2
