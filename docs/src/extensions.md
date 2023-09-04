# Package extensions

## SymmetryReduceBZ.jl

Loading [SymmetryReduceBZ.jl](https://github.com/jerjorg/SymmetryReduceBZ.jl)
provides a specialized method of [`load_bz`](@ref) that when provided atom
species and positions can compute the [`IBZ`](@ref).

## HDF5.jl

Loading [HDF5.jl](https://github.com/JuliaIO/HDF5.jl) provides a specialized
method of [`batchsolve`](@ref) that accepts an H5 archive or group in the first
argument and writes the integral results and timings to a dataset.

## WannierIO.jl

Loading [WannierIO.jl](https://github.com/qiaojunfeng/WannierIO.jl) provides a
specialized method of [`load_bz`](@ref) that loads the BZ defined in a
`seedname.wout` file.

## AtomsBase.jl

Loading [AtomsBase.jl](https://github.com/qiaojunfeng/WannierIO.jl) provides a
specialized method of [`load_bz`](@ref) to load the BZ of an `AtomsBase.AbstractSystem`