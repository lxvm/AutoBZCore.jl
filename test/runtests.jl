using Test, Aqua
using AutoBZCore

Aqua.test_all(AutoBZCore)

@testset "AutoBZCore - SciML" include("sciml.jl")
@testset "AutoBZCore - AutoBZ" include("autobz.jl")
@testset "AutoBZCore - Fourier" include("fourier.jl")
@testset "HDF5Ext" include("hdf5.jl")
@testset "SymmetryReduceBZExt" include("test_ibz.jl")
