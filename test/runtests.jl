using Test

include("utils.jl")
@testset "aqua" include("aqua.jl")
@testset "interface" include("interface_tests.jl")
@testset "brillouin" include("brillouin.jl")
@testset "fourier" include("fourier.jl")
@testset "HDF5Ext" include("hdf5ext.jl")
@testset "SymmetryReduceBZExt" include("test_ibz.jl")
@testset "AtomsBaseExt" include("atomsbaseext.jl")
@testset "WannierIOExt" include("wannierioext.jl")
@testset "AutoBZCore - DOS" include("dos.jl")
