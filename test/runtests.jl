using Test

include("utils.jl")
@testset "aqua" include("aqua.jl")
@testset "interface" include("interface_tests.jl")
@testset "brillouin" include("brillouin.jl")
@testset "fourier" include("fourier.jl")
@testset "SymmetryReduceBZExt" include("test_ibz.jl")
# @testset "AtomsBaseExt" include("atomsbaseext.jl")
# @testset "WannierIOExt" include("wannierioext.jl")
@testset "DOS" include("dos.jl")
