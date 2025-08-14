using Test
using Unitful
using UnitfulAtomic
using AtomsBase
using AutoBZCore
using SymmetryReduceBZ
using LinearAlgebra: det

# do the example of getting the volume of the bz of silicon
bounding_box = 10.26 / 2 * [[0, 0, 1], [1, 0, 1], [1, 1, 0]]u"bohr"
silicon = periodic_system([:Si =>  ones(3)/8,
                        :Si => -ones(3)/8],
                        bounding_box, fractional=true)
A = reinterpret(reshape,eltype(eltype(bounding_box)),AtomsBase.bounding_box(silicon))
recip_vol = det(AutoBZCore.canonical_reciprocal_basis(A))
fbz = load_bz(FBZ(), silicon)
fprob = AutoBZCore.AutoBZProblem((x,p) -> 1.0, fbz)
ibz = load_bz(IBZ(), silicon)
iprob = AutoBZCore.AutoBZProblem(TrivialRep(), IntegralFunction((x,p) -> 1.0), ibz)
for alg in (IAI(), PTR())
    @test recip_vol ≈ AutoBZCore.solve(fprob, alg).value
    @test recip_vol ≈ AutoBZCore.solve(iprob, alg).value
end
