using Unitful
using UnitfulAtomic
using AtomsBase
# do the example of getting the volume of the bz of silicon
bounding_box = 10.26 / 2 * [[0, 0, 1], [1, 0, 1], [1, 1, 0]]u"bohr"
silicon = periodic_system([:Si =>  ones(3)/8,
                        :Si => -ones(3)/8],
                        bounding_box, fractional=true)
A = reinterpret(reshape,eltype(eltype(bounding_box)),AtomsBase.bounding_box(silicon))
using AutoBZCore
recip_vol = det(AutoBZCore.canonical_reciprocal_basis(A))
fbz = load_bz(FBZ(), silicon)
fprob = AutoBZCore.IntegralProblem((x,p) -> 1.0, fbz)
using SymmetryReduceBZ
ibz = load_bz(IBZ(), silicon)
iprob = AutoBZCore.IntegralProblem((x,p) -> 1.0, ibz)
for alg in (IAI(), PTR(), TAI())
    @test recip_vol ≈ AutoBZCore.solve(fprob, alg).u
    @test recip_vol ≈ AutoBZCore.solve(iprob, alg).u
end
