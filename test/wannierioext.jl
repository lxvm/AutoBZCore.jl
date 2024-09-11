using Test
using WannierIO
using AutoBZCore
using SymmetryReduceBZ
using LinearAlgebra: det

# TODO use artefacts to provide an input wout file
path = joinpath(dirname(dirname(pathof(AutoBZCore))), "aps_example/svo.wout")
fbz = load_bz(FBZ(), path)
ibz = load_bz(IBZ(), path)
@test det(fbz.B) ≈ det(ibz.B)
fprob = AutoBZCore.AutoBZProblem((x,p) -> 1.0, fbz)
iprob = AutoBZCore.AutoBZProblem(TrivialRep(), IntegralFunction((x,p) -> 1.0), ibz)
for alg in (IAI(), PTR())
    @test det(fbz.B) ≈ AutoBZCore.solve(fprob, alg).value
    @test det(ibz.B) ≈ AutoBZCore.solve(iprob, alg).value
end
