using Test
using WannierIO
using AutoBZCore
using SymmetryReduceBZ

# TODO use artefacts to provide an input wout file
fbz = load_bz(FBZ(), "svo.wout")
ibz = load_bz(IBZ(), "svo.wout")
