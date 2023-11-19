using WannierIO
# use artefacts to provide an input wout file
using AutoBZCore
fbz = load_bz(FBZ(), "svo.wout")
using SymmetryReduceBZ
ibz = load_bz(IBZ(), "svo.wout")
