using LinearAlgebra
using CircularArrays
function init_cacheval(h, domain, p, alg::LT)
    h isa FourierSeries || throw(ArgumentError("LT currently supports Fourier series Hamiltonians"))
    p isa SymmetricBZ || throw(ArgumentError("LT supports BZ parameters from load_bz"))
    kalg = MonkhorstPack(npt = alg.npt, syms = p.syms)
    dom = canonical_ptr_basis(p.B)
    rule = init_fourier_rule(h, dom, kalg, SerialExecutor())
    if rule isa FourierPTR
        tabeigen = CircularArray(real.(getfield.(eigen.(rule.s), :values)))
    elseif rule isa FourierMonkhorstPack
        throw(ArgumentError("LT does not support symmetrized BZ yet."))
    end
    return rule, tabeigen
end


function dos_solve(h, domain, p, alg::LT, cacheval;
    abstol = 1e-8, reltol = nothing, maxiters = nothing)
    domain isa Number || throw(ArgumentError("LT supports domains of individual eigenvalues"))
    E = domain
    bz = p
    rule, tabeigen = cacheval
    temp = rule isa FourierPTR ? rule.s[1] : rule.wxs[1][2].s
    J = Int(sqrt(length(temp)))
    d = ndims(bz)
    result = parse_LT(d, J, rule, tabeigen, E)
    return DOSSolution(result, Success, (;))

end

function parse_LT(d, J, rule, tabeigen, E)
    if d == 1
        result = LT1D(rule, tabeigen, J, E)
    elseif d == 2
        result = LT2D(rule, tabeigen, J, E)
    elseif d == 3
        result = LT3D(rule, tabeigen, J, E)
    else
        result = ErrorException("The Linear Tetrahedron method is not defined in that case")
    end
    return result
end

function LT1D((ε1, ε2), E::Real)
    if (ε1 <= E <= ε2)
        return 1 / abs(ε1 - ε2)
    end
    return 0
end

function LT1D(rule::FourierPTR, tabeigen, J, E::Real)
    dos = 0
    for index in eachindex(rule.p)
        for j in 1:J
            dos += LT1D(Tuple(sort(reduce(vcat, getindex.(tabeigen[index:index+1], j)))), E)
        end
    end
    return dos / length(rule.p)
end




#Formula for the eigenvalues sorted in 2D 
function LT2D((ε1, ε2, ε3), E)
    ε2 <= E < ε3 ? (ε3 - E) / ((ε3 - ε2) * (ε3 - ε1)) :
    ε1 <= E < ε2 ? (E - ε1) / ((ε2 - ε1) * (ε3 - ε1)) : 0

end

#In 2D the squares are split into two triangles. We iterate over each point of the mesh, then we consider the associated lower-right square, that is split into two triangles.
#We add the contribution of both triangles to the DOS.  
function LT2D(rule::FourierPTR, tabeigen, J, E::Real)
    dos = 0

    @views for index in CartesianIndices(rule.p)
        for j in 1:J
            dos += LT2D(Tuple(sort([tabeigen[index][j], tabeigen[index+CartesianIndex((1, 0))][j], tabeigen[index+CartesianIndex((0, 1))][j]])), E)
            dos += LT2D(Tuple(sort([tabeigen[index+CartesianIndex((1, 0))][j], tabeigen[index+CartesianIndex((0, 1))][j], tabeigen[index+CartesianIndex((1, 1))][j]])), E)
        end

    end
    return dos / (length(rule.p))
end

#Formula for the eigenvalues sorted in 3D 
function LT3D((ε1, ε2, ε3, ε4), E)
    ε3 <= E < ε4 ? 3 * (ε4 - E)^2 / ((ε4 - ε1) * (ε4 - ε2) * (ε4 - ε3)) :
    ε2 <= E < ε3 ? 1 / ((ε3 - ε1) * (ε4 - ε1)) * (3(ε2 - ε1) - 6(ε2 - E) - 3 * (ε2 - E)^2 * (ε3 - ε1 + ε4 - ε2) / ((ε3 - ε2) * (ε4 - ε2))) :
    ε1 <= E < ε2 ? 3 * (E - ε1)^2 / ((ε2 - ε1) * (ε3 - ε1) * (ε4 - ε1)) : 0
end

#In 3D the cubes are split into six tetrahedra. We iterate over each point of the mesh, then we consider the associated lower-right cube, that is split into six tetrahedra.
#We add the contribution of all tetrahedras to the DOS.  
function LT3D(rule::FourierPTR, tabeigen, J, E::Real)
    dos = 0

    @views for index in CartesianIndices(rule.p)
        for j in 1:J
            dos += LT3D(sort([tabeigen[index][j], tabeigen[index+CartesianIndex((1, 0, 0))][j], tabeigen[index+CartesianIndex((0, 1, 0))][j], tabeigen[index+CartesianIndex((0, 1, 1))][j]]), E)

            dos += LT3D(sort([tabeigen[index][j], tabeigen[index+CartesianIndex((1, 0, 0))][j], tabeigen[index+CartesianIndex((0, 0, 1))][j], tabeigen[index+CartesianIndex((0, 1, 1))][j]]), E)

            dos += LT3D(sort([tabeigen[index+CartesianIndex((0, 1, 0))][j], tabeigen[index+CartesianIndex((1, 0, 0))][j], tabeigen[index+CartesianIndex((1, 1, 0))][j], tabeigen[index+CartesianIndex((0, 1, 1))][j]]), E)

            dos += LT3D(sort([tabeigen[index+CartesianIndex((0, 0, 1))][j], tabeigen[index+CartesianIndex((1, 0, 0))][j], tabeigen[index+CartesianIndex((1, 0, 1))][j], tabeigen[index+CartesianIndex((0, 1, 1))][j]]), E)

            dos += LT3D(sort([tabeigen[index+CartesianIndex((1, 1, 0))][j], tabeigen[index+CartesianIndex((1, 0, 0))][j], tabeigen[index+CartesianIndex((1, 1, 1))][j], tabeigen[index+CartesianIndex((0, 1, 1))][j]]), E)

            dos += LT3D(sort([tabeigen[index+CartesianIndex((1, 0, 1))][j], tabeigen[index+CartesianIndex((1, 0, 0))][j], tabeigen[index+CartesianIndex((1, 1, 1))][j], tabeigen[index+CartesianIndex((0, 1, 1))][j]]), E)
        end

    end
    return dos / (6 * length(rule.p))
end

