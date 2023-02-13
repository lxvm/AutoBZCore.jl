"""
    fourier_ptr!(rule::Vector, f::AbstracFourierSeries, fbz, npt)

Returns a vector `rule` containing tuples `(f(x), w)` where `x, w` are the nodes
and weights of the symmetrized PTR quadrature rule.
"""
@generated function fourier_ptr!(rule, f::AbstractFourierSeries{N}, bz::FullBZ{<:Any,N}, npt) where N
    f_N = Symbol(:f_, N)
    quote
        $f_N = f
        resize!(rule, npt^N)
        box = boundingbox(bz)
        dvol = equispace_dvol(bz, npt)
        Base.Cartesian.@nloops $N i _ -> Base.OneTo(npt) (d -> d==1 ? nothing : f_{d-1} = contract(f_d, (box[d][2]-box[d][1])*(i_d-1)/npt + box[d][1], Val(d))) begin
            rule[Base.Cartesian.@ncall $N equispace_index npt d -> i_d] = (f_1((box[1][2]-box[1][1])*(i_1-1)/npt + box[1][1]), dvol)
        end
        rule
    end
end

@generated function fourier_ptr!(rule, f::AbstractFourierSeries{N}, bz::AbstractBZ{N}, npt) where N
    f_N = Symbol(:f_, N)
    quote
        $f_N = f
        flag, wsym, nsym = equispace_ptr(bz, npt)
        n = 0
        box = boundingbox(bz)
        dvol = equispace_dvol(bz, npt)
        resize!(rule, nsym)
        Base.Cartesian.@nloops $N i flag (d -> d==1 ? nothing : f_{d-1} = contract(f_d, (box[d][2]-box[d][1])*(i_d-1)/npt + box[d][1], Val(d))) begin
            (Base.Cartesian.@nref $N flag i) || continue
            n += 1
            rule[n] = (f_1((box[1][2]-box[1][1])*(i_1-1)/npt + box[1][1]), dvol*wsym[n])
            n >= nsym && break
        end
        rule
    end
end