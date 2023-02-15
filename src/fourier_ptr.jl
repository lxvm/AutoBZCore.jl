"""
    fourier_ptr!(rule, npt, f::AbstractFourierSeries, ::Val{d}, ::Type{T}, syms) where {d,T}

Modifies and returns the NamedTuple `rule` with fields `x,w` to contain the
Fourier series evaluated at the symmetrized PTR grid points
"""
# no symmetries
@generated function fourier_ptr!(rule, npt, f::AbstractFourierSeries{N}, ::Val{N}, ::Type{T}, ::Nothing) where {N,T}
    f_N = Symbol(:f_, N)
    quote
        $f_N = f
        resize!(rule.x, npt^N)
        box = period(f)
        n = 0
        Base.Cartesian.@nloops $N i _ -> Base.OneTo(npt) (d -> d==1 ? nothing : f_{d-1} = contract(f_d, box[d]*(i_d-1)/npt, Val(d))) begin
            n += 1
            rule.x[n] = f_1(box[1]*(i_1-1)/npt)
        end
        rule
    end
end

# general symmetries
@generated function fourier_ptr!(rule, npt, f::AbstractFourierSeries{N}, ::Val{N}, ::Type{T}, syms) where {N,T}
    f_N = Symbol(:f_, N)
    quote
        $f_N = f
        flag, wsym, nsym = ptr_(Val(N), npt, syms)
        n = 0
        box = period(f)
        resize!(rule.x, nsym)
        resize!(rule.w, nsym)
        Base.Cartesian.@nloops $N i flag (d -> d==1 ? nothing : f_{d-1} = contract(f_d, box[d]*(i_d-1)/npt, Val(d))) begin
            (Base.Cartesian.@nref $N flag i) || continue
            n += 1
            rule.x[n] = f_1(box[1]*(i_1-1)/npt)
            rule.w[n] = wsym[n]
            n >= nsym && break
        end
        rule
    end
end