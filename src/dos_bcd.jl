using LinearAlgebra
function init_cacheval(h, domain, p, alg::BCD)
    h isa FourierSeries || throw(ArgumentError("BCD currently supports Fourier series Hamiltonians"))
    p isa SymmetricBZ || throw(ArgumentError("BCD supports BZ parameters from load_bz"))
    d2h = HessianSeries(h)
    wd2h = FourierSeriesEvaluators.workspace_allocate(d2h, FourierSeriesEvaluators.period(d2h))
    kalg = MonkhorstPack(npt=alg.npt,syms=p.syms)
    dom = canonical_ptr_basis(p.B)
    rule = init_fourier_rule(wd2h, dom, kalg)
    if rule isa FourierPTR
        if typeof(rule.s[1][1]) <: Number
            tabeigen = eigen.(getindex.(rule.s, 1))
        else
            tabeigen = eigen.(Hermitian.(getindex.(rule.s, 1)))
        end
    elseif rule isa FourierMonkhorstPack
        if typeof(rule.wxs[1][2].s) <: Number
            tabeigen = eigen.(getindex.(getproperty.(getindex.(rule.wxs, 2),:s),1))
        else
            tabeigen = eigen.(Hermitian.(getindex.(getproperty.(getindex.(rule.wxs, 2),:s),1)))
        end
    end
    
    return rule, tabeigen
end


function dos_solve(h, domain, p, alg::BCD, cacheval;
    abstol=1e-8, reltol=nothing, maxiters=nothing)
    domain isa Number || throw(ArgumentError("BCD supports domains of individual eigenvalues"))
    E = domain
    bz = p
    rule, tabeigen = cacheval
    (; npt, α, ΔE, η) = alg
    d = ndims(bz)
    temp= rule isa FourierPTR ? rule.s[1][1] : rule.wxs[1][2].s[1] 
    J = size(temp)[1]
    result = do_BCD(rule, tabeigen, h, E, α, ΔE, η, Val(d), Val(J))
    return DOSSolution(result, Success, (;))

end

function bcd_integrand(h,k,H,d1H,d2H,evals,evecs,::Val{d},::Val{J},E,α,η,ΔE) where {d,J}
    def = SVector{length(k)}([tr(d1H[i] * exp(-((H - E * I) / ΔE)^2)) for i in eachindex(d1H)])
    ddef = SMatrix{d,d}(divided_difference_contour(zeros(ComplexF64, d, d), H,d1H,d2H , evals, evecs, Val(J), Val(d), E, ΔE))
    return -imag(tr(inv((E + im * η) * I - h(k - im * α * def))) * det(I - im * α * ddef))
end
function do_BCD(rule::FourierPTR, tabeigen, h, E, α, ΔE, η, d, J)
    DOS = 0.0
    for index in CartesianIndices(rule.p)
        H,d1H,d2H = rule.s[index]
        k = rule.p[index][2]
        DOS += bcd_integrand(h,k,H,d1H,d2H,tabeigen[index].values, tabeigen[index].vectors,d,J,E,α,η,ΔE)
    end
    return DOS / (π * length(rule.p))
end
function do_BCD(rule::FourierMonkhorstPack, tabeigen, h, E, α, ΔE, η, ::Val{d}, J) where {d}
    DOS = 0.0
    for index in CartesianIndices(rule.wxs)
        H,d1H,d2H = rule.wxs[index][2].s
        k = rule.wxs[index][2].x
        w = rule.wxs[index][1]
        DOS += w*bcd_integrand(h,k,H,d1H,d2H,tabeigen[index].values, tabeigen[index].vectors,Val(d),J,E,α,η,ΔE)
    end
    return DOS / (π * rule.npt^d)
end
function divided_difference_gaussian(x, y)
    if x == y
        return -2x * exp(-x * x)
    elseif x == -y
        return 0
    end
    return exp(-y * y) * (expm1((y - x) * (x + y)) / (x - y))
end

function divided_difference_contour(sum, h, d1h, d2h, tabε, tabψ, ::Val{J}, ::Val{d}, E, ΔE) where {d} where {J}

    @views for n in 1:J
        for m in 1:J
            for i in 1:d
                for j in 1:d
                    sum[i, j] += divided_difference_gaussian((tabε[n] - E) / ΔE, (tabε[m] - E) / ΔE) / (ΔE) * (tabψ[:, n]' * d1h[i] * tabψ[:, m]) * (tabψ[:, m]' * d1h[j] * tabψ[:, n])
                end

            end
        end
    end
    for i in 1:d
        for j in 1:d+1-i
            #sum[i, j+i-1] += dot(d2h[i][j]', exp(-((h - E * I) / ΔE)^2))
            sum[i, j+i-1] += tr(d2h[i][j] * exp(-((h - E * I) / ΔE)^2))
        end
    end
    return Hermitian(sum)
end