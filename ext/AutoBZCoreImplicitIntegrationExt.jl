module AutoBZCoreImplicitIntegrationExt

using AutoBZCore
using FourierSeriesEvaluators
using ImplicitIntegration
using LinearAlgebra
using StaticArrays

function AutoBZCore.init_cacheval(h, domain, p, alg::ImplicitIntegrationJL)
    h isa FourierSeries || throw(ArgumentError("GGR currently supports Fourier series Hamiltonians"))
    p isa SymmetricBZ || throw(ArgumentError("GGR supports BZ parameters from load_bz"))
    j = JacobianSeries(h)
    wj = FourierSeriesEvaluators.workspace_allocate(j, FourierSeriesEvaluators.period(j))
    return wj
end


function AutoBZCore.dos_solve(h, domain, p, alg::ImplicitIntegrationJL, cacheval;
    abstol=1e-8, reltol=nothing, maxiters=nothing)
    domain isa Number || throw(ArgumentError("GGR supports domains of individual eigenvalues"))
    p isa SymmetricBZ || throw(ArgumentError("GGR supports BZ parameters from load_bz"))
    E = domain
    bz = p
    j = cacheval
    ϕ = k -> begin
        H = h(k)-E*I
        real(StaticArrays._det(StaticArrays.Size(H), H))
    end
    f = k -> begin
        hk, vk = j(k)
        values, U = eigen(Hermitian(hk))
        resid, n = findmin(x -> abs(x-E), values)
        u = U[:,n]
        return inv(sqrt(mapreduce(v -> abs2(dot(u,v,u)/u'u), +, vk))) # compute 1/|∇ϵₙ|
    end
    result = integrate(f, ϕ, map(zero, FourierSeriesEvaluators.period(h)), FourierSeriesEvaluators.period(h); surface=true, tol=abstol)
    return AutoBZCore.DOSSolution(result.val, AutoBZCore.Success, (; loginfo=result.logger))
end


end
