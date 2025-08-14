using Test
using AutoBZCore
using IteratedIntegration: AbstractIteratedLimits, TetrahedralLimits, load_limits
using IteratedIntegration: segments, fixandeliminate, eliminate, measure, interior_point
using QuadGK: quadgk
using FourierSeriesEvaluators: FourierSeries


struct AbstolLogger <: AutoBZCore.StatsAlgorithm end
AbstolLogger(alg) = AutoBZCore.SolverStats(alg, AbstolLogger())

function AutoBZCore.init_stats_cacheval(::AbstolLogger, args...)
    Vector{Any}(undef, 0)
end
function AutoBZCore.stats_step!(cache, ::AbstolLogger, batch, args...)
    x = args[begin]
    out = args[end]
    if out isa AutoBZCore.CommonSolutionStats
        haskey(out.stats, :abstol) && push!(cache, (x, out.stats.abstol...))
    end
    return
end
function AutoBZCore.stats_reset(::AbstolLogger, channel)
    for c in channel.data
        empty!(c)
    end
end
function AutoBZCore.stats_summary(::AbstolLogger, channel, sol)
    abstollog = vec(stack(channel.data))
    return (; sol.stats..., abstol=(sol.stats.abstol, abstollog))
end


record_error_abstol(; error, abstol, stats...) = (; error, abstol)

dims = 3
for f in [IntegralFunction((x, p) -> 1.0, 1.0), FourierIntegralFunction((x, s, p) -> 1.0, FourierSeries(integer_lattice(dims); period=2pi), 1.0)]
quad = AbstolLogger(QuadGKJL(order=3, stats=record_error_abstol))

abstol=1e-5
for (dom, dom2) in [
    (AutoBZCore.HyperCube(fill(0.0, dims), 2 .* (1:dims)), AutoBZCore.HyperCube(fill(0.0, dims), 3 .* (1:dims))),
    (TetrahedralLimits((2.0,3.0,5.0)), TetrahedralLimits((6.0,9.0,15.0))),
    ]
    prob = IntegralProblem(f, dom)
    let tolalg = AutoBZCore.StaticTolAlg(AutoBZCore.project)
        alg = NestedQuad(quad; tolalg)
        solver = init(prob, alg; abstol)
        sol = solve!(solver)
        lims = dom isa AbstractIteratedLimits ? dom : load_limits(dom)
        m1 = measure(quadgk, eliminate(lims, 1:dims-1))
        m2 = measure(quadgk, eliminate(lims, 1:dims-2))
        tol, nexttol = sol.stats.abstol
        @test tol ≈ abstol
        for (z, _tol, _nexttol) in nexttol
            @test _tol ≈ abstol/m1
            for (y, __tol, __nexttol) in _nexttol
                @test __tol ≈ abstol/m2
            end
        end
        # checks that abstol get updated in all dimensions
        abstol2 = 7*abstol
        solver.kwargs = (; solver.kwargs..., abstol=abstol2)
        sol = solve!(solver)
        tol, nexttol = sol.stats.abstol
        @test tol ≈ abstol2
        for (z, _tol, _nexttol) in nexttol
            @test _tol ≈ abstol2/m1
            for (y, __tol, __nexttol) in _nexttol
                @test __tol ≈ abstol2/m2
            end
        end
        # checks that changes in domain are not reflected in tolerance and only original is used
        solver.dom = dom2
        sol = solve!(solver)
        tol, nexttol = sol.stats.abstol
        @test tol ≈ abstol2
        for (z, _tol, _nexttol) in nexttol
            @test _tol ≈ abstol2/m1
            for (y, __tol, __nexttol) in _nexttol
                @test __tol ≈ abstol2/m2
            end
        end
    end
    let tolalg = AutoBZCore.StaticTolAlg(AutoBZCore.bbox)
        alg = NestedQuad(quad; tolalg)
        solver = init(prob, alg; abstol)
        sol = solve!(solver)
        tol, nexttol = sol.stats.abstol
        @test tol ≈ abstol
        lims = dom isa AbstractIteratedLimits ? dom : load_limits(dom)
        m1 = (s = segments(lims, dims); abs(s[2]-s[1]))
        m2 = (s = segments(lims, dims-1); abs(s[2]-s[1]))
        for (z, _tol, _nexttol) in nexttol
            @test _tol ≈ abstol/m1
            for (y, __tol, __nexttol) in _nexttol
                @test __tol ≈ abstol/m1/m2
            end
        end
        # checks that abstol get updated in all dimensions
        abstol2 = 7*abstol
        solver.kwargs = (; solver.kwargs..., abstol=abstol2)
        sol = solve!(solver)
        tol, nexttol = sol.stats.abstol
        @test tol ≈ abstol2
        for (z, _tol, _nexttol) in nexttol
            @test _tol ≈ abstol2/m1
            for (y, __tol, __nexttol) in _nexttol
                @test __tol ≈ abstol2/m1/m2
            end
        end
        # checks that changes in domain are not reflected in tolerance and only original is used
        solver.dom = dom2
        sol = solve!(solver)
        tol, nexttol = sol.stats.abstol
        @test tol ≈ abstol2
        for (z, _tol, _nexttol) in nexttol
            @test _tol ≈ abstol2/m1
            for (y, __tol, __nexttol) in _nexttol
                @test __tol ≈ abstol2/m1/m2
            end
        end
    end
    let tolalg = AutoBZCore.AdaptiveTolAlg()
        alg = NestedQuad(quad; tolalg)
        solver = init(prob, alg; abstol)
        sol = solve!(solver)
        tol, nexttol = sol.stats.abstol
        @test tol ≈ abstol
        lims = dom isa AbstractIteratedLimits ? dom : load_limits(dom)
        m1 = (s = segments(lims, dims); abs(s[2]-s[1]))
        for (z, _tol, _nexttol) in nexttol
            _lims = fixandeliminate(lims, z, Val(dims))
            m2 = (s = segments(_lims, dims-1); abs(s[2]-s[1]))
            @test _tol ≈ abstol/m1
            for (y, __tol, __nexttol) in _nexttol
                @test __tol ≈ abstol/m1/m2
            end
        end
        # checks that abstol get updated in all dimensions
        abstol2 = 7*abstol
        solver.kwargs = (; solver.kwargs..., abstol=abstol2)
        sol = solve!(solver)
        tol, nexttol = sol.stats.abstol
        @test tol ≈ abstol2
        lims = dom isa AbstractIteratedLimits ? dom : load_limits(dom)
        m1 = (s = segments(lims, dims); abs(s[2]-s[1]))
        for (z, _tol, _nexttol) in nexttol
            _lims = fixandeliminate(lims, z, Val(dims))
            m2 = (s = segments(_lims, dims-1); abs(s[2]-s[1]))
            @test _tol ≈ abstol2/m1
            for (y, __tol, __nexttol) in _nexttol
                @test __tol ≈ abstol2/m1/m2
            end
        end
        # checks that changes in domain are reflected in tolerance
        solver.dom = dom2
        sol = solve!(solver)
        tol, nexttol = sol.stats.abstol
        @test tol ≈ abstol2
        lims = dom2 isa AbstractIteratedLimits ? dom2 : load_limits(dom2)
        m1 = (s = segments(lims, dims); abs(s[2]-s[1]))
        for (z, _tol, _nexttol) in nexttol
            _lims = fixandeliminate(lims, z, Val(dims))
            m2 = (s = segments(_lims, dims-1); abs(s[2]-s[1]))
            @test _tol ≈ abstol2/m1
            for (y, __tol, __nexttol) in _nexttol
                @test __tol ≈ abstol2/m1/m2
            end
        end
    end
    let tolalg = AutoBZCore.v03TolAlg()
        alg = NestedQuad(quad; tolalg)
        solver = init(prob, alg; abstol)
        sol = solve!(solver)
        tol, nexttol = sol.stats.abstol
        @test tol ≈ abstol
        lims = dom isa AbstractIteratedLimits ? dom : load_limits(dom)
        for (z, _tol, _nexttol) in nexttol
            _lims = fixandeliminate(lims, z, Val(dims))
            m1 = (s = segments(_lims, dims-1); abs(s[2]-s[1]))
            @test _tol ≈ abstol/m1
            for (y, __tol, __nexttol) in _nexttol
                __lims = fixandeliminate(_lims, y, Val(dims-1))
                m2 = (s = segments(__lims, dims-2); abs(s[2]-s[1]))
                @test __tol ≈ abstol/m1/m2
            end
        end
        # checks that abstol get updated in all dimensions
        abstol2 = 7*abstol
        solver.kwargs = (; solver.kwargs..., abstol=abstol2)
        sol = solve!(solver)
        tol, nexttol = sol.stats.abstol
        @test tol ≈ abstol2
        lims = dom isa AbstractIteratedLimits ? dom : load_limits(dom)
        for (z, _tol, _nexttol) in nexttol
            _lims = fixandeliminate(lims, z, Val(dims))
            m1 = (s = segments(_lims, dims-1); abs(s[2]-s[1]))
            @test _tol ≈ abstol2/m1
            for (y, __tol, __nexttol) in _nexttol
                __lims = fixandeliminate(_lims, y, Val(dims-1))
                m2 = (s = segments(__lims, dims-2); abs(s[2]-s[1]))
                @test __tol ≈ abstol2/m1/m2
            end
        end
        # checks that changes in domain are reflected in tolerance
        solver.dom = dom2
        sol = solve!(solver)
        tol, nexttol = sol.stats.abstol
        @test tol ≈ abstol2
        lims = dom2 isa AbstractIteratedLimits ? dom2 : load_limits(dom2)
        for (z, _tol, _nexttol) in nexttol
            _lims = fixandeliminate(lims, z, Val(dims))
            m1 = (s = segments(_lims, dims-1); abs(s[2]-s[1]))
            @test _tol ≈ abstol2/m1
            for (y, __tol, __nexttol) in _nexttol
                __lims = fixandeliminate(_lims, y, Val(dims-1))
                m2 = (s = segments(__lims, dims-2); abs(s[2]-s[1]))
                @test __tol ≈ abstol2/m1/m2
            end
        end
    end
    let tolalg = AutoBZCore.v04TolAlg()
        alg = NestedQuad(quad; tolalg)
        solver = init(prob, alg; abstol)
        sol = solve!(solver)
        tol, nexttol = sol.stats.abstol
        @test tol ≈ abstol
        lims = dom isa AbstractIteratedLimits ? dom : load_limits(dom)
        s1 = segments(lims, dims)
        m1 = abs(s1[end]-s1[begin])
        z = (s1[end]+s1[begin])/2
        _lims = fixandeliminate(lims, z, Val(dims))
        s2 = segments(_lims, dims-1)
        m2 = abs(s2[2]-s2[1])
        for (z, _tol, _nexttol) in nexttol
            @test _tol ≈ abstol/m1
            for (y, __tol, __nexttol) in _nexttol
                @test __tol ≈ abstol/m1/m2
            end
        end
        # checks that abstol get updated in only outermost dimension
        abstol2 = 7*abstol
        solver.kwargs = (; solver.kwargs..., abstol=abstol2)
        sol = solve!(solver)
        tol, nexttol = sol.stats.abstol
        @test tol ≈ abstol2
        for (z, _tol, _nexttol) in nexttol
            @test _tol ≈ abstol/m1
            for (y, __tol, __nexttol) in _nexttol
                @test __tol ≈ abstol/m1/m2
            end
        end
        # checks that changes in domain are not reflected in tolerance and only original is used
        solver.dom = dom2
        sol = solve!(solver)
        tol, nexttol = sol.stats.abstol
        @test tol ≈ abstol2
        for (z, _tol, _nexttol) in nexttol
            @test _tol ≈ abstol/m1
            for (y, __tol, __nexttol) in _nexttol
                @test __tol ≈ abstol/m1/m2
            end
        end
    end
end
end