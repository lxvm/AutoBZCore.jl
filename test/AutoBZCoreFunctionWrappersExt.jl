using Test
using AutoBZCore
using FunctionWrappers

f = IntegralFunction((x, p) -> (1 / (p - cos(x))), 0.0im)
prob1 = IntegralProblem(f, (0.0, 2pi), 1.0+im)
_solve! = function (solver, x, p)
    solver.p = p-cos(x)
    sol = solve!(solver)
    return AutoBZCore.CommonSolutionStats(sol.value, sol.stats)
end
g1 = CommonSolveIntegralFunction(_solve!, prob1, QuadGKJL(), 0.0im, DefaultSpecialize())
g2 = CommonSolveIntegralFunction(_solve!, prob1, QuadGKJL(), 0.0im, FunctionWrapperSpecialize())
prob21 = IntegralProblem(g1, (0.0, 2pi), 0.5+0.1im)
prob22 = IntegralProblem(g2, (0.0, 2pi), 0.5+0.1im)
sol1 = solve(prob21, QuadGKJL())
sol2 = solve(prob22, QuadGKJL())
@test sol1.value ≈ sol2.value