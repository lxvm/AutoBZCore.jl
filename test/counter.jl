using Test
using AutoBZCore
using AutoBZCore: IntegralProblem, solve

# important that this integrand require no subdivision for correct counts
foop = (x, p) -> sum(y -> sin(y*p), x)
p = 2.0

alg = AuxQuadGKJL(order=7)
evalsperseg = 2*alg.order+1
# TODO: check also against a cubature
# cub = HCubatureJL()
# cubevals = 7

# test counting for ordinary integrands
for dim in 1:3
    prob = IntegralProblem(foop, CubicLimits(zeros(dim), ones(dim)), p)
    @test solve(prob, NestedQuad(alg)).numevals == -1   # default: no counting
    @test solve(prob, EvalCounter(NestedQuad(alg))).numevals == evalsperseg^dim
    @test solve(prob, NestedQuad(EvalCounter(alg))).numevals == evalsperseg^dim
end

# test counting for nested integrands without inner counters, which do not have necessary
# info for counting all evals
ncfoop = NestedIntegrand() do x, p
    f_ = (y, p) -> foop((x..., y...), p)
    prob = IntegralProblem(f_, 0.0, 1.0, p)
    solve(prob, alg)
end

for dim in 1:3
    prob = IntegralProblem(ncfoop, CubicLimits(zeros(dim), ones(dim)), p)
    @test solve(prob, NestedQuad(alg)).numevals == -1   # default: no counting
    @test solve(prob, EvalCounter(NestedQuad(alg))).numevals == -1 # no counting without inner counter
    @test solve(prob, NestedQuad(EvalCounter(alg))).numevals == evalsperseg^dim # only counts outer dim
end

# test counting for nested integrands with inner counters
wcfoop = NestedIntegrand() do x, p
    f_ = (y, p) -> foop((x..., y...), p)
    prob = IntegralProblem(f_, 0.0, 1.0, p)
    solve(prob, EvalCounter(alg))
end

for dim in 1:3
    prob = IntegralProblem(wcfoop, CubicLimits(zeros(dim), ones(dim)), p)
    @test solve(prob, NestedQuad(alg)).numevals == -1   # default: no counting
    @test solve(prob, EvalCounter(NestedQuad(alg))).numevals == evalsperseg^(dim+1) # counts all
    @test solve(prob, NestedQuad(EvalCounter(alg))).numevals == evalsperseg^dim # counts outer
end

# nest singly nested batch integrands (results should be same as before)

## without inner counters
for dim in 1:3
    g = NestedBatchIntegrand((ncfoop,), Float64[], Float64[])
    prob = IntegralProblem(g, CubicLimits(zeros(dim), ones(dim)), p)
    @test solve(prob, NestedQuad(alg)).numevals == -1   # default: no counting
    @test solve(prob, EvalCounter(NestedQuad(alg))).numevals == -1 # no counting without inner counter
    @test solve(prob, NestedQuad(EvalCounter(alg))).numevals == evalsperseg^dim # only counts outer dim
end

## with inner counters
for dim in 1:3
    g = NestedBatchIntegrand((wcfoop,), Float64[], Float64[])
    prob = IntegralProblem(g, CubicLimits(zeros(dim), ones(dim)), p)
    @test solve(prob, NestedQuad(alg)).numevals == -1   # default: no counting
    @test solve(prob, EvalCounter(NestedQuad(alg))).numevals == evalsperseg^(dim+1) # counts all
    @test solve(prob, NestedQuad(EvalCounter(alg))).numevals == evalsperseg^dim # counts outer
end

# nest multiply nested batch integrands (results should be same as before)

## without inner counters
for dim in 1:3
    g = NestedBatchIntegrand((NestedBatchIntegrand((ncfoop,), Float64[], Float64[]),), Float64[], Float64[])
    prob = IntegralProblem(g, CubicLimits(zeros(dim), ones(dim)), p)
    @test solve(prob, NestedQuad(alg)).numevals == -1   # default: no counting
    @test solve(prob, EvalCounter(NestedQuad(alg))).numevals == -1 # no counting without inner counter
    @test solve(prob, NestedQuad(EvalCounter(alg))).numevals == evalsperseg^dim # only counts outer dim
end

## with inner counters
for dim in 1:3
    g = NestedBatchIntegrand((NestedBatchIntegrand((wcfoop,), Float64[], Float64[]),), Float64[], Float64[])
    prob = IntegralProblem(g, CubicLimits(zeros(dim), ones(dim)), p)
    @test solve(prob, NestedQuad(alg)).numevals == -1   # default: no counting
    @test solve(prob, EvalCounter(NestedQuad(alg))).numevals == evalsperseg^(dim+1) # counts all
    @test solve(prob, NestedQuad(EvalCounter(alg))).numevals == evalsperseg^dim # counts outer
end

#TODO: test the above with FourierIntegrands
