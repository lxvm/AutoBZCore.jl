function init_cacheval(h, domain, p, alg::GGR)
    h isa FourierSeries || throw(ArgumentError("GGR currently supports Fourier series Hamiltonians"))
    p isa SymmetricBZ || throw(ArgumentError("GGR supports BZ parameters from load_bz"))
    bz = p

    j = JacobianSeries(h)
    w = workspace_allocate(j, period(j))
    kalg = MonkhorstPack(npt=alg.npt, syms=bz.syms)
    dom = canonical_ptr_basis(bz.B)
    rule = init_fourier_rule(w, dom, kalg)
    return get_ggr_data(rule, period(j))
end

function get_ggr_data(rule, t)
    next = iterate(rule)
    isnothing(next) && throw(ArgumentError("GGR - no data in rule"))
    (w0, x0), state = next
    h0, V0 = x0.s
    e0, U0 = eigen(Hermitian(h0))
    v0 = map(*, map(real∘diag, map(v -> U0'*v*U0, V0)), t)  # multiply by period to get standardized velocities
    n = 1
    energies = Vector{typeof(e0)}(undef, length(rule))
    velocities = Vector{typeof(v0)}(undef, length(rule))
    weights = Vector{typeof(w0)}(undef, length(rule))
    energies[n] = e0
    velocities[n] = v0
    weights[n] = w0

    n += 1
    next = iterate(rule, state)
    while !isnothing(next)
        (w, x), state = next
        h, V = x.s
        e, U = eigen(Hermitian(h))
        v = map(*, map(real∘diag, map(v -> U'*v*U, V)), t)
        energies[n] = e
        velocities[n] = v
        weights[n] = w

        n += 1
        next = iterate(rule, state)
    end
    return weights, energies, velocities
end

function dos_solve(h, domain, p, alg::GGR, cacheval;
    abstol=nothing, reltol=nothing, maxiters=nothing)
    domain isa Number || throw(ArgumentError("GGR supports domains of individual eigenvalues"))
    p isa SymmetricBZ || throw(ArgumentError("GGR supports BZ parameters from load_bz"))
    E = domain
    bz = p

    A = sum_ggr(ndims(bz.lims), alg.npt, E, cacheval...)

    return DOSSolution(A, Success, (;))
end

function sum_ggr(ndim, npt, E, weights, energies, velocities)
    @assert ndim == length(first(velocities))
    b = 1/2npt
    formula = (args...) -> ggr_formula(b, E, args...)
    mapreduce(+, weights, energies, velocities) do w, es, vs
        AutoSymPTR.mymul(w, mapreduce(formula, +, es, vs...))
    end
end

ggr_formula(b, E, e) = throw(ArgumentError("GGR implemented for up to 3d BZ"))
ggr_formula(b, E, e, vs...) = ggr_formula(b, E, e)
# TODO: in higher dimensions, we can compute the area of the equi-frequency surface in a
# box with polyhedral manipulations, i.e.:
# - in plane coordinates, e+t*v, compute the convex hull in t due to intersection with E and
#   bounds by sides of box
# - use any method to compute area of the convex hull in t, such as iterated integration.

function ggr_formula(b, E, e, v1)
    v1 = abs(v1)
    Δω = abs(E - e)
    ω₁ = b * v1
    return zero(E) <= Δω <= ω₁ ? 1/v1 : zero(1/v1)
end
function ggr_formula(b, E, e, v1, v2)
    v2, v1 = extrema((abs(v1), abs(v2)))
    Δω = abs(E - e)
    ω₁ = b * abs(v1 - v2)
    ω₃ = b * (v1 + v2)
    # 2b is the line element
    return zero(E) <= Δω <= ω₁  ? 2b/v1 :
            ω₁ <= Δω <= ω₃      ? (b*(v1 + v2) - Δω)/(v1*v2) : zero(4b/v1)
end
function ggr_formula(b, E, e, v1, v2, v3)
    v3, v2, v1 = sort(SVector(abs(v1), abs(v2), abs(v3)))
    Δω = abs(E - e)
    ω₁ = b * abs(v1 - v2 - v3)
    ω₂ = b * (v1 - v2 + v3)
    ω₃ = b * (v1 + v2 - v3)
    ω₄ = b * (v1 + v2 + v3)
    v = hypot(v1, v2, v3)
    # 4b^2 is the area element
    (v1 >= v2 + v3 && zero(E) <= Δω <= ω₁)  ? 4b^2/v1   :
    (v1 <= v2 + v3 && zero(E) <= Δω <= ω₁)  ? (2b^2*(v1*v2 + v2*v3 + v3*v1) - (Δω^2 + (v*b)^2))/(v1*v2*v3) :
    ω₁ <= Δω <= ω₂  ? (b^2*(v1*v2 + 3*v2*v3 + v3*v1) - b*Δω*(-v1 + v2 + v3) - (Δω^2 + (v*b)^2)/2)/(v1*v2*v3) :
    ω₂ <= Δω <= ω₃  ? 2b*(b*(v1+v2) - Δω)/(v1*v2) : # this formula was incorrect in the Liu et al paper, correct in the Gilat paper
    ω₃ <= Δω <= ω₄  ? (b*(v1+v2+v3) - Δω)^2/(2*v1*v2*v3) : zero(4b^2/v1)
end
