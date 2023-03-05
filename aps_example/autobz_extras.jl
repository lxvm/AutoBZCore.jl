# this is a minimal version some utilities in AutoBZ.jl

using LinearAlgebra, StaticArrays, FourierSeriesEvaluators, AutoBZCore

"""
    parse_hamiltonian(filename)

Parse an ab-initio Hamiltonian output from Wannier90 into `filename`, extracting
the fields `(date_time, num_wann, nrpts, degen, irvec, C)`
"""
parse_hamiltonian(filename) = open(filename) do file
    date_time = readline(file)
    
    num_wann = parse(Int, readline(file))
    nrpts = parse(Int, readline(file))
    
    entries_per_line = 15
    degen = Vector{Int}(undef, nrpts)
    for j in 1:ceil(Int, nrpts/entries_per_line)
        col = split(readline(file)) # degeneracy of Wigner-Seitz grid points
        for i in eachindex(col)
            i > entries_per_line && error("parsing found more entries than expected")
            degen[i+(j-1)*entries_per_line] = parse(Int, col[i])
        end
    end

    C = Vector{SMatrix{num_wann,num_wann,ComplexF64,num_wann^2}}(undef, nrpts)
    irvec = Vector{SVector{3,Int}}(undef, nrpts)
    c = Matrix{ComplexF64}(undef, num_wann, num_wann)
    for k in 1:nrpts
        for j in 1:num_wann^2
            col = split(readline(file))
            irvec[k] = SVector{3,Int}(parse(Int, col[1]), parse(Int, col[2]), parse(Int, col[3]))
            m = parse(Int, col[4])
            n = parse(Int, col[5])
            re = parse(Float64, col[6])
            im = parse(Float64, col[7])
            c[m,n] = complex(re, im)
        end
        C[k] = SMatrix{num_wann,num_wann,ComplexF64,num_wann^2}(c)
    end
    date_time, num_wann, nrpts, degen, irvec, C
end

"""
    load_hamiltonian(filename; period=1.0, compact=:N)

Load an ab-initio Hamiltonian output from Wannier90 into `filename` as an
evaluatable `FourierSeries` whose periodicity can be set by the keyword argument
`period` which defaults to setting the period along each dimension to `1.0`. To
define different periods for different dimensions, pass an `SVector` as the
`period`. To store Hermitian Fourier coefficients in compact form, use the
keyword `compact` to specify:
- `:N`: do not store the coefficients in compact form
- `:L`: store the lower triangle of the coefficients
- `:U`: store the upper triangle of the coefficients
- `:S`: store the lower triangle of the symmetrized coefficients, `(c+c')/2`
"""
function load_hamiltonian(filename; period=1.0, compact=:N)
    date_time, num_wann, nrpts, degen, irvec, C_ = parse_hamiltonian(filename)
    C = load_coefficients(Val{compact}(), num_wann, irvec, C_)[1]
    InplaceFourierSeries(C; period=period, offset=map(s -> -div(s,2)-1, size(C)))
end

load_coefficients(compact, num_wann, irvec, cs...) = load_coefficients(compact, num_wann, irvec, cs)
@generated function load_coefficients(compact, num_wann, irvec, cs::NTuple{N}) where N
    T_full = :(SMatrix{num_wann,num_wann,ComplexF64,num_wann^2})
    T_compact = :(SHermitianCompact{num_wann,ComplexF64,StaticArrays.triangularnumber(num_wann)})
    if compact === Val{:N}
        T = T_full; expr = :(c[i])
    elseif compact === Val{:L}
        T = T_compact; expr = :($T(c[i]))
    elseif compact === Val{:U}
        T = T_compact; expr = :($T(c[i]'))
    elseif compact === Val{:S}
        T = T_compact; expr = :($T(0.5*(c[i]+c[i]')))
    end
    quote
        nmodes = zeros(Int, 3)
        for idx in irvec
            @inbounds for i in 1:3
                if (n = abs(idx[i])) > nmodes[i]
                    nmodes[i] = n
                end
            end
        end
        Cs = Base.Cartesian.@ntuple $N _ -> zeros($T, (2nmodes .+ 1)...)
        for (i,idx) in enumerate(irvec)
            idx_ = CartesianIndex((idx .+ nmodes .+ 1)...)
            for (j,c) in enumerate(cs)
                Cs[j][idx_] = $expr
            end
        end
        Cs
    end
end


"""
    parse_position_operator(filename)

Parse a position operator output from Wannier90 into `filename`, extracting the
fields `(date_time, num_wann, nrpts, irvec, X, Y, Z)`
"""
parse_position_operator(filename) = open(filename) do file
    date_time = readline(file)

    num_wann = parse(Int, readline(file))
    nrpts = parse(Int, readline(file))

    T = SMatrix{num_wann,num_wann,ComplexF64,num_wann^2}
    X = Vector{T}(undef, nrpts)
    Y = Vector{T}(undef, nrpts)
    Z = Vector{T}(undef, nrpts)
    irvec = Vector{SVector{3,Int}}(undef, nrpts)
    x = Matrix{ComplexF64}(undef, num_wann, num_wann)
    y = Matrix{ComplexF64}(undef, num_wann, num_wann)
    z = Matrix{ComplexF64}(undef, num_wann, num_wann)
    for k in 1:nrpts
        for j in 1:num_wann^2
            col = split(readline(file))
            irvec[k] = SVector{3,Int}(parse(Int, col[1]), parse(Int, col[2]), parse(Int, col[3]))
            m = parse(Int, col[4])
            n = parse(Int, col[5])
            re_x = parse(Float64, col[6])
            im_x = parse(Float64, col[7])
            x[m,n] = complex(re_x, im_x)
            re_y = parse(Float64, col[8])
            im_y = parse(Float64, col[9])
            y[m,n] = complex(re_y, im_y)
            re_z = parse(Float64, col[10])
            im_z = parse(Float64, col[11])
            z[m,n] = complex(re_z, im_z)
        end
        X[k] = T(x)
        Y[k] = T(y)
        Z[k] = T(z)
    end
    date_time, num_wann, nrpts, irvec, X, Y, Z
end


"""
    parse_wout(filename; iprint=1)

returns the lattice vectors `a` and reciprocal lattice vectors `b`
"""
parse_wout(filename; iprint=1) = open(filename) do file
    iprint != 1 && throw(ArgumentError("Verbosity setting iprint not implemented"))

    # header
    while (l = strip(readline(file))) != "SYSTEM"
        continue
    end

    # system
    readline(file)
    readline(file)
    readline(file)
    ## lattice vectors
    c = Matrix{Float64}(undef, 3, 3)
    for i in 1:3
        col = split(readline(file))
        popfirst!(col)
        @. c[:,i] = parse(Float64, col)
    end
    A = SMatrix{3,3,Float64,9}(c)

    readline(file)
    readline(file)
    readline(file)
    readline(file)
    ## reciprocal lattice vectors
    for i in 1:3
        col = split(readline(file))
        popfirst!(col)
        @. c[:,i] = parse(Float64, col)
    end
    B = SMatrix{3,3,Float64,9}(c)


    readline(file)
    readline(file)
    readline(file) # site fractional coordinate cartesian coordinate (unit)
    readline(file)
    # lattice
    species = String[]
    site = Int[]
    frac_lat = SVector{3,Float64}[]
    cart_lat = SVector{3,Float64}[]
    while true
        col = split(readline(file))
        length(col) == 11 || break
        push!(species, col[2])
        push!(site, parse(Int, col[3]))
        push!(frac_lat, parse.(Float64, col[4:6]))
        push!(cart_lat, parse.(Float64, col[8:10]))
    end
    # projections
    # k-point grid
    # main
    # wannierise
    # disentangle
    # plotting
    # k-mesh
    # etc...

    return A, B, species, site, frac_lat, cart_lat
end

parse_sym(filename) = open(filename) do file
    nsymmetry = parse(Int, readline(file))
    readline(file)
    point_sym = Vector{SMatrix{3,3,Float64,9}}(undef, nsymmetry)
    translate = Vector{SVector{3,Float64}}(undef, nsymmetry)
    S = Matrix{Float64}(undef, (3,4))
    for i in 1:nsymmetry
        for j in 1:4
            col = split(readline(file))
            S[:,j] .= parse.(Float64, col)
        end
        point_sym[i] = S[:,1:3]
        translate[i] = S[:,4]
        readline(file)
    end
    return nsymmetry, point_sym, translate
end

"""
    load_wannier90_data(seedname; gauge=:Wannier, vkind=:none, vcomp=:whole, compact=:N)

Reads Wannier90 output files with the given `seedname` to return the Hamiltonian
(optionally with band velocities if `vkind` is specified as either `:covariant`
or `:gradient`) and the full Brillouin zone limits. The keyword `compact` is
available if to compress the Fourier series if its Fourier coefficients are
known to be Hermitian. Returns `(w, fbz)` containing the Wannier interpolant and
the full BZ limits.
"""
function load_wannier90_data(seedname::String; compact=:N, atol=1e-5)
    A, B, = parse_wout(seedname * ".wout")

    # use fractional lattice coordinates for the Fourier series
    w = load_hamiltonian(seedname * "_hr.dat"; compact=compact)

    fbz = FullBZ(A, B, AutoBZCore.lattice_bz_limits(B); atol=atol)

    w, fbz
end

cubic_sym_ibz(bz::SymmetricBZ; kwargs...) = cubic_sym_ibz(bz.A, bz.B; kwargs...)
function cubic_sym_ibz(A::M, B::M; kwargs...) where {N,T,M<:SMatrix{N,N,T}}
    lims = AutoBZCore.TetrahedralLimits(ntuple(n -> 1/2, Val{N}()))
    syms = vec(collect(cube_automorphisms(Val{N}())))
    SymmetricBZ(A, B, lims, syms; kwargs...)
end

"""
    cube_automorphisms(::Val{d}) where d

return a generator of the symmetries of the cube in `d` dimensions including the
identity.
"""
cube_automorphisms(n::Val{d}) where {d} = (S*P for S in sign_flip_matrices(n), P in permutation_matrices(n))
n_cube_automorphisms(d) = n_sign_flips(d) * n_permutations(d)

sign_flip_tuples(n::Val{d}) where {d} = Iterators.product(ntuple(_ -> (1,-1), n)...)
sign_flip_matrices(n::Val{d}) where {d} = (Diagonal(SVector{d,Int}(A)) for A in sign_flip_tuples(n))
n_sign_flips(d::Integer) = 2^d

function permutation_matrices(t::Val{n}) where {n}
    permutations = permutation_tuples(ntuple(identity, t))
    (StaticArrays.sacollect(SMatrix{n,n,Int,n^2}, ifelse(j == p[i], 1, 0) for i in 1:n, j in 1:n) for p in permutations)
end
permutation_tuples(C::NTuple{N,T}) where {N,T} = @inbounds((C[i], p...)::NTuple{N,T} for i in eachindex(C) for p in permutation_tuples(C[[j for j in eachindex(C) if j != i]]))
permutation_tuples(C::NTuple{1}) = C
n_permutations(n::Integer) = factorial(n)
