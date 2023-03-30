module HDF5Ext

using Printf: @sprintf

if isdefined(Base, :get_extension)
    using AutoBZCore: IntegralSolver, MixedParameters
    using HDF5: h5open, write_dataset, read_dataset, Group, H5DataStore, create_dataset, create_group
    import AutoBZCore: batchsolve
else
    using ..AutoBZCore: IntegralSolver, MixedParameters
    using ..HDF5: h5open, write_dataset, read_dataset, Group, H5DataStore, create_dataset, create_group
    import ..AutoBZCore: batchsolve
end


"""
    read_h5_to_nt(filename)

Loads the h5 archive from `filename` and reads its datasets into a `NamedTuple`
and its groups into `NamedTuple`s recursively.
"""
read_h5_to_nt(filename) = h5open(read_h5_to_nt_, filename, "r")
read_h5_to_nt_(h5) = NamedTuple([Pair(Symbol(key), ((val = h5[key]) isa Group) ? read_h5_to_nt_(val) : read_dataset(h5, key)) for key in keys(h5)])

"""
    write_nt_to_h5(nt::NamedTuple, filename)

Takes a `NamedTuple` and writes its values, which must be arrays, into an h5
archive at `filename` with dataset names corresponding to the tuple names.
If a value is a `NamedTuple`, its datasets are written to h5 groups recursively.
"""
write_nt_to_h5(nt::NamedTuple, filename) = h5open(filename, "w") do h5
    write_nt_to_h5_(nt, h5)
end
function write_nt_to_h5_(nt::NamedTuple, h5)
    for key in keys(nt)
        if (val = nt[key]) isa NamedTuple
            write_nt_to_h5_(val, create_group(h5, string(key)))
        else
            write_dataset(h5, string(key), val)
        end
    end
end

# parallelization

# returns (dset, ax) to allow for array-valued data types
function autobz_create_dataset(parent, path, T::Type, dims_::Tuple{Vararg{Int}})
    dims = ((ndims(T) == 0 ? () : size(T))..., dims_...)
    ax = ntuple(_-> :, Val(ndims(T)))
    return create_dataset(parent, path, eltype(T), dims), ax
end


function param_group(parent, T, dims)
    return create_dataset(parent, "p", T, dims)
end
function param_group(parent, ::Type{T}, dims) where {T<:Tuple}
    g = create_group(parent, "params")
    for (i, S) in enumerate(T.parameters)
        create_dataset(g, string(i), S, dims)
    end
    return g
end
function param_group(parent, ::Type{MixedParameters{T,NamedTuple{K,V}}}, dims) where {T,K,V}
    g = create_group(parent, "args")
    for (i, S) in enumerate(T.parameters)
        create_dataset(g, string(i), S, dims)
    end
    q = create_group(parent, "kwargs")
    for (key, val) in zip(K,V.parameters)
        create_dataset(q, string(key), val, dims)
    end
    return (g,q)
end
function param_record(group, p, i)
    group[i] = p
end
function param_record(group, p::Tuple, i)
    for (j,e) in enumerate(p)
        group[string(j)][i] = e
    end
end
function param_record((g, q), p::MixedParameters, i)
    for (j,e) in getfield(p, :args)
        g[string(j)][i] = e
    end
    for (k,v) in pairs(getfield(p, :kwargs))
        q[string(k)][i] = v
    end
end

"""
    batchsolve(h5::H5DataStore, f::IntegralSolver, ps, T=Base.promote_op(f, eltype(ps)); nthreads=Threads.nthreads())

Batchsolver
"""
function batchsolve(h5::H5DataStore, f::IntegralSolver, ps, T=Base.promote_op(f, eltype(ps)), verb=true; nthreads=Threads.nthreads())
    dims = tuple(length(ps))

    gI, ax = autobz_create_dataset(h5, "I", T, dims)
    gE = create_dataset(h5, "E", Float64, dims)
    gt = create_dataset(h5, "t", Float64, dims)
    gr = create_dataset(h5, "retcode", Int32, dims)
    gp = param_group(h5, eltype(ps), dims)

    function h5callback(f, i, p, sol, t)
        verb && @info @sprintf "parameter %i finished in %e (s)" i t
        gI[ax...,i] = sol
        gE[i] = isnothing(sol.resid) ? NaN : convert(Float64, sol.resid)
        gt[i] = t
        gr[i] = Integer(sol.retcode)
        param_record(gp, p, i)
    end

    @info "Started parameter sweep"
    t = time()
    sol = batchsolve(f, ps, T; callback=h5callback, nthreads=nthreads)
    t = time()-t
    @info @sprintf "Finished parameter sweep in %e (s) CPU time and %e (s) wall clock time" sum(read(gt)) t
    sol
end

end