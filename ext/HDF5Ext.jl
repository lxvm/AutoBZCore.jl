module HDF5Ext

using Printf: @sprintf
using LinearAlgebra: norm

using AutoBZCore: IntegralSolver, MixedParameters, solver_type, AuxValue, IntegralSolution
using HDF5: h5open, write_dataset, read_dataset, Group, H5DataStore, create_dataset, create_group, delete_object, name
import AutoBZCore: batchsolve


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
function autobz_create_dataset(parent, path, T::Type, dims_)
    dims = ((ndims(T) == 0 ? () : size(T))..., dims_...)
    ax = ntuple(_-> :, Val(ndims(T)))
    return create_dataset(parent, path, eltype(T), dims), ax
end
function autobz_create_dataset(parent, path, ::Type{AuxValue{T}}, dims) where T
    # split auxvalue into two groups for easier interoperability with other languages, since
    # the HDF5 compound type could be a challenge
    g = create_group(parent, "I")
    gval, axval = autobz_create_dataset(g, "val", T, dims)
    gaux, axaux = autobz_create_dataset(g, "aux", T, dims)
    return (gval, gaux), (axval, axaux)
end

set_value(g, i::CartesianIndex, val) = g[i.I...] = val
set_value(parent, ax::Tuple, i::CartesianIndex, sol) = parent[ax...,i.I...] = sol
function set_value((gval, gaux)::Tuple, (axval, axaux)::Tuple, i::CartesianIndex, sol::AuxValue)
    set_value(gval, axval, i, sol.val)
    set_value(gaux, axaux, i, sol.aux)
    return nothing
end
# special cases for 0-d arrays with scalar fields, for which rewrite is not supported
function set_value(g, ::CartesianIndex{0}, val)
    p = parent(g)
    s = basename(name(g))
    delete_object(p, s)
    write_dataset(p, s, val)
    return nothing
end
set_value(parent, ::Tuple{}, i::CartesianIndex{0}, sol) = set_value(parent, i, sol)



param_group(parent, T, dims) = create_dataset(parent, "p", T, dims)
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
    set_value(group, i, p)
    return nothing
end
function param_record(group, p::Tuple, i)
    for (j,e) in enumerate(p)
        set_value(group[string(j)], i, e)
    end
    return nothing
end
function param_record((g, q), p::MixedParameters, i)
    for (j,e) in enumerate(getfield(p, :args))
        set_value(g[string(j)], i, e)
    end
    for (k,v) in pairs(getfield(p, :kwargs))
        set_value(q[string(k)], i, v)
    end
    return nothing
end

const liblock = ReentrantLock()

"""
    batchsolve(h5::H5DataStore, f::IntegralSolver, ps, [T]; verb=true, nthreads=Threads.nthreads())

Batchsolver
"""
function batchsolve(h5::H5DataStore, f::IntegralSolver, ps::AbstractArray, T=solver_type(f, ps[begin]); flush=true, verb=true, nthreads=Threads.nthreads())
    isconcretetype(T) || throw(ArgumentError("Result type of integrand is abstract or could not be inferred. Please provide the concrete return type to save to HDF5"))
    len = length(ps)
    dims = size(ps)
    gI, ax = autobz_create_dataset(h5, "I", T, dims)
    gE = create_dataset(h5, "E", Float64, dims)
    gt = create_dataset(h5, "t", Float64, dims)
    gr = create_dataset(h5, "retcode", Int32, dims)
    gp = param_group(h5, eltype(ps), dims)
    flush && Base.flush(h5)

    function h5callback(_, i, n, p, sol, t)
        lock(liblock)
        try
            verb && @info @sprintf "%5i / %i done in %e (s)" n len t
            set_value(gI, ax, i, sol.u)
            set_value(gE, i, isnothing(sol.resid) ? NaN : convert(Float64, T<:AuxValue ? sol.resid.val : sol.resid))
            set_value(gt, i, t)
            set_value(gr, i, Integer(sol.retcode))
            param_record(gp, p, i)
            flush && Base.flush(h5)
        finally
            unlock(liblock)
        end
    end

    verb && @info "Started parameter sweep"
    t = time()
    sol = batchsolve(f, ps, T; callback=h5callback, nthreads=nthreads)
    t = time()-t
    verb && @info @sprintf "Finished parameter sweep in %e (s) CPU time and %e (s) wall clock time" sum(read(gt)) t

    return sol
end

end
