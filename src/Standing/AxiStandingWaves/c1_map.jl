# ─── C1 inter-element continuity map ───

"""
    C1Map

Algebraic constraint map enforcing C⁰ + C¹ continuity at inter-element junctions
and ∂ζ/∂r = 0 at the axis (r = 0). Provides zero-allocation expand/restrict
operations between free DOFs and full nodal vectors.

The constraint system `x_constrained = expansion_matrix * x_free` is precomputed
during construction, making expand/restrict operations pure matrix-vector multiplies.

# Fields
- `N_full::Int` — total number of free-surface nodes
- `N_free::Int` — number of independent (free) DOFs
- `free_idx::Vector{Int}` — indices of free nodes in the full vector
- `junction_idx::Vector{Int}` — indices of constrained (junction + axis) nodes
- `expansion_matrix::Matrix{Float64}` — precomputed `(I-M)⁻¹B` for constraint evaluation
- `has_r0::Bool` — whether axis constraint (r=0) is present

# See also
[`build_c1_map`](@ref), [`expand_c1`](@ref), [`restrict_c1`](@ref)
"""
struct C1Map
    N_full::Int
    N_free::Int
    free_idx::Vector{Int}
    junction_idx::Vector{Int}
    expansion_matrix::Matrix{Float64}
    has_r0::Bool
end

function Base.show(io::IO, cm::C1Map)
    print(io, "C1Map(N_full=$(cm.N_full), N_free=$(cm.N_free), ",
          "$(length(cm.junction_idx)) constrained)")
end

function Base.show(io::IO, ::MIME"text/plain", cm::C1Map)
    println(io, "C1Map:")
    println(io, "  Full nodes:  $(cm.N_full)")
    println(io, "  Free DOFs:   $(cm.N_free)")
    println(io, "  Constrained: $(cm.N_full - cm.N_free)")
    print(io, "  Axis (r=0):  $(cm.has_r0 ? "constrained" : "not present")")
end

"""
    build_c1_map(mesh::CylindricalBasin) → C1Map

Build the C¹ continuity constraint map for the free-surface nodes of `mesh`.
Constraints enforce:
- Value continuity (C⁰) at inter-element junctions
- Derivative continuity (C¹) at inter-element junctions
- Zero radial derivative at the axis of symmetry (r = 0)
"""
function build_c1_map(mesh::CylindricalBasin)
    N_F = mesh.n_sf
    junction_nodes = Int[]
    dep_coeffs_list = Vector{Float64}[]
    dep_indices_list = Vector{Int}[]

    # Axis: dζ/dr = 0 at r=0
    e1 = mesh.elements[1]; n = e1.Q + 1
    if e1.nodes_rz[1][1] < 1e-12
        push!(junction_nodes, 1)
        coeffs = Float64[]; indices = Int[]
        @inbounds for k in 2:n
            push!(indices, k)
            push!(coeffs, -e1.D[1, k] / e1.D[1, 1])
        end
        push!(dep_coeffs_list, coeffs)
        push!(dep_indices_list, indices)
    end

    # Junctions: C0+C1
    offset = 0
    for j in 1:(mesh.n_fe - 1)
        e_left = mesh.elements[j]; e_right = mesh.elements[j + 1]
        n_left = e_left.Q + 1; n_right = e_right.Q + 1
        idx_l = offset + n_left; idx_r = idx_l + 1
        denom = e_left.D[end, n_left] - e_right.D[1, 1]
        coeffs = Float64[]; indices = Int[]
        @inbounds for k in 1:(n_left-1)
            push!(indices, offset + k)
            push!(coeffs, -e_left.D[end, k] / denom)
        end
        @inbounds for k in 2:n_right
            push!(indices, idx_r + k - 1)
            push!(coeffs, e_right.D[1, k] / denom)
        end
        push!(junction_nodes, idx_l)
        push!(junction_nodes, idx_r)
        push!(dep_coeffs_list, coeffs)
        push!(dep_indices_list, indices)
        offset += n_left
    end

    free_idx = setdiff(1:N_F, junction_nodes)
    N_free = length(free_idx)
    nc = length(dep_coeffs_list)

    has_r0 = !isempty(junction_nodes) && junction_nodes[1] == 1

    if nc == 0
        return C1Map(N_F, N_free, free_idx, junction_nodes, zeros(0, N_free), has_r0)
    end

    # Map full index → position in free_idx (0 if constrained)
    full_to_free = zeros(Int, N_F)
    @inbounds for (k, idx) in enumerate(free_idx)
        full_to_free[idx] = k
    end

    # Map constrained node → constraint index
    node_to_c = Dict{Int,Int}()
    if has_r0
        node_to_c[1] = 1; ji = 2
        for ci in 2:nc
            node_to_c[junction_nodes[ji]] = ci
            node_to_c[junction_nodes[ji+1]] = ci
            ji += 2
        end
    else
        ji = 1
        for ci in 1:nc
            node_to_c[junction_nodes[ji]] = ci
            node_to_c[junction_nodes[ji+1]] = ci
            ji += 2
        end
    end

    # Build (I - M) and B where constrained = (I-M)⁻¹ B * v_free
    M_mat = zeros(nc, nc)
    B_mat = zeros(nc, N_free)
    @inbounds for ci in 1:nc
        for (c, idx) in zip(dep_coeffs_list[ci], dep_indices_list[ci])
            cj = get(node_to_c, idx, 0)
            if cj > 0
                M_mat[ci, cj] += c
            else
                fi = full_to_free[idx]
                fi > 0 && (B_mat[ci, fi] += c)
            end
        end
    end

    expansion_matrix = (I - M_mat) \ B_mat

    C1Map(N_F, N_free, free_idx, junction_nodes, expansion_matrix, has_r0)
end

"""
    expand_c1!(v, cm::C1Map, v_free)

In-place: fill `v` (length `N_full`) from `v_free` (length `N_free`),
computing constrained node values via the precomputed expansion matrix.
"""
function expand_c1!(v::AbstractVector{T}, cm::C1Map, v_free::AbstractVector{T}) where {T}
    # Place free values
    @inbounds for (k, idx) in enumerate(cm.free_idx)
        v[idx] = v_free[k]
    end

    nc = size(cm.expansion_matrix, 1)
    nc == 0 && return v

    # Compute constrained values: expansion_matrix * v_free
    @inbounds if cm.has_r0
        # Constraint 1 → axis node
        s = zero(T)
        for j in 1:cm.N_free
            s += cm.expansion_matrix[1, j] * v_free[j]
        end
        v[1] = s
        ji = 2
        for ci in 2:nc
            s = zero(T)
            for j in 1:cm.N_free
                s += cm.expansion_matrix[ci, j] * v_free[j]
            end
            v[cm.junction_idx[ji]] = s
            v[cm.junction_idx[ji+1]] = s
            ji += 2
        end
    else
        ji = 1
        for ci in 1:nc
            s = zero(T)
            for j in 1:cm.N_free
                s += cm.expansion_matrix[ci, j] * v_free[j]
            end
            v[cm.junction_idx[ji]] = s
            v[cm.junction_idx[ji+1]] = s
            ji += 2
        end
    end
    v
end

"""
    expand_c1(cm::C1Map, v_free) → v_full

Allocating version of [`expand_c1!`](@ref). Returns a new vector of length `N_full`.
"""
function expand_c1(cm::C1Map, v_free::AbstractVector{T}) where {T}
    v = Vector{T}(undef, cm.N_full)
    expand_c1!(v, cm, v_free)
end

"""
    restrict_c1!(v_free, cm::C1Map, v)

In-place: extract free DOFs from full vector `v` into `v_free`.
"""
function restrict_c1!(v_free::AbstractVector, cm::C1Map, v::AbstractVector)
    @inbounds for (k, idx) in enumerate(cm.free_idx)
        v_free[k] = v[idx]
    end
    v_free
end

"""
    restrict_c1(cm::C1Map, v) → v_free

Allocating version of [`restrict_c1!`](@ref). Returns a vector of length `N_free`.
"""
restrict_c1(cm::C1Map, v::AbstractVector) = v[cm.free_idx]
