# ─── Spectral element mesh for cylindrical basin ───

"""Boundary element type classification."""
@enum ElemType FreeSurface Wall Bottom

"""
    Element

A single spectral element on the boundary contour, with Chebyshev–Lobatto
collocation nodes, differentiation matrix, and quadrature weights.

# Fields
- `type::ElemType` — FreeSurface, Wall, or Bottom
- `Q::Int` — polynomial order
- `s_start::Float64`, `s_end::Float64` — arc-length interval
- `nodes_y::Vector{Float64}` — reference nodes on [-1, 1]
- `nodes_rz::Vector{Tuple{Float64,Float64}}` — physical (r, z) coordinates
- `nr::Float64`, `nz::Float64` — outward unit normal components
- `D::Matrix{Float64}` — physical-space differentiation matrix
- `weights::Vector{Float64}` — quadrature weights
- `jacobian::Float64` — mapping Jacobian (s_end - s_start) / 2
"""
struct Element
    type::ElemType
    Q::Int
    s_start::Float64
    s_end::Float64
    nodes_y::Vector{Float64}
    nodes_rz::Vector{Tuple{Float64,Float64}}
    nr::Float64
    nz::Float64
    D::Matrix{Float64}
    weights::Vector{Float64}
    jacobian::Float64
end

"""
    CylindricalBasin <: AbstractMesh

Spectral element mesh for a circular cylindrical basin of radius `R` and depth `h`.

# Fields
- `elements::Vector{Element}` — all boundary elements (free surface + wall + bottom)
- `R::Float64` — basin radius
- `h::Float64` — basin depth
- `n_sf::Int` — total free-surface nodes
- `n_body::Int` — total body (wall + bottom) nodes
- `sf_range::UnitRange{Int}` — global indices of free-surface nodes
- `body_range::UnitRange{Int}` — global indices of body nodes
- `n_fe::Int` — number of free-surface elements

# Constructors
```julia
CylindricalBasin(R, h; n_fe=8, Q=8, Q_wall=16, Q_bottom=16)
CylindricalBasin(R, h, breakpoints; Q=8, Q_wall=16, Q_bottom=16)
```
"""
struct CylindricalBasin <: AbstractMesh
    elements::Vector{Element}
    R::Float64
    h::Float64
    n_sf::Int
    n_body::Int
    sf_range::UnitRange{Int}
    body_range::UnitRange{Int}
    n_fe::Int
end

function Base.show(io::IO, mesh::CylindricalBasin)
    print(io, "CylindricalBasin(R=$(mesh.R), h=$(mesh.h), ",
          "$(mesh.n_fe) elem, N_sf=$(mesh.n_sf), N_body=$(mesh.n_body))")
end

function Base.show(io::IO, ::MIME"text/plain", mesh::CylindricalBasin)
    println(io, "CylindricalBasin:")
    println(io, "  Radius R = $(mesh.R)")
    println(io, "  Depth  h = $(mesh.h)")
    println(io, "  Free-surface: $(mesh.n_fe) elements, $(mesh.n_sf) nodes")
    println(io, "  Body (wall+bottom): $(mesh.n_body) nodes")
    Q_sf = mesh.elements[1].Q
    Q_w = mesh.elements[mesh.n_fe + 1].Q
    Q_b = mesh.elements[end].Q
    print(io, "  Polynomial orders: Q_sf=$Q_sf, Q_wall=$Q_w, Q_bottom=$Q_b")
end

# ─── Chebyshev utilities ───

"""Chebyshev–Lobatto nodes on [-1, 1] for polynomial order Q."""
function chebyshev_nodes(Q::Int)
    [cos((Q - j) * π / Q) for j in 0:Q]
end

"""Chebyshev differentiation matrix on nodes `y`."""
function chebyshev_diffmat(y::Vector{Float64})
    Q = length(y) - 1
    c = [j == 1 || j == Q + 1 ? 2.0 : 1.0 for j in 1:Q+1]
    D = zeros(Q + 1, Q + 1)
    @inbounds for j in 1:Q+1, k in 1:Q+1
        j != k && (D[j, k] = (c[j] / c[k]) * (-1)^(j + k) / (y[j] - y[k]))
    end
    @inbounds for j in 1:Q+1
        D[j, j] = -sum(D[j, k] for k in 1:Q+1 if k != j)
    end
    D
end

"""Clenshaw–Curtis quadrature weights on Q+1 Chebyshev–Lobatto nodes."""
function chebyshev_weights(Q::Int)
    Q == 0 && return [2.0]
    θ = [j * π / Q for j in 0:Q]
    w = zeros(Q + 1)
    @inbounds for j in 0:Q
        s = 0.0
        for k in 0:div(Q, 2)
            bk = (k == 0 || k == div(Q, 2)) ? 1.0 : 2.0
            s += bk / (1 - 4k^2) * cos(2k * θ[j+1])
        end
        cj = (j == 0 || j == Q) ? 1.0 : 2.0
        w[j+1] = cj / Q * s
    end
    reverse(w)
end

"""Construct a single boundary element."""
function make_element(type::ElemType, Q::Int, s0::Float64, s1::Float64,
                      nr::Float64, nz::Float64, coord_fn::Function)
    y = chebyshev_nodes(Q)
    J = (s1 - s0) / 2.0
    nodes_rz = [coord_fn(s0 + (yi + 1.0) / 2.0 * (s1 - s0)) for yi in y]
    D_phys = chebyshev_diffmat(y) ./ J
    w = chebyshev_weights(Q) .* J
    Element(type, Q, s0, s1, y, nodes_rz, nr, nz, D_phys, w, J)
end

# ─── Constructors ───

"""
    CylindricalBasin(R, h; n_fe=8, Q=8, Q_wall=16, Q_bottom=16)

Construct a spectral element mesh with uniform free-surface element spacing.

# Arguments
- `R::Float64`: basin radius (must be positive)
- `h::Float64`: basin depth (must be positive)
- `n_fe::Int=8`: number of free-surface elements
- `Q::Int=8`: polynomial order on free surface
- `Q_wall::Int=16`: polynomial order on sidewall
- `Q_bottom::Int=16`: polynomial order on bottom

# Examples
```julia
mesh = CylindricalBasin(1.0, 0.5; n_fe=8, Q=8)
```
"""
function CylindricalBasin(R::Float64, h::Float64;
                          n_fe::Int=8, Q::Int=8, Q_wall::Int=16, Q_bottom::Int=16)
    @assert R > 0 "Radius must be positive, got R=$R"
    @assert h > 0 "Depth must be positive, got h=$h"
    @assert n_fe >= 1 "Need at least 1 free-surface element, got n_fe=$n_fe"
    @assert Q >= 2 "Polynomial order must be ≥ 2, got Q=$Q"
    bp = collect(range(0.0, R, length=n_fe + 1))
    CylindricalBasin(R, h, bp; Q=Q, Q_wall=Q_wall, Q_bottom=Q_bottom)
end

"""
    CylindricalBasin(R, h, breakpoints; Q=8, Q_wall=16, Q_bottom=16)

Construct a spectral element mesh with custom free-surface breakpoints for graded meshing.
`breakpoints` must start at 0 and end at R.

# Examples
```julia
bp = [0.0, 0.05, 0.12, 0.22, 0.35, 0.50, 0.68, 0.84, 1.0]
mesh = CylindricalBasin(1.0, 0.5, bp; Q=8)
```
"""
function CylindricalBasin(R::Float64, h::Float64, bp::Vector{Float64};
                          Q::Int=8, Q_wall::Int=16, Q_bottom::Int=16)
    @assert bp[1] ≈ 0.0 && bp[end] ≈ R "breakpoints must span [0, R], got [$(bp[1]), $(bp[end])]"
    @assert length(bp) >= 2 "Need at least 2 breakpoints"
    n_fe = length(bp) - 1
    elements = Element[]
    for j in 1:n_fe
        push!(elements, make_element(FreeSurface, Q, bp[j], bp[j+1],
                                     0.0, 1.0, s -> (s, 0.0)))
    end
    push!(elements, make_element(Wall, Q_wall, 0.0, h, 1.0, 0.0, s -> (R, -s)))
    push!(elements, make_element(Bottom, Q_bottom, 0.0, R, 0.0, -1.0, s -> (R - s, -h)))
    n_sf = sum(e.Q + 1 for e in elements if e.type == FreeSurface)
    n_body = sum(e.Q + 1 for e in elements if e.type != FreeSurface)
    CylindricalBasin(elements, R, h, n_sf, n_body, 1:n_sf, (n_sf+1):(n_sf+n_body), n_fe)
end

"""
    surface_nodes(mesh::CylindricalBasin) → Vector{Float64}

Return the radial coordinates of all free-surface collocation nodes.
"""
function surface_nodes(mesh::CylindricalBasin)
    r = Float64[]
    for e in mesh.elements
        e.type == FreeSurface && append!(r, [c[1] for c in e.nodes_rz])
    end
    r
end
