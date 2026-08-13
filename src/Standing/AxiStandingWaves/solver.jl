# ─── HOSE solver: BVP cascade, DtN map, ODE RHS ───
# ─── HOSE solver: BVP cascade, DtN map, ODE RHS ───

# ─── Type aliases for concreteness ───

"""
    ProfileTuple

Concrete type alias for standing wave profile data at a single amplitude.

```julia
ProfileTuple = NamedTuple{
    (:kA, :A, :T, :dT, :zeta),
    Tuple{Float64, Float64, Float64, Float64, Vector{Float64}}
}
```

Fields:
- `kA::Float64` — steepness (wavenumber × amplitude)
- `A::Float64` — wave amplitude
- `T::Float64` — period
- `dT::Float64` — relative period shift (%)
- `zeta::Vector{Float64}` — free-surface elevation profile

Using a concrete type (all element types specified) enables type inference to
determine `.profiles[i].kA` has type `Float64` at compile time, eliminating
dispatch overhead when accessing profile fields.
"""
const ProfileTuple = NamedTuple{
    (:kA, :A, :T, :dT, :zeta),
    Tuple{Float64, Float64, Float64, Float64, Vector{Float64}}
}

# ─── Precomputed operators ───

"""
    PrecomputedOps

Cached differentiation matrices and index information for the bottom and wall
elements, used in the HOSE cascade to compute vertical derivatives and enforce
boundary conditions without re-deriving element geometry at every RHS evaluation.

# Fields
- `D2_bot::Matrix{Float64}` — second derivative matrix on the bottom element
- `D_bot::Matrix{Float64}` — first derivative matrix on the bottom element
- `d_end_last_sf::Vector{Float64}` — last row of differentiation matrix of last free-surface element
- `d_end_coeff::Float64` — coefficient for enforcing ∂ζ/∂r = 0 at the wall contact
- `last_sf_idx0::Int` — global start index of last free-surface element
- `last_sf_n::Int` — number of nodes in the last free-surface element
- `bot_n::Int` — number of nodes on the bottom element
- `wall_n::Int` — number of nodes on the wall element
- `bot_r::Vector{Float64}` — radial coordinates of bottom element nodes
- `bot_inv_r::Vector{Float64}` — reciprocal radial coordinates (0 at axis)
"""
struct PrecomputedOps
    D2_bot::Matrix{Float64}
    D_bot::Matrix{Float64}
    d_end_last_sf::Vector{Float64}
    d_end_coeff::Float64
    last_sf_idx0::Int
    last_sf_n::Int
    bot_n::Int
    wall_n::Int
    bot_r::Vector{Float64}
    bot_inv_r::Vector{Float64}
end

# ─── Workspace (pre-allocated arrays for zero-allocation inner loops) ───

"""
    Workspace{T}

Pre-allocated arrays for zero-allocation inner loops in BVP solves and HOSE cascade.
Parameterized on element type `T` to support both `Float64` and `ForwardDiff.Dual`.

# Fields
- `bvp_rhs::Vector{T}` — right-hand side vector for the BVP linear system
- `bvp_x::Vector{T}` — solution vector for the BVP linear system
- `dz_phi::Vector{Vector{Vector{T}}}` — hierarchical z-derivative storage `∂ᵏΦ/∂zᵏ` for cascade
- `f_m::Vector{T}` — temporary buffer for Dirichlet data assembly
- `zeta_f::Vector{T}` — expanded free-surface elevation
- `phi_s_f::Vector{T}` — expanded surface potential
- `W_f::Vector{T}` — vertical velocity at free surface
- `zeta_r_f::Vector{T}` — radial derivative of surface elevation
- `phi_r_f::Vector{T}` — radial derivative of surface potential
- `zeta_pow::Vector{Vector{T}}` — powers of ζ divided by factorial: ζˡ/l!
- `neumann_body::Vector{T}` — Neumann data on body boundary
- `residual_buf::Vector{T}` — pre-allocated residual buffer for shooting (length N_free + 1)
- `u0_buf::Vector{T}` — pre-allocated initial condition buffer (length 2*N_F)
- `temp_smooth::Vector{T}` — temporary buffer for smoothing operations (length N_F)
"""
struct Workspace{T}
    bvp_rhs::Vector{T}
    bvp_x::Vector{T}
    dz_phi::Vector{Vector{Vector{T}}}
    f_m::Vector{T}
    zeta_f::Vector{T}
    phi_s_f::Vector{T}
    W_f::Vector{T}
    zeta_r_f::Vector{T}
    phi_r_f::Vector{T}
    zeta_pow::Vector{Vector{T}}
    neumann_body::Vector{T}
    residual_buf::Vector{T}
    u0_buf::Vector{T}
    temp_smooth::Vector{T}
end

"""Construct a `Workspace{T}` with `N` total nodes, `N_F` free-surface nodes, `N_B` body nodes, HOSE order `M`, and `N_free` free DOFs from C1 map."""
function Workspace{T}(N::Int, N_F::Int, N_B::Int, M::Int, N_free::Int) where {T}
    dz_phi = [[zeros(T, N_F) for _ in 0:M] for _ in 1:M]
    zeta_pow = [zeros(T, N_F) for _ in 1:M]
    Workspace{T}(zeros(T, N), zeros(T, N), dz_phi, zeros(T, N_F),
        zeros(T, N_F), zeros(T, N_F), zeros(T, N_F),
        zeros(T, N_F), zeros(T, N_F), zeta_pow, zeros(T, N_B),
        zeros(T, N_free + 1), zeros(T, 2 * N_F), zeros(T, N_F))
end

"""Construct a `Workspace{T}` with `N` total nodes, `N_F` free-surface nodes, `N_B` body nodes, and HOSE order `M` (legacy: uses N_F as N_free fallback)."""
function Workspace{T}(N::Int, N_F::Int, N_B::Int, M::Int) where {T}
    Workspace{T}(N, N_F, N_B, M, N_F)
end

# ─── HOSESolver ───

"""
    HOSESolver <: AbstractSolver

Precomputed BIE system for HOSE time-stepping. Contains the LU-factored
influence matrix, right-hand side matrices, and pre-allocated workspace.

Construct with `HOSESolver(mesh; order=3, gravity=9.81)`.

# Fields
- `mesh::CylindricalBasin` — the discretized domain
- `order::Int` — HOSE perturbation order M (typically 3–5)
- `gravity::Float64` — gravitational acceleration
"""
struct HOSESolver <: AbstractSolver
    mesh::CylindricalBasin
    LU_L::Matrix{Float64}
    LU_U::Matrix{Float64}
    LU_p::Vector{Int}
    RHS_mat::Matrix{Float64}
    RHS_body::Matrix{Float64}
    order::Int
    gravity::Float64
    workspace::Workspace{Float64}
    ops::PrecomputedOps
    S_cascade::Union{Matrix{Float64}, Nothing}  # inter-cascade smoothing matrix
end

function Base.show(io::IO, s::HOSESolver)
    print(io, "HOSESolver(order=$(s.order), g=$(s.gravity), ",
          "N_sf=$(s.mesh.n_sf), N_body=$(s.mesh.n_body))")
end

function Base.show(io::IO, ::MIME"text/plain", s::HOSESolver)
    println(io, "HOSESolver:")
    println(io, "  HOSE order M = $(s.order)")
    println(io, "  Gravity g = $(s.gravity)")
    println(io, "  BVPs per RHS eval: $(s.order * (s.order + 1) ÷ 2)")
    print(io, "  Mesh: ", s.mesh)
end

# ─── Type-stable workspace cache for Dual numbers (ForwardDiff) ───

"""
    getWorkspace(solver::HOSESolver, ::Type{T}) → Workspace{T}

Return a type-stable `Workspace{T}` for the given solver.

- For `T === Float64`: returns `solver.workspace` directly (zero overhead, no allocation).
- For other `T` (e.g., `ForwardDiff.Dual`): creates or retrieves a task-local `Workspace{T}`
  cached via `task_local_storage()`, so each thread/task gets its own correctly-typed workspace
  without re-allocating per Jacobian column.

The return type is guaranteed concrete: `::Workspace{T}`. The cache is invalidated whenever the
solver's dimensions (order `M`, mesh size, or `N_free`) change, so it is safe to call with
solvers of differing `order` or mesh size within the same task.
"""
@inline function getWorkspace(solver::HOSESolver, ::Type{T}) where T
    if T === Float64
        return solver.workspace::Workspace{Float64}
    end
    # AD path: task-local caching
    tls_key = :_axisw_workspace
    tls = task_local_storage()
    ws = get(tls, tls_key, nothing)
    N_free = size(solver.workspace.residual_buf, 1) - 1  # infer from Float64 workspace
    if ws isa Workspace{T} &&
       length(ws.dz_phi) == solver.order &&
       length(ws.bvp_rhs) == solver.mesh.n_sf + solver.mesh.n_body &&
       length(ws.neumann_body) == solver.mesh.n_body &&
       length(ws.residual_buf) == N_free + 1
        return ws::Workspace{T}
    end
    N = solver.mesh.n_sf + solver.mesh.n_body
    N_F = solver.mesh.n_sf
    N_B = solver.mesh.n_body
    M = solver.order
    new_ws = Workspace{T}(N, N_F, N_B, M, N_free)
    task_local_storage(tls_key, new_ws)
    return new_ws::Workspace{T}
end

# Keep old name as internal alias for backward compatibility
@inline _get_workspace(solver::HOSESolver, ::Type{T}) where {T} = getWorkspace(solver, T)

# ─── BVP solve (AD-compatible manual LU forward/back substitution) ───

"""
    solve_bvp!(solver, phi_n_sf, dirichlet_sf, ws; neumann_body=nothing)

Solve one BVP: given Dirichlet data on the free surface (and optionally Neumann
data on the body), compute the normal derivative on the free surface.
Uses the pre-factored LU system for O(N²) cost.
"""
function solve_bvp!(solver::HOSESolver, phi_n_sf::AbstractVector,
                    dirichlet_sf::AbstractVector, ws::Workspace;
                    neumann_body::Union{AbstractVector,Nothing}=nothing)
    mul!(ws.bvp_rhs, solver.RHS_mat, dirichlet_sf)
    neumann_body !== nothing && mul!(ws.bvp_rhs, solver.RHS_body, neumann_body, 1.0, 1.0)
    _lu_solve!(ws.bvp_x, solver.LU_L, solver.LU_U, solver.LU_p, ws.bvp_rhs)
    N_B = solver.mesh.n_body
    @inbounds for i in 1:solver.mesh.n_sf
        phi_n_sf[i] = ws.bvp_x[N_B + i]
    end
    phi_n_sf
end

"""Solve `LUx = Pb` via manual forward/back substitution, compatible with ForwardDiff Dual numbers."""
function _lu_solve!(x::AbstractVector, L::Matrix{Float64}, U::Matrix{Float64},
                    p::Vector{Int}, rhs::AbstractVector)
    N = length(rhs)
    # Forward substitution (L)
    @inbounds for i in 1:N
        s = rhs[p[i]]
        for j in 1:i-1
            s -= L[i, j] * x[j]
        end
        x[i] = s
    end
    # Back substitution (U)
    @inbounds for i in N:-1:1
        s = x[i]
        for j in i+1:N
            s -= U[i, j] * x[j]
        end
        x[i] = s / U[i, i]
    end
    x
end

# ─── Precomputed operators ───

"""Build the `PrecomputedOps` struct caching differentiation matrices and indices for the HOSE cascade."""
function _build_precomputed(mesh::CylindricalBasin)
    bot_e = mesh.elements[mesh.n_fe + 2]
    wall_e = mesh.elements[mesh.n_fe + 1]
    last_sf = mesh.elements[mesh.n_fe]
    n_bot = bot_e.Q + 1
    D2 = bot_e.D * bot_e.D
    n_last = last_sf.Q + 1
    d_end = last_sf.D[end, :]
    idx0 = mesh.n_sf - n_last + 1
    bot_r = [bot_e.nodes_rz[i][1] for i in 1:n_bot]
    bot_inv_r = [ri > 1e-12 ? 1.0 / ri : 0.0 for ri in bot_r]
    PrecomputedOps(D2, bot_e.D, d_end, -1.0 / d_end[end], idx0, n_last,
                   n_bot, wall_e.Q + 1, bot_r, bot_inv_r)
end

"""Enforce ∂f/∂r = 0 at the wall contact point (r = R) by adjusting the last free-surface node value."""
function _enforce_zero_dr_at_wall!(solver::HOSESolver, f::AbstractVector)
    ops = solver.ops
    s = zero(eltype(f))
    @inbounds for k in 1:ops.last_sf_n - 1
        s += ops.d_end_last_sf[k] * f[ops.last_sf_idx0 + k - 1]
    end
    f[solver.mesh.n_sf] = s * ops.d_end_coeff
    f
end

"""Compute the Neumann body data from ∂²Φ/∂z² = -(∂²Φ/∂r² + (1/r)∂Φ/∂r) on the bottom using the BVP solution."""
function _compute_zderiv_neumann!(solver::HOSESolver, neumann::AbstractVector, bvp_x::AbstractVector)
    ops = solver.ops
    fill!(neumann, zero(eltype(neumann)))
    phi_bot = @view bvp_x[(ops.wall_n + 1):(ops.wall_n + ops.bot_n)]
    @inbounds for i in 1:ops.bot_n
        dphi_dr = zero(eltype(neumann))
        d2phi_dr2 = zero(eltype(neumann))
        for j in 1:ops.bot_n
            dphi_dr -= ops.D_bot[i, j] * phi_bot[j]
            d2phi_dr2 += ops.D2_bot[i, j] * phi_bot[j]
        end
        inv_r = ops.bot_inv_r[i]
        neumann[ops.wall_n + i] = inv_r > 0.0 ? d2phi_dr2 + dphi_dr * inv_r : 2 * d2phi_dr2
    end
    neumann
end

# ─── Inter-cascade smoothing helper ───

"""
    _apply_smoothing!(f, S, N_F, tmp)

Apply smoothing matrix S to vector f in-place. Works with Dual numbers (ForwardDiff).
The S matrix is Float64; the multiplication S*f is type-stable for any eltype(f).
Uses a pre-allocated `tmp` buffer to avoid read-after-write hazard (zero allocations).
"""
@inline function _apply_smoothing!(f::AbstractVector{T}, S::Matrix{Float64}, N_F::Int, tmp::AbstractVector{T}) where T
    # Two-pass approach using pre-allocated tmp (ws.temp_smooth) to avoid
    # read-after-write hazard: compute S*f into tmp, then copy back to f.
    # This eliminates heap allocation for the intermediate result.
    @inbounds for i in 1:N_F
        s = zero(T)
        for j in 1:N_F
            s += S[i, j] * f[j]
        end
        tmp[i] = s
    end
    @inbounds for i in 1:N_F
        f[i] = tmp[i]
    end
    nothing
end

# ─── HOSE W computation ───

"""
    compute_W!(solver, W, ws, zeta, phi_s)

Compute the vertical velocity W = ∂Φ/∂z|_{z=ζ} via the HOSE cascade of M(M+1)/2
BVP solves. Modifies `W` in-place. Uses pre-allocated workspace `ws`.
"""
function compute_W!(solver::HOSESolver, W::AbstractVector, ws::Workspace,
                    zeta::AbstractVector, phi_s::AbstractVector)
    M = solver.order; N_F = solver.mesh.n_sf
    dz = ws.dz_phi; zp = ws.zeta_pow
    S = solver.S_cascade  # nothing if smoothing disabled

    # Precompute ζ^l / l!
    @inbounds for i in 1:N_F
        zp[1][i] = zeta[i]
    end
    for l in 2:M
        @inbounds @simd for i in 1:N_F
            zp[l][i] = zp[l-1][i] * zeta[i] / l
        end
    end

    # Cascade: solve BVPs sequentially
    for m in 1:M
        f_m = dz[m][1]
        if m == 1
            @inbounds @simd for i in 1:N_F
                f_m[i] = phi_s[i]
            end
        else
            fill!(f_m, zero(eltype(f_m)))
            for l in 1:m-1
                dzl = dz[m - l][l + 1]
                @inbounds @simd for i in 1:N_F
                    f_m[i] -= zp[l][i] * dzl[i]
                end
            end
        end
        solve_bvp!(solver, dz[m][2], f_m, ws)
        for k in 2:(M - m + 1)
            _enforce_zero_dr_at_wall!(solver, dz[m][k])
            if iseven(k)
                _compute_zderiv_neumann!(solver, ws.neumann_body, ws.bvp_x)
                solve_bvp!(solver, dz[m][k + 1], dz[m][k], ws; neumann_body=ws.neumann_body)
            else
                solve_bvp!(solver, dz[m][k + 1], dz[m][k], ws)
            end
        end
    end

    # Assemble W from all BVP outputs
    fill!(W, zero(eltype(W)))
    for m in 1:M
        @inbounds @simd for i in 1:N_F
            W[i] += dz[m][2][i]
        end
        for l in 1:(M - m)
            @inbounds @simd for i in 1:N_F
                W[i] += zp[l][i] * dz[m][l + 2][i]
            end
        end
    end

    # Apply smoothing to final W to suppress spurious high-mode content
    if S !== nothing
        _apply_smoothing!(W, S, N_F, ws.temp_smooth)
    end

    W
end

# Keep old name as internal alias
const _compute_W! = compute_W!

# ─── Surface derivative ───

"""Compute the radial derivative df/dr on the free surface using element-wise spectral differentiation."""
function _surface_derivative!(solver::HOSESolver, df::AbstractVector, f::AbstractVector)
    idx = 1
    for e in solver.mesh.elements
        e.type != FreeSurface && continue
        n = e.Q + 1
        mul!(view(df, idx:idx+n-1), e.D, view(f, idx:idx+n-1))
        e.nodes_rz[1][1] < 1e-12 && (df[idx] = zero(eltype(df)))
        idx += n
    end
    df
end

# ─── Utilities for ODE norm with Duals ───

"""Extract the real (primal) value from a ForwardDiff Dual number."""
_realval(x::ForwardDiff.Dual) = ForwardDiff.value(x)
"""Extract the real value from a plain Real number."""
_realval(x::Real) = Float64(x)

"""Compute the L² norm of an array of possibly-Dual numbers, extracting real values."""
function _dualnorm(u::AbstractArray, t)
    s = 0.0
    @inbounds for i in eachindex(u)
        s += abs2(_realval(u[i]))
    end
    sqrt(s)
end
"""Scalar norm for a possibly-Dual number."""
_dualnorm(u::Number, t) = abs(_realval(u))

# ─── HOSERhs functor ───

"""
    HOSERhs{S,C}

Callable functor implementing the HOSE ODE right-hand side for the free-surface evolution.
Use as `rhs(du, u, p, t)` with any OrdinaryDiffEq solver.

State vector `u = [ζ; φˢ]` (length 2N_F). Exact C¹ projection applied at
every evaluation to maintain inter-element continuity.

Pre-allocated buffers eliminate allocations in the hot path (zero-alloc for Float64).

# Construction
```julia
rhs = HOSERhs(solver)             # auto-builds C1Map
rhs = HOSERhs(solver, c1map)      # with explicit C1Map
```

# Usage
```julia
sol = ODE.solve(ODE.ODEProblem(rhs, u0, (0.0, T)), ODE.Tsit5();
                abstol=1e-12, reltol=1e-11)
```
"""
struct HOSERhs{S<:HOSESolver, C<:C1Map}
    solver::S
    c1map::C
    _buf_free::Vector{Float64}   # pre-alloc buffer for free DOF extraction (N_free)
end

"""Construct a `HOSERhs` functor with an explicit `C1Map`."""
function HOSERhs(solver::HOSESolver, cm::C1Map)
    HOSERhs(solver, cm, Vector{Float64}(undef, cm.N_free))
end

"""Construct a `HOSERhs` functor, automatically building the `C1Map` from the solver's mesh."""
HOSERhs(solver::HOSESolver) = HOSERhs(solver, build_c1_map(solver.mesh))

function Base.show(io::IO, rhs::HOSERhs)
    print(io, "HOSERhs(order=$(rhs.solver.order), N_sf=$(rhs.solver.mesh.n_sf))")
end

"""Evaluate the HOSE ODE right-hand side: `du/dt = f(u)` where `u = [ζ; φˢ]`."""
function (rhs::HOSERhs)(du, u, p, t)
    # Zero-allocation design: all temporaries (zeta_f, phi_s_f, W_f, zeta_r_f, phi_r_f)
    # are written into pre-allocated Workspace buffers obtained via getWorkspace.
    # For Float64 this returns solver.workspace (no allocation); for Dual types it
    # uses a task-local cached workspace. No heap allocation occurs in the Float64 path.
    solver = rhs.solver
    cm = rhs.c1map
    N_F = solver.mesh.n_sf
    T = eltype(u)
    ws = _get_workspace(solver, T)

    # Extract free DOFs and expand (zero-alloc for Float64 path)
    @inbounds for (k, idx) in enumerate(cm.free_idx)
        ws.f_m[k] = u[idx]  # reuse f_m as temp buffer for ζ free DOFs
    end
    expand_c1!(ws.zeta_f, cm, @view(ws.f_m[1:cm.N_free]))

    @inbounds for (k, idx) in enumerate(cm.free_idx)
        ws.f_m[k] = u[N_F + idx]  # reuse for φˢ free DOFs
    end
    expand_c1!(ws.phi_s_f, cm, @view(ws.f_m[1:cm.N_free]))

    compute_W!(solver, ws.W_f, ws, ws.zeta_f, ws.phi_s_f)
    _surface_derivative!(solver, ws.zeta_r_f, ws.zeta_f)
    _surface_derivative!(solver, ws.phi_r_f, ws.phi_s_f)

    # Free-surface rate equations with @fastmath (safe here: NaN check is upstream in Newton)
    g = solver.gravity
    @inbounds @fastmath for i in 1:N_F
        zr = ws.zeta_r_f[i]; pr = ws.phi_r_f[i]; w = ws.W_f[i]
        zr2p1 = 1 + zr * zr
        du[i] = -zr * pr + zr2p1 * w
        du[N_F + i] = -g * ws.zeta_f[i] - 0.5 * pr * pr + 0.5 * zr2p1 * w * w
    end

    # Project rates to C1 (extract free DOFs from du, expand back)
    @inbounds for (k, idx) in enumerate(cm.free_idx)
        ws.f_m[k] = du[idx]
    end
    expand_c1!(view(du, 1:N_F), cm, @view(ws.f_m[1:cm.N_free]))

    @inbounds for (k, idx) in enumerate(cm.free_idx)
        ws.f_m[k] = du[N_F + idx]
    end
    expand_c1!(view(du, N_F+1:2N_F), cm, @view(ws.f_m[1:cm.N_free]))
    nothing
end
