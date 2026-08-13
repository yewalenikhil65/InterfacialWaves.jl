# ──────────────────────────────────────────────────────────────────────────────
# assembly.jl — BIE influence matrix assembly and HOSESolver constructor
# ──────────────────────────────────────────────────────────────────────────────

# Log-weighted Gauss rule on [0,1]: ∫₀¹ f(t)(-log t) dt ≈ Σ wᵢ f(tᵢ)

"""Compute log-Gauss quadrature nodes and weights at precompile time."""
function _computeLogGaussConstant()::Tuple{NTuple{16,Float64}, NTuple{16,Float64}}
    nodes, weights = gauss(t -> -log(t), 16, 0.0, 1.0; rtol=1e-12)
    ntuple(i -> nodes[i], Val(16)), ntuple(i -> weights[i], Val(16))
end

# Pre-computed log-Gauss nodes and weights stored as immutable NTuple
const LogGaussNodesWeights_CONST::Tuple{NTuple{16,Float64}, NTuple{16,Float64}} = _computeLogGaussConstant()

"""
    logGaussQuadrature() → Tuple{NTuple{16,Float64}, NTuple{16,Float64}}

Return the pre-computed log-weighted Gauss quadrature rule on [0,1] for singular integration.
The nodes and weights are computed at precompile time and stored as an immutable constant.
"""
@inline function logGaussQuadrature()
    LogGaussNodesWeights_CONST
end

"""Map parameter `s` (within element arc-length interval) to physical (r, z) coordinates."""
function _s_to_rz(src_e::Element, s::Float64)
    t = (s - src_e.s_start) / (src_e.s_end - src_e.s_start)
    src_e.type == FreeSurface && return (s, 0.0)
    src_e.type == Wall && return (src_e.nodes_rz[1][1],
        src_e.nodes_rz[1][2] + t * (src_e.nodes_rz[end][2] - src_e.nodes_rz[1][2]))
    (src_e.nodes_rz[1][1] + t * (src_e.nodes_rz[end][1] - src_e.nodes_rz[1][1]),
     src_e.nodes_rz[1][2])
end

"""Convert local node index `k` in element `e` to its arc-length parameter value."""
_node_to_s(e::Element, k::Int) = e.s_start + (e.nodes_y[k] + 1.0) / 2.0 * (e.s_end - e.s_start)

"""Evaluate the k-th Lagrange basis polynomial of element `src_e` at arc-length parameter `s`."""
function _lagrange_basis(src_e::Element, k::Int, s::Float64)
    y = 2.0 * (s - src_e.s_start) / (src_e.s_end - src_e.s_start) - 1.0
    val = 1.0
    @inbounds for j in 1:src_e.Q+1
        j != k && (val *= (y - src_e.nodes_y[j]) / (src_e.nodes_y[k] - src_e.nodes_y[j]))
    end
    val
end

"""Integrate Green's function G₀ against Lagrange basis `k_local` on `src_e`, splitting at singularity `s_sing` with log subtraction."""
function _integrate_split_G(ri, zi, src_e, s_sing, k_local, segbuf)
    s0, s1 = src_e.s_start, src_e.s_end
    rp_sing, _ = _s_to_rz(src_e, s_sing)
    if rp_sing < 1e-14
        integrand(s) = begin
            rp, zp = _s_to_rz(src_e, s)
            G = ring_G(ri, zi, rp, zp)
            isinf(G) ? 0.0 : G * rp * _lagrange_basis(src_e, k_local, s)
        end
        v, _ = quadgk(integrand, s0, s_sing, s1; rtol=1e-12, order=7, segbuf)
        return v
    end
    t_unit, w_unit = logGaussQuadrature()
    val = 0.0
    for (s_lo, s_hi) in ((s0, s_sing), (s_sing, s1))
        L = s_hi - s_lo; L < 1e-15 && continue
        from_left = abs(s_lo - s_sing) < 1e-15
        log_sum = 0.0
        @inbounds for i in eachindex(t_unit)
            u = L * t_unit[i]
            s = from_left ? s_sing + u : s_hi - u
            log_sum += w_unit[i] * _lagrange_basis(src_e, k_local, s)
        end
        logL_over_pi = log(L) / π
        remainder(u) = begin
            s = from_left ? s_sing + u : s_hi - u
            rp, zp = _s_to_rz(src_e, s)
            (ring_G_remainder(ri, zi, rp, zp, u) - logL_over_pi) * _lagrange_basis(src_e, k_local, s)
        end
        smooth, _ = quadgk(remainder, 0.0, L; rtol=1e-12, order=7, segbuf)
        val += smooth + (L / π) * log_sum
    end
    val
end

"""Integrate normal derivative ∂G₀/∂n' against Lagrange basis `k_local` on `src_e`, splitting at singularity `s_sing`."""
function _integrate_split_Gn(ri, zi, src_e, s_sing, k_local, segbuf)
    s0, s1 = src_e.s_start, src_e.s_end
    nr, nz = src_e.nr, src_e.nz
    integrand(s) = begin
        rp, zp = _s_to_rz(src_e, s)
        Gn = ring_dGdn(ri, zi, rp, zp, nr, nz)
        (isnan(Gn) || isinf(Gn)) ? 0.0 : Gn * rp * _lagrange_basis(src_e, k_local, s)
    end
    v, _ = quadgk(integrand, s0, s_sing, s1; rtol=1e-12, order=15, segbuf)
    v
end

"""
    integrateGreenFunction(ri, zi, src_e, k_local, segbuf) → Float64

Integrate the ring-source Green's function G₀ against Lagrange basis `k_local`
on element `src_e` using adaptive Gauss–Kronrod quadrature (rtol=1e-12, order=15).

Computes: ∫ G₀(ri, zi; r(s), z(s)) ψₖ(s) r(s) ds

where `ψₖ` is the k-th Lagrange basis polynomial.

This replaces the Symbol dispatch pattern: `_integrate_quadgk(..., :G, ...)`.

# Arguments
- `ri::Float64`, `zi::Float64`: field point coordinates
- `src_e::Element`: source element
- `k_local::Int`: local Lagrange basis index (1 ≤ k_local ≤ Q+1)
- `segbuf::QuadGK.SegmentBuffer`: pre-allocated quadrature buffer

# Returns
- `::Float64`: the integral value

# References
This is a semantic dispatch function replacing Symbol-based dispatch.
Numerical results are identical to `_integrate_quadgk(..., :G, ...)` to within
floating-point round-off.
"""
function integrateGreenFunction(ri, zi, src_e, k_local, segbuf)
    s0, s1 = src_e.s_start, src_e.s_end
    integrand(s) = begin
        rp, zp = _s_to_rz(src_e, s)
        G = ring_G(ri, zi, rp, zp)
        Lk = _lagrange_basis(src_e, k_local, s)
        isinf(G) ? 0.0 : G * rp * Lk
    end
    v, _ = quadgk(integrand, s0, s1; rtol=1e-12, order=15, segbuf)
    v
end

"""
    integrateGreenNormalDerivative(ri, zi, src_e, k_local, segbuf) → Float64

Integrate the normal derivative of the ring-source Green's function ∂G₀/∂n'
against Lagrange basis `k_local` on element `src_e` using adaptive Gauss–Kronrod
quadrature (rtol=1e-12, order=15).

Computes: ∫ (∂G₀/∂n')(ri, zi; r(s), z(s)) ψₖ(s) r(s) ds

where `n'` is the outward normal at the source point and `ψₖ` is the k-th
Lagrange basis polynomial.

This replaces the Symbol dispatch pattern: `_integrate_quadgk(..., :Gn, ...)`.

# Arguments
- `ri::Float64`, `zi::Float64`: field point coordinates
- `src_e::Element`: source element
- `k_local::Int`: local Lagrange basis index (1 ≤ k_local ≤ Q+1)
- `segbuf::QuadGK.SegmentBuffer`: pre-allocated quadrature buffer

# Returns
- `::Float64`: the integral value

# References
This is a semantic dispatch function replacing Symbol-based dispatch.
Numerical results are identical to `_integrate_quadgk(..., :Gn, ...)` to within
floating-point round-off.
"""
function integrateGreenNormalDerivative(ri, zi, src_e, k_local, segbuf)
    s0, s1 = src_e.s_start, src_e.s_end
    nr, nz = src_e.nr, src_e.nz
    integrand(s) = begin
        rp, zp = _s_to_rz(src_e, s)
        Gn = ring_dGdn(ri, zi, rp, zp, nr, nz)
        Lk = _lagrange_basis(src_e, k_local, s)
        (isnan(Gn) || isinf(Gn)) ? 0.0 : Gn * rp * Lk
    end
    v, _ = quadgk(integrand, s0, s1; rtol=1e-12, order=15, segbuf)
    v
end



"""Fill influence matrix columns for a regular (non-self) element pair, choosing quadrature strategy based on distance."""
function _fill_regular!(S, K, i, cols, ri, zi, src_e, segbuf)
    r1, z1 = src_e.nodes_rz[1]
    rn, zn = src_e.nodes_rz[end]
    d_start = sqrt((ri - r1)^2 + (zi - z1)^2)
    d_end   = sqrt((ri - rn)^2 + (zi - zn)^2)
    elem_len = src_e.s_end - src_e.s_start
    if d_start < 1e-12 || d_end < 1e-12
        s_sing = d_start < 1e-12 ? src_e.s_start : src_e.s_end
        for (k, j) in enumerate(cols)
            S[i, j] = _integrate_split_G(ri, zi, src_e, s_sing, k, segbuf)
            K[i, j] = _integrate_split_Gn(ri, zi, src_e, s_sing, k, segbuf)
        end
    elseif min(d_start, d_end) < 0.5 * elem_len
        for (k, j) in enumerate(cols)
            S[i, j] = integrateGreenFunction(ri, zi, src_e, k, segbuf)
            K[i, j] = integrateGreenNormalDerivative(ri, zi, src_e, k, segbuf)
        end
    else
        @inbounds for (k, j) in enumerate(cols)
            rp, zp = src_e.nodes_rz[k]
            Gv, Gnv = ring_G_and_dGdn(ri, zi, rp, zp, src_e.nr, src_e.nz)
            S[i, j] = Gv * rp * src_e.weights[k]
            K[i, j] = Gnv * rp * src_e.weights[k]
        end
    end
end

"""Fill influence matrix columns for a self-element (field point on source element), using singular quadrature."""
function _fill_self!(S, K, i, cols, ri, zi, src_e, segbuf)
    i_local = 0
    @inbounds for k in 1:src_e.Q+1
        cols[k] == i && (i_local = k; break)
    end
    s_sing = i_local > 0 ? _node_to_s(src_e, i_local) : 0.5 * (src_e.s_start + src_e.s_end)
    for (k, j) in enumerate(cols)
        S[i, j] = _integrate_split_G(ri, zi, src_e, s_sing, k, segbuf)
        K[i, j] = _integrate_split_Gn(ri, zi, src_e, s_sing, k, segbuf)
    end
end

"""Compute the solid angle coefficient c(x) for node `i`: 0.5 at corners, 1.0 on smooth boundaries."""
function _solid_angle(i, all_rz, mesh)
    ri, zi = all_rz[i]
    at_corner = (abs(ri - mesh.R) < 1e-12 && abs(zi) < 1e-12) ||
                (abs(ri - mesh.R) < 1e-12 && abs(zi + mesh.h) < 1e-12)
    at_corner ? 0.5 : 1.0
end

"""Replace inter-element junction rows in LHS/RHS with C⁰ + C¹ continuity constraints and wall matching."""
function _apply_continuity!(LHS, RHS_mat, RHS_body, mesh)
    N_B = mesh.n_body; N_F = mesh.n_sf
    offset = 0
    for j in 1:(mesh.n_fe - 1)
        e_left = mesh.elements[j]; e_right = mesh.elements[j + 1]
        n_left = e_left.Q + 1; n_right = e_right.Q + 1
        row_l = offset + n_left; row_r = row_l + 1
        LHS[row_l, :] .= 0.0; RHS_mat[row_l, :] .= 0.0; RHS_body[row_l, :] .= 0.0
        LHS[row_l, N_B + row_l] = 1.0; LHS[row_l, N_B + row_r] = -1.0
        LHS[row_r, :] .= 0.0; RHS_mat[row_r, :] .= 0.0; RHS_body[row_r, :] .= 0.0
        for k in 1:n_left; LHS[row_r, N_B + offset + k] = e_left.D[end, k]; end
        for k in 1:n_right; LHS[row_r, N_B + offset + n_left + k] = -e_right.D[1, k]; end
        offset += n_left
    end
    LHS[N_F, :] .= 0.0; RHS_mat[N_F, :] .= 0.0; RHS_body[N_F, :] .= 0.0
    LHS[N_F, 1] = 1.0; RHS_mat[N_F, N_F] = 1.0
end

"""
    assemble_system(mesh::CylindricalBasin) → (LHS, RHS_mat, RHS_body)

Assemble BIE influence matrices with adaptive singular quadrature.
Returns the LHS matrix (to be factored) and two RHS matrices for
Dirichlet (free-surface) and Neumann (body) boundary data.
"""
function assemble_system(mesh::CylindricalBasin)
    elems = mesh.elements
    N_F = mesh.n_sf; N_B = mesh.n_body; N = N_F + N_B
    all_rz = Tuple{Float64,Float64}[]
    elem_idx = Int[]
    for (ie, e) in enumerate(elems)
        for node in e.nodes_rz
            push!(all_rz, node)
            push!(elem_idx, ie)
        end
    end
    S = zeros(N, N); K = zeros(N, N)
    Threads.@threads for i in 1:N
        segbuf = get!(() -> QuadGK.alloc_segbuf(), task_local_storage(), :segbuf)
        ri, zi = all_rz[i]
        col = 0
        for (je, src_e) in enumerate(elems)
            n_src = src_e.Q + 1; cols = (col + 1):(col + n_src)
            if elem_idx[i] == je
                _fill_self!(S, K, i, cols, ri, zi, src_e, segbuf)
            else
                _fill_regular!(S, K, i, cols, ri, zi, src_e, segbuf)
            end
            col += n_src
        end
        K[i, i] += _solid_angle(i, all_rz, mesh)
    end
    sf = mesh.sf_range; bd = mesh.body_range
    LHS = zeros(N, N); RHS_mat = zeros(N, N_F); RHS_body = zeros(N, N_B)
    @inbounds for i in 1:N
        for (jj, j) in enumerate(bd); LHS[i, jj] = K[i, j]; end
        for (jj, j) in enumerate(sf); LHS[i, N_B + jj] = -S[i, j]; end
        for (jj, j) in enumerate(sf); RHS_mat[i, jj] = -K[i, j]; end
        for (jj, j) in enumerate(bd); RHS_body[i, jj] = S[i, j]; end
    end
    _apply_continuity!(LHS, RHS_mat, RHS_body, mesh)
    LHS, RHS_mat, RHS_body
end

"""
    HOSESolver(mesh::CylindricalBasin; order=3, gravity=9.81, cascade_smoothing=false) → HOSESolver

Assemble the BIE system and return a solver ready for time-stepping.

# Arguments
- `mesh::CylindricalBasin`: spectral element mesh
- `order::Int=3`: HOSE truncation order M (typically 3–5)
- `gravity::Float64=9.81`: gravitational acceleration
- `cascade_smoothing::Bool=false`: if true, apply Zhu (2000) cubic-spline smoothing
  between cascade levels in `compute_W!` to cap high-mode amplification

# Examples
```julia
mesh = CylindricalBasin(1.0, 0.5; n_fe=8, Q=8)
solver = HOSESolver(mesh; order=3, gravity=9.81)
solver_smooth = HOSESolver(mesh; order=3, cascade_smoothing=true)
```
"""
function HOSESolver(mesh::CylindricalBasin; order::Int=3, gravity::Float64=9.81,
                    cascade_smoothing::Bool=false)
    @info "Assembling BIE" N_F=mesh.n_sf N_B=mesh.n_body
    LHS, RHS_mat, RHS_body = assemble_system(mesh)
    @info "Factorizing..."
    LHS_fact = lu(LHS)
    ws = Workspace{Float64}(mesh.n_sf + mesh.n_body, mesh.n_sf, mesh.n_body, order)
    ops = _build_precomputed(mesh)
    S_cascade = cascade_smoothing ? build_smoothing_matrix(mesh) : nothing
    cascade_smoothing && @info "Inter-cascade smoothing enabled"
    @info "HOSESolver assembly complete" order=order
    HOSESolver(mesh, LHS_fact.L, LHS_fact.U, LHS_fact.p, RHS_mat, RHS_body,
               order, gravity, ws, ops, S_cascade)
end
