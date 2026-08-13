# ──────────────────────────────────────────────────────────────────────────────
# multiple_shooting.jl — Block-bidiagonal Newton solver for high-steepness continuation
# ──────────────────────────────────────────────────────────────────────────────

"""
    MultipleShooting <: AbstractShootingMethod

Configuration for multiple shooting continuation. Splits the half-period [0, T/2]
into `K` sub-intervals, computing block-bidiagonal Jacobians independently for
each block and solving the assembled sparse system via UMFPACK.

# Fields
- `K::Int`: Number of sub-intervals (default 16). More intervals reduce per-block
  sensitivity growth at the cost of more unknowns. Recommended: 8–32.
- `n_steps::Int`: Total RK4 steps across the half-period (distributed equally among blocks).
  Each block gets `n_steps ÷ K` steps.
- `maxiters::Int`: Maximum Newton iterations per continuation step (default 15).
- `tol::Float64`: Newton convergence tolerance on `‖F‖` (default 1e-9).

# Background

Single shooting computes the Jacobian of the full half-period map, whose condition
number grows as ~exp(λ_max · T/2). Multiple shooting reduces per-block integration
to T/(2K), giving per-block Jacobian condition ~exp(λ_max · T/(2K)). The full system
is block-bidiagonal and sparse, solvable in O(K·n³) via sparse LU.
"""
struct MultipleShooting <: AbstractShootingMethod
    K::Int
    n_steps::Int
    maxiters::Int
    tol::Float64
end

"""Construct a `MultipleShooting` configuration with keyword arguments."""
MultipleShooting(; K::Int=16, n_steps::Int=512, maxiters::Int=15, tol::Float64=1e-9) =
    MultipleShooting(K, n_steps, maxiters, tol)

function Base.show(io::IO, ms::MultipleShooting)
    print(io, "MultipleShooting(K=$(ms.K), n_steps=$(ms.n_steps), ",
          "maxiters=$(ms.maxiters), tol=$(ms.tol))")
end

# ══════════════════════════════════════════════════════════════════════════════
# BLOCK INTEGRATOR
# ══════════════════════════════════════════════════════════════════════════════

"""
    _ms_integrate_block(rhs!, N_F, ζ, φ, dt, n_sub, integrator) → (ζ_end, φ_end)

Integrate one sub-interval of duration `dt` using the solver and settings from `integrator`.
AD-compatible via SciMLSensitivity's ForwardDiffSensitivity.
"""
function _ms_integrate_block(rhs!::HOSERhs, N_F::Int,
                             ζ::AbstractVector{T}, φ::AbstractVector{T},
                             dt, n_sub::Int, integrator::DecoupledIntegrator) where T
    # Pre-allocate u0 instead of vcat (avoids allocation for Float64 path)
    u0 = Vector{T}(undef, 2N_F)
    @inbounds for i in 1:N_F
        u0[i] = ζ[i]
        u0[N_F + i] = φ[i]
    end
    sol = ODE.solve(
        ODE.ODEProblem(rhs!, u0, (zero(T(dt)), T(dt))),
        integrator.solver;
        adaptive=(integrator.n_steps == 0),
        dt=T(dt) / n_sub,
        abstol=integrator.abstol,
        reltol=integrator.reltol,
        save_everystep=false,
        maxiters=10_000_000,
        sensealg=integrator.sensealg,
        unstable_check=(_...) -> false)
    u = sol.u[end]
    @view(u[1:N_F]), @view(u[N_F+1:2N_F])
end

# ══════════════════════════════════════════════════════════════════════════════
# RESIDUAL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# BLOCK-BY-BLOCK JACOBIAN COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

"""Compute per-block Jacobians (Mⱼ, cⱼ) via ForwardDiff for the multiple-shooting system, parallelized across blocks."""
function _ms_compute_block_jacobians(x::Vector{Float64}, rhs!::HOSERhs,
                                     cm::C1Map, N_F::Int, K::Int, n_sub::Int,
                                     integrator::DecoupledIntegrator)
    N_free = cm.N_free
    T_val = x[end]

    Ms = Vector{Matrix{Float64}}(undef, K)
    cs = Vector{Vector{Float64}}(undef, K)

    # Block 1: input = [ζ₀_free; T], φ₀ = 0 (N_free+1 inputs → use smaller chunk)
    ζ0_free = x[1:N_free]
    input1 = vcat(ζ0_free, T_val)

    function _eval_block1(v)
        ζf = v[1:N_free]; Tv = v[end]
        dt = Tv / (2K)
        ζ = expand_c1(cm, ζf)
        φ = zeros(eltype(v), N_F)
        ζe, φe = _ms_integrate_block(rhs!, N_F, ζ, φ, dt, n_sub, integrator)
        vcat(restrict_c1(cm, ζe), restrict_c1(cm, φe))
    end

    # Use chunk_size=8 for optimal SIMD utilization
    cfg1 = ForwardDiff.JacobianConfig(_eval_block1, input1, ForwardDiff.Chunk{8}())
    J1 = ForwardDiff.jacobian(_eval_block1, input1, cfg1)
    Ms[1] = J1[:, 1:N_free]
    cs[1] = J1[:, end]

    # Blocks 2..K: input = [sⱼ₋₁_free; T] (2N_free+1 inputs)
    # These are independent — parallelize with Threads.@spawn
    tasks = Vector{Task}(undef, K - 1)
    for j in 2:K
        off = N_free + (j - 2) * 2N_free
        sj_prev = x[off+1:off+2N_free]
        input_j = vcat(sj_prev, T_val)

        tasks[j-1] = Threads.@spawn begin
            # Each thread gets its own closure (captures local copies)
            function _eval_block(v)
                sf = v[1:2N_free]; Tv = v[end]
                dt = Tv / (2K)
                ζf = sf[1:N_free]; φf = sf[N_free+1:2N_free]
                ζ = expand_c1(cm, ζf)
                φ = expand_c1(cm, φf)
                ζe, φe = _ms_integrate_block(rhs!, N_F, ζ, φ, dt, n_sub, integrator)
                vcat(restrict_c1(cm, ζe), restrict_c1(cm, φe))
            end
            cfg = ForwardDiff.JacobianConfig(_eval_block, input_j, ForwardDiff.Chunk{8}())
            Jj = ForwardDiff.jacobian(_eval_block, input_j, cfg)
            (Jj[:, 1:2N_free], Jj[:, end])
        end
    end

    # Collect results
    for j in 2:K
        Mj, cj = fetch(tasks[j-1])
        Ms[j] = Mj
        cs[j] = cj
    end

    Ms, cs
end

# ══════════════════════════════════════════════════════════════════════════════
# BOUNDARY CONDITION JACOBIAN
# ══════════════════════════════════════════════════════════════════════════════

"""Compute the boundary condition Jacobians (G₀, G_K, g_p) for the last block and amplitude constraint."""
function _ms_compute_bc_jacobians(x::Vector{Float64}, rhs!::HOSERhs,
                                  cm::C1Map, N_F::Int, K::Int, n_sub::Int,
                                  integrator::DecoupledIntegrator)
    N_free = cm.N_free
    T_val = x[end]
    ζ0_free = x[1:N_free]

    # G₀: only the amplitude constraint row depends on ζ₀_free
    G0 = zeros(N_free + 1, N_free)
    G0[end, :] .= ForwardDiff.gradient(v -> maximum(expand_c1(cm, v)), ζ0_free)

    # G_K and g_p: differentiate BC output w.r.t. [s_{K-1}; T]
    off_last = N_free + (K - 2) * 2N_free
    s_last = x[off_last+1:off_last+2N_free]
    input_bc = vcat(s_last, T_val)

    function _bc_block(v)
        sf = v[1:2N_free]; Tv = v[end]
        dt = Tv / (2K)
        ζf = sf[1:N_free]; φf = sf[N_free+1:2N_free]
        ζ = expand_c1(cm, ζf)
        φ = expand_c1(cm, φf)
        ζe, φe = _ms_integrate_block(rhs!, N_F, ζ, φ, dt, n_sub, integrator)
        bc = Vector{eltype(v)}(undef, N_free + 1)
        bc[1:N_free] .= restrict_c1(cm, φe)
        bc[end] = -minimum(ζe)
        bc
    end
    cfg_bc = ForwardDiff.JacobianConfig(_bc_block, input_bc, ForwardDiff.Chunk{8}())
    J_bc = ForwardDiff.jacobian(_bc_block, input_bc, cfg_bc)
    GK = J_bc[:, 1:2N_free]
    gp = J_bc[:, end]

    G0, GK, gp
end

# ══════════════════════════════════════════════════════════════════════════════
# NONLINEARSOLVE.JL INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════

"""
    _ms_residual!(F, x, p)

In-place MS residual for NonlinearSolve.jl. Parameters `p` is a NamedTuple with
fields: rhs!, cm, N_F, K, n_sub, A_target, integrator.
"""
function _ms_residual!(F::Vector{Float64}, x::Vector{Float64}, p)
    (; rhs_fn, cm, N_F, K, n_sub, A_target, integrator) = p
    N_free = cm.N_free
    T_val = x[end]
    dt_block = T_val / (2K)
    ζ0_free = @view x[1:N_free]
    ζ0 = expand_c1(cm, ζ0_free)
    φ0 = zeros(N_F)

    ζp, φp = ζ0, φ0
    for j in 1:K
        ζe, φe = _ms_integrate_block(rhs_fn, N_F, ζp, φp, dt_block, n_sub, integrator)
        if j < K
            off = N_free + (j - 1) * 2N_free
            ζj_free = @view x[off+1:off+N_free]
            φj_free = @view x[off+N_free+1:off+2N_free]
            ro = (j - 1) * 2N_free
            F[ro+1:ro+N_free] .= restrict_c1(cm, ζe) .- ζj_free
            F[ro+N_free+1:ro+2N_free] .= restrict_c1(cm, φe) .- φj_free
            ζp = expand_c1(cm, ζj_free)
            φp = expand_c1(cm, φj_free)
        else
            ro = (K - 1) * 2N_free
            F[ro+1:ro+N_free] .= restrict_c1(cm, φe)
            F[ro+N_free+1] = maximum(ζ0) - minimum(ζe) - 2A_target
        end
    end
    F
end

"""
    _ms_jac!(J, x, p)

In-place block-bidiagonal Jacobian for NonlinearSolve.jl.
Fills the sparse matrix `J` directly into `nzval` using a precomputed index map.
No intermediate sparse matrix is constructed.
"""
function _ms_jac!(J::SparseMatrixCSC{Float64,Int}, x::Vector{Float64}, p)
    (; rhs_fn, cm, N_F, K, n_sub, integrator, nzmap) = p
    N_free = cm.N_free
    n = 2N_free

    Ms, cs = _ms_compute_block_jacobians(x, rhs_fn, cm, N_F, K, n_sub, integrator)
    G0, GK, gp = _ms_compute_bc_jacobians(x, rhs_fn, cm, N_F, K, n_sub, integrator)

    # Zero all nonzero values
    fill!(J.nzval, 0.0)

    # Fill block entries using precomputed nzval index map
    # Block matching rows (j = 1..K-1)
    for j in 1:K-1
        Mj = Ms[j]
        cj = cs[j]
        n_cols_M = j == 1 ? N_free : n

        # Mⱼ block
        map_M = nzmap.M_maps[j]
        @inbounds for c in 1:n_cols_M, r in 1:n
            idx = map_M[r, c]
            idx > 0 && (J.nzval[idx] = Mj[r, c])
        end

        # -I block
        map_I = nzmap.I_maps[j]
        @inbounds for i in 1:n
            J.nzval[map_I[i]] = -1.0
        end

        # T-sensitivity column
        map_c = nzmap.c_maps[j]
        @inbounds for i in 1:n
            J.nzval[map_c[i]] = cj[i]
        end
    end

    # BC rows: G₀
    @inbounds for c in 1:N_free, r in 1:(N_free + 1)
        idx = nzmap.G0_map[r, c]
        idx > 0 && (J.nzval[idx] = G0[r, c])
    end

    # BC rows: G_K
    @inbounds for c in 1:n, r in 1:(N_free + 1)
        idx = nzmap.GK_map[r, c]
        idx > 0 && (J.nzval[idx] = GK[r, c])
    end

    # BC rows: g_p
    @inbounds for i in 1:(N_free + 1)
        J.nzval[nzmap.gp_map[i]] = gp[i]
    end

    J
end

"""
    NZValMap

Precomputed mapping from block matrix entries to positions in `J.nzval`.
Computed once at the start of a continuation run.
"""
struct NZValMap
    M_maps::Vector{Matrix{Int}}    # M_maps[j][r, c] → nzval index for block j
    I_maps::Vector{Vector{Int}}    # I_maps[j][i] → nzval index for -I diagonal
    c_maps::Vector{Vector{Int}}    # c_maps[j][i] → nzval index for T-column
    G0_map::Matrix{Int}            # G0_map[r, c] → nzval index
    GK_map::Matrix{Int}            # GK_map[r, c] → nzval index
    gp_map::Vector{Int}            # gp_map[i] → nzval index
end

"""
    _ms_build_jac_prototype(N_free, K) → (SparseMatrixCSC, NZValMap)

Build the sparse Jacobian prototype and precompute the nzval index map for
zero-allocation Jacobian filling.
"""
function _ms_build_jac_prototype(N_free::Int, K::Int)
    n = 2N_free
    n0 = N_free
    n_eq = (K - 1) * n + N_free + 1
    n_unk = n0 + (K - 1) * n + 1

    I_idx = Int[]; J_idx = Int[]
    sizehint!(I_idx, K * n * n + n_eq)
    sizehint!(J_idx, K * n * n + n_eq)

    # Matching rows for blocks 1..K-1
    for j in 1:K-1
        ro = (j - 1) * n
        if j == 1
            for c in 1:n0, r in 1:n
                push!(I_idx, ro + r); push!(J_idx, c)
            end
            for i in 1:n
                push!(I_idx, ro + i); push!(J_idx, n0 + i)
            end
        else
            co_prev = n0 + (j - 2) * n
            for c in 1:n, r in 1:n
                push!(I_idx, ro + r); push!(J_idx, co_prev + c)
            end
            co_cur = n0 + (j - 1) * n
            for i in 1:n
                push!(I_idx, ro + i); push!(J_idx, co_cur + i)
            end
        end
        for i in 1:n
            push!(I_idx, ro + i); push!(J_idx, n_unk)
        end
    end

    # BC rows
    ro_bc = (K - 1) * n
    for c in 1:n0, r in 1:(N_free + 1)
        push!(I_idx, ro_bc + r); push!(J_idx, c)
    end
    co_last = n0 + (K - 2) * n
    for c in 1:n, r in 1:(N_free + 1)
        push!(I_idx, ro_bc + r); push!(J_idx, co_last + c)
    end
    for i in 1:(N_free + 1)
        push!(I_idx, ro_bc + i); push!(J_idx, n_unk)
    end

    J = sparse(I_idx, J_idx, ones(length(I_idx)), n_eq, n_unk)

    # Build the nzval index map by looking up each entry's position in J
    M_maps = Vector{Matrix{Int}}(undef, K - 1)
    I_maps = Vector{Vector{Int}}(undef, K - 1)
    c_maps = Vector{Vector{Int}}(undef, K - 1)

    for j in 1:K-1
        ro = (j - 1) * n
        n_cols_M = j == 1 ? n0 : n
        co_M = j == 1 ? 0 : (n0 + (j - 2) * n)

        Mmap = zeros(Int, n, n_cols_M)
        for c in 1:n_cols_M, r in 1:n
            Mmap[r, c] = _nzval_index(J, ro + r, co_M + c)
        end
        M_maps[j] = Mmap

        co_I = j == 1 ? n0 : (n0 + (j - 1) * n)
        Imap = zeros(Int, n)
        for i in 1:n
            Imap[i] = _nzval_index(J, ro + i, co_I + i)
        end
        I_maps[j] = Imap

        cmap = zeros(Int, n)
        for i in 1:n
            cmap[i] = _nzval_index(J, ro + i, n_unk)
        end
        c_maps[j] = cmap
    end

    G0_map = zeros(Int, N_free + 1, N_free)
    for c in 1:n0, r in 1:(N_free + 1)
        G0_map[r, c] = _nzval_index(J, ro_bc + r, c)
    end

    GK_map = zeros(Int, N_free + 1, n)
    for c in 1:n, r in 1:(N_free + 1)
        GK_map[r, c] = _nzval_index(J, ro_bc + r, co_last + c)
    end

    gp_map = zeros(Int, N_free + 1)
    for i in 1:(N_free + 1)
        gp_map[i] = _nzval_index(J, ro_bc + i, n_unk)
    end

    nzmap = NZValMap(M_maps, I_maps, c_maps, G0_map, GK_map, gp_map)
    J, nzmap
end

"""Look up the nzval position for entry (row, col) in a SparseMatrixCSC."""
function _nzval_index(J::SparseMatrixCSC, row::Int, col::Int)
    for idx in nzrange(J, col)
        rowvals(J)[idx] == row && return idx
    end
    return 0  # not in pattern (should not happen)
end

# ══════════════════════════════════════════════════════════════════════════════
# INITIAL GUESS CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

"""
    _ms_build_guess_stable(ζ0_free, T_val, rhs, cm, N_F, K, integrator)

Build a multiple-shooting state vector by integrating the full half-period with
adaptive Tsit5 (high accuracy, no ForwardDiff involved), then sampling the
dense output at K-1 internal node points. This avoids the sequential fixed-step
block-by-block integration that diverges at high steepness.
"""
function _ms_build_guess_stable(ζ0_free::Vector{Float64}, T_val::Float64,
                                rhs!::HOSERhs, cm::C1Map, N_F::Int, K::Int,
                                integrator::DecoupledIntegrator)
    N_free = cm.N_free
    T_half = T_val / 2
    dt_block = T_half / K

    # Build initial condition
    ζ0 = expand_c1(cm, ζ0_free)
    u0 = vcat(ζ0, zeros(N_F))

    # Use adaptive Tsit5 for the seed — no AD here, so adaptive is safe.
    # High tolerances ensure accurate sampling at node points.
    sol = ODE.solve(
        ODE.ODEProblem(rhs!, u0, (0.0, T_half)),
        ODE.Tsit5();
        abstol=1e-12,
        reltol=1e-11,
        save_everystep=true,
        maxiters=10_000_000)

    # Check if solve failed
    if sol.retcode != ODE.ReturnCode.Success
        # Fallback: try with more conservative tolerances
        sol = ODE.solve(
            ODE.ODEProblem(rhs!, u0, (0.0, T_half)),
            ODE.Vern7();
            abstol=1e-10,
            reltol=1e-9,
            save_everystep=true,
            maxiters=10_000_000)
    end

    # Assemble MS state vector by sampling the dense output at node times
    x_ms = zeros(N_free + (K - 1) * 2N_free + 1)
    x_ms[1:N_free] .= ζ0_free

    for j in 1:K-1
        t_j = j * dt_block
        u_j = sol(t_j)
        ζ_j = u_j[1:N_F]
        φ_j = u_j[N_F+1:2N_F]
        off = N_free + (j - 1) * 2N_free
        x_ms[off+1:off+N_free] .= restrict_c1(cm, ζ_j)
        x_ms[off+N_free+1:off+2N_free] .= restrict_c1(cm, φ_j)
    end
    x_ms[end] = T_val
    x_ms
end

"""Construct an initial guess for multiple-shooting Newton via Lagrange extrapolation or integration from linear mode."""
function _ms_initial_guess(history::Vector{Tuple{Float64,Vector{Float64}}},
                           A::Float64, k::Float64, cm::C1Map, K::Int, n_sub::Int,
                           rhs!::HOSERhs, N_F::Int, T_lin::Float64,
                           r_vals::Vector{Float64}, integrator::DecoupledIntegrator)
    N_free = cm.N_free
    n = length(history)

    if n >= 3
        A1, x1 = history[end-2]; A2, x2 = history[end-1]; A3, x3 = history[end]
        L1 = ((A - A2) * (A - A3)) / ((A1 - A2) * (A1 - A3))
        L2 = ((A - A1) * (A - A3)) / ((A2 - A1) * (A2 - A3))
        L3 = ((A - A1) * (A - A2)) / ((A3 - A1) * (A3 - A2))
        return L1 .* x1 .+ L2 .* x2 .+ L3 .* x3
    elseif n == 2
        A_p, x_p = history[end]; A_p2, x_p2 = history[end-1]
        return x_p .+ (A - A_p) / (A_p - A_p2) .* (x_p .- x_p2)
    elseif n == 1
        A_p, x_p = history[end]
        x0 = copy(x_p)
        x0[1:N_free] .*= A / A_p
        return x0
    else
        ζ_full = [A * besselj(0, k * r) for r in r_vals]
        ζ0_free = restrict_c1(cm, ζ_full)
        return _ms_build_guess_stable(ζ0_free, T_lin, rhs!, cm, N_F, K, integrator)
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# CONTINUATION DISPATCH
# ══════════════════════════════════════════════════════════════════════════════

"""
    continuation(solver, method::MultipleShooting; kwargs...) → StandingWaveResult

Trace nonlinear standing waves using multiple shooting for high steepness.

# Arguments
- `solver::HOSESolver`: assembled BIE system
- `method::MultipleShooting`: configuration (K, n_steps, maxiters, tol)
- `ν::Int`: radial mode number
- `kA_range`: iterable of target steepness values
- `integrator::DecoupledIntegrator`: time integrator settings
- `nl_alg`: NonlinearSolve algorithm (default `NewtonRaphson()`). Accepts any algorithm
  from NonlinearSolve.jl: `TrustRegion()`, `LevenbergMarquardt()`, `NewtonRaphson()`, etc.
- `verbose::Bool=true`: print progress
- `seed`: optional `Vector{Tuple{Float64,Vector{Float64}}}` to seed history
"""
function continuation(solver::HOSESolver, method::MultipleShooting;
                      ν::Int, kA_range,
                      integrator::DecoupledIntegrator=DecoupledIntegrator(),
                      nl_alg=NLS.NewtonRaphson(),
                      verbose::Bool=true,
                      seed::Union{Nothing,Vector{Tuple{Float64,Vector{Float64}}}}=nothing)

    R = solver.mesh.R; h = solver.mesh.h; g = solver.gravity
    N_F = solver.mesh.n_sf
    r_vals = surface_nodes(solver.mesh)
    cm = build_c1_map(solver.mesh)
    N_free = cm.N_free

    # Wavenumber
    k0R = approx_besselroots(1, ν)[ν]
    k = k0R / R
    ω = sqrt(g * k * tanh(k * h))
    T_lin = 2π / ω

    K = method.K
    n_sub = method.n_steps ÷ K

    if verbose
        n_unk = N_free + (K - 1) * 2N_free + 1
        @info "Multiple shooting continuation" ν K order=solver.order T_lin=round(T_lin, digits=5) N_free n_unk n_sub solver_alg=nameof(typeof(integrator.solver)) nl_alg=nameof(typeof(nl_alg)) kA_min=first(kA_range) kA_max=last(kA_range)
    end

    rhs_fn = HOSERhs(solver, cm)

    # Precompute sparse Jacobian prototype and nzval index map
    jac_proto, nzmap = _ms_build_jac_prototype(N_free, K)

    profiles = ProfileTuple[]
    history = Tuple{Float64, Vector{Float64}}[]
    kA_targets = collect(Float64, kA_range)

    if seed !== nothing
        for (A_s, x_s) in seed
            push!(history, (A_s, x_s))
        end
    end

    idx = 1
    while idx <= length(kA_targets)
        kA = kA_targets[idx]
        A = kA / k
        A_target = A

        x0 = _ms_initial_guess(history, A, k, cm, K, n_sub, rhs_fn, N_F, T_lin, r_vals, integrator)

        # Build NonlinearProblem with custom Jacobian
        p = (; rhs_fn, cm, N_F, K, n_sub, A_target, integrator, nzmap)
        nlfunc = NLS.NonlinearFunction(_ms_residual!; jac=_ms_jac!, jac_prototype=copy(jac_proto))
        nlprob = NLS.NonlinearProblem(nlfunc, Float64.(x0), p)

        t0 = time()
        sol = NLS.solve(nlprob, nl_alg; abstol=method.tol, maxiters=method.maxiters)
        dt = round(time() - t0, digits=1)

        x_sol = sol.u
        resid = norm(sol.resid)
        conv = (sol.retcode == NLS.ReturnCode.Success)

        T_val = x_sol[end]
        dT = (T_val - T_lin) / T_lin * 100

        if conv
            ζ_full = expand_c1(cm, x_sol[1:N_free])
            ζ_min_T2 = ζ_full[1] - 2A
            kh_eff = abs(k * ζ_min_T2)
            push!(profiles, (kA=kA, A=A, T=T_val, dT=dT, zeta=ζ_full)::ProfileTuple)
            push!(history, (A, copy(x_sol)))
            verbose && @info "Continuation step" kA=round(kA,digits=4) kh=round(kh_eff,digits=4) dT=round(dT,digits=4) resid=round(resid,sigdigits=2) time_s=dt
            idx += 1
        else
            kA_prev = isempty(profiles) ? 0.0 : profiles[end].kA
            gap = kA - kA_prev
            if gap > 0.002
                kA_mid = (kA_prev + kA) / 2
                verbose && @info "Continuation step failed" kA=round(kA,digits=4) action="bisecting" kA_mid=round(kA_mid,digits=4)
                insert!(kA_targets, idx, kA_mid)
            else
                verbose && @info "Continuation step failed" kA=round(kA,digits=4) action="stopping" resid=round(resid,sigdigits=2) time_s=dt
                break
            end
        end
    end

    StandingWaveResult(profiles, r_vals, ν, k, T_lin)
end

# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC SEED CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

"""
    ms_seed(result::StandingWaveResult, K::Int, solver::HOSESolver;
            integrator=DecoupledIntegrator(solver=ODE.RK4(), n_steps=512),
            n_profiles::Int=2) → Vector{Tuple{Float64, Vector{Float64}}}

Build a multiple-shooting seed from converged profiles in `result`, suitable for
passing as the `seed` keyword to [`continuation`](@ref) with a different `K`.

Takes the last `n_profiles` converged profiles, integrates each forward through
`K-1` sub-intervals using the same RHS + integrator, and returns the `(A, x_ms)`
history vector that the continuation solver uses for Lagrange extrapolation.

This is the standard workflow for multi-phase sweeps: compute at low K, then
re-seed at higher K to push to steeper waves.

# Arguments
- `result::StandingWaveResult`: a completed continuation run
- `K::Int`: target number of shooting intervals for the next phase
- `solver::HOSESolver`: the assembled BIE system (must match `result`)
- `integrator::DecoupledIntegrator`: time integrator (default: RK4, 512 steps)
- `n_profiles::Int=2`: number of trailing profiles to seed from (≥2 enables
  linear extrapolation in the first step of the next phase)

# Returns
A `Vector{Tuple{Float64, Vector{Float64}}}` ready for `continuation(...; seed=...)`.

# Example
```julia
res_K4 = continuation(solver, MultipleShooting(K=4, n_steps=512); ν=1, kA_range=0.05:0.05:0.50)
seed = ms_seed(res_K4, 8, solver)
res_K8 = continuation(solver, MultipleShooting(K=8, n_steps=512); ν=1, kA_range=0.52:0.02:0.70, seed=seed)
```
"""
function ms_seed(result::StandingWaveResult, K::Int, solver::HOSESolver;
                 integrator::DecoupledIntegrator=DecoupledIntegrator(solver=ODE.RK4(), n_steps=512),
                 n_profiles::Int=2)
    ms_seed(result.profiles, result.k, K, solver; integrator, n_profiles)
end

"""
    ms_seed(profiles::Vector{<:NamedTuple}, k::Float64, K::Int, solver::HOSESolver;
            integrator=DecoupledIntegrator(solver=ODE.RK4(), n_steps=512),
            n_profiles::Int=2) → Vector{Tuple{Float64, Vector{Float64}}}

Build a seed from a raw profile vector (e.g. from `result.profiles`) and a known
wavenumber `k`. Useful when combining profiles from multiple continuation runs.
"""
function ms_seed(profiles::Vector{<:NamedTuple}, k::Float64, K::Int, solver::HOSESolver;
                 integrator::DecoupledIntegrator=DecoupledIntegrator(solver=ODE.RK4(), n_steps=512),
                 n_profiles::Int=2)
    isempty(profiles) && return Tuple{Float64, Vector{Float64}}[]

    cm = build_c1_map(solver.mesh)
    N_F = solver.mesh.n_sf
    rhs = HOSERhs(solver, cm)

    n = min(n_profiles, length(profiles))
    idx_start = length(profiles) - n + 1

    seeds = Vector{Tuple{Float64, Vector{Float64}}}(undef, n)
    @inbounds for i in 1:n
        p = profiles[idx_start + i - 1]
        A = p.kA / k
        ζ0_free = restrict_c1(cm, p.zeta)
        x_ms = _ms_build_guess_stable(ζ0_free, p.T, rhs, cm, N_F, K, integrator)
        seeds[i] = (A, x_ms)
    end
    seeds
end