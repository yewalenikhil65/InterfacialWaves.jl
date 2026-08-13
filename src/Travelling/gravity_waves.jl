# =============================================================================
# Gravity Waves — all variants (Original, Efficient, Analytical Jacobian)
# =============================================================================

# =============================================================================
# Original: full FFT, allocating
# =============================================================================

@views function GravityWaves(du, u, p)
    ξ, 𝞊, Ehw_val, N, h, d = p

    Y = u[1:end-1]
    F = u[end]

    Ŷ = fft(Y)

    Yξ = real(ifft(d .* Ŷ))
    Xξ = 1.0 .- real(ifft(h .* d .* Ŷ))

    J = @. Xξ^2 + Yξ^2

    ψξ = Yξ
    ϕξ = -real(ifft(h .* fft(ψξ)))

    Fϕ̂ξ = (1.0 ./ d) .* fft(ϕξ)
    Fϕ̂ξ[1] = 0.0 + im*0.0
    Fϕ̂ξ[N÷2 + 1] = 0.0 + im*0.0
    ϕ = real(ifft(Fϕ̂ξ))
    ϕ = ϕ .- solve(SampledIntegralProblem(ϕ .* Xξ, ξ), TrapezoidalRule()).u

    du[1:length(ξ)] .= @. ((-F^2)*(Xξ*ϕξ + Yξ*ψξ)/J) + ((F^2)*(ϕξ^2 + ψξ^2)/(2*J)) + Y
    du[length(ξ) + 1] = WaveEnergyParameter(ψξ, -ϕ, Y, Xξ, J, ξ, F, 0.0, Ehw_val) - 𝞊

    return nothing
end

# =============================================================================
# Efficient: rfft/irfft, zero-allocation spectral operations
# =============================================================================

"""
    GravityWavesEfficient(du, u, p)

Efficient counterpart of `GravityWaves` using `SpectralWorkspace`.

Pass `p = (ξ, 𝞊, Ehw_val, N, ws)` to NonlinearSolve.  
Note: `h` and `d` are no longer needed in `p` — they are stored inside `ws`.
"""
@views function GravityWavesEfficient(du, u, p)
    ξ, 𝞊, Ehw_val, N, ws = p
    Y = u[1:end-1]
    F = u[end]
    r = ws.r

    # Spectrum of Y
    transform!(ws, Y)

    # Spatial derivatives from Y-spectrum
    derivative!(ws, r[1], 1)                              # Yξ
    hilbert_derivative!(ws, r[2], 1)                      # H·d·Y
    @inbounds @simd for i in eachindex(r[2])
        r[2][i] = 1.0 - r[2][i]                          # Xξ = 1 - H(Yξ)
    end
    @inbounds @simd for i in eachindex(r[3], r[1], r[2])
        r[3][i] = r[2][i]^2 + r[1][i]^2                  # J = Xξ² + Yξ²
    end

    # ϕξ = -H(ψξ) = -H(Yξ)
    transform!(ws, r[1])                                  # spectrum of Yξ
    neg_hilbert!(ws, r[4])                                # ϕξ

    # ϕ = spectral_integrate(ϕξ) + ∫(ϕ·Xξ)dξ
    transform!(ws, r[4])                                  # spectrum of ϕξ
    integrate!(ws, r[5])                                  # ϕ (before mean correction)
    r[5] .-= trapezoid!(ws, r[5], r[2])                  # ϕ -= ∫ϕ·Xξ dξ

    Yξ, Xξ, J, ϕξ, ϕ = r[1], r[2], r[3], r[4], r[5]

    # Bernoulli equation
    @inbounds @simd for i in eachindex(ξ)
        du[i] = ((-F^2) * (Xξ[i] * ϕξ[i] + Yξ[i] * Yξ[i]) / J[i]) +
                ((F^2) * (ϕξ[i]^2 + Yξ[i]^2) / (2 * J[i])) + Y[i]
    end

    # Energy constraint
    @inbounds @simd for i in eachindex(r[5])
        r[5][i] = -r[5][i]
    end
    du[length(ξ) + 1] = energy!(ws, Yξ, r[5], Y, Xξ, J, F, 0.0, Ehw_val) - 𝞊

    return nothing
end

# =============================================================================
# Analytical Jacobian — maximum performance
# =============================================================================

"""
    GravityWavesJacobian!{T}

Precomputed operator matrices + workspace for computing the analytical Jacobian 
of the gravity wave Bernoulli equation. Used with NonlinearSolve's `jac` kwarg.

The Jacobian is structured as:
    ∂R/∂u = [∂R/∂Y  ∂R/∂F]

where ∂R/∂Y is decomposed using the chain rule through spectral operators:
    ∂R_i/∂Y_j = (∂R_i/∂Yξ_i)·D_ij + (∂R_i/∂Xξ_i)·(-HD_ij) + 
                 (∂R_i/∂ϕξ_i)·(NHD_ij) + (∂R_i/∂J_i)·(∂J_i/∂Y_j) + δ_ij

This gives an O(N²) Jacobian fill — same as one dense matrix multiply — avoiding 
the O(N² log N) cost of finite differences or the O(N³) cost of ForwardDiff through
dense matrices.
"""
struct GravityWavesJacobian!{T<:AbstractFloat}
    N::Int
    D::Matrix{T}       # derivative: Yξ = D·Y
    HD::Matrix{T}      # Hilbert-derivative: HD·Y (Xξ = 1 - HD·Y)  
    NHD::Matrix{T}     # neg-Hilbert of derivative: ϕξ = NHD·Y (i.e., -H(D·Y))
    IOp_NHD::Matrix{T} # integration of ϕξ operator: Iop·NHD (for ϕ base)
    trap_wt::Vector{T}
    simpson_wt::Vector{T}
    # Preallocated Jacobian buffer
    Jbuf::Matrix{T}
end

"""
    GravityWavesJacobian!(N; x=ξ, T=Float64)

Precompute all operator matrices needed for the analytical Jacobian.
"""
function GravityWavesJacobian!(N::Integer; x=nothing, T::Type{<:AbstractFloat}=Float64)
    if x === nothing
        x = [T(-1/2) + T(i)/N for i in 0:N-1]
    end
    length(x) == N || throw(DimensionMismatch("grid length must equal N"))

    # Full-spectrum operators
    k = vcat(0:(N÷2)-1, 0, (-N÷2)+1:-1)
    d = 2π * im .* T.(k)
    h = im .* T.(sign.(k))

    D   = zeros(T, N, N)
    HD  = zeros(T, N, N)
    NHD = zeros(T, N, N)
    IOp_NHD = zeros(T, N, N)

    e_j = zeros(Complex{T}, N)
    for j in 1:N
        fill!(e_j, zero(Complex{T}))
        e_j[j] = one(Complex{T})
        ê = fft(e_j)

        D[:, j]  = real(ifft(d .* ê))
        HD[:, j] = real(ifft(h .* d .* ê))

        # NHD = -H(D·Y) = ifft(-h · d · fft(Y))
        NHD[:, j] = real(ifft(-h .* d .* ê))

        # IOp_NHD: spectral integration of (-h·d·ê)
        nhd_spec = -h .* d .* ê
        int_spec = copy(nhd_spec)
        int_spec[1] = zero(Complex{T})
        for i in 2:N
            if i == N÷2 + 1
                int_spec[i] = zero(Complex{T})
            else
                int_spec[i] = nhd_spec[i] / d[i]
            end
        end
        IOp_NHD[:, j] = real(ifft(int_spec))
    end

    tw = T.(trap_weights(x))
    sw = T.(simpson_weights(x))
    Jbuf = zeros(T, N + 1, N + 1)

    return GravityWavesJacobian!{T}(N, D, HD, NHD, IOp_NHD, tw, sw, Jbuf)
end

"""
    gravity_wave_jacobian!(J, u, p)

Compute the analytical Jacobian of the gravity wave residual in-place.

`p = (ξ, 𝞊, Ehw_val, N, ws, jw)` where `jw::GravityWavesJacobian!`.

The residual is:
    R_i = (-F²)(Xξ_i·ϕξ_i + Yξ_i²)/J_i + (F²)(ϕξ_i² + Yξ_i²)/(2J_i) + Y_i
    R_{N+1} = energy - 𝞊

Derivatives w.r.t. Y_j:
    ∂Yξ_i/∂Y_j = D[i,j]
    ∂Xξ_i/∂Y_j = -HD[i,j]
    ∂ϕξ_i/∂Y_j = NHD[i,j]
    ∂ϕ_i/∂Y_j  = IOp_NHD[i,j] + mean_correction_jacobian
    ∂J_i/∂Y_j  = 2·Xξ_i·(-HD[i,j]) + 2·Yξ_i·D[i,j]
"""
function gravity_wave_jacobian!(Jac, u, p)
    ξ, 𝞊, Ehw_val, N, ws, jw = p

    Y = @view u[1:N]
    F = u[N+1]

    # --- Evaluate state quantities using the efficient workspace ---
    r = ws.r
    transform!(ws, Y)
    derivative!(ws, r[1], 1)                              # Yξ
    hilbert_derivative!(ws, r[2], 1)
    @inbounds @simd for i in eachindex(r[2])
        r[2][i] = 1.0 - r[2][i]                          # Xξ
    end
    @inbounds @simd for i in eachindex(r[3], r[1], r[2])
        r[3][i] = r[2][i]^2 + r[1][i]^2                  # J
    end
    transform!(ws, r[1])
    neg_hilbert!(ws, r[4])                                # ϕξ

    transform!(ws, r[4])
    integrate!(ws, r[5])                                  # ϕ_base (before mean correction)

    # Mean correction: c = -∫ϕ_base·Xξ dξ
    c = -trapezoid!(ws, r[5], r[2])
    # ϕ = ϕ_base + c
    @inbounds @simd for i in 1:N
        r[5][i] += c
    end

    Yξ  = r[1]
    Xξ  = r[2]
    J   = r[3]
    ϕξ  = r[4]
    ϕ   = r[5]

    F2 = F^2

    # --- Fill Jacobian: ∂R[1:N]/∂Y[1:N] ---
    @inbounds for j in 1:N
        for i in 1:N
            Ji = J[i]
            Ji2 = Ji * Ji

            # Partial derivatives of state vars w.r.t. Y_j at point i
            dYξ  = jw.D[i, j]
            dXξ  = -jw.HD[i, j]
            dϕξ  = jw.NHD[i, j]
            dJ   = 2.0 * Xξ[i] * dXξ + 2.0 * Yξ[i] * dYξ

            # ∂/∂Y_j of [ (-F²)(Xξ·ϕξ + Yξ²)/J ]
            num_A = Xξ[i] * ϕξ[i] + Yξ[i]^2
            dA = (-F2) * ((dXξ * ϕξ[i] + Xξ[i] * dϕξ + 2.0 * Yξ[i] * dYξ) * Ji - num_A * dJ) / Ji2

            # ∂/∂Y_j of [ (F²)(ϕξ² + Yξ²)/(2J) ]
            num_B = ϕξ[i]^2 + Yξ[i]^2
            dB = (F2 / 2.0) * ((2.0 * ϕξ[i] * dϕξ + 2.0 * Yξ[i] * dYξ) * Ji - num_B * dJ) / Ji2

            Jac[i, j] = dA + dB + (i == j ? 1.0 : 0.0)
        end
    end

    # --- ∂R[1:N]/∂F ---
    @inbounds for i in 1:N
        num_A = Xξ[i] * ϕξ[i] + Yξ[i]^2
        num_B = ϕξ[i]^2 + Yξ[i]^2
        Jac[i, N+1] = (-2.0 * F) * num_A / J[i] + (F) * num_B / J[i]
    end

    # --- ∂R[N+1]/∂Y[1:N] (energy constraint) ---
    # Mean correction: c = -Σ_k trap_wt[k] · ϕ_base[k] · Xξ[k]
    # ∂c/∂Y_j = -Σ_k trap_wt[k] · [ IOp_NHD[k,j]·Xξ[k] + ϕ_base[k]·(-HD[k,j]) ]
    ϕ_base = r[6]  # reuse buffer
    @inbounds @simd for i in 1:N
        ϕ_base[i] = ϕ[i] - c  # recover ϕ_base
    end

    @inbounds for j in 1:N
        # Compute ∂c/∂Y_j
        dc = 0.0
        for k in 1:N
            dc -= jw.trap_wt[k] * (jw.IOp_NHD[k, j] * Xξ[k] + ϕ_base[k] * (-jw.HD[k, j]))
        end

        # ∂E/∂Y_j = Σ_i simpson_wt[i] · ∂I_i/∂Y_j
        dE = 0.0
        for i in 1:N
            dYξ_ij = jw.D[i, j]
            dXξ_ij = -jw.HD[i, j]
            dϕ_ij  = jw.IOp_NHD[i, j] + dc  # ∂ϕ_i/∂Y_j

            dI = (1.0 / Ehw_val) * (
                (F2 / 2.0) * (dYξ_ij * (-ϕ[i]) + Yξ[i] * (-dϕ_ij)) +
                0.5 * (dXξ_ij * Y[i]^2 + (i == j ? Xξ[i] * 2.0 * Y[i] : 0.0))
            )
            dE += jw.simpson_wt[i] * dI
        end
        Jac[N+1, j] = dE
    end

    # --- ∂R[N+1]/∂F ---
    dE_dF = 0.0
    @inbounds for i in 1:N
        dE_dF += jw.simpson_wt[i] * (1.0 / Ehw_val) * F * Yξ[i] * (-ϕ[i])
    end
    Jac[N+1, N+1] = dE_dF

    return nothing
end

"""
    GravityWavesAnalyticalJac(du, u, p)

Gravity wave residual with the same interface as `GravityWavesEfficient`, 
but designed to be used with the analytical Jacobian.

Pass `p = (ξ, 𝞊, Ehw_val, N, ws, jw)` where `jw::GravityWavesJacobian!`.
The residual computation uses the efficient `ws` workspace; the `jw` is only 
used by the Jacobian function.
"""
@views function GravityWavesAnalyticalJac(du, u, p)
    ξ, 𝞊, Ehw_val, N, ws, jw = p

    Y = u[1:end-1]
    F = u[end]
    r = ws.r

    # Same computation as GravityWavesEfficient
    transform!(ws, Y)
    derivative!(ws, r[1], 1)
    hilbert_derivative!(ws, r[2], 1)
    @inbounds @simd for i in eachindex(r[2])
        r[2][i] = 1.0 - r[2][i]
    end
    @inbounds @simd for i in eachindex(r[3], r[1], r[2])
        r[3][i] = r[2][i]^2 + r[1][i]^2
    end

    transform!(ws, r[1])
    neg_hilbert!(ws, r[4])

    transform!(ws, r[4])
    integrate!(ws, r[5])
    r[5] .-= trapezoid!(ws, r[5], r[2])

    Yξ, Xξ, J, ϕξ, ϕ = r[1], r[2], r[3], r[4], r[5]

    @inbounds @simd for i in eachindex(ξ)
        du[i] = ((-F^2) * (Xξ[i] * ϕξ[i] + Yξ[i] * Yξ[i]) / J[i]) +
                ((F^2) * (ϕξ[i]^2 + Yξ[i]^2) / (2 * J[i])) + Y[i]
    end

    @inbounds @simd for i in eachindex(r[5])
        r[5][i] = -r[5][i]
    end
    du[length(ξ) + 1] = energy!(ws, Yξ, r[5], Y, Xξ, J, F, 0.0, Ehw_val) - 𝞊

    return nothing
end

# =============================================================================
# Phase-fixed analytical formulation — even Stokes-wave subspace
# =============================================================================

"""
    GravityWavePhaseWorkspace(N; T=Float64)

Workspace for the phase-fixed gravity-wave formulation.  It parameterizes an
N-point, periodic, even profile by its values at `ξ ∈ [-1/2, 0]`; the values on
`(0, 1/2)` are filled by reflection.  This removes all odd (translation/phase)
directions from Newton's linear system while retaining the full residual and
analytical Jacobian internally.
"""
struct GravityWavePhaseWorkspace{T<:AbstractFloat}
    N::Int
    M::Int                         # independent even-profile values: N ÷ 2 + 1
    u_full::Vector{T}              # [Y_full; F]
    r_full::Vector{T}              # full gravity-wave residual
    jac_full::Matrix{T}            # full analytical Jacobian
end

function GravityWavePhaseWorkspace(N::Integer; T::Type{<:AbstractFloat}=Float64)
    iseven(N) || throw(ArgumentError("phase-fixed gravity waves require an even N"))
    M = N ÷ 2 + 1
    return GravityWavePhaseWorkspace{T}(N, M, zeros(T, N + 1),
        zeros(T, N + 1), zeros(T, N + 1, N + 1))
end

"""Expand reduced even-profile unknowns `[Y[-1/2:0]; F]` to `[Y_full; F]`."""
@inline function _expand_even_gravity_state!(u_full, u_reduced,
        phase_ws::GravityWavePhaseWorkspace)
    N, M = phase_ws.N, phase_ws.M
    @inbounds begin
        for i in 1:M
            u_full[i] = u_reduced[i]
        end
        # Points i=1 (-1/2) and i=M (0) are self-reflections.  Every other
        # negative-grid point is mirrored onto its positive-grid counterpart.
        for i in 2:M-1
            u_full[N - i + 2] = u_reduced[i]
        end
        u_full[N + 1] = u_reduced[M + 1]
    end
    return u_full
end

"""
    GravityWavesPhaseFixedAnalyticalJac(du, u, p)

Phase-fixed gravity-wave residual on the even Stokes-wave subspace.  The
underlying physics residual remains the full `GravityWavesAnalyticalJac`
residual; only its independent even rows are retained.

Pass `p = (ξ, 𝞊, Ehw_val, N, ws, jw, phase_ws)`.
"""
function GravityWavesPhaseFixedAnalyticalJac(du, u, p)
    ξ, 𝞊, Ehw_val, N, ws, jw, phase_ws = p
    _expand_even_gravity_state!(phase_ws.u_full, u, phase_ws)

    full_p = (ξ, 𝞊, Ehw_val, N, ws, jw)
    GravityWavesAnalyticalJac(phase_ws.r_full, phase_ws.u_full, full_p)

    @inbounds for i in 1:phase_ws.M
        du[i] = phase_ws.r_full[i]
    end
    du[phase_ws.M + 1] = phase_ws.r_full[N + 1]
    return nothing
end

"""
    gravity_wave_phase_fixed_jacobian!(Jac, u, p)

Project the full analytical gravity-wave Jacobian onto the even-profile
subspace.  A reduced variable at an interior negative-grid point changes both
that point and its reflected positive-grid point, so its column is the sum of
the corresponding full-Jacobian columns.
"""
function gravity_wave_phase_fixed_jacobian!(Jac, u, p)
    ξ, 𝞊, Ehw_val, N, ws, jw, phase_ws = p
    M = phase_ws.M
    _expand_even_gravity_state!(phase_ws.u_full, u, phase_ws)

    full_p = (ξ, 𝞊, Ehw_val, N, ws, jw)
    gravity_wave_jacobian!(phase_ws.jac_full, phase_ws.u_full, full_p)
    Jfull = phase_ws.jac_full

    @inbounds for j in 1:M
        # Endpoints ξ=-1/2 and ξ=0 have no distinct reflected counterpart.
        jmirror = (j == 1 || j == M) ? 0 : N - j + 2
        for i in 1:M
            Jac[i, j] = Jfull[i, j] + (jmirror == 0 ? 0.0 : Jfull[i, jmirror])
        end
        Jac[M + 1, j] = Jfull[N + 1, j] +
            (jmirror == 0 ? 0.0 : Jfull[N + 1, jmirror])
    end

    @inbounds for i in 1:M
        Jac[i, M + 1] = Jfull[i, N + 1]
    end
    Jac[M + 1, M + 1] = Jfull[N + 1, N + 1]
    return nothing
end

function _expand_even_gravity_solutions(solutions_reduced,
        phase_ws::GravityWavePhaseWorkspace{T}) where {T}
    solutions = Vector{Vector{T}}(undef, length(solutions_reduced))
    for i in eachindex(solutions_reduced)
        _expand_even_gravity_state!(phase_ws.u_full, solutions_reduced[i], phase_ws)
        solutions[i] = copy(phase_ws.u_full)
    end
    return solutions
end

# =============================================================================
# solve() dispatch for GravityWaveProblem
# =============================================================================

"""
    solve(prob::GravityWaveProblem, method::Collocation;
          abstol=1e-10, maxiters=200, phase_fixed=iseven(prob.N), callback=nothing)

Solve a pure gravity wave problem using the conformal-mapping collocation method
with energy continuation.

`method = Collocation()` uses the analytical Jacobian (default).
`method = Collocation(; jacobian=:finitediff)` uses `AutoFiniteDiff()` with the efficient rfft residual.
`method = Collocation(; jacobian=:krylov)` uses matrix-free Newton-Krylov (GMRES) — never forms the Jacobian.

For even `N`, `phase_fixed=true` (default) solves on the even Stokes-wave subspace,
removing the translational null mode. (Not used with `:krylov` — the Krylov solver
operates on the full system since it does not need to invert a dense matrix.)
"""
function CommonSolve.solve(prob::GravityWaveProblem{T}, method::Collocation;
        abstol::Real=1e-10, maxiters::Int=200,
        phase_fixed::Bool=iseven(prob.N), callback=nothing) where {T}

    N = prob.N
    if phase_fixed && isodd(N)
        throw(ArgumentError("phase_fixed=true requires an even N"))
    end
    grid = WaveGrid(N; T=T)

    # Build workspaces
    ws = SpectralWorkspace(grid)

    # Initial condition (small-amplitude linear wave)
    ϵ₀ = T(1e-5)
    Y = @. ϵ₀ * cos(2π * grid.ξ)
    F = sqrt(T(1) / T(2π))  # linear dispersion (B=0)

    # Compute initial energy and prepend to schedule
    𝞊₀ = initial_energy(ws, Y, F, T(0), Ehw)
    𝜖 = vcat(𝞊₀, prob.𝜖)

    if method.jacobian == :finitediff
        # Efficient residual + AutoFiniteDiff
        solutions, retcodes = continuation(
            GravityWavesEfficient, vcat(Y, F), 𝜖,
            ε -> (grid.ξ, ε, Ehw, N, ws);
            solver = NewtonRaphson(; autodiff=AutoFiniteDiff()),
            abstol = abstol, maxiters = maxiters,
            callback = callback
        )
        return WaveSolution{T, typeof(prob)}(grid, solutions, retcodes, 𝜖, prob)
    end

    if method.jacobian == :krylov
        # Matrix-free Newton-Krylov: GMRES for the linear solve, Jv via FD
        solutions, retcodes = continuation(
            GravityWavesEfficient, vcat(Y, F), 𝜖,
            ε -> (grid.ξ, ε, Ehw, N, ws);
            solver = NewtonRaphson(; linsolve=KrylovJL_GMRES(),
                                     autodiff=AutoFiniteDiff()),
            abstol = abstol, maxiters = maxiters,
            callback = callback
        )
        return WaveSolution{T, typeof(prob)}(grid, solutions, retcodes, 𝜖, prob)
    end

    # Analytical Jacobian path (default)
    jw = GravityWavesJacobian!(N; x=grid.ξ, T=T)

    if phase_fixed
        phase_ws = GravityWavePhaseWorkspace(N; T=T)
        nf = NonlinearFunction{true}(GravityWavesPhaseFixedAnalyticalJac;
            jac=gravity_wave_phase_fixed_jacobian!)
        u0 = vcat(Y[1:phase_ws.M], F)

        solutions_reduced, retcodes = continuation(
            nf, u0, 𝜖,
            ε -> (grid.ξ, ε, Ehw, N, ws, jw, phase_ws);
            solver = NewtonRaphson(),
            abstol = abstol, maxiters = maxiters,
            callback = callback
        )
        solutions = _expand_even_gravity_solutions(solutions_reduced, phase_ws)
        return WaveSolution{T, typeof(prob)}(grid, solutions, retcodes, 𝜖, prob)
    end

    # Unconstrained full-grid analytical Jacobian
    nf = NonlinearFunction{true}(GravityWavesAnalyticalJac; jac=gravity_wave_jacobian!)
    solutions, retcodes = continuation(
        nf, vcat(Y, F), 𝜖,
        ε -> (grid.ξ, ε, Ehw, N, ws, jw);
        solver = NewtonRaphson(),
        abstol = abstol, maxiters = maxiters,
        callback = callback
    )
    return WaveSolution{T, typeof(prob)}(grid, solutions, retcodes, 𝜖, prob)
end

"""
    solve(prob::GravityWaveProblem, method::LonguetHiggins;
          abstol=1e-10, maxiters=70)

Solve a pure gravity wave problem using the Longuet-Higgins (1978) Toeplitz
formulation with steepness (`ak`) continuation.

`method = LH()` or `LH(; jacobian=:analytical)` uses the hand-coded Jacobian.
`method = LH(; jacobian=:finitediff)` uses `AutoFiniteDiff()`.
`method = LH(; jacobian=:krylov)` uses matrix-free Newton-Krylov (GMRES).

# Returns
A `WaveSolution` with Fourier coefficients (full a₀ convention internally)
and phase speeds accessible via `sol.c`, `sol.ak`, `sol.a`.
"""
function CommonSolve.solve(prob::GravityWaveProblem{T}, method::LonguetHiggins;
        abstol::Real=1e-10, maxiters::Int=70) where {T}

    N = prob.N
    M = N + 1  # number of unknowns

    ak_schedule = prob.ak_schedule
    if isempty(ak_schedule)
        throw(ArgumentError(
            "GravityWaveProblem has no steepness schedule. " *
            "Use GravityWaveProblem(N; ak=0.43) to set a steepness target for LonguetHiggins()."))
    end

    # Initial condition: small-amplitude Stokes expansion at ak = 0.01
    ak_init = T(0.01)
    x0 = zeros(T, M)
    x0[2] = ak_init
    x0[3] = T(0.5) * ak_init^2

    # Build solver depending on jacobian choice
    if method.jacobian == :analytical
        jac_fn! = (J, x, p) -> LonguetHigginsJacobian!(J, x, p)
        nf = NonlinearFunction{true}(LonguetHigginsResidual!; jac=jac_fn!)
        solver = NewtonRaphson()
    elseif method.jacobian == :krylov
        nf = NonlinearFunction{true}(LonguetHigginsResidual!)
        solver = NewtonRaphson(; linsolve=KrylovJL_GMRES(),
                                 autodiff=AutoFiniteDiff())
    else
        nf = NonlinearFunction{true}(LonguetHigginsResidual!)
        solver = NewtonRaphson(; autodiff=AutoFiniteDiff())
    end

    # Full schedule: start at ak_init, then follow user schedule
    ak_full = vcat(ak_init, ak_schedule)

    # Run continuation
    n_steps = length(ak_full)
    solutions = Vector{Vector{T}}(undef, n_steps)
    retcodes = Vector{Any}(undef, n_steps)
    ak_values = Vector{T}(undef, n_steps)

    x_current = copy(x0)

    for i in 1:n_steps
        p_i = (ak=ak_full[i],)
        prob_nl = NonlinearProblem(nf, x_current, p_i)
        sol = solve(prob_nl, solver; abstol=abstol, maxiters=maxiters)

        solutions[i] = copy(sol.u)
        retcodes[i] = sol.retcode
        ak_values[i] = ak_full[i]

        if sol.retcode == ReturnCode.Success || sol.retcode == ReturnCode.Stalled
            x_current = sol.u
        else
            @warn "LH step $i (ak=$(ak_full[i])): solver returned $(sol.retcode)"
        end
    end

    return WaveSolution{T, typeof(prob)}(nothing, solutions, retcodes, ak_values, prob)
end

# Convenience: solve(prob) without method defaults to Collocation()
function CommonSolve.solve(prob::GravityWaveProblem{T}; kwargs...) where {T}
    return CommonSolve.solve(prob, Collocation(); kwargs...)
end

# =============================================================================
# Third-order Stokes expansion for pure gravity waves
# =============================================================================

"""
    Third()

Third-order Stokes expansion for deep-water gravity waves.
Returns a `WaveSolution` with Fourier coefficients stored for the profile:

```math
\\eta(x) = a\\cos(kx) + \\frac{ak}{2}\\cos(2kx) + \\frac{3(ak)^2}{8}\\cos(3kx)
```

Use with `solve(GravityProblem(N; ak=...), Third())`.
"""
struct Third end

function CommonSolve.solve(prob::GravityWaveProblem{T}, ::Third) where {T}
    ak_target = isempty(prob.ak_schedule) ? T(0.1) : prob.ak_schedule[end]
    N = prob.N

    # Third-order Stokes coefficients (deep water, k=1 normalization)
    # η = a₁cos(x) + a₂cos(2x) + a₃cos(3x)
    # where a₁ = ak, a₂ = ak²/2, a₃ = 3ak³/8
    a1 = ak_target
    a2 = ak_target^2 / 2
    a3 = 3 * ak_target^3 / 8

    # Phase speed: c²/c₀² = 1 + ak² (to third order)
    # In LH normalization c₀ = 1, so c = sqrt(1 + ak²)
    c = sqrt(one(T) + ak_target^2)

    # Store as LH-style coefficients: [H₀/2, H₁, H₂, H₃, 0, 0, ...]
    coeffs = zeros(T, N + 1)
    coeffs[1] = zero(T)   # H₀/2 = 0 (zero mean)
    coeffs[2] = a1         # H₁ = ak
    if N >= 2
        coeffs[3] = a2     # H₂ = ak²/2
    end
    if N >= 3
        coeffs[4] = a3     # H₃ = 3ak³/8
    end

    # Pack as solution vector (LH format)
    u = copy(coeffs)
    u[1] *= 2  # undo the a₀/2 convention for storage

    return WaveSolution{T, typeof(prob)}(
        nothing, [u], Any[ReturnCode.Success], T[ak_target], prob)
end
