# =============================================================================
# Gravity-Capillary Waves — Original and Efficient variants
# =============================================================================

# =============================================================================
# Original: full FFT, allocating
# =============================================================================

@views function GravityCapillaryWaves(du, u, p)
    B, ξ, 𝞊, Ehw_val, N, h, d = p 

    Y = u[1:end-1]      # Interface  
    F = u[end]          # Froude Number

    Ŷ = fft(Y);         # fft(interface)

    Yξ = real(ifft(d .* Ŷ ))
    Yξξ = real(ifft(d.^2 .* Ŷ ))
    Xξ = 1.0 .- real(ifft(h.* d .* Ŷ ))
    Xξξ = -real(ifft(h.* d.^2 .* Ŷ ))

    J = @. Xξ^2 + Yξ^2  
    
    ψξ = Yξ
    ϕξ = -real(ifft(h .* fft(ψξ)))
    
    Fϕ̂ξ = (1 ./ d).*fft(ϕξ)
    Fϕ̂ξ[1] = 0.0 + im*0.0
    Fϕ̂ξ[N÷2 + 1] = 0.0 + im*0.0
    ϕ = real(ifft(Fϕ̂ξ))
    ϕ = ϕ .- solve(SampledIntegralProblem(ϕ .* Xξ , ξ), TrapezoidalRule()).u
   
    du[1:length(ξ)] .= @. ((-F^2)*(Xξ*ϕξ + Yξ*ψξ) /J) + ((F^2)*(ϕξ^2 + ψξ^2)/(2*J)) + Y + (B*(Yξ*Xξξ - Yξξ*Xξ)/(J^(3/2)))
    du[length(ξ) + 1] = WaveEnergyParameter(ψξ, -ϕ, Y, Xξ, J, ξ, F, B, Ehw_val) - 𝞊
    
    return nothing
end 

# =============================================================================
# Efficient: rfft/irfft, zero-allocation spectral operations
# =============================================================================

"""
    GravityCapillaryWavesEfficient(du, u, p)

Efficient counterpart of `GravityCapillaryWaves` using `SpectralWorkspace`.

Pass `p = (B, ξ, 𝞊, Ehw_val, N, ws)` to NonlinearSolve.
"""
@views function GravityCapillaryWavesEfficient(du, u, p)
    B, ξ, 𝞊, Ehw_val, N, ws = p
    Y = u[1:end-1]
    F = u[end]
    r = ws.r

    # Spectrum of Y
    transform!(ws, Y)

    # Spatial derivatives from Y-spectrum
    derivative!(ws, r[1], 1)                              # Yξ
    derivative!(ws, r[2], 2)                              # Yξξ
    hilbert_derivative!(ws, r[3], 1)                      # H·d·Y
    @inbounds @simd for i in eachindex(r[3])
        r[3][i] = 1.0 - r[3][i]                          # Xξ = 1 - H(Yξ)
    end
    hilbert_derivative!(ws, r[4], 2)                      # H·d²·Y
    @inbounds @simd for i in eachindex(r[4])
        r[4][i] = -r[4][i]                               # Xξξ = -H(Yξξ)
    end
    @inbounds @simd for i in eachindex(r[5], r[1], r[3])
        r[5][i] = r[3][i]^2 + r[1][i]^2                  # J = Xξ² + Yξ²
    end

    # ϕξ = -H(ψξ) = -H(Yξ)
    transform!(ws, r[1])                                  # spectrum of Yξ
    neg_hilbert!(ws, r[6])                                # ϕξ

    # ϕ = spectral_integrate(ϕξ) + ∫(ϕ·Xξ)dξ
    transform!(ws, r[6])                                  # spectrum of ϕξ
    integrate!(ws, r[7])                                  # ϕ (before mean correction)
    r[7] .-= trapezoid!(ws, r[7], r[3])                  # ϕ -= ∫ϕ·Xξ dξ

    Yξ, Yξξ, Xξ, Xξξ, J, ϕξ, ϕ = r[1], r[2], r[3], r[4], r[5], r[6], r[7]

    # Bernoulli equation with capillary term
    @inbounds @simd for i in eachindex(ξ)
        du[i] = ((-F^2) * (Xξ[i] * ϕξ[i] + Yξ[i] * Yξ[i]) / J[i]) +
                ((F^2) * (ϕξ[i]^2 + Yξ[i]^2) / (2 * J[i])) + Y[i] +
                B * (Yξ[i] * Xξξ[i] - Yξξ[i] * Xξ[i]) / J[i]^(3/2)
    end

    # Energy constraint
    @inbounds @simd for i in eachindex(r[7])
        r[7][i] = -r[7][i]
    end
    du[length(ξ) + 1] = energy!(ws, Yξ, r[7], Y, Xξ, J, F, B, Ehw_val) - 𝞊

    return nothing
end

# =============================================================================
# solve() dispatch for GravityCapillaryWaveProblem
# =============================================================================

# =============================================================================
# Analytical Jacobian for Gravity-Capillary Waves
# =============================================================================

"""
    GravityCapillaryJacobian!{T}

Precomputed operator matrices for the gravity-capillary analytical Jacobian.
Extends the pure-gravity workspace with the second-derivative operators needed
for the curvature term B·(Yξ·Xξξ - Yξξ·Xξ)/J^{3/2}.
"""
struct GravityCapillaryJacobian!{T<:AbstractFloat}
    N::Int
    D::Matrix{T}        # ∂Yξ/∂Y
    D2::Matrix{T}       # ∂Yξξ/∂Y (second derivative)
    HD::Matrix{T}       # ∂(1-Xξ)/∂Y = H·D  (so ∂Xξ/∂Y = -HD)
    HD2::Matrix{T}      # ∂(-Xξξ)/∂Y = H·D²  (so ∂Xξξ/∂Y = -HD2)
    NHD::Matrix{T}      # ∂ϕξ/∂Y = -H·D
    IOp_NHD::Matrix{T}  # integration of NHD (for ϕ)
    trap_wt::Vector{T}
    simpson_wt::Vector{T}
end

"""
    GravityCapillaryJacobian!(N; x=ξ, T=Float64)

Precompute operator matrices for the gravity-capillary analytical Jacobian.
"""
function GravityCapillaryJacobian!(N::Integer; x=nothing, T::Type{<:AbstractFloat}=Float64)
    if x === nothing
        x = [T(-1/2) + T(i)/N for i in 0:N-1]
    end
    length(x) == N || throw(DimensionMismatch("grid length must equal N"))

    # WaveGrid's first-derivative convention zeroes the Nyquist mode because
    # a real Nyquist coefficient has no real first derivative on this grid.
    # The rFFT residual nevertheless retains its real second derivative, so
    # D² must use |k_Nyq|=N/2 even though D itself has k_Nyq=0.
    k_first = vcat(0:(N÷2)-1, 0, (-N÷2)+1:-1)
    k_second = vcat(0:(N÷2)-1, N÷2, (-N÷2)+1:-1)
    d = 2π * im .* T.(k_first)
    d2 = (2π * im) .^ 2 .* T.(k_second).^2
    h = im .* T.(sign.(k_first))

    D   = zeros(T, N, N)
    D2  = zeros(T, N, N)
    HD  = zeros(T, N, N)
    HD2 = zeros(T, N, N)
    NHD = zeros(T, N, N)
    IOp_NHD = zeros(T, N, N)

    e_j = zeros(Complex{T}, N)
    for j in 1:N
        fill!(e_j, zero(Complex{T}))
        e_j[j] = one(Complex{T})
        ê = fft(e_j)

        D[:, j]   = real(ifft(d .* ê))
        D2[:, j]  = real(ifft(d2 .* ê))
        HD[:, j]  = real(ifft(h .* d .* ê))
        HD2[:, j] = real(ifft(h .* d2 .* ê))
        NHD[:, j] = real(ifft(-h .* d .* ê))

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

    return GravityCapillaryJacobian!{T}(N, D, D2, HD, HD2, NHD, IOp_NHD, tw, sw)
end

"""
    gravity_capillary_jacobian!(Jac, u, p)

Analytical Jacobian of the gravity-capillary wave residual.

`p = (B, ξ, 𝞊, Ehw_val, N, ws, jw)` where `jw::GravityCapillaryJacobian!`.

Residual:
    R_i = (-F²)(Xξ·ϕξ + Yξ²)/J + (F²/2)(ϕξ² + Yξ²)/J + Y + B·κ/J^{3/2}
where κ = Yξ·Xξξ - Yξξ·Xξ (signed curvature numerator).
"""
function gravity_capillary_jacobian!(Jac, u, p)
    B, ξ, 𝞊, Ehw_val, N, ws, jw = p

    Y = @view u[1:N]
    F = u[N+1]
    r = ws.r

    # Evaluate state quantities
    transform!(ws, Y)
    derivative!(ws, r[1], 1)                              # Yξ
    derivative!(ws, r[2], 2)                              # Yξξ
    hilbert_derivative!(ws, r[3], 1)
    @inbounds @simd for i in eachindex(r[3])
        r[3][i] = 1.0 - r[3][i]                          # Xξ
    end
    hilbert_derivative!(ws, r[4], 2)
    @inbounds @simd for i in eachindex(r[4])
        r[4][i] = -r[4][i]                               # Xξξ
    end
    @inbounds @simd for i in eachindex(r[5], r[1], r[3])
        r[5][i] = r[3][i]^2 + r[1][i]^2                  # J
    end

    # ϕξ
    transform!(ws, r[1])
    neg_hilbert!(ws, r[6])                                # ϕξ

    # ϕ
    transform!(ws, r[6])
    integrate!(ws, r[7])
    c_mean = -trapezoid!(ws, r[7], r[3])
    @inbounds @simd for i in 1:N
        r[7][i] += c_mean
    end

    Yξ  = r[1]; Yξξ = r[2]; Xξ = r[3]; Xξξ = r[4]; J = r[5]; ϕξ = r[6]; ϕ = r[7]
    F2 = F^2

    # --- Fill Jacobian: ∂R[1:N]/∂Y[1:N] ---
    @inbounds for j in 1:N
        for i in 1:N
            Ji = J[i]
            Ji2 = Ji * Ji
            sqrtJi = sqrt(Ji)
            Ji32 = Ji * sqrtJi      # J^{3/2}
            Ji52 = Ji2 * sqrtJi     # J^{5/2}

            # Partial derivatives of state vars w.r.t. Y_j
            dYξ  = jw.D[i, j]
            dYξξ = jw.D2[i, j]
            dXξ  = -jw.HD[i, j]
            dXξξ = -jw.HD2[i, j]
            dϕξ  = jw.NHD[i, j]
            dJ   = 2.0 * Xξ[i] * dXξ + 2.0 * Yξ[i] * dYξ

            # Pure gravity terms (same as gravity_wave_jacobian!)
            num_A = Xξ[i] * ϕξ[i] + Yξ[i]^2
            dA = (-F2) * ((dXξ * ϕξ[i] + Xξ[i] * dϕξ + 2.0 * Yξ[i] * dYξ) * Ji - num_A * dJ) / Ji2

            num_B = ϕξ[i]^2 + Yξ[i]^2
            dB = (F2 / 2.0) * ((2.0 * ϕξ[i] * dϕξ + 2.0 * Yξ[i] * dYξ) * Ji - num_B * dJ) / Ji2

            # Curvature term: B·κ/J^{3/2} where κ = Yξ·Xξξ - Yξξ·Xξ
            κ = Yξ[i] * Xξξ[i] - Yξξ[i] * Xξ[i]
            dκ = dYξ * Xξξ[i] + Yξ[i] * dXξξ - dYξξ * Xξ[i] - Yξξ[i] * dXξ
            # ∂/∂Y_j [κ/J^{3/2}] = dκ/J^{3/2} - (3/2)·κ·dJ/J^{5/2}
            dC = B * (dκ / Ji32 - 1.5 * κ * dJ / Ji52)

            Jac[i, j] = dA + dB + dC + (i == j ? 1.0 : 0.0)
        end
    end

    # --- ∂R[1:N]/∂F ---
    @inbounds for i in 1:N
        num_A = Xξ[i] * ϕξ[i] + Yξ[i]^2
        num_B = ϕξ[i]^2 + Yξ[i]^2
        # Curvature term doesn't depend on F
        Jac[i, N+1] = (-2.0 * F) * num_A / J[i] + F * num_B / J[i]
    end

    # --- ∂R[N+1]/∂Y[1:N] (energy constraint) ---
    ϕ_base = r[8]
    @inbounds @simd for i in 1:N
        ϕ_base[i] = ϕ[i] - c_mean
    end

    @inbounds for j in 1:N
        # ∂c/∂Y_j (mean correction)
        dc = 0.0
        for k in 1:N
            dc -= jw.trap_wt[k] * (jw.IOp_NHD[k, j] * Xξ[k] + ϕ_base[k] * (-jw.HD[k, j]))
        end

        # ∂E/∂Y_j (energy constraint with capillary term)
        dE = 0.0
        for i in 1:N
            dYξ_ij = jw.D[i, j]
            dXξ_ij = -jw.HD[i, j]
            dϕ_ij  = jw.IOp_NHD[i, j] + dc
            dJ_ij  = 2.0 * Xξ[i] * dXξ_ij + 2.0 * Yξ[i] * dYξ_ij

            # Energy integrand: (1/Ehw)[(F²/2)ψξ·ϕ + B(√J - Xξ) + (1/2)Xξ·Y²]
            # where ψξ = Yξ and ϕ is stored as -ϕ in the energy call
            dI = (1.0 / Ehw_val) * (
                (F2 / 2.0) * (dYξ_ij * (-ϕ[i]) + Yξ[i] * (-dϕ_ij)) +
                B * (dJ_ij / (2.0 * sqrt(J[i])) - dXξ_ij) +
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
    GravityCapillaryWavesAnalyticalJac(du, u, p)

Gravity-capillary wave residual designed to be paired with `gravity_capillary_jacobian!`.
Pass `p = (B, ξ, 𝞊, Ehw_val, N, ws, jw)`.
"""
@views function GravityCapillaryWavesAnalyticalJac(du, u, p)
    B, ξ, 𝞊, Ehw_val, N, ws, jw = p

    Y = u[1:end-1]
    F = u[end]
    r = ws.r

    transform!(ws, Y)
    derivative!(ws, r[1], 1)                              # Yξ
    derivative!(ws, r[2], 2)                              # Yξξ
    hilbert_derivative!(ws, r[3], 1)
    @inbounds @simd for i in eachindex(r[3])
        r[3][i] = 1.0 - r[3][i]                          # Xξ
    end
    hilbert_derivative!(ws, r[4], 2)
    @inbounds @simd for i in eachindex(r[4])
        r[4][i] = -r[4][i]                               # Xξξ
    end
    @inbounds @simd for i in eachindex(r[5], r[1], r[3])
        r[5][i] = r[3][i]^2 + r[1][i]^2                  # J
    end

    transform!(ws, r[1])
    neg_hilbert!(ws, r[6])                                # ϕξ
    transform!(ws, r[6])
    integrate!(ws, r[7])
    r[7] .-= trapezoid!(ws, r[7], r[3])

    Yξ = r[1]; Yξξ = r[2]; Xξ = r[3]; Xξξ = r[4]; J = r[5]; ϕξ = r[6]; ϕ = r[7]

    @inbounds @simd for i in eachindex(ξ)
        du[i] = ((-F^2) * (Xξ[i] * ϕξ[i] + Yξ[i]^2) / J[i]) +
                ((F^2) * (ϕξ[i]^2 + Yξ[i]^2) / (2 * J[i])) + Y[i] +
                B * (Yξ[i] * Xξξ[i] - Yξξ[i] * Xξ[i]) / J[i]^(3/2)
    end

    @inbounds @simd for i in eachindex(r[7])
        r[7][i] = -r[7][i]
    end
    du[length(ξ) + 1] = energy!(ws, Yξ, r[7], Y, Xξ, J, F, B, Ehw_val) - 𝞊

    return nothing
end
# =============================================================================
# Gravity-capillary continuation schedule
# =============================================================================

"""
    gravity_capillary_continuation_schedule(ε_max)

Proven `TravellingStokes` energy-continuation schedule for the full,
translationally invariant gravity-capillary system:

```julia
[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.11:0.01:0.99...]
```

The solve dispatch prepends the energy of the linear initial wave. The schedule
is capped at `min(ε_max, 0.99)`, exactly as in the established workflow.
"""
function gravity_capillary_continuation_schedule(ε_max::T) where {T<:AbstractFloat}
    ε_cap = min(ε_max, T(0.99))
    ε_cap > zero(T) || return T[]

    schedule = T[]
    for ε in (T(1e-7), T(1e-6), T(1e-5), T(1e-4), T(1e-3), T(1e-2), T(1e-1))
        ε <= ε_cap && push!(schedule, ε)
    end
    if ε_cap >= T(0.11)
        append!(schedule, T.(T(0.11):T(0.01):ε_cap))
        # Ensure a non-grid endpoint is retained.
        if schedule[end] < ε_cap
            push!(schedule, ε_cap)
        end
    elseif isempty(schedule) || schedule[end] < ε_cap
        push!(schedule, ε_cap)
    end
    return schedule
end
# =============================================================================
# Pure-gravity seeded gravity-capillary solves
# =============================================================================

"""
    _solve_gravity_capillary_from_gravity_seeds(prob, grid, ws;
                                                 jacobian=:finitediff,
                                                 abstol, maxiters, callback)

Implement the target-state workflow for fixed Bond number `B`:

1. continue the *pure-gravity* collocation branch over `prob.𝜖`;
2. take the final pure-gravity state at `ε_target = last(prob.𝜖)`;
3. use that state as the initial iterate for exactly **one** gravity-capillary
   FastLM solve at `(B, ε_target)`.

`jacobian=:finitediff` uses `AutoFiniteDiff()`. `jacobian=:analytical` uses
only `gravity_capillary_jacobian!`. Both variants use QR least squares to
retain the full translationally invariant system.

No gravity-capillary energy continuation is generated. The energy sequence is
used only to construct the pure-gravity seed branch.
"""
function _solve_gravity_capillary_from_gravity_seeds(
        prob::GravityCapillaryWaveProblem{T}, grid::WaveGrid{T}, ws::SpectralWorkspace;
        jacobian::Symbol=:finitediff, abstol::Real, maxiters::Int,
        callback=nothing) where {T}

    isempty(prob.𝜖) && return Vector{Vector{T}}(), Any[]

    # Continue pure gravity to the sole requested target energy.
    gravity_prob = GravityWaveProblem(prob.N, prob.𝜖; T=T)
    gravity_sol = solve(gravity_prob, Collocation();
        abstol=abstol, maxiters=maxiters)

    # The pure-gravity solve prepends its own linear-energy state.
    gravity_index = length(prob.𝜖) + 1
    gravity_rc = gravity_sol.retcodes[gravity_index]
    ε_target = prob.𝜖[end]

    if gravity_rc ∉ (ReturnCode.Success, ReturnCode.Stalled)
        @warn "Pure-gravity continuation did not reach gravity-capillary target energy" ε_target gravity_retcode=gravity_rc
        return [copy(gravity_sol.solutions[gravity_index])], Any[gravity_rc]
    end

    # One target-B solve, seeded by the pure-gravity state at ε_target.
    u0 = copy(gravity_sol.solutions[gravity_index])
    if jacobian === :analytical
        # A supplied jacobian takes precedence: do not request autodiff.
        fast_lm = FastLevenbergMarquardtJL(:qr)
        jw = GravityCapillaryJacobian!(prob.N; x=grid.ξ, T=T)
        nf = NonlinearFunction{true}(GravityCapillaryWavesAnalyticalJac;
            jac=gravity_capillary_jacobian!)
        gc_prob = NonlinearProblem(nf, u0,
            (prob.B, grid.ξ, ε_target, Ehw, prob.N, ws, jw))
    elseif jacobian === :finitediff
        fast_lm = FastLevenbergMarquardtJL(:qr; autodiff=AutoFiniteDiff())
        residual = NonlinearFunction{true}(GravityCapillaryWavesEfficient)
        gc_prob = NonlinearProblem{true}(
            residual, u0, (prob.B, grid.ξ, ε_target, Ehw, prob.N, ws))
    else
        throw(ArgumentError("gravity-seeded target solver supports :finitediff or :analytical"))
    end

    gc_sol = solve(gc_prob, fast_lm;
        abstol=abstol, maxiters=maxiters)

    if gc_sol.retcode ∉ (ReturnCode.Success, ReturnCode.Stalled)
        @warn "Gravity-capillary LM target solve did not converge from pure-gravity seed" ε_target retcode=gc_sol.retcode residual=norm(gc_sol.resid)
    end
    callback !== nothing && callback(1, gc_sol, ε_target)

    return [copy(gc_sol.u)], Any[gc_sol.retcode]
end



# =============================================================================
# solve() dispatch for GravityCapillaryWaveProblem
# =============================================================================

"""
    solve(prob::GravityCapillaryWaveProblem, method::Collocation;
          abstol=1e-10, maxiters=100, callback=nothing)

Solve a gravity-capillary **target state** on the full translationally invariant
system. The package first continues pure gravity on the `TravellingStokes`
energy schedule, then seeds one target-
`B` solve from the final pure-gravity state at `ε_target`.

The default `solve(prob)` route is
`Collocation(; jacobian=:finitediff)` with
`FastLevenbergMarquardtJL(; autodiff=AutoFiniteDiff())`. It returns one
`WaveSolution` state at `ε_target`, rather than a gravity-capillary energy
continuation.

# Example
```julia
prob = GravityCapillaryWaveProblem(512, 0.01)
sol = solve(prob, Collocation())
```
"""
function CommonSolve.solve(prob::GravityCapillaryWaveProblem{T},
        method::Collocation;
        abstol::Real=1e-10, maxiters::Int=100, callback=nothing) where {T}

    N = prob.N
    B = prob.B
    grid = WaveGrid(N; T=T)
    ws = SpectralWorkspace(grid)

    # Initial condition
    ϵ₀ = T(1e-5)
    Y = @. ϵ₀ * cos(2π * grid.ξ)
    F = sqrt((B * 4π^2 + 1) / T(2π))  # linear dispersion with surface tension

    𝞊₀ = initial_energy(ws, Y, F, B, Ehw)
    𝜖 = vcat(𝞊₀, prob.𝜖)
    u0 = vcat(Y, F)

    if method.jacobian == :analytical
        solutions, retcodes = _solve_gravity_capillary_from_gravity_seeds(
            prob, grid, ws;
            jacobian=:analytical,
            abstol=abstol, maxiters=maxiters, callback=callback)
        𝜖 = T[prob.𝜖[end]]
    elseif method.jacobian == :krylov
        solutions, retcodes = continuation(
            GravityCapillaryWavesEfficient, u0, 𝜖,
            ε -> (B, grid.ξ, ε, Ehw, N, ws);
            solver = NewtonRaphson(; linsolve=KrylovJL_GMRES(),
                                     autodiff=AutoFiniteDiff()),
            abstol = abstol, maxiters = maxiters,
            callback = callback
        )
    else  # :finitediff — pure-gravity seeded TravellingStokes workflow
        solutions, retcodes = _solve_gravity_capillary_from_gravity_seeds(
            prob, grid, ws;
            jacobian=:finitediff,
            abstol=abstol, maxiters=maxiters, callback=callback)
        # The returned solution contains only the single target-B state.
        𝜖 = T[prob.𝜖[end]]
    end

    return WaveSolution{T, typeof(prob)}(grid, solutions, retcodes, 𝜖, prob)
end

# Primary public route: finite-difference Levenberg–Marquardt on the full system.
function CommonSolve.solve(prob::GravityCapillaryWaveProblem{T}; kwargs...) where {T}
    return CommonSolve.solve(prob, Collocation(; jacobian=:finitediff); kwargs...)
end
