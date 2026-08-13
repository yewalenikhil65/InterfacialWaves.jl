# =============================================================================
# Viscous Gravity-Capillary Waves — Original and Efficient variants
# =============================================================================

# =============================================================================
# Fixed-B variant: B is a user-specified parameter (not derived from F, Re)
# When Re=Inf and P=0, this reduces to inviscid gravity-capillary waves.
# =============================================================================

"""
    ViscousGravityCapillaryWaves_fixedB(du, u, p)

Viscous gravity-capillary wave residual with Bond number B as a fixed parameter.

Unlike `ViscousGravityCapillaryWaves` where B = (1/G^{1/3})*(F/Re)^{4/3},
here B is specified directly. For Re=Inf, the viscous terms vanish and this
reduces to inviscid gravity-capillary waves.

State vector: `u = [Y..., F, P]` (N+2 unknowns, N+1 equations — use LevenbergMarquardt).
Parameters: `p = (Re, B, ξ, 𝞊, Ehw_val, N, h, d)`
"""
@views function ViscousGravityCapillaryWaves_fixedB(du, u, p)
    Re, B, ξ, 𝞊, Ehw_val, N, h, d = p

    Y = u[1:end-2]
    F = u[end-1]
    P = u[end]

    Ŷ = fft(Y)

    Yξ  = real(ifft(d .* Ŷ))
    Yξξ = real(ifft(d.^2 .* Ŷ))
    Xξ  = 1.0 .- real(ifft(h .* d .* Ŷ))
    Xξξ = -real(ifft(h .* d.^2 .* Ŷ))

    J = @. Xξ^2 + Yξ^2

    if isinf(Re)
        ψξ = Yξ
    else
        ψξ = @. Yξ + (2.0 / Re) * ((Xξ * Yξξ - Yξ * Xξξ) / Xξ^2)
    end
    ϕξ  = -real(ifft(h .* fft(ψξ)))
    ϕξξ = real(ifft(d .* fft(ϕξ)))
    ψξξ = real(ifft(h .* fft(ϕξξ)))

    Fϕ̂ξ = (1 ./ d) .* fft(ϕξ)
    Fϕ̂ξ[1] = 0.0 + im * 0.0
    Fϕ̂ξ[N÷2+1] = 0.0 + im * 0.0
    ϕ = real(ifft(Fϕ̂ξ))
    ϕ = ϕ .- solve(SampledIntegralProblem(ϕ .* Xξ, ξ), TrapezoidalRule()).u

    if isinf(Re)
        du[1:length(ξ)] .= @. ((-F^2) * (Xξ * ϕξ + Yξ * ψξ) / J) +
            ((F^2) * (ϕξ^2 + ψξ^2) / (2 * J)) + Y +
            (B * (Yξ * Xξξ - Yξξ * Xξ) / (J^(3/2))) + P * (Yξ / Xξ)
    else
        du[1:length(ξ)] .= @. ((-F^2) * (Xξ * ϕξ + Yξ * ψξ) / J) +
            ((F^2) * (ϕξ^2 + ψξ^2) / (2 * J)) + Y +
            (B * (Yξ * Xξξ - Yξξ * Xξ) / (J^(3/2))) + P * (Yξ / Xξ) +
            (2 * F^2 / Re) * (((Yξ^2 - Xξ^2) * ϕξξ - 2.0 * Xξ * Yξ * ψξξ) / J^2 +
            (ϕξ * (Xξξ * Xξ * (Xξ^2 - 3 * Yξ^2) + Yξξ * Yξ * (3 * Xξ^2 - Yξ^2))) / J^3 +
            (ψξ * (Xξξ * Yξ * (3 * Xξ^2 - Yξ^2) + Yξξ * Xξ * (3 * Yξ^2 - Xξ^2))) / J^3)
    end

    du[length(ξ) + 1] = WaveEnergyParameter(ψξ, -ϕ, Y, Xξ, J, ξ, F, B, Ehw_val) - 𝞊

    return nothing
end

"""
    ViscousGravityCapillaryWavesEfficient_fixedB(du, u, p)

Efficient (zero-allocation) variant with B as a fixed parameter.

Parameters: `p = (Re, B, ξ, 𝞊, Ehw_val, N, ws)`
"""
@views function ViscousGravityCapillaryWavesEfficient_fixedB(du, u, p)
    Re, B, ξ, 𝞊, Ehw_val, N, ws = p
    Y = u[1:end-2]
    F = u[end-1]
    P = u[end]
    r = ws.r
    inviscid = isinf(Re)

    # Spectrum of Y
    transform!(ws, Y)

    # Spatial derivatives
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

    # ψξ (with viscous correction if finite Re)
    if inviscid
        @inbounds @simd for i in eachindex(r[6], r[1])
            r[6][i] = r[1][i]                             # ψξ = Yξ
        end
    else
        @inbounds @simd for i in eachindex(r[6], r[1], r[2], r[3], r[4])
            r[6][i] = r[1][i] + (2.0 / Re) * (r[3][i] * r[2][i] - r[1][i] * r[4][i]) / r[3][i]^2
        end
    end

    # ϕξ = -H(ψξ)
    transform!(ws, r[6])
    neg_hilbert!(ws, r[7])                                # ϕξ

    # ϕξξ = d(ϕξ)
    transform!(ws, r[7])
    derivative!(ws, r[8], 1)                              # ϕξξ

    # ψξξ = H(ϕξξ)
    transform!(ws, r[8])
    hilbert!(ws, r[9])                                    # ψξξ

    # ϕ = spectral_integrate(ϕξ) - ∫ϕ·Xξ dξ
    transform!(ws, r[7])
    integrate!(ws, r[10])
    r[10] .-= trapezoid!(ws, r[10], r[3])

    Yξ, Yξξ, Xξ, Xξξ, J = r[1], r[2], r[3], r[4], r[5]
    ψξ, ϕξ, ϕξξ, ψξξ, ϕ = r[6], r[7], r[8], r[9], r[10]

    # Bernoulli residual
    if inviscid
        @inbounds @simd for i in eachindex(ξ)
            du[i] = ((-F^2) * (Xξ[i] * ϕξ[i] + Yξ[i] * ψξ[i]) / J[i]) +
                    ((F^2) * (ϕξ[i]^2 + ψξ[i]^2) / (2 * J[i])) + Y[i] +
                    B * (Yξ[i] * Xξξ[i] - Yξξ[i] * Xξ[i]) / J[i]^(3/2) +
                    P * (Yξ[i] / Xξ[i])
        end
    else
        @inbounds @simd for i in eachindex(ξ)
            du[i] = ((-F^2) * (Xξ[i] * ϕξ[i] + Yξ[i] * ψξ[i]) / J[i]) +
                    ((F^2) * (ϕξ[i]^2 + ψξ[i]^2) / (2 * J[i])) + Y[i] +
                    B * (Yξ[i] * Xξξ[i] - Yξξ[i] * Xξ[i]) / J[i]^(3/2) +
                    P * (Yξ[i] / Xξ[i]) +
                    (2 * F^2 / Re) * (
                        ((Yξ[i]^2 - Xξ[i]^2) * ϕξξ[i] - 2.0 * Xξ[i] * Yξ[i] * ψξξ[i]) / J[i]^2 +
                        (ϕξ[i] * (Xξξ[i] * Xξ[i] * (Xξ[i]^2 - 3 * Yξ[i]^2) + Yξξ[i] * Yξ[i] * (3 * Xξ[i]^2 - Yξ[i]^2))) / J[i]^3 +
                        (ψξ[i] * (Xξξ[i] * Yξ[i] * (3 * Xξ[i]^2 - Yξ[i]^2) + Yξξ[i] * Xξ[i] * (3 * Yξ[i]^2 - Xξ[i]^2))) / J[i]^3
                    )
        end
    end

    # Energy constraint
    @inbounds @simd for i in eachindex(r[10])
        r[10][i] = -r[10][i]
    end
    du[length(ξ) + 1] = energy!(ws, ψξ, r[10], Y, Xξ, J, F, B, Ehw_val) - 𝞊

    return nothing
end

# =============================================================================
# Original: full FFT, allocating (B derived from F, Re, G)
# =============================================================================

@views function ViscousGravityCapillaryWaves(du, u, p)
    Re, Mo, ξ, 𝞊, Ehw_val, N, h, d = p 

    Y = u[1:end-2]      # Interface  
    F = u[end-1]          # Froude Number
    P = u[end]          # Pressure term

    B = _viscous_derived_B(Mo, Re, F)

    Ŷ = fft(Y);         # fft(interface)

    Yξ = real(ifft(d .* Ŷ ))
    Yξξ = real(ifft(d.^2 .* Ŷ ))
    Xξ = 1.0 .- real(ifft(h.* d .* Ŷ ))
    Xξξ = -real(ifft(h.* d.^2 .* Ŷ ))

    J = @. Xξ^2 + Yξ^2  
    
    ψξ = @. Yξ + (2.0 / Re)*((Xξ*Yξξ - Yξ*Xξξ)/Xξ^2)
    ϕξ = -real(ifft(h .* fft(ψξ)))
    ϕξξ = real(ifft(d .* fft(ϕξ)))
    ψξξ = real(ifft(h .* fft(ϕξξ)))
    
    
    Fϕ̂ξ = (1 ./ d).*fft(ϕξ)
    Fϕ̂ξ[1] = 0.0 + im*0.0
    Fϕ̂ξ[N÷2 + 1] = 0.0 + im*0.0
    ϕ = real(ifft(Fϕ̂ξ))
    ϕ = ϕ .- solve(SampledIntegralProblem(ϕ .* Xξ , ξ), TrapezoidalRule()).u
   
    du[1:length(ξ)] .= @. ((-F^2)*(Xξ*ϕξ + Yξ*ψξ) /J) + ((F^2)*(ϕξ^2 + ψξ^2)/(2*J)) + Y + (B*(Yξ*Xξξ - Yξξ*Xξ)/(J^(3/2))) + P*(Yξ/Xξ) + (2*F^2 / Re)*(((Yξ^2 - Xξ^2)*ϕξξ - 2.0*Xξ*Yξ*ψξξ)/J^2 
    + (ϕξ*(Xξξ*Xξ*(Xξ^2 - 3*Yξ^2) + Yξξ*Yξ*(3*Xξ^2 - Yξ^2)))/J^3 + 
    (ψξ*(Xξξ*Yξ*(3*Xξ^2 - Yξ^2) + Yξξ*Xξ*(3*Yξ^2 - Xξ^2)))/J^3)

    du[length(ξ) + 1] = WaveEnergyParameter(ψξ, -ϕ, Y, Xξ, J, ξ, F, B, Ehw_val) - 𝞊
    
    return nothing
end 

# =============================================================================
# Efficient: rfft/irfft, zero-allocation spectral operations
# =============================================================================

"""
    ViscousGravityCapillaryWavesEfficient(du, u, p)

Efficient counterpart of `ViscousGravityCapillaryWaves` using `SpectralWorkspace`.

Pass `p = (Re, Mo, ξ, 𝞊, Ehw_val, N, ws)` to NonlinearSolve.
`Mo` is the Morton number and `B = Mo^(-1/3) * abs(F/Re)^(4/3)`.
"""
@views function ViscousGravityCapillaryWavesEfficient(du, u, p)
    Re, Mo, ξ, 𝞊, Ehw_val, N, ws = p
    Y = u[1:end-2]
    F = u[end-1]
    P = u[end]
    B = _viscous_derived_B(Mo, Re, F)
    r = ws.r

    # Spectrum of Y
    transform!(ws, Y)

    # Spatial derivatives from Y-spectrum
    derivative!(ws, r[1], 1)                              # Yξ
    derivative!(ws, r[2], 2)                              # Yξξ
    hilbert_derivative!(ws, r[3], 1)                      # H·d·Y
    @inbounds @simd for i in eachindex(r[3])
        r[3][i] = 1.0 - r[3][i]                          # Xξ
    end
    hilbert_derivative!(ws, r[4], 2)                      # H·d²·Y
    @inbounds @simd for i in eachindex(r[4])
        r[4][i] = -r[4][i]                               # Xξξ
    end
    @inbounds @simd for i in eachindex(r[5], r[1], r[3])
        r[5][i] = r[3][i]^2 + r[1][i]^2                  # J
    end

    # ψξ = Yξ + (2/Re)·(Xξ·Yξξ - Yξ·Xξξ)/Xξ² (viscous correction)
    @inbounds @simd for i in eachindex(r[6], r[1], r[2], r[3], r[4])
        r[6][i] = r[1][i] + (2.0 / Re) * (r[3][i] * r[2][i] - r[1][i] * r[4][i]) / r[3][i]^2
    end

    # ϕξ = -H(ψξ)
    transform!(ws, r[6])                                  # spectrum of ψξ
    neg_hilbert!(ws, r[7])                                # ϕξ

    # ϕξξ = d(ϕξ)
    transform!(ws, r[7])                                  # spectrum of ϕξ
    derivative!(ws, r[8], 1)                              # ϕξξ

    # ψξξ = H(ϕξξ)
    transform!(ws, r[8])                                  # spectrum of ϕξξ
    hilbert!(ws, r[9])                                    # ψξξ

    # ϕ = spectral_integrate(ϕξ) + ∫(ϕ·Xξ)dξ
    transform!(ws, r[7])                                  # spectrum of ϕξ
    integrate!(ws, r[10])                                 # ϕ (before mean correction)
    r[10] .-= trapezoid!(ws, r[10], r[3])                # ϕ -= ∫ϕ·Xξ dξ

    Yξ, Yξξ, Xξ, Xξξ, J = r[1], r[2], r[3], r[4], r[5]
    ψξ, ϕξ, ϕξξ, ψξξ, ϕ = r[6], r[7], r[8], r[9], r[10]

    # Full Bernoulli equation with viscous + capillary + pressure terms
    @inbounds @simd for i in eachindex(ξ)
        du[i] = ((-F^2) * (Xξ[i] * ϕξ[i] + Yξ[i] * ψξ[i]) / J[i]) +
                ((F^2) * (ϕξ[i]^2 + ψξ[i]^2) / (2 * J[i])) + Y[i] +
                B * (Yξ[i] * Xξξ[i] - Yξξ[i] * Xξ[i]) / J[i]^(3/2) +
                P * (Yξ[i] / Xξ[i]) +
                (2 * F^2 / Re) * (
                    ((Yξ[i]^2 - Xξ[i]^2) * ϕξξ[i] - 2.0 * Xξ[i] * Yξ[i] * ψξξ[i]) / J[i]^2 +
                    (ϕξ[i] * (Xξξ[i] * Xξ[i] * (Xξ[i]^2 - 3 * Yξ[i]^2) + Yξξ[i] * Yξ[i] * (3 * Xξ[i]^2 - Yξ[i]^2))) / J[i]^3 +
                    (ψξ[i] * (Xξξ[i] * Yξ[i] * (3 * Xξ[i]^2 - Yξ[i]^2) + Yξξ[i] * Xξ[i] * (3 * Yξ[i]^2 - Xξ[i]^2))) / J[i]^3
                )
    end

    # Energy constraint
    @inbounds @simd for i in eachindex(r[10])
        r[10][i] = -r[10][i]
    end
    du[length(ξ) + 1] = energy!(ws, ψξ, r[10], Y, Xξ, J, F, B, Ehw_val) - 𝞊

    return nothing
end

# =============================================================================

# =============================================================================
# Analytical Jacobian for finite-Re, solved-pressure viscous waves
# =============================================================================

"""
    ViscousGravityCapillaryJacobian!(N; x=nothing, T=Float64)

Precomputed spectral derivative matrices and reusable directional buffers for
the rectangular `(N+1) × (N+2)` Jacobian of the unpinned finite-Re system
with state `[Y..., F, P]`.
"""
struct ViscousGravityCapillaryJacobian!{T<:AbstractFloat}
    operators::GravityCapillaryJacobian!{T}
    dψξ::Vector{T}
    dϕξ::Vector{T}
    dϕξξ::Vector{T}
    dψξξ::Vector{T}
    dϕbase::Vector{T}
end

function ViscousGravityCapillaryJacobian!(N::Integer; x=nothing,
        T::Type{<:AbstractFloat}=Float64)
    ops = GravityCapillaryJacobian!(N; x=x, T=T)
    return ViscousGravityCapillaryJacobian!{T}(
        ops, zeros(T, N), zeros(T, N), zeros(T, N), zeros(T, N), zeros(T, N))
end

@views function ViscousGravityCapillaryWavesDerivedBAnalyticalJac(du, u, p)
    Re, Mo, ξ, 𝞊, Ehw_val, N, ws, jw = p
    return ViscousGravityCapillaryWavesEfficient(du, u,
        (Re, Mo, ξ, 𝞊, Ehw_val, N, ws))
end

@views function ViscousGravityCapillaryWavesFixedBAnalyticalJac(du, u, p)
    Re, B, ξ, 𝞊, Ehw_val, N, ws, jw = p
    return ViscousGravityCapillaryWavesEfficient_fixedB(du, u,
        (Re, B, ξ, 𝞊, Ehw_val, N, ws))
end

"""
    _viscous_gravity_capillary_jacobian!(Jac, u, p, B, dB_dF)

Fill the analytical Jacobian of the finite-Re residual.  `dB_dF` is zero for
the fixed-B formulation and `(4/3)B/F` on the positive-F derived-B branch.
"""
function _viscous_gravity_capillary_jacobian!(Jac, u, p, B, dB_dF)
    Re, _, ξ, 𝞊, Ehw_val, N, ws, jw = p
    Re > 0 || throw(ArgumentError("analytical viscous Jacobian requires finite positive Re"))
    ops = jw.operators
    Y = @view u[1:N]
    F = u[N + 1]
    P = u[N + 2]
    r = ws.r

    # Base state, matching the efficient residual exactly.
    transform!(ws, Y)
    derivative!(ws, r[1], 1)                              # Yξ
    derivative!(ws, r[2], 2)                              # Yξξ
    hilbert_derivative!(ws, r[3], 1)
    @inbounds @simd for i in 1:N
        r[3][i] = 1 - r[3][i]                             # Xξ
    end
    hilbert_derivative!(ws, r[4], 2)
    @inbounds @simd for i in 1:N
        r[4][i] = -r[4][i]                                # Xξξ
        r[5][i] = r[3][i]^2 + r[1][i]^2                  # J
    end

    Yξ, Yξξ, Xξ, Xξξ, J = r[1], r[2], r[3], r[4], r[5]
    @inbounds @simd for i in 1:N
        r[6][i] = Yξ[i] + (2 / Re) *
            (Xξ[i] * Yξξ[i] - Yξ[i] * Xξξ[i]) / Xξ[i]^2 # ψξ
    end
    transform!(ws, r[6]); neg_hilbert!(ws, r[7])           # ϕξ
    transform!(ws, r[7]); derivative!(ws, r[8], 1)         # ϕξξ
    transform!(ws, r[8]); hilbert!(ws, r[9])               # ψξξ
    transform!(ws, r[7]); integrate!(ws, r[10])            # ϕbase
    cmean = -trapezoid!(ws, r[10], Xξ)
    @inbounds @simd for i in 1:N
        r[10][i] += cmean                                 # ϕ
    end

    ψξ, ϕξ, ϕξξ, ψξξ, ϕ = r[6], r[7], r[8], r[9], r[10]
    F2 = F^2
    νfac = 2 * F2 / Re

    # F and P columns.  These do not require a directional spectral solve.
    @inbounds for i in 1:N
        Ji = J[i]
        κ = Yξ[i] * Xξξ[i] - Yξξ[i] * Xξ[i]
        a = (Xξ[i] * ϕξ[i] + Yξ[i] * ψξ[i]) / Ji
        b = (ϕξ[i]^2 + ψξ[i]^2) / (2 * Ji)
        q1 = ((Yξ[i]^2 - Xξ[i]^2) * ϕξξ[i] - 2 * Xξ[i] * Yξ[i] * ψξξ[i]) / Ji^2
        l = Xξξ[i] * Xξ[i] * (Xξ[i]^2 - 3 * Yξ[i]^2) +
            Yξξ[i] * Yξ[i] * (3 * Xξ[i]^2 - Yξ[i]^2)
        m = Xξξ[i] * Yξ[i] * (3 * Xξ[i]^2 - Yξ[i]^2) +
            Yξξ[i] * Xξ[i] * (3 * Yξ[i]^2 - Xξ[i]^2)
        v = q1 + ϕξ[i] * l / Ji^3 + ψξ[i] * m / Ji^3
        Jac[i, N + 1] = -2 * F * a + 2 * F * b +
            dB_dF * κ / Ji^(3 / 2) + (4 * F / Re) * v
        Jac[i, N + 2] = Yξ[i] / Xξ[i]
    end

    # Y columns.  The nonlinear ψξ variation is propagated through the exact
    # rFFT operators for ϕξ, ϕξξ, ψξξ and the integrated mean correction.
    @inbounds for j in 1:N
        for i in 1:N
            dyξ = ops.D[i, j]
            dyξξ = ops.D2[i, j]
            dxξ = -ops.HD[i, j]
            dxξξ = -ops.HD2[i, j]
            q = Xξ[i] * Yξξ[i] - Yξ[i] * Xξξ[i]
            dq = dxξ * Yξξ[i] + Xξ[i] * dyξξ - dyξ * Xξξ[i] - Yξ[i] * dxξξ
            jw.dψξ[i] = dyξ + (2 / Re) * (dq / Xξ[i]^2 - 2 * q * dxξ / Xξ[i]^3)
        end
        transform!(ws, jw.dψξ); neg_hilbert!(ws, jw.dϕξ)
        transform!(ws, jw.dϕξ); derivative!(ws, jw.dϕξξ, 1)
        transform!(ws, jw.dϕξξ); hilbert!(ws, jw.dψξξ)
        transform!(ws, jw.dϕξ); integrate!(ws, jw.dϕbase)

        # d[-∫ϕbase Xξ] and hence dϕ.
        dcmean = zero(eltype(u))
        for i in 1:N
            dxξ = -ops.HD[i, j]
            ϕbase = ϕ[i] - cmean
            dcmean -= ops.trap_wt[i] * (jw.dϕbase[i] * Xξ[i] + ϕbase * dxξ)
        end

        dE = zero(eltype(u))
        for i in 1:N
            dyξ = ops.D[i, j]
            dyξξ = ops.D2[i, j]
            dxξ = -ops.HD[i, j]
            dxξξ = -ops.HD2[i, j]
            dJ = 2 * Xξ[i] * dxξ + 2 * Yξ[i] * dyξ
            Ji = J[i]
            Ji2 = Ji^2
            Ji3 = Ji^3
            Ji4 = Ji^4
            sqrtJi = sqrt(Ji)
            κ = Yξ[i] * Xξξ[i] - Yξξ[i] * Xξ[i]
            dκ = dyξ * Xξξ[i] + Yξ[i] * dxξξ - dyξξ * Xξ[i] - Yξξ[i] * dxξ

            na = Xξ[i] * ϕξ[i] + Yξ[i] * ψξ[i]
            dna = dxξ * ϕξ[i] + Xξ[i] * jw.dϕξ[i] +
                dyξ * ψξ[i] + Yξ[i] * jw.dψξ[i]
            da = dna / Ji - na * dJ / Ji2

            nb = ϕξ[i]^2 + ψξ[i]^2
            db = (ϕξ[i] * jw.dϕξ[i] + ψξ[i] * jw.dψξ[i]) / Ji -
                nb * dJ / (2 * Ji2)
            dc = dκ / Ji^(3 / 2) - 1.5 * κ * dJ / Ji^(5 / 2)
            dp = P * (dyξ / Xξ[i] - Yξ[i] * dxξ / Xξ[i]^2)

            q1num = (Yξ[i]^2 - Xξ[i]^2) * ϕξξ[i] - 2 * Xξ[i] * Yξ[i] * ψξξ[i]
            dq1num = (2 * Yξ[i] * dyξ - 2 * Xξ[i] * dxξ) * ϕξξ[i] +
                (Yξ[i]^2 - Xξ[i]^2) * jw.dϕξξ[i] -
                2 * (dxξ * Yξ[i] + Xξ[i] * dyξ) * ψξξ[i] -
                2 * Xξ[i] * Yξ[i] * jw.dψξξ[i]
            dv1 = dq1num / Ji2 - 2 * q1num * dJ / Ji3

            a1 = Xξ[i]^2 - 3 * Yξ[i]^2
            a2 = 3 * Xξ[i]^2 - Yξ[i]^2
            a3 = 3 * Yξ[i]^2 - Xξ[i]^2
            da1 = 2 * Xξ[i] * dxξ - 6 * Yξ[i] * dyξ
            da2 = 6 * Xξ[i] * dxξ - 2 * Yξ[i] * dyξ
            da3 = 6 * Yξ[i] * dyξ - 2 * Xξ[i] * dxξ
            l = Xξξ[i] * Xξ[i] * a1 + Yξξ[i] * Yξ[i] * a2
            dl = dxξξ * Xξ[i] * a1 + Xξξ[i] * dxξ * a1 + Xξξ[i] * Xξ[i] * da1 +
                dyξξ * Yξ[i] * a2 + Yξξ[i] * dyξ * a2 + Yξξ[i] * Yξ[i] * da2
            m = Xξξ[i] * Yξ[i] * a2 + Yξξ[i] * Xξ[i] * a3
            dm = dxξξ * Yξ[i] * a2 + Xξξ[i] * dyξ * a2 + Xξξ[i] * Yξ[i] * da2 +
                dyξξ * Xξ[i] * a3 + Yξξ[i] * dxξ * a3 + Yξξ[i] * Xξ[i] * da3
            dv2 = (jw.dϕξ[i] * l + ϕξ[i] * dl) / Ji3 - 3 * ϕξ[i] * l * dJ / Ji4
            dv3 = (jw.dψξ[i] * m + ψξ[i] * dm) / Ji3 - 3 * ψξ[i] * m * dJ / Ji4

            Jac[i, j] = -F2 * da + F2 * db + B * dc + dp + νfac * (dv1 + dv2 + dv3) +
                (i == j ? one(eltype(u)) : zero(eltype(u)))

            dϕ = jw.dϕbase[i] + dcmean
            denergy = (1 / Ehw_val) * (
                (F2 / 2) * (jw.dψξ[i] * (-ϕ[i]) + ψξ[i] * (-dϕ)) +
                B * (dJ / (2 * sqrtJi) - dxξ) +
                0.5 * (dxξ * Y[i]^2 + (i == j ? 2 * Xξ[i] * Y[i] : 0))
            )
            dE += ops.simpson_wt[i] * denergy
        end
        Jac[N + 1, j] = dE
    end

    # Energy row F and P columns.
    dEF = zero(eltype(u))
    @inbounds for i in 1:N
        dEF += ops.simpson_wt[i] * (1 / Ehw_val) * (
            F * ψξ[i] * (-ϕ[i]) + dB_dF * (sqrt(J[i]) - Xξ[i]))
    end
    Jac[N + 1, N + 1] = dEF
    Jac[N + 1, N + 2] = zero(eltype(Jac))
    return nothing
end

function viscous_gravity_capillary_derivedB_jacobian!(Jac, u, p)
    Re, Mo, ξ, 𝞊, Ehw_val, N, ws, jw = p
    F = u[N + 1]
    B = _viscous_derived_B(Mo, Re, F)
    return _viscous_gravity_capillary_jacobian!(Jac, u, p, B,
        _viscous_derived_dB_dF(Mo, Re, F))
end

function viscous_gravity_capillary_fixedB_jacobian!(Jac, u, p)
    Re, B, ξ, 𝞊, Ehw_val, N, ws, jw = p
    return _viscous_gravity_capillary_jacobian!(Jac, u, p, B, zero(eltype(u)))
end
# Unpinned finite-Re, solved-pressure energy continuation
# =============================================================================

"""
    _viscous_seed_energy(residual, u0, p0, N, T)

Evaluate the energy of the finite-Re residual at its small-amplitude seed.  The
last residual component equals `energy - ε`, so use `ε=0` to obtain the
consistent finite-viscosity seed energy without imposing a phase condition.
"""
function _viscous_seed_energy(residual, u0, p0, N::Int, ::Type{T}) where {T}
    r0 = zeros(T, N + 1)
    residual(r0, u0, p0)
    return r0[end]
end

"""
    _solve_viscous_from_pure_gravity(prob, residual, make_params;
                                      analytical_residual, analytical_jacobian,
                                      jacobian, abstol, maxiters, callback)

Solve one viscous target state from a pure-gravity collocation state at the same
terminal energy. The initial viscous vector is
`[Y_gravity..., F_gravity, P0]`; the viscous nonlinear system solves for the
final pressure coefficient `P`.
"""
# Solve one viscous target state from a pure-gravity collocation target.
# The viscous initial state is [Y_gravity..., F_gravity, P0].
function _solve_viscous_from_pure_gravity(
        prob::AbstractViscousGravityCapillaryProblem{T}, residual, make_params;
        analytical_residual, analytical_jacobian, jacobian::Symbol,
        abstol::Real, maxiters::Int, callback=nothing) where {T}

    ε_target = prob.𝜖[end]
    gravity_prob = GravityWaveProblem(prob.N; ε_max=ε_target, T=T)
    gravity_sol = solve(gravity_prob, Collocation(; jacobian=:analytical);
        abstol=abstol, maxiters=maxiters)
    gravity_ok = gravity_sol.retcodes[end] ∈
        (ReturnCode.Success, ReturnCode.Stalled)
    gravity_ok || error(
        "pure-gravity initialization did not converge at target energy " *
        "$(ε_target): $(gravity_sol.retcodes[end])")

    grid = WaveGrid(prob.N; T=T)
    u0 = vcat(copy(gravity_sol.Y), gravity_sol.F, prob.P0)

    if jacobian === :analytical
        nf = NonlinearFunction{true}(analytical_residual; jac=analytical_jacobian)
        an_ws = SpectralWorkspace(grid)
        an_jw = ViscousGravityCapillaryJacobian!(prob.N; x=grid.ξ, T=T)
        target_params = make_params(ε_target, grid, an_ws, an_jw)
        target_problem = NonlinearProblem(nf, copy(u0), target_params)
        target_sol = solve(target_problem, FastLevenbergMarquardtJL(:qr);
            abstol=abstol, maxiters=maxiters)
        # Guard against false convergence from FastLM
        if !(norm(target_sol.resid) ≤ abstol)
            retry_problem = NonlinearProblem(nf, copy(u0), target_params)
            target_sol = solve(retry_problem, LevenbergMarquardt();
                abstol=abstol, maxiters=maxiters)
        end
    else
        fd_ws = SpectralWorkspace(grid)
        target_params = make_params(ε_target, grid, fd_ws)
        nf = NonlinearFunction{true}(residual)
        target_problem = NonlinearProblem{true}(nf, copy(u0), target_params)
        target_sol = solve(target_problem,
            LevenbergMarquardt(; autodiff=AutoFiniteDiff());
            abstol=abstol, maxiters=maxiters)
    end

    if target_sol.retcode ∉ (ReturnCode.Success, ReturnCode.Stalled)
        @warn "Pure-gravity-seeded viscous target solve did not converge" ε_target retcode=target_sol.retcode residual=norm(target_sol.resid)
    end
    callback !== nothing && callback(1, target_sol, ε_target)

    return WaveSolution{T, typeof(prob)}(
        grid, [copy(target_sol.u)], Any[target_sol.retcode], T[ε_target], prob)
end

function _solve_viscous_energy_branch(prob::AbstractViscousGravityCapillaryProblem{T},
        residual, make_params;
        analytical_residual=nothing, analytical_jacobian=nothing,
        jacobian::Symbol=:finitediff, initial_state::Symbol=:pure_gravity,
        abstol::Real=1e-10, maxiters::Int=100,
        callback=nothing) where {T}
    initial_state ∈ (:pure_gravity, :linear) ||
        throw(ArgumentError("initial_state must be :pure_gravity or :linear"))
    jacobian ∈ (:finitediff, :analytical) ||
        throw(ArgumentError("jacobian must be :finitediff or :analytical"))
    if initial_state === :pure_gravity
        return _solve_viscous_from_pure_gravity(
            prob, residual, make_params;
            analytical_residual=analytical_residual,
            analytical_jacobian=analytical_jacobian,
            jacobian=jacobian, abstol=abstol, maxiters=maxiters,
            callback=callback)
    end
    N = prob.N
    grid = WaveGrid(N; T=T)
    ws = SpectralWorkspace(grid)
    jw = jacobian === :analytical ? ViscousGravityCapillaryJacobian!(N; x=grid.ξ, T=T) : nothing

    # Same finite-Re seed strategy as trial_parallel.jl, but with the package's
    # minus mean correction and unpinned least-squares solves.
    Y0 = @. prob.amplitude * cos(2π * grid.ξ)
    u0 = vcat(Y0, prob.F0, prob.P0)
    p0 = jacobian === :analytical ? make_params(zero(T), grid, ws, jw) :
        make_params(zero(T), grid, ws)
    residual_for_energy = jacobian === :analytical ? analytical_residual : residual
    ε0 = _viscous_seed_energy(residual_for_energy, u0, p0, N, T)
    ε = vcat(T(ε0), prob.𝜖)

    if jacobian === :analytical
        nf = NonlinearFunction{true}(analytical_residual; jac=analytical_jacobian)
        seed_problem = NonlinearProblem(nf, u0, p0)
        # The first low-energy solve is intentionally standard LM; all later
        # prescribed-energy solves use the QR FastLM backend.
        seed_sol = solve(seed_problem, LevenbergMarquardt();
            abstol=abstol, maxiters=maxiters)
        if seed_sol.retcode ∉ (ReturnCode.Success, ReturnCode.Stalled)
            @warn "Finite-Re analytical viscous seed solve did not converge" ε0 retcode=seed_sol.retcode residual=norm(seed_sol.resid)
        end
        u_seed = seed_sol.retcode ∈ (ReturnCode.Success, ReturnCode.Stalled) ? seed_sol.u : u0

        if callback !== nothing
            callback(1, seed_sol, ε0) === false &&
                return WaveSolution{T, typeof(prob)}(grid, [copy(seed_sol.u)], Any[seed_sol.retcode], T[ε0], prob)
        end
        subsequent_solutions, subsequent_retcodes = continuation(
            nf, u_seed, prob.𝜖,
            e -> make_params(e, grid, ws, jw);
            solver=FastLevenbergMarquardtJL(:qr),
            fallback_solver=LevenbergMarquardt(),
            abstol=abstol, maxiters=maxiters,
            callback=callback === nothing ? nothing :
                ((i, sol, e) -> callback(i + 1, sol, e)))
        solutions = vcat([copy(seed_sol.u)], subsequent_solutions)
        retcodes = vcat(Any[seed_sol.retcode], subsequent_retcodes)
        return WaveSolution{T, typeof(prob)}(grid, solutions, retcodes, ε, prob)
    end

    seed_problem = NonlinearProblem{true}(NonlinearFunction{true}(residual), u0, p0)
    seed_sol = solve(seed_problem,
        LevenbergMarquardt(; autodiff=AutoFiniteDiff());
        abstol=abstol, maxiters=maxiters)
    if seed_sol.retcode ∉ (ReturnCode.Success, ReturnCode.Stalled)
        @warn "Finite-Re viscous seed solve did not converge" ε0 retcode=seed_sol.retcode residual=norm(seed_sol.resid)
    end
    u_seed = seed_sol.retcode ∈ (ReturnCode.Success, ReturnCode.Stalled) ? seed_sol.u : u0

    if callback !== nothing
        callback(1, seed_sol, ε0) === false &&
            return WaveSolution{T, typeof(prob)}(grid, [copy(seed_sol.u)], Any[seed_sol.retcode], T[ε0], prob)
    end
    subsequent_solutions, subsequent_retcodes = continuation(
        residual, u_seed, prob.𝜖,
        e -> make_params(e, grid, ws);
        solver=FastLevenbergMarquardtJL(:qr; autodiff=AutoFiniteDiff()),
        fallback_solver=LevenbergMarquardt(; autodiff=AutoFiniteDiff()),
        abstol=abstol, maxiters=maxiters,
        callback=callback === nothing ? nothing :
            ((i, sol, e) -> callback(i + 1, sol, e)))
    solutions = vcat([copy(seed_sol.u)], subsequent_solutions)
    retcodes = vcat(Any[seed_sol.retcode], subsequent_retcodes)
    return WaveSolution{T, typeof(prob)}(grid, solutions, retcodes, ε, prob)
end

function CommonSolve.solve(prob::ViscousGravityCapillaryDerivedBProblem{T};
        jacobian::Symbol=:finitediff, initial_state::Symbol=:pure_gravity,
        abstol::Real=1e-10, maxiters::Int=100,
        callback=nothing) where {T}
    return _solve_viscous_energy_branch(prob, ViscousGravityCapillaryWavesEfficient,
        (ε, grid, ws, args...) -> (prob.Re, prob.Mo, grid.ξ, ε, Ehw, prob.N, ws, args...);
        analytical_residual=ViscousGravityCapillaryWavesDerivedBAnalyticalJac,
        analytical_jacobian=viscous_gravity_capillary_derivedB_jacobian!,
        jacobian=jacobian, initial_state=initial_state,
        abstol=abstol, maxiters=maxiters, callback=callback)
end

function CommonSolve.solve(prob::ViscousGravityCapillaryFixedBProblem{T};
        jacobian::Symbol=:finitediff, initial_state::Symbol=:pure_gravity,
        abstol::Real=1e-10, maxiters::Int=100,
        callback=nothing) where {T}
    return _solve_viscous_energy_branch(prob, ViscousGravityCapillaryWavesEfficient_fixedB,
        (ε, grid, ws, args...) -> (prob.Re, prob.B, grid.ξ, ε, Ehw, prob.N, ws, args...);
        analytical_residual=ViscousGravityCapillaryWavesFixedBAnalyticalJac,
        analytical_jacobian=viscous_gravity_capillary_fixedB_jacobian!,
        jacobian=jacobian, initial_state=initial_state,
        abstol=abstol, maxiters=maxiters, callback=callback)
end
