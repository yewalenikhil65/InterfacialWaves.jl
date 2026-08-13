"""
    Linear stability analysis for finite-amplitude interfacial waves.
    
    Reference: Murashige & Choi (2022), J. Fluid Mech. 938, A13.
    Solves the generalized eigenvalue problem (4.32) for growth rates σ.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Steady wave interface data
# ─────────────────────────────────────────────────────────────────────────────

struct SteadyWaveData
    x11::Vector{Float64}   # x̃_{1,1}^{(0)} at ξ̂_ℓ
    y11::Vector{Float64}   # ỹ_{1,1}^{(0)} at ξ̂_ℓ
    x22::Vector{Float64}   # x̃_{2,2}^{(0)} at ξ̂_ℓ
    y22::Vector{Float64}   # ỹ_{2,2}^{(0)} at ξ̂_ℓ
    J1::Vector{Float64}    # J₁^{(0)} at ξ̂_ℓ
    J2::Vector{Float64}    # J₂^{(0)} at ξ̂_ℓ
    J1_1::Vector{Float64}  # ∂J₁/∂ξ₁ at interface
    J2_2::Vector{Float64}  # ∂J₂/∂ξ₂ at interface
    γ0::Vector{Float64}    # γ^{(0)}(ξ̂_ℓ)
    ξ̂::Vector{Float64}     # collocation points
end

"""
    compute_steady_data(a1_n, a2_n, c_n, N) -> SteadyWaveData

Compute all steady wave interface quantities at 2N collocation points ξ̂_ℓ = ℓπ/N.
"""
function compute_steady_data(a1_n::AbstractVector, a2_n::AbstractVector, 
                             c_n::AbstractVector, N::Int)
    npts = 2N
    ξ̂ = [ℓ * π / N for ℓ in 0:npts-1]
    
    # γ^{(0)}(ξ̂) = Σ c_n sin(nξ̂)
    γ0 = zeros(npts)
    for n in eachindex(c_n)
        @. γ0 += c_n[n] * sin(n * ξ̂)
    end
    
    # ξ₁ = ξ̂ + γ, ξ₂ = ξ̂ - γ
    ξ1 = ξ̂ .+ γ0
    ξ2 = ξ̂ .- γ0
    
    # ∂z₁/∂ξ₁ = 1 + i Σ (in) a_{1n} exp(inξ₁)  at η₁=0
    # ∂z₂/∂ξ₂ = 1 + i Σ (-in) a_{2n} exp(-inξ₂) at η₂=0
    Na1 = length(a1_n)
    Na2 = length(a2_n)
    
    x11 = ones(npts)
    y11 = zeros(npts)
    x22 = ones(npts)
    y22 = zeros(npts)
    
    # Second derivatives for J_{j,j}
    x11_2 = zeros(npts)  # ∂²x₁/∂ξ₁²
    y11_2 = zeros(npts)  # ∂²y₁/∂ξ₁²
    x22_2 = zeros(npts)
    y22_2 = zeros(npts)
    
    for n in 0:Na1-1
        a = a1_n[n+1]
        abs(a) < 1e-13 && continue
        # dz₁/dξ₁ = 1 + i·Σ(in)·a_n·exp(inξ₁) = 1 - Σ n·a_n·exp(inξ₁)
        # = 1 - Σ n·a_n·cos(nξ₁) - i·Σ n·a_n·sin(nξ₁)
        @. x11 += -n * a * cos(n * ξ1)
        @. y11 += -n * a * sin(n * ξ1)
        # d²z₁/dξ₁² = -Σ (in)·n·a_n·exp(inξ₁) = -i·Σ n²·a_n·exp(inξ₁)
        # = -i·Σ n²·a_n·[cos(nξ₁) + i·sin(nξ₁)]
        # = Σ n²·a_n·sin(nξ₁) - i·Σ n²·a_n·cos(nξ₁)
        @. x11_2 +=  n^2 * a * sin(n * ξ1)
        @. y11_2 += -n^2 * a * cos(n * ξ1)
    end
    
    for n in 0:Na2-1
        a = a2_n[n+1]
        abs(a) < 1e-13 && continue
        # dz₂/dξ₂ = 1 + Σ n·a_n·exp(-inξ₂)
        # Re part: 1 + Σ n·a_n·cos(nξ₂), Im part: -Σ n·a_n·sin(nξ₂)
        @. x22 +=  n * a * cos(n * ξ2)
        @. y22 += -n * a * sin(n * ξ2)
        # d²z₂/dξ₂² = Σ (-in)·n·a_n·exp(-inξ₂) = Σ n²·a_n·(-i)·exp(-inξ₂)
        # Re part: -Σ n²·a_n·sin(nξ₂) ... wait, let me be precise:
        # d²z₂/dξ₂² = d/dξ₂[1 + Σ n·a_n·exp(-inξ₂)] = Σ (-in)·n·a_n·exp(-inξ₂)
        #            = -i·Σ n²·a_n·[cos(nξ₂) - i·sin(nξ₂)]
        #            = -Σ n²·a_n·sin(nξ₂) - i·Σ n²·a_n·cos(nξ₂)  ... no:
        # = -i·Σ n²·a_n·cos(nξ₂) + i²·Σ n²·a_n·sin(nξ₂)
        # = -Σ n²·a_n·sin(nξ₂) - i·Σ n²·a_n·cos(nξ₂)  ... let me just expand:
        # (-i)·exp(-inξ₂) = (-i)(cos(nξ₂) - i·sin(nξ₂)) = -i·cos(nξ₂) + i²·sin(nξ₂)
        #                  = -sin(nξ₂) - i·cos(nξ₂)
        # So d²z₂/dξ₂² = Σ n²·a_n·[-sin(nξ₂) - i·cos(nξ₂)]
        @. x22_2 += -n^2 * a * sin(n * ξ2)
        @. y22_2 += -n^2 * a * cos(n * ξ2)
    end
    
    J1 = @. x11^2 + y11^2
    J2 = @. x22^2 + y22^2
    
    # J_{1,1} = 2(x̃_{1,1}·x̃_{1,11} + ỹ_{1,1}·ỹ_{1,11})
    J1_1 = @. 2(x11 * x11_2 + y11 * y11_2)
    J2_2 = @. 2(x22 * x22_2 + y22 * y22_2)
    
    return SteadyWaveData(x11, y11, x22, y22, J1, J2, J1_1, J2_2, γ0, ξ̂)
end

# ─────────────────────────────────────────────────────────────────────────────
# Coefficient functions (Appendix B)
# ─────────────────────────────────────────────────────────────────────────────

"""
Evaluate all coefficient functions at all collocation points for a given m.
Returns vectors of length npts for each coefficient.
"""
function evaluate_coefficients!(
    A11m, B11m, A21m, A22m, B21m, B22m,
    A31m, A32m, B31m, B32m, C3m,
    A41m, A42m, C4m,
    A51m, A52m, B51m, B52m,
    m::Int, p::Real, ρ_::Real, c::Real, data::SteadyWaveData)
    
    npts = length(data.ξ̂)
    
    @inbounds for ℓ in 1:npts
        pm = p + m
        s = sign(pm)   # sgn(p+m)
        abs_pm = abs(pm)
        
        γ0 = data.γ0[ℓ]
        x11 = data.x11[ℓ]
        y11 = data.y11[ℓ]
        x22 = data.x22[ℓ]
        y22 = data.y22[ℓ]
        J1 = data.J1[ℓ]
        J2 = data.J2[ℓ]
        J1_1 = data.J1_1[ℓ]
        J2_2 = data.J2_2[ℓ]
        
        Ep = exp(im * pm * γ0)       # exp(i(p+m)γ⁰)
        Em = exp(-im * pm * γ0)      # exp(-i(p+m)γ⁰)
        Emξ = exp(im * m * data.ξ̂[ℓ])  # exp(imξ̂)
        
        # B1: kinematic BC upper
        A11m[ℓ] = (x11 - im * s * y11) * Ep * Emξ
        B11m[ℓ] = -abs_pm * Ep * Emξ
        
        # B2: dynamic BC LHS
        A21m[ℓ] = ρ_ / J1 * (im * s * x11 + y11) * Ep * Emξ
        A22m[ℓ] = -1.0 / J2 * (-im * s * x22 + y22) * Em * Emξ
        B21m[ℓ] = -ρ_ * Ep * Emξ
        B22m[ℓ] = Em * Emξ
        
        # B3: dynamic BC RHS
        A31m[ℓ] = ρ_ * (-1.0 / J1^2 * (im * s * x11 + y11) * im * pm + 1.0 / c^2) * Ep * Emξ
        A32m[ℓ] = (1.0 / J2^2 * (-im * s * x22 + y22) * im * pm - 1.0 / c^2) * Em * Emξ
        B31m[ℓ] = ρ_ / J1 * im * pm * Ep * Emξ
        B32m[ℓ] = -1.0 / J2 * im * pm * Em * Emξ
        C3m[ℓ] = (-0.5 * J2_2 / J2^2 + y22 / c^2 + ρ_ * (-0.5 * J1_1 / J1^2 + y11 / c^2)) * Emξ
        
        # B4: contact x-direction
        A41m[ℓ] = s * Ep * Emξ
        A42m[ℓ] = s * Em * Emξ
        C4m[ℓ] = im * (x11 + x22) * Emξ
        
        # B5: combined contact / streamfunction
        A51m[ℓ] = (x11 + x22 - im * s * (y11 + y22)) * Ep * Emξ
        A52m[ℓ] = (x11 + x22 + im * s * (y11 + y22)) * Em * Emξ
        B51m[ℓ] = A41m[ℓ]
        B52m[ℓ] = -A42m[ℓ]
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# FFT-based matrix assembly
# ─────────────────────────────────────────────────────────────────────────────

"""
    assemble_matrices(data, p, ρ_, c, M) -> (A11, B11, ..., B52)

Assemble all (2M+1)×(2M+1) Fourier coefficient matrices via FFT.
"""
function assemble_matrices(data::SteadyWaveData, p::Real, ρ_::Real, c::Real, M::Int)
    npts = length(data.ξ̂)
    sz = 2M + 1
    
    # Allocate output matrices
    mA11 = zeros(ComplexF64, sz, sz)
    mB11 = zeros(ComplexF64, sz, sz)
    mA21 = zeros(ComplexF64, sz, sz)
    mA22 = zeros(ComplexF64, sz, sz)
    mB21 = zeros(ComplexF64, sz, sz)
    mB22 = zeros(ComplexF64, sz, sz)
    mA31 = zeros(ComplexF64, sz, sz)
    mA32 = zeros(ComplexF64, sz, sz)
    mB31 = zeros(ComplexF64, sz, sz)
    mB32 = zeros(ComplexF64, sz, sz)
    mC3  = zeros(ComplexF64, sz, sz)
    mA41 = zeros(ComplexF64, sz, sz)
    mA42 = zeros(ComplexF64, sz, sz)
    mC4  = zeros(ComplexF64, sz, sz)
    mA51 = zeros(ComplexF64, sz, sz)
    mA52 = zeros(ComplexF64, sz, sz)
    mB51 = zeros(ComplexF64, sz, sz)
    mB52 = zeros(ComplexF64, sz, sz)
    
    # Temp vectors for coefficient evaluation
    A11m = Vector{ComplexF64}(undef, npts)
    B11m = similar(A11m)
    A21m = similar(A11m); A22m = similar(A11m)
    B21m = similar(A11m); B22m = similar(A11m)
    A31m = similar(A11m); A32m = similar(A11m)
    B31m = similar(A11m); B32m = similar(A11m)
    C3m  = similar(A11m)
    A41m = similar(A11m); A42m = similar(A11m)
    C4m  = similar(A11m)
    A51m = similar(A11m); A52m = similar(A11m)
    B51m = similar(A11m); B52m = similar(A11m)
    
    for (col, m) in enumerate(-M:M)
        evaluate_coefficients!(
            A11m, B11m, A21m, A22m, B21m, B22m,
            A31m, A32m, B31m, B32m, C3m,
            A41m, A42m, C4m,
            A51m, A52m, B51m, B52m,
            m, p, ρ_, c, data)
        
        # FFT each and extract k = -M:M
        extract_column!(mA11, col, A11m, M, npts)
        extract_column!(mB11, col, B11m, M, npts)
        extract_column!(mA21, col, A21m, M, npts)
        extract_column!(mA22, col, A22m, M, npts)
        extract_column!(mB21, col, B21m, M, npts)
        extract_column!(mB22, col, B22m, M, npts)
        extract_column!(mA31, col, A31m, M, npts)
        extract_column!(mA32, col, A32m, M, npts)
        extract_column!(mB31, col, B31m, M, npts)
        extract_column!(mB32, col, B32m, M, npts)
        extract_column!(mC3,  col, C3m,  M, npts)
        extract_column!(mA41, col, A41m, M, npts)
        extract_column!(mA42, col, A42m, M, npts)
        extract_column!(mC4,  col, C4m,  M, npts)
        extract_column!(mA51, col, A51m, M, npts)
        extract_column!(mA52, col, A52m, M, npts)
        extract_column!(mB51, col, B51m, M, npts)
        extract_column!(mB52, col, B52m, M, npts)
    end
    
    return (mA11, mB11, mA21, mA22, mB21, mB22,
            mA31, mA32, mB31, mB32, mC3,
            mA41, mA42, mC4, mA51, mA52, mB51, mB52)
end

"""
FFT the values vector and place k=-M:M coefficients into column `col` of matrix `mat`.
"""
function extract_column!(mat::Matrix{ComplexF64}, col::Int, 
                          vals::Vector{ComplexF64}, M::Int, npts::Int)
    coeffs = fft(vals) / npts
    for (row, k) in enumerate(-M:M)
        idx = mod(k, npts) + 1
        mat[row, col] = coeffs[idx]
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Elimination and eigenvalue solve
# ─────────────────────────────────────────────────────────────────────────────

"""
    solve_stability(a1_n, a2_n, c_n, c, ρ_; p=0.5, M=60, N=128) -> (σ, eigvecs, L, R)

Solve the linear stability eigenvalue problem for given steady wave and Floquet exponent p.

Returns eigenvalues σ (complex) and eigenvectors.
Unstable if any Re(σ) > 0.
"""
function solve_stability(a1_n::AbstractVector, a2_n::AbstractVector,
                          c_n::AbstractVector, c::Real, ρ_::Real;
                          p::Real=0.5, M::Int=60, N::Int=128)
    
    # Step 1: Compute steady wave data at collocation points
    data = compute_steady_data(a1_n, a2_n, c_n, N)
    
    # Step 2: Assemble all Fourier coefficient matrices
    (mA11, mB11, mA21, mA22, mB21, mB22,
     mA31, mA32, mB31, mB32, mC3,
     mA41, mA42, mC4, mA51, mA52, mB51, mB52) = assemble_matrices(data, p, ρ_, c, M)
    
    # Step 3: Elimination — express a₂, b₂, c in terms of a₁, b₁
    A53 = mA52 \ mA51       # a₂ = A53 * a₁
    B53 = mB52 \ mB51       # b₂ = B53 * b₁
    C5  = mC4  \ (mA41 + mA42 * A53)  # c = C5 * a₁
    
    # Step 4: Form combined matrices (eq. 4.33)
    A23 = mA21 + mA22 * A53
    B23 = mB21 + mB22 * B53
    A33 = mA31 + mA32 * A53 + mC3 * C5
    B33 = mB31 + mB32 * B53
    
    # Step 5: Assemble generalized eigenvalue problem σ L x = R x
    sz = 2M + 1
    Z = zeros(ComplexF64, sz, sz)
    
    L = [mA11  Z;  A23  B23]   # (4M+2) × (4M+2)
    R = [Z  mB11;  A33  B33]   # (4M+2) × (4M+2)
    
    # Step 6: Solve generalized eigenvalue problem
    F = eigen(R, L)
    
    return F.values, F.vectors, L, R
end

"""
    growth_rates(σ) -> Vector{Float64}

Extract growth rates (real parts) from eigenvalues.
"""
growth_rates(σ::AbstractVector) = real.(σ)

"""
    dominant_mode(eigvec, M) -> Int

Find the dominant mode number μ for an eigenvector.
The eigenvector has structure [a₁; b₁] where a₁ has indices m = -M:M.
"""
function dominant_mode(eigvec::AbstractVector, M::Int)
    sz = 2M + 1
    a1 = @view eigvec[1:sz]
    _, idx = findmax(abs.(a1))
    return idx - M - 1   # convert 1-based index to m ∈ {-M,...,M}
end

"""
    classify_eigenvalues(σ_vals, eigvecs, M) -> Vector{@NamedTuple{σ, μ, σ_r}}

Classify all eigenvalues by dominant mode number and growth rate.
"""
function classify_eigenvalues(σ_vals::AbstractVector, eigvecs::AbstractMatrix, M::Int)
    n = length(σ_vals)
    results = [(σ = σ_vals[i], 
                μ = dominant_mode(eigvecs[:, i], M),
                σ_r = real(σ_vals[i])) for i in 1:n]
    return sort(results, by = x -> x.μ)
end

# ─────────────────────────────────────────────────────────────────────────────
# Analytical approximations (Section 5)
# ─────────────────────────────────────────────────────────────────────────────

"""
    σ_KH(μ, p, ρ_, Δq_crest, c, c0) -> Float64

Approximate growth rate for wave-induced KH instability (eq. 5.1).
"""
function σ_KH(μ::Int, p::Real, ρ_::Real, Δq_crest::Real, c_wave::Real, c0::Real)
    pm = μ + p
    inside = pm^2 * ρ_ / (1 + ρ_)^2 * Δq_crest^2 - abs(pm)
    return real(sqrt(complex(inside))) * (c0 / c_wave)
end

"""
    μ_c_KH(p, ρ_, Δq_crest, c, c0) -> Float64

Critical mode number for KH instability onset (eq. 5.2).
"""
function μ_c_KH(p::Real, ρ_::Real, Δq_crest::Real, c_wave::Real, c0::Real)
    return (1 + ρ_)^2 / ρ_ * 1.0 / Δq_crest^2 * (c0 / c_wave)^2 - p
end

"""
    σ_NLS(p, h, ρ_) -> Float64

Approximate growth rate for modulational instability (eq. 5.6).
"""
function σ_NLS(p::Real, h::Real, ρ_::Real)
    inside = 2.0 * (1 + ρ_^2) / (1 + ρ_)^2 * h^2 - p^2
    return real(p / 8 * sqrt(complex(inside)))
end
