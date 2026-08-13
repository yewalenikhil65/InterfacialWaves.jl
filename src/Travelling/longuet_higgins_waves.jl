# =============================================================================
# Longuet-Higgins (1978) Toeplitz Stokes Wave Formulation
#
# Parameterizes the Stokes wave by its Fourier coefficients x = [a₀, a₁, ..., aₙ]
# with the nonlinear system:
#   f(x) = SymmetricToeplitz(x) · v(x)
# where v(x) = [1, 1·x₂, 2·x₃, ..., N·xₙ₊₁].
#
# The constraint is the steepness: ∑ x[even indices] = ak.
# Phase speed: c = √(-f(x)[1]).
#
# Reference: Longuet-Higgins (1978), "The instabilities of gravity waves of
# finite amplitude in deep water. II. Subharmonics."
# =============================================================================

# Note: The residual and Jacobian are implemented with explicit O(N²) loops,
# matching the structure of SymmetricToeplitz(x) * v(x) without requiring
# the ToeplitzMatrices package as a dependency.

# =============================================================================
# Residual functions
# =============================================================================

"""
    LonguetHigginsResidual!(out, x, p)

In-place Stokes wave residual using the Longuet-Higgins Toeplitz formulation.

# System (M = N+1 unknowns, M equations):
- `out[1] = ∑ x[even] - ak`  (steepness constraint)
- `out[i] = [SymmetricToeplitz(x) · v(x)]ᵢ` for i = 2:M  (Bernoulli + kinematic)

`p` is a named tuple `(ak=...,)`.
"""
function LonguetHigginsResidual!(out::AbstractVector{T}, x::AbstractVector{T}, p) where T
    (; ak) = p
    M = length(x)

    # Steepness constraint: ∑ x[even indices] = ak
    sum_even = zero(T)
    @inbounds for i in 2:2:M
        sum_even += x[i]
    end
    out[1] = sum_even - ak

    # Toeplitz product rows 2:M (the actual wave equations)
    @inbounds for i in 2:M
        val = zero(T)
        for j in 1:M
            v_j = j == 1 ? one(T) : T(j - 1) * x[j]
            val += x[abs(i - j) + 1] * v_j
        end
        out[i] = val
    end

    return out
end

"""
    LonguetHigginsJacobian!(J, x, p)

In-place analytical Jacobian of `LonguetHigginsResidual!`.

The Jacobian has structure:
- Row 1: `J[1,k] = 1` if k is even, `0` otherwise (steepness constraint).
- Row i≥2: `J[i,k] = ∂fᵢ/∂xₖ` where f = SymmetricToeplitz(x) · v(x).

The derivative `∂fᵢ/∂xₖ` decomposes as:
- Term 1: derivative of the Toeplitz matrix entries (v fixed)
- Term 2: derivative of v_k (Toeplitz row fixed)

Specifically: `∂fᵢ/∂xₖ = v(i+k-1) + v(i-k+1) + x[|i-k|+1] · (k-1)`
where `v(j)` reads from the v-vector with bounds checking.
"""
function LonguetHigginsJacobian!(J::AbstractMatrix{T}, x::AbstractVector{T}, p) where T
    M = length(x)

    @inbounds for k in 1:M
        # Row 1: steepness constraint
        J[1, k] = iseven(k) ? one(T) : zero(T)

        # v'(k) = k - 1 (derivative of v_k w.r.t. x_k)
        vpk = T(k - 1)

        for i in 2:M
            # Term 1: ∂(Toeplitz row i · v) / ∂x_k from the Toeplitz matrix structure
            if k == 1
                # ∂T[i,j]/∂x_1 = δ_{|i-j|, 0} = δ_{i,j}, so term1 = v_i
                t1 = (i >= 2 && i <= M) ? T(i - 1) * x[i] : (i == 1 ? one(T) : zero(T))
            else
                # ∂T[i,j]/∂x_k contributes where |i-j|+1 = k, i.e., j = i+k-1 or j = i-k+1
                ipk = i + k - 1
                imk = i - k + 1
                t1a = (2 <= ipk <= M) ? T(ipk - 1) * x[ipk] : (ipk == 1 ? one(T) : zero(T))
                t1b = (2 <= imk <= M) ? T(imk - 1) * x[imk] : (imk == 1 ? one(T) : zero(T))
                t1 = t1a + t1b
            end

            # Term 2: Toeplitz row i, column k × ∂v_k/∂x_k
            J[i, k] = t1 + x[abs(i - k) + 1] * vpk
        end
    end

    return J
end

# =============================================================================
# Phase speed computation
# =============================================================================

"""
    longuet_higgins_phase_speed(x::AbstractVector)

Compute phase speed from the Longuet-Higgins solution vector.

`x` uses full `a₀` convention. The phase speed is `c = √(-f(x)[1])`.
For a converged solution, `f(x)[1] < 0` (it equals `-c²`).
"""
function longuet_higgins_phase_speed(x::AbstractVector{T}) where T
    M = length(x)
    # Compute f(x)[1] = row 1 of SymmetricToeplitz(x) · v(x)
    # Row 1 of SymmetricToeplitz(x) is: [x[1], x[2], x[3], ..., x[M]]
    # (because |1-j|+1 = j for all j=1,...,M)
    # v = [1, 1·x[2], 2·x[3], ..., (M-1)·x[M]]
    val = x[1]  # x[1] * v[1] = x[1] * 1
    @inbounds for j in 2:M
        val += x[j] * T(j - 1) * x[j]
    end
    # val = f(x)[1] = -c² for a converged solution
    return sqrt(abs(val))
end

"""
    longuet_higgins_phase_speed_half(coeffs::AbstractVector)

Compute phase speed from coefficients in `a₀/2` convention (first entry is a₀/2).
"""
function longuet_higgins_phase_speed_half(coeffs::AbstractVector{T}) where T
    x = copy(coeffs)
    x[1] *= 2
    return longuet_higgins_phase_speed(x)
end

# =============================================================================
# Steepness continuation schedule
# =============================================================================

"""
    AK_STOKES_LIMIT

Maximum steepness of a Stokes wave in deep water (the "highest wave"):
`ak ≈ 0.4434` (Schwartz & Fenton 1982).  Beyond this value the wave
develops a 120° corner at its crest and no smooth periodic solution exists.
"""
const AK_STOKES_LIMIT = 0.4434

"""
    default_ak_schedule(ak_max; phase_boundary=0.42)

Generate a steepness continuation schedule for the Longuet-Higgins method.

If `ak_max > AK_STOKES_LIMIT` (≈ 0.4434), a warning is issued and the
schedule is clamped to `AK_STOKES_LIMIT`.

Two phases:
- `ak ≤ 0.42`: coarse steps (Δak = 0.005)
- `ak > 0.42`: micro-steps (Δak = 5e-5) approaching the limiting wave
"""
function default_ak_schedule(ak_max::T; phase_boundary::T=T(0.42)) where T
    if ak_max > T(AK_STOKES_LIMIT)
        @warn "Requested steepness ak=$ak_max exceeds the Stokes limiting steepness " *
              "(ak_max = $AK_STOKES_LIMIT). No smooth periodic deep-water gravity wave " *
              "exists beyond this value. Falling back to ak = $AK_STOKES_LIMIT."
        ak_max = T(AK_STOKES_LIMIT)
    end
    schedule = T[]

    # Phase 1: coarse steps from 0.01 to min(ak_max, phase_boundary)
    ak_phase1_end = min(ak_max, phase_boundary)
    append!(schedule, T(0.01):T(0.005):ak_phase1_end)

    # Phase 2: micro-steps beyond the phase boundary
    if ak_max > phase_boundary
        append!(schedule, (phase_boundary + T(5e-5)):T(5e-5):ak_max)
    end

    return schedule
end
