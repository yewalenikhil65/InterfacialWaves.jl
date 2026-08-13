# ─── Ring-source Green's function for axisymmetric BIE ───

"""
    ring_G(r, z, rp, zp) → Float64

Axisymmetric ring-source Green's function G₀(r,z; r',z').
Evaluates `(2/π) K(m) / √A` where `m = 4rr'/A`, `A = (r+r')² + (z-z')²`.

Returns `Inf` when the source and field points coincide.
"""
function ring_G(r::Float64, z::Float64, rp::Float64, zp::Float64)
    dz = z - zp
    A = (r + rp)^2 + dz^2
    A < 1e-30 && return Inf
    m = clamp(4r * rp / A, 0.0, 1.0 - 1e-15)
    (2.0 / π) * ellipk(m) / sqrt(A)
end

"""
    ring_grad_source(r, z, rp, zp) → (∂G₀/∂r', ∂G₀/∂z')

Gradient of the ring-source Green's function with respect to source coordinates.
"""
function ring_grad_source(r::Float64, z::Float64, rp::Float64, zp::Float64)
    dz = z - zp
    sr = r + rp
    A = sr^2 + dz^2
    A < 1e-30 && return (NaN, NaN)
    invA = 1.0 / A
    sqA = sqrt(A)
    m = clamp(4r * rp * invA, 0.0, 1.0 - 1e-15)
    K = ellipk(m)
    E = ellipe(m)
    dKdm = m < 1e-15 ? π / 8.0 : (E / (1.0 - m) - K) / (2.0 * m)
    c1 = (2.0 / π) / sqA
    c2 = (1.0 / π) * K / (A * sqA)
    dAdrp = 2.0 * sr
    dAdzp = -2.0 * dz
    dmdrp = 4.0 * r * invA - m * invA * dAdrp
    dmdzp = -m * invA * dAdzp
    (c1 * dKdm * dmdrp - c2 * dAdrp, c1 * dKdm * dmdzp - c2 * dAdzp)
end

"""
    ring_dGdn(r, z, rp, zp, nrp, nzp) → Float64

Normal derivative of the ring-source Green's function at the source point
(outward normal components `nrp`, `nzp`).
"""
function ring_dGdn(r::Float64, z::Float64, rp::Float64, zp::Float64,
                   nrp::Float64, nzp::Float64)
    gr, gz = ring_grad_source(r, z, rp, zp)
    gr * nrp + gz * nzp
end

"""
    ring_G_and_dGdn(r, z, rp, zp, nrp, nzp) → (G₀, ∂G₀/∂n')

Combined evaluation of Green's function and its normal derivative (avoids
redundant computation of elliptic integrals).
"""
function ring_G_and_dGdn(r::Float64, z::Float64, rp::Float64, zp::Float64,
                         nrp::Float64, nzp::Float64)
    dz = z - zp
    sr = r + rp
    A = sr^2 + dz^2
    A < 1e-30 && return (Inf, NaN)
    invA = 1.0 / A
    sqA = sqrt(A)
    m = clamp(4r * rp * invA, 0.0, 1.0 - 1e-15)
    K = ellipk(m)
    E = ellipe(m)
    G = (2.0 / π) * K / sqA
    dKdm = m < 1e-15 ? π / 8.0 : (E / (1.0 - m) - K) / (2.0 * m)
    c1 = (2.0 / π) / sqA
    c2 = (1.0 / π) * K / (A * sqA)
    dAdrp = 2.0 * sr
    dAdzp = -2.0 * dz
    dmdrp = 4.0 * r * invA - m * invA * dAdrp
    dmdzp = -m * invA * dAdzp
    gr = c1 * dKdm * dmdrp - c2 * dAdrp
    gz = c1 * dKdm * dmdzp - c2 * dAdzp
    (G, gr * nrp + gz * nzp)
end

"""
    ring_G_remainder(r, z, rp, zp, u) → Float64

Smooth remainder of G₀ after log-singularity subtraction for self-element quadrature.
Uses A&S 17.3.34 polynomial for `K_reg = K(m) + ½ln(m')` to avoid catastrophic
cancellation when `m' = 1-m < 10⁻⁶`.
"""
function ring_G_remainder(r::Float64, z::Float64, rp::Float64, zp::Float64, u::Float64)
    dz = z - zp
    sr = r + rp
    dr = r - rp
    A = sr^2 + dz^2
    A < 1e-30 && return (1.0 / π) * log(8.0 * max(r, rp))
    sqA = sqrt(A)
    mp = clamp((dr^2 + dz^2) / A, 1e-30, 1.0)
    m = 1.0 - mp
    K_reg = _K_reg(m, mp)
    abs_u = abs(u)
    smooth_part = rp * (2.0 / π) * K_reg / sqA
    abs_u < 1e-30 && return smooth_part
    (1.0 / π) * (log(abs_u) - rp / sqA * log(mp)) + smooth_part
end

# A&S 17.3.34: polynomial approximation for K(m) + ½ln(1-m) when m→1.
# Required because ellipk(m) + 0.5*log(1-m) suffers catastrophic cancellation
# for (1-m) < 10⁻⁶.
const _KA = (1.38629436112, 0.09666344259, 0.03590092383, 0.03742563713, 0.01451196212)
const _KB = (0.5, 0.12498593597, 0.06880248576, 0.03328355346, 0.00441787012)

"""Compute K(m) + ½ln(1-m) using A&S 17.3.34 polynomial when m' = 1-m < 10⁻⁶, otherwise directly."""
function _K_reg(m::Float64, mp::Float64)
    if mp > 1e-6
        return ellipk(m) + 0.5 * log(mp)
    end
    lnmp = log(mp)
    val = _KA[1]
    mpn = mp
    @inbounds for n in 1:4
        val += (_KA[n+1] - _KB[n+1] * lnmp) * mpn
        mpn *= mp
    end
    val
end
