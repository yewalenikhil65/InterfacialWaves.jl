# ─── Zhu (2000) cubic-spline smoothing ───

"""
    build_smoothing_matrix(mesh::CylindricalBasin; N_uni::Int=0) → Matrix{Float64}

Precompute the linear smoothing operator `S` such that `f_smooth = S * f`.

Following Zhu (2000 §2.5.1): evaluate f at a uniform grid (via cubic spline
from Chebyshev nodes), then fit a cubic spline through those uniform points
and evaluate back at Chebyshev nodes. The uniform grid acts as a low-pass
filter — content above its Nyquist frequency is suppressed.

Since cubic spline interpolation is linear in the data, the entire operation
is a matrix-vector product, compatible with ForwardDiff.

# Arguments
- `mesh::CylindricalBasin`: the basin mesh
- `N_uni::Int=0`: number of uniform grid points (default: 2× unique nodes)
"""
function build_smoothing_matrix(mesh::CylindricalBasin; N_uni::Int=0)
    r_cheb = surface_nodes(mesh)
    N_F = length(r_cheb)

    # Identify unique node positions (element boundaries share nodes)
    unique_idx = Int[1]
    for i in 2:N_F
        r_cheb[i] - r_cheb[unique_idx[end]] > 1e-14 && push!(unique_idx, i)
    end
    r_unique = r_cheb[unique_idx]
    N_u = length(r_unique)

    N_uni = N_uni > 0 ? N_uni : 2 * N_u
    r_uni = collect(range(r_unique[1], r_unique[end], length=N_uni))

    # Build S column by column: S[:,j] = smooth(e_j)
    S = zeros(N_F, N_F)
    f_unique = zeros(N_u)
    @inbounds for j in 1:N_F
        # Unit vector in direction j
        fill!(f_unique, 0.0)
        for (k, idx) in enumerate(unique_idx)
            f_unique[k] = (idx == j) ? 1.0 : 0.0
        end
        # Chebyshev → uniform via spline
        itp1 = CubicSpline(f_unique, r_unique)
        f_uni = [itp1(r) for r in r_uni]
        # Uniform → Chebyshev via spline
        itp2 = CubicSpline(f_uni, r_uni)
        for i in 1:N_F
            S[i, j] = itp2(r_cheb[i])
        end
    end
    S
end

"""
    smooth_chebyshev!(f::AbstractVector, S::Matrix{Float64})

Apply precomputed smoothing matrix `S` to `f` in-place.
"""
function smooth_chebyshev!(f::AbstractVector, S::Matrix{Float64})
    f .= S * f
end
