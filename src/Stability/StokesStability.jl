# =============================================================================
# Stokes Wave Linear Stability — Numerical Path (ApproxFun Multiplication operators)
# Reference: Longuet-Higgins (1978) §§3,6 and Appendix
# Source: AccurateStokesWave/src/new_stabilityAPI_fixed.jl (validated)
# =============================================================================

"""Y(ψ=0) = H₀ + H₁ cos(θ) + H₂ cos(2θ) + ..."""
_Y_fun(H::Vector{Float64}) = Fun(CosSpace(), H)

"""Yψ and Yϕ at ψ=0 from C coefficients."""
function _Yψ_Yϕ(C::Vector{Float64})
    return (Fun(CosSpace(), C), Fun(SinSpace(), -C[2:end]))
end

"""Second derivatives Yψψ and Yϕψ at ψ=0."""
function _Yψψ_Yϕψ(C::Vector{Float64}, phase_speed::Float64)
    return (Fun(CosSpace(), (1/phase_speed) * (C .* (0:length(C)-1))),
            Fun(SinSpace(), (-1/phase_speed) * (C[2:end] .* (1:length(C)-1))))
end

function _create_mult_pre_PQRS(Y, Yψ, Yϕ)
    return (Multiplication(Y, CosSpace()),
            Multiplication(Y, SinSpace()),
            Multiplication(Yϕ, SinSpace()),
            Multiplication(Yψ, CosSpace()))
end

function _compute_PQRS(Yϕ, Yψ, Yψψ, Yϕψ, M_Y_Cos, M_Y_Sin, M₂, M₃)
    P = M₂ * Yϕ + M₃ * Yψ
    Q = 2.0 * M_Y_Cos * Yψ
    R = 2.0 * M_Y_Sin * Yϕ
    M₅ = Multiplication(P, CosSpace())
    M₇ = Multiplication(Q, CosSpace())
    M10 = Multiplication(R, SinSpace())
    S = M₅ * Yψ + M₇ * Yψψ + M10 * Yϕψ
    return P, Q, R, S, M₅, M₇, M10
end

function _create_mult_post_PQRS(Yϕ, Yψ, P, Q, R, S)
    return (Multiplication(Yϕ, CosSpace()),
            Multiplication(Yψ, SinSpace()),
            Multiplication(P, SinSpace()),
            Multiplication(Q, SinSpace()),
            Multiplication(R, CosSpace()),
            Multiplication(S, CosSpace()),
            Multiplication(S, SinSpace()))
end

# =============================================================================
# M_Matrices!: validated direct matrix assembly
# =============================================================================

"""
    M_Matrices!(A, B, H, C, phase_speed; n_choose=5)

Assemble the generalized eigenvalue problem `A x = λ B x` for Stokes wave
linear stability (Longuet-Higgins 1978). Uses ApproxFun's Multiplication
operators for exact Fourier coefficient convolutions.
"""
function M_Matrices!(A::AbstractMatrix, B::AbstractMatrix,
        H::Vector{Float64}, C::Vector{Float64}, phase_speed::Float64;
        n_choose::Int=5)

    Y_ = _Y_fun(H)
    Yψ, Yϕ = _Yψ_Yϕ(C)
    M_Y_Cos, M_Y_Sin, M₂, M₃ = _create_mult_pre_PQRS(Y_, Yψ, Yϕ)
    Yψψ, Yϕψ = _Yψψ_Yϕψ(C, phase_speed)
    P, Q, R, S, M₅, M₇, M10 = _compute_PQRS(Yϕ, Yψ, Yψψ, Yϕψ, M_Y_Cos, M_Y_Sin, M₂, M₃)
    M₁, M₄, M₆, M₈, M₉, M11, M12 = _create_mult_post_PQRS(Yϕ, Yψ, P, Q, R, S)

    n = n_choose

    # --- A matrix assembly ---
    copyto!(A, CartesianIndices((1:n+1, 1:n+1)),
            M₃[1:n+1, 1:n+1], CartesianIndices((1:n+1, 1:n+1)))
    A[1:n+1, 2:n+1] .+= -M₂[1:n+1, 1:n]

    copyto!(A, CartesianIndices((n+2:2n+1, n+2:2n+2)),
            M₁[1:n, 1:n+1], CartesianIndices((1:n, 1:n+1)))
    A[n+2:2n+1, n+3:2n+2] .+= M₄[1:n, 1:n]

    copyto!(A, CartesianIndices((2+2n:2+3n, 2+n:2+2n)),
            M₃[1:n+1, 1:n+1], CartesianIndices((1:n+1, 1:n+1)))
    A[2+2n:2+3n, 3+n:2+2n] .+= -M₂[1:n+1, 1:n]
    A[2+2n:2+3n, 3+2n:2+3n] .+= M₅[1:n+1, 2:n+1]

    A[1, :] .*= 2.0
    A[2n+2, :] .*= 2.0

    copyto!(A, CartesianIndices((3n+3:4n+2, 1:n+1)),
            -M₁[1:n, 1:n+1], CartesianIndices((1:n, 1:n+1)))
    A[3n+3:4n+2, 2:n+1] .+= -M₄[1:n, 1:n]

    copyto!(A, CartesianIndices((3n+3:4n+2, 3n+3:4n+2)),
            M₆[1:n, 1:n], CartesianIndices((1:n, 1:n)))

    # --- B matrix assembly ---
    copyto!(B, CartesianIndices((1:n+1, 2+n:2+2n)),
            M₅[1:n+1, 1:n+1], CartesianIndices((1:n+1, 1:n+1)))
    B[1:n+1, 2+n:2+2n] .+= (M₇[1:n+1, 1:n+1] .* permutedims((1.0/phase_speed)*(0:n)))
    B[1:n+1, 3+n:2+2n] .+= M10[1:n+1, 1:n] .* permutedims((-1.0/phase_speed)*(1:n))

    copyto!(B, CartesianIndices((1:n+1, 1+2n+2:2n+2+n)),
            M11[1:n+1, 1:n+1], CartesianIndices((1:n+1, 2:n+1)))

    copyto!(B, CartesianIndices((n+2:2n+1, 3n+3:4n+2)),
            M12[1:n, 1:n], CartesianIndices((1:n, 1:n)))

    copyto!(B, CartesianIndices((n+2:2n+1, 2:1+n)),
            -M₆[1:n, 1:n], CartesianIndices((1:n, 1:n)))
    B[n+2:2n+1, 2:1+n] .+= M₈[1:n, 1:n] .* permutedims(-(1:n)/phase_speed)
    B[n+2:2n+1, 2:1+n] .+= M₉[1:n, 2:n+1] .* permutedims(-(1:n)/phase_speed)

    B[2+2n:2+3n, 3n+2:4n+2] .= Diagonal(-(0:n)/phase_speed)
    B[3+3n:4n+2, 3+2n:3n+2] .= Diagonal((1:n)/phase_speed)

    B[1, :] .*= 2.0
    B[2n+2, :] .*= 2.0

    return A, B
end

# =============================================================================
# NormalModeAnalysis: top-level entry from LH coefficients
# =============================================================================

"""
    NormalModeAnalysis(coeffs, phase_speed; n_choose=150, m=1)

Compute stability matrices for perturbation class `m`.

- `coeffs`: LH Fourier coefficients [H₀, H₁, H₂, ...] (a₀/2 convention)
- `phase_speed`: phase speed c
- `n_choose`: truncation for perturbation modes
- `m`: perturbation class (m=1 superharmonic, m>1 subharmonic)
"""
function NormalModeAnalysis(coeffs::AbstractVector{Float64}, phase_speed::Float64;
        n_choose::Int=150, m::Int=1)
    @assert m > 0

    H = zeros(length(coeffs))
    ind_to_keep = findall(rem.(eachindex(coeffs) .- 1, m) .== 0)
    H[ind_to_keep] .= coeffs[1:length(ind_to_keep)]
    H ./= m

    c_prime = phase_speed / √m

    C = copy(H)
    C[1] = 1.0 / c_prime
    for k in 2:length(C)
        C[k] = (k - 1) * H[k] / c_prime
    end

    A = zeros(4n_choose + 2, 4n_choose + 2)
    B = zeros(4n_choose + 2, 4n_choose + 2)
    M_Matrices!(A, B, H, C, c_prime; n_choose=n_choose)

    return A, B, H
end

# =============================================================================
# QEP eigenvalue solver (memory-optimized)
# =============================================================================

"""
    qep_eigen(A, B, n, phase_speed, m) -> (λ, Φ)

Solve the quadratic eigenvalue problem for Stokes-wave stability.
Returns eigenvalues λ and eigenvectors Φ.
"""
function qep_eigen(A::AbstractMatrix, B::AbstractMatrix, n::Int,
        phase_speed::Float64, m::Int)
    c_prime = phase_speed / √m
    half = 2n + 1
    sz = 4n + 2

    col_perm = vcat(1:n+1, 3n+3:sz, n+2:2n+2, 2n+3:3n+2)
    row_perm = vcat(1:n+1, 3n+3:sz, 2n+2:3n+2, n+2:2n+1)

    Ap = Matrix{Float64}(undef, sz, sz)
    Bp = Matrix{Float64}(undef, sz, sz)
    @inbounds for (jnew, jold) in enumerate(col_perm)
        scale = jold > 2n + 2 ? (-c_prime) : 1.0
        for (inew, iold) in enumerate(row_perm)
            Ap[inew, jnew] = A[iold, jold] * scale
            Bp[inew, jnew] = B[iold, jold] * scale
        end
    end

    B11 = @view Ap[1:half, 1:half]
    B22 = @view Ap[half+1:sz, half+1:sz]
    A12 = @view Bp[1:half, half+1:sz]
    A21 = @view Bp[half+1:sz, 1:half]

    zr = findfirst(i -> norm(@view(A21[i, :])) < 1e-14, 1:half)
    constraint = @view B22[zr, :]
    pivot_col = argmax(abs.(constraint))
    pivot_val = constraint[pivot_col]
    other_cols = setdiff(1:half, pivot_col)
    keep = setdiff(1:half, zr)
    hm1 = half - 1

    g_ratio = constraint[other_cols] / pivot_val
    A12T = A12[:, other_cols] .- (@view(A12[:, pivot_col])) * g_ratio'
    B22T = Ap[half .+ keep, half .+ other_cols] .- (@view(Ap[half .+ keep, half + pivot_col])) * g_ratio'
    A21k = Matrix(A21[keep, :])

    B22T_lu = lu(B22T)
    tmp = B22T_lu \ Matrix(A21k)
    M = B11 \ (A12T * tmp)
    μ, V1 = eigen(M)

    λ_pos = sqrt.(Complex.(μ))
    λ_all = Vector{ComplexF64}(undef, sz)
    Φ_all = zeros(ComplexF64, sz, sz)

    T_mat = zeros(half, hm1)
    for (idx, j) in enumerate(other_cols)
        T_mat[j, idx] = 1.0
    end
    T_mat[pivot_col, :] .= -(constraint[other_cols] / pivot_val)

    rhs = Vector{ComplexF64}(undef, hm1)
    v2r = Vector{ComplexF64}(undef, hm1)
    v2_pos = Vector{ComplexF64}(undef, half)

    for k in 1:half
        λp = λ_pos[k]
        if abs(λp) > 1e-14
            mul!(rhs, A21k, @view(V1[:, k]))
            v2r .= B22T_lu \ rhs
            v2r .*= (1.0 / λp)
        else
            v2r .= 0
        end
        mul!(v2_pos, T_mat, v2r)

        λ_all[k] = λp
        @inbounds for (i, j) in enumerate(col_perm)
            Φ_all[j, k] = i <= half ? Complex(V1[i, k]) : v2_pos[i - half]
        end
        @inbounds for j in 2n+3:sz
            Φ_all[j, k] *= (-c_prime)
        end

        kn = half + k
        λ_all[kn] = -λp
        @inbounds for (i, j) in enumerate(col_perm)
            Φ_all[j, kn] = i <= half ? Complex(V1[i, k]) : -v2_pos[i - half]
        end
        @inbounds for j in 2n+3:sz
            Φ_all[j, kn] *= (-c_prime)
        end
    end

    return λ_all, Φ_all
end

# =============================================================================
# Perturbed surface reconstruction
# =============================================================================

"""
    perturbed_surface(θ, coeffs, phase_speed, Φ, n_choose; ε=0.05)

Reconstruct the perturbed free surface from an eigenvector.
Returns `(x_base, y_base, x_pert, y_pert)`.
"""
function perturbed_surface(θ::AbstractVector, coeffs::Vector{Float64},
        phase_speed::Float64, Φ::AbstractVector, n_choose::Int; ε::Float64=0.05)
    Nc = length(coeffs) - 1
    a = real.(Φ[1:n_choose+1])
    b = real.(Φ[n_choose+2:2n_choose+2])
    cd = real.(Φ[2n_choose+3:3n_choose+2])
    dd = real.(Φ[3n_choose+3:4n_choose+2])

    Npts = length(θ)
    ξ_hat = fill(a[1], Npts)
    η_hat = fill(b[1], Npts)
    F_hat = zeros(Npts)

    for nn in 1:n_choose
        cn = cos.(nn .* θ); sn = sin.(nn .* θ)
        ξ_hat .+= a[nn+1] .* cn .+ b[nn+1] .* sn
        η_hat .+= b[nn+1] .* cn .- a[nn+1] .* sn
        nn <= length(cd) && (F_hat .+= cd[nn] .* cn)
        nn <= length(dd) && (F_hat .+= dd[nn] .* sn)
    end

    x_base = copy(θ)
    y_base = fill(coeffs[1], Npts)
    Yψ_base = fill(1.0 / phase_speed, Npts)
    for nn in 1:Nc
        cn = cos.(nn .* θ); sn = sin.(nn .* θ)
        noc = nn / phase_speed
        x_base .+= coeffs[nn+1] .* sn
        y_base .+= coeffs[nn+1] .* cn
        Yψ_base .+= noc .* coeffs[nn+1] .* cn
    end
    y_base .-= coeffs[1]

    x_pert = x_base .+ ε .* ξ_hat
    y_pert = y_base .+ ε .* (η_hat .+ F_hat .* Yψ_base)
    return x_base, y_base, x_pert, y_pert
end
